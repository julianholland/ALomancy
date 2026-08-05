import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Union

from expyre.func import ExPyRe

from alomancy.configs.remote_info import RemoteInfo

logger = logging.getLogger(__name__)

_expyre_db_lock = threading.Lock()
_expyre_db_patched = False

# Shared across every RemoteJobExecutor instance in this process (not just
# one instance's threads): guards every ssh-invoking call ExPyRe makes on a
# job's behalf -- job.start() (mkdir/stage-input/sbatch submit) and
# sync_remote_results_status() (squeue status + rsync results, called from
# inside get_results()'s polling loop). Both shell out over the same shared
# multiplexed control connection to a given HPC host; letting
# max_concurrent_jobs threads hit either one simultaneously can exceed
# whatever session/connection cap the remote sshd enforces, at which point
# the excess sessions silently fall back to a fresh, separately-authenticated
# connection that hangs forever if that auth needs interactive input nobody
# is present to provide. The lock only wraps the brief moment of the actual
# subprocess call, not get_results()'s check_interval sleeps, so each job's
# own polling cadence -- and hence real completion order -- stays independent.
_ssh_call_lock = threading.Lock()
_expyre_sync_patched = False


def _ensure_expyre_sync_serialized() -> None:
    """Serialize ExPyRe's remote status/file sync calls behind _ssh_call_lock.

    sync_remote_results_status() is where get_results()'s polling loop
    actually shells out: system.scheduler.status(...) (ssh squeue-equivalent)
    and system.get_remotes(...) (ssh/rsync). Every job being monitored runs
    its own independent get_results() loop in its own thread, so without
    this, up to max_concurrent_jobs threads can trigger these simultaneously
    -- this was observed in production as a stuck ``squeue`` subprocess
    exactly like the stuck job.start() calls this module already guards
    against. Idempotent; patches the class once regardless of how many
    ExPyRe instances/threads call it.
    """
    global _expyre_sync_patched
    if _expyre_sync_patched:
        return

    orig_sync = ExPyRe.sync_remote_results_status

    def _locked_sync(self, *args, **kwargs):
        with _ssh_call_lock:
            return orig_sync(self, *args, **kwargs)

    ExPyRe.sync_remote_results_status = _locked_sync
    _expyre_sync_patched = True


def _ensure_expyre_db_thread_safe() -> None:
    """Make expyre's module-level JobsDB connection safe for concurrent threads.

    ExPyRe routes job.start()/job.get_results() through a single module-level
    sqlite3 connection (expyre.config.db) opened with the default
    check_same_thread=True. Every read/write funnels through JobsDB._execute,
    a single choke point (a self-contained ``with self.db: ...`` transaction
    per call). Reopening the connection with check_same_thread=False and
    serializing just that call behind one lock makes concurrent access safe
    without giving up real wall-clock concurrency of the blocking waits
    between checks. Idempotent; a no-op if expyre hasn't been configured yet.
    """
    global _expyre_db_patched
    if _expyre_db_patched:
        return
    from expyre import config as expyre_config

    db = expyre_config.db
    if db is None:
        return

    import sqlite3

    db.db = sqlite3.connect(db.db_filename, check_same_thread=False)
    orig_execute = db._execute

    def _locked_execute(cmd, *args, **kwargs):
        with _expyre_db_lock:
            return orig_execute(cmd, *args, **kwargs)

    db._execute = _locked_execute
    _expyre_db_patched = True


class RemoteJobExecutor:
    """
    General-purpose remote job submission utility.

    Handles submitting arbitrary functions to remote compute resources
    using the ExPyRe framework.
    """

    def __init__(self, remote_info: RemoteInfo):
        self.remote_info = remote_info
        self.jobs = []

    def submit_job(
        self,
        function: Callable,
        function_kwargs: dict[str, Any],
        input_files: list[Union[str, Path]] | None = None,
        output_files: list[Union[str, Path]] | None = None,
        job_name: str | None | None = None,
        **expyre_kwargs,
    ) -> ExPyRe:
        if input_files is None:
            input_files = []
        if output_files is None:
            output_files = []
        if job_name is None:
            job_name = self.remote_info.job_name
        input_files = [str(f) for f in (input_files or [])]
        output_files = [str(f) for f in (output_files or [])]

        final_input_files = input_files or self.remote_info.input_files
        final_output_files = output_files or getattr(
            self.remote_info, "output_files", []
        )

        job = ExPyRe(
            name=job_name or self.remote_info.job_name,
            pre_run_commands=self.remote_info.pre_cmds,
            post_run_commands=getattr(self.remote_info, "post_cmds", []),
            env_vars=getattr(self.remote_info, "env_vars", {}),
            input_files=final_input_files,
            output_files=final_output_files,
            function=function,
            kwargs=function_kwargs,
            **expyre_kwargs,
        )

        self.jobs.append(job)
        return job

    def submit_multiple_jobs(
        self,
        function: Callable,
        job_configs: list[dict[str, Any]],
        common_input_files: list[Union[str, Path]] | None = None,
        common_output_pattern: str | None | None = None,
        job_name_pattern: str | None | None = None,
    ) -> list[ExPyRe]:
        if common_input_files is None:
            common_input_files = []
        if job_name_pattern is None:
            job_name_pattern = self.remote_info.job_name

        jobs = []
        common_input_files = common_input_files or []

        for i, config in enumerate(job_configs):
            job_input_files = list(common_input_files)
            if "input_files" in config:
                job_input_files.extend(config["input_files"])

            job_output_files = config.get("output_files", [])
            if common_output_pattern:
                job_output_files.append(common_output_pattern.format(job_id=i))
            logger.debug("Job %d output files: %s", i, job_output_files)

            job_name = config.get("job_name")
            if not job_name and job_name_pattern:
                job_name = job_name_pattern.format(job_id=i)

            job_function = config.get("function", function)

            job = self.submit_job(
                function=job_function,
                function_kwargs=config["function_kwargs"],
                input_files=job_input_files,
                output_files=job_output_files,
                job_name=job_name,
            )
            jobs.append(job)

        return jobs

    def _run_single_job(self, index: int, job: ExPyRe) -> tuple[int, Any]:
        """Start one job and block until it resolves. Runs inside a worker
        thread of run_all_jobs_bounded's pool -- never lets an exception
        propagate, so one job's failure frees its slot without blocking or
        cancelling siblings.

        job.start() is serialized across threads via the module-level
        _ssh_call_lock (see its docstring); job.get_results()'s own
        ssh-invoking calls are serialized the same way via the
        sync_remote_results_status() monkeypatch, applied once by
        _ensure_expyre_sync_serialized() in run_all_jobs_bounded(). The lock
        is only held for the moment of each subprocess call, not for
        get_results()'s check_interval sleeps in between, so each job's own
        polling cadence -- and hence real completion order -- stays
        independent: a fast job's result still becomes available immediately
        rather than waiting behind an earlier-submitted, still-running one."""
        job_name = getattr(job, "name", f"job_{index}")
        try:
            with _ssh_call_lock:
                start_wall_time = time.time()
                job.start(
                    resources=self.remote_info.resources,
                    system_name=self.remote_info.sys_name,
                    header_extra=getattr(self.remote_info, "header_extra", []),
                    exact_fit=getattr(self.remote_info, "exact_fit", True),
                    partial_node=getattr(self.remote_info, "partial_node", False),
                )
            logger.debug("Job %d submitted to queue: %s", index + 1, job_name)

            result, _stdout, _stderr = job.get_results(
                timeout=self.remote_info.timeout,
                check_interval=getattr(self.remote_info, "check_interval", 10),
            )
            logger.info("Job %d completed successfully.", index + 1)
            try:
                started_file = job.stage_dir / "_expyre_job_started"
                if started_file.exists():
                    queue_s = max(0.0, started_file.stat().st_mtime - start_wall_time)
                    logger.info("Job %d queue_time=%.1f s.", index + 1, queue_s)
            except Exception as _qe:
                logger.debug(
                    "Could not compute queue time for job %d: %s", index + 1, _qe
                )
            return index, result
        except Exception as exc:
            logger.warning("Job %d failed: %s", index + 1, exc)
            return index, None

    def run_all_jobs_bounded(self) -> list[Any]:
        """Start and wait for all submitted jobs, keeping at most
        remote_info.max_concurrent_jobs started at once. The instant one
        finishes (success or failure), the next pending job is started to
        fill its slot -- ThreadPoolExecutor provides this rolling-window
        scheduling for free. Returned results stay index-aligned with the
        job_configs passed to submit_multiple_jobs, regardless of the order
        jobs actually complete in. Every individual ssh-invoking call (job
        start, and each job's own status/result sync while being monitored)
        is serialized regardless of max_concurrent_jobs -- see _ssh_call_lock
        -- but the surrounding wait/poll cadence keeps real concurrency."""
        if not self.jobs:
            return []

        _ensure_expyre_db_thread_safe()
        _ensure_expyre_sync_serialized()

        max_concurrent_jobs = getattr(self.remote_info, "max_concurrent_jobs", 20)
        if not max_concurrent_jobs or max_concurrent_jobs < 1:
            logger.warning(
                "max_concurrent_jobs=%s is invalid; falling back to 1 "
                "(serial submission).",
                max_concurrent_jobs,
            )
            max_concurrent_jobs = 1
        max_workers = min(max_concurrent_jobs, len(self.jobs))

        results: list[Any] = [None] * len(self.jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_single_job, i, job): i
                for i, job in enumerate(self.jobs)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as exc:
                    logger.error(
                        "Job %d worker raised unexpectedly: %s", index + 1, exc
                    )

        return results

    def cleanup_jobs(self) -> None:
        for job in self.jobs:
            job.mark_processed()

    def run_and_wait(
        self,
        function: Callable,
        job_configs: list[dict[str, Any]],
        **kwargs,
    ) -> list[Any]:
        logger.debug("run_and_wait working directory: %s", os.getcwd())
        self.submit_multiple_jobs(function, job_configs, **kwargs)
        results = self.run_all_jobs_bounded()
        self.cleanup_jobs()
        return results
