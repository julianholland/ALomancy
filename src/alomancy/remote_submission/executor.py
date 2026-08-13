import fcntl
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from expyre.func import ExPyRe

from alomancy.configs.remote_info import RemoteInfo

logger = logging.getLogger(__name__)

_expyre_db_lock = threading.Lock()
_expyre_db_patched = False

# Kept open for the lifetime of this process once acquired -- see
# acquire_local_expyre_lock. A plain reference (rather than closing it) is
# what keeps the OS-level flock held.
_expyre_root_lock_handle = None
_expyre_root_lock_guard = threading.Lock()


def acquire_local_expyre_lock() -> None:
    """Exclusively lock this process's resolved ExPyRe local_stage_dir,
    raising immediately if another alomancy process already holds it.

    _ensure_expyre_db_thread_safe/_get_ssh_call_lock above only serialize
    *threads within this one process* -- a plain threading.Lock gives zero
    protection between two separate alomancy processes (e.g. concurrent AL
    runs on different machines) that resolve to the same local_stage_dir,
    which happens whenever they share a jobs.db over a shared home
    directory (see alomancy/__init__.py's _ensure_local_expyre_root, which
    gives each run its own local_stage_dir by default and makes this a
    last-resort guard rather than the primary defense). A second process
    mutating that same sqlite file/job-stage-dir tree outside this
    process's control can make a job's row vanish out from under this
    process's poll loop, surfacing as a bare IndexError deep in expyre
    (func.py's "list(config.db.jobs(id=...))[0]" pattern) with no
    indication of why -- exactly the failure mode this guard exists to
    turn into an immediate, actionable error instead.

    fcntl.flock (not a hand-rolled PID file) is used because the OS
    releases it automatically the moment this process's file descriptor
    closes -- including on a crash or kill -9 -- so a dead process can
    never leave a stale lock blocking a legitimate restart. POSIX-only,
    which matches this codebase's Linux/HPC-only environment.

    Call once, early (pre_run_checks, before any remote submission); the
    lock is held for the rest of this process's lifetime by keeping the
    open file handle referenced in _expyre_root_lock_handle. A no-op if
    this process already holds it, or if expyre has no local_stage_dir
    resolved yet (e.g. no HPC configured) -- nothing to guard in that case.

    The check-and-set of _expyre_root_lock_handle is itself guarded by
    _expyre_root_lock_guard (double-checked locking, matching
    _get_ssh_call_lock's pattern above): fcntl.flock is scoped per *open
    file description*, not per-process, so if this function ever ran from
    two threads in the same process at once, both could see the handle as
    unset and each open() a fresh file description to the same lock_path --
    the second would then fail to flock() against the first and raise
    "another alomancy process already holds the lock" against its own
    process. In current usage pre_run_checks() only ever calls this once
    from the main thread before any worker threads start, so this can't
    actually happen today, but the guard costs nothing and keeps this
    function safe under any future call pattern.
    """
    global _expyre_root_lock_handle
    if _expyre_root_lock_handle is not None:
        return

    with _expyre_root_lock_guard:
        if _expyre_root_lock_handle is not None:
            return

        from expyre import config as expyre_config

        local_stage_dir = expyre_config.local_stage_dir
        if local_stage_dir is None:
            return

        lock_path = Path(local_stage_dir) / "alomancy.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")  # noqa: SIM115 -- must outlive this function to hold the lock
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = (
                fh.read().strip() or "<unknown, lock held before this field existed>"
            )
            fh.close()
            raise RuntimeError(
                f"Another alomancy process already holds the lock on "
                f"{local_stage_dir} (recorded holder: {holder}). Running two "
                "alomancy processes against the same ExPyRe local_stage_dir/"
                "jobs.db at once can corrupt job-tracking state. If that "
                f"process has genuinely exited without cleaning up, delete "
                f"{lock_path} and retry."
            ) from None

        fh.seek(0)
        fh.truncate()
        fh.write(
            f"pid={os.getpid()} host={socket.gethostname()} "
            f"started={datetime.now().isoformat()}\n"
        )
        fh.flush()
        _expyre_root_lock_handle = fh


# One lock per HPC system_name (not one global lock): guards every
# ssh-invoking call ExPyRe makes on a job's behalf for that host --
# job.start() (mkdir/stage-input/sbatch submit) and
# sync_remote_results_status() (squeue status + rsync results, called from
# inside get_results()'s polling loop). Both shell out over that host's own
# shared multiplexed control connection; letting max_concurrent_jobs threads
# hit either one simultaneously can exceed whatever session/connection cap
# the remote sshd enforces, at which point the excess sessions silently fall
# back to a fresh, separately-authenticated connection that hangs forever if
# that auth needs interactive input nobody is present to provide. Scoping per
# system_name (rather than one lock shared process-wide) means two
# RemoteJobExecutors targeting different, unrelated HPC hosts never serialize
# against each other's ssh traffic -- only calls that actually share one
# host's connection contend for the same lock. Each lock only wraps the brief
# moment of the actual subprocess call, not get_results()'s check_interval
# sleeps, so each job's own polling cadence -- and hence real completion
# order -- stays independent.
_ssh_call_locks: dict[str, threading.Lock] = {}
_ssh_call_locks_guard = threading.Lock()
_expyre_sync_patched = False

# Per-host counters used to coalesce redundant sync_remote_results_status()
# calls -- see _ensure_expyre_sync_serialized.
_sync_generations: dict[str, int] = {}


def _get_ssh_call_lock(sys_name: str) -> threading.Lock:
    """Return the shared ssh-call lock for one HPC system, creating it on
    first use (double-checked under _ssh_call_locks_guard, which is only
    ever held for the instant of a dict lookup/insert -- never for the ssh
    call itself)."""
    lock = _ssh_call_locks.get(sys_name)
    if lock is not None:
        return lock
    with _ssh_call_locks_guard:
        lock = _ssh_call_locks.get(sys_name)
        if lock is None:
            lock = threading.Lock()
            _ssh_call_locks[sys_name] = lock
        return lock


def _acquire_ssh_call_lock_or_raise(
    sys_name: str, timeout: float | None, description: str
) -> threading.Lock:
    """Acquire the per-host ssh-call lock, raising TimeoutError rather than
    blocking forever if it isn't free within `timeout` seconds (None blocks
    indefinitely -- Lock.acquire's own default). Caller must release() the
    returned lock, typically via try/finally.

    A wait this long means the thread currently holding the lock isn't just
    busy -- it's been longer than this job's own expected total runtime
    (get_remote_info sets `timeout` from the job's max_time/max_go_time) just
    to get a turn to touch ssh at all. That points at the lock holder being
    genuinely stuck: most likely the shared multiplexed control connection
    to this host died (network blip, remote sshd restart, ServerAliveCountMax
    teardown) and the next ssh call fell back to a fresh connection needing
    interactive auth (password/OTP) that nobody is present to provide -- see
    executor.py's module docstring comment above _ssh_call_locks. Giving up
    here can't un-stick that original hung ssh subprocess (this lock timeout
    doesn't touch it), but it stops every *other* job on the host from
    silently hanging alongside it: they fail loudly and promptly instead,
    matching how a mid-run hang would have surfaced before ssh calls were
    serialized at all -- one stuck job, not the whole host.
    """
    lock = _get_ssh_call_lock(sys_name)
    effective_timeout = -1 if timeout is None else timeout
    if not lock.acquire(timeout=effective_timeout):
        raise TimeoutError(
            f"Timed out after {effective_timeout:.0f}s waiting for the "
            f"ssh-call lock for host '{sys_name}' ({description}). Another "
            "job's ssh call on this host appears stuck -- most likely the "
            "shared control connection died and a fresh connection needs "
            "interactive auth (password/OTP) nobody is present to provide. "
            "Treating this job as failed rather than waiting indefinitely."
        )
    return lock


def _ensure_expyre_sync_serialized() -> None:
    """Serialize ExPyRe's remote status/file sync calls behind a per-host
    lock from _get_ssh_call_lock, and coalesce redundant calls.

    sync_remote_results_status() is where get_results()'s polling loop
    actually shells out: system.scheduler.status(...) (ssh squeue-equivalent)
    and system.get_remotes(...) (ssh/rsync). Every job being monitored runs
    its own independent get_results() loop in its own thread, so without
    this, up to max_concurrent_jobs threads can trigger these simultaneously
    -- this was observed in production as a stuck ``squeue`` subprocess
    exactly like the stuck job.start() calls this module already guards
    against.

    The calls this codebase makes always use sync_all=True (ExPyRe's
    default, never overridden here), meaning one call already syncs every
    job on that host, not just the caller's own. So once a thread has waited
    for the lock, another thread may have already done a full sync for that
    host in the meantime, making its own call redundant -- purely wasted
    ssh/rsync round-trips, and enough of them queued up can push a polling
    round past check_interval. _sync_generations makes each thread check,
    right after acquiring the lock, whether a sync has completed since it
    decided it needed one; if so, it skips its own call rather than
    repeating an already-fresh result.

    Lock acquisition is bounded by the job's own _alomancy_lock_timeout
    (stashed on the ExPyRe instance by _run_single_job right after start(),
    from remote_info.lock_timeout) -- see _acquire_ssh_call_lock_or_raise.

    Idempotent; patches the class once regardless of how many ExPyRe
    instances/threads call it.
    """
    global _expyre_sync_patched
    if _expyre_sync_patched:
        return

    orig_sync = ExPyRe.sync_remote_results_status

    def _locked_sync(self, *args, **kwargs):
        sys_name = self.system_name
        seen_generation = _sync_generations.get(sys_name, 0)
        lock = _acquire_ssh_call_lock_or_raise(
            sys_name,
            getattr(self, "_alomancy_lock_timeout", None),
            f"sync_remote_results_status() for job {getattr(self, 'id', '?')}",
        )
        try:
            if _sync_generations.get(sys_name, 0) != seen_generation:
                # Another thread already ran a full sync for this host
                # while we waited for the lock -- nothing left to do.
                return None
            result = orig_sync(self, *args, **kwargs)
            _sync_generations[sys_name] = _sync_generations.get(sys_name, 0) + 1
            return result
        finally:
            lock.release()

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

    `_execute` returns a live sqlite3.Cursor, not fetched rows -- and every
    caller in jobsdb.py only ever consumes it as an iterable afterward
    (`list(self._execute(...))` in add/remove/update, `for row in
    self._execute(...)` in jobs()), never touching cursor-specific methods.
    The first version of this lock released `_expyre_db_lock` the instant
    `self.db.execute(cmd)` returned the cursor -- *before* the caller
    actually iterated it. Because check_same_thread=False only disables
    sqlite3's same-thread assertion, not genuine thread-safety of
    concurrent cursor use on one shared connection, a second thread's own
    `execute()` call on that same connection could land in the gap between
    "cursor returned" and "cursor iterated", corrupting the first thread's
    still-pending read and making it silently yield zero rows. That is
    exactly what surfaced in production, from a single process with many
    concurrent job-monitoring threads, as a bare
    `list(config.db.jobs(id=...))[0]` -> IndexError deep in expyre's
    get_results() (func.py) -- with no relation to cross-process
    contention. `_locked_execute` below fully materializes the cursor into
    a list *before* releasing the lock, so by the time any other thread can
    touch the shared connection again, this thread's rows are already
    safely in a plain Python list with no further dependency on shared
    connection state.
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
            return list(orig_execute(cmd, *args, **kwargs))

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

        job.start() is serialized across threads via the per-host lock from
        _get_ssh_call_lock (see its docstring); job.get_results()'s own
        ssh-invoking calls are serialized the same way via the
        sync_remote_results_status() monkeypatch, applied once by
        _ensure_expyre_sync_serialized() in run_all_jobs_bounded(). The lock
        is only held for the moment of each subprocess call, not for
        get_results()'s check_interval sleeps in between, so each job's own
        polling cadence -- and hence real completion order -- stays
        independent: a fast job's result still becomes available immediately
        rather than waiting behind an earlier-submitted, still-running one.

        Lock acquisition is bounded by remote_info.lock_timeout (see
        _acquire_ssh_call_lock_or_raise) rather than waiting forever, so one
        job stuck on an unattended interactive-auth prompt can't silently
        freeze every other job on the same host indefinitely. The same
        timeout is stashed on the job instance for sync_remote_results_status
        to reuse later, since that call only has `self` to work with, not
        this RemoteJobExecutor."""
        job_name = getattr(job, "name", f"job_{index}")
        lock_timeout = getattr(self.remote_info, "lock_timeout", None)
        try:
            lock = _acquire_ssh_call_lock_or_raise(
                self.remote_info.sys_name,
                lock_timeout,
                f"job.start() for job {index + 1} ({job_name})",
            )
            try:
                start_wall_time = time.time()
                job.start(
                    resources=self.remote_info.resources,
                    system_name=self.remote_info.sys_name,
                    header_extra=getattr(self.remote_info, "header_extra", []),
                    exact_fit=getattr(self.remote_info, "exact_fit", True),
                    partial_node=getattr(self.remote_info, "partial_node", False),
                )
                job._alomancy_lock_timeout = lock_timeout
            finally:
                lock.release()
            logger.debug("Job %d submitted to queue: %s", index + 1, job_name)

            result, stdout, stderr = job.get_results(
                timeout=self.remote_info.timeout,
                check_interval=getattr(self.remote_info, "check_interval", 10),
            )
            logger.info("Job %d completed successfully.", index + 1)
            # A job "succeeding" only means the remote function returned
            # without raising -- it can still have logged warnings (e.g. a
            # near-total silent prediction-failure rate, previously
            # invisible because this stdout/stderr was captured into
            # _stdout/_stderr and immediately discarded) to its own
            # stdout/stderr. Log it at DEBUG, matching the failure path
            # below, so results/alomancy.log (which always captures DEBUG
            # regardless of console verbosity) has it for postmortem.
            if stdout:
                logger.debug("Job %d stdout:\n%s", index + 1, stdout)
            if stderr:
                logger.debug("Job %d stderr:\n%s", index + 1, stderr)
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
            logger.debug("Job %d failure traceback:", index + 1, exc_info=exc)
            self._salvage_partial_output(index, job)
            return index, None

    @staticmethod
    def _salvage_partial_output(index: int, job: ExPyRe) -> None:
        """Best-effort recovery of a died/failed job's on-disk output.

        ExPyRe's own output_files stage-out (the "if self.status ==
        'succeeded'" block in expyre's get_results()) only runs for jobs
        that finish cleanly. A job that dies -- e.g. ExPyReJobDiedError, a
        hard crash (segfault, missing GPU driver, OOM-kill) with no
        _succeeded/_error sentinel ever written -- never reaches that code,
        so its output is discarded even when the data genuinely exists:
        get_remotes() (called by sync_remote_results_status() on every
        get_results() polling iteration while remote status isn't yet
        'done', including the final iteration where it transitions to
        'done') rsyncs the job's *entire* remote stage directory back into
        its local stage_dir regardless of eventual success -- unconditional,
        not filtered by output_files. Whatever a crashing remote process
        managed to write before dying (e.g. run_md's MD trajectory, flushed
        to disk on every snapshot) is therefore usually already sitting in
        job.stage_dir; this just performs the same local-to-cwd copy
        ExPyRe's own succeeded path would have done (ExPyRe._copy, reusing
        the same output_files list ExPyRe itself already recorded in
        job.stage_dir/_expyre_output_files at job construction time), for
        whichever output files actually exist.

        This matters most for structure_generation: a hard crash mid-MD-run
        (e.g. a numerically unstable structure from a gap in the
        committee's PES -- exactly the kind of structure active learning
        most needs) would otherwise silently drop that run's entire
        trajectory, discarding the most informative candidate along with
        it. Never raises: a failed salvage attempt (nothing was written
        yet, or the glob doesn't match) is logged at DEBUG and ignored, so
        it can never mask or replace the original job failure above.
        """
        marker = job.stage_dir / "_expyre_output_files"
        if not marker.exists():
            return
        output_files = [
            line.strip() for line in marker.read_text().splitlines() if line.strip()
        ]
        for out_file in output_files:
            try:
                ExPyRe._copy(job.stage_dir, Path.cwd(), out_file)
                logger.info(
                    "Job %d died/failed but salvaged partial output: %s",
                    index + 1,
                    out_file,
                )
            except Exception as copy_exc:
                logger.debug(
                    "Job %d: nothing to salvage for output %s (%s).",
                    index + 1,
                    out_file,
                    copy_exc,
                )

    def run_all_jobs_bounded(self) -> list[Any]:
        """Start and wait for all submitted jobs, keeping at most
        remote_info.max_concurrent_jobs started at once. The instant one
        finishes (success or failure), the next pending job is started to
        fill its slot -- ThreadPoolExecutor provides this rolling-window
        scheduling for free. Returned results stay index-aligned with the
        job_configs passed to submit_multiple_jobs, regardless of the order
        jobs actually complete in. Every individual ssh-invoking call (job
        start, and each job's own status/result sync while being monitored)
        is serialized per HPC host regardless of max_concurrent_jobs -- see
        _get_ssh_call_lock -- but the surrounding wait/poll cadence keeps
        real concurrency."""
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
