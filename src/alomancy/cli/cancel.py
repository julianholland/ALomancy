"""`alomancy cancel`: stop a run, cancel its remote jobs and, after a
second confirmation, delete its ExPyRe folders locally and remotely.

Run from the run's directory (where `alomancy run` was started). The run's
process is found through the lock it holds on `.expyre/alomancy.lock`
(remote_submission/executor.py writes `pid=... host=...` into it), and its
jobs through the run's own ExPyRe job database (`.expyre/jobs.db`).
Cancelling goes through ExPyRe itself (the call `xpr rm -c` makes), but
ExPyRe swallows a failed cancel, so every job is then checked against the
scheduler until it is gone; only jobs verified gone are marked failed, and
the folders are offered for deletion only when nothing is still alive.
"""

import contextlib
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import socket
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alomancy.cli.nuke import wipe_local_expyre
from alomancy.utils.remote_ssh import run_ssh_command

LOCK_NAME = "alomancy.lock"
ONGOING_STATUSES = ("created", "submitted", "started")
# ExPyRe scheduler statuses (expyre/schedulers/*.py) of a job that still
# holds or may still take a node. "other" (e.g. Slurm's CANCELLED, listed by
# squeue for a few minutes after scancel) and "done" (no longer listed)
# count as stopped.
ALIVE_STATUSES = frozenset({"queued", "held", "running"})
UNVERIFIED = "unverified"
STOPPED = "stopped"
_SSH_TIMEOUT = 60.0
_CANCEL_COMMANDS = {"slurm": "scancel", "pbs": "qdel", "sge": "qdel"}


class NotARunDirectoryError(RuntimeError):
    """The directory `alomancy cancel` was run in is not an alomancy run."""


@dataclass
class RunProcess:
    """The process holding a run's lock, as recorded in the lock file."""

    pid: int | None
    host: str | None


@dataclass
class CancelReport:
    """Final state of each cancelled job: STOPPED, UNVERIFIED, or the
    scheduler status it was still in (one of ALIVE_STATUSES)."""

    final: dict[str, str] = field(default_factory=dict)  # job id -> state
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def all_stopped(self) -> bool:
        return all(state == STOPPED for state in self.final.values())

    def not_stopped(self) -> dict[str, str]:
        return {i: s for i, s in self.final.items() if s != STOPPED}


def is_run_directory(run_dir: Path) -> bool:
    """True if `run_dir` is where an `alomancy run` was started: it has the
    run's saved config, or the lock written at its first remote job. A
    `.expyre/jobs.db` alone proves nothing, since importing alomancy seeds
    `.expyre/` (and expyre then creates an empty jobs.db) in any directory."""
    return (run_dir / "results" / "run_config.yaml").exists() or (
        run_dir / ".expyre" / LOCK_NAME
    ).exists()


def _parse_lock_record(text: str) -> RunProcess:
    pid = re.search(r"\bpid=(\d+)", text)
    host = re.search(r"\bhost=(\S+)", text)
    return RunProcess(
        pid=int(pid.group(1)) if pid else None,
        host=host.group(1) if host else None,
    )


def _lock_is_free(expyre_dir: Path) -> bool:
    lock = expyre_dir / LOCK_NAME
    if not lock.exists():
        return True
    # flock locks belong to an open file description, so probing with our
    # own handle never disturbs the run's lock.
    with open(lock, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
    return True


def find_run_process(expyre_dir: Path) -> RunProcess | None:
    """The live process holding the run's lock, or None if no process
    holds it (the run is not running, or never started a remote job)."""
    if _lock_is_free(expyre_dir):
        return None
    return _parse_lock_record((expyre_dir / LOCK_NAME).read_text())


def stop_run_process(
    proc: RunProcess,
    expyre_dir: Path,
    *,
    grace: float = 15.0,
    poll: float = 0.2,
) -> tuple[bool, str]:
    """SIGTERM the run's process, SIGKILL it if it outlives `grace` seconds.
    Success means the run's lock is free again (the kernel releases it when
    the process dies, whatever killed it). Returns (stopped, message)."""
    if proc.pid is None:
        return False, (
            f"{expyre_dir / LOCK_NAME} is locked but records no pid; stop the "
            "run's process by hand."
        )
    this_host = socket.gethostname()
    if proc.host is not None and proc.host != this_host:
        return False, (
            f"The run is running on {proc.host} (pid {proc.pid}), not on "
            f"{this_host}; run 'alomancy cancel' there."
        )
    cmdline = Path(f"/proc/{proc.pid}/cmdline")
    if cmdline.exists() and b"alomancy" not in cmdline.read_bytes():
        return False, (
            f"pid {proc.pid} is not an alomancy process (the pid was reused?); "
            f"not signalling it. Check who holds {expyre_dir / LOCK_NAME}."
        )

    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        with contextlib.suppress(ProcessLookupError):
            os.kill(proc.pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if _lock_is_free(expyre_dir):
                return True, f"Stopped the run (pid {proc.pid}, {sig.name})."
            time.sleep(poll)
    return False, f"pid {proc.pid} still holds {expyre_dir / LOCK_NAME}."


def ongoing_jobs(db: Any) -> list[dict[str, Any]]:
    """Jobs in the run's ExPyRe DB that may still be queued or running
    (what `xpr ls -s created,submitted,started` lists)."""
    return list(db.jobs(status=list(ONGOING_STATUSES), readable=True))


def confirm_stopped(
    scheduler: Any,
    remote_ids: list[str],
    *,
    rounds: int = 6,
    interval: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Ask the scheduler until every job is gone, re-cancelling survivors
    between rounds. Returns each id's final state: STOPPED, the alive
    status it is still in, or UNVERIFIED if the scheduler couldn't be asked
    (never assumed stopped)."""
    final: dict[str, str] = {}
    pending = list(remote_ids)
    statuses: dict[str, str] = {}
    for attempt in range(rounds):
        if attempt:
            sleep(interval)
        try:
            statuses = scheduler.status(pending)
        except Exception:
            final.update(dict.fromkeys(pending, UNVERIFIED))
            return final
        for remote_id in pending:
            if statuses.get(remote_id, "done") not in ALIVE_STATUSES:
                final[remote_id] = STOPPED
        pending = [i for i in pending if i not in final]
        if not pending:
            return final
        with contextlib.suppress(Exception):
            scheduler.cancel(pending)
    final.update({i: statuses[i] for i in pending})
    return final


def cancel_jobs(
    records: list[dict[str, Any]],
    *,
    systems: dict[str, Any] | None = None,
    db: Any = None,
    rounds: int = 6,
    interval: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> CancelReport:
    """Cancel the jobs through ExPyRe, verify each is gone from its
    scheduler, and mark only the verified ones failed in the DB (a restart
    then resubmits them rather than waiting on a cancelled job). Survivors
    and unverifiable jobs keep their status, so the DB still matches the
    cluster."""
    from expyre import config as expyre_config
    from expyre.func import ExPyRe

    systems = expyre_config.systems if systems is None else systems
    db = expyre_config.db if db is None else db

    for job in ExPyRe.from_jobsdb(records):
        job.cancel()

    report = CancelReport(records={r["id"]: r for r in records})
    by_system: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["remote_id"] is None:
            # Never submitted: nothing exists on the cluster.
            report.final[record["id"]] = STOPPED
        elif record["system"] not in systems:
            report.final[record["id"]] = UNVERIFIED
        else:
            by_system.setdefault(record["system"], []).append(record)

    for system_name, system_records in by_system.items():
        remote_ids = [r["remote_id"] for r in system_records]
        states = confirm_stopped(
            systems[system_name].scheduler,
            remote_ids,
            rounds=rounds,
            interval=interval,
            sleep=sleep,
        )
        for record in system_records:
            report.final[record["id"]] = states[record["remote_id"]]

    for job_id, state in report.final.items():
        if state == STOPPED:
            db.update(job_id, status="failed")
    return report


def _run_suffix(local_stage_dir: Path) -> str:
    """The run-specific part of every system's remote rundir, as expyre
    builds it (expyre/config.py: `rundir_extra`), minus the hostname
    prefix, which depends on the environment of the shell that started the
    run."""
    base = (
        local_stage_dir.parent
        if local_stage_dir.name in (".expyre", "_expyre")
        else local_stage_dir
    )
    return str(base).replace("/", "_")


def remote_run_dir_glob(rundir: str, local_stage_dir: Path) -> str:
    """Shell glob matching this run's folder under a system's `rundir`:
    `<rundir>/<any host>-<run path with / as _>`."""
    return (
        shlex.quote(rundir.rstrip("/") + "/")
        + "*"
        + shlex.quote("-" + _run_suffix(local_stage_dir))
    )


_HPC_LOG_LINE = re.compile(r"HPC: (\S+), Job: ")


def hpcs_used(run_dir: Path) -> set[str]:
    """The HPC systems this run submitted jobs to, from the line
    configs/remote_info.py logs for every remote job ("HPC: <name>, Job:
    <phase>"; the file log is always at DEBUG level). Empty if the run has no
    log."""
    log = run_dir / "results" / "alomancy.log"
    if not log.exists():
        return set()
    with open(log, errors="replace") as fh:
        return {m.group(1) for line in fh if (m := _HPC_LOG_LINE.search(line))}


def remote_targets(expyre_dir: Path, systems: set[str]) -> dict[str, set[str]]:
    """{ssh host: {rundir, ...}} for the named systems of the run's
    `.expyre/config.json` (deduplicated: systems often share host and
    rundir). A host-less system runs locally and has no remote folder."""
    config = json.loads((expyre_dir / "config.json").read_text())
    targets: dict[str, set[str]] = {}
    for name, system in config.get("systems", {}).items():
        host = system.get("host")
        if name not in systems:
            continue
        if host is None:
            continue
        targets.setdefault(host, set()).add(system.get("rundir") or "run_expyre")
    return targets


def remote_run_dirs(
    expyre_dir: Path,
    systems: set[str],
    *,
    ssh: Callable[[str, str, float], tuple[bool, str, str]] = run_ssh_command,
) -> tuple[dict[str, list[str]], list[str]]:
    """This run's existing folders on the hosts of `systems` (the HPCs it
    submitted to), and the hosts that couldn't be listed."""
    found: dict[str, list[str]] = {}
    unreachable: list[str] = []
    suffix = "-" + _run_suffix(expyre_dir)
    for host, rundirs in sorted(remote_targets(expyre_dir, systems).items()):
        globs = " ".join(remote_run_dir_glob(r, expyre_dir) for r in sorted(rundirs))
        ok, stdout, _ = ssh(host, f"ls -d -- {globs} 2>/dev/null; true", _SSH_TIMEOUT)
        if not ok:
            unreachable.append(host)
            continue
        paths = [p for p in stdout.split() if p.endswith(suffix)]
        if paths:
            found[host] = paths
    return found, unreachable


def _ask(question: str, input_fn: Callable[[str], str]) -> bool:
    return input_fn(f"{question} (y/n) ").strip().lower() == "y"


def require_run_directory(run_dir: Path, command: str) -> None:
    """Raise NotARunDirectoryError unless `run_dir` is an alomancy run,
    first removing the `.expyre/` that importing alomancy seeded there in
    this process (so a command run in the wrong place leaves nothing
    behind; a pre-existing `.expyre/` is never touched)."""
    import alomancy

    if is_run_directory(run_dir):
        return
    seeded = alomancy._SEEDED_EXPYRE_ROOT
    if seeded is not None and seeded.resolve() == (run_dir / ".expyre").resolve():
        shutil.rmtree(seeded, ignore_errors=True)
    raise NotARunDirectoryError(
        f"{run_dir} is not an alomancy run directory (no "
        f"results/run_config.yaml or .expyre/{LOCK_NAME}). Run 'alomancy "
        f"{command}' from the folder you ran 'alomancy run' in."
    )


def _check_run_directory(run_dir: Path) -> Path:
    """The run's ExPyRe directory, after checking `run_dir` is a run and
    expyre resolved this run's own job DB. Changes nothing except removing
    an `.expyre/` this process seeded in a directory that isn't a run."""
    from expyre import config as expyre_config

    require_run_directory(run_dir, "cancel")
    expyre_dir = run_dir / ".expyre"
    if not expyre_dir.is_dir():
        raise NotARunDirectoryError(
            f"{run_dir} has no .expyre/ directory: this run never submitted "
            "a remote job, so there is nothing to cancel. Stop its process "
            "by hand."
        )
    resolved = expyre_config.local_stage_dir
    if resolved is None or Path(resolved).resolve() != expyre_dir.resolve():
        raise NotARunDirectoryError(
            f"expyre is using {resolved}, not this run's {expyre_dir} (is "
            "EXPYRE_ROOT set?); refusing to cancel another run's jobs."
        )
    return expyre_dir


def _print_jobs(records: Iterable[dict[str, Any]], final: dict[str, str]) -> None:
    for record in records:
        state = final.get(record["id"], record["status"])
        print(
            f"  {record['system'] or '-':<12} {record['remote_id'] or '-':<12} "
            f"{state:<11} {record['name']}"
        )


def _cancel_hints(report: CancelReport, expyre_dir: Path) -> list[str]:
    config = json.loads((expyre_dir / "config.json").read_text())
    systems = config.get("systems", {})
    ids_by_system: dict[str, list[str]] = {}
    for job_id in report.not_stopped():
        record = report.records[job_id]
        ids_by_system.setdefault(record["system"], []).append(record["remote_id"])
    hints = []
    for system_name, ids in ids_by_system.items():
        system = systems.get(system_name, {})
        command = _CANCEL_COMMANDS.get(system.get("scheduler", ""), "<cancel>")
        target = f"ssh {system['host']} " if system.get("host") else ""
        hints.append(f"  {target}{command} {' '.join(ids)}")
    return hints


def cancel_run(
    run_dir: Path,
    *,
    assume_yes: bool = False,
    input_fn: Callable[[str], str] = input,
    ssh: Callable[[str, str, float], tuple[bool, str, str]] = run_ssh_command,
    grace: float = 15.0,
    rounds: int = 6,
    interval: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """`alomancy cancel`. Returns the exit status: 0 done (or declined),
    1 something could not be stopped, 2 not a run directory."""
    try:
        expyre_dir = _check_run_directory(run_dir)
    except NotARunDirectoryError as exc:
        print(f"alomancy cancel: {exc}")
        return 2

    from expyre import config as expyre_config

    proc = find_run_process(expyre_dir)
    records = ongoing_jobs(expyre_config.db)

    print(f"Run: {run_dir}")
    if proc is None:
        print("  No live alomancy process holds this run's lock.")
    else:
        print(f"  Live process: pid {proc.pid} on {proc.host}")
    print(f"  Queued/running jobs in {expyre_dir / 'jobs.db'}: {len(records)}")
    _print_jobs(records, {})

    if (
        (proc is not None or records)
        and not assume_yes
        and not _ask(f"Stop this run and cancel these {len(records)} job(s)?", input_fn)
    ):
        print("Aborting; nothing changed.")
        return 0

    if proc is not None:
        stopped, message = stop_run_process(proc, expyre_dir, grace=grace)
        print(message)
        if not stopped:
            print("Not cancelling jobs while the run is alive: it would resubmit them.")
            return 1

    if records:
        print("Cancelling and checking each job with its scheduler...")
        report = cancel_jobs(
            records,
            db=expyre_config.db,
            rounds=rounds,
            interval=interval,
            sleep=sleep,
        )
        _print_jobs(records, report.final)
        if not report.all_stopped:
            print(
                f"{len(report.not_stopped())} job(s) could not be confirmed "
                "stopped; they keep their status in jobs.db. Cancel them with:"
            )
            for hint in _cancel_hints(report, expyre_dir):
                print(hint)
            print("Not offering to delete folders while jobs may still be running.")
            return 1
        print(f"All {len(records)} job(s) confirmed stopped and marked failed.")

    used = hpcs_used(run_dir)
    configured = set(json.loads((expyre_dir / "config.json").read_text())["systems"])
    remote, unreachable = remote_run_dirs(expyre_dir, used & configured, ssh=ssh)
    stage_dirs = [p for p in expyre_dir.iterdir() if p.is_dir()]
    print("This run's ExPyRe folders:")
    print(
        f"  local:  {expyre_dir} ({len(stage_dirs)} stage folder(s) and "
        "jobs.db; config.json is kept)"
    )
    for host, paths in remote.items():
        for path in paths:
            print(f"  {host}: {path}")
    for host in unreachable:
        print(f"  {host}: could not be listed over ssh (left alone)")
    if not used:
        print(
            f"  remote: none ({run_dir / 'results' / 'alomancy.log'} names no "
            "HPC jobs were submitted to)"
        )
    for name in sorted(used - configured):
        print(f"  {name}: used by this run but not in .expyre/config.json (left alone)")
    if not _ask("Delete these folders?", input_fn):
        print("Folders kept.")
        return 0

    wipe_local_expyre(expyre_dir)
    print(f"Deleted the job state in {expyre_dir}.")
    status = 0
    for host, paths in remote.items():
        quoted = " ".join(shlex.quote(p) for p in paths)
        ok, _, stderr = ssh(host, f"rm -rf -- {quoted}", _SSH_TIMEOUT)
        if ok:
            print(f"Deleted {len(paths)} folder(s) on {host}.")
        else:
            print(f"Could not delete on {host}: {stderr.strip()}")
            status = 1
    return status
