"""Unit tests for alomancy.cli.cancel."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from expyre import config as expyre_config
from expyre.jobsdb import JobsDB
from expyre.schedulers.base import Scheduler
from expyre.system import System

import alomancy
from alomancy.cli.cancel import (
    LOCK_NAME,
    STOPPED,
    UNVERIFIED,
    cancel_jobs,
    cancel_run,
    confirm_stopped,
    find_run_process,
    hpcs_used,
    is_run_directory,
    remote_run_dir_glob,
    remote_run_dirs,
    remote_targets,
    stop_run_process,
)
from alomancy.cli.nuke import wipe_local_expyre

_LOCK_HOLDER = """
import fcntl, os, socket, sys, time
fh = open(os.environ["ALOMANCY_TEST_LOCK"], "a+")
fcntl.flock(fh, fcntl.LOCK_EX)
fh.seek(0); fh.truncate()
fh.write(f"pid={os.getpid()} host={socket.gethostname()} started=now\\n")
fh.flush()
print("ready", flush=True)
time.sleep(600)
"""


class ScriptedScheduler(Scheduler):
    """A scheduler whose status() replays a script (one entry per call: a
    {remote_id: status} dict, or an exception to raise); records cancels."""

    def __init__(self, script: list[Any]):
        super().__init__(host=None)
        self.script = list(script)
        self.cancelled: list[list[str]] = []

    def status(self, remote_ids, verbose=False):
        step = self.script.pop(0) if self.script else {}
        if isinstance(step, Exception):
            raise step
        return {i: step.get(i, "done") for i in remote_ids}

    def cancel(self, remote_ids, verbose=False):
        if isinstance(remote_ids, str):
            remote_ids = [remote_ids]
        self.cancelled.append(list(remote_ids))


def _system(scheduler: Scheduler) -> System:
    return System(host=None, partitions=None, scheduler=lambda host: scheduler)


def _make_run(tmp_path: Path, systems: dict[str, Any] | None = None) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "results" / "run_config.yaml").write_text("general: {}\n")
    expyre_dir = run_dir / ".expyre"
    expyre_dir.mkdir()
    config = {"systems": systems or {}}
    (expyre_dir / "config.json").write_text(json.dumps(config))
    return run_dir


def _add_job(db: JobsDB, tmp: Path, name: str, status: str, remote_id: str | None):
    stage = tmp / ".expyre" / f"run_{name}"
    stage.mkdir()
    db.add(name, name, str(stage), status=status, system="sys", remote_id=remote_id)


def _status(db: JobsDB, job_id: str) -> str:
    return next(j for j in db.jobs() if j["id"] == job_id)["status"]


def _start_lock_holder(lock: Path, marker: bool = True) -> subprocess.Popen:
    # The lock path goes in the environment, not argv: tmp paths carry the
    # test's name, which may itself contain "alomancy".
    args = [sys.executable, "-c", _LOCK_HOLDER]
    if marker:
        args.append("alomancy-test")  # an alomancy run's cmdline names alomancy
    env = {**os.environ, "ALOMANCY_TEST_LOCK": str(lock)}
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True, env=env)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


@pytest.fixture
def expyre_state(monkeypatch, tmp_path):
    """Point expyre's module state at a fake run with a scripted scheduler."""

    def _install(run_dir: Path, scheduler: Scheduler) -> JobsDB:
        db = JobsDB(run_dir / ".expyre" / "jobs.db")
        monkeypatch.setattr(expyre_config, "db", db)
        monkeypatch.setattr(expyre_config, "systems", {"sys": _system(scheduler)})
        monkeypatch.setattr(expyre_config, "local_stage_dir", run_dir / ".expyre")
        return db

    return _install


@pytest.fixture
def children():
    procs: list[subprocess.Popen] = []
    yield procs
    for proc in procs:
        proc.kill()
        proc.wait()


def _no_input(prompt: str) -> str:
    raise AssertionError(f"unexpected prompt: {prompt}")


def _answers(*answers: str):
    queue = list(answers)
    asked: list[str] = []

    def _input(prompt: str) -> str:
        asked.append(prompt)
        return queue.pop(0)

    _input.asked = asked  # type: ignore[attr-defined]
    return _input


# --- run process ---------------------------------------------------------


@pytest.mark.unit
def test_free_lock_means_no_live_run(tmp_path):
    (tmp_path / LOCK_NAME).write_text("pid=1 host=x\n")
    assert find_run_process(tmp_path) is None
    assert find_run_process(tmp_path / "missing") is None


@pytest.mark.unit
def test_lock_holder_is_found_and_stopped(tmp_path, children):
    child = _start_lock_holder(tmp_path / LOCK_NAME)
    children.append(child)

    proc = find_run_process(tmp_path)
    assert proc is not None
    assert proc.pid == child.pid
    assert proc.host == socket.gethostname()

    stopped, message = stop_run_process(proc, tmp_path, grace=5.0)
    assert stopped, message
    assert find_run_process(tmp_path) is None
    assert child.wait(timeout=5) != 0


@pytest.mark.unit
def test_refuses_a_run_on_another_host(tmp_path, children):
    child = _start_lock_holder(tmp_path / LOCK_NAME)
    children.append(child)
    proc = find_run_process(tmp_path)
    assert proc is not None
    proc.host = "elsewhere"

    stopped, message = stop_run_process(proc, tmp_path, grace=0.5)
    assert not stopped
    assert "elsewhere" in message
    assert child.poll() is None


@pytest.mark.unit
@pytest.mark.skipif(
    not Path("/proc/self/cmdline").exists() or "alomancy" in sys.executable,
    reason="needs /proc and an interpreter path without 'alomancy'",
)
def test_refuses_a_pid_that_is_not_alomancy(tmp_path, children):
    child = _start_lock_holder(tmp_path / LOCK_NAME, marker=False)
    children.append(child)
    proc = find_run_process(tmp_path)
    assert proc is not None

    stopped, message = stop_run_process(proc, tmp_path, grace=0.5)
    assert not stopped
    assert "not an alomancy process" in message
    assert child.poll() is None


# --- cancelling and verifying -----------------------------------------------


@pytest.mark.unit
def test_confirm_stopped_recancels_until_gone():
    scheduler = ScriptedScheduler(
        [{"1": "queued", "2": "running"}, {"2": "running"}, {}]
    )
    sleeps: list[float] = []
    final = confirm_stopped(scheduler, ["1", "2"], interval=3.0, sleep=sleeps.append)

    assert final == {"1": STOPPED, "2": STOPPED}
    assert scheduler.cancelled == [["1", "2"], ["2"]]
    assert sleeps == [3.0, 3.0]


@pytest.mark.unit
def test_confirm_stopped_reports_survivors_and_unreachable_schedulers():
    stuck = ScriptedScheduler([{"1": "running"}] * 3)
    assert confirm_stopped(stuck, ["1"], rounds=3, sleep=lambda s: None) == {
        "1": "running"
    }

    broken = ScriptedScheduler([RuntimeError("ssh failed")])
    assert confirm_stopped(broken, ["1"], sleep=lambda s: None) == {"1": UNVERIFIED}


@pytest.mark.unit
def test_only_verified_jobs_are_marked_failed(tmp_path, expyre_state):
    run_dir = _make_run(tmp_path)
    scheduler = ScriptedScheduler([{"11": "running"}] * 2)
    db = expyre_state(run_dir, scheduler)
    _add_job(db, run_dir, "gone", "submitted", "10")
    _add_job(db, run_dir, "stuck", "started", "11")
    _add_job(db, run_dir, "unsent", "created", None)
    _add_job(db, run_dir, "finished", "succeeded", "12")
    records = list(db.jobs(status=["created", "submitted", "started"]))

    report = cancel_jobs(records, rounds=2, sleep=lambda s: None)

    assert report.final == {"gone": STOPPED, "stuck": "running", "unsent": STOPPED}
    assert not report.all_stopped
    assert _status(db, "gone") == "failed"
    assert _status(db, "unsent") == "failed"
    assert _status(db, "stuck") == "started"
    assert _status(db, "finished") == "succeeded"


# --- folders --------------------------------------------------------------


@pytest.mark.unit
def test_remote_glob_matches_expyre_run_folder_layout():
    local = Path(
        "/work/home/jholl/repos/ALomancy_paper/src/alomancy_paper/good_c/.expyre"
    )
    glob = remote_run_dir_glob("/ptmp/jholl/alomancy_expyre_cpu/", local)
    assert glob == (
        "/ptmp/jholl/alomancy_expyre_cpu/*"
        "-_work_home_jholl_repos_ALomancy_paper_src_alomancy_paper_good_c"
    )


@pytest.mark.unit
def test_remote_targets_dedupe_shared_rundirs_and_skip_local(tmp_path):
    run_dir = _make_run(
        tmp_path,
        {
            "cpu": {"host": "raven", "rundir": "/ptmp/a", "scheduler": "slurm"},
            "gpu": {"host": "raven", "rundir": "/ptmp/a", "scheduler": "slurm"},
            "other": {"host": "raven", "rundir": "/ptmp/b", "scheduler": "slurm"},
            "local": {"host": None, "scheduler": "local"},
        },
    )
    systems = {"cpu", "gpu", "other", "local"}
    assert remote_targets(run_dir / ".expyre", systems) == {
        "raven": {"/ptmp/a", "/ptmp/b"}
    }
    # Only systems the run used are searched.
    assert remote_targets(run_dir / ".expyre", {"other"}) == {"raven": {"/ptmp/b"}}
    assert remote_targets(run_dir / ".expyre", set()) == {}


@pytest.mark.unit
def test_hpcs_used_reads_the_run_log(tmp_path):
    run_dir = _make_run(tmp_path)
    assert hpcs_used(run_dir) == set()
    (run_dir / "results" / "alomancy.log").write_text(
        "2026-10-26 12:28:00 [DEBUG   ] alomancy.configs.remote_info: "
        "HPC: fhi-raccoon, Job: training\n"
        "2026-10-26 18:07:00 [DEBUG   ] alomancy.configs.remote_info: "
        "HPC: raven, Job: high_accuracy_evaluation\n"
        "2026-10-26 18:07:00 [INFO    ] something about HPC: notalog\n"
    )
    assert hpcs_used(run_dir) == {"fhi-raccoon", "raven"}


@pytest.mark.unit
def test_remote_run_dirs_keeps_only_this_runs_folders(tmp_path):
    run_dir = _make_run(tmp_path, {"cpu": {"host": "raven", "rundir": "/r"}})
    suffix = str(run_dir).replace("/", "_")

    def ssh(host, command, timeout):
        return True, f"/r/h-{suffix}\n/r/unrelated\n", ""

    found, unreachable = remote_run_dirs(run_dir / ".expyre", {"cpu"}, ssh=ssh)
    assert found == {"raven": [f"/r/h-{suffix}"]}
    assert unreachable == []

    found, unreachable = remote_run_dirs(
        run_dir / ".expyre", {"cpu"}, ssh=lambda h, c, t: (False, "", "no route")
    )
    assert found == {}
    assert unreachable == ["raven"]


@pytest.mark.unit
def test_wipe_keeps_config_and_lock(tmp_path):
    for name in ("config.json", LOCK_NAME, "jobs.db"):
        (tmp_path / name).write_text("x")
    (tmp_path / "run_job").mkdir()

    wipe_local_expyre(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "alomancy.lock",
        "config.json",
    ]


# --- outside a run --------------------------------------------------------


@pytest.mark.unit
def test_outside_a_run_errors_and_removes_the_seeded_expyre(tmp_path, monkeypatch):
    seeded = tmp_path / ".expyre"
    seeded.mkdir()
    (seeded / "config.json").write_text("{}")
    (seeded / "jobs.db").write_text("")
    monkeypatch.setattr(alomancy, "_SEEDED_EXPYRE_ROOT", seeded)

    assert not is_run_directory(tmp_path)
    assert cancel_run(tmp_path, input_fn=_no_input) == 2
    assert not seeded.exists()


@pytest.mark.unit
def test_outside_a_run_keeps_an_expyre_it_did_not_create(tmp_path, monkeypatch):
    existing = tmp_path / ".expyre"
    existing.mkdir()
    (existing / "config.json").write_text("{}")
    monkeypatch.setattr(alomancy, "_SEEDED_EXPYRE_ROOT", None)

    assert cancel_run(tmp_path, input_fn=_no_input) == 2
    assert (existing / "config.json").exists()


@pytest.mark.unit
def test_results_subfolder_is_not_a_run(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr(alomancy, "_SEEDED_EXPYRE_ROOT", None)
    assert is_run_directory(run_dir)
    assert cancel_run(run_dir / "results", input_fn=_no_input) == 2


@pytest.mark.unit
def test_refuses_when_expyre_resolved_another_directory(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr(expyre_config, "local_stage_dir", tmp_path / "other")
    assert cancel_run(run_dir, input_fn=_no_input) == 2


# --- the whole command ----------------------------------------------------


def _run_with_live_process(tmp_path, expyre_state, children, scheduler, log=True):
    run_dir = _make_run(
        tmp_path,
        {
            "sys": {"host": "raven", "rundir": "/r"},
            "unused": {"host": "other-hpc", "rundir": "/r"},
        },
    )
    if log:
        (run_dir / "results" / "alomancy.log").write_text(
            "[DEBUG   ] alomancy.configs.remote_info: HPC: sys, Job: training\n"
        )
    db = expyre_state(run_dir, scheduler)
    _add_job(db, run_dir, "go_0", "submitted", "10")
    child = _start_lock_holder(run_dir / ".expyre" / LOCK_NAME)
    children.append(child)
    return run_dir, db, child


class _Ssh:
    def __init__(self, run_dir: Path):
        self.remote = f"/r/h-{str(run_dir).replace('/', '_')}"
        self.commands: list[str] = []
        self.hosts: list[str] = []

    def __call__(self, host, command, timeout):
        self.commands.append(command)
        self.hosts.append(host)
        if command.startswith("ls"):
            return True, self.remote + "\n", ""
        return True, "", ""


@pytest.mark.unit
def test_declining_changes_nothing(tmp_path, expyre_state, children):
    run_dir, db, child = _run_with_live_process(
        tmp_path, expyre_state, children, ScriptedScheduler([])
    )
    assert cancel_run(run_dir, input_fn=_answers("n")) == 0
    assert child.poll() is None
    assert _status(db, "go_0") == "submitted"


@pytest.mark.unit
def test_cancel_stops_run_and_jobs_and_keeps_folders_on_no(
    tmp_path, expyre_state, children
):
    scheduler = ScriptedScheduler([{"10": "queued"}, {}])
    run_dir, db, child = _run_with_live_process(
        tmp_path, expyre_state, children, scheduler
    )
    ssh = _Ssh(run_dir)
    answers = _answers("y", "n")

    status = cancel_run(
        run_dir, input_fn=answers, ssh=ssh, grace=5.0, sleep=lambda s: None
    )

    assert status == 0
    assert child.wait(timeout=5) != 0
    assert _status(db, "go_0") == "failed"
    # ExPyRe's cancel, then one re-cancel while the job was still queued
    assert scheduler.cancelled == [["10"], ["10"]]
    assert len(answers.asked) == 2
    assert (run_dir / ".expyre" / "run_go_0").exists()
    assert not any(c.startswith("rm") for c in ssh.commands)


@pytest.mark.unit
def test_cancel_then_delete_wipes_local_and_remote(tmp_path, expyre_state, children):
    run_dir, _db, _child = _run_with_live_process(
        tmp_path, expyre_state, children, ScriptedScheduler([{}])
    )
    ssh = _Ssh(run_dir)

    status = cancel_run(
        run_dir, assume_yes=True, input_fn=_answers("y"), ssh=ssh, grace=5.0
    )

    assert status == 0
    assert sorted(p.name for p in (run_dir / ".expyre").iterdir()) == [
        "alomancy.lock",
        "config.json",
    ]
    assert ssh.commands[-1] == f"rm -rf -- {ssh.remote}"
    # Only the HPC named in the run's log is searched or touched.
    assert set(ssh.hosts) == {"raven"}


@pytest.mark.unit
def test_without_a_log_no_remote_folder_is_touched(tmp_path, expyre_state, children):
    run_dir, _db, _child = _run_with_live_process(
        tmp_path, expyre_state, children, ScriptedScheduler([{}]), log=False
    )
    ssh = _Ssh(run_dir)

    status = cancel_run(
        run_dir, assume_yes=True, input_fn=_answers("y"), ssh=ssh, grace=5.0
    )

    assert status == 0
    assert ssh.hosts == []
    assert sorted(p.name for p in (run_dir / ".expyre").iterdir()) == [
        "alomancy.lock",
        "config.json",
    ]


@pytest.mark.unit
def test_surviving_jobs_block_deletion(tmp_path, expyre_state, children):
    scheduler = ScriptedScheduler([{"10": "running"}] * 6)
    run_dir, db, _child = _run_with_live_process(
        tmp_path, expyre_state, children, scheduler
    )
    answers = _answers("y")

    status = cancel_run(
        run_dir, input_fn=answers, ssh=_Ssh(run_dir), grace=5.0, sleep=lambda s: None
    )

    assert status == 1
    assert len(answers.asked) == 1  # never asked to delete
    assert _status(db, "go_0") == "submitted"
    assert (run_dir / ".expyre" / "run_go_0").exists()
