"""Unit tests for alomancy.cli.status."""

import json
import os
import subprocess
import sys
import time

import pytest
from rich.console import Console

import alomancy
from alomancy.cli.status import show_status
from alomancy.remote_submission.progress import JobTracker

_LOCK_HOLDER = """
import fcntl, os, socket, time
fh = open(os.environ["ALOMANCY_TEST_LOCK"], "a+")
fcntl.flock(fh, fcntl.LOCK_EX)
fh.seek(0); fh.truncate()
fh.write(f"pid={os.getpid()} host={socket.gethostname()}\\n"); fh.flush()
print("ready", flush=True)
time.sleep(600)
"""


def _run_dir(tmp_path, snapshot=None):
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "results" / "run_config.yaml").write_text("general: {}\n")
    (run_dir / ".expyre").mkdir()
    if snapshot is not None:
        (run_dir / "results" / "remote_status.json").write_text(json.dumps(snapshot))
    return run_dir


def _snapshot(updated_at=None):
    tracker = JobTracker()
    batch = tracker.register(
        "high_accuracy_evaluation", [("go_0", "raven"), ("go_1", "raven")], loop=0
    )
    tracker.submitted(batch, 0, "31549232", "id0")
    tracker.synced("raven", {"id0": "queued"})
    return tracker.snapshot(updated_at)


def _output(run_dir, **kwargs) -> tuple[int, str]:
    console = Console(record=True, width=100, color_system=None)
    status = show_status(run_dir, console=console, **kwargs)
    return status, console.export_text()


@pytest.fixture
def live_run(tmp_path):
    def _start(run_dir):
        env = {
            **os.environ,
            "ALOMANCY_TEST_LOCK": str(run_dir / ".expyre" / "alomancy.lock"),
        }
        proc = subprocess.Popen(
            [sys.executable, "-c", _LOCK_HOLDER, "alomancy-test"],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        procs.append(proc)

    procs: list[subprocess.Popen] = []
    yield _start
    for proc in procs:
        proc.kill()
        proc.wait()


@pytest.mark.unit
def test_shows_the_panel_of_a_live_run(tmp_path, live_run):
    run_dir = _run_dir(tmp_path, _snapshot())
    live_run(run_dir)

    status, text = _output(run_dir)
    assert status == 0
    assert "high_accuracy_evaluation · loop 0 ── 2 jobs" in text
    assert "raven" in text
    assert "not running" not in text
    assert "stale" not in text


@pytest.mark.unit
def test_says_when_the_run_is_not_running(tmp_path):
    run_dir = _run_dir(tmp_path, _snapshot())
    status, text = _output(run_dir)
    assert status == 0
    assert "run not running (last update" in text


@pytest.mark.unit
def test_flags_a_stale_status_file(tmp_path, live_run):
    run_dir = _run_dir(tmp_path, _snapshot(time.time() - 120))
    live_run(run_dir)
    _, text = _output(run_dir)
    assert "status file stale" in text


@pytest.mark.unit
def test_no_status_file_yet(tmp_path):
    run_dir = _run_dir(tmp_path)
    status, text = _output(run_dir)
    assert status == 0
    assert "No remote jobs reported yet (the run is not running)." in text


@pytest.mark.unit
def test_outside_a_run_directory(tmp_path, monkeypatch):
    seeded = tmp_path / ".expyre"
    seeded.mkdir()
    (seeded / "config.json").write_text("{}")
    monkeypatch.setattr(alomancy, "_SEEDED_EXPYRE_ROOT", seeded)

    status, text = _output(tmp_path)
    assert status == 2
    assert "Run 'alomancy status' from the folder" in " ".join(text.split())
    assert not seeded.exists()
