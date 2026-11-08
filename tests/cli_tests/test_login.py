"""Unit tests for alomancy.cli.login (logging in again to a run's HPC)."""

import json
import socket

import pytest
from rich.console import Console

from alomancy.cli.login import login, lost_systems, system_host
from alomancy.remote_submission.progress import JobTracker
from alomancy.utils.ssh_master import control_path_template


def _run_dir(tmp_path, auth_lost=(), lock_host=None):
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "results" / "run_config.yaml").write_text("general: {}\n")
    expyre = run_dir / ".expyre"
    expyre.mkdir()
    (expyre / "config.json").write_text(
        json.dumps(
            {
                "systems": {
                    "raven": {"host": "raven.mpcdf"},
                    "raven_gpu": {"host": "raven.mpcdf"},
                    "fhi-raccoon": {"host": "fhi-raccoon"},
                }
            }
        )
    )
    if lock_host is not None:
        (expyre / "alomancy.lock").write_text(f"pid=1 host={lock_host}\n")
    tracker = JobTracker()
    for system in auth_lost:
        tracker.mark_auth_lost(system, system)
    (run_dir / "results" / "remote_status.json").write_text(
        json.dumps(tracker.snapshot())
    )
    return run_dir


class _Opener:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, host, template, *, interactive):
        self.calls.append((host, template, interactive))
        return self.ok


def _login(run_dir, system=None, opener=None):
    console = Console(record=True, width=120, color_system=None)
    status = login(run_dir, system, console=console, opener=opener or _Opener())
    return status, console.export_text()


@pytest.mark.unit
def test_outside_a_run_directory(tmp_path):
    opener = _Opener()
    status, text = _login(tmp_path, opener=opener)
    assert status == 2
    assert "alomancy login" in text
    assert opener.calls == []


@pytest.mark.unit
def test_logs_in_to_every_hpc_the_run_reports_lost(tmp_path):
    run_dir = _run_dir(tmp_path, auth_lost=("raven", "raven_gpu", "fhi-raccoon"))
    assert lost_systems(run_dir) == ["fhi-raccoon", "raven", "raven_gpu"]
    opener = _Opener()
    status, text = _login(run_dir, opener=opener)
    assert status == 0
    template = control_path_template(run_dir)
    # raven and raven_gpu share a host: one login.
    assert opener.calls == [
        ("fhi-raccoon", template, True),
        ("raven.mpcdf", template, True),
    ]
    assert "raven, raven_gpu (raven.mpcdf)" in text


@pytest.mark.unit
def test_nothing_to_do_without_a_lost_login(tmp_path):
    opener = _Opener()
    status, text = _login(_run_dir(tmp_path), opener=opener)
    assert status == 0
    assert "No HPC of this run needs a login" in text
    assert opener.calls == []


@pytest.mark.unit
def test_a_named_hpc_and_a_failed_login(tmp_path):
    run_dir = _run_dir(tmp_path)
    opener = _Opener(ok=False)
    status, text = _login(run_dir, "fhi-raccoon", opener=opener)
    assert status == 1
    assert [c[0] for c in opener.calls] == ["fhi-raccoon"]
    assert "Login to fhi-raccoon failed" in text


@pytest.mark.unit
def test_an_unknown_hpc(tmp_path):
    status, text = _login(_run_dir(tmp_path), "nowhere")
    assert status == 1
    assert "Unknown HPC 'nowhere'" in text
    assert system_host(_run_dir(tmp_path / "x"), "raven") == "raven.mpcdf"


@pytest.mark.unit
def test_refuses_on_another_machine(tmp_path):
    run_dir = _run_dir(tmp_path, auth_lost=("raven",), lock_host="elsewhere")
    assert socket.gethostname() != "elsewhere"
    opener = _Opener()
    status, text = _login(run_dir, opener=opener)
    assert status == 1
    assert "runs on elsewhere" in text
    assert opener.calls == []
