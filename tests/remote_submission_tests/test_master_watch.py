"""The master watch's reopen backoff (remote_submission/executor.py
check_ssh_masters): a host whose login needs a password or OTP must not
get a failed non-interactive login every 30 s while nobody is there."""

import pytest

from alomancy.remote_submission import executor, progress
from alomancy.utils import ssh_master


@pytest.fixture
def watched(monkeypatch, tmp_path):
    """One watched host whose master is down; counts reopen attempts."""
    masters = ssh_master.RunMasters()
    masters.configure(tmp_path, {"raven_cpu": "raven"})
    monkeypatch.setattr(ssh_master, "MASTERS", masters)
    monkeypatch.setattr(ssh_master, "external_calls_disabled", lambda: False)
    monkeypatch.setattr(executor, "_REOPEN_BACKOFF", {})
    state = {"alive": False, "reopens": 0, "reopen_works": False}

    def master_alive(host, template):
        return state["alive"]

    def open_master(host, template, interactive=True):
        assert interactive is False
        state["reopens"] += 1
        state["alive"] = state["reopen_works"]
        return state["alive"]

    monkeypatch.setattr(ssh_master, "master_alive", master_alive)
    monkeypatch.setattr(ssh_master, "open_master", open_master)
    monkeypatch.setattr(executor, "_note_ssh_auth_lost", lambda system: None)
    monkeypatch.setattr(executor, "_note_ssh_auth_restored", lambda system: None)
    return state


@pytest.mark.unit
def test_failed_reopens_back_off(watched):
    interval = progress.SSH_WATCH_INTERVAL
    # One watch tick every interval for 10 minutes.
    for tick in range(int(600 / interval) + 1):
        executor.check_ssh_masters(now=tick * interval)
    # Attempts at 0, 30, 90, 210, 450 s (delays 30, 60, 120, 240...), not 21.
    assert watched["reopens"] == 5


@pytest.mark.unit
def test_backoff_is_capped_and_reset_by_a_live_master(watched):
    for tick in range(400):  # well past the cap
        executor.check_ssh_masters(now=tick * progress.SSH_WATCH_INTERVAL)
    _, delay = executor._REOPEN_BACKOFF["raven"]
    assert delay == executor.REOPEN_BACKOFF_MAX
    # A login from the panel brings the master back: seen at once (the
    # liveness check is local and runs every tick), and the backoff resets.
    watched["alive"] = True
    executor.check_ssh_masters(now=400 * progress.SSH_WATCH_INTERVAL)
    assert "raven" not in executor._REOPEN_BACKOFF


@pytest.mark.unit
def test_a_key_login_host_reopens_on_the_first_tick(watched):
    watched["reopen_works"] = True
    executor.check_ssh_masters(now=0.0)
    assert watched["reopens"] == 1 and watched["alive"]
    assert "raven" not in executor._REOPEN_BACKOFF
