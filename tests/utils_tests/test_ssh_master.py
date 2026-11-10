"""Unit tests for alomancy.utils.ssh_master (the run's own ssh masters) and
the matching alomancy-rsh options. Real ssh is never run: the commands go to
a stand-in `ssh` script on PATH that records its arguments."""

import os
from pathlib import Path

import pytest

from alomancy.utils import ssh_master
from alomancy.utils.rsh import build_argv
from alomancy.utils.ssh_master import (
    CONTROL_DIR_ENV,
    RUN_TAG_ENV,
    RunMasters,
    control_dir,
    control_path_template,
    master_alive,
    open_master,
    run_tag,
    start_run_masters,
)


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """An `ssh` on PATH that logs each call's arguments and answers
    `-O check` with the exit code in $FAKE_SSH_CHECK (default 0), and
    `-G` with a controlpath line."""
    log = tmp_path / "ssh_calls"
    script = tmp_path / "bin" / "ssh"
    script.parent.mkdir()
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {log}\n'
        'case " $* " in\n'
        '  *" -O check "*) exit "${FAKE_SSH_CHECK:-0}" ;;\n'
        f'  *" -G "*) echo "controlpath {tmp_path}/socket" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    return log


@pytest.mark.unit
def test_run_tag_is_stable_per_run_and_differs_between_runs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert run_tag(tmp_path / "a") == run_tag(tmp_path / "a" / ".." / "a")
    assert run_tag(tmp_path / "a") != run_tag(tmp_path / "b")
    assert len(run_tag(tmp_path / "a")) == 6


@pytest.mark.unit
def test_control_path_fits_a_unix_socket(tmp_path):
    template = control_path_template(tmp_path, base=Path("/home/u/.alomancy/cm"))
    assert template == f"/home/u/.alomancy/cm/{run_tag(tmp_path)}-%C"
    # %C expands to 40 characters, ssh adds a 17-character temporary suffix.
    assert len(template) - 2 + 40 + 17 < 104


@pytest.mark.unit
def test_a_long_home_falls_back_to_a_short_directory():
    long_base = Path("/") / ("x" * 60) / ".alomancy" / "cm"
    assert control_dir(long_base) == Path("/tmp") / f"alomancy-cm-{os.getuid()}"


@pytest.mark.unit
def test_rsh_uses_the_runs_socket_only_when_the_run_set_one(tmp_path):
    plain = build_argv(["host", "true"], env={})
    assert plain == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "host",
        "true",
    ]
    socket = tmp_path / "socket"
    socket.touch()
    owned = build_argv(
        ["host", "true"],
        env={CONTROL_DIR_ENV: "/home/u/.alomancy/cm", RUN_TAG_ENV: "abc123"},
        socket_lookup=lambda args, template: socket,
    )
    assert owned[-2:] == ["host", "true"]
    assert "ControlPath=/home/u/.alomancy/cm/abc123-%C" in owned
    assert "ControlMaster=no" in owned
    assert "BatchMode=yes" in owned


@pytest.mark.unit
def test_rsh_and_masters_agree_on_the_socket(tmp_path):
    env: dict[str, str] = {}
    start_run_masters(tmp_path, {"raven": "raven"}, env=env)
    socket = tmp_path / "socket"
    socket.touch()
    argv = build_argv(["raven", "true"], env=env, socket_lookup=lambda a, t: socket)
    assert f"ControlPath={control_path_template(tmp_path)}" in argv


@pytest.mark.unit
def test_start_under_test_mode_runs_no_ssh_and_leaves_the_environment_alone(
    tmp_path, fake_ssh
):
    before = dict(os.environ)
    start_run_masters(tmp_path, {"raven": "raven"})
    assert not fake_ssh.exists()
    assert os.environ.get(CONTROL_DIR_ENV) == before.get(CONTROL_DIR_ENV)


@pytest.mark.unit
def test_master_alive_asks_ssh_with_the_runs_socket(tmp_path, fake_ssh, monkeypatch):
    template = control_path_template(tmp_path)
    assert master_alive("raven", template)
    assert f"-o ControlPath={template} -O check raven" in fake_ssh.read_text()
    monkeypatch.setenv("FAKE_SSH_CHECK", "255")
    assert not master_alive("raven", template)


@pytest.mark.unit
def test_open_master_skips_a_live_master(tmp_path, fake_ssh):
    assert open_master("raven", str(tmp_path / "cm" / "x-%C"))
    assert " -M " not in f" {fake_ssh.read_text()} "


@pytest.mark.unit
def test_open_master_removes_a_stale_socket_and_starts_a_master(
    tmp_path, fake_ssh, monkeypatch
):
    monkeypatch.setenv("FAKE_SSH_CHECK", "255")
    stale = tmp_path / "socket"
    stale.write_text("")
    template = str(tmp_path / "cm" / "x-%C")
    assert not open_master("raven", template, interactive=False)
    assert not stale.exists()
    calls = fake_ssh.read_text().splitlines()
    master = next(c for c in calls if c.startswith("-M -N -f"))
    assert f"ControlPath={template}" in master
    assert "ControlPersist=yes" in master
    assert "BatchMode=yes" in master
    assert master.endswith("raven")
    assert (tmp_path / "cm").stat().st_mode & 0o777 == 0o700


@pytest.mark.unit
def test_run_masters_map_hosts_to_systems_and_log_in_once_per_call(tmp_path):
    masters = RunMasters()
    assert not masters.login("raven")  # not configured: nothing to log in to
    masters.configure(tmp_path, {"raven": "raven", "raven_gpu": "raven", "x": "y"})
    assert masters.systems_for("raven") == ["raven", "raven_gpu"]
    seen = []

    def opener(host, template, *, interactive):
        seen.append((host, template, interactive))
        return True

    assert masters.login("raven", opener=opener)
    assert seen == [("raven", control_path_template(tmp_path), True)]


@pytest.mark.unit
def test_own_master_can_be_turned_off(monkeypatch):
    assert ssh_master.own_master_enabled()
    monkeypatch.setenv("ALOMANCY_OWN_SSH_MASTER", "0")
    assert not ssh_master.own_master_enabled()


@pytest.mark.unit
def test_rsh_leaves_the_users_ssh_alone_when_the_runs_master_is_missing(tmp_path):
    """Forcing the run's socket for a host whose master never opened sent
    every background call past the user's own authenticated socket, to a
    BatchMode login that fails on password/OTP hosts."""
    env = {CONTROL_DIR_ENV: str(tmp_path), RUN_TAG_ENV: "abc123"}
    missing = build_argv(
        ["raven", "true"], env=env, socket_lookup=lambda a, t: tmp_path / "gone"
    )
    assert not any("ControlPath" in arg for arg in missing)
    unknown = build_argv(["raven", "true"], env=env, socket_lookup=lambda a, t: None)
    assert not any("ControlPath" in arg for arg in unknown)


@pytest.mark.unit
def test_rsh_finds_the_host_and_asks_ssh_for_its_socket(tmp_path, fake_ssh):
    """The socket's %C depends on host, port and user, so ssh -G gets the
    same options the call has, up to the host."""
    from alomancy.utils.rsh import _host_index, run_socket_path

    assert _host_index(["raven", "true"]) == 0
    assert _host_index(["-p", "22", "-l", "u", "raven", "ls"]) == 4
    assert _host_index(["-p22", "-T", "raven", "ls"]) == 2
    assert _host_index(["-T"]) is None
    path = run_socket_path(["-p", "22", "raven", "ls"], "/cm/abc-%C")
    assert path == tmp_path / "socket"
    call = fake_ssh.read_text().splitlines()[-1]
    assert call == "-G -o ControlPath=/cm/abc-%C -p 22 raven"
