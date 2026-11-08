"""Shared helpers for shelling out over ssh to a configured HPC host.

Used both by the workflow-startup HPC summary (checking what alomancy
version is installed remotely) and by the `alomancy upgrade-hpc` CLI
(upgrading it). Every ssh-invoking call here takes an explicit timeout --
this cluster environment commonly has no passwordless key auth, so a fresh,
non-multiplexed ssh connection can block indefinitely on an interactive
password/OTP prompt nobody is present to answer. See
remote_submission/executor.py's module docstring for the full story on why
that matters; this module is the CLI/summary-side counterpart of the same
lesson.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from alomancy.configs.global_config import EXPYRE_CONFIG, LEGACY_EXPYRE_CONFIG

logger = logging.getLogger(__name__)


def run_ssh_command(
    host: str, command: str, timeout: float, control_path: str | None = None
) -> tuple[bool, str, str]:
    """Run `command` on `host` over ssh, bounded by `timeout` seconds.

    `timeout` has no default -- every call site must consciously choose one.
    Returns (success, stdout, stderr); never raises. Note: ssh opens
    /dev/tty directly for a password/OTP prompt even though stdout/stderr
    are captured here, so a host needing interactive auth can still block
    for up to `timeout` seconds rather than failing immediately -- that's
    exactly the scenario this timeout exists to bound. Interactive on
    purpose: its callers are CLI commands someone is watching. A workflow's
    unattended calls go through ExPyRe and `alomancy-rsh` instead.

    `control_path`: a run's own master socket (utils/ssh_master.py), used
    without becoming a master, so a command for a running run (e.g.
    `alomancy cancel`) needs no new login while the run's connection is up.
    """
    options = (
        []
        if control_path is None
        else ["-o", f"ControlPath={control_path}", "-o", "ControlMaster=no"]
    )
    try:
        proc = subprocess.run(
            ["ssh", *options, host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        return False, "", str(exc)
    return proc.returncode == 0, proc.stdout, proc.stderr


def derive_python_from_venv(venv_cmd: str) -> str | None:
    """Extract the python executable path from a venv activation command.

    Handles: source /path/to/venv/bin/activate -> /path/to/venv/bin/python
    """
    m = re.search(r"source\s+(.+)/bin/activate", venv_cmd)
    if m:
        return f"{m.group(1)}/bin/python"
    return None


def resolve_hpc_host(
    hpc_name: str,
    expyre_config_path: Path = EXPYRE_CONFIG,
    legacy_expyre_config_path: Path = LEGACY_EXPYRE_CONFIG,
) -> str | None:
    """Look up the ssh host/alias for `hpc_name` in ALomancy's canonical
    ExPyRe systems config (~/.alomancy/expyre_config.json by default),
    falling back to the pre-isolation "~/.expyre/config.json" location if
    the canonical file doesn't exist yet -- e.g. `alomancy add-hpc` hasn't
    been re-run since upgrading to the version that introduced the split
    (see cli/add_hpc.py's write_expyre_config, which performs the
    equivalent migration the moment the wizard writes anything).

    Returns None (never raises) if the system name isn't found in either
    file, or both are missing/malformed -- callers must degrade gracefully
    rather than crash a workflow run or abort a whole upgrade batch over
    one bad profile.
    """
    for candidate in (expyre_config_path, legacy_expyre_config_path):
        try:
            with open(candidate) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(
                "Could not read ExPyRe config %s while resolving ssh host "
                "for hpc_name=%r: %s",
                candidate,
                hpc_name,
                exc,
            )
            continue
        try:
            host = data.get("systems", {}).get(hpc_name, {}).get("host")
        except (AttributeError, TypeError):
            host = None
        if host is not None:
            return host
    return None


def get_remote_alomancy_version(
    host: str, python_path: str, timeout: float = 10.0
) -> str | None:
    """Query the alomancy version installed on `host`'s `python_path`.

    Returns None on any failure (ssh error, timeout, alomancy not
    importable, empty output) -- never raises.
    """
    success, stdout, _stderr = run_ssh_command(
        host,
        f'{python_path} -c "import alomancy; print(alomancy.__version__)"',
        timeout=timeout,
    )
    version = stdout.strip()
    if success and version:
        return version
    return None


def ensure_ssh_connectivity(hpc_profiles: dict[str, dict]) -> None:
    """Establish (and, if needed, interactively authenticate) an ssh
    connection to every distinct HPC host used by this run, once, up
    front -- called from pre_run_checks() before any remote submission
    begins, while whoever launched the run is presumably still watching
    the terminal to answer a password/OTP prompt if one appears.

    Deliberately unbounded and non-captured, unlike every other ssh call
    in this module: `subprocess.run(["ssh", host, "true"])` with no
    `timeout=` and no `capture_output=`, so a password/OTP prompt prints
    to the real terminal and reads the real stdin, and this simply blocks
    until the user answers it (or the connection succeeds/fails outright).
    That is the exact interactive-prompt-and-wait experience this
    function exists to move to start-of-run: without it, the first ssh
    call needing auth happens whenever a background RemoteJobExecutor
    thread first touches that host, possibly hours into an unattended
    run with nobody present to answer -- see remote_submission/
    executor.py's _get_ssh_call_lock docstring for that failure mode.

    If the host's ssh config uses connection multiplexing (ControlMaster/
    ControlPersist), authenticating here establishes the shared control
    connection once; every later ssh call from RemoteJobExecutor then
    reuses it silently for as long as it stays alive, instead of each one
    risking its own fresh, potentially-blocking connection attempt.

    Never raises: a host that's unreachable for a reason other than
    pending auth (bad hostname, network down) is logged as a warning and
    left for the actual remote submission to fail loudly on later --
    this is a best-effort convenience, not a hard gate on startup.
    Skipped outright under ALOMANCY_TEST_MODE/ALOMANCY_MOCK_EXTERNAL
    (set autouse for the whole test suite, see tests/conftest.py), same
    as every other real-network call in this module and in
    core/active_learning_workflow.py -- tests must never shell out to
    ssh or block waiting on a password prompt.
    """
    if (
        os.getenv("ALOMANCY_TEST_MODE") == "1"
        or os.getenv("ALOMANCY_MOCK_EXTERNAL") == "1"
    ):
        return

    seen_hosts: set[str] = set()
    for hpc_name in hpc_profiles:
        host = resolve_hpc_host(hpc_name)
        if not host or host in seen_hosts:
            continue
        seen_hosts.add(host)

        logger.info(
            "Checking ssh connectivity to '%s' (%s) -- if prompted, enter your "
            "password/OTP now.",
            hpc_name,
            host,
        )
        try:
            proc = subprocess.run(["ssh", host, "true"])
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning(
                "Could not check ssh connectivity to '%s' (%s): %s. Remote "
                "submission to this host may fail later.",
                hpc_name,
                host,
                exc,
            )
            continue
        if proc.returncode != 0:
            logger.warning(
                "ssh connectivity check for '%s' (%s) exited %d. Remote "
                "submission to this host may fail later.",
                hpc_name,
                host,
                proc.returncode,
            )
        else:
            logger.info("ssh connectivity to '%s' (%s) OK.", hpc_name, host)


def _find_rsh_wrapper() -> str | None:
    wrapper = Path(sys.executable).parent / "alomancy-rsh"
    return str(wrapper) if wrapper.exists() else shutil.which("alomancy-rsh")


def use_noninteractive_ssh(
    env: MutableMapping[str, str] = os.environ,
    systems: dict[str, Any] | None = None,
    wrapper: str | None = None,
) -> None:
    """Route a workflow's unattended ExPyRe ssh calls through `alomancy-rsh`
    (BatchMode), so a call made after the ControlMaster died fails instead
    of blocking forever on a hidden password prompt (see utils/rsh.py).
    Called after ensure_ssh_connectivity, whose login stays interactive.

    ExPyRe reads the ssh command from two places: rsync copies use
    $EXPYRE_RSH, while each System and its scheduler (squeue, sbatch,
    scancel) keep the `remsh_cmd` from the systems config, which
    `alomancy add-hpc` writes as "ssh". Both are switched; a user-set
    EXPYRE_RSH or a system with a custom remsh_cmd is left alone.
    *systems* defaults to expyre.config.systems.

    Raises ValueError only if the wrapper's path contains whitespace:
    ExPyRe splits remsh_cmd and EXPYRE_RSH on whitespace, so every
    background ssh and rsync would fail. Called before any job is
    submitted."""
    wrapper = wrapper or _find_rsh_wrapper()
    if wrapper is None:
        logger.warning(
            "alomancy-rsh not found next to %s or on PATH (reinstall alomancy). "
            "Background ssh calls may hang on a password prompt if the ssh "
            "connection drops.",
            sys.executable,
        )
        return
    if any(ch.isspace() for ch in wrapper):
        raise ValueError(
            f"alomancy-rsh is installed at {wrapper!r}, a path with whitespace "
            "in it. ExPyRe splits its ssh command on whitespace, so every "
            "background ssh and rsync (squeue, sbatch, staging files) would "
            "fail. Install alomancy in an environment whose path has no "
            "spaces, or reach this one through a symlink without spaces and "
            "run python from there."
        )
    if systems is None:
        try:
            from expyre import config as expyre_config

            systems = expyre_config.systems or {}
        except Exception as exc:  # never break a run start over this
            logger.debug("Could not read ExPyRe systems: %s", exc)
            systems = {}
    env.setdefault("EXPYRE_RSH", wrapper)
    for name, system in systems.items():
        if getattr(system, "remsh_cmd", None) != "ssh":
            continue
        system.remsh_cmd = wrapper
        scheduler = getattr(system, "scheduler", None)
        if scheduler is not None and getattr(scheduler, "remsh_cmd", None) == "ssh":
            scheduler.remsh_cmd = wrapper
        logger.debug("ExPyRe system %s: background ssh uses %s.", name, wrapper)


def get_alomancy_version_for_profile(profile: dict) -> str | None:
    """Best-effort remote-installed-version lookup for one HPC profile dict.

    Skipped outright under ALOMANCY_TEST_MODE/ALOMANCY_MOCK_EXTERNAL (set
    autouse for the whole test suite, see tests/conftest.py) so tests never
    make a real ssh call, mirroring _fetch_latest_pypi_version's identical
    guard in core/active_learning_workflow.py. Never raises -- a
    workflow's startup summary must not crash a run over an unreachable
    HPC host.
    """
    if (
        os.getenv("ALOMANCY_TEST_MODE") == "1"
        or os.getenv("ALOMANCY_MOCK_EXTERNAL") == "1"
    ):
        return None

    try:
        host = resolve_hpc_host(profile.get("hpc_name", ""))
        if not host:
            return None
        pre_cmds = profile.get("pre_cmds") or []
        if not pre_cmds:
            return None
        python_path = derive_python_from_venv(pre_cmds[0])
        if not python_path:
            return None
        return get_remote_alomancy_version(host, python_path)
    except Exception as exc:
        logger.debug(
            "Could not determine remote alomancy version for profile %r: %s",
            profile.get("hpc_name"),
            exc,
        )
        return None
