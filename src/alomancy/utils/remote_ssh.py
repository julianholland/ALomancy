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
import subprocess
from pathlib import Path

from alomancy.configs.global_config import EXPYRE_CONFIG, LEGACY_EXPYRE_CONFIG

logger = logging.getLogger(__name__)


def run_ssh_command(host: str, command: str, timeout: float) -> tuple[bool, str, str]:
    """Run `command` on `host` over ssh, bounded by `timeout` seconds.

    `timeout` has no default -- every call site must consciously choose one.
    Returns (success, stdout, stderr); never raises. Note: ssh opens
    /dev/tty directly for a password/OTP prompt even though stdout/stderr
    are captured here, so a host needing interactive auth can still block
    for up to `timeout` seconds rather than failing immediately -- that's
    exactly the scenario this timeout exists to bound.
    """
    try:
        proc = subprocess.run(
            ["ssh", host, command],
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


def get_alomancy_version_for_profile(profile: dict) -> str | None:
    """Best-effort remote-installed-version lookup for one HPC profile dict.

    Skipped outright under ALOMANCY_TEST_MODE/ALOMANCY_MOCK_EXTERNAL (set
    autouse for the whole test suite, see tests/conftest.py) so tests never
    make a real ssh call, mirroring _fetch_latest_pypi_version's identical
    guard in core/base_active_learning.py. Never raises -- a workflow's
    startup summary must not crash a run over an unreachable HPC host.
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
