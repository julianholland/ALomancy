"""`alomancy login [HPC]`: log in again to an HPC whose ssh connection the
run in the current directory lost, from any terminal on the run's machine.

The run owns one ssh master per HPC host (utils/ssh_master.py); its socket
is named after the run directory, so this reopens exactly the master the
run uses, interactively (password/OTP in this terminal). The run's next
poll then succeeds on its own and its panel clears the login banner.
Without an HPC name it logs in to every HPC the run's status file
(`results/remote_status.json`) reports as needing a login. This is the
way back in for a run without a terminal (nohup, tmux in the background);
a run with its live panel can also log in with the L key."""

import json
import socket
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from alomancy.cli.cancel import NotARunDirectoryError, require_run_directory, run_host
from alomancy.cli.status import STATUS_PATH
from alomancy.utils.remote_ssh import resolve_hpc_host
from alomancy.utils.ssh_master import control_path_template, open_master


def lost_systems(run_dir: Path) -> list[str]:
    """The HPCs the run's status file reports as needing a login."""
    try:
        snapshot = json.loads((run_dir / STATUS_PATH).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return sorted((snapshot.get("auth_lost") or {}).keys())


def system_host(run_dir: Path, system: str) -> str | None:
    """The ssh host of `system`, from the run's own ExPyRe config, else
    ALomancy's canonical one."""
    try:
        config = json.loads((run_dir / ".expyre" / "config.json").read_text())
        host = config.get("systems", {}).get(system, {}).get("host")
        if host:
            return str(host)
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    host = resolve_hpc_host(system)
    return None if host is None else str(host)


def login(
    run_dir: Path,
    system: str | None = None,
    *,
    console: Console | None = None,
    opener: Callable[..., bool] = open_master,
) -> int:
    """Log in to `system` (or every HPC needing one) for the run in
    `run_dir`. Returns the exit status: 0 logged in (or nothing to do),
    1 a login failed or this isn't the run's machine, 2 not a run."""
    console = console or Console()
    try:
        require_run_directory(run_dir, "login")
    except NotARunDirectoryError as exc:
        console.print(f"alomancy login: {exc}", markup=False)
        return 2
    made_on = run_host(run_dir / ".expyre")
    if made_on != socket.gethostname():
        console.print(
            f"alomancy login: this run runs on {made_on}; its ssh connections "
            "live there. Log in from a terminal on that machine.",
            markup=False,
        )
        return 1
    systems = [system] if system else lost_systems(run_dir)
    if not systems:
        console.print(
            "No HPC of this run needs a login. Name one to log in anyway: "
            "alomancy login <HPC>",
            markup=False,
        )
        return 0
    template = control_path_template(run_dir)
    status = 0
    for host in sorted({h for s in systems if (h := system_host(run_dir, s))}):
        names = ", ".join(s for s in systems if system_host(run_dir, s) == host)
        console.print(f"Logging in to {names} ({host}) for this run.", markup=False)
        if opener(host, template, interactive=True):
            console.print(
                f"ssh to {names} works again; the run picks it up on its next poll.",
                markup=False,
            )
        else:
            console.print(f"Login to {host} failed.", markup=False)
            status = 1
    unknown = [s for s in systems if system_host(run_dir, s) is None]
    for name in unknown:
        console.print(f"Unknown HPC {name!r}: no ssh host configured.", markup=False)
        status = 1
    return status
