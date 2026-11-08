"""The run's own ssh connection (ControlMaster) to each HPC host.

At run start alomancy opens one master connection per HPC host, with a
socket path of its own (`control_path`), and logs in interactively there
(password/OTP reach the real terminal). Every background ssh call
(`alomancy-rsh`, utils/rsh.py) then goes through that socket with
``ControlMaster=no``, so it never starts a login of its own. This works
whether or not the user's ~/.ssh/config sets up multiplexing, and alomancy
always knows which socket to check (`master_alive`) and how to log in
again (`open_master`, from the live panel's L key or `alomancy login`).

The socket lives under ~/.alomancy/cm/ (a unix socket path is limited to
about 104 characters, so it can't sit under the run directory) and is
named after the run directory, so each run has its own master per host
and a restart of the same run finds it still alive. `close_master` ends
it when the run finishes; a crashed or interrupted run leaves it up, so
the restart needs no new login (`alomancy cancel` closes it).

`ALOMANCY_OWN_SSH_MASTER=0` turns all of this off: the run then relies on
the user's own ssh config, as before.

No call here runs under ALOMANCY_TEST_MODE/ALOMANCY_MOCK_EXTERNAL, like
every other real ssh call (see utils/remote_ssh.py).
"""

import hashlib
import logging
import os
import subprocess
import threading
from collections.abc import Callable, MutableMapping
from pathlib import Path

from alomancy.configs.global_config import ALOMANCY_DIR

logger = logging.getLogger(__name__)

CONTROL_DIR = ALOMANCY_DIR / "cm"
# Read by alomancy-rsh: the run's socket directory and run tag.
CONTROL_DIR_ENV = "ALOMANCY_SSH_CONTROL_DIR"
RUN_TAG_ENV = "ALOMANCY_SSH_RUN_TAG"
OWN_MASTER_ENV = "ALOMANCY_OWN_SSH_MASTER"
# ssh's %C: a 40-character hash of local host, remote host, port and user.
_PERCENT_C_LEN = 40
# ssh adds ".XXXXXXXXXXXXXXXX" while it creates the socket.
_SSH_TEMP_SUFFIX_LEN = 17
_MAX_SOCKET_PATH = 104
# The master's own keepalives: tolerate ~5 min of network trouble before
# giving up (the user's config often drops it after a minute).
MASTER_OPTIONS = [
    "-o",
    "ControlPersist=yes",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=10",
]
CHECK_TIMEOUT = 15.0


def own_master_enabled() -> bool:
    return os.environ.get(OWN_MASTER_ENV, "1") != "0"


def external_calls_disabled() -> bool:
    return (
        os.getenv("ALOMANCY_TEST_MODE") == "1"
        or os.getenv("ALOMANCY_MOCK_EXTERNAL") == "1"
    )


def run_tag(run_dir: str | Path) -> str:
    """Six hex characters naming the run (its resolved directory)."""
    resolved = str(Path(run_dir).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:6]


def control_dir(base: Path | None = None) -> Path:
    """The socket directory: ~/.alomancy/cm, or /tmp/alomancy-cm-<uid> when
    the home path is too long for a unix socket."""
    base = CONTROL_DIR if base is None else base
    longest = len(str(base)) + 1 + 6 + 1 + _PERCENT_C_LEN + _SSH_TEMP_SUFFIX_LEN
    if longest >= _MAX_SOCKET_PATH:
        base = Path("/tmp") / f"alomancy-cm-{os.getuid()}"
    return base


def control_path_template(run_dir: str | Path, base: Path | None = None) -> str:
    """The ControlPath option for this run, with ssh's own %C left for ssh
    to expand (one socket per run and host)."""
    return str(control_dir(base) / f"{run_tag(run_dir)}-%C")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def master_alive(host: str, template: str) -> bool:
    """Whether the run's master to *host* is up (``ssh -O check``)."""
    try:
        proc = subprocess.run(
            ["ssh", "-o", f"ControlPath={template}", "-O", "check", host],
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0


def _socket_path(host: str, template: str) -> Path | None:
    """The expanded socket path for *host* (``ssh -G``), or None."""
    try:
        proc = subprocess.run(
            ["ssh", "-G", "-o", f"ControlPath={template}", host],
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == "controlpath" and value:
            return Path(value)
    return None


def open_master(host: str, template: str, *, interactive: bool = True) -> bool:
    """Open the run's master connection to *host*; returns whether it is up.

    Interactive: no captured output and no timeout, so a password/OTP prompt
    reaches the real terminal and this waits for the answer. Not
    interactive: BatchMode, so only a key login can succeed (used by the
    master watch to reopen a key-auth host silently). A stale socket left
    by a dead master is removed first, since ssh -M would otherwise run
    without becoming a master."""
    if master_alive(host, template):
        return True
    _ensure_dir(Path(template).parent)
    stale = _socket_path(host, template)
    if stale is not None and stale.exists():
        stale.unlink(missing_ok=True)
    command = [
        "ssh",
        "-M",
        "-N",
        "-f",
        "-o",
        f"ControlPath={template}",
        *MASTER_OPTIONS,
    ]
    try:
        if interactive:
            subprocess.run([*command, host])
        else:
            subprocess.run(
                [*command, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host],
                capture_output=True,
                timeout=60,
                stdin=subprocess.DEVNULL,
            )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("Opening the ssh master to %s failed: %s", host, exc)
        return False
    return master_alive(host, template)


def close_master(host: str, template: str) -> None:
    """End the run's master to *host* (``ssh -O exit``); never raises."""
    try:
        subprocess.run(
            ["ssh", "-o", f"ControlPath={template}", "-O", "exit", host],
            capture_output=True,
            timeout=CHECK_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("Closing the ssh master to %s failed: %s", host, exc)


# --- the run's masters ---------------------------------------------------------


class RunMasters:
    """The HPC systems of this run and the ssh host each one uses, with the
    socket template of the run. One per process (MASTERS)."""

    def __init__(self) -> None:
        self.template: str | None = None
        self.hosts: dict[str, str] = {}  # system -> ssh host
        self._login_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.template is not None

    def configure(self, run_dir: str | Path, hosts: dict[str, str]) -> None:
        self.template = control_path_template(run_dir)
        self.hosts = dict(hosts)

    def systems_for(self, host: str) -> list[str]:
        return sorted(s for s, h in self.hosts.items() if h == host)

    def login(
        self,
        host: str,
        opener: Callable[..., bool] = open_master,
    ) -> bool:
        """Log in to *host* interactively, one prompt at a time per process."""
        if self.template is None:
            return False
        with self._login_lock:
            return opener(host, self.template, interactive=True)

    def close_all(self) -> None:
        if self.template is None or external_calls_disabled():
            return
        for host in sorted(set(self.hosts.values())):
            close_master(host, self.template)


MASTERS = RunMasters()


def start_run_masters(
    run_dir: str | Path,
    hosts: dict[str, str],
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Open the run's master to every distinct host, logging in where
    needed (called from pre_run_checks while the person who launched the
    run is watching). Never raises: an unreachable host is a warning and
    the real submission fails loudly later. Sets the environment
    alomancy-rsh reads (CONTROL_DIR_ENV, RUN_TAG_ENV) in *env* (default:
    this process's environment, which ExPyRe's subprocesses inherit).
    Under ALOMANCY_TEST_MODE/ALOMANCY_MOCK_EXTERNAL it changes nothing
    unless given an *env* of its own."""
    if external_calls_disabled() and env is None:
        return
    env_map: MutableMapping[str, str] = os.environ if env is None else env
    MASTERS.configure(run_dir, hosts)
    template = control_path_template(run_dir)
    env_map[CONTROL_DIR_ENV] = str(Path(template).parent)
    env_map[RUN_TAG_ENV] = run_tag(run_dir)
    if external_calls_disabled():
        return
    for host in sorted(set(hosts.values())):
        systems = ", ".join(MASTERS.systems_for(host))
        if master_alive(host, template):
            logger.info("ssh connection to %s (%s) already open.", systems, host)
            continue
        logger.info(
            "Opening the run's ssh connection to %s (%s) -- if prompted, enter "
            "your password/OTP now.",
            systems,
            host,
        )
        if open_master(host, template, interactive=True):
            logger.info("ssh connection to %s (%s) OK.", systems, host)
        else:
            logger.warning(
                "Could not open an ssh connection to %s (%s). Remote submission "
                "to this host may fail later; log in again with `alomancy login` "
                "in the run directory.",
                systems,
                host,
            )
