"""`alomancy-rsh`: the ssh that ExPyRe runs for a workflow's background calls.

ExPyRe shells out to `$EXPYRE_RSH` (default `ssh`) for every squeue poll
and rsync. On a host that only accepts password/OTP login, a call made after
the shared ControlMaster connection has died starts a fresh login and blocks
reading a password from /dev/tty, which the TUI hides and nobody answers:
the run hangs forever. BatchMode makes such a call fail at once (exit 255)
instead, so the executor can wait for re-authentication
(remote_submission/executor.py, `ssh_auth_lost`). While the ControlMaster
is up, the options change nothing.

When the run owns its ssh masters (utils/ssh_master.py), the run sets
ALOMANCY_SSH_CONTROL_DIR and ALOMANCY_SSH_RUN_TAG, and every call goes
through the run's socket with ControlMaster=no: it uses the master the run
logged in with and never tries to become one itself.

It is a separate executable rather than `EXPYRE_RSH="ssh -o ..."` because
ExPyRe builds `rsync -e <EXPYRE_RSH>` by joining strings and splitting on
spaces, so a multi-word command would break every rsync.
"""

import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

NONINTERACTIVE_SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]

# ssh's own login-failure messages. "Permission denied" alone would also
# match a remote file error (rsync's "opendir ... Permission denied (13)"),
# which must keep failing the job rather than wait for a login.
_SSH_AUTH_FAILURE = re.compile(
    r"Permission denied \((?:publickey|password|keyboard-interactive|gssapi"
    r"|hostbased)[^)]*\)|Host key verification failed"
)


def is_ssh_auth_failure(error: BaseException | str) -> bool:
    """Whether *error* is ssh failing to log in (no live ControlMaster and no
    non-interactive way to authenticate), as opposed to any other failure."""
    return bool(_SSH_AUTH_FAILURE.search(str(error)))


# ssh options that take a value (the next argument, or attached: -p22).
_SSH_OPTIONS_WITH_VALUE = set("BbcDEeFIiJLlmOoPpQRSWw")


def _host_index(args: list[str]) -> int | None:
    """Index of the destination host in ssh *args* (the first argument
    that is neither an option nor an option's value), or None."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            return i + 1 if i + 1 < len(args) else None
        if arg.startswith("-") and len(arg) > 1:
            letter = arg[-1] if len(arg) == 2 else arg[1]
            takes_value = len(arg) == 2 and letter in _SSH_OPTIONS_WITH_VALUE
            i += 2 if takes_value else 1
            continue
        return i
    return None


def run_socket_path(args: list[str], template: str) -> Path | None:
    """The run's socket for the host *args* connect to: ssh -G (local only,
    no connection) expands the template's %C with that host, port and user
    as these arguments give them. None when it can't be worked out."""
    index = _host_index(args)
    if index is None:
        return None
    try:
        proc = subprocess.run(
            ["ssh", "-G", "-o", f"ControlPath={template}", *args[: index + 1]],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == "controlpath" and value:
            return Path(value)
    return None


def build_argv(
    args: list[str],
    env: Mapping[str, str] = os.environ,
    socket_lookup: Callable[[list[str], str], Path | None] = run_socket_path,
) -> list[str]:
    """The ssh command line for *args* (what ExPyRe or rsync passed).

    The run's own master socket is used only when it exists for the host
    being reached: forcing it for a host whose master failed to open (or
    was never opened) would override the user's own ssh multiplexing, and
    on a password/OTP host fail where their authenticated socket works."""
    options = list(NONINTERACTIVE_SSH_OPTIONS)
    control_dir = env.get("ALOMANCY_SSH_CONTROL_DIR")
    tag = env.get("ALOMANCY_SSH_RUN_TAG")
    if control_dir and tag:
        template = f"{control_dir}/{tag}-%C"
        socket = socket_lookup(args, template)
        if socket is not None and socket.exists():
            options += [
                "-o",
                f"ControlPath={template}",
                "-o",
                "ControlMaster=no",
            ]
    return ["ssh", *options, *args]


def main() -> None:
    os.execvp("ssh", build_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
