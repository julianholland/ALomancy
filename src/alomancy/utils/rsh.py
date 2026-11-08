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
import sys
from collections.abc import Mapping

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


def build_argv(args: list[str], env: Mapping[str, str] = os.environ) -> list[str]:
    """The ssh command line for *args* (what ExPyRe or rsync passed)."""
    options = list(NONINTERACTIVE_SSH_OPTIONS)
    control_dir = env.get("ALOMANCY_SSH_CONTROL_DIR")
    tag = env.get("ALOMANCY_SSH_RUN_TAG")
    if control_dir and tag:
        options += [
            "-o",
            f"ControlPath={control_dir}/{tag}-%C",
            "-o",
            "ControlMaster=no",
        ]
    return ["ssh", *options, *args]


def main() -> None:
    os.execvp("ssh", build_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
