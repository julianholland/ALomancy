"""`alomancy-rsh`: the ssh that ExPyRe runs for a workflow's background calls.

ExPyRe shells out to `$EXPYRE_RSH` (default `ssh`) for every squeue poll
and rsync. On a host that only accepts password/OTP login, a call made after
the shared ControlMaster connection has died starts a fresh login and blocks
reading a password from /dev/tty, which the TUI hides and nobody answers:
the run hangs forever. BatchMode makes such a call fail at once (exit 255)
instead, so the executor can wait for re-authentication
(remote_submission/executor.py, `ssh_auth_lost`). While the ControlMaster
is up, the options change nothing.

It is a separate executable rather than `EXPYRE_RSH="ssh -o ..."` because
ExPyRe builds `rsync -e <EXPYRE_RSH>` by joining strings and splitting on
spaces, so a multi-word command would break every rsync.
"""

import os
import re
import sys

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


def main() -> None:
    os.execvp("ssh", ["ssh", *NONINTERACTIVE_SSH_OPTIONS, *sys.argv[1:]])


if __name__ == "__main__":
    main()
