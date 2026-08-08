import os
import shutil
import sys
from pathlib import Path

import matplotlib

from alomancy.configs.global_config import EXPYRE_CONFIG, LEGACY_EXPYRE_CONFIG


def _ensure_local_expyre_root() -> None:
    """Give each ALomancy run its own isolated ExPyRe job-tracking state.

    expyre.config resolves its configuration -- and, critically, jobs.db
    and every job's stage directory (JobsDB(local_stage_dir / 'jobs.db'),
    and ExPyRe's own tempfile.mkdtemp(dir=config.local_stage_dir)) -- by
    walking from the current working directory up to $HOME, merging every
    ".expyre"/"_expyre" directory found along the way (deeper overrides
    shallower), and using the *deepest* one as local_stage_dir. Without a
    run-local ".expyre", that's always "~/.expyre": every alomancy process
    running anywhere that shares this $HOME (including, on a cluster with a
    shared home directory, processes on other machines entirely) shares one
    jobs.db and one pool of job stage directories. expyre's own
    "list(config.db.jobs(id=...))[0]" pattern (func.py) assumes a job's row
    is always present; a second process mutating that same file can make a
    row disappear out from under a first process's poll loop, surfacing as
    a bare "IndexError: list index out of range" deep in expyre.

    Rather than relying on expyre's own merge-on-every-import walk (which
    would still leave every run reading live off a shared "~/.expyre" if a
    run-local directory were seeded with an empty stub), this copies
    ALomancy's canonical HPC systems config -- EXPYRE_CONFIG
    ("~/.alomancy/expyre_config.json", written by `alomancy add-hpc`;
    falling back to LEGACY_EXPYRE_CONFIG, "~/.expyre/config.json", for
    installs that haven't re-run the wizard since upgrading) -- into a
    fresh "<rundir>/.expyre/config.json" the first time a run's own
    directory is imported from. That copy makes this directory the
    deepest, and therefore local_stage_dir, for every subsequent job this
    run submits. It's a one-time snapshot, not a live reference: editing
    the master config later doesn't retroactively affect a run whose
    ".expyre/" already exists (delete that run's ".expyre/config.json" and
    re-import to pick up a fresh copy). Idempotent and non-fatal: does
    nothing if this directory already has a ".expyre" or "_expyre" (a
    rerun, or a user who set one up deliberately), if neither config
    source exists yet (nothing configured -- nothing to isolate), or if the
    copy fails for any OS-level reason (e.g. a permissions issue) --
    package import must never hard-fail over this best-effort setup step.

    Skipped under pytest (matches expyre's own "if 'pytest' not in
    sys.modules" guard) so test runs never litter the repo/test cwd with a
    stray directory, and skipped if EXPYRE_ROOT is set to anything other
    than the default "@" -- an explicit EXPYRE_ROOT bypasses expyre's
    walk-and-merge entirely, so a run-local ".expyre" would never be
    consulted anyway. The guarded-vs-core logic split (this function just
    checks whether to run at all; _seed_local_expyre_root does the actual
    work) exists so tests can exercise the core logic directly without
    fighting the pytest-detection guard, which must stay on for real runs.
    """
    if "pytest" in sys.modules:
        return
    if os.environ.get("EXPYRE_ROOT", "@") != "@":
        return
    _seed_local_expyre_root(Path.cwd())


def _seed_local_expyre_root(cwd: Path) -> None:
    """Core logic for _ensure_local_expyre_root, without the pytest/
    EXPYRE_ROOT guard -- see that function's docstring."""
    if (cwd / ".expyre").exists() or (cwd / "_expyre").exists():
        return

    master = EXPYRE_CONFIG if EXPYRE_CONFIG.exists() else LEGACY_EXPYRE_CONFIG
    if not master.exists():
        return

    try:
        (cwd / ".expyre").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(master, cwd / ".expyre" / "config.json")
    except OSError:
        pass


_ensure_local_expyre_root()

# Force the non-interactive Agg backend before any submodule can import
# matplotlib.pyplot. Without this, matplotlib auto-selects an interactive
# backend (e.g. TkAgg) whenever a DISPLAY is available. RemoteJobExecutor
# polls remote jobs from a ThreadPoolExecutor (executor.py), and if a
# Figure created on the main thread (analysis/plotting.py, mlip_plots.py,
# timing_plots.py) is later garbage-collected on one of those worker
# threads, Tkinter's __del__ finalizers try to call back into the Tk
# interpreter from the wrong thread, raising "main thread is not in main
# loop" and aborting the whole process with SIGABRT -- killing every
# in-flight remote job along with it. Agg never creates GUI/Tk state, so
# it has no thread affinity to violate.
matplotlib.use("Agg")
