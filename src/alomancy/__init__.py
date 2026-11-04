import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

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

    The run's directory is then *pinned*: EXPYRE_ROOT is set to it, so
    expyre reads that one directory only, with no walk and no merge.
    Without the pin, every parent ".expyre" still has its config.json
    merged in underneath the run's copy: it can add systems, override
    settings, or set "local_stage_dir" and move the run's jobs.db and
    stage dirs into itself. The walk only stops at a directory equal to
    $HOME, so when the run's path doesn't literally start with $HOME
    (e.g. $HOME=/home/u, cwd /work/home/u/run), it goes up to "/" and
    also picks up the shared "~/.expyre". A run ".expyre" without a
    config.json (a hand-made stub that relies on merging) is not pinned.

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
    _pin_local_expyre_root(Path.cwd())


def _pin_local_expyre_root(cwd: Path) -> None:
    """Seed *cwd*'s ExPyRe directory, then point EXPYRE_ROOT at it when it
    has a config.json -- see _ensure_local_expyre_root's docstring."""
    root = _seed_local_expyre_root(cwd)
    if root is not None and (root / "config.json").exists():
        os.environ["EXPYRE_ROOT"] = str(root)


def _seed_local_expyre_root(cwd: Path) -> Path | None:
    """Core logic for _ensure_local_expyre_root, without the pytest/
    EXPYRE_ROOT guard -- see that function's docstring. Returns the run's
    ExPyRe directory, or None if there is none."""
    for name in (".expyre", "_expyre"):
        if (cwd / name).exists():
            return cwd / name

    master = EXPYRE_CONFIG if EXPYRE_CONFIG.exists() else LEGACY_EXPYRE_CONFIG
    if not master.exists():
        return None

    try:
        (cwd / ".expyre").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(master, cwd / ".expyre" / "config.json")
    except OSError:
        return None
    return cwd / ".expyre"


_ensure_local_expyre_root()

# Import expyre only after the local .expyre root is seeded above: any
# import touching the expyre namespace (even a submodule) first runs
# expyre/__init__.py, which does "from .config import systems" and
# eagerly resolves expyre's own config -- importing this any earlier would
# resolve against a pre-isolation config, defeating the seeding above.
from expyre.subprocess import FailedSubprocessWarning  # noqa: E402

logger = logging.getLogger(__name__)

# Job statuses after which ExPyRe never touches the remote directory again.
_EXPYRE_FINISHED_STATUSES = ("processed", "cleaned")


def _short_rundir_extra(run_dir: Path, home: Path) -> str:
    """The run's own part of its remote directory name: its path below
    $HOME, flattened ("Simulations_alomancy_test_5"), or the whole path
    when it isn't under $HOME. $HOME is often a symlink (/home/u ->
    /work/home/u) while the run path is the resolved one, so both forms
    are tried."""
    for base in (home, home.resolve()):
        try:
            rel = run_dir.relative_to(base)
        except ValueError:
            continue
        if rel.parts:
            return "_".join(rel.parts)
    return str(run_dir).replace("/", "_").strip("_")


def _shorten_rundir_extra(
    systems: dict,
    local_stage_dir: Path,
    db: Any,
    hostname: str,
    home: Path,
) -> bool:
    """Replace the "<hostname>-<flattened local path>" segment ExPyRe
    appends to every system's remote rundir with _short_rundir_extra.

    expyre/config.py's init builds that segment (rundir_extra) from
    $HOSTNAME ("unkownhost" when unset) and the full local run path. It
    adds nothing but length: the remote DFT run directory hit VASP's
    ~255-character path limit (see submitters.MAX_REMOTE_OUTPUT_DIR_LEN).
    Each run still gets its own remote directory.

    Left alone while the run's jobs.db has an unfinished job: ExPyRe finds
    a resumed job's remote directory from the current remote_rundir, so
    jobs submitted under the old layout must keep it. Also left alone when
    the segment isn't where expyre is expected to put it (a different
    expyre version). Returns whether anything was changed.
    """
    run_dir = (
        local_stage_dir.parent
        if local_stage_dir.name in (".expyre", "_expyre")
        else local_stage_dir
    )
    old_extra = hostname + "-" + str(run_dir).replace("/", "_")
    new_extra = _short_rundir_extra(run_dir, home)

    if db is not None:
        unfinished = [
            job for job in db.jobs() if job["status"] not in _EXPYRE_FINISHED_STATUSES
        ]
        if unfinished:
            logger.debug(
                "Keeping ExPyRe's remote rundir layout (.../%s): %d unfinished "
                "job(s) in jobs.db were submitted under it.",
                old_extra,
                len(unfinished),
            )
            return False

    changed = False
    for system in systems.values():
        rundir = system.remote_rundir
        if rundir is None or not rundir.endswith("/" + old_extra):
            continue
        system.remote_rundir = rundir.removesuffix(old_extra) + new_extra
        changed = True
    return changed


def _shorten_expyre_remote_rundirs() -> None:
    """Apply _shorten_rundir_extra to the loaded ExPyRe config. Skipped
    under pytest (expyre's config isn't loaded there). Never fails the
    package import."""
    if "pytest" in sys.modules:
        return
    try:
        from expyre import config as expyre_config

        if expyre_config.local_stage_dir is None:
            return
        _shorten_rundir_extra(
            expyre_config.systems or {},
            Path(expyre_config.local_stage_dir),
            expyre_config.db,
            os.environ.get("HOSTNAME", "unkownhost"),
            Path.home(),
        )
    except Exception:  # best effort; never fail the import
        logger.debug("Could not shorten ExPyRe remote rundirs", exc_info=True)


_shorten_expyre_remote_rundirs()

_RSYNC_RETRY_WARNING_PATTERN = (
    r'(Failed|Succeeded) to run "[^"]*\brsync\b.*on attempt \d+.*trying again'
)


def _register_rsync_retry_warning_filter() -> None:
    """Silence expyre's per-retry-attempt subprocess warnings for a known,
    harmless race: while an mlip_committee job is still 'running', ExPyRe's
    get_results() polls sync_remote_results_status() every check_interval,
    which rsyncs the job's entire remote stage directory back -- including
    checkpoints/ -- regardless of what output_files ALomancy requested
    (that list is only consulted once, locally, after the job succeeds).
    MACE's own checkpoint housekeeping (mace/tools/checkpoint.py's
    CheckpointIO.save) deletes the previous checkpoint file right after
    writing the new one, so a mid-training sync can catch a checkpoint file
    mid-delete, surfacing as rsync "Stale file handle"/"file has vanished"
    errors. expyre's own subprocess retry (subprocess_run, 3 attempts/5s
    apart by default) already recovers from this transparently; only the
    retry-in-progress warnings are filtered here -- a genuine, unrecoverable
    subprocess failure still raises a RuntimeError after exhausting all
    retries, so no real failure is hidden by this filter. Scoped to rsync
    specifically so unrelated subprocess retries (e.g. flaky ssh during job
    submission/status polling) stay visible.

    Split out from module level (rather than a bare warnings.filterwarnings
    call at import time) so tests can re-apply it inside their own
    warnings.catch_warnings() block -- pytest's warnings plugin resets the
    filter list to "always" around each test, which would otherwise wipe
    out the filter this registers at import time.
    """
    warnings.filterwarnings(
        "ignore",
        message=_RSYNC_RETRY_WARNING_PATTERN,
        category=FailedSubprocessWarning,
    )


_register_rsync_retry_warning_filter()

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

__all__ = ["ALomancy", "__version__"]


def __getattr__(name: str) -> Any:
    # Lazy: importing alomancy.core here would pull in expyre, matplotlib
    # and the MLIP stack at package import, before the setup above is
    # guaranteed to have run for every import path.
    if name == "ALomancy":
        from alomancy.core.entry import ALomancy

        return ALomancy
    if name == "__version__":
        # Written by the build backend (hatch-vcs) at build/install time;
        # missing only in a source tree that was never installed.
        try:
            from alomancy.version import __version__
        except ImportError:
            return "unknown"
        return __version__
    raise AttributeError(f"module 'alomancy' has no attribute {name!r}")
