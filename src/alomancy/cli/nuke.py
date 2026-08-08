import shutil
from pathlib import Path


def resolve_default_expyre_dir() -> Path:
    """Resolve the ExPyRe directory `alomancy nuke` should target when the
    user doesn't pass --expyre-dir explicitly.

    Prefers expyre.config.local_stage_dir -- the directory expyre actually
    resolved from the current working directory (see
    alomancy/__init__.py's _ensure_local_expyre_root, which gives each run
    its own local_stage_dir by default) -- so nuking from within a project
    directory only ever touches that project's own job state. Falls back
    to the legacy "~/.expyre" if expyre hasn't resolved any config yet
    (e.g. no HPC configured), matching this command's behavior before
    per-run isolation existed.
    """
    from expyre import config as expyre_config

    if expyre_config.local_stage_dir is not None:
        return Path(expyre_config.local_stage_dir)
    return Path("~/.expyre").expanduser()


def nuke_expyre_results(
    expyre_results_dir: Path | None = None,
) -> None:
    """
    Delete all local ExPyRe job state (job cache, unsynced stage dirs) in the
    given directory, keeping config.json untouched. Defaults to
    resolve_default_expyre_dir() if no directory is given.
    """
    if expyre_results_dir is None:
        expyre_results_dir = resolve_default_expyre_dir()

    if not expyre_results_dir.exists():
        print(f"{expyre_results_dir} does not exist, nothing to do.")
        return

    print(
        f"Are you sure you want to delete all local ExPyRe job state in {expyre_results_dir}? "
        "This will mean you lose all unsynced data not currently in your results file. "
        "config.json will be left untouched. (y/n)"
    )
    answer = input()
    if answer.lower() != "y":
        print("Aborting.")
        return

    for item in expyre_results_dir.iterdir():
        if item.name == "config.json":
            continue
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
