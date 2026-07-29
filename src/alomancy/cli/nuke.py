import shutil
from pathlib import Path


def nuke_expyre_results(
    expyre_results_dir: Path = Path("~/.expyre").expanduser(),
) -> None:
    """
    Delete all local ExPyRe job state (job cache, unsynced stage dirs) in the
    given directory, keeping config.json untouched.
    """
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
