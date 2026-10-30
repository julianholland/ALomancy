"""`alomancy status [--watch]`: the live remote-job panel of the run in the
current directory, read from the status file the run rewrites every few
seconds (`results/remote_status.json`, remote_submission/progress.py).
Only reads files, so it is safe to run alongside the run, and Ctrl-C
exits at once."""

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console, RenderableType
from rich.live import Live
from rich.text import Text

from alomancy.cli.cancel import (
    NotARunDirectoryError,
    find_run_process,
    require_run_directory,
)
from alomancy.remote_submission.progress import render

STATUS_PATH = Path("results") / "remote_status.json"
STALE_AFTER = 30.0  # s without an update while the run is alive


def _read(run_dir: Path) -> dict[str, Any] | None:
    try:
        snapshot: dict[str, Any] = json.loads((run_dir / STATUS_PATH).read_text())
        return snapshot
    except (OSError, json.JSONDecodeError):
        return None


def status_view(run_dir: Path, width: int = 100) -> RenderableType:
    """The panel for `run_dir`, with a note when the run isn't running or
    its status file has stopped updating."""
    snapshot = _read(run_dir)
    alive = find_run_process(run_dir / ".expyre") is not None
    if snapshot is None:
        state = "the run is running" if alive else "the run is not running"
        return Text(f"No remote jobs reported yet ({state}).")
    updated_at = float(snapshot["updated_at"])
    updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(updated_at))
    note = None
    if not alive:
        note = f"run not running (last update {updated})"
    elif time.time() - updated_at > STALE_AFTER:
        note = f"status file stale (last update {updated})"
    elif snapshot.get("finished"):
        note = "waiting for the next batch of remote jobs"
    view: RenderableType = render(snapshot, width=width, note=note)
    return view


def show_status(
    run_dir: Path,
    *,
    watch: bool = False,
    interval: float = 5.0,
    console: Console | None = None,
) -> int:
    """Print the panel once, or redraw it every `interval` s until Ctrl-C.
    Returns the exit status (2 outside a run directory)."""
    console = console or Console()
    try:
        require_run_directory(run_dir, "status")
    except NotARunDirectoryError as exc:
        console.print(f"alomancy status: {exc}", markup=False)
        return 2
    if not watch:
        console.print(status_view(run_dir, console.width))
        return 0
    try:
        with Live(
            status_view(run_dir, console.width), console=console, auto_refresh=False
        ) as live:
            while True:
                time.sleep(interval)
                live.update(status_view(run_dir, console.width), refresh=True)
    except KeyboardInterrupt:
        pass
    return 0
