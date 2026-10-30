"""Live status of the remote jobs a run is waiting on.

RemoteJobExecutor (executor.py) registers each batch of jobs with the
module-level TRACKER and updates it as jobs are submitted, synced and
finished. ProgressMonitor turns the tracker's snapshot into:

- a panel that redraws itself in the run's terminal (rich), with log lines
  scrolling above it, when stdout is a terminal and ALOMANCY_LIVE != "0";
- an INFO summary line every few minutes otherwise (nohup, redirected);
- results/remote_status.json, rewritten every few seconds, which
  `alomancy status [--watch]` renders from any terminal.

One snapshot dict feeds all three, so they always agree. "Last data" for an
HPC is the last time a sync with it (queue status + rsync of results)
succeeded; it is flagged stale after STALE_INTERVALS check intervals.
"""

import json
import logging
import os
import re
import socket
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

# Where the run writes its status; None disables the file (tests/conftest.py
# does that, so tests run from the repository never write into it). The file
# is only written when its directory already exists: a run always has one.
STATUS_FILE: Path | None = Path("results") / "remote_status.json"
STATUS_INTERVAL = 5.0  # s between status-file writes
SUMMARY_INTERVAL = 300.0  # s between summary lines without a terminal
STALE_INTERVALS = 3  # check intervals without a sync before "last data" is stale
_STATES = ("done", "running", "queued", "to_submit", "failed")
_BAR = {"done": "█", "running": "▓", "queued": "▒", "to_submit": "░"}
_BAR_STYLE = {
    "done": "green",
    "running": "yellow",
    "queued": "blue",
    "to_submit": "dim",
}
_LOOP_RE = re.compile(r"al_loop_(\d+)")


def loop_from_paths(paths: Iterable[Any]) -> int | None:
    """The AL loop a job belongs to, from its output paths
    (results/al_loop_N/...), or None."""
    for path in paths:
        match = _LOOP_RE.search(str(path))
        if match:
            return int(match.group(1))
    return None


@dataclass
class JobRow:
    index: int
    item: str
    system: str | None
    state: str = "to_submit"
    remote_id: str | None = None
    submitted_at: float | None = None
    running_since: float | None = None
    finished_at: float | None = None
    expyre_id: str | None = None


@dataclass
class Batch:
    phase: str
    loop: int | None
    check_interval: float
    jobs: list[JobRow]
    started_at: float = field(default_factory=time.time)


class JobTracker:
    """Thread-safe registry of the batches of remote jobs being waited on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batches: dict[int, Batch] = {}
        self._next_id = 0
        self.last_sync: dict[str, float] = {}
        self.last_sync_error: dict[str, str] = {}

    # --- batches ------------------------------------------------------
    def register(
        self,
        phase: str,
        jobs: list[tuple[str, str | None]],
        *,
        loop: int | None = None,
        check_interval: float = 30.0,
    ) -> int:
        """Register a batch; `jobs` is [(item, system), ...] in job order."""
        rows = [JobRow(i, item, system) for i, (item, system) in enumerate(jobs)]
        with self._lock:
            batch_id = self._next_id
            self._next_id += 1
            self._batches[batch_id] = Batch(phase, loop, check_interval, rows)
        return batch_id

    def unregister(self, batch_id: int) -> None:
        with self._lock:
            self._batches.pop(batch_id, None)

    def active(self) -> bool:
        with self._lock:
            return bool(self._batches)

    # --- job events ----------------------------------------------------
    def _row(self, batch_id: int, index: int) -> JobRow | None:
        batch = self._batches.get(batch_id)
        return batch.jobs[index] if batch and index < len(batch.jobs) else None

    def submitted(
        self,
        batch_id: int,
        index: int,
        remote_id: str | None,
        expyre_id: str | None = None,
    ) -> None:
        with self._lock:
            row = self._row(batch_id, index)
            if row is not None:
                row.state = "queued"
                row.remote_id = remote_id
                row.expyre_id = expyre_id
                row.submitted_at = time.time()
                row.running_since = None

    def finished(self, batch_id: int, index: int, ok: bool) -> None:
        with self._lock:
            row = self._row(batch_id, index)
            if row is not None:
                row.state = "done" if ok else "failed"
                row.finished_at = time.time()

    def synced(self, system: str, remote_statuses: dict[str, str]) -> None:
        """A sync with `system` succeeded; `remote_statuses` maps expyre job
        ids to expyre's remote_status (queued/held/running/done/...)."""
        now = time.time()
        with self._lock:
            self.last_sync[system] = now
            self.last_sync_error.pop(system, None)
            for batch in self._batches.values():
                for row in batch.jobs:
                    if row.system != system or row.state not in ("queued", "running"):
                        continue
                    status = remote_statuses.get(row.expyre_id or "")
                    if status in ("queued", "held"):
                        row.state = "queued"
                    elif status == "running":
                        if row.state != "running":
                            row.running_since = now
                        row.state = "running"

    def sync_failed(self, system: str, error: BaseException) -> None:
        with self._lock:
            self.last_sync_error[system] = f"{type(error).__name__}: {error}"[:200]

    # --- snapshot --------------------------------------------------------
    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        """JSON-ready state of every active batch."""
        now = time.time() if now is None else now
        with self._lock:
            batches = [
                _batch_snapshot(b, self.last_sync, self.last_sync_error, now)
                for b in self._batches.values()
            ]
        return {
            "updated_at": now,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "batches": batches,
        }


def _batch_snapshot(
    batch: Batch,
    last_sync: dict[str, float],
    last_sync_error: dict[str, str],
    now: float,
) -> dict[str, Any]:
    counts = dict.fromkeys(_STATES, 0)
    hpcs: dict[str, dict[str, Any]] = {}
    for row in batch.jobs:
        counts[row.state] += 1
        system = row.system or "local"
        hpc = hpcs.setdefault(
            system, {**dict.fromkeys(_STATES, 0), "oldest_queued_s": None}
        )
        hpc[row.state] += 1
        if row.state == "queued" and row.submitted_at is not None:
            waited = now - row.submitted_at
            if hpc["oldest_queued_s"] is None or waited > hpc["oldest_queued_s"]:
                hpc["oldest_queued_s"] = waited
    for system, hpc in hpcs.items():
        synced = last_sync.get(system)
        hpc["last_data_s"] = None if synced is None else now - synced
        waiting = hpc["queued"] + hpc["running"]
        hpc["stale"] = bool(
            waiting
            and (
                (
                    synced is None
                    and now - batch.started_at > STALE_INTERVALS * batch.check_interval
                )
                or (
                    synced is not None
                    and now - synced > STALE_INTERVALS * batch.check_interval
                )
            )
        )
        hpc["error"] = last_sync_error.get(system)
    running = [r for r in batch.jobs if r.state == "running" and r.running_since]
    slowest = max(running, key=lambda r: now - (r.running_since or now), default=None)
    return {
        "phase": batch.phase,
        "loop": batch.loop,
        "total": len(batch.jobs),
        "started_at": batch.started_at,
        "check_interval": batch.check_interval,
        "counts": counts,
        "hpcs": hpcs,
        "slowest_running": None
        if slowest is None
        else {
            "item": slowest.item,
            "system": slowest.system,
            "remote_id": slowest.remote_id,
            "running_s": now - (slowest.running_since or now),
        },
    }


TRACKER = JobTracker()


# --- rendering ---------------------------------------------------------------


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} m {seconds % 60:02d} s"
    if seconds < 86400:
        return f"{seconds // 3600} h {seconds // 60 % 60:02d} m"
    return f"{seconds // 86400} d {seconds // 3600 % 24:02d} h"


def bar_segments(counts: dict[str, int], width: int) -> dict[str, int]:
    """Characters per state for a bar of `width`; failed jobs count as done
    (they are finished), and every non-empty state gets at least one."""
    shown = {s: counts.get(s, 0) for s in _BAR}
    shown["done"] += counts.get("failed", 0)
    total = sum(shown.values())
    if total == 0 or width <= 0:
        return dict.fromkeys(_BAR, 0)
    raw = {s: n * width / total for s, n in shown.items()}
    seg = {s: (max(1, int(v)) if shown[s] else 0) for s, v in raw.items()}
    # Hand out (or take back) the rounding remainder by largest fraction.
    order = sorted(_BAR, key=lambda s: raw[s] - int(raw[s]), reverse=True)
    diff = width - sum(seg.values())
    i = 0
    while diff != 0 and i < 10 * width:
        state = order[i % len(order)]
        if diff > 0 and shown[state]:
            seg[state] += 1
            diff -= 1
        elif diff < 0 and seg[state] > 1:
            seg[state] -= 1
            diff += 1
        i += 1
    return seg


def _batch_renderable(batch: dict[str, Any], width: int) -> RenderableType:
    counts = batch["counts"]
    loop = "" if batch.get("loop") is None else f" · loop {batch['loop']}"
    header = Text(
        f"── {batch['phase']}{loop} ── {batch['total']} jobs ──", style="bold"
    )

    bar = Text()
    bar_width = max(10, min(50, width - 78))
    for state, n in bar_segments(counts, bar_width).items():
        bar.append(_BAR[state] * n, style=_BAR_STYLE[state])
    summary = (
        f"  done {counts['done']} · running {counts['running']} · "
        f"queued {counts['queued']} · to submit {counts['to_submit']}"
    )
    bar.append(summary)
    if counts["failed"]:
        bar.append(f" · failed {counts['failed']}", style="red")

    table = Table(box=None, padding=(0, 1), show_edge=False)
    for column in ("HPC", "done", "run", "queue", "fail", "last data", "oldest queued"):
        table.add_column(column, justify="left" if column == "HPC" else "right")
    for system, hpc in sorted(batch["hpcs"].items()):
        last = format_duration(hpc["last_data_s"])
        last = "never" if hpc["last_data_s"] is None else f"{last} ago"
        last_text = Text(last)
        if hpc["stale"]:
            last_text = Text(f"{last} ⚠", style="yellow")
        if hpc.get("error"):
            last_text.append(" (sync failing)", style="red")
        table.add_row(
            system,
            str(hpc["done"]),
            str(hpc["running"]),
            str(hpc["queued"]),
            str(hpc["failed"]),
            last_text,
            format_duration(hpc["oldest_queued_s"]),
        )

    parts: list[RenderableType] = [header, bar, Text(""), table]
    slowest = batch.get("slowest_running")
    if slowest:
        parts.append(
            Text(
                f" slowest running: {slowest['item']} ({slowest['system']}, "
                f"{slowest['remote_id']}) {format_duration(slowest['running_s'])}"
            )
        )
    return Group(*parts)


def render(
    snapshot: dict[str, Any], *, width: int = 100, note: str | None = None
) -> RenderableType:
    """The status panel for a TRACKER snapshot (live or read from the
    status file)."""
    parts: list[RenderableType] = []
    for batch in snapshot.get("batches", []):
        parts.extend([_batch_renderable(batch, width), Text("")])
    if not snapshot.get("batches"):
        parts.append(Text("No remote jobs running."))
    updated = datetime.fromtimestamp(snapshot["updated_at"]).strftime("%H:%M:%S")
    footer = Text(f" updated {updated}", style="dim")
    if note:
        footer.append(f" · {note}", style="yellow")
    parts.append(footer)
    return Group(*parts)


def summary_line(snapshot: dict[str, Any]) -> str:
    """One line per snapshot, for logs without a terminal."""
    lines = []
    for batch in snapshot.get("batches", []):
        c = batch["counts"]
        loop = "" if batch.get("loop") is None else f" loop {batch['loop']}"
        hpcs = ", ".join(
            f"{system}: last data "
            + (
                "never"
                if hpc["last_data_s"] is None
                else f"{format_duration(hpc['last_data_s'])} ago"
            )
            + (
                ""
                if hpc["oldest_queued_s"] is None
                else f", oldest queued {format_duration(hpc['oldest_queued_s'])}"
            )
            + (" (stale)" if hpc["stale"] else "")
            for system, hpc in sorted(batch["hpcs"].items())
        )
        lines.append(
            f"{batch['phase']}{loop}: {c['done']} done, {c['running']} running, "
            f"{c['queued']} queued, {c['to_submit']} to submit, {c['failed']} failed"
            f" | {hpcs}"
        )
    return "; ".join(lines)


def write_status(snapshot: dict[str, Any], path: Path | None = None) -> None:
    """Atomically replace the status file (if its directory exists)."""
    path = STATUS_FILE if path is None else path
    if path is None or not path.parent.is_dir():
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(snapshot, indent=1))
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not write %s: %s", path, exc)
        tmp.unlink(missing_ok=True)


# --- the monitor ---------------------------------------------------------------


def live_enabled() -> bool:
    return os.environ.get("ALOMANCY_LIVE", "1") != "0" and sys.stdout.isatty()


class ProgressMonitor:
    """While active: redraw the live panel (terminal) or log a summary every
    `summary_every` s (no terminal), and rewrite the status file every
    `status_every` s. Nested monitors (a batch started while another runs)
    share the outermost one's panel, which renders every batch."""

    _depth = 0
    _depth_lock = threading.Lock()

    def __init__(
        self,
        tracker: JobTracker = TRACKER,
        *,
        live: bool | None = None,
        tick: float = 1.0,
        status_every: float = STATUS_INTERVAL,
        summary_every: float = SUMMARY_INTERVAL,
        status_file: Path | None = None,
    ):
        self.tracker = tracker
        self.live = live
        self.tick = tick
        self.status_every = status_every
        self.summary_every = summary_every
        self.status_file = status_file
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._restore: list[tuple[logging.StreamHandler, Any]] = []
        self._outer = False

    def __enter__(self) -> "ProgressMonitor":
        with ProgressMonitor._depth_lock:
            self._outer = ProgressMonitor._depth == 0
            ProgressMonitor._depth += 1
        if not self._outer:
            return self
        live = live_enabled() if self.live is None else self.live
        if live:
            self._start_live()
        write_status(self.tracker.snapshot(), self.status_file)
        self._thread = threading.Thread(
            target=self._loop, args=(live,), name="alomancy-progress", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        with ProgressMonitor._depth_lock:
            ProgressMonitor._depth -= 1
        if not self._outer:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._live is not None:
            self._live.update(render(self.tracker.snapshot()))
            self._live.stop()
            for handler, stream in self._restore:
                handler.setStream(stream)
        final = self.tracker.snapshot()
        final["finished"] = True
        write_status(final, self.status_file)

    def _start_live(self) -> None:
        real_stdout = sys.stdout
        self._live = Live(
            render(self.tracker.snapshot()),
            refresh_per_second=1,
            redirect_stdout=True,
            redirect_stderr=True,
            transient=False,
        )
        self._live.start()
        # alomancy's console handler holds the real stdout; point it at
        # rich's proxy so log lines print above the panel.
        for handler in logging.getLogger("alomancy").handlers:
            if type(handler) is logging.StreamHandler and (
                handler.stream is real_stdout or handler.stream is sys.__stdout__
            ):
                self._restore.append((handler, handler.stream))
                handler.setStream(sys.stdout)

    def _loop(self, live: bool) -> None:
        last_status = last_summary = time.monotonic()
        while not self._stop.wait(self.tick):
            snapshot = self.tracker.snapshot()
            now = time.monotonic()
            if self._live is not None:
                width = self._live.console.width
                self._live.update(render(snapshot, width=width))
            elif not live and now - last_summary >= self.summary_every:
                logger.info("Remote jobs: %s", summary_line(snapshot))
                last_summary = now
            if now - last_status >= self.status_every:
                write_status(snapshot, self.status_file)
                last_status = now
