"""Live status of a run: what it is, what it is doing, and its remote jobs.

The run (core/entry.py's ALomancy.run) holds one ProgressMonitor for its
whole length; RemoteJobExecutor (executor.py) registers each batch of jobs
with the module-level TRACKER and updates it as jobs are submitted, synced
and finished, and the workflow tells it the run's settings, loop, phase and
best model. ProgressMonitor turns the tracker's snapshot into:

- two panels that redraw themselves in the run's terminal (rich), with log
  lines scrolling above them, when stdout is a terminal and
  ALOMANCY_LIVE != "0": "ALomancy" (Run Details, Settings, Best Model) and
  "Current task" (Errors, Status, Options). Keys: L logs in again to an
  HPC whose ssh login was lost (terminal_handoff), ? shows help;
- an INFO summary line every few minutes otherwise (nohup, redirected);
- results/remote_status.json, rewritten every few seconds, which
  `alomancy status [--watch]` renders from any terminal.

One snapshot dict feeds all three, so they always agree. "Last data" for an
HPC is the last time a sync with it (queue status + rsync of results)
succeeded; it is flagged stale after STALE_INTERVALS check intervals.

While the run owns its ssh masters (utils/ssh_master.py), the monitor also
checks each one every SSH_WATCH_INTERVAL s, reopens a key-auth host on its
own and raises the login banner as soon as a password/OTP host's master
dies, before any poll fails.
"""

import io
import json
import logging
import os
import re
import select
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from alomancy.analysis.palette import ALOMANCY_PURPLE, PALETTE, PENDING_COLOR

logger = logging.getLogger(__name__)

# Where the run writes its status; None disables the file (tests/conftest.py
# does that, so tests run from the repository never write into it). The file
# is only written when its directory already exists: a run always has one.
STATUS_FILE: Path | None = Path("results") / "remote_status.json"
STATUS_INTERVAL = 5.0  # s between status-file writes
SUMMARY_INTERVAL = 300.0  # s between summary lines without a terminal
STALE_INTERVALS = 3  # check intervals without a sync before "last data" is stale
SSH_WATCH_INTERVAL = 30.0  # s between checks of the run's ssh masters
FINISHED_KEEP = 300.0  # s a finished batch stays on the panel (one line)
_STATES = ("done", "running", "queued", "to_submit", "failed")
_BAR = {"done": "█", "running": "▓", "queued": "▒", "to_submit": "░"}
# The ALomancy colour scheme (analysis/palette.py), as in the plots.
_CYAN, _EMERALD, _AMBER, _ROSE = PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]
_BAR_STYLE = {
    "done": ALOMANCY_PURPLE,
    "running": _CYAN,
    "queued": _AMBER,
    "to_submit": PENDING_COLOR,
}
_FAILED_STYLE = _ROSE
_WARNING_STYLE = _AMBER
_OK_STYLE = _EMERALD
_TITLE_STYLE = f"bold {ALOMANCY_PURPLE}"
_LABEL_STYLE = f"bold {PENDING_COLOR}"
_KEY_STYLE = f"bold {_CYAN}"
_LOOP_RE = re.compile(r"al_loop_(\d+)")
# ssh states shown per HPC
SSH_OK = "ok"
SSH_LOGIN_NEEDED = "login needed"


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
    """Thread-safe registry of the run's state and the batches of remote
    jobs being waited on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batches: dict[int, Batch] = {}
        self._next_id = 0
        self.last_sync: dict[str, float] = {}
        self.last_sync_error: dict[str, str] = {}
        # system -> (since, ssh host) while ssh to it can't log in
        self.auth_lost: dict[str, tuple[float, str]] = {}
        # system -> SSH_OK / SSH_LOGIN_NEEDED (from the master watch)
        self.ssh_state: dict[str, str] = {}
        self.finished_batches: list[dict[str, Any]] = []
        # The run: static settings (set_run_info), loop, phase, best model.
        self.run_info: dict[str, Any] | None = None
        self.loop: dict[str, Any] | None = None
        self.phase: dict[str, Any] | None = None
        self.best_model: dict[str, Any] | None = None

    def reset(self) -> None:
        """Forget everything (tests; a new run in the same process)."""
        self.__init__()  # type: ignore[misc]

    # --- the run ---------------------------------------------------------
    def set_run_info(self, info: dict[str, Any]) -> None:
        with self._lock:
            self.run_info = {"started_at": time.time(), **info}

    def set_loop(
        self,
        loop: int | None,
        *,
        final: bool = False,
        started_at: float | None = None,
    ) -> None:
        with self._lock:
            self.loop = (
                None
                if loop is None
                else {
                    "loop": loop,
                    "final": final,
                    "started_at": time.time() if started_at is None else started_at,
                }
            )

    def set_phase(self, phase: str | None) -> None:
        with self._lock:
            self.phase = (
                None if phase is None else {"name": phase, "started_at": time.time()}
            )

    def set_best_model(self, metadata: dict[str, Any] | None) -> None:
        with self._lock:
            self.best_model = metadata

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
        now = time.time()
        with self._lock:
            batch = self._batches.pop(batch_id, None)
            if batch is None:
                return
            done = sum(r.state == "done" for r in batch.jobs)
            failed = sum(r.state == "failed" for r in batch.jobs)
            self.finished_batches.append(
                {
                    "phase": batch.phase,
                    "loop": batch.loop,
                    "total": len(batch.jobs),
                    "done": done,
                    "failed": failed,
                    "duration_s": now - batch.started_at,
                    "finished_at": now,
                }
            )

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

    def synced(self, system: str, remote_statuses: dict[str, str]) -> bool:
        """A sync with `system` succeeded; `remote_statuses` maps expyre job
        ids to expyre's remote_status (queued/held/running/done/...).
        Returns whether this ends an ssh login outage (mark_auth_lost)."""
        now = time.time()
        with self._lock:
            self.last_sync[system] = now
            self.last_sync_error.pop(system, None)
            restored = self.auth_lost.pop(system, None) is not None
            if system in self.ssh_state:
                self.ssh_state[system] = SSH_OK
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
        return restored

    def sync_failed(self, system: str, error: BaseException) -> None:
        with self._lock:
            self.last_sync_error[system] = f"{type(error).__name__}: {error}"[:200]

    def mark_auth_lost(self, system: str, host: str) -> bool:
        """ssh to `system` can't log in (its master connection is gone);
        shown until the next successful sync or login. Returns whether this
        starts a new outage, so the caller warns once per outage."""
        with self._lock:
            self.ssh_state[system] = SSH_LOGIN_NEEDED
            if system in self.auth_lost:
                return False
            self.auth_lost[system] = (time.time(), host)
            return True

    def auth_restored(self, system: str) -> bool:
        """ssh to `system` works again (a login or a reopened master).
        Returns whether this ends an outage."""
        with self._lock:
            self.ssh_state[system] = SSH_OK
            return self.auth_lost.pop(system, None) is not None

    def set_ssh_ok(self, system: str) -> None:
        with self._lock:
            if system not in self.auth_lost:
                self.ssh_state[system] = SSH_OK

    # --- snapshot --------------------------------------------------------
    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        """JSON-ready state of the run and every active batch."""
        now = time.time() if now is None else now
        with self._lock:
            batches = [
                _batch_snapshot(b, self.last_sync, self.last_sync_error, now)
                for b in self._batches.values()
            ]
            auth_lost = {
                system: {"host": host, "since_s": now - since}
                for system, (since, host) in self.auth_lost.items()
            }
            self.finished_batches = [
                b
                for b in self.finished_batches
                if now - b["finished_at"] < FINISHED_KEEP
            ]
            finished = [dict(b) for b in self.finished_batches]
            run = _run_snapshot(
                self.run_info, self.loop, self.phase, self.best_model, now
            )
            ssh = dict(self.ssh_state)
        return {
            "updated_at": now,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "run": run,
            "batches": batches,
            "finished_batches": finished,
            "auth_lost": auth_lost,
            "ssh": ssh,
        }


def _run_snapshot(
    info: dict[str, Any] | None,
    loop: dict[str, Any] | None,
    phase: dict[str, Any] | None,
    best_model: dict[str, Any] | None,
    now: float,
) -> dict[str, Any] | None:
    if info is None and loop is None and phase is None:
        return None
    run: dict[str, Any] = dict(info or {})
    if "started_at" in run:
        run["elapsed_s"] = now - run["started_at"]
    run["loop"] = (
        None
        if loop is None
        else {
            "loop": loop["loop"],
            "final": loop["final"],
            "elapsed_s": now - loop["started_at"],
        }
    )
    run["phase"] = (
        None
        if phase is None
        else {"name": phase["name"], "elapsed_s": now - phase["started_at"]}
    )
    run["best_model"] = best_model
    return run


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
        "elapsed_s": now - batch.started_at,
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


def section(title: str) -> Rule:
    """A sub-section heading: ``── Title ─────``, the title in the brand
    violet, the line dim."""
    heading = Text("── ", style="dim")
    heading.append(title, style=_TITLE_STYLE)
    return Rule(heading, align="left", characters="─", style="dim")


def _label_table() -> Table:
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style=_LABEL_STYLE, no_wrap=True)
    table.add_column()
    return table


def _loop_label(batch_loop: int | None) -> str:
    return "" if batch_loop is None else f" · loop {batch_loop}"


def batch_header(batch: dict[str, Any]) -> Text:
    return Text(
        f"{batch['phase']}{_loop_label(batch.get('loop'))} · {batch['total']} jobs",
        style=_TITLE_STYLE,
    )


def batch_bar(batch: dict[str, Any], width: int) -> Text:
    counts = batch["counts"]
    bar = Text()
    bar_width = max(10, min(50, width - 80))
    for state, n in bar_segments(counts, bar_width).items():
        bar.append(_BAR[state] * n, style=_BAR_STYLE[state])
    # Each label in its segment's colour, so the summary is the legend.
    for i, (state, label) in enumerate(
        (
            ("done", "done"),
            ("running", "running"),
            ("queued", "queued"),
            ("to_submit", "to submit"),
        )
    ):
        bar.append("  " if i == 0 else " · ")
        bar.append(label, style=_BAR_STYLE[state])
        bar.append(f" {counts[state]}")
    if counts["failed"]:
        bar.append(" · ")
        bar.append(f"failed {counts['failed']}", style=_FAILED_STYLE)
    return bar


def _ssh_cell(system: str, snapshot: dict[str, Any]) -> Text:
    lost = sorted((snapshot.get("auth_lost") or {}).keys())
    if system in lost:
        return Text(
            f"{SSH_LOGIN_NEEDED} [{lost.index(system) + 1}]", style=_FAILED_STYLE
        )
    state = (snapshot.get("ssh") or {}).get(system)
    if state == SSH_OK:
        return Text(SSH_OK, style=_OK_STYLE)
    return Text("-", style="dim")


def _batch_renderable(
    batch: dict[str, Any], width: int, snapshot: dict[str, Any] | None = None
) -> RenderableType:
    snapshot = snapshot or {}
    table = Table(box=None, padding=(0, 1), show_edge=False, header_style="bold")
    for column in (
        "HPC",
        "ssh",
        "done",
        "run",
        "queue",
        "fail",
        "last data",
        "oldest queued",
    ):
        table.add_column(
            column, justify="left" if column in ("HPC", "ssh") else "right"
        )
    for system, hpc in sorted(batch["hpcs"].items()):
        last = format_duration(hpc["last_data_s"])
        last = "never" if hpc["last_data_s"] is None else f"{last} ago"
        last_text = Text(last)
        if hpc["stale"]:
            last_text = Text(f"{last} !", style=_WARNING_STYLE)
        if hpc.get("error"):
            last_text.append(" (sync failing)", style=_FAILED_STYLE)
        table.add_row(
            system,
            _ssh_cell(system, snapshot),
            str(hpc["done"]),
            str(hpc["running"]),
            str(hpc["queued"]),
            Text(str(hpc["failed"]), style=_FAILED_STYLE if hpc["failed"] else ""),
            last_text,
            format_duration(hpc["oldest_queued_s"]),
        )

    parts: list[RenderableType] = [batch_header(batch), batch_bar(batch, width), table]
    slowest = batch.get("slowest_running")
    if slowest:
        parts.append(
            Text(
                f" slowest running: {slowest['item']} ({slowest['system']}, "
                f"{slowest['remote_id']}) {format_duration(slowest['running_s'])}",
                style="dim",
            )
        )
    return Group(*parts)


def _best_model_lines(best: dict[str, Any] | None) -> Table:
    table = _label_table()
    if not best:
        table.add_row("model", Text("none yet", style="dim"))
        return table
    trainer = best.get("trainer")
    split = best.get("selected_on_split")
    model = f"loop {best.get('al_loop')} · fit_{best.get('fit_idx')}"
    if trainer:
        model += f" · {display_name(str(trainer))}"
    if split:
        model += f" · selected on {split}"
    table.add_row("model", model)
    errors = (best.get("errors") or {}).get(split or "", {})
    parts = []
    if errors.get("mae_e_per_atom") is not None:
        parts.append(f"E {1000 * errors['mae_e_per_atom']:.1f} meV/atom")
    if errors.get("mae_f") is not None:
        parts.append(f"F {1000 * errors['mae_f']:.0f} meV/Å")
    if errors.get("mae_stress") is not None:
        parts.append(f"S {1000 * errors['mae_stress']:.1f} meV/Å³")
    if parts:
        table.add_row(f"MAE ({split})", "   ".join(parts))
    return table


def display_name(name: str) -> str:
    """Short module names read as acronyms (mace → MACE, qe → QE, md → MD);
    longer ones are shown as configured."""
    return name.upper() if len(name) <= 4 else name


def _run_panel(run: dict[str, Any], width: int) -> Panel:
    details = _label_table()
    if run.get("name"):
        where = run["name"]
        if run.get("directory"):
            where += f" · {run['directory']}"
        details.add_row("run", where)
    if run.get("workflow"):
        details.add_row("workflow", run["workflow"])
    loop = run.get("loop")
    total = run.get("num_of_al_loops")
    loop_text = Text()
    if loop is None:
        loop_text.append("preparing (initial training set)")
    elif loop.get("final"):
        loop_text.append("final training")
    else:
        loop_text.append(f"{loop['loop']} / {total}" if total else str(loop["loop"]))
    if loop is not None:
        loop_text.append("   loop time ", style="dim")
        loop_text.append(format_duration(loop["elapsed_s"]))
    if run.get("elapsed_s") is not None:
        loop_text.append("   run time ", style="dim")
        loop_text.append(format_duration(run["elapsed_s"]))
    details.add_row("loop", loop_text)
    if run.get("elements"):
        details.add_row("elements", ", ".join(run["elements"]))

    parts: list[RenderableType] = [section("Run Details"), details]
    settings = run.get("settings") or []
    if settings and width >= 80:
        table = Table(
            box=box.SIMPLE_HEAD,
            show_edge=False,
            pad_edge=False,
            header_style=_TITLE_STYLE,
            border_style="dim",
            expand=True,
        )
        table.add_column("Block", style=_LABEL_STYLE, no_wrap=True)
        table.add_column("Information", ratio=1)
        table.add_column("HPC", no_wrap=True)
        for row in settings:
            table.add_row(row["block"], row["information"], row.get("hpc") or "")
        parts += [Text(""), section("Settings"), table]
    parts += [Text(""), section("Best Model"), _best_model_lines(run.get("best_model"))]
    return Panel(
        Group(*parts),
        title=Text("ALomancy", style=_TITLE_STYLE),
        title_align="left",
        border_style=ALOMANCY_PURPLE,
        box=box.ROUNDED,
    )


def error_lines(snapshot: dict[str, Any]) -> list[Text]:
    """What is going wrong right now: lost ssh logins, failing syncs and
    failed jobs."""
    lines = [
        Text(f"! {line}", style=f"bold {_FAILED_STYLE}")
        for line in auth_lost_lines(snapshot)
    ]
    lost = set((snapshot.get("auth_lost") or {}).keys())
    failing: dict[str, str] = {}
    for batch in snapshot.get("batches", []):
        for system, hpc in batch["hpcs"].items():
            if hpc.get("error") and system not in lost:
                failing.setdefault(system, hpc["error"])
        if batch["counts"]["failed"]:
            lines.append(
                Text(
                    f"! {batch['counts']['failed']} of {batch['total']} jobs failed in "
                    f"{batch['phase']}{_loop_label(batch.get('loop'))} "
                    "(see results/alomancy.log)",
                    style=_FAILED_STYLE,
                )
            )
    for system, error in sorted(failing.items()):
        lines.append(
            Text(f"! sync with {system} failing: {error}", style=_WARNING_STYLE)
        )
    return lines


def _status_line(run: dict[str, Any] | None, snapshot: dict[str, Any]) -> Text:
    line = Text()
    phase = (run or {}).get("phase")
    loop = (run or {}).get("loop")
    if phase:
        line.append(phase["name"], style=_TITLE_STYLE)
        if loop is not None:
            line.append(
                " · final training" if loop.get("final") else f" · loop {loop['loop']}"
            )
        line.append(f" · running {format_duration(phase['elapsed_s'])}")
    elif run is not None and loop is None:
        line.append("preparing the run", style=_TITLE_STYLE)
    elif run is not None:
        line.append("between steps", style=_TITLE_STYLE)
    updated = datetime.fromtimestamp(snapshot["updated_at"]).strftime("%H:%M:%S")
    if line.plain:
        line.append(" · ")
    line.append(f"updated {updated}", style="dim")
    return line


def _options_lines(snapshot: dict[str, Any], keys: bool) -> Text:
    lost = sorted((snapshot.get("auth_lost") or {}).keys())
    text = Text()
    if keys:
        if lost:
            text.append("[L]", style=_KEY_STYLE)
            text.append(f" log in to {lost[0]}   ")
            if len(lost) > 1:
                text.append(f"[1-{min(9, len(lost))}]", style=_KEY_STYLE)
                text.append(" log in to that HPC   ")
        text.append("[?]", style=_KEY_STYLE)
        text.append(" help")
        if lost:
            text.append("   (from another terminal: alomancy login)", style="dim")
    elif lost:
        text.append("log in again: ")
        text.append("alomancy login", style=_KEY_STYLE)
        text.append(" in the run directory")
    else:
        text.append("nothing to do", style="dim")
    return text


HELP_TEXT = (
    "L  log in again to the first HPC whose ssh login was lost\n"
    "1-9  log in again to that HPC (numbers as in the ssh column)\n"
    "?  show or hide this help\n"
    "Ctrl-C  stop the run (a restart picks up where it stopped)"
)


def _task_panel(
    snapshot: dict[str, Any], width: int, note: str | None, keys: bool, show_help: bool
) -> Panel:
    run = snapshot.get("run")
    parts: list[RenderableType] = []
    errors = error_lines(snapshot)
    if errors:
        parts += [section("Errors"), *errors, Text("")]
    parts += [section("Status"), _status_line(run, snapshot)]
    batches = snapshot.get("batches", [])
    for batch in batches:
        parts += [Text(""), _batch_renderable(batch, width, snapshot)]
    if not batches:
        parts.append(
            Text(
                "running locally" if run is not None else "No remote jobs running.",
                style="dim",
            )
        )
    finished = snapshot.get("finished_batches") or []
    if finished:
        parts.append(Text(""))
    for done in finished:
        line = Text("✓ ", style=_OK_STYLE)
        line.append(
            f"{done['phase']}{_loop_label(done.get('loop'))} · "
            f"{done['done']}/{done['total']} done in {format_duration(done['duration_s'])}"
        )
        if done["failed"]:
            line.append(f" · {done['failed']} failed", style=_FAILED_STYLE)
        parts.append(line)
    if note:
        parts.append(Text(note, style=_WARNING_STYLE))
    parts += [Text(""), section("Options"), _options_lines(snapshot, keys)]
    if show_help:
        parts.append(Text(HELP_TEXT, style="dim"))
    return Panel(
        Group(*parts),
        title=Text("Current task", style=_TITLE_STYLE),
        title_align="left",
        border_style=ALOMANCY_PURPLE,
        box=box.ROUNDED,
    )


def render(
    snapshot: dict[str, Any],
    *,
    width: int = 100,
    note: str | None = None,
    keys: bool = False,
    show_help: bool = False,
) -> RenderableType:
    """The two panels for a TRACKER snapshot (live or read from the status
    file). `keys`: the run's own terminal, where the key bindings work."""
    parts: list[RenderableType] = []
    run = snapshot.get("run")
    if run is not None:
        parts.append(_run_panel(run, width))
    parts.append(_task_panel(snapshot, width, note, keys, show_help))
    return Group(*parts)


def auth_lost_lines(snapshot: dict[str, Any]) -> list[str]:
    """One line per HPC whose ssh login was lost, saying what to do."""
    return [
        f"ssh to {system} needs a login ({format_duration(info['since_s'])}): "
        "press L in the run's terminal or run `alomancy login` in the run "
        "directory; jobs keep waiting."
        for system, info in sorted((snapshot.get("auth_lost") or {}).items())
    ]


def summary_line(snapshot: dict[str, Any]) -> str:
    """One line per snapshot, for logs without a terminal."""
    lines = list(auth_lost_lines(snapshot))
    run = snapshot.get("run") or {}
    head = []
    if run.get("loop") is not None:
        loop = run["loop"]
        total = run.get("num_of_al_loops")
        head.append(
            "final training"
            if loop.get("final")
            else f"loop {loop['loop']}" + (f"/{total}" if total else "")
        )
    if run.get("phase"):
        head.append(
            f"{run['phase']['name']} {format_duration(run['phase']['elapsed_s'])}"
        )
    best = run.get("best_model") or {}
    split = best.get("selected_on_split")
    mae_f = ((best.get("errors") or {}).get(split or "") or {}).get("mae_f")
    if mae_f is not None:
        head.append(f"best mae_f {mae_f:.3f}")
    if head:
        lines.append(" · ".join(head))
    for batch in snapshot.get("batches", []):
        c = batch["counts"]
        loop_text = "" if batch.get("loop") is None else f" loop {batch['loop']}"
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
            f"{batch['phase']}{loop_text}: {c['done']} done, {c['running']} running, "
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


# --- keys ------------------------------------------------------------------------


def handle_key(key: str, snapshot: dict[str, Any]) -> tuple[str, str | None]:
    """What a key press does: ("login", system), ("help", None) or
    ("none", None)."""
    lost = sorted((snapshot.get("auth_lost") or {}).keys())
    if key in ("l", "L"):
        return ("login", lost[0]) if lost else ("none", None)
    if key.isdigit() and key != "0":
        index = int(key) - 1
        return ("login", lost[index]) if index < len(lost) else ("none", None)
    if key == "?":
        return ("help", None)
    return ("none", None)


# --- the monitor ---------------------------------------------------------------


def live_enabled() -> bool:
    return os.environ.get("ALOMANCY_LIVE", "1") != "0" and sys.stdout.isatty()


def _keys_enabled() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


class ProgressMonitor:
    """While active: redraw the live panels (terminal) or log a summary
    every `summary_every` s (no terminal), rewrite the status file every
    `status_every` s and, while the run owns ssh masters, watch them.
    Nested monitors (a batch started while another runs) share the
    outermost one's panels.

    With `defer_live`, the panels first appear once the run has told the
    tracker about itself (TRACKER.set_run_info, end of pre_run_checks) or a
    batch starts, so the start-of-run ssh logins reach a plain terminal."""

    _depth = 0
    _depth_lock = threading.Lock()
    _active: "ProgressMonitor | None" = None

    def __init__(
        self,
        tracker: JobTracker = TRACKER,
        *,
        live: bool | None = None,
        tick: float = 1.0,
        status_every: float = STATUS_INTERVAL,
        summary_every: float = SUMMARY_INTERVAL,
        status_file: Path | None = None,
        defer_live: bool = False,
        watch_ssh: bool = False,
    ):
        self.tracker = tracker
        self.live = live
        self.tick = tick
        self.status_every = status_every
        self.summary_every = summary_every
        self.status_file = status_file
        self.defer_live = defer_live
        self.watch_ssh = watch_ssh
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._key_thread: threading.Thread | None = None
        self._watch_thread: threading.Thread | None = None
        self._live: Live | None = None
        self._want_live = False
        self._keys = False
        self._show_help = False
        self._restore: list[tuple[logging.StreamHandler, Any]] = []
        self._outer = False
        self._render_lock = threading.RLock()
        self._paused = threading.Event()
        self._keys_idle = threading.Event()
        self._term: tuple[int, Any] | None = None
        self._buffer: io.StringIO | None = None

    # --- lifecycle -------------------------------------------------------
    def __enter__(self) -> "ProgressMonitor":
        with ProgressMonitor._depth_lock:
            self._outer = ProgressMonitor._depth == 0
            ProgressMonitor._depth += 1
            if self._outer:
                ProgressMonitor._active = self
        if not self._outer:
            # The outer monitor may still be waiting for its first content.
            outer = ProgressMonitor._active
            if outer is not None:
                outer._maybe_start_live()
            return self
        self._want_live = live_enabled() if self.live is None else self.live
        if self._want_live and not self.defer_live:
            self._start_live()
        write_status(self.tracker.snapshot(), self.status_file)
        self._thread = threading.Thread(
            target=self._loop, name="alomancy-progress", daemon=True
        )
        self._thread.start()
        if self.watch_ssh:
            self._watch_thread = threading.Thread(
                target=self._watch_loop, name="alomancy-ssh-watch", daemon=True
            )
            self._watch_thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        with ProgressMonitor._depth_lock:
            ProgressMonitor._depth -= 1
            if self._outer:
                ProgressMonitor._active = None
        if not self._outer:
            return
        self._stop.set()
        for thread in (self._thread, self._key_thread, self._watch_thread):
            if thread is not None:
                thread.join(timeout=5)
        with self._render_lock:
            if self._live is not None:
                self._live.update(self._render(self.tracker.snapshot()))
                self._live.stop()
                self._restore_handlers()
                self._live = None
        self._restore_terminal()
        final = self.tracker.snapshot()
        final["finished"] = True
        write_status(final, self.status_file)

    # --- live panel ------------------------------------------------------
    def _render(self, snapshot: dict[str, Any]) -> RenderableType:
        width = self._live.console.width if self._live is not None else 100
        return render(snapshot, width=width, keys=self._keys, show_help=self._show_help)

    def _maybe_start_live(self) -> None:
        if (
            self._want_live
            and self._live is None
            and not self._paused.is_set()
            and (self.tracker.run_info is not None or self.tracker.active())
        ):
            self._start_live()

    def _start_live(self) -> None:
        with self._render_lock:
            if self._live is not None:
                return
            self._keys = _keys_enabled()
            real_stdout = sys.stdout
            self._live = Live(
                self._render(self.tracker.snapshot()),
                refresh_per_second=1,
                redirect_stdout=True,
                redirect_stderr=True,
                transient=False,
            )
            self._live.start()
            # alomancy's console handler holds the real stdout; point it at
            # rich's proxy so log lines print above the panels.
            self._restore = []
            for handler in logging.getLogger("alomancy").handlers:
                if type(handler) is logging.StreamHandler and (
                    handler.stream is real_stdout or handler.stream is sys.__stdout__
                ):
                    self._restore.append((handler, handler.stream))
                    handler.setStream(sys.stdout)
        if self._keys and self._key_thread is None:
            self._key_thread = threading.Thread(
                target=self._key_loop, name="alomancy-keys", daemon=True
            )
            self._key_thread.start()

    def _restore_handlers(self) -> None:
        for handler, stream in self._restore:
            handler.setStream(stream)

    def _loop(self) -> None:
        last_status = last_summary = time.monotonic()
        while not self._stop.wait(self.tick):
            snapshot = self.tracker.snapshot()
            now = time.monotonic()
            self._maybe_start_live()
            with self._render_lock:
                if self._live is not None and not self._paused.is_set():
                    self._live.update(self._render(snapshot))
            if (
                not self._want_live
                and now - last_summary >= self.summary_every
                and (snapshot["batches"] or snapshot["auth_lost"])
            ):
                logger.info("Remote jobs: %s", summary_line(snapshot))
                last_summary = now
            if now - last_status >= self.status_every:
                write_status(snapshot, self.status_file)
                last_status = now

    # --- handing the terminal to an interactive ssh ------------------------
    @contextmanager
    def _handoff(self) -> Iterator[None]:
        """Stop the panels, hold back log lines and give the terminal its
        normal mode, so an interactive ssh prompt is seen and answered;
        then restore everything."""
        in_key_thread = threading.current_thread() is self._key_thread
        self._paused.set()
        if self._key_thread is not None and not in_key_thread:
            self._keys_idle.wait(timeout=2.0)
        with self._render_lock:
            was_live = self._live is not None
            if was_live:
                assert self._live is not None
                self._live.stop()
                self._live = None
                self._buffer = io.StringIO()
                for handler, _ in self._restore:
                    handler.setStream(self._buffer)
        self._restore_terminal()
        try:
            yield
        finally:
            if self._keys and self._key_thread is not None:
                self._set_cbreak()
            buffered = self._buffer.getvalue() if self._buffer is not None else ""
            self._buffer = None
            self._restore_handlers()
            self._paused.clear()
            if was_live:
                self._start_live()
            if buffered:
                sys.stdout.write(buffered)
                sys.stdout.flush()

    # --- keys ------------------------------------------------------------
    def _set_cbreak(self) -> None:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            if self._term is None:
                self._term = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)
        except (ImportError, OSError, ValueError, AttributeError):
            self._keys = False

    def _restore_terminal(self) -> None:
        if self._term is None:
            return
        try:
            import termios

            fd, attrs = self._term
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except (ImportError, OSError, ValueError):
            pass

    def _key_loop(self) -> None:
        self._set_cbreak()
        try:
            fd = sys.stdin.fileno()
        except (OSError, ValueError, AttributeError):
            return
        while not self._stop.is_set() and self._keys:
            if self._paused.is_set():
                self._keys_idle.set()
                time.sleep(0.1)
                continue
            self._keys_idle.clear()
            try:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                key = os.read(fd, 1).decode(errors="ignore")
            except (OSError, ValueError):
                return
            self.on_key(key)
        self._keys_idle.set()
        self._restore_terminal()

    def on_key(self, key: str) -> None:
        action, system = handle_key(key, self.tracker.snapshot())
        if action == "help":
            self._show_help = not self._show_help
        elif action == "login" and system is not None:
            login = _login_handler()
            with self._handoff():
                login(system)

    # --- ssh master watch -----------------------------------------------------
    def _watch_loop(self) -> None:
        check = _watch_handler()
        while not self._stop.wait(SSH_WATCH_INTERVAL):
            if self._paused.is_set():
                continue
            try:
                check()
            except Exception as exc:  # the watch must never end the run
                logger.debug("ssh master check failed: %s", exc)


@contextmanager
def terminal_handoff() -> Iterator[None]:
    """Give the terminal to an interactive command (an ssh login) while the
    run's live panels are up; a no-op otherwise."""
    monitor = ProgressMonitor._active
    if monitor is None:
        yield
        return
    with monitor._handoff():
        yield


def _login_handler() -> Callable[[str], bool]:
    # Imported here: executor imports this module.
    from alomancy.remote_submission import executor

    login: Callable[[str], bool] = executor.relogin
    return login


def _watch_handler() -> Callable[[], None]:
    from alomancy.remote_submission import executor

    check: Callable[[], None] = executor.check_ssh_masters
    return check
