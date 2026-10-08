"""Which of a loop's events still stand, so the report doesn't present
failures a restart or a retry already fixed.

Two rules, in this order:

1. **Latest attempt per step.** Every event is tagged with the run attempt
   (one per process, ``logging_config.ATTEMPT_ID``) and the step it ran in
   (``@phase`` name; None for loop wrap-up). For each step, only the events
   of the latest attempt that ran it are kept. A step completed in an
   earlier attempt is not re-run (its restart sentinel), so its events,
   warnings included, are kept from that attempt.
2. **Retries that succeeded.** Job and retry events name the item they
   concern (``fit_0``, ``md_run_3``, ``structure_17``). An item's outcome is
   its last success (``job_succeeded``) or failure event; a failure or
   retry event whose items all ended in success is dropped.

Event files written before events carried ``attempt``/``phase`` are tagged
from ``alomancy.log``: each "ALomancy Workflow Summary" header starts an
attempt, and the steps' "Phase <name> marked complete" lines (with the
fixed step order) say which step an event belongs to. They have no items,
so only rule 1 applies to them.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# The loop's steps, in the order a loop runs them.
PHASE_ORDER = ("train_mlip", "generate_structures", "high_accuracy_eval")
# Bookkeeping events: used here, never counted or shown in the report.
INTERNAL_EVENTS = frozenset({"phase_started", "phase_completed", "job_succeeded"})
_SUCCESS = "job_succeeded"
_FAILURES = frozenset(
    {"job_failed", "job_died", "md_unfinished", "fit_missing", "fit_not_evaluated"}
)
# Dropped when every item they name ended in success.
_SUPERSEDABLE = frozenset(
    {
        "job_failed",
        "job_died",
        "job_resumed",
        "job_resubmitted",
        "fit_retry",
        "fit_not_evaluated",
        "md_no_steps",
    }
)

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_COMPLETE = re.compile(r"Phase (\w+) marked complete for al_loop_(\d+)\.")
_ATTEMPT_HEADER = "ALomancy Workflow Summary"


def _time(event: dict) -> str:
    return str(event.get("time", "")).replace("T", " ")


def _attempt_order(attempt: str) -> str:
    # Attempt ids start with their start time (ISO), so they sort by time.
    return attempt.replace("T", " ")


def _items(event: dict) -> list[str]:
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return []
    items = list(data.get("items") or [])
    if data.get("item") is not None:
        items.append(data["item"])
    return [str(i) for i in items]


def _log_timeline(log_file: Path) -> tuple[list[str], dict[int, list[tuple[str, str]]]]:
    """Attempt start times, and per loop the (time, step) completions."""
    starts: list[str] = []
    completions: dict[int, list[tuple[str, str]]] = defaultdict(list)
    last_ts = ""
    with log_file.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if match := _TS.match(line):
                last_ts = match.group(1)
            if _ATTEMPT_HEADER in line:
                starts.append(last_ts)
            elif match := _COMPLETE.search(line):
                completions[int(match.group(2))].append((last_ts, match.group(1)))
    return starts, completions


def _tag_from_log(events: list[dict], loop: int, log_file: Path | None) -> None:
    """Give untagged events of *loop* an attempt and a step, from the log."""
    untagged = [e for e in events if "attempt" not in e]
    if not untagged:
        return
    if log_file is None or not log_file.exists():
        for event in untagged:
            event["attempt"], event["phase"] = "unknown", None
        return
    starts, completions = _log_timeline(log_file)
    loop_completions = completions.get(loop, [])
    for event in untagged:
        t = _time(event)
        start = max((s for s in starts if s <= t), default="")
        later = [s for s in starts if s > start]
        end = min(later) if later else "9999"
        done_before = {p for ts, p in loop_completions if ts < start}
        done_now = {p: ts for ts, p in loop_completions if start <= ts < end}
        phase = None
        for p in (p for p in PHASE_ORDER if p not in done_before):
            if p not in done_now or done_now[p] >= t:
                phase = p
                break
        event["attempt"], event["phase"] = f"{start}-log", phase


def _latest_attempt_per_phase(events: list[dict]) -> list[dict]:
    started: dict[Any, set[str]] = defaultdict(set)
    seen: dict[Any, set[str]] = defaultdict(set)
    for event in events:
        seen[event.get("phase")].add(event["attempt"])
        if event.get("event") == "phase_started":
            started[event.get("phase")].add(event["attempt"])
    latest = {
        phase: max(started.get(phase) or attempts, key=_attempt_order)
        for phase, attempts in seen.items()
    }
    return [e for e in events if e["attempt"] == latest[e.get("phase")]]


def _drop_resolved(events: list[dict]) -> list[dict]:
    outcome: dict[str, bool] = {}
    for event in events:
        code = event.get("event")
        if code == _SUCCESS or code in _FAILURES:
            for item in _items(event):
                outcome[item] = code == _SUCCESS
    kept = []
    for event in events:
        items = _items(event)
        if (
            event.get("event") in _SUPERSEDABLE
            and items
            and all(outcome.get(i) for i in items)
        ):
            continue
        kept.append(event)
    return kept


def current_events(events: list[dict], loop: int, log_file: Path | None) -> list[dict]:
    """The events of *loop* that still stand (see module docstring), in
    order, without the internal bookkeeping events."""
    loop_events = [dict(e) for e in events if e.get("loop") == loop]
    _tag_from_log(loop_events, loop, log_file)
    kept = _drop_resolved(_latest_attempt_per_phase(loop_events))
    return [e for e in kept if e.get("event") not in INTERNAL_EVENTS]
