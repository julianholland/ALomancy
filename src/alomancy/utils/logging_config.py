import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

EVENTS_FILENAME = "events.jsonl"

# The AL loop currently running (None outside a loop). A plain module global
# rather than a ContextVar: RemoteJobExecutor logs from ThreadPoolExecutor
# worker threads, which would not inherit a ContextVar set on the main thread.
_current_loop: int | None = None


def set_current_loop(loop: int | None) -> None:
    """Tag every event logged from now on with *loop* (see JsonlEventHandler)."""
    global _current_loop
    _current_loop = loop


# The step (@phase name, e.g. "generate_structures") currently running, None
# outside one; a module global for the same reason as _current_loop.
_current_phase: str | None = None

# One per process: every event this run attempt logs carries it, so the loop
# report can keep only a step's latest attempt (analysis/report/
# current_events.py) instead of counting failures a restart already fixed.
ATTEMPT_ID = f"{datetime.now():%Y-%m-%dT%H:%M:%S}-{os.getpid()}"


def set_current_phase(phase: str | None) -> None:
    """Tag every event logged from now on with *phase* (see JsonlEventHandler)."""
    global _current_phase
    _current_phase = phase


class JsonlEventHandler(logging.Handler):
    """Append warnings, errors and coded events to a JSON-lines file.

    A record is written when its level is WARNING or above, or when it
    carries an ``event`` code (``logger.info(..., extra={"event": "x",
    "data": {...}})``). Each line holds time, level, logger, message,
    event (None if uncoded), loop (see set_current_loop), phase (see
    set_current_phase), attempt (ATTEMPT_ID) and data. The loop report
    (analysis/report) counts these instead of parsing log text.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(level=logging.DEBUG)
        self.path = Path(path).resolve()
        self._lock_file = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "event", None)
        if record.levelno < logging.WARNING and event is None:
            return
        try:
            line: dict[str, Any] = {
                "time": datetime.fromtimestamp(record.created).isoformat(
                    timespec="seconds"
                ),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "event": event,
                "loop": _current_loop,
                "phase": _current_phase,
                "attempt": ATTEMPT_ID,
                "data": getattr(record, "data", None),
            }
            text = json.dumps(line, default=str)
            with self._lock_file, self.path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            self.handleError(record)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Every event in a JSON-lines event file; [] if it doesn't exist.
    Unparseable lines (e.g. a line cut short by a crash) are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def setup_logging(
    verbose: int = 0, log_file: str | None = "results/alomancy.log"
) -> None:
    """Configure the alomancy logger hierarchy.

    verbose=0 → console shows WARNING+  (silent during normal runs)
    verbose=1 → console shows INFO      (step-level progress)
    verbose=2 → console shows DEBUG     (per-job detail, ExPyRe stdout/stderr)

    The file handler always captures DEBUG regardless of verbose, so every
    run produces a complete timestamped record even when the console is quiet.
    Next to the log file, events.jsonl collects warnings and coded events
    (JsonlEventHandler) for the loop report.
    """
    root = logging.getLogger("alomancy")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console_level = (
        logging.WARNING
        if verbose == 0
        else logging.INFO
        if verbose == 1
        else logging.DEBUG
    )
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(JsonlEventHandler(Path(log_file).with_name(EVENTS_FILENAME)))

    # Route expyre's own logging through our handlers so HPC job events
    # appear in the same log file.
    expyre_logger = logging.getLogger("expyre")
    expyre_logger.setLevel(logging.DEBUG if verbose >= 2 else logging.WARNING)
    expyre_logger.propagate = False
    expyre_logger.handlers.clear()
    for h in root.handlers:
        expyre_logger.addHandler(h)

    root.propagate = False
