import logging
import sys
from pathlib import Path


def setup_logging(verbose: int = 0, log_file: str | None = "results/alomancy.log") -> None:
    """Configure the alomancy logger hierarchy.

    verbose=0 → console shows WARNING+  (silent during normal runs)
    verbose=1 → console shows INFO      (step-level progress)
    verbose=2 → console shows DEBUG     (per-job detail, ExPyRe stdout/stderr)

    The file handler always captures DEBUG regardless of verbose, so every
    run produces a complete timestamped record even when the console is quiet.
    """
    root = logging.getLogger("alomancy")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console_level = (
        logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
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

    # Route expyre's own logging through our handlers so HPC job events
    # appear in the same log file.
    expyre_logger = logging.getLogger("expyre")
    expyre_logger.setLevel(logging.DEBUG if verbose >= 2 else logging.WARNING)
    expyre_logger.propagate = False
    expyre_logger.handlers.clear()
    for h in root.handlers:
        expyre_logger.addHandler(h)

    root.propagate = False
