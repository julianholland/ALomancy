import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alomancy.analysis.colors import (
    PALETTE,
    STAGE2_COLOR,
    add_logo_watermark,
    setup_alomancy_style,
)

logger = logging.getLogger(__name__)

_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
_TS_FMT = "%Y-%m-%d %H:%M:%S"

_LOOP_START_RE = re.compile(_TS + r".*Starting AL loop (\d+)")
_N_TRAIN_RE = re.compile(_TS + r".*Training set size: (\d+)")
_SUBMITTED_RE = re.compile(_TS + r".*Submitted \d+ jobs to queue\.")
_QUEUE_TIME_RE = re.compile(_TS + r".*Job \d+ queue_time=([\d.]+) s\.")
_GEN_START_RE = re.compile(
    _TS + r".* structures selected for structure generation step\."
)
_GEN_END_RE = re.compile(
    _TS + r".*Selected \d+ structures for DFT calculations based on force std dev\."
)
_DFT_END_RE = re.compile(
    _TS + r".*High-accuracy evaluation completed for \d+ structures\."
)
_LOOP_END_RE = re.compile(_TS + r".*Completed AL loop (\d+),")

_QUEUE_COLOR = "#cccccc"
_PHASE_LABELS = [
    "training_plots",
    "generate_structures",
    "high_accuracy_evaluation",
    "postprocess",
]


def _ts(s: str) -> datetime:
    return datetime.strptime(s, _TS_FMT)


def parse_timing_log(log_file: str | Path) -> pd.DataFrame:
    """Parse phase timings from an alomancy.log file.

    Infers phase boundaries from existing INFO/DEBUG messages — no new timing
    messages required. The log file is append-only across restarts; duplicate
    loop entries are resolved by last-write-wins.

    Returns a DataFrame sorted by loop with columns for wall-clock seconds per
    phase and estimated queue seconds (NaN when no queue messages were logged).
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return pd.DataFrame()

    # Per-loop state
    loops: dict[int, dict] = {}
    active_loop: int | None = None

    def _fresh() -> dict:
        return {
            "loop_start": None,
            "n_train": None,
            "gen_start": None,
            "gen_end": None,
            "dft_end": None,
            "loop_end": None,
            # list of (ts: datetime, queue_s: float)
            "queue_samples": [],
        }

    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _LOOP_START_RE.search(line)
            if m:
                ts, n = _ts(m.group(1)), int(m.group(2))
                active_loop = n
                loops[n] = _fresh()
                loops[n]["loop_start"] = ts
                continue

            if active_loop is None:
                continue

            m = _N_TRAIN_RE.search(line)
            if m:
                loops[active_loop]["n_train"] = int(m.group(2))
                continue

            m = _QUEUE_TIME_RE.search(line)
            if m:
                loops[active_loop]["queue_samples"].append(
                    (_ts(m.group(1)), float(m.group(2)))
                )
                continue

            m = _GEN_START_RE.search(line)
            if m and loops[active_loop]["gen_start"] is None:
                loops[active_loop]["gen_start"] = _ts(m.group(1))
                continue

            m = _GEN_END_RE.search(line)
            if m and loops[active_loop]["gen_end"] is None:
                loops[active_loop]["gen_end"] = _ts(m.group(1))
                continue

            m = _DFT_END_RE.search(line)
            if m and loops[active_loop]["dft_end"] is None:
                loops[active_loop]["dft_end"] = _ts(m.group(1))
                continue

            m = _LOOP_END_RE.search(line)
            if m:
                n = int(m.group(2))
                if n in loops:
                    loops[n]["loop_end"] = _ts(m.group(1))

    if not loops:
        return pd.DataFrame()

    def _secs(t0: datetime, end: datetime | None) -> float:
        return float("nan") if end is None else (end - t0).total_seconds()

    def _interval(a: datetime | None, b: datetime | None) -> float:
        return float("nan") if (a is None or b is None) else (b - a).total_seconds()

    def _mean_q(samples: list, lo: datetime | None, hi: datetime | None) -> float:
        if lo is None:
            vals = [qs for _, qs in samples]
        elif hi is None:
            vals = [qs for ts, qs in samples if ts < lo]
        else:
            vals = [qs for ts, qs in samples if lo <= ts < hi]
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for loop_idx, d in sorted(loops.items()):
        t0 = d["loop_start"]
        if t0 is None:
            continue

        gen_start = d["gen_start"]
        gen_end = d["gen_end"]
        dft_end = d["dft_end"]
        loop_end = d["loop_end"]
        samples = d["queue_samples"]

        training_plots_s = _secs(t0, gen_start)
        generate_structures_s = _interval(gen_start, gen_end)
        high_accuracy_evaluation_s = _interval(gen_end, dft_end)
        postprocess_s = _interval(dft_end, loop_end)
        total_s = _secs(t0, loop_end)

        # Attribute queue_samples to phases by timestamp
        train_q = _mean_q(samples, lo=None, hi=gen_start)
        gen_q = _mean_q(samples, lo=gen_start, hi=gen_end)
        dft_q = _mean_q(samples, lo=gen_end, hi=dft_end)

        rows.append(
            {
                "loop": loop_idx,
                "n_train": d["n_train"],
                "total_s": total_s,
                "training_plots_s": training_plots_s,
                "training_plots_queue_s": train_q,
                "generate_structures_s": generate_structures_s,
                "generate_structures_queue_s": gen_q,
                "high_accuracy_evaluation_s": high_accuracy_evaluation_s,
                "high_accuracy_evaluation_queue_s": dft_q,
                "postprocess_s": postprocess_s,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("loop").reset_index(drop=True)


def timing_plots(log_file: str | Path, directory: str | Path) -> None:
    """Generate a combined timing plot from an alomancy.log file.

    Produces a single PNG in *directory* (``timing_combined.png``): a stacked
    bar chart of wall-clock time per phase per AL loop (with queue/running
    sub-segments, as the old ``timing_phases.png`` showed), overlaid with a
    line on a secondary y-axis showing the number of training structures per
    loop (the annotation the old ``timing_total.png`` carried as text
    labels). The legend combines both axes' handles into one call, placed at
    ``upper left``.

    The figure width is fixed regardless of the number of loops -- unlike
    the old per-plot ``max(4, len(loops) * 1.2 + 1)`` sizing, which grew
    unbounded as a run accumulated loops. Bars get more cramped with many
    loops on a fixed-width figure; that is an accepted, explicitly requested
    tradeoff over an ever-widening image.
    """
    import matplotlib.pyplot as plt

    df = parse_timing_log(log_file)
    if df.empty:
        logger.warning("No timing data found in %s — skipping timing plot.", log_file)
        return

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    setup_alomancy_style()

    loops = df["loop"].tolist()
    x = np.arange(len(loops))

    # --- phase breakdown stacked bar chart (primary content) ---
    phase_cols = [
        ("training_plots_s", "training_plots_queue_s", "Training + plots"),
        ("generate_structures_s", "generate_structures_queue_s", "Structure gen"),
        ("high_accuracy_evaluation_s", "high_accuracy_evaluation_queue_s", "DFT"),
        ("postprocess_s", None, "Post-process"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(loops))
    queue_patch_added = False

    for pi, (col, q_col, label) in enumerate(phase_cols):
        phase_h = (
            df[col].to_numpy() / 3600.0
            if col in df.columns
            else np.full(len(loops), np.nan)
        )
        q_h = (
            df[q_col].to_numpy() / 3600.0
            if (q_col is not None and q_col in df.columns)
            else np.full(len(loops), np.nan)
        )

        # Clamp: queue cannot exceed phase
        run_h = np.where(
            ~np.isnan(phase_h) & ~np.isnan(q_h),
            np.maximum(0.0, phase_h - q_h),
            phase_h,
        )
        q_plot = np.where(~np.isnan(phase_h) & ~np.isnan(q_h), q_h, 0.0)
        phase_no_nan = np.where(np.isnan(phase_h), 0.0, phase_h)

        # Queue sub-segment (light gray, bottom of this phase)
        if np.any(q_plot > 0):
            ax.bar(
                x,
                q_plot,
                bottom=bottoms,
                color=_QUEUE_COLOR,
                label="Queued" if not queue_patch_added else "_nolegend_",
            )
            queue_patch_added = True
            ax.bar(x, run_h, bottom=bottoms + q_plot, color=PALETTE[pi], label=label)
        else:
            ax.bar(x, phase_no_nan, bottom=bottoms, color=PALETTE[pi], label=label)

        bottoms = bottoms + phase_no_nan

    ax.set_xticks(x)
    ax.set_xticklabels([f"Loop {li}" for li in loops])
    ax.set_xlabel("AL loop")
    ax.set_ylabel("Wall-clock time (hours)")
    ax.set_title("Phase breakdown and training-set size per AL loop")

    # --- training-set size overlay (secondary y-axis line) ---
    ax2 = ax.twinx()
    n_train = df["n_train"].to_numpy(dtype=float)
    ax2.plot(
        x,
        n_train,
        color=STAGE2_COLOR,
        marker="o",
        linewidth=1.5,
        label="Training structures",
    )
    ax2.set_ylabel("Training structures")
    ax2.grid(False)

    # twinx() legends must be combined manually -- ax.legend() alone would
    # silently drop ax2's line from the legend.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig.tight_layout()
    add_logo_watermark(fig)
    combined_path = directory / "timing_combined.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    logger.info("Saved combined timing plot to %s", combined_path)
