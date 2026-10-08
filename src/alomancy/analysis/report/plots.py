"""Figures drawn for the loop report. Every function saves its figure and
closes it (Agg only, never shown -- see CLAUDE.md), and returns nothing.

Style: ALomancy's brand palette (analysis/colors.py) in fixed order; one
axis per figure; thin marks with a white gap between adjacent fills;
excluded data in greys so it reads as background; a legend only when
there is more than one series.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from alomancy.analysis.colors import (
    DIAGONAL_COLOR,
    PALETTE,
    STAGE2_COLOR,
    add_logo_watermark,
    setup_alomancy_style,
)

logger = logging.getLogger(__name__)

# Training-set composition: status -> colour, in drawing order. Included
# data in brand colours, excluded data in greys.
STATUS_COLORS = {
    "train": PALETTE[0],
    "test": PALETTE[1],
    "diagnostic": PALETTE[2],
    "unsplit": "#9CA3AF",
    "redundant": "#D1D5DB",
    "quality_filtered": "#6B7280",
}
STATUS_LABELS = {
    "train": "Train",
    "test": "Test",
    "diagnostic": "Diagnostic",
    "unsplit": "Unsplit",
    "redundant": "Excluded: redundant",
    "quality_filtered": "Excluded: quality filter",
}
_GAP = {"edgecolor": "white", "linewidth": 1.5}


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    add_logo_watermark(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def histogram(
    values: list[float],
    path: Path,
    *,
    title: str,
    xlabel: str,
    marker: float | None = None,
    marker_label: str | None = None,
    log_y: bool = False,
) -> None:
    """One-series histogram, optionally with a labelled vertical line."""
    setup_alomancy_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(30, max(5, len(values) // 2)), color=PALETTE[0], **_GAP)
    if marker is not None:
        ax.axvline(marker, color=STAGE2_COLOR, linestyle="--", linewidth=1.5)
        if marker_label:
            ax.annotate(
                marker_label,
                xy=(marker, 1),
                xycoords=("data", "axes fraction"),
                xytext=(-4, -12),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color="#374151",
            )
    if log_y:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y")
    _save(fig, path)


def bar_per_item(
    values: list[float], path: Path, *, title: str, xlabel: str, ylabel: str
) -> None:
    """One bar per item (e.g. frames per MD run)."""
    setup_alomancy_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(values)), values, color=PALETTE[0], **_GAP)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")
    _save(fig, path)


# Points drawn per MD run; logs have one row per step (tens of thousands).
_MD_MAX_POINTS = 2000


def md_runs(
    temperatures: list[np.ndarray],
    completed_steps: list[int],
    path: Path,
    *,
    run_labels: list[str],
    title: str,
    target_temperature: float,
    requested_steps: int,
    equilibration_steps: int = 0,
) -> None:
    """Temperature against MD step for every run on one axis, one thin
    line and colour per run, with the target temperature as a black dashed
    line and the end of equilibration dotted. A run that stopped early says
    where in the legend."""
    setup_alomancy_style()
    n = len(temperatures)
    # The brand palette in its fixed order; beyond it, evenly spaced colours
    # from one perceptual map so every run still has its own.
    colors = (
        PALETTE[:n]
        if n <= len(PALETTE)
        else [plt.get_cmap("turbo")(x) for x in np.linspace(0.05, 0.95, n)]
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for temps, steps, label, color in zip(
        temperatures, completed_steps, run_labels, colors, strict=True
    ):
        stride = max(1, len(temps) // _MD_MAX_POINTS)
        stopped = f" (stopped at {steps:,})" if steps < requested_steps else ""
        ax.plot(
            np.arange(len(temps))[::stride],
            temps[::stride],
            color=color,
            lw=0.7,
            alpha=0.85,
            label=f"{label}{stopped}",
        )
    ax.axhline(
        target_temperature,
        color="black",
        ls="--",
        lw=1.2,
        label=f"target {target_temperature:g} K",
        zorder=5,
    )
    if equilibration_steps:
        ax.axvline(
            equilibration_steps,
            color="black",
            ls=":",
            lw=1,
            label="end of equilibration",
            zorder=5,
        )
    ax.set_xlim(0, equilibration_steps + requested_steps)
    ax.set_xlabel("MD step")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    ax.grid(True)
    # Below the plot: the top-right corner holds the logo watermark.
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=8,
        ncol=min(6, n + 1 + bool(equilibration_steps)),
        frameon=False,
    )
    for handle in legend.get_lines():
        handle.set_linewidth(2)
    _save(fig, path)


def redundancy_probe(record: dict[str, Any], path: Path, *, title: str) -> None:
    """Unique structures against the duplicate tolerance, from redundancy
    removal's tolerance probe: the plateaus it found shaded, the tolerance
    it chose as a black dashed line, and on a right-hand axis the number of
    structures removed between consecutive tolerance steps (the negative
    gradient of the unique count: near zero on a plateau)."""
    setup_alomancy_style()
    tolerances = np.asarray(record["tolerances"], dtype=float)
    counts = np.asarray(record["unique_counts"], dtype=float)
    n = int(record.get("n_structures") or counts.max())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (start, end) in enumerate(record.get("plateaus") or []):
        ax.axvspan(
            start,
            end,
            color=PALETTE[1],
            alpha=0.15,
            lw=0,
            label="plateau" if i == 0 else None,
        )
    ax.plot(tolerances, counts, color=PALETTE[0], lw=1.2, label="unique structures")
    ax.axhline(n, color="#9CA3AF", ls=":", lw=1, label=f"all {n:,} structures")
    chosen = record.get("chosen_tolerance")
    if chosen is not None:
        flagged = int(record.get("n_flagged") or 0)
        ax.axvline(
            chosen,
            color="black",
            ls="--",
            lw=1.2,
            label=(
                f"chosen tolerance {chosen:.4f}: {n - flagged:,} unique "
                f"({flagged:,} flagged)"
            ),
        )
    else:
        ax.text(
            0.98,
            0.05,
            "no plateau found: nothing flagged",
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
        )
    descriptor = (record.get("descriptor") or {}).get("key", "descriptor")
    ax.set_xlabel(f"Tolerance (Euclidean distance between {descriptor} descriptors)")
    ax.set_ylabel("Unique structures")
    ax.set_title(title)
    ax.grid(True)

    ax_removed = ax.twinx()
    if len(counts) > 1:
        removed = -np.diff(counts)
        ax_removed.plot(
            tolerances[1:],
            removed,
            color=PALETTE[3],
            lw=0.6,
            alpha=0.8,
            label="structures removed per step",
        )
    ax_removed.set_ylabel("Structures removed per step", color=PALETTE[3])
    ax_removed.tick_params(axis="y", colors=PALETTE[3])
    ax_removed.set_ylim(bottom=0)
    ax_removed.grid(False)
    # One legend for both axes; a bare legend() after twinx() drops the
    # second axis's line.
    lines, labels = ax.get_legend_handles_labels()
    removed_lines, removed_labels = ax_removed.get_legend_handles_labels()
    # Below the plot: the curve crosses most of the axes, and the top-right
    # corner holds the logo watermark.
    ax.legend(
        lines + removed_lines,
        labels + removed_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    _save(fig, path)


def composition(
    composition: dict[str, dict[str, int]], path: Path, *, title: str
) -> None:
    """Horizontal stacked bars: structures per config_type by status."""
    setup_alomancy_style()
    config_types = sorted(composition, key=lambda c: sum(composition[c].values()))
    statuses = [
        s for s in STATUS_COLORS if any(s in composition[c] for c in config_types)
    ]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * len(config_types) + 1.8)))
    left = np.zeros(len(config_types))
    for status in statuses:
        widths = np.array([composition[c].get(status, 0) for c in config_types])
        ax.barh(
            config_types,
            widths,
            left=left,
            color=STATUS_COLORS[status],
            label=STATUS_LABELS[status],
            **_GAP,
        )
        left += widths
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Structures")
    ax.set_title(title)
    ax.grid(True, axis="x")
    if len(statuses) > 1:
        # Below the axes, so it never covers a bar.
        ax.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=len(statuses),
            frameon=False,
        )
    _save(fig, path)


def parity(
    predictions: dict[str, tuple], path: Path, *, title: str, energy_label: str
) -> None:
    """Energy and force parity for one model, train and test overlaid.

    *predictions* is GlobalDatabase.get_model_predictions' shape:
    {split: (e_ref, e_pred, f_ref, f_pred)}."""
    setup_alomancy_style()
    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(10, 4.6))
    splits = [s for s in ("train", "test") if s in predictions]
    for i, split in enumerate(splits):
        e_ref, e_pred, f_ref, f_pred = predictions[split]
        color = PALETTE[i]
        e_mae = (
            float(np.mean(np.abs(np.asarray(e_pred) - np.asarray(e_ref))))
            if len(e_ref)
            else float("nan")
        )
        f_mae = (
            float(np.mean(np.abs(np.asarray(f_pred) - np.asarray(f_ref))))
            if len(f_ref)
            else float("nan")
        )
        ax_e.scatter(
            e_ref,
            e_pred,
            s=10,
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=f"{split} (MAE {e_mae * 1000:.1f} meV/atom)",
        )
        ax_f.scatter(
            f_ref,
            f_pred,
            s=6,
            alpha=0.4,
            color=color,
            edgecolors="none",
            label=f"{split} (MAE {f_mae:.3f} eV/Å)",
        )
    for ax, unit in (
        (ax_e, f"{energy_label} (eV/atom)"),
        (ax_f, "force component (eV/Å)"),
    ):
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=DIAGONAL_COLOR, linewidth=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"DFT {unit}")
        ax.set_ylabel(f"Model {unit}")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    _save(fig, path)


def training_curve(df: Any, path: Path, *, title: str) -> None:
    """Validation force MAE per epoch for one fit (a polars frame with
    "epoch" and "mae_f" columns, from a trainer's TrainingHistory)."""
    setup_alomancy_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        df["epoch"].to_numpy(), df["mae_f"].to_numpy(), color=PALETTE[0], linewidth=2
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation force MAE (eV/Å)")
    ax.set_title(title)
    ax.grid(True)
    _save(fig, path)
