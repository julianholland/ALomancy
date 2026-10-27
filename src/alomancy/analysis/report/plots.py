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
    """Structures kept against the duplicate tolerance, from redundancy
    removal's tolerance probe: the plateaus it found shaded, the
    flagged-fraction cap as a horizontal line, the tolerance it chose as a
    black dashed line, and on a right-hand axis the number of structures
    removed between consecutive tolerance steps (the negative gradient of
    the kept count: zero on a plateau)."""
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
    ax.plot(tolerances, counts, color=PALETTE[0], lw=1.2, label="structures kept")
    ax.axhline(n, color="#9CA3AF", ls=":", lw=1, label=f"all {n:,} structures")
    cap = (record.get("settings") or {}).get("max_flagged_fraction")
    if cap is not None and cap < 1:
        ax.axhline(
            n * (1 - cap),
            color=PALETTE[2],
            ls="-.",
            lw=1,
            label=f"cap: at most {cap:.0%} flagged",
        )
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
            "lowest plateau flags more than the cap: nothing flagged"
            if record.get("outcome") == "cap_exceeded"
            else "no plateau found: nothing flagged",
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
        )
    # The sweep runs until one structure is left; past the point where 95%
    # are flagged it's a long flat tail, so it's cut there (keeping the
    # chosen tolerance in view).
    if len(counts) > 1:
        tail = np.nonzero(counts <= max(1.0, 0.05 * n))[0]
        right = tolerances[tail[0]] * 1.1 if tail.size else tolerances[-1]
        if chosen is not None:
            right = max(right, chosen * 1.2)
        ax.set_xlim(tolerances[0], min(right, tolerances[-1]))
    descriptor = (record.get("descriptor") or {}).get("key", "descriptor")
    ax.set_xlabel(f"Tolerance (Euclidean distance between {descriptor} descriptors)")
    ax.set_ylabel("Structures kept")
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


# Relative error is meaningless near zero: energies with |formation energy|
# below this (eV/atom) and atoms with |F| below this (eV/Å) are left out.
PERCENT_ENERGY_FLOOR = 0.05
PERCENT_FORCE_FLOOR = 0.1


def percent_errors(
    predictions: dict[str, tuple], *, formation_energy: bool = True
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    """Relative errors for the percentage-error plot and the report's
    summary line, {split: {"energy"|"force": (x, percent, n_below_floor)}}.

    Energy: x = DFT formation energy per atom, percent = signed 100 (pred -
    ref) / |ref|; left out unless *formation_energy* (relative to a total
    energy of hundreds of eV/atom every error looks like 0%). Force: per
    atom, x = |F_ref|, percent = 100 |F_pred - F_ref| / |F_ref| (unsigned).
    Points below the floors are left out and counted. A force array that
    isn't whole atoms (length not a multiple of 3) gives no force entry for
    that split."""
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]] = {}
    for split in ("train", "test"):
        if split not in predictions:
            continue
        e_ref, e_pred, f_ref, f_pred = (
            np.asarray(v, dtype=float).reshape(-1) for v in predictions[split]
        )
        entry: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        if formation_energy:
            keep = np.abs(e_ref) >= PERCENT_ENERGY_FLOOR
            entry["energy"] = (
                e_ref[keep],
                100 * (e_pred[keep] - e_ref[keep]) / np.abs(e_ref[keep]),
                int((~keep).sum()),
            )
        if f_ref.size % 3 == 0 and f_ref.size == f_pred.size:
            ref_vec, pred_vec = f_ref.reshape(-1, 3), f_pred.reshape(-1, 3)
            ref_norm = np.linalg.norm(ref_vec, axis=1)
            keep = ref_norm >= PERCENT_FORCE_FLOOR
            err = np.linalg.norm(pred_vec[keep] - ref_vec[keep], axis=1)
            entry["force"] = (
                ref_norm[keep],
                100 * err / ref_norm[keep],
                int((~keep).sum()),
            )
        else:
            logger.debug(
                "%s forces are not whole atoms (%d components); no force "
                "percentage error.",
                split,
                f_ref.size,
            )
        out[split] = entry
    return out


def _medians(errors: dict[str, dict]) -> dict[str, dict[str, float]]:
    return {
        split: {
            quantity: float(np.median(np.abs(pct)))
            for quantity, (_x, pct, _n) in entry.items()
            if pct.size
        }
        for split, entry in errors.items()
    }


def percent_error_medians(
    predictions: dict[str, tuple], *, formation_energy: bool = True
) -> dict[str, dict[str, float]]:
    """Median |percent error| per split and quantity (only those with
    points above the floor), for the report's summary line."""
    return _medians(percent_errors(predictions, formation_energy=formation_energy))


def percent_error(
    predictions: dict[str, tuple],
    path: Path,
    *,
    title: str,
    formation_energy: bool = True,
) -> dict[str, dict[str, float]] | None:
    """Signed energy and unsigned force percentage error against the DFT
    value for one model, train and test overlaid (see percent_errors).
    Returns the medians the legend shows (percent_error_medians), or None
    and writes nothing when no point is above the floors."""
    errors = percent_errors(predictions, formation_energy=formation_energy)
    if not any(pct.size for entry in errors.values() for _x, pct, _n in entry.values()):
        return None
    medians = _medians(errors)
    setup_alomancy_style()
    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(10, 4.6))
    for i, (split, entry) in enumerate(errors.items()):
        color = PALETTE[i]
        for ax, quantity, size, alpha in (
            (ax_e, "energy", 10, 0.6),
            (ax_f, "force", 6, 0.4),
        ):
            if quantity not in entry:
                continue
            x, pct, n_below = entry[quantity]
            label = split
            if pct.size:
                label += f" (median {medians[split][quantity]:.1f}%"
                label += f", {n_below} below floor)" if n_below else ")"
            ax.scatter(
                x, pct, s=size, alpha=alpha, color=color, edgecolors="none", label=label
            )
    # Force error is unsigned and spans decades: log on both axes.
    has_force = any(
        entry["force"][1].size for entry in errors.values() if "force" in entry
    )
    if has_force:
        ax_f.set_xscale("log")
        ax_f.set_yscale("log")
    else:
        ax_f.text(
            0.5,
            0.5,
            f"No atom with |F| ≥ {PERCENT_FORCE_FLOOR} eV/Å",
            transform=ax_f.transAxes,
            ha="center",
            va="center",
        )
    if formation_energy:
        # Signed energy error: symlog keeps a few huge outliers from
        # flattening the rest.
        ax_e.set_yscale("symlog", linthresh=10)
        ax_e.axhline(0, color=DIAGONAL_COLOR, linewidth=1, zorder=0)
    else:
        ax_e.text(
            0.5,
            0.5,
            "No isolated-atom energy for every element:\nno formation energies "
            "to take a relative error of",
            transform=ax_e.transAxes,
            ha="center",
            va="center",
        )
    energy_levels = (-10, 10) if formation_energy else ()
    for ax, levels in ((ax_e, energy_levels), (ax_f, (10,) if has_force else ())):
        for level in levels:
            ax.axhline(level, color=DIAGONAL_COLOR, linewidth=0.8, linestyle="--")
    for ax, xlabel, ylabel in (
        (ax_e, "DFT formation energy (eV/atom)", "Energy error (%)"),
        (ax_f, "DFT |F| per atom (eV/Å)", "Force error |ΔF|/|F| (%)"),
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    _save(fig, path)
    return medians


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
