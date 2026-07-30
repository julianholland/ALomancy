import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alomancy.analysis.colors import (
    DIAGONAL_COLOR,
    PALETTE,
    STAGE2_COLOR,
    add_logo_watermark,
    setup_alomancy_style,
)

logger = logging.getLogger(__name__)


def _get_stage_two_epoch(mlip_committee_job_dict: dict) -> int:
    mace_kwargs = mlip_committee_job_dict.get("mace_fit_kwargs", {})
    if "start_swa" in mace_kwargs:
        return int(mace_kwargs["start_swa"])
    max_ep = mlip_committee_job_dict.get("max_num_epochs") or mace_kwargs.get(
        "max_num_epochs", 80
    )
    return math.floor(max_ep * 0.8)


def _parse_training_jsonl(
    fit_dir: Path, name: str, fit_seed: int
) -> pd.DataFrame | None:
    txt_path = fit_dir / "results" / f"{name}_run-{fit_seed}_train.txt"
    if not txt_path.exists():
        # Seed can differ from expected value — fall back to any matching file
        candidates = sorted((fit_dir / "results").glob("*_train.txt"))
        if not candidates:
            logger.warning("Training metrics file not found: %s", txt_path)
            return None
        txt_path = candidates[0]
        logger.debug("Using training metrics file: %s", txt_path)

    rows = []
    with txt_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("mode") == "eval" and record.get("epoch") is not None:
                rows.append(record)

    if not rows:
        logger.warning("No eval records found in %s", txt_path)
        return None

    df = pd.DataFrame(rows).set_index("epoch")
    return df


def _parse_used_epoch(fit_dir: Path, name: str, fit_seed: int) -> int | None:
    log_path = fit_dir / "logs" / f"{name}_run-{fit_seed}.log"
    if not log_path.exists():
        candidates = sorted((fit_dir / "logs").glob("*.log"))
        if not candidates:
            logger.warning("Log file not found: %s", log_path)
            return None
        log_path = candidates[0]
        logger.debug("Using log file: %s", log_path)

    pattern = re.compile(r"Loaded Stage two model from epoch (\d+)")
    with log_path.open() as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    return None


def plot_training_curves(
    base_name: str,
    mlip_committee_job_dict: dict,
    seed: int,
    plots_dir: Path,
) -> None:
    name = mlip_committee_job_dict["name"]
    n_fits = mlip_committee_job_dict["size_of_committee"]
    stage2_epoch = _get_stage_two_epoch(mlip_committee_job_dict)

    setup_alomancy_style()
    colors = PALETTE

    # --- collect per-fit data ---
    fit_data: list[tuple[int, pd.DataFrame, int | None]] = []
    for i in range(n_fits):
        fit_dir = Path("results", base_name, name, f"fit_{i}")
        df = _parse_training_jsonl(fit_dir, name, seed + i)
        if df is None:
            logger.warning("Skipping fit_%d: no training data.", i)
            continue
        used_ep = _parse_used_epoch(fit_dir, name, seed + i)
        fit_data.append((i, df, used_ep))

    if not fit_data:
        logger.warning(
            "No fit data available for %s — skipping training curve plots.", base_name
        )
        return

    # --- Plot 2: MAE curves ---
    fig_mae, (ax_e, ax_f) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig_mae.suptitle(f"{name} — Training MAE  [{base_name}]")

    for i, df, used_ep in fit_data:
        color = colors[i % len(colors)]
        label = f"fit_{i} (seed {seed + i})"
        if "mae_e_per_atom" in df.columns:
            ax_e.plot(
                df.index, df["mae_e_per_atom"], color=color, label=label, linewidth=1.2
            )
        if "mae_f" in df.columns:
            ax_f.plot(df.index, df["mae_f"], color=color, label=label, linewidth=1.2)
        if used_ep is not None:
            ax_e.axvline(used_ep, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
            ax_f.axvline(used_ep, color=color, linestyle=":", linewidth=1.0, alpha=0.8)

    for ax in (ax_e, ax_f):
        ax.axvline(
            stage2_epoch,
            color=STAGE2_COLOR,
            linestyle="--",
            linewidth=1.2,
            label="Stage 2",
        )
        ax.grid(True)
        ax.legend(fontsize=8)

    ax_e.set_ylabel("Energy MAE (eV/atom)")
    ax_f.set_ylabel("Force MAE (eV/Å)")
    ax_f.set_xlabel("Epoch")

    mae_path = plots_dir / f"training_mae_{base_name}.png"
    fig_mae.tight_layout()
    add_logo_watermark(fig_mae)
    fig_mae.savefig(mae_path, dpi=150)
    plt.close(fig_mae)
    logger.info("Saved training MAE plot to %s", mae_path)

    # --- Plot 3: Loss curves ---
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    fig_loss.suptitle(f"{name} — Training Loss  [{base_name}]")

    for i, df, used_ep in fit_data:
        color = colors[i % len(colors)]
        label = f"fit_{i} (seed {seed + i})"
        if "loss" in df.columns:
            ax_loss.plot(df.index, df["loss"], color=color, label=label, linewidth=1.2)
        if used_ep is not None:
            ax_loss.axvline(
                used_ep, color=color, linestyle=":", linewidth=1.0, alpha=0.8
            )

    ax_loss.axvline(
        stage2_epoch, color=STAGE2_COLOR, linestyle="--", linewidth=1.2, label="Stage 2"
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True)
    ax_loss.legend(fontsize=8)

    loss_path = plots_dir / f"training_loss_{base_name}.png"
    fig_loss.tight_layout()
    add_logo_watermark(fig_loss)
    fig_loss.savefig(loss_path, dpi=150)
    plt.close(fig_loss)
    logger.info("Saved training loss plot to %s", loss_path)


def _parse_eval_xyz(path: Path, e0: dict[str, float] | None = None) -> tuple | None:
    """Read a MACE eval predictions xyz and return (e_dft, e_pred, f_dft, f_pred).

    Written by mace_fit after training completes on the remote node. Looks for
    mace_energy / mace_forces keys. Returns None if the file has no usable rows.

    Energy values are per-atom eV/atom. If `e0` (element -> isolated-atom
    energy) is given, each structure's per-element E0 sum is subtracted before
    dividing by atom count, yielding formation energy per atom instead of raw
    energy per atom. If any structure contains an element missing from `e0`,
    the whole file falls back to raw per-atom energy (logging one warning)
    rather than mixing formation- and raw-energy points in one figure.
    """
    from ase.io import read as ase_read

    try:
        atoms_list = list(ase_read(str(path), ":", format="extxyz"))
    except Exception as exc:
        logger.warning("Failed to read eval xyz %s: %s", path, exc)
        return None

    rows = [
        atoms
        for atoms in atoms_list
        if "REF_energy" in atoms.info and "mace_energy" in atoms.info
    ]
    if not rows:
        return None

    use_e0 = e0 is not None
    e0_map: dict[str, float] = e0 if e0 is not None else {}
    if use_e0:
        missing = {
            s for atoms in rows for s in atoms.get_chemical_symbols() if s not in e0_map
        }
        if missing:
            logger.warning(
                "E0 dict missing energies for elements %s in %s — falling back to "
                "raw per-atom energy instead of formation energy.",
                sorted(missing),
                path,
            )
            use_e0 = False

    e_dft, e_pred, f_dft, f_pred = [], [], [], []
    for atoms in rows:
        n = len(atoms)
        shift = sum(e0_map[s] for s in atoms.get_chemical_symbols()) if use_e0 else 0.0
        e_dft.append((atoms.info["REF_energy"] - shift) / n)
        e_pred.append((float(atoms.info["mace_energy"]) - shift) / n)
        if "REF_forces" in atoms.arrays and "mace_forces" in atoms.arrays:
            f_dft.extend(atoms.arrays["REF_forces"].flatten().tolist())
            f_pred.extend(atoms.arrays["mace_forces"].flatten().tolist())

    return (
        np.array(e_dft),
        np.array(e_pred),
        np.array(f_dft),
        np.array(f_pred),
    )


def _draw_parity_figure(
    results_per_fit: list,
    n_fits: int,
    name: str,
    seed: int,
    set_label: str,
    base_name: str,
    plots_dir: Path,
    file_suffix: str,
    energy_label: str = "energy",
) -> None:
    fig, axes = plt.subplots(n_fits, 2, figsize=(8, 4 * n_fits), squeeze=False)
    fig.suptitle(f"{name} — {set_label} Set Parity  [{base_name}]", y=1.01)

    for i, result in enumerate(results_per_fit):
        ax_e, ax_f = axes[i, 0], axes[i, 1]
        row_title = f"fit_{i}  (seed {seed + i})"

        if result is None:
            for ax in (ax_e, ax_f):
                ax.text(
                    0.5,
                    0.5,
                    "No predictions available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            e_dft_arr, e_pred_arr, f_dft_arr, f_pred_arr = result
            color = PALETTE[i % len(PALETTE)]

            if len(e_dft_arr):
                mae_e = np.mean(np.abs(e_dft_arr - e_pred_arr))
                lim = (
                    min(e_dft_arr.min(), e_pred_arr.min()),
                    max(e_dft_arr.max(), e_pred_arr.max()),
                )
                ax_e.scatter(
                    e_dft_arr, e_pred_arr, s=6, alpha=0.5, color=color, rasterized=True
                )
                ax_e.plot(lim, lim, color=DIAGONAL_COLOR, linestyle="--", linewidth=0.8)
                ax_e.text(
                    0.05,
                    0.95,
                    f"MAE = {mae_e:.4f} eV/atom",
                    transform=ax_e.transAxes,
                    va="top",
                    fontsize=8,
                )

            if len(f_dft_arr):
                mae_f = np.mean(np.abs(f_dft_arr - f_pred_arr))
                lim_f = (
                    min(f_dft_arr.min(), f_pred_arr.min()),
                    max(f_dft_arr.max(), f_pred_arr.max()),
                )
                ax_f.scatter(
                    f_dft_arr, f_pred_arr, s=2, alpha=0.2, color=color, rasterized=True
                )
                ax_f.plot(
                    lim_f, lim_f, color=DIAGONAL_COLOR, linestyle="--", linewidth=0.8
                )
                ax_f.text(
                    0.05,
                    0.95,
                    f"MAE = {mae_f:.4f} eV/Å",
                    transform=ax_f.transAxes,
                    va="top",
                    fontsize=8,
                )

        ax_e.set_xlabel(f"DFT {energy_label} (eV/atom)")
        ax_e.set_ylabel(f"Model {energy_label} (eV/atom)")
        ax_e.set_title(f"{row_title} — Energy")
        ax_f.set_xlabel("DFT forces (eV/Å)")
        ax_f.set_ylabel("Model forces (eV/Å)")
        ax_f.set_title(f"{row_title} — Forces")

    fig.tight_layout()
    add_logo_watermark(fig)
    path = plots_dir / f"fit_parity_{file_suffix}_{base_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s parity plot to %s", set_label.lower(), path)


def plot_dft_vs_model(
    base_name: str,
    mlip_committee_job_dict: dict,
    seed: int,
    plots_dir: Path,
    db: Any = None,
    loop_idx: int | None = None,
) -> None:
    setup_alomancy_style()
    name = mlip_committee_job_dict["name"]
    n_fits = mlip_committee_job_dict["size_of_committee"]

    e0: dict[str, float] | None = None
    if db is not None:
        try:
            e0 = db.get_isolated_atom_energies() or None
        except Exception as exc:  # defensive: DB problems must not break plotting
            logger.warning("Failed to compute isolated-atom E0 dict: %s", exc)
            e0 = None
        else:
            if e0 is None:
                logger.warning(
                    "No IsolatedAtom energies in GlobalDatabase — parity plots will "
                    "show raw per-atom energy, not formation energy."
                )
    energy_label = "formation energy" if e0 else "energy"

    train_results: list = []
    test_results: list = []

    for i in range(n_fits):
        # Primary: use stored DB predictions — no model load or GPU needed.
        if db is not None and loop_idx is not None:
            stored = db.get_mace_predictions(loop_idx, i, e0=e0)
            if stored is not None:
                train_results.append(stored.get("train"))
                test_results.append(stored.get("test"))
                logger.info(
                    "Using stored DB predictions for loop %d fit %d.", loop_idx, i
                )
                continue

        # Secondary: read from eval xyz files written by mace_fit on the remote node.
        fit_dir = Path("results", base_name, name, f"fit_{i}")
        train_xyz = fit_dir / "train_pred.xyz"
        test_xyz = fit_dir / "test_pred.xyz"
        if train_xyz.exists() or test_xyz.exists():
            train_results.append(
                _parse_eval_xyz(train_xyz, e0=e0) if train_xyz.exists() else None
            )
            test_results.append(
                _parse_eval_xyz(test_xyz, e0=e0) if test_xyz.exists() else None
            )
            logger.info(
                "Using eval xyz files for parity plot, loop %s fit %d.",
                loop_idx if loop_idx is not None else "?",
                i,
            )
            continue

        # No predictions available — skip this fit rather than running local inference.
        logger.info(
            "No predictions available for fit_%d (loop %s) — parity plot will be blank. "
            "Predictions are written during remote training from alomancy v0.4.2 onwards.",
            i,
            loop_idx if loop_idx is not None else "?",
        )
        train_results.append(None)
        test_results.append(None)

    if not train_results and not test_results:
        return

    has_train = any(r is not None for r in train_results)
    has_test = any(r is not None for r in test_results)

    if has_train:
        _draw_parity_figure(
            train_results,
            n_fits,
            name,
            seed,
            "Training",
            base_name,
            plots_dir,
            "train",
            energy_label=energy_label,
        )
    if has_test:
        _draw_parity_figure(
            test_results,
            n_fits,
            name,
            seed,
            "Test",
            base_name,
            plots_dir,
            "test",
            energy_label=energy_label,
        )
