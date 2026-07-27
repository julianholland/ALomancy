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


_MAX_PARITY_STRUCTURES = 200


def _load_and_subsample(xyz_path: Path, seed: int, label: str) -> list | None:
    from ase.io import read

    if not xyz_path.exists():
        logger.warning("%s set not found at %s — skipping.", label, xyz_path)
        return None
    atoms = list(read(str(xyz_path), ":", format="extxyz"))
    logger.info("Loaded %d %s structures for parity plots.", len(atoms), label)
    if len(atoms) > _MAX_PARITY_STRUCTURES:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(atoms), _MAX_PARITY_STRUCTURES, replace=False))
        atoms = [atoms[i] for i in idx]
        logger.info(
            "Subsampled %s set to %d structures.", label, _MAX_PARITY_STRUCTURES
        )
    return atoms


def _run_inference(calc: object, atoms_list: list) -> tuple:
    e_dft, e_pred, f_dft, f_pred = [], [], [], []
    for atoms in atoms_list:
        if "REF_energy" not in atoms.info:
            continue
        a = atoms.copy()
        a.calc = calc
        try:
            n = len(a)
            e_dft.append(atoms.info["REF_energy"] / n)
            e_pred.append(a.get_potential_energy() / n)
            if "REF_forces" in atoms.arrays:
                f_dft.extend(atoms.arrays["REF_forces"].flatten().tolist())
                f_pred.extend(a.get_forces().flatten().tolist())
        except Exception as exc:
            logger.debug("Inference failed for one structure: %s", exc)
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
                    "Model missing",
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

        ax_e.set_xlabel("DFT energy (eV/atom)")
        ax_e.set_ylabel("Model energy (eV/atom)")
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

    train_results: list = []
    test_results: list = []

    for i in range(n_fits):
        # Use stored DB predictions when available — no model load or GPU needed.
        if db is not None and loop_idx is not None:
            stored = db.get_mace_predictions(loop_idx, i)
            if stored is not None:
                train_results.append(stored.get("train"))
                test_results.append(stored.get("test"))
                logger.info(
                    "Using stored DB predictions for loop %d fit %d.", loop_idx, i
                )
                continue

        # Fall back to live inference.
        try:
            from mace.calculators import MACECalculator
        except ImportError:
            logger.warning("mace not importable — skipping parity plots.")
            return

        train_atoms = _load_and_subsample(
            Path("results", base_name, "train_set.xyz"), seed, "train"
        )
        test_atoms = _load_and_subsample(
            Path("results", base_name, "test_set.xyz"), seed, "test"
        )

        if train_atoms is None and test_atoms is None:
            return

        model_path = (
            Path("results", base_name, name, f"fit_{i}")
            / f"{name}_stagetwo_compiled.model"
        )
        if not model_path.exists():
            logger.warning(
                "Model not found: %s — skipping fit_%d parity.", model_path, i
            )
            train_results.append(None)
            test_results.append(None)
            continue

        try:
            calc = MACECalculator(
                model_paths=[str(model_path)],
                device="cpu",
                default_dtype="float64",
            )
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", model_path, exc)
            train_results.append(None)
            test_results.append(None)
            continue

        train_results.append(_run_inference(calc, train_atoms) if train_atoms else None)
        test_results.append(_run_inference(calc, test_atoms) if test_atoms else None)

    if not train_results and not test_results:
        return

    has_train = any(r is not None for r in train_results)
    has_test = any(r is not None for r in test_results)

    if has_train:
        _draw_parity_figure(
            train_results, n_fits, name, seed, "Training", base_name, plots_dir, "train"
        )
    if has_test:
        _draw_parity_figure(
            test_results, n_fits, name, seed, "Test", base_name, plots_dir, "test"
        )
