"""MaceTrainer: the MACE backend of ALomancyTrainer (mlip/base.py).

Only MACE-specific work lives here: building MACE's arguments from
``training.mace_kwargs``, the E0s format, stage-two (SWA) timing, stress
training defaults, and which of MACE's output files is which. The standard
``train`` entry point, isolated-atom-energy resolution, evaluation,
restart checks and clean-up are inherited.

MACE writes two models per fit: ``<name>_stagetwo.model`` (uncompiled, used
for evaluation, parity plots, MD and prediction -- the TorchScript-compiled
model's forces failed for every multi-atom structure on one production
GPU/CUDA/PyTorch combination) and ``<name>_stagetwo_compiled.model``
(copied to results/best_model/). MACE writes the compiled one inside a bare
``except Exception: pass``, so it can be missing even after a successful fit.
MACE also writes stage-one ``<name>.model``/``<name>_compiled.model``; no
code reads them, so they are cleaned up with ``checkpoints/``.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from ase.io import read
from mace import tools
from mace.calculators import MACECalculator
from mace.cli.run_train import run

from alomancy.mlip.base import (
    ALomancyTrainer,
    TrainingHistory,
    isolated_atom_reference_energies_by_z,
)
from alomancy.utils.training_schedule import resolve_epochs

logger = logging.getLogger(__name__)

# Defaults for training.mace_kwargs, merged under the user's settings. Also
# shown, fully resolved, by the workflow's config summary.
# max_num_epochs: "dynamic" (utils/training_schedule.py) rather than MACE's
# own 2048, which assumes separately tuned early stopping. E0s has no entry:
# it defaults to the database's IsolatedAtom energies at train time.
_MACE_KWARGS_DEFAULTS: dict[str, Any] = {
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "max_num_epochs": "dynamic",
    "compute_stress": False,
    "model": "MACE",
    "correlation": 3,
    "device": "cuda",
    "ema": None,
    "energy_weight": 1,
    "forces_weight": 10,
    "error_table": "PerAtomMAE",
    "eval_interval": 1,
    "max_L": 2,
    "num_channels": 128,
    "num_interactions": 2,
    "patience": 30,
    "r_max": 5.0,
    "restart_latest": None,
    "save_cpu": None,
    "scheduler_patience": 15,
    "swa": None,
    "batch_size": 16,
    "valid_batch_size": 16,
    "distributed": None,
}

# Stage two (SWA) starts at this fraction of the epochs.
_STAGE_TWO_FRACTION = 0.8

if (
    importlib.util.find_spec("torch._native") is not None
    and "TRITON_CACHE_DIR" not in os.environ
):
    logger.warning(
        "This PyTorch version uses triton for native GPU ops (torch._native). "
        "On HPC nodes without python3-dev headers, triton kernel compilation will "
        "fail at training time. Set TRITON_CACHE_DIR to a persistent path and "
        "pre-warm the cache in an interactive GPU job. See CLAUDE.md for details."
    )


def _write_resolved_mace_epochs(fit_dir: Path, mace_fit_params: dict) -> None:
    """Save the resolved max_num_epochs/start_swa to resolved_mace_epochs.json,
    fixed or dynamic alike, so the training-curve plots (mlip_plots.py) can
    mark the real stage-two epoch without re-deriving it."""
    payload = {
        "max_num_epochs": mace_fit_params["max_num_epochs"],
        "start_swa": mace_fit_params["start_swa"],
    }
    with open(fit_dir / "resolved_mace_epochs.json", "w") as fh:
        json.dump(payload, fh)


def _apply_compute_stress_defaults(mace_fit_params: dict, compute_stress: bool) -> None:
    """Enable stress training when requested, in place.

    setdefault, so a user's own loss/stress_key (e.g. "huber", "universal")
    wins. A bare stress_key does nothing on its own: MACE only trains on
    stress when loss is "stress"/"huber"/"universal".
    """
    if compute_stress:
        mace_fit_params.setdefault("stress_key", "REF_stresses")
        mace_fit_params.setdefault("loss", "stress")


def _mace_e0s_arg(e0s: dict) -> str:
    """{element symbol or atomic number: energy (eV)} -> the string MACE's
    E0s argument expects: a dict literal keyed by atomic number, e.g.
    ``"{1: -13.6, 8: -432.1}"``."""
    return str(isolated_atom_reference_energies_by_z(e0s, "mace_kwargs.E0s"))


class MaceTrainer(ALomancyTrainer):
    """MACE backend (training.trainer: mace, settings in training.mace_kwargs)."""

    NAME = "mace"
    KWARGS_KEY = "mace_kwargs"
    KWARGS_DEFAULTS = _MACE_KWARGS_DEFAULTS
    ISOLATED_ATOM_ENERGIES_KWARG = "E0s"

    def model_path(self, fit_dir: Path) -> Path:
        return Path(fit_dir) / f"{self.name}_stagetwo.model"

    def compiled_model_path(self, fit_dir: Path) -> Path:
        return Path(fit_dir) / f"{self.name}_stagetwo_compiled.model"

    def deployable_model_path(self, fit_dir: Path) -> Path | None:
        """The compiled model, or None (best_model/ is then left as it was)."""
        path = self.compiled_model_path(fit_dir)
        return path if path.exists() else None

    def cleanup_paths(self, fit_dir: Path) -> list[Path]:
        """Once the compiled model exists: checkpoints/ (MACE only needs it to
        restore its best state before writing that model) and the stage-one
        models <name>.model / <name>_compiled.model (about 88 MB each),
        which nothing reads -- evaluation, MD and prediction use the
        uncompiled stage-two model, results/best_model/ the compiled one."""
        if not self.compiled_model_path(fit_dir).exists():
            return []
        fit_dir = Path(fit_dir)
        return [
            fit_dir / "checkpoints",
            fit_dir / f"{self.name}.model",
            fit_dir / f"{self.name}_compiled.model",
        ]

    def format_isolated_atom_energies(self, energies: dict) -> str:
        return _mace_e0s_arg(energies)

    def get_calculator(
        self, model_path: str | Path, *, device: str | None = None
    ) -> Any:
        import torch

        device = (
            device
            or self.config.get("device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        return MACECalculator(
            model_paths=[str(model_path)],
            device=device,
            default_dtype=self.config.get("default_dtype", "float64"),
        )

    def report_section(self, stats: dict, **kwargs: Any) -> Any:
        from alomancy.analysis.report.sections import trainer_section

        return trainer_section(stats, trainer=self, **kwargs)

    def training_history(self, fit_dir: Path, seed: int) -> TrainingHistory | None:
        """MACE's per-epoch validation records (``results/*_train.txt``), the
        stage-two start (resolved_mace_epochs.json) and the epoch MACE
        restored its stage-two model from (``logs/*.log``)."""
        from alomancy.analysis.mlip_plots import (
            _parse_training_jsonl,
            _parse_used_epoch,
        )

        fit_dir = Path(fit_dir)
        frame = _parse_training_jsonl(fit_dir, self.name, seed)
        if frame is None:
            return None
        stage_two = None
        sidecar = fit_dir / "resolved_mace_epochs.json"
        if sidecar.exists():
            try:
                stage_two = int(json.loads(sidecar.read_text())["start_swa"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Could not read %s: %s", sidecar, exc)
        return TrainingHistory(
            frame=frame.with_columns(split=pl.lit("valid")),
            stage_two_epoch=stage_two,
            selected_epoch=_parse_used_epoch(fit_dir, self.name, seed),
        )

    def fit(
        self,
        train_path: Path,
        valid_path: Path | None,
        test_path: Path,
        seed: int,
        fit_dir: Path,
        *,
        isolated_atom_energies: Any,
    ) -> Path | None:
        """Run MACE's own training in *fit_dir*; return the uncompiled model."""
        mace_kwargs = dict(self.kwargs)
        if "seed" in mace_kwargs:
            raise ValueError(
                "mace_kwargs must not set 'seed' -- it is derived and passed "
                "separately (general.seed + fit index)."
            )
        if isolated_atom_energies is not None:
            mace_kwargs["E0s"] = isolated_atom_energies

        n_train = len(read(train_path, ":", format="extxyz"))
        n_valid = len(read(valid_path, ":", format="extxyz")) if valid_path else 0
        logger.info(
            "Read %d training structure(s), %d validation structure(s).",
            n_train,
            n_valid,
        )
        # Control values, not MACE arguments: popped so they never reach MACE.
        # Epochs count train + valid, the pool this loop trains from.
        epochs = resolve_epochs(
            mace_kwargs.pop("max_num_epochs", None),
            mace_kwargs.get("batch_size", 16),
            n_train + n_valid,
        )
        compute_stress = mace_kwargs.pop("compute_stress", False)

        mace_fit_params = {
            **mace_kwargs,
            "train_file": str(train_path),
            "test_file": str(test_path),
            "name": self.name,
            "max_num_epochs": epochs,
            "start_swa": int(np.floor(epochs * _STAGE_TWO_FRACTION)),
            "seed": seed,
        }
        if "start_swa" in mace_kwargs:
            mace_fit_params["start_swa"] = mace_kwargs["start_swa"]
        if valid_path is not None:
            mace_fit_params["valid_file"] = str(valid_path)
        _apply_compute_stress_defaults(mace_fit_params, compute_stress)
        _write_resolved_mace_epochs(fit_dir, mace_fit_params)

        logger.debug("MACE fit parameters:")
        for key, value in mace_fit_params.items():
            logger.debug("  %s: %s", key, value)

        parser = tools.build_default_arg_parser()
        args = parser.parse_args(["--name", self.name])
        for key, value in mace_fit_params.items():
            setattr(args, key, value)

        orig_dir = os.getcwd()
        try:
            os.chdir(fit_dir)
            run(args)
        finally:
            os.chdir(orig_dir)
        model = self.model_path(fit_dir)
        return model if model.exists() else None
