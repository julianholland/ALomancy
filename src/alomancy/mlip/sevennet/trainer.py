"""SevenNetTrainer: the SevenNet backend of ALomancyTrainer (mlip/base.py).

Only SevenNet-specific work lives here: building SevenNet's arguments from
``training.sevennet_kwargs``, the E0s format, stage-two (SWA) timing, stress
training defaults, and which of MACE's output files is which. The standard
``train`` entry point, isolated-atom-energy resolution, evaluation,
restart checks and clean-up are inherited.

MACE writes two models per fit: ``<name>_stagetwo.model`` (uncompiled, used
for evaluation, parity plots, MD and prediction -- the TorchScript-compiled
model's forces failed for every multi-atom structure on one production
GPU/CUDA/PyTorch combination) and ``<name>_stagetwo_compiled.model``
(copied to results/best_model/). MACE writes the compiled one inside a bare
``except Exception: pass``, so it can be missing even after a successful fit.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from mace import tools
from mace.calculators import MACECalculator
from mace.cli.run_train import run

from alomancy.mlip.base import ALomancyTrainer
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
    from ase.data import atomic_numbers

    converted: dict[int, float] = {}
    for key, energy in e0s.items():
        if isinstance(key, str) and key in atomic_numbers:
            z = atomic_numbers[key]
        elif isinstance(key, int) and not isinstance(key, bool) and key > 0:
            z = key
        elif isinstance(key, str) and key.isdigit() and int(key) > 0:
            z = int(key)
        else:
            raise ValueError(
                f"mace_kwargs.E0s key {key!r} is not an element symbol or "
                "atomic number."
            )
        value = float(energy)
        if not np.isfinite(value):
            raise ValueError(f"mace_kwargs.E0s[{key!r}] is not finite: {energy!r}.")
        converted[z] = value
    return str(converted)


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
        """checkpoints/, once the compiled model exists: MACE only needs it to
        restore its best state before writing that model."""
        if not self.compiled_model_path(fit_dir).exists():
            return []
        return [Path(fit_dir) / "checkpoints"]

    def format_isolated_atom_energies(self, energies: dict) -> str:
        return _mace_e0s_arg(energies)

    def get_calculator(self, model_path: str | Path) -> Any:
        import torch

        device = self.config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return MACECalculator(
            model_paths=[str(model_path)],
            device=device,
            default_dtype=self.config.get("default_dtype", "float64"),
        )

    def report_section(self, stats: dict, **kwargs: Any) -> Any:
        from alomancy.analysis.report.sections import mace_section

        return mace_section(stats, **kwargs)

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
