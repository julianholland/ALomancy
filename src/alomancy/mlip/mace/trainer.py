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

Fine-tuning a foundation model (``mace_kwargs.foundation_model``) is naive
by default: one head, every weight trainable, no replay data. Its defaults
(``_MACE_NAIVE_FINETUNE_DEFAULTS``) follow the naive-fine-tuning
recommendations of

    T. L. Tompa, E. Varga-Umbrich, I. Batatia, A. M. Elena, N. Bernstein and
    G. Csányi, "Fine-tuning MLIP foundation models: strategies for accuracy
    and transferability", arXiv:2606.12704 (2026),
    https://doi.org/10.48550/arXiv.2606.12704

as summarised in MACE's fine-tuning guidance
(https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html):
learning rate 1e-3, EMA with decay 0.999, gradient clipping at 1.0, no
weight decay (it pulls parameters away from the pretrained solution), and
constant loss weights (energy 10, forces 10) throughout, since a two-stage
force-then-energy schedule can destabilise fine-tuning. Stage two still
runs, so MACE writes the ``_stagetwo`` models ALomancy reads, but with the
stage-one weights and learning rate. The epoch count (dynamic, as from
scratch) and the E0s (isolated-atom DFT) are ALomancy's own choices, not
the study's. Multihead (replay) fine-tuning is opt-in
(``multiheads_finetuning: true`` or a ``pt_train_file``).
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

# Merged between _MACE_KWARGS_DEFAULTS and the user's mace_kwargs for a naive
# fine-tune of a foundation model (_is_naive_finetune): Tompa et al. 2026,
# arXiv:2606.12704, naive row of MACE's fine-tuning guidance (module
# docstring). amsgrad, rms_forces_scaling and float64 are MACE's own
# defaults already.
_MACE_NAIVE_FINETUNE_DEFAULTS: dict[str, Any] = {
    "multiheads_finetuning": False,
    "lr": 1e-3,
    "ema": True,
    "ema_decay": 0.999,
    "clip_grad": 1.0,
    "weight_decay": 0.0,
    "energy_weight": 10,
    "forces_weight": 10,
}
# Stage-two settings a naive fine-tune keeps equal to stage one (constant
# loss weights and learning rate), unless set in mace_kwargs. Values are
# (stage-one key, MACE's default for it).
_STAGE_TWO_MIRRORS: dict[str, tuple[str, float]] = {
    "swa_energy_weight": ("energy_weight", 1.0),
    "swa_forces_weight": ("forces_weight", 100.0),
    "swa_stress_weight": ("stress_weight", 1.0),
    "swa_lr": ("lr", 0.01),
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


def _write_resolved_mace_epochs(
    fit_dir: Path, mace_fit_params: dict, replay: dict[str, int] | None = None
) -> None:
    """Save the resolved max_num_epochs/start_swa to resolved_mace_epochs.json,
    fixed or dynamic alike, so the training-curve plots (mlip_plots.py) can
    mark the real stage-two epoch without re-deriving it. A multihead fit
    also records the epoch size it was resolved from (_replay_epoch_size)."""
    payload: dict[str, Any] = {
        "max_num_epochs": mace_fit_params["max_num_epochs"],
        "start_swa": mace_fit_params["start_swa"],
    }
    if replay is not None:
        payload.update(replay)
    with open(fit_dir / "resolved_mace_epochs.json", "w") as fh:
        json.dump(payload, fh)


# MACE 0.3.16's own defaults for the multihead settings read below
# (mace/tools/arg_parser.py); a value in mace_kwargs overrides each.
_MACE_MULTIHEAD_DEFAULTS: dict[str, Any] = {
    "multiheads_finetuning": True,
    "num_samples_pt": 10000,
    "valid_fraction": 0.1,
    "real_pt_data_ratio_threshold": 0.1,
}
# Foundation models MACE fine-tunes with Materials Project replay data
# without a pt_train_file (run_train.py), and the pt_train_file keywords it
# downloads and subsamples itself (multihead_tools.prepare_pt_head).
_MP_FOUNDATION_MODELS = ("small", "medium", "large")
_REPLAY_KEYWORDS = ("mp", "omat", "matpes_pbe", "matpes_r2scan")


def _mace_setting(mace_kwargs: dict, key: str) -> Any:
    return mace_kwargs.get(key, _MACE_MULTIHEAD_DEFAULTS[key])


def _truthy(value: Any) -> bool:
    """MACE parses its booleans with str2bool; a YAML string "False" would
    otherwise be truthy once set directly on its namespace."""
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "no", "0", "")
    return bool(value)


def _is_naive_finetune(user_kwargs: dict) -> bool:
    """A foundation model fine-tuned without replay: the user set
    foundation_model, and neither multiheads_finetuning: true nor a
    pt_train_file (which only means something for multihead)."""
    if user_kwargs.get("foundation_model") is None:
        return False
    if user_kwargs.get("pt_train_file") is not None:
        return False
    multihead = user_kwargs.get("multiheads_finetuning")
    return multihead is None or not _truthy(multihead)


def _multihead_replay(mace_kwargs: dict) -> str | None:
    """The replay data MACE will fine-tune with (a keyword or a file path),
    or None when it trains a single head. Mirrors run_train.py: with a
    foundation model, multihead mode is on by default, but MACE silently
    turns it off for a foundation model outside small/medium/large unless
    pt_train_file is given -- logged here, since that's easy to miss.
    A user ``heads`` dict is left to MACE (returns None)."""
    foundation = mace_kwargs.get("foundation_model")
    if foundation is None or "heads" in mace_kwargs:
        return None
    if not _truthy(_mace_setting(mace_kwargs, "multiheads_finetuning")):
        return None
    pt_train_file = mace_kwargs.get("pt_train_file")
    if pt_train_file is None:
        if foundation in _MP_FOUNDATION_MODELS:
            return "mp"
        logger.warning(
            "foundation_model %s is not a Materials Project model and "
            "mace_kwargs.pt_train_file is unset, so MACE turns multihead "
            "fine-tuning off and fine-tunes every weight on this data alone. "
            'Set pt_train_file (e.g. "mp" or a replay .xyz) for replay '
            "fine-tuning.",
            foundation,
            extra={
                "event": "multihead_finetuning_disabled",
                "data": {"foundation_model": str(foundation)},
            },
        )
        return None
    return str(pt_train_file)


def _atomic_numbers_arg(mace_kwargs: dict, elements: list[str] | None) -> None:
    """Set MACE's atomic_numbers from general.elements, in place, for a
    multihead fit: MACE filters its replay data by these elements only
    (assemble_replay_data passes args.atomic_numbers), and any
    filter_type_pt but "none" raises without them. A user value wins; one
    missing an element of general.elements is warned about."""
    import ast

    from ase.data import atomic_numbers

    if not elements:
        return
    wanted = sorted({atomic_numbers[el] for el in elements})
    given = mace_kwargs.get("atomic_numbers")
    if given is None:
        mace_kwargs["atomic_numbers"] = str(wanted)
        return
    given_z = ast.literal_eval(given) if isinstance(given, str) else given
    missing = sorted(set(wanted) - set(given_z))
    if missing:
        logger.warning(
            "mace_kwargs.atomic_numbers %s misses %s from general.elements %s: "
            "MACE's replay data won't cover them.",
            given,
            missing,
            elements,
        )


def _replay_structures(source: str, fit_dir: Path, mace_kwargs: dict) -> int:
    """How many replay structures MACE trains the pretraining head on,
    before its validation split. A keyword source is subsampled to
    num_samples_pt: padded up to it by default, and without padding either
    cut to exactly it or MACE raises, so the number is exact whenever
    training runs. A file source is used whole (MACE resolves it from the
    fit directory)."""
    if source in _REPLAY_KEYWORDS:
        return int(_mace_setting(mace_kwargs, "num_samples_pt"))
    from ase.io import iread

    path = Path(source)
    if not path.is_absolute():
        path = fit_dir / path
    return sum(1 for _ in iread(path, index=":"))


def _replay_epoch_size(
    mace_kwargs: dict, source: str, n_train: int, fit_dir: Path
) -> dict[str, int]:
    """The structures one MACE multihead epoch goes through, by head,
    mirroring MACE 0.3.16:
    - replay train split: MACE keeps valid_fraction of the replay data for
      validation unless pt_valid_file is given
      (data.utils.random_train_valid_split:
      min(n - int(valid_fraction * n), n - 1));
    - fine-tuning copies: when this data is under real_pt_data_ratio_threshold
      of the replay train split, MACE adds int(threshold / ratio) copies of
      it (run_train.py, "increasing the number of configurations in the
      fine-tuning heads")."""
    n_replay = _replay_structures(source, fit_dir, mace_kwargs)
    if mace_kwargs.get("pt_valid_file") or n_replay < 2:
        n_replay_train = n_replay
    else:
        valid_fraction = float(_mace_setting(mace_kwargs, "valid_fraction"))
        n_replay_train = min(n_replay - int(valid_fraction * n_replay), n_replay - 1)
    copies = 0
    if n_replay_train > 0:
        threshold = float(_mace_setting(mace_kwargs, "real_pt_data_ratio_threshold"))
        ratio = n_train / n_replay_train
        if 0 < ratio < threshold:
            copies = int(threshold / ratio)
    n_fine_tuning = n_train * (1 + copies)
    return {
        "replay_structures": n_replay,
        "replay_train_structures": n_replay_train,
        "fine_tuning_copies": copies,
        "fine_tuning_train_structures": n_fine_tuning,
        "epoch_train_structures": n_replay_train + n_fine_tuning,
    }


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

    def __init__(self, config: dict, name: str = "training") -> None:
        """As ALomancyTrainer, plus the naive fine-tuning defaults merged
        between the MACE defaults and the user's mace_kwargs when a
        foundation model is fine-tuned without replay (module docstring)."""
        super().__init__(config, name=name)
        self._user_kwargs: dict[str, Any] = dict(config.get(self.KWARGS_KEY) or {})
        self.naive_finetune = _is_naive_finetune(self._user_kwargs)
        if self.naive_finetune:
            self.kwargs = {
                **self.KWARGS_DEFAULTS,
                **_MACE_NAIVE_FINETUNE_DEFAULTS,
                **self._user_kwargs,
            }

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
        elements: list[str] | None = None,
    ) -> Path | None:
        """Run MACE's own training in *fit_dir*; return the uncompiled model.
        In multihead fine-tuning, *elements* (general.elements) become
        MACE's atomic_numbers and the replay data counts toward the dynamic
        epoch size."""
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
        # Epochs count train + valid, the pool this loop trains from, plus in
        # multihead fine-tuning what MACE adds to every epoch: the replay
        # train split and its extra copies of this data.
        n_epoch = n_train + n_valid
        replay = None
        source = _multihead_replay(mace_kwargs)
        if source is not None:
            _atomic_numbers_arg(mace_kwargs, elements)
            replay = _replay_epoch_size(mace_kwargs, source, n_train, Path(fit_dir))
            n_epoch = replay["epoch_train_structures"] + n_valid
            logger.info(
                "Multihead fine-tuning with replay data %s: %d replay "
                "structure(s), %d in its train split; this data x%d (MACE's "
                "copies for a small fine-tuning set); %d structure(s) per epoch.",
                source,
                replay["replay_structures"],
                replay["replay_train_structures"],
                1 + replay["fine_tuning_copies"],
                n_epoch,
            )
        # Control values, not MACE arguments: popped so they never reach MACE.
        epochs = resolve_epochs(
            mace_kwargs.pop("max_num_epochs", None),
            mace_kwargs.get("batch_size", 16),
            n_epoch,
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
        if self.naive_finetune:
            # Constant loss weights and learning rate through stage two.
            for swa_key, (key, mace_default) in _STAGE_TWO_MIRRORS.items():
                if swa_key not in self._user_kwargs:
                    mace_fit_params[swa_key] = mace_fit_params.get(key, mace_default)
        if valid_path is not None:
            mace_fit_params["valid_file"] = str(valid_path)
        _apply_compute_stress_defaults(mace_fit_params, compute_stress)
        _write_resolved_mace_epochs(fit_dir, mace_fit_params, replay)

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
