"""SevenNetTrainer: the SevenNet backend of ALomancyTrainer (mlip/base.py).

Only SevenNet-specific work lives here: turning our extxyz labels into
SevenNet graphs, building the model and training configs from
``training.sevennet_kwargs``, the missing-stress masking, and keeping one
slim model. The epoch loop itself is SevenNet's own ``processing_epoch_v2``
(what ``sevenn train`` runs since sevenn 0.9.6), handed our trainer,
loaders and error recorder.
The standard ``train`` entry point, isolated-atom-energy resolution,
evaluation, restart checks and clean-up are inherited.

``sevennet_kwargs`` mirrors SevenNet's own ``input.yaml``: ``model``,
``train`` and ``data`` sections using SevenNet's key names, each merged over
the defaults below. A few values are always derived and may not be set:

- ``data.shift``: the per-element isolated-atom reference energies
  (``sevennet_kwargs.isolated_atom_reference_energies``, else the database's
  IsolatedAtom energies), so the network learns the energy relative to them,
  like MACE's E0s;
- ``model.chemical_species``: the elements of those reference energies;
- ``train.random_seed``: the fit's seed (general.seed + fit index).

``train.epoch`` is ``"dynamic"`` by default (utils/training_schedule.py).

Files written in the fit directory (``<name>`` is the trainer's name,
"training" in the workflow):

- ``<name>_sevennet.pth``: the model (weights, config, epoch; no optimizer
  state, about 0.5 MB for the defaults), from the best validation epoch, or
  the last epoch without a valid split;
- ``lc.csv``: SevenNet's per-epoch errors (one row per epoch, columns per
  split), read by ``training_history``;
- ``log.sevenn``: SevenNet's own human-readable training log;
- ``<name>_sevennet_fit.json``: resolved epochs, kept epoch, elements and
  the shift/scale used.

SevenNet's ``checkpoint_*.pth`` files are deleted once the model is saved
(``cleanup_paths`` also removes any that ExPyRe synced back first). Graphs
are built in memory, so there is no ``sevenn_data/``.
"""

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from ase.data import chemical_symbols
from ase.io import read

from alomancy.mlip.base import (
    ALomancyTrainer,
    TrainingHistory,
    isolated_atom_reference_energies_by_z,
)
from alomancy.utils.training_schedule import resolve_epochs

logger = logging.getLogger(__name__)

# e3nn 0.4.4 (pinned by mace-torch, so also what SevenNet runs on here)
# torch.load()s its own constants file, which fails under torch>=2.6's
# weights_only default. MACE sets the same variable on import; SevenNet
# alone (no MACE import first) would otherwise fail to import at all.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Defaults for training.sevennet_kwargs, section by section (merged under the
# user's settings per section). Taken from SevenNet's example input.yaml,
# except: epoch is "dynamic", and stress training is opt-in, like MACE's
# compute_stress. SevenNet's own defaults fill anything not listed.
_SEVENNET_KWARGS_DEFAULTS: dict[str, Any] = {
    "model": {
        "cutoff": 5.0,
        "channel": 32,
        "lmax": 2,
        "num_convolution_layer": 3,
        "is_parity": False,
        "self_connection_type": "nequip",
        "conv_denominator": "avg_num_neigh",
    },
    "train": {
        "epoch": "dynamic",
        "device": "cuda",
        "is_train_stress": False,
        "optimizer": "adam",
        "optim_param": {"lr": 0.005},
        "scheduler": "exponentiallr",
        "scheduler_param": {"gamma": 0.99},
        "force_loss_weight": 0.1,
        "stress_loss_weight": 1e-6,
    },
    "data": {
        "batch_size": 4,
        "scale": "force_rms",
    },
}
_SECTIONS = ("model", "train", "data")
_REFERENCE_ENERGIES_KEY = "isolated_atom_reference_energies"

# Always recorded, so training_history has energy/force MAE and the loss
# that picks the kept epoch. Stress is left out: structures without stress
# carry NaN labels, and evaluate() reports stress errors after the fit.
_REQUIRED_ERROR_RECORD = [["Energy", "MAE"], ["Force", "MAE"], ["TotalLoss", "None"]]
# SevenNet's own per-epoch files (fixed names, written by processing_epoch_v2
# and its Logger).
_SEVENNET_CSV = "lc.csv"
_SEVENNET_LOG = "log.sevenn"
# ErrorRecorder metric names -> TrainingHistory columns. In lc.csv they are
# prefixed with the loader name ("trainset_" / "validset_").
_SPLIT_PREFIXES = {"trainset_": "train", "validset_": "valid"}
_HISTORY_COLUMNS = {
    "Energy_MAE": "mae_e_per_atom",
    "Force_MAE": "mae_f",
    "TotalLoss": "loss",
}


def _sevenn_stress(ref_stress: Any) -> np.ndarray:
    """ASE Voigt stress (xx, yy, zz, yz, xz, xy; eV/Å³, ASE sign) -> the
    6-vector SevenNet trains on: (xx, yy, zz, xy, yz, zx), opposite sign.
    The same conversion SevenNet applies itself to a stress_key label."""
    return -np.asarray(ref_stress, dtype=float)[[0, 1, 2, 5, 3, 4]]


def _read_graphs(path: Path, cutoff: float) -> list:
    """SevenNet graphs from one of our extxyz split files: REF_energy,
    REF_forces and (where present) REF_stresses as labels. Structures
    without stress get NaN stress, which the loss ignores.

    IsolatedAtom structures are kept, unlike for MACE: SevenNet still passes
    a lone atom's own features through its readout, so its energy is not
    pinned to the shift and needs the training data to anchor it."""
    from sevenn.train.dataload import graph_build

    atoms_list = []
    for atoms in read(path, ":", format="extxyz"):
        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            raise ValueError(
                f"A structure in {path} has no REF_energy/REF_forces "
                f"(config_type={atoms.info.get('config_type')!r})."
            )
        stress = atoms.info.get("REF_stresses")
        atoms.info["y_energy"] = float(atoms.info["REF_energy"])
        atoms.arrays["y_force"] = np.asarray(atoms.arrays["REF_forces"], dtype=float)
        atoms.info["y_stress"] = (
            _sevenn_stress(stress) if stress is not None else np.full(6, np.nan)
        )
        atoms_list.append(atoms)
    return graph_build(atoms_list, cutoff, transfer_info=False)


class _FiniteOnly:
    """Wraps a loss criterion so entries with a non-finite label or
    prediction (missing stress) are ignored. With nothing left it returns
    zero (*skip_empty* False, for the training loss) or an empty tensor
    (*skip_empty* True, so an error metric doesn't count the batch)."""

    def __init__(self, criterion: Any, skip_empty: bool = False) -> None:
        self.criterion = criterion
        self.skip_empty = skip_empty

    def __call__(self, a: Any, b: Any) -> Any:
        import torch

        keep = torch.isfinite(a) & torch.isfinite(b)
        if bool(keep.all()):
            return self.criterion(a, b)
        if not bool(keep.any()):
            return a.new_zeros(0) if self.skip_empty else (a * 0.0).sum()
        return self.criterion(a[keep], b[keep])


def _mask_missing_stress(trainer: Any, recorder: Any) -> None:
    """Make the stress loss, and the stress term of the recorded TotalLoss
    (which picks the kept epoch), ignore structures without stress."""
    from sevenn.error_recorder import CombinedError, CustomError

    for loss_def, _ in trainer.loss_functions:
        if loss_def.name == "Stress":
            loss_def.criterion = _FiniteOnly(loss_def.criterion)
    for metric in recorder.metrics:
        if isinstance(metric, CombinedError):
            for sub, _ in metric.metrics:
                if isinstance(sub, CustomError) and sub.name == "Stress":
                    sub.func = _FiniteOnly(sub.func, skip_empty=True)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto" or (
        device.startswith("cuda") and not torch.cuda.is_available()
    ):
        if device != "auto":
            logger.warning(
                "sevennet_kwargs.train.device=%r but CUDA is unavailable; using cpu.",
                device,
            )
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _statistic(value: Any, stats: dict, allowed: dict, setting: str) -> float:
    """A float as given, or the named dataset statistic."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if value in allowed:
        return float(allowed[value](stats))
    raise ValueError(
        f"sevennet_kwargs.{setting} must be a number or one of {sorted(allowed)}, "
        f"got {value!r}."
    )


def _force_rms(stats: dict) -> float:
    import sevenn._keys as KEY

    return float((stats[KEY.FORCE]["mean"] ** 2 + stats[KEY.FORCE]["std"] ** 2) ** 0.5)


_SCALES = {
    "force_rms": _force_rms,
    "per_atom_energy_std": lambda s: s["per_atom_energy"]["std"],
}
_CONV_DENOMINATORS = {
    "avg_num_neigh": lambda s: s["num_neighbor"]["mean"],
    "sqrt_avg_num_neigh": lambda s: s["num_neighbor"]["mean"] ** 0.5,
}


class SevenNetTrainer(ALomancyTrainer):
    """SevenNet backend (training.trainer: sevennet, settings in training.sevennet_kwargs)."""

    NAME = "sevennet"
    KWARGS_KEY = "sevennet_kwargs"
    KWARGS_DEFAULTS = _SEVENNET_KWARGS_DEFAULTS
    ISOLATED_ATOM_ENERGIES_KWARG = _REFERENCE_ENERGIES_KEY

    def __init__(self, config: dict, name: str = "training") -> None:
        super().__init__(config, name)
        user = config.get(self.KWARGS_KEY) or {}
        unknown = set(user) - {*_SECTIONS, _REFERENCE_ENERGIES_KEY}
        if unknown:
            raise ValueError(
                f"Unknown sevennet_kwargs key(s) {sorted(unknown)}: expected the "
                f"sections {list(_SECTIONS)} (as in SevenNet's input.yaml) and "
                f"optionally {_REFERENCE_ENERGIES_KEY}."
            )
        # The base merge is shallow; merge each section over its defaults so
        # e.g. model: {channel: 64} keeps the other model defaults.
        for section in _SECTIONS:
            self.kwargs[section] = {
                **_SEVENNET_KWARGS_DEFAULTS[section],
                **(user.get(section) or {}),
            }
        derived = {
            ("train", "random_seed"): "it is the fit's seed (general.seed + fit index)",
            ("data", "shift"): f"it is set from {_REFERENCE_ENERGIES_KEY}",
            (
                "model",
                "chemical_species",
            ): f"it is the elements of {_REFERENCE_ENERGIES_KEY}",
        }
        for (section, key), why in derived.items():
            if key in self.kwargs[section]:
                raise ValueError(
                    f"sevennet_kwargs.{section}.{key} must not be set: {why}."
                )

    def model_path(self, fit_dir: Path) -> Path:
        return Path(fit_dir) / f"{self.name}_sevennet.pth"

    def cleanup_paths(self, fit_dir: Path) -> list[Path]:
        """SevenNet's checkpoints. fit() already deletes them on the remote
        node, but ExPyRe's additive sync may have copied some back first;
        the local clean-up pass removes those."""
        return _checkpoints(fit_dir)

    def _fit_info_path(self, fit_dir: Path) -> Path:
        return Path(fit_dir) / f"{self.name}_sevennet_fit.json"

    def format_isolated_atom_energies(self, energies: dict) -> dict[int, float]:
        return isolated_atom_reference_energies_by_z(
            energies, f"sevennet_kwargs.{_REFERENCE_ENERGIES_KEY}"
        )

    def get_calculator(
        self, model_path: str | Path, *, device: str | None = None
    ) -> Any:
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator(
            str(model_path),
            device=device or _resolve_device(self.kwargs["train"]["device"]),
        )

    def report_section(self, stats: dict, **kwargs: Any) -> Any:
        from alomancy.analysis.report.sections import trainer_section

        return trainer_section(stats, trainer=self, **kwargs)

    def training_history(self, fit_dir: Path, seed: int) -> TrainingHistory | None:  # noqa: ARG002
        """SevenNet's lc.csv (one row per epoch, columns per loader such as
        trainset_Force_MAE), reshaped to one row per epoch per split, plus
        the kept epoch from <name>_sevennet_fit.json."""
        log = Path(fit_dir) / _SEVENNET_CSV
        if not log.exists():
            return None
        try:
            wide = pl.read_csv(log)
        except Exception as exc:
            logger.warning("Could not read %s: %s", log, exc)
            return None
        parts = []
        for prefix, split in _SPLIT_PREFIXES.items():
            columns = {
                c: _HISTORY_COLUMNS[c[len(prefix) :]]
                for c in wide.columns
                if c.startswith(prefix) and c[len(prefix) :] in _HISTORY_COLUMNS
            }
            if columns:
                parts.append(
                    wide.select(["epoch", *columns])
                    .rename(columns)
                    .with_columns(split=pl.lit(split))
                )
        if not parts:
            return None
        frame = pl.concat(parts, how="diagonal").sort(["epoch", "split"])
        selected = None
        info = self._fit_info_path(fit_dir)
        if info.exists():
            try:
                selected = json.loads(info.read_text()).get("selected_epoch")
            except (OSError, ValueError) as exc:
                logger.warning("Could not read %s: %s", info, exc)
        return TrainingHistory(frame=frame, selected_epoch=selected)

    def fit(
        self,
        train_path: Path,
        valid_path: Path | None,
        test_path: Path,  # noqa: ARG002 -- never used for training or selection
        seed: int,
        fit_dir: Path,
        *,
        isolated_atom_energies: Any,
    ) -> Path | None:
        """Train SevenNet in *fit_dir*; return the kept checkpoint."""
        import sevenn
        import sevenn.util as util
        from sevenn._const import (
            DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG,
            DEFAULT_TRAINING_CONFIG,
        )
        from sevenn.error_recorder import ErrorRecorder
        from sevenn.model_build import build_E3_equivariant_model
        from sevenn.scripts.processing_epoch import processing_epoch_v2
        from sevenn.sevenn_logger import Logger
        from sevenn.train.graph_dataset import _run_stat
        from sevenn.train.trainer import Trainer
        from torch_geometric.loader import DataLoader

        if not isinstance(isolated_atom_energies, dict) or not isolated_atom_energies:
            raise ValueError(
                f"SevenNet needs per-element {_REFERENCE_ENERGIES_KEY} as an "
                "{element: eV} dict (sevennet_kwargs or the database's IsolatedAtom "
                f"structures); got {isolated_atom_energies!r}."
            )
        _seed_everything(seed)
        model_kwargs = deepcopy(self.kwargs["model"])
        train_kwargs = deepcopy(self.kwargs["train"])
        data_kwargs = self.kwargs["data"]
        cutoff = float(model_kwargs["cutoff"])

        train_graphs = _read_graphs(train_path, cutoff)
        valid_graphs = _read_graphs(valid_path, cutoff) if valid_path else []
        if not train_graphs:
            raise ValueError(f"No trainable structures in {train_path}.")
        logger.info(
            "Built %d training graph(s), %d validation graph(s).",
            len(train_graphs),
            len(valid_graphs),
        )
        batch_size = int(data_kwargs["batch_size"])
        epochs = resolve_epochs(
            train_kwargs.pop("epoch"), batch_size, len(train_graphs) + len(valid_graphs)
        )

        # Model: elements and per-element shift from the reference energies;
        # scale and conv_denominator from the training graphs' statistics.
        import sevenn._keys as KEY

        stats = _run_stat(train_graphs, y_keys=[KEY.PER_ATOM_ENERGY, KEY.FORCE])
        species = sorted(chemical_symbols[z] for z in isolated_atom_energies)
        shift = [0.0] * len(chemical_symbols)
        for z, energy in isolated_atom_energies.items():
            shift[z] = float(energy)
        model_cfg = {
            **deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG),
            **model_kwargs,
            **util.chemical_species_preprocess(species),
            "shift": shift,
            "scale": _statistic(data_kwargs["scale"], stats, _SCALES, "data.scale"),
            "conv_denominator": _statistic(
                model_kwargs["conv_denominator"],
                stats,
                _CONV_DENOMINATORS,
                "model.conv_denominator",
            ),
            "version": sevenn.__version__,
        }
        error_record = [list(e) for e in train_kwargs.get("error_record", [])]
        error_record = _REQUIRED_ERROR_RECORD + [
            e for e in error_record if e not in _REQUIRED_ERROR_RECORD
        ]
        train_cfg = {
            **deepcopy(DEFAULT_TRAINING_CONFIG),
            **train_kwargs,
            "epoch": epochs,
            "random_seed": seed,
            "device": _resolve_device(train_kwargs["device"]),
            "error_record": error_record,
        }
        logger.debug("SevenNet model config: %s", model_cfg)
        logger.debug("SevenNet training config: %s", train_cfg)

        model = build_E3_equivariant_model(model_cfg)
        trainer = Trainer.from_config(model, train_cfg)
        # SevenNet copies this recorder once per loader, masking included.
        recorder = ErrorRecorder.from_config(train_cfg)
        if train_cfg["is_train_stress"]:
            _mask_missing_stress(trainer, recorder)
        loaders = {
            "trainset": DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        }
        if valid_graphs:
            loaders["validset"] = DataLoader(valid_graphs, batch_size=batch_size)

        # A retry into the same directory must start clean: SevenNet would
        # otherwise write lc0.csv next to an old lc.csv.
        for stale in (*_checkpoints(fit_dir), *fit_dir.glob("lc*.csv")):
            stale.unlink()
        (fit_dir / _SEVENNET_LOG).unlink(missing_ok=True)

        # SevenNet's own epoch loop: writes lc.csv every epoch,
        # checkpoint_best.pth on a new best validation TotalLoss, and (with
        # per_epoch=epochs) checkpoint_<epochs>.pth at the end -- the
        # fallback when there is no validation split. Its scheduler step
        # gets the best validation loss so far (only reducelronplateau
        # uses it).
        checkpoint_config = {**model_cfg, **train_cfg}
        with Logger().switch_file(str(fit_dir / _SEVENNET_LOG)):
            processing_epoch_v2(
                checkpoint_config,
                trainer,
                loaders,
                error_recorder=recorder,
                total_epoch=epochs,
                per_epoch=epochs,
                working_dir=str(fit_dir),
            )

        selected_epoch = _keep_one_model(fit_dir, epochs, self.model_path(fit_dir))
        self._fit_info_path(fit_dir).write_text(
            json.dumps(
                {
                    "epoch": epochs,
                    "selected_epoch": selected_epoch,
                    "chemical_species": species,
                    "shift": {
                        chemical_symbols[z]: float(e)
                        for z, e in isolated_atom_energies.items()
                    },
                    "scale": model_cfg["scale"],
                    "conv_denominator": model_cfg["conv_denominator"],
                },
                indent=2,
            )
            + "\n"
        )
        logger.info(
            "SevenNet trained %d epoch(s); kept epoch %s (per-epoch log: %s).",
            epochs,
            selected_epoch,
            fit_dir / _SEVENNET_LOG,
        )
        model_path = self.model_path(fit_dir)
        return model_path if model_path.exists() else None


def _checkpoints(fit_dir: Path) -> list[Path]:
    """SevenNet's checkpoint_*.pth files (best, periodic and epoch 0)."""
    return sorted(Path(fit_dir).glob("checkpoint_*.pth"))


def _keep_one_model(fit_dir: Path, epochs: int, model_path: Path) -> int | None:
    """Save the kept checkpoint (best validation epoch, else the last) as a
    slim *model_path* -- weights, config and epoch only, no optimizer or
    scheduler state, which is all SevenNetCalculator reads -- then delete
    every checkpoint_*.pth. Returns the kept epoch, or None if SevenNet
    wrote no checkpoint."""
    import torch

    model_path.unlink(missing_ok=True)
    best = fit_dir / "checkpoint_best.pth"
    kept = best if best.exists() else fit_dir / f"checkpoint_{epochs}.pth"
    selected = None
    if kept.exists():
        checkpoint = torch.load(kept, map_location="cpu", weights_only=False)
        selected = int(checkpoint["epoch"])
        torch.save(
            {
                "model_state_dict": checkpoint["model_state_dict"],
                "config": checkpoint["config"],
                "epoch": selected,
            },
            model_path,
        )
    for path in _checkpoints(fit_dir):
        path.unlink()
    return selected
