"""ALomancyTrainer: the generic base of every MLIP trainer backend.

A backend (``mlip/mace/trainer.py``'s ``MaceTrainer``; next, SevenNet)
subclasses ``ALomancyTrainer``, declares its settings as class attributes
and implements three methods:

- ``fit(...)``: train one model, return the model file's path (or None);
- ``model_path(fit_dir)``: where ``fit`` puts that file;
- ``get_calculator(model_path)``: an ASE calculator for a trained model.

Everything else is shared and lives here: the standard ``train`` entry
point, resolving the per-element isolated-atom energies, evaluating the
model on every split (``{split}_pred.xyz`` + ``evaluation_metrics.json``),
restart checks and clean-up.

``train`` always takes the three split paths and the seed and returns the
model's path, or None if no model was produced. Metrics are not returned:
the workflow reads them back from ``evaluation_metrics.json``
(``read_existing_result``), checksum-verified, on every path.

ExPyRe pickles remote functions by reference, so the remote worker is the
module-level ``run_training``, never a method: it receives only plain data
(the trainer's registry name, its config dict, paths, the seed) and builds
the trainer on the remote node. Register a trainer with
``register("mlip_trainer", "<name>", "<module>", trainer_class="<Class>")``.
"""

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase.io import read, write

from alomancy.mlip.evaluation import (
    prediction_metrics,
    read_evaluation,
    save_evaluation,
)

logger = logging.getLogger(__name__)

EVALUATION_FILENAME = "evaluation_metrics.json"
_SPLITS = ("train", "valid", "test")


class ALomancyTrainer(ABC):
    """Generic MLIP trainer; see the module docstring."""

    #: Registry name (training.trainer).
    NAME: ClassVar[str] = ""
    #: training.<KWARGS_KEY> holds the backend's own settings.
    KWARGS_KEY: ClassVar[str] = ""
    #: Defaults for training.<KWARGS_KEY>; shown in the config summary.
    KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = {}
    #: The backend's name for the per-element isolated-atom energies (MACE
    #: "E0s", SevenNet "elemwise_reference_energies"); None if not needed.
    ISOLATED_ATOM_ENERGIES_KWARG: ClassVar[str | None] = None

    def __init__(self, config: dict, name: str = "training") -> None:
        """*config* is the workflow's ``training`` section; *name* is the
        fit directory's parent name, used by backends in file names."""
        self.config = config
        self.name = name
        self.kwargs: dict[str, Any] = {
            **self.KWARGS_DEFAULTS,
            **(config.get(self.KWARGS_KEY) or {}),
        }

    # -- What a backend implements ------------------------------------------

    @abstractmethod
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
        """Train one model into *fit_dir*; return the model file (or None).

        Paths are absolute. *isolated_atom_energies* is already resolved and
        formatted for the backend (see resolve_isolated_atom_energies), or
        None when nothing was given and the backend should fall back to its
        own behaviour."""

    @abstractmethod
    def model_path(self, fit_dir: Path) -> Path:
        """Where fit() writes the model that evaluation, MD and prediction use."""

    @abstractmethod
    def get_calculator(self, model_path: str | Path) -> Any:
        """An ASE calculator for a trained model. Only call it inside a
        remote job, never in the local driver process."""

    # -- Overridable, with generic defaults ----------------------------------

    def deployable_model_path(self, fit_dir: Path) -> Path | None:
        """The file copied to results/best_model/ (None: don't update it)."""
        path = self.model_path(fit_dir)
        return path if path.exists() else None

    def cleanup_paths(self, fit_dir: Path) -> list[Path]:  # noqa: ARG002
        """Files or directories to delete once a fit is finished."""
        return []

    def format_isolated_atom_energies(self, energies: dict) -> Any:
        """Convert a resolved {symbol or Z: eV} dict to the backend's format."""
        return energies

    def report_section(self, stats: dict, **kwargs: Any) -> Any:  # noqa: ARG002
        """This trainer's loop-report section (analysis/report); None = none."""
        return None

    def effective_kwargs(self) -> dict[str, Any]:
        """The fully resolved backend settings, for the config summary."""
        effective = dict(self.kwargs)
        key = self.ISOLATED_ATOM_ENERGIES_KWARG
        if key and key not in effective:
            effective[key] = "<resolved at train time from IsolatedAtom structures>"
        return effective

    # -- Shared behaviour ------------------------------------------------------

    def resolve_isolated_atom_energies(
        self,
        elements: list[str] | None,
        isolated_atom_energies: dict[str, float] | None,
    ) -> Any:
        """The per-element isolated-atom energies to train with.

        1. The backend setting (``<KWARGS_KEY>.<ISOLATED_ATOM_ENERGIES_KWARG>``,
           e.g. ``mace_kwargs.E0s``) wins. A string (e.g. "average", a .json
           path) passes through untouched.
        2. Otherwise the database's IsolatedAtom energies.
        3. Neither, while *elements* is given: ValueError, before training.
        A dict must cover every element in *elements* (by symbol or atomic
        number). Returns None when the backend doesn't use these energies,
        or when there is nothing to pass and no *elements* to check.
        """
        key = self.ISOLATED_ATOM_ENERGIES_KWARG
        if key is None:
            return None
        energies = self.kwargs.get(key)
        if energies is None:
            if isolated_atom_energies:
                energies = dict(isolated_atom_energies)
            elif elements:
                raise ValueError(
                    f"{self.KWARGS_KEY}.{key} is not set, and no IsolatedAtom "
                    "structures with REF_energy were found in the GlobalDatabase "
                    f"to default it from. Either set {key} explicitly in "
                    f"{self.KWARGS_KEY} (e.g. an {{element: energy}} dict), or "
                    "make sure IsolatedAtom structures for every element in "
                    f"general.elements ({elements}) have been DFT-evaluated first."
                )
            else:
                return None
        if not isinstance(energies, dict):
            return energies
        if elements:
            from ase.data import atomic_numbers

            missing = [
                el
                for el in elements
                if el not in energies and atomic_numbers.get(el) not in energies
            ]
            if missing:
                raise ValueError(
                    f"{self.KWARGS_KEY}.{key} is missing an entry for element(s) "
                    f"{missing} (from general.elements={elements}). These "
                    "reference energies can't be inferred by the model -- add "
                    f"them to {self.KWARGS_KEY}.{key}, or make sure an "
                    "IsolatedAtom structure for each element has been "
                    "DFT-evaluated into the GlobalDatabase."
                )
        return self.format_isolated_atom_energies(energies)

    def train(
        self,
        train_path: str | Path,
        valid_path: str | Path | None,
        test_path: str | Path,
        seed: int,
        *,
        fit_dir: str | Path,
        elements: list[str] | None = None,
        isolated_atom_energies: dict[str, float] | None = None,
    ) -> str | None:
        """Train, evaluate and tidy up one model. Returns the model's path,
        or None if no model was produced."""
        fit_dir = Path(fit_dir).resolve()
        fit_dir.mkdir(parents=True, exist_ok=True)
        train_file = Path(train_path).resolve()
        test_file = Path(test_path).resolve()
        valid_file = Path(valid_path).resolve() if valid_path else None
        for path in (train_file, test_file, valid_file):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}.")

        energies = self.resolve_isolated_atom_energies(elements, isolated_atom_energies)
        model = self.fit(
            train_file,
            valid_file,
            test_file,
            seed,
            fit_dir,
            isolated_atom_energies=energies,
        )
        if model is None or not Path(model).exists():
            logger.warning("No model produced in %s.", fit_dir)
            return None
        splits = {"train": train_file, "test": test_file}
        if valid_file is not None:
            splits["valid"] = valid_file
        self.evaluate(Path(model), splits, fit_dir)
        self.cleanup(fit_dir)
        return str(model)

    def evaluate(
        self, model_path: Path, splits: dict[str, Path], fit_dir: Path
    ) -> dict:
        """Predict every structure of every split with the trained model,
        write ``{split}_pred.xyz`` (``model_energy``, ``model_forces`` and,
        where DFT stress exists, ``model_stress``) and record the metrics in
        evaluation_metrics.json together with the model's checksum.

        Per-structure failures are counted, not fatal; the first one is
        logged with its traceback. Returns {split: metrics}.
        """
        try:
            calc = self.get_calculator(model_path)
        except Exception as exc:
            logger.warning("Could not load %s for evaluation: %s", model_path, exc)
            return {}

        split_results: dict[str, dict] = {}
        for tag, xyz_path in splits.items():
            try:
                atoms_list = list(read(xyz_path, ":", format="extxyz"))
            except Exception as exc:
                logger.debug("Could not read %s for evaluation: %s", xyz_path, exc)
                continue
            out = []
            n_ok = n_failed = n_no_stress = 0
            for atoms in atoms_list:
                a = atoms.copy()
                a.info.pop("model_energy", None)
                a.info.pop("model_stress", None)
                a.arrays.pop("model_forces", None)
                a.calc = calc
                try:
                    a.info["model_energy"] = float(a.get_potential_energy())
                    a.arrays["model_forces"] = a.get_forces()
                    n_ok += 1
                    # Only where DFT stress exists, and never fatal (e.g.
                    # non-periodic cells have none).
                    if "REF_stresses" in a.info:
                        try:
                            a.info["model_stress"] = a.get_stress()
                        except Exception:
                            n_no_stress += 1
                except Exception as exc:
                    n_failed += 1
                    if n_failed == 1:
                        logger.warning(
                            "Prediction failed for structure %d/%d (config_type=%s): %s",
                            n_ok + n_failed,
                            len(atoms_list),
                            atoms.info.get("config_type"),
                            exc,
                            exc_info=True,
                        )
                    else:
                        # The same failure on every structure would flood the
                        # log; the summary line below still gives the count.
                        logger.debug(
                            "Prediction failed for structure %d/%d: %s",
                            n_ok + n_failed,
                            len(atoms_list),
                            exc,
                        )
                finally:
                    # Keep only the explicit model_* fields: ASE would also
                    # try to write the calculator's results, some of which
                    # aren't per-atom and break extxyz writing.
                    a.calc = None
                out.append(a)
            if n_no_stress:
                logger.info(
                    "%s predictions: model stress unavailable for %d structure(s).",
                    tag,
                    n_no_stress,
                )
            if n_failed:
                logger.warning(
                    "%s predictions: %d succeeded, %d failed out of %d structures.",
                    tag,
                    n_ok,
                    n_failed,
                    len(atoms_list),
                )
            try:
                write(fit_dir / f"{tag}_pred.xyz", out, format="extxyz")
            except Exception as exc:
                logger.warning("Failed to write %s_pred.xyz: %s", tag, exc)
            try:
                split_results[tag] = prediction_metrics(out)
            except ValueError as exc:
                split_results[tag] = {
                    "complete": False,
                    "reason": str(exc),
                    "n_structures": len(out),
                }
        save_evaluation(fit_dir, model_path, split_results)
        return split_results

    def cleanup(self, fit_dir: Path) -> None:
        """Delete cleanup_paths(fit_dir); failures are logged, never raised."""
        for path in self.cleanup_paths(fit_dir):
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logger.info("Removed %s after a finished fit.", path)
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", path, exc)

    def output_paths(self, fit_dir: Path) -> list[Path]:
        """Files that exist once a fit has genuinely finished: the model and
        its evaluation. Used by the workflow's restart checks."""
        return [self.model_path(fit_dir), fit_dir / EVALUATION_FILENAME]

    def read_existing_result(self, fit_dir: Path) -> tuple[str, str | None, dict]:
        """(model_path, deployable_model_path or None, {split: metrics}) for
        a finished fit, every split checksum-verified against the model
        (read_evaluation). Raises ValueError if no split verifies."""
        metrics: dict = {}
        for split in _SPLITS:
            try:
                metrics[split], _ = read_evaluation(fit_dir, split)
            except (FileNotFoundError, KeyError, ValueError):
                continue
        if not metrics:
            raise ValueError(
                f"No valid checkpoint-verified evaluation found for {fit_dir} -- "
                "cannot reconstruct a cached result."
            )
        deployable = self.deployable_model_path(fit_dir)
        return (
            str(self.model_path(fit_dir)),
            str(deployable) if deployable is not None else None,
            metrics,
        )


def get_trainer(name: str, config: dict, fit_name: str = "training") -> ALomancyTrainer:
    """The trainer registered as *name* (training.trainer), built for *config*."""
    from alomancy.registry import resolve

    trainer_class: type[ALomancyTrainer] = resolve("mlip_trainer", name).trainer_class
    return trainer_class(config, name=fit_name)


def run_training(
    trainer: str,
    config: dict,
    train_atoms_path: str,
    valid_atoms_path: str | None,
    test_atoms_path: str,
    seed: int,
    *,
    fit_dir: str,
    fit_name: str = "training",
    elements: list[str] | None = None,
    isolated_atom_energies: dict[str, float] | None = None,
) -> str | None:
    """The remote worker the workflow submits once per fit: builds the
    trainer by registry name on the remote node and runs its train()."""
    return get_trainer(trainer, config, fit_name).train(
        train_atoms_path,
        valid_atoms_path,
        test_atoms_path,
        seed,
        fit_dir=fit_dir,
        elements=elements,
        isolated_atom_energies=isolated_atom_energies,
    )


def read_predictions(fit_dir: Path) -> dict[int, dict]:
    """Per-structure predictions from a fit's train_pred.xyz/test_pred.xyz,
    as {global_db_id: {"energy": float, "forces": list}}; {} if none."""
    preds: dict[int, dict] = {}
    for tag in ("train", "test"):
        xyz = fit_dir / f"{tag}_pred.xyz"
        if not xyz.exists():
            continue
        try:
            atoms_list = list(read(xyz, ":", format="extxyz"))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", xyz, exc)
            continue
        for atoms in atoms_list:
            gid = atoms.info.get("global_db_id")
            if gid is None or "model_energy" not in atoms.info:
                continue
            forces = atoms.arrays.get("model_forces")
            preds[int(gid)] = {
                "energy": float(atoms.info["model_energy"]),
                "forces": np.asarray(forces).tolist() if forces is not None else [],
            }
    return preds
