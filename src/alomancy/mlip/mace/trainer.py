"""MLIP trainer module: the MACE backend for the modular AL architecture.

Implements the trainer entry points the shared registry dispatches to
(``train``, ``get_calculator``, ``output_paths``, ``read_existing_result``).
Unlike ``mlip.mace.mace_wfl.mace_fit`` (which this supersedes once a
skeleton is wired up to call it), a trainer module has zero committee
awareness: it trains exactly one model from an already-built, explicit
train/valid/test split and returns everything downstream needs about that
one fit. All committee-ness -- looping N times, aggregating metrics,
quality-gating, selecting the best member, storing predictions in the
GlobalDatabase, local checkpoint cleanup -- lives in the skeleton, not here.

``train`` is the literal remote-executed worker: the skeleton drives the
N-times committee loop itself, calling ``submit_n(train, [...])`` directly
rather than this module doing any of its own remote submission.
``base_name``/``name``/``hpc``/``max_time`` are explicit keyword arguments
shared across every module category's entry points (uniform interface),
even though ``hpc``/``max_time`` themselves are unused inside a remote
worker -- see the architecture plan's discussion of the trainer vs.
generator/evaluator submission shapes.
"""

import importlib.util
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write
from mace import tools
from mace.calculators import MACECalculator
from mace.cli.run_train import run

from alomancy.mlip.evaluation import (
    prediction_metrics,
    read_evaluation,
    save_evaluation,
)

logger = logging.getLogger(__name__)

_DYNAMIC_EPOCHS_TARGET_SAMPLES = 200_000
_DYNAMIC_EPOCHS_CAP = 300
_DYNAMIC_EPOCHS_FLOOR = 20

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


def _fit_dir(base_name: str, name: str, fit_idx: int) -> Path:
    return Path("results", base_name, name, f"fit_{fit_idx}")


def _compute_dynamic_epochs(batch_size: int, n_training_structures: int) -> int:
    """epochs = ceil(200_000 * batch_size / n_training_structures), clamped to
    [20, 300].

    The cap prevents an absurd epoch count for a small early-loop training
    set; the floor prevents a data-rich late-loop training set from being
    pushed down to a near-zero stage-two (SWA) phase --
    start_swa = floor(0.8 * epochs), so a floor of 20 guarantees at least 16
    SWA epochs.
    """
    if n_training_structures <= 0:
        raise ValueError(
            f"n_training_structures must be positive, got {n_training_structures}."
        )
    raw = math.ceil(_DYNAMIC_EPOCHS_TARGET_SAMPLES * batch_size / n_training_structures)
    epochs = max(_DYNAMIC_EPOCHS_FLOOR, min(_DYNAMIC_EPOCHS_CAP, raw))
    if epochs != raw:
        logger.warning(
            "Dynamic epoch formula produced %d epochs (batch_size=%d, "
            "n_training_structures=%d); clamped to %d.",
            raw,
            batch_size,
            n_training_structures,
            epochs,
        )
    return epochs


def _write_resolved_mace_epochs(mlip_dir: Path, mace_fit_params: dict) -> None:
    """Persist the actually-resolved max_num_epochs/start_swa to
    resolved_mace_epochs.json in the fit directory -- see mace_wfl.py's
    version of this function for why (mlip_plots.py reads it back)."""
    payload = {
        "max_num_epochs": mace_fit_params["max_num_epochs"],
        "start_swa": mace_fit_params["start_swa"],
    }
    with open(mlip_dir / "resolved_mace_epochs.json", "w") as fh:
        json.dump(payload, fh)


def _apply_compute_stress_defaults(mace_fit_params: dict, compute_stress: bool) -> None:
    """Mutate mace_fit_params in place to enable stress training, if requested.

    setdefault (not direct assignment) so a user who already set
    loss/stress_key explicitly (e.g. "huber", "universal") keeps their own
    choice. A bare stress_key alone has no training effect on its own --
    MACE only trains on stress when loss is "stress"/"huber"/"universal".
    """
    if compute_stress:
        mace_fit_params.setdefault("stress_key", "REF_stresses")
        mace_fit_params.setdefault("loss", "stress")


def _remove_checkpoints_dir_if_model_exists(
    model_path: Path, checkpoints_dir: Path, location: str = ""
) -> None:
    """Delete checkpoints_dir once model_path exists; leave it alone (with a
    warning) if the model is missing, so a checkpoint is never deleted out
    from under a fit that hasn't actually finished training."""
    if not model_path.exists():
        logger.warning(
            "Stagetwo compiled model not found; leaving %s%s in place.",
            location,
            checkpoints_dir,
        )
        return
    if not checkpoints_dir.exists():
        return
    try:
        shutil.rmtree(checkpoints_dir)
        logger.info("Removed %s%s after successful fit.", location, checkpoints_dir)
    except OSError as exc:
        logger.warning("Failed to remove %s%s: %s", location, checkpoints_dir, exc)


def _evaluate_and_save_predictions(
    model_path: Path, split_paths: dict[str, Path]
) -> dict:
    """Evaluate the trained stagetwo model on every split in split_paths;
    write {tag}_pred.xyz for each and record the whole set via
    save_evaluation. Returns the splits dict (same shape save_evaluation
    persists to evaluation_metrics.json), which becomes train()'s own
    metrics_dict return value.

    Called while os.chdir'd into the fit directory. Deliberately evaluates
    the UNCOMPILED model (plain torch.save'd nn.Module), not the
    TorchScript-compiled one used for MD/DFT-adjacent inference -- see
    mace_wfl.py's version of this function for the production incident
    (silent force-evaluation failure under TorchScript) this avoids.
    """
    if not model_path.exists():
        logger.warning(
            "Stagetwo (uncompiled) model not found; skipping eval predictions."
        )
        return {}

    logger.info("Using %s for post-training eval predictions.", model_path.name)

    try:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        calc = MACECalculator(
            model_paths=[str(model_path.resolve())],
            device=device,
            default_dtype="float64",
        )
    except Exception as exc:
        logger.warning("Failed to load MACECalculator for post-training eval: %s", exc)
        return {}

    split_results: dict = {}
    for tag, xyz_path in split_paths.items():
        try:
            atoms_list = list(read(xyz_path, ":", format="extxyz"))
        except Exception as exc:
            logger.debug("Could not read %s for eval predictions: %s", xyz_path, exc)
            continue

        out = []
        n_ok = 0
        n_failed = 0
        for atoms in atoms_list:
            a = atoms.copy()
            a.info.pop("model_energy", None)
            a.arrays.pop("model_forces", None)
            a.calc = calc
            try:
                a.info["model_energy"] = float(a.get_potential_energy())
                a.arrays["model_forces"] = a.get_forces()
                n_ok += 1
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
                    logger.debug(
                        "Prediction failed for structure %d/%d: %s",
                        n_ok + n_failed,
                        len(atoms_list),
                        exc,
                    )
            finally:
                a.calc = None
            out.append(a)

        if n_failed:
            logger.warning(
                "%s predictions: %d succeeded, %d failed out of %d structures.",
                tag,
                n_ok,
                n_failed,
                len(atoms_list),
            )

        try:
            write(f"{tag}_pred.xyz", out, format="extxyz")
            logger.info("Saved %d %s prediction(s) to %s_pred.xyz.", len(out), tag, tag)
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

    save_evaluation(Path.cwd(), model_path, split_results)
    return split_results


def output_paths(
    config: dict,
    *,
    base_name: str,
    name: str,
    fit_idx: int,  # noqa: ARG001
) -> list[Path]:
    """Files that exist once this fit has genuinely completed -- used by the
    skeleton's restart mechanism. The uncompiled model (what every current
    consumer -- eval predictions, MD, EZGA -- actually uses) plus the
    checkpoint-verified evaluation_metrics.json that read_existing_result
    actually trusts. The compiled model is deliberately NOT included here:
    MACE silently swallows a failed compile (bare except: pass), so gating
    restart on its existence would make a fit whose training genuinely
    succeeded look incomplete forever.
    """
    fit_dir = _fit_dir(base_name, name, fit_idx)
    return [
        fit_dir / f"{name}_stagetwo.model",
        fit_dir / "evaluation_metrics.json",
    ]


def read_existing_result(
    config: dict,
    *,
    base_name: str,
    name: str,
    fit_idx: int,  # noqa: ARG001
) -> tuple[str, str | None, dict]:
    """Reconstruct this fit's (model_path, compiled_model_path, metrics_dict)
    from real, already-on-disk output for the skeleton's restart mechanism.

    Goes through read_evaluation's checkpoint-SHA-256 integrity check for
    every split it can find, rather than trusting a cached blob -- a stale
    or corrupted on-disk state raises here instead of silently being
    treated as a valid cached result. Raises ValueError if no split
    evaluates successfully at all.
    """
    fit_dir = _fit_dir(base_name, name, fit_idx)
    model_path = fit_dir / f"{name}_stagetwo.model"
    compiled_model_path = fit_dir / f"{name}_stagetwo_compiled.model"

    metrics: dict = {}
    for split in ("train", "valid", "test"):
        try:
            metrics[split], _ = read_evaluation(fit_dir, split)
        except (FileNotFoundError, KeyError, ValueError):
            continue
    if not metrics:
        raise ValueError(
            f"No valid checkpoint-verified evaluation found for {fit_dir} -- "
            "cannot reconstruct a cached result."
        )
    return (
        str(model_path),
        str(compiled_model_path) if compiled_model_path.exists() else None,
        metrics,
    )


def get_calculator(model_path: str, config: dict) -> Any:
    """Build a calculator from a trained model path.

    Must only be called from inside a remote worker (e.g. MD's own per-seed
    job, or the skeleton's committee-scoring job) -- never the local driver
    process, since a live calculator must never cross the ExPyRe boundary.
    """
    import torch

    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return MACECalculator(
        model_paths=[str(model_path)],
        device=device,
        default_dtype=config.get("default_dtype", "float64"),
    )


def train(
    train_atoms_path: str,
    valid_atoms_path: str | None,
    test_atoms_path: str,
    config: dict,
    fit_seed: int,
    *,
    base_name: str,
    name: str,
    fit_idx: int,
    hpc: dict,  # noqa: ARG001 -- unused here; uniform across module categories
    max_time: str,  # noqa: ARG001 -- unused here; uniform across module categories
    elements: list[str] | None = None,
    isolated_atom_e0s: dict[str, float] | None = None,
) -> tuple[str, str | None, dict]:
    """Train one MACE model. The remote-executed worker function the
    skeleton drives its N-times committee loop with (one call = one fit).

    Splits are passed as file paths already built by the skeleton -- this
    function has no awareness of how the split was constructed, and never
    re-derives one from a seed. `valid_atoms_path` is None when the
    skeleton's eligible pool for validation was too small to carve one out
    (mirrors mace_fit's `_select_validation_split` legitimately skipping
    the carve-out) -- not an error, just no `valid_file` passed to MACE.

    `elements` (workflow.elements, the shared element list) is used for a
    safety-net check (see below). `isolated_atom_e0s` ({symbol: REF_energy},
    from GlobalDatabase.get_isolated_atom_energies() -- computed locally by
    the skeleton and passed down as a plain dict, never a live DB, which
    must not cross the ExPyRe boundary) is the fallback E0s source when
    `mace_kwargs.E0s` isn't set explicitly: MACE cannot infer this
    physical reference value on its own, but isolated-atom DFT energies
    already in the database are exactly what it needs. Safety net: if E0s
    is neither given explicitly nor available from the database for every
    element in `elements`, this raises rather than letting MACE either fail
    obscurely or silently train against a wrong/missing reference.
    """
    mace_kwargs = dict(config.get("mace_kwargs", {}))
    if "seed" in mace_kwargs:
        raise ValueError(
            "mace_kwargs must not set 'seed' -- it is derived and passed "
            "separately (fit_seed)."
        )
    # Default to this codebase's own standard REF_energy/REF_forces keys
    # (see CLAUDE.md's "Never use bare 'energy' as an info key" convention)
    # rather than requiring every config to repeat them explicitly.
    mace_kwargs.setdefault("energy_key", "REF_energy")
    mace_kwargs.setdefault("forces_key", "REF_forces")

    if "E0s" not in mace_kwargs:
        if isolated_atom_e0s:
            mace_kwargs["E0s"] = isolated_atom_e0s
        elif elements:
            # Safety net: no E0s given and no isolated-atom reference
            # energies available either -- MACE cannot proceed without one
            # or the other (it has no way to infer this physical value).
            raise ValueError(
                "mace_kwargs.E0s is not set, and no IsolatedAtom "
                "structures with REF_energy were found in the "
                "GlobalDatabase to default it from. Either set E0s "
                "explicitly in mace_kwargs (e.g. 'average', or a "
                "{element: energy} dict), or make sure IsolatedAtom "
                "structures for every element in workflow.elements "
                f"({elements}) have been DFT-evaluated first."
            )

    e0s = mace_kwargs.get("E0s")
    if elements and isinstance(e0s, dict):
        # Coverage check only applies to an explicit/defaulted {element:
        # energy} dict -- a string value like "average" (MACE's own
        # built-in linear-regression E0 estimate) has nothing to check here.
        from ase.data import atomic_numbers

        missing = [
            el for el in elements if el not in e0s and atomic_numbers.get(el) not in e0s
        ]
        if missing:
            raise ValueError(
                f"mace_kwargs.E0s is missing an entry for element(s) "
                f"{missing} (from workflow.elements={elements}). E0s is a "
                "physical reference value MACE cannot infer on its own -- "
                "either add it explicitly to mace_kwargs.E0s, or make "
                "sure an IsolatedAtom structure for that element has been "
                "DFT-evaluated into the GlobalDatabase."
            )

    fit_dir = _fit_dir(base_name, name, fit_idx)
    logger.info("Creating MLIP directory: %s", fit_dir)
    fit_dir.mkdir(exist_ok=True, parents=True)

    # Resolve to absolute paths before chdir'ing into fit_dir below --
    # unlike mace_fit's fragile "../../train_set.xyz"-style relative paths,
    # this works regardless of where the skeleton chose to write the
    # shared split files.
    train_path = Path(train_atoms_path).resolve()
    test_path = Path(test_atoms_path).resolve()
    valid_path = Path(valid_atoms_path).resolve() if valid_atoms_path else None

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}.")
    all_training = list(read(train_path, ":", format="extxyz"))
    n_valid = 0
    if valid_path is not None:
        if not valid_path.exists():
            raise FileNotFoundError(f"Validation file not found: {valid_path}.")
        n_valid = len(list(read(valid_path, ":", format="extxyz")))
    logger.info(
        "Read %d training structure(s), %d validation structure(s).",
        len(all_training),
        n_valid,
    )

    batch_size = mace_kwargs.get("batch_size", 16)
    # Popped (not just read) so these control values never leak through the
    # **mace_kwargs spread below into mace_fit_params -- max_num_epochs is
    # resolved to a real epoch count first, and compute_stress is consumed
    # by _apply_compute_stress_defaults, not passed to MACE directly.
    configured_epochs = mace_kwargs.pop("max_num_epochs", None)
    compute_stress = mace_kwargs.pop("compute_stress", False)
    if configured_epochs is None:
        epochs = 80
    elif configured_epochs == "dynamic":
        # Uses the full pre-split pool passed to this loop's training
        # (train + valid), matching mace_fit's historical use of
        # len(all_training) *before* its own per-fit carve-out -- using
        # only the post-split train count would silently shift the
        # resolved epoch count by ~1/(1 - valid_fraction).
        epochs = _compute_dynamic_epochs(batch_size, len(all_training) + n_valid)
        logger.info(
            "Dynamic max_num_epochs resolved to %d (batch_size=%d, "
            "n_training_structures=%d).",
            epochs,
            batch_size,
            len(all_training) + n_valid,
        )
    else:
        epochs = configured_epochs

    mace_fit_params = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "model": "MACE",
        "correlation": 3,
        "device": "cuda",
        "ema": None,
        "energy_weight": 1,
        "forces_weight": 10,
        "error_table": "PerAtomMAE",
        "eval_interval": 1,
        "max_L": 2,
        "max_num_epochs": epochs,
        "name": name,
        "num_channels": 128,
        "num_interactions": 2,
        "patience": 30,
        "r_max": 5.0,
        "restart_latest": None,
        "save_cpu": None,
        "scheduler_patience": 15,
        "start_swa": int(np.floor(epochs * 0.8)),
        "swa": None,
        "batch_size": batch_size,
        "valid_batch_size": 16,
        "distributed": None,
        "seed": fit_seed,
        **mace_kwargs,
    }
    if valid_path is not None:
        mace_fit_params["valid_file"] = str(valid_path)

    _apply_compute_stress_defaults(mace_fit_params, compute_stress)
    _write_resolved_mace_epochs(fit_dir, mace_fit_params)

    logger.debug("MACE fit parameters:")
    for key, value in mace_fit_params.items():
        logger.debug("  %s: %s", key, value)

    parser = tools.build_default_arg_parser()
    args = parser.parse_args(["--name", mace_fit_params["name"]])
    for key, value in mace_fit_params.items():
        setattr(args, key, value)

    orig_dir = os.getcwd()
    try:
        os.chdir(fit_dir)
        run(args)

        model_path = Path(f"{name}_stagetwo.model")
        compiled_model_path = Path(f"{name}_stagetwo_compiled.model")

        split_paths = {"train": train_path, "test": test_path}
        if valid_path is not None:
            split_paths["valid"] = valid_path
        metrics = _evaluate_and_save_predictions(model_path, split_paths)

        _remove_checkpoints_dir_if_model_exists(
            compiled_model_path, Path("checkpoints")
        )
    finally:
        os.chdir(orig_dir)

    # Check existence via the fit_dir-qualified path, not the bare relative
    # one used above while chdir'd in -- orig_dir has been restored by now.
    compiled_model_full_path = fit_dir / compiled_model_path
    return (
        str(fit_dir / model_path),
        str(compiled_model_full_path) if compiled_model_full_path.exists() else None,
        metrics,
    )
