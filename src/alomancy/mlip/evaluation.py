"""Metrics for exported checkpoints, with explicit split and data identity."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from ase.stress import full_3x3_to_voigt_6_stress

from alomancy.utils.dataset_curation import geometry_digest, structure_domain

logger = logging.getLogger(__name__)


def _voigt_stress(stress: object) -> np.ndarray | None:
    """Stress as a Voigt 6-vector (ASE's get_stress() default), accepting a
    full 3x3 tensor too; None when missing or malformed."""
    if stress is None:
        return None
    arr = np.asarray(stress, dtype=float)
    if arr.shape == (3, 3):
        arr = full_3x3_to_voigt_6_stress(arr)
    if arr.shape != (6,) or not np.isfinite(arr).all():
        return None
    return arr


def prediction_metrics(atoms_list: list) -> dict:
    """Energy/force (and, where available, stress) errors for one split.

    Reported overall, per ``structure_domain`` (``"domains"``) and per
    ``config_type`` (``"config_types"``). Stress errors (eV/Angstrom^3) use
    only structures carrying both ``REF_stresses`` and ``model_stress``;
    stress is optional and never makes a split incomplete.
    """
    if not atoms_list:
        raise ValueError("Cannot evaluate an empty split")
    energy_errors, force_errors, identities = [], [], []
    stress_errors: list[np.ndarray | None] = []
    domains: dict[str, list[int]] = {}
    config_types: dict[str, list[int]] = {}
    for atoms in atoms_list:
        n = len(atoms)
        e = atoms.info.get("REF_energy")
        p = atoms.info.get("model_energy")
        f = atoms.arrays.get("REF_forces")
        pf = atoms.arrays.get("model_forces")
        if (
            not n
            or e is None
            or p is None
            or f is None
            or pf is None
            or f.shape != (n, 3)
            or pf.shape != (n, 3)
            or not np.isfinite([e, p]).all()
            or not np.isfinite(f).all()
            or not np.isfinite(pf).all()
        ):
            raise ValueError("Missing or invalid predictions/reference labels")
        energy_errors.append((float(p) - float(e)) / n)
        force_errors.append((pf - f).ravel())
        ref_stress = _voigt_stress(atoms.info.get("REF_stresses"))
        model_stress = _voigt_stress(atoms.info.get("model_stress"))
        stress_errors.append(
            model_stress - ref_stress
            if ref_stress is not None and model_stress is not None
            else None
        )
        identity = hashlib.sha256()
        identity.update(geometry_digest(atoms).encode())
        identity.update(np.asarray([e], dtype=np.float64).tobytes())
        identity.update(np.asarray(f, dtype=np.float64).tobytes())
        identities.append(identity.hexdigest())
        index = len(energy_errors) - 1
        domains.setdefault(structure_domain(atoms), []).append(index)
        config_types.setdefault(
            str(atoms.info.get("config_type", "unknown")), []
        ).append(index)

    def aggregate(indices: list[int]) -> dict:
        e = np.asarray([energy_errors[i] for i in indices])
        f = np.concatenate([force_errors[i] for i in indices])
        result = {
            "n_structures": len(indices),
            "mae_e_per_atom": float(np.abs(e).mean()),
            "rmse_e_per_atom": float(np.sqrt(np.mean(e**2))),
            "mae_f": float(np.abs(f).mean()),
            "rmse_f": float(np.sqrt(np.mean(f**2))),
        }
        stresses: list[np.ndarray] = [
            err for i in indices if (err := stress_errors[i]) is not None
        ]
        if stresses:
            st: np.ndarray = np.concatenate(stresses)
            result.update(
                {
                    "n_structures_with_stress": len(stresses),
                    "mae_stress": float(np.abs(st).mean()),
                    "rmse_stress": float(np.sqrt(np.mean(st**2))),
                }
            )
        return result

    return {
        **aggregate(list(range(len(atoms_list)))),
        "complete": True,
        "data_id": hashlib.sha256("\n".join(sorted(identities)).encode()).hexdigest(),
        "domains": {domain: aggregate(ids) for domain, ids in domains.items()},
        "config_types": {ct: aggregate(ids) for ct, ids in config_types.items()},
    }


def save_evaluation(fit_dir: Path, model_path: Path, splits: dict) -> None:
    record = {
        "schema_version": 2,
        "checkpoint": model_path.name,
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "units": {
            "energy": "eV/atom",
            "forces": "eV/Angstrom",
            "stress": "eV/Angstrom^3",
        },
        "splits": splits,
    }
    target = fit_dir / "evaluation_metrics.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def read_evaluation(fit_dir: Path, split: str) -> tuple[dict, Path]:
    record = json.loads((fit_dir / "evaluation_metrics.json").read_text())
    model = fit_dir / record["checkpoint"]
    if model.parent != fit_dir or not model.is_file():
        raise ValueError("Evaluation checkpoint is missing or invalid")
    if hashlib.sha256(model.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after evaluation; re-evaluate it")
    result = record["splits"][split]
    if not result.get("complete") or not result.get("n_structures"):
        raise ValueError(f"Incomplete evaluation on {split}")
    for metric in ("mae_f", "mae_e_per_atom", "rmse_f", "rmse_e_per_atom"):
        if not np.isfinite(result[metric]):
            raise ValueError(f"Non-finite {metric}")
    return result, model


def rank_committee(
    fit_dirs: dict[int, Path], metric: str = "mae_f", label: str = "committee"
) -> tuple[int, Path, str]:
    """Pick the fit with the lowest *metric* on the committee's common split.

    Uses the shared ``"valid"`` split when every fit has one, and falls back
    to ``"test"`` only when NO fit has ``"valid"`` (normal for a small
    eligible pool, where no validation split is carved). Fits disagreeing on
    having ``"valid"``, or missing ``"test"`` in the fallback, is a genuine
    evaluation failure and raises ``RuntimeError``. *label* names the
    committee in error messages.

    Returns ``(best_fit_idx, checkpoint_path, split_used)``.
    """

    def _try_read(fit_dir: Path, split: str) -> tuple[float, Path] | None:
        try:
            metrics, model_path = read_evaluation(fit_dir, split)
        except (FileNotFoundError, KeyError, ValueError):
            return None
        return float(metrics[metric]), model_path

    n_fits = len(fit_dirs)
    valid_scores: dict[int, tuple[float, Path]] = {}
    test_scores: dict[int, tuple[float, Path]] = {}
    for idx, fit_dir in fit_dirs.items():
        valid_result = _try_read(fit_dir, "valid")
        if valid_result is not None:
            valid_scores[idx] = valid_result
        test_result = _try_read(fit_dir, "test")
        if test_result is not None:
            test_scores[idx] = test_result

    if valid_scores:
        if len(valid_scores) < n_fits:
            raise RuntimeError(
                f"{label}: {n_fits - len(valid_scores)} of {n_fits} committee "
                "fit(s) are missing complete checkpoint validation on the "
                "'valid' split while others have it. Check remote job logs "
                "for evaluation failures."
            )
        scores, split_used = valid_scores, "valid"
    else:
        if len(test_scores) < n_fits or not test_scores:
            raise RuntimeError(
                f"{label}: {n_fits - len(test_scores)} of {n_fits} committee "
                "fit(s) are missing complete checkpoint validation (neither "
                "'valid' nor 'test' evaluation is available). Check remote "
                "job logs for evaluation failures."
            )
        scores, split_used = test_scores, "test"

    best_fit = min(scores, key=lambda i: scores[i][0])
    return best_fit, scores[best_fit][1], split_used


def best_fit_test_metrics(committee_dir: Path, metric: str = "mae_f") -> dict | None:
    """Test-split metrics of one AL loop's best committee member.

    The best member is chosen exactly as for MD (``rank_committee``: lowest
    *metric* on the common ``"valid"`` split, else ``"test"``), so the
    MAE-vs-loop plot tracks the model actually carried forward. Returns
    ``None`` when no fit in *committee_dir* has an evaluation; raises
    ``RuntimeError`` (from ``rank_committee``) or ``ValueError`` (best fit
    has no usable ``"test"`` split) when the evaluations are inconsistent.
    """
    fit_dirs = {
        int(p.parent.name.rsplit("_", 1)[1]): p.parent
        for p in committee_dir.glob("fit_*/evaluation_metrics.json")
    }
    if not fit_dirs:
        return None
    best_fit, _, split_used = rank_committee(fit_dirs, metric, label=str(committee_dir))
    record, _ = read_evaluation(fit_dirs[best_fit], "test")
    return {
        "mae_f": float(record["mae_f"]),
        "mae_e_per_atom": float(record["mae_e_per_atom"]),
        "best_fit_idx": best_fit,
        "selection_split": split_used,
    }


def loop_metrics_frame(loops: list[int], rows: list[dict]) -> pl.DataFrame:
    """One row per AL loop, with the loop number as the first column
    ("al_loop"); rows may carry different keys (missing ones are null)."""
    return pl.DataFrame(
        [{"al_loop": loop, **row} for loop, row in zip(loops, rows, strict=True)],
        infer_schema_length=None,
    )


def metrics_by_loop(
    name: str,
    *,
    strict: bool,
    expected_fits: int | None = None,
    results_dir: Path = Path("results"),
) -> pl.DataFrame:
    """One row per AL loop with its best model's test-split metrics
    (best_fit_test_metrics), the loop number in the "al_loop" column.

    Reads each loop's ``results/al_loop_N/<name>/fit_*/evaluation_metrics.json``;
    loops with none are left out. With ``strict=False`` (the live run's
    MAE-vs-loop plot) a loop whose evaluations are inconsistent is skipped
    with a warning; with ``strict=True`` (``alomancy results --replot``) it
    raises, and so does a loop where fewer than *expected_fits* fits have
    an evaluation.
    """
    rows = []
    loops = []
    loop_dirs = sorted(
        results_dir.glob("al_loop_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    )
    for loop_dir in loop_dirs:
        committee_dir = loop_dir / name
        try:
            if strict and expected_fits is not None:
                evaluated = {
                    p.parent.name
                    for p in committee_dir.glob("fit_*/evaluation_metrics.json")
                }
                if evaluated and evaluated != {
                    f"fit_{i}" for i in range(expected_fits)
                }:
                    raise RuntimeError(
                        f"{committee_dir}: evaluations found for {sorted(evaluated)}, "
                        f"expected fit_0..fit_{expected_fits - 1}."
                    )
            row = best_fit_test_metrics(committee_dir)
        except (RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
            if strict:
                raise
            logger.warning(
                "Skipping %s in the MAE-vs-loop metrics: %s", loop_dir.name, exc
            )
            continue
        if row is None:
            continue
        rows.append(row)
        loops.append(int(loop_dir.name.rsplit("_", 1)[1]))
    return loop_metrics_frame(loops, rows)


def check_quality_gate(workdir: Path, committee: dict) -> None:
    """Require all members to pass validation limits before exploration."""
    gate = committee["quality_gate"]
    required = gate.get("domains", {})
    if not required:
        raise ValueError("quality_gate requires per-domain validation limits")
    identities = set()
    for fit in range(committee["num_of_models_in_committee"]):
        metrics, _ = read_evaluation(
            workdir / committee["name"] / f"fit_{fit}", "valid"
        )
        identities.add(metrics["data_id"])
        for domain, limits in required.items():
            observed = metrics["domains"].get(domain)
            if observed is None:
                raise RuntimeError(f"Validation missing domain {domain} in fit_{fit}")
            for key, limit in limits.items():
                if key not in {"mae_f", "mae_e_per_atom", "rmse_f", "rmse_e_per_atom"}:
                    raise ValueError(f"Unknown quality metric {key}")
                if not np.isfinite(limit) or limit <= 0:
                    raise ValueError("Quality limits must be finite and positive")
                if not np.isfinite(observed[key]):
                    raise ValueError(f"Non-finite {domain} {key} in fit_{fit}")
                if observed[key] > limit:
                    raise RuntimeError(
                        f"fit_{fit} {domain} {key}={observed[key]:.6g} exceeds {limit}"
                    )
    if len(identities) != 1:
        raise RuntimeError("Committee members must share the same validation set")
