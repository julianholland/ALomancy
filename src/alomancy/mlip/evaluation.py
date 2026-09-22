"""Metrics for exported checkpoints, with explicit split and data identity."""

import hashlib
import json
from pathlib import Path

import numpy as np

from alomancy.utils.dataset_curation import geometry_digest, structure_domain


def prediction_metrics(atoms_list: list) -> dict:
    if not atoms_list:
        raise ValueError("Cannot evaluate an empty split")
    energy_errors, force_errors, identities = [], [], []
    domains: dict[str, list[int]] = {}
    for atoms in atoms_list:
        n = len(atoms)
        e = atoms.info.get("REF_energy")
        p = atoms.info.get("mace_energy")
        f = atoms.arrays.get("REF_forces")
        pf = atoms.arrays.get("mace_forces")
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
        identity = hashlib.sha256()
        identity.update(geometry_digest(atoms).encode())
        identity.update(np.asarray([e], dtype=np.float64).tobytes())
        identity.update(np.asarray(f, dtype=np.float64).tobytes())
        identities.append(identity.hexdigest())
        domains.setdefault(structure_domain(atoms), []).append(len(energy_errors) - 1)

    def aggregate(indices: list[int]) -> dict:
        e = np.asarray([energy_errors[i] for i in indices])
        f = np.concatenate([force_errors[i] for i in indices])
        return {
            "n_structures": len(indices),
            "mae_e_per_atom": float(np.abs(e).mean()),
            "rmse_e_per_atom": float(np.sqrt(np.mean(e**2))),
            "mae_f": float(np.abs(f).mean()),
            "rmse_f": float(np.sqrt(np.mean(f**2))),
        }

    return {
        **aggregate(list(range(len(atoms_list)))),
        "complete": True,
        "data_id": hashlib.sha256("\n".join(sorted(identities)).encode()).hexdigest(),
        "domains": {domain: aggregate(ids) for domain, ids in domains.items()},
    }


def save_evaluation(fit_dir: Path, model_path: Path, splits: dict) -> None:
    record = {
        "schema_version": 1,
        "checkpoint": model_path.name,
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "units": {"energy": "eV/atom", "forces": "eV/Angstrom"},
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


def check_quality_gate(workdir: Path, committee: dict) -> None:
    """Require all members to pass validation limits before exploration."""
    gate = committee["quality_gate"]
    required = gate.get("domains", {})
    if not required:
        raise ValueError("quality_gate requires per-domain validation limits")
    identities = set()
    for fit in range(committee["size_of_committee"]):
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
