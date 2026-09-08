"""Prepare an extra dataset using a Pd FCC-derived reference and domain filters.

The minimum energy of a distorted FCC-derived bulk is an ESTIMATE of the
bulk chemical potential, not a converged equilibrium EOS calculation. It is
sufficient to define a reproducible filtering coordinate; REF_energy and
MACE isolated-atom E0s are unchanged. The reference status records this limit.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write

from alomancy.utils.dataset_curation import (
    annotate_structure,
    geometry_digest,
    grouped_split,
    validate_policy,
)


def is_fcc_derived_bulk(atoms: Atoms, max_rms_displacement: float = 0.25) -> bool:
    """Recognize equal-sided primitive-FCC supercells with a small rattle.

    Only explicit bulk_rattle configurations are candidates. HCP init_MP is
    never assumed FCC based on its name. Other cell representations require
    an explicit external reference instead of a guessed classification.
    """
    if atoms.info.get("config_type") != "bulk_rattle" or set(
        atoms.get_chemical_symbols()
    ) != {"Pd"}:
        return False
    n = round(len(atoms) ** (1 / 3))
    lengths, angles = atoms.cell.cellpar()[:3], atoms.cell.cellpar()[3:]
    if (
        n**3 != len(atoms)
        or not np.allclose(lengths, lengths[0], rtol=1e-5)
        or not np.allclose(angles, 60, atol=1e-3)
    ):
        return False
    scaled = atoms.get_scaled_positions()
    sites = np.round(scaled * n).astype(int)
    if len({tuple(site % n) for site in sites}) != len(atoms):
        return False
    displacement = (scaled - sites / n) @ atoms.cell.array
    return (
        float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))) <= max_rms_displacement
    )


def estimate_pd_reference(atoms_list: list, max_force: float = 3.0) -> dict:
    """Low-force, low-energy FCC-derived bulk reference from the extra dataset.

    Preselection uses E/N (equivalently formation energy relative to a fixed
    isolated Pd atom). Final formation energies use the resulting bulk mu.
    """
    candidates = []
    isolated = [
        a
        for a in atoms_list
        if len(a) == 1
        and a.info.get("config_type") == "IsolatedAtom"
        and a.get_chemical_symbols() == ["Pd"]
    ]
    if not isolated:
        raise ValueError("Pd isolated-atom reference is missing")
    e0 = float(isolated[0].info["REF_energy"])
    if not np.isfinite(e0):
        raise ValueError("Non-finite isolated-atom reference")
    for i, a in enumerate(atoms_list):
        if not is_fcc_derived_bulk(a):
            continue
        e, f = a.info.get("REF_energy"), a.arrays.get("REF_forces")
        if (
            e is None
            or f is None
            or not np.isfinite(e)
            or not np.isfinite(f).all()
            or f.shape != (len(a), 3)
            or np.linalg.norm(f, axis=1).max() >= max_force
        ):
            continue
        candidates.append((float(e) / len(a) - e0, i, a))
    if not candidates:
        raise ValueError(
            "No suitable FCC-derived reference: supply a separately computed FCC reference"
        )
    formation, index, selected = min(candidates, key=lambda item: (item[0], item[1]))
    mu = formation + e0
    return {
        "id": "Pd_FCC_dataset_" + geometry_digest(selected)[:16],
        "status": "estimated",
        "chemical_potentials": {"Pd": mu},
        "method": "minimum_E_per_atom_of_low_force_FCC_derived_bulk",
        "source_index": int(selected.info.get("source_index", index)),
        "source_geometry_sha256": geometry_digest(selected),
        "candidate_count": len(candidates),
        "preselection_max_force": max_force,
        "preselection_isolated_atom_energy": e0,
        "equilibrium_verified": False,
        "note": "Estimate from distorted bulk data; refine with an FCC equation of state using the run protocol.",
    }


def prepare_dataset(source: Path, config: Path, output: Path) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "Prepared dataset already exists; use a new output directory"
        )
    data = cast(list[Atoms], read(source, ":", format="extxyz"))
    settings = yaml.safe_load(config.read_text())
    policy = settings["dataset_curation"]
    if not policy.get("reference"):
        policy["reference"] = estimate_pd_reference(
            data, settings.get("reference_selection", {}).get("max_force", 3.0)
        )
    validate_policy(policy)
    archive: list[Atoms] = []
    selected: list[Atoms] = []
    rejected: list[Atoms] = []
    rows = []
    for index, a in enumerate(data):
        meta = annotate_structure(a, policy)
        a.info.update(meta)
        archive.append(a)
        (selected if meta["is_training_eligible"] else rejected).append(a)
        rows.append(
            {
                "index": index,
                "source_index": a.info.get("source_index", index),
                "domain": meta["domain"],
                "natoms": len(a),
                "energy": a.info["REF_energy"],
                "formation_energy": meta.get("REF_formation_energy"),
                "formation_energy_per_atom": meta.get("REF_formation_energy_per_atom"),
                "max_force": meta.get("REF_max_force"),
                "accepted": meta["is_training_eligible"],
                "reasons": ";".join(meta["filter_reasons"]),
            }
        )
    if not selected:
        raise ValueError("All structures rejected")
    train, test = grouped_split(
        selected, settings["initialization"]["test_to_train_ratio"], 803
    )
    if not test:
        raise ValueError("No independent test groups remain after filtering")
    output.mkdir(parents=True, exist_ok=True)
    for name, structures in (
        ("annotated_archive", archive),
        ("selected", selected),
        ("rejected", rejected),
        ("train", train),
        ("test", test),
    ):
        if structures:
            write(output / f"{name}.xyz", structures, format="extxyz")
    with (output / "filter_report.csv").open("w") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "reference": policy["reference"],
        "policy": policy,
        "total": len(data),
        "selected": len(selected),
        "rejected": len(rejected),
        "train": len(train),
        "test": len(test),
        "domains_before": dict(Counter(a.info["domain"] for a in data)),
        "domains_selected": dict(Counter(a.info["domain"] for a in selected)),
    }
    (output / "reference.json").write_text(
        json.dumps(policy["reference"], indent=2) + "\n"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # A run-local resolved config freezes the actual reference used for filtering.
    settings["initialization"]["extra_datasets"] = [
        str((output / "selected.xyz").resolve())
    ]
    (output / "resolved_init.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False)
    )
    manifest["output_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare_dataset(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
