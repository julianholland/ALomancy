"""Recover archived labels from matching, converged VASP outputs.

Usage: python -m alomancy.utils.recover_dft_labels SOURCE RAW_ROOT OUTPUT_DIR
The input dataset and DFT output directories are read only. OUTPUT_DIR must
not contain previous recovery artifacts. Unresolved records are quarantined.
"""

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms
from ase.io import read, write

from alomancy.utils.dataset_curation import geometry_digest, structure_domain
from alomancy.utils.dft_utils import refresh_dft_labels

logger = logging.getLogger(__name__)


def vasp_converged(outcar: Path) -> bool:
    """Require completed output and explicit electronic convergence.

    Conservative for historical files without VASP's termination marker:
    unconfirmed results are quarantined, never silently declared converged.
    """
    if not outcar.is_file():
        return False
    text = outcar.read_text(errors="replace")
    exits = re.findall(r"[^\n]*aborting loop[^\n]*", text)
    return bool(
        exits
        and "because EDIFF is reached" in exits[-1]
        and "General timing and accounting" in text
    )


def recover_dataset(source: Path, raw_root: Path, output_dir: Path) -> dict:
    targets = [
        output_dir / name
        for name in ("recovered.xyz", "quarantine.xyz", "recovery.csv", "summary.json")
    ]
    if any(p.exists() for p in targets):
        raise FileExistsError("Recovery output already exists; use a new directory")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    atoms_list = cast(list[Atoms], read(source, ":", format="extxyz"))
    seed_domains = {
        a.info.get("global_db_id"): structure_domain(a)
        for a in atoms_list
        if a.info.get("config_type") != "high_sd"
    }
    for seed_file in (raw_root / "initialization").glob("*_set.xyz"):
        for seed in read(seed_file, ":", format="extxyz"):
            if seed.info.get("config_type") != "high_sd":
                seed_domains[seed.info.get("global_db_id")] = structure_domain(seed)
    needed = {
        geometry_digest(a) for a in atoms_list if a.info.get("config_type") == "high_sd"
    }
    matches: dict[str, list[tuple[Path, Atoms]]] = defaultdict(list)
    parent_by_geometry = {}
    for path in sorted(
        raw_root.glob(
            "al_loop_*/high_accuracy_evaluation/**/high_accuracy_evaluation.xyz"
        )
    ):
        atoms = cast(Atoms, read(path, format="extxyz"))
        key = geometry_digest(atoms)
        parent_by_geometry[key] = atoms.info.get("global_db_id")
        if key in needed:
            matches[key].append((path, atoms))
    # Trace seeds introduced by earlier AL cycles back to their original seed.
    ancestry = {}
    for seed_file in sorted(raw_root.glob("al_loop_*/*_set.xyz")):
        for seed in cast(list[Atoms], read(seed_file, ":", format="extxyz")):
            seed_id = seed.info.get("global_db_id")
            parent_id = parent_by_geometry.get(geometry_digest(seed))
            if seed_id is not None and parent_id is not None and seed_id != parent_id:
                ancestry[seed_id] = parent_id
            domain = structure_domain(seed)
            if domain != "unknown":
                seed_domains[seed_id] = domain

    def root_seed(seed_id: int) -> int:
        seen = set()
        while seed_id in ancestry and seed_id not in seen:
            seen.add(seed_id)
            seed_id = ancestry[seed_id]
        return seed_id

    accepted, quarantine, rows = [], [], []
    corrected = 0
    for i, original in enumerate(atoms_list):
        atoms = original.copy()
        # Remove operational state inherited from previous active-learning runs.
        for key in list(atoms.info):
            if key.startswith("mace_") or key in {
                "global_db_id",
                "split",
                "is_duplicate",
                "is_high_force",
                "is_training_eligible",
                "filter_reasons",
            }:
                atoms.info.pop(key)
        atoms.info["source_index"] = i
        atoms.info["source_dataset"] = str(source.resolve())
        atoms.info["source_dataset_sha256"] = source_digest
        atoms.info["domain"] = structure_domain(atoms)
        reason = "original_initialization_labels"
        raw_source: Path | None = None
        old_energy = float(original.info["REF_energy"])
        if original.info.get("config_type") == "high_sd":
            candidates = [
                (p, a)
                for p, a in matches[geometry_digest(original)]
                if vasp_converged(p.with_name("OUTCAR"))
            ]
            if not candidates:
                reason = "no_matching_converged_DFT_output"
                atoms.info["dft_converged"] = False
                quarantine.append(atoms)
            else:
                raw_source, fresh = candidates[0]
                e, f = (
                    fresh.get_potential_energy(),
                    fresh.get_forces(apply_constraint=False),
                )
                agree = all(
                    abs(a.get_potential_energy() - e) < 1e-5
                    and np.allclose(
                        a.get_forces(apply_constraint=False), f, atol=1e-5, rtol=0
                    )
                    for _, a in candidates
                )
                if not agree:
                    raise ValueError(f"Conflicting DFT outputs for source frame {i}")
                # Retain source provenance, but take both labels from the DFT calculator.
                atoms.calc = fresh.calc
                refresh_dft_labels(atoms, str(raw_source.resolve()))
                atoms.calc = None
                atoms.info["dft_converged"] = True
                atoms.info["recovered_from"] = str(raw_source.resolve())
                atoms.info["recovered_previous_REF_energy"] = old_energy
                # The inherited global DB id identifies the source seed; job_id is a candidate index.
                parent = fresh.info.get("global_db_id")
                if parent is not None:
                    parent = root_seed(int(parent))
                if parent is not None:
                    atoms.info["split_group"] = (
                        f"{raw_root.resolve()}:seed_db_id:{parent}"
                    )
                if parent in seed_domains:
                    atoms.info["domain"] = seed_domains[parent]
                corrected += 1
                reason = "recovered_converged_DFT"
                accepted.append(atoms)
        else:
            parent = original.info.get("global_db_id")
            if parent is not None:
                parent = root_seed(int(parent))
            if parent is not None:
                atoms.info["split_group"] = f"{raw_root.resolve()}:seed_db_id:{parent}"
            atoms.info["dft_convergence_status"] = "not_audited"
            accepted.append(atoms)
        rows.append(
            {
                "source_index": i,
                "config_type": original.info.get("config_type"),
                "status": reason,
                "raw_source": str(raw_source) if raw_source else "",
                "old_energy": old_energy,
                "new_energy": float(atoms.info["REF_energy"]),
                "domain": atoms.info["domain"],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if accepted:
        write(targets[0], accepted, format="extxyz")
    if quarantine:
        write(targets[1], quarantine, format="extxyz")
    with targets[2].open("w") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(source.resolve()),
        "source_sha256": source_digest,
        "input": len(atoms_list),
        "recovered_high_sd": corrected,
        "accepted": len(accepted),
        "quarantined": len(quarantine),
        "note": "Initial labels retained; original DFT protocol compatibility and convergence still require audit.",
    }
    targets[3].write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Recovery summary: %s",
        recover_dataset(args.source, args.raw_root, args.output_dir),
    )


if __name__ == "__main__":
    main()
