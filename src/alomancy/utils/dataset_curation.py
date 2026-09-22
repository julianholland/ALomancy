"""Reference annotation and reversible selection of labelled atomistic data.

Energies are eV, formation energies per atom are eV/atom, forces are eV/Angstrom.
REF_energy is never shifted. The bulk reference is distinct from MACE E0s.
"""

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from ase import Atoms


def geometry_digest(atoms: Atoms) -> str:
    """Ordered geometry fingerprint (not a symmetry-equivalence test)."""
    h = hashlib.sha256()
    for array in (
        atoms.numbers,
        np.round(atoms.positions, 7),
        np.round(atoms.cell.array, 7),
        atoms.pbc,
    ):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def structure_domain(atoms: Atoms) -> str:
    """Prefer explicit provenance; do not infer dimensionality from VASP PBC."""
    if atoms.info.get("domain") and atoms.info["domain"] != "unknown":
        return str(atoms.info["domain"])
    declared = atoms.info.get("structure_type", atoms.info.get("structure type"))
    if declared in {"bulk", "surface", "cluster", "amorphous"}:
        return str(declared)
    if len(atoms) == 1:
        return "isolated_atom"
    ct = str(atoms.info.get("config_type", ""))
    if ct.startswith("surface"):
        return "surface"
    if ct in {"init_MP", "bulk_rattle", "init_stretch_compress"}:
        return "bulk"
    if ct == "init_amorphous":
        return "amorphous"
    if ct == "init_dimer" or len(atoms) == 2:
        return "dimer"
    if ct == "init_trimer" or len(atoms) == 3:
        return "trimer"
    # No specific rule for this config_type (e.g. "high_sd", an AL-loop
    # candidate) -- fall back to the config_type itself rather than
    # "unknown", so require_known_domain doesn't silently treat every
    # currently-unmapped-but-tagged structure as having no provenance at
    # all. A structure with no config_type either genuinely has none.
    return ct or "unknown"


def validate_policy(policy: dict) -> None:
    allowed = {"reference", "default", "domains", "require_known_domain"}
    if set(policy) - allowed:
        raise ValueError(f"Unknown dataset_curation keys: {set(policy) - allowed}")
    reference = policy.get("reference")
    if reference is not None:
        if not reference.get("id") or not reference.get("chemical_potentials"):
            raise ValueError("Reference requires id and chemical_potentials")
        if reference.get("status") not in {"verified", "estimated"}:
            raise ValueError(
                "Formation-energy filtering requires a verified or explicitly estimated reference"
            )
        if not all(
            np.isfinite(float(v)) for v in reference["chemical_potentials"].values()
        ):
            raise ValueError("Chemical potentials must be finite eV/atom values")
    rules = [policy.get("default", {}), *policy.get("domains", {}).values()]
    for rule in rules:
        if set(rule) - {"max_force", "formation_energy_per_atom"}:
            raise ValueError(f"Unknown filter keys: {set(rule)}")
        if "max_force" in rule:
            f = float(rule["max_force"])
            if not np.isfinite(f) or f <= 0:
                raise ValueError("max_force must be positive and finite")
        if "formation_energy_per_atom" in rule:
            bounds = rule["formation_energy_per_atom"]
            if not reference:
                raise ValueError("Energy window requires an explicit bulk reference")
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1]
            ):
                raise ValueError("Energy window must be finite [minimum, maximum]")


def annotate_structure(atoms: Atoms, policy: dict) -> dict:
    """Return fresh metadata; rejection never deletes or changes DFT labels."""
    domain = structure_domain(atoms)
    reasons = []
    metadata: dict[str, Any] = {
        "domain": domain,
        "curation_policy_id": hashlib.sha256(
            json.dumps(policy, sort_keys=True).encode()
        ).hexdigest(),
    }
    if policy.get("require_known_domain", False) and domain == "unknown":
        reasons.append("unknown_domain")
    if atoms.info.get("dft_converged") is False:
        reasons.append("unconverged_dft")
    n = len(atoms)
    e = atoms.info.get("REF_energy")
    f = atoms.arrays.get("REF_forces")
    valid_e = n > 0 and e is not None and np.isfinite(e)
    valid_f = f is not None and f.shape == (n, 3) and n > 0 and np.isfinite(f).all()
    if not valid_e:
        reasons.append("invalid_energy")
    if not valid_f:
        reasons.append("invalid_forces")
    rule = {**policy.get("default", {}), **policy.get("domains", {}).get(domain, {})}
    if valid_f:
        assert f is not None
        fmax = float(np.linalg.norm(f, axis=1).max())
        metadata["REF_max_force"] = fmax
        if "max_force" in rule and fmax >= rule["max_force"]:
            reasons.append("high_force")
    reference = policy.get("reference")
    if reference and valid_e:
        assert e is not None
        composition = Counter(atoms.get_chemical_symbols())
        mu = reference["chemical_potentials"]
        if set(composition) - set(mu):
            raise ValueError(
                f"Missing chemical potential for {set(composition) - set(mu)}"
            )
        ef = float(e) - sum(
            count * float(mu[element]) for element, count in composition.items()
        )
        metadata.update(
            REF_formation_energy=ef,
            REF_formation_energy_per_atom=ef / n,
            formation_reference_id=reference["id"],
        )
        bounds = rule.get("formation_energy_per_atom")
        if bounds and not bounds[0] <= ef / n <= bounds[1]:
            reasons.append("formation_energy")
    metadata.update(is_training_eligible=not reasons, filter_reasons=reasons)
    return metadata


def curate_database(db: Any, policy: dict) -> dict:
    """Recompute flags for ALL splits. Archived structures remain accessible."""
    validate_policy(policy)
    updates = {}
    counts: Counter[str] = Counter()
    for i, atoms in enumerate(db.get_all_as_atoms()):
        metadata = annotate_structure(atoms, policy)
        updates[i] = metadata
        counts["accepted" if metadata["is_training_eligible"] else "rejected"] += 1
    if updates:
        db.partition.set_metadata_bulk(updates, use_indices=True)
    return dict(counts)


def grouped_split(
    atoms_list: list[Atoms], fraction: float, seed: int
) -> tuple[list[Atoms], list[Atoms]]:
    """Split whole groups; shared geometry OR split_group links frames.

    Groups are stratified by domain. Isolated references and singleton domains
    stay in training. Large trajectory groups can make the fraction approximate.
    """
    if not 0 <= fraction < 1:
        raise ValueError("Split fraction must be in [0, 1)")
    parents = list(range(len(atoms_list)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, a in enumerate(atoms_list):
        keys = [("geometry", geometry_digest(a))]
        if a.info.get("split_group"):
            keys.append(("group", str(a.info["split_group"])))
        for k in keys:
            if k in seen:
                parents[find(i)] = find(seen[k])
            else:
                seen[k] = i
    groups = defaultdict(list)
    for i in range(len(atoms_list)):
        groups[find(i)].append(i)
    strata = defaultdict(list)
    for ids in groups.values():
        domains = {structure_domain(atoms_list[i]) for i in ids}
        if "isolated_atom" in domains:
            continue
        strata[tuple(sorted(domains))].append(ids)
    rng = np.random.default_rng(seed)
    held_out = set()
    for candidates in strata.values():
        if len(candidates) < 2 or fraction == 0:
            continue
        target = max(1, round(sum(map(len, candidates)) * fraction))
        size = 0
        # Always leave at least one group of each stratum in training.
        for j in rng.permutation(len(candidates))[:-1]:
            if size >= target:
                break
            held_out.update(candidates[j])
            size += len(candidates[j])
    return (
        [a for i, a in enumerate(atoms_list) if i not in held_out],
        [a for i, a in enumerate(atoms_list) if i in held_out],
    )
