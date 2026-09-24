"""Initialiser module: wraps create_initialization_atoms_list/
compute_initialization_needs for the modular AL architecture.

Produces unlabeled structures only -- no DFT evaluation, no DB writes, no
train/test split. It does not depend on the evaluator category (see the
architecture plan's initialiser decision): the skeleton is what calls the
DFT evaluator on this module's output afterward, exactly as it already
does for AL-loop-generated structures. This also means the on-disk
fast-path check (pre-existing initial_train/test files), the DB-first/
extra-datasets-second needs computation, clean_structures, db.add_structures,
and the initial train/test split -- all currently inline in
standard_active_learning.initialize_training_set -- move to skeleton-level
orchestration, not here.

Self-contained config surface: reads only initialiser-specific settings
(creation_kwargs), never reaching into another module's section.

Bootstrap generation is structurally a different problem from the AL-loop
structure_generator abstraction (no trained model, no uncertainty-based
selection) -- generate() here does not take seed_atoms/model_path.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ase import Atoms
from ase.io import read

from alomancy.initialize.initialization_structure_list import (
    compute_initialization_needs,
    create_initialization_atoms_list,
)

if TYPE_CHECKING:
    from alomancy.database.global_database import GlobalDatabase

logger = logging.getLogger(__name__)


def compute_needs(db: "GlobalDatabase", config: dict) -> dict:
    """Compare DB contents against initialization targets (config's
    creation_kwargs) and return what still needs to be generated -- see
    compute_initialization_needs for the returned dict's shape.
    """
    creation_kwargs = config["creation_kwargs"]
    return compute_initialization_needs(
        db=db,
        elements=creation_kwargs["elements"],
        _single_atoms=creation_kwargs.get("single_atoms", True),
        mp_structures=creation_kwargs.get("mp_structures", True),
        num_dimers_per_combo=creation_kwargs.get("num_dimers_per_combo", 10),
        num_trimers_per_combo=creation_kwargs.get("num_trimers_per_combo", 5),
        num_amorphous=creation_kwargs.get("num_amorphous", 100),
    )


def _generated_structures_path(base_name: str) -> Path:
    return Path("results", base_name, "initialization_structures_generated.xyz")


def output_paths(config: dict, *, base_name: str, name: str) -> list[Path]:  # noqa: ARG001
    """Restart check: the file create_initialization_atoms_list already
    writes unconditionally."""
    return [_generated_structures_path(base_name)]


def read_existing_result(config: dict, *, base_name: str, name: str) -> list[Atoms]:  # noqa: ARG001
    path = _generated_structures_path(base_name)
    if not path.exists():
        raise ValueError(f"No cached initialiser result at {path}.")
    return list(read(path, ":", format="extxyz"))


def generate(
    config: dict,
    *,
    base_name: str,
    name: str,  # noqa: ARG001 -- unused (no remote submission); uniform across module categories
    hpc: dict,  # noqa: ARG001 -- unused (no remote submission); uniform across module categories
    max_time: str,  # noqa: ARG001 -- unused (no remote submission); uniform across module categories
    needs: dict | None = None,
) -> list[Atoms]:
    """Generate bootstrap structures (dimers/trimers/amorphous/MP/isolated
    atoms), reading targets from config's creation_kwargs.

    When `needs` is given (from compute_needs), only the missing subset
    (relative to what's already in the DB) is generated -- enabling
    idempotent restarts. When None, the full target set is generated.
    hpc/max_time are accepted for interface uniformity across module
    categories but unused: initialisation runs entirely in the local
    driver process (the Materials Project fetch is a plain network call,
    not an ExPyRe remote job).
    """
    work_dir = Path("results", base_name)
    work_dir.mkdir(exist_ok=True, parents=True)
    creation_kwargs = config["creation_kwargs"]

    return create_initialization_atoms_list(
        work_dir=str(work_dir),
        elements=creation_kwargs["elements"],
        mp_structures=(
            needs["mp_structures"]
            if needs is not None
            else creation_kwargs.get("mp_structures", True)
        ),
        single_atoms=(
            bool(needs["isolated_atoms"])
            if needs is not None
            else creation_kwargs.get("single_atoms", True)
        ),
        num_dimers_per_combo=creation_kwargs.get("num_dimers_per_combo", 10),
        num_trimers_per_combo=creation_kwargs.get("num_trimers_per_combo", 5),
        num_amorphous=creation_kwargs.get("num_amorphous", 100),
        num_stretch_compress_per_mp=creation_kwargs.get(
            "num_stretch_compress_per_mp", 5
        ),
        densities_list=creation_kwargs.get("densities_list"),
        deform_xyz=creation_kwargs.get("deform_xyz", False),
        max_deformation=creation_kwargs.get("max_deformation", 0.2),
        max_atom_number=creation_kwargs.get("max_atom_number", 20),
        amorphous_atom_number=creation_kwargs.get("amorphous_atom_number", 20),
        mp_max_energy_above_hull=creation_kwargs.get("mp_max_energy_above_hull", 0.1),
        composition_list=creation_kwargs.get("composition_list"),
        seed=creation_kwargs.get("seed", 803),
        isolated_atoms_override=(
            (needs["isolated_atoms"] or None) if needs is not None else None
        ),
        dimer_override=(needs["dimer_override"] or None) if needs is not None else None,
        trimer_override=(needs["trimer_override"] or None)
        if needs is not None
        else None,
        amorphous_override=(
            (needs["amorphous_override"] or None) if needs is not None else None
        ),
    )
