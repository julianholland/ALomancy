import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def clean_structures(
    structures: list[Atoms],
    config_type: str,
    override_config_type: bool = False,
    already_computed: bool = True,
    extra_metadata: dict | None = None,
) -> list[Atoms]:
    """
    adds DFT results to copy of structures info dictionary.
    """
    cleaned_structures = []
    for structure in structures:
        # copy structure with just the right information
        structure_copy = Atoms(
            symbols=structure.get_chemical_symbols(),
            positions=structure.get_positions(),
            cell=structure.get_cell(),
            pbc=structure.get_pbc(),
        )
        structure_copy.info = (
            structure.info.copy()
        )  # start with a copy of the original info dictionary

        structure_copy.info.update(
            extra_metadata or {}
        )  # add any extra metadata if provided

        if already_computed:
            if (
                "REF_energy" not in structure.info
                or "REF_forces" not in structure.arrays
            ):
                try:
                    energy = structure.get_potential_energy()
                    forces = structure.get_forces()
                except Exception as e:
                    raise ValueError(
                        "Structure is marked as already_computed but is missing REF_energy or REF_forces, and they could not be computed. Original error: "
                        + str(e)
                    ) from e
            else:
                energy = structure.info["REF_energy"]
                forces = structure.arrays["REF_forces"]

            structure_copy.info["REF_energy"] = energy
            structure_copy.arrays["REF_forces"] = forces

        if override_config_type or "config_type" not in structure.info:
            logger.debug("Setting config_type to '%s'.", config_type)
            structure_copy.info["config_type"] = config_type

        cleaned_structures.append(structure_copy)

    return cleaned_structures


def filter_structures_by_min_bond_distance(
    structures: list[Atoms], min_distance: float = 0.5
) -> list[Atoms]:
    """Exclude structures with any pairwise atomic distance below min_distance (Å).

    Guards against submitting an unphysical/exploded structure (e.g. an MD
    instability, or a badly-generated dimer/trimer/stretch-compress
    structure) to expensive DFT. `get_all_distances(mic=True)` is safe to
    call unconditionally: it uses the minimum-image convention for
    periodic structures and falls back to plain distances for non-periodic
    ones, correctly covering every config_type this codebase produces.
    Single-atom structures always pass through unfiltered — there is no
    pairwise distance to check, and `np.triu_indices(1, k=1)` correctly
    returns an empty index set rather than raising.
    """
    filtered = []
    n_excluded = 0
    for structure in structures:
        if len(structure) < 2:
            filtered.append(structure)
            continue
        distance_matrix = structure.get_all_distances(mic=True)
        upper = distance_matrix[np.triu_indices(len(structure), k=1)]
        if upper.min() >= min_distance:
            filtered.append(structure)
        else:
            n_excluded += 1

    if n_excluded:
        logger.warning(
            "Excluded %d/%d structure(s) with a bond shorter than %.2f Å "
            "from DFT submission.",
            n_excluded,
            len(structures),
            min_distance,
        )

    return filtered
