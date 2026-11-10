import logging

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

logger = logging.getLogger(__name__)

# Default rattle size, as a fraction of the structure's nearest-neighbour
# distance (about 0.08 Angstrom for fcc Pt), when rattle_standard_deviation
# isn't set: small enough to stay near the parent structure, large enough
# to give forces well above DFT noise, for any element.
DEFAULT_RATTLE_FRACTION = 0.03

# A 1-atom periodic cell is rattled as this supercell instead.
SINGLE_ATOM_SUPERCELL = (2, 2, 2)


def nearest_neighbour_distance(atoms: Atoms) -> float | None:
    """The shortest interatomic distance in *atoms*, periodic images
    included; None when no two atoms are within 10 Angstrom."""
    for cutoff in (3.0, 5.0, 10.0):
        distances = neighbor_list("d", atoms, cutoff)
        if len(distances):
            return float(np.min(distances))
    return None


def create_rattle_atoms_list(
    atoms: Atoms,
    rattle_standard_deviation: float | None,
    num_structures: int,
    seed: int,
) -> list[Atoms]:
    """
    Apply ASE's built-in Atoms.rattle to independent copies of an atoms
    object.

    A 1-atom cell is never rattled as it is: moving its only atom is a pure
    translation (same energy, zero forces), a duplicate label. A periodic
    1-atom cell is rattled as a 2x2x2 supercell instead (info: supercell,
    parent_config_type); a non-periodic single atom gives nothing.

    Parameters:
    atoms (Atoms): The atoms object to rattle.
    rattle_standard_deviation (float | None): Standard deviation (Angstrom)
        of the per-atom Gaussian displacement, passed to Atoms.rattle's
        stdev. None: DEFAULT_RATTLE_FRACTION of the nearest-neighbour
        distance (of the supercell, for a 1-atom cell).
    num_structures (int): The number of rattled copies to generate.
    seed (int): Base seed; copy i uses seed + i so the num_structures
        copies are independent perturbations, not identical ones.

    Returns:
    list[Atoms]: A list of rattled atoms objects.
    """
    if num_structures <= 0:
        return []
    parent_config_type = atoms.info.get("config_type")
    base = atoms
    supercell = None
    if len(atoms) == 1:
        if not atoms.pbc.all():
            logger.debug(
                "Not rattling the single non-periodic atom %s: it would only "
                "translate.",
                atoms.get_chemical_formula(),
            )
            return []
        base = atoms.repeat(SINGLE_ATOM_SUPERCELL)
        supercell = "x".join(str(n) for n in SINGLE_ATOM_SUPERCELL)

    stdev = rattle_standard_deviation
    if stdev is None:
        nn = nearest_neighbour_distance(base)
        if nn is None:
            logger.debug(
                "No neighbours within 10 Angstrom in %s: not rattled.",
                base.get_chemical_formula(),
            )
            return []
        stdev = DEFAULT_RATTLE_FRACTION * nn

    rattled_atoms_list = []
    for i in range(num_structures):
        rattled_atoms = base.copy()
        rattled_atoms.rattle(stdev=stdev, seed=seed + i)
        rattled_atoms.info["config_type"] = "init_rattle"
        rattled_atoms.info["needs_relaxation"] = False
        rattled_atoms.info["rattle_standard_deviation"] = float(stdev)
        if parent_config_type is not None:
            rattled_atoms.info["parent_config_type"] = parent_config_type
        if supercell is not None:
            rattled_atoms.info["supercell"] = supercell
        rattled_atoms_list.append(rattled_atoms)

    return rattled_atoms_list
