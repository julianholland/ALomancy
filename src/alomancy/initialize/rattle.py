import logging

from ase import Atoms

logger = logging.getLogger(__name__)


def create_rattle_atoms_list(
    atoms: Atoms,
    rattle_standard_deviation: float,
    num_structures: int,
    seed: int,
) -> list[Atoms]:
    """
    Apply ASE's built-in Atoms.rattle to independent copies of an atoms
    object.

    Parameters:
    atoms (Atoms): The atoms object to rattle.
    rattle_standard_deviation (float): Standard deviation (Angstrom) of the
        per-atom Gaussian position displacement, passed straight through to
        Atoms.rattle's stdev.
    num_structures (int): The number of rattled copies to generate.
    seed (int): Base seed; copy i uses seed + i so the num_structures
        copies are independent perturbations, not identical ones.

    Returns:
    list[Atoms]: A list of rattled atoms objects.
    """
    rattled_atoms_list = []
    for i in range(num_structures):
        rattled_atoms = atoms.copy()
        rattled_atoms.rattle(stdev=rattle_standard_deviation, seed=seed + i)
        rattled_atoms.info["config_type"] = "init_rattle"
        rattled_atoms.info["needs_relaxation"] = False
        rattled_atoms_list.append(rattled_atoms)

    return rattled_atoms_list
