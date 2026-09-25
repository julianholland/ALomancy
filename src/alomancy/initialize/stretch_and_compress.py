import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def create_stretch_compress_atoms_list(
    atoms: Atoms,
    max_lattice_deformation: float,
    num_structures: int,
) -> list[Atoms]:
    """
    Apply an isotropic stretch/compression deformation to the cell of an
    atoms object.

    Parameters:
    atoms (Atoms): The atoms object to deform.
    max_lattice_deformation (float): The maximum deformation to apply (as a
        fraction of the original cell size).
    num_structures (int): The number of deformation steps to generate.

    Returns:
    list[Atoms]: A list of deformed atoms objects.

    Example:
    atoms = Atoms("H2O", positions=[(0, 0, 0), (0.76, 0.58, 0), (-0.76, 0.58, 0)], cell=[3,3,3])
    deformed_atoms_list = create_stretch_compress_atoms_list(atoms, max_lattice_deformation=0.1, num_structures=5)

    """
    deformed_atoms_list = []
    if num_structures > 0:
        for i in np.linspace(
            1 - max_lattice_deformation, 1 + max_lattice_deformation, num_structures
        ):
            deformed_atoms = atoms.copy()
            cell_multiplier = np.eye(3) * i
            new_cell = deformed_atoms.cell * cell_multiplier
            deformed_atoms.set_cell(new_cell, scale_atoms=True)
            deformed_atoms.info["config_type"] = "init_stretch_compress"
            deformed_atoms.info["deformation"] = f"{i:.3f}"
            deformed_atoms.info["needs_relaxation"] = False
            deformed_atoms_list.append(deformed_atoms)

    return deformed_atoms_list


if __name__ == "__main__":
    atoms = Atoms(
        "H2O", positions=[(0, 0, 0), (0.76, 0.58, 0), (-0.76, 0.58, 0)], cell=[3, 3, 3]
    )
    deformed_atoms_list = create_stretch_compress_atoms_list(
        atoms, max_lattice_deformation=0.4, num_structures=5
    )
    logger.debug(
        "Stretch/compress positions: %s",
        [deformed_atoms.positions.tolist() for deformed_atoms in deformed_atoms_list],
    )
