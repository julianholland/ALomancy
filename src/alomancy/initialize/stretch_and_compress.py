from ase import Atoms
import numpy as np

def create_stretch_compress_atoms_list(atoms: Atoms, deform_xyz: bool | list[bool], max_deformation: float, num_structures: int) -> list[Atoms]:
    """
    Apply a stretch/compression deformation to the cell of an atoms object.

    Parameters:
    atoms (Atoms): The atoms object to deform.
    deform_xyz (bool | list[bool]): Whether to deform in x, y, z directions. If a list, it should be of length 3.
    max_deformation (float): The maximum deformation to apply (as a fraction of the original cell size).
    steps (int): The number of deformation steps to generate.

    Returns:
    list[Atoms]: A list of deformed atoms objects.

    Example:
    atoms = Atoms("H2O", positions=[(0, 0, 0), (0.76, 0.58, 0), (-0.76, 0.58, 0)], cell=[3,3,3])
    deformed_atoms_list = stretch_compress_atoms(atoms, deform_xyz=[True, False, False], max_deformation=0.1, steps=5)
    
    """
    deformed_atoms_list = []
    if num_structures > 0: 
        for i in np.linspace(1-max_deformation, 1+max_deformation, num_structures):
            deformed_atoms = atoms.copy()
            if deform_xyz is True:
                cell_multiplier = np.eye(3) * i
            elif deform_xyz is False:
                cell_multiplier = np.eye(3)
            elif isinstance(deform_xyz, list) and len(deform_xyz) == 3:
                cell_multiplier = np.diag([i if deform else 1 for deform in deform_xyz])
            else:
                raise ValueError("deform_xyz must be a bool or a list of three bools.")

            new_cell = deformed_atoms.cell * cell_multiplier
            deformed_atoms.set_cell(new_cell, scale_atoms=True)
            deformed_atoms.info['config_type'] = 'init_stretch_compress'
            deformed_atoms.info['deformation'] = f'{i:.3f}'
            deformed_atoms.info['needs_relaxation'] = False
            deformed_atoms_list.append(deformed_atoms)
            print([deformed_atoms.info for deformed_atoms in deformed_atoms_list])

    return deformed_atoms_list
    
if __name__ == "__main__":
    atoms = Atoms("H2O", positions=[(0, 0, 0), (0.76, 0.58, 0), (-0.76, 0.58, 0)], cell =[3,3,3])
    deformed_atoms_list = create_stretch_compress_atoms_list(atoms, deform_xyz=[True, False, False], max_deformation=0.4, num_structures=5)
    print([deformed_atoms.positions for deformed_atoms in deformed_atoms_list])