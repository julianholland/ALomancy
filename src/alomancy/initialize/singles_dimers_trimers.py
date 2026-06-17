import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii
from ase.io import write


def create_single_atoms_list(element: str) -> list:
    """Create a single atoms object in a list."""
    cell = [vdw_radii[atomic_numbers[element]]*3]*3  # Ensure enough space
    atom = Atoms(element, positions=[(0, 0, 0)])
    atom.cell=cell
    atom.info['config_type'] = 'IsolatedAtom'
    atom.info['needs_relaxation'] = False
    return [atom]

def create_dimer_atoms_list(element_a: str, element_b: str, num_dimers: int) -> list:
    """Create a list of atoms objects of two atoms at various distances."""
    dimer_atoms = []
    vdw_list = [vdw_radii[atomic_numbers[element_a]], vdw_radii[atomic_numbers[element_b]]]
    distance_range = (0.2 * sum(vdw_list), 1.0 * sum(vdw_list))
    cell = [distance_range[1]*3]*3  # Ensure enough space
    if num_dimers > 0:
        for d in np.linspace(distance_range[0], distance_range[1], num_dimers):
            a1 = Atoms(element_a, positions=[(0, 0, 0)])
            a2 = Atoms(element_b, positions=[(0, 0, d)])
            dimer = a1 + a2
            dimer.cell=cell
            dimer.info['config_type'] = 'init_dimer'
            dimer.info['distance'] = f'{d:.3f}'
            dimer.info['needs_relaxation'] = False
            dimer_atoms.append(dimer)
    return dimer_atoms

def create_trimer_atoms_list(element_a: str, element_b: str, element_c: str, num_trimers: int) -> list:
    """Create a list of atoms objects of three atoms at various distances."""
    trimer_atoms = []
    vdw_list = [vdw_radii[atomic_numbers[element_a]], vdw_radii[atomic_numbers[element_b]], vdw_radii[atomic_numbers[element_c]]]
    distance_range = (0.2 * sum(vdw_list), 1.0 * sum(vdw_list))
    cell = [distance_range[1]*3]*3  # Ensure enough space
    if num_trimers > 1:
        for dx in np.linspace(distance_range[0], distance_range[1], int(np.floor(np.sqrt(num_trimers-2)))):
            for dy in np.linspace(distance_range[0], distance_range[1], int(np.floor(np.sqrt(num_trimers-2)))):
                a1 = Atoms(element_a, positions=[(0, 0, 0)])
                a2 = Atoms(element_b, positions=[(0, dy, 2*dx)])
                a3 = Atoms(element_c, positions=[(0, 2*dy, dx)])
                trimer = a1 + a2 + a3
                trimer.cell = cell
                trimer.info['config_type'] = 'init_trimer'
                trimer.info['deformation'] = f'{dx:.3f}_{dy:.3f}'
                trimer.info['needs_relaxation'] = False
                trimer_atoms.append(trimer)

        # add some linear trimers if we don't have enough
        for dx in np.linspace(distance_range[0], distance_range[1], num_trimers - len(trimer_atoms)):
                a1 = Atoms(element_a, positions=[(0, 0, 0)])
                a2 = Atoms(element_b, positions=[(0, 0, dx)])
                a3 = Atoms(element_c, positions=[(0, 0, 2*dx)])
                trimer = a1 + a2 + a3
                trimer.cell = cell
                trimer.info['config_type'] = 'init_trimer'
                trimer.info['deformation'] = f'{dx:.3f}_0.000'
                trimer.info['needs_relaxation'] = False
                trimer_atoms.append(trimer)

    return trimer_atoms

if __name__ == "__main__":
    dimers = create_dimer_atoms_list("H", "O",  50)
    write("dimers.xyz", dimers)
    trimers = create_trimer_atoms_list("H", "O", "H", 50)
    write("trimers.xyz", trimers)
