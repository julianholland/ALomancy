import logging
import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
import itertools

logger = logging.getLogger(__name__)

def create_amorphous_atoms_list(elements: list[str], atom_number: int, density:float, num_structures: int, seed: int, composition_list: None | list[list[str]] = None) -> list[Atoms]:
    """
    Create a list of amorphous atoms objects with given elements, atom number, and density.
    This is done by randomly placing atoms in a box and then relaxing the structure using a 
    Lennard-Jones potential to ensure no atoms are too close.

    Parameters:
    elements (list[str]): List of element symbols to use in the amorphous structures.
    atom_number (int): Number of atoms in each structure.
    density (float): Density of the structure in g/cm^3.
    num_structures (int): Number of amorphous structures to generate.
    seed (int): Random seed for reproducibility.
    composition_list (None | list[list[str]]): Optional list of specific compositions to use. If None, random compositions will be generated.

    Returns:
    list[Atoms]: A list of amorphous atoms objects.

    Example:
    create_amorphous_atoms_list(
        elements = ["H", "O"],
        atom_number = 10,
        density = 1.0  # g/cm^3
        num_structures = 5,
        seed = 803,
        composition_list = None
    )

    """
    amorphous_atoms_list = []
    if num_structures > 0:
        rng= np.random.default_rng(seed)
        if composition_list is None:
            
            composition_tuple_list = list(itertools.combinations_with_replacement(elements, atom_number))
            composition_list = [list(comp) for comp in composition_tuple_list]

        if len(composition_list) < num_structures:
            extra_composition_indices=list(rng.choice(range(len(composition_list)), num_structures - len(composition_list)))
            extra_compositions=[composition_list[i] for i in extra_composition_indices]
            composition_list.extend(extra_compositions)

        if len(composition_list) > num_structures:
            composition_indices=list(rng.choice(range(len(composition_list)), num_structures))
            composition_list = [composition_list[i] for i in composition_indices]

        def calculate_side_length(composition: list[str]) -> float:
            molecular_weight=sum(atomic_masses[atomic_numbers[el]] for el in composition) # in amu
            volume = molecular_weight / (density * 0.6022)  # in Å^3
            return volume ** (1/3)  # Assuming cubic cell for simplicity

        for composition in composition_list:
            cell = np.eye(3) * calculate_side_length(composition)
            atoms=  Atoms(symbols=composition,
                        positions=rng.random((atom_number, 3)) * cell[0,0],
                        cell=cell,
                        pbc=True)
            amorphous_atoms_list.append(atoms)

        # ensure no atoms are too close
        for atoms in amorphous_atoms_list:
            lj = LennardJones(sigma=1.0, epsilon=0.0103)  # parameters for Argon, adjust as needed
            atoms.calc = lj
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.05, steps=200)
            atoms.info['config_type'] = 'init_amorphous'
            atoms.info['needs_relaxation'] = True

    return amorphous_atoms_list

if __name__ == "__main__":
    elements = ["H", "O"]
    atom_number = 3
    density = 1.0  # g/cm^3
    num_structures = 50
    amorphous_atoms_list = create_amorphous_atoms_list(elements, atom_number, density, num_structures, composition_list=None, seed=803)
    logger.debug("Amorphous structures generated: %s", [atoms.get_chemical_formula() for atoms in amorphous_atoms_list])