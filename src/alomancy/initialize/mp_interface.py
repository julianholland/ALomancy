import os

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
import itertools
# from pymatgen.


def retrieve_mp_material_docs(
    elements: list, max_energy_above_hull: float, max_num_atoms: int, 
) -> list:
    element_permutations = []
    for i in range(len(elements)):
        n_elements = list(itertools.combinations(elements, i + 1))
        element_permutations.extend(n_elements)

    print(f"Element permutations: {element_permutations}")
    docs_list = []
    for el in element_permutations:
        mp_api_key = "0qR6VlWDWYQ3rzzwrBqKlnVsXpWcOOyp"
        print(mp_api_key)
        with MPRester(mp_api_key) as mpr:
            docs = mpr.materials.summary.search(
                elements=el,
                num_elements=len(el),
                energy_above_hull=(0, max_energy_above_hull),
                num_sites=(2, max_num_atoms),
            )
            docs_list.extend(docs)

    return docs_list


def docs_to_atoms(docs: list) -> list:
    toatoms = AseAtomsAdaptor()
    atoms_list = []
    for doc in docs:
        atoms_list.append(toatoms.get_atoms(doc.structure, msonable=False))

    return atoms_list

def atoms_list_from_mp(elements: list, max_energy_above_hull: float, max_num_atoms: int, relax_structures: bool = False) -> list:
    docs = retrieve_mp_material_docs(elements, max_energy_above_hull, max_num_atoms)
    atoms_list = docs_to_atoms(docs)
    for atoms in atoms_list:
        atoms.info['config_type'] = 'init_MP'
        atoms.info['needs_relaxation'] = relax_structures

    return atoms_list

if __name__ == "__main__":
    elements = ["C", "Na", "O"]
    max_energy_above_hull = 0.1
    max_num_atoms = 20
    atoms_list = atoms_list_from_mp(elements, max_energy_above_hull, max_num_atoms)
    print(f"Retrieved {len(atoms_list)} documents from Materials Project.")

