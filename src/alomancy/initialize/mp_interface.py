import itertools
import logging
import os

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

# from pymatgen.
logger = logging.getLogger(__name__)
mp_api_key = os.getenv("MP_API_KEY")
if not mp_api_key:
    raise ValueError("MP_API_KEY environment variable not set. Please set it to your Materials Project API key.")

def retrieve_mp_material_docs(
    elements: list, max_energy_above_hull: float, max_num_atoms: int,
) -> list:
    element_permutations = []
    for i in range(len(elements)):
        n_elements = list(itertools.combinations(elements, i + 1))
        element_permutations.extend(n_elements)


    docs_list = []
    for el in element_permutations:
        logger.info("Retrieving structures from Materials Project: elements=%s, hull_cutoff=%.2f eV, max_atoms=%d", el, max_energy_above_hull, max_num_atoms)

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
    logger.info("Retrieved %d structures from Materials Project.", len(atoms_list))

