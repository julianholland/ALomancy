import itertools
from pathlib import Path
from alomancy.initialize.amorphize import create_amorphous_atoms_list
from alomancy.initialize.singles_dimers_trimers import (
    create_single_atoms_list,
    create_dimer_atoms_list,
    create_trimer_atoms_list,
)
from alomancy.initialize.mp_interface import atoms_list_from_mp
from alomancy.initialize.stretch_and_compress import create_stretch_compress_atoms_list

from ase.io import write, read

import numpy as np
import warnings


def create_initialization_atoms_list(
    elements: list[str],
    mp_structures: bool = True,
    single_atoms: bool = True,
    target_non_mp_structures: int = 100,
    d_t_s_a_ratio: list[float] = [1, 1, 1, 1],
    densities_list: None |list[float] = None,
    deform_xyz: bool | list[bool] = False,
    max_deformation: float = 0.2,
    max_atom_number: int = 20,
    read_from_prior_file: None | str = None,
    save_file_name: None | str = None,
    seed: int = 803,
) -> list:
    assert len(elements) > 0, "At least one element must be specified."

    if read_from_prior_file is not None:
        assert Path(read_from_prior_file).is_file(), (
            f"File {read_from_prior_file} does not exist."
        )
        prior_atoms_list = read(read_from_prior_file, ":")
        if prior_atoms_list and len(prior_atoms_list) > target_non_mp_structures:
            print(f"Retrieved {len(prior_atoms_list)} structures from prior file.")
            return prior_atoms_list
        else:
            target_non_mp_structures -= len(prior_atoms_list)
            print(
                f"Retrieved {len(prior_atoms_list)} structures from prior file. Will create {target_non_mp_structures} additional structures."
            )
    else:
        mp_atoms_list = []
        if mp_structures:
            mp_atoms_list = atoms_list_from_mp(
                elements=elements,
                max_energy_above_hull=0.1,
                max_num_atoms=max_atom_number,
            )
            print(f"Retrieved {len(mp_atoms_list)} structures from Materials Project.")

        (
            dimer_target_structures,
            trimer_target_structures,
            stretch_target_structures,
            amorphous_target_structures,
        ) = np.floor(
            np.array(d_t_s_a_ratio) / np.sum(d_t_s_a_ratio) * target_non_mp_structures
        )
        print(
            f"Creating {dimer_target_structures} dimer structures, {trimer_target_structures} trimer structures, {stretch_target_structures} stretch/compress structures, and {amorphous_target_structures} amorphous structures."
        )

        dimer_composition_list = list(
            itertools.combinations_with_replacement(elements, 2)
        )
        dimer_atoms_list = []
        for dimer in dimer_composition_list:
            dimer_structures = create_dimer_atoms_list(
                element_a=dimer[0],
                element_b=dimer[1],
                num_dimers=int(
                    np.floor(dimer_target_structures / len(dimer_composition_list))
                ),
            )
            dimer_atoms_list.extend(dimer_structures)
        print(f"Created {len(dimer_atoms_list)} dimer structures.")

        trimer_composition_list = list(
            itertools.combinations_with_replacement(elements, 3)
        )
        trimer_atoms_list = []
        for trimer in trimer_composition_list:
            trimer_structures = create_trimer_atoms_list(
                element_a=trimer[0],
                element_b=trimer[1],
                element_c=trimer[2],
                num_trimers=int(
                    np.floor(trimer_target_structures / len(trimer_composition_list))
                ),
            )
            trimer_atoms_list.extend(trimer_structures)
        print(f"Created {len(trimer_atoms_list)} trimer structures.")

        amorphous_atoms_list = []
        for density in densities_list or [1.0]:
            print(f"Creating amorphous structures with density {density} g/cm^3.")
            amorphous_atoms_list.extend(
                create_amorphous_atoms_list(
                    elements=elements,
                    atom_number=max_atom_number,
                    density=density,
                    num_structures=int(
                        np.floor(amorphous_target_structures / len(densities_list) or 1)
                    ),
                    seed=seed,
                )
            )
        print(f"Created {len(amorphous_atoms_list)} amorphous structures.")

        stretch_compress_atoms_list = []
        if mp_structures and len(mp_atoms_list) > 0:
            stretch_compress_per_structure = int(
                np.floor(stretch_target_structures / len(mp_atoms_list))
            )
            if stretch_compress_per_structure <= 1:
                stretch_compress_per_structure = 2
                sc_mp_atoms_list = mp_atoms_list[
                    : int(np.floor(stretch_target_structures / 2))
                ]
                warnings.warn(
                    f"Not enough allotted structures to create stretch/compress structures for all MP structures. Only creating {len(sc_mp_atoms_list)} stretch/compress structures with max compression and extension."
                )
            else:
                sc_mp_atoms_list = mp_atoms_list
            
            for mp in sc_mp_atoms_list:
                stretched_compressed_structures = create_stretch_compress_atoms_list(
                    atoms=mp,
                    deform_xyz=deform_xyz,
                    max_deformation=max_deformation,
                    num_structures=stretch_compress_per_structure,
                )
                stretch_compress_atoms_list.extend(stretched_compressed_structures)
        print(
            f"Created {len(stretch_compress_atoms_list)} stretch/compress structures from {len(mp_atoms_list)} MP structures."
        )

        single_atoms_list = []
        single_composition_list = list(
            itertools.combinations_with_replacement(elements, 1)
        )
        for single in single_composition_list:
            single_structure = create_single_atoms_list(element=single[0])
            single_atoms_list.extend(single_structure)

        total_atoms_list = []
        if single_atoms:
            total_atoms_list.extend(single_atoms_list)
        if mp_structures:
            total_atoms_list.extend(mp_atoms_list)

        total_atoms_list.extend(dimer_atoms_list)
        total_atoms_list.extend(trimer_atoms_list)
        total_atoms_list.extend(amorphous_atoms_list)
        total_atoms_list.extend(stretch_compress_atoms_list)

        print(f"Created {len(total_atoms_list)} total structures for initialization.")
        if save_file_name is not None:
            write(save_file_name, total_atoms_list)
            print(f"Saved initialization structures to {save_file_name}.")
        return total_atoms_list


if __name__ == "__main__":
    atoms_list = create_initialization_atoms_list(
        elements=["C", "Na", "O"],
        mp_structures=True,
        target_non_mp_structures=100,
        d_t_s_a_ratio=[1, 1, 1, 1],
        max_atom_number=20,
        densities_list=[1.3],
        deform_xyz=True,
        max_deformation=0.4,
        seed=803,
    )
    # print(f"Created {len(atoms_list)} total structures for initialization.")
