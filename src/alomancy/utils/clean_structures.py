from ase import Atoms


def clean_structures(structures, base_name, high_accuracy_evaluation_job_dict):
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
        structure_copy.info["REF_energy"] = structure.get_potential_energy()
        structure_copy.arrays["REF_forces"] = structure.get_forces()
        structure_copy.info[
            "config_type"
        ] = f"{high_accuracy_evaluation_job_dict['name']}_{base_name}"
        cleaned_structures.append(structure_copy)

    return cleaned_structures
