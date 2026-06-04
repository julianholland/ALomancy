from ase import Atoms

def clean_structures(structures: list[Atoms], config_type: str, override_config_type: bool = False, already_computed: bool = True) -> list[Atoms]:
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
        if already_computed:
            structure_copy.info["REF_energy"] = structure.get_potential_energy()
            structure_copy.arrays["REF_forces"] = structure.get_forces()

        if override_config_type or "config_type" not in structure.info:
            structure_copy.info[
                "config_type"
            ] = config_type

        cleaned_structures.append(structure_copy)

    return cleaned_structures
