import logging

from ase import Atoms

logger = logging.getLogger(__name__)

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
        structure_copy.info = structure.info.copy()  # start with a copy of the original info dictionary

        if already_computed:
            if "REF_energy" not in structure.info or "REF_forces" not in structure.arrays:
                try:
                    energy = structure.get_potential_energy()
                    forces = structure.get_forces()
                except Exception as e:
                    raise ValueError(
                        "Structure is marked as already_computed but is missing REF_energy or REF_forces, and they could not be computed. Original error: " + str(e)
                    ) from e
            else:
                energy = structure.info["REF_energy"]
                forces = structure.arrays["REF_forces"]
            
            structure_copy.info["REF_energy"] = energy
            structure_copy.arrays["REF_forces"] = forces

        if override_config_type or "config_type" not in structure.info:
            logger.debug("Setting config_type to '%s'.", config_type)
            structure_copy.info[
                "config_type"
            ] = config_type

        cleaned_structures.append(structure_copy)

    return cleaned_structures
