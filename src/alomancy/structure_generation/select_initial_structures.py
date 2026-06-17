import logging
import warnings

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def select_initial_structures(
    base_name,
    structure_generation_job_dict: dict,
    train_atoms_list: list[Atoms],
    max_number_of_concurrent_jobs: int = 5,
    chem_formula_list: list[str] | None = None,
    selectable_configs: list[str] | None = None,
    atom_number_range: tuple[int, int] = (0, 0),
    enforce_chemical_diversity: bool = False,
):
    """randomly selects structures from a train set based on a chemical formula or
    max number of atoms and number of mds to run.

    Probably should be moved to its own file.
    Args:
        base_name (str): Base name for the job.
        structure_generation_job_dict (dict): Dictionary containing job parameters.
        train_xyzs (list[Atoms]): List of Atoms objects from the training set.
        desired_initial_structures (int): Number of initial structures to select.
        chem_formula (list[str]): List of chemical formulas to filter structures. If empty, no filtering is applied.
        max_atoms (Optional[int]): Maximum number of atoms in the selected structures. If None, no filtering is applied.
        enforce_chemical_diversity (bool): Whether to enforce chemical diversity in selection.

    Returns:
        list[Atoms]: Selected Atoms objects for structure generation.
    """
    # Handle None default for mutable argument
    atom_number_range = tuple(atom_number_range)
    if atom_number_range != (0, 0):
        assert (
            atom_number_range[0] <= atom_number_range[1]
        ), "atom_number_range must be a tuple of two integers where the first is less than or equal to the second"
        if atom_number_range[0] < 2:
            warnings.warn(
                f"atom_number_range minimum value is {atom_number_range[0]}, which allows single-atom structures. "
                "This can lead to problems with some structure generators like MD simulations, as single atoms "
                "cannot form proper molecular dynamics trajectories. Consider setting the minimum to 2 or higher.",
                UserWarning,
                stacklevel=3,
            )

    if chem_formula_list is None:
        chem_formula_list = []

    if atom_number_range != (0, 0) and len(chem_formula_list) > 0:
        filtered_structures = [
            s
            for s in train_atoms_list
            if s.get_chemical_formula() in chem_formula_list
            and len(s) <= atom_number_range[1]
            and len(s) >= atom_number_range[0]
        ]

    elif atom_number_range != (0, 0):
        filtered_structures = [
            s
            for s in train_atoms_list
            if len(s) <= atom_number_range[1] and len(s) >= atom_number_range[0]
        ]

    elif len(chem_formula_list) > 0:
        filtered_structures = [
            s for s in train_atoms_list if s.get_chemical_formula() in chem_formula_list
        ]
    else:
        filtered_structures = train_atoms_list
    
    if selectable_configs is not None:
        filtered_structures = [
            s for s in filtered_structures if s.info.get("config_type") in selectable_configs
        ]

    assert (
        len(filtered_structures) >= max_number_of_concurrent_jobs
    ), f"Not enough structures to select {max_number_of_concurrent_jobs} from. Available: {len(filtered_structures)}"

    if not enforce_chemical_diversity:
        initial_atoms = [
            filtered_structures[x]
            for x in np.random.choice(
                np.array(range(len(filtered_structures))),
                max_number_of_concurrent_jobs,
                replace=False,
            )
        ]
        mark_structures_for_dft(initial_atoms, base_name, structure_generation_job_dict["name"])
        return initial_atoms
    
    # Ensure chemical diversity by selecting unique chemical formulas
    # If there are fewer unique formulas than `max_number_of_concurrent_jobs`, select all
    
    unique_chemical_formulas = {
        s.get_chemical_formula() for s in filtered_structures
    }
    if len(unique_chemical_formulas) <= max_number_of_concurrent_jobs:
        list_of_formulas = list(unique_chemical_formulas)
        extra_formulas = [
            np.random.choice(list(unique_chemical_formulas), replace=False)
            for _ in range(max_number_of_concurrent_jobs - len(list_of_formulas))
        ]
        list_of_formulas.extend(extra_formulas)


    else:
        # select formulas with probability inversely proportional to their frequency in the dataset to promote diversity
        all_chemical_formulas = [s.get_chemical_formula() for s in filtered_structures]
        formula_counts = {formula: all_chemical_formulas.count(formula) for formula in set(all_chemical_formulas)}
        formula_probabilities = {formula: 1/count for formula, count in formula_counts.items()}
        list_of_formulas = np.random.choice(
            list(unique_chemical_formulas),
            max_number_of_concurrent_jobs,
            replace=False,
            p=[formula_probabilities[formula]/sum(formula_probabilities.values()) for formula in unique_chemical_formulas]
        )

    initial_atoms = []
    for chemical_formula in list_of_formulas:
        formula_structures = [
            s
            for s in filtered_structures
            if s.get_chemical_formula() == chemical_formula
        ]
        selected_structure = np.random.choice(
            np.array(range(len(formula_structures)))
        )
        initial_atoms.append(formula_structures[selected_structure])


    mark_structures_for_dft(initial_atoms, base_name, structure_generation_job_dict["name"])

    logger.debug("Structures selected for MD: %s", [a.get_chemical_formula() for a in initial_atoms])

    return initial_atoms

def mark_structures_for_dft(atoms_list: list[Atoms], base_name: str, job_name: str) -> None:
    for atoms in atoms_list:
        atoms.info["job_id"] = atoms.info.get("job_id", -1)
        atoms.info["config_type"] = f"{base_name}_{job_name}"