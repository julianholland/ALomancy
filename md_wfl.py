from ase.md.langevin import Langevin
from mace.calculators import MACECalculator
from ase.io import write
from ase.units import fs
from ase import Atom, Atoms
import numpy as np
import pandas as pd
from pathlib import Path
from typing_extensions import Optional
from tqdm import tqdm


def select_md_structures(
    base_name,
    job_name: str,
    train_xyzs: list[Atoms],
    number_of_mds: int = 5,
    chem_formula_list: list[str] = [],
    atom_number_range: tuple[int, int] = (0, 0),
    enforce_chemical_diversity: bool = False,
    verbose: int = 0,
):
    """randomly selects structures from a train set based on a chemical formula or
    max number of atoms and number of mds to run.

    Probably should be moved to its own file.
    Args:
        base_name (str): Base name for the job.
        job_name (str): Name of the job.
        train_xyzs (list[Atoms]): List of Atoms objects from the training set.
        number_of_mds (int): Number of MD structures to select.
        chem_formula (list[str]): List of chemical formulas to filter structures. If empty, no filtering is applied.
        max_atoms (Optional[int]): Maximum number of atoms in the selected structures. If None, no filtering is applied.
        enforce_chemical_diversity (bool): Whether to enforce chemical diversity in selection.

    Returns:
        list[Atoms]: Selected Atoms objects for MD runs.
    """

    if atom_number_range != (0, 0) and len(chem_formula_list) > 0:
        filtered_structures = [
            s
            for s in train_xyzs
            if s.get_chemical_formula() in chem_formula_list
            and len(s) <= atom_number_range[1]
            and len(s) >= atom_number_range[0]
        ]
    elif atom_number_range != (0, 0):
        filtered_structures = [
            s
            for s in train_xyzs
            if len(s) <= atom_number_range[1] and len(s) >= atom_number_range[0]
        ]
    elif len(chem_formula_list) > 0:
        filtered_structures = [
            s for s in train_xyzs if s.get_chemical_formula() in chem_formula_list
        ]
    else:
        filtered_structures = train_xyzs

    assert len(filtered_structures) >= number_of_mds, (
        f"Not enough structures to select {number_of_mds} from. Available: {len(filtered_structures)}"
    )

    if enforce_chemical_diversity:
        # Ensure chemical diversity by selecting unique chemical formulas
        # If there are fewer unique formulas than `number_of_mds`, select all
        # Otherwise, randomly select `number_of_mds` unique formulas
        unique_chemical_formulas = set(
            s.get_chemical_formula() for s in filtered_structures
        )

        if len(unique_chemical_formulas) <= number_of_mds:
            list_of_formulas = list(unique_chemical_formulas)
            extra_formulas = [
                np.random.choice(list(unique_chemical_formulas), replace=False)
                for _ in range(number_of_mds - len(list_of_formulas))
            ]
            list_of_formulas.extend(extra_formulas)

        else:
            list_of_formulas = np.random.choice(
                list(unique_chemical_formulas), number_of_mds, replace=False
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

    else:
        initial_atoms = [
            filtered_structures[x]
            for x in np.random.choice(
                np.array(range(len(filtered_structures))),
                number_of_mds,
                replace=False,
            )
        ]

    for i, atoms in enumerate(initial_atoms):
        atoms.info["job_id"] = i
        atoms.info["config_type"] = f"{base_name}_{job_name}"

    if verbose > 0:
        print(
            f"The following structures were selected for MD: {[a.get_chemical_formula() for a in initial_atoms]}"
        )

    return initial_atoms


def primary_run(
    input_atoms_list,
    out_dir,
    base_mace=0,
    device="cuda",
    steps=100,
    temperature=300,
    md_name="md_run",
    number_of_structures: int = 50,
    timestep_fs: float = 0.5,
    verbose: int = 0,
):
    assert number_of_structures > 0, "Number of structures must be greater than 0"
    assert steps > number_of_structures / len(input_atoms_list), (
        "Number of steps must be greater than the number of structures divided by the number of intended MD runs"
    )
    calc = MACECalculator(
        model_paths=[base_mace],
        device=device,
    )

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    atom_traj_list = []
    for atoms in input_atoms_list:
        initial_structure = atoms.copy()
        initial_structure.calc = calc

        dyn = Langevin(
            atoms=initial_structure,
            timestep=timestep_fs * fs,
            temperature_K=temperature,
            friction=0.002,
            logfile=str(Path(out_dir, f"{md_name}.log")),
        )

        snapshot_interval = steps * len(input_atoms_list) // (number_of_structures * 5)

        dyn.attach(
            lambda: write(
                str(Path(out_dir, f"{md_name}_{initial_structure.info['job_id']}.xyz")),
                dyn.atoms.copy(),
                append=True,
            ),
            interval=snapshot_interval,
        )
        dyn.attach(
            lambda: atom_traj_list.append(dyn.atoms.copy()), interval=snapshot_interval
        )
        dyn.run(steps=steps)
    if verbose > 0:
        print(f"MD run {md_name} completed, {len(atom_traj_list)} structures generated.")
    
    


def flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0] * 3))


def std_deviation_of_forces(
    structure_forces_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    md_dir,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Calculate the standard deviation of forces for each structure in the dictionary.

    Parameters
    ----------
    structure_force_dict : dict
        A dictionary where keys are fit names and values are dictionaries with structure names as keys and forces as values.

        e.g.:
        {
            'base_mace': {
                'structure_0': {'forces': np.ndarray, 'energy': float},
                'structure_1': {'forces': np.ndarray, 'energy': float},
                ...
            },
            'fit_1': {
                ...
            },
        }

    Returns
    -------
    list
        A list of standard deviations of forces for each structure.
    """
    number_of_structures = len(structure_forces_dict["base_mace"])
    std_dev_array = np.zeros((number_of_structures, 3))
    for structure in range(number_of_structures):
        forces_array = np.concatenate(
            [
                structure_forces_dict[fit][f"structure_{structure}"]["forces"]
                for fit in structure_forces_dict
            ],
            axis=0,
        )
        std_dev_per_force_fragment = np.std(forces_array, axis=0)
        energy_array = np.array(
            [
                structure_forces_dict[fit][f"structure_{structure}"]["energy"]
                for fit in structure_forces_dict
            ]
        )
        std_dev_per_energy = np.std(energy_array)

        if verbose > 0:
            print(
                f"Structure {structure}, max std dev: {np.max(std_dev_per_force_fragment)}, mean std dev: {np.mean(std_dev_per_force_fragment)}, std dev of energy: {std_dev_per_energy}, energies: {energy_array}"
            )

        std_dev_array[structure, :] = np.array(
            [
                np.max(std_dev_per_force_fragment),
                np.mean(std_dev_per_force_fragment),
                std_dev_per_energy,
            ]
        )

    df = pd.DataFrame(
        std_dev_array, columns=["max_std_dev", "mean_std_dev", "std_dev_energy"]
    ).sort_values(by="max_std_dev", ascending=False)

    df.to_csv(str(Path(md_dir, "std_dev_forces.csv")), index=True)

    return df


def get_forces_for_all_maces(
    structure_list: list[Atoms],
    base_name: str,
    job_dict: dict[str, dict[str, str]],
    base_mace: str,
    fits_to_use: list[int] = [0],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Get forces for all MACE models specified in fits_to_use.
    """
    calc = MACECalculator(model_paths=base_mace, device="cpu")

    for atoms in structure_list:
        atoms.calc = calc
    structure_forces_dict = {
        "base_mace": {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(structure_list[i].get_forces()),
                "energy": np.array(structure_list[i].get_potential_energy()),
            }
            for i in range(len(structure_list))
        }
    }

    for i in fits_to_use:
        calc = MACECalculator(
            model_paths=str(
                Path(
                    "results",
                    base_name,
                    f"MACE/fit_{i}/{job_dict['mace_committee']['name']}_stagetwo.model",
                )
            ),
            device="cpu",
            default_dtype="float64",
        )
        for atoms in tqdm(structure_list):
            atoms.calc = calc
        structure_forces_dict[f"fit_{i}"] = {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(structure_list[i].get_forces()),
                "energy": structure_list[i].get_potential_energy(),
            }
            for i in range(len(structure_list))
        }

    return structure_forces_dict
