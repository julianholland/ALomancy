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


def run_md(
    structure_generation_job_dict: dict,
    initial_structure: Atoms,
    total_md_runs: int,
    out_dir,
    model_path,
    steps=100,
    temperature=300,
    timestep_fs: float = 0.5,
    verbose: int = 0,
):
    assert structure_generation_job_dict["desired_number_of_structures"] > 0, (
        "Number of structures must be greater than 0"
    )
    assert (
        steps
        > structure_generation_job_dict["desired_number_of_structures"] / total_md_runs
    ), (
        "Number of steps must be greater than the number of structures divided by the number of intended MD runs"
    )
    # further asserting needed here to avoid:
    # for i in range(steps // snapshot_interval):
    #                ~~~~~~^^~~~~~~~~~~~~~~~~~~
    # ZeroDivisionError: integer division or modulo by zero

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    atom_traj_list = []

    md_structure = initial_structure.copy()
    md_structure.calc = MACECalculator(
        model_paths=model_path,
        device="cuda",
        default_dtype="float64",
    )

    dyn = Langevin(
        atoms=md_structure,
        timestep=timestep_fs * fs,
        temperature_K=temperature,
        friction=0.002,
        logfile=str(
            Path(
                out_dir,
                f"{structure_generation_job_dict['name']}_{md_structure.info['job_id']}.log",
            )
        ),
    )

    snapshot_interval = (
        steps
        * total_md_runs
        // structure_generation_job_dict["desired_number_of_structures"]
    )

    for _ in range(steps // snapshot_interval):
        # recording
        write(
            str(Path(out_dir, f"{structure_generation_job_dict['name']}.xyz")),
            dyn.atoms.copy(),
            append=True,
        )
        atom_traj_list.append(dyn.atoms.copy())

        # force check
        max_forces = np.max(np.abs(dyn.atoms.get_forces()), axis=0)
        if np.any(max_forces > 1000):
            print(
                f"Stopping MD run {structure_generation_job_dict['name']} due to excessive forces: {max_forces}"
            )
            break
        # run
        dyn.run(steps=snapshot_interval)

    if verbose > 0:
        print(
            f"MD run {structure_generation_job_dict['name']} completed, {len(atom_traj_list)} structures generated."
        )


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
