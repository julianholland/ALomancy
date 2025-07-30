from ase.md.langevin import Langevin
from mace.calculators import MACECalculator
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.units import fs
import numpy as np
import pandas as pd
from pathlib import Path


def primary_run(
    input_atoms_list,
    out_dir,
    base_mace=0,
    device="cuda",
    steps=100,
    temperature=300,
    md_name="md_run",
):

    calc = MACECalculator(
        model_paths=[str(Path(f"MACE_model/fit_{base_mace}/test_stagetwo.model"))],
        device=device,
    )

    Path.mkdir(out_dir, exist_ok=True)

    atom_traj_list = []
    for atoms in input_atoms_list:
        initial_structure = atoms.copy()
        initial_structure.calc = calc

        dyn = Langevin(
            atoms=initial_structure,
            timestep=0.5 * fs,
            temperature_K=temperature,
            friction=0.002,
            logfile=str(Path(out_dir, f"{md_name}.log")),
        )

        dyn.attach(lambda: write(f'{md_name}.xyz', dyn.atoms.copy(), append=True), interval=50)
        dyn.attach(lambda: atom_traj_list.append(dyn.atoms.copy()), interval=50)
        dyn.run(steps=steps)
    print(f'MD run completed, {len(atom_traj_list)} structures generated.')
    return atom_traj_list

def flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0] * 3))


def std_deviation_of_forces(
    structure_forces_dict: dict[dict[dict[np.ndarray[float], float]]], verbose: int = 0
) -> list[float]:
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

    return pd.DataFrame(
        std_dev_array, columns=["max_std_dev", "mean_std_dev", "std_dev_energy"]
    ).sort_values(by="max_std_dev", ascending=False)


def get_forces_for_all_maces(
    structure_list, base_mace=0, fits_to_use=[0]
) -> np.ndarray:
    calc = MACECalculator(
        model_paths=f"MACE_model/fit_{base_mace}/test_stagetwo.model", device="cpu"
    )

    for atoms in structure_list:
        atoms.calc = calc
    structure_forces_dict = {
        "base_mace": {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(structure_list[i].get_forces()),
                "energy": structure_list[i].get_potential_energy(),
            }
            for i in range(len(structure_list))
        }
    }

    for i in fits_to_use:
        calc = MACECalculator(
            model_paths=f"MACE_model/fit_{i}/test_stagetwo.model",
            device="cpu",
            default_dtype="float64",
        )
        for atoms in structure_list:
            atoms.calc = calc
        structure_forces_dict[f"fit_{i}"] = {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(structure_list[i].get_forces()),
                "energy": structure_list[i].get_potential_energy(),
            }
            for i in range(len(structure_list))
        }

    return structure_forces_dict