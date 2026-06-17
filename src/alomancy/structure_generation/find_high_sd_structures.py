import logging
from pathlib import Path

import numpy as np
import polars as pl
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_high_sd_structures(
    structure_list: list[Atoms],
    base_name: str,
    job_dict: dict[str, dict[str, str]],
    structure_forces_dict: dict,
    read_xyz: bool = True,
) -> list[Atoms]:
    desired_structures = job_dict["structure_generation"][
        "desired_number_of_structures"
    ]

    assert desired_structures > 0, "Number of structures must be greater than 0"
    assert (
        len(structure_list) >= desired_structures
    ), f"Not enough structures to select {desired_structures} from. Available: {len(structure_list)}"

    std_dev_csv_name = Path(
        "results",
        base_name,
        job_dict["structure_generation"]["name"],
        "std_dev_forces.csv",
    )
    high_sd_structures_xyz_name = Path(
        "results",
        base_name,
        job_dict["structure_generation"]["name"],
        "high_sd_structures.xyz",
    )

    if (
        read_xyz
        and Path.exists(std_dev_csv_name)
        and Path.exists(high_sd_structures_xyz_name)
    ):
        std_dev_df = pl.read_csv(std_dev_csv_name, ignore_errors=True)
        high_sd_structures = read(high_sd_structures_xyz_name, ":", format="extxyz")

    else:
        std_dev_df = std_deviation_of_forces(
            structure_forces_dict=structure_forces_dict,
            structure_generation_dir=Path(
                "results", base_name, job_dict["structure_generation"]["name"]
            ),
        )
        index_list = std_dev_df["structure_index"][:desired_structures].to_list()
        high_sd_structures = [structure_list[i] for i in index_list]

        for structure in high_sd_structures:
            write(
                str(
                    Path(
                        "results",
                        base_name,
                        job_dict["structure_generation"]["name"],
                        "high_sd_structures.xyz",
                    )
                ),
                structure.copy(),
                append=True,
            )

    logger.info(
        "Selected %d structures for DFT calculations based on force std dev.",
        len(high_sd_structures),
    )
    logger.debug("Std dev top structures:\n%s", std_dev_df[:desired_structures])
    logger.debug("Total mean std dev: %s", std_dev_df["mean_std_dev"].mean())

    return high_sd_structures


def find_sd_of_all_structures(
    structure_list: list[Atoms],
    base_name: str,
    job_dict: dict[str, dict[str, str]],
    list_of_other_calculators: list,
    forces_name: str = "REF_forces",
    energy_name: str = "REF_energy",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    structure_forces_dict = {
        "base_mlip": {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(
                    structure_list[i].arrays[forces_name]
                ),
                "energy": np.array(structure_list[i].info[energy_name]),
            }
            for i in range(len(structure_list))
        }
    }

    for i in range(len(list_of_other_calculators)):
        calculator = list_of_other_calculators[i]
        for atoms in tqdm(structure_list):
            atoms.calc = calculator

        structure_forces_dict[f"fit_{i}"] = {
            f"structure_{j}": {
                "forces": flatten_array_of_forces(structure_list[j].get_forces()),
                "energy": structure_list[j].get_potential_energy(),
            }
            for j in range(len(structure_list))
        }

    return std_deviation_of_forces(
        structure_forces_dict=structure_forces_dict,
        structure_generation_dir=Path(
            "results", base_name, job_dict["structure_generation"]["name"]
        ),
    )


def flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0] * 3))


# def std_deviation_of_forces(
#     structure_forces_dict: dict[str, dict[str, dict[str, np.ndarray]]],
#     structure_generation_dir: Path,
#     verbose: int = 0,
# ) -> pd.DataFrame:
#     """
#     Calculate the standard deviation of forces for each structure in the dictionary.

#     Parameters
#     ----------
#     structure_force_dict : dict
#         A dictionary where keys are fit names and values are dictionaries with structure names as keys and forces as values.

#         e.g.:
#         {
#             'base_mace': {
#                 'structure_0': {'forces': np.ndarray, 'energy': float},
#                 'structure_1': {'forces': np.ndarray, 'energy': float},
#                 ...
#             },
#             'fit_1': {
#                 ...
#             },
#         }

#     Returns
#     -------
#     list
#         A list of standard deviations of forces for each structure.
#     """
#     number_of_structures = len(structure_forces_dict["base_mlip"])
#     std_dev_array = np.zeros((number_of_structures, 3))
#     for structure in range(number_of_structures):
#         forces_array = np.concatenate(
#             [
#                 structure_forces_dict[fit][f"structure_{structure}"]["forces"]
#                 for fit in structure_forces_dict
#             ],
#             axis=0,
#         )
#         std_dev_per_force_fragment = np.std(forces_array, axis=0)
#         energy_array = np.array(
#             [
#                 structure_forces_dict[fit][f"structure_{structure}"]["energy"]
#                 for fit in structure_forces_dict
#             ]
#         )
#         std_dev_per_energy = np.std(energy_array)

#         if verbose > 0:
#             print(
#                 f"Structure {structure}, max std dev: {np.max(std_dev_per_force_fragment)}, mean std dev: {np.mean(std_dev_per_force_fragment)}, std dev of energy: {std_dev_per_energy}, energies: {energy_array}"
#             )

#         std_dev_array[structure, :] = np.array(
#             [
#                 np.max(std_dev_per_force_fragment),
#                 np.mean(std_dev_per_force_fragment),
#                 std_dev_per_energy,
#             ]
#         )

#     df = pd.DataFrame(
#         std_dev_array, columns=["max_std_dev", "mean_std_dev", "std_dev_energy"]
#     ).sort_values(by="max_std_dev", ascending=False)

#     df.to_csv(str(Path(structure_generation_dir, "std_dev_forces.csv")), index=True)

#     return df



def std_deviation_of_forces(
    structure_forces_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    structure_generation_dir: Path,
) -> pl.DataFrame:
    """
    Calculate the standard deviation of forces for each structure in the dictionary.
    """
    logger.debug("Calculators in forces dict: %s", list(structure_forces_dict.keys()))
    number_of_structures = len(structure_forces_dict["base_mlip"])
    rows = []

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

        logger.debug(
            "Structure %d: max_std_dev=%.4f, mean_std_dev=%.4f, std_dev_energy=%.4f",
            structure,
            float(np.max(std_dev_per_force_fragment)),
            float(np.mean(std_dev_per_force_fragment)),
            float(std_dev_per_energy),
        )

        rows.append(
            {
                "structure_index": structure,
                "max_std_dev": float(np.max(std_dev_per_force_fragment)),
                "mean_std_dev": float(np.mean(std_dev_per_force_fragment)),
                "std_dev_energy": float(std_dev_per_energy),
            }
        )

    df = pl.DataFrame(rows).sort("max_std_dev", descending=True)
    df.write_csv(Path(structure_generation_dir) / "std_dev_forces.csv")
    logger.debug("Std dev DataFrame:\n%s", df)

    return df
