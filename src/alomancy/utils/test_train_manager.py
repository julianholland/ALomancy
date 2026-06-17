from pathlib import Path
from warnings import warn
import logging

from ase import Atoms
from ase.io import read
import numpy as np
from alomancy.utils.clean_structures import clean_structures
from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

logger = logging.getLogger(__name__)


def split_atoms_list_into_test_and_train(
    atoms_list: list[Atoms], test_fraction: float, seed: int
) -> tuple[list[Atoms], list[Atoms]]:
    """
    Split a list of Atoms objects into training and test sets.

    Args:
        atoms_list (list[Atoms]): List of Atoms objects to split.
        test_fraction (float): Fraction of the data to use for the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[list[Atoms], list[Atoms]]: A tuple containing the training and test sets.
    """
    
    rng = np.random.default_rng(seed=seed)
    shuffled_indices = rng.permutation(len(atoms_list))
    split_index = int(len(atoms_list) * (1 - test_fraction))
    train_indices = shuffled_indices[:split_index]
    test_indices = shuffled_indices[split_index:]

    train_set = [atoms_list[i] for i in train_indices]
    test_set = [atoms_list[i] for i in test_indices]

    return train_set, test_set

def extend_test_and_train_sets_with_extra_dataset(
    extra_dataset: str | Path,
    train_xyzs: list[Atoms],
    test_xyzs: list[Atoms],
    test_fraction: float,
    seed: int,
    filter_out_config_types: list[str] | None = ["IsolatedAtom"],
    fall_back_config_type: None | str = None,
) -> tuple[list[Atoms], list[Atoms]]:

    extra_dataset_atoms = [
        a
        for a in read_atoms_file_if_enabled(True, extra_dataset)
        if filter_out_config_types is None
        or a.info.get("config_type") not in filter_out_config_types
    ]

    if fall_back_config_type is None:
        fall_back_config_type = f"undefined_from_{Path(extra_dataset).name}"
    
    if extra_dataset_atoms is not None:
        extra_dataset_atoms = clean_structures(
            extra_dataset_atoms,
            fall_back_config_type,
            override_config_type=False,
            already_computed=True,
        )
        
        inelegible_configs= ["IsolatedAtom"]
        elegible_extra_dataset_atoms = [
            a for a in extra_dataset_atoms if a.info.get("config_type") not in inelegible_configs
        ]
        extra_dataset_train, extra_dataset_test = split_atoms_list_into_test_and_train(
            elegible_extra_dataset_atoms, test_fraction, seed
        )

        train_xyzs.extend(extra_dataset_train + [a for a in extra_dataset_atoms if a.info.get("config_type") in inelegible_configs])
        test_xyzs.extend(extra_dataset_test)
        logger.info("Added %d structures from %s to training set and %d to test set.", len(extra_dataset_train), extra_dataset, len(extra_dataset_test))
        logger.warning("Remove %s from extra_datasets to avoid duplicates upon restart.", extra_dataset)
    else:
        logger.warning("Could not read dataset from %s. Check path and format.", extra_dataset)

    return train_xyzs, test_xyzs


def add_new_training_data(
    base_name: str,
    high_accuracy_eval_job_dict: dict,
    train_xyzs: list[Atoms],
):
    """
    Add new training data from DFT calculations to the existing training data.

    Args:
        base_name (str): Base name for the job.
        high_accuracy_eval_job_dict (dict): Dictionary containing job names for different runs.
    """
    path_list = list(
        Path.glob(
            Path("results", base_name, high_accuracy_eval_job_dict["name"]),
            f"{high_accuracy_eval_job_dict['name']}_*_out_structures.xyz",
        )
    )
    new_dft_structures = []

    for path in path_list:
        new_dft_structures.extend(
            read(
                str(path),
                ":",
            )
        )

    return train_xyzs + new_dft_structures
