from pathlib import Path

from ase import Atoms
from ase.io import read


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
