from pathlib import Path
from ase.io import read
from ase import Atoms


def add_new_training_data(
    base_name: str,
    job_dict: dict,
    train_xyzs: list[Atoms],
):
    """
    Add new training data from DFT calculations to the existing training data.

    Args:
        base_name (str): Base name for the job.
        job_name_dict (dict): Dictionary containing job names for different runs.
    """

    new_dft_structures = list(
        read(
            str(
                Path(
                    "results",
                    base_name,
                    "DFT",
                    f"{job_dict['dft_run']['name']}_out_structures.xyz",
                )
            ),
            ":",
        )
    )

    return train_xyzs + new_dft_structures
