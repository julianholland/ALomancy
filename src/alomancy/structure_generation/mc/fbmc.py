from pathlib import Path

from ase.atoms import Atoms
from ase.constraints import FixCom
from ase.io import read
from mace.calculators import MACECalculator
from quansino.mc.fbmc import ForceBias
from quansino.constraints import FixRot

def run_fbmc(
    structure_generation_job_dict: dict,
    input_structure: Atoms,
    out_dir: str,
    model_path: str,
    delta: float = 0.2,
    temperature: float = 298.15,
    steps: int = 100,
    verbose: int = 0,
):
    concurrent_runs = structure_generation_job_dict["structure_selection_kwargs"][
        "max_number_of_concurrent_jobs"
    ]

    assert (
        structure_generation_job_dict["desired_number_of_structures"] > 0
    ), "Number of structures must be greater than 0"
    assert (
        steps
        > structure_generation_job_dict["desired_number_of_structures"]
        / concurrent_runs
    ), "Number of steps must be greater than the number of structures divided by the number of intended MD runs"
    assert (
        steps
        * concurrent_runs
        // structure_generation_job_dict["desired_number_of_structures"]
        * 5
    ) > 0, "Snapshot interval must be greater than 0"

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    snapshot_interval = 1
    # (
    #     steps  # total steps to run the calculation for
    #     * concurrent_runs  # to divide the total number of structures by the number of concurrent runs
    #     // structure_generation_job_dict[
    #         "desired_number_of_structures"
    #     ]  # to return 5 times the number of structures we want
    #     // 5  # to get the top 20% of structures
    # )
    if verbose > 0:
        print(
            f"Snapshot interval set to {snapshot_interval} steps for {structure_generation_job_dict['name']}."
        )
    device = "cuda" if structure_generation_job_dict["hpc"]["gpu"] else "cpu"

    fbmc_structure = input_structure.copy()
    fbmc_structure.calc = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype="float64",
    )
    fbmc_structure.set_constraint([FixCom(), FixRot()])
    fbmc_object = ForceBias(
        fbmc_structure,
        temperature=temperature,
        delta=delta,
        trajectory=str(Path(out_dir, f"{structure_generation_job_dict['name']}.xyz")),
        logfile=str(Path(out_dir, f"{structure_generation_job_dict['name']}.log")),
        logging_interval=snapshot_interval,
        logging_mode="w"
    )

    fbmc_object.run(steps)

    atom_traj_list = read(
        str(Path(out_dir, f"{structure_generation_job_dict['name']}.xyz")), ":"
    )
    if verbose > 0:
        print(
            f"{structure_generation_job_dict['name']} run completed, {len(atom_traj_list)} structures generated."
        )


