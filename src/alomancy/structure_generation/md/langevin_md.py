from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from ase.md.langevin import Langevin
from ase.units import fs
from mace.calculators import MACECalculator


def run_langevin_md(
    structure_generation_job_dict: dict,
    initial_structure: Atoms,
    out_dir,
    model_path,
    steps=100,
    temperature=300,
    timestep_fs: float = 0.5,
    friction: float = 0.002,
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
        friction=friction,
        logfile=str(
            Path(
                out_dir,
                f"{structure_generation_job_dict['name']}_{md_structure.info['job_id']}.log",
            )
        ),
    )

    snapshot_interval = (
        steps  # total steps to run the calculation for
        * concurrent_runs  # times total number of concurrent runs
        // structure_generation_job_dict[
            "desired_number_of_structures"
        ]  # to get the total number of structures wanted
        // 5  # to get 5 times the number of structures we want to filter later
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

