import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

logger = logging.getLogger(__name__)


def generate_kpts(
    cell: np.ndarray, periodic_3d: bool = True, kspacing: float = 0.1
) -> np.ndarray:
    cell_lengths = np.linalg.norm(cell, axis=1)
    kpts = np.ceil(2 * np.pi / (cell_lengths * kspacing)).astype(int)
    return kpts if periodic_3d else np.array([kpts[0], kpts[1], 1])


def _build_srun_command(para_info_dict: dict, executable_and_flags: str) -> str:
    # --mem=0 means "use all memory already granted to the job" (documented
    # Slurm sentinel), not "request 0 memory". This step runs nested inside
    # a job whose own #SBATCH header may already set --mem; if it were, the
    # enclosing batch script's process itself is accounted as the job's
    # first step and, on some Slurm configs, is credited with the *entire*
    # job memory allocation. A nested srun step that then asks for its own
    # explicit sub-amount (e.g. --mem=60GB) competes with that reservation
    # and fails immediately with "Unable to create step ... Memory required
    # by task is not available", regardless of how large the job's total
    # --mem is. --mem=0 avoids the conflict by inheriting rather than
    # re-requesting.
    return (
        f"srun --ntasks={para_info_dict['ranks_per_system']} "
        f"--tasks-per-node={para_info_dict['ranks_per_node']} "
        f"--cpus-per-task={para_info_dict['threads_per_rank']} "
        f"--distribution=block:block "
        f"--hint=nomultithread "
        f"--mem=0 "
        f"{executable_and_flags}"
    )


def _write_dft_result(atoms: Atoms, out_dir: str, name: str) -> None:
    write(Path(out_dir, f"{name}.xyz"), atoms, format="extxyz")
    logger.debug("Writing structures to %s as %s.xyz", out_dir, name)


def _run_sp(
    input_structure: Atoms,
    out_dir: str,
    job_dict: dict,
    create_calc_fn: Callable,
) -> Atoms:
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    input_structure.calc = create_calc_fn(input_structure, job_dict, out_dir)
    input_structure.get_potential_energy()
    _write_dft_result(input_structure, out_dir, job_dict["name"])
    return input_structure


def _run_go(
    input_structure: Atoms,
    out_dir: str,
    job_dict: dict,
    create_calc_fn: Callable,
    opt_prefix: str = "opt",
) -> Atoms:
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    input_structure.calc = create_calc_fn(input_structure, job_dict, out_dir)
    opt = BFGS(
        input_structure,
        logfile=str(Path(out_dir, f"{opt_prefix}.log")),
        trajectory=str(Path(out_dir, f"{opt_prefix}.traj")),
    )
    opt.run(fmax=0.05, steps=200)
    _write_dft_result(input_structure, out_dir, job_dict["name"])
    return input_structure
