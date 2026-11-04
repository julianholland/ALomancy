import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ase import Atoms

from alomancy.configs.remote_info import RemoteInfo
from alomancy.remote_submission.executor import RemoteJobExecutor

logger = logging.getLogger(__name__)

ASE_OUTPUT_PREFIX = "ase_output"

# VASP keeps file paths in a ~256-character Fortran buffer: a run directory
# of 255 characters left no room for the filename, so VASP tried to open the
# directory itself ("forrtl: severe (30): open failure ... Is a directory").
# 240 leaves room for "/vasprun.xml" and similar. QE has the same kind of
# limit.
MAX_REMOTE_OUTPUT_DIR_LEN = 240

# ExPyRe names each job's stage directory (local and remote)
# f"run_{name}_{arghash}_{suffix}": arghash is the urlsafe base64 of a
# sha256 digest (always 44 characters) and suffix is tempfile.mkdtemp's
# random part (always 8). Only their lengths matter here.
_EXPYRE_ARGHASH_LEN = 44
_EXPYRE_MKDTEMP_SUFFIX_LEN = 8


def dft_job_name(function: Callable, index: int) -> str:
    """Short ExPyRe job name for one DFT structure, e.g. "sp_vasp_0".

    ExPyRe adds its own "run_" prefix, so the function's is dropped, and the
    section name ("high_accuracy_evaluation") is left out: every character
    here ends up in the remote run directory (MAX_REMOTE_OUTPUT_DIR_LEN).
    """
    return f"{function.__name__.removeprefix('run_')}_{index}"


def projected_remote_output_dir(remote_rundir: str, job_name: str, out_dir: str) -> str:
    """The remote directory a job's *out_dir* (relative to the job's stage
    directory) will have, with placeholder characters standing in for
    ExPyRe's hash and random suffix. Exact in length, which is what
    check_remote_path_length needs."""
    stage_dir_name = (
        f"run_{job_name}_{'#' * _EXPYRE_ARGHASH_LEN}_{'#' * _EXPYRE_MKDTEMP_SUFFIX_LEN}"
    )
    return f"{remote_rundir.rstrip('/')}/{stage_dir_name}/{out_dir.strip('/')}/"


def check_remote_path_length(path: str, *, remote_rundir: str, context: str) -> None:
    """Raise ValueError if *path* (from projected_remote_output_dir) is too
    long for VASP/QE to open files in."""
    if len(path) <= MAX_REMOTE_OUTPUT_DIR_LEN:
        return
    note = (
        ""
        if remote_rundir.startswith("/")
        else " The remote rundir is relative to the remote $HOME, which is "
        "not counted, so the real path is longer still."
    )
    raise ValueError(
        f"{context}: the remote DFT run directory would be {len(path)} "
        f"characters, over the {MAX_REMOTE_OUTPUT_DIR_LEN} that VASP/QE can "
        f"handle (they fail with 'open failure ... Is a directory').{note}\n"
        f"  {path}\n"
        f"The first part, {remote_rundir!r}, is the HPC rundir (in "
        "~/.alomancy/expyre_config.json, set by `alomancy add-hpc`) plus this "
        "run's directory below $HOME. Shorten the rundir, or run from a "
        "directory with a shorter path."
    )


def expyre_remote_rundir(sys_name: str) -> str | None:
    """The remote directory ExPyRe puts *sys_name*'s jobs in, or None if
    ExPyRe has no such system (it then fails on its own at submission) or
    runs it locally."""
    from expyre import config as expyre_config

    system = (expyre_config.systems or {}).get(sys_name)
    if system is None or system.remote_rundir is None:
        return None
    return str(system.remote_rundir)


def _noop(**_kwargs: Any) -> None:
    logger.warning("No function provided for remote execution. This is a no-op.")
    return None


def ase_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    input_atoms_list: list[Atoms],
    function: Callable | None = None,
    per_structure_function: list[Callable] | None = None,
    batch: int = 0,
    function_kwargs: dict[str, Any] | None = None,
) -> None:
    """Submit one job per structure in input_atoms_list to a shared
    RemoteJobExecutor pool (bounded by remote_info.max_num_of_concurrent_jobs).

    per_structure_function, when given, must be the same length as
    input_atoms_list and picks which function each structure's job runs
    (e.g. a mix of geometry-optimisation and single-point evaluation jobs
    sharing one submission queue instead of running as separate batches).
    When omitted, every structure uses the single shared `function`.
    Every job is named by dft_job_name ("go_vasp_3", "sp_qe_0"), and its
    remote run directory is checked against MAX_REMOTE_OUTPUT_DIR_LEN
    before anything is submitted.
    """
    if per_structure_function is not None and len(per_structure_function) != len(
        input_atoms_list
    ):
        raise ValueError(
            "per_structure_function must be the same length as input_atoms_list "
            f"({len(per_structure_function)} != {len(input_atoms_list)})"
        )

    ase_dir = Path("results", base_name, "high_accuracy_evaluation", f"batch_{batch}")
    remote_rundir = expyre_remote_rundir(remote_info.sys_name)

    job_configs = []
    for i, atoms in enumerate(input_atoms_list):
        job_function = (
            per_structure_function[i] if per_structure_function is not None else None
        ) or function
        out_dir = str(Path(f"{ase_dir}/{ASE_OUTPUT_PREFIX}_{i}"))
        job_config: dict[str, Any] = {
            "function_kwargs": {
                "input_structure": atoms,
                "out_dir": out_dir,
                **(function_kwargs or {}),
            },
            # The structure this job labels, for the loop report's retry
            # bookkeeping (analysis/report/current_events.py).
            "item": f"structure_{atoms.info.get('job_id', f'{batch}_{i}')}",
        }
        if per_structure_function is not None:
            job_config["function"] = job_function
        if job_function is not None:
            job_config["job_name"] = dft_job_name(job_function, i)
            # Checked for every job before any is submitted, so a run whose
            # paths are too long fails here rather than in the queue.
            if remote_rundir is not None:
                check_remote_path_length(
                    projected_remote_output_dir(
                        remote_rundir, job_config["job_name"], out_dir
                    ),
                    remote_rundir=remote_rundir,
                    context=f"DFT job {job_config['job_name']}",
                )
        job_configs.append(job_config)

    ase_dir.mkdir(exist_ok=True, parents=True)
    executor = RemoteJobExecutor(remote_info)

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
        common_output_pattern=str(Path(ase_dir, ASE_OUTPUT_PREFIX + "_{job_id}")),
    )
