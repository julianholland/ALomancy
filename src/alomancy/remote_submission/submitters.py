import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ase import Atoms

from alomancy.configs.remote_info import RemoteInfo
from alomancy.remote_submission.executor import RemoteJobExecutor

logger = logging.getLogger(__name__)

ASE_OUTPUT_PREFIX = "ase_output"


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
    RemoteJobExecutor pool (bounded by remote_info.max_concurrent_jobs).

    per_structure_function, when given, must be the same length as
    input_atoms_list and picks which function each structure's job runs
    (e.g. a mix of geometry-optimisation and single-point evaluation jobs
    sharing one submission queue instead of running as separate batches).
    Each such job also gets a job name prefixed with the function's
    __name__ so the two kinds are distinguishable in logs/ExPyRe state.
    When omitted, every structure uses the single shared `function`,
    unchanged from before.
    """
    if per_structure_function is not None and len(per_structure_function) != len(
        input_atoms_list
    ):
        raise ValueError(
            "per_structure_function must be the same length as input_atoms_list "
            f"({len(per_structure_function)} != {len(input_atoms_list)})"
        )

    ase_dir = Path("results", base_name, "high_accuracy_evaluation", f"batch_{batch}")
    ase_dir.mkdir(exist_ok=True, parents=True)
    executor = RemoteJobExecutor(remote_info)

    job_configs = []
    for i, atoms in enumerate(input_atoms_list):
        job_config: dict[str, Any] = {
            "function_kwargs": {
                "input_structure": atoms,
                "out_dir": str(Path(f"{ase_dir}/{ASE_OUTPUT_PREFIX}_{i}")),
                **(function_kwargs or {}),
            }
        }
        if per_structure_function is not None:
            job_function = per_structure_function[i]
            job_config["function"] = job_function
            job_config["job_name"] = (
                f"{job_function.__name__}_{remote_info.job_name}_{i}"
            )
        job_configs.append(job_config)

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
        common_output_pattern=str(Path(ase_dir, ASE_OUTPUT_PREFIX + "_{job_id}")),
    )


def md_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    target_file: str,
    input_atoms_list: list[Atoms],
    function: Callable | None = None,
    function_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    workdir = Path("results", base_name)
    md_dir = Path(workdir, "structure_generation")

    def find_target_files():
        return list(Path.glob(md_dir, f"md_output_*/{target_file}"))

    target_file_list = find_target_files()
    n_existing = len(target_file_list)

    if n_existing >= len(input_atoms_list):
        logger.info(
            "All %d structure generation runs finished. Skipping submission.",
            len(input_atoms_list),
        )
        return target_file_list

    elif target_file_list:
        logger.info(
            "Found %d existing structure generation runs. Reusing them.",
            n_existing,
        )
        input_atoms_list = input_atoms_list[n_existing:]

    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {
            "function_kwargs": {
                "initial_structure": atoms,
                "out_dir": str(Path(f"{md_dir}/md_output_{n_existing + i}")),
                **(function_kwargs or {}),
            }
        }
        for i, atoms in enumerate(input_atoms_list)
    ]

    logger.debug(
        "MD output directories: %s",
        [job_config["function_kwargs"]["out_dir"] for job_config in job_configs],
    )

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
        common_output_pattern=str(Path(md_dir, "md_output_{job_id}")),
    )

    return find_target_files()


def all_maces_remote_submitter(
    remote_info: RemoteInfo,
    function: Callable | None = None,
    function_kwargs: dict[str, Any] | None = None,
) -> dict:
    executor = RemoteJobExecutor(remote_info)
    job_configs = [{"function_kwargs": {**(function_kwargs or {})}}]

    forces_dict = executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
    )[0]

    return forces_dict


def committee_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    function: Callable,
    seed: int = 803,
    size_of_committee: int = 5,
    function_kwargs: dict[str, Any] | None = None,
) -> None:
    mace_dir = Path("results", base_name)
    mace_dir.mkdir(exist_ok=True, parents=True)

    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {"function_kwargs": {"seed": seed + i, "fit_idx": i, **(function_kwargs or {})}}
        for i in range(size_of_committee)
    ]

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
        common_output_pattern=str(Path(mace_dir, "mlip_committee", "fit_{job_id}")),
    )
