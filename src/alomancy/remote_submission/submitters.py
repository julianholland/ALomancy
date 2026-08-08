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

    # output_files is set explicitly per job (keyed by the real n_existing + i
    # directory name) rather than via common_output_pattern's positional
    # job_id: submit_multiple_jobs derives job_id from each job's *position*
    # in job_configs (0..len-1), which only matches n_existing + i when
    # n_existing == 0. Whenever some structure-generation runs already exist
    # and get skipped above, that positional index silently diverges from
    # the directory the remote MD job actually writes to (out_dir), causing
    # ExPyRe's stage-out step to glob for the wrong directory and fail with
    # "does not match any files" even though the job succeeded.
    job_configs = [
        {
            "function_kwargs": {
                "initial_structure": atoms,
                "out_dir": str(Path(f"{md_dir}/md_output_{n_existing + i}")),
                **(function_kwargs or {}),
            },
            "output_files": [str(Path(f"{md_dir}/md_output_{n_existing + i}"))],
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
    )

    return find_target_files()


def all_maces_remote_submitter(
    remote_info: RemoteInfo,
    function: Callable | None = None,
    function_kwargs: dict[str, Any] | None = None,
    job_name: str | None = None,
) -> dict:
    """Submit the single job that evaluates every committee MACE model on a
    batch of candidate structures (the "mace evaluation" step of
    structure_generation), producing the per-structure force std-dev used
    to select high-uncertainty structures for DFT.

    job_name defaults to ``f"mace_eval_{remote_info.job_name}"`` rather than
    the bare ``remote_info.job_name`` used by the MD jobs submitted just
    before this in the same structure_generation phase — without an
    explicit override, both would land in ExPyRe's job state under the
    identical name (job_dict["structure_generation"]["name"]), making them
    indistinguishable in logs and queue listings.
    """
    if job_name is None:
        job_name = f"mace_eval_{remote_info.job_name}"

    n_structures = len((function_kwargs or {}).get("structure_list", []))
    logger.info(
        "Submitting MACE committee evaluation job '%s' to score %d candidate "
        "structure(s) for uncertainty.",
        job_name,
        n_structures,
    )

    executor = RemoteJobExecutor(remote_info)
    job_configs = [
        {"function_kwargs": {**(function_kwargs or {})}, "job_name": job_name}
    ]

    forces_dict = executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
    )[0]

    logger.info("MACE committee evaluation job '%s' finished.", job_name)

    return forces_dict


def committee_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    function: Callable,
    seed: int = 803,
    size_of_committee: int = 5,
    function_kwargs: dict[str, Any] | None = None,
    fit_indices: list[int] | None = None,
) -> None:
    """Submit `size_of_committee` committee training jobs, one per fit index.

    fit_indices
        Explicit committee-member indices to (re)train, e.g. to backfill only
        the specific fits missing from a prior partial run. Defaults to
        ``range(size_of_committee)`` (every member, indices 0..N-1).
        Each job's output directory is keyed off its own index (not its
        position in this list) so it matches the `fit_{fit_idx}` directory
        `mace_fit` itself writes to (`mlip/mace_wfl.py`) — using the
        shared `common_output_pattern`/positional `job_id` mechanism here
        would stage/sync the wrong directory whenever fit_indices is a
        non-contiguous subset (e.g. backfilling just fit_2 and fit_4).
    """
    mace_dir = Path("results", base_name)
    mace_dir.mkdir(exist_ok=True, parents=True)

    executor = RemoteJobExecutor(remote_info)

    indices = fit_indices if fit_indices is not None else list(range(size_of_committee))

    job_configs = [
        {
            "function_kwargs": {
                "seed": seed + i,
                "fit_idx": i,
                **(function_kwargs or {}),
            },
            "output_files": [str(Path(mace_dir, "mlip_committee", f"fit_{i}"))],
        }
        for i in indices
    ]

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
    )
