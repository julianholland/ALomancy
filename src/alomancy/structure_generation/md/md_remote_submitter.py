import logging
from pathlib import Path
from typing import Any, Callable

from ase import Atoms
from alomancy.configs.remote_info import RemoteInfo

from alomancy.utils.remote_job_executor import RemoteJobExecutor

logger = logging.getLogger(__name__)


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

    if len(target_file_list) >= len(input_atoms_list):
        logger.info(
            f"All {len(input_atoms_list)} structure generation runs finished. Skipping submission."
        )
        return target_file_list

    elif len(target_file_list) != 0:
        logger.info(
            f"Found {len(target_file_list)} existing structure generation runs. Reusing them."
        )
        input_atoms_list = input_atoms_list[len(target_file_list) :]

    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {
            "function_kwargs": {
                "initial_structure": input_atoms_list[i],
                "out_dir": str(Path(f"{md_dir}/md_output_{i}")),
                **function_kwargs,
            }
        }
        for i in range(len(input_atoms_list))
    ]

    logger.debug("MD output directories: %s", [job_config["function_kwargs"]["out_dir"] for job_config in job_configs])

    executor.run_and_wait(
        function=function,
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
        job_configs = [
            {
                "function_kwargs": {
                    **function_kwargs,
                }
            }
        ]

        forces_dict= executor.run_and_wait(
            function=function,
            job_configs=job_configs,
        )[0]

        return forces_dict