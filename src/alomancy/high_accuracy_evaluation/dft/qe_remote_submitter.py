from pathlib import Path
from typing import Any, Callable

from ase import Atoms

from alomancy.configs.remote_info import RemoteInfo
from alomancy.utils.remote_job_executor import RemoteJobExecutor


def qe_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    input_atoms_list: list[Atoms],
    function: Callable | None = None,
    function_kwargs: dict[str, Any] | None = None,
) -> None:
    # actual path of structure should be results/base_name/qe_output_i/target_file
    qe_dir = Path("results", base_name)
    qe_dir.mkdir(exist_ok=True, parents=True)
    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {
            "function_kwargs": {
                "input_structure": input_atoms_list[i],
                "out_dir": str(Path(f"{qe_dir}/qe_output_{i}")),
                **(function_kwargs or {}),
            }
        }
        for i in range(len(input_atoms_list))
    ]

    # run_and_wait expects a callable; provide a no-op if None was passed
    def _noop(**_kwargs: Any) -> None:
        print("No function provided for remote execution. This is a no-op.")
        return None

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
        common_output_pattern=str(Path(qe_dir, "qe_output_{job_id}")), # f"{remote_info.job_name}.xyz for just the file
    )


    # if not find_target_files() and results:
    #     output_name = Path(target_file).name
    #     for job_id, result in enumerate(results):
    #         if isinstance(result, Atoms):
    #             local_output_dir = Path(qe_dir, f"qe_output_{job_id}")
    #             local_output_dir.mkdir(exist_ok=True, parents=True)
    #             write(
    #                 Path(local_output_dir, output_name),
    #                 result,
    #                 format="extxyz",
    #             )

    # print('find_target_files(): after execution', find_target_files())
    # return find_target_files()
