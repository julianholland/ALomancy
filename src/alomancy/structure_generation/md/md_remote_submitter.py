from pathlib import Path
from wfl.autoparallelize.remoteinfo import RemoteInfo
from typing import List, Callable
from alomancy.utils.remote_job_executor import RemoteJobExecutor
from ase import Atoms


def md_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    target_file: str,
    input_atoms_list: List[Atoms],
    function: Callable = None,
    function_kwargs=None,
) -> List[str]:
    workdir = Path("results", base_name)
    md_dir = Path(workdir, "structure_generation")

    def find_target_files():
        return list(Path.glob(md_dir, f"md_output_*/{target_file}"))

    target_file_list = find_target_files()

    if len(target_file_list) >= len(input_atoms_list):
        print(
            f"All {len(input_atoms_list)} structure generation runs finished. Skipping submission."
        )
        return target_file_list

    elif len(target_file_list) != 0:
        print(
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

    print([job_config["function_kwargs"]["out_dir"] for job_config in job_configs])

    executor.run_and_wait(
        function=function,
        job_configs=job_configs,
        common_output_pattern=str(Path(md_dir, "md_output_{job_id}")),
    )

    return find_target_files()
