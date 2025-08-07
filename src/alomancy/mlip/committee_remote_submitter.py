from pathlib import Path
from wfl.autoparallelize.remoteinfo import RemoteInfo
from typing import List, Callable
from alomancy.utils.remote_job_executor import RemoteJobExecutor


def committee_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    target_file: str,
    seed: int = 803,
    size_of_committee: int = 5,
    function: Callable = None,
    function_kwargs=None,
)-> List[str]:
    workdir = Path("results", base_name)

    def find_target_files():
        return list(Path.glob(Path(workdir, 'mlip_committee'), f'fit_*/{target_file}'))

    target_file_list = find_target_files()

    if len(target_file_list) >= size_of_committee:
        print(f"All {size_of_committee} committee members already trained. Skipping submission.")
        return target_file_list
    
    elif len(target_file_list) != 0:
        print(f"Found {len(target_file_list)} existing committee members. Reusing them.")
        size_of_committee -= len(target_file_list)
        seed += len(target_file_list)

    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {"function_kwargs": {"seed": seed + i, "fit_idx": i, **function_kwargs}}
        for i in range(size_of_committee)
    ]

    executor.run_and_wait(
        function=function,
        job_configs=job_configs,
        common_output_pattern=str(Path(workdir, "mlip_committee", "fit_{job_id}"))
    )

    return find_target_files()
