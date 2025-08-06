from pathlib import Path
from wfl.autoparallelize.remoteinfo import RemoteInfo
from typing import List, Callable
from alomancy.utils.remote_job_executor import RemoteJobExecutor


def committee_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    seed: int = 803,
    size_of_committee: int = 5,
    function: Callable = None,
    function_kwargs=None,
):
    workdir = Path("results", base_name)
    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {"function_kwargs": {"seed": seed + i, **function_kwargs}}
        for i in range(size_of_committee)
    ]

    results = executor.run_and_wait(
        function=function,
        job_configs=job_configs,
        common_output_pattern=str(Path(workdir, "mlip_committee", "fit_{job_id}")),
    )

    return results
