import logging
from pathlib import Path
from typing import Any, Callable

from alomancy.configs.remote_info import RemoteInfo
from alomancy.utils.remote_job_executor import RemoteJobExecutor

logger = logging.getLogger(__name__)


def committee_remote_submitter(
    remote_info: RemoteInfo,
    base_name: str,
    _target_file: str,
    function: Callable,
    seed: int = 803,
    size_of_committee: int = 5,
    function_kwargs: dict[str, Any] | None = None,
) -> None:
    mace_dir = Path("results", base_name)
    mace_dir.mkdir(exist_ok=True, parents=True)

      # fill in some params from standard function arguments


    # set number of threads in queued job, only if user hasn't set them
    # if not any(
    #     var.split("=")[0] == "WFL_MACE_FIT_OMP_NUM_THREADS"
    #     for var in remote_info.env_vars
    # ):
    #     remote_info.env_vars.append(
    #         "WFL_MACE_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE"
    #     )
    # if not any(
    #     var.split("=")[0] == "WFL_NUM_PYTHON_SUBPROCESSES"
    #     for var in remote_info.env_vars
    # ):
    #     remote_info.env_vars.append(
    #         "WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE"
    #     )
    # def find_target_files() -> list[str]:
    #     files = list(Path.glob(Path(workdir, "mlip_committee"), f"fit_*/{target_file}"))
    #     return [ str(file) for file in files ]

    # target_file_list = find_target_files()

    # if len(target_file_list) >= size_of_committee:
    #     print(
    #         f"All {size_of_committee} committee members already trained. Skipping submission."
    #     )
    #     return target_file_list

    # elif len(target_file_list) != 0:
    #     print(
    #         f"Found {len(target_file_list)} existing committee members. Reusing them."
    #     )
    #     size_of_committee -= len(target_file_list)
    #     seed += len(target_file_list)

    executor = RemoteJobExecutor(remote_info)

    job_configs = [
        {"function_kwargs": {"seed": seed + i, "fit_idx": i, **(function_kwargs or {})}}
        for i in range(size_of_committee)
    ]

        # run_and_wait expects a callable; provide a no-op if None was passed
    def _noop(**_kwargs: Any) -> None:
        logger.warning("No function provided for remote execution. This is a no-op.")
        return None

    executor.run_and_wait(
        function=(function or _noop),
        job_configs=job_configs,
        common_output_pattern=str(Path(mace_dir, "mlip_committee", "fit_{job_id}")),
    )


