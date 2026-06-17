import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

from expyre.func import ExPyRe

from alomancy.configs.remote_info import RemoteInfo

logger = logging.getLogger(__name__)


class RemoteJobExecutor:
    """
    General-purpose remote job submission utility.

    Handles submitting arbitrary functions to remote compute resources
    using the ExPyRe framework.
    """

    def __init__(self, remote_info: RemoteInfo):
        self.remote_info = remote_info
        self.jobs = []

    def submit_job(
        self,
        function: Callable,
        function_kwargs: dict[str, Any],
        input_files: list[Union[str, Path]] | None = None,
        output_files: list[Union[str, Path]] | None = None,
        job_name: Optional[str] | None = None,
        **expyre_kwargs,
    ) -> ExPyRe:
        if input_files is None:
            input_files = []
        if output_files is None:
            output_files = []
        if job_name is None:
            job_name = self.remote_info.job_name
        input_files = [str(f) for f in (input_files or [])]
        output_files = [str(f) for f in (output_files or [])]

        final_input_files = input_files or self.remote_info.input_files
        final_output_files = output_files or getattr(self.remote_info, "output_files", [])

        job = ExPyRe(
            name=job_name or self.remote_info.job_name,
            pre_run_commands=self.remote_info.pre_cmds,
            post_run_commands=getattr(self.remote_info, "post_cmds", []),
            env_vars=getattr(self.remote_info, "env_vars", {}),
            input_files=final_input_files,
            output_files=final_output_files,
            function=function,
            kwargs=function_kwargs,
            **expyre_kwargs,
        )

        self.jobs.append(job)
        return job

    def submit_multiple_jobs(
        self,
        function: Callable,
        job_configs: list[dict[str, Any]],
        common_input_files: list[Union[str, Path]] | None = None,
        common_output_pattern: Optional[str] | None = None,
        job_name_pattern: Optional[str] | None = None,
    ) -> list[ExPyRe]:
        if common_input_files is None:
            common_input_files = []
        if job_name_pattern is None:
            job_name_pattern = self.remote_info.job_name

        jobs = []
        common_input_files = common_input_files or []

        for i, config in enumerate(job_configs):
            job_input_files = list(common_input_files)
            if "input_files" in config:
                job_input_files.extend(config["input_files"])

            job_output_files = config.get("output_files", [])
            if common_output_pattern:
                job_output_files.append(common_output_pattern.format(job_id=i))
            logger.debug("Job %d output files: %s", i, job_output_files)

            job_name = config.get("job_name")
            if not job_name and job_name_pattern:
                job_name = job_name_pattern.format(job_id=i)

            job = self.submit_job(
                function=function,
                function_kwargs=config["function_kwargs"],
                input_files=job_input_files,
                output_files=job_output_files,
                job_name=job_name,
            )
            jobs.append(job)

        return jobs

    def start_all_jobs(self, **start_kwargs) -> None:
        for job in self.jobs:
            job.start(
                resources=self.remote_info.resources,
                system_name=self.remote_info.sys_name,
                header_extra=getattr(self.remote_info, "header_extra", []),
                exact_fit=getattr(self.remote_info, "exact_fit", True),
                partial_node=getattr(self.remote_info, "partial_node", False),
                **start_kwargs,
            )

    def wait_for_all_jobs(self) -> list[Any]:
        results = []

        for i, job in enumerate(self.jobs):
            stdout, stderr = None, None
            job_name = getattr(job, "name", f"job_{i}")
            logger.debug("Waiting for job %d/%d: %s", i + 1, len(self.jobs), job_name)

            try:
                result, stdout, stderr = job.get_results(
                    timeout=self.remote_info.timeout,
                    check_interval=getattr(self.remote_info, "check_interval", 10),
                )
                results.append(result)
                logger.info("Job %d completed successfully.", i + 1)

            except Exception as exc:
                logger.warning("Job %d failed: %s", i + 1, exc)
                logger.debug("Job %d stdout:\n%s", i + 1, stdout)
                logger.debug("Job %d stderr:\n%s", i + 1, stderr)
                results.append(None)

        return results

    def cleanup_jobs(self) -> None:
        for job in self.jobs:
            job.mark_processed()

    def run_and_wait(
        self,
        function: Callable,
        job_configs: list[dict[str, Any]],
        **kwargs,
    ) -> list[Any]:
        logger.debug("run_and_wait working directory: %s", os.getcwd())
        self.submit_multiple_jobs(function, job_configs, **kwargs)
        self.start_all_jobs()
        self.wait_for_all_jobs()

        # final call essential to sync results locally from the remote
        results = self.wait_for_all_jobs()
        self.cleanup_jobs()
        return results
