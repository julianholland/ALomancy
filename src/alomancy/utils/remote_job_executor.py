from pathlib import Path
from typing import Any, Callable, Optional, Union

from expyre.func import ExPyRe
from wfl.autoparallelize.remoteinfo import RemoteInfo


class RemoteJobExecutor:
    """
    General-purpose remote job submission utility.

    Handles submitting arbitrary functions to remote compute resources
    using the ExPyRe framework.
    """

    def __init__(self, remote_info: RemoteInfo):
        """
        Initialize the remote executor.

        Parameters
        ----------
        remote_info : RemoteInfo
            Configuration for remote execution
        """
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
        """
        Submit a single job to remote execution.

        Parameters
        ----------
        function : Callable
            The function to execute remotely
        function_kwargs : Dict[str, Any]
            Keyword arguments to pass to the function
        input_files : List[Union[str, Path]], optional
            Files to transfer to remote system
        output_files : List[Union[str, Path]], optional
            Files to transfer back from remote system
        job_name : str, optional
            Name for this specific job (overrides remote_info.job_name)
        **expyre_kwargs
            Additional ExPyRe-specific arguments

        Returns
        -------
        ExPyRe
            The ExPyRe job object
        """

        if input_files is None:
            input_files = []
        if output_files is None:
            output_files = []
        if job_name is None:
            job_name = self.remote_info.job_name
        # Convert paths to strings
        input_files = [str(f) for f in (input_files or [])]
        output_files = [str(f) for f in (output_files or [])]

        # Use provided files or fall back to remote_info defaults
        final_input_files = input_files or self.remote_info.input_files
        final_output_files = output_files or getattr(
            self.remote_info, "output_files", []
        )

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
        """
        Submit multiple similar jobs with different parameters.

        Parameters
        ----------
        function : Callable
            The function to execute remotely
        job_configs : List[Dict[str, Any]]
            List of job configurations, each containing:
            - function_kwargs: Dict of kwargs for the function
            - input_files: Optional list of input files (in addition to common)
            - output_files: Optional list of output files
            - job_name: Optional specific job name
        common_input_files : List[Union[str, Path]], optional
            Input files common to all jobs
        common_output_pattern : str, optional
            Pattern for output files, use {job_id} for job index
        job_name_pattern : str, optional
            Pattern for job names, use {job_id} for job index

        Returns
        -------
        List[ExPyRe]
            List of ExPyRe job objects
        """
        if common_input_files is None:
            common_input_files = []
        if job_name_pattern is None:
            job_name_pattern = self.remote_info.job_name

        jobs = []
        common_input_files = common_input_files or []

        for i, config in enumerate(job_configs):
            # Prepare input files
            job_input_files = list(common_input_files)
            if "input_files" in config:
                job_input_files.extend(config["input_files"])

            # Prepare output files
            job_output_files = config.get("output_files", [])
            if common_output_pattern:
                job_output_files.append(common_output_pattern.format(job_id=i))

            # Prepare job name
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
        """
        Start all submitted jobs.

        Parameters
        ----------
        **start_kwargs
            Additional arguments for job.start()
        """
        for job in self.jobs:
            job.start(
                resources=self.remote_info.resources,
                system_name=self.remote_info.sys_name,
                header_extra=getattr(self.remote_info, "header_extra", []),
                exact_fit=getattr(self.remote_info, "exact_fit", True),
                partial_node=getattr(self.remote_info, "partial_node", False),
                **start_kwargs,
            )

    def wait_for_all_jobs(self, verbose: bool = True) -> list[Any]:
        """
        Wait for all jobs to complete and gather results.

        Parameters
        ----------
        verbose : bool
            Whether to print job completion status

        Returns
        -------
        List[Any]
            List of results from all jobs
        """
        results = []

        for i, job in enumerate(self.jobs):
            if verbose:
                job_name = getattr(job, "name", f"job_{i}")
                print(f"Waiting for job {i + 1}/{len(self.jobs)}: {job_name}")

            try:
                result, stdout, stderr = job.get_results(
                    timeout=self.remote_info.timeout,
                    check_interval=getattr(self.remote_info, "check_interval", 10),
                )
                results.append(result)

                if verbose:
                    print(f"Job {i + 1} completed successfully")

            except Exception as exc:
                if verbose:
                    print(f"Job {i + 1} failed with error: {exc}")
                    print("stdout", "-" * 30)
                    print(stdout)
                    print("stderr", "-" * 30)
                    print(stderr)
                results.append(None)

        return results

    def cleanup_jobs(self) -> None:
        """Mark all jobs as processed for cleanup."""
        for job in self.jobs:
            job.mark_processed()

    def run_and_wait(
        self,
        function: Callable,
        job_configs: list[dict[str, Any]],
        verbose: bool = True,
        **kwargs,
    ) -> list[Any]:
        """
        Convenience method to submit, start, and wait for multiple jobs.

        Parameters
        ----------
        function : Callable
            The function to execute remotely
        job_configs : List[Dict[str, Any]]
            List of job configurations
        verbose : bool
            Whether to print progress
        **kwargs
            Additional arguments for submit_multiple_jobs

        Returns
        -------
        List[Any]
            Results from all jobs
        """
        self.submit_multiple_jobs(function, job_configs, **kwargs)
        self.start_all_jobs()
        results = self.wait_for_all_jobs(verbose=verbose)
        self.cleanup_jobs()
        return results


# Convenience functions for backward compatibility
# def submit_committee_jobs(
#     remote_info: RemoteInfo,
#     function: Callable,
#     function_kwargs: Dict[str, Any],
#     base_name: str,
#     size_of_committee: int = 5,
#     **kwargs
# ) -> List[Any]:
#     """
#     Submit committee of jobs (like MACE ensemble training).

#     This maintains backward compatibility with your original use case.
#     """
#     executor = RemoteJobExecutor(remote_info)

#     workdir = Path("results", base_name)

#     # Create job configs for committee
#     job_configs = []
#     for i in range(size_of_committee):
#         job_configs.append({
#             'function_kwargs': {
#                 **function_kwargs,
#                 'seed': function_kwargs.get('seed', 803) + i,  # Different seed per job
#                 'output_dir': str(workdir / "MACE" / f"fit_{i}")
#             },
#             'output_files': [str(workdir / "MACE" / f"fit_{i}")]
#         })

#     # Common input files
#     common_input_files = [
#         "mace_wfl.py",
#         str(workdir / "train_set.xyz"),
#         str(workdir / "test_set.xyz"),
#     ]

#     return executor.run_and_wait(
#         function=function,
#         job_configs=job_configs,
#         common_input_files=common_input_files,
#         job_name_pattern=f"{remote_info.job_name}_{{job_id}}",
#         **kwargs
#     )
