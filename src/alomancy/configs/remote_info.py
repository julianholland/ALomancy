import copy
import logging
import pathlib  # wizadry to solve pickle problem
import sys

from expyre.resources import Resources

sys.modules["pathlib._local"] = pathlib


logger = logging.getLogger(__name__)


class RemoteInfo:
    """Create a RemoteInfo object

    Parameters
    ----------
    sys_name: str
        name of system to run on
    job_name: str
        name for job (unique within this project)
    resources: dict or Resources
        expyre.resources.Resources or kwargs for its constructor
    num_inputs_per_queued_job: int, default -100
        num_inputs_per_python_subprocess for each job. If negative will be multiplied by iterable_autopara_wrappable
        num_inputs_per_python_subprocess
    pre_cmds: list(str)
        commands to run before starting job
    post_cmds: list(str)
        commands to run after finishing job
    env_vars: list(str)
        environment variables to set before starting job
    input_files: list(str)
        input_files to stage in starting job
    output_files: list(str)
        output_files to stage out when job is done
    header_extra: list(str), optional
        extra lines to add to queuing system header
    exact_fit: bool, default True
        require exact fit to node size
    partial_node: bool, default True
        allow jobs that take less than a whole node, overrides exact_fit
    timeout: int
        time to wait in get_results before giving up
    check_interval: int
        check_interval arg to pass to get_results
    ignore_failed_jobs: bool, default False
        skip failures in remote jobs
    resubmit_killed_jobs: bool, default False
        resubmit jobs that were killed without an exit status (out of walltime or crashed),
        hoping that other parameters such as walltime or memory have been changed to make run complete this time
    max_concurrent_jobs: int, default 20
        cap on how many jobs a RemoteJobExecutor keeps started (occupying a scheduler slot)
        at once; the next queued job starts the instant a running one finishes. This is a
        property of the HPC system/account, sourced from the HPC profile in
        ~/.alomancy/hpc_config.yaml (see get_remote_info) -- unrelated to
        structure_generation's max_number_of_concurrent_jobs, which controls how many seed
        structures are selected for MD, not scheduler concurrency.
    lock_timeout: float or None, default None
        max seconds a RemoteJobExecutor worker thread will wait to acquire the
        per-host ssh-call lock (see executor._get_ssh_call_lock) before giving
        up on its own job rather than waiting forever. None waits indefinitely
        (matches the old, unbounded behavior). get_remote_info sets this from
        the job's own max_time -- if a thread has waited longer than the job's
        entire expected walltime just for a turn to touch ssh, something is
        stuck (e.g. the shared control connection died and a fresh connection
        needs interactive auth nobody can provide), not merely busy.
    hash_ignore: list(str), default []
        list of arguments to ignore when doing hash of remote function arguments to determine if it's already been done
    """

    def __init__(
        self,
        sys_name,
        job_name,
        resources,
        num_inputs_per_queued_job=-100,
        pre_cmds=None,
        post_cmds=None,
        env_vars=None,
        input_files=None,
        output_files=None,
        header_extra=None,
        exact_fit=True,
        partial_node=False,
        timeout=3600,
        check_interval=30,
        ignore_failed_jobs=False,
        resubmit_killed_jobs=False,
        max_concurrent_jobs=20,
        lock_timeout=None,
        hash_ignore=None,
    ):
        if hash_ignore is None:
            hash_ignore = []
        if header_extra is None:
            header_extra = []
        if output_files is None:
            output_files = []
        if input_files is None:
            input_files = []
        if env_vars is None:
            env_vars = []
        if post_cmds is None:
            post_cmds = []
        if pre_cmds is None:
            pre_cmds = []
        self.sys_name = sys_name
        self.job_name = job_name
        self.resources = copy.deepcopy(resources)
        self.num_inputs_per_queued_job = num_inputs_per_queued_job
        self.pre_cmds = pre_cmds.copy()
        self.post_cmds = post_cmds.copy()
        self.env_vars = env_vars.copy()
        self.input_files = input_files.copy()
        self.output_files = output_files.copy()
        self.header_extra = header_extra.copy()

        self.exact_fit = exact_fit
        self.partial_node = partial_node
        self.timeout = timeout
        self.check_interval = check_interval
        self.ignore_failed_jobs = ignore_failed_jobs
        self.resubmit_killed_jobs = resubmit_killed_jobs
        self.max_concurrent_jobs = max_concurrent_jobs
        self.lock_timeout = lock_timeout
        self.hash_ignore = hash_ignore.copy()

    def __str__(self):
        return (
            f"{self.sys_name} {self.job_name} {self.resources} {self.num_inputs_per_queued_job} {self.exact_fit} "
            f"{self.partial_node} {self.timeout} {self.check_interval}"
        )


_DEFAULT_MAX_CONCURRENT_JOBS = 20


def _resolve_max_concurrent_jobs(job_dict: dict) -> int:
    """Resolve the concurrency cap for a job dict.

    Precedence: an explicit ``max_concurrent_jobs`` on the HPC profile
    (``job_dict["hpc"]``) always wins when present -- it is the sole
    authoritative, current-format home for this setting. The legacy
    job-dict-level ``max_batch_size`` key is only consulted as a fallback
    when the profile doesn't define the new key, so a leftover key a user
    forgot to delete can't silently override a value they've since migrated
    into their HPC profile. ``max_batch_size`` is deprecated and will be
    removed in ALomancy 1.0.0 -- see docs/deprecations.md.
    """
    hpc = job_dict["hpc"]
    explicit_cap = hpc.get("max_concurrent_jobs")
    legacy_cap = job_dict.get("max_batch_size")

    if legacy_cap is not None:
        if explicit_cap is not None:
            logger.warning(
                "'max_batch_size' on job '%s' is deprecated and ignored because "
                "HPC profile '%s' already defines max_concurrent_jobs=%s. Remove "
                "'max_batch_size' from your job config -- it will be removed "
                "entirely in ALomancy 1.0.0. See docs/deprecations.md.",
                job_dict.get("name"),
                hpc.get("hpc_name"),
                explicit_cap,
            )
        else:
            logger.warning(
                "'max_batch_size' on job '%s' is deprecated. Using its value "
                "(%s) as max_concurrent_jobs because HPC profile '%s' does not "
                "define max_concurrent_jobs. Move this setting into "
                "~/.alomancy/hpc_config.yaml under the '%s' profile's hpc dict "
                "instead; 'max_batch_size' will be removed entirely in "
                "ALomancy 1.0.0. See docs/deprecations.md.",
                job_dict.get("name"),
                legacy_cap,
                hpc.get("hpc_name"),
                hpc.get("hpc_name"),
            )

    if explicit_cap is not None:
        return explicit_cap
    if legacy_cap is not None:
        return legacy_cap
    return _DEFAULT_MAX_CONCURRENT_JOBS


def get_remote_info(job_dict, input_files: list[str] | None = None) -> RemoteInfo:
    """
    Returns a RemoteInfo object for running MACE fits on a GPU cluster.
    """
    if input_files is None:
        input_files = []

    logger.debug("HPC: %s, Job: %s", job_dict["hpc"]["hpc_name"], job_dict["name"])
    return RemoteInfo(
        sys_name=job_dict["hpc"]["hpc_name"],
        job_name=job_dict["name"],
        num_inputs_per_queued_job=1,
        timeout=36000 * 3,
        input_files=input_files,
        pre_cmds=job_dict["hpc"].get("pre_cmds", []),
        resources=Resources(
            max_time=job_dict["max_time"],
            num_nodes=1,
            partitions=job_dict["hpc"]["partitions"],
        ),
        max_concurrent_jobs=_resolve_max_concurrent_jobs(job_dict),
        # lock_timeout intentionally left at RemoteInfo's own default (None
        # -- wait indefinitely for the per-host ssh-call lock). This used
        # to be set to time_to_sec(job_dict["max_time"]), so a job queued
        # behind one stuck on an interactive password/OTP prompt would
        # give up and fail loudly after its own max_time elapsed rather
        # than just waiting -- the actual stuck call was, and still is,
        # never bounded either way, so that revert didn't make the stuck
        # call itself succeed any faster, it only turned "everything
        # waits quietly for you to type the password" into "everything
        # else fails around the one call that's waiting for you to type
        # the password." Restored to match the pre-0.5.2 behavior by
        # request. pre_run_checks()'s ensure_ssh_connectivity call is the
        # real fix for the underlying problem this was trying to guard
        # against -- it authenticates every HPC host up front, before any
        # remote submission begins, while a person is presumably still at
        # the terminal to answer a prompt.
    )
