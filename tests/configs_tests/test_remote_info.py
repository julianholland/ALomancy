"""Unit tests for RemoteInfo/get_remote_info max_concurrent_jobs resolution."""

import pytest


def _job_dict(hpc_extra=None):
    hpc = {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]}
    if hpc_extra:
        hpc.update(hpc_extra)
    return {
        "name": "high_accuracy_evaluation",
        "max_time": "10m",
        "hpc": hpc,
    }


@pytest.mark.unit
def test_default_when_nothing_set():
    from alomancy.configs.remote_info import get_remote_info

    info = get_remote_info(_job_dict())
    assert info.max_concurrent_jobs == 20


@pytest.mark.unit
def test_lock_timeout_defaults_to_none():
    """lock_timeout (the cap on how long a RemoteJobExecutor worker waits
    for the per-host ssh-call lock, see remote_submission/executor.py) is
    left at RemoteInfo's own default of None -- wait indefinitely --
    rather than being derived from the job's max_time. A prior version
    set it from max_time, so a job queued behind one stuck on an
    interactive ssh password/OTP prompt would give up and fail loudly
    instead of just waiting for the prompt to be answered, which is not
    the desired behavior: pre_run_checks()'s ensure_ssh_connectivity
    authenticates every HPC host up front instead, while a person is
    presumably still at the terminal to answer a prompt, rather than
    guessing at how long a wait is "too long" mid-run."""
    from alomancy.configs.remote_info import get_remote_info

    info = get_remote_info(_job_dict())
    assert info.lock_timeout is None


@pytest.mark.unit
def test_hpc_profile_value_used():
    from alomancy.configs.remote_info import get_remote_info

    info = get_remote_info(_job_dict(hpc_extra={"max_concurrent_jobs": 7}))
    assert info.max_concurrent_jobs == 7


@pytest.mark.unit
def test_remote_info_default_constructor_arg():
    from alomancy.configs.remote_info import RemoteInfo

    info = RemoteInfo(sys_name="s", job_name="j", resources={})
    assert info.max_concurrent_jobs == 20
    assert info.lock_timeout is None
