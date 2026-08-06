"""Unit tests for RemoteInfo/get_remote_info max_concurrent_jobs resolution."""

import logging

import pytest


def _job_dict(hpc_extra=None, max_batch_size=None):
    hpc = {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]}
    if hpc_extra:
        hpc.update(hpc_extra)
    job_dict = {
        "name": "high_accuracy_evaluation",
        "max_time": "10m",
        "hpc": hpc,
    }
    if max_batch_size is not None:
        job_dict["max_batch_size"] = max_batch_size
    return job_dict


@pytest.mark.unit
def test_default_when_nothing_set():
    from alomancy.configs.remote_info import get_remote_info

    info = get_remote_info(_job_dict())
    assert info.max_concurrent_jobs == 20


@pytest.mark.unit
def test_lock_timeout_derived_from_max_time():
    """lock_timeout (the cap on how long a RemoteJobExecutor worker waits
    for the per-host ssh-call lock, see remote_submission/executor.py) is
    sourced from the same max_time string Resources() uses for the actual
    Slurm walltime request -- so a job that's waited longer than its own
    expected total runtime just for a turn to touch ssh is treated as stuck
    rather than merely busy."""
    from alomancy.configs.remote_info import get_remote_info

    info = get_remote_info(_job_dict())
    assert info.lock_timeout == 600  # "10m" from _job_dict()


@pytest.mark.unit
def test_hpc_profile_value_used():
    from alomancy.configs.remote_info import get_remote_info

    info = get_remote_info(_job_dict(hpc_extra={"max_concurrent_jobs": 7}))
    assert info.max_concurrent_jobs == 7


@pytest.mark.unit
def test_legacy_max_batch_size_used_as_fallback(caplog):
    from alomancy.configs.remote_info import get_remote_info

    with caplog.at_level(logging.WARNING, logger="alomancy.configs.remote_info"):
        info = get_remote_info(_job_dict(max_batch_size=5))

    assert info.max_concurrent_jobs == 5
    assert any(
        "Using its value" in r.message and "1.0.0" in r.message for r in caplog.records
    )


@pytest.mark.unit
def test_hpc_profile_wins_over_legacy_max_batch_size(caplog):
    from alomancy.configs.remote_info import get_remote_info

    with caplog.at_level(logging.WARNING, logger="alomancy.configs.remote_info"):
        info = get_remote_info(
            _job_dict(hpc_extra={"max_concurrent_jobs": 12}, max_batch_size=5)
        )

    assert info.max_concurrent_jobs == 12
    assert any("ignored" in r.message and "1.0.0" in r.message for r in caplog.records)


@pytest.mark.unit
def test_remote_info_default_constructor_arg():
    from alomancy.configs.remote_info import RemoteInfo

    info = RemoteInfo(sys_name="s", job_name="j", resources={})
    assert info.max_concurrent_jobs == 20
    assert info.lock_timeout is None
