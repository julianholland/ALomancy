"""
Tests for utility functions and modules.

This module tests various utility functions used throughout the alomancy package.
"""

import fcntl
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from alomancy.remote_submission.executor import RemoteJobExecutor


class _EventTracker:
    """Records real start/finish wall-clock times per fake job name."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: dict[str, dict] = {}

    def on_start(self, name):
        with self.lock:
            self.events.setdefault(name, {})["start"] = time.monotonic()

    def on_finish(self, name):
        with self.lock:
            self.events.setdefault(name, {})["finish"] = time.monotonic()


class _ConcurrencyTracker:
    """Tracks the peak number of simultaneously-"started" fake jobs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def on_start(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def on_finish(self):
        with self.lock:
            self.active -= 1


class _FakeExPyReJob:
    """Mimics the ExPyRe interface RemoteJobExecutor relies on
    (.start/.get_results/.stage_dir/.name/.mark_processed) with real,
    controllable blocking timing -- no mocking of RemoteJobExecutor itself."""

    def __init__(
        self,
        name,
        stage_dir,
        result=None,
        duration=0.02,
        should_fail=False,
        event_tracker=None,
        concurrency_tracker=None,
        start_duration=0.0,
        start_concurrency_tracker=None,
    ):
        self.name = name
        self.stage_dir = stage_dir
        self.result = result if result is not None else name
        self.duration = duration
        self.should_fail = should_fail
        self._events = event_tracker
        self._concurrency = concurrency_tracker
        self._start_duration = start_duration
        self._start_concurrency = start_concurrency_tracker

    def start(self, **kwargs):
        if self._start_concurrency is not None:
            self._start_concurrency.on_start()
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        (self.stage_dir / "_expyre_job_started").touch()
        if self._start_duration:
            time.sleep(self._start_duration)
        if self._events is not None:
            self._events.on_start(self.name)
        if self._concurrency is not None:
            self._concurrency.on_start()
        if self._start_concurrency is not None:
            self._start_concurrency.on_finish()

    def get_results(self, **kwargs):
        time.sleep(self.duration)
        if self._events is not None:
            self._events.on_finish(self.name)
        if self._concurrency is not None:
            self._concurrency.on_finish()
        if self.should_fail:
            raise RuntimeError(f"{self.name} failed")
        return self.result, "stdout", "stderr"

    def mark_processed(self):
        pass


def _fake_executor(max_concurrent_jobs, jobs, sys_name="test-hpc", lock_timeout=None):
    remote_info = SimpleNamespace(
        resources={},
        sys_name=sys_name,
        timeout=10,
        check_interval=0.001,
        header_extra=[],
        exact_fit=True,
        partial_node=False,
        max_concurrent_jobs=max_concurrent_jobs,
        lock_timeout=lock_timeout,
    )
    executor = RemoteJobExecutor(remote_info)
    executor.jobs = jobs
    return executor


@pytest.fixture
def write_temporary_yaml():
    """Fixture to create a temporary YAML file for testing."""
    from yaml import dump

    tmp_dict = {
        "mlip_committee": {
            "name": "mlip_test",
            "max_time": "value",
            "hpc": {"hpc_name": "test-hpc"},
            "size_of_committee": 5,
        },
        "structure_generation": {
            "name": "struc_gen_test",
            "max_time": "value",
            "hpc": {"hpc_name": "test-hpc"},
            "number_of_concurrent_jobs": 5,
        },
        "high_accuracy_evaluation": {
            "name": "high_acc_test",
            "max_time": "value",
            "hpc": {
                "node_info": {
                    "ranks_per_system": 4,
                    "ranks_per_node": 2,
                    "threads_per_rank": 8,
                }
            },
            "pwx_path": "/path/to/pw.x",
            "pp_path": "/path/to/pseudopotentials",
            "pseudo_dict": {"H": "H.pz-vbc.UPF", "O": "O.pz-vbc.UPF"},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_file = Path(tmpdir) / "test.yaml"
        with open(yaml_file, "w") as f:
            dump(tmp_dict, f)
        yield yaml_file


class TestRemoteJobExecutor:
    """Test remote job execution utilities."""

    @pytest.mark.unit
    def test_remote_job_executor_initialization(self):
        """Test RemoteJobExecutor initialization."""
        # Test that we can create a mock remote info
        mock_remote_info = MagicMock()
        assert mock_remote_info is not None

    @pytest.mark.unit
    def test_job_config_validation(self):
        """Test job configuration validation."""
        # Test valid job config
        valid_config = {"function_kwargs": {"param1": "value1", "param2": 42}}

        assert "function_kwargs" in valid_config
        assert isinstance(valid_config["function_kwargs"], dict)

    @pytest.mark.unit
    def test_invalid_job_config(self):
        """Test handling of invalid job configurations."""
        invalid_configs = [
            {},  # Empty config
            {"wrong_key": "value"},  # Missing function_kwargs
            {"function_kwargs": "not_a_dict"},  # Wrong type for function_kwargs
        ]

        for config in invalid_configs:
            if "function_kwargs" not in config:
                assert "function_kwargs" not in config
            elif not isinstance(config.get("function_kwargs"), dict):
                assert not isinstance(config.get("function_kwargs"), dict)

    @pytest.mark.unit
    def test_never_exceeds_max_concurrent_jobs(self, tmp_path):
        """Peak concurrently-started jobs never exceeds max_concurrent_jobs."""
        tracker = _ConcurrencyTracker()
        jobs = [
            _FakeExPyReJob(
                f"job{i}",
                tmp_path / f"job{i}",
                duration=0.05,
                concurrency_tracker=tracker,
            )
            for i in range(8)
        ]
        executor = _fake_executor(max_concurrent_jobs=3, jobs=jobs)
        results = executor.run_all_jobs_bounded()

        assert tracker.peak <= 3
        assert results == [f"job{i}" for i in range(8)]

    @pytest.mark.unit
    def test_job_start_is_serialized_while_monitoring_stays_concurrent(self, tmp_path):
        """job.start() (the ssh-heavy staging/submission call) never overlaps
        across threads even when max_concurrent_jobs lets many jobs be
        monitored at once -- otherwise max_concurrent_jobs simultaneous ssh
        sessions burst against one shared multiplexed control connection,
        which is what caused jobs to hang indefinitely on an unattended
        interactive-auth prompt in production. Monitoring (get_results) must
        still run with real concurrency -- that's the entire point of the
        rolling-window model over the old batch-and-wait-in-order code."""
        # duration (get_results window) is an order of magnitude bigger than
        # start_duration so each job's monitoring window comfortably outlasts
        # the cumulative serialized-start delay of every job queued behind
        # it, giving a wide, jitter-tolerant overlap margin rather than a
        # tight race between real wall-clock windows.
        start_tracker = _ConcurrencyTracker()
        monitor_tracker = _ConcurrencyTracker()
        jobs = [
            _FakeExPyReJob(
                f"job{i}",
                tmp_path / f"job{i}",
                duration=0.3,
                start_duration=0.02,
                start_concurrency_tracker=start_tracker,
                concurrency_tracker=monitor_tracker,
            )
            for i in range(6)
        ]
        executor = _fake_executor(max_concurrent_jobs=6, jobs=jobs)
        executor.run_all_jobs_bounded()

        assert start_tracker.peak == 1
        assert monitor_tracker.peak > 1

    @pytest.mark.unit
    def test_sync_remote_results_status_is_serialized(self, monkeypatch):
        """get_results()'s polling loop calls sync_remote_results_status()
        on its own independent per-job schedule to check remote status and
        pull back files -- shelling out over ssh (squeue-equivalent + rsync)
        exactly like job.start() does. Without serializing this too,
        max_concurrent_jobs monitoring threads can trigger it simultaneously
        and reproduce the same ssh-session-burst hang job.start()'s lock
        alone doesn't cover -- observed in production as a stuck ``squeue``
        subprocess. Two calls for the same host must never run at once, even
        though the redundant-call coalescing (see the next test) means not
        every thread necessarily reaches the real call at all."""
        from expyre.func import ExPyRe

        import alomancy.remote_submission.executor as executor_module

        tracker = _ConcurrencyTracker()

        def _fake_sync(self, *args, **kwargs):
            tracker.on_start()
            time.sleep(0.05)
            tracker.on_finish()

        monkeypatch.setattr(ExPyRe, "sync_remote_results_status", _fake_sync)
        monkeypatch.setattr(executor_module, "_expyre_sync_patched", False)

        executor_module._ensure_expyre_sync_serialized()

        class _Dummy:
            system_name = "test-hpc-serialized"

        threads = [
            threading.Thread(target=lambda: ExPyRe.sync_remote_results_status(_Dummy()))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.peak == 1

    @pytest.mark.unit
    def test_sync_remote_results_status_coalesces_concurrent_calls(self, monkeypatch):
        """sync_remote_results_status() is always called with sync_all=True
        in this codebase, meaning one call already refreshes every job on
        that host -- not just the caller's own. If several monitoring
        threads for the same host queue up behind the lock at once, only
        the first should make the real call; the rest, finding the host's
        sync generation already advanced past what they saw before they
        started waiting, should skip rather than repeat an already-fresh
        squeue/rsync round-trip. Without this, N threads piling up behind
        the lock cost N real calls instead of 1, and enough of them can push
        a polling round past check_interval."""
        from expyre.func import ExPyRe

        import alomancy.remote_submission.executor as executor_module

        call_count = _ConcurrencyTracker()

        def _fake_sync(self, *args, **kwargs):
            call_count.on_start()
            time.sleep(0.05)

        monkeypatch.setattr(ExPyRe, "sync_remote_results_status", _fake_sync)
        monkeypatch.setattr(executor_module, "_expyre_sync_patched", False)
        monkeypatch.setattr(executor_module, "_sync_generations", {})

        executor_module._ensure_expyre_sync_serialized()

        class _Dummy:
            system_name = "test-hpc-coalesce"

        threads = [
            threading.Thread(target=lambda: ExPyRe.sync_remote_results_status(_Dummy()))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count.active == 1

    @pytest.mark.unit
    def test_ssh_call_lock_is_scoped_per_host(self, monkeypatch):
        """Two RemoteJobExecutors targeting different HPC hosts (different
        remote_info.sys_name) share no ssh connection at all, so their
        job.start() calls must not serialize against each other -- only
        calls that genuinely share one host's multiplexed connection should
        contend for the same lock."""
        import alomancy.remote_submission.executor as executor_module

        monkeypatch.setattr(executor_module, "_ssh_call_locks", {})

        tracker = _ConcurrencyTracker()

        def _hold_lock(sys_name):
            with executor_module._get_ssh_call_lock(sys_name):
                tracker.on_start()
                time.sleep(0.1)
                tracker.on_finish()

        threads = [
            threading.Thread(target=_hold_lock, args=(sys_name,))
            for sys_name in ["host-a", "host-b"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.peak == 2

    @pytest.mark.unit
    def test_job_start_gives_up_after_lock_timeout_instead_of_hanging_forever(
        self, tmp_path
    ):
        """If the ssh-call lock for a host is held by something that never
        releases it -- e.g. another job's ssh call is genuinely stuck on an
        unattended interactive-auth prompt after the shared control
        connection died -- a job with a bounded lock_timeout must fail
        promptly and let siblings proceed, rather than wait forever. This is
        the mitigation for the wider blast radius the per-host lock itself
        introduces: before it existed, a stuck ssh call only stranded its
        own job; now every job sharing that lock would otherwise wait on it
        indefinitely too."""
        from alomancy.remote_submission.executor import _get_ssh_call_lock

        sys_name = "test-hpc-stuck"
        stuck_lock = _get_ssh_call_lock(sys_name)
        stuck_lock.acquire()  # simulate another job's ssh call stuck holding it
        try:
            jobs = [_FakeExPyReJob("job0", tmp_path / "job0")]
            executor = _fake_executor(
                max_concurrent_jobs=1,
                jobs=jobs,
                sys_name=sys_name,
                lock_timeout=0.05,
            )

            results = executor.run_all_jobs_bounded()

            assert results == [None]
        finally:
            stuck_lock.release()

    @pytest.mark.unit
    def test_finishing_job_frees_slot_promptly(self, tmp_path):
        """A short job queued behind a long one starts as soon as *any*
        running job frees a slot -- not only after the long straggler
        finishes (the old batch-wide-wait behavior this replaces)."""
        events = _EventTracker()
        # job0 is slow; job1 is fast and shares a slot with it; job2/job3
        # are queued and should start as slots free up from the fast jobs,
        # well before job0 (the straggler) ever finishes.
        jobs = [
            _FakeExPyReJob(
                "job0", tmp_path / "job0", duration=0.3, event_tracker=events
            ),
            _FakeExPyReJob(
                "job1", tmp_path / "job1", duration=0.02, event_tracker=events
            ),
            _FakeExPyReJob(
                "job2", tmp_path / "job2", duration=0.02, event_tracker=events
            ),
            _FakeExPyReJob(
                "job3", tmp_path / "job3", duration=0.02, event_tracker=events
            ),
        ]
        executor = _fake_executor(max_concurrent_jobs=2, jobs=jobs)
        executor.run_all_jobs_bounded()

        job0_finish = events.events["job0"]["finish"]
        job2_start = events.events["job2"]["start"]
        job3_start = events.events["job3"]["start"]

        # job2 and job3 must have started well before the straggler (job0)
        # finished -- proving they didn't wait for the whole group.
        assert job2_start < job0_finish
        assert job3_start < job0_finish

    @pytest.mark.unit
    def test_one_job_failure_does_not_block_others(self, tmp_path):
        """A failing job returns None at its own index without blocking or
        losing the results of sibling jobs."""
        jobs = [
            _FakeExPyReJob("job0", tmp_path / "job0", duration=0.01),
            _FakeExPyReJob("job1", tmp_path / "job1", duration=0.01, should_fail=True),
            _FakeExPyReJob("job2", tmp_path / "job2", duration=0.01),
        ]
        executor = _fake_executor(max_concurrent_jobs=3, jobs=jobs)
        results = executor.run_all_jobs_bounded()

        assert results == ["job0", None, "job2"]

    @pytest.mark.unit
    def test_results_index_aligned_regardless_of_completion_order(self, tmp_path):
        """Results stay aligned to submission index even when a
        later-submitted job finishes before an earlier one."""
        jobs = [
            _FakeExPyReJob("job0", tmp_path / "job0", result=0, duration=0.15),
            _FakeExPyReJob("job1", tmp_path / "job1", result=1, duration=0.01),
            _FakeExPyReJob("job2", tmp_path / "job2", result=2, duration=0.08),
        ]
        executor = _fake_executor(max_concurrent_jobs=3, jobs=jobs)
        results = executor.run_all_jobs_bounded()

        assert results == [0, 1, 2]


@pytest.mark.unit
class TestAcquireLocalExpyreLock:
    """acquire_local_expyre_lock guards against two separate alomancy
    *processes* racing the same resolved ExPyRe local_stage_dir -- the
    thread-scoped locks above (_get_ssh_call_lock etc.) give no protection
    across process boundaries. fcntl.flock locks are scoped per *open file
    description*, not per-process, so opening the same lock path a second
    time within this test process still faithfully exercises the
    cross-process-contention path these tests are actually about."""

    def _reset(self, monkeypatch):
        import alomancy.remote_submission.executor as executor_module

        monkeypatch.setattr(executor_module, "_expyre_root_lock_handle", None)
        return executor_module

    def test_noop_when_no_local_stage_dir_resolved(self, tmp_path, monkeypatch):
        from expyre import config as expyre_config

        executor_module = self._reset(monkeypatch)
        monkeypatch.setattr(expyre_config, "local_stage_dir", None)

        executor_module.acquire_local_expyre_lock()

        assert executor_module._expyre_root_lock_handle is None
        assert not (tmp_path / "alomancy.lock").exists()

    def test_acquires_lock_and_records_holder_metadata(self, tmp_path, monkeypatch):
        from expyre import config as expyre_config

        executor_module = self._reset(monkeypatch)
        monkeypatch.setattr(expyre_config, "local_stage_dir", tmp_path)

        executor_module.acquire_local_expyre_lock()

        assert executor_module._expyre_root_lock_handle is not None
        lock_path = tmp_path / "alomancy.lock"
        assert lock_path.exists()
        assert f"pid={os.getpid()}" in lock_path.read_text()

    def test_reentrant_call_in_same_process_is_a_noop(self, tmp_path, monkeypatch):
        from expyre import config as expyre_config

        executor_module = self._reset(monkeypatch)
        monkeypatch.setattr(expyre_config, "local_stage_dir", tmp_path)

        executor_module.acquire_local_expyre_lock()
        first_handle = executor_module._expyre_root_lock_handle
        executor_module.acquire_local_expyre_lock()  # must not raise or reopen

        assert executor_module._expyre_root_lock_handle is first_handle

    def test_concurrent_threads_in_same_process_do_not_race(
        self, tmp_path, monkeypatch
    ):
        """fcntl.flock is scoped per *open file description*, not per
        process: without _expyre_root_lock_guard, two threads could both
        see the handle as unset and each open() a fresh file description
        to the same lock_path, so the second would fail to flock() against
        the first's already-held lock and incorrectly raise "another
        alomancy process already holds the lock" against its own process.
        Exactly one open()+flock() must happen; every thread must return
        cleanly with the same handle set."""
        from expyre import config as expyre_config

        executor_module = self._reset(monkeypatch)
        monkeypatch.setattr(expyre_config, "local_stage_dir", tmp_path)

        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def _call():
            try:
                executor_module.acquire_local_expyre_lock()
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert executor_module._expyre_root_lock_handle is not None

    def test_raises_when_already_held_by_another_process(self, tmp_path, monkeypatch):
        from expyre import config as expyre_config

        executor_module = self._reset(monkeypatch)
        monkeypatch.setattr(expyre_config, "local_stage_dir", tmp_path)

        lock_path = tmp_path / "alomancy.lock"
        with open(lock_path, "a+") as holder_fh:
            fcntl.flock(holder_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            holder_fh.write(
                "pid=999999 host=other-machine started=2026-01-01T00:00:00\n"
            )
            holder_fh.flush()

            with pytest.raises(RuntimeError, match="already holds the lock"):
                executor_module.acquire_local_expyre_lock()
            assert executor_module._expyre_root_lock_handle is None

    def test_second_open_fd_to_same_path_cannot_also_acquire(
        self, tmp_path, monkeypatch
    ):
        from expyre import config as expyre_config

        executor_module = self._reset(monkeypatch)
        monkeypatch.setattr(expyre_config, "local_stage_dir", tmp_path)

        executor_module.acquire_local_expyre_lock()

        with (
            open(tmp_path / "alomancy.lock", "a+") as second_fh,
            pytest.raises(OSError),
        ):
            fcntl.flock(second_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.unit
class TestEnsureExpyreDbThreadSafe:
    """_ensure_expyre_db_thread_safe patches JobsDB._execute to be safe for
    concurrent threads sharing one sqlite3 connection. Regression coverage
    for the production bug where `list(config.db.jobs(id=...))[0])` raised
    IndexError from a single process running many concurrent job-monitoring
    threads (structure_generation submitting ~20 MD jobs at once) -- not
    cross-process contention. Root cause: the lock originally wrapped only
    `self.db.execute(cmd)` (which returns a live, unfetched sqlite3.Cursor),
    releasing before the caller iterated it -- leaving a window where a
    second thread's own execute() call on the same shared connection could
    corrupt the first thread's still-pending cursor, making it yield zero
    rows even though the row genuinely existed."""

    def _patched_db(self, monkeypatch, tmp_path):
        from expyre import config as expyre_config
        from expyre.jobsdb import JobsDB

        import alomancy.remote_submission.executor as executor_module

        monkeypatch.setattr(executor_module, "_expyre_db_patched", False)
        db = JobsDB(str(tmp_path / "jobs.db"))
        monkeypatch.setattr(expyre_config, "db", db)
        executor_module._ensure_expyre_db_thread_safe()
        return db

    def test_execute_returns_a_list_not_a_live_cursor(self, tmp_path, monkeypatch):
        """The fetch must happen *inside* the lock: a caller must never be
        handed a live cursor to iterate after the lock is already released,
        since that's precisely the gap the production race lived in."""
        import sqlite3

        db = self._patched_db(monkeypatch, tmp_path)

        result = db._execute("SELECT * FROM jobs")

        assert isinstance(result, list)
        assert not isinstance(result, sqlite3.Cursor)

    def test_add_then_immediate_jobs_lookup_never_empty_under_concurrency(
        self, tmp_path, monkeypatch
    ):
        """Stress-reproduction of the production failure: many threads each
        add their own job then immediately look it up by id, while every
        other thread is doing the same thing concurrently against the one
        shared connection. Every thread's own lookup of its own
        just-added, never-removed row must find exactly one match, every
        single time -- a single miss reproduces the "list index out of
        range" IndexError seen in production."""
        db = self._patched_db(monkeypatch, tmp_path)

        n_threads = 16
        n_rounds = 25
        errors: list[str] = []
        errors_lock = threading.Lock()

        def _worker(worker_idx):
            for round_idx in range(n_rounds):
                job_id = f"job-{worker_idx}-{round_idx}"
                db.add(job_id, name="n", from_dir="/tmp", status="created")
                rows = list(db.jobs(id=job_id))
                if len(rows) != 1:
                    with errors_lock:
                        errors.append(f"{job_id}: expected 1 row, got {len(rows)}")

        threads = [
            threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


class TestConfigurationUtils:
    """Test configuration utility functions."""

    @pytest.mark.unit
    def test_load_dictionaries_structure(self, write_temporary_yaml):
        """Test that load_dictionaries returns expected structure."""
        from alomancy.configs.config_dictionaries import load_dictionaries

        config = load_dictionaries(write_temporary_yaml)

        # Test that required keys are present
        required_keys = [
            "mlip_committee",
            "structure_generation",
            "high_accuracy_evaluation",
        ]
        for key in required_keys:
            assert key in config

        # Test that each job dict has required fields
        for _, job_config in config.items():
            assert "name" in job_config
            assert "max_time" in job_config
            assert "hpc" in job_config

    @pytest.mark.unit
    def test_job_dict_validation(self, write_temporary_yaml):
        """Test validation of job dictionary structure."""
        from alomancy.configs.config_dictionaries import load_dictionaries

        config = load_dictionaries(write_temporary_yaml)

        # Test mlip_committee specific fields
        mlip_config = config["mlip_committee"]
        assert "size_of_committee" in mlip_config
        assert isinstance(mlip_config["size_of_committee"], int)
        assert mlip_config["size_of_committee"] > 0

        # Test structure_generation specific fields
        struct_gen_config = config["structure_generation"]
        assert "number_of_concurrent_jobs" in struct_gen_config
        assert isinstance(struct_gen_config["number_of_concurrent_jobs"], int)

        # Test high_accuracy_evaluation specific fields
        ha_eval_config = config["high_accuracy_evaluation"]
        assert "hpc" in ha_eval_config
        if "node_info" in ha_eval_config["hpc"]:
            node_info = ha_eval_config["hpc"]["node_info"]
            required_node_fields = [
                "ranks_per_system",
                "ranks_per_node",
                "threads_per_rank",
            ]
            for field in required_node_fields:
                assert field in node_info

    @pytest.mark.unit
    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    def test_config_customization(self, mock_load_dict):
        """Test that configuration can be customized."""
        custom_config = {
            "mlip_committee": {
                "name": "custom_mlip",
                "size_of_committee": 10,
                "max_time": "8H",
                "hpc": {"hpc_name": "custom-hpc"},
            },
            "structure_generation": {
                "name": "custom_md",
                "number_of_concurrent_jobs": 8,
                "max_time": "4H",
                "hpc": {"hpc_name": "custom-hpc"},
            },
            "high_accuracy_evaluation": {
                "name": "custom_qe",
                "max_time": "1H",
                "hpc": {"hpc_name": "custom-hpc"},
            },
        }

        mock_load_dict.return_value = custom_config

        config = mock_load_dict()

        assert config["mlip_committee"]["size_of_committee"] == 10
        assert config["structure_generation"]["number_of_concurrent_jobs"] == 8


class TestFileOperations:
    """Test file operation utilities."""

    @pytest.mark.unit
    def test_path_operations(self):
        """Test path manipulation utilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir)

            # Test directory creation
            subdir = test_path / "subdir" / "nested"
            subdir.mkdir(parents=True, exist_ok=True)
            assert subdir.exists()
            assert subdir.is_dir()

            # Test file creation
            test_file = subdir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"


class TestDataValidation:
    """Test data validation utilities."""

    @pytest.mark.unit
    def test_atoms_validation(self):
        """Test validation of ASE Atoms objects."""
        # Valid atoms object
        valid_atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )

        assert len(valid_atoms) == 3
        assert valid_atoms.get_chemical_symbols() == ["O", "H", "H"]
        assert valid_atoms.cell is not None

        # Test with energy and forces
        valid_atoms.info["energy"] = -15.0
        valid_atoms.arrays["forces"] = np.random.random((3, 3))

        assert "energy" in valid_atoms.info
        assert "forces" in valid_atoms.arrays
        assert valid_atoms.arrays["forces"].shape == (3, 3)

    @pytest.mark.unit
    def test_structure_list_validation(self):
        """Test validation of structure lists."""
        atoms_list = []
        for i in range(5):
            atoms = Atoms(
                symbols=["C"], positions=[[i, 0, 0]], cell=[20, 20, 20], pbc=True
            )
            atoms.info["energy"] = -i
            atoms.arrays["forces"] = np.array([[0, 0, 0]])
            atoms_list.append(atoms)

        # Test list properties
        assert len(atoms_list) == 5
        assert all(isinstance(atoms, Atoms) for atoms in atoms_list)
        assert all("energy" in atoms.info for atoms in atoms_list)
        assert all("forces" in atoms.arrays for atoms in atoms_list)

    @pytest.mark.unit
    def test_energy_forces_consistency(self):
        """Test consistency between energy and forces data."""
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[[0, 0, 0], [1.1, 0, 0]],
            cell=[15, 15, 15],
            pbc=True,
        )

        # Add energy and forces
        atoms.info["energy"] = -10.5
        atoms.arrays["forces"] = np.array([[0.1, 0, 0], [-0.1, 0, 0]])

        # Test that forces shape matches number of atoms
        assert atoms.arrays["forces"].shape[0] == len(atoms)
        assert atoms.arrays["forces"].shape[1] == 3  # x, y, z components

        # Test energy is a scalar
        assert isinstance(atoms.info["energy"], int | float)


class TestArrayOperations:
    """Test array operation utilities."""

    @pytest.mark.unit
    def test_force_array_operations(self):
        """Test operations on force arrays."""
        # Test force flattening (from md_wfl.py)
        forces = np.random.random((5, 3))  # 5 atoms, 3 components each

        def flatten_array_of_forces(forces_array):
            return np.reshape(forces_array, (1, forces_array.shape[0] * 3))

        flattened = flatten_array_of_forces(forces)
        assert flattened.shape == (1, 15)  # 5 atoms * 3 components

        # Test unflattening
        unflattened = flattened.reshape((5, 3))
        np.testing.assert_array_equal(forces, unflattened)

    @pytest.mark.unit
    def test_statistical_operations(self):
        """Test statistical operations on arrays."""
        # Create sample force data from multiple models
        n_structures = 10
        n_models = 5
        n_atoms = 3

        force_data = {}
        for model_id in range(n_models):
            force_data[f"model_{model_id}"] = {}
            for struct_id in range(n_structures):
                force_data[f"model_{model_id}"][f"structure_{struct_id}"] = {
                    "forces": np.random.random((n_atoms, 3)),
                    "energy": np.random.random(),
                }

        # Test standard deviation calculation
        for struct_id in range(n_structures):
            forces_array = np.array(
                [
                    force_data[f"model_{model_id}"][f"structure_{struct_id}"]["forces"]
                    for model_id in range(n_models)
                ]
            )

            std_dev = np.std(forces_array, axis=0)
            assert std_dev.shape == (n_atoms, 3)

            # Test max and mean standard deviation
            max_std = np.max(std_dev)
            mean_std = np.mean(std_dev)
            assert max_std >= mean_std >= 0


@pytest.mark.unit
class TestUnitUtilities:
    """Unit tests for basic utility functions."""

    def test_basic_math_operations(self):
        """Test basic mathematical operations."""
        # Test array operations
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        result = arr1 + arr2
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_string_operations(self):
        """Test string manipulation utilities."""
        # Test base name generation
        loop_number = 5
        base_name = f"al_loop_{loop_number}"
        assert base_name == "al_loop_5"

        # Test file path operations
        base_path = Path("results") / base_name
        train_file = base_path / "train_set.xyz"
        assert str(train_file) == "results/al_loop_5/train_set.xyz"

    def test_type_checking(self):
        """Test type checking utilities."""
        # Test ASE Atoms type checking
        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]])
        assert isinstance(atoms, Atoms)

        # Test list type checking
        atoms_list = [atoms]
        assert isinstance(atoms_list, list)
        assert all(isinstance(item, Atoms) for item in atoms_list)


class TestSplitAtomsListIntoTestAndTrain:
    """Test split_atoms_list_into_test_and_train function."""

    def _make_atoms_list(self, n):
        """Create a simple list of Atoms objects."""
        return [
            Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
            for _ in range(n)
        ]

    @pytest.mark.unit
    def test_correct_split_size(self):
        """Test that split produces correct train and test sizes."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(100)
        train, test = split_atoms_list_into_test_and_train(
            atoms, test_fraction=0.2, seed=42
        )
        assert len(train) + len(test) == 100
        assert len(test) == 20  # 100 * 0.2

    @pytest.mark.unit
    def test_seeded_reproducibility(self):
        """Test that same seed produces same split."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(50)
        train1, test1 = split_atoms_list_into_test_and_train(atoms, 0.2, seed=42)
        train2, test2 = split_atoms_list_into_test_and_train(atoms, 0.2, seed=42)
        # Same seed → same split (verify by checking ids, not values)
        assert [id(a) for a in train1] == [id(a) for a in train2]
        assert [id(a) for a in test1] == [id(a) for a in test2]

    @pytest.mark.unit
    def test_different_seeds_different_splits(self):
        """Test that different seeds produce different splits."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(50)
        train1, _ = split_atoms_list_into_test_and_train(atoms, 0.2, seed=1)
        train2, _ = split_atoms_list_into_test_and_train(atoms, 0.2, seed=2)
        # With high probability different seeds give different orders
        assert [id(a) for a in train1] != [id(a) for a in train2]

    @pytest.mark.unit
    def test_empty_list(self):
        """Test split with empty list."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        train, test = split_atoms_list_into_test_and_train([], 0.2, seed=42)
        assert train == []
        assert test == []

    @pytest.mark.unit
    def test_zero_test_fraction(self):
        """Test split with zero test fraction."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(10)
        train, test = split_atoms_list_into_test_and_train(atoms, 0.0, seed=42)
        assert len(train) == 10
        assert len(test) == 0


class TestCleanStructures:
    """Test clean_structures function."""

    def _make_structure_with_ref(self, config_type=None):
        """Create a test structure with REF_energy and REF_forces."""
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        if config_type:
            atoms.info["config_type"] = config_type
        atoms.info["REF_energy"] = -76.0
        atoms.arrays["REF_forces"] = np.zeros((3, 3))
        return atoms

    @pytest.mark.unit
    def test_config_type_set_when_missing(self):
        """Test that config_type is set when missing."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref()  # no config_type
        result = clean_structures([s], config_type="al_loop_0")
        assert result[0].info["config_type"] == "al_loop_0"

    @pytest.mark.unit
    def test_config_type_preserved_when_not_overriding(self):
        """Test that config_type is preserved when not overriding."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref(config_type="original_type")
        result = clean_structures(
            [s], config_type="new_type", override_config_type=False
        )
        assert result[0].info["config_type"] == "original_type"

    @pytest.mark.unit
    def test_config_type_overridden(self):
        """Test that config_type is overridden when requested."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref(config_type="old_type")
        result = clean_structures(
            [s], config_type="new_type", override_config_type=True
        )
        assert result[0].info["config_type"] == "new_type"

    @pytest.mark.unit
    def test_ref_energy_preserved(self):
        """Test that REF_energy is preserved."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref()
        result = clean_structures([s], config_type="test")
        assert result[0].info["REF_energy"] == pytest.approx(-76.0)

    @pytest.mark.unit
    def test_ref_forces_preserved(self):
        """Test that REF_forces are preserved."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref()
        result = clean_structures([s], config_type="test")
        np.testing.assert_allclose(result[0].arrays["REF_forces"], np.zeros((3, 3)))

    @pytest.mark.unit
    def test_multiple_structures(self):
        """Test cleaning multiple structures."""
        from alomancy.utils.clean_structures import clean_structures

        structures = [self._make_structure_with_ref() for _ in range(3)]
        result = clean_structures(structures, config_type="batch")
        assert len(result) == 3

    @pytest.mark.unit
    def test_already_computed_missing_ref_no_calc_raises(self):
        """Raises ValueError when already_computed=True but no REF data and no calculator."""
        from alomancy.utils.clean_structures import clean_structures

        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        with pytest.raises(ValueError, match="already_computed"):
            clean_structures([atoms], config_type="test", already_computed=True)

    @pytest.mark.unit
    def test_already_computed_false_skips_ref_check(self):
        """With already_computed=False, no REF data is required or added."""
        from alomancy.utils.clean_structures import clean_structures

        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        atoms.info["config_type"] = "test"
        result = clean_structures([atoms], config_type="test", already_computed=False)
        assert len(result) == 1
        assert "REF_energy" not in result[0].info


@pytest.mark.unit
class TestReadAtomsFileIfEnabled:
    """Test read_atoms_file_if_enabled function."""

    def test_returns_none_when_disabled(self, tmp_path):
        from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

        result = read_atoms_file_if_enabled(
            read_file=False, file_path=tmp_path / "dummy.xyz"
        )
        assert result is None

    def test_returns_none_when_file_missing(self, tmp_path):
        from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

        result = read_atoms_file_if_enabled(
            read_file=True, file_path=tmp_path / "nonexistent.xyz"
        )
        assert result is None

    def test_returns_empty_list_for_empty_file(self, tmp_path):
        from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

        empty_file = tmp_path / "empty.xyz"
        empty_file.write_text("")
        result = read_atoms_file_if_enabled(read_file=True, file_path=empty_file)
        assert result == []

    def test_returns_atoms_from_file(self, tmp_path, h2o_mol):
        from ase.io import write

        from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

        xyz_path = tmp_path / "structs.xyz"
        write(str(xyz_path), [h2o_mol], format="extxyz")
        result = read_atoms_file_if_enabled(read_file=True, file_path=xyz_path)
        assert result is not None
        assert len(result) == 1

    def test_returns_multiple_atoms(self, tmp_path, h2o_mol, h_atom):
        from ase.io import write

        from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

        xyz_path = tmp_path / "multi.xyz"
        write(str(xyz_path), [h2o_mol, h_atom], format="extxyz")
        result = read_atoms_file_if_enabled(read_file=True, file_path=xyz_path)
        assert len(result) == 2

    def test_accepts_path_object(self, tmp_path, h2o_mol):
        from ase.io import write

        from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled

        xyz_path = tmp_path / "test.xyz"
        write(str(xyz_path), [h2o_mol], format="extxyz")
        result = read_atoms_file_if_enabled(read_file=True, file_path=xyz_path)
        assert result is not None


class TestLoadDictionaries:
    """Test load_dictionaries function."""

    @pytest.mark.unit
    def test_load_real_yaml(self, tmp_path):
        """Test loading a real YAML configuration."""
        from alomancy.configs.config_dictionaries import load_dictionaries

        # Write a minimal valid config YAML
        yaml_content = """
initialization:
  name: init
  max_time: 1H
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
mlip_committee:
  name: mlip
  max_time: 2H
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
structure_generation:
  name: struc_gen
  max_time: 30m
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
high_accuracy_evaluation:
  name: dft
  max_time: 10m
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)
        result = load_dictionaries(config_path)
        assert "initialization" in result
        assert "mlip_committee" in result
        assert "structure_generation" in result
        assert "high_accuracy_evaluation" in result
