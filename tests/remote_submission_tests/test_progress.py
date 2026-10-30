"""Unit tests for alomancy.remote_submission.progress (live job status)."""

import json
import logging
import time
from types import SimpleNamespace

import pytest
from expyre import config as expyre_config
from expyre.jobsdb import JobsDB
from rich.console import Console

from alomancy.remote_submission import executor as executor_module
from alomancy.remote_submission import progress
from alomancy.remote_submission.executor import RemoteJobExecutor
from alomancy.remote_submission.progress import (
    JobTracker,
    ProgressMonitor,
    bar_segments,
    format_duration,
    loop_from_paths,
    render,
    summary_line,
    write_status,
)


def _text(renderable, width=100) -> str:
    console = Console(record=True, width=width, color_system=None)
    console.print(renderable)
    return console.export_text()


def _tracker_with_batch(n=4, systems=("raven",), check_interval=10.0):
    tracker = JobTracker()
    jobs = [(f"go_{i}", systems[i % len(systems)]) for i in range(n)]
    batch = tracker.register(
        "high_accuracy_evaluation", jobs, loop=0, check_interval=check_interval
    )
    return tracker, batch


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


# --- tracker -------------------------------------------------------------------


@pytest.mark.unit
def test_job_states_and_counts():
    tracker, batch = _tracker_with_batch(n=5, systems=("raven", "raccoon"))
    tracker.submitted(batch, 0, "100", "id0")
    tracker.submitted(batch, 1, "101", "id1")
    tracker.submitted(batch, 2, "102", "id2")
    tracker.synced("raven", {"id0": "running", "id2": "held"})
    tracker.finished(batch, 1, ok=True)

    snap = tracker.snapshot()["batches"][0]
    assert snap["phase"] == "high_accuracy_evaluation"
    assert snap["loop"] == 0
    assert snap["counts"] == {
        "done": 1,
        "running": 1,
        "queued": 1,
        "to_submit": 2,
        "failed": 0,
    }
    assert snap["hpcs"]["raven"]["running"] == 1
    assert snap["hpcs"]["raven"]["queued"] == 1
    assert snap["hpcs"]["raccoon"]["done"] == 1
    assert snap["slowest_running"]["item"] == "go_0"
    assert snap["slowest_running"]["remote_id"] == "100"

    tracker.finished(batch, 0, ok=False)
    snap = tracker.snapshot()["batches"][0]
    assert snap["counts"]["failed"] == 1
    assert snap["slowest_running"] is None


@pytest.mark.unit
def test_synced_reads_remote_statuses_from_a_real_jobs_db(tmp_path, monkeypatch):
    db = JobsDB(tmp_path / "jobs.db")
    db.add(
        "id0", "go_0", str(tmp_path), status="submitted", system="raven", remote_id="1"
    )
    db.add(
        "id1", "go_1", str(tmp_path), status="started", system="raven", remote_id="2"
    )
    db.add(
        "id2", "go_2", str(tmp_path), status="succeeded", system="raven", remote_id="3"
    )
    db.update("id0", remote_status="queued")
    db.update("id1", remote_status="running")
    monkeypatch.setattr(expyre_config, "db", db)

    statuses = executor_module._remote_statuses("raven")
    assert statuses == {"id0": "queued", "id1": "running"}

    tracker, batch = _tracker_with_batch(n=2)
    tracker.submitted(batch, 0, "1", "id0")
    tracker.submitted(batch, 1, "2", "id1")
    tracker.synced("raven", statuses)
    counts = tracker.snapshot()["batches"][0]["counts"]
    assert counts["queued"] == 1
    assert counts["running"] == 1


@pytest.mark.unit
def test_oldest_queued_and_stale_last_data():
    tracker, batch = _tracker_with_batch(n=2, check_interval=10.0)
    tracker.submitted(batch, 0, "1", "id0")
    tracker.synced("raven", {"id0": "queued"})
    now = time.time()

    fresh = tracker.snapshot(now + 5)["batches"][0]["hpcs"]["raven"]
    assert not fresh["stale"]
    assert 4 < fresh["oldest_queued_s"] < 7

    stale = tracker.snapshot(now + 31)["batches"][0]["hpcs"]["raven"]
    assert stale["stale"]
    assert stale["last_data_s"] > 30


@pytest.mark.unit
def test_a_failed_sync_is_recorded_without_advancing_last_data():
    tracker, batch = _tracker_with_batch(n=1)
    tracker.submitted(batch, 0, "1", "id0")
    tracker.synced("raven", {})
    first = tracker.last_sync["raven"]
    tracker.sync_failed("raven", RuntimeError("ssh: connection refused"))

    assert tracker.last_sync["raven"] == first
    hpc = tracker.snapshot()["batches"][0]["hpcs"]["raven"]
    assert "connection refused" in hpc["error"]
    tracker.synced("raven", {})
    assert tracker.snapshot()["batches"][0]["hpcs"]["raven"]["error"] is None


@pytest.mark.unit
def test_unregistered_batches_disappear():
    tracker, batch = _tracker_with_batch()
    assert tracker.active()
    tracker.unregister(batch)
    assert not tracker.active()
    assert tracker.snapshot()["batches"] == []


@pytest.mark.unit
def test_loop_from_paths():
    assert loop_from_paths(["results/al_loop_3/high_accuracy_evaluation/x"]) == 3
    assert loop_from_paths(["results/initialization/x"]) is None


# --- rendering -----------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "counts",
    [
        {"done": 31, "running": 12, "queued": 8, "to_submit": 49, "failed": 0},
        {"done": 0, "running": 1, "queued": 0, "to_submit": 99, "failed": 0},
        {"done": 3, "running": 0, "queued": 0, "to_submit": 0, "failed": 2},
    ],
)
def test_bar_segments_fill_the_width(counts):
    segments = bar_segments(counts, 40)
    assert sum(segments.values()) == 40
    for state in ("running", "queued", "to_submit"):
        assert (segments[state] > 0) == (counts[state] > 0)


@pytest.mark.unit
def test_render_shows_phase_counts_hpcs_and_stale_flag():
    tracker, batch = _tracker_with_batch(n=3)
    tracker.submitted(batch, 0, "31549232", "id0")
    tracker.synced("raven", {"id0": "running"})
    tracker.submitted(batch, 1, "31549233", "id1")
    text = _text(render(tracker.snapshot(time.time() + 40)))

    assert "high_accuracy_evaluation · loop 0 ── 3 jobs" in text
    assert "done 0 · running 1 · queued 1 · to submit 1" in text
    assert "raven" in text
    assert "⚠" in text
    assert "slowest running: go_0 (raven, 31549232)" in text


@pytest.mark.unit
def test_summary_line():
    tracker, batch = _tracker_with_batch(n=2)
    tracker.submitted(batch, 0, "1", "id0")
    tracker.synced("raven", {"id0": "queued"})
    line = summary_line(tracker.snapshot())
    assert line.startswith(
        "high_accuracy_evaluation loop 0: 0 done, 0 running, 1 queued, 1 to submit"
    )
    assert "raven: last data 0 s ago, oldest queued 0 s" in line


@pytest.mark.unit
def test_format_duration():
    assert format_duration(None) == "-"
    assert format_duration(12) == "12 s"
    assert format_duration(61 * 3600 + 4 * 60) == "2 d 13 h"
    assert format_duration(3 * 3600 + 5 * 60) == "3 h 05 m"


# --- status file and monitor ----------------------------------------------------


@pytest.mark.unit
def test_status_file_is_written_atomically(tmp_path):
    tracker, _ = _tracker_with_batch()
    path = tmp_path / "results" / "remote_status.json"
    write_status(tracker.snapshot(), path)  # no results/ yet: skipped
    assert not path.exists()

    path.parent.mkdir()
    write_status(tracker.snapshot(), path)
    assert json.loads(path.read_text())["batches"][0]["total"] == 4
    assert [p.name for p in path.parent.iterdir()] == ["remote_status.json"]


@pytest.mark.unit
def test_monitor_logs_summaries_without_a_terminal(tmp_path):
    tracker, _ = _tracker_with_batch()
    path = tmp_path / "remote_status.json"
    handler = _Collector()
    progress.logger.addHandler(handler)
    level = progress.logger.level
    progress.logger.setLevel(logging.INFO)
    try:
        monitor = ProgressMonitor(
            tracker, live=False, tick=0.02, summary_every=0.05, status_file=path
        )
        with monitor:
            time.sleep(0.3)
    finally:
        progress.logger.removeHandler(handler)
        progress.logger.setLevel(level)

    assert any(m.startswith("Remote jobs: high_accuracy") for m in handler.messages)
    assert json.loads(path.read_text())["finished"] is True
    assert monitor._thread is not None
    assert not monitor._thread.is_alive()


# --- executor integration --------------------------------------------------------


class _Job:
    """The ExPyRe surface run_all_jobs_bounded uses, with real timing."""

    def __init__(self, name, stage_dir, fail=False):
        self.name = name
        self.id = f"{name}_id"
        self.remote_id = None
        self.stage_dir = stage_dir
        self.status = "created"
        self.fail = fail
        self.get_results_kwargs: dict = {}
        self._alomancy_item = name
        self._alomancy_loop = 2

    def start(self, **kwargs):
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.remote_id = f"r_{self.name}"
        self.status = "submitted"

    def get_results(self, **kwargs):
        self.get_results_kwargs = kwargs
        time.sleep(0.05)
        if self.fail:
            self.status = "failed"
            raise RuntimeError("remote error")
        self.status = "succeeded"
        return self.name, "", ""


@pytest.mark.unit
def test_executor_reports_to_the_tracker_and_silences_expyre(tmp_path, monkeypatch):
    seen: list[dict] = []
    original = progress.TRACKER.finished

    def finished(batch_id, index, ok):
        original(batch_id, index, ok)
        seen.append(progress.TRACKER.snapshot()["batches"][0])

    monkeypatch.setattr(progress.TRACKER, "finished", finished)
    status_file = tmp_path / "results" / "remote_status.json"
    status_file.parent.mkdir()
    monkeypatch.setattr(progress, "STATUS_FILE", status_file)

    jobs = [_Job("a", tmp_path / "a"), _Job("b", tmp_path / "b", fail=True)]
    remote_info = SimpleNamespace(
        resources={},
        sys_name="test-hpc",
        job_name="high_accuracy_evaluation",
        timeout=10,
        check_interval=0.001,
        max_num_of_concurrent_jobs=1,
    )
    executor = RemoteJobExecutor(remote_info)
    executor.jobs = jobs
    assert executor.run_all_jobs_bounded() == ["a", None]

    assert all(job.get_results_kwargs["quiet"] is True for job in jobs)
    assert seen[-1]["counts"]["done"] == 1
    assert seen[-1]["counts"]["failed"] == 1
    assert seen[-1]["loop"] == 2
    assert seen[-1]["hpcs"]["test-hpc"]["failed"] == 1
    assert not progress.TRACKER.active()
    final = json.loads(status_file.read_text())
    assert final["finished"] is True
