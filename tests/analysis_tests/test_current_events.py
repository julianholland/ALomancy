"""analysis/report/current_events.py: the loop report shows only failures
that still stand -- a step's latest attempt, minus failures whose retry of
the same item succeeded."""

from pathlib import Path

import pytest

from alomancy.analysis.report.current_events import current_events


def _event(time, code, *, attempt, phase, loop=0, level="INFO", data=None):
    return {
        "time": time,
        "level": level,
        "message": code or "",
        "event": code,
        "loop": loop,
        "phase": phase,
        "attempt": attempt,
        "data": data,
    }


A1, A2 = "2026-10-05T14:00:00-1", "2026-10-05T16:00:00-2"


def _codes(events):
    return [e["event"] for e in events]


@pytest.mark.unit
def test_a_step_rerun_by_a_later_attempt_shows_only_that_attempt():
    events = [
        _event(
            "2026-10-05T14:01:00",
            "phase_started",
            attempt=A1,
            phase="generate_structures",
        ),
        _event(
            "2026-10-05T14:02:00",
            "job_failed",
            attempt=A1,
            phase="generate_structures",
            level="WARNING",
        ),
        _event(
            "2026-10-05T14:03:00",
            "md_unfinished",
            attempt=A1,
            phase="generate_structures",
            level="WARNING",
        ),
        _event(
            "2026-10-05T16:01:00",
            "phase_started",
            attempt=A2,
            phase="generate_structures",
        ),
        _event(
            "2026-10-05T16:30:00", "md_summary", attempt=A2, phase="generate_structures"
        ),
    ]
    assert _codes(current_events(events, 0, None)) == ["md_summary"]


@pytest.mark.unit
def test_a_step_completed_earlier_keeps_its_warnings():
    """Training finished in attempt 1 and wasn't re-run, so its warning
    stays even though attempt 2 ran later steps."""
    events = [
        _event("2026-10-05T14:01:00", "phase_started", attempt=A1, phase="train_mlip"),
        _event(
            "2026-10-05T14:05:00",
            "fit_missing",
            attempt=A1,
            phase="train_mlip",
            level="WARNING",
            data={"items": ["fit_2"]},
        ),
        _event(
            "2026-10-05T16:01:00",
            "phase_started",
            attempt=A2,
            phase="generate_structures",
        ),
        _event(
            "2026-10-05T16:30:00", "md_summary", attempt=A2, phase="generate_structures"
        ),
    ]
    assert _codes(current_events(events, 0, None)) == ["fit_missing", "md_summary"]


@pytest.mark.unit
def test_a_failure_whose_retry_succeeded_is_dropped():
    events = [
        _event(
            "t1", "job_failed", attempt=A1, phase="train_mlip", data={"item": "fit_0"}
        ),
        _event(
            "t2", "job_failed", attempt=A1, phase="train_mlip", data={"item": "fit_1"}
        ),
        _event(
            "t3",
            "fit_retry",
            attempt=A1,
            phase="train_mlip",
            data={"items": ["fit_0", "fit_1"]},
        ),
        _event(
            "t4",
            "job_succeeded",
            attempt=A1,
            phase="train_mlip",
            data={"item": "fit_0"},
        ),
        _event(
            "t5", "job_failed", attempt=A1, phase="train_mlip", data={"item": "fit_1"}
        ),
        _event(
            "t6",
            "fit_missing",
            attempt=A1,
            phase="train_mlip",
            data={"items": ["fit_1"]},
        ),
    ]
    kept = current_events(events, 0, None)
    # fit_0 recovered on retry; fit_1 never did, so its failures and the
    # retry that included it stay.
    assert [(e["event"], (e["data"] or {}).get("item")) for e in kept] == [
        ("job_failed", "fit_1"),
        ("fit_retry", None),
        ("job_failed", "fit_1"),
        ("fit_missing", None),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replacement_stepped", "kept"),
    [(True, []), (False, ["md_no_steps", "md_unfinished"])],
)
def test_md_replacements(replacement_stepped, kept):
    """A run that took no MD step is replaced; the replacement fills the
    same item. md_no_steps stands only if md_unfinished names the run."""
    events = [
        _event(
            "t1",
            "job_succeeded",
            attempt=A1,
            phase="generate_structures",
            data={"item": "md_run_3"},
        ),
        _event(
            "t2",
            "md_no_steps",
            attempt=A1,
            phase="generate_structures",
            data={"items": ["md_run_3"]},
        ),
        _event(
            "t3",
            "job_succeeded",
            attempt=A1,
            phase="generate_structures",
            data={"item": "md_run_3"},
        ),
    ]
    if not replacement_stepped:
        events.append(
            _event(
                "t4",
                "md_unfinished",
                attempt=A1,
                phase="generate_structures",
                data={"items": ["md_run_3"]},
            )
        )
    assert _codes(current_events(events, 0, None)) == kept


@pytest.mark.unit
def test_other_loops_and_internal_events_are_left_out():
    events = [
        _event("t1", "phase_started", attempt=A1, phase="train_mlip"),
        _event("t2", "jobs_started", attempt=A1, phase="train_mlip"),
        _event("t3", "jobs_started", attempt=A1, phase="train_mlip", loop=1),
        _event("t4", "phase_completed", attempt=A1, phase="train_mlip"),
    ]
    assert _codes(current_events(events, 0, None)) == ["jobs_started"]


def _old(time, code, level="INFO"):
    """An event as written before events carried attempt/phase."""
    return {
        "time": time,
        "level": level,
        "message": "",
        "event": code,
        "loop": 0,
        "data": None,
    }


@pytest.mark.unit
def test_old_event_files_are_sorted_into_attempts_from_the_log(tmp_path):
    """Shaped like the SevenNet test run: attempt 1 failed training;
    attempt 2 trained, then MD failed; attempt 3 re-ran MD successfully,
    then DFT. Only the last attempt of each step remains."""
    log = tmp_path / "alomancy.log"
    lines = []

    def attempt(ts):
        lines.extend(
            [
                f"{ts} [INFO    ] alomancy.core.active_learning_workflow: ",
                "=" * 70,
                "ALomancy Workflow Summary (v1.0.1)",
            ]
        )

    def complete(ts, phase):
        lines.append(
            f"{ts} [DEBUG   ] alomancy.core.active_learning_workflow: "
            f"Phase {phase} marked complete for al_loop_0."
        )

    attempt("2026-10-03 15:45:00")
    attempt("2026-10-03 15:53:00")
    complete("2026-10-04 15:19:42", "train_mlip")
    attempt("2026-10-05 16:39:36")
    complete("2026-10-05 18:20:40", "generate_structures")
    complete("2026-10-06 03:06:32", "high_accuracy_eval")
    log.write_text("\n".join(lines) + "\n")

    events = [
        _old("2026-10-03T15:51:19", "job_died", "WARNING"),
        _old("2026-10-03T15:51:19", "fit_retry", "WARNING"),
        _old("2026-10-03T15:58:22", "jobs_started"),
        _old("2026-10-04T15:20:01", "jobs_started"),
        _old("2026-10-05T14:16:59", "job_failed", "WARNING"),
        _old("2026-10-05T14:18:42", "md_no_steps", "WARNING"),
        _old("2026-10-05T14:20:41", "md_unfinished", "WARNING"),
        _old("2026-10-05T16:44:39", "jobs_started"),
        _old("2026-10-05T18:19:42", "md_summary"),
        _old("2026-10-05T18:20:49", "jobs_started"),
        _old("2026-10-06T03:06:32", "dft_summary"),
        _old("2026-10-06T03:10:47", "redundancy_flagged"),
    ]

    kept = current_events(events, 0, Path(log))

    assert [(e["event"], e["phase"]) for e in kept] == [
        ("jobs_started", "train_mlip"),
        ("jobs_started", "generate_structures"),
        ("md_summary", "generate_structures"),
        ("jobs_started", "high_accuracy_eval"),
        ("dft_summary", "high_accuracy_eval"),
        ("redundancy_flagged", None),
    ]
