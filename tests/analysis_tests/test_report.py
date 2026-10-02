"""Tests for analysis/report: the per-loop Markdown report, its statistics,
the editable trigger -> suggestion table, the JSONL event log it reads,
and the `alomancy report` CLI."""

import json
import logging
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from alomancy.analysis.report import write_loop_report
from alomancy.analysis.report.rules import (
    DEFAULT_SUGGESTIONS,
    evaluate,
    load_suggestions,
)
from alomancy.analysis.report.stats import collect_loop_stats
from alomancy.analysis.report.triggers import TRIGGERS
from alomancy.core.active_learning_workflow import LoopContext
from alomancy.core.entry import ALomancy
from alomancy.mlip.evaluation import prediction_metrics, save_evaluation
from alomancy.utils.logging_config import read_events, set_current_loop

_LOGGER = logging.getLogger("alomancy.tests.report")


# ---------------------------------------------------------------------------
# A small finished loop: database, one evaluated fit, DFT results, events
# ---------------------------------------------------------------------------


def _labelled(
    d: float, config_type: str, al_loop: int | None = None, force: float = 0.0
) -> Atoms:
    a = Atoms("H2", positions=[[0, 0, 0], [d, 0, 0]], cell=[10] * 3, pbc=True)
    a.info["config_type"] = config_type
    a.info["REF_energy"] = -1.0 - d
    forces = np.zeros((2, 3))
    forces[0, 0] = force
    a.arrays["REF_forces"] = forces
    if al_loop is not None:
        a.info["al_loop"] = al_loop
    return a


def _write_fit(loop: int, fit_idx: int, test_error: float) -> None:
    fit_dir = Path(f"results/al_loop_{loop}/training/fit_{fit_idx}")
    fit_dir.mkdir(parents=True)
    model = fit_dir / "training_stagetwo.model"
    model.write_bytes(f"model {loop}-{fit_idx}".encode())

    def split(error: float) -> dict:
        a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
        a.info.update(REF_energy=-8.0, model_energy=-8.0 + 2 * error, config_type="d")
        a.set_array("REF_forces", np.zeros((2, 3)))
        a.set_array("model_forces", np.ones((2, 3)) * error)
        return prediction_metrics([a])

    save_evaluation(fit_dir, model, {"train": split(0.05), "test": split(test_error)})


def _write_dft_results(loop: int) -> None:
    results = []
    for steps, converged in ((50, False), (10, True), (12, True)):
        a = _labelled(0.8, "random_selection", loop, force=0.04)
        a.info.update(
            geometry_converged=converged,
            geometry_steps=steps,
            geometry_max_steps=50,
            dft_wall_time_s=600.0,
        )
        results.append(a)
    sp = _labelled(0.9, "random_selection", loop)
    sp.info["dft_wall_time_s"] = 60.0
    results.append(sp)
    write(
        f"results/al_loop_{loop}/high_accuracy_eval_results.xyz",
        results,
        format="extxyz",
    )


@pytest.fixture
def jobs_dict(minimal_jobs_dict):
    minimal_jobs_dict["general"] = {
        "al_workflow": "random_selection",
        "elements": ["H"],
        "num_of_al_loops": 2,
        "plots": True,
        "dataset_kwargs": {"target_config_types": ["IsolatedAtom"], "test_ratio": 0.1},
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["num_of_models_in_committee"]
    minimal_jobs_dict["structure_generation"]["desired_num_of_structures"] = 4
    return minimal_jobs_dict


@pytest.fixture
def finished_loop(tmp_path, monkeypatch, jobs_dict, shared_db):
    """Loop 0 of a random_selection run, finished, in tmp_path."""
    monkeypatch.chdir(tmp_path)
    wf = ALomancy(jobs_dict).workflow
    wf.db = shared_db
    shared_db.add_structures(
        [_labelled(0.7 + 0.1 * i, "init_dimer") for i in range(3)],
        split="train",
        skip_duplicates=False,
    )
    shared_db.add_structures(
        [_labelled(1.5, "init_MP")], split="test", skip_duplicates=False
    )
    shared_db.add_structures(
        [_labelled(0.8 + 0.01 * i, "random_selection", 0) for i in range(4)],
        split="train",
        skip_duplicates=False,
    )
    shared_db.flag_as_duplicates([4])
    shared_db.partition.set_metadata_bulk(
        {5: {"is_quality_filtered": True, "quality_filter_reasons": ["high_force"]}},
        use_indices=True,
    )
    _write_fit(0, 0, test_error=0.3)
    _write_dft_results(0)
    set_current_loop(0)
    _LOGGER.info(
        "summary",
        extra={"event": "dft_summary", "data": {"submitted": 5, "returned": 4}},
    )
    _LOGGER.info("jobs", extra={"event": "jobs_started", "data": {"n": 10}})
    _LOGGER.warning("Job 3 failed: boom", extra={"event": "job_failed"})
    _LOGGER.warning("something odd happened")
    set_current_loop(None)
    return wf


# ---------------------------------------------------------------------------
# Remote DFT fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_go_and_sp_record_steps_and_wall_time(tmp_path):
    from ase.calculators.emt import EMT

    from alomancy.utils.dft_utils import _run_go, _run_sp

    def dimer() -> Atoms:
        return Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10] * 3, pbc=True)

    go = _run_go(
        dimer(),
        str(tmp_path / "go"),
        {"name": "go", "max_num_of_relax_steps": 7},
        lambda a, j, d: EMT(),
    )
    assert isinstance(go.info["geometry_steps"], int)
    assert 0 < go.info["geometry_steps"] <= 7
    assert go.info["geometry_max_steps"] == 7
    assert go.info["geometry_fmax_target"] == pytest.approx(0.05)
    assert go.info["dft_wall_time_s"] >= 0

    sp = _run_sp(dimer(), str(tmp_path / "sp"), {"name": "sp"}, lambda a, j, d: EMT())
    assert sp.info["dft_wall_time_s"] >= 0
    assert "geometry_steps" not in sp.info


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_event_log_records_coded_events_and_warnings(finished_loop):
    events = read_events("results/events.jsonl")
    by_message = {e["message"]: e for e in events}

    assert by_message["Job 3 failed: boom"]["event"] == "job_failed"
    assert by_message["Job 3 failed: boom"]["loop"] == 0
    assert by_message["something odd happened"]["event"] is None
    assert by_message["summary"]["data"] == {"submitted": 5, "returned": 4}
    # Plain INFO records without an event code are not written.
    _LOGGER.info("routine progress")
    assert all(
        e["message"] != "routine progress" for e in read_events("results/events.jsonl")
    )


@pytest.mark.unit
def test_read_events_skips_truncated_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\n{"event": \n')
    assert read_events(path) == [{"event": "a"}]
    assert read_events(tmp_path / "missing.jsonl") == []


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stats_composition_redundancy_and_filters(finished_loop):
    dataset = collect_loop_stats(finished_loop, 0)["dataset"]

    assert dataset["composition"]["init_dimer"] == {"train": 3}
    assert dataset["composition"]["init_MP"] == {"test": 1}
    assert dataset["composition"]["random_selection"] == {
        "train": 2,
        "redundant": 1,
        "quality_filtered": 1,
    }
    assert dataset["redundant_total"] == 1
    assert dataset["new_this_loop"] == {
        "train": 2,
        "redundant": 1,
        "quality_filtered": 1,
    }
    assert dataset["new_quality_filter_reasons"] == {"high_force": 1}


@pytest.mark.unit
def test_stats_old_loop_ignores_later_structures(finished_loop, shared_db):
    shared_db.add_structures(
        [_labelled(2.0, "random_selection", 1)], split="train", skip_duplicates=False
    )
    assert (
        "random_selection"
        in collect_loop_stats(finished_loop, 0)["dataset"]["composition"]
    )
    assert collect_loop_stats(finished_loop, 0)["dataset"]["totals"]["train"] == 5


@pytest.mark.unit
def test_stats_dft_and_model(finished_loop):
    stats = collect_loop_stats(finished_loop, 0)
    dft = stats["dft"]

    assert (dft["submitted"], dft["returned"]) == (5, 4)
    assert (dft["n_go"], dft["n_sp"], dft["go_not_converged"]) == (3, 1, 1)
    assert dft["go_steps"]["mean"] == pytest.approx(24.0)
    assert dft["go_step_budget"] == 50
    assert dft["wall_time_s"]["go"]["mean"] == pytest.approx(600.0)
    assert dft["wall_time_s"]["sp"]["mean"] == pytest.approx(60.0)
    assert stats["model"]["best_fit_idx"] == 0
    assert stats["model"]["errors"]["test"]["mae_f"] == pytest.approx(0.3)
    assert stats["events"]["by_event"]["job_failed"] == 1
    json.dumps(stats)  # report.json must be JSON-safe


@pytest.mark.unit
def test_stats_without_new_dft_fields(finished_loop):
    results = [_labelled(0.8, "random_selection", 0)]
    results[0].info["geometry_converged"] = True
    write("results/al_loop_0/high_accuracy_eval_results.xyz", results, format="extxyz")

    dft = collect_loop_stats(finished_loop, 0)["dft"]

    assert dft["go_steps"] is None
    assert dft["wall_time_s"] == {}


# ---------------------------------------------------------------------------
# Triggers and the suggestions file
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_packaged_suggestions_cover_every_trigger_and_nothing_else():
    entries = yaml.safe_load(DEFAULT_SUGGESTIONS.read_text())
    assert set(entries) == set(TRIGGERS)
    for name, entry in entries.items():
        assert entry["severity"] in ("info", "warning", "error"), name
        assert float(entry["threshold"]) >= 0, name
        assert entry["suggestion"].strip(), name


@pytest.mark.unit
def test_findings_from_a_finished_loop(finished_loop):
    stats = collect_loop_stats(finished_loop, 0)
    findings, notes = evaluate(stats, load_suggestions())
    fired = {f.trigger: f for f in findings}

    # 1/3 GO hit the budget; 1/5 DFT structures missing; 1/10 jobs failed;
    # 1/4 new structures redundant (under the 50% threshold).
    assert "go_not_converged" in fired
    assert "1/3 geometry optimisations (33%)" in fired["go_not_converged"].message
    assert "dft_partial_failure" in fired
    assert "remote_jobs_failed" in fired
    assert "redundancy_removed_new" not in fired
    assert notes == []
    assert [f.severity for f in findings] == sorted(
        (f.severity for f in findings), key=("error", "warning", "info").index
    )


@pytest.mark.unit
def test_override_file_replaces_one_entry_and_keeps_the_rest(finished_loop, tmp_path):
    override = tmp_path / "mine.yaml"
    override.write_text(
        "go_not_converged:\n  threshold: 0.5\n"
        "dft_partial_failure:\n  suggestion: 'Lost {n} of {total}.'\n"
    )
    suggestions = load_suggestions(override)
    findings, _ = evaluate(collect_loop_stats(finished_loop, 0), suggestions)
    fired = {f.trigger: f for f in findings}

    assert "go_not_converged" not in fired  # 33% < new 50% threshold
    assert fired["dft_partial_failure"].message == "Lost 1 of 5."
    assert fired["dft_partial_failure"].severity == "error"  # kept from packaged


@pytest.mark.unit
def test_bad_suggestions_warn_instead_of_raising(finished_loop, tmp_path):
    override = tmp_path / "bad.yaml"
    override.write_text(
        "no_such_trigger:\n  threshold: 0\n  suggestion: x\n"
        "go_not_converged:\n  suggestion: 'uses {missing_value}'\n"
    )
    suggestions = load_suggestions(override)
    del suggestions["fit_retries"]

    findings, notes = evaluate(collect_loop_stats(finished_loop, 0), suggestions)

    assert any("no_such_trigger" in n for n in notes)
    assert any("fit_retries" in n for n in notes)
    fired = {f.trigger: f for f in findings}
    assert fired["go_not_converged"].message == "uses {missing_value}"


@pytest.mark.unit
def test_trigger_silent_when_not_applicable():
    stats = {
        "dft": {"n_go": 0, "go_not_converged": 0, "returned": 0, "submitted": None},
        "dataset": {"new_this_loop": {}, "new_quality_filter_reasons": {}},
        "events": {"by_event": {}, "data": {}},
        "model": None,
        "timing": None,
    }
    assert all(fn(stats) is None for fn in TRIGGERS.values())


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _image_links(markdown: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)


@pytest.mark.unit
def test_report_markdown_sections_and_plots(finished_loop):
    path = write_loop_report(finished_loop, 0)
    markdown = path.read_text()

    for heading in (
        "# ALomancy report: al_loop_0",
        "## Headline",
        "## Issues & suggestions",
        "## Trends",
        "## Best model",
        "## Training set",
        "## DFT",
        "## High-accuracy evaluation: QE",
        "## Warnings and events",
    ):
        assert heading in markdown
    assert "go_not_converged" in markdown
    assert "something odd happened" in markdown
    links = _image_links(markdown)
    assert "plots/training_set_composition.png" in links
    assert "plots/dft_geometry_steps.png" in links
    for link in links:
        assert (path.parent / link).exists(), link
    latest = Path("results/reports/latest.md").read_text()
    for link in _image_links(latest):
        assert (Path("results/reports") / link).exists(), link
    assert json.loads((path.parent / "report.json").read_text())["loop"] == 0


@pytest.mark.unit
def test_report_without_plots_has_text_only(finished_loop):
    finished_loop.plots = False
    markdown = write_loop_report(finished_loop, 0).read_text()
    assert _image_links(markdown) == []
    assert "## Training set" in markdown
    assert not Path("results/reports/al_loop_0/plots").exists()


@pytest.mark.unit
def test_trends_table_has_a_row_per_loop(finished_loop):
    write_loop_report(finished_loop, 0)
    Path("results/al_loop_1").mkdir()
    markdown = write_loop_report(finished_loop, 1).read_text()
    trends = markdown.split("## Trends")[1].split("\n\n")[1]
    rows = [line for line in trends.splitlines() if re.match(r"\| \d+ \|", line)]
    assert [r.split("|")[1].strip() for r in rows] == ["0", "1"]
    assert "| loop 0 | loop 1 |" in markdown


# ---------------------------------------------------------------------------
# Wiring and CLI
# ---------------------------------------------------------------------------


def _ctx(loop: int = 0) -> LoopContext:
    return LoopContext(
        loop=loop,
        base_name=f"al_loop_{loop}",
        workdir=Path("results", f"al_loop_{loop}"),
        train=[],
        test=[],
        train_only=False,
        plots_dir=Path("results/current_plots", f"al_loop_{loop}"),
    )


@pytest.mark.unit
def test_finish_loop_writes_the_report(finished_loop):
    with patch.object(finished_loop, "_curate_dataset"):
        finished_loop.finish_loop(_ctx(0))
    assert Path("results/reports/al_loop_0/report.md").exists()
    assert Path("results/al_loop_0/loop.done").exists()


@pytest.mark.unit
def test_report_failure_never_fails_the_loop(finished_loop):
    with (
        patch.object(finished_loop, "_curate_dataset"),
        patch(
            "alomancy.analysis.report.collect_loop_stats",
            side_effect=RuntimeError("boom"),
        ),
    ):
        finished_loop.finish_loop(_ctx(0))
    assert Path("results/al_loop_0/loop.done").exists()
    assert any(
        "Could not write the report" in e["message"]
        for e in read_events("results/events.jsonl")
    )


@pytest.mark.unit
def test_report_disabled(finished_loop):
    finished_loop.report = False
    with patch.object(finished_loop, "_curate_dataset"):
        finished_loop.finish_loop(_ctx(0))
    assert not Path("results/reports").exists()


@pytest.mark.unit
def test_cli_rebuilds_reports_from_saved_config(finished_loop, tmp_path, shared_db):
    from alomancy.cli.report import write_reports

    finished_loop._write_run_config()
    Path("results/al_loop_0/loop.done").touch()
    with patch(
        "alomancy.core.active_learning_workflow.GlobalDatabase", return_value=shared_db
    ):
        (path,) = write_reports(tmp_path / "results")

    assert path == Path("results/reports/al_loop_0/report.md")
    assert "## Training set" in (tmp_path / path).read_text()
