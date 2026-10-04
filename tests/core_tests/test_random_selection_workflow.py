"""Tests for core/random_selection_workflow.py -- the single-model,
random-selection baseline and the smallest ActiveLearningWorkflow child."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from alomancy.core.active_learning_workflow import (
    LoopContext,
    TrainedModel,
    build_workflow,
)
from alomancy.core.random_selection_workflow import RandomSelectionWorkflow

_PARENT = "alomancy.core.active_learning_workflow"


@pytest.fixture
def jobs_dict(minimal_jobs_dict):
    minimal_jobs_dict["general"] = {
        "al_workflow": "random_selection",
        "elements": ["H"],
        "num_of_al_loops": 2,
        "plots": False,
        "dataset_kwargs": {"target_config_types": ["IsolatedAtom"], "test_ratio": 0.1},
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["num_of_models_in_committee"]
    minimal_jobs_dict["general"]["num_of_structures_per_loop"] = 3
    return minimal_jobs_dict


def _workflow(jobs_dict, shared_db) -> RandomSelectionWorkflow:
    wf = build_workflow(jobs_dict)
    assert isinstance(wf, RandomSelectionWorkflow)
    wf.db = shared_db
    return wf


def _ctx(loop: int = 0) -> LoopContext:
    base_name = f"al_loop_{loop}"
    return LoopContext(
        loop=loop,
        base_name=base_name,
        workdir=Path("results", base_name),
        train=[],
        test=[],
        train_only=False,
        plots_dir=Path("results/current_plots", base_name),
    )


_MODEL = TrainedModel(
    fit_idx=0,
    seed=803,
    model_path="model_0.pt",
    compiled_model_path=None,
    metrics={},
    fit_dir=Path("results/al_loop_0/training/fit_0"),
)


def _candidates(n: int) -> list[Atoms]:
    out = []
    for i in range(n):
        a = Atoms("H2", positions=[[0, 0, 0], [0.7 + 0.01 * i, 0, 0]], cell=[5] * 3)
        a.info["candidate"] = i
        out.append(a)
    return out


@pytest.mark.unit
def test_build_workflow_dispatches_random_selection(jobs_dict, shared_db):
    wf = _workflow(jobs_dict, shared_db)
    assert wf.NEW_STRUCTURE_CONFIG_TYPE == "random_selection"
    assert wf.workflow_kwargs == {}


@pytest.mark.unit
def test_committee_block_is_an_unknown_key_here(jobs_dict, shared_db):
    """random_selection has no kwargs block; committee settings are flagged
    as unrecognised rather than silently ignored."""
    jobs_dict["general"]["committee_uncertainty_kwargs"] = {
        "num_of_models_in_committee": 5
    }
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger(_PARENT)
    logger.addHandler(handler)
    try:
        _workflow(jobs_dict, shared_db)
    finally:
        logger.removeHandler(handler)
    assert any("committee_uncertainty_kwargs" in r.getMessage() for r in records)


@pytest.mark.unit
def test_select_random_is_seeded_and_saved(tmp_path, jobs_dict, monkeypatch, shared_db):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)

    with patch.object(wf, "generate_candidates", return_value=_candidates(10)):
        first = wf.select_random(_ctx(0), _MODEL)

    ids = [a.info["candidate"] for a in first]
    assert len(ids) == 3
    rng = np.random.default_rng(803 + 0)
    assert ids == sorted(rng.choice(10, size=3, replace=False))
    saved = read(
        "results/al_loop_0/structure_generation/random_selection_structures.xyz", ":"
    )
    assert [a.info["candidate"] for a in saved] == ids


@pytest.mark.unit
def test_select_random_takes_all_when_too_few(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)

    with patch.object(wf, "generate_candidates", return_value=_candidates(2)):
        selected = wf.select_random(_ctx(0), _MODEL)

    assert len(selected) == 2


@pytest.mark.unit
def test_select_random_reloads_when_done(tmp_path, jobs_dict, monkeypatch, shared_db):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)
    with patch.object(wf, "generate_candidates", return_value=_candidates(10)):
        first = wf.select_random(_ctx(0), _MODEL)

    resumed = _workflow(jobs_dict, shared_db)
    with patch.object(resumed, "generate_candidates") as gen:
        again = resumed.select_random(_ctx(0), _MODEL)

    gen.assert_not_called()
    assert [a.info["candidate"] for a in again] == [a.info["candidate"] for a in first]


@pytest.mark.unit
def test_run_trains_one_model_per_loop(tmp_path, jobs_dict, monkeypatch, shared_db):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)
    calls = []

    with (
        patch.object(wf, "_initialize_training_set", return_value=([], [])),
        patch.object(
            wf,
            "train_models",
            side_effect=lambda ctx, seeds, **kw: (
                calls.append(("train", seeds, kw)) or [_MODEL]
            ),
        ),
        patch.object(
            wf,
            "select_random",
            side_effect=lambda ctx, model: (
                calls.append(("select", model.fit_idx)) or []
            ),
        ),
        patch(f"{_PARENT}._evaluator_orchestrate", return_value=[]),
        patch(f"{_PARENT}.write"),
    ):
        wf.run()

    assert calls == [
        ("train", [803], {}),
        ("select", 0),
        ("train", [803], {}),
        ("select", 0),
    ]


@pytest.mark.unit
def test_new_structures_labelled_random_selection(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)
    labelled = _candidates(1)[0]
    labelled.info["REF_energy"] = -1.0
    labelled.arrays["REF_forces"] = np.zeros((2, 3))

    with patch(f"{_PARENT}._evaluator_orchestrate", return_value=[labelled]):
        (result,) = wf.high_accuracy_evaluate(_ctx(1), _candidates(1))

    assert result.info["config_type"] == "random_selection"
    assert result.info["al_loop"] == 1
