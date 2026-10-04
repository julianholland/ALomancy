"""Tests for core/committee_uncertainty_workflow.py -- the committee child:
its settings, its loop order and its uncertainty-based selection. The
generic helpers it calls are tested in test_active_learning_workflow.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from alomancy.core.active_learning_workflow import LoopContext, TrainedModel
from alomancy.core.committee_uncertainty_workflow import CommitteeUncertaintyWorkflow

_PARENT = "alomancy.core.active_learning_workflow"
_MODULE = "alomancy.core.committee_uncertainty_workflow"


@pytest.fixture
def jobs_dict(minimal_jobs_dict):
    minimal_jobs_dict["general"] = {
        "al_workflow": "committee_uncertainty",
        "elements": ["H"],
        "num_of_al_loops": 2,
        "plots": False,
        "dataset_kwargs": {"target_config_types": ["IsolatedAtom"], "test_ratio": 0.1},
        "committee_uncertainty_kwargs": {"num_of_models_in_committee": 3},
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["num_of_models_in_committee"]
    return minimal_jobs_dict


def _workflow(jobs_dict, shared_db) -> CommitteeUncertaintyWorkflow:
    wf = CommitteeUncertaintyWorkflow(jobs_dict=jobs_dict)
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


def _model(fit_idx: int) -> TrainedModel:
    return TrainedModel(
        fit_idx=fit_idx,
        seed=803 + fit_idx,
        model_path=f"model_{fit_idx}.pt",
        compiled_model_path=None,
        metrics={},
        fit_dir=Path("results/al_loop_0/training", f"fit_{fit_idx}"),
    )


def _atoms(x: float = 0.9) -> Atoms:
    a = Atoms("H2", positions=[[0, 0, 0], [x, 0, 0]], cell=[5] * 3, pbc=True)
    a.info["config_type"] = "high_sd"
    a.info["REF_energy"] = -1.0
    a.arrays["REF_forces"] = np.zeros((2, 3))
    return a


# -- settings ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("size", [2, 0, 3.5, "3", True])
def test_committee_smaller_than_three_or_not_int_raises(jobs_dict, shared_db, size):
    jobs_dict["general"]["committee_uncertainty_kwargs"][
        "num_of_models_in_committee"
    ] = size
    with pytest.raises(ValueError, match="num_of_models_in_committee"):
        _workflow(jobs_dict, shared_db)


@pytest.mark.unit
def test_committee_size_defaults_to_three(jobs_dict, shared_db):
    del jobs_dict["general"]["committee_uncertainty_kwargs"]
    assert _workflow(jobs_dict, shared_db).workflow_kwargs == {
        "num_of_models_in_committee": 3
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("test_ratio", "general.dataset_kwargs.test_ratio"),
        ("target_config_types", "general.dataset_kwargs.target_config_types"),
        ("valid_fraction", "general.dataset_kwargs.valid_fraction"),
        ("fixed_test", "general.dataset_kwargs.fixed_test"),
        ("train_only", "general.train_only"),
    ],
)
def test_split_keys_in_committee_block_raise_moved_error(
    jobs_dict, shared_db, old, new
):
    jobs_dict["general"]["committee_uncertainty_kwargs"][old] = 0.1
    with pytest.raises(
        ValueError, match=rf"committee_uncertainty_kwargs\.{old} -> {new}"
    ):
        _workflow(jobs_dict, shared_db)


# -- selection ----------------------------------------------------------------------


@pytest.mark.unit
def test_select_uncertain_predicts_with_best_model_first(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    """The best model drives generation and is scored as "base_mlip"; every
    other model is scored under its own fit label."""
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)
    models = [_model(0), _model(1), _model(2)]
    candidates = [_atoms(0.8), _atoms(0.9)]
    prediction = {"forces": [np.zeros((1, 6))] * 2, "energies": [0.0, 0.0]}
    captured = {}

    def fake_find(
        structure_list, base_name, job_dict, structure_forces_dict, num_of_structures
    ):
        captured["labels"] = list(structure_forces_dict)
        captured["name"] = job_dict["structure_generation"]["name"]
        captured["num_of_structures"] = num_of_structures
        return structure_list[:1]

    with (
        patch.object(wf, "best_model", return_value=models[1]),
        patch.object(wf, "generate_candidates", return_value=candidates) as gen,
        patch.object(wf, "predict", return_value=[prediction] * 3) as predict,
        patch(f"{_MODULE}.find_high_sd_structures", side_effect=fake_find),
    ):
        selected = wf.select_uncertain(_ctx(0), models)

    assert gen.call_args.args[1] is models[1]
    assert [m.fit_idx for m in predict.call_args.args[1]] == [1, 0, 2]
    assert captured["labels"] == ["base_mlip", "fit_0", "fit_2"]
    assert captured["name"] == "structure_generation"
    assert captured["num_of_structures"] == wf.num_of_structures_per_loop
    assert len(selected) == 1
    assert Path("results/al_loop_0/generate_structures.done").exists()


@pytest.mark.unit
def test_select_uncertain_reloads_file_when_done(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    monkeypatch.chdir(tmp_path)
    sg_dir = Path("results/al_loop_0/structure_generation")
    sg_dir.mkdir(parents=True)
    write(sg_dir / "high_sd_structures.xyz", [_atoms(), _atoms(1.0)], format="extxyz")
    Path("results/al_loop_0/generate_structures.done").write_text("done\n")
    wf = _workflow(jobs_dict, shared_db)

    with patch.object(wf, "generate_candidates") as gen:
        result = wf.select_uncertain(_ctx(0), [_model(0)])

    gen.assert_not_called()
    assert len(result) == 2


# -- run() order ------------------------------------------------------------------------


@pytest.mark.unit
def test_run_trains_full_committee_then_selects(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)
    calls = []

    with (
        patch.object(wf, "_initialize_training_set", return_value=([], [])),
        patch.object(
            wf,
            "train_models",
            side_effect=lambda ctx, seeds, **kw: (
                calls.append(("train", ctx.loop, seeds, kw)) or []
            ),
        ),
        patch.object(
            wf,
            "select_uncertain",
            side_effect=lambda ctx, models: calls.append(("select", ctx.loop)) or [],
        ),
        patch(f"{_PARENT}._evaluator_orchestrate", return_value=[]),
        patch(f"{_PARENT}.write"),
    ):
        wf.run()

    assert calls == [
        ("train", 0, [803, 804, 805], {"min_successful": 3}),
        ("select", 0),
        ("train", 1, [803, 804, 805], {"min_successful": 3}),
        ("select", 1),
    ]
    assert Path("results/al_loop_1/loop.done").exists()


@pytest.mark.unit
def test_resumes_existing_mid_loop_results_without_redoing_work(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    """Regression: a results tree written by the pre-refactor workflow
    (loop 0 done; loop 1 trained and selected, DFT not yet run) resumes
    straight at DFT -- no training, no structure generation."""
    monkeypatch.chdir(tmp_path)
    Path("results/al_loop_0").mkdir(parents=True)
    Path("results/al_loop_0/loop.done").write_text("done\n")
    loop1 = Path("results/al_loop_1")
    (loop1 / "structure_generation").mkdir(parents=True)
    (loop1 / "train_mlip.done").write_text("done\n")
    (loop1 / "generate_structures.done").write_text("done\n")
    selected = [_atoms(0.8), _atoms(0.9), _atoms(1.0)]
    write(
        loop1 / "structure_generation/high_sd_structures.xyz", selected, format="extxyz"
    )
    wf = _workflow(jobs_dict, shared_db)

    fake_trainer = MagicMock()
    fake_trainer.read_existing_result.side_effect = lambda *a, **kw: (
        f"fit_{kw['fit_idx']}.model",
        None,
        {},
    )
    evaluated = {}

    def fake_evaluate(structures, config, **kwargs):
        evaluated["n"] = len(structures)
        return []

    with (
        patch(f"{_PARENT}.resolve", return_value=fake_trainer),
        patch(f"{_PARENT}.submit_n") as submit_n,
        patch.object(wf, "generate_candidates") as generate,
        patch(f"{_PARENT}._evaluator_orchestrate", side_effect=fake_evaluate),
    ):
        wf.run()

    submit_n.assert_not_called()
    generate.assert_not_called()
    assert evaluated["n"] == 3
    assert (loop1 / "loop.done").exists()


@pytest.mark.unit
def test_train_only_stops_after_first_training(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    monkeypatch.chdir(tmp_path)
    jobs_dict["general"]["train_only"] = True
    wf = _workflow(jobs_dict, shared_db)

    with (
        patch.object(wf, "_initialize_training_set", return_value=([], [])),
        patch.object(wf, "train_models", return_value=[]) as train,
        patch.object(wf, "select_uncertain") as select,
        patch(f"{_PARENT}.write"),
    ):
        wf.run()

    assert train.call_count == 1
    select.assert_not_called()


@pytest.mark.unit
def test_selected_structures_read_back_with_config_type(tmp_path):
    """high_sd_structures.xyz round-trips through extxyz (the loader's
    source) with its config_type intact."""
    path = tmp_path / "high_sd_structures.xyz"
    write(path, [_atoms()], format="extxyz")
    (atoms,) = read(path, ":", format="extxyz")
    assert atoms.info["config_type"] == "high_sd"


@pytest.mark.unit
def test_moved_and_removed_keys_reported_together(jobs_dict, shared_db):
    jobs_dict["general"]["committee_uncertainty_kwargs"]["test_ratio"] = 0.1
    jobs_dict["general"]["high_force_threshold"] = 5.0
    with pytest.raises(ValueError) as exc_info:
        _workflow(jobs_dict, shared_db)
    message = str(exc_info.value)
    assert "general.dataset_kwargs.test_ratio" in message
    assert "high_force_threshold" in message


@pytest.mark.unit
def test_resumes_after_dft_without_rerunning_any_step(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    """Loop 1 trained, selected and DFT-evaluated before the stop: the
    restart only adds the saved DFT results and finishes the loop."""
    monkeypatch.chdir(tmp_path)
    Path("results/al_loop_0").mkdir(parents=True)
    Path("results/al_loop_0/loop.done").write_text("done\n")
    loop1 = Path("results/al_loop_1")
    (loop1 / "structure_generation").mkdir(parents=True)
    for sentinel in ("train_mlip", "generate_structures", "high_accuracy_eval"):
        (loop1 / f"{sentinel}.done").write_text("done\n")
    write(
        loop1 / "structure_generation/high_sd_structures.xyz",
        [_atoms()],
        format="extxyz",
    )
    evaluated = [_atoms(0.8), _atoms(0.9)]
    for atoms in evaluated:  # the evaluator saves structures with DFT results
        atoms.calc = SinglePointCalculator(
            atoms, energy=-1.0, forces=np.zeros((len(atoms), 3))
        )
    write(loop1 / "high_accuracy_eval_results.xyz", evaluated, format="extxyz")
    wf = _workflow(jobs_dict, shared_db)
    fake_trainer = MagicMock()
    fake_trainer.read_existing_result.side_effect = lambda *a, **kw: (
        f"fit_{kw['fit_idx']}.model",
        None,
        {},
    )
    added = {}

    with (
        patch(f"{_PARENT}.resolve", return_value=fake_trainer),
        patch(f"{_PARENT}.submit_n") as submit_n,
        patch.object(wf, "generate_candidates") as generate,
        patch(f"{_PARENT}._evaluator_orchestrate") as orchestrate,
        patch.object(
            wf,
            "add_to_dataset",
            side_effect=lambda ctx, structures: added.update(n=len(structures)),
        ),
    ):
        wf.run()

    submit_n.assert_not_called()
    generate.assert_not_called()
    orchestrate.assert_not_called()
    assert added["n"] == 2
    assert (loop1 / "loop.done").exists()
