"""Unit tests for alomancy.structure_generation.mlip_relax and the
workflow step that uses it (ActiveLearningWorkflow.relax_selected)."""

import logging
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from alomancy.core.active_learning_workflow import LoopContext, TrainedModel
from alomancy.core.committee_uncertainty_workflow import CommitteeUncertaintyWorkflow
from alomancy.mlip.base import ALomancyTrainer
from alomancy.registry import _REGISTRY, register
from alomancy.structure_generation.mlip_relax import (
    MLIP_RELAXATION_DEFAULTS,
    relax_with_model,
    resolve_mlip_relaxation,
)


class _Collapse(Calculator):
    """Pulls every atom onto the first: a model that relaxes into
    unphysically short bonds."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), changes=all_changes):
        super().calculate(atoms, properties, changes)
        d = self.atoms.positions - self.atoms.positions[0]
        self.results = {"energy": 0.5 * float((d**2).sum()), "forces": -d}


class RelaxTrainer(ALomancyTrainer):
    NAME = "relax_test"
    KWARGS_KEY = "relax_test_kwargs"
    KWARGS_DEFAULTS: ClassVar[dict] = {"collapse": False}

    def model_path(self, fit_dir: Path) -> Path:
        return Path(fit_dir) / "relax.model"

    def get_calculator(self, model_path, *, device=None):
        return _Collapse() if self.kwargs.get("collapse") else EMT()

    def fit(self, *args, **kwargs):
        return None


@pytest.fixture
def relax_trainer():
    register("mlip_trainer", "relax_test", __name__, trainer_class="RelaxTrainer")
    yield
    del _REGISTRY["mlip_trainer"]["relax_test"]


def _cu_dimer(x=1.8):
    return Atoms("Cu2", positions=[[0, 0, 0], [x, 0, 0]], cell=[10, 10, 10], pbc=True)


# --- config -----------------------------------------------------------------


@pytest.mark.unit
def test_null_is_off_and_empty_mapping_takes_the_defaults():
    assert resolve_mlip_relaxation(None) is None
    assert resolve_mlip_relaxation({}) == MLIP_RELAXATION_DEFAULTS
    assert resolve_mlip_relaxation({"fmax": 0.1})["fmax"] == 0.1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (True, "must be a mapping"),
        ({"fmx": 0.1}, "Unknown"),
        ({"fmax": 0}, "fmax must be a positive"),
        ({"fmax": True}, "fmax must be a positive"),
        ({"steps": 0}, "steps must be a positive integer"),
        ({"steps": 1.5}, "steps must be a positive integer"),
        ({"optimizer": "SGD"}, "optimizer must be one of"),
    ],
)
def test_malformed_config_raises(cfg, match):
    with pytest.raises(ValueError, match=match):
        resolve_mlip_relaxation(cfg)


# --- remote worker ------------------------------------------------------------


@pytest.mark.unit
def test_relax_with_model_relaxes_and_tags(relax_trainer):
    start = _cu_dimer()
    (relaxed,) = relax_with_model(
        [start],
        "unused.model",
        "relax_test",
        {"relax_test_kwargs": {}},
        fmax=0.01,
        steps=200,
        optimizer="FIRE",
    )

    assert relaxed is not start
    assert relaxed.calc is None
    assert relaxed.info["mlip_relaxed"] is True
    assert relaxed.info["mlip_relax_converged"] is True
    assert relaxed.info["mlip_relax_steps"] > 0
    check = relaxed.copy()
    check.calc = EMT()
    assert np.abs(check.get_forces()).max() < 0.01
    assert np.allclose(start.positions[1], [1.8, 0, 0])  # input untouched


@pytest.mark.unit
def test_relax_with_model_reports_unconverged(relax_trainer):
    (relaxed,) = relax_with_model(
        [_cu_dimer(1.5)],
        "unused.model",
        "relax_test",
        {"relax_test_kwargs": {}},
        fmax=1e-6,
        steps=2,
        optimizer="BFGS",
    )
    assert relaxed.info["mlip_relax_converged"] is False
    assert relaxed.info["mlip_relax_steps"] == 2


# --- workflow step --------------------------------------------------------------

_WF_MODULE = "alomancy.core.active_learning_workflow"


class _Events(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.events: list[str] = []

    def emit(self, record):
        if getattr(record, "event", None):
            self.events.append(record.event)


def _workflow(jobs_dict, shared_db, *, relaxation, collapse=False):
    jobs_dict["structure_generation"]["mlip_relaxation"] = relaxation
    general = jobs_dict.setdefault("general", {})
    general.setdefault(
        "dataset_kwargs", {"test_ratio": 0.1, "target_config_types": ["IsolatedAtom"]}
    )
    general.update({"num_of_al_loops": 1, "plots": False})
    wf = CommitteeUncertaintyWorkflow(jobs_dict=jobs_dict)
    wf.db = shared_db
    wf.training_config = {
        "trainer": "relax_test",
        "relax_test_kwargs": {"collapse": collapse},
    }
    return wf


def _ctx(loop=0):
    base_name = f"al_loop_{loop}"
    return LoopContext(
        loop=loop,
        base_name=base_name,
        workdir=Path("results", base_name),
        train=[],
        test=[],
        train_only=False,
        plots_dir=Path("results", "current_plots", base_name),
    )


_MODEL = TrainedModel(
    fit_idx=0,
    seed=803,
    model_path="model_0.pt",
    compiled_model_path=None,
    metrics={},
    fit_dir=Path("results/al_loop_0/training/fit_0"),
)


def _run_locally(function, job_configs, remote_info, **_kwargs):
    """submit_n without the HPC: the remote function runs here."""
    return [function(**jc["function_kwargs"]) for jc in job_configs]


def _ctx_with_train(loop=0):
    ctx = _ctx(loop)
    ctx.train.append(_cu_dimer())
    return ctx


def _generate(wf, candidates, **patches):
    """generate_candidates with a generator returning *candidates*."""
    generator = MagicMock()
    generator.generate.return_value = candidates
    with (
        patch(f"{_WF_MODULE}.resolve", return_value=generator),
        patch(f"{_WF_MODULE}.get_remote_info"),
        patch(
            f"{_WF_MODULE}.submit_n",
            side_effect=patches.get("submit", _run_locally),
        ) as submit,
    ):
        return wf.generate_candidates(_ctx_with_train(), _MODEL), submit


@pytest.mark.unit
def test_off_leaves_the_candidates_unrelaxed(
    tmp_path, monkeypatch, minimal_jobs_dict, shared_db
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(minimal_jobs_dict, shared_db, relaxation=None)
    out, submit = _generate(wf, [_cu_dimer()])
    submit.assert_not_called()
    assert "mlip_relaxed" not in out[0].info


@pytest.mark.unit
def test_candidates_are_relaxed_before_selection_and_cached(
    tmp_path, monkeypatch, minimal_jobs_dict, shared_db, relax_trainer
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(minimal_jobs_dict, shared_db, relaxation={"fmax": 0.01})
    out, _ = _generate(wf, [_cu_dimer(), _cu_dimer(2.0)])

    assert [a.info["mlip_relaxed"] for a in out] == [True, True]
    assert Path("results/al_loop_0/mlip_relax.done").exists()

    restarted = _workflow(minimal_jobs_dict, shared_db, relaxation={"fmax": 0.01})
    again, submit = _generate(restarted, [_cu_dimer(), _cu_dimer(2.0)])
    submit.assert_not_called()
    for loaded, relaxed in zip(again, out, strict=True):
        assert np.allclose(loaded.positions, relaxed.positions)


@pytest.mark.unit
def test_candidates_that_collapse_are_dropped(
    tmp_path, monkeypatch, minimal_jobs_dict, shared_db, relax_trainer
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(minimal_jobs_dict, shared_db, relaxation={}, collapse=True)
    handler = _Events()
    al_logger = logging.getLogger(_WF_MODULE)
    al_logger.addHandler(handler)
    try:
        out, _ = _generate(wf, [_cu_dimer()])
    finally:
        al_logger.removeHandler(handler)

    assert out == []
    assert "mlip_relax_dropped" in handler.events


@pytest.mark.unit
def test_a_failed_job_keeps_the_unrelaxed_candidates(
    tmp_path, monkeypatch, minimal_jobs_dict, shared_db
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(minimal_jobs_dict, shared_db, relaxation={})
    handler = _Events()
    al_logger = logging.getLogger(_WF_MODULE)
    al_logger.addHandler(handler)
    try:
        out, _ = _generate(
            wf, [_cu_dimer(), _cu_dimer(2.0)], submit=lambda *a, **k: [None]
        )
    finally:
        al_logger.removeHandler(handler)

    assert len(out) == 2
    assert not any("mlip_relaxed" in a.info for a in out)
    assert "mlip_relax_fallback" in handler.events


@pytest.mark.unit
def test_the_committee_scores_the_relaxed_candidates(
    tmp_path, monkeypatch, minimal_jobs_dict, shared_db, relax_trainer
):
    """The user's order: MD frames -> MLIP relaxation -> committee
    uncertainty -> select."""
    monkeypatch.chdir(tmp_path)
    wf = _workflow(minimal_jobs_dict, shared_db, relaxation={"fmax": 0.01})
    scored: list[Atoms] = []

    def fake_predict(ctx, models, structures):
        scored.extend(structures)
        return [
            {
                "forces": [np.zeros((1, 3 * len(a))) + k for a in structures],
                "energies": [np.array(0.0) for _ in structures],
            }
            for k in range(len(models))
        ]

    models = [
        TrainedModel(
            fit_idx=i,
            seed=803 + i,
            model_path=f"model_{i}.pt",
            compiled_model_path=None,
            metrics={},
            fit_dir=Path(f"results/al_loop_0/training/fit_{i}"),
        )
        for i in range(3)
    ]
    generator = MagicMock()
    generator.generate.return_value = [_cu_dimer(), _cu_dimer(2.0), _cu_dimer(1.6)]
    with (
        patch(f"{_WF_MODULE}.resolve", return_value=generator),
        patch(f"{_WF_MODULE}.get_remote_info"),
        patch(f"{_WF_MODULE}.submit_n", side_effect=_run_locally),
        patch.object(wf, "best_model", return_value=models[0]),
        patch.object(wf, "predict", side_effect=fake_predict),
    ):
        wf.select_uncertain(_ctx_with_train(), models)

    assert len(scored) == 3
    assert all(a.info.get("mlip_relaxed") for a in scored)


@pytest.mark.unit
def test_bad_config_fails_at_construction(minimal_jobs_dict, shared_db):
    with pytest.raises(ValueError, match=r"mlip_relaxation\.optimizer"):
        _workflow(minimal_jobs_dict, shared_db, relaxation={"optimizer": "SGD"})
