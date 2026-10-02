"""Tests for core/active_learning_workflow.py -- the generic AL workflow
parent -- exercised through its CommitteeUncertaintyWorkflow child (the
parent is abstract). Committee-specific selection and loop order live in
test_committee_uncertainty_workflow.py.

The loop-control logic in run() (restart, plotting call sites, train_only/
fixed_test branching, DB writes) was originally copied largely as-is from
the now-removed BaseActiveLearningWorkflow.run() -- these tests focus on
what's new: registry-resolved module dispatch, the shared validation
split, the partial-aware per-fit restart mechanism, generalized committee
scoring, and cross-loop metrics aggregation.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from alomancy.core.active_learning_workflow import (
    LoopContext,
    TrainedModel,
    _flatten_settings,
    _is_user_specified,
    _resolve_effective_phase_dict,
    _select_validation_split,
    build_workflow,
    phase,
)
from alomancy.core.committee_uncertainty_workflow import CommitteeUncertaintyWorkflow
from alomancy.mlip.base import ALomancyTrainer, run_training
from alomancy.mlip.evaluation import prediction_metrics, save_evaluation
from alomancy.mlip.predict import predict_with_model

_MODULE = "alomancy.core.active_learning_workflow"


# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_jobs_dict(minimal_jobs_dict):
    """minimal_jobs_dict plus the new `general` section (renamed from
    `workflow`; decision 14: split-building parameters and
    num_of_models_in_committee -- renamed from size_of_committee -- move
    here from initialization/mlip_committee, nested under
    committee_uncertainty_kwargs to match the <dispatch_value>_kwargs
    convention; mlip_committee is renamed training)."""
    committee_size = minimal_jobs_dict["mlip_committee"]["num_of_models_in_committee"]
    minimal_jobs_dict["general"] = {
        "al_workflow": "committee_uncertainty",
        "elements": ["H"],
        "dataset_kwargs": {
            "target_config_types": ["IsolatedAtom"],
            "test_ratio": 0.1,
            "valid_fraction": 0.05,
        },
        "committee_uncertainty_kwargs": {
            "num_of_models_in_committee": committee_size,
        },
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["num_of_models_in_committee"]
    minimal_jobs_dict["training"]["trainer"] = "mace"
    return minimal_jobs_dict


def _make_workflow(tmp_path, jobs_dict, shared_db, **general_overrides):
    """CommitteeUncertaintyWorkflow now takes only jobs_dict -- every
    former constructor kwarg lives under jobs_dict["general"] instead
    (see the class's own module docstring). db is the one exception (a
    live object can't be a config value): set post-construction via the
    lazy `db` property/setter, which never pays for the real
    GlobalDatabase(db_path) construction this replaces."""
    general = jobs_dict.setdefault("general", {})
    general.setdefault(
        "dataset_kwargs",
        {"test_ratio": 0.1, "target_config_types": ["IsolatedAtom"]},
    )
    general.update(
        {
            "num_of_al_loops": 2,
            "plots": False,
            **general_overrides,
        }
    )
    wf = CommitteeUncertaintyWorkflow(jobs_dict=jobs_dict)
    wf.db = shared_db
    return wf


def _ctx(loop: int = 0, train=None, test=None) -> LoopContext:
    base_name = f"al_loop_{loop}"
    return LoopContext(
        loop=loop,
        base_name=base_name,
        workdir=Path("results", base_name),
        train=train or [],
        test=test or [],
        train_only=False,
        plots_dir=Path("results", "current_plots", base_name),
    )


def _atoms(symbol="H", config_type="init_dimer"):
    a = Atoms(symbol, positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    a.info["config_type"] = config_type
    a.info["REF_energy"] = 1.0
    a.arrays["REF_forces"] = np.zeros((1, 3))
    return a


# ---------------------------------------------------------------------------
# build_workflow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildWorkflow:
    def test_dispatches_to_committee_uncertainty(
        self, tmp_path, workflow_jobs_dict, shared_db
    ):
        wf = build_workflow(jobs_dict=workflow_jobs_dict)
        wf.db = shared_db
        assert isinstance(wf, CommitteeUncertaintyWorkflow)

    def test_defaults_to_committee_uncertainty_when_absent(
        self, tmp_path, minimal_jobs_dict, shared_db
    ):
        minimal_jobs_dict["general"] = {
            "dataset_kwargs": {
                "test_ratio": 0.1,
                "target_config_types": ["IsolatedAtom"],
            },
        }
        wf = build_workflow(jobs_dict=minimal_jobs_dict)
        wf.db = shared_db
        assert isinstance(wf, CommitteeUncertaintyWorkflow)

    @pytest.mark.parametrize("missing", ["test_ratio", "target_config_types"])
    def test_raises_at_construction_when_required_committee_key_missing(
        self, tmp_path, minimal_jobs_dict, missing
    ):
        """Regression: a missing test_ratio used to surface as a KeyError
        only after an AL loop's DFT had finished, hours into a run."""
        committee = {"test_ratio": 0.1, "target_config_types": ["IsolatedAtom"]}
        del committee[missing]
        minimal_jobs_dict["general"] = {
            "dataset_kwargs": committee,
        }
        with pytest.raises(ValueError, match=missing):
            build_workflow(jobs_dict=minimal_jobs_dict)

    @pytest.mark.parametrize(
        ("path", "old", "new"),
        [
            (("general",), "number_of_al_loops", "num_of_al_loops"),
            (
                ("general", "committee_uncertainty_kwargs"),
                "number_models_in_committee",
                "num_of_models_in_committee",
            ),
            (
                ("structure_generation", "structure_selection_kwargs"),
                "max_number_of_concurrent_jobs",
                "num_of_md_starts",
            ),
            (
                ("high_accuracy_evaluation",),
                "relax_max_steps",
                "max_num_of_relax_steps",
            ),
            (
                ("initialization", "amorphous_kwargs"),
                "num_amorphous",
                "num_of_amorphous_structures",
            ),
            (("initialization", "mp_kwargs"), "max_atom_number", "max_num_of_atoms"),
        ],
    )
    def test_raises_on_renamed_key(
        self, tmp_path, workflow_jobs_dict, shared_db, path, old, new
    ):
        node = workflow_jobs_dict
        for part in path:
            node = node.setdefault(part, {})
        node[old] = 3
        with pytest.raises(ValueError, match=rf"{old} -> .*{new}"):
            _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

    def test_renamed_key_error_lists_every_old_key(
        self, tmp_path, workflow_jobs_dict, shared_db
    ):
        workflow_jobs_dict["general"]["number_of_al_loops"] = 3
        workflow_jobs_dict.setdefault("initialization", {})["dimer_kwargs"] = {
            "num_dimers_per_combo": 2
        }
        with pytest.raises(ValueError) as exc_info:
            _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        assert "number_of_al_loops" in str(exc_info.value)
        assert "num_dimers_per_combo" in str(exc_info.value)

    def test_warns_on_unrecognised_general_key(
        self, tmp_path, minimal_jobs_dict, shared_db
    ):
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        # The module's own logger, not "alomancy": setup_logging (called in
        # __init__, before this warning) replaces the root alomancy handlers.
        alomancy_logger = logging.getLogger("alomancy.core.active_learning_workflow")
        alomancy_logger.addHandler(handler)
        try:
            _make_workflow(
                tmp_path,
                minimal_jobs_dict,
                shared_db,
                skip_initializatoin=False,
            )
        finally:
            alomancy_logger.removeHandler(handler)
        messages = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        assert any("skip_initializatoin" in m for m in messages)

    def test_raises_on_unknown_al_workflow(self, minimal_jobs_dict):
        minimal_jobs_dict["general"] = {"al_workflow": "furthest_point_sampling"}
        with pytest.raises(ValueError, match=r"Unknown general\.al_workflow"):
            build_workflow(jobs_dict=minimal_jobs_dict)


# ---------------------------------------------------------------------------
# _select_validation_split (the workflow's per-loop validation carve-out)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectValidationSplit:
    def _dimers(self, n: int, config_type: str) -> list[Atoms]:
        return [
            Atoms(
                ["H"],
                positions=[[i, 0, 0]],
                cell=[5, 5, 5],
                pbc=True,
                info={"config_type": config_type},
            )
            for i in range(n)
        ]

    def test_carves_correct_fraction(self):
        eligible = self._dimers(100, "dimer")
        ineligible = self._dimers(10, "IsolatedAtom")
        all_training = eligible + ineligible
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.05, rng=np.random.default_rng(42)
        )
        assert len(valid) == 5
        assert len(new_train) == len(all_training) - 5

    def test_no_overlap(self):
        all_training = self._dimers(50, "dimer")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.1, rng=np.random.default_rng(42)
        )
        train_ids = {id(a) for a in new_train}
        valid_ids = {id(a) for a in valid}
        assert train_ids.isdisjoint(valid_ids)

    def test_ineligible_always_in_train(self):
        eligible = self._dimers(20, "dimer")
        ineligible = self._dimers(5, "IsolatedAtom")
        all_training = eligible + ineligible
        new_train, _valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.2, rng=np.random.default_rng(42)
        )
        ineligible_ids = {id(a) for a in ineligible}
        assert ineligible_ids.issubset({id(a) for a in new_train})

    def test_empty_eligible_returns_all_training(self):
        all_training = self._dimers(10, "IsolatedAtom")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.1, rng=np.random.default_rng(42)
        )
        assert valid == []
        assert len(new_train) == len(all_training)

    def test_rounds_to_zero_returns_all_training(self):
        all_training = self._dimers(1, "dimer")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.05, rng=np.random.default_rng(42)
        )
        assert valid == []
        assert len(new_train) == 1


# ---------------------------------------------------------------------------
# predict_with_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoreStructuresWithMember:
    def test_resolves_calculator_and_scores_each_structure(self):
        fake_calc = MagicMock()
        structures = [_atoms(), _atoms()]
        for atoms in structures:
            atoms.calc = None

        with patch("alomancy.mlip.predict.get_trainer") as mock_get_trainer:
            mock_get_trainer.return_value = MagicMock(
                get_calculator=MagicMock(return_value=fake_calc)
            )
            with (
                patch.object(Atoms, "get_forces", lambda self: np.ones((1, 3))),
                patch.object(Atoms, "get_potential_energy", lambda self: -1.0),
            ):
                result = predict_with_model(
                    structures, "model.pt", "mace", {"device": "cpu"}
                )

        mock_get_trainer.assert_called_once_with("mace", {"device": "cpu"})
        mock_get_trainer.return_value.get_calculator.assert_called_once_with("model.pt")
        assert len(result["forces"]) == 2
        assert len(result["energies"]) == 2
        assert all(atoms.calc is fake_calc for atoms in structures)


# ---------------------------------------------------------------------------
# _cross_loop_metrics_dataframe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossLoopMetricsDataframe:
    @staticmethod
    def _split(error: float) -> dict:
        a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
        a.info.update(REF_energy=-8.0, model_energy=-8.0 + 2 * error, config_type="d")
        a.set_array("REF_forces", np.zeros((2, 3)))
        a.set_array("model_forces", np.ones((2, 3)) * error)
        return prediction_metrics([a])

    def _write_fit(
        self, fit_dir: Path, error: float, valid_error: float | None = None
    ) -> None:
        fit_dir.mkdir(parents=True)
        model = fit_dir / "committee_stagetwo.model"
        model.write_bytes(b"checkpoint")
        splits = {"test": self._split(error)}
        if valid_error is not None:
            splits["valid"] = self._split(valid_error)
        save_evaluation(fit_dir, model, splits)

    def test_one_best_model_row_per_loop(self, tmp_path, monkeypatch, shared_db):
        monkeypatch.chdir(tmp_path)
        for loop in range(2):
            for fit_idx, error in enumerate([0.3, 0.1, 0.2]):
                self._write_fit(
                    Path(f"results/al_loop_{loop}/committee/fit_{fit_idx}"), error
                )

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        df = wf._cross_loop_metrics_dataframe("committee")

        assert df["al_loop"].to_list() == [0, 1]
        assert df["best_fit_idx"].to_list() == [1, 1]
        assert df["mae_f"].to_list() == pytest.approx([0.1, 0.1])
        assert "mae_f_std_dev" not in df.columns

    def test_best_chosen_on_valid_but_reports_test(
        self, tmp_path, monkeypatch, shared_db
    ):
        """The plotted model is the one MD uses (valid-best), not whichever
        happens to score best on test."""
        monkeypatch.chdir(tmp_path)
        committee = Path("results/al_loop_0/committee")
        self._write_fit(committee / "fit_0", error=0.05, valid_error=0.4)
        self._write_fit(committee / "fit_1", error=0.2, valid_error=0.1)

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        df = wf._cross_loop_metrics_dataframe("committee")

        row = df.row(0, named=True)
        assert row["al_loop"] == 0
        assert row["best_fit_idx"] == 1
        assert row["mae_f"] == pytest.approx(0.2)
        assert row["selection_split"] == "valid"

    def test_skips_loop_with_inconsistent_evaluations(
        self, tmp_path, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._write_fit(Path("results/al_loop_0/committee/fit_0"), 0.1, valid_error=0.1)
        self._write_fit(Path("results/al_loop_0/committee/fit_1"), 0.1)
        self._write_fit(Path("results/al_loop_1/committee/fit_0"), 0.2)

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        df = wf._cross_loop_metrics_dataframe("committee")

        assert df["al_loop"].to_list() == [1]

    def test_empty_when_no_loops(self, tmp_path, monkeypatch, shared_db):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        df = wf._cross_loop_metrics_dataframe("committee")
        assert len(df) == 0

    def test_skips_loop_with_no_evaluations(self, tmp_path, monkeypatch, shared_db):
        monkeypatch.chdir(tmp_path)
        Path("results/al_loop_0/committee").mkdir(parents=True)
        self._write_fit(Path("results/al_loop_1/committee/fit_0"), 0.1)

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        df = wf._cross_loop_metrics_dataframe("committee")
        assert len(df) == 1


# ---------------------------------------------------------------------------
# _update_best_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateBestModel:
    @staticmethod
    def _write_fit(
        loop: int, fit_idx: int, valid_error: float, compiled: bool = True
    ) -> Path:
        fit_dir = Path(f"results/al_loop_{loop}/training/fit_{fit_idx}")
        fit_dir.mkdir(parents=True)
        model = fit_dir / "training_stagetwo.model"
        model.write_bytes(f"model {loop}-{fit_idx}".encode())
        if compiled:
            (fit_dir / "training_stagetwo_compiled.model").write_bytes(
                f"compiled {loop}-{fit_idx}".encode()
            )

        def structure(error, config_type):
            a = Atoms("Pd2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4] * 3, pbc=True)
            a.info.update(
                REF_energy=-8.0,
                model_energy=-8.0 + 2 * error,
                config_type=config_type,
                REF_stresses=np.zeros(6),
                model_stress=np.full(6, error / 10),
            )
            a.set_array("REF_forces", np.zeros((2, 3)))
            a.set_array("model_forces", np.ones((2, 3)) * error)
            return a

        save_evaluation(
            fit_dir,
            model,
            {
                "valid": prediction_metrics([structure(valid_error, "init_MP")]),
                "test": prediction_metrics(
                    [structure(0.2, "init_MP"), structure(0.4, "high_sd")]
                ),
            },
        )
        return fit_dir

    def test_copies_valid_best_compiled_model_with_metadata(
        self, tmp_path, monkeypatch, workflow_jobs_dict, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        for fit_idx, error in enumerate([0.3, 0.1, 0.2]):
            self._write_fit(0, fit_idx, error)

        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        wf._update_best_model("al_loop_0", 3)

        best_dir = Path("results/best_model")
        assert (best_dir / "ALomancy_best_model.model").read_bytes() == b"compiled 0-1"
        metadata = json.loads((best_dir / "model_metadata.json").read_text())
        assert metadata["al_loop"] == 0
        assert metadata["fit_idx"] == 1
        assert metadata["selected_on_split"] == "valid"
        test_errors = metadata["errors"]["test"]
        assert set(test_errors["config_types"]) == {"init_MP", "high_sd"}
        assert test_errors["config_types"]["high_sd"]["mae_f"] == pytest.approx(0.4)
        assert test_errors["config_types"]["high_sd"]["mae_stress"] == pytest.approx(
            0.04
        )
        assert "mae_e_per_atom" in test_errors

    def test_later_loop_replaces_previous_best_model(
        self, tmp_path, monkeypatch, workflow_jobs_dict, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        for loop in range(2):
            self._write_fit(loop, 0, 0.1)
            self._write_fit(loop, 1, 0.2)
        Path("results/best_model").mkdir(parents=True)
        Path("results/best_model/old_name.model").write_bytes(b"stale")

        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        wf._update_best_model("al_loop_0", 2)
        wf._update_best_model("al_loop_1", 2)

        best_dir = Path("results/best_model")
        assert sorted(p.name for p in best_dir.glob("*.model")) == [
            "ALomancy_best_model.model"
        ]
        assert (best_dir / "ALomancy_best_model.model").read_bytes() == b"compiled 1-0"
        metadata = json.loads((best_dir / "model_metadata.json").read_text())
        assert metadata["al_loop"] == 1

    def test_missing_compiled_model_keeps_previous(
        self, tmp_path, monkeypatch, workflow_jobs_dict, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._write_fit(0, 0, 0.1)
        self._write_fit(1, 0, 0.1, compiled=False)

        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        wf._update_best_model("al_loop_0", 1)
        wf._update_best_model("al_loop_1", 1)

        best_dir = Path("results/best_model")
        assert (best_dir / "ALomancy_best_model.model").read_bytes() == b"compiled 0-0"
        assert (
            json.loads((best_dir / "model_metadata.json").read_text())["al_loop"] == 0
        )


# ---------------------------------------------------------------------------
# _store_predictions_and_cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStorePredictionsAndCleanup:
    def test_stores_predictions_and_removes_checkpoints(
        self, tmp_path, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        fit_dir = Path("results/al_loop_0/committee/fit_0")
        checkpoints_dir = fit_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "epoch_1.pt").write_bytes(b"x")
        # MACE deletes checkpoints/ only once its compiled model exists.
        (fit_dir / "training_stagetwo_compiled.model").write_bytes(b"compiled")

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        results = {0: ("model.pt", "model_compiled.pt", {"test": {}})}

        with (
            patch(
                f"{_MODULE}.read_predictions",
                return_value={0: {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]}},
            ),
            patch.object(wf.db, "store_model_predictions") as mock_store,
        ):
            wf._store_predictions_and_cleanup("al_loop_0", "committee", results)

        mock_store.assert_called_once_with(
            0, 0, {0: {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]}}
        )
        assert not checkpoints_dir.exists()

    def test_no_cleanup_when_compiled_model_missing(
        self, tmp_path, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        fit_dir = Path("results/al_loop_0/committee/fit_0")
        checkpoints_dir = fit_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        results = {0: ("model.pt", None, {"test": {}})}

        with patch(f"{_MODULE}.read_predictions", return_value={}):
            wf._store_predictions_and_cleanup("al_loop_0", "committee", results)

        assert checkpoints_dir.exists()


# ---------------------------------------------------------------------------
# train_models
# ---------------------------------------------------------------------------


def _fit_index(fit_dir) -> int:
    return int(Path(fit_dir).name.rsplit("_", 1)[1])


class _FakeTrainer:
    """Stand-in ALomancyTrainer whose "outputs" are marker files, so
    train_models' success rule (outputs exist and read back) can be
    exercised: `submit_n` below writes a fit's marker when that job
    succeeds, the way a real remote job syncs back its model and
    evaluation."""

    @staticmethod
    def marker(fit_idx: int) -> Path:
        return Path(f"fit_{fit_idx}.outputs")

    def output_paths(self, fit_dir):
        return [self.marker(_fit_index(fit_dir))]

    def read_existing_result(self, fit_dir):
        return (f"fit_{_fit_index(fit_dir)}.pt", None, {})

    def deployable_model_path(self, fit_dir):
        return None

    def cleanup(self, fit_dir):
        pass

    def submit_n(self, outcome):
        """A fake submit_n. `outcome(fit_idx, attempt)` is "ok" (job
        returns and writes outputs), "fail" (job returns None) or
        "unevaluated" (job returns a result but writes no outputs)."""
        calls: list[list[int]] = []

        def fake(function, job_configs, remote_info, **kwargs):
            fit_indices = [
                _fit_index(jc["function_kwargs"]["fit_dir"]) for jc in job_configs
            ]
            attempt = len(calls)
            calls.append(fit_indices)
            results = []
            for fit_idx in fit_indices:
                result = outcome(fit_idx, attempt)
                if result == "ok":
                    self.marker(fit_idx).write_text("done")
                results.append(None if result == "fail" else "x.pt")
            return results

        return fake, calls


@pytest.mark.unit
class TestTrainMlip:
    def _prepare_loop_dir(self, base_name: str, n_train: int = 5) -> None:
        workdir = Path("results", base_name)
        workdir.mkdir(parents=True)
        write(
            str(workdir / "train_set.xyz"),
            [_atoms() for _ in range(n_train)],
            format="extxyz",
        )
        write(str(workdir / "test_set.xyz"), [_atoms()], format="extxyz")

    def test_submits_all_fits_when_none_cached(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        fake_submit, calls = trainer.submit_n(lambda fit_idx, attempt: "ok")

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
            patch.object(wf, "_store_predictions_and_cleanup"),
        ):
            models = wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        assert calls == [[0, 1, 2]]
        assert [m.model_path for m in models] == ["fit_0.pt", "fit_1.pt", "fit_2.pt"]
        # isolated_atom_energies is computed once locally from the DB and
        # passed to every fit so the trainer can default its reference
        # energies; each job runs the generic run_training worker.
        assert mock_submit_n.call_args.args[0] is run_training
        job_configs = mock_submit_n.call_args.args[1]
        kwargs = job_configs[0]["function_kwargs"]
        assert kwargs["isolated_atom_energies"] == wf.db.get_isolated_atom_energies()
        assert kwargs["trainer"] == "mace"
        assert kwargs["fit_dir"] == str(Path("results/al_loop_0/training/fit_0"))
        assert kwargs["seed"] == 803

    def test_reuses_cached_fits_and_only_submits_missing(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        trainer.marker(0).write_text("cached")
        fake_submit, calls = trainer.submit_n(lambda fit_idx, attempt: "ok")

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit),
            patch(f"{_MODULE}.get_remote_info"),
            patch.object(wf, "_store_predictions_and_cleanup"),
        ):
            wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        assert calls == [[1, 2]]  # fit_0 was cached

    def test_failed_fit_retried_once_and_recovers(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        fake_submit, calls = trainer.submit_n(
            lambda fit_idx, attempt: "fail" if (fit_idx, attempt) == (1, 0) else "ok"
        )

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit),
            patch(f"{_MODULE}.get_remote_info"),
            patch.object(wf, "_store_predictions_and_cleanup"),
        ):
            models = wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        assert calls == [[0, 1, 2], [1]]
        assert [m.fit_idx for m in models] == [0, 1, 2]

    def test_unevaluated_fit_counts_as_failed_and_is_retried(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        """A job that returns but leaves no evaluation (e.g. the trainer's
        evaluation step bailed out) is retried, not treated as trained."""
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        fake_submit, calls = trainer.submit_n(
            lambda fit_idx, attempt: (
                "unevaluated" if (fit_idx, attempt) == (2, 0) else "ok"
            )
        )

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit),
            patch(f"{_MODULE}.get_remote_info"),
            patch.object(wf, "_store_predictions_and_cleanup"),
        ):
            models = wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        assert calls == [[0, 1, 2], [2]]
        assert len(models) == 3

    def test_raises_when_retry_still_leaves_too_few(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        fake_submit, calls = trainer.submit_n(
            lambda fit_idx, attempt: "fail" if fit_idx == 0 else "ok"
        )

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit),
            patch(f"{_MODULE}.get_remote_info"),
            pytest.raises(RuntimeError, match="only 2 trained model"),
        ):
            wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        assert calls == [[0, 1, 2], [0]]

    def test_recognizes_real_checkpoint_evaluation_on_restart(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        """End-to-end restart recognition through the real mlip_trainer
        registry entry (trainer.py's output_paths/read_existing_result),
        not a mocked-away resolve() -- ported from the now-removed
        standard_active_learning.py/test_checkpoint_evaluation.py's
        test_train_only_recognizes_evaluated_stage_one_checkpoint, which
        exercised the same real-on-disk-checkpoint scenario against
        ActiveLearningStandardMACE.train_mlip before that class existed
        here as CommitteeUncertaintyWorkflow._train_mlip."""
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        for fit_idx in range(3):
            fit_dir = Path("results/al_loop_0/training", f"fit_{fit_idx}")
            fit_dir.mkdir(parents=True)
            model = fit_dir / "training_stagetwo.model"
            model.write_bytes(b"stage one checkpoint")
            metrics = prediction_metrics([_atoms_scored()])
            save_evaluation(fit_dir, model, {"test": metrics})

        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        with (
            patch(f"{_MODULE}.submit_n") as mock_submit_n,
            patch.object(wf, "_store_predictions_and_cleanup"),
            patch.object(
                wf, "_cross_loop_metrics_dataframe", return_value=pl.DataFrame()
            ),
        ):
            wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        mock_submit_n.assert_not_called()

    def test_raises_when_fewer_than_three_succeed(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        fake_entry = MagicMock()
        fake_entry.output_paths.return_value = [Path("does_not_exist.model")]

        with (
            patch.object(wf, "trainer", return_value=fake_entry),
            patch(f"{_MODULE}.submit_n", return_value=[None, None, None]),
            patch(f"{_MODULE}.get_remote_info"),
            pytest.raises(RuntimeError, match="only 0 trained model"),
        ):
            wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

    def test_phase_done_reloads_cached_models_without_training(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        Path("results/al_loop_0").mkdir(parents=True)
        Path("results/al_loop_0/train_mlip.done").write_text("done\n")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        fake_entry = MagicMock()
        fake_entry.read_existing_result.side_effect = lambda fit_dir: (
            f"model_{_fit_index(fit_dir)}.pt",
            None,
            {},
        )

        with (
            patch.object(wf, "trainer", return_value=fake_entry),
            patch(f"{_MODULE}.submit_n") as mock_submit_n,
        ):
            models = wf.train_models(_ctx(0), wf.seeds(3), min_successful=3)

        mock_submit_n.assert_not_called()
        assert [m.fit_idx for m in models] == [0, 1, 2]
        assert [m.seed for m in models] == [803, 804, 805]
        assert models[1].model_path == "model_1.pt"


def _atoms_scored():
    a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a.info.update(REF_energy=-8.0, model_energy=-8.0, config_type="d")
    a.set_array("REF_forces", np.zeros((2, 3)))
    a.set_array("model_forces", np.zeros((2, 3)))
    return a


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _model(fit_idx: int) -> TrainedModel:
    return TrainedModel(
        fit_idx=fit_idx,
        seed=803 + fit_idx,
        model_path=f"model_{fit_idx}.pt",
        compiled_model_path=None,
        metrics={},
        fit_dir=Path("results/al_loop_0/training", f"fit_{fit_idx}"),
    )


@pytest.mark.unit
class TestPredict:
    def test_one_batch_one_result_per_model(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        structures = [_atoms(), _atoms()]

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            return [
                {
                    "forces": [np.zeros((1, 3)), np.zeros((1, 3))],
                    "energies": [1.0, 2.0],
                    "model": jc["function_kwargs"]["model_path"],
                }
                for jc in job_configs
            ]

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            result = wf.predict(_ctx(0), [_model(2), _model(0)], structures)

        mock_submit.assert_called_once()
        assert mock_submit.call_args.args[0] is predict_with_model
        assert [r["model"] for r in result] == ["model_2.pt", "model_0.pt"]

    def test_raises_when_a_model_fails(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)

        with (
            patch(f"{_MODULE}.submit_n", return_value=[None]),
            patch(f"{_MODULE}.get_remote_info"),
            pytest.raises(RuntimeError, match="Prediction failed for fit_0"),
        ):
            wf.predict(_ctx(0), [_model(0)], [_atoms()])


# ---------------------------------------------------------------------------
# generate_candidates / high_accuracy_evaluate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateCandidates:
    def test_defaults_desired_number_of_structures_when_absent(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        """find_high_sd_structures/run_md (old, shared) both require this
        key with no default of their own -- the parent must apply one
        consistent default regardless of which generator module runs."""
        monkeypatch.chdir(tmp_path)
        del minimal_jobs_dict["structure_generation"]["desired_num_of_structures"]
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        fake_generator = MagicMock()
        fake_generator.generate.return_value = []

        with patch(f"{_MODULE}.resolve", return_value=fake_generator):
            wf.generate_candidates(_ctx(0, train=[_atoms()]), _model(0))

        assert wf.jobs_dict["structure_generation"]["desired_num_of_structures"] == 50

    def test_runs_generator_with_model_and_filters_short_bonds(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        collapsed = Atoms(
            "H2", positions=[[0, 0, 0], [0.1, 0, 0]], cell=[5] * 3, pbc=True
        )
        fake_generator = MagicMock()
        fake_generator.generate.return_value = [_atoms(), collapsed]

        with patch(f"{_MODULE}.resolve", return_value=fake_generator):
            result = wf.generate_candidates(_ctx(0, train=[_atoms()]), _model(1))

        assert fake_generator.generate.call_args.args[1] == "model_1.pt"
        assert len(result) == 1


@pytest.mark.unit
class TestHighAccuracyEvaluate:
    @pytest.mark.parametrize(
        ("force_ceiling", "relax"), [("default", True), (5.0, True), (None, False)]
    )
    def test_force_ceiling_decides_relaxation(
        self,
        tmp_path,
        minimal_jobs_dict,
        monkeypatch,
        shared_db,
        force_ceiling,
        relax,
    ):
        monkeypatch.chdir(tmp_path)
        if force_ceiling != "default":
            minimal_jobs_dict["high_accuracy_evaluation"]["force_ceiling"] = (
                force_ceiling
            )
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        stale = _atoms()
        stale.info["needs_relaxation"] = not relax  # inherited flag is overwritten
        seen = {}

        def fake_evaluate(structures, config, **kwargs):
            seen["structures"] = structures
            seen["config"] = config
            return []

        with patch(f"{_MODULE}._evaluator_orchestrate", side_effect=fake_evaluate):
            wf.high_accuracy_evaluate(_ctx(0), [stale])

        assert seen["structures"][0].info["needs_relaxation"] is relax
        assert seen["structures"][0].info["job_id"] == 0
        assert ("fmax" in seen["config"]) is relax

    def test_done_step_reloads_saved_results_without_dft(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        """high_accuracy_eval.done is honoured through @phase: the saved
        results are reloaded and relabelled, no DFT is submitted."""
        monkeypatch.chdir(tmp_path)
        loop_dir = Path("results/al_loop_2")
        loop_dir.mkdir(parents=True)
        saved = [_atoms(config_type="anything"), _atoms(config_type="anything")]
        for atoms in saved:  # the evaluator saves structures with DFT results
            atoms.calc = SinglePointCalculator(
                atoms, energy=-1.0, forces=np.zeros((len(atoms), 3))
            )
        write(str(loop_dir / "high_accuracy_eval_results.xyz"), saved)
        (loop_dir / "high_accuracy_eval.done").write_text("done\n")
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)

        with patch(f"{_MODULE}._evaluator_orchestrate") as orchestrate:
            result = wf.high_accuracy_evaluate(_ctx(2), [_atoms()])

        orchestrate.assert_not_called()
        assert len(result) == 2
        assert {a.info["config_type"] for a in result} == {"high_sd"}
        assert {a.info["al_loop"] for a in result} == {2}

    def test_second_call_in_same_loop_raises(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)

        with patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]):
            wf.high_accuracy_evaluate(_ctx(0), [_atoms()])
            with pytest.raises(RuntimeError, match="high_accuracy_eval"):
                wf.high_accuracy_evaluate(_ctx(0), [_atoms()])

        assert Path("results/al_loop_0/high_accuracy_eval.done").exists()

    def test_labels_with_workflow_config_type_and_loop(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        labelled = _atoms(config_type="whatever")

        with patch(f"{_MODULE}._evaluator_orchestrate", return_value=[labelled]):
            (result,) = wf.high_accuracy_evaluate(_ctx(3), [_atoms()])

        assert result.info["config_type"] == "high_sd"
        assert result.info["al_loop"] == 3


# ---------------------------------------------------------------------------
# _initialize_training_set
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeTrainingSet:
    def test_db_path_calls_initialiser_and_evaluator(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)

        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        fake_initialiser = MagicMock()
        fake_initialiser.compute_needs.return_value = {
            "isolated_atoms": ["H"],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": False,
        }
        fake_initialiser.generate.return_value = [_atoms(config_type="IsolatedAtom")]

        evaluated = _atoms(config_type="IsolatedAtom")

        with (
            patch(f"{_MODULE}.resolve", return_value=fake_initialiser),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[evaluated]),
            patch(
                f"{_MODULE}.clean_structures",
                side_effect=lambda structs, *a, **kw: structs,
            ),
        ):
            train_xyzs, test_xyzs = wf._initialize_training_set("initialization")

        fake_initialiser.generate.assert_called_once()
        assert len(train_xyzs) + len(test_xyzs) >= 1


# ---------------------------------------------------------------------------
# general.start_from -- warm/cold start modes (docs/starting_a_run.md)
# ---------------------------------------------------------------------------


def _labelled(i: int, config_type: str | None = "init_amorphous") -> Atoms:
    """A distinct 2-atom structure with ALomancy's canonical DFT labels."""
    a = Atoms(
        "H2", positions=[[0, 0, 0], [0.7 + 0.05 * i, 0, 0]], cell=[6] * 3, pbc=True
    )
    a.info["REF_energy"] = -1.0 - 0.01 * i
    a.arrays["REF_forces"] = np.zeros((2, 3))
    if config_type is not None:
        a.info["config_type"] = config_type
    return a


def _no_needs_initialiser() -> MagicMock:
    initialiser = MagicMock()
    initialiser.compute_needs.return_value = {
        "isolated_atoms": [],
        "dimer_override": {},
        "trimer_override": {},
        "amorphous_override": 0,
        "mp_structures": False,
    }
    return initialiser


@pytest.mark.unit
class TestParseStartFrom:
    @pytest.mark.parametrize(
        ("start_from", "mode"),
        [
            (None, "cold"),
            ({}, "cold"),
            ({"train_xyz": "a.xyz", "test_xyz": "b.xyz"}, "split_files"),
            ({"xyz": "a.xyz"}, "single_xyz"),
            ({"xyz": ["a.xyz", "b.xyz"]}, "single_xyz"),
            ({"database": "old/global_database"}, "database"),
        ],
    )
    def test_mode_detected_from_keys(self, start_from, mode):
        from alomancy.core.active_learning_workflow import _parse_start_from

        assert _parse_start_from(start_from)[0] == mode

    @pytest.mark.parametrize(
        ("start_from", "match"),
        [
            ({"train_xyz": "a.xyz"}, "must be given together"),
            ({"xyz": "a.xyz", "database": "db"}, "more than one data source"),
            ({"train_xyz": "a", "test_xyz": "b", "xyz": "c"}, "more than one"),
            ({"databse": "db"}, "Unknown general.start_from"),
            ({"database": "db", "metadata_map": {"energy": "e"}}, "only applies"),
            ({"xyz": []}, "path or a list"),
        ],
    )
    def test_invalid_start_from_raises(self, start_from, match):
        from alomancy.core.active_learning_workflow import _parse_start_from

        with pytest.raises(ValueError, match=match):
            _parse_start_from(start_from)

    @pytest.mark.parametrize(
        ("section", "key", "replacement"),
        [
            ("general", "initial_train_file_path", "start_from.train_xyz"),
            ("general", "initial_test_file_path", "start_from.test_xyz"),
            ("general", "skip_initialization", "start_from"),
            ("initialization", "extra_datasets", "start_from.xyz"),
            ("initialization", "reset_extra_splits", "nothing"),
            ("general", "high_force_threshold", "force_ceiling.*train_filter"),
        ],
    )
    def test_removed_keys_raise_with_replacement(
        self, tmp_path, workflow_jobs_dict, shared_db, section, key, replacement
    ):
        workflow_jobs_dict.setdefault(section, {})[key] = "x"
        with pytest.raises(ValueError, match=rf"{key} -> .*{replacement}"):
            _make_workflow(tmp_path, workflow_jobs_dict, shared_db)


def _starting_a_run_examples() -> list[dict]:
    import re

    import yaml

    page = Path(__file__).parents[2] / "docs" / "starting_a_run.md"
    blocks = re.findall(r"```yaml\n(.*?)```", page.read_text(), flags=re.S)
    return [yaml.safe_load(block) for block in blocks]


@pytest.mark.unit
def test_docs_start_examples_build_workflows(tmp_path, minimal_jobs_dict, shared_db):
    """Every YAML example in docs/starting_a_run.md is a valid config, and
    together they cover all four start modes."""
    from alomancy.core.committee_uncertainty_workflow import build_workflow

    examples = _starting_a_run_examples()
    assert examples
    modes = set()
    for example in examples:
        jobs_dict = {**minimal_jobs_dict, "general": example["general"]}
        jobs_dict["general"]["log_file"] = str(tmp_path / "docs.log")
        wf = build_workflow(jobs_dict=jobs_dict)
        wf.db = shared_db
        modes.add(wf.start_mode)
    assert modes == {"split_files", "single_xyz", "database", "cold"}


@pytest.mark.unit
class TestStartModes:
    def _init(self, wf, initialiser=None, evaluated=None):
        initialiser = initialiser or _no_needs_initialiser()
        with (
            patch(f"{_MODULE}.resolve", return_value=initialiser),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=evaluated or []),
            patch(
                f"{_MODULE}.clean_structures",
                side_effect=lambda structs, *a, **kw: structs,
            ),
        ):
            return wf._initialize_training_set("initialization")

    @staticmethod
    def _targets(jobs_dict, *types):
        jobs_dict["general"]["dataset_kwargs"]["target_config_types"] = list(types)

    def test_split_files_keep_their_split(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        write("train.xyz", [_labelled(i) for i in range(3)], format="extxyz")
        write("test.xyz", [_labelled(i) for i in range(3, 5)], format="extxyz")
        self._targets(workflow_jobs_dict, "init_amorphous")
        wf = _make_workflow(
            tmp_path,
            workflow_jobs_dict,
            shared_db,
            start_from={"train_xyz": "train.xyz", "test_xyz": "test.xyz"},
        )

        train, test = self._init(wf)

        assert (len(train), len(test)) == (3, 2)
        splits = sorted(a.info["split"] for a in shared_db.get_all_as_atoms())
        assert splits == ["test"] * 2 + ["train"] * 3
        assert Path("results/initialization/train_set.xyz").exists()

    def test_foreign_split_files_get_normalized_labels(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        foreign = []
        for i in range(2):
            a = _labelled(i, config_type=None)
            a.info["dft_energy"] = a.info.pop("REF_energy")
            a.arrays["dft_forces"] = a.arrays.pop("REF_forces")
            a.info["label"] = "bulk"
            foreign.append(a)
        write("train.xyz", foreign[:1], format="extxyz")
        write("test.xyz", foreign[1:], format="extxyz")
        wf = _make_workflow(
            tmp_path,
            workflow_jobs_dict,
            shared_db,
            start_from={"train_xyz": "train.xyz", "test_xyz": "test.xyz"},
        )

        train, test = self._init(wf)

        for atoms in train + test:
            assert atoms.info["config_type"] == "bulk"
            assert atoms.info["REF_energy"] < 0
            assert atoms.arrays["REF_forces"].shape == (2, 3)

    def test_single_xyz_split_by_test_ratio_over_targets(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        structures = [_labelled(i, "init_amorphous") for i in range(10)]
        structures += [_labelled(i, "init_dimer") for i in range(10, 13)]
        write("data.xyz", structures, format="extxyz")
        self._targets(workflow_jobs_dict, "init_amorphous")
        workflow_jobs_dict["general"]["dataset_kwargs"]["test_ratio"] = 0.2
        wf = _make_workflow(
            tmp_path, workflow_jobs_dict, shared_db, start_from={"xyz": "data.xyz"}
        )

        train, test = self._init(wf)

        assert len(train) + len(test) == 13
        assert test and all(a.info["config_type"] == "init_amorphous" for a in test)

    def test_single_xyz_of_external_structures_raises(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        write("data.xyz", [_labelled(i, None) for i in range(5)], format="extxyz")
        self._targets(workflow_jobs_dict, "init_amorphous")
        wf = _make_workflow(
            tmp_path, workflow_jobs_dict, shared_db, start_from={"xyz": "data.xyz"}
        )

        with pytest.raises(ValueError, match=r"'external'.*target_config_types"):
            self._init(wf)

    def test_single_xyz_external_allowed_when_targeted(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        write("data.xyz", [_labelled(i, None) for i in range(10)], format="extxyz")
        self._targets(workflow_jobs_dict, "external")
        wf = _make_workflow(
            tmp_path, workflow_jobs_dict, shared_db, start_from={"xyz": "data.xyz"}
        )

        train, test = self._init(wf)

        assert test
        assert {a.info["config_type"] for a in train + test} == {"external"}

    def test_reimport_is_idempotent(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        write("data.xyz", [_labelled(i) for i in range(6)], format="extxyz")
        self._targets(workflow_jobs_dict, "init_amorphous")
        wf = _make_workflow(
            tmp_path, workflow_jobs_dict, shared_db, start_from={"xyz": "data.xyz"}
        )

        self._init(wf)
        size = shared_db.size
        self._init(wf)

        assert shared_db.size == size == 6

    def test_database_copy_keeps_splits_and_leaves_source_alone(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        from alomancy.database.global_database import GlobalDatabase

        monkeypatch.chdir(tmp_path)
        old = GlobalDatabase(str(tmp_path / "old_run" / "global_database"))
        with_prediction = _labelled(0)
        with_prediction.info["model_energy_loop_0_fit_0"] = -1.0
        old.add_structures([with_prediction, _labelled(1)], split="train")
        old.add_structures([_labelled(2)], split="test")
        self._targets(workflow_jobs_dict, "init_amorphous")
        wf = _make_workflow(
            tmp_path,
            workflow_jobs_dict,
            shared_db,
            start_from={"database": "old_run/global_database"},
        )

        train, test = self._init(wf)
        self._init(wf)  # re-import is a no-op

        assert (len(train), len(test)) == (2, 1)
        assert shared_db.size == 3
        copied = shared_db.get_all_as_atoms()
        assert sorted(a.info["source_global_db_id"] for a in copied) == [0, 1, 2]
        assert not any(k.startswith("model_") for a in copied for k in a.info)
        assert GlobalDatabase(str(tmp_path / "old_run" / "global_database")).size == 3

    def test_warm_start_generates_only_missing_targets(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        write("data.xyz", [_labelled(i) for i in range(6)], format="extxyz")
        self._targets(workflow_jobs_dict, "init_amorphous")
        wf = _make_workflow(
            tmp_path, workflow_jobs_dict, shared_db, start_from={"xyz": "data.xyz"}
        )
        initialiser = _no_needs_initialiser()
        initialiser.compute_needs.return_value = {
            **initialiser.compute_needs.return_value,
            "isolated_atoms": ["H"],
        }
        isolated = _atoms(config_type="IsolatedAtom")
        initialiser.generate.return_value = [isolated]

        train, test = self._init(wf, initialiser=initialiser, evaluated=[isolated])

        initialiser.compute_needs.assert_called_once()
        initialiser.generate.assert_called_once()
        assert shared_db.size == 7
        assert len(train) + len(test) == 7

    def test_warm_start_with_nothing_missing_skips_generation(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        write("data.xyz", [_labelled(i) for i in range(6)], format="extxyz")
        self._targets(workflow_jobs_dict, "init_amorphous")
        wf = _make_workflow(
            tmp_path, workflow_jobs_dict, shared_db, start_from={"xyz": "data.xyz"}
        )
        initialiser = _no_needs_initialiser()

        self._init(wf, initialiser=initialiser)

        initialiser.generate.assert_not_called()


# ---------------------------------------------------------------------------
# run() -- loop control flow, originally copied from the now-removed
# BaseActiveLearningWorkflow; these confirm the copy wires correctly to
# this skeleton's own internal method names.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQualityFilterAndForceCeilingConfig:
    @pytest.mark.parametrize(
        ("force_ceiling", "expected_fmax"),
        [("default", 100.0), (5.0, 5.0), (None, None)],
    )
    def test_force_ceiling_sets_dft_relaxation_target(
        self,
        tmp_path,
        workflow_jobs_dict,
        monkeypatch,
        shared_db,
        force_ceiling,
        expected_fmax,
    ):
        monkeypatch.chdir(tmp_path)
        if force_ceiling != "default":
            workflow_jobs_dict["high_accuracy_evaluation"]["force_ceiling"] = (
                force_ceiling
            )
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db, num_of_al_loops=1)
        configs = []

        def fake_evaluate(structures, config, **kwargs):
            configs.append(config)
            return []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_models", return_value=[]),
            patch.object(wf, "select_uncertain", return_value=[_atoms()]),
            patch(f"{_MODULE}._evaluator_orchestrate", side_effect=fake_evaluate),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert configs[0].get("fmax") == expected_fmax

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), "100", True])
    def test_invalid_force_ceiling_raises(
        self, tmp_path, workflow_jobs_dict, shared_db, bad
    ):
        workflow_jobs_dict["high_accuracy_evaluation"]["force_ceiling"] = bad
        with pytest.raises(ValueError, match="force_ceiling"):
            _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

    @pytest.mark.parametrize("name", ["train_filter", "test_filter"])
    def test_invalid_split_filter_raises(
        self, tmp_path, workflow_jobs_dict, shared_db, name
    ):
        workflow_jobs_dict["general"][name] = {"max_force": -1}
        with pytest.raises(ValueError, match=f"general.{name}.max_force"):
            _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

    def test_filters_resolved_with_defaults_and_not_unknown_keys(
        self, tmp_path, workflow_jobs_dict, shared_db
    ):
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        module_logger = logging.getLogger("alomancy.core.active_learning_workflow")
        module_logger.addHandler(handler)
        try:
            wf = _make_workflow(
                tmp_path,
                workflow_jobs_dict,
                shared_db,
                test_filter={"formation_energy_per_atom": [None, 1.0]},
            )
        finally:
            module_logger.removeHandler(handler)

        assert wf.train_filter == {
            "max_force": 100.0,
            "formation_energy_per_atom": None,
        }
        assert wf.test_filter == {
            "max_force": None,
            "formation_energy_per_atom": [None, 1.0],
        }
        assert not any("unrecognised" in r.getMessage() for r in records)


@pytest.mark.unit
class TestRun:
    def _wf_with_mocks(self, tmp_path, workflow_jobs_dict, shared_db, **overrides):
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db, **overrides)
        return wf

    def test_loop_count(self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db):
        monkeypatch.chdir(tmp_path)
        wf = self._wf_with_mocks(tmp_path, workflow_jobs_dict, shared_db)
        train_calls = []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(
                wf,
                "train_models",
                side_effect=lambda *a, **kw: train_calls.append(1) or [],
            ),
            patch.object(wf, "select_uncertain", return_value=[]),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert len(train_calls) == 2  # num_of_al_loops=2

    def test_start_loop_respected(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = self._wf_with_mocks(
            tmp_path, workflow_jobs_dict, shared_db, num_of_al_loops=4, start_loop=2
        )
        train_calls = []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(
                wf,
                "train_models",
                side_effect=lambda *a, **kw: train_calls.append(1) or [],
            ),
            patch.object(wf, "select_uncertain", return_value=[]),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert len(train_calls) == 2  # loops 2 and 3 only

    def test_base_names_correct_for_loops(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = self._wf_with_mocks(tmp_path, workflow_jobs_dict, shared_db)
        train_calls = []

        def track_train(ctx, *args, **kwargs):
            train_calls.append(ctx.base_name)
            return []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_models", side_effect=track_train),
            patch.object(wf, "select_uncertain", return_value=[]),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert train_calls == ["al_loop_0", "al_loop_1"]

    def test_train_only_stops_before_generation(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        workflow_jobs_dict["general"]["train_only"] = True
        wf = self._wf_with_mocks(tmp_path, workflow_jobs_dict, shared_db)
        generate_calls = []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_models", return_value=[]),
            patch.object(
                wf,
                "select_uncertain",
                side_effect=lambda *a, **kw: generate_calls.append(1) or [],
            ),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert generate_calls == []

    def test_resumes_from_last_complete_loop(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        Path("results/al_loop_0").mkdir(parents=True)
        (Path("results/al_loop_0") / "loop.done").write_text("done\n")

        wf = self._wf_with_mocks(
            tmp_path, workflow_jobs_dict, shared_db, num_of_al_loops=3
        )
        init_calls = []
        train_calls = []

        with (
            patch.object(
                wf,
                "_initialize_training_set",
                side_effect=lambda *a, **kw: init_calls.append(1) or ([], []),
            ),
            patch.object(
                wf,
                "train_models",
                side_effect=lambda *a, **kw: train_calls.append(1) or [],
            ),
            patch.object(wf, "select_uncertain", return_value=[]),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        # Resumed past loop 0 -- initialize_training_set is skipped entirely,
        # and only loops 1, 2 run.
        assert init_calls == []
        assert len(train_calls) == 2


# ---------------------------------------------------------------------------
# _is_user_specified / _resolve_effective_phase_dict / display_workflow_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsUserSpecified:
    def test_absent_key_is_not_user_specified(self):
        assert _is_user_specified({}, "mace_kwargs.max_num_epochs") is False

    def test_present_nested_key_is_user_specified(self):
        raw = {"mace_kwargs": {"max_num_epochs": 40}}
        assert _is_user_specified(raw, "mace_kwargs.max_num_epochs") is True

    def test_sibling_key_not_present_is_not_user_specified(self):
        raw = {"mace_kwargs": {"batch_size": 8}}
        assert _is_user_specified(raw, "mace_kwargs.max_num_epochs") is False

    def test_nested_key_written_by_user_is_user_specified(self):
        raw = {"qe_kwargs": {"system": {"input_dft": "pbe"}}}
        assert _is_user_specified(raw, "qe_kwargs.system.input_dft") is True

    def test_nested_sibling_default_is_not_user_specified(self):
        # get_qe_input_data merges per namelist, so ecutwfc keeps its
        # default even though the user wrote another key under "system".
        raw = {"qe_kwargs": {"system": {"input_dft": "pbe"}}}
        assert _is_user_specified(raw, "qe_kwargs.system.ecutwfc") is False

    def test_top_level_key_present(self):
        assert _is_user_specified({"trainer": "mace"}, "trainer") is True

    def test_top_level_key_absent(self):
        assert _is_user_specified({}, "trainer") is False


@pytest.mark.unit
class TestResolveEffectivePhaseDict:
    def test_initialization_merges_defaults_per_namespace(self):
        phase_dict = {
            "dimer_kwargs": {"num_of_dimers_per_combo": 3},
            "mp_kwargs": {"enabled": False},
        }
        effective = _resolve_effective_phase_dict("initialization", phase_dict)
        assert effective["dimer_kwargs"]["num_of_dimers_per_combo"] == 3
        assert effective["dimer_kwargs"]["enabled"] is True  # default, not overridden
        assert effective["mp_kwargs"]["enabled"] is False
        assert effective["mp_kwargs"]["max_num_of_atoms"] == 20  # default
        assert "stretch_compress_targets_kwargs" in effective
        assert "isolated_atom_kwargs" in effective

    def test_training_merges_mace_defaults_and_e0s_placeholder(self):
        phase_dict = {"trainer": "mace", "mace_kwargs": {"max_num_epochs": 40}}
        effective = _resolve_effective_phase_dict("training", phase_dict)
        mk = effective["mace_kwargs"]
        assert mk["max_num_epochs"] == 40
        assert mk["energy_key"] == "REF_energy"  # default
        assert mk["E0s"] == "<resolved at train time from IsolatedAtom structures>"

    def test_training_e0s_not_placeholder_when_user_sets_it(self):
        phase_dict = {"trainer": "mace", "mace_kwargs": {"E0s": {"H": -1.0}}}
        effective = _resolve_effective_phase_dict("training", phase_dict)
        assert effective["mace_kwargs"]["E0s"] == {"H": -1.0}

    def test_structure_generation_merges_md_defaults_and_desired_number(self):
        phase_dict = {"generator": "md", "md_kwargs": {"steps": 500}}
        effective = _resolve_effective_phase_dict("structure_generation", phase_dict)
        assert effective["md_kwargs"]["steps"] == 500
        assert effective["md_kwargs"]["temperature"] == 300  # default
        assert effective["desired_num_of_structures"] == 50  # default

    def test_structure_generation_respects_existing_desired_number(self):
        phase_dict = {"generator": "md", "desired_num_of_structures": 10}
        effective = _resolve_effective_phase_dict("structure_generation", phase_dict)
        assert effective["desired_num_of_structures"] == 10

    def test_structure_generation_ezga_defaults(self):
        phase_dict = {"generator": "ezga", "ezga_kwargs": {"population_size": 10}}
        effective = _resolve_effective_phase_dict("structure_generation", phase_dict)
        ek = effective["ezga_kwargs"]
        assert ek["population_size"] == 10
        assert ek["max_generations"] == 2  # default

    def test_high_accuracy_evaluation_merges_qe_defaults(self):
        phase_dict = {
            "evaluator": "qe",
            "qe_kwargs": {"system": {"input_dft": "pbesol"}},
        }
        effective = _resolve_effective_phase_dict(
            "high_accuracy_evaluation", phase_dict
        )
        system = effective["qe_kwargs"]["system"]
        assert system["input_dft"] == "pbesol"
        assert system["ecutwfc"] == 40.0  # per-namelist merge keeps defaults
        assert effective["qe_kwargs"]["control"]["calculation"] == "scf"  # default

    def test_high_accuracy_evaluation_merges_vasp_defaults(self):
        phase_dict = {"evaluator": "vasp", "vasp_kwargs": {"encut": 600}}
        effective = _resolve_effective_phase_dict(
            "high_accuracy_evaluation", phase_dict
        )
        vk = effective["vasp_kwargs"]
        assert vk["encut"] == 600
        assert vk["pp"] == "PBE"  # default

    def test_does_not_mutate_input_phase_dict(self):
        phase_dict = {"generator": "md", "md_kwargs": {"steps": 500}}
        _resolve_effective_phase_dict("structure_generation", phase_dict)
        assert phase_dict == {"generator": "md", "md_kwargs": {"steps": 500}}

    def test_general_merges_dataset_and_workflow_kwargs_defaults(self):
        phase_dict = {
            "al_workflow": "committee_uncertainty",
            "elements": ["H"],
            "dataset_kwargs": {"test_ratio": 0.2},
        }
        effective = _resolve_effective_phase_dict("general", phase_dict)
        assert effective["dataset_kwargs"]["test_ratio"] == 0.2
        assert effective["dataset_kwargs"]["valid_fraction"] == 0.05  # default
        assert (
            effective["committee_uncertainty_kwargs"]["num_of_models_in_committee"] == 3
        )
        assert effective["elements"] == ["H"]  # untouched sibling

    def test_general_defaults_al_workflow_to_committee_uncertainty(self):
        phase_dict = {"elements": ["H"]}
        effective = _resolve_effective_phase_dict("general", phase_dict)
        assert "committee_uncertainty_kwargs" in effective
        assert (
            effective["committee_uncertainty_kwargs"]["num_of_models_in_committee"] == 3
        )

    def test_general_random_selection_has_no_workflow_block(self):
        effective = _resolve_effective_phase_dict(
            "general", {"al_workflow": "random_selection"}
        )
        assert "committee_uncertainty_kwargs" not in effective
        assert "dataset_kwargs" in effective


@pytest.mark.unit
class TestDisplayWorkflowSummary:
    def _capture(self, fn):
        # setup_logging sets propagate=False on the "alomancy" logger, so
        # records are captured by attaching a handler directly to it,
        # rather than via pytest's caplog (see CLAUDE.md's logging note).
        import logging

        al_logger = logging.getLogger("alomancy")
        records: list[str] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _Collector()
        al_logger.addHandler(handler)
        try:
            fn()
        finally:
            al_logger.removeHandler(handler)
        return "\n".join(records)

    def test_marks_user_specified_values_and_shows_defaults(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        workflow_jobs_dict["structure_generation"]["md_kwargs"] = {"steps": 500}
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        summary = self._capture(wf.display_workflow_summary)

        assert "md_kwargs.steps: 500  [user-specified]" in summary
        # temperature wasn't set by the user -- shown, but unmarked.
        assert "md_kwargs.temperature: 300\n" in summary + "\n"
        assert "md_kwargs.temperature: 300  [user-specified]" not in summary

    def test_max_time_never_marked(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        workflow_jobs_dict["training"]["max_time"] = "12:00:00"
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        summary = self._capture(wf.display_workflow_summary)

        assert "max_time: 12:00:00\n" in summary + "\n"
        assert "max_time: 12:00:00  [user-specified]" not in summary

    def test_empty_but_present_initialization_still_shown(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        """An empty initialization section ("initialization: {}") is a
        valid, meaningful config now -- every one of its settings defaults
        to enabled -- so it must still show its resolved defaults, not be
        silently skipped the way a genuinely absent section is."""
        monkeypatch.chdir(tmp_path)
        workflow_jobs_dict["initialization"] = {}
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        summary = self._capture(wf.display_workflow_summary)

        assert "--- Initialisation (initialization) ---" in summary
        assert "dimer_kwargs.enabled: True" in summary

    def test_general_block_shown_first_with_defaults_and_markers(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        summary = self._capture(wf.display_workflow_summary)

        general_pos = summary.index("--- General ---")
        first_phase_pos = summary.index("--- Initialisation")
        assert general_pos < first_phase_pos
        # test_ratio was set by workflow_jobs_dict's fixture -- user-specified.
        assert "dataset_kwargs.test_ratio: 0.1  [user-specified]" in summary
        # num_of_models_in_committee was also set by the fixture.
        assert "committee_uncertainty_kwargs.num_of_models_in_committee:" in summary
        # grouped_splits wasn't set by the user -- shown, but unmarked.
        assert "dataset_kwargs.grouped_splits: False\n" in summary + "\n"
        assert "dataset_kwargs.grouped_splits: False  [user-specified]" not in summary
        assert "elements: ['H']  [user-specified]" in summary


@pytest.mark.unit
class TestFlattenSettingsDepth:
    def test_flattens_three_levels_deep(self):
        d = {"a": {"b": {"c": 1}}}
        assert _flatten_settings(d) == [("a.b.c", 1)]


# ---------------------------------------------------------------------------
# phase decorator, abstract parent, train_models edge cases
# ---------------------------------------------------------------------------


class _Probe(CommitteeUncertaintyWorkflow):
    """A child with one extra phase, to exercise the decorator directly."""

    ran: list

    @phase("probe", load=lambda self, ctx, value: f"loaded {value}")
    def probe(self, ctx, value):
        self.ran.append(value)
        return f"ran {value}"


@pytest.mark.unit
class TestPhaseDecorator:
    def _probe(self, tmp_path, workflow_jobs_dict, shared_db):
        workflow_jobs_dict.setdefault("general", {})
        wf = _Probe(jobs_dict=workflow_jobs_dict)
        wf.db = shared_db
        wf.ran = []
        return wf

    def test_runs_then_marks_done(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = self._probe(tmp_path, workflow_jobs_dict, shared_db)

        assert wf.probe(_ctx(0), 1) == "ran 1"
        assert wf.ran == [1]
        assert Path("results/al_loop_0/probe.done").exists()

    def test_done_phase_uses_loader(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        Path("results/al_loop_0").mkdir(parents=True)
        Path("results/al_loop_0/probe.done").write_text("done\n")
        wf = self._probe(tmp_path, workflow_jobs_dict, shared_db)

        assert wf.probe(_ctx(0), 2) == "loaded 2"
        assert wf.ran == []

    def test_second_call_in_same_loop_raises(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = self._probe(tmp_path, workflow_jobs_dict, shared_db)
        wf.probe(_ctx(0), 1)

        with pytest.raises(RuntimeError, match="already ran in al_loop_0"):
            wf.probe(_ctx(0), 2)
        # A different loop, or an explicit phase_name, is fine.
        assert wf.probe(_ctx(1), 3) == "ran 3"
        assert wf.probe(_ctx(0), 4, phase_name="probe_again") == "ran 4"
        assert Path("results/al_loop_0/probe_again.done").exists()


@pytest.mark.unit
def test_parent_cannot_be_instantiated(workflow_jobs_dict):
    from alomancy.core.active_learning_workflow import ActiveLearningWorkflow

    with pytest.raises(TypeError, match="abstract"):
        ActiveLearningWorkflow(jobs_dict=workflow_jobs_dict)


@pytest.mark.unit
class TestTrainModelsEdgeCases:
    def _prepare(self):
        workdir = Path("results/al_loop_0")
        workdir.mkdir(parents=True)
        write(str(workdir / "train_set.xyz"), [_atoms() for _ in range(5)])
        write(str(workdir / "test_set.xyz"), [_atoms()])

    def test_single_seed_single_model(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare()
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        fake_submit, calls = trainer.submit_n(lambda fit_idx, attempt: "ok")

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit) as submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            (model,) = wf.train_models(_ctx(0), [42])

        job_configs = submit_n.call_args.args[1]
        assert calls == [[0]]
        assert job_configs[0]["function_kwargs"]["seed"] == 42
        assert model.seed == 42 and model.fit_idx == 0
        recorded = Path("results/al_loop_0/training/fit_0/fit_seed.json")
        assert json.loads(recorded.read_text()) == {"seed": 42}

    def test_min_successful_one_tolerates_failures(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare()
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        trainer = _FakeTrainer()
        fake_submit, calls = trainer.submit_n(
            lambda fit_idx, attempt: "fail" if fit_idx == 0 else "ok"
        )

        with (
            patch.object(wf, "trainer", return_value=trainer),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit),
            patch(f"{_MODULE}.get_remote_info"),
        ):
            models = wf.train_models(_ctx(0), wf.seeds(2))

        assert calls == [[0, 1], [0]]
        assert [m.fit_idx for m in models] == [1]

    def test_cached_fit_with_other_seed_is_reused_with_warning(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare()
        fit_dir = Path("results/al_loop_0/training/fit_0")
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit_seed.json").write_text(json.dumps({"seed": 1}))
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        fake_entry = MagicMock()
        fake_entry.output_paths.return_value = [fit_dir]
        fake_entry.read_existing_result.return_value = ("cached.pt", None, {})
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        module_logger = logging.getLogger(_MODULE)
        module_logger.addHandler(handler)
        try:
            with (
                patch.object(wf, "trainer", return_value=fake_entry),
                patch(f"{_MODULE}.submit_n") as submit_n,
            ):
                (model,) = wf.train_models(_ctx(0), [803])
        finally:
            module_logger.removeHandler(handler)

        submit_n.assert_not_called()
        assert model.model_path == "cached.pt"
        assert any(
            "trained with seed 1" in r.getMessage()
            for r in records
            if r.levelno == logging.WARNING
        )


# ---------------------------------------------------------------------------
# End to end through the generic trainer path (no MACE)
# ---------------------------------------------------------------------------


class _EMTTrainer(ALomancyTrainer):
    """A backend-free trainer: fit() writes a placeholder model, the
    "model" is ASE's EMT, and a fake compiled copy is what gets deployed."""

    NAME = "emt_e2e"
    KWARGS_KEY = "emt_e2e_kwargs"

    def model_path(self, fit_dir):
        return Path(fit_dir) / f"{self.name}.model"

    def deployable_model_path(self, fit_dir):
        path = Path(fit_dir) / f"{self.name}_deployed.model"
        return path if path.exists() else None

    def get_calculator(self, model_path):
        from ase.calculators.emt import EMT

        return EMT()

    def fit(
        self,
        train_path,
        valid_path,
        test_path,
        seed,
        fit_dir,
        *,
        isolated_atom_energies,
    ):
        (fit_dir / f"{self.name}_deployed.model").write_bytes(
            f"deployed {seed}".encode()
        )
        model = self.model_path(fit_dir)
        model.write_bytes(f"model {seed}".encode())
        return model


@pytest.mark.unit
def test_train_models_end_to_end_through_a_registered_trainer(
    tmp_path, workflow_jobs_dict, monkeypatch, shared_db
):
    from alomancy.registry import _REGISTRY, register

    monkeypatch.chdir(tmp_path)
    register("mlip_trainer", "emt_e2e", __name__, trainer_class="_EMTTrainer")
    try:
        workflow_jobs_dict["training"]["trainer"] = "emt_e2e"
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        cu = [
            Atoms(
                "Cu2",
                positions=[[0, 0, 0], [2.3 + 0.05 * i, 0, 0]],
                cell=[8] * 3,
                pbc=True,
            )
            for i in range(6)
        ]
        for a in cu:
            a.info.update(config_type="init_dimer", REF_energy=0.5)
            a.arrays["REF_forces"] = np.zeros((2, 3))
        shared_db.add_structures(cu[:5], split="train", skip_duplicates=False)
        shared_db.add_structures(cu[5:], split="test", skip_duplicates=False)
        workdir = Path("results/al_loop_0")
        workdir.mkdir(parents=True)
        write(workdir / "train_set.xyz", shared_db.get_train_atoms(), format="extxyz")
        write(workdir / "test_set.xyz", shared_db.get_test_atoms(), format="extxyz")

        def in_process(function, job_configs, remote_info, **kwargs):
            return [function(**jc["function_kwargs"]) for jc in job_configs]

        with (
            patch(f"{_MODULE}.submit_n", side_effect=in_process),
            patch(f"{_MODULE}.get_remote_info"),
        ):
            models = wf.train_models(_ctx(0), wf.seeds(2))
    finally:
        del _REGISTRY["mlip_trainer"]["emt_e2e"]

    assert [m.fit_idx for m in models] == [0, 1]
    fit_dir = Path("results/al_loop_0/training/fit_0")
    for name in (
        "training.model",
        "train_pred.xyz",
        "test_pred.xyz",
        "evaluation_metrics.json",
    ):
        assert (fit_dir / name).exists(), name
    assert models[0].compiled_model_path == str(fit_dir / "training_deployed.model")
    assert (
        Path("results/best_model/ALomancy_best_model.model")
        .read_bytes()
        .startswith(b"deployed")
    )
    assert shared_db.get_model_predictions(0, 0) is not None
