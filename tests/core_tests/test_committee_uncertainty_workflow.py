"""Tests for core/committee_uncertainty_workflow.py -- the concrete AL
skeleton for the modular architecture.

The loop-control logic in run() (restart, plotting call sites, train_only/
fixed_test branching, DB writes) was originally copied largely as-is from
the now-removed BaseActiveLearningWorkflow.run() -- these tests focus on
what's new: registry-resolved module dispatch, the shared validation
split, the partial-aware per-fit restart mechanism, generalized committee
scoring, and cross-loop metrics aggregation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import write

from alomancy.core.committee_uncertainty_workflow import (
    CommitteeUncertaintyWorkflow,
    _flatten_settings,
    _is_user_specified,
    _resolve_effective_phase_dict,
    _score_structures_with_member,
    _select_validation_split,
    build_workflow,
)
from alomancy.mlip.evaluation import prediction_metrics, save_evaluation

_MODULE = "alomancy.core.committee_uncertainty_workflow"


# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_jobs_dict(minimal_jobs_dict):
    """minimal_jobs_dict plus the new `general` section (renamed from
    `workflow`; decision 14: split-building parameters and
    number_models_in_committee -- renamed from size_of_committee -- move
    here from initialization/mlip_committee, nested under
    committee_uncertainty_kwargs to match the <dispatch_value>_kwargs
    convention; mlip_committee is renamed training)."""
    committee_size = minimal_jobs_dict["mlip_committee"]["size_of_committee"]
    minimal_jobs_dict["general"] = {
        "al_workflow": "committee_uncertainty",
        "elements": ["H"],
        "committee_uncertainty_kwargs": {
            "target_config_types": ["IsolatedAtom"],
            "test_ratio": 0.1,
            "valid_fraction": 0.05,
            "number_models_in_committee": committee_size,
        },
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["size_of_committee"]
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
    general.update(
        {
            "initial_train_file_path": str(tmp_path / "train.xyz"),
            "initial_test_file_path": str(tmp_path / "test.xyz"),
            "number_of_al_loops": 2,
            "plots": False,
            **general_overrides,
        }
    )
    wf = CommitteeUncertaintyWorkflow(jobs_dict=jobs_dict)
    wf.db = shared_db
    return wf


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
        workflow_jobs_dict["general"].update(
            {
                "initial_train_file_path": str(tmp_path / "train.xyz"),
                "initial_test_file_path": str(tmp_path / "test.xyz"),
            }
        )
        wf = build_workflow(jobs_dict=workflow_jobs_dict)
        wf.db = shared_db
        assert isinstance(wf, CommitteeUncertaintyWorkflow)

    def test_defaults_to_committee_uncertainty_when_absent(
        self, tmp_path, minimal_jobs_dict, shared_db
    ):
        minimal_jobs_dict["general"] = {
            "initial_train_file_path": str(tmp_path / "train.xyz"),
            "initial_test_file_path": str(tmp_path / "test.xyz"),
        }
        wf = build_workflow(jobs_dict=minimal_jobs_dict)
        wf.db = shared_db
        assert isinstance(wf, CommitteeUncertaintyWorkflow)

    def test_raises_on_unknown_al_workflow(self, minimal_jobs_dict):
        minimal_jobs_dict["general"] = {"al_workflow": "furthest_point_sampling"}
        with pytest.raises(ValueError, match=r"Unknown general\.al_workflow"):
            build_workflow(jobs_dict=minimal_jobs_dict)


# ---------------------------------------------------------------------------
# _select_validation_split (same logic as mace_wfl's, new skeleton-owned copy)
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
# _score_structures_with_member
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoreStructuresWithMember:
    def test_resolves_calculator_and_scores_each_structure(self):
        fake_calc = MagicMock()
        structures = [_atoms(), _atoms()]
        for atoms in structures:
            atoms.calc = None

        with patch(f"{_MODULE}.resolve") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                get_calculator=MagicMock(return_value=fake_calc)
            )
            with (
                patch.object(Atoms, "get_forces", lambda self: np.ones((1, 3))),
                patch.object(Atoms, "get_potential_energy", lambda self: -1.0),
            ):
                result = _score_structures_with_member(
                    structures, "model.pt", "mace", {"device": "cpu"}
                )

        mock_resolve.assert_called_once_with("mlip_trainer", "mace")
        assert len(result["forces"]) == 2
        assert len(result["energies"]) == 2
        assert all(atoms.calc is fake_calc for atoms in structures)


# ---------------------------------------------------------------------------
# _cross_loop_metrics_dataframe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossLoopMetricsDataframe:
    def _write_fit(self, fit_dir: Path, error: float) -> None:
        fit_dir.mkdir(parents=True)
        model = fit_dir / "committee_stagetwo.model"
        model.write_bytes(b"checkpoint")
        a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
        a.info.update(REF_energy=-8.0, model_energy=-8.0 + 2 * error, config_type="d")
        a.set_array("REF_forces", np.zeros((2, 3)))
        a.set_array("model_forces", np.ones((2, 3)) * error)
        save_evaluation(fit_dir, model, {"test": prediction_metrics([a])})

    def test_aggregates_across_loops_and_fits(self, tmp_path, monkeypatch, shared_db):
        monkeypatch.chdir(tmp_path)
        for loop in range(2):
            for fit_idx, error in enumerate([0.1, 0.2, 0.3]):
                self._write_fit(
                    Path(f"results/al_loop_{loop}/committee/fit_{fit_idx}"), error
                )

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        df = wf._cross_loop_metrics_dataframe("committee")

        assert len(df) == 2
        assert "mae_f" in df.columns
        assert "mae_f_std_dev" in df.columns

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

        wf = _make_workflow(tmp_path, {"initialization": {}}, shared_db)
        results = {0: ("model.pt", "model_compiled.pt", {"test": {}})}

        with (
            patch(
                f"{_MODULE}.read_mace_eval_predictions",
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

        with patch(f"{_MODULE}.read_mace_eval_predictions", return_value={}):
            wf._store_predictions_and_cleanup("al_loop_0", "committee", results)

        assert checkpoints_dir.exists()


# ---------------------------------------------------------------------------
# _train_mlip
# ---------------------------------------------------------------------------


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

        fake_entry = MagicMock()
        fake_entry.output_paths.return_value = [Path("does_not_exist.model")]
        fake_result = (
            "model.pt",
            None,
            {"test": prediction_metrics([_atoms_scored()])},
        )

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            return [fake_result] * len(job_configs)

        with (
            patch(f"{_MODULE}.resolve", return_value=fake_entry),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
            patch.object(wf, "_store_predictions_and_cleanup"),
            patch.object(
                wf, "_cross_loop_metrics_dataframe", return_value=pd.DataFrame()
            ),
        ):
            wf._train_mlip("al_loop_0")

        job_configs = mock_submit_n.call_args.args[1]
        assert (
            len(job_configs)
            == workflow_jobs_dict["general"]["committee_uncertainty_kwargs"][
                "number_models_in_committee"
            ]
        )
        # isolated_atom_e0s is computed once locally from the DB and passed
        # to every fit so trainer.train() can default mace_fit_kwargs.E0s
        # when the config doesn't set it explicitly.
        assert (
            job_configs[0]["function_kwargs"]["isolated_atom_e0s"]
            == wf.db.get_isolated_atom_energies()
        )

    def test_reuses_cached_fits_and_only_submits_missing(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        self._prepare_loop_dir("al_loop_0")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        cached_result = (
            "cached.pt",
            None,
            {"test": prediction_metrics([_atoms_scored()])},
        )
        fresh_result = (
            "fresh.pt",
            None,
            {"test": prediction_metrics([_atoms_scored()])},
        )

        fake_entry = MagicMock()

        def fake_output_paths(config, *, base_name, name, fit_idx):
            # fit_0 is "cached" (path exists), fit_1/fit_2 are missing.
            return [Path("cached.marker")] if fit_idx == 0 else [Path("missing.marker")]

        fake_entry.output_paths.side_effect = fake_output_paths
        fake_entry.read_existing_result.return_value = cached_result

        with (
            patch("pathlib.Path.exists", lambda self: self.name == "cached.marker"),
            patch(f"{_MODULE}.resolve", return_value=fake_entry),
            patch(
                f"{_MODULE}.submit_n",
                return_value=[fresh_result, fresh_result],
            ) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
            patch.object(wf, "_store_predictions_and_cleanup"),
            patch.object(
                wf, "_cross_loop_metrics_dataframe", return_value=pd.DataFrame()
            ),
        ):
            wf._train_mlip("al_loop_0")

        job_configs = mock_submit_n.call_args.args[1]
        assert len(job_configs) == 2  # only fit_1 and fit_2 submitted
        fit_indices_submitted = {jc["function_kwargs"]["fit_idx"] for jc in job_configs}
        assert fit_indices_submitted == {1, 2}

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
                wf, "_cross_loop_metrics_dataframe", return_value=pd.DataFrame()
            ),
        ):
            wf._train_mlip("al_loop_0")

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
            patch(f"{_MODULE}.resolve", return_value=fake_entry),
            patch(f"{_MODULE}.submit_n", return_value=[None, None, None]),
            patch(f"{_MODULE}.get_remote_info"),
            pytest.raises(RuntimeError, match="only 0 trained model"),
        ):
            wf._train_mlip("al_loop_0")

    def test_phase_done_reloads_cached_dataframe(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        Path("results/al_loop_0").mkdir(parents=True)
        Path("results/al_loop_0/train_mlip.done").write_text("done\n")
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)

        with (
            patch(f"{_MODULE}.resolve") as mock_resolve,
            patch.object(
                wf,
                "_cross_loop_metrics_dataframe",
                return_value=pd.DataFrame({"a": [1]}),
            ),
        ):
            result = wf._train_mlip("al_loop_0")

        mock_resolve.assert_not_called()
        assert len(result) == 1


def _atoms_scored():
    a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a.info.update(REF_energy=-8.0, model_energy=-8.0, config_type="d")
    a.set_array("REF_forces", np.zeros((2, 3)))
    a.set_array("model_forces", np.zeros((2, 3)))
    return a


# ---------------------------------------------------------------------------
# _score_committee
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoreCommittee:
    def test_builds_structure_forces_dict_with_base_mlip_and_fit_labels(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        structures = [_atoms(), _atoms()]

        fake_entry = MagicMock()
        fake_entry.read_existing_result.side_effect = lambda *a, **kw: (
            f"model_{kw['fit_idx']}.pt",
            None,
            {},
        )

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            return [
                {"forces": [np.zeros((1, 3)), np.zeros((1, 3))], "energies": [1.0, 2.0]}
                for _ in job_configs
            ]

        with (
            patch(f"{_MODULE}.resolve", return_value=fake_entry),
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n),
            patch(f"{_MODULE}.get_remote_info"),
        ):
            result = wf._score_committee(
                structures,
                "al_loop_0",
                0,
                [1, 2],
                "mace",
                {"name": "committee"},
                "committee",
                {},
                "1H",
            )

        assert set(result) == {"base_mlip", "fit_1", "fit_2"}
        assert set(result["base_mlip"]) == {"structure_0", "structure_1"}

    def test_raises_when_a_member_scoring_fails(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        fake_entry = MagicMock()
        fake_entry.read_existing_result.return_value = ("m.pt", None, {})

        with (
            patch(f"{_MODULE}.resolve", return_value=fake_entry),
            patch(f"{_MODULE}.submit_n", return_value=[None]),
            patch(f"{_MODULE}.get_remote_info"),
            pytest.raises(RuntimeError, match="Committee scoring failed"),
        ):
            wf._score_committee(
                [_atoms()],
                "al_loop_0",
                0,
                [],
                "mace",
                {"name": "committee"},
                "committee",
                {},
                "1H",
            )


# ---------------------------------------------------------------------------
# _generate_structures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateStructures:
    def test_reuses_existing_high_sd_file(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        sg_dir = Path("results/al_loop_0/structure_generation")
        sg_dir.mkdir(parents=True)
        write(
            str(sg_dir / "high_sd_structures.xyz"),
            [_atoms(), _atoms()],
            format="extxyz",
        )

        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        result = wf._generate_structures("al_loop_0", [])

        assert len(result) == 2
        assert all(a.info.get("needs_relaxation") is True for a in result)

    def test_defaults_desired_number_of_structures_when_absent(
        self, tmp_path, minimal_jobs_dict, monkeypatch, shared_db
    ):
        """find_high_sd_structures/run_md (old, shared) both require this
        key with no default of their own -- the skeleton must apply one
        consistent default regardless of which generator module runs."""
        monkeypatch.chdir(tmp_path)
        del minimal_jobs_dict["structure_generation"]["desired_number_of_structures"]
        sg_dir = Path("results/al_loop_0/structure_generation")
        sg_dir.mkdir(parents=True)
        write(str(sg_dir / "high_sd_structures.xyz"), [_atoms()], format="extxyz")

        wf = _make_workflow(tmp_path, minimal_jobs_dict, shared_db)
        wf._generate_structures("al_loop_0", [])

        assert (
            wf.jobs_dict["structure_generation"]["desired_number_of_structures"] == 50
        )

    def test_full_path_calls_generator_and_scores_committee(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        train_atoms = [_atoms() for _ in range(5)]
        generated = [_atoms(), _atoms()]

        fake_generator = MagicMock()
        fake_generator.generate.return_value = generated

        with (
            patch(f"{_MODULE}.resolve", return_value=fake_generator),
            patch(
                f"{_MODULE}.select_best_committee_model",
                return_value=(0, Path("model.pt")),
            ),
            patch.object(
                wf,
                "_score_committee",
                return_value={
                    "base_mlip": {
                        f"structure_{i}": {"forces": np.zeros((1, 3)), "energy": 0.0}
                        for i in range(2)
                    },
                    "fit_1": {
                        f"structure_{i}": {"forces": np.zeros((1, 3)), "energy": 0.0}
                        for i in range(2)
                    },
                },
            ),
            patch(
                f"{_MODULE}.find_high_sd_structures",
                return_value=generated,
            ),
        ):
            result = wf._generate_structures("al_loop_0", train_atoms)

        fake_generator.generate.assert_called_once()
        assert len(result) == 2
        assert all("job_id" in a.info for a in result)


# ---------------------------------------------------------------------------
# _initialize_training_set
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeTrainingSet:
    def test_fast_path_loads_existing_files(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        train_path = tmp_path / "train.xyz"
        test_path = tmp_path / "test.xyz"
        write(str(train_path), [_atoms(), _atoms()], format="extxyz")
        write(str(test_path), [_atoms()], format="extxyz")

        wf = _make_workflow(tmp_path, workflow_jobs_dict, shared_db)
        with patch(f"{_MODULE}.resolve") as mock_resolve:
            train_xyzs, test_xyzs = wf._initialize_training_set("initialization")

        mock_resolve.assert_not_called()
        assert len(train_xyzs) == 2
        assert len(test_xyzs) == 1

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
# run() -- loop control flow, originally copied from the now-removed
# BaseActiveLearningWorkflow; these confirm the copy wires correctly to
# this skeleton's own internal method names.
# ---------------------------------------------------------------------------


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
                "_train_mlip",
                side_effect=lambda *a, **kw: train_calls.append(1) or pd.DataFrame(),
            ),
            patch.object(wf, "_generate_structures", return_value=[]),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert len(train_calls) == 2  # number_of_al_loops=2

    def test_start_loop_respected(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        wf = self._wf_with_mocks(
            tmp_path, workflow_jobs_dict, shared_db, number_of_al_loops=4, start_loop=2
        )
        train_calls = []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(
                wf,
                "_train_mlip",
                side_effect=lambda *a, **kw: train_calls.append(1) or pd.DataFrame(),
            ),
            patch.object(wf, "_generate_structures", return_value=[]),
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

        def track_train(base_name, *args, **kwargs):
            train_calls.append(base_name)
            return pd.DataFrame()

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(wf, "_train_mlip", side_effect=track_train),
            patch.object(wf, "_generate_structures", return_value=[]),
            patch(f"{_MODULE}._evaluator_orchestrate", return_value=[]),
            patch(f"{_MODULE}.write"),
        ):
            wf.run()

        assert train_calls == ["al_loop_0", "al_loop_1"]

    def test_train_only_stops_before_generation(
        self, tmp_path, workflow_jobs_dict, monkeypatch, shared_db
    ):
        monkeypatch.chdir(tmp_path)
        workflow_jobs_dict["general"]["committee_uncertainty_kwargs"]["train_only"] = (
            True
        )
        wf = self._wf_with_mocks(tmp_path, workflow_jobs_dict, shared_db)
        generate_calls = []

        with (
            patch.object(wf, "_initialize_training_set", return_value=([], [])),
            patch.object(wf, "_train_mlip", return_value=pd.DataFrame()),
            patch.object(
                wf,
                "_generate_structures",
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
            tmp_path, workflow_jobs_dict, shared_db, number_of_al_loops=3
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
                "_train_mlip",
                side_effect=lambda *a, **kw: train_calls.append(1) or pd.DataFrame(),
            ),
            patch.object(wf, "_generate_structures", return_value=[]),
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

    def test_whole_subdict_user_supplied_marks_every_key_beneath(self):
        # get_qe_input_data's shallow per-section merge replaces "system"
        # wholesale -- any key surviving under it in the effective dict came
        # entirely from the user, even if they only wrote one of several.
        raw = {"qe_kwargs": {"system": {"input_dft": "pbe"}}}
        assert _is_user_specified(raw, "qe_kwargs.system.input_dft") is True

    def test_top_level_key_present(self):
        assert _is_user_specified({"trainer": "mace"}, "trainer") is True

    def test_top_level_key_absent(self):
        assert _is_user_specified({}, "trainer") is False


@pytest.mark.unit
class TestResolveEffectivePhaseDict:
    def test_initialization_merges_defaults_per_namespace(self):
        phase_dict = {
            "dimer_kwargs": {"num_dimers_per_combo": 3},
            "mp_kwargs": {"enabled": False},
        }
        effective = _resolve_effective_phase_dict("initialization", phase_dict)
        assert effective["dimer_kwargs"]["num_dimers_per_combo"] == 3
        assert effective["dimer_kwargs"]["enabled"] is True  # default, not overridden
        assert effective["mp_kwargs"]["enabled"] is False
        assert effective["mp_kwargs"]["max_atom_number"] == 20  # default
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
        assert effective["desired_number_of_structures"] == 50  # default

    def test_structure_generation_respects_existing_desired_number(self):
        phase_dict = {"generator": "md", "desired_number_of_structures": 10}
        effective = _resolve_effective_phase_dict("structure_generation", phase_dict)
        assert effective["desired_number_of_structures"] == 10

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
        assert system == {"input_dft": "pbesol"}  # shallow replace, matches runtime
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

    def test_general_merges_committee_uncertainty_kwargs_defaults(self):
        phase_dict = {
            "al_workflow": "committee_uncertainty",
            "elements": ["H"],
            "committee_uncertainty_kwargs": {"test_ratio": 0.2},
        }
        effective = _resolve_effective_phase_dict("general", phase_dict)
        cuk = effective["committee_uncertainty_kwargs"]
        assert cuk["test_ratio"] == 0.2
        assert cuk["number_models_in_committee"] == 3  # default
        assert cuk["valid_fraction"] == 0.05  # default
        assert effective["elements"] == ["H"]  # untouched sibling

    def test_general_defaults_al_workflow_to_committee_uncertainty(self):
        phase_dict = {"elements": ["H"]}
        effective = _resolve_effective_phase_dict("general", phase_dict)
        assert "committee_uncertainty_kwargs" in effective
        assert (
            effective["committee_uncertainty_kwargs"]["number_models_in_committee"] == 3
        )


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
        assert (
            "committee_uncertainty_kwargs.test_ratio: 0.1  [user-specified]" in summary
        )
        # number_models_in_committee was also set by the fixture.
        assert "committee_uncertainty_kwargs.number_models_in_committee:" in summary
        # grouped_splits wasn't set by the user -- shown, but unmarked.
        assert "committee_uncertainty_kwargs.grouped_splits: False\n" in summary + "\n"
        assert (
            "committee_uncertainty_kwargs.grouped_splits: False  [user-specified]"
            not in summary
        )
        assert "elements: ['H']  [user-specified]" in summary


@pytest.mark.unit
class TestFlattenSettingsDepth:
    def test_flattens_three_levels_deep(self):
        d = {"a": {"b": {"c": 1}}}
        assert _flatten_settings(d) == [("a.b.c", 1)]
