"""Tests for core/committee_uncertainty_workflow.py -- the concrete AL
skeleton for the modular architecture.

The loop-control logic in run() (restart, plotting call sites, train_only/
fixed_test branching, DB writes) is copied largely as-is from
BaseActiveLearningWorkflow.run() (still covered by test_base_active_
learning.py) -- these tests focus on what's new: registry-resolved module
dispatch, the shared validation split, the partial-aware per-fit restart
mechanism, generalized committee scoring, and cross-loop metrics
aggregation.
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
    """minimal_jobs_dict plus the new `workflow` section (decision 14: split-
    building parameters and size_of_committee move here from
    initialization/mlip_committee; mlip_committee is renamed training)."""
    committee_size = minimal_jobs_dict["mlip_committee"]["size_of_committee"]
    minimal_jobs_dict["workflow"] = {
        "al_workflow": "committee_uncertainty",
        "target_config_types": ["IsolatedAtom"],
        "test_ratio": 0.1,
        "valid_fraction": 0.05,
        "size_of_committee": committee_size,
        "elements": ["H"],
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["size_of_committee"]
    minimal_jobs_dict["training"]["trainer"] = "mace"
    return minimal_jobs_dict


def _make_workflow(tmp_path, jobs_dict, shared_db, **overrides):
    kwargs = {
        "initial_train_file_path": str(tmp_path / "train.xyz"),
        "initial_test_file_path": str(tmp_path / "test.xyz"),
        "jobs_dict": jobs_dict,
        "number_of_al_loops": 2,
        "db": shared_db,
        "plots": False,
    }
    kwargs.update(overrides)
    return CommitteeUncertaintyWorkflow(**kwargs)


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
        wf = build_workflow(
            jobs_dict=workflow_jobs_dict,
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            db=shared_db,
        )
        assert isinstance(wf, CommitteeUncertaintyWorkflow)

    def test_defaults_to_committee_uncertainty_when_absent(
        self, tmp_path, minimal_jobs_dict, shared_db
    ):
        wf = build_workflow(
            jobs_dict=minimal_jobs_dict,
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            db=shared_db,
        )
        assert isinstance(wf, CommitteeUncertaintyWorkflow)

    def test_raises_on_unknown_al_workflow(
        self, tmp_path, minimal_jobs_dict, shared_db
    ):
        minimal_jobs_dict["workflow"] = {"al_workflow": "furthest_point_sampling"}
        with pytest.raises(ValueError, match=r"Unknown workflow\.al_workflow"):
            build_workflow(
                jobs_dict=minimal_jobs_dict,
                initial_train_file_path=str(tmp_path / "train.xyz"),
                initial_test_file_path=str(tmp_path / "test.xyz"),
                db=shared_db,
            )


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
                f"{_MODULE}._read_mace_eval_predictions",
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

        with patch(f"{_MODULE}._read_mace_eval_predictions", return_value={}):
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
        assert len(job_configs) == workflow_jobs_dict["workflow"]["size_of_committee"]

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
# run() -- loop control flow, largely copied from BaseActiveLearningWorkflow
# (test_base_active_learning.py covers that version); these confirm the copy
# still wires correctly to this skeleton's own internal method names.
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
        workflow_jobs_dict["workflow"]["train_only"] = True
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
