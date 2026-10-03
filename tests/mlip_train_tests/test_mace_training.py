import typing
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alomancy.core.active_learning_workflow import _select_validation_split
from alomancy.mlip.evaluation import (
    metrics_by_loop,
    prediction_metrics,
    rank_committee,
    save_evaluation,
)
from alomancy.remote_submission import submitters
from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train


@pytest.mark.unit
def test_committee_uses_common_split_seed_and_distinct_fit_indices(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeExecutor:
        def __init__(self, _remote_info):
            pass

        def run_and_wait(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submitters, "RemoteJobExecutor", FakeExecutor)

    submitters.committee_remote_submitter(
        remote_info={},
        base_name="al_loop_0",
        function=lambda: None,
        seed=803,
        num_of_models_in_committee=3,
    )

    configs = captured["job_configs"]
    assert [c["function_kwargs"]["seed"] for c in configs] == [803, 803, 803]
    assert [c["function_kwargs"]["fit_idx"] for c in configs] == [0, 1, 2]


class TestEvaluationMetrics:
    """Test the mathematical relationships between evaluation metrics using numpy directly."""

    @pytest.mark.unit
    def test_mae_less_than_or_equal_rmse(self):
        # By Cauchy-Schwarz inequality, MAE <= RMSE always holds
        predictions = np.array([1.1, 2.3, 3.0, 4.5])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        errors = predictions - targets
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        assert mae <= rmse + 1e-10

    @pytest.mark.unit
    def test_zero_error_zero_metrics(self):
        predictions = np.array([1.0, 2.0, 3.0])
        errors = predictions - predictions
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        assert mae == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)

    @pytest.mark.unit
    def test_mae_calculation(self):
        predictions = np.array([1.5, 2.5])
        targets = np.array([1.0, 2.0])
        errors = predictions - targets
        mae = np.mean(np.abs(errors))
        assert mae == pytest.approx(0.5)

    @pytest.mark.unit
    def test_rmse_calculation(self):
        predictions = np.array([1.5, 2.5])
        targets = np.array([1.0, 2.0])
        errors = predictions - targets
        rmse = np.sqrt(np.mean(errors**2))
        assert rmse == pytest.approx(0.5)


class TestCommitteePredictionVariance:
    """Test standard deviation calculation across committee members — pure numpy."""

    @pytest.mark.unit
    def test_identical_predictions_zero_variance(self):
        forces = np.array([[1.0, 0.0, 0.0]])  # shape (1, 3)
        # All 3 committee members return same forces
        all_forces = np.concatenate([forces, forces, forces], axis=0)  # (3, 3)
        std_dev = np.std(all_forces, axis=0)
        assert np.max(std_dev) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_different_predictions_nonzero_variance(self):
        forces_a = np.array([[1.0, 0.0, 0.0]])
        forces_b = np.array([[2.0, 0.0, 0.0]])
        all_forces = np.concatenate([forces_a, forces_b], axis=0)  # (2, 3)
        std_dev = np.std(all_forces, axis=0)
        assert std_dev[0] > 0.0
        assert std_dev[1] == pytest.approx(0.0)
        assert std_dev[2] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_max_std_exceeds_mean_std_with_outlier(self):
        # One force component has high variance, others near-zero
        forces = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ]
        )
        std_dev = np.std(forces, axis=0)
        assert np.max(std_dev) > np.mean(std_dev)


class TestTrainTestSplit:
    """Test split_atoms_list_into_test_and_train with real data."""

    def _atoms_list(self, n):
        return [
            Atoms(["H"], positions=[[i, 0, 0]], cell=[5, 5, 5], pbc=True)
            for i in range(n)
        ]

    @pytest.mark.unit
    def test_no_overlap_between_train_and_test(self):
        atoms = self._atoms_list(20)
        train, test = split_atoms_list_into_test_and_train(atoms, 0.2, seed=42)
        train_ids = {id(a) for a in train}
        test_ids = {id(a) for a in test}
        assert train_ids.isdisjoint(test_ids)

    @pytest.mark.unit
    def test_all_atoms_accounted_for(self):
        atoms = self._atoms_list(20)
        train, test = split_atoms_list_into_test_and_train(atoms, 0.2, seed=42)
        assert len(train) + len(test) == 20

    @pytest.mark.unit
    def test_fraction_boundary(self):
        # test_fraction=0.3, 10 atoms -> 3 test, 7 train
        atoms = self._atoms_list(10)
        train, test = split_atoms_list_into_test_and_train(atoms, 0.3, seed=42)
        assert len(test) == 3
        assert len(train) == 7


class TestSelectValidationSplit:
    """Tests for _select_validation_split — the per-fit validation set carver."""

    def _atoms(self, n: int, config_type: str) -> list[Atoms]:
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

    @pytest.mark.unit
    def test_carves_correct_fraction(self):
        eligible = self._atoms(100, "dimer")
        ineligible = self._atoms(10, "IsolatedAtom")
        all_training = eligible + ineligible
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.05, rng=np.random.default_rng(42)
        )
        # 5% of 100 eligible = 5 go to valid
        assert len(valid) == 5
        assert len(new_train) == len(all_training) - 5

    @pytest.mark.unit
    def test_all_accounted_for(self):
        all_training = self._atoms(40, "dimer") + self._atoms(10, "IsolatedAtom")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.1, rng=np.random.default_rng(42)
        )
        assert len(new_train) + len(valid) == len(all_training)

    @pytest.mark.unit
    def test_no_overlap(self):
        all_training = self._atoms(50, "dimer")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.1, rng=np.random.default_rng(42)
        )
        train_ids = {id(a) for a in new_train}
        valid_ids = {id(a) for a in valid}
        assert train_ids.isdisjoint(valid_ids)

    @pytest.mark.unit
    def test_ineligible_always_in_train(self):
        eligible = self._atoms(20, "dimer")
        ineligible = self._atoms(5, "IsolatedAtom")
        all_training = eligible + ineligible
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.2, rng=np.random.default_rng(42)
        )
        ineligible_ids = {id(a) for a in ineligible}
        assert ineligible_ids.issubset({id(a) for a in new_train})
        assert not any(id(a) in ineligible_ids for a in valid)

    @pytest.mark.unit
    def test_empty_eligible_returns_all_training(self):
        # No structures with matching config_type
        all_training = self._atoms(10, "IsolatedAtom")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.1, rng=np.random.default_rng(42)
        )
        assert valid == []
        assert len(new_train) == len(all_training)

    @pytest.mark.unit
    def test_rounds_to_zero_returns_all_training(self):
        # 5% of 1 structure floors to 0
        all_training = self._atoms(1, "dimer")
        new_train, valid = _select_validation_split(
            all_training, ["dimer"], valid_fraction=0.05, rng=np.random.default_rng(42)
        )
        assert valid == []
        assert len(new_train) == 1

    @pytest.mark.unit
    def test_reproducible_with_same_seed(self):
        all_training = self._atoms(50, "dimer")
        _, valid_a = _select_validation_split(
            all_training, ["dimer"], 0.1, np.random.default_rng(7)
        )
        _, valid_b = _select_validation_split(
            all_training, ["dimer"], 0.1, np.random.default_rng(7)
        )
        assert [id(a) for a in valid_a] == [id(b) for b in valid_b]

    @pytest.mark.unit
    def test_different_seeds_different_splits(self):
        all_training = self._atoms(100, "dimer")
        _, valid_a = _select_validation_split(
            all_training, ["dimer"], 0.1, np.random.default_rng(1)
        )
        _, valid_b = _select_validation_split(
            all_training, ["dimer"], 0.1, np.random.default_rng(2)
        )
        # With 10 out of 100, it would be astronomically unlikely to get same selection
        assert {id(a) for a in valid_a} != {id(b) for b in valid_b}

    @pytest.mark.unit
    def test_multiple_eligible_config_types(self):
        dimers = self._atoms(20, "dimer")
        high_sd = self._atoms(20, "high_sd")
        isolated = self._atoms(5, "IsolatedAtom")
        all_training = dimers + high_sd + isolated
        new_train, valid = _select_validation_split(
            all_training,
            ["dimer", "high_sd"],
            valid_fraction=0.1,
            rng=np.random.default_rng(42),
        )
        # 10% of 40 eligible = 4 in valid
        assert len(valid) == 4
        assert len(new_train) + len(valid) == len(all_training)
        # IsolatedAtom always in train
        isolated_ids = {id(a) for a in isolated}
        assert isolated_ids.issubset({id(a) for a in new_train})
        assert not any(id(a) in isolated_ids for a in valid)


class TestRankCommittee:
    """rank_committee: the single committee-ranking rule -- lowest error on
    the common checkpoint-evaluation split, read from each fit's
    evaluation_metrics.json."""

    N_FITS: typing.ClassVar[int] = 3

    def _rank(self, base: Path, metric: str = "mae_f") -> tuple[int, Path, str]:
        fit_dirs = {i: self._fit_dir(base, i) for i in range(self.N_FITS)}
        return rank_committee(fit_dirs, metric=metric)

    @staticmethod
    def _predicted(e_error: float, f_error: float | None = None) -> Atoms:
        """A dimer whose per-atom energy MAE is e_error and force MAE is f_error."""
        f_error = e_error if f_error is None else f_error
        a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
        a.info.update(
            REF_energy=-8.0, model_energy=-8.0 + 2 * e_error, config_type="init_dimer"
        )
        a.set_array("REF_forces", np.zeros((2, 3)))
        a.set_array("model_forces", np.ones((2, 3)) * f_error)
        return a

    def _fit_dir(self, base: Path, fit_idx: int) -> Path:
        return base / "results" / "al_loop_0" / "mlip_committee" / f"fit_{fit_idx}"

    def _write_evaluation(
        self,
        base: Path,
        fit_idx: int,
        splits: dict[str, Atoms | list[Atoms]],
    ) -> Path:
        """Write a fit's stagetwo model plus the evaluation_metrics.json
        that ALomancyTrainer.evaluate produces on the remote node.

        rank_committee reads this file (via read_evaluation,
        which also checks the model exists and matches its recorded
        checksum) rather than any *_test.txt training log.
        """
        fit_dir = self._fit_dir(base, fit_idx)
        fit_dir.mkdir(parents=True, exist_ok=True)
        model = fit_dir / "mlip_committee_stagetwo.model"
        model.write_bytes(f"checkpoint {fit_idx}".encode())
        save_evaluation(
            fit_dir,
            model,
            {
                split: prediction_metrics(atoms if isinstance(atoms, list) else [atoms])
                for split, atoms in splits.items()
            },
        )
        return model

    @pytest.mark.unit
    def test_selects_fit_with_lowest_mae_f(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for i, error in enumerate([0.30, 0.10, 0.20]):
            self._write_evaluation(tmp_path, i, {"test": self._predicted(error)})

        best_idx, _, _ = self._rank(tmp_path)
        assert best_idx == 1

    @pytest.mark.unit
    def test_returns_correct_model_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for i, error in enumerate([0.30, 0.05, 0.20]):
            self._write_evaluation(tmp_path, i, {"test": self._predicted(error)})

        _, model_path, _ = self._rank(tmp_path)
        assert "fit_1" in str(model_path)
        assert model_path.name == "mlip_committee_stagetwo.model"

    @pytest.mark.unit
    def test_selects_by_requested_metric(self, tmp_path, monkeypatch):
        """fit_0 has the lowest force error, fit_2 the lowest energy error —
        which one wins must follow the `metric` argument."""
        monkeypatch.chdir(tmp_path)
        errors = [(0.30, 0.05), (0.20, 0.20), (0.05, 0.30)]
        for i, (e_err, f_err) in enumerate(errors):
            self._write_evaluation(tmp_path, i, {"test": self._predicted(e_err, f_err)})

        by_force, _, _ = self._rank(tmp_path)
        by_energy, _, _ = self._rank(tmp_path, metric="mae_e_per_atom")
        assert by_force == 0
        assert by_energy == 2

    @pytest.mark.unit
    def test_prefers_valid_split_over_test_split(self, tmp_path, monkeypatch):
        """When every fit has a 'valid' split it decides the ranking, even if
        'test' would rank the fits differently."""
        monkeypatch.chdir(tmp_path)
        for i, (valid_err, test_err) in enumerate([(0.30, 0.01), (0.10, 0.50)]):
            self._write_evaluation(
                tmp_path,
                i,
                {
                    "valid": self._predicted(valid_err),
                    "test": self._predicted(test_err),
                },
            )
        self._write_evaluation(
            tmp_path,
            2,
            {"valid": self._predicted(0.20), "test": self._predicted(0.30)},
        )

        best_idx, _, _ = self._rank(tmp_path)
        assert best_idx == 1

    @pytest.mark.unit
    def test_raises_when_no_fit_has_an_evaluation(self, tmp_path, monkeypatch):
        """Models on disk without evaluation_metrics.json are not enough — the
        old behaviour of quietly defaulting to fit_0 is gone."""
        monkeypatch.chdir(tmp_path)
        for i in range(3):
            fit_dir = self._fit_dir(tmp_path, i)
            fit_dir.mkdir(parents=True)
            (fit_dir / "mlip_committee_stagetwo.model").touch()

        with pytest.raises(RuntimeError, match="complete checkpoint validation"):
            self._rank(tmp_path)

    @pytest.mark.unit
    def test_raises_when_no_fit_directories_exist(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError, match="3 of 3 committee fit"):
            self._rank(tmp_path)

    @pytest.mark.unit
    def test_raises_rather_than_returning_a_failed_fit_0(self, tmp_path, monkeypatch):
        """Regression test for the production crash this guards against: fit_0
        failed (no model, no evaluation) while fit_1/fit_2 trained fine. The
        old unconditional 'default to fit_0' fallback returned fit_0's
        nonexistent model path; a committee with a missing member must now
        raise instead of silently picking from whoever remains."""
        monkeypatch.chdir(tmp_path)
        for i in (1, 2):
            self._write_evaluation(tmp_path, i, {"test": self._predicted(0.1 * i)})

        with pytest.raises(RuntimeError, match="1 of 3 committee fit"):
            self._rank(tmp_path)

    @pytest.mark.unit
    def test_raises_when_checkpoint_changed_after_evaluation(
        self, tmp_path, monkeypatch
    ):
        """A model whose bytes no longer match the checksum recorded with its
        metrics must not be trusted — the fit counts as missing."""
        monkeypatch.chdir(tmp_path)
        models = [
            self._write_evaluation(
                tmp_path, i, {"test": self._predicted(0.1 * (i + 1))}
            )
            for i in range(3)
        ]
        models[0].write_bytes(b"retrained after evaluation")

        with pytest.raises(RuntimeError, match="1 of 3 committee fit"):
            self._rank(tmp_path)


class TestMetricsByLoop:
    """metrics_by_loop: one row per AL loop with its best model's test-split
    metrics, from evaluation_metrics.json only (the old *_train.txt
    fallback is gone)."""

    @staticmethod
    def _write_fit(
        loop: int, fit_idx: int, test_error: float, valid_error=None
    ) -> None:
        fit_dir = Path(f"results/al_loop_{loop}/training/fit_{fit_idx}")
        fit_dir.mkdir(parents=True)
        model = fit_dir / "training_stagetwo.model"
        model.write_bytes(f"model {loop}-{fit_idx}".encode())
        splits = {
            "test": prediction_metrics([TestRankCommittee._predicted(test_error)])
        }
        if valid_error is not None:
            splits["valid"] = prediction_metrics(
                [TestRankCommittee._predicted(valid_error)]
            )
        save_evaluation(fit_dir, model, splits)

    @pytest.mark.unit
    def test_one_row_per_loop_with_best_fit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for loop in range(3):
            for fit_idx, error in enumerate([0.3, 0.1, 0.2]):
                self._write_fit(loop, fit_idx, error)

        df = metrics_by_loop("training", strict=True, expected_fits=3)

        assert df["al_loop"].to_list() == [0, 1, 2]
        assert df["best_fit_idx"].to_list() == [1, 1, 1]
        assert df["mae_f"].to_list() == pytest.approx([0.1, 0.1, 0.1])
        assert "mae_e_per_atom" in df.columns

    @pytest.mark.unit
    def test_empty_when_no_loops(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert len(metrics_by_loop("training", strict=True)) == 0

    @pytest.mark.unit
    def test_loop_without_evaluations_is_left_out(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fit(0, 0, 0.1)
        (tmp_path / "results/al_loop_1/training/fit_0/results").mkdir(parents=True)
        (tmp_path / "results/al_loop_1/training/fit_0/results/x_train.txt").write_text(
            "[('mae_f', '0.05')]\n"
        )

        df = metrics_by_loop("training", strict=True)

        assert df["al_loop"].to_list() == [0]

    @pytest.mark.unit
    def test_strict_raises_on_a_missing_fit_lenient_skips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fit(0, 0, 0.1)
        self._write_fit(0, 2, 0.2)
        self._write_fit(1, 0, 0.1)
        self._write_fit(1, 1, 0.1)
        self._write_fit(1, 2, 0.1)

        with pytest.raises(RuntimeError, match=r"expected fit_0\.\.fit_2"):
            metrics_by_loop("training", strict=True, expected_fits=3)

    @pytest.mark.unit
    def test_lenient_skips_inconsistent_loop(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fit(0, 0, 0.1, valid_error=0.1)
        self._write_fit(0, 1, 0.1)  # no valid split: inconsistent committee
        self._write_fit(1, 0, 0.2)

        df = metrics_by_loop("training", strict=False)
        assert df["al_loop"].to_list() == [1]
        with pytest.raises(RuntimeError):
            metrics_by_loop("training", strict=True)
