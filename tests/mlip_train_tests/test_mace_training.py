from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train


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


class TestGetMaceEvalInfo:
    """Tests for get_mace_eval_info reading MACE train.txt result files."""

    def _write_train_txt(self, results_dir: Path, mae_f: float, mae_e: float) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        line = str([("mae_f", str(mae_f)), ("mae_e", str(mae_e))])
        (results_dir / "results_train.txt").write_text(f"epoch step\n{line}\n")

    @pytest.mark.unit
    def test_returns_dataframe_with_mae_columns(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        self._write_train_txt(
            tmp_path / "results" / "al_loop_0" / "mlip_committee" / "fit_0" / "results",
            mae_f=0.05,
            mae_e=0.01,
        )
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert "mae_f" in df.columns
        assert "mae_e" in df.columns

    @pytest.mark.unit
    def test_averages_multiple_fits(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        for i in range(3):
            self._write_train_txt(
                tmp_path / "results" / "al_loop_0" / "mlip_committee" / f"fit_{i}" / "results",
                mae_f=0.1 * (i + 1),
                mae_e=0.01,
            )
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert df["mae_f"].iloc[0] == pytest.approx(np.mean([0.1, 0.2, 0.3]))

    @pytest.mark.unit
    def test_empty_dataframe_when_no_al_loop_dirs(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert len(df) == 0

    @pytest.mark.unit
    def test_one_row_per_al_loop(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        for loop in range(3):
            self._write_train_txt(
                tmp_path / "results" / f"al_loop_{loop}" / "mlip_committee" / "fit_0" / "results",
                mae_f=0.1 * (loop + 1),
                mae_e=0.01,
            )
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert len(df) == 3

    @pytest.mark.unit
    def test_loop_with_no_results_files_skipped(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        # Loop 0 has results; loop 1 directory exists but is empty
        self._write_train_txt(
            tmp_path / "results" / "al_loop_0" / "mlip_committee" / "fit_0" / "results",
            mae_f=0.05,
            mae_e=0.01,
        )
        (tmp_path / "results" / "al_loop_1" / "mlip_committee").mkdir(parents=True)
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert len(df) == 1
