import json
import logging
import typing
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alomancy.mlip.mace_wfl import _select_validation_split
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

    def _write_train_txt(
        self, results_dir: Path, mae_f: float, mae_e_per_atom: float = 0.01
    ) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        line = str([("mae_f", str(mae_f)), ("mae_e_per_atom", str(mae_e_per_atom))])
        (results_dir / "results_train.txt").write_text(f"epoch step\n{line}\n")

    @pytest.mark.unit
    def test_returns_dataframe_with_mae_columns(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        self._write_train_txt(
            tmp_path / "results" / "al_loop_0" / "mlip_committee" / "fit_0" / "results",
            mae_f=0.05,
            mae_e_per_atom=0.01,
        )
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert "mae_f" in df.columns
        assert "mae_e_per_atom" in df.columns

    @pytest.mark.unit
    def test_averages_multiple_fits(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import get_mace_eval_info

        monkeypatch.chdir(tmp_path)
        for i in range(3):
            self._write_train_txt(
                tmp_path
                / "results"
                / "al_loop_0"
                / "mlip_committee"
                / f"fit_{i}"
                / "results",
                mae_f=0.1 * (i + 1),
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
                tmp_path
                / "results"
                / f"al_loop_{loop}"
                / "mlip_committee"
                / "fit_0"
                / "results",
                mae_f=0.1 * (loop + 1),
                mae_e_per_atom=0.01,
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
            mae_e_per_atom=0.01,
        )
        (tmp_path / "results" / "al_loop_1" / "mlip_committee").mkdir(parents=True)
        df = get_mace_eval_info({"name": "mlip_committee"})
        assert len(df) == 1


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


class TestSelectBestCommitteeModel:
    """Tests for select_best_committee_model — picks the fit with lowest test mae_f."""

    JOB_DICT: typing.ClassVar[dict] = {"name": "mlip_committee", "size_of_committee": 3}

    def _write_test_txt_python_format(
        self, results_dir: Path, mae_f: float, mae_e: float = 0.01
    ) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        line = str([("mae_f", str(mae_f)), ("mae_e", str(mae_e))])
        (results_dir / "results_test.txt").write_text(f"header\n{line}\n")

    def _write_test_txt_json_format(
        self, results_dir: Path, mae_f: float, mae_e_per_atom: float = 0.01
    ) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        record = json.dumps(
            {
                "mode": "test",
                "epoch": 79,
                "mae_f": mae_f,
                "mae_e_per_atom": mae_e_per_atom,
            }
        )
        (results_dir / "results_test.txt").write_text(record + "\n")

    def _fit_dir(self, base: Path, fit_idx: int) -> Path:
        return (
            base
            / "results"
            / "al_loop_0"
            / "mlip_committee"
            / f"fit_{fit_idx}"
            / "results"
        )

    @pytest.mark.unit
    def test_selects_fit_with_lowest_mae_f(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import select_best_committee_model

        monkeypatch.chdir(tmp_path)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 0), mae_f=0.30)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 1), mae_f=0.10)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 2), mae_f=0.20)

        best_idx, _ = select_best_committee_model("al_loop_0", self.JOB_DICT, seed=803)
        assert best_idx == 1

    @pytest.mark.unit
    def test_returns_correct_model_path(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import select_best_committee_model

        monkeypatch.chdir(tmp_path)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 0), mae_f=0.30)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 1), mae_f=0.05)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 2), mae_f=0.20)

        _, model_path = select_best_committee_model(
            "al_loop_0", self.JOB_DICT, seed=803
        )
        assert "fit_1" in str(model_path)
        assert model_path.name == "mlip_committee_stagetwo.model"

    @pytest.mark.unit
    def test_falls_back_to_fit_0_when_no_test_files(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import select_best_committee_model

        monkeypatch.chdir(tmp_path)
        best_idx, model_path = select_best_committee_model(
            "al_loop_0", self.JOB_DICT, seed=803
        )
        assert best_idx == 0
        assert "fit_0" in str(model_path)

    @pytest.mark.unit
    def test_falls_back_to_fit_0_when_metric_missing(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import select_best_committee_model

        monkeypatch.chdir(tmp_path)
        for i in range(3):
            d = self._fit_dir(tmp_path, i)
            d.mkdir(parents=True, exist_ok=True)
            # Write a file with a different metric key — no 'mae_f'
            (d / "results_test.txt").write_text(
                json.dumps({"mae_e_per_atom": 0.01}) + "\n"
            )

        best_idx, _ = select_best_committee_model("al_loop_0", self.JOB_DICT, seed=803)
        assert best_idx == 0

    @pytest.mark.unit
    def test_handles_json_format(self, tmp_path, monkeypatch):
        from alomancy.mlip.get_mace_eval_info import select_best_committee_model

        monkeypatch.chdir(tmp_path)
        self._write_test_txt_json_format(self._fit_dir(tmp_path, 0), mae_f=0.25)
        self._write_test_txt_json_format(self._fit_dir(tmp_path, 1), mae_f=0.08)
        self._write_test_txt_json_format(self._fit_dir(tmp_path, 2), mae_f=0.15)

        best_idx, _ = select_best_committee_model("al_loop_0", self.JOB_DICT, seed=803)
        assert best_idx == 1

    @pytest.mark.unit
    def test_skips_fits_missing_test_file_picks_best_of_rest(
        self, tmp_path, monkeypatch
    ):
        from alomancy.mlip.get_mace_eval_info import select_best_committee_model

        monkeypatch.chdir(tmp_path)
        # fit_0 has no test file; fit_1 and fit_2 do
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 1), mae_f=0.12)
        self._write_test_txt_python_format(self._fit_dir(tmp_path, 2), mae_f=0.08)

        best_idx, _ = select_best_committee_model("al_loop_0", self.JOB_DICT, seed=803)
        assert best_idx == 2


class TestSaveMaceEvalPredictions:
    """_save_mace_eval_predictions runs on the remote GPU node right after
    training, evaluating the trained model on every train/test structure.
    Regression coverage for a bug where a near-total per-structure
    prediction failure (e.g. 1 succeeding out of 1405 structures, observed
    in production) was completely invisible: the per-structure exception
    was only logger.debug'd inside a freshly spawned remote process where
    setup_logging() is never called (so there's no handler for DEBUG-level
    records), and RemoteJobExecutor discarded a successful job's
    stdout/stderr entirely -- so nothing ever reached results/alomancy.log.
    That silently degraded parity plots to a single trivial (0, 0) point
    (whichever one structure's prediction happened to succeed) with no
    error anywhere to explain why."""

    @staticmethod
    def _collect_alomancy_logs():
        """setup_logging sets propagate=False on the root "alomancy" logger
        elsewhere in the process, so pytest's caplog can't reliably see
        these records -- attach a handler directly, matching the pattern in
        test_base_active_learning.py's test_seed_logs_message."""
        al_logger = logging.getLogger("alomancy")
        al_logger.setLevel(logging.DEBUG)
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collector()
        handler.setLevel(logging.DEBUG)
        al_logger.addHandler(handler)
        return al_logger, handler, records

    def _write_structures(self, path: Path, n: int) -> None:
        from ase.io import write

        structures = []
        for i in range(n):
            a = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
            a.info["config_type"] = f"s{i}"
            a.info["REF_energy"] = 1.0
            structures.append(a)
        write(str(path), structures, format="extxyz")

    @pytest.mark.unit
    def test_first_failure_gets_warning_with_traceback_rest_are_debug(
        self, tmp_path, monkeypatch
    ):
        from unittest.mock import MagicMock, patch

        from alomancy.mlip.mace_wfl import _save_mace_eval_predictions

        monkeypatch.chdir(tmp_path)
        (tmp_path / "test_name_stagetwo_compiled.model").touch()
        self._write_structures(tmp_path / "train.xyz", 3)

        call_count = {"n": 0}

        def fake_get_potential_energy(self):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError(f"boom {call_count['n']}")
            return 1.23

        monkeypatch.setattr(Atoms, "get_potential_energy", fake_get_potential_energy)
        monkeypatch.setattr(
            Atoms, "get_forces", lambda self: np.zeros((1, 3)), raising=False
        )

        al_logger, handler, records = self._collect_alomancy_logs()
        try:
            with patch("mace.calculators.MACECalculator") as mock_calc_cls:
                mock_calc_cls.return_value = MagicMock()
                _save_mace_eval_predictions("test_name", "train.xyz")
        finally:
            al_logger.removeHandler(handler)

        warning_failures = [
            r
            for r in records
            if r.levelno == logging.WARNING
            and "Prediction failed for structure" in r.getMessage()
        ]
        debug_failures = [
            r
            for r in records
            if r.levelno == logging.DEBUG
            and "Prediction failed for structure" in r.getMessage()
        ]
        summary = [
            r
            for r in records
            if "predictions:" in r.getMessage() and "succeeded" in r.getMessage()
        ]

        # Only the first failure gets a WARNING-level, full-traceback log;
        # subsequent identical failures drop to DEBUG so 1000+ structures
        # failing the same way doesn't flood the log.
        assert len(warning_failures) == 1
        assert warning_failures[0].exc_info is not None
        assert len(debug_failures) == 1

        assert len(summary) == 1
        assert "1 succeeded, 2 failed out of 3 structures" in summary[0].getMessage()

    @pytest.mark.unit
    def test_no_failure_logs_when_all_predictions_succeed(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        from alomancy.mlip.mace_wfl import _save_mace_eval_predictions

        monkeypatch.chdir(tmp_path)
        (tmp_path / "test_name_stagetwo_compiled.model").touch()
        self._write_structures(tmp_path / "train.xyz", 3)

        monkeypatch.setattr(Atoms, "get_potential_energy", lambda self: 1.23)
        monkeypatch.setattr(
            Atoms, "get_forces", lambda self: np.zeros((1, 3)), raising=False
        )

        al_logger, handler, records = self._collect_alomancy_logs()
        try:
            with patch("mace.calculators.MACECalculator") as mock_calc_cls:
                mock_calc_cls.return_value = MagicMock()
                _save_mace_eval_predictions("test_name", "train.xyz")
        finally:
            al_logger.removeHandler(handler)

        failure_records = [r for r in records if "Prediction failed" in r.getMessage()]
        assert failure_records == []
        assert (tmp_path / "train_pred.xyz").exists()
