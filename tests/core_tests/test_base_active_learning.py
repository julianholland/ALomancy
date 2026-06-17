"""
Tests for the base active learning workflow.

This module tests the BaseActiveLearningWorkflow abstract class and its core functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write

from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
from alomancy.database.global_database import GlobalDatabase


# =============================================================================
# ConcreteWorkflow: Stub implementation for testing
# =============================================================================

class ConcreteWorkflow(BaseActiveLearningWorkflow):
    """Concrete implementation of BaseActiveLearningWorkflow for testing."""

    def initialize_training_set(self, base_name, **kwargs):
        """Return empty lists for testing."""
        return [], []

    def train_mlip(self, base_name, mlip_committee_job_dict, **kwargs):
        """Return empty DataFrame for testing."""
        return pd.DataFrame()

    def generate_structures(self, base_name, job_dict, train_data, **kwargs):
        """Return empty list for testing."""
        return []

    def high_accuracy_evaluation(
        self, base_name, high_accuracy_eval_job_dict, structures, **kwargs
    ):
        """Return empty list for testing."""
        return []


# =============================================================================
# Test Classes
# =============================================================================

class TestConstructor:
    """Tests for BaseActiveLearningWorkflow constructor."""

    @pytest.mark.unit
    def test_default_params(self, tmp_path, minimal_jobs_dict):
        """Test that default parameters are set correctly."""
        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )
        assert wf.number_of_al_loops == 5
        assert wf.verbose == 0
        assert wf.start_loop == 0
        assert wf.seed == 803
        assert isinstance(wf.db, GlobalDatabase)

    @pytest.mark.unit
    def test_custom_params(self, tmp_path, minimal_jobs_dict):
        """Test that custom parameters override defaults."""
        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            number_of_al_loops=3,
            verbose=1,
            start_loop=1,
            plots=False,
            seed=42,
            db_path=str(tmp_path / "custom_db"),
        )
        assert wf.number_of_al_loops == 3
        assert wf.verbose == 1
        assert wf.start_loop == 1
        assert wf.plots is False
        assert wf.seed == 42

    @pytest.mark.unit
    def test_custom_db_path(self, tmp_path, minimal_jobs_dict):
        """Test that custom db_path is used."""
        custom_db = str(tmp_path / "custom_db")
        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=custom_db,
        )
        assert isinstance(wf.db, GlobalDatabase)

    @pytest.mark.unit
    def test_abstract_methods_required(self, tmp_path, minimal_jobs_dict):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseActiveLearningWorkflow(
                initial_train_file_path=str(tmp_path / "train.xyz"),
                initial_test_file_path=str(tmp_path / "test.xyz"),
                jobs_dict=minimal_jobs_dict,
            )

    @pytest.mark.unit
    def test_paths_stored_as_path_objects(self, tmp_path, minimal_jobs_dict):
        """Test that file paths are converted to Path objects."""
        train_path = str(tmp_path / "train.xyz")
        test_path = str(tmp_path / "test.xyz")
        wf = ConcreteWorkflow(
            initial_train_file_path=train_path,
            initial_test_file_path=test_path,
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )
        assert isinstance(wf.initial_train_file_path, Path)
        assert isinstance(wf.initial_test_file_path, Path)


class TestSeedDbFromExtraDataset:
    """Tests for _seed_db_from_extra_dataset method."""

    @pytest.mark.unit
    def test_seeds_structures_into_db(self, tmp_path, minimal_jobs_dict, h_atom, h2o_mol):
        """Test that extra dataset structures are added to the database."""
        xyz_path = tmp_path / "extra.xyz"
        write(str(xyz_path), [h_atom, h2o_mol], format="extxyz")

        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )
        wf._seed_db_from_extra_dataset(str(xyz_path))
        assert wf.db.size == 2

    @pytest.mark.unit
    def test_dedup_on_seed_isolated_atom(self, tmp_path, minimal_jobs_dict, h_atom):
        """Test that duplicate IsolatedAtoms are deduplicated on seed."""
        # Two H IsolatedAtoms in same file — only 1 should be added
        xyz_path = tmp_path / "extra.xyz"
        write(str(xyz_path), [h_atom, h_atom.copy()], format="extxyz")

        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )
        wf._seed_db_from_extra_dataset(str(xyz_path))
        assert wf.db.size == 1

    @pytest.mark.unit
    def test_seed_logs_message(self, tmp_path, minimal_jobs_dict, h_atom):
        """Test that seeding emits a log record at INFO level."""
        import logging

        xyz_path = tmp_path / "extra.xyz"
        write(str(xyz_path), [h_atom], format="extxyz")

        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
            log_file=None,
        )
        # setup_logging sets propagate=False on the "alomancy" logger, so we
        # capture records by attaching a handler directly to it for this test.
        al_logger = logging.getLogger("alomancy")
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collector()
        al_logger.addHandler(handler)
        try:
            wf._seed_db_from_extra_dataset(str(xyz_path))
        finally:
            al_logger.removeHandler(handler)

        messages = " ".join(r.getMessage() for r in records)
        assert "Seeded DB from" in messages
        assert str(xyz_path) in messages

    @pytest.mark.unit
    def test_seed_with_single_atom_file(self, tmp_path, minimal_jobs_dict, h_atom):
        """Test seeding with a single-atom file."""
        xyz_path = tmp_path / "single.xyz"
        write(str(xyz_path), h_atom, format="extxyz")

        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )
        wf._seed_db_from_extra_dataset(str(xyz_path))
        assert wf.db.size == 1


class TestRunWorkflowStructure:
    """Tests for run() method structure and execution flow."""

    def _make_workflow(self, tmp_path, minimal_jobs_dict, **kwargs):
        """Helper to create a ConcreteWorkflow with standard parameters."""
        return ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            number_of_al_loops=2,
            db_path=str(tmp_path / "db"),
            **kwargs,
        )

    @pytest.mark.unit
    def test_extra_datasets_seeded_before_initialize(
        self, tmp_path, minimal_jobs_dict, h_atom
    ):
        """Test that extra datasets are seeded before initialize_training_set."""
        # Setup: one extra_dataset file
        extra = tmp_path / "extra.xyz"
        write(str(extra), [h_atom], format="extxyz")
        minimal_jobs_dict["initialization"]["extra_datasets"] = [str(extra)]

        call_order = []
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting

        original_seed = wf._seed_db_from_extra_dataset
        def tracking_seed(path):
            call_order.append(("seed", path))
            return original_seed(path)

        original_init = wf.initialize_training_set
        def tracking_init(base_name, **kwargs):
            call_order.append(("init", base_name))
            return [], []

        wf._seed_db_from_extra_dataset = tracking_seed
        wf.initialize_training_set = tracking_init

        with patch("alomancy.core.base_active_learning.write"):
            with patch.object(wf, "train_mlip", return_value=pd.DataFrame()):
                with patch.object(wf, "generate_structures", return_value=[]):
                    with patch.object(wf, "high_accuracy_evaluation", return_value=[]):
                        wf.run()

        # Check that seed was called before init
        assert call_order[0][0] == "seed"
        assert call_order[1][0] == "init"

    @pytest.mark.unit
    def test_loop_count(self, tmp_path, minimal_jobs_dict):
        """Test that run() executes the correct number of AL loops."""
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting
        train_call_count = []

        with patch.object(wf, "initialize_training_set", return_value=([], [])):
            with patch.object(
                wf, "train_mlip",
                side_effect=lambda *a, **kw: train_call_count.append(1) or pd.DataFrame()
            ):
                with patch.object(wf, "generate_structures", return_value=[]):
                    with patch.object(wf, "high_accuracy_evaluation", return_value=[]):
                        with patch("alomancy.core.base_active_learning.write"):
                            wf.run()

        assert len(train_call_count) == 2  # number_of_al_loops=2

    @pytest.mark.unit
    def test_start_loop_respected(self, tmp_path, minimal_jobs_dict):
        """Test that start_loop parameter is respected."""
        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            number_of_al_loops=4,
            start_loop=2,
            plots=False,
            db_path=str(tmp_path / "db"),
        )
        train_calls = []

        with patch.object(wf, "initialize_training_set", return_value=([], [])):
            with patch.object(
                wf, "train_mlip",
                side_effect=lambda *a, **kw: train_calls.append(1) or pd.DataFrame()
            ):
                with patch.object(wf, "generate_structures", return_value=[]):
                    with patch.object(wf, "high_accuracy_evaluation", return_value=[]):
                        with patch("alomancy.core.base_active_learning.write"):
                            wf.run()

        assert len(train_calls) == 2  # loops 2 and 3 only

    @pytest.mark.unit
    def test_base_names_correct_for_loops(self, tmp_path, minimal_jobs_dict):
        """Test that base_name is correct for each loop."""
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting
        train_calls = []

        def track_train(base_name, *args, **kwargs):
            train_calls.append(base_name)
            return pd.DataFrame()

        with patch.object(wf, "initialize_training_set", return_value=([], [])):
            with patch.object(wf, "train_mlip", side_effect=track_train):
                with patch.object(wf, "generate_structures", return_value=[]):
                    with patch.object(wf, "high_accuracy_evaluation", return_value=[]):
                        with patch("alomancy.core.base_active_learning.write"):
                            wf.run()

        assert train_calls[0] == "al_loop_0"
        assert train_calls[1] == "al_loop_1"

    @pytest.mark.unit
    def test_abstract_methods_called_in_sequence(self, tmp_path, minimal_jobs_dict):
        """Test that abstract methods are called in the correct sequence."""
        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            number_of_al_loops=1,
            plots=False,
            db_path=str(tmp_path / "db"),
        )
        call_sequence = []

        def track_init(base_name, **kwargs):
            call_sequence.append("init")
            return [], []

        def track_train(base_name, *args, **kwargs):
            call_sequence.append("train")
            return pd.DataFrame()

        def track_gen(base_name, *args, **kwargs):
            call_sequence.append("gen")
            return []

        def track_eval(base_name, *args, **kwargs):
            call_sequence.append("eval")
            return []

        with patch.object(wf, "initialize_training_set", side_effect=track_init):
            with patch.object(wf, "train_mlip", side_effect=track_train):
                with patch.object(wf, "generate_structures", side_effect=track_gen):
                    with patch.object(wf, "high_accuracy_evaluation", side_effect=track_eval):
                        with patch("alomancy.core.base_active_learning.write"):
                            wf.run()

        # Check sequence: init first, then train, gen, eval for each loop
        assert call_sequence[0] == "init"
        assert call_sequence[1] == "train"
        assert call_sequence[2] == "gen"
        assert call_sequence[3] == "eval"

    @pytest.mark.unit
    def test_workdir_created_for_each_loop(self, tmp_path, minimal_jobs_dict):
        """Test that work directories are created for each loop."""
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting

        with patch.object(wf, "initialize_training_set", return_value=([], [])):
            with patch.object(wf, "train_mlip", return_value=pd.DataFrame()):
                with patch.object(wf, "generate_structures", return_value=[]):
                    with patch.object(wf, "high_accuracy_evaluation", return_value=[]):
                        with patch("alomancy.core.base_active_learning.write"):
                            wf.run()

        # Verify that the expected directories exist
        assert Path("results/al_loop_0").exists()
        assert Path("results/al_loop_1").exists()

    @pytest.mark.unit
    def test_train_test_files_written(self, tmp_path, minimal_jobs_dict):
        """Test that train and test set files are written for each loop."""
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting

        with patch.object(wf, "initialize_training_set", return_value=([], [])):
            with patch.object(wf, "train_mlip", return_value=pd.DataFrame()):
                with patch.object(wf, "generate_structures", return_value=[]):
                    with patch.object(wf, "high_accuracy_evaluation", return_value=[]):
                        with patch("alomancy.core.base_active_learning.write") as mock_write:
                            wf.run()

        # Check that write was called for train and test files
        # At minimum: 2 loops * 2 files (train + test) = 4 calls
        assert mock_write.call_count >= 4
