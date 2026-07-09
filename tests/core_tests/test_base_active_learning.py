"""
Tests for the base active learning workflow.

This module tests the BaseActiveLearningWorkflow abstract class and its core functionality.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
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
    def test_seeds_structures_into_db(
        self, tmp_path, minimal_jobs_dict, h_atom, h2o_mol
    ):
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
    def test_run_delegates_extra_dataset_seeding_to_initialize(
        self, tmp_path, minimal_jobs_dict, h_atom
    ):
        """Test that run() does not pre-seed extra_datasets; seeding is initialize_training_set's responsibility."""
        extra = tmp_path / "extra.xyz"
        write(str(extra), [h_atom], format="extxyz")
        minimal_jobs_dict["initialization"]["extra_datasets"] = [str(extra)]

        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False

        seed_calls_from_run = []
        original_seed = wf._seed_db_from_extra_dataset

        def tracking_seed(path):
            seed_calls_from_run.append(path)
            return original_seed(path)

        wf._seed_db_from_extra_dataset = tracking_seed

        with (
            patch("alomancy.core.base_active_learning.write"),
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_mlip", return_value=pd.DataFrame()),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=[]),
        ):
            wf.run()

        # run() must not call _seed_db_from_extra_dataset directly
        assert len(seed_calls_from_run) == 0

    @pytest.mark.unit
    def test_loop_count(self, tmp_path, minimal_jobs_dict):
        """Test that run() executes the correct number of AL loops."""
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting
        train_call_count = []

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(
                wf,
                "train_mlip",
                side_effect=lambda *a, **kw: (
                    train_call_count.append(1) or pd.DataFrame()
                ),
            ),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=[]),
            patch("alomancy.core.base_active_learning.write"),
        ):
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

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(
                wf,
                "train_mlip",
                side_effect=lambda *a, **kw: train_calls.append(1) or pd.DataFrame(),
            ),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=[]),
            patch("alomancy.core.base_active_learning.write"),
        ):
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

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_mlip", side_effect=track_train),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=[]),
            patch("alomancy.core.base_active_learning.write"),
        ):
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

        with (
            patch.object(wf, "initialize_training_set", side_effect=track_init),
            patch.object(wf, "train_mlip", side_effect=track_train),
            patch.object(wf, "generate_structures", side_effect=track_gen),
            patch.object(wf, "high_accuracy_evaluation", side_effect=track_eval),
            patch("alomancy.core.base_active_learning.write"),
        ):
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

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_mlip", return_value=pd.DataFrame()),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=[]),
            patch("alomancy.core.base_active_learning.write"),
        ):
            wf.run()

        # Verify that the expected directories exist
        assert Path("results/al_loop_0").exists()
        assert Path("results/al_loop_1").exists()

    @pytest.mark.unit
    def test_train_test_files_written(self, tmp_path, minimal_jobs_dict):
        """Test that train and test set files are written for each loop."""
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        wf.plots = False  # Disable plotting

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_mlip", return_value=pd.DataFrame()),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=[]),
            patch("alomancy.core.base_active_learning.write") as mock_write,
        ):
            wf.run()

        # Check that write was called for train and test files
        # At minimum: 2 loops * 2 files (train + test) = 4 calls
        assert mock_write.call_count >= 4

    @pytest.mark.unit
    def test_al_loop_structures_tagged_as_high_sd(self, tmp_path, minimal_jobs_dict):
        """New structures from high_accuracy_evaluation get config_type='high_sd'."""
        from ase import Atoms

        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            number_of_al_loops=1,
            plots=False,
            db_path=str(tmp_path / "db"),
        )

        def make_evaluated_structure():
            a = Atoms(
                "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5], pbc=True
            )
            a.info["REF_energy"] = -1.0
            a.arrays["REF_forces"] = np.zeros((2, 3))
            return a

        evaluated = [make_evaluated_structure() for _ in range(3)]
        captured_train: list = []

        def track_train(base_name, *a, **kw):
            captured_train.extend(a[0] if a else kw.get("train_data", []))
            return pd.DataFrame()

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_mlip", side_effect=track_train),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(wf, "high_accuracy_evaluation", return_value=evaluated),
            patch("alomancy.core.base_active_learning.write"),
        ):
            wf.run()

        # After the loop, structures added to train_xyzs must carry config_type="high_sd"
        # (verified by re-running; the next train call would see them — but here we check
        # via the DB which clean_structures feeds into)
        all_db = wf.db.get_all_as_atoms()
        assert all(a.info.get("config_type") == "high_sd" for a in all_db)

    @pytest.mark.unit
    def test_al_loop_metadata_stored_in_structures(self, tmp_path, minimal_jobs_dict):
        """Structures from loop N carry al_loop=N in their info dict."""
        from ase import Atoms

        wf = ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            number_of_al_loops=2,
            plots=False,
            db_path=str(tmp_path / "db"),
        )

        def make_structure():
            a = Atoms(
                "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5], pbc=True
            )
            a.info["REF_energy"] = -1.0
            a.arrays["REF_forces"] = np.zeros((2, 3))
            return a

        call_count = [0]

        def evaluated_for_loop(*a, **kw):
            loop_idx = call_count[0]
            call_count[0] += 1
            structs = [make_structure() for _ in range(2)]
            for s in structs:
                s.info["_loop"] = loop_idx
            return structs

        with (
            patch.object(wf, "initialize_training_set", return_value=([], [])),
            patch.object(wf, "train_mlip", return_value=pd.DataFrame()),
            patch.object(wf, "generate_structures", return_value=[]),
            patch.object(
                wf, "high_accuracy_evaluation", side_effect=evaluated_for_loop
            ),
            patch("alomancy.core.base_active_learning.write"),
        ):
            wf.run()

        all_db = wf.db.get_all_as_atoms()
        loop_values = {a.info["al_loop"] for a in all_db}
        assert loop_values == {0, 1}


class TestLoadInitialTrainTestSets:
    """Tests for load_initial_train_test_sets."""

    def _make_workflow(self, tmp_path, minimal_jobs_dict):
        return ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

    @pytest.mark.unit
    def test_raises_file_not_found_when_missing(self, tmp_path, minimal_jobs_dict):
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        with pytest.raises(FileNotFoundError):
            wf.load_initial_train_test_sets()

    @pytest.mark.unit
    def test_loads_structures_from_existing_files(
        self, tmp_path, minimal_jobs_dict, h_atom
    ):
        write(str(tmp_path / "train.xyz"), [h_atom, h_atom], format="extxyz")
        write(str(tmp_path / "test.xyz"), [h_atom], format="extxyz")
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        train, test = wf.load_initial_train_test_sets()
        assert len(train) == 2
        assert len(test) == 1

    @pytest.mark.unit
    def test_dummy_run_caps_train_at_500(self, tmp_path, minimal_jobs_dict, h_atom):
        write(str(tmp_path / "train.xyz"), [h_atom] * 600, format="extxyz")
        write(str(tmp_path / "test.xyz"), [h_atom] * 300, format="extxyz")
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        train, test = wf.load_initial_train_test_sets(dummy_run=True)
        assert len(train) == 500
        assert len(test) == 200

    @pytest.mark.unit
    def test_raises_when_only_train_missing(self, tmp_path, minimal_jobs_dict, h_atom):
        write(str(tmp_path / "test.xyz"), [h_atom], format="extxyz")
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        with pytest.raises(FileNotFoundError):
            wf.load_initial_train_test_sets()

    @pytest.mark.unit
    def test_raises_when_only_test_missing(self, tmp_path, minimal_jobs_dict, h_atom):
        write(str(tmp_path / "train.xyz"), [h_atom], format="extxyz")
        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        with pytest.raises(FileNotFoundError):
            wf.load_initial_train_test_sets()


class TestProcessStructure:
    """Tests for process_structure."""

    def _make_workflow(self, tmp_path, minimal_jobs_dict):
        return ConcreteWorkflow(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

    @pytest.mark.unit
    def test_extracts_ref_energy(self, tmp_path, minimal_jobs_dict):
        from ase.calculators.emt import EMT

        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        atoms = Atoms(
            "Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True
        )
        atoms.calc = EMT()
        result = wf.process_structure(atoms)
        assert "REF_energy" in result.info
        assert isinstance(result.info["REF_energy"], float)

    @pytest.mark.unit
    def test_extracts_ref_forces(self, tmp_path, minimal_jobs_dict):
        from ase.calculators.emt import EMT

        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        atoms = Atoms(
            "Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True
        )
        atoms.calc = EMT()
        result = wf.process_structure(atoms)
        assert "REF_forces" in result.arrays
        assert result.arrays["REF_forces"].shape == (2, 3)

    @pytest.mark.unit
    def test_returns_copy_not_same_object(self, tmp_path, minimal_jobs_dict):
        from ase.calculators.emt import EMT

        wf = self._make_workflow(tmp_path, minimal_jobs_dict)
        atoms = Atoms(
            "Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True
        )
        atoms.calc = EMT()
        result = wf.process_structure(atoms)
        assert result is not atoms
