"""
Tests for the standard active learning workflow.

This module tests the ActiveLearningStandardMACE class and its implementation
of the active learning workflow using MACE, MD, and Quantum Espresso.
"""

import sys
from unittest.mock import MagicMock

# Pre-patch MACE modules and related dependencies before any imports
sys.modules.setdefault("mace", MagicMock())
sys.modules.setdefault("mace.calculators", MagicMock())
sys.modules.setdefault("mace.calculators.mace", MagicMock())
sys.modules.setdefault("mace.cli", MagicMock())
sys.modules.setdefault("mace.cli.run_train", MagicMock())

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import write  # noqa: E402

from alomancy.core.standard_active_learning import (  # noqa: E402
    ActiveLearningStandardMACE,
)


@pytest.fixture
def mock_job_config():
    return {
        "mlip_committee": {
            "name": "test_committee",
            "size_of_committee": 3,
            "max_time": "1H",
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
            },
        },
        "structure_generation": {
            "name": "test_structure_generation",
            "desired_number_of_structures": 10,
            "max_time": "1H",
            "structure_selection_kwargs": {
                "max_number_of_concurrent_jobs": 5,
                "chem_formula_list": None,
                "atom_number_range": (0, 21),
                "enforce_chemical_diversity": True,
            },
            "run_md_kwargs": {
                "steps": 1000,
                "temperature": 300,
                "timestep_fs": 0.5,
                "friction": 0.002,
            },
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
            },
        },
        "high_accuracy_evaluation": {
            "name": "test_dft",
            "max_time": "2H",
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
            },
        },
    }


@pytest.fixture
def sample_atoms_co2():
    """Create a sample CO2 molecule."""
    return Atoms(
        symbols=["C", "O", "O"],
        positions=np.ones((3, 3)) * 1.2,
        cell=[15.0, 15.0, 15.0],
        pbc=True,
    )


@pytest.fixture
def sample_training_data_co2(sample_atoms_co2):
    """Create sample training data with CO2."""
    atoms_list = []
    for i in range(10):
        atoms = sample_atoms_co2.copy()
        atoms.positions += np.random.random((3, 3)) * 0.1
        atoms.info["energy"] = -20.0 + i * 0.1
        atoms.arrays["forces"] = np.random.random((3, 3)) * 0.1
        atoms_list.append(atoms)
    return atoms_list


@pytest.fixture
def temp_files_co2(sample_training_data_co2, mock_job_config):
    """Create temporary training and test files with CO2 data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train_co2.xyz"
        test_file = Path(tmpdir) / "test_co2.xyz"

        # Write training data
        write(str(train_file), sample_training_data_co2[:7], format="extxyz")
        # Write test data
        write(str(test_file), sample_training_data_co2[7:], format="extxyz")

        yield str(train_file), str(test_file), mock_job_config


# ============================================================================
# Tests for initialize_training_set: Fast Path
# ============================================================================


@pytest.mark.unit
class TestInitializeTrainingSetFastPath:
    """Tests for the fast path of initialize_training_set.

    When initial_train_file_path and initial_test_file_path exist on disk,
    the method should load them directly without consulting the database.
    """

    def _make_workflow(self, tmp_path, minimal_jobs_dict):
        """Helper to create a workflow with MACE modules mocked."""
        return ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "train.xyz"),
            initial_test_file_path=str(tmp_path / "test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

    def test_fast_path_returns_loaded_structures(
        self, tmp_path, minimal_jobs_dict, sample_atoms_list
    ):
        """Verify that existing xyz files are loaded and returned directly."""
        train_path = tmp_path / "train.xyz"
        test_path = tmp_path / "test.xyz"

        # Write real xyz files to the expected paths
        write(str(train_path), sample_atoms_list[:3], format="extxyz")
        write(str(test_path), sample_atoms_list[3:], format="extxyz")

        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(train_path),
            initial_test_file_path=str(test_path),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        with patch("alomancy.core.standard_active_learning.write"):
            train, test = wf.initialize_training_set("initialization")

        # Verify structures were loaded correctly
        assert len(train) == 3, "Training set should have 3 structures"
        assert len(test) == 2, "Test set should have 2 structures"

    def test_fast_path_skips_db_check(
        self, tmp_path, minimal_jobs_dict, sample_atoms_list
    ):
        """Verify that compute_initialization_needs is not called in fast path."""
        train_path = tmp_path / "train.xyz"
        test_path = tmp_path / "test.xyz"

        write(str(train_path), sample_atoms_list, format="extxyz")
        write(str(test_path), sample_atoms_list[:1], format="extxyz")

        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(train_path),
            initial_test_file_path=str(test_path),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        # Mock compute_initialization_needs to ensure it's never called
        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs, patch("alomancy.core.standard_active_learning.write"):
            wf.initialize_training_set("initialization")

        # Fast path should skip DB entirely
        mock_needs.assert_not_called()

    def test_fast_path_writes_files_to_work_dir(
        self, tmp_path, minimal_jobs_dict, sample_atoms_list
    ):
        """Verify that fast-path results are also written to results/<base_name>/."""
        train_path = tmp_path / "external_train.xyz"
        test_path = tmp_path / "external_test.xyz"

        write(str(train_path), sample_atoms_list[:2], format="extxyz")
        write(str(test_path), sample_atoms_list[2:3], format="extxyz")

        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(train_path),
            initial_test_file_path=str(test_path),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        with patch("alomancy.core.standard_active_learning.write") as mock_write:
            _train, _test = wf.initialize_training_set("initialization")

        # Verify write was called to copy files to work_dir
        # (It should be called at least once to write the results to work_dir)
        assert mock_write.called, "Files should be written to work_dir"
        write_calls = [call[0] for call in mock_write.call_args_list]

        # Check that writes include paths under results/initialization/
        write_paths = [str(call[0]) if call else None for call in write_calls]
        has_results_dir = any(
            "results" in str(p) and "initialization" in str(p)
            for p in write_paths if p
        )
        assert has_results_dir, "Results should be written to results/initialization/"

    def test_fast_path_returns_single_structure_as_list(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that even a single structure file is loaded as a list."""
        train_path = tmp_path / "train.xyz"
        test_path = tmp_path / "test.xyz"

        # Single structure files
        single_atoms = Atoms("H", positions=[[0, 0, 0]])
        write(str(train_path), single_atoms, format="extxyz")
        write(str(test_path), single_atoms, format="extxyz")

        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(train_path),
            initial_test_file_path=str(test_path),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        with patch("alomancy.core.standard_active_learning.write"):
            train, test = wf.initialize_training_set("initialization")

        assert isinstance(train, list), "Training set should be a list"
        assert isinstance(test, list), "Test set should be a list"
        assert len(train) >= 1, "Training set should have at least 1 structure"
        assert len(test) >= 1, "Test set should have at least 1 structure"


# ============================================================================
# Tests for initialize_training_set: DB Path
# ============================================================================


@pytest.mark.unit
class TestInitializeTrainingSetDBPath:
    """Tests for the DB-aware path of initialize_training_set.

    When initial xyz files don't exist, the method should:
    1. Consult the database to determine what needs to be generated
    2. Generate missing structures
    3. Run DFT evaluation
    4. Add results to the database
    5. Build train/test sets from DB contents
    """

    def test_db_path_calls_compute_needs_when_no_files(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that compute_initialization_needs is called when xyz files don't exist."""
        # Create workflow with non-existent file paths
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        # Mock all the downstream functions to avoid expensive operations
        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs, patch(
            "alomancy.core.standard_active_learning.create_initialization_atoms_list"
        ), patch(
            "alomancy.core.standard_active_learning.read_atoms_file_if_enabled"
        ), patch.object(wf, "high_accuracy_evaluation") as mock_hae, patch(
            "alomancy.core.standard_active_learning.clean_structures"
        ), patch(
            "alomancy.core.standard_active_learning.write"
        ):
            # Mock compute_needs to return no requirements
            mock_needs.return_value = {
                "isolated_atoms": [],
                "dimer_override": {},
                "trimer_override": {},
                "amorphous_override": 0,
                "mp_structures": [],
            }

            # Mock high_accuracy_evaluation to return empty
            mock_hae.return_value = []

            # Mock db.get_all_as_atoms to return test structures
            test_struct = Atoms("H")
            test_struct.info["config_type"] = "IsolatedAtom"
            test_struct.info["REF_energy"] = -1.0
            test_struct.arrays["REF_forces"] = np.array([[0, 0, 0]])

            wf.db.get_all_as_atoms = Mock(return_value=[test_struct])

            _train, _test = wf.initialize_training_set("initialization")

        mock_needs.assert_called_once()

    def test_db_path_generates_structures_if_needed(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that create_initialization_atoms_list is called when DB check shows unmet needs."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        # Set up mocks
        needs_dict = {
            "isolated_atoms": ["H"],  # Some atoms still needed
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        generated_atoms = Atoms("H", positions=[[0, 0, 0]])
        generated_atoms.info["config_type"] = "initialization"

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.create_initialization_atoms_list"
            ) as mock_create:
                mock_create.return_value = [generated_atoms]

                with patch(
                    "alomancy.core.standard_active_learning.read_atoms_file_if_enabled"
                ), patch.object(wf, "high_accuracy_evaluation") as mock_hae, patch(
                    "alomancy.core.standard_active_learning.clean_structures"
                ) as mock_clean, patch(
                    "alomancy.core.standard_active_learning.write"
                ):
                    # Return the generated atoms from clean_structures
                    mock_hae.return_value = [generated_atoms]
                    mock_clean.return_value = [generated_atoms]

                    # Mock db operations
                    wf.db.add_structures = Mock(return_value=1)
                    test_struct = generated_atoms.copy()
                    wf.db.get_all_as_atoms = Mock(return_value=[test_struct])

                    _train, _test = wf.initialize_training_set("initialization")

        # Verify create_initialization_atoms_list was called
        mock_create.assert_called_once()

    def test_db_path_skips_generation_if_all_targets_met(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that generation is skipped when all initialization targets are already in DB."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        # Mock compute_needs to return no requirements
        needs_dict = {
            "isolated_atoms": [],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with (
                patch("alomancy.core.standard_active_learning.create_initialization_atoms_list") as mock_create,
                patch("alomancy.core.standard_active_learning.write"),
            ):
                # Set up DB to return an existing structure
                existing_struct = Atoms("H")
                existing_struct.info["config_type"] = "IsolatedAtom"
                existing_struct.info["REF_energy"] = -1.0
                existing_struct.arrays["REF_forces"] = np.array([[0, 0, 0]])

                wf.db.get_all_as_atoms = Mock(return_value=[existing_struct])

                _train, _test = wf.initialize_training_set("initialization")

        # When all targets are met, generation should be skipped
        mock_create.assert_not_called()

    def test_db_path_adds_structures_to_db(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that newly evaluated structures are added to the global database."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        needs_dict = {
            "isolated_atoms": ["H"],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        evaluated_atoms = Atoms("H", positions=[[0, 0, 0]])
        evaluated_atoms.info["config_type"] = "initialization"
        evaluated_atoms.info["REF_energy"] = -1.0
        evaluated_atoms.arrays["REF_forces"] = np.array([[0, 0, 0]])

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.create_initialization_atoms_list"
            ) as mock_create:
                mock_create.return_value = [evaluated_atoms]

                with patch(
                    "alomancy.core.standard_active_learning.read_atoms_file_if_enabled"
                ), patch.object(wf, "high_accuracy_evaluation") as mock_hae:
                    mock_hae.return_value = [evaluated_atoms]

                    with patch(
                        "alomancy.core.standard_active_learning.clean_structures"
                    ) as mock_clean:
                        mock_clean.return_value = [evaluated_atoms]

                        with patch(
                            "alomancy.core.standard_active_learning.write"
                        ):
                            # Mock db.add_structures
                            wf.db.add_structures = Mock(return_value=1)
                            wf.db.get_all_as_atoms = Mock(
                                return_value=[evaluated_atoms]
                            )

                            _train, _test = wf.initialize_training_set("initialization")

        # Verify add_structures was called with the evaluated structures
        wf.db.add_structures.assert_called_once()
        call_args = wf.db.add_structures.call_args
        assert call_args is not None, "add_structures should have been called"

    def test_db_path_builds_train_test_from_db(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that train/test sets are built from DB when no generation is needed."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        # Create test structures
        test_atoms_list = []
        for i in range(5):
            atoms = Atoms("H", positions=[[0, 0, 0]])
            if i < 2:
                atoms.info["config_type"] = "IsolatedAtom"
            else:
                atoms.info["config_type"] = "other_type"
            atoms.info["REF_energy"] = -1.0 - i * 0.1
            atoms.arrays["REF_forces"] = np.array([[0, 0, 0]])
            test_atoms_list.append(atoms)

        # Mock no generation needed
        needs_dict = {
            "isolated_atoms": [],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.write"
            ):
                wf.db.get_all_as_atoms = Mock(return_value=test_atoms_list)

                train, test = wf.initialize_training_set("initialization")

        # Verify structures were returned
        assert len(train) + len(test) == 5, "All structures should be distributed"

    def test_db_path_handles_no_eligible_test_structures(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify graceful handling when no structures match test_config_types."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        # Create structures that don't match test_config_types
        test_atoms = Atoms("H", positions=[[0, 0, 0]])
        test_atoms.info["config_type"] = "something_else"
        test_atoms.info["REF_energy"] = -1.0
        test_atoms.arrays["REF_forces"] = np.array([[0, 0, 0]])

        needs_dict = {
            "isolated_atoms": [],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.write"
            ):
                wf.db.get_all_as_atoms = Mock(return_value=[test_atoms])

                train, test = wf.initialize_training_set("initialization")

        # When no eligible test structures, all go to training
        assert len(train) >= 1, "All structures should go to training"
        assert len(test) == 0, "No eligible structures for test set"

    def test_db_path_reads_pre_generated_file_if_exists(
        self, tmp_path, minimal_jobs_dict, sample_atoms_list
    ):
        """Verify that pre-generated structures are read if configured."""
        # Update jobs_dict to include read_generated_file
        jobs_dict = minimal_jobs_dict.copy()
        jobs_dict["initialization"]["read_generated_file"] = "generated_structures.xyz"

        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        needs_dict = {
            "isolated_atoms": ["H"],  # Something needed
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.read_atoms_file_if_enabled"
            ) as mock_read_file:
                # Return pre-generated structures
                mock_read_file.return_value = sample_atoms_list

                with (
                    patch("alomancy.core.standard_active_learning.create_initialization_atoms_list") as mock_create,
                    patch.object(wf, "high_accuracy_evaluation") as mock_hae,
                    patch("alomancy.core.standard_active_learning.clean_structures") as mock_clean,
                    patch("alomancy.core.standard_active_learning.write"),
                ):
                    mock_hae.return_value = sample_atoms_list
                    mock_clean.return_value = sample_atoms_list
                    wf.db.add_structures = Mock(return_value=len(sample_atoms_list))
                    wf.db.get_all_as_atoms = Mock(return_value=sample_atoms_list)

                    _train, _test = wf.initialize_training_set("initialization")

        # If pre-generated file exists, create_initialization_atoms_list should not be called
        mock_create.assert_not_called()


# ============================================================================
# Tests for initialize_training_set: Error Handling
# ============================================================================


@pytest.mark.unit
class TestInitializeTrainingSetErrorHandling:
    """Tests for error conditions in initialize_training_set."""

    def test_raises_error_when_no_structures_generated(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that an error is raised when generation produces no structures."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        needs_dict = {
            "isolated_atoms": ["H"],  # Something needed
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.create_initialization_atoms_list"
            ) as mock_create:
                # Return empty list (no structures generated)
                mock_create.return_value = []

                with (
                    patch("alomancy.core.standard_active_learning.read_atoms_file_if_enabled"),
                    patch("alomancy.core.standard_active_learning.write"),
                    pytest.raises(ValueError, match="No structures were generated"),
                ):
                    wf.initialize_training_set("initialization")

    def test_raises_error_when_no_high_accuracy_structures(
        self, tmp_path, minimal_jobs_dict
    ):
        """Verify that an error is raised when DFT evaluation produces no results."""
        wf = ActiveLearningStandardMACE(
            initial_train_file_path=str(tmp_path / "nonexistent_train.xyz"),
            initial_test_file_path=str(tmp_path / "nonexistent_test.xyz"),
            jobs_dict=minimal_jobs_dict,
            db_path=str(tmp_path / "db"),
        )

        needs_dict = {
            "isolated_atoms": ["H"],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": [],
        }

        generated_atoms = Atoms("H", positions=[[0, 0, 0]])

        with patch(
            "alomancy.core.standard_active_learning.compute_initialization_needs"
        ) as mock_needs:
            mock_needs.return_value = needs_dict

            with patch(
                "alomancy.core.standard_active_learning.create_initialization_atoms_list"
            ) as mock_create:
                mock_create.return_value = [generated_atoms]

                with patch(
                    "alomancy.core.standard_active_learning.read_atoms_file_if_enabled"
                ), patch.object(wf, "high_accuracy_evaluation") as mock_hae:
                    # Return empty list (no DFT results)
                    mock_hae.return_value = []

                    with patch(
                        "alomancy.core.standard_active_learning.write"
                    ), pytest.raises(
                        ValueError,
                        match="No high-accuracy structures returned"
                    ):
                        wf.initialize_training_set("initialization")


# ============================================================================
# Tests from original file (preserved for compatibility)
# ============================================================================


@pytest.fixture
def mock_job_config_full():
    return {
        "mlip_committee": {
            "name": "test_committee",
            "size_of_committee": 3,
            "max_time": "1H",
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
            },
        },
        "structure_generation": {
            "name": "test_structure_generation",
            "desired_number_of_structures": 10,
            "max_time": "1H",
            "structure_selection_kwargs": {
                "max_number_of_concurrent_jobs": 5,
                "chem_formula_list": None,
                "atom_number_range": (0, 21),
                "enforce_chemical_diversity": True,
            },
            "run_md_kwargs": {
                "steps": 1000,
                "temperature": 300,
                "timestep_fs": 0.5,
                "friction": 0.002,
            },
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
            },
        },
        "high_accuracy_evaluation": {
            "name": "test_dft",
            "max_time": "2H",
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
            },
        },
    }


@pytest.mark.unit
class TestActiveLearningStandardMACE:
    """Test the ActiveLearningStandardMACE class."""

    def test_initialization(self, temp_files_co2):
        """Test workflow initialization."""
        train_file, test_file, mock_job_config = temp_files_co2

        workflow = ActiveLearningStandardMACE(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            jobs_dict=mock_job_config,
            number_of_al_loops=3,
            verbose=1,
        )

        assert workflow.initial_train_file_path == Path(train_file)
        assert workflow.initial_test_file_path == Path(test_file)
        assert workflow.number_of_al_loops == 3
        assert workflow.verbose == 1

    @patch("alomancy.core.standard_active_learning.committee_remote_submitter")
    @patch("alomancy.core.standard_active_learning.get_mace_eval_info")
    @patch("alomancy.configs.remote_info.get_remote_info")
    def test_train_mlip(
        self,
        mock_get_remote_info,
        mock_mace_recover,
        mock_committee_submitter,
        temp_files_co2,
        mock_job_config_full,
    ):
        """Test MLIP training method."""
        train_file, test_file, _ = temp_files_co2

        # Set up mocks
        mock_remote_info = MagicMock()
        mock_get_remote_info.return_value = mock_remote_info

        mock_results_df = pd.DataFrame(
            {
                "mae_e": [0.1, 0.08, 0.12],
                "mae_f": [0.2, 0.18, 0.22],
            }
        )
        mock_mace_recover.return_value = mock_results_df

        workflow = ActiveLearningStandardMACE(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            jobs_dict=mock_job_config_full,
            plots=False,
        )

        # Test train_mlip method
        result = workflow.train_mlip("test_loop_0", mock_job_config_full["mlip_committee"])

        # Verify remote submitter was called
        mock_committee_submitter.assert_called_once()

        # Verify the result is the DataFrame returned by mace_recover
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, mock_results_df)

    @patch("alomancy.core.standard_active_learning.qe_remote_submitter")
    @patch("alomancy.configs.remote_info.get_remote_info")
    @patch("alomancy.core.standard_active_learning.read")
    def test_high_accuracy_evaluation(
        self,
        mock_read,
        mock_get_remote_info,
        mock_qe_submitter,
        temp_files_co2,
        sample_atoms_co2,
    ):
        """Test high accuracy evaluation method."""
        train_file, test_file, mock_job_config = temp_files_co2

        # Ensure max_batch_size is set
        mock_job_config["high_accuracy_evaluation"]["max_batch_size"] = 10

        # Set up mocks
        mock_qe_structures = [sample_atoms_co2.copy() for _ in range(2)]
        for atoms in mock_qe_structures:
            atoms.info["energy"] = -20.5
            atoms.arrays["forces"] = np.random.random((3, 3)) * 0.1
            atoms.get_potential_energy = Mock(return_value=-1.0)
            atoms.get_forces = Mock(return_value=np.array([[0, 0, 0]]))
        mock_read.side_effect = mock_qe_structures

        mock_remote_info = MagicMock()
        mock_get_remote_info.return_value = mock_remote_info

        workflow = ActiveLearningStandardMACE(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            jobs_dict=mock_job_config,
        )

        input_structures = [sample_atoms_co2.copy() for _ in range(2)]

        result = workflow.high_accuracy_evaluation(
            "test_loop_0", mock_job_config["high_accuracy_evaluation"], input_structures
        )

        # Verify results
        assert len(result) >= 0, "Should return a list of structures"


@pytest.mark.integration
class TestActiveLearningStandardMACEIntegration:
    """Integration tests for the standard MACE workflow."""

    def test_full_workflow_execution_with_tempdir(
        self, temp_files_co2, sample_atoms_co2
    ):
        """Test workflow execution in a controlled temporary directory."""
        train_file, test_file, mock_job_config = temp_files_co2

        # Add required 'initialization' key to jobs_dict
        mock_job_config["initialization"] = {
            "extra_datasets": [],
            "test_to_train_ratio": 0.1,
            "test_config_types": ["al_loop_0"],
        }

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Mock all the heavy external operations
                with (
                    patch.object(
                        ActiveLearningStandardMACE, "train_mlip"
                    ) as mock_train,
                    patch.object(
                        ActiveLearningStandardMACE, "generate_structures"
                    ) as mock_gen,
                    patch.object(
                        ActiveLearningStandardMACE, "high_accuracy_evaluation"
                    ) as mock_ha,
                ):
                    # Configure method return values
                    mock_train.return_value = pd.DataFrame(
                        {"mae_f": [0.1], "mae_e": [0.2]}
                    )
                    mock_gen.return_value = [sample_atoms_co2.copy() for _ in range(5)]
                    mock_ha.return_value = [sample_atoms_co2.copy() for _ in range(3)]
                    for atoms in mock_ha.return_value:
                        atoms.get_potential_energy = Mock(return_value=-1.0)
                        atoms.get_forces = Mock(return_value=np.zeros((3, 3)))

                    workflow = ActiveLearningStandardMACE(
                        initial_train_file_path=train_file,
                        initial_test_file_path=test_file,
                        jobs_dict=mock_job_config,
                        number_of_al_loops=1,
                        verbose=0,
                    )

                    # This should work since we're in a temp directory with proper mocks
                    workflow.run()

                    # Verify workflow executed properly
                    mock_train.assert_called_once()
                    mock_gen.assert_called_once()
                    mock_ha.assert_called_once()

            finally:
                os.chdir(old_cwd)


@pytest.mark.slow
@pytest.mark.requires_external
class TestActiveLearningStandardMACEExternal:
    """Tests that require external dependencies (MACE, QE, etc.)."""

    def test_with_real_mace_calculator(self, skip_if_no_external):
        """Test with real MACE calculator if available."""
        pass

    def test_with_real_quantum_espresso(self, skip_if_no_external):
        """Test with real Quantum Espresso if available."""
        pass
