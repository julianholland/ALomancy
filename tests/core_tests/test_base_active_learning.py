"""
Tests for the base active learning workflow.

This module tests the BaseActiveLearningWorkflow abstract class and its core functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write
from yaml import dump

from alomancy.core.base_active_learning import BaseActiveLearningWorkflow


class ConcreteActiveLearningWorkflow(BaseActiveLearningWorkflow):
    """Concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mlip_calls = []
        self.evaluate_mlip_calls = []
        self.generate_structures_calls = []
        self.high_accuracy_evaluation_calls = []

    def train_mlip(self, base_name, mlip_committee_job_dict, **kwargs):
        self.train_mlip_calls.append((base_name, mlip_committee_job_dict, kwargs))
        return "mock_model.pt"

    def evaluate_mlip(self, mlip_committee_job_dict, **kwargs):
        self.evaluate_mlip_calls.append((mlip_committee_job_dict, kwargs))
        return pd.DataFrame({"rmse": [0.1], "mae": [0.05]})

    def generate_structures(self, base_name, job_dict, train_data, **kwargs):
        self.generate_structures_calls.append((base_name, job_dict, train_data, kwargs))
        # Return some mock structures
        mock_structures = []
        for i in range(2):
            atoms = Atoms(symbols=["H"], positions=[[0, 0, i]])
            atoms.info["job_id"] = i
            mock_structures.append(atoms)
        return mock_structures

    def high_accuracy_evaluation(
        self, base_name, high_accuracy_eval_job_dict, structures, **kwargs
    ):
        self.high_accuracy_evaluation_calls.append(
            (base_name, high_accuracy_eval_job_dict, structures, kwargs)
        )
        # Add energy and forces to structures
        for atoms in structures:
            atoms.info["energy"] = -1.0
            atoms.arrays["forces"] = np.array([[0, 0, 0]])
            # Mock the get_potential_energy method
            atoms.get_potential_energy = Mock(return_value=-1.0)
            atoms.get_forces = Mock(return_value=np.array([[0, 0, 0]]))
        return structures


@pytest.fixture
def sample_atoms_h2o():
    """Create a sample H2O molecule."""
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


@pytest.fixture
def sample_training_data(sample_atoms_h2o):
    """Create sample training data."""
    atoms_list = []
    for i in range(5):
        atoms = sample_atoms_h2o.copy()
        atoms.positions += np.random.random((3, 3)) * 0.1
        atoms.info["energy"] = -10.0 + i * 0.1
        atoms.calc = EMT()
        atoms.arrays["forces"] = np.random.random((3, 3)) * 0.1
        atoms_list.append(atoms)
    return atoms_list


@pytest.fixture
def temp_files(sample_training_data):
    """Create temporary training and test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train.xyz"
        test_file = Path(tmpdir) / "test.xyz"
        config_file = Path(tmpdir) / "config.yaml"

        # Write training data
        write(str(train_file), sample_training_data[:3], format="extxyz")
        # Write test data
        write(str(test_file), sample_training_data[3:], format="extxyz")

        sample_config = {
            "mlip_committee": {
                "name": "test_mlip",
                "size_of_committee": 3,
                "max_time": "1H",
                "hpc": {
                    "hpc_name": "test-hpc",
                    "pre_cmds": ["echo 'test'"],
                    "partitions": ["test"],
                },
            },
            "structure_generation": {
                "name": "test_md",
                "number_of_concurrent_jobs": 2,
                "max_time": "30m",
                "hpc": {
                    "hpc_name": "test-hpc",
                    "pre_cmds": ["echo 'test'"],
                    "partitions": ["test"],
                },
            },
            "high_accuracy_evaluation": {
                "name": "test_qe",
                "pwx_path": "/path/to/pwx",
                "pp_path": "/path/to/pp",
                "pseudo_dict": {"O": "/path/to/O.pseudo", "H": "/path/to/H.pseudo"},
                "node_info": {
                    "ranks_per_system": 16,
                    "ranks_per_node": 4,
                    "threads_per_rank": 1,
                    "max_mem_per_node": "8GB",
                },
            },
        }

        with open(config_file, "w") as f:
            dump(sample_config, f)

        yield str(train_file), str(test_file), str(config_file)


class TestBaseActiveLearningWorkflow:
    """Test the BaseActiveLearningWorkflow class."""

    def test_initialization(self, temp_files):
        """Test workflow initialization."""
        train_file, test_file, config_file = temp_files

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
            number_of_al_loops=3,
            verbose=1,
            start_loop=1,
        )

        assert workflow.initial_train_file == Path(train_file)
        assert workflow.initial_test_file == Path(test_file)
        assert workflow.config_file_path == str(Path(config_file))
        assert workflow.number_of_al_loops == 3
        assert workflow.verbose == 1
        assert workflow.start_loop == 1

    def test_initialization_defaults(self, temp_files):
        """Test workflow initialization with defaults."""
        train_file, test_file, config_file = temp_files

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
        )

        assert workflow.number_of_al_loops == 5
        assert workflow.verbose == 0
        assert workflow.start_loop == 0

    @patch("alomancy.core.base_active_learning.load_dictionaries")
    def test_load_dictionaries_called(self, mock_load_dict, temp_files):
        """Test that load_dictionaries is called during initialization."""
        train_file, test_file, config_file = temp_files
        mock_load_dict.return_value = {"test": "dict"}

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
        )

        mock_load_dict.assert_called_once_with(config_file)
        assert workflow.jobs_dict == {"test": "dict"}

    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    @patch("alomancy.utils.clean_structures.clean_structures")
    def test_run_workflow_structure(
        self, mock_clean_structures, mock_load_dict, temp_files
    ):
        """Test the overall structure of running the AL workflow."""
        train_file, test_file, config_file = temp_files
        mock_load_dict.return_value = {
            "mlip_committee": {"name": "test_mlip"},
            "structure_generation": {"name": "test_md"},
            "high_accuracy_evaluation": {"name": "test_qe"},
        }

        # Mock clean_structures to return the input structures
        mock_clean_structures.side_effect = lambda structures, *args: structures

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
            number_of_al_loops=2,
            start_loop=0,
        )

        # Mock the results directory creation and file operations
        with (
            patch("pathlib.Path.mkdir"),
            patch("alomancy.core.base_active_learning.write"),  # Add this patch
        ):
            workflow.run()

        # Check that all abstract methods were called the expected number of times
        assert len(workflow.train_mlip_calls) == 2  # 2 loops
        assert len(workflow.evaluate_mlip_calls) == 2  # 2 loops
        assert len(workflow.generate_structures_calls) == 2  # 2 loops
        assert len(workflow.high_accuracy_evaluation_calls) == 2  # 2 loops

        # Check that base_name is correct for each loop
        assert workflow.train_mlip_calls[0][0] == "al_loop_0"
        assert workflow.train_mlip_calls[1][0] == "al_loop_1"

    # @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    # @patch("alomancy.utils.clean_structures.clean_structures")  # Add this patch
    # def test_run_workflow_structure(
    #     self, mock_clean_structures, mock_load_dict, temp_files
    # ):
    #     """Test the overall structure of running the AL workflow."""
    #     train_file, test_file, config_file = temp_files
    #     mock_load_dict.return_value = {
    #         "mlip_committee": {"name": "test_mlip"},
    #         "structure_generation": {"name": "test_md"},
    #         "high_accuracy_evaluation": {"name": "test_qe"},
    #     }

    #     # Mock clean_structures to return the input structures
    #     mock_clean_structures.side_effect = lambda structures, *args: structures

    #     workflow = ConcreteActiveLearningWorkflow(
    #         initial_train_file_path=train_file,
    #         initial_test_file_path=test_file,
    #         config_file_path=config_file,
    #         number_of_al_loops=2,
    #         start_loop=0,
    #     )

    #     # Mock the results directory creation and file operations
    #     with (
    #         patch("pathlib.Path.mkdir"),
    #     ):
    #         workflow.run()

    #     # Check that all abstract methods were called the expected number of times
    #     assert len(workflow.train_mlip_calls) == 2  # 2 loops
    #     assert len(workflow.evaluate_mlip_calls) == 2  # 2 loops
    #     assert len(workflow.generate_structures_calls) == 2  # 2 loops
    #     assert len(workflow.high_accuracy_evaluation_calls) == 2  # 2 loops

    #     # Check that base_name is correct for each loop
    #     assert workflow.train_mlip_calls[0][0] == "al_loop_0"
    #     assert workflow.train_mlip_calls[1][0] == "al_loop_1"

    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    def test_run_with_start_loop(self, mock_load_dict, temp_files):
        """Test running with a non-zero start loop."""
        train_file, test_file, config_file = temp_files
        mock_load_dict.return_value = {
            "mlip_committee": {"name": "test_mlip"},
            "structure_generation": {"name": "test_md"},
            "high_accuracy_evaluation": {"name": "test_qe"},
        }

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
            number_of_al_loops=3,
            start_loop=1,
        )

        with (
            patch("pathlib.Path.mkdir"),
            patch("alomancy.core.base_active_learning.write"),
        ):
            workflow.run()

        # Should only run loops 1 and 2 (3-1=2 loops)
        assert len(workflow.train_mlip_calls) == 2
        assert workflow.train_mlip_calls[0][0] == "al_loop_1"
        assert workflow.train_mlip_calls[1][0] == "al_loop_2"

    def test_abstract_methods_must_be_implemented(self, temp_files):
        """Test that abstract methods must be implemented."""
        train_file, test_file, config_file = temp_files

        # Should not be able to instantiate the abstract base class
        with pytest.raises(TypeError):
            BaseActiveLearningWorkflow(
                initial_train_file_path=train_file,
                initial_test_file_path=test_file,
                config_file_path=config_file,
            )  # ignore: para

    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    def test_verbose_output(self, mock_load_dict, temp_files, capsys):
        """Test verbose output during workflow execution."""
        train_file, test_file, config_file = temp_files
        mock_load_dict.return_value = {
            "mlip_committee": {"name": "test_mlip"},
            "structure_generation": {"name": "test_md"},
            "high_accuracy_evaluation": {"name": "test_qe"},
        }

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
            number_of_al_loops=1,
            verbose=1,
        )

        with (
            patch("pathlib.Path.mkdir"),
            patch("alomancy.core.base_active_learning.write"),
        ):
            workflow.run()

        captured = capsys.readouterr()
        assert "Starting AL loop 0" in captured.out
        assert "Training set size:" in captured.out
        assert "Test set size:" in captured.out
        assert "AL Loop 0 evaluation results:" in captured.out
        assert "Completed AL loop 0" in captured.out

    def test_file_not_found_error(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            workflow = ConcreteActiveLearningWorkflow(
                initial_train_file_path="nonexistent_train.xyz",
                initial_test_file_path="nonexistent_test.xyz",
                config_file_path="nonexistent_config.yaml",
            )
            workflow.run()

    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    def test_insufficient_training_data(self, mock_load_dict, temp_files):
        """Test error when insufficient training data is provided."""
        train_file, test_file, config_file = temp_files
        mock_load_dict.return_value = {
            "mlip_committee": {"name": "test_mlip"},
            "structure_generation": {"name": "test_md"},
            "high_accuracy_evaluation": {"name": "test_qe"},
        }

        # Create a file with only one structure (should fail assertion)
        with tempfile.TemporaryDirectory() as tmpdir:
            single_train_file = Path(tmpdir) / "single_train.xyz"
            single_atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]])
            write(str(single_train_file), [single_atoms], format="extxyz")

            workflow = ConcreteActiveLearningWorkflow(
                initial_train_file_path=str(single_train_file),
                initial_test_file_path=test_file,
                config_file_path=config_file,
            )

            with pytest.raises(
                AssertionError, match="More than one training structure required"
            ):
                workflow.run()


@pytest.mark.integration
class TestActiveLearningIntegration:
    """Integration tests for the active learning workflow."""

    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    def test_full_workflow_integration(self, mock_load_dict, temp_files):
        """Test a complete workflow execution."""
        train_file, test_file, config_file = temp_files
        mock_load_dict.return_value = {
            "mlip_committee": {"name": "test_mlip"},
            "structure_generation": {"name": "test_md"},
            "high_accuracy_evaluation": {"name": "test_qe"},
        }

        workflow = ConcreteActiveLearningWorkflow(
            initial_train_file_path=train_file,
            initial_test_file_path=test_file,
            config_file_path=config_file,
            number_of_al_loops=2,
            verbose=0,
        )

        with (
            patch("pathlib.Path.mkdir"),
            patch("alomancy.core.base_active_learning.write") as mock_write,
        ):
            workflow.run()

        # Verify the workflow completed all steps
        assert len(workflow.train_mlip_calls) == 2
        assert len(workflow.evaluate_mlip_calls) == 2
        assert len(workflow.generate_structures_calls) == 2
        assert len(workflow.high_accuracy_evaluation_calls) == 2

        # Verify files were written for each loop
        # Should write train_set.xyz and test_set.xyz for each loop
        assert mock_write.call_count >= 4  # At least 2 loops * 2 files per loop
