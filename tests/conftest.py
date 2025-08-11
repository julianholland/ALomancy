"""
Alomancy Test Configuration and Fixtures

This module provides common fixtures and configuration for testing the alomancy package.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from ase import Atoms

try:
    import pytest
except ImportError:
    pytest = None


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a directory for test data."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_atoms():
    """Provide a simple H2O molecule for testing."""
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


@pytest.fixture
def sample_atoms_list(sample_atoms):
    """Provide a list of sample atoms for testing."""
    atoms_list = []
    for i in range(5):
        atoms = sample_atoms.copy()
        # Add some variation
        atoms.positions += np.random.random((3, 3)) * 0.1
        atoms.info["energy"] = -10.0 + i * 0.1
        atoms.arrays["forces"] = np.random.random((3, 3)) * 0.1
        atoms_list.append(atoms)
    return atoms_list


@pytest.fixture
def mock_job_dict():
    """Provide a mock job dictionary for testing."""
    return {
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
            "max_time": "10m",
            "hpc": {
                "hpc_name": "test-hpc",
                "pre_cmds": ["echo 'test'"],
                "partitions": ["test"],
                "node_info": {
                    "ranks_per_system": 4,
                    "ranks_per_node": 4,
                    "threads_per_rank": 1,
                    "max_mem_per_node": "8GB",
                },
                "pwx_path": "/usr/bin/pw.x",
                "pp_path": "/test/pps",
                "pseudo_dict": {"O": "O.pbe.UPF", "H": "H.pbe.UPF"},
            },
        },
    }


@pytest.fixture
def mock_config_dict(mock_job_dict):
    """Mock the config dictionaries loading."""
    return mock_job_dict


@pytest.fixture
def sample_train_file(temp_dir, sample_atoms_list):
    """Create a sample training file."""
    train_file = temp_dir / "train.xyz"
    # Mock writing atoms to file - in reality would use ase.io.write
    train_file.write_text("# Sample training data")
    return str(train_file)


@pytest.fixture
def sample_test_file(temp_dir, sample_atoms_list):
    """Create a sample test file."""
    test_file = temp_dir / "test.xyz"
    # Mock writing atoms to file - in reality would use ase.io.write
    test_file.write_text("# Sample test data")
    return str(test_file)


@pytest.fixture
def mock_remote_info():
    """Mock RemoteInfo object."""
    mock = MagicMock()
    mock.host = "test-host"
    mock.username = "test-user"
    return mock


@pytest.fixture
def mock_mace_calculator():
    """Mock MACE calculator."""
    mock_calc = MagicMock()
    mock_calc.calculate.return_value = {"energy": -10.0, "forces": np.random.random((3, 3))}
    return mock_calc


@pytest.fixture
def mock_espresso_calculator():
    """Mock Espresso calculator."""
    mock_calc = MagicMock()
    mock_calc.calculate.return_value = {"energy": -10.0, "forces": np.random.random((3, 3))}
    return mock_calc


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Set up test environment variables and paths."""
    # Set up test environment
    monkeypatch.setenv("ALOMANCY_TEST_MODE", "1")
    monkeypatch.setenv("ALOMANCY_TEST_DATA_DIR", str(temp_dir))
    
    # Mock external dependencies by default
    monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "1")


@pytest.fixture
def skip_if_no_external():
    """Skip test if external dependencies are not available."""
    if os.getenv("ALOMANCY_MOCK_EXTERNAL", "1") == "1":
        pytest.skip("External dependencies not available in test environment")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_if_no_mace():
    """Skip test if MACE is not available."""
    try:
        import mace  # noqa: F401
    except ImportError:
        pytest.skip("MACE not available")


@pytest.fixture
def mock_wfl_autoparallelize(monkeypatch):
    """Mock wfl autoparallelize functionality."""
    mock_remote_info = MagicMock()
    
    # Mock the RemoteInfo class
    mock_remote_info_class = MagicMock()
    mock_remote_info_class.return_value = mock_remote_info
    
    monkeypatch.setattr("wfl.autoparallelize.remoteinfo.RemoteInfo", mock_remote_info_class)
    
    return mock_remote_info


class MockActiveLearningWorkflow:
    """Mock implementation of BaseActiveLearningWorkflow for testing."""
    
    def __init__(self, *args, **kwargs):
        self.initial_train_file = Path("mock_train.xyz")
        self.initial_test_file = Path("mock_test.xyz")
        self.number_of_al_loops = 2
        self.verbose = 0
        self.start_loop = 0
        self.jobs_dict = {}
    
    def train_mlip(self, base_name, mlip_committee_job_dict, **kwargs):
        return "mock_model.pt"
    
    def evaluate_mlip(self, base_name, mlip_committee_job_dict, **kwargs):
        import pandas as pd
        return pd.DataFrame({"rmse": [0.1], "mae": [0.05]})
    
    def generate_structures(self, base_name, job_dict, train_data, **kwargs):
        return [Atoms(symbols=["H"], positions=[[0, 0, 0]])]
    
    def high_accuracy_evaluation(self, base_name, high_accuracy_eval_job_dict, structures, **kwargs):
        for atoms in structures:
            atoms.info["energy"] = -1.0
            atoms.arrays["forces"] = np.array([[0, 0, 0]])
        return structures


@pytest.fixture
def mock_workflow():
    """Provide a mock workflow for testing."""
    return MockActiveLearningWorkflow()
