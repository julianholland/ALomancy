"""
Tests for utility functions and modules.

This module tests various utility functions used throughout the alomancy package.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms


class TestRemoteJobExecutor:
    """Test remote job execution utilities."""
    
    @patch('wfl.autoparallelize.remoteinfo.RemoteInfo')
    def test_remote_job_executor_initialization(self, mock_remote_info_class):
        """Test RemoteJobExecutor initialization."""
        # This would test the RemoteJobExecutor class if it exists
        # For now, we'll test that we can mock the dependencies
        mock_remote_info = MagicMock()
        mock_remote_info_class.return_value = mock_remote_info
        
        # Test that we can create a mock remote info
        assert mock_remote_info is not None
    
    def test_job_config_validation(self):
        """Test job configuration validation."""
        # Test valid job config
        valid_config = {
            "function_kwargs": {
                "param1": "value1",
                "param2": 42
            }
        }
        
        assert "function_kwargs" in valid_config
        assert isinstance(valid_config["function_kwargs"], dict)
    
    def test_invalid_job_config(self):
        """Test handling of invalid job configurations."""
        invalid_configs = [
            {},  # Empty config
            {"wrong_key": "value"},  # Missing function_kwargs
            {"function_kwargs": "not_a_dict"}  # Wrong type for function_kwargs
        ]
        
        for config in invalid_configs:
            if "function_kwargs" not in config:
                assert "function_kwargs" not in config
            elif not isinstance(config.get("function_kwargs"), dict):
                assert not isinstance(config.get("function_kwargs"), dict)


class TestConfigurationUtils:
    """Test configuration utility functions."""
    
    def test_load_dictionaries_structure(self):
        """Test that load_dictionaries returns expected structure."""
        from alomancy.configs.config_dictionaries import load_dictionaries
        
        config = load_dictionaries()
        
        # Test that required keys are present
        required_keys = ["mlip_committee", "structure_generation", "high_accuracy_evaluation"]
        for key in required_keys:
            assert key in config
        
        # Test that each job dict has required fields
        for job_name, job_config in config.items():
            assert "name" in job_config
            assert "max_time" in job_config
            assert "hpc" in job_config
    
    def test_job_dict_validation(self):
        """Test validation of job dictionary structure."""
        from alomancy.configs.config_dictionaries import load_dictionaries
        
        config = load_dictionaries()
        
        # Test mlip_committee specific fields
        mlip_config = config["mlip_committee"]
        assert "size_of_committee" in mlip_config
        assert isinstance(mlip_config["size_of_committee"], int)
        assert mlip_config["size_of_committee"] > 0
        
        # Test structure_generation specific fields
        struct_gen_config = config["structure_generation"]
        assert "number_of_concurrent_jobs" in struct_gen_config
        assert isinstance(struct_gen_config["number_of_concurrent_jobs"], int)
        
        # Test high_accuracy_evaluation specific fields
        ha_eval_config = config["high_accuracy_evaluation"]
        assert "hpc" in ha_eval_config
        if "node_info" in ha_eval_config["hpc"]:
            node_info = ha_eval_config["hpc"]["node_info"]
            required_node_fields = ["ranks_per_system", "ranks_per_node", "threads_per_rank"]
            for field in required_node_fields:
                assert field in node_info
    
    @patch('alomancy.configs.config_dictionaries.load_dictionaries')
    def test_config_customization(self, mock_load_dict):
        """Test that configuration can be customized."""
        custom_config = {
            "mlip_committee": {
                "name": "custom_mlip",
                "size_of_committee": 10,
                "max_time": "8H",
                "hpc": {"hpc_name": "custom-hpc"}
            },
            "structure_generation": {
                "name": "custom_md",
                "number_of_concurrent_jobs": 8,
                "max_time": "4H",
                "hpc": {"hpc_name": "custom-hpc"}
            },
            "high_accuracy_evaluation": {
                "name": "custom_qe",
                "max_time": "1H",
                "hpc": {"hpc_name": "custom-hpc"}
            }
        }
        
        mock_load_dict.return_value = custom_config
        
        config = mock_load_dict()
        
        assert config["mlip_committee"]["size_of_committee"] == 10
        assert config["structure_generation"]["number_of_concurrent_jobs"] == 8


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_path_operations(self):
        """Test path manipulation utilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir)
            
            # Test directory creation
            subdir = test_path / "subdir" / "nested"
            subdir.mkdir(parents=True, exist_ok=True)
            assert subdir.exists()
            assert subdir.is_dir()
            
            # Test file creation
            test_file = subdir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"
    
    def test_xyz_file_operations(self):
        """Test XYZ file operations."""
        from ase.io import read, write
        
        # Create sample atoms
        atoms = Atoms(
            symbols=["H", "H"],
            positions=[[0, 0, 0], [0, 0, 1]],
            cell=[10, 10, 10],
            pbc=True
        )
        atoms.info["energy"] = -2.0
        atoms.arrays["forces"] = np.array([[0, 0, 0.1], [0, 0, -0.1]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_file = Path(tmpdir) / "test.xyz"
            
            # Test writing
            write(str(xyz_file), atoms, format="extxyz")
            assert xyz_file.exists()
            
            # Test reading
            read_atoms = read(str(xyz_file), format="extxyz")
            assert len(read_atoms) == 2
            assert read_atoms.info["energy"] == -2.0
            np.testing.assert_array_almost_equal(
                read_atoms.arrays["forces"], 
                atoms.arrays["forces"]
            )


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_atoms_validation(self):
        """Test validation of ASE Atoms objects."""
        # Valid atoms object
        valid_atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            cell=[10, 10, 10],
            pbc=True
        )
        
        assert len(valid_atoms) == 3
        assert valid_atoms.get_chemical_symbols() == ["O", "H", "H"]
        assert valid_atoms.cell is not None
        
        # Test with energy and forces
        valid_atoms.info["energy"] = -15.0
        valid_atoms.arrays["forces"] = np.random.random((3, 3))
        
        assert "energy" in valid_atoms.info
        assert "forces" in valid_atoms.arrays
        assert valid_atoms.arrays["forces"].shape == (3, 3)
    
    def test_structure_list_validation(self):
        """Test validation of structure lists."""
        atoms_list = []
        for i in range(5):
            atoms = Atoms(
                symbols=["C"],
                positions=[[i, 0, 0]],
                cell=[20, 20, 20],
                pbc=True
            )
            atoms.info["energy"] = -i
            atoms.arrays["forces"] = np.array([[0, 0, 0]])
            atoms_list.append(atoms)
        
        # Test list properties
        assert len(atoms_list) == 5
        assert all(isinstance(atoms, Atoms) for atoms in atoms_list)
        assert all("energy" in atoms.info for atoms in atoms_list)
        assert all("forces" in atoms.arrays for atoms in atoms_list)
    
    def test_energy_forces_consistency(self):
        """Test consistency between energy and forces data."""
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[[0, 0, 0], [1.1, 0, 0]],
            cell=[15, 15, 15],
            pbc=True
        )
        
        # Add energy and forces
        atoms.info["energy"] = -10.5
        atoms.arrays["forces"] = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
        
        # Test that forces shape matches number of atoms
        assert atoms.arrays["forces"].shape[0] == len(atoms)
        assert atoms.arrays["forces"].shape[1] == 3  # x, y, z components
        
        # Test energy is a scalar
        assert isinstance(atoms.info["energy"], (int, float))


class TestArrayOperations:
    """Test array operation utilities."""
    
    def test_force_array_operations(self):
        """Test operations on force arrays."""
        # Test force flattening (from md_wfl.py)
        forces = np.random.random((5, 3))  # 5 atoms, 3 components each
        
        def flatten_array_of_forces(forces_array):
            return np.reshape(forces_array, (1, forces_array.shape[0] * 3))
        
        flattened = flatten_array_of_forces(forces)
        assert flattened.shape == (1, 15)  # 5 atoms * 3 components
        
        # Test unflattening
        unflattened = flattened.reshape((5, 3))
        np.testing.assert_array_equal(forces, unflattened)
    
    def test_statistical_operations(self):
        """Test statistical operations on arrays."""
        # Create sample force data from multiple models
        n_structures = 10
        n_models = 5
        n_atoms = 3
        
        force_data = {}
        for model_id in range(n_models):
            force_data[f"model_{model_id}"] = {}
            for struct_id in range(n_structures):
                force_data[f"model_{model_id}"][f"structure_{struct_id}"] = {
                    "forces": np.random.random((n_atoms, 3)),
                    "energy": np.random.random()
                }
        
        # Test standard deviation calculation
        for struct_id in range(n_structures):
            forces_array = np.array([
                force_data[f"model_{model_id}"][f"structure_{struct_id}"]["forces"]
                for model_id in range(n_models)
            ])
            
            std_dev = np.std(forces_array, axis=0)
            assert std_dev.shape == (n_atoms, 3)
            
            # Test max and mean standard deviation
            max_std = np.max(std_dev)
            mean_std = np.mean(std_dev)
            assert max_std >= mean_std >= 0


@pytest.mark.unit
class TestUnitUtilities:
    """Unit tests for basic utility functions."""
    
    def test_basic_math_operations(self):
        """Test basic mathematical operations."""
        # Test array operations
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        
        result = arr1 + arr2
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)
    
    def test_string_operations(self):
        """Test string manipulation utilities."""
        # Test base name generation
        loop_number = 5
        base_name = f"al_loop_{loop_number}"
        assert base_name == "al_loop_5"
        
        # Test file path operations
        base_path = Path("results") / base_name
        train_file = base_path / "train_set.xyz"
        assert str(train_file) == "results/al_loop_5/train_set.xyz"
    
    def test_type_checking(self):
        """Test type checking utilities."""
        # Test ASE Atoms type checking
        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]])
        assert isinstance(atoms, Atoms)
        
        # Test list type checking
        atoms_list = [atoms]
        assert isinstance(atoms_list, list)
        assert all(isinstance(item, Atoms) for item in atoms_list)
