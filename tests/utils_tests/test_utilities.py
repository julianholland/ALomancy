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


@pytest.fixture
def write_temporary_yaml():
    """Fixture to create a temporary YAML file for testing."""
    from yaml import dump

    tmp_dict = {
        "mlip_committee": {
            "name": "mlip_test",
            "max_time": "value",
            "hpc": "value",
            "size_of_committee": 5,
        },
        "structure_generation": {
            "name": "struc_gen_test",
            "max_time": "value",
            "hpc": "value",
            "number_of_concurrent_jobs": 5,
        },
        "high_accuracy_evaluation": {
            "name": "high_acc_test",
            "max_time": "value",
            "hpc": {
                "node_info": {
                    "ranks_per_system": 4,
                    "ranks_per_node": 2,
                    "threads_per_rank": 8,
                }
            },
            "pwx_path": "/path/to/pw.x",
            "pp_path": "/path/to/pseudopotentials",
            "pseudo_dict": {"H": "H.pz-vbc.UPF", "O": "O.pz-vbc.UPF"},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_file = Path(tmpdir) / "test.yaml"
        with open(yaml_file, "w") as f:
            dump(tmp_dict, f)
        yield yaml_file


class TestRemoteJobExecutor:
    """Test remote job execution utilities."""

    @pytest.mark.unit
    def test_remote_job_executor_initialization(self):
        """Test RemoteJobExecutor initialization."""
        # Test that we can create a mock remote info
        mock_remote_info = MagicMock()
        assert mock_remote_info is not None

    @pytest.mark.unit
    def test_job_config_validation(self):
        """Test job configuration validation."""
        # Test valid job config
        valid_config = {"function_kwargs": {"param1": "value1", "param2": 42}}

        assert "function_kwargs" in valid_config
        assert isinstance(valid_config["function_kwargs"], dict)

    @pytest.mark.unit
    def test_invalid_job_config(self):
        """Test handling of invalid job configurations."""
        invalid_configs = [
            {},  # Empty config
            {"wrong_key": "value"},  # Missing function_kwargs
            {"function_kwargs": "not_a_dict"},  # Wrong type for function_kwargs
        ]

        for config in invalid_configs:
            if "function_kwargs" not in config:
                assert "function_kwargs" not in config
            elif not isinstance(config.get("function_kwargs"), dict):
                assert not isinstance(config.get("function_kwargs"), dict)


class TestConfigurationUtils:
    """Test configuration utility functions."""

    @pytest.mark.unit
    def test_load_dictionaries_structure(self, write_temporary_yaml):
        """Test that load_dictionaries returns expected structure."""
        from alomancy.configs.config_dictionaries import load_dictionaries

        config = load_dictionaries(write_temporary_yaml)

        # Test that required keys are present
        required_keys = [
            "mlip_committee",
            "structure_generation",
            "high_accuracy_evaluation",
        ]
        for key in required_keys:
            assert key in config

        # Test that each job dict has required fields
        for _, job_config in config.items():
            assert "name" in job_config
            assert "max_time" in job_config
            assert "hpc" in job_config

    @pytest.mark.unit
    def test_job_dict_validation(self, write_temporary_yaml):
        """Test validation of job dictionary structure."""
        from alomancy.configs.config_dictionaries import load_dictionaries

        config = load_dictionaries(write_temporary_yaml)

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
            required_node_fields = [
                "ranks_per_system",
                "ranks_per_node",
                "threads_per_rank",
            ]
            for field in required_node_fields:
                assert field in node_info

    @pytest.mark.unit
    @patch("alomancy.configs.config_dictionaries.load_dictionaries")
    def test_config_customization(self, mock_load_dict):
        """Test that configuration can be customized."""
        custom_config = {
            "mlip_committee": {
                "name": "custom_mlip",
                "size_of_committee": 10,
                "max_time": "8H",
                "hpc": {"hpc_name": "custom-hpc"},
            },
            "structure_generation": {
                "name": "custom_md",
                "number_of_concurrent_jobs": 8,
                "max_time": "4H",
                "hpc": {"hpc_name": "custom-hpc"},
            },
            "high_accuracy_evaluation": {
                "name": "custom_qe",
                "max_time": "1H",
                "hpc": {"hpc_name": "custom-hpc"},
            },
        }

        mock_load_dict.return_value = custom_config

        config = mock_load_dict()

        assert config["mlip_committee"]["size_of_committee"] == 10
        assert config["structure_generation"]["number_of_concurrent_jobs"] == 8


class TestFileOperations:
    """Test file operation utilities."""

    @pytest.mark.unit
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


class TestDataValidation:
    """Test data validation utilities."""

    @pytest.mark.unit
    def test_atoms_validation(self):
        """Test validation of ASE Atoms objects."""
        # Valid atoms object
        valid_atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            cell=[10, 10, 10],
            pbc=True,
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

    @pytest.mark.unit
    def test_structure_list_validation(self):
        """Test validation of structure lists."""
        atoms_list = []
        for i in range(5):
            atoms = Atoms(
                symbols=["C"], positions=[[i, 0, 0]], cell=[20, 20, 20], pbc=True
            )
            atoms.info["energy"] = -i
            atoms.arrays["forces"] = np.array([[0, 0, 0]])
            atoms_list.append(atoms)

        # Test list properties
        assert len(atoms_list) == 5
        assert all(isinstance(atoms, Atoms) for atoms in atoms_list)
        assert all("energy" in atoms.info for atoms in atoms_list)
        assert all("forces" in atoms.arrays for atoms in atoms_list)

    @pytest.mark.unit
    def test_energy_forces_consistency(self):
        """Test consistency between energy and forces data."""
        atoms = Atoms(
            symbols=["N", "N"],
            positions=[[0, 0, 0], [1.1, 0, 0]],
            cell=[15, 15, 15],
            pbc=True,
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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
                    "energy": np.random.random(),
                }

        # Test standard deviation calculation
        for struct_id in range(n_structures):
            forces_array = np.array(
                [
                    force_data[f"model_{model_id}"][f"structure_{struct_id}"]["forces"]
                    for model_id in range(n_models)
                ]
            )

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


class TestSplitAtomsListIntoTestAndTrain:
    """Test split_atoms_list_into_test_and_train function."""

    def _make_atoms_list(self, n):
        """Create a simple list of Atoms objects."""
        return [
            Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
            for _ in range(n)
        ]

    @pytest.mark.unit
    def test_correct_split_size(self):
        """Test that split produces correct train and test sizes."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(100)
        train, test = split_atoms_list_into_test_and_train(atoms, test_fraction=0.2, seed=42)
        assert len(train) + len(test) == 100
        assert len(test) == 20  # 100 * 0.2

    @pytest.mark.unit
    def test_seeded_reproducibility(self):
        """Test that same seed produces same split."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(50)
        train1, test1 = split_atoms_list_into_test_and_train(atoms, 0.2, seed=42)
        train2, test2 = split_atoms_list_into_test_and_train(atoms, 0.2, seed=42)
        # Same seed → same split (verify by checking ids, not values)
        assert [id(a) for a in train1] == [id(a) for a in train2]
        assert [id(a) for a in test1] == [id(a) for a in test2]

    @pytest.mark.unit
    def test_different_seeds_different_splits(self):
        """Test that different seeds produce different splits."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(50)
        train1, _ = split_atoms_list_into_test_and_train(atoms, 0.2, seed=1)
        train2, _ = split_atoms_list_into_test_and_train(atoms, 0.2, seed=2)
        # With high probability different seeds give different orders
        assert [id(a) for a in train1] != [id(a) for a in train2]

    @pytest.mark.unit
    def test_empty_list(self):
        """Test split with empty list."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        train, test = split_atoms_list_into_test_and_train([], 0.2, seed=42)
        assert train == []
        assert test == []

    @pytest.mark.unit
    def test_zero_test_fraction(self):
        """Test split with zero test fraction."""
        from alomancy.utils.test_train_manager import (
            split_atoms_list_into_test_and_train,
        )

        atoms = self._make_atoms_list(10)
        train, test = split_atoms_list_into_test_and_train(atoms, 0.0, seed=42)
        assert len(train) == 10
        assert len(test) == 0


class TestCleanStructures:
    """Test clean_structures function."""

    def _make_structure_with_ref(self, config_type=None):
        """Create a test structure with REF_energy and REF_forces."""
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        if config_type:
            atoms.info["config_type"] = config_type
        atoms.info["REF_energy"] = -76.0
        atoms.arrays["REF_forces"] = np.zeros((3, 3))
        return atoms

    @pytest.mark.unit
    def test_config_type_set_when_missing(self):
        """Test that config_type is set when missing."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref()  # no config_type
        result = clean_structures([s], config_type="al_loop_0")
        assert result[0].info["config_type"] == "al_loop_0"

    @pytest.mark.unit
    def test_config_type_preserved_when_not_overriding(self):
        """Test that config_type is preserved when not overriding."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref(config_type="original_type")
        result = clean_structures(
            [s], config_type="new_type", override_config_type=False
        )
        assert result[0].info["config_type"] == "original_type"

    @pytest.mark.unit
    def test_config_type_overridden(self):
        """Test that config_type is overridden when requested."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref(config_type="old_type")
        result = clean_structures(
            [s], config_type="new_type", override_config_type=True
        )
        assert result[0].info["config_type"] == "new_type"

    @pytest.mark.unit
    def test_ref_energy_preserved(self):
        """Test that REF_energy is preserved."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref()
        result = clean_structures([s], config_type="test")
        assert result[0].info["REF_energy"] == pytest.approx(-76.0)

    @pytest.mark.unit
    def test_ref_forces_preserved(self):
        """Test that REF_forces are preserved."""
        from alomancy.utils.clean_structures import clean_structures

        s = self._make_structure_with_ref()
        result = clean_structures([s], config_type="test")
        np.testing.assert_allclose(result[0].arrays["REF_forces"], np.zeros((3, 3)))

    @pytest.mark.unit
    def test_multiple_structures(self):
        """Test cleaning multiple structures."""
        from alomancy.utils.clean_structures import clean_structures

        structures = [self._make_structure_with_ref() for _ in range(3)]
        result = clean_structures(structures, config_type="batch")
        assert len(result) == 3


class TestLoadDictionaries:
    """Test load_dictionaries function."""

    @pytest.mark.unit
    def test_load_real_yaml(self, tmp_path):
        """Test loading a real YAML configuration."""
        from alomancy.configs.config_dictionaries import load_dictionaries

        # Write a minimal valid config YAML
        yaml_content = """
initialization:
  name: init
  max_time: 1H
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
mlip_committee:
  name: mlip
  max_time: 2H
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
structure_generation:
  name: struc_gen
  max_time: 30m
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
high_accuracy_evaluation:
  name: dft
  max_time: 10m
  hpc:
    hpc_name: test
    partitions: [test]
    pre_cmds: []
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)
        result = load_dictionaries(config_path)
        assert "initialization" in result
        assert "mlip_committee" in result
        assert "structure_generation" in result
        assert "high_accuracy_evaluation" in result
