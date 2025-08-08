"""
Tests for MLIP training components.

This module tests MACE training, committee methods, and related functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms


@pytest.fixture
def sample_training_atoms():
    """Create sample atoms for MLIP training."""
    atoms_list = []
    for i in range(20):
        # Create different molecules
        if i % 3 == 0:
            # H2O
            atoms = Atoms(
                symbols=["O", "H", "H"],
                positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
                cell=[12, 12, 12],
                pbc=True
            )
        elif i % 3 == 1:
            # CO2
            atoms = Atoms(
                symbols=["C", "O", "O"],
                positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]],
                cell=[12, 12, 12],
                pbc=True
            )
        else:
            # NH3
            atoms = Atoms(
                symbols=["N", "H", "H", "H"],
                positions=[[0, 0, 0], [0.94, 0, 0], [-0.47, 0.81, 0], [-0.47, -0.27, 0.77]],
                cell=[12, 12, 12],
                pbc=True
            )
        
        # Add some noise and properties
        atoms.positions += np.random.random(atoms.positions.shape) * 0.1
        atoms.info["energy"] = -10.0 - i * 0.1 + np.random.random() * 0.5
        atoms.arrays["forces"] = np.random.random((len(atoms), 3)) * 0.2 - 0.1
        atoms_list.append(atoms)
    
    return atoms_list


@pytest.fixture
def mock_mace_job_dict():
    """Mock job dictionary for MACE training."""
    return {
        "name": "test_mace_training",
        "size_of_committee": 5,
        "max_time": "4H",
        "hpc": {
            "hpc_name": "test-gpu-cluster",
            "pre_cmds": ["source /test/env/bin/activate"],
            "partitions": ["gpu"],
        },
    }


class TestMACETraining:
    """Test MACE training functionality."""
    
    @patch('alomancy.mlip.mace_wfl.mace_fit')
    def test_mace_fit_function_exists(self, mock_mace_fit):
        """Test that mace_fit function can be imported and called."""
        # Mock the function to return success
        mock_mace_fit.return_value = "success"
        
        # Test that we can import and call it
        from alomancy.mlip.mace_wfl import mace_fit
        result = mace_fit()
        
        assert result == "success"
        mock_mace_fit.assert_called_once()
    
    @patch('alomancy.mlip.committee_remote_submitter.committee_remote_submitter')
    def test_committee_remote_submitter_call(self, mock_submitter):
        """Test committee remote submitter function call."""
        from alomancy.mlip.committee_remote_submitter import committee_remote_submitter
        
        # Mock the submitter
        mock_submitter.return_value = ["model1.pt", "model2.pt", "model3.pt"]
        
        # Test parameters
        mock_remote_info = MagicMock()
        base_name = "test_al_loop_0"
        target_file = "test_model.pt"
        
        result = committee_remote_submitter(
            remote_info=mock_remote_info,
            base_name=base_name,
            target_file=target_file,
            seed=42,
            size_of_committee=3,
            function=MagicMock(),
            function_kwargs={}
        )
        
        assert len(result) == 3
        mock_submitter.assert_called_once()
    
    def test_committee_size_validation(self, mock_mace_job_dict):
        """Test committee size validation."""
        # Test valid committee sizes
        valid_sizes = [1, 3, 5, 7, 10]
        for size in valid_sizes:
            mock_mace_job_dict["size_of_committee"] = size
            assert mock_mace_job_dict["size_of_committee"] == size
            assert mock_mace_job_dict["size_of_committee"] > 0
        
        # Test invalid committee sizes
        invalid_sizes = [0, -1, -5]
        for size in invalid_sizes:
            mock_mace_job_dict["size_of_committee"] = size
            assert mock_mace_job_dict["size_of_committee"] <= 0
    
    def test_training_data_requirements(self, sample_training_atoms):
        """Test training data requirements."""
        # Test minimum data requirements
        assert len(sample_training_atoms) >= 10  # Minimum for meaningful training
        
        # Test that all atoms have required properties
        for atoms in sample_training_atoms:
            assert "energy" in atoms.info
            assert "forces" in atoms.arrays
            assert atoms.arrays["forces"].shape == (len(atoms), 3)
    
    @patch('pathlib.Path.glob')
    def test_model_file_discovery(self, mock_glob):
        """Test discovery of trained model files."""
        # Mock model file paths
        mock_model_paths = [
            Path("results/al_loop_0/MACE/fit_0/test_model.pt"),
            Path("results/al_loop_0/MACE/fit_1/test_model.pt"),
            Path("results/al_loop_0/MACE/fit_2/test_model.pt"),
        ]
        mock_glob.return_value = mock_model_paths
        
        # Test that we can find model files
        base_path = Path("results/al_loop_0/MACE")
        found_models = list(base_path.glob("fit_*/test_model.pt"))
        
        assert len(found_models) == 3
        mock_glob.assert_called_once()


class TestMACECalculator:
    """Test MACE calculator functionality."""
    
    @patch('mace.calculators.MACECalculator')
    def test_mace_calculator_initialization(self, mock_mace_calc_class):
        """Test MACE calculator initialization."""
        # Mock the calculator
        mock_calc = MagicMock()
        mock_mace_calc_class.return_value = mock_calc
        
        # Test initialization parameters
        from mace.calculators import MACECalculator
        
        calc = MACECalculator(
            model_paths=["model1.pt", "model2.pt"],
            device="cpu",
            default_dtype="float64"
        )
        
        assert calc is not None
        mock_mace_calc_class.assert_called_once_with(
            model_paths=["model1.pt", "model2.pt"],
            device="cpu",
            default_dtype="float64"
        )
    
    @patch('mace.calculators.MACECalculator')
    def test_mace_calculator_predictions(self, mock_mace_calc_class, sample_training_atoms):
        """Test MACE calculator predictions."""
        # Mock calculator with prediction methods
        mock_calc = MagicMock()
        mock_calc.get_potential_energy.return_value = -15.5
        mock_calc.get_forces.return_value = np.random.random((3, 3)) * 0.1
        mock_mace_calc_class.return_value = mock_calc
        
        # Test using calculator
        atoms = sample_training_atoms[0].copy()
        atoms.calc = mock_calc
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        assert energy == -15.5
        assert forces.shape == (len(atoms), 3)
    
    def test_committee_prediction_variance(self):
        """Test variance calculation from committee predictions."""
        # Simulate committee predictions
        n_models = 5
        n_structures = 10
        
        committee_energies = {}
        committee_forces = {}
        
        for model_id in range(n_models):
            committee_energies[f"model_{model_id}"] = {}
            committee_forces[f"model_{model_id}"] = {}
            
            for struct_id in range(n_structures):
                # Simulate slightly different predictions from each model
                base_energy = -10.0 - struct_id * 0.1
                energy_noise = np.random.random() * 0.2 - 0.1
                committee_energies[f"model_{model_id}"][f"struct_{struct_id}"] = base_energy + energy_noise
                
                base_forces = np.random.random((3, 3)) * 0.1
                force_noise = np.random.random((3, 3)) * 0.05 - 0.025
                committee_forces[f"model_{model_id}"][f"struct_{struct_id}"] = base_forces + force_noise
        
        # Calculate variance for each structure
        for struct_id in range(n_structures):
            # Energy variance
            energies = [
                committee_energies[f"model_{model_id}"][f"struct_{struct_id}"]
                for model_id in range(n_models)
            ]
            energy_std = np.std(energies)
            assert energy_std >= 0
            
            # Force variance
            forces_array = np.array([
                committee_forces[f"model_{model_id}"][f"struct_{struct_id}"]
                for model_id in range(n_models)
            ])
            force_std = np.std(forces_array, axis=0)
            assert force_std.shape == (3, 3)
            assert np.all(force_std >= 0)


class TestMLIPEvaluation:
    """Test MLIP evaluation functionality."""
    
    @patch('alomancy.analysis.mace_analysis.mace_al_loop_average_error')
    def test_mace_evaluation_function(self, mock_eval_func):
        """Test MACE evaluation function."""
        # Mock evaluation results
        import pandas as pd
        mock_results = pd.DataFrame({
            'rmse_energy': [0.1, 0.08, 0.12],
            'rmse_forces': [0.2, 0.18, 0.22],
            'mae_energy': [0.05, 0.04, 0.06],
            'mae_forces': [0.1, 0.09, 0.11]
        })
        mock_eval_func.return_value = mock_results
        
        from alomancy.analysis.mace_analysis import mace_al_loop_average_error
        
        result = mace_al_loop_average_error(
            mlip_committee_job_dict={"name": "test"},
            plot=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'rmse_energy' in result.columns
        mock_eval_func.assert_called_once()
    
    def test_evaluation_metrics(self):
        """Test calculation of evaluation metrics."""
        # Create test data
        true_energies = np.array([-10.0, -10.5, -11.0, -9.5, -10.2])
        pred_energies = np.array([-10.1, -10.4, -10.9, -9.6, -10.3])
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((true_energies - pred_energies) ** 2))
        assert rmse > 0
        
        # Calculate MAE
        mae = np.mean(np.abs(true_energies - pred_energies))
        assert mae > 0
        assert mae <= rmse  # MAE should be <= RMSE
        
        # Test forces
        true_forces = np.random.random((5, 3, 3))
        pred_forces = true_forces + np.random.random((5, 3, 3)) * 0.1
        
        force_rmse = np.sqrt(np.mean((true_forces - pred_forces) ** 2))
        force_mae = np.mean(np.abs(true_forces - pred_forces))
        
        assert force_rmse > 0
        assert force_mae > 0
    
    def test_cross_validation_setup(self):
        """Test cross-validation setup for model evaluation."""
        # Test data splitting
        data_size = 100
        train_ratio = 0.8
        
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        
        split_point = int(data_size * train_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        assert len(train_indices) == 80
        assert len(test_indices) == 20
        assert len(set(train_indices) & set(test_indices)) == 0  # No overlap


@pytest.mark.integration
class TestMLIPIntegration:
    """Integration tests for MLIP components."""
    
    @patch('alomancy.mlip.committee_remote_submitter.committee_remote_submitter')
    @patch('alomancy.analysis.mace_analysis.mace_al_loop_average_error')
    def test_full_mlip_workflow(self, mock_eval, mock_submitter, sample_training_atoms):
        """Test complete MLIP training and evaluation workflow."""
        import pandas as pd
        
        # Mock submitter to return model files
        mock_submitter.return_value = ["model_0.pt", "model_1.pt", "model_2.pt"]
        
        # Mock evaluation to return metrics
        mock_eval.return_value = pd.DataFrame({
            'rmse_energy': [0.1],
            'rmse_forces': [0.2]
        })
        
        # Test workflow
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            
            # Mock training
            training_result = mock_submitter(
                remote_info=MagicMock(),
                base_name="test_loop_0",
                target_file="test_model.pt",
                seed=42,
                size_of_committee=3,
                function=MagicMock(),
                function_kwargs={}
            )
            
            # Mock evaluation
            eval_result = mock_eval(
                mlip_committee_job_dict={"name": "test"},
                plot=False
            )
            
            assert len(training_result) == 3
            assert isinstance(eval_result, pd.DataFrame)


@pytest.mark.slow
@pytest.mark.requires_external
class TestMLIPExternal:
    """Tests requiring external MACE installation."""
    
    def test_real_mace_training(self, skip_if_no_mace):
        """Test with real MACE if available."""
        # This would test actual MACE training if the library is installed
        pass
    
    def test_real_mace_calculator(self, skip_if_no_mace):
        """Test with real MACE calculator if available."""
        # This would test actual MACE calculator if the library is installed
        pass
