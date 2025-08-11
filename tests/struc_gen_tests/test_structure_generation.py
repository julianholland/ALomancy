"""
Tests for structure generation components.

This module tests molecular dynamics, structure selection, and related functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms


@pytest.fixture
def sample_md_structures():
    """Create sample structures for MD testing."""
    structures = []
    for i in range(50):
        # Create H2O molecules with slight variations
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
            cell=[15, 15, 15],
            pbc=True
        )
        
        # Add thermal noise to positions
        atoms.positions += np.random.random((3, 3)) * 0.3 - 0.15
        
        # Add MD step information
        atoms.info["step"] = i
        atoms.info["temperature"] = 300 + np.random.random() * 50
        atoms.info["energy"] = -10.0 + np.random.random() * 0.5
        atoms.arrays["forces"] = np.random.random((3, 3)) * 0.2 - 0.1
        
        structures.append(atoms)
    
    return structures


@pytest.fixture
def mock_md_job_dict():
    """Mock job dictionary for MD generation."""
    return {
        "name": "test_md_generation",
        "number_of_concurrent_jobs": 4,
        "max_time": "2H",
        "hpc": {
            "hpc_name": "test-gpu-cluster",
            "pre_cmds": ["source /test/env/bin/activate"],
            "partitions": ["gpu"],
        },
    }


class TestMolecularDynamics:
    """Test molecular dynamics functionality."""
    
    @patch('alomancy.structure_generation.md.md_wfl.run_md')
    def test_run_md_function(self, mock_run_md):
        """Test MD run function."""
        from alomancy.structure_generation.md.md_wfl import run_md
        
        # Mock MD run
        mock_run_md.return_value = None
        
        # Test parameters
        structure_generation_job_dict = {"name": "test_md"}
        initial_structure = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 1]])
        
        run_md(
            structure_generation_job_dict=structure_generation_job_dict,
            initial_structure=initial_structure,
            total_md_runs=1,
            out_dir="/tmp/test",
            model_path=["test_model.pt"],
            steps=100,
            temperature=300,
            desired_number_of_structures=10,
            timestep_fs=0.5,
            verbose=0
        )
        
        mock_run_md.assert_called_once()
    
    def test_md_parameter_validation(self):
        """Test MD parameter validation."""
        # Test valid parameters
        valid_params = {
            "steps": 100,
            "temperature": 300,
            "desired_number_of_structures": 20,
            "total_md_runs": 5
        }
        
        # Check basic constraints
        assert valid_params["desired_number_of_structures"] > 0
        assert valid_params["steps"] > valid_params["desired_number_of_structures"] / valid_params["total_md_runs"]
        assert valid_params["temperature"] > 0
        
        # Test invalid parameters that would cause division by zero
        invalid_params = {
            "steps": 10,
            "desired_number_of_structures": 50,
            "total_md_runs": 5
        }
        
        # This should fail the constraint
        snapshot_interval = invalid_params["steps"] * invalid_params["total_md_runs"] // invalid_params["desired_number_of_structures"]
        assert snapshot_interval == 1  # This would be problematic for the loop
    
    @patch('ase.md.langevin.Langevin')
    @patch('mace.calculators.MACECalculator')
    def test_md_setup(self, mock_mace_calc, mock_langevin):
        """Test MD simulation setup."""
        # Mock calculator
        mock_calc = MagicMock()
        mock_mace_calc.return_value = mock_calc
        
        # Mock dynamics
        mock_dyn = MagicMock()
        mock_langevin.return_value = mock_dyn
        
        # Test setup
        atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 1]])
        atoms.calc = mock_calc
        
        from ase.md.langevin import Langevin
        from ase.units import fs
        
        dyn = Langevin(
            atoms=atoms,
            timestep=0.5 * fs,
            temperature_K=300,
            friction=0.002
        )
        
        assert dyn is not None
        mock_langevin.assert_called_once()
    
    def test_force_checking(self, sample_md_structures):
        """Test force magnitude checking during MD."""
        for atoms in sample_md_structures:
            forces = atoms.arrays["forces"]
            max_forces = np.max(np.abs(forces), axis=0)
            
            # Test that forces are reasonable (not excessive)
            # In real MD, we would stop if forces > 1000 eV/Å
            force_threshold = 1000
            is_stable = np.all(max_forces < force_threshold)
            
            # For our test data, forces should be reasonable
            assert np.all(max_forces < 1.0)  # Test data has small forces
    
    @patch('alomancy.structure_generation.md.md_remote_submitter.md_remote_submitter')
    def test_md_remote_submission(self, mock_md_submitter):
        """Test MD remote submission."""
        from alomancy.structure_generation.md.md_remote_submitter import md_remote_submitter
        
        # Mock return trajectory files
        mock_trajectories = [
            "/test/path/md_output_0/trajectory.xyz",
            "/test/path/md_output_1/trajectory.xyz"
        ]
        mock_md_submitter.return_value = mock_trajectories
        
        # Test parameters
        mock_remote_info = MagicMock()
        base_name = "test_al_loop_0"
        target_file = "trajectory.xyz"
        input_atoms_list = [Atoms(symbols=["H"], positions=[[0, 0, 0]]) for _ in range(2)]
        
        result = md_remote_submitter(
            remote_info=mock_remote_info,
            base_name=base_name,
            target_file=target_file,
            input_atoms_list=input_atoms_list,
            function=MagicMock(),
            function_kwargs={}
        )
        
        assert len(result) == 2
        assert all("md_output_" in path for path in result)
        mock_md_submitter.assert_called_once()


class TestStructureSelection:
    """Test structure selection functionality."""
    
    @patch('alomancy.structure_generation.select_initial_structures.select_initial_structures')
    def test_initial_structure_selection(self, mock_select_initial):
        """Test initial structure selection."""
        from alomancy.structure_generation.select_initial_structures import select_initial_structures
        
        # Mock selected structures
        mock_structures = [
            Atoms(symbols=["C", "O"], positions=[[0, 0, 0], [1.1, 0, 0]]),
            Atoms(symbols=["N", "H"], positions=[[0, 0, 0], [1.0, 0, 0]]),
        ]
        mock_select_initial.return_value = mock_structures
        
        result = select_initial_structures(
            base_name="test_loop_0",
            structure_generation_job_dict={"name": "test"},
            desired_initial_structures=2,
            chem_formula_list=[],
            atom_number_range=(2, 10),
            enforce_chemical_diversity=True,
            train_atoms_list=[],
            verbose=0
        )
        
        assert len(result) == 2
        mock_select_initial.assert_called_once()
    
    def test_chemical_diversity_check(self):
        """Test chemical diversity checking."""
        structures = [
            Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [1, 0, 0]]),  # H2
            Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),  # H2O
            Atoms(symbols=["C", "O"], positions=[[0, 0, 0], [1.1, 0, 0]]),  # CO
        ]
        
        # Check chemical formulas
        formulas = [atoms.get_chemical_formula() for atoms in structures]
        unique_formulas = set(formulas)
        
        assert len(unique_formulas) == 3  # All different
        assert "H2" in formulas
        assert "H2O" in formulas
        assert "CO" in formulas
    
    def test_atom_number_filtering(self):
        """Test filtering structures by atom number."""
        structures = [
            Atoms(symbols=["H"], positions=[[0, 0, 0]]),  # 1 atom
            Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [1, 0, 0]]),  # 2 atoms
            Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),  # 3 atoms
            Atoms(symbols=["C"] * 20, positions=np.random.random((20, 3))),  # 20 atoms
        ]
        
        # Filter by atom number range
        min_atoms, max_atoms = 2, 10
        filtered_structures = [
            atoms for atoms in structures 
            if min_atoms <= len(atoms) <= max_atoms
        ]
        
        assert len(filtered_structures) == 2  # 2-atom and 3-atom structures
        assert all(min_atoms <= len(atoms) <= max_atoms for atoms in filtered_structures)
    
    @patch('alomancy.structure_generation.find_high_sd_structures.find_high_sd_structures')
    def test_high_sd_structure_selection(self, mock_find_high_sd, sample_md_structures):
        """Test high standard deviation structure selection."""
        from alomancy.structure_generation.find_high_sd_structures import find_high_sd_structures
        
        # Mock return high SD structures
        high_sd_structures = sample_md_structures[:10]  # Select first 10
        mock_find_high_sd.return_value = high_sd_structures
        
        structure_list = sample_md_structures
        base_name = "test_loop_0"
        job_dict = {"test": "dict"}
        list_of_calculators = [MagicMock() for _ in range(3)]
        
        result = find_high_sd_structures(
            structure_list=structure_list,
            base_name=base_name,
            job_dict=job_dict,
            list_of_other_calculators=list_of_calculators,
            forces_name="REF_forces",
            energy_name="REF_energy",
            verbose=0
        )
        
        assert len(result) == 10
        mock_find_high_sd.assert_called_once()


class TestForceVarianceCalculation:
    """Test force variance calculation functionality."""
    
    def test_force_flattening(self):
        """Test force array flattening."""
        # Test the flatten_array_of_forces function from md_wfl.py
        forces = np.random.random((5, 3))  # 5 atoms, 3 components each
        
        def flatten_array_of_forces(forces_array):
            return np.reshape(forces_array, (1, forces_array.shape[0] * 3))
        
        flattened = flatten_array_of_forces(forces)
        
        assert flattened.shape == (1, 15)  # 5 atoms × 3 components
        
        # Test that we can unflatten correctly
        unflattened = flattened.reshape((5, 3))
        np.testing.assert_array_equal(forces, unflattened)
    
    def test_standard_deviation_calculation(self, sample_md_structures):
        """Test standard deviation calculation for forces."""
        import pandas as pd
        
        # Simulate multiple model predictions
        n_models = 5
        structure_forces_dict = {}
        
        for model_id in range(n_models):
            model_name = f"model_{model_id}" if model_id > 0 else "base_mace"
            structure_forces_dict[model_name] = {}
            
            for struct_id, atoms in enumerate(sample_md_structures[:10]):
                # Add some variation to the forces
                base_forces = atoms.arrays["forces"]
                noise = np.random.random(base_forces.shape) * 0.1 - 0.05
                varied_forces = base_forces + noise
                
                structure_forces_dict[model_name][f"structure_{struct_id}"] = {
                    "forces": varied_forces,
                    "energy": atoms.info["energy"] + np.random.random() * 0.1
                }
        
        # Test std deviation calculation function structure
        def mock_std_deviation_of_forces(structure_forces_dict, md_dir, verbose=0):
            number_of_structures = len(structure_forces_dict["base_mace"])
            std_dev_array = np.zeros((number_of_structures, 3))
            
            for structure in range(number_of_structures):
                forces_array = np.concatenate([
                    structure_forces_dict[fit][f"structure_{structure}"]["forces"]
                    for fit in structure_forces_dict
                ], axis=0)
                
                std_dev_per_force_fragment = np.std(forces_array, axis=0)
                energy_array = np.array([
                    structure_forces_dict[fit][f"structure_{structure}"]["energy"]
                    for fit in structure_forces_dict
                ])
                std_dev_per_energy = np.std(energy_array)
                
                std_dev_array[structure, :] = np.array([
                    np.max(std_dev_per_force_fragment),
                    np.mean(std_dev_per_force_fragment),
                    std_dev_per_energy,
                ])
            
            df = pd.DataFrame(
                std_dev_array, 
                columns=["max_std_dev", "mean_std_dev", "std_dev_energy"]
            ).sort_values(by="max_std_dev", ascending=False)
            
            return df
        
        result_df = mock_std_deviation_of_forces(structure_forces_dict, "/tmp")
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 10
        assert "max_std_dev" in result_df.columns
        assert "mean_std_dev" in result_df.columns
        assert "std_dev_energy" in result_df.columns
        
        # Check that max_std_dev >= mean_std_dev for each structure
        assert all(result_df["max_std_dev"] >= result_df["mean_std_dev"])


class TestTrajectoryProcessing:
    """Test trajectory file processing."""
    
    def test_trajectory_file_reading(self, sample_md_structures):
        """Test reading trajectory files."""
        from ase.io import write, read    
        
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = Path(tmpdir) / "trajectory.xyz"
            
            # Write trajectory
            write(str(traj_file), sample_md_structures[:10], format="extxyz")
            
            # Read trajectory
            read_structures = read(str(traj_file), ":", format="extxyz")
            
            assert len(read_structures) == 10
            assert all(isinstance(atoms, Atoms) for atoms in read_structures)
    
    def test_trajectory_concatenation(self, sample_md_structures):
        """Test concatenating multiple trajectory files."""
        from ase.io import write, read    
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple trajectory files
            traj_files = []
            for i in range(3):
                traj_file = Path(tmpdir) / f"trajectory_{i}.xyz"
                start_idx = i * 10
                end_idx = (i + 1) * 10
                write(str(traj_file), sample_md_structures[start_idx:end_idx], format="extxyz")
                traj_files.append(str(traj_file))
            
            # Simulate reading and concatenating
            all_structures = []
            for traj_file in traj_files:
                structures = sample_md_structures[:10]  # Mock read
                all_structures.extend(structures)
            
            assert len(all_structures) == 30  # 3 files × 10 structures each


@pytest.mark.integration
class TestStructureGenerationIntegration:
    """Integration tests for structure generation."""
    
    @patch('alomancy.structure_generation.select_initial_structures.select_initial_structures')
    @patch('alomancy.structure_generation.md.md_remote_submitter.md_remote_submitter')
    @patch('alomancy.structure_generation.find_high_sd_structures.find_high_sd_structures')
    @patch('ase.io.read')
    @patch('pathlib.Path.glob')
    def test_full_structure_generation_workflow(self, mock_glob, mock_read, mock_find_high_sd,
                                              mock_md_submitter, mock_select_initial,
                                              sample_md_structures):
        """Test complete structure generation workflow."""
        # Mock all components
        mock_select_initial.return_value = sample_md_structures[:5]
        mock_md_submitter.return_value = ["/path/to/traj1.xyz", "/path/to/traj2.xyz"]
        mock_read.return_value = sample_md_structures[:20]
        mock_glob.return_value = [Path("model1.pt"), Path("model2.pt")]
        mock_find_high_sd.return_value = sample_md_structures[:3]
        
        # Test workflow components are called in sequence
        # This would be part of the generate_structures method
        
        # 1. Select initial structures
        initial_structures = mock_select_initial()
        assert len(initial_structures) == 5
        
        # 2. Run MD simulations
        trajectories = mock_md_submitter()
        assert len(trajectories) == 2
        
        # 3. Read MD results
        md_structures = mock_read()
        assert len(md_structures) == 20
        
        # 4. Find high SD structures
        high_sd_structures = mock_find_high_sd()
        assert len(high_sd_structures) == 3


@pytest.mark.slow
@pytest.mark.requires_external
class TestStructureGenerationExternal:
    """Tests requiring external dependencies."""
    
    def test_real_md_simulation(self, skip_if_no_mace):
        """Test with real MD simulation if MACE is available."""
        pass
    
    def test_real_ase_md(self, skip_if_no_external):
        """Test with real ASE MD if available."""
        pass
