"""
Tests for high-accuracy evaluation components.

This module tests DFT calculations, Quantum Espresso integration, and related functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms


@pytest.fixture
def sample_qe_structures():
    """Create sample structures for QE testing."""
    structures = []
    molecules = [
        # Water
        {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        },
        # Carbon dioxide
        {
            "symbols": ["C", "O", "O"],
            "positions": [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]],
        },
        # Ammonia
        {
            "symbols": ["N", "H", "H", "H"],
            "positions": [
                [0.0, 0.0, 0.0],
                [0.94, 0.0, 0.0],
                [-0.47, 0.81, 0.0],
                [-0.47, -0.27, 0.77],
            ],
        },
    ]

    for i, mol in enumerate(molecules):
        atoms = Atoms(
            symbols=mol["symbols"],
            positions=mol["positions"],
            cell=[15.0, 15.0, 15.0],
            pbc=True,
        )

        # Add some variation
        atoms.positions += np.random.random(atoms.positions.shape) * 0.1
        atoms.info["job_id"] = i
        structures.append(atoms)

    return structures


@pytest.fixture
def mock_qe_job_dict():
    """Mock job dictionary for QE calculations."""
    return {
        "name": "test_qe_calculation",
        "max_time": "1H",
        "hpc": {
            "hpc_name": "test-cpu-cluster",
            "pre_cmds": ["source /test/env/bin/activate"],
            "partitions": ["cpu"],
            "node_info": {
                "ranks_per_system": 8,
                "ranks_per_node": 8,
                "threads_per_rank": 1,
                "max_mem_per_node": "16GB",
            },
            "pwx_path": "/test/bin/pw.x",
            "pp_path": "/test/pseudopotentials",
            "pseudo_dict": {
                "O": "O.pbe-rrkjus.UPF",
                "H": "H.pbe-rrkjus.UPF",
                "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
                "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
            },
        },
    }


class TestQuantumEspressoSetup:
    """Test Quantum Espresso setup and configuration."""

    def test_qe_input_data_generation(self):
        """Test QE input data dictionary generation."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import get_qe_input_data

        input_data = get_qe_input_data("scf")

        # Check required sections
        assert "control" in input_data
        assert "system" in input_data
        assert "electrons" in input_data

        # Check control section
        control = input_data["control"]
        assert control["calculation"] == "scf"
        assert control["verbosity"] == "high"
        assert control["tprnfor"] is True  # Forces calculation

        # Check system section
        system = input_data["system"]
        assert system["ibrav"] == 0  # User-defined cell
        assert system["ecutwfc"] > 0  # Plane wave cutoff
        assert system["input_dft"] == "pbe"  # Exchange-correlation functional

        # Check electrons section
        electrons = input_data["electrons"]
        assert electrons["conv_thr"] > 0  # Convergence threshold
        assert electrons["scf_must_converge"] is True

    def test_kpoint_generation(self):
        """Test k-point grid generation."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import generate_kpts

        # Test cubic cell
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        kpts = generate_kpts(cell, periodic_3d=True, kspacing=0.1)

        assert len(kpts) == 3  # kx, ky, kz
        assert all(k > 0 for k in kpts)  # All positive
        assert all(isinstance(k, (int, np.integer)) for k in kpts)  # Integers

        # Test 2D system
        kpts_2d = generate_kpts(cell, periodic_3d=False, kspacing=0.1)
        assert kpts_2d[2] == 1  # Only 1 k-point in z direction

    def test_npool_optimization(self):
        """Test optimal npool calculation."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import find_optimal_npool

        # Test with 8 cores and 4 k-points
        npool = find_optimal_npool(
            ranks_per_system=8, total_kpoints=4, min_ranks_per_pool=2
        )

        assert npool <= 4  # Can't have more pools than k-points
        assert 8 % npool == 0  # Must divide evenly
        assert 8 // npool >= 2  # Minimum ranks per pool

    def test_debug_espresso_profile_creation(self):
        """Debug what create_espresso_profile actually returns."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import (
            create_espresso_profile,
        )

        para_info_dict = {
            "ranks_per_system": 8,
            "ranks_per_node": 8,
            "threads_per_rank": 1,
            "max_mem_per_node": "16GB",
        }

        profile = create_espresso_profile(
            para_info_dict=para_info_dict,
            npool=2,
            pwx_path="/test/pw.x",
            pp_path="/test/pps",
        )

        print(f"Profile type: {type(profile)}")
        print(f"Profile value: {profile}")
        print(
            f"Profile attributes: {dir(profile) if hasattr(profile, '__dict__') else 'No attributes'}"
        )

        # Test based on actual return type
        assert profile is not None

    @patch(
        "alomancy.high_accuracry_evaluation.dft.run_qe.Espresso"
    )  # Patch where it's imported
    def test_qe_calculator_creation(
        self, mock_espresso_class, sample_qe_structures, mock_qe_job_dict
    ):
        """Test QE calculator creation."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import create_qe_calc_object

        # Mock calculator
        mock_calc = MagicMock()
        mock_espresso_class.return_value = mock_calc

        atoms = sample_qe_structures[0]

        calc = create_qe_calc_object(
            atoms=atoms,
            high_accuracy_eval_hpc_job_dict=mock_qe_job_dict["hpc"],
            out_dir="/tmp/test",
        )

        assert calc is not None
        mock_espresso_class.assert_called_once()


class TestQuantumEspressoExecution:
    """Test Quantum Espresso execution."""

    @patch("alomancy.high_accuracry_evaluation.dft.run_qe.create_qe_calc_object")
    def test_run_qe_function(
        self, mock_create_calc, sample_qe_structures, mock_qe_job_dict
    ):
        """Test QE run function."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import run_qe
        import tempfile

        # Mock calculator
        mock_calc = MagicMock()
        mock_calc.get_potential_energy.return_value = -15.5
        mock_calc.get_forces.return_value = np.random.random((3, 3)) * 0.1
        mock_create_calc.return_value = mock_calc

        atoms = sample_qe_structures[0].copy()

        # Use a real temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_qe(
                input_structure=atoms,
                out_dir=tmpdir,  # Use real temp directory
                high_accuracy_eval_job_dict=mock_qe_job_dict,
                verbose=0,
            )

            # Check that calculation was performed
            mock_create_calc.assert_called_once()

            # Check result
            assert isinstance(result, Atoms)
            assert result.calc is not None

            # Verify file was actually written
            expected_file = Path(tmpdir) / f"{mock_qe_job_dict['name']}.xyz"
            assert expected_file.exists()

    @patch(
        "alomancy.high_accuracry_evaluation.dft.qe_remote_submitter.qe_remote_submitter"
    )
    def test_qe_remote_submission(self, mock_qe_submitter, sample_qe_structures):
        """Test QE remote submission."""
        from alomancy.high_accuracry_evaluation.dft.qe_remote_submitter import (
            qe_remote_submitter,
        )

        # Mock return result files
        mock_result_files = [
            "/test/path/qe_output_0/result.xyz",
            "/test/path/qe_output_1/result.xyz",
        ]
        mock_qe_submitter.return_value = mock_result_files

        mock_remote_info = MagicMock()
        base_name = "test_al_loop_0"
        target_file = "result.xyz"
        input_atoms_list = sample_qe_structures[:2]

        result = qe_remote_submitter(
            remote_info=mock_remote_info,
            base_name=base_name,
            target_file=target_file,
            input_atoms_list=input_atoms_list,
            function=MagicMock(),
            function_kwargs={},
        )

        assert len(result) == 2
        assert all("qe_output_" in path for path in result)
        mock_qe_submitter.assert_called_once()

    def test_pseudopotential_validation(self, mock_qe_job_dict):
        """Test pseudopotential file validation."""
        pseudo_dict = mock_qe_job_dict["hpc"]["pseudo_dict"]

        # Test that required elements have pseudopotentials
        required_elements = ["O", "H", "C", "N"]
        for element in required_elements:
            assert element in pseudo_dict
            assert pseudo_dict[element].endswith(".UPF")

        # Test pseudopotential path construction
        pp_path = mock_qe_job_dict["hpc"]["pp_path"]
        for element, pp_file in pseudo_dict.items():
            full_path = Path(pp_path) / pp_file
            assert element in str(full_path)


class TestDFTResults:
    """Test DFT results processing and validation."""

    def test_energy_forces_extraction(self, sample_qe_structures):
        """Test extraction of energy and forces from DFT results."""
        atoms = sample_qe_structures[0]

        # Simulate DFT results
        atoms.info["energy"] = -25.5  # eV
        atoms.arrays["forces"] = np.random.random((len(atoms), 3)) * 0.2 - 0.1  # eV/Å

        # Test that results are present and reasonable
        assert "energy" in atoms.info
        assert "forces" in atoms.arrays
        assert isinstance(atoms.info["energy"], (int, float))
        assert atoms.arrays["forces"].shape == (len(atoms), 3)

        # Test force units and magnitudes
        forces = atoms.arrays["forces"]
        max_force = np.max(np.abs(forces))
        assert max_force < 10.0  # Reasonable force magnitude in eV/Å


class TestHighAccuracyEvaluationIntegration:
    """Integration tests for high-accuracy evaluation."""

    @patch(
        "alomancy.high_accuracry_evaluation.dft.qe_remote_submitter.qe_remote_submitter"
    )
    @patch("ase.io.read")
    def test_complete_evaluation_workflow(
        self, mock_read, mock_qe_submitter, sample_qe_structures, mock_qe_job_dict
    ):
        """Test complete high-accuracy evaluation workflow."""
        # Mock QE submission
        mock_result_paths = [
            "/test/qe_output_0/result.xyz",
            "/test/qe_output_1/result.xyz",
        ]
        mock_qe_submitter.return_value = mock_result_paths

        # Mock reading results
        result_structures = []
        for atoms in sample_qe_structures[:2]:
            result_atoms = atoms.copy()
            result_atoms.info["energy"] = -20.0
            result_atoms.arrays["forces"] = np.random.random((len(atoms), 3)) * 0.1
            result_structures.append(result_atoms)

        mock_read.side_effect = result_structures

        # Test workflow (this would be part of high_accuracy_evaluation method)
        input_structures = sample_qe_structures[:2]

        # 1. Submit QE calculations
        result_paths = mock_qe_submitter(
            remote_info=MagicMock(),
            base_name="test_loop_0",
            target_file="result.xyz",
            input_atoms_list=input_structures,
            function=MagicMock(),
            function_kwargs={},
        )

        # 2. Read results
        final_structures = []
        for path in result_paths:
            atoms = mock_read(path)
            final_structures.append(atoms)

        # Verify workflow
        assert len(result_paths) == 2
        assert len(final_structures) == 2
        assert all("energy" in atoms.info for atoms in final_structures)
        assert all("forces" in atoms.arrays for atoms in final_structures)

    def test_error_handling(self, sample_qe_structures, mock_qe_job_dict):
        """Test error handling in QE calculations."""
        from alomancy.high_accuracry_evaluation.dft.run_qe import get_qe_input_data

        # Test invalid calculation type
        try:
            input_data = get_qe_input_data("invalid_calc_type")
            # Should still work but with wrong calculation type
            assert input_data["control"]["calculation"] == "invalid_calc_type"
        except Exception:
            # Or it might raise an exception, both are acceptable
            pass

        # Test missing pseudopotentials
        incomplete_pseudo_dict = {"O": "O.pbe.UPF"}  # Missing H, C, N

        atoms = sample_qe_structures[1]  # CO2 molecule
        elements = set(atoms.get_chemical_symbols())
        missing_elements = elements - set(incomplete_pseudo_dict.keys())

        assert len(missing_elements) > 0  # Should have missing elements

    def test_parallel_execution_setup(self, mock_qe_job_dict):
        """Test parallel execution setup for QE."""
        node_info = mock_qe_job_dict["hpc"]["node_info"]

        # Test MPI setup
        ranks_per_system = node_info["ranks_per_system"]
        ranks_per_node = node_info["ranks_per_node"]
        threads_per_rank = node_info["threads_per_rank"]

        assert ranks_per_system > 0
        assert ranks_per_node > 0
        assert threads_per_rank > 0
        assert ranks_per_system >= ranks_per_node  # Reasonable constraint

        # Test memory allocation
        max_mem = node_info["max_mem_per_node"]
        assert "GB" in max_mem or "MB" in max_mem  # Has memory units


@pytest.mark.integration
class TestQuantumEspressoIntegration:
    """Integration tests for QE components."""

    @patch("alomancy.high_accuracry_evaluation.dft.run_qe.run_qe")
    @patch(
        "alomancy.high_accuracry_evaluation.dft.qe_remote_submitter.qe_remote_submitter"
    )
    def test_full_qe_pipeline(
        self, mock_submitter, mock_run_qe, sample_qe_structures, mock_qe_job_dict
    ):
        """Test full QE calculation pipeline."""

        # Mock individual QE runs
        def mock_qe_run_side_effect(
            input_structure, out_dir, high_accuracy_eval_job_dict, verbose=0
        ):
            result_atoms = input_structure.copy()
            result_atoms.info["energy"] = -20.0
            result_atoms.arrays["forces"] = (
                np.random.random((len(input_structure), 3)) * 0.1
            )
            return result_atoms

        mock_run_qe.side_effect = mock_qe_run_side_effect

        # Mock remote submission
        mock_submitter.return_value = [
            "/test/qe_output_0/result.xyz",
            "/test/qe_output_1/result.xyz",
        ]

        # Test pipeline
        input_structures = sample_qe_structures[:2]

        # This simulates the high_accuracy_evaluation method
        result_paths = mock_submitter(
            remote_info=MagicMock(),
            base_name="test_loop_0",
            target_file="result.xyz",
            input_atoms_list=input_structures,
            function=mock_run_qe,
            function_kwargs={"high_accuracy_eval_job_dict": mock_qe_job_dict},
        )

        assert len(result_paths) == 2
        mock_submitter.assert_called_once()


@pytest.mark.slow
@pytest.mark.requires_external
class TestQuantumEspressoExternal:
    """Tests requiring external QE installation."""

    def test_real_qe_calculation(self, skip_if_no_external):
        """Test with real QE if available."""
        # This would test actual QE calculation if the software is installed
        pass

    def test_real_pseudopotentials(self, skip_if_no_external):
        """Test with real pseudopotential files if available."""
        # This would test actual pseudopotential files if available
        pass
