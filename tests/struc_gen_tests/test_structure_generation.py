"""
Tests for structure generation components.

This module tests molecular dynamics, structure selection, and related functionality.
"""

import tempfile
import warnings
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
            pbc=True,
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


# ============================================================================
# Tests for flatten_array_of_forces
# ============================================================================


@pytest.mark.unit
class TestFlattenArrayOfForces:
    """Test the flatten_array_of_forces function."""

    def test_shape_n3_to_1_3n(self):
        """Test that (N, 3) forces reshape to (1, N*3)."""
        from alomancy.structure_generation.find_high_sd_structures import (
            flatten_array_of_forces,
        )

        forces = np.ones((5, 3))
        result = flatten_array_of_forces(forces)
        assert result.shape == (1, 15)

    def test_single_atom(self):
        """Test flattening forces for a single atom."""
        from alomancy.structure_generation.find_high_sd_structures import (
            flatten_array_of_forces,
        )

        forces = np.array([[1.0, 2.0, 3.0]])
        result = flatten_array_of_forces(forces)
        assert result.shape == (1, 3)

    def test_values_preserved(self):
        """Test that values are preserved during flattening."""
        from alomancy.structure_generation.find_high_sd_structures import (
            flatten_array_of_forces,
        )

        forces = np.arange(6, dtype=float).reshape(2, 3)
        result = flatten_array_of_forces(forces)
        np.testing.assert_allclose(result.flatten(), forces.flatten())


# ============================================================================
# Tests for std_deviation_of_forces
# ============================================================================


@pytest.mark.unit
class TestStdDeviationOfForces:
    """Test the std_deviation_of_forces function."""

    @staticmethod
    def _flatten_forces(forces):
        """Helper to flatten forces for test data."""
        return np.reshape(forces, (1, forces.shape[0] * 3))

    def _build_forces_dict(self, n_structures=3, n_atoms=2, n_models=3, rng=None):
        """Build a structure_forces_dict with controlled random forces."""
        if rng is None:
            rng = np.random.default_rng(42)
        keys = ["base_mlip"] + [f"fit_{i}" for i in range(n_models - 1)]
        d = {}
        for key in keys:
            d[key] = {}
            for s in range(n_structures):
                forces = self._flatten_forces(rng.random((n_atoms, 3)))
                d[key][f"structure_{s}"] = {
                    "forces": forces,
                    "energy": float(rng.random()),
                }
        return d

    def test_returns_polars_dataframe(self, tmp_path):
        """Test that std_deviation_of_forces returns a Polars DataFrame."""
        import polars as pl

        from alomancy.structure_generation.find_high_sd_structures import (
            std_deviation_of_forces,
        )

        d = self._build_forces_dict()
        df = std_deviation_of_forces(d, tmp_path)
        assert isinstance(df, pl.DataFrame)

    def test_correct_columns(self, tmp_path):
        """Test that returned DataFrame has correct columns."""
        from alomancy.structure_generation.find_high_sd_structures import (
            std_deviation_of_forces,
        )

        d = self._build_forces_dict()
        df = std_deviation_of_forces(d, tmp_path)
        assert set(df.columns) == {
            "structure_index",
            "max_std_dev",
            "mean_std_dev",
            "std_dev_energy",
        }

    def test_identical_predictions_zero_std(self, tmp_path):
        """Test that identical model predictions give zero std dev."""
        from alomancy.structure_generation.find_high_sd_structures import (
            std_deviation_of_forces,
        )

        # All models return same forces -> std_dev should be 0
        forces = np.ones((2, 3))
        flat = self._flatten_forces(forces)
        d = {
            "base_mlip": {"structure_0": {"forces": flat.copy(), "energy": -1.0}},
            "fit_0": {"structure_0": {"forces": flat.copy(), "energy": -1.0}},
        }
        df = std_deviation_of_forces(d, tmp_path)
        assert df["max_std_dev"][0] == pytest.approx(0.0)
        assert df["std_dev_energy"][0] == pytest.approx(0.0)

    def test_sorted_descending_by_max_std_dev(self, tmp_path):
        """Test that results are sorted by max_std_dev in descending order."""
        from alomancy.structure_generation.find_high_sd_structures import (
            std_deviation_of_forces,
        )

        # Structure 0 has low variance, structure 1 has high variance
        d = {
            "base_mlip": {
                "structure_0": {"forces": np.array([[0.1, 0.1, 0.1]]), "energy": -1.0},
                "structure_1": {"forces": np.array([[0.0, 0.0, 0.0]]), "energy": -2.0},
            },
            "fit_0": {
                "structure_0": {"forces": np.array([[0.1, 0.1, 0.1]]), "energy": -1.0},
                "structure_1": {"forces": np.array([[10.0, 10.0, 10.0]]), "energy": -3.0},
            },
        }
        df = std_deviation_of_forces(d, tmp_path)
        # structure_1 should be first (highest std dev)
        assert df["structure_index"][0] == 1

    def test_csv_written(self, tmp_path):
        """Test that CSV file is written."""
        from alomancy.structure_generation.find_high_sd_structures import (
            std_deviation_of_forces,
        )

        d = self._build_forces_dict()
        std_deviation_of_forces(d, tmp_path)
        assert (tmp_path / "std_dev_forces.csv").exists()

    def test_row_count_equals_n_structures(self, tmp_path):
        """Test that output has one row per structure."""
        from alomancy.structure_generation.find_high_sd_structures import (
            std_deviation_of_forces,
        )

        d = self._build_forces_dict(n_structures=5)
        df = std_deviation_of_forces(d, tmp_path)
        assert len(df) == 5


# ============================================================================
# Tests for select_initial_structures
# ============================================================================


@pytest.mark.unit
class TestSelectInitialStructures:
    """Test the select_initial_structures function."""

    @staticmethod
    def _make_atoms(symbols, config_type="train"):
        """Create an Atoms object for testing."""
        n = len(symbols)
        atoms = Atoms(
            symbols=symbols, positions=np.eye(n, 3) * 2, cell=[10, 10, 10], pbc=True
        )
        atoms.info["config_type"] = config_type
        return atoms

    def test_formula_filter(self):
        """Test that chemical formula filter works correctly."""
        from alomancy.structure_generation.select_initial_structures import (
            select_initial_structures,
        )

        np.random.seed(42)
        structures = [self._make_atoms(["H", "H"]) for _ in range(5)] + [
            self._make_atoms(["O", "O"]) for _ in range(5)
        ]
        job_dict = {"name": "test_md"}
        result = select_initial_structures(
            base_name="test",
            structure_generation_job_dict=job_dict,
            train_atoms_list=structures,
            max_number_of_concurrent_jobs=3,
            chem_formula_list=["H2"],
        )
        assert len(result) == 3
        assert all(a.get_chemical_formula() == "H2" for a in result)

    def test_atom_number_range_filter(self):
        """Test that atom count range filter works correctly."""
        from alomancy.structure_generation.select_initial_structures import (
            select_initial_structures,
        )

        np.random.seed(42)
        structures = (
            [self._make_atoms(["H"]) for _ in range(5)]  # 1 atom
            + [self._make_atoms(["H", "O"]) for _ in range(10)]  # 2 atoms
            + [self._make_atoms(["H", "O", "O"]) for _ in range(5)]  # 3 atoms
        )
        job_dict = {"name": "test_md"}
        result = select_initial_structures(
            base_name="test",
            structure_generation_job_dict=job_dict,
            train_atoms_list=structures,
            max_number_of_concurrent_jobs=4,
            atom_number_range=(2, 2),
        )
        assert len(result) == 4
        assert all(len(a) == 2 for a in result)

    def test_config_type_filter(self):
        """Test that config_type filter and marking works correctly."""
        from alomancy.structure_generation.select_initial_structures import (
            select_initial_structures,
        )

        np.random.seed(42)
        structures = [self._make_atoms(["H", "H"], config_type="al_loop_0") for _ in range(10)] + [
            self._make_atoms(["H", "H"], config_type="init_dimer") for _ in range(5)
        ]
        job_dict = {"name": "test_md"}
        result = select_initial_structures(
            base_name="test",
            structure_generation_job_dict=job_dict,
            train_atoms_list=structures,
            max_number_of_concurrent_jobs=3,
            selectable_configs=["al_loop_0"],
        )
        # The returned atoms should have config_type set to {base_name}_{job_name}
        assert all(
            a.info.get("config_type") == "test_test_md" for a in result
        )

    def test_less_than_two_atoms_warning(self):
        """Test warning is raised for atom_number_range starting below 2."""
        from alomancy.structure_generation.select_initial_structures import (
            select_initial_structures,
        )

        structures = [self._make_atoms(["H"]) for _ in range(10)]
        job_dict = {"name": "test_md"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            select_initial_structures(
                base_name="test",
                structure_generation_job_dict=job_dict,
                train_atoms_list=structures,
                max_number_of_concurrent_jobs=3,
                atom_number_range=(1, 5),
            )
        assert any(issubclass(warning.category, UserWarning) for warning in w)


# ============================================================================
# Tests for mark_structures_for_dft
# ============================================================================


@pytest.mark.unit
class TestMarkStructuresForDft:
    """Test the mark_structures_for_dft function."""

    def test_sets_config_type(self):
        """Test that config_type is set correctly."""
        from alomancy.structure_generation.select_initial_structures import (
            mark_structures_for_dft,
        )

        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        mark_structures_for_dft(
            [atoms], base_name="al_loop_0", job_name="structure_generation"
        )
        assert atoms.info["config_type"] == "al_loop_0_structure_generation"

    def test_sets_job_id_default(self):
        """Test that job_id is initialized to -1."""
        from alomancy.structure_generation.select_initial_structures import (
            mark_structures_for_dft,
        )

        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        mark_structures_for_dft([atoms], base_name="test", job_name="md")
        assert atoms.info["job_id"] == -1

    def test_preserves_existing_job_id(self):
        """Test that existing job_id is preserved."""
        from alomancy.structure_generation.select_initial_structures import (
            mark_structures_for_dft,
        )

        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        atoms.info["job_id"] = 42
        mark_structures_for_dft([atoms], base_name="test", job_name="md")
        assert atoms.info["job_id"] == 42

    def test_marks_multiple_structures(self):
        """Test marking multiple structures at once."""
        from alomancy.structure_generation.select_initial_structures import (
            mark_structures_for_dft,
        )

        atoms_list = [
            Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
            for _ in range(3)
        ]
        mark_structures_for_dft(atoms_list, base_name="test", job_name="dft")
        assert all(a.info["config_type"] == "test_dft" for a in atoms_list)


# ============================================================================
# Tests for trajectory I/O
# ============================================================================


@pytest.mark.unit
class TestTrajectoryIO:
    """Test trajectory file input/output."""

    def test_xyz_round_trip(self, tmp_path):
        """Test writing and reading a single structure."""
        from ase.io import read, write

        atoms = Atoms(
            symbols=["H", "O"],
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        path = tmp_path / "traj.xyz"
        write(str(path), [atoms], format="extxyz")
        loaded = read(str(path), ":", format="extxyz")
        assert len(loaded) == 1
        np.testing.assert_allclose(loaded[0].get_positions(), atoms.get_positions())

    def test_multiple_frames(self, tmp_path):
        """Test writing and reading multiple frames."""
        from ase.io import read, write

        frames = [
            Atoms(symbols=["H"], positions=[[i, 0, 0]], cell=[5, 5, 5], pbc=True)
            for i in range(5)
        ]
        path = tmp_path / "multi.xyz"
        write(str(path), frames, format="extxyz")
        loaded = read(str(path), ":", format="extxyz")
        assert len(loaded) == 5

    def test_preserves_structure_properties(self, tmp_path):
        """Test that structure properties are preserved in I/O."""
        from ase.io import read, write

        atoms = Atoms(
            symbols=["C", "H"],
            positions=[[0, 0, 0], [1.1, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        atoms.info["config_type"] = "test"
        atoms.info["REF_energy"] = -5.0

        path = tmp_path / "prop.xyz"
        write(str(path), [atoms], format="extxyz")
        loaded = read(str(path), ":", format="extxyz")[0]

        assert loaded.info["config_type"] == "test"
        assert loaded.info["REF_energy"] == -5.0


# ============================================================================
# Integration and other tests (existing, preserved)
# ============================================================================


class TestMolecularDynamics:
    """Test molecular dynamics functionality."""

    @patch("alomancy.structure_generation.md.md_wfl.run_md")
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
            verbose=0,
        )

        mock_run_md.assert_called_once()

    def test_md_parameter_validation(self):
        """Test MD parameter validation."""
        # Test valid parameters
        valid_params = {
            "steps": 100,
            "temperature": 300,
            "desired_number_of_structures": 20,
            "total_md_runs": 5,
        }

        # Check basic constraints
        assert valid_params["desired_number_of_structures"] > 0
        assert (
            valid_params["steps"]
            > valid_params["desired_number_of_structures"]
            / valid_params["total_md_runs"]
        )
        assert valid_params["temperature"] > 0

        # Test invalid parameters that would cause division by zero
        invalid_params = {
            "steps": 10,
            "desired_number_of_structures": 50,
            "total_md_runs": 5,
        }

        # This should fail the constraint
        snapshot_interval = (
            invalid_params["steps"]
            * invalid_params["total_md_runs"]
            // invalid_params["desired_number_of_structures"]
        )
        assert snapshot_interval == 1  # This would be problematic for the loop

    @patch("ase.md.langevin.Langevin")
    @patch("mace.calculators.MACECalculator")
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
            atoms=atoms, timestep=0.5 * fs, temperature_K=300, friction=0.002
        )

        assert dyn is not None
        mock_langevin.assert_called_once()

    @patch("alomancy.structure_generation.md.md_remote_submitter.md_remote_submitter")
    def test_md_remote_submission(self, mock_md_submitter):
        """Test MD remote submission."""
        from alomancy.structure_generation.md.md_remote_submitter import (
            md_remote_submitter,
        )

        # Mock return trajectory files
        mock_trajectories = [
            "/test/path/md_output_0/trajectory.xyz",
            "/test/path/md_output_1/trajectory.xyz",
        ]
        mock_md_submitter.return_value = mock_trajectories

        # Test parameters
        mock_remote_info = MagicMock()
        base_name = "test_al_loop_0"
        target_file = "trajectory.xyz"
        input_atoms_list = [
            Atoms(symbols=["H"], positions=[[0, 0, 0]]) for _ in range(2)
        ]

        result = md_remote_submitter(
            remote_info=mock_remote_info,
            base_name=base_name,
            target_file=target_file,
            input_atoms_list=input_atoms_list,
            function=MagicMock(),
            function_kwargs={},
        )

        assert len(result) == 2
        assert all("md_output_" in path for path in result)
        mock_md_submitter.assert_called_once()


class TestStructureSelection:
    """Test structure selection functionality."""

    @patch(
        "alomancy.structure_generation.select_initial_structures.select_initial_structures"
    )
    def test_initial_structure_selection(self, mock_select_initial):
        """Test initial structure selection."""
        from alomancy.structure_generation.select_initial_structures import (
            select_initial_structures,
        )

        # Mock selected structures
        mock_structures = [
            Atoms(symbols=["C", "O"], positions=[[0, 0, 0], [1.1, 0, 0]]),
            Atoms(symbols=["N", "H"], positions=[[0, 0, 0], [1.0, 0, 0]]),
        ]
        mock_select_initial.return_value = mock_structures

        result = select_initial_structures(
            base_name="test_loop_0",
            structure_generation_job_dict={"name": "test"},
            max_number_of_concurrent_jobs=2,
            chem_formula_list=[],
            atom_number_range=(2, 10),
            enforce_chemical_diversity=True,
            train_atoms_list=[],
            verbose=0,
        )

        assert len(result) == 2
        mock_select_initial.assert_called_once()

    def test_chemical_diversity_check(self):
        """Test chemical diversity checking."""
        structures = [
            Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [1, 0, 0]]),  # H2
            Atoms(
                symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            ),  # H2O
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
            Atoms(
                symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            ),  # 3 atoms
            Atoms(symbols=["C"] * 20, positions=np.random.random((20, 3))),  # 20 atoms
        ]

        # Filter by atom number range
        min_atoms, max_atoms = 2, 10
        filtered_structures = [
            atoms for atoms in structures if min_atoms <= len(atoms) <= max_atoms
        ]

        assert len(filtered_structures) == 2  # 2-atom and 3-atom structures
        assert all(
            min_atoms <= len(atoms) <= max_atoms for atoms in filtered_structures
        )

    @patch(
        "alomancy.structure_generation.find_high_sd_structures.find_high_sd_structures"
    )
    def test_high_sd_structure_selection(self, mock_find_high_sd, sample_md_structures):
        """Test high standard deviation structure selection."""
        from alomancy.structure_generation.find_high_sd_structures import (
            find_high_sd_structures,
        )

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
            verbose=0,
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

        assert flattened.shape == (1, 15)  # 5 atoms x 3 components

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
                    "energy": atoms.info["energy"] + np.random.random() * 0.1,
                }

        # Test std deviation calculation function structure
        def mock_std_deviation_of_forces(structure_forces_dict, md_dir, verbose=0):
            number_of_structures = len(structure_forces_dict["base_mace"])
            std_dev_array = np.zeros((number_of_structures, 3))

            for structure in range(number_of_structures):
                forces_array = np.concatenate(
                    [
                        structure_forces_dict[fit][f"structure_{structure}"]["forces"]
                        for fit in structure_forces_dict
                    ],
                    axis=0,
                )

                std_dev_per_force_fragment = np.std(forces_array, axis=0)
                energy_array = np.array(
                    [
                        structure_forces_dict[fit][f"structure_{structure}"]["energy"]
                        for fit in structure_forces_dict
                    ]
                )
                std_dev_per_energy = np.std(energy_array)

                std_dev_array[structure, :] = np.array(
                    [
                        np.max(std_dev_per_force_fragment),
                        np.mean(std_dev_per_force_fragment),
                        std_dev_per_energy,
                    ]
                )

            df = pd.DataFrame(
                std_dev_array, columns=["max_std_dev", "mean_std_dev", "std_dev_energy"]
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
        from ase.io import read, write

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
        from ase.io import write

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple trajectory files
            traj_files = []
            for i in range(3):
                traj_file = Path(tmpdir) / f"trajectory_{i}.xyz"
                start_idx = i * 10
                end_idx = (i + 1) * 10
                write(
                    str(traj_file),
                    sample_md_structures[start_idx:end_idx],
                    format="extxyz",
                )
                traj_files.append(str(traj_file))

            # Simulate reading and concatenating
            all_structures = []
            for _ in traj_files:
                structures = sample_md_structures[:10]  # Mock read
                all_structures.extend(structures)

            assert len(all_structures) == 30  # 3 files x 10 structures each


@pytest.mark.integration
class TestStructureGenerationIntegration:
    """Integration tests for structure generation."""

    @patch(
        "alomancy.structure_generation.select_initial_structures.select_initial_structures"
    )
    @patch("alomancy.structure_generation.md.md_remote_submitter.md_remote_submitter")
    @patch(
        "alomancy.structure_generation.find_high_sd_structures.find_high_sd_structures"
    )
    @patch("ase.io.read")
    @patch("pathlib.Path.glob")
    def test_full_structure_generation_workflow(
        self,
        mock_glob,
        mock_read,
        mock_find_high_sd,
        mock_md_submitter,
        mock_select_initial,
        sample_md_structures,
    ):
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

    def test_real_md_simulation(self, skip_if_no_external):
        """Test with real MD simulation if MACE is available."""
        pass

    def test_real_ase_md(self, skip_if_no_external):
        """Test with real ASE MD if available."""
        pass
