"""Tests for amorphize initialization function."""

import numpy as np
import pytest


@pytest.mark.unit
class TestCreateAmorphousAtomsList:
    def test_returns_correct_count(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=3, seed=42
        )
        assert len(result) == 3

    def test_zero_structures_returns_empty(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=0, seed=42
        )
        assert result == []

    def test_config_type(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=2, seed=42
        )
        assert all(a.info["config_type"] == "init_amorphous" for a in result)

    def test_needs_relaxation_true(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=2, seed=42
        )
        assert all(a.info["needs_relaxation"] is True for a in result)

    def test_correct_atom_count_per_structure(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H", "O"], atom_number=3, density=1.0, num_structures=2, seed=42
        )
        assert all(len(a) == 3 for a in result)

    def test_reproducibility_with_same_seed(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        r1 = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=2, seed=99
        )
        r2 = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=2, seed=99
        )
        for a1, a2 in zip(r1, r2):
            np.testing.assert_allclose(a1.positions, a2.positions)

    def test_custom_composition_list(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H", "O"],
            atom_number=2,
            density=1.0,
            num_structures=2,
            seed=42,
            composition_list=[["H", "O"], ["H", "H"]],
        )
        assert len(result) == 2
        assert all(len(a) == 2 for a in result)

    def test_pbc_set(self):
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H"], atom_number=2, density=1.0, num_structures=1, seed=42
        )
        assert all(result[0].get_pbc())

    def test_fewer_compositions_than_structures_fills_by_sampling(self):
        """When compositions < num_structures, extras are sampled from existing."""
        from alomancy.initialize.amorphize import create_amorphous_atoms_list
        result = create_amorphous_atoms_list(
            elements=["H"],
            atom_number=2,
            density=1.0,
            num_structures=5,
            seed=42,
            composition_list=[["H", "H"]],
        )
        assert len(result) == 5
