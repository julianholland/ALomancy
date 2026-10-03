"""Tests for the rattle initialization function."""

import numpy as np
import pytest
from ase import Atoms


def _make_h2o():
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
        cell=[5, 5, 5],
        pbc=True,
    )


@pytest.mark.unit
class TestCreateRattleAtomsList:
    def test_returns_correct_count(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 5, seed=803)
        assert len(result) == 5

    def test_zero_structures_returns_empty(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 0, seed=803)
        assert result == []

    def test_config_type(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 4, seed=803)
        assert all(a.info["config_type"] == "init_rattle" for a in result)

    def test_needs_relaxation_false(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 4, seed=803)
        assert all(a.info["needs_relaxation"] is False for a in result)

    def test_positions_perturbed(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.05, 3, seed=803)
        for a in result:
            assert not np.allclose(a.positions, atoms.positions)

    def test_original_atoms_not_modified(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        orig_positions = atoms.positions.copy()
        create_rattle_atoms_list(atoms, 0.05, 5, seed=803)
        np.testing.assert_allclose(atoms.positions, orig_positions)

    def test_copies_are_independent_perturbations(self):
        """Each of the num_structures copies uses a distinct seed (seed + i),
        so they should not all be identical."""
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.05, 5, seed=803)
        positions = [a.positions.copy() for a in result]
        assert not all(np.allclose(positions[0], p) for p in positions[1:])

    def test_deterministic_given_same_seed(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result_a = create_rattle_atoms_list(atoms, 0.05, 3, seed=42)
        result_b = create_rattle_atoms_list(atoms, 0.05, 3, seed=42)
        for a, b in zip(result_a, result_b):
            np.testing.assert_allclose(a.positions, b.positions)

    def test_cell_unchanged(self):
        """Rattle perturbs positions only, never the cell."""
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.05, 3, seed=803)
        for a in result:
            np.testing.assert_allclose(a.cell.array, atoms.cell.array)
