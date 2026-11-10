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
    @pytest.mark.unit
    def test_returns_correct_count(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 5, seed=803)
        assert len(result) == 5

    @pytest.mark.unit
    def test_zero_structures_returns_empty(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 0, seed=803)
        assert result == []

    @pytest.mark.unit
    def test_config_type(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 4, seed=803)
        assert all(a.info["config_type"] == "init_rattle" for a in result)

    @pytest.mark.unit
    def test_needs_relaxation_false(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.01, 4, seed=803)
        assert all(a.info["needs_relaxation"] is False for a in result)

    @pytest.mark.unit
    def test_positions_perturbed(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.05, 3, seed=803)
        for a in result:
            assert not np.allclose(a.positions, atoms.positions)

    @pytest.mark.unit
    def test_original_atoms_not_modified(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        orig_positions = atoms.positions.copy()
        create_rattle_atoms_list(atoms, 0.05, 5, seed=803)
        np.testing.assert_allclose(atoms.positions, orig_positions)

    @pytest.mark.unit
    def test_copies_are_independent_perturbations(self):
        """Each of the num_structures copies uses a distinct seed (seed + i),
        so they should not all be identical."""
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.05, 5, seed=803)
        positions = [a.positions.copy() for a in result]
        assert not all(np.allclose(positions[0], p) for p in positions[1:])

    @pytest.mark.unit
    def test_deterministic_given_same_seed(self):
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result_a = create_rattle_atoms_list(atoms, 0.05, 3, seed=42)
        result_b = create_rattle_atoms_list(atoms, 0.05, 3, seed=42)
        for a, b in zip(result_a, result_b):
            np.testing.assert_allclose(a.positions, b.positions)

    @pytest.mark.unit
    def test_cell_unchanged(self):
        """Rattle perturbs positions only, never the cell."""
        from alomancy.initialize.rattle import create_rattle_atoms_list

        atoms = _make_h2o()
        result = create_rattle_atoms_list(atoms, 0.05, 3, seed=803)
        for a in result:
            np.testing.assert_allclose(a.cell.array, atoms.cell.array)


@pytest.mark.unit
def test_single_atom_cell_is_rattled_as_a_2x2x2_supercell():
    """Rattling a 1-atom periodic cell only translates its atom (same
    energy, zero forces): a duplicate label. It is rattled as a supercell."""
    from ase.build import bulk

    from alomancy.initialize.rattle import create_rattle_atoms_list

    fcc = bulk("Pt", "fcc", a=3.92)
    fcc.info["config_type"] = "init_MP"
    result = create_rattle_atoms_list(fcc, 0.05, 3, seed=803)
    assert len(result) == 3
    for a in result:
        assert len(a) == 8
        assert a.info["supercell"] == "2x2x2"
        assert a.info["parent_config_type"] == "init_MP"
        ideal = fcc.repeat((2, 2, 2)).positions
        assert not np.allclose(a.positions - a.positions[0], ideal - ideal[0])


@pytest.mark.unit
def test_single_non_periodic_atom_is_not_rattled():
    from alomancy.initialize.rattle import create_rattle_atoms_list

    assert create_rattle_atoms_list(Atoms("Pt"), 0.05, 3, seed=803) == []


@pytest.mark.unit
def test_default_rattle_size_scales_with_the_bond_length():
    from ase.build import bulk

    from alomancy.initialize.rattle import create_rattle_atoms_list

    fcc = bulk("Pt", "fcc", a=3.92)
    (rattled,) = create_rattle_atoms_list(fcc, None, 1, seed=803)
    # 3% of the fcc nearest-neighbour distance a / sqrt(2) = 2.772 A.
    assert rattled.info["rattle_standard_deviation"] == pytest.approx(
        0.03 * 3.92 / np.sqrt(2)
    )
    (fixed,) = create_rattle_atoms_list(fcc, 0.2, 1, seed=803)
    assert fixed.info["rattle_standard_deviation"] == 0.2
    # Multi-atom cells are rattled as they are.
    h2o = _make_h2o()
    assert len(create_rattle_atoms_list(h2o, None, 1, seed=803)[0]) == 3
