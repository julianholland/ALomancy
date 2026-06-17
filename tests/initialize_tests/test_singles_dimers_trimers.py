"""Tests for singles_dimers_trimers initialization functions."""

import pytest
from ase.data import atomic_numbers, vdw_radii


@pytest.mark.unit
class TestCreateSingleAtomsList:
    def test_returns_list_of_one(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list
        result = create_single_atoms_list("H")
        assert len(result) == 1

    def test_config_type_is_isolated_atom(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list
        result = create_single_atoms_list("H")
        assert result[0].info["config_type"] == "IsolatedAtom"

    def test_correct_element(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list
        result = create_single_atoms_list("O")
        assert result[0].get_chemical_symbols() == ["O"]

    def test_cell_based_on_vdw_radius(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list
        result = create_single_atoms_list("H")
        expected = vdw_radii[atomic_numbers["H"]] * 3
        assert result[0].cell[0][0] == pytest.approx(expected)

    def test_needs_relaxation_false(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list
        result = create_single_atoms_list("H")
        assert result[0].info["needs_relaxation"] is False

    def test_different_elements(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list
        for elem in ["C", "N", "O", "Na"]:
            result = create_single_atoms_list(elem)
            assert result[0].get_chemical_symbols() == [elem]


@pytest.mark.unit
class TestCreateDimerAtomsList:
    def test_returns_correct_count(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "O", 5)
        assert len(result) == 5

    def test_zero_dimers_returns_empty(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "O", 0)
        assert result == []

    def test_config_type(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "H", 3)
        assert all(a.info["config_type"] == "init_dimer" for a in result)

    def test_two_atoms_per_dimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "O", 4)
        assert all(len(a) == 2 for a in result)

    def test_correct_element_symbols(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "O", 4)
        for a in result:
            syms = sorted(a.get_chemical_symbols())
            assert syms == ["H", "O"]

    def test_distances_span_range(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "H", 10)
        distances = [a.get_distance(0, 1) for a in result]
        assert min(distances) < max(distances)

    def test_distance_stored_in_info(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "O", 3)
        assert all("distance" in a.info for a in result)

    def test_homonuclear_dimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("C", "C", 5)
        assert len(result) == 5
        for a in result:
            assert all(s == "C" for s in a.get_chemical_symbols())

    def test_needs_relaxation_false(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "H", 3)
        assert all(a.info["needs_relaxation"] is False for a in result)

    def test_single_dimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("N", "O", 1)
        assert len(result) == 1
        assert result[0].get_chemical_symbols() in [["N", "O"], ["O", "N"]]

    def test_distance_minimum_fraction_of_vdw(self):
        """Minimum distance should be 20% of sum of vdW radii."""
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list
        result = create_dimer_atoms_list("H", "H", 10)
        vdw_sum = sum(vdw_radii[atomic_numbers["H"]] for _ in range(2))
        min_distance = 0.2 * vdw_sum
        distances = [a.get_distance(0, 1) for a in result]
        assert float(min(distances)) >= float(min_distance) - 1e-10


@pytest.mark.unit
class TestCreateTrimerAtomsList:
    def test_returns_atoms_with_three_per_trimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all(len(a) == 3 for a in result)

    def test_zero_trimers_returns_empty(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "O", "H", 0)
        assert result == []

    def test_one_trimer_returns_empty(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "O", "H", 1)
        assert result == []

    def test_config_type(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all(a.info["config_type"] == "init_trimer" for a in result)

    def test_needs_relaxation_false(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all(a.info["needs_relaxation"] is False for a in result)

    def test_deformation_info_stored(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all("deformation" in a.info for a in result)

    def test_exact_count(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        for n in [5, 10, 20, 50]:
            result = create_trimer_atoms_list("H", "O", "O", n)
            assert len(result) == n, f"Expected {n}, got {len(result)}"

    def test_two_atom_count(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("H", "H", "H", 2)
        assert len(result) == 2

    def test_homonuclear_trimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list
        result = create_trimer_atoms_list("C", "C", "C", 5)
        for a in result:
            assert all(s == "C" for s in a.get_chemical_symbols())
