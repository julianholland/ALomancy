"""Tests for singles_dimers_trimers initialization functions."""

import pytest
from ase.data import atomic_numbers, covalent_radii, vdw_radii


@pytest.mark.unit
class TestCreateSingleAtomsList:
    @pytest.mark.unit
    def test_returns_list_of_one(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list

        result = create_single_atoms_list("H")
        assert len(result) == 1

    @pytest.mark.unit
    def test_config_type_is_isolated_atom(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list

        result = create_single_atoms_list("H")
        assert result[0].info["config_type"] == "IsolatedAtom"

    @pytest.mark.unit
    def test_correct_element(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list

        result = create_single_atoms_list("O")
        assert result[0].get_chemical_symbols() == ["O"]

    @pytest.mark.unit
    def test_cell_is_large_enough_to_isolate_the_atom(self):
        """At least 15 Angstrom: 3 x the vdW radius alone (about 5 Angstrom)
        left a QE atom interacting with its periodic images."""
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list

        assert create_single_atoms_list("H")[0].cell[0][0] == pytest.approx(15.0)
        # Larger atoms keep 3 x their vdW radius when that is bigger.
        big = vdw_radii[atomic_numbers["Cs"]] * 3
        expected = max(big, 15.0)
        assert create_single_atoms_list("Cs")[0].cell[0][0] == pytest.approx(expected)

    @pytest.mark.unit
    def test_needs_relaxation_false(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list

        result = create_single_atoms_list("H")
        assert result[0].info["needs_relaxation"] is False

    @pytest.mark.unit
    def test_different_elements(self):
        from alomancy.initialize.singles_dimers_trimers import create_single_atoms_list

        for elem in ["C", "N", "O", "Na"]:
            result = create_single_atoms_list(elem)
            assert result[0].get_chemical_symbols() == [elem]


@pytest.mark.unit
class TestCreateDimerAtomsList:
    @pytest.mark.unit
    def test_returns_correct_count(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "O", 5)
        assert len(result) == 5

    @pytest.mark.unit
    def test_zero_dimers_returns_empty(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "O", 0)
        assert result == []

    @pytest.mark.unit
    def test_config_type(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "H", 3)
        assert all(a.info["config_type"] == "init_dimer" for a in result)

    @pytest.mark.unit
    def test_two_atoms_per_dimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "O", 4)
        assert all(len(a) == 2 for a in result)

    @pytest.mark.unit
    def test_correct_element_symbols(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "O", 4)
        for a in result:
            syms = sorted(a.get_chemical_symbols())
            assert syms == ["H", "O"]

    @pytest.mark.unit
    def test_distances_span_range(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "H", 10)
        distances = [a.get_distance(0, 1) for a in result]
        assert min(distances) < max(distances)

    @pytest.mark.unit
    def test_distance_stored_in_info(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "O", 3)
        assert all("distance" in a.info for a in result)

    @pytest.mark.unit
    def test_homonuclear_dimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("C", "C", 5)
        assert len(result) == 5
        for a in result:
            assert all(s == "C" for s in a.get_chemical_symbols())

    @pytest.mark.unit
    def test_needs_relaxation_false(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("H", "H", 3)
        assert all(a.info["needs_relaxation"] is False for a in result)

    @pytest.mark.unit
    def test_single_dimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        result = create_dimer_atoms_list("N", "O", 1)
        assert len(result) == 1
        assert result[0].get_chemical_symbols() in [["N", "O"], ["O", "N"]]

    @pytest.mark.unit
    def test_distances_follow_the_covalent_radii(self):
        """0.5-2.0 x the covalent radii sum by default: Pt-Pt starts at 1.36
        Angstrom, not the 0.70 of the former 0.2 x vdW (overlapping PAW
        spheres, forces above 100 eV/Angstrom, failed SCFs)."""
        from alomancy.initialize.singles_dimers_trimers import create_dimer_atoms_list

        pair = 2 * covalent_radii[atomic_numbers["Pt"]]
        result = create_dimer_atoms_list("Pt", "Pt", 10)
        distances = [a.get_distance(0, 1) for a in result]
        assert min(distances) == pytest.approx(0.5 * pair)
        assert max(distances) == pytest.approx(2.0 * pair)
        assert min(distances) > 1.3

    @pytest.mark.unit
    def test_distance_range_override(self):
        from alomancy.initialize.singles_dimers_trimers import (
            create_dimer_atoms_list,
            create_trimer_atoms_list,
        )

        pair = 2 * covalent_radii[atomic_numbers["Pt"]]
        result = create_dimer_atoms_list("Pt", "Pt", 3, distance_range=[0.8, 1.5])
        distances = [a.get_distance(0, 1) for a in result]
        assert min(distances) == pytest.approx(0.8 * pair)
        assert max(distances) == pytest.approx(1.5 * pair)
        trimers = create_trimer_atoms_list(
            "Pt", "Pt", "Pt", 5, distance_range=[0.8, 1.5]
        )
        shortest = min(
            min(t.get_distance(i, j) for i in range(3) for j in range(i + 1, 3))
            for t in trimers
        )
        assert shortest >= 0.8 * pair - 1e-9


@pytest.mark.unit
class TestCreateTrimerAtomsList:
    @pytest.mark.unit
    def test_returns_atoms_with_three_per_trimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all(len(a) == 3 for a in result)

    @pytest.mark.unit
    def test_zero_trimers_returns_empty(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "O", "H", 0)
        assert result == []

    @pytest.mark.unit
    def test_one_trimer_returns_empty(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "O", "H", 1)
        assert result == []

    @pytest.mark.unit
    def test_config_type(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all(a.info["config_type"] == "init_trimer" for a in result)

    @pytest.mark.unit
    def test_needs_relaxation_false(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all(a.info["needs_relaxation"] is False for a in result)

    @pytest.mark.unit
    def test_deformation_info_stored(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "O", "H", 10)
        assert all("deformation" in a.info for a in result)

    @pytest.mark.unit
    def test_exact_count(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        for n in [5, 10, 20, 50]:
            result = create_trimer_atoms_list("H", "O", "O", n)
            assert len(result) == n, f"Expected {n}, got {len(result)}"

    @pytest.mark.unit
    def test_two_atom_count(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("H", "H", "H", 2)
        assert len(result) == 2

    @pytest.mark.unit
    def test_homonuclear_trimer(self):
        from alomancy.initialize.singles_dimers_trimers import create_trimer_atoms_list

        result = create_trimer_atoms_list("C", "C", "C", 5)
        for a in result:
            assert all(s == "C" for s in a.get_chemical_symbols())
