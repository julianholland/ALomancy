from unittest.mock import MagicMock

import pytest

from alomancy.initialize.initialization_structure_list import (
    compute_initialization_needs,
)


def make_mock_db(all_counts):
    """Create a mock DB that returns the given count dict from count_all_by_config_type_and_formula."""
    db = MagicMock()
    db.count_all_by_config_type_and_formula.return_value = all_counts
    return db


class TestComputeInitializationNeeds:
    """Tests for compute_initialization_needs function."""

    @pytest.mark.unit
    def test_empty_db_needs_everything(self):
        """When DB is empty, all targets should be needed."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert set(needs["isolated_atoms"]) == {"H", "O"}
        # All 3 dimer combos needed at full count
        assert needs["dimer_override"].get("H2") == 10
        assert needs["dimer_override"].get("HO") == 10
        assert needs["dimer_override"].get("O2") == 10
        assert needs["amorphous_override"] == 100
        assert needs["mp_structures"] is True

    @pytest.mark.unit
    def test_isolated_atoms_fully_present(self):
        """When all isolated atoms are in DB, none should be needed."""
        db = make_mock_db({"IsolatedAtom": {"H": 1, "O": 1}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["isolated_atoms"] == []

    @pytest.mark.unit
    def test_isolated_atoms_partially_present(self):
        """When only some isolated atoms are in DB, missing ones should be listed."""
        db = make_mock_db({"IsolatedAtom": {"H": 1}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["isolated_atoms"] == ["O"]

    @pytest.mark.unit
    def test_dimer_partial_count(self):
        """When dimers partially exist, compute remaining count needed."""
        # H2 has 7 already, HO has 0, O2 has 10
        db = make_mock_db({"init_dimer": {"H2": 7, "O2": 10}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["dimer_override"]["H2"] == 3   # 10 - 7
        assert needs["dimer_override"]["HO"] == 10  # 10 - 0
        assert "O2" not in needs["dimer_override"]  # already at target

    @pytest.mark.unit
    def test_all_dimers_satisfied(self):
        """When all dimers meet or exceed target, return empty dimer_override."""
        db = make_mock_db({"init_dimer": {"H2": 10, "HO": 15, "O2": 10}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["dimer_override"] == {}

    @pytest.mark.unit
    def test_amorphous_partial(self):
        """When amorphous count is below target, compute remainder needed."""
        db = make_mock_db({"init_amorphous": {"HO": 50}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["amorphous_override"] == 50  # 100 - 50

    @pytest.mark.unit
    def test_amorphous_satisfied(self):
        """When amorphous total meets or exceeds target, need zero more."""
        db = make_mock_db({"init_amorphous": {"HO": 100, "H2O": 50}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["amorphous_override"] == 0  # 150 >= 100

    @pytest.mark.unit
    def test_mp_not_needed_when_present(self):
        """When init_MP is in DB, mp_structures should be False."""
        db = make_mock_db({"init_MP": {"NaCl": 5}})
        needs = compute_initialization_needs(db, ["Na", "Cl"], True, True, 10, 5, 100)
        assert needs["mp_structures"] is False

    @pytest.mark.unit
    def test_mp_disabled(self):
        """When mp_structures parameter is False, result should be False."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H", "O"], True, False, 10, 5, 100)
        assert needs["mp_structures"] is False

    @pytest.mark.unit
    def test_mp_needed_when_db_empty(self):
        """When DB is empty and mp_structures is True, should need fetch."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        assert needs["mp_structures"] is True

    @pytest.mark.unit
    def test_trimer_partial_count(self):
        """When trimers partially exist, compute remaining count needed."""
        # H3 has 3/5
        db = make_mock_db({"init_trimer": {"H3": 3}})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 5)
        assert needs["trimer_override"]["H3"] == 2  # 5 - 3

    @pytest.mark.unit
    def test_single_element(self):
        """Test with single element: one dimer combo (H-H) and one trimer (H-H-H)."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H"], True, True, 10, 5, 50)
        assert needs["isolated_atoms"] == ["H"]
        assert "H2" in needs["dimer_override"]
        assert needs["dimer_override"]["H2"] == 10
        assert "H3" in needs["trimer_override"]
        assert needs["trimer_override"]["H3"] == 5

    @pytest.mark.unit
    def test_all_trimer_combos_generated_for_two_elements(self):
        """Test that all 4 trimer combos are checked for two-element system."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)
        # For elements ["H", "O"], trimers are: H3, H2O, HO2, O3
        assert "H3" in needs["trimer_override"]
        assert "H2O" in needs["trimer_override"]
        assert "HO2" in needs["trimer_override"]
        assert "O3" in needs["trimer_override"]

    @pytest.mark.unit
    def test_dimer_exceeds_target(self):
        """When dimer count already exceeds target, should not be in override."""
        db = make_mock_db({"init_dimer": {"H2": 20}})
        needs = compute_initialization_needs(db, ["H"], True, True, 10, 5, 50)
        assert "H2" not in needs["dimer_override"]

    @pytest.mark.unit
    def test_trimer_exceeds_target(self):
        """When trimer count already exceeds target, should not be in override."""
        db = make_mock_db({"init_trimer": {"H3": 100}})
        needs = compute_initialization_needs(db, ["H"], True, True, 10, 5, 50)
        assert "H3" not in needs["trimer_override"]

    @pytest.mark.unit
    def test_amorphous_exceeds_target(self):
        """When amorphous total exceeds target, amorphous_override should be 0."""
        db = make_mock_db({"init_amorphous": {"H2": 200}})
        needs = compute_initialization_needs(db, ["H"], True, True, 10, 5, 100)
        assert needs["amorphous_override"] == 0

    @pytest.mark.unit
    def test_complex_mixed_state(self):
        """Test a complex scenario with mixed present and absent structures."""
        all_counts = {
            "IsolatedAtom": {"H": 1},  # H present, O missing
            "init_dimer": {"H2": 8},   # H2 needs 2 more, HO needs 10, O2 needs 10
            "init_trimer": {"H3": 5},  # H3 complete, H2O needs 5, HO2 needs 5, O3 needs 5
            "init_amorphous": {"HO": 75},  # 25 more needed out of 100
            # init_MP not present, so MP needed
        }
        db = make_mock_db(all_counts)
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 10, 5, 100)

        assert needs["isolated_atoms"] == ["O"]
        assert needs["dimer_override"]["H2"] == 2
        assert needs["dimer_override"]["HO"] == 10
        assert needs["dimer_override"]["O2"] == 10
        assert "H3" not in needs["trimer_override"]  # already satisfied
        assert needs["trimer_override"]["H2O"] == 5
        assert needs["trimer_override"]["HO2"] == 5
        assert needs["trimer_override"]["O3"] == 5
        assert needs["amorphous_override"] == 25
        assert needs["mp_structures"] is True

    @pytest.mark.unit
    def test_three_element_system(self):
        """Test with three elements to verify combo generation scales correctly."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H", "C", "N"], True, True, 10, 5, 100)

        # Dimers: HC, H2, HN, C2, CN, N2 (6 total from combinations_with_replacement)
        assert len(needs["dimer_override"]) == 6

        # Trimers: H3, H2C, H2N, HC2, HCN, HN2, C3, C2N, CN2, N3 (10 total)
        assert len(needs["trimer_override"]) == 10

        # All should need the target count
        for count in needs["dimer_override"].values():
            assert count == 10
        for count in needs["trimer_override"].values():
            assert count == 5

    @pytest.mark.unit
    def test_return_dict_keys(self):
        """Test that the returned dict has exactly the expected keys."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H"], True, True, 10, 5, 100)
        expected_keys = {
            "isolated_atoms",
            "dimer_override",
            "trimer_override",
            "amorphous_override",
            "mp_structures",
        }
        assert set(needs.keys()) == expected_keys

    @pytest.mark.unit
    def test_isolated_atoms_multiple_missing(self):
        """Test isolated atoms with multiple elements missing."""
        db = make_mock_db({"IsolatedAtom": {"C": 1}})
        needs = compute_initialization_needs(db, ["H", "C", "O"], True, True, 10, 5, 100)
        assert set(needs["isolated_atoms"]) == {"H", "O"}

    @pytest.mark.unit
    def test_zero_target_counts(self):
        """Test with zero target counts for dimers/trimers/amorphous."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H", "O"], True, True, 0, 0, 0)
        assert needs["dimer_override"] == {}
        assert needs["trimer_override"] == {}
        assert needs["amorphous_override"] == 0

    @pytest.mark.unit
    def test_mp_structures_false_overrides_empty_db(self):
        """Ensure mp_structures=False takes precedence even with empty DB."""
        db = make_mock_db({})
        needs = compute_initialization_needs(db, ["H"], True, False, 10, 5, 100)
        assert needs["mp_structures"] is False
        # But other fields should still reflect needs
        assert "H2" in needs["dimer_override"]
