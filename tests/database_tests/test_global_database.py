"""Tests for GlobalDatabase class."""

import numpy as np
import pytest
from ase import Atoms

from alomancy.database.global_database import GlobalDatabase

# ---------------------------------------------------------------------------
# Helper function (copied from conftest for use in this test module)
# ---------------------------------------------------------------------------


def make_atoms(
    symbols: list,
    config_type=None,
    ref_energy=None,
    ref_forces=None,
    needs_relaxation=False,
    cell=10.0,
):
    """Create a test Atoms object with optional metadata."""
    n = len(symbols)
    positions = np.eye(n, 3) * 2.0
    atoms = Atoms(symbols=symbols, positions=positions, cell=[cell] * 3, pbc=True)
    if config_type is not None:
        atoms.info["config_type"] = config_type
    if ref_energy is not None:
        atoms.info["REF_energy"] = ref_energy
    if ref_forces is not None:
        atoms.arrays["REF_forces"] = np.array(ref_forces)
    if needs_relaxation:
        atoms.info["needs_relaxation"] = True
    return atoms


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestAddStructures:
    """Tests for add_structures method with deduplication logic."""

    @pytest.mark.unit
    def test_dedup_isolated_atom(self, tmp_path):
        """Add same IsolatedAtom H twice, assert db.size == 1."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h_atom.copy()])
        assert db.size == 1

    @pytest.mark.unit
    def test_dedup_init_mp(self, tmp_path):
        """Add same formula init_MP twice, assert size == 1."""
        a1 = make_atoms(["Na", "Cl"], config_type="init_MP", ref_energy=-5.0)
        a2 = make_atoms(["Na", "Cl"], config_type="init_MP", ref_energy=-5.1)
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a1, a2])
        assert db.size == 1

    @pytest.mark.unit
    def test_no_dedup_init_dimer(self, tmp_path):
        """Two dimers with same formula are both added (count-based, not exact dedup)."""
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2_dimer, h2_dimer.copy()])
        assert db.size == 2

    @pytest.mark.unit
    def test_no_dedup_al_loop(self, tmp_path):
        """al_loop_0 structures always added."""
        h2o_mol = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        h2o_mol.info["config_type"] = "al_loop_0"
        h2o_mol.info["REF_energy"] = -76.0
        h2o_mol.arrays["REF_forces"] = np.zeros((3, 3))

        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2o_mol, h2o_mol.copy()])
        assert db.size == 2

    @pytest.mark.unit
    def test_skip_duplicates_false_adds_all(self, tmp_path):
        """Even IsolatedAtom added twice when skip_duplicates=False."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h_atom.copy()], skip_duplicates=False)
        assert db.size == 2

    @pytest.mark.unit
    def test_returns_added_count(self, tmp_path):
        """3 structures, 1 IsolatedAtom duplicate -> should add 2."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        o_atom = make_atoms(
            ["O"],
            config_type="IsolatedAtom",
            ref_energy=-432.0,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        count = db.add_structures([h_atom, o_atom, h_atom.copy()])
        assert count == 2

    @pytest.mark.unit
    def test_ref_forces_round_trip(self, tmp_path):
        """REF_forces survive storage and retrieval."""
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2_dimer])
        retrieved = db.get_all_as_atoms()
        assert len(retrieved) == 1
        assert "REF_forces" in retrieved[0].arrays
        np.testing.assert_allclose(
            retrieved[0].arrays["REF_forces"], h2_dimer.arrays["REF_forces"], atol=1e-6
        )

    @pytest.mark.unit
    def test_custom_dedup_list(self, tmp_path):
        """When config_types_to_dedup=["init_dimer"], dimers are deduped."""
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures(
            [h2_dimer, h2_dimer.copy()], config_types_to_dedup=["init_dimer"]
        )
        assert db.size == 1


class TestCounting:
    """Tests for counting methods."""

    @pytest.mark.unit
    def test_count_all_by_config_type_and_formula(self, tmp_path):
        """Test detailed counting by config_type and formula."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h2_dimer, h2_dimer.copy()], skip_duplicates=False)
        counts = db.count_all_by_config_type_and_formula()
        assert counts["IsolatedAtom"]["H"] == 1
        assert counts["init_dimer"]["H2"] == 2

    @pytest.mark.unit
    def test_count_by_config_type(self, tmp_path):
        """Test counting aggregated by config_type only."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        o_atom = make_atoms(
            ["O"],
            config_type="IsolatedAtom",
            ref_energy=-432.0,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, o_atom, h2_dimer], skip_duplicates=False)
        counts = db.count_by_config_type()
        assert counts["IsolatedAtom"] == 2
        assert counts["init_dimer"] == 1

    @pytest.mark.unit
    def test_count_empty_db(self, tmp_path):
        """Test counting on an empty database."""
        db = GlobalDatabase(str(tmp_path / "db"))
        assert db.count_all_by_config_type_and_formula() == {}
        assert db.count_by_config_type() == {}


class TestRetrieval:
    """Tests for retrieval methods."""

    @pytest.mark.unit
    def test_get_structures_by_config_type(self, tmp_path):
        """Test filtering structures by config_type."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        o_atom = make_atoms(
            ["O"],
            config_type="IsolatedAtom",
            ref_energy=-432.0,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, o_atom, h2_dimer], skip_duplicates=False)
        isolated = db.get_structures_by_config_type(["IsolatedAtom"])
        assert len(isolated) == 2
        assert all(a.info.get("config_type") == "IsolatedAtom" for a in isolated)

    @pytest.mark.unit
    def test_get_all_as_atoms(self, tmp_path):
        """Test retrieving all structures as Atoms objects."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h2_dimer], skip_duplicates=False)
        all_atoms = db.get_all_as_atoms()
        assert len(all_atoms) == 2
        assert all(isinstance(a, Atoms) for a in all_atoms)

    @pytest.mark.unit
    def test_size_property(self, tmp_path):
        """Test the size property increases correctly."""
        h_atom = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        h2_dimer = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        h2o_mol = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        h2o_mol.info["config_type"] = "al_loop_0"
        h2o_mol.info["REF_energy"] = -76.0
        h2o_mol.arrays["REF_forces"] = np.zeros((3, 3))

        db = GlobalDatabase(str(tmp_path / "db"))
        assert db.size == 0
        db.add_structures([h_atom], skip_duplicates=False)
        assert db.size == 1
        db.add_structures([h2_dimer, h2o_mol], skip_duplicates=False)
        assert db.size == 3


class TestCountByConfigTypeAndFormula:
    """Tests for count_by_config_type_and_formula (single config_type lookup)."""

    @pytest.mark.unit
    def test_returns_per_formula_count(self, tmp_path):
        h2 = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        o2 = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2, h2.copy(), o2], skip_duplicates=False)
        counts = db.count_by_config_type_and_formula("init_dimer")
        assert counts["H2"] == 2
        assert counts["O2"] == 1

    @pytest.mark.unit
    def test_empty_for_missing_config_type(self, tmp_path):
        db = GlobalDatabase(str(tmp_path / "db"))
        assert db.count_by_config_type_and_formula("nonexistent") == {}

    @pytest.mark.unit
    def test_ignores_other_config_types(self, tmp_path):
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6)
        h2 = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h2], skip_duplicates=False)
        counts = db.count_by_config_type_and_formula("IsolatedAtom")
        assert "H2" not in counts
        assert counts.get("H") == 1


class TestPrepareForStorageEdgeCases:
    @pytest.mark.unit
    def test_returns_none_when_no_energy_source(self, tmp_path):
        """Atoms with no REF_energy and no calculator → _prepare_for_storage returns None."""
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        atoms.info["config_type"] = "test"
        assert GlobalDatabase._prepare_for_storage(atoms) is None

    @pytest.mark.unit
    def test_stored_without_forces(self, tmp_path):
        """Atoms with energy but no forces are stored; no crash on retrieval."""
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        atoms.info["config_type"] = "test"
        atoms.info["REF_energy"] = -13.6
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([atoms])
        assert db.size == 1
        retrieved = db.get_all_as_atoms()
        assert retrieved[0].info["REF_energy"] == pytest.approx(-13.6)


class TestSplitTagging:
    """Tests for add_structures(split=...) and get_train/test_atoms."""

    @pytest.mark.unit
    def test_add_with_train_split_tag(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a], split="train", skip_duplicates=False)
        train = db.get_train_atoms()
        assert len(train) == 1
        assert train[0].info.get("split") == "train"

    @pytest.mark.unit
    def test_get_train_excludes_test_split(self, tmp_path):
        h2 = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        o2 = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2], split="train", skip_duplicates=False)
        db.add_structures([o2], split="test", skip_duplicates=False)
        assert len(db.get_train_atoms()) == 1
        assert len(db.get_test_atoms()) == 1

    @pytest.mark.unit
    def test_get_test_atoms_filters_correctly(self, tmp_path):
        a = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        b = make_atoms(
            ["O"],
            config_type="IsolatedAtom",
            ref_energy=-432.0,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a], split="train", skip_duplicates=False)
        db.add_structures([b], split="test", skip_duplicates=False)
        test = db.get_test_atoms()
        assert len(test) == 1
        assert test[0].get_chemical_formula() == "O"

    @pytest.mark.unit
    def test_get_train_atoms_excludes_duplicates_by_default(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        b = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a, b], split="train", skip_duplicates=False)
        # Flag first structure as duplicate
        db.flag_as_duplicates([0])
        assert len(db.get_train_atoms()) == 1
        assert len(db.get_train_atoms(exclude_duplicates=False)) == 2

    @pytest.mark.unit
    def test_untagged_structures_absent_from_split_queries(self, tmp_path):
        a = make_atoms(
            ["H"],
            config_type="IsolatedAtom",
            ref_energy=-13.6,
            ref_forces=[[0.0, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        # Add without split tag
        db.add_structures([a], skip_duplicates=False)
        assert len(db.get_train_atoms()) == 0
        assert len(db.get_test_atoms()) == 0
        assert db.size == 1


class TestGetSplitPartition:
    """Tests for get_split_partition."""

    @pytest.mark.unit
    def test_returns_only_train_split(self, tmp_path):
        h2 = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        o2 = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2], split="train", skip_duplicates=False)
        db.add_structures([o2], split="test", skip_duplicates=False)
        train_p = db.get_split_partition("train")
        assert len(train_p) == 1

    @pytest.mark.unit
    def test_empty_when_no_train_structures(self, tmp_path):
        a = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6)
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a], split="test", skip_duplicates=False)
        train_p = db.get_split_partition("train")
        assert len(train_p) == 0


class TestFlagAsDuplicates:
    """Tests for flag_as_duplicates and the is_duplicate metadata field."""

    @pytest.mark.unit
    def test_flags_specific_container(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        b = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a, b], split="train", skip_duplicates=False)
        db.flag_as_duplicates([0])
        containers = list(db.partition.list_containers())
        assert containers[0].AtomPositionManager.metadata.get("is_duplicate") is True
        assert not containers[1].AtomPositionManager.metadata.get("is_duplicate", False)

    @pytest.mark.unit
    def test_flagged_excluded_from_get_train(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a], split="train", skip_duplicates=False)
        db.flag_as_duplicates([0])
        assert len(db.get_train_atoms()) == 0
        assert len(db.get_train_atoms(exclude_duplicates=False)) == 1


class TestUpdateSplitsPostHoc:
    """Tests for update_splits_post_hoc."""

    @pytest.mark.unit
    def test_tags_matching_structures(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        b = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        # Add without split tags (simulates DB path from initialize_training_set)
        db.add_structures([a, b], skip_duplicates=False)
        n = db.update_splits_post_hoc(train_atoms=[a], test_atoms=[b])
        assert n == 2
        assert len(db.get_train_atoms()) == 1
        assert len(db.get_test_atoms()) == 1

    @pytest.mark.unit
    def test_skips_already_tagged_structures(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a], split="train", skip_duplicates=False)
        # Call again — already tagged, should return 0 updates
        n = db.update_splits_post_hoc(train_atoms=[a], test_atoms=[])
        assert n == 0

    @pytest.mark.unit
    def test_returns_zero_when_no_match(self, tmp_path):
        a = make_atoms(
            ["H", "H"],
            config_type="init_dimer",
            ref_energy=-31.0,
            ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        )
        b = make_atoms(
            ["O", "O"],
            config_type="init_dimer",
            ref_energy=-50.0,
            ref_forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
        )
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([a], skip_duplicates=False)
        # b not in DB — no match possible
        n = db.update_splits_post_hoc(train_atoms=[b], test_atoms=[])
        assert n == 0


class TestApplyTrainTestSplit:
    """Tests for apply_train_test_split — migrating a DB without split tags."""

    def _make_db(self, tmp_path, n: int) -> GlobalDatabase:
        db = GlobalDatabase(str(tmp_path / "db"))
        for k in range(n):
            a = make_atoms(
                ["H", "H"],
                config_type="high_sd",
                ref_energy=-float(k),
                ref_forces=[[0.0, 0.0, float(k)], [0.0, 0.0, -float(k)]],
            )
            db.add_structures([a], skip_duplicates=False)
        return db

    @pytest.mark.unit
    def test_all_structures_tagged(self, tmp_path):
        """Every container gets a split tag; total equals DB size."""
        db = self._make_db(tmp_path, 10)
        n_train, n_test = db.apply_train_test_split(test_fraction=0.2, seed=42)
        assert n_train + n_test == 10

    @pytest.mark.unit
    def test_test_fraction_honoured(self, tmp_path):
        """Approximately test_fraction of structures end up as test."""
        db = self._make_db(tmp_path, 20)
        n_train, n_test = db.apply_train_test_split(test_fraction=0.1, seed=1)
        # 10% of 20 = 2 → at least 1 test (min=1)
        assert n_test == 2
        assert n_train == 18

    @pytest.mark.unit
    def test_at_least_one_test_structure(self, tmp_path):
        """Even with a tiny fraction the result always has ≥1 test structure."""
        db = self._make_db(tmp_path, 5)
        _, n_test = db.apply_train_test_split(test_fraction=0.01, seed=0)
        assert n_test >= 1

    @pytest.mark.unit
    def test_already_tagged_containers_skipped(self, tmp_path):
        """Containers with an existing split tag are left untouched."""
        db = GlobalDatabase(str(tmp_path / "db"))
        a = make_atoms(
            ["H", "H"],
            config_type="high_sd",
            ref_energy=-1.0,
            ref_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        b = make_atoms(
            ["O", "O"],
            config_type="high_sd",
            ref_energy=-2.0,
            ref_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        db.add_structures([a], split="train", skip_duplicates=False)  # already tagged
        db.add_structures([b], skip_duplicates=False)  # untagged

        n_train, n_test = db.apply_train_test_split(test_fraction=0.5, seed=0)
        # only the untagged b was processed
        assert n_train + n_test == 1

    @pytest.mark.unit
    def test_returns_zero_when_all_already_tagged(self, tmp_path):
        """Returns (0, 0) when every container already has a split tag."""
        db = self._make_db(tmp_path, 3)
        db.apply_train_test_split(test_fraction=0.3, seed=0)
        n_train, n_test = db.apply_train_test_split(test_fraction=0.3, seed=0)
        assert (n_train, n_test) == (0, 0)

    @pytest.mark.unit
    def test_split_is_persistent(self, tmp_path):
        """Tags survive a fresh GlobalDatabase handle (on-disk persistence)."""
        db = GlobalDatabase(str(tmp_path / "db"))
        a = make_atoms(
            ["H", "H"],
            config_type="high_sd",
            ref_energy=-1.0,
            ref_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        db.add_structures([a], skip_duplicates=False)
        db.apply_train_test_split(
            test_fraction=0.0, seed=0
        )  # 0% → all train (min 1 test)

        db2 = GlobalDatabase(str(tmp_path / "db"))
        all_splits = [
            c.AtomPositionManager.metadata.get("split")
            for c in db2.partition.list_containers()
        ]
        assert all(s in ("train", "test") for s in all_splits)

    @pytest.mark.unit
    def test_seeded_split_is_reproducible(self, tmp_path):
        """Same seed produces the same train/test assignment."""
        db1 = self._make_db(tmp_path / "a", 10)
        db2 = self._make_db(tmp_path / "b", 10)
        db1.apply_train_test_split(test_fraction=0.3, seed=99)
        db2.apply_train_test_split(test_fraction=0.3, seed=99)

        splits1 = [
            c.AtomPositionManager.metadata.get("split")
            for c in db1.partition.list_containers()
        ]
        splits2 = [
            c.AtomPositionManager.metadata.get("split")
            for c in db2.partition.list_containers()
        ]
        assert splits1 == splits2

    @pytest.mark.unit
    def test_eligible_config_types_restricts_test_set(self, tmp_path):
        """Structures not in eligible_config_types always go to train; only eligible ones can be test."""
        db = GlobalDatabase(str(tmp_path / "db"))
        # 3 eligible, 2 ineligible (IsolatedAtom)
        for ct in ["high_sd", "high_sd", "init_amorphous"]:
            a = make_atoms(
                ["H", "H"],
                config_type=ct,
                ref_energy=-1.0,
                ref_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            )
            db.add_structures([a], skip_duplicates=False)
        for _ in range(2):
            a = make_atoms(
                ["H"],
                config_type="IsolatedAtom",
                ref_energy=-0.5,
                ref_forces=[[0.0, 0.0, 0.0]],
            )
            db.add_structures([a], skip_duplicates=False)

        db.apply_train_test_split(
            test_fraction=0.33,
            seed=0,
            eligible_config_types=["high_sd", "init_amorphous"],
        )

        test_atoms = db.get_test_atoms()
        test_config_types = {a.info.get("config_type") for a in test_atoms}
        # IsolatedAtom must never appear in the test set
        assert "IsolatedAtom" not in test_config_types

    @pytest.mark.unit
    def test_ineligible_structures_always_train(self, tmp_path):
        """Every structure whose config_type is not in eligible_config_types is tagged train."""
        db = GlobalDatabase(str(tmp_path / "db"))
        for ct in ["IsolatedAtom", "init_MP", "high_sd"]:
            a = make_atoms(
                ["H", "H"],
                config_type=ct,
                ref_energy=-1.0,
                ref_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            )
            db.add_structures([a], skip_duplicates=False)

        db.apply_train_test_split(
            test_fraction=0.5,
            seed=0,
            eligible_config_types=["high_sd", "init_amorphous"],
        )

        all_containers = list(db.partition.list_containers())
        for c in all_containers:
            ct = c.AtomPositionManager.metadata.get("config_type")
            split = c.AtomPositionManager.metadata.get("split")
            if ct not in ("high_sd", "init_amorphous"):
                assert split == "train", f"{ct} should be train-only, got {split}"

    @pytest.mark.unit
    def test_no_eligible_structures_all_go_to_train(self, tmp_path):
        """When no untagged structures match eligible_config_types, all go to train, n_test=0."""
        db = GlobalDatabase(str(tmp_path / "db"))
        for _ in range(3):
            a = make_atoms(
                ["H"],
                config_type="IsolatedAtom",
                ref_energy=-0.5,
                ref_forces=[[0.0, 0.0, 0.0]],
            )
            db.add_structures([a], skip_duplicates=False)

        n_train, n_test = db.apply_train_test_split(
            test_fraction=0.3,
            seed=0,
            eligible_config_types=["high_sd"],
        )
        assert n_test == 0
        assert n_train == 3


class TestGlobalDbId:
    """Tests for global_db_id tagging on add_structures and assign_global_db_ids."""

    def _make_atoms(self, energy: float = -1.0) -> "Atoms":
        return make_atoms(
            ["H", "H"],
            config_type="high_sd",
            ref_energy=energy,
            ref_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

    @pytest.mark.unit
    def test_id_in_atoms_info(self, tmp_path):
        """global_db_id appears in atoms.info after a round-trip through the DB."""
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([self._make_atoms()], skip_duplicates=False)
        atoms = db.get_all_as_atoms()
        assert "global_db_id" in atoms[0].info
        assert isinstance(atoms[0].info["global_db_id"], int)

    @pytest.mark.unit
    def test_ids_are_sequential_from_zero(self, tmp_path):
        """N structures added in one batch get IDs 0 … N-1."""
        db = GlobalDatabase(str(tmp_path / "db"))
        n = 5
        db.add_structures(
            [self._make_atoms(float(k)) for k in range(n)], skip_duplicates=False
        )
        ids = {a.info["global_db_id"] for a in db.get_all_as_atoms()}
        assert ids == set(range(n))

    @pytest.mark.unit
    def test_ids_unique_across_batches(self, tmp_path):
        """Two separate add_structures calls produce non-overlapping sequential IDs."""
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures(
            [self._make_atoms(0.0), self._make_atoms(1.0)], skip_duplicates=False
        )
        db.add_structures(
            [self._make_atoms(2.0), self._make_atoms(3.0)], skip_duplicates=False
        )
        ids = [a.info["global_db_id"] for a in db.get_all_as_atoms()]
        assert sorted(ids) == [0, 1, 2, 3]

    @pytest.mark.unit
    def test_id_survives_reload(self, tmp_path):
        """IDs persist after reopening the DB from disk."""
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures(
            [self._make_atoms(0.0), self._make_atoms(1.0)], skip_duplicates=False
        )

        db2 = GlobalDatabase(str(tmp_path / "db"))
        ids = {a.info["global_db_id"] for a in db2.get_all_as_atoms()}
        assert ids == {0, 1}

    @pytest.mark.unit
    def test_assign_global_db_ids_backfills(self, tmp_path):
        """assign_global_db_ids tags containers that have no global_db_id."""
        db = GlobalDatabase(str(tmp_path / "db"))
        # Bypass add_structures to simulate a pre-existing DB without IDs

        for k in range(3):
            sr = GlobalDatabase._prepare_for_storage(self._make_atoms(float(k)))
            db.partition.add([sr])

        n = db.assign_global_db_ids()
        assert n == 3
        ids = {a.info["global_db_id"] for a in db.get_all_as_atoms()}
        assert ids == {0, 1, 2}

    @pytest.mark.unit
    def test_assign_global_db_ids_skips_already_tagged(self, tmp_path):
        """Containers that already have global_db_id are left untouched."""
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures(
            [self._make_atoms(0.0), self._make_atoms(1.0)], skip_duplicates=False
        )
        # All containers already tagged by add_structures
        n = db.assign_global_db_ids()
        assert n == 0

    @pytest.mark.unit
    def test_assign_global_db_ids_idempotent(self, tmp_path):
        """Calling assign_global_db_ids twice leaves IDs unchanged on the second call."""
        db = GlobalDatabase(str(tmp_path / "db"))

        for k in range(2):
            sr = GlobalDatabase._prepare_for_storage(self._make_atoms(float(k)))
            db.partition.add([sr])

        db.assign_global_db_ids()
        ids_after_first = [a.info["global_db_id"] for a in db.get_all_as_atoms()]

        n = db.assign_global_db_ids()
        ids_after_second = [a.info["global_db_id"] for a in db.get_all_as_atoms()]

        assert n == 0
        assert ids_after_first == ids_after_second
