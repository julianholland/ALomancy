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
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
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
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
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
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h_atom.copy()], skip_duplicates=False)
        assert db.size == 2

    @pytest.mark.unit
    def test_returns_added_count(self, tmp_path):
        """3 structures, 1 IsolatedAtom duplicate -> should add 2."""
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        o_atom = make_atoms(["O"], config_type="IsolatedAtom", ref_energy=-432.0,
                           ref_forces=[[0.0, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        count = db.add_structures([h_atom, o_atom, h_atom.copy()])
        assert count == 2

    @pytest.mark.unit
    def test_ref_forces_round_trip(self, tmp_path):
        """REF_forces survive storage and retrieval."""
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2_dimer])
        retrieved = db.get_all_as_atoms()
        assert len(retrieved) == 1
        assert "REF_forces" in retrieved[0].arrays
        np.testing.assert_allclose(retrieved[0].arrays["REF_forces"],
                                   h2_dimer.arrays["REF_forces"], atol=1e-6)

    @pytest.mark.unit
    def test_custom_dedup_list(self, tmp_path):
        """When config_types_to_dedup=["init_dimer"], dimers are deduped."""
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h2_dimer, h2_dimer.copy()],
                           config_types_to_dedup=["init_dimer"])
        assert db.size == 1


class TestCounting:
    """Tests for counting methods."""

    @pytest.mark.unit
    def test_count_all_by_config_type_and_formula(self, tmp_path):
        """Test detailed counting by config_type and formula."""
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h2_dimer, h2_dimer.copy()], skip_duplicates=False)
        counts = db.count_all_by_config_type_and_formula()
        assert counts["IsolatedAtom"]["H"] == 1
        assert counts["init_dimer"]["H2"] == 2

    @pytest.mark.unit
    def test_count_by_config_type(self, tmp_path):
        """Test counting aggregated by config_type only."""
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        o_atom = make_atoms(["O"], config_type="IsolatedAtom", ref_energy=-432.0,
                           ref_forces=[[0.0, 0.0, 0.0]])
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
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
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        o_atom = make_atoms(["O"], config_type="IsolatedAtom", ref_energy=-432.0,
                           ref_forces=[[0.0, 0.0, 0.0]])
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, o_atom, h2_dimer], skip_duplicates=False)
        isolated = db.get_structures_by_config_type(["IsolatedAtom"])
        assert len(isolated) == 2
        assert all(a.info.get("config_type") == "IsolatedAtom" for a in isolated)

    @pytest.mark.unit
    def test_get_all_as_atoms(self, tmp_path):
        """Test retrieving all structures as Atoms objects."""
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        db = GlobalDatabase(str(tmp_path / "db"))
        db.add_structures([h_atom, h2_dimer], skip_duplicates=False)
        all_atoms = db.get_all_as_atoms()
        assert len(all_atoms) == 2
        assert all(isinstance(a, Atoms) for a in all_atoms)

    @pytest.mark.unit
    def test_size_property(self, tmp_path):
        """Test the size property increases correctly."""
        h_atom = make_atoms(["H"], config_type="IsolatedAtom", ref_energy=-13.6,
                           ref_forces=[[0.0, 0.0, 0.0]])
        h2_dimer = make_atoms(["H", "H"], config_type="init_dimer", ref_energy=-31.0,
                             ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
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
