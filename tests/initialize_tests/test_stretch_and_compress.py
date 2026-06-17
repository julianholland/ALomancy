"""Tests for stretch_and_compress initialization function."""

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
class TestCreateStretchCompressAtomsList:
    def test_returns_correct_count(self):
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.1, 5)
        assert len(result) == 5

    def test_zero_structures_returns_empty(self):
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.1, 0)
        assert result == []

    def test_config_type(self):
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.2, 4)
        assert all(a.info["config_type"] == "init_stretch_compress" for a in result)

    def test_needs_relaxation_false(self):
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.2, 4)
        assert all(a.info["needs_relaxation"] is False for a in result)

    def test_deformation_stored_in_info(self):
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.2, 4)
        assert all("deformation" in a.info for a in result)

    def test_cell_varies_with_true(self):
        """With deform_xyz=True, each structure should have a different cell volume."""
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.3, 5)
        volumes = [a.get_volume() for a in result]
        assert len(set(volumes)) > 1

    def test_cell_unchanged_with_false(self):
        """With deform_xyz=False, all cells should equal the original."""
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        orig_vol = atoms.get_volume()
        result = create_stretch_compress_atoms_list(atoms, False, 0.3, 5)
        for a in result:
            assert a.get_volume() == pytest.approx(orig_vol)

    def test_list_deform_xyz_behaves_as_true(self):
        """Per the implementation, a non-empty list is truthy and triggers uniform scaling."""
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result_list = create_stretch_compress_atoms_list(atoms, [True, False, False], 0.2, 5)
        result_true = create_stretch_compress_atoms_list(atoms, True, 0.2, 5)
        # Both should produce the same cells because bool([...]) is True
        for a_l, a_t in zip(result_list, result_true):
            np.testing.assert_allclose(a_l.cell.array, a_t.cell.array)

    def test_original_atoms_not_modified(self):
        """Source atoms object should not be mutated."""
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        orig_cell = atoms.cell.array.copy()
        create_stretch_compress_atoms_list(atoms, True, 0.5, 5)
        np.testing.assert_allclose(atoms.cell.array, orig_cell)

    def test_atoms_scaled_with_cell(self):
        """Atom positions should scale proportionally with the cell."""
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.1, 3)
        # The middle structure (index 1) should have scale ~1.0 (no deformation)
        mid = result[1]
        # Scaled atoms: fractional coords should be preserved
        orig_frac = atoms.get_scaled_positions()
        mid_frac = mid.get_scaled_positions()
        np.testing.assert_allclose(mid_frac, orig_frac, atol=1e-10)

    def test_single_structure(self):
        from alomancy.initialize.stretch_and_compress import (
            create_stretch_compress_atoms_list,
        )
        atoms = _make_h2o()
        result = create_stretch_compress_atoms_list(atoms, True, 0.2, 1)
        assert len(result) == 1
