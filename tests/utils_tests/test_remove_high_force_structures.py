"""Unit tests for remove_high_force_structures_from_partition."""

import numpy as np
import pytest
from ase import Atoms

from alomancy.database.global_database import GlobalDatabase
from alomancy.utils.remove_high_force_structures import (
    remove_high_force_structures_from_partition,
)


def _make_s2(max_force: float, ref_energy: float = -10.0) -> Atoms:
    """Build an S2 Atoms with a uniform force of magnitude max_force on each atom."""
    atoms = Atoms(
        symbols=["S", "S"],
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.info["config_type"] = "high_sd"
    atoms.info["REF_energy"] = ref_energy
    # Force vector (max_force, 0, 0) on both atoms
    atoms.arrays["REF_forces"] = np.array([[max_force, 0.0, 0.0], [max_force, 0.0, 0.0]])
    return atoms


@pytest.mark.unit
def test_high_force_structures_flagged(tmp_path):
    """Structures exceeding the threshold are flagged; low-force ones are not."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(150.0), _make_s2(50.0), _make_s2(200.0)],
        split="train",
        skip_duplicates=False,
    )

    remove_high_force_structures_from_partition(db, force_threshold=100.0)

    train = db.get_train_atoms()
    assert len(train) == 1
    assert abs(train[0].arrays["REF_forces"][0, 0] - 50.0) < 1e-6


@pytest.mark.unit
def test_high_force_structures_kept_in_archive(tmp_path):
    """Flagged structures are never deleted — get_train_atoms(exclude_high_force=False) returns all."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(150.0), _make_s2(50.0)],
        split="train",
        skip_duplicates=False,
    )

    remove_high_force_structures_from_partition(db, force_threshold=100.0)

    assert db.size == 2
    assert len(db.get_train_atoms(exclude_high_force=False)) == 2


@pytest.mark.unit
def test_exact_threshold_boundary(tmp_path):
    """A structure with max force exactly at the threshold is flagged (>=)."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(100.0), _make_s2(99.9)],
        split="train",
        skip_duplicates=False,
    )

    remove_high_force_structures_from_partition(db, force_threshold=100.0)

    train = db.get_train_atoms()
    assert len(train) == 1
    assert abs(train[0].arrays["REF_forces"][0, 0] - 99.9) < 0.01


@pytest.mark.unit
def test_test_split_structures_unaffected(tmp_path):
    """Structures tagged as test are never flagged regardless of their forces."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures([_make_s2(500.0)], split="test", skip_duplicates=False)
    db.add_structures([_make_s2(50.0)], split="train", skip_duplicates=False)

    remove_high_force_structures_from_partition(db, force_threshold=100.0)

    # Test structure stays in test set; train structure is unaffected (below threshold)
    assert len(db.get_test_atoms()) == 1
    assert len(db.get_train_atoms()) == 1


@pytest.mark.unit
def test_empty_train_split_no_error(tmp_path):
    """No train-split structures → function returns without raising."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures([_make_s2(500.0)], split="test", skip_duplicates=False)
    # Should not raise
    remove_high_force_structures_from_partition(db, force_threshold=100.0)


@pytest.mark.unit
def test_all_below_threshold_none_flagged(tmp_path):
    """When all structures are below the threshold, nothing is flagged."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(10.0), _make_s2(20.0), _make_s2(30.0)],
        split="train",
        skip_duplicates=False,
    )

    remove_high_force_structures_from_partition(db, force_threshold=100.0)

    assert len(db.get_train_atoms()) == 3


@pytest.mark.unit
def test_flag_as_high_force_method(tmp_path):
    """GlobalDatabase.flag_as_high_force sets is_high_force on the correct container."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(10.0), _make_s2(200.0), _make_s2(30.0)],
        split="train",
        skip_duplicates=False,
    )

    db.flag_as_high_force([1])  # flag the second container (index 1)

    train = db.get_train_atoms()
    assert len(train) == 2
    # The unflagged ones have forces 10 and 30
    force_mags = sorted(abs(a.arrays["REF_forces"][0, 0]) for a in train)
    assert force_mags == pytest.approx([10.0, 30.0])


@pytest.mark.unit
def test_exclude_high_force_false_returns_all(tmp_path):
    """get_train_atoms(exclude_high_force=False) returns all train structures including flagged."""
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(200.0), _make_s2(50.0)],
        split="train",
        skip_duplicates=False,
    )
    remove_high_force_structures_from_partition(db, force_threshold=100.0)

    assert len(db.get_train_atoms(exclude_high_force=True)) == 1
    assert len(db.get_train_atoms(exclude_high_force=False)) == 2
