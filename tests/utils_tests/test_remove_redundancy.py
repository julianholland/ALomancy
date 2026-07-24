"""Unit tests for remove_redundancy_from_partition."""

import numpy as np
import pytest
from ase import Atoms

from alomancy.database.global_database import GlobalDatabase


def _make_s2(positions, ref_energy=-10.0):
    """Build an S2 Atoms with given positions and a REF_energy."""
    atoms = Atoms(
        symbols=["S", "S"],
        positions=positions,
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.info["config_type"] = "init_amorphous"
    atoms.info["REF_energy"] = ref_energy
    atoms.arrays["REF_forces"] = np.zeros((2, 3))
    return atoms


@pytest.mark.unit
def test_near_duplicates_flagged(tmp_path):
    """3 near-identical + 2 distinct: near1 and near2 flagged, base kept as representative → 3 unique."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    near1 = [[0.0001, 0.0, 0.0], [2.0001, 0.0, 0.0]]
    near2 = [[0.0002, 0.0, 0.0], [2.0002, 0.0, 0.0]]
    dist1 = [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]]  # different bond length
    dist2 = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]  # clearly different

    structures = [
        _make_s2(base),
        _make_s2(near1),
        _make_s2(near2),
        _make_s2(dist1),
        _make_s2(dist2),
    ]
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(structures, split="train", skip_duplicates=False)

    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    # DistanceMatrix keeps one representative per near-duplicate group:
    # base is kept; near1 and near2 are flagged; dist1 and dist2 are distinct.
    unique = db.get_train_atoms()
    assert len(unique) == 3  # base + dist1 + dist2


@pytest.mark.unit
def test_all_structures_kept_in_archive(tmp_path):
    """Flagged duplicates are still in DB; exclude_duplicates=False returns all."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    near = [[0.0001, 0.0, 0.0], [2.0001, 0.0, 0.0]]
    dist = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]

    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(base), _make_s2(near), _make_s2(dist)],
        split="train",
        skip_duplicates=False,
    )
    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    assert db.size == 3
    assert len(db.get_train_atoms(exclude_duplicates=False)) == 3


@pytest.mark.unit
def test_non_config_list_structures_unaffected(tmp_path):
    """Structures whose config_type is NOT in config_list are never flagged."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    near = [[0.0001, 0.0, 0.0], [2.0001, 0.0, 0.0]]

    always_train = _make_s2(base)
    always_train.info["config_type"] = "IsolatedAtom"

    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures([always_train], split="train", skip_duplicates=False)
    db.add_structures([_make_s2(base), _make_s2(near)], split="train", skip_duplicates=False)

    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    train = db.get_train_atoms()
    # 1 IsolatedAtom (unaffected) + 1 unique init_amorphous = 2
    assert len(train) == 2


@pytest.mark.unit
def test_empty_train_split_no_error(tmp_path):
    """No train-split structures → function returns without raising."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    db = GlobalDatabase(str(tmp_path / "db"))
    a = _make_s2([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    db.add_structures([a], split="test", skip_duplicates=False)
    # Should not raise
    remove_redundancy_from_partition(db, config_list=["init_amorphous"])


@pytest.mark.unit
def test_config_list_not_in_train_no_error(tmp_path):
    """config_list doesn't match any train structures → function returns without raising."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    db = GlobalDatabase(str(tmp_path / "db"))
    a = _make_s2([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    db.add_structures([a], split="train", skip_duplicates=False)
    # config_list has no overlap with "init_amorphous"
    remove_redundancy_from_partition(db, config_list=["high_sd"])
    # Structure should be unaffected
    assert len(db.get_train_atoms()) == 1
