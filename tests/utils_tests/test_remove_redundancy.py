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
    """Structures whose config_type is NOT in config_list are never flagged.

    Only 2 structures match config_list here -- too few for
    NaturalTolerancePlateauProbe to probe a tolerance at all (it needs at
    least datapoints_to_calculate_gradient=3 probe steps), so
    remove_redundancy_from_partition skips flagging entirely rather than
    crashing. Both init_amorphous structures survive; this isn't a real
    dedup decision, just "not enough data to make one"."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    near = [[0.0001, 0.0, 0.0], [2.0001, 0.0, 0.0]]

    always_train = _make_s2(base)
    always_train.info["config_type"] = "IsolatedAtom"

    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures([always_train], split="train", skip_duplicates=False)
    db.add_structures(
        [_make_s2(base), _make_s2(near)], split="train", skip_duplicates=False
    )

    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    train = db.get_train_atoms()
    # 1 IsolatedAtom (unaffected) + 2 init_amorphous (neither flagged: too
    # few structures to probe a tolerance) = 3
    assert len(train) == 3


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
def test_no_plateau_found_skips_flagging(tmp_path, monkeypatch):
    """When the tolerance probe runs but finds no plateau, it emits a
    "No plateaus found" warning and returns an arbitrary fallback tolerance
    (per deduplicate_lib's own calculate_tolerance implementation) -- this
    must not be used to flag duplicates; remove_redundancy_from_partition
    should skip flagging entirely instead."""
    import warnings

    from deduplicate_lib.plugins.tolerance_calculators import (
        natural_tolerance_plateau_probe,
    )

    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    near1 = [[0.0001, 0.0, 0.0], [2.0001, 0.0, 0.0]]
    near2 = [[0.0002, 0.0, 0.0], [2.0002, 0.0, 0.0]]

    structures = [_make_s2(base), _make_s2(near1), _make_s2(near2)]
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(structures, split="train", skip_duplicates=False)

    def _fake_calculate_tolerance(self, condition="longest"):
        warnings.warn(
            "No plateaus found in tolerance probe. Consider adding in "
            "perturbed structures and/or increasing dataset size and/or "
            "increaseing probe steps.\nReturning average of all same and "
            "all different tolerance as fallback.",
            stacklevel=2,
        )
        return 0.5

    monkeypatch.setattr(
        natural_tolerance_plateau_probe.NaturalTolerancePlateauProbe,
        "calculate_tolerance",
        _fake_calculate_tolerance,
    )

    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    # Nothing flagged despite near1/near2 being near-duplicates of base --
    # the fallback tolerance is never applied.
    assert len(db.get_train_atoms()) == 3


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
