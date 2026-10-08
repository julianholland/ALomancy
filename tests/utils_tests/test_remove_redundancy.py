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


def _distinct_s2(n: int) -> list[Atoms]:
    return [_make_s2([[0.0, 0.0, 0.0], [1.5 + 0.5 * i, 0.0, 0.0]]) for i in range(n)]


def _count_descriptor_calls(monkeypatch) -> list[int]:
    """Wrap make_char_vec so each call is counted (real computation still runs)."""
    from alomancy.utils import remove_redundancy

    calls: list[int] = []
    real = remove_redundancy.make_char_vec

    def counting(structure, dimensions=128):
        calls.append(1)
        return real(structure, dimensions=dimensions)

    monkeypatch.setattr(remove_redundancy, "make_char_vec", counting)
    return calls


@pytest.mark.unit
def test_descriptors_cached_in_db_and_reused(tmp_path, monkeypatch):
    """A second call (next loop / restart, fresh GlobalDatabase handle)
    computes no descriptors; only newly added structures get computed."""
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    db_path = str(tmp_path / "db")
    db = GlobalDatabase(db_path)
    db.add_structures(_distinct_s2(4), split="train", skip_duplicates=False)
    calls = _count_descriptor_calls(monkeypatch)

    remove_redundancy_from_partition(db, config_list=["init_amorphous"])
    assert len(calls) == 4

    calls.clear()
    reopened = GlobalDatabase(db_path)
    remove_redundancy_from_partition(reopened, config_list=["init_amorphous"])
    assert len(calls) == 0

    reopened.add_structures(
        [_make_s2([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])],
        split="train",
        skip_duplicates=False,
    )
    remove_redundancy_from_partition(reopened, config_list=["init_amorphous"])
    assert len(calls) == 1


@pytest.mark.unit
def test_cached_descriptor_matches_fresh_computation(tmp_path):
    from alomancy.global_descriptor.atomic_distance_descriptor import make_char_vec
    from alomancy.utils.remove_redundancy import (
        descriptor_key,
        remove_redundancy_from_partition,
    )

    structures = _distinct_s2(3)
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(structures, split="train", skip_duplicates=False)
    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    containers = list(GlobalDatabase(str(tmp_path / "db")).partition.list_containers())
    assert len(containers) == len(structures)
    for container in containers:
        apm = container.AtomPositionManager
        np.testing.assert_allclose(
            apm.metadata[descriptor_key(128)], make_char_vec(apm, dimensions=128)
        )


@pytest.mark.unit
def test_cached_descriptors_never_reach_atoms_info(tmp_path):
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(_distinct_s2(3), split="train", skip_duplicates=False)
    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    for atoms in db.get_train_atoms(exclude_duplicates=False):
        assert not any(key.startswith("char_vec") for key in atoms.info)


@pytest.mark.unit
def test_descriptor_cache_is_dimension_specific(tmp_path, monkeypatch):
    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(_distinct_s2(3), split="train", skip_duplicates=False)
    remove_redundancy_from_partition(db, config_list=["init_amorphous"])

    calls = _count_descriptor_calls(monkeypatch)
    remove_redundancy_from_partition(db, config_list=["init_amorphous"], dimensions=64)
    assert len(calls) == 3


@pytest.mark.unit
def test_probe_record_is_saved_for_the_report(tmp_path, capsys):
    """probe_path gets the full sweep of unique structures against
    tolerance, the plateaus and the outcome; the library's plateau-log
    print no longer reaches stdout."""
    import json

    from alomancy.utils.remove_redundancy import remove_redundancy_from_partition

    base = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    near1 = [[0.0001, 0.0, 0.0], [2.0001, 0.0, 0.0]]
    near2 = [[0.0002, 0.0, 0.0], [2.0002, 0.0, 0.0]]
    dist1 = [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]]
    dist2 = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    db = GlobalDatabase(str(tmp_path / "db"))
    db.add_structures(
        [_make_s2(p) for p in (base, near1, near2, dist1, dist2)],
        split="train",
        skip_duplicates=False,
    )
    probe_path = tmp_path / "al_loop_0" / "redundancy_probe.json"

    remove_redundancy_from_partition(
        db, config_list=["init_amorphous"], probe_path=probe_path
    )

    record = json.loads(probe_path.read_text())
    assert record["n_structures"] == 5
    assert record["tolerances"] == sorted(record["tolerances"])
    assert len(record["tolerances"]) == len(record["unique_counts"]) > 1
    assert record["descriptor"] == {
        "key": "char_vec_128",
        "dimensions": 128,
        "metric": "euclidean",
    }
    if record["outcome"] == "applied":
        assert record["plateaus"]
        assert record["chosen_tolerance"] is not None
        assert record["n_flagged"] == 5 - len(db.get_train_atoms())
    else:
        assert record["chosen_tolerance"] is None
        assert record["n_flagged"] == 0
    assert "plateau log" not in capsys.readouterr().out
