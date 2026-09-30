"""Unit tests for utils/split_filter.py -- general.train_filter /
general.test_filter quality filters (forces, formation energy per atom)."""

import logging

import numpy as np
import pytest
from ase import Atoms

from alomancy.database.global_database import GlobalDatabase
from alomancy.utils.split_filter import (
    apply_split_filter,
    resolve_split_filter,
    validate_split_filter,
)

# Isolated-atom reference energies stored in the DB for formation energies.
_E0 = {"S": -5.0}


def _s2(max_force: float = 1.0, energy: float = -10.0, x: float = 2.0) -> Atoms:
    """S2 with force (max_force, 0, 0) on both atoms. Formation energy per
    atom vs _E0 is (energy - 2 * -5) / 2 = (energy + 10) / 2."""
    atoms = Atoms("S2", positions=[[0, 0, 0], [x, 0, 0]], cell=[10.0] * 3, pbc=True)
    atoms.info["config_type"] = "high_sd"
    atoms.info["REF_energy"] = energy
    atoms.arrays["REF_forces"] = np.array([[max_force, 0, 0], [max_force, 0, 0]])
    return atoms


def _isolated_s() -> Atoms:
    atoms = Atoms("S", positions=[[0, 0, 0]], cell=[10.0] * 3, pbc=True)
    atoms.info["config_type"] = "IsolatedAtom"
    atoms.info["REF_energy"] = _E0["S"]
    atoms.arrays["REF_forces"] = np.zeros((1, 3))
    return atoms


@pytest.fixture
def db(tmp_path):
    database = GlobalDatabase(str(tmp_path / "db"))
    database.add_structures([_isolated_s()], split="train")
    return database


def _energies(atoms_list):
    # The DB returns REF_energy as a one-element array.
    return sorted(
        round(float(np.asarray(a.info["REF_energy"]).reshape(-1)[0]), 6)
        for a in atoms_list
    )


# -- defaults ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("cfg", [None, {}, {"max_force": None}])
def test_test_filter_off_by_default(db, cfg):
    db.add_structures([_s2(500.0), _s2(1.0)], split="test", skip_duplicates=False)

    apply_split_filter(db, "test", resolve_split_filter(cfg, "test"))

    assert len(db.get_test_atoms()) == 2


@pytest.mark.unit
def test_train_default_excludes_high_force_only(db):
    db.add_structures(
        [_s2(150.0), _s2(50.0, energy=500.0)], split="train", skip_duplicates=False
    )

    counts = apply_split_filter(db, "train", resolve_split_filter(None, "train"))

    kept = [a for a in db.get_train_atoms() if a.info["config_type"] == "high_sd"]
    # the 500 eV structure has an absurd formation energy but the train
    # energy window is off by default
    assert _energies(kept) == [500.0]
    assert counts["high_force"] == 1


@pytest.mark.unit
def test_explicit_null_disables_train_default(db):
    db.add_structures([_s2(150.0)], split="train", skip_duplicates=False)

    apply_split_filter(db, "train", resolve_split_filter({"max_force": None}, "train"))

    assert len(db.get_train_atoms()) == 2  # isolated atom + the 150 eV/Å one


# -- force criterion ----------------------------------------------------------


@pytest.mark.unit
def test_max_force_boundary_is_inclusive(db):
    db.add_structures(
        [_s2(100.0, energy=-11.0), _s2(99.9, energy=-12.0)],
        split="test",
        skip_duplicates=False,
    )

    apply_split_filter(db, "test", {"max_force": 100.0})

    assert _energies(db.get_test_atoms()) == [-12.0]


@pytest.mark.unit
def test_filter_only_touches_its_own_split(db):
    db.add_structures([_s2(500.0, energy=-11.0)], split="train", skip_duplicates=False)
    db.add_structures([_s2(500.0, energy=-12.0)], split="test", skip_duplicates=False)

    apply_split_filter(db, "test", {"max_force": 100.0})

    assert db.get_test_atoms() == []
    assert -11.0 in _energies(db.get_train_atoms())


@pytest.mark.unit
def test_filtered_structures_stay_in_archive(db):
    db.add_structures([_s2(500.0)], split="test", skip_duplicates=False)

    apply_split_filter(db, "test", {"max_force": 100.0})

    assert db.size == 2
    assert len(db.get_test_atoms(exclude_quality_filtered=False)) == 1
    (flagged,) = db.get_test_atoms(exclude_quality_filtered=False)
    assert flagged.info["quality_filter_reasons"] == ["high_force"]


@pytest.mark.unit
def test_invalid_forces_are_excluded(db):
    atoms = _s2()
    atoms.arrays["REF_forces"] = np.array([[np.nan, 0, 0], [0, 0, 0]])
    db.add_structures([atoms], split="test", skip_duplicates=False)

    counts = apply_split_filter(db, "test", {"max_force": 100.0})

    assert counts["invalid_forces"] == 1
    assert db.get_test_atoms() == []


# -- formation energy criterion -----------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("window", "kept"),
    [
        # E_f/atom of the three structures: -12 eV -> -1.0, -10 -> 0.0, -4 -> 3.0
        ([-1.0, 1.0], [-12.0, -10.0]),  # bounds are inclusive
        ([-0.5, 1.0], [-10.0]),
        ([None, 1.0], [-12.0, -10.0]),
        ([-0.5, None], [-10.0, -4.0]),
    ],
)
def test_formation_energy_window(db, window, kept):
    db.add_structures(
        [_s2(energy=-12.0, x=2.0), _s2(energy=-10.0, x=2.1), _s2(energy=-4.0, x=2.2)],
        split="test",
        skip_duplicates=False,
    )

    apply_split_filter(db, "test", {"formation_energy_per_atom": window})

    assert _energies(db.get_test_atoms()) == kept


@pytest.mark.unit
def test_formation_energy_recorded(db):
    db.add_structures([_s2(energy=-12.0)], split="test", skip_duplicates=False)

    apply_split_filter(db, "test", {"formation_energy_per_atom": [None, 10.0]})

    (atoms,) = db.get_test_atoms()
    value = atoms.info["REF_formation_energy_per_atom_e0"]
    assert isinstance(value, float)
    assert value == pytest.approx(-1.0)


@pytest.mark.unit
def test_missing_isolated_atom_energy_skips_energy_filter(tmp_path):
    db = GlobalDatabase(str(tmp_path / "db"))  # no IsolatedAtom stored
    db.add_structures([_s2(energy=1000.0)], split="test", skip_duplicates=False)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    module_logger = logging.getLogger("alomancy.utils.split_filter")
    module_logger.addHandler(handler)
    try:
        apply_split_filter(db, "test", {"formation_energy_per_atom": [None, 1.0]})
    finally:
        module_logger.removeHandler(handler)

    assert len(db.get_test_atoms()) == 1
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "['S']" in warnings[0]


# -- switching and independence ---------------------------------------------------


@pytest.mark.unit
def test_switching_off_restores_structures(db):
    db.add_structures([_s2(500.0)], split="test", skip_duplicates=False)
    apply_split_filter(db, "test", {"max_force": 100.0})
    assert db.get_test_atoms() == []

    apply_split_filter(db, "test", resolve_split_filter(None, "test"))

    assert len(db.get_test_atoms()) == 1


@pytest.mark.unit
def test_train_and_test_filters_independent(db):
    db.add_structures([_s2(60.0, energy=-11.0)], split="train", skip_duplicates=False)
    db.add_structures([_s2(60.0, energy=-12.0)], split="test", skip_duplicates=False)

    apply_split_filter(db, "train", {"max_force": 100.0})
    apply_split_filter(db, "test", {"max_force": 50.0})

    assert -11.0 in _energies(db.get_train_atoms())
    assert db.get_test_atoms() == []


@pytest.mark.unit
def test_legacy_is_high_force_flag_ignored(db):
    legacy = _s2(1.0)
    legacy.info["is_high_force"] = True
    db.add_structures([legacy], split="train", skip_duplicates=False)

    assert len(db.get_train_atoms()) == 2


# -- validation ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        ({"max_forc": 1.0}, "Unknown general.test_filter"),
        ({"max_force": 0}, "positive number"),
        ({"max_force": -5.0}, "positive number"),
        ({"max_force": float("inf")}, "positive number"),
        ({"max_force": True}, "positive number"),
        ({"formation_energy_per_atom": 1.0}, r"\[min, max\]"),
        ({"formation_energy_per_atom": [1.0]}, r"\[min, max\]"),
        ({"formation_energy_per_atom": [2.0, 1.0]}, "above its maximum"),
        ({"formation_energy_per_atom": ["a", 1.0]}, "finite numbers or null"),
        ("strict", "must be a mapping"),
    ],
)
def test_validate_rejects_bad_config(cfg, match):
    with pytest.raises(ValueError, match=match):
        validate_split_filter(cfg, "test_filter")


@pytest.mark.unit
@pytest.mark.parametrize(
    "cfg",
    [
        None,
        {},
        {"max_force": 50},
        {"formation_energy_per_atom": [None, 1.0]},
        {"max_force": None, "formation_energy_per_atom": [-2.0, 2.0]},
    ],
)
def test_validate_accepts_good_config(cfg):
    validate_split_filter(cfg, "train_filter")
