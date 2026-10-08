"""drop_unwritable_info: empty atoms.info values and extxyz round trips."""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from alomancy.utils.clean_structures import (
    drop_unwritable_info,
    recover_swallowed_model_energy,
)

_EMPTY_VALUES = {
    "empty_list": [],
    "empty_tuple": (),
    "empty_array": np.array([]),
    "empty_string": "",
}


def _atoms(key: str, value) -> Atoms:
    atoms = Atoms("H", cell=[3.0] * 3, pbc=True)
    atoms.info[key] = value
    # Written straight after the empty value, so it is the token a bare
    # "key=" would swallow on read.
    atoms.info["model_energy"] = -1.5
    return atoms


def _twice_through_extxyz(atoms: Atoms, tmp_path, clean: bool) -> Atoms:
    """write -> read -> write -> read, as the AL loop does (DB export, then
    split/pred files written from what was read)."""
    for step in ("first", "second"):
        if clean:
            drop_unwritable_info(atoms)
        path = tmp_path / f"{step}.xyz"
        write(path, atoms, format="extxyz")
        atoms = read(path, format="extxyz")
    return atoms


@pytest.mark.unit
def test_empty_list_swallows_next_key_without_cleaning(tmp_path):
    """The mechanism behind the missing test parity plots. If a future ASE
    round-trips empty lists, this fails and drop_unwritable_info can go."""
    result = _twice_through_extxyz(_atoms("reasons", []), tmp_path, clean=False)
    assert "model_energy" not in result.info
    assert result.info["reasons"] == "model_energy=-1.5"


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(_EMPTY_VALUES))
def test_next_key_survives_after_cleaning(tmp_path, key):
    result = _twice_through_extxyz(
        _atoms(key, _EMPTY_VALUES[key]), tmp_path, clean=True
    )
    assert result.info["model_energy"] == pytest.approx(-1.5)
    assert key not in result.info


@pytest.mark.unit
def test_keeps_non_empty_values():
    atoms = Atoms("H")
    atoms.info.update(
        {"reasons": ["high_force"], "zero": 0, "flag": False, "name": "x"}
    )
    drop_unwritable_info(atoms)
    assert atoms.info == {
        "reasons": ["high_force"],
        "zero": 0,
        "flag": False,
        "name": "x",
    }


@pytest.mark.unit
def test_swallowed_model_energy_is_recovered_exactly(tmp_path):
    """Files written before the fix still hold the value, inside the key
    before it; reading it back needs no re-evaluation."""
    damaged = _twice_through_extxyz(_atoms("reasons", []), tmp_path, clean=False)
    assert "model_energy" not in damaged.info

    assert recover_swallowed_model_energy(damaged) is True
    assert damaged.info["model_energy"] == pytest.approx(-1.5)
    assert "reasons" not in damaged.info


@pytest.mark.unit
def test_intact_structures_are_left_alone():
    atoms = Atoms("H")
    atoms.info.update(model_energy=-2.0, note="model_energy=-9")
    assert recover_swallowed_model_energy(atoms) is False
    assert atoms.info == {"model_energy": -2.0, "note": "model_energy=-9"}
