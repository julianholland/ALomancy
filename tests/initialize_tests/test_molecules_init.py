"""initialize/molecules.py: molecules in vacuum and periodic boxes of
molecules at a density range, plus their wiring into
create_initialization_atoms_list / compute_initialization_needs."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase.geometry import get_distances

from alomancy.initialize.initialization_structure_list import (
    compute_initialization_needs,
    create_initialization_atoms_list,
)
from alomancy.initialize.molecules import (
    create_molecule_atoms_list,
    create_molecules_in_box_atoms_list,
    molecule_counts,
)
from alomancy.initialize.stretch_and_compress import create_stretch_compress_atoms_list

_WATER_METHANOL = {"H2O": 2, "smiles-CO": 1}


def _boxes(**overrides):
    kwargs = {
        "molecules_ratio": _WATER_METHANOL,
        "density_range": [0.8, 1.1],
        "num_of_molecules": 12,
        "num_of_structures": 4,
        "min_distance": 1.5,
        "max_attempts": 1000,
        "seed": 803,
        **overrides,
    }
    return create_molecules_in_box_atoms_list(**kwargs)


def _density(atoms):
    return atoms.get_masses().sum() / atoms.get_volume() / 0.6022


@pytest.mark.unit
def test_vacuum_molecule_is_one_non_periodic_equilibrium_structure():
    (atoms,) = create_molecule_atoms_list("smiles-CCO", vacuum=10.0)
    assert atoms.get_chemical_formula() == "C2H6O"
    assert not atoms.pbc.any()
    assert atoms.cell.lengths().min() >= 10.0
    assert atoms.info["config_type"] == "init_molecule"
    assert atoms.info["molecule"] == "smiles-CCO"
    assert atoms.info["needs_relaxation"] is False


@pytest.mark.unit
def test_stretch_compress_of_a_vacuum_molecule_scales_its_bonds():
    (atoms,) = create_molecule_atoms_list("CO", vacuum=10.0)
    bond = atoms.get_distance(0, 1)
    stretched = create_stretch_compress_atoms_list(atoms, 0.2, 3)
    assert [s.get_distance(0, 1) / bond for s in stretched] == pytest.approx(
        [0.8, 1.0, 1.2]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "ratio, total, expected",
    [
        ({"a": 2, "b": 1}, 24, {"a": 16, "b": 8}),
        ({"a": 1, "b": 100}, 5, {"a": 1, "b": 4}),
        ({"a": 1, "b": 1, "c": 1}, 4, {"a": 2, "b": 1, "c": 1}),
    ],
)
def test_counts_follow_the_ratio_with_every_molecule_at_least_once(
    ratio, total, expected
):
    assert molecule_counts(ratio, total) == expected


@pytest.mark.unit
def test_fewer_molecules_than_species_raises():
    with pytest.raises(ValueError, match="fewer than the 3 molecules"):
        molecule_counts({"a": 1, "b": 1, "c": 1}, 2)


@pytest.mark.unit
def test_boxes_hold_the_counts_at_random_exact_densities_within_the_range():
    boxes = _boxes()
    densities = [b.info["density"] for b in boxes]
    assert all(0.8 <= d <= 1.1 for d in densities)
    assert len(set(densities)) == 4
    assert [_density(b) for b in boxes] == pytest.approx(densities, rel=1e-3)
    for box in boxes:
        # 8 H2O + 4 CH3OH
        assert box.get_chemical_formula() == "C4H32O12"
        assert box.pbc.all()
        assert box.info["config_type"] == "init_molecules_in_box"
        assert box.info["box_composition"] == "H2O:2,smiles-CO:1"
        assert box.info["needs_relaxation"] is True


@pytest.mark.unit
def test_atoms_of_different_molecules_keep_min_distance_under_minimum_image():
    box = _boxes(num_of_structures=1, density_range=[1.0, 1.0])[0]
    # Molecules are appended whole: SAGE builds H2O as O,H,H and CH3OH
    # (smiles-CO) as C,O,H,H,H,H, so the first symbol gives each one's size.
    molecule_of, symbols, index = [], box.get_chemical_symbols(), 0
    while index < len(box):
        size = {"O": 3, "C": 6}[symbols[index]]
        molecule_of += [index] * size
        index += size
    molecule_of = np.array(molecule_of)
    _, distances = get_distances(box.positions, cell=box.cell, pbc=True)
    other = molecule_of[:, None] != molecule_of[None, :]
    assert distances[other].min() >= 1.5 - 1e-9


@pytest.mark.unit
def test_same_attempt_same_boxes_and_a_new_attempt_new_ones():
    """attempt = how many boxes the database holds: a restart replacing a box
    that failed DFT draws a new box (new density too), never a copy of one
    already labelled; the same database state gives the same boxes."""
    first, second = _boxes(), _boxes()
    assert all(
        np.allclose(a.positions, b.positions)
        for a, b in zip(first, second, strict=True)
    )
    restarted = _boxes(num_of_structures=1, attempt=3)
    assert len(restarted) == 1
    assert restarted[0].info["density"] not in [b.info["density"] for b in first]
    assert not np.allclose(_boxes(seed=1)[0].positions, first[0].positions)


@pytest.mark.unit
def test_impossible_density_raises():
    with pytest.raises(ValueError, match="could not place every molecule"):
        _boxes(density_range=[5.0, 5.0], num_of_structures=1, max_attempts=50)


@pytest.mark.unit
def test_box_smaller_than_a_molecule_raises():
    with pytest.raises(ValueError, match="too small"):
        _boxes(
            molecules_ratio={"smiles-CCCCCCCC": 1},
            num_of_molecules=1,
            num_of_structures=1,
        )


def _mock_db(all_counts):
    db = MagicMock()
    db.count_all_by_config_type_and_formula.return_value = all_counts
    return db


_BOX = {
    "molecules": _WATER_METHANOL,
    "density_range": [0.8, 1.1],
    "num_of_molecules": 12,
    "num_of_structures": 4,
}


@pytest.mark.unit
def test_needs_count_molecules_and_boxes_per_formula():
    db = _mock_db(
        {"init_molecule": {"CO": 1}, "init_molecules_in_box": {"C4H32O12": 1}}
    )
    needs = compute_initialization_needs(
        db,
        ["C", "H", "O"],
        False,
        False,
        0,
        0,
        0,
        molecules=["CO", "smiles-O"],
        box_compositions=[_BOX],
    )
    assert needs["molecule_override"] == {"H2O": 1}
    assert needs["box_override"] == {"C4H32O12": 3}


@pytest.mark.unit
def test_needs_are_empty_when_the_db_has_everything():
    db = _mock_db(
        {"init_molecule": {"H2O": 1}, "init_molecules_in_box": {"C4H32O12": 4}}
    )
    needs = compute_initialization_needs(
        db, ["C", "H", "O"], False, False, 0, 0, 0, ["smiles-O"], [_BOX]
    )
    assert needs["molecule_override"] == {}
    assert needs["box_override"] == {}


def _create(tmp_path, **kwargs):
    return create_initialization_atoms_list(
        work_dir=str(tmp_path),
        elements=["C", "H", "O"],
        mp_structures=False,
        single_atoms=False,
        num_of_dimers_per_combo=0,
        num_of_trimers_per_combo=0,
        num_of_amorphous_structures=0,
        num_of_stretch_compress_per_target=0,
        **kwargs,
    )


@pytest.mark.unit
def test_create_makes_molecules_and_boxes(tmp_path):
    atoms = _create(tmp_path, molecules=["CO", "smiles-O"], box_compositions=[_BOX])
    types = [a.info["config_type"] for a in atoms]
    assert types.count("init_molecule") == 2
    assert types.count("init_molecules_in_box") == 4
    assert (tmp_path / "initialization_structures_generated.xyz").exists()


@pytest.mark.unit
def test_create_with_overrides_makes_only_what_is_missing(tmp_path):
    atoms = _create(
        tmp_path,
        molecules=["CO", "smiles-O"],
        box_compositions=[_BOX],
        molecule_override={"H2O": 1},
        box_override={"C4H32O12": 1},
    )
    molecules = [a for a in atoms if a.info["config_type"] == "init_molecule"]
    boxes = [a for a in atoms if a.info["config_type"] == "init_molecules_in_box"]
    assert [m.info["molecule"] for m in molecules] == ["smiles-O"]
    # One new box at a random density in the range, not a copy of one of
    # the three the database already holds (attempt 3, not 0).
    assert len(boxes) == 1
    assert 0.8 <= boxes[0].info["density"] <= 1.1
    first_attempt = _boxes(num_of_molecules=_BOX["num_of_molecules"])
    assert boxes[0].info["density"] not in [b.info["density"] for b in first_attempt]


@pytest.mark.unit
def test_molecules_reach_stretch_compress_through_target_config_types(tmp_path):
    atoms = create_initialization_atoms_list(
        work_dir=str(tmp_path),
        elements=["C", "O"],
        mp_structures=False,
        single_atoms=False,
        num_of_dimers_per_combo=0,
        num_of_trimers_per_combo=0,
        num_of_amorphous_structures=0,
        num_of_stretch_compress_per_target=3,
        target_config_types=["init_molecule"],
        molecules=["CO"],
    )
    types = [a.info["config_type"] for a in atoms]
    assert types.count("init_stretch_compress") == 3


_AROMATICS = {"smiles-OB(O)c1ccccc1": "C6H7BO2", "smiles-Brc1ccccc1": "C6H5Br"}
_AROMATIC_ELEMENTS = ["C", "H", "B", "O", "Br"]


@pytest.mark.unit
def test_aromatic_smiles_molecules_validate_and_generate(tmp_path, monkeypatch):
    from alomancy.initialize.initialiser_interface import generate, validate_kwargs

    monkeypatch.chdir(tmp_path)
    config = {
        "isolated_atom_kwargs": {"enabled": False},
        "dimer_kwargs": {"enabled": False},
        "trimer_kwargs": {"enabled": False},
        "amorphous_kwargs": {"enabled": False},
        "mp_kwargs": {"enabled": False},
        "molecule_kwargs": {"enabled": True, "molecules": list(_AROMATICS)},
    }
    validate_kwargs(config, elements=_AROMATIC_ELEMENTS)
    atoms = generate(
        config, base_name="init", name="initialization", elements=_AROMATIC_ELEMENTS
    )
    assert {a.info["molecule"]: a.get_chemical_formula() for a in atoms} == _AROMATICS
    assert all(a.info["config_type"] == "init_molecule" for a in atoms)


@pytest.mark.unit
def test_aromatic_smiles_pack_into_a_box():
    from collections import Counter

    from alomancy.utils.molecules import molecule_symbols

    (box,) = _boxes(
        molecules_ratio=dict.fromkeys(_AROMATICS, 1),
        num_of_molecules=6,
        num_of_structures=1,
        density_range=[1.0, 1.0],
    )
    expected = Counter()
    for name in _AROMATICS:
        for symbol, count in molecule_symbols(name).items():
            expected[symbol] += 3 * count
    assert Counter(box.get_chemical_symbols()) == expected
    assert _density(box) == pytest.approx(1.0, rel=1e-3)
