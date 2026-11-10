"""Common single-element crystal structures (initialize/crystals.py,
initialization.crystal_structures_kwargs, config_type init_crystal)."""

import logging

import numpy as np
import pytest
from ase import Atoms

from alomancy.database.global_database import GlobalDatabase
from alomancy.initialize import initialiser_interface
from alomancy.initialize.crystals import (
    CRYSTAL_STRUCTURES,
    IDEAL_HCP_C_OVER_A,
    build_crystal,
    create_crystal_atoms_list,
    default_elements,
    reference_volume_per_atom,
)
from alomancy.initialize.initialiser_interface import (
    compute_needs,
    kwargs_defaults,
    validate_kwargs,
)
from alomancy.utils.dataset_curation import structure_domain
from alomancy.utils.split_filter import apply_split_filter

_ATOMS = {"fcc": 1, "bcc": 1, "hcp": 2, "sc": 1, "diamond": 2, "a15": 8, "omega": 3}


@pytest.mark.unit
def test_every_structure_has_the_reference_volume_per_atom():
    volume = reference_volume_per_atom("Pt")
    assert volume == pytest.approx(3.92**3 / 4)
    for structure in CRYSTAL_STRUCTURES:
        atoms = build_crystal("Pt", structure)
        assert len(atoms) == _ATOMS[structure]
        assert atoms.get_volume() / len(atoms) == pytest.approx(volume, abs=1e-6)
        assert atoms.pbc.all()
        assert atoms.info["config_type"] == "init_crystal"
        assert atoms.info["crystal_structure"] == structure
        assert atoms.info["needs_relaxation"] is False
        assert structure_domain(atoms) == "bulk"
    fcc = build_crystal("Pt", "fcc")
    # The reference structure comes out at the reference lattice constant.
    assert np.linalg.norm(fcc.cell[0]) * np.sqrt(2) == pytest.approx(3.92)
    assert fcc.info["lattice_source"] == "reference"
    assert build_crystal("Pt", "bcc").info["lattice_source"] == "equal_volume"


@pytest.mark.unit
def test_hcp_c_over_a():
    ti = build_crystal("Ti", "hcp")
    assert ti.cell.cellpar()[2] / ti.cell.cellpar()[0] == pytest.approx(1.588)
    assert ti.info["lattice_source"] == "reference"
    pt = build_crystal("Pt", "hcp")
    assert pt.cell.cellpar()[2] / pt.cell.cellpar()[0] == pytest.approx(
        IDEAL_HCP_C_OVER_A
    )


@pytest.mark.unit
def test_user_values_override_in_order():
    fcc = build_crystal("Pt", "fcc", lattice=3.97)
    assert np.linalg.norm(fcc.cell[0]) * np.sqrt(2) == pytest.approx(3.97)
    assert fcc.info["lattice_source"] == "user"
    hcp = build_crystal("Pt", "hcp", lattice={"a": 2.8, "c": 4.6})
    assert hcp.cell.cellpar()[0] == pytest.approx(2.8)
    assert hcp.cell.cellpar()[2] == pytest.approx(4.6)
    bcc = build_crystal("Pt", "bcc", volume_per_atom=16.0)
    assert bcc.get_volume() == pytest.approx(16.0)
    assert bcc.info["lattice_source"] == "user"
    # lattice_constants win over volume_per_atom for that structure only.
    built = create_crystal_atoms_list(
        ["Pt"],
        ["fcc", "bcc"],
        lattice_constants={"Pt": {"fcc": 3.97}},
        volume_per_atom={"Pt": 16.0},
    )
    fcc, bcc = built
    assert np.linalg.norm(fcc.cell[0]) * np.sqrt(2) == pytest.approx(3.97)
    assert bcc.get_volume() == pytest.approx(16.0)


@pytest.mark.unit
def test_reference_volumes_beyond_ase_bulk():
    assert reference_volume_per_atom("Mn") == pytest.approx(8.89**3 / 58)
    assert reference_volume_per_atom("Ga") == pytest.approx(4.51**3 * 1.001 * 1.695 / 8)
    assert reference_volume_per_atom("Pm") is None
    with pytest.raises(ValueError, match="no reference crystal for Pm"):
        build_crystal("Pm", "fcc")
    assert len(build_crystal("Pm", "fcc", volume_per_atom=33.0)) == 1


@pytest.mark.unit
def test_default_elements_are_the_metals():
    assert default_elements(["Pt", "C", "O", "Sn", "Si"]) == ["Pt", "Sn"]


@pytest.mark.unit
def test_on_by_default_off_without_periodic_boundary_conditions():
    assert kwargs_defaults()["crystal_structures_kwargs"]["enabled"] is True
    assert (
        kwargs_defaults(evaluator_supports_periodic=False)["crystal_structures_kwargs"][
            "enabled"
        ]
        is False
    )
    with pytest.raises(ValueError, match="crystal_structures_kwargs"):
        validate_kwargs(
            {"crystal_structures_kwargs": {"enabled": True}},
            elements=["Pt"],
            evaluator_supports_periodic=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({"structures": ["fcc", "bct"]}, "unknown"),
        ({"elements": ["Au"]}, "not in general.elements"),
        ({"lattice_constants": {"Pt": {"fcc": -1}}}, "positive number"),
        ({"lattice_constants": {"Pt": {"hcp": {"a": 2.8, "c": 0}}}}, "a, c"),
        ({"volume_per_atom": {"Pt": "big"}}, "positive number"),
        ({"elements": ["Pm"]}, "no reference crystal"),
    ],
)
def test_validation_raises(block, match):
    with pytest.raises(ValueError, match=match):
        validate_kwargs(
            {
                "amorphous_kwargs": {"enabled": False},
                "crystal_structures_kwargs": block,
            },
            elements=["Pt", "Pm"],
        )


@pytest.mark.unit
def test_validation_logs_each_lattice_and_its_source():
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = initialiser_interface.logger
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        validate_kwargs({"amorphous_kwargs": {"enabled": False}}, elements=["Pt", "C"])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    lines = [r.getMessage() for r in records if "init_crystal" in r.getMessage()]
    assert len(lines) == 1  # Pt only: C is not a metal
    assert "fcc a=3.920 A (reference" in lines[0]
    assert "bcc" in lines[0] and "equal_volume" in lines[0]


@pytest.mark.unit
def test_needs_count_each_structure_and_restart(tmp_path):
    db = GlobalDatabase(str(tmp_path / "db"))
    config = {"amorphous_kwargs": {"enabled": False}}
    needs = compute_needs(db, config, ["Pt", "C"])
    assert needs["crystal_override"] == {"Pt": list(CRYSTAL_STRUCTURES)}
    fcc = build_crystal("Pt", "fcc")
    fcc.info["REF_energy"] = -6.0
    fcc.arrays["REF_forces"] = np.zeros((1, 3))
    db.add_structures([fcc])
    needs = compute_needs(db, config, ["Pt", "C"])
    assert needs["crystal_override"] == {
        "Pt": [s for s in CRYSTAL_STRUCTURES if s != "fcc"]
    }
    built = create_crystal_atoms_list(["Pt"], [], override=needs["crystal_override"])
    for atoms in built:
        atoms.info["REF_energy"] = -5.0
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    db.add_structures(built)
    assert compute_needs(db, config, ["Pt", "C"])["crystal_override"] == {}


@pytest.mark.unit
def test_crystals_are_always_kept_by_the_train_filter(tmp_path):
    """The unstable phases are far above the ground state by design: the
    train filter's windows must not drop them, whatever the user's
    exempt_config_types."""
    db = GlobalDatabase(str(tmp_path / "db"))
    atom = Atoms("Pt", cell=[15] * 3)
    atom.info.update(config_type="IsolatedAtom", REF_energy=-1.0)
    atom.arrays["REF_forces"] = np.zeros((1, 3))
    crystal = build_crystal("Pt", "sc")
    crystal.info["REF_energy"] = 50.0  # far outside any window
    crystal.arrays["REF_forces"] = np.zeros((1, 3))
    dimer = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.3]], cell=[15] * 3)
    dimer.info.update(config_type="init_dimer", REF_energy=80.0)
    dimer.arrays["REF_forces"] = np.zeros((2, 3))
    db.add_structures([atom, crystal, dimer], split="train", skip_duplicates=False)
    for exempt in ([], ["IsolatedAtom"]):
        result = apply_split_filter(
            db,
            "train",
            {
                "formation_energy_per_atom": [-20.0, 5.0],
                "exempt_config_types": exempt,
            },
        )
        flagged = {
            c.AtomPositionManager.metadata["config_type"]
            for c in db.partition.list_containers()
            if c.AtomPositionManager.metadata.get("is_quality_filtered")
        }
        assert "init_crystal" not in flagged
        assert "init_dimer" in flagged
        assert result["excluded"] >= 1
