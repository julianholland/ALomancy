"""Tests for high_accuracy_evaluation/level_of_theory.py, the QE/VASP
level_of_theory entry points and utils/dft_utils.resolve_parallel_resources."""

import pytest
from ase import Atoms
from ase.io import read, write

from alomancy.high_accuracy_evaluation.dft import run_qe, run_vasp
from alomancy.high_accuracy_evaluation.level_of_theory import (
    LEVEL_OF_THEORY_ID_KEY,
    LEVEL_OF_THEORY_KEY,
    LevelOfTheory,
    differing_fields,
    functional_rung,
    stamp_level_of_theory,
)
from alomancy.utils.dft_utils import parse_memory_mb, resolve_parallel_resources


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pbe", ("PBE", "GGA")),
        ("R2SCAN", ("r2SCAN", "meta-GGA")),
        ("hse06", ("HSE06", "range-separated hybrid")),
        ("PBE-D3BJ", ("PBE", "GGA")),
        ("B2PLYP", ("B2PLYP", "double-hybrid")),
        ("made-up", ("made-up", None)),
        (None, (None, None)),
    ],
)
def test_functional_rung(name, expected):
    assert functional_rung(name) == expected


@pytest.mark.unit
def test_precision_changes_keep_the_id_identity_changes_alter_it():
    base = run_qe.level_of_theory({}, {})
    tighter = run_qe.level_of_theory(
        {"qe_kwargs": {"electrons": {"conv_thr": 1e-8}, "system": {"degauss": 0.02}}},
        {},
    )
    assert tighter.identity_id() == base.identity_id()
    assert tighter.to_json() != base.to_json()
    other = run_qe.level_of_theory(
        {"qe_kwargs": {"system": {"input_dft": "pbesol"}}}, {}
    )
    assert other.identity_id() != base.identity_id()
    assert differing_fields(base.identity(), other.identity()) == ["method"]


@pytest.mark.unit
def test_qe_level_of_theory():
    level = run_qe.level_of_theory(
        {"qe_kwargs": {"system": {"vdw_corr": "grimme-d3", "dftd3_version": 4}}},
        {"pseudo_dict": {"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF"}},
    )
    assert (level.code, level.method, level.rung) == ("qe", "PBE", "GGA")
    assert level.dispersion == "D3(BJ)"
    assert level.basis_type == "plane_wave"
    assert level.cutoff_eV == pytest.approx(40 * 13.6057, rel=1e-4)
    assert level.core_treatment == "paw"
    assert level.smearing["scheme"] == "cold"


@pytest.mark.unit
def test_vasp_level_of_theory():
    level = run_vasp.level_of_theory(
        {"vasp_kwargs": {"lhfcalc": True, "hfscreen": 0.2, "ivdw": 12, "ispin": 2}}
    )
    assert level.method == "HSE06" and level.rung == "range-separated hybrid"
    assert level.dispersion == "D3(BJ)"
    assert level.spin_treatment == "collinear"
    assert level.core_treatment == "paw"
    assert run_vasp.level_of_theory({"vasp_kwargs": {"gga": "PS"}}).method == "PBEsol"


@pytest.mark.unit
def test_stamp_survives_an_extxyz_round_trip(tmp_path):
    level = run_vasp.level_of_theory({})
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    stamp_level_of_theory([atoms], level)
    write(tmp_path / "a.xyz", atoms, format="extxyz")
    back = read(tmp_path / "a.xyz", format="extxyz")
    assert back.info[LEVEL_OF_THEORY_ID_KEY] == level.identity_id()
    restored = LevelOfTheory.from_json(back.info[LEVEL_OF_THEORY_KEY])
    assert restored == level


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "mb"),
    [("120GB", 122880), ("90G", 92160), ("512M", 512), (2048, 2048), ("1.5T", 1572864)],
)
def test_parse_memory_mb(value, mb):
    assert parse_memory_mb(value) == mb


@pytest.mark.unit
@pytest.mark.parametrize("value", ["lots", "-4G", "", None, True])
def test_parse_memory_mb_rejects(value):
    with pytest.raises(ValueError):
        parse_memory_mb(value)


@pytest.mark.unit
def test_parallel_resources_defaults_and_overrides():
    node = {"ranks_per_system": 4, "ranks_per_node": 8, "max_mem_per_node": "16G"}
    default = resolve_parallel_resources(node)
    assert (default.nprocs, default.mem_per_proc_mb) == (4, 3072)
    override = resolve_parallel_resources(node, nprocs=8, mem_per_proc_mb=2000)
    assert (override.nprocs, override.mem_per_proc_mb) == (8, 2000)
    with pytest.raises(ValueError, match="doesn't fit"):
        resolve_parallel_resources(node, nprocs=9)
    with pytest.raises(ValueError, match="more than the node"):
        resolve_parallel_resources(node, nprocs=8, mem_per_proc_mb=3000)
    with pytest.raises(ValueError, match="max_mem_per_node is required"):
        resolve_parallel_resources({"ranks_per_system": 4, "ranks_per_node": 8})
