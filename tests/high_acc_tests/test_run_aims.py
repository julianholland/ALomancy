"""Tests for the FHI-aims backend (high_accuracy_evaluation/dft/run_aims.py).

No FHI-aims binary is needed: the calculator is built and control.in /
geometry.in written with ASE's own AimsTemplate against a small fake
species_defaults tree, which is what a real job would run.
"""

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule

from alomancy.high_accuracy_evaluation.dft import run_aims
from alomancy.high_accuracy_evaluation.dft.run_aims import (
    create_aims_calc_object,
    level_of_theory,
    validate_kwargs,
)
from alomancy.utils.dft_utils import pad_non_periodic_axes

NODE_INFO = {
    "ranks_per_system": 8,
    "ranks_per_node": 16,
    "threads_per_rank": 1,
    "max_mem_per_node": "64GB",
}

_SPECIES = """\
  species        {symbol}
    nucleus      {z}
    mass         {mass}
    l_hartree    {setting_lh}
#  "First tier" - improvements: -1.00 meV
     hydro 2 p 1.7
#  "Second tier" - improvements: -0.10 meV
     hydro 3 d 6
"""


@pytest.fixture
def hpc(tmp_path):
    root = tmp_path / "species_defaults" / "defaults_2020"
    for setting, lh in (("intermediate", 6), ("light", 4)):
        folder = root / setting
        folder.mkdir(parents=True)
        for symbol, z, mass in (("H", 1, 1.008), ("C", 6, 12.011), ("O", 8, 15.999)):
            (folder / f"{z:02d}_{symbol}_default").write_text(
                _SPECIES.format(symbol=symbol, z=z, mass=mass, setting_lh=lh)
            )
        # Cu for the slab test.
        (folder / "29_Cu_default").write_text(
            _SPECIES.format(symbol="Cu", z=29, mass=63.546, setting_lh=lh)
        )
    return {
        "hpc_name": "test-hpc",
        "aims_path": "/opt/aims/aims.x",
        "aims_species_dir": str(root),
        "pre_cmds": ["ulimit -s unlimited"],
        "node_info": NODE_INFO,
    }


def _job(hpc: dict, aims_kwargs: dict | None = None) -> dict:
    return {
        "name": "high_accuracy_evaluation",
        "hpc": hpc,
        "aims_kwargs": aims_kwargs or {},
    }


def _write(atoms: Atoms, job: dict, out) -> tuple[str, str, object]:
    out.mkdir(exist_ok=True)
    calc = create_aims_calc_object(atoms, job, str(out))
    calc.template.write_input(
        profile=calc.profile,
        directory=out,
        atoms=atoms,
        parameters=calc.parameters,
        properties=["energy"],
    )
    return (out / "control.in").read_text(), (out / "geometry.in").read_text(), calc


def _keywords(control: str) -> dict[str, str]:
    """control.in's keyword lines before the species blocks."""
    head = control.split("species ")[0]
    lines = [
        ln.strip() for ln in head.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    return {ln.split()[0]: " ".join(ln.split()[1:]) for ln in lines}


@pytest.mark.unit
def test_periodic_input_has_intermediate_defaults_and_one_scf(hpc, tmp_path):
    atoms = bulk("C", "diamond", a=3.57)
    control, geometry, calc = _write(atoms, _job(hpc), tmp_path / "run")
    kw = _keywords(control)
    assert kw["xc"] == "pbe"
    assert kw["relativistic"] == "atomic_zora scalar"
    assert kw["occupation_type"] == "gaussian 0.01"
    assert kw["spin"] == "none"
    # Forces and stress are requested even though only the energy was asked
    # for: one SCF yields every label.
    assert kw["compute_forces"] == ".true."
    assert kw["compute_analytical_stress"] == ".true."
    assert kw["k_grid"] == "17 17 17"
    assert "k_offset" not in kw
    # The intermediate species file is copied in.
    assert "l_hartree    6" in control
    assert "lattice_vector" in geometry
    assert calc.profile.command.startswith("env OMP_NUM_THREADS=1 srun --ntasks=8")
    assert calc.profile.command.endswith("/opt/aims/aims.x")


@pytest.mark.unit
def test_cluster_input_has_no_cell_k_grid_or_stress(hpc, tmp_path):
    control, geometry, _ = _write(molecule("H2O"), _job(hpc), tmp_path / "run")
    kw = _keywords(control)
    assert "k_grid" not in kw
    assert "compute_analytical_stress" not in kw
    assert kw["compute_forces"] == ".true."
    assert "lattice_vector" not in geometry


@pytest.mark.unit
def test_isolated_atom_runs_without_a_cell(hpc, tmp_path):
    atoms = Atoms("C", positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=True)
    atoms.info["config_type"] = "IsolatedAtom"
    control, geometry, _ = _write(atoms, _job(hpc), tmp_path / "run")
    assert not atoms.pbc.any()
    assert "k_grid" not in _keywords(control)
    assert "lattice_vector" not in geometry


@pytest.mark.unit
def test_mixed_pbc_is_padded_along_the_open_axis_only(hpc, tmp_path):
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=None)
    slab.pbc = [True, True, False]
    in_plane = slab.cell[:2].copy()
    control, _, _ = _write(slab, _job(hpc), tmp_path / "run")
    assert slab.pbc.all()
    assert np.allclose(slab.cell[:2], in_plane)
    z = slab.positions[:, 2]
    assert z.min() == pytest.approx(10.0) and slab.cell[2, 2] == pytest.approx(
        z.max() + 10.0
    )
    assert "k_grid" in _keywords(control)


@pytest.mark.unit
def test_vasp_padding_keeps_a_slabs_in_plane_cell():
    """The shared helper VASP now uses: centring along all axes used to
    resize the periodic in-plane cell of a slab too."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=None)
    slab.pbc = [True, True, False]
    in_plane = slab.cell[:2].copy()
    pad_non_periodic_axes(slab, vacuum=8.0)
    assert slab.pbc.all()
    assert np.allclose(slab.cell[:2], in_plane)


@pytest.mark.unit
def test_user_settings_charge_k_spacing_and_light_species(hpc, tmp_path):
    atoms = molecule("H2O")
    atoms.info["charge"] = 1
    control, _, _ = _write(
        atoms,
        _job(hpc, {"species_defaults": "light", "xc": "pbe0", "spin": "collinear"}),
        tmp_path / "run",
    )
    kw = _keywords(control)
    assert kw["charge"] == "1"
    assert kw["xc"] == "pbe0" and kw["spin"] == "collinear"
    assert "l_hartree    4" in control
    coarse, _, _ = _write(
        bulk("C", "diamond", a=3.57), _job(hpc, {"k_spacing": 0.5}), tmp_path / "k"
    )
    # Primitive cell vectors are a/sqrt(2) = 2.52 A: ceil(2 pi / (2.52 * 0.5)) = 5.
    assert _keywords(coarse)["k_grid"] == "5 5 5"


@pytest.mark.unit
def test_smearing_replaces_the_default_occupation_type(hpc, tmp_path):
    control, _, _ = _write(
        molecule("H2O"), _job(hpc, {"smearing": ["fermi-dirac", 0.05]}), tmp_path / "r"
    )
    assert control.count("occupation_type") == 1
    assert "fermi 0.05" in control


@pytest.mark.unit
def test_level_of_theory(hpc):
    level = level_of_theory({"aims_kwargs": {}}, hpc)
    assert level.code == "aims"
    assert (level.method, level.rung, level.method_class) == ("PBE", "GGA", "dft")
    assert level.basis_type == "numeric_atom_centred"
    assert level.basis == "defaults_2020/intermediate"
    assert level.core_treatment == "all_electron"
    assert level.pseudopotentials is None
    assert level.relativity == "atomic_ZORA"
    # "zora scalar <threshold>" used to give "scalar_scalar_ZORA".
    zora = level_of_theory({"aims_kwargs": {"relativistic": "zora scalar 1e-12"}}, hpc)
    assert zora.relativity == "scalar_ZORA"
    none = level_of_theory({"aims_kwargs": {"relativistic": "none"}}, hpc)
    assert none.relativity == "none"
    assert level.spin_treatment == "restricted"
    assert level.smearing == {"scheme": "gaussian", "width_eV": 0.01}
    assert level.k_spacing == 0.15
    hse = level_of_theory(
        {
            "aims_kwargs": {
                "xc": "hse06 0.11",
                "hse_unit": "bohr-1",
                "many_body_dispersion": "",
            }
        },
        hpc,
    )
    assert (hse.method, hse.rung, hse.dispersion) == (
        "HSE06",
        "range-separated hybrid",
        "MBD",
    )
    light = level_of_theory({"aims_kwargs": {"species_defaults": "light"}}, hpc)
    assert light.identity_id() != level.identity_id()
    tier = level_of_theory({"aims_kwargs": {"tier": 2}}, hpc)
    assert tier.basis == "defaults_2020/intermediate tier 2"
    # k-point spacing is precision, not identity.
    coarse = level_of_theory({"aims_kwargs": {"k_spacing": 0.3}}, hpc)
    assert coarse.identity_id() == level.identity_id()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("aims_kwargs", "match"),
    [
        ({"k_grid": [4, 4, 4]}, "k_grid can't be set"),
        ({"kpts": [4, 4, 4]}, "kpts can't be set"),
        ({"species_dir": "/x"}, "aims_species_dir"),
        ({"plus_u": {"C": "2p 1.0"}}, "plus_u can't be set"),
        ({"species_defaults": "tighter"}, "species_defaults must be one of"),
        ({"k_spacing": 0}, "k_spacing must be a positive number"),
        ({"tier": "two"}, "tier must be an integer"),
        ({"compute_forces": False}, "compute_forces can't be false"),
        ({"xc": "hse06 0.11"}, "needs hse_unit"),
    ],
)
def test_validate_kwargs_raises(hpc, aims_kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_kwargs({"aims_kwargs": aims_kwargs}, hpc)


@pytest.mark.unit
def test_validate_kwargs_needs_the_profile_paths(hpc):
    with pytest.raises(ValueError, match="no aims_species_dir"):
        validate_kwargs({"aims_kwargs": {}}, {**hpc, "aims_species_dir": None})
    validate_kwargs({"aims_kwargs": {}}, hpc)
    # No profile at hand: only the kwargs are checked.
    validate_kwargs({"aims_kwargs": {}}, None)


@pytest.mark.unit
def test_validate_kwargs_warns_on_stack_limit_and_meta_gga(hpc):
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    run_aims.logger.addHandler(handler)
    try:
        validate_kwargs({"aims_kwargs": {}}, {**hpc, "pre_cmds": []})
        validate_kwargs({"aims_kwargs": {"xc": "scan"}}, hpc)
        validate_kwargs(
            {"aims_kwargs": {"xc": "scan", "compute_analytical_stress": False}}, hpc
        )
    finally:
        run_aims.logger.removeHandler(handler)
    events = [getattr(r, "event", None) for r in records]
    assert events.count("aims_stack_limit") == 1
    assert events.count("aims_meta_gga_stress") == 1
