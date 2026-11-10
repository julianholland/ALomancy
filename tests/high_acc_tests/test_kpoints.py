"""k-point grids across evaluators (utils/dft_utils.kpoint_grid): a user
density overrides ALomancy's default spacing, an explicit grid raises, and
non-periodic axes (slabs, molecules, isolated atoms) get one k-point."""

import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule

from alomancy.high_accuracy_evaluation.dft import run_aims, run_qe, run_vasp
from alomancy.utils.dft_utils import generate_kpts, kpoint_grid

NODE_INFO = {
    "ranks_per_system": 8,
    "ranks_per_node": 16,
    "threads_per_rank": 1,
    "max_mem_per_node": "64GB",
}


def _slab() -> Atoms:
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=None)
    slab.pbc = [True, True, False]
    return slab


@pytest.mark.unit
def test_bulk_grid_matches_the_former_formula():
    cell = bulk("Pt", "fcc", a=3.92, cubic=True)
    assert kpoint_grid(cell, 0.15) == [
        int(k) for k in generate_kpts(cell.cell, periodic_3d=True, kspacing=0.15)
    ]


@pytest.mark.unit
def test_non_periodic_axes_get_one_k_point():
    slab = _slab()
    grid = kpoint_grid(slab, 0.15)
    assert grid[2] == 1 and grid[0] > 1 and grid[1] > 1
    h2o = molecule("H2O", vacuum=6.0)
    assert kpoint_grid(h2o, 0.15) == [1, 1, 1]
    with pytest.raises(ValueError, match="positive"):
        kpoint_grid(h2o, 0)


@pytest.mark.unit
class TestVasp:
    def _job(self, vasp_kwargs):
        return {
            "hpc": {
                "vasp_path": "/opt/vasp_std",
                "node_info": NODE_INFO,
                "pseudo_dict": {},
            },
            "vasp_input_kwargs": vasp_kwargs,
        }

    def test_user_kspacing_overrides_and_leaves_the_incar(self):
        cell = bulk("Pt", "fcc", a=3.92, cubic=True)
        calc = run_vasp.create_vasp_calc_object(
            cell.copy(), self._job({"kspacing": 0.3, "kgamma": ".TRUE."}), "/tmp/out"
        )
        assert list(calc.input_params["kpts"]) == kpoint_grid(cell, 0.3)
        assert calc.input_params["gamma"] is True
        assert calc.float_params.get("kspacing") is None
        default = run_vasp.create_vasp_calc_object(cell.copy(), self._job({}), "/tmp/o")
        assert list(default.input_params["kpts"]) == kpoint_grid(
            cell, run_vasp.KSPACING
        )
        assert default.input_params["gamma"] is False

    def test_slab_and_molecule_grids_come_from_the_pbc_before_padding(self):
        slab = _slab()
        expected = kpoint_grid(slab, run_vasp.KSPACING)
        calc = run_vasp.create_vasp_calc_object(slab, self._job({}), "/tmp/out")
        assert slab.pbc.all()  # padded for VASP
        assert list(calc.input_params["kpts"]) == expected
        assert expected[2] == 1
        h2o = molecule("H2O")
        calc = run_vasp.create_vasp_calc_object(h2o, self._job({}), "/tmp/out")
        assert list(calc.input_params["kpts"]) == [1, 1, 1]

    def test_explicit_grid_raises(self):
        with pytest.raises(ValueError, match="kpts can't be set"):
            run_vasp.validate_kwargs({"vasp_kwargs": {"kpts": [4, 4, 4]}}, None)
        run_vasp.validate_kwargs({"vasp_kwargs": {"kspacing": 0.2}}, None)

    def test_level_of_theory_records_the_spacing(self):
        level = run_vasp.level_of_theory({"vasp_kwargs": {"kspacing": 0.3}})
        assert level.k_spacing == 0.3


@pytest.mark.unit
class TestQe:
    def _job(self, qe_kwargs):
        return {
            "hpc": {
                "pwx_path": "/opt/pw.x",
                "pp_path": "/opt/pp",
                "pseudo_dict": {"Pt": "Pt.UPF", "H": "H.UPF", "O": "O.UPF"},
                "node_info": NODE_INFO,
            },
            "qe_input_kwargs": qe_kwargs,
        }

    def test_k_spacing_overrides_and_is_not_a_namelist(self):
        cell = bulk("Pt", "fcc", a=3.92, cubic=True)
        calc = run_qe.create_qe_calc_object(cell, self._job({"k_spacing": 0.3}), "/o")
        assert list(calc.parameters["kpts"]) == kpoint_grid(cell, 0.3)
        assert "k_spacing" not in calc.parameters["input_data"]

    def test_isolated_atom_runs_at_gamma(self):
        atom = Atoms("H", cell=[15, 15, 15])
        calc = run_qe.create_qe_calc_object(atom, self._job({}), "/o")
        assert list(calc.parameters["kpts"]) == [1, 1, 1]

    def test_invalid_k_spacing_raises(self):
        with pytest.raises(ValueError, match="k_spacing must be"):
            run_qe.validate_kwargs({"qe_kwargs": {"k_spacing": "dense"}}, None)
        assert (
            run_qe.level_of_theory({"qe_kwargs": {"k_spacing": 0.25}}).k_spacing == 0.25
        )


@pytest.mark.unit
class TestAims:
    def test_k_grid_density_overrides_k_spacing(self):
        assert run_aims.effective_k_spacing({"k_grid_density": 5}) == pytest.approx(0.2)
        assert run_aims.effective_k_spacing({}) == run_aims.KSPACING
        cell = bulk("Pt", "fcc", a=3.92, cubic=True)
        params = run_aims.get_aims_parameters({"k_grid_density": 5}, cell)
        assert params["k_grid"] == kpoint_grid(cell, 0.2)
        assert "k_grid_density" not in params

    def test_slab_grid_is_in_plane_only(self):
        slab = _slab()
        expected = kpoint_grid(slab, run_aims.KSPACING)
        k_grid = run_aims.prepare_structure(slab)
        assert slab.pbc.all() and k_grid == expected and k_grid[2] == 1

    def test_both_densities_or_a_fixed_grid_raise(self):
        with pytest.raises(ValueError, match="not both"):
            run_aims.validate_kwargs(
                {"aims_kwargs": {"k_spacing": 0.2, "k_grid_density": 5}}, None
            )
        with pytest.raises(ValueError, match="k_grid can't be set"):
            run_aims.validate_kwargs({"aims_kwargs": {"k_grid": [4, 4, 4]}}, None)
        run_aims.validate_kwargs({"aims_kwargs": {"k_grid_density": 5}}, None)
        level = run_aims.level_of_theory({"aims_kwargs": {"k_grid_density": 5}})
        assert level.k_spacing == pytest.approx(0.2)
