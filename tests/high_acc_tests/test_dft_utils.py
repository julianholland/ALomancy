"""Tests for backend-agnostic DFT utilities in dft_utils.py."""

from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    _write_dft_result,
)

_PARA_INFO = {
    "ranks_per_system": 72,
    "ranks_per_node": 36,
    "threads_per_rank": 2,
    "max_mem_per_node": "90G",
}


def _cu_dimer():
    return Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True)


@pytest.mark.unit
class TestBuildSrunCommand:
    def test_executable_in_command(self):
        cmd = _build_srun_command(_PARA_INFO, "pw.x -nk 4")
        assert "pw.x -nk 4" in cmd

    def test_ntasks(self):
        assert "--ntasks=72" in _build_srun_command(_PARA_INFO, "pw.x")

    def test_tasks_per_node(self):
        assert "--tasks-per-node=36" in _build_srun_command(_PARA_INFO, "pw.x")

    def test_cpus_per_task(self):
        assert "--cpus-per-task=2" in _build_srun_command(_PARA_INFO, "pw.x")

    def test_mem(self):
        assert "--mem=90G" in _build_srun_command(_PARA_INFO, "pw.x")

    def test_distribution(self):
        assert "--distribution=block:block" in _build_srun_command(_PARA_INFO, "pw.x")

    def test_hint_nomultithread(self):
        assert "--hint=nomultithread" in _build_srun_command(_PARA_INFO, "pw.x")


@pytest.mark.unit
class TestWriteDftResult:
    def test_file_created(self, tmp_path):
        _write_dft_result(_cu_dimer(), str(tmp_path), "sp")
        assert (tmp_path / "sp.xyz").exists()

    def test_written_atoms_readable(self, tmp_path):
        from ase.io import read

        atoms = _cu_dimer()
        atoms.info["config_type"] = "test"
        _write_dft_result(atoms, str(tmp_path), "result")
        loaded = read(str(tmp_path / "result.xyz"), format="extxyz")
        assert loaded.get_chemical_formula() == "Cu2"
        assert loaded.info.get("config_type") == "test"

    def test_name_used_as_filename_stem(self, tmp_path):
        _write_dft_result(_cu_dimer(), str(tmp_path), "my_custom_name")
        assert (tmp_path / "my_custom_name.xyz").exists()


@pytest.mark.unit
class TestRunSp:
    def test_returns_atoms(self, tmp_path):
        result = _run_sp(
            _cu_dimer(), str(tmp_path / "out"), {"name": "sp"}, lambda a, j, d: EMT()
        )
        assert isinstance(result, Atoms)

    def test_returns_same_object(self, tmp_path):
        atoms = _cu_dimer()
        result = _run_sp(
            atoms, str(tmp_path / "out"), {"name": "x"}, lambda a, j, d: EMT()
        )
        assert result is atoms

    def test_output_xyz_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_sp(_cu_dimer(), out, {"name": "result"}, lambda a, j, d: EMT())
        assert (Path(out) / "result.xyz").exists()

    def test_creates_nested_directory(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "c")
        _run_sp(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert Path(out).exists()

    def test_calc_fn_called_with_atoms_jobdict_outdir(self, tmp_path):
        recorded = {}

        def record_calc(atoms, job_dict, out_dir):
            recorded["atoms"] = atoms
            recorded["job_dict"] = job_dict
            recorded["out_dir"] = out_dir
            return EMT()

        atoms = _cu_dimer()
        job_dict = {"name": "sp"}
        out = str(tmp_path / "out")
        _run_sp(atoms, out, job_dict, record_calc)
        assert recorded["atoms"] is atoms
        assert recorded["job_dict"] is job_dict
        assert recorded["out_dir"] == out


@pytest.mark.unit
class TestRunGo:
    def test_returns_atoms(self, tmp_path):
        result = _run_go(
            _cu_dimer(), str(tmp_path / "out"), {"name": "go"}, lambda a, j, d: EMT()
        )
        assert isinstance(result, Atoms)

    def test_output_xyz_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(_cu_dimer(), out, {"name": "result"}, lambda a, j, d: EMT())
        assert (Path(out) / "result.xyz").exists()

    def test_default_opt_prefix_log_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert (Path(out) / "opt.log").exists()

    def test_custom_opt_prefix_log_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(
            _cu_dimer(),
            out,
            {"name": "x"},
            lambda a, j, d: EMT(),
            opt_prefix="vasp_opt",
        )
        assert (Path(out) / "vasp_opt.log").exists()

    def test_custom_opt_prefix_traj_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(
            _cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT(), opt_prefix="qe_opt"
        )
        assert (Path(out) / "qe_opt.traj").exists()

    def test_creates_nested_directory(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "c")
        _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert Path(out).exists()
