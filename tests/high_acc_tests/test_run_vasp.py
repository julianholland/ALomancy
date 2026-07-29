"""Tests for VASP DFT backend (run_vasp.py)."""

import os

import pytest
from ase import Atoms


def _job_dict():
    return {
        "hpc": {
            "vasp_path": "/path/to/vasp_std",
            "node_info": {
                "ranks_per_system": 36,
                "ranks_per_node": 36,
                "threads_per_rank": 1,
                "max_mem_per_node": "90G",
            },
            "pseudo_dict": {},
        },
    }


def _cu_atoms():
    return Atoms("Cu", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)


@pytest.mark.unit
class TestCreateVaspCommand:
    def _para_info(self):
        return {
            "ranks_per_system": 36,
            "ranks_per_node": 36,
            "threads_per_rank": 1,
            "max_mem_per_node": "90G",
        }

    def test_vasp_path_in_command(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import create_vasp_command

        cmd = create_vasp_command(self._para_info(), "/usr/bin/vasp_std")
        assert "/usr/bin/vasp_std" in cmd

    def test_ntasks_in_command(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import create_vasp_command

        assert "--ntasks=36" in create_vasp_command(self._para_info(), "vasp_std")


@pytest.mark.unit
class TestGetVaspInputKwargs:
    def test_required_defaults_present(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import get_vasp_input_kwargs

        result = get_vasp_input_kwargs({})
        for key in ("encut", "ediff", "nsw", "ibrion", "prec"):
            assert key in result

    def test_single_point_nsw_zero(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import get_vasp_input_kwargs

        assert get_vasp_input_kwargs({})["nsw"] == 0

    def test_single_point_ibrion_minus_one(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import get_vasp_input_kwargs

        assert get_vasp_input_kwargs({})["ibrion"] == -1

    def test_overrides_applied(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import get_vasp_input_kwargs

        result = get_vasp_input_kwargs({"encut": 700, "sigma": 0.2})
        assert result["encut"] == 700
        assert result["sigma"] == 0.2

    def test_unset_defaults_preserved(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import get_vasp_input_kwargs

        result = get_vasp_input_kwargs({"encut": 700})
        assert result["prec"] == "Accurate"
        assert result["lwave"] is False
        assert result["lcharg"] is False


@pytest.mark.unit
class TestCreateVaspCalcObject:
    def test_returns_vasp_instance(self):
        from ase.calculators.vasp import Vasp

        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        calc = create_vasp_calc_object(_cu_atoms(), _job_dict(), "/tmp/out")
        assert isinstance(calc, Vasp)

    def test_sp_ibrion_minus_one(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        calc = create_vasp_calc_object(_cu_atoms(), _job_dict(), "/tmp/out")
        assert calc.int_params["ibrion"] == -1
        assert calc.int_params["nsw"] == 0

    def test_relaxation_sets_ibrion_and_nsw(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        calc = create_vasp_calc_object(
            _cu_atoms(), _job_dict(), "/tmp/out", is_relaxation=True
        )
        assert calc.int_params["ibrion"] == 2
        assert calc.int_params["nsw"] == 200

    def test_ncore_derived_from_ranks_per_node(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        # ranks_per_node=36 -> ncore = int(36**0.5) = 6
        calc = create_vasp_calc_object(_cu_atoms(), _job_dict(), "/tmp/out")
        assert calc.int_params.get("ncore") == 6

    def test_vasp_input_kwargs_override_defaults(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        job = _job_dict()
        job["vasp_input_kwargs"] = {"encut": 700}
        calc = create_vasp_calc_object(_cu_atoms(), job, "/tmp/out")
        assert calc.float_params.get("encut") == 700

    def test_pp_path_sets_vasp_pp_path_env_var(self, monkeypatch):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        monkeypatch.delenv("VASP_PP_PATH", raising=False)
        job = _job_dict()
        job["hpc"]["pp_path"] = "/u/antia/pps/potpaw_PBE"
        create_vasp_calc_object(_cu_atoms(), job, "/tmp/out")
        assert os.environ["VASP_PP_PATH"] == "/u/antia/pps/potpaw_PBE"

    def test_no_pp_path_leaves_env_var_untouched(self, monkeypatch):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import (
            create_vasp_calc_object,
        )

        monkeypatch.delenv("VASP_PP_PATH", raising=False)
        create_vasp_calc_object(_cu_atoms(), _job_dict(), "/tmp/out")
        assert "VASP_PP_PATH" not in os.environ
