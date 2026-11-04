"""Tests for backend-agnostic DFT utilities in dft_utils.py."""

import logging
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.emt import EMT
from ase.io import read

import alomancy.utils.dft_utils as dft_utils
from alomancy.utils.dft_utils import (
    _build_srun_command,
    _run_go,
    _run_sp,
    _snapshot,
    _write_dft_result,
    refresh_dft_labels,
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
    @pytest.mark.unit
    def test_executable_in_command(self):
        cmd = _build_srun_command(_PARA_INFO, "pw.x -nk 4")
        assert "pw.x -nk 4" in cmd

    @pytest.mark.unit
    def test_ntasks(self):
        assert "--ntasks=72" in _build_srun_command(_PARA_INFO, "pw.x")

    @pytest.mark.unit
    def test_tasks_per_node(self):
        assert "--tasks-per-node=36" in _build_srun_command(_PARA_INFO, "pw.x")

    @pytest.mark.unit
    def test_cpus_per_task(self):
        assert "--cpus-per-task=2" in _build_srun_command(_PARA_INFO, "pw.x")

    @pytest.mark.unit
    def test_mem_is_zero_sentinel(self):
        # --mem=0 means "inherit all memory already granted to the job", not
        # "request zero memory" -- see comment in _build_srun_command.
        # max_mem_per_node is deliberately not echoed here; requesting an
        # explicit sub-amount from a nested srun step competes with the
        # enclosing batch step's own memory reservation and can fail with
        # "Unable to create step ... Memory required by task is not
        # available" regardless of how much the job was granted.
        assert "--mem=0 " in _build_srun_command(_PARA_INFO, "pw.x")
        assert "--mem=90G" not in _build_srun_command(_PARA_INFO, "pw.x")

    @pytest.mark.unit
    def test_distribution(self):
        assert "--distribution=block:block" in _build_srun_command(_PARA_INFO, "pw.x")

    @pytest.mark.unit
    def test_hint_nomultithread(self):
        assert "--hint=nomultithread" in _build_srun_command(_PARA_INFO, "pw.x")


@pytest.mark.unit
class TestWriteDftResult:
    @pytest.mark.unit
    def test_file_created(self, tmp_path):
        _write_dft_result(_cu_dimer(), str(tmp_path), "sp")
        assert (tmp_path / "sp.xyz").exists()

    @pytest.mark.unit
    def test_written_atoms_readable(self, tmp_path):
        from ase.io import read

        atoms = _cu_dimer()
        atoms.info["config_type"] = "test"
        _write_dft_result(atoms, str(tmp_path), "result")
        loaded = read(str(tmp_path / "result.xyz"), format="extxyz")
        assert loaded.get_chemical_formula() == "Cu2"
        assert loaded.info.get("config_type") == "test"

    @pytest.mark.unit
    def test_name_used_as_filename_stem(self, tmp_path):
        _write_dft_result(_cu_dimer(), str(tmp_path), "my_custom_name")
        assert (tmp_path / "my_custom_name.xyz").exists()


@pytest.mark.unit
class TestRunSp:
    @pytest.mark.unit
    def test_returns_atoms(self, tmp_path):
        result = _run_sp(
            _cu_dimer(), str(tmp_path / "out"), {"name": "sp"}, lambda a, j, d: EMT()
        )
        assert isinstance(result, Atoms)

    @pytest.mark.unit
    def test_returns_same_object(self, tmp_path):
        atoms = _cu_dimer()
        result = _run_sp(
            atoms, str(tmp_path / "out"), {"name": "x"}, lambda a, j, d: EMT()
        )
        assert result is atoms

    @pytest.mark.unit
    def test_output_xyz_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_sp(_cu_dimer(), out, {"name": "result"}, lambda a, j, d: EMT())
        assert (Path(out) / "result.xyz").exists()

    @pytest.mark.unit
    def test_creates_nested_directory(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "c")
        _run_sp(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert Path(out).exists()

    @pytest.mark.unit
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
    @pytest.mark.unit
    def test_returns_atoms(self, tmp_path):
        result = _run_go(
            _cu_dimer(), str(tmp_path / "out"), {"name": "go"}, lambda a, j, d: EMT()
        )
        assert isinstance(result, Atoms)

    @pytest.mark.unit
    def test_output_xyz_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(_cu_dimer(), out, {"name": "result"}, lambda a, j, d: EMT())
        assert (Path(out) / "result.xyz").exists()

    @pytest.mark.unit
    def test_default_opt_prefix_log_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert (Path(out) / "opt.log").exists()

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_custom_opt_prefix_traj_written(self, tmp_path):
        out = str(tmp_path / "out")
        _run_go(
            _cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT(), opt_prefix="qe_opt"
        )
        assert (Path(out) / "qe_opt.traj").exists()

    @pytest.mark.unit
    def test_creates_nested_directory(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "c")
        _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert Path(out).exists()

    @pytest.mark.unit
    def test_converged_structure_flagged_true(self, tmp_path):
        out = str(tmp_path / "out")
        result = _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        assert result.info["geometry_converged"] is True

    @pytest.mark.unit
    def test_fmax_and_steps_default_when_absent_from_job_dict(
        self, tmp_path, monkeypatch
    ):
        """fmax/max_num_of_relax_steps default to today's hardcoded values (0.05,
        200) when the job dict doesn't set them -- preserves existing
        behavior for every caller that doesn't opt into a custom target."""
        captured = {}

        class _Capturing:
            nsteps = 0  # ASE optimizers count completed steps here

            def __init__(self, atoms, logfile=None, trajectory=None):
                pass

            def run(self, fmax, steps):
                captured["fmax"] = fmax
                captured["steps"] = steps
                return True

        monkeypatch.setattr(dft_utils, "BFGS", _Capturing)
        out = str(tmp_path / "out")
        _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())

        assert captured["fmax"] == 0.05
        assert captured["steps"] == 200

    @pytest.mark.unit
    def test_fmax_and_steps_read_from_job_dict_when_present(
        self, tmp_path, monkeypatch
    ):
        """fmax/max_num_of_relax_steps are read from the job dict when present --
        the channel high_accuracy_evaluation.force_ceiling uses to drive
        relaxation targets."""
        captured = {}

        class _Capturing:
            nsteps = 0  # ASE optimizers count completed steps here

            def __init__(self, atoms, logfile=None, trajectory=None):
                pass

            def run(self, fmax, steps):
                captured["fmax"] = fmax
                captured["steps"] = steps
                return True

        monkeypatch.setattr(dft_utils, "BFGS", _Capturing)
        out = str(tmp_path / "out")
        _run_go(
            _cu_dimer(),
            out,
            {"name": "x", "fmax": 5.0, "max_num_of_relax_steps": 500},
            lambda a, j, d: EMT(),
        )

        assert captured["fmax"] == 5.0
        assert captured["steps"] == 500

    @pytest.mark.unit
    def test_non_convergence_warns_and_keeps_structure(self, tmp_path, monkeypatch):
        """Non-convergence must not discard the completed DFT computation --
        it downgrades to a logged warning and geometry_converged=False,
        keeping the best structure BFGS found rather than raising."""

        class _NeverConverges:
            nsteps = 0  # ASE optimizers count completed steps here

            def __init__(self, atoms, logfile=None, trajectory=None):
                pass

            def run(self, fmax, steps):
                return False

        monkeypatch.setattr(dft_utils, "BFGS", _NeverConverges)

        al_logger = logging.getLogger("alomancy")
        al_logger.setLevel(logging.WARNING)
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collector()
        handler.setLevel(logging.WARNING)
        al_logger.addHandler(handler)
        try:
            out = str(tmp_path / "out")
            result = _run_go(_cu_dimer(), out, {"name": "x"}, lambda a, j, d: EMT())
        finally:
            al_logger.removeHandler(handler)

        assert result.info["geometry_converged"] is False
        assert (Path(out) / "x.xyz").exists()
        assert any("did not reach fmax" in r.getMessage() for r in records)


@pytest.mark.unit
class TestGoInitialFrame:
    """Step 0 of a geometry optimisation is a full DFT calculation of the
    unrelaxed structure; _run_go keeps it in <name>_initial.xyz."""

    @pytest.mark.unit
    def test_initial_frame_holds_the_unrelaxed_geometry_and_its_labels(self, tmp_path):
        out = tmp_path / "out"
        start = _cu_dimer()
        expected = start.copy()
        expected.calc = EMT()
        result = _run_go(start.copy(), str(out), {"name": "x"}, lambda a, j, d: EMT())

        initial = read(out / "x_initial.xyz", format="extxyz")
        assert np.allclose(initial.positions, expected.positions)
        assert initial.info["REF_energy"] == pytest.approx(
            expected.get_potential_energy()
        )
        assert np.allclose(initial.arrays["REF_forces"], expected.get_forces())
        assert initial.info["geometry_step"] == 0
        assert initial.info["geometry_initial"] is True
        for key in ("geometry_converged", "geometry_steps", "dft_wall_time_s"):
            assert key not in initial.info
        assert result.info["geometry_step"] == result.info["geometry_steps"] > 0
        final = read(out / "x.xyz", format="extxyz")
        assert "geometry_initial" not in final.info
        assert not np.allclose(final.positions, initial.positions)

    @pytest.mark.unit
    def test_no_initial_frame_when_the_structure_did_not_move(self, tmp_path):
        out = tmp_path / "out"
        single = Atoms("Cu", cell=[10, 10, 10], pbc=True)
        _run_go(single, str(out), {"name": "x"}, lambda a, j, d: EMT())
        assert (out / "x.xyz").exists()
        assert not (out / "x_initial.xyz").exists()

    @pytest.mark.unit
    def test_snapshot_never_asks_the_calculator_again(self):
        """A property the calculator didn't compute (QE stress with tstress
        off) is left out rather than triggering a rerun."""

        class NoStressEMT(EMT):
            calls = 0

            def calculate(self, atoms=None, properties=("energy",), changes=()):
                NoStressEMT.calls += 1
                if NoStressEMT.calls > 1:
                    raise AssertionError("recalculated")
                super().calculate(atoms, properties, changes)
                self.results.pop("stress", None)

        atoms = _cu_dimer()
        atoms.calc = NoStressEMT()
        atoms.get_forces()
        copy = _snapshot(atoms)

        assert np.allclose(copy.get_forces(), atoms.calc.results["forces"])
        with pytest.raises(PropertyNotImplementedError):
            copy.get_stress()
        assert NoStressEMT.calls == 1


class _CountingNoStressEMT(EMT):
    """A calculator that, like QE with tstress off, never reports stress;
    counts how often it actually runs."""

    def __init__(self):
        super().__init__()
        self.runs = 0

    def calculate(self, atoms=None, properties=("energy",), changes=()):
        self.runs += 1
        super().calculate(atoms, properties, changes)
        self.results.pop("stress", None)


@pytest.mark.unit
class TestDftLabelStress:
    @pytest.mark.unit
    def test_missing_stress_never_reruns_the_calculation(self):
        """Asking for stress the calculation didn't produce used to rerun
        it (one extra pw.x per DFT job with tstress off)."""
        atoms = _cu_dimer()
        calc = _CountingNoStressEMT()
        atoms.calc = calc
        atoms.get_forces()

        refresh_dft_labels(atoms)

        assert calc.runs == 1
        assert "REF_stresses" not in atoms.info

    @pytest.mark.unit
    def test_stale_stress_from_the_seed_is_dropped(self):
        atoms = _cu_dimer()
        atoms.info["REF_stresses"] = np.ones(6)  # inherited from an MD seed
        atoms.calc = _CountingNoStressEMT()
        atoms.get_forces()
        refresh_dft_labels(atoms)
        assert "REF_stresses" not in atoms.info

    @pytest.mark.unit
    def test_computed_stress_becomes_the_label(self):
        atoms = _cu_dimer()
        atoms.info["REF_stresses"] = np.ones(6)
        atoms.calc = EMT()
        expected = atoms.get_stress()
        refresh_dft_labels(atoms)
        assert np.allclose(atoms.info["REF_stresses"], expected)
