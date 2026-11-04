"""Direct (non-mocked-away) tests of remote_submission/submitters.py's
ase_remote_submitter (training, prediction and MD go through
executor.submit_n). Each test here mocks only
RemoteJobExecutor.run_and_wait (the actual remote/ssh boundary), letting the
submitter's own job_config-construction logic run for real.
"""

from unittest.mock import MagicMock

import pytest
from ase import Atoms

from alomancy.remote_submission.submitters import (
    MAX_REMOTE_OUTPUT_DIR_LEN,
    ase_remote_submitter,
    check_remote_path_length,
    dft_job_name,
    expyre_remote_rundir,
    projected_remote_output_dir,
)


def _fake_run_and_wait(captured: dict):
    def _run(self, function, job_configs, **kwargs):
        captured["job_configs"] = job_configs
        return [None] * len(job_configs)

    return _run


class TestAseRemoteSubmitter:
    @pytest.mark.unit
    def test_per_structure_function_length_mismatch_raises(self):
        remote_info = MagicMock()
        remote_info.job_name = "high_accuracy_evaluation"

        with pytest.raises(ValueError, match="per_structure_function"):
            ase_remote_submitter(
                remote_info=remote_info,
                base_name="al_loop_0",
                input_atoms_list=[Atoms("H"), Atoms("H")],
                per_structure_function=[MagicMock()],
            )

    @pytest.mark.unit
    def test_per_structure_function_picks_go_or_sp_per_structure(
        self, tmp_path, monkeypatch
    ):
        """GO and SP structures share one submission queue, distinguished
        only by which function each job's config points at and a short
        {fn.__name__ without run_}_{i} job name."""
        monkeypatch.chdir(tmp_path)
        captured: dict = {}
        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _fake_run_and_wait(captured),
        )

        def run_go(**kwargs):
            pass

        def run_sp(**kwargs):
            pass

        remote_info = MagicMock()
        remote_info.job_name = "high_accuracy_evaluation"

        ase_remote_submitter(
            remote_info=remote_info,
            base_name="al_loop_0",
            input_atoms_list=[Atoms("H"), Atoms("H")],
            per_structure_function=[run_go, run_sp],
        )

        job_configs = captured["job_configs"]
        assert job_configs[0]["job_name"] == "go_0"
        assert job_configs[1]["job_name"] == "sp_1"
        assert job_configs[0]["function"] is run_go
        assert job_configs[1]["function"] is run_sp


# --- Remote path length (VASP's ~255-character path buffer) ---

_ADVR_OLD_RUNDIR = (
    "/raven/ptmp/advr/alomancy_scratch/"
    "unkownhost-_work_home_advr_Simulations_alomancy_test_5"
)
_ADVR_NEW_RUNDIR = "/raven/ptmp/advr/alomancy_scratch/Simulations_alomancy_test_5"
_ADVR_OUT_DIR = "results/initialization/high_accuracy_evaluation/batch_0/ase_output_0"


def _system(rundir: str):
    from expyre.system import System

    return System(host="raven", partitions={}, scheduler="slurm", rundir=rundir)


@pytest.fixture
def expyre_systems(monkeypatch):
    """expyre's config isn't loaded under pytest; give it real Systems."""
    from expyre import config as expyre_config

    systems: dict = {}
    monkeypatch.setattr(expyre_config, "systems", systems)
    return systems


class TestRemotePathLength:
    @pytest.mark.unit
    def test_dft_job_name_drops_run_prefix_and_section(self):
        from alomancy.high_accuracy_evaluation.dft.run_vasp import run_sp_vasp

        assert dft_job_name(run_sp_vasp, 0) == "sp_vasp_0"

    @pytest.mark.unit
    def test_projection_reproduces_the_failing_advr_path_length(self):
        """The real failure: VASP was handed this 255-character directory."""
        path = projected_remote_output_dir(
            _ADVR_OLD_RUNDIR, "run_sp_vasp_high_accuracy_evaluation_0", _ADVR_OUT_DIR
        )
        assert len(path) == 255
        with pytest.raises(ValueError, match="255 characters"):
            check_remote_path_length(
                path, remote_rundir=_ADVR_OLD_RUNDIR, context="test"
            )

    @pytest.mark.unit
    def test_new_layout_fits(self):
        path = projected_remote_output_dir(_ADVR_NEW_RUNDIR, "sp_vasp_0", _ADVR_OUT_DIR)
        assert len(path) == 199
        check_remote_path_length(path, remote_rundir=_ADVR_NEW_RUNDIR, context="test")

    @pytest.mark.unit
    def test_limit_is_inclusive(self):
        ok = "/" + "a" * (MAX_REMOTE_OUTPUT_DIR_LEN - 1)
        check_remote_path_length(ok, remote_rundir="/a", context="test")
        with pytest.raises(ValueError):
            check_remote_path_length(ok + "a", remote_rundir="/a", context="test")

    @pytest.mark.unit
    def test_relative_rundir_says_home_not_counted(self):
        with pytest.raises(ValueError, match="remote \\$HOME"):
            check_remote_path_length("x" * 300, remote_rundir="scratch", context="t")

    @pytest.mark.unit
    def test_submitter_raises_before_submitting(
        self, tmp_path, monkeypatch, expyre_systems
    ):
        monkeypatch.chdir(tmp_path)
        captured: dict = {}
        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _fake_run_and_wait(captured),
        )
        expyre_systems["raven_cpu"] = _system("/scratch/" + "x" * 200)
        remote_info = MagicMock()
        remote_info.sys_name = "raven_cpu"

        def run_sp_vasp(**kwargs):
            pass

        with pytest.raises(ValueError, match="DFT job sp_vasp_0"):
            ase_remote_submitter(
                remote_info=remote_info,
                base_name="initialization",
                input_atoms_list=[Atoms("H")],
                function=run_sp_vasp,
            )
        assert "job_configs" not in captured

    @pytest.mark.unit
    def test_submitter_names_single_function_jobs(
        self, tmp_path, monkeypatch, expyre_systems
    ):
        monkeypatch.chdir(tmp_path)
        captured: dict = {}
        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _fake_run_and_wait(captured),
        )
        expyre_systems["raven_cpu"] = _system(_ADVR_NEW_RUNDIR)
        remote_info = MagicMock()
        remote_info.sys_name = "raven_cpu"

        def run_sp_qe(**kwargs):
            pass

        ase_remote_submitter(
            remote_info=remote_info,
            base_name="al_loop_0",
            input_atoms_list=[Atoms("H"), Atoms("H")],
            function=run_sp_qe,
        )
        names = [c["job_name"] for c in captured["job_configs"]]
        assert names == ["sp_qe_0", "sp_qe_1"]

    @pytest.mark.unit
    def test_expyre_remote_rundir_unknown_system_is_none(self, expyre_systems):
        assert expyre_remote_rundir("nope") is None
