"""Direct (non-mocked-away) tests of remote_submission/submitters.py's
ase_remote_submitter (training, prediction and MD go through
executor.submit_n). Each test here mocks only
RemoteJobExecutor.run_and_wait (the actual remote/ssh boundary), letting the
submitter's own job_config-construction logic run for real.
"""

from unittest.mock import MagicMock

import pytest
from ase import Atoms

from alomancy.remote_submission.submitters import ase_remote_submitter


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
        only by which function each job's config points at and a
        {fn.__name__}_{job_name}_{i} job name."""
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
        assert job_configs[0]["job_name"] == "run_go_high_accuracy_evaluation_0"
        assert job_configs[1]["job_name"] == "run_sp_high_accuracy_evaluation_1"
        assert job_configs[0]["function"] is run_go
        assert job_configs[1]["function"] is run_sp
