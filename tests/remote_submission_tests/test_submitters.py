"""Direct (non-mocked-away) tests of remote_submission/submitters.py.

Previously these four functions were only ever exercised indirectly, via
mocks in the tests of their callers (train_mlip, generate_structures,
high_accuracy_evaluation) -- see docs/remote_submission_architecture.md
sections 4 and 7 for the full audit. Each test here mocks only
RemoteJobExecutor.run_and_wait (the actual remote/ssh boundary), letting the
submitter's own job_config-construction logic run for real.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from alomancy.remote_submission.submitters import (
    all_maces_remote_submitter,
    ase_remote_submitter,
    committee_remote_submitter,
)


def _fake_run_and_wait(captured: dict):
    def _run(self, function, job_configs, **kwargs):
        captured["job_configs"] = job_configs
        return [None] * len(job_configs)

    return _run


class TestCommitteeRemoteSubmitter:
    @pytest.mark.unit
    def test_fit_indices_output_files_keyed_by_own_index_not_position(
        self, tmp_path, monkeypatch
    ):
        """fit_indices=[2, 4] (a non-contiguous backfill subset, e.g.
        train_mlip retrying only the committee members lost to a prior
        failure) must produce output_files keyed by each fit's OWN index
        (fit_2, fit_4), not by its position in the fit_indices list -- the
        latter would wrongly write fit_0/fit_1, silently overwriting two
        unrelated, already-successful committee members. Regression test
        for the exact bug class committee_remote_submitter's own docstring
        claims to guard against; this was previously untested at any
        level."""
        monkeypatch.chdir(tmp_path)
        captured: dict = {}
        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _fake_run_and_wait(captured),
        )

        remote_info = MagicMock()
        remote_info.job_name = "mlip_committee"

        committee_remote_submitter(
            remote_info=remote_info,
            base_name="al_loop_6",
            function=MagicMock(),
            seed=803,
            fit_indices=[2, 4],
        )

        job_configs = captured["job_configs"]
        assert len(job_configs) == 2
        expected_fit_2 = str(Path("results", "al_loop_6", "mlip_committee", "fit_2"))
        expected_fit_4 = str(Path("results", "al_loop_6", "mlip_committee", "fit_4"))
        assert job_configs[0]["output_files"] == [expected_fit_2]
        assert job_configs[0]["function_kwargs"]["fit_idx"] == 2
        assert job_configs[0]["function_kwargs"]["seed"] == 803 + 2
        assert job_configs[1]["output_files"] == [expected_fit_4]
        assert job_configs[1]["function_kwargs"]["fit_idx"] == 4
        assert job_configs[1]["function_kwargs"]["seed"] == 803 + 4

    @pytest.mark.unit
    def test_default_fit_indices_covers_full_committee_in_order(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        captured: dict = {}
        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _fake_run_and_wait(captured),
        )

        remote_info = MagicMock()
        remote_info.job_name = "mlip_committee"

        committee_remote_submitter(
            remote_info=remote_info,
            base_name="al_loop_0",
            function=MagicMock(),
            seed=803,
            size_of_committee=3,
        )

        fit_idxs = [c["function_kwargs"]["fit_idx"] for c in captured["job_configs"]]
        assert fit_idxs == [0, 1, 2]


class TestAllMacesRemoteSubmitter:
    @pytest.mark.unit
    def test_job_name_defaults_to_mace_eval_prefixed(self, monkeypatch):
        """Regression coverage for the v0.6.0 fix: without a distinct
        prefix, this job's name collides with the MD jobs submitted just
        before it in the same structure_generation phase (both would
        otherwise be remote_info.job_name verbatim)."""
        captured: dict = {}

        def _run(self, function, job_configs, **kwargs):
            captured["job_configs"] = job_configs
            return [{"structure_0": 0.1}]

        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _run,
        )

        remote_info = MagicMock()
        remote_info.job_name = "structure_generation"

        result = all_maces_remote_submitter(
            remote_info=remote_info,
            function=MagicMock(),
            function_kwargs={"structure_list": [MagicMock(), MagicMock()]},
        )

        assert (
            captured["job_configs"][0]["job_name"] == "mace_eval_structure_generation"
        )
        assert result == {"structure_0": 0.1}

    @pytest.mark.unit
    def test_explicit_job_name_overrides_default(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            "alomancy.remote_submission.executor.RemoteJobExecutor.run_and_wait",
            _fake_run_and_wait(captured),
        )

        remote_info = MagicMock()
        remote_info.job_name = "structure_generation"

        all_maces_remote_submitter(
            remote_info=remote_info,
            function=MagicMock(),
            job_name="custom_name",
        )

        assert captured["job_configs"][0]["job_name"] == "custom_name"


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
