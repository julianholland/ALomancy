"""Construction-time check that the longest remote DFT run directory fits
VASP/QE's path limit (ActiveLearningWorkflow._validate_remote_dft_path_length)."""

from types import SimpleNamespace

import pytest
from expyre.system import System

from alomancy.core.active_learning_workflow import ActiveLearningWorkflow


@pytest.fixture
def expyre_systems(monkeypatch):
    from expyre import config as expyre_config

    systems: dict = {}
    monkeypatch.setattr(expyre_config, "systems", systems)
    return systems


def _workflow(jobs_dict):
    return SimpleNamespace(start_loop=0, num_of_al_loops=5, jobs_dict=jobs_dict)


def _hae(evaluator="vasp"):
    return {"evaluator": evaluator, "hpc": {"hpc_name": "raven_cpu"}}


def _validate(workflow, hae):
    ActiveLearningWorkflow._validate_remote_dft_path_length(workflow, hae)


@pytest.mark.unit
def test_long_rundir_fails_at_construction(expyre_systems):
    expyre_systems["raven_cpu"] = System(
        host="raven",
        partitions={},
        scheduler="slurm",
        rundir="/raven/ptmp/advr/alomancy_scratch",
        rundir_extra="unkownhost-_work_home_advr_" + "x" * 40,
    )
    with pytest.raises(ValueError, match="longest possible path"):
        _validate(_workflow({"initialization": {}}), _hae())


@pytest.mark.unit
def test_shortened_rundir_passes(expyre_systems):
    expyre_systems["raven_cpu"] = System(
        host="raven",
        partitions={},
        scheduler="slurm",
        rundir="/raven/ptmp/advr/alomancy_scratch",
        rundir_extra="Simulations_alomancy_test_5",
    )
    _validate(_workflow({"initialization": {}}), _hae())


@pytest.mark.unit
def test_unknown_hpc_is_skipped(expyre_systems):
    _validate(_workflow({}), _hae())
