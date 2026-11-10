"""Construction-time check that the longest remote DFT run directory fits
the evaluator's path limit (VASP's; QE has none) (ActiveLearningWorkflow._validate_remote_dft_path_length)."""

from types import SimpleNamespace

import pytest
from expyre.system import System

from alomancy.core.active_learning_workflow import ActiveLearningWorkflow
from alomancy.high_accuracy_evaluation.dft.run_vasp import (
    MAX_OUTPUT_DIR_LEN as VASP_LIMIT,
)


@pytest.fixture
def expyre_systems(monkeypatch):
    from expyre import config as expyre_config

    systems: dict = {}
    monkeypatch.setattr(expyre_config, "systems", systems)
    return systems


def _workflow(jobs_dict, num_of_al_loops=5, per_loop=50):
    return SimpleNamespace(
        start_loop=0,
        num_of_al_loops=num_of_al_loops,
        num_of_structures_per_loop=per_loop,
        jobs_dict=jobs_dict,
    )


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


def _rundir_system(length: int) -> System:
    return System(
        host="raven", partitions={}, scheduler="slurm", rundir="/" + "r" * (length - 1)
    )


@pytest.mark.unit
def test_projection_uses_the_last_dft_loop_not_start_plus_total(expyre_systems):
    """al_loop_{num_of_al_loops} only trains, and start_loop doesn't add to
    the loop count: a run with 10 loops projects al_loop_9. The rundir is
    sized so al_loop_9 just fits and the old al_loop_15 projection didn't."""
    from alomancy.remote_submission.submitters import (
        projected_remote_output_dir,
    )

    tail = "results/al_loop_9/high_accuracy_evaluation/batch_9/ase_output_999"
    rundir_len = VASP_LIMIT - len(projected_remote_output_dir("", "go_vasp_999", tail))
    expyre_systems["raven_cpu"] = _rundir_system(rundir_len)
    workflow = _workflow({}, num_of_al_loops=10)
    workflow.start_loop = 5
    _validate(workflow, _hae())


@pytest.mark.unit
def test_index_width_follows_num_of_structures_per_loop(expyre_systems):
    from alomancy.remote_submission.submitters import (
        projected_remote_output_dir,
    )

    tail = "results/al_loop_4/high_accuracy_evaluation/batch_9/ase_output_999"
    rundir_len = VASP_LIMIT - len(projected_remote_output_dir("", "go_vasp_999", tail))
    expyre_systems["raven_cpu"] = _rundir_system(rundir_len)
    _validate(_workflow({}, per_loop=50), _hae())  # 3-digit index: fits
    with pytest.raises(ValueError):
        _validate(_workflow({}, per_loop=5000), _hae())  # 4 digits: over


@pytest.mark.unit
def test_qe_has_no_path_limit(expyre_systems):
    """Only VASP's Fortran path buffer is checked: the same long rundir
    that fails for VASP passes for QE."""
    expyre_systems["raven_cpu"] = _rundir_system(300)
    with pytest.raises(ValueError, match="longest possible path"):
        _validate(_workflow({}), _hae("vasp"))
    _validate(_workflow({}), _hae("qe"))
