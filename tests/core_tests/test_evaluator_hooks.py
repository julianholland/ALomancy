"""The workflow's use of the evaluator registry hooks: construction-time
evaluator validation (ORCA), evaluator-dependent initialisation defaults,
and the database check before initialisation (level of theory, periodic
structures with a non-periodic evaluator)."""

import logging

import numpy as np
import pytest
from ase import Atoms

from alomancy.core import active_learning_workflow
from alomancy.core.committee_uncertainty_workflow import CommitteeUncertaintyWorkflow
from alomancy.high_accuracy_evaluation.dft import run_qe
from alomancy.high_accuracy_evaluation.level_of_theory import stamp_level_of_theory

_HPC = {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]}
_ORCA_HPC = {
    **_HPC,
    "orca_path": "/opt/orca/orca",
    "node_info": {
        "ranks_per_system": 8,
        "ranks_per_node": 16,
        "threads_per_rank": 1,
        "max_mem_per_node": "64GB",
    },
}


def _jobs_dict(evaluator: str = "qe", **evaluator_settings) -> dict:
    phase = {"max_time": "1H", "hpc": dict(_HPC)}
    hae = {
        "max_time": "10m",
        "max_vdw_volume_ratio": None,
        "hpc": dict(_ORCA_HPC if evaluator == "orca" else _HPC),
        "evaluator": evaluator,
        **evaluator_settings,
    }
    return {
        "general": {
            "al_workflow": "committee_uncertainty",
            "elements": ["H", "O"],
            "num_of_al_loops": 1,
            "plots": False,
            "dataset_kwargs": {
                "target_config_types": ["IsolatedAtom"],
                "test_ratio": 0.1,
            },
            "committee_uncertainty_kwargs": {"num_of_models_in_committee": 3},
        },
        "initialization": dict(phase),
        "training": {**phase, "trainer": "mace"},
        "structure_generation": dict(phase),
        "high_accuracy_evaluation": hae,
    }


def _workflow(jobs_dict, shared_db) -> CommitteeUncertaintyWorkflow:
    wf = CommitteeUncertaintyWorkflow(jobs_dict=jobs_dict)
    wf.db = shared_db
    return wf


def _labelled(periodic: bool, level=None) -> Atoms:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6, 6, 6])
    atoms.pbc = periodic
    atoms.info["config_type"] = "init_dimer"
    atoms.info["REF_energy"] = -1.0
    atoms.arrays["REF_forces"] = np.zeros((2, 3))
    if level is not None:
        stamp_level_of_theory([atoms], level)
    return atoms


def _records():
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    return records, handler


@pytest.mark.unit
def test_orca_kwargs_are_validated_at_construction(shared_db):
    with pytest.raises(ValueError, match="doesn't fit the job"):
        _workflow(
            _jobs_dict(
                "orca", orca_kwargs={"orcasimpleinput": "B3LYP def2-SVP", "nprocs": 64}
            ),
            shared_db,
        )


@pytest.mark.unit
def test_orca_runs_reject_periodic_initialization_blocks(shared_db):
    jobs = _jobs_dict("orca", orca_kwargs={"orcasimpleinput": "B3LYP def2-SVP"})
    jobs["initialization"]["amorphous_kwargs"] = {"enabled": True}
    with pytest.raises(ValueError, match="amorphous_cluster_kwargs"):
        _workflow(jobs, shared_db)


@pytest.mark.unit
def test_orca_runs_default_to_non_periodic_initialization(shared_db):
    wf = _workflow(
        _jobs_dict("orca", orca_kwargs={"orcasimpleinput": "B3LYP def2-SVP"}),
        shared_db,
    )
    assert wf.evaluator_supports_periodic is False
    needs = active_learning_workflow.resolve("initialiser", "default").compute_needs(
        shared_db,
        wf.jobs_dict["initialization"],
        ["H", "O"],
        evaluator_supports_periodic=wf.evaluator_supports_periodic,
    )
    assert needs["amorphous_override"] == 0 and needs["mp_structures"] is False
    assert needs["amorphous_cluster_override"] > 0


@pytest.mark.unit
def test_correlated_cap_below_configured_cluster_size_raises(shared_db):
    jobs = _jobs_dict(
        "orca",
        orca_kwargs={
            "orcasimpleinput": "RI-MP2 def2-SVP def2-SVP/C",
            "max_num_of_atoms_correlated": 5,
        },
    )
    with pytest.raises(ValueError, match="up to 5 atoms"):
        _workflow(jobs, shared_db)


@pytest.mark.unit
def test_periodic_structures_in_the_database_raise_for_orca(shared_db):
    wf = _workflow(
        _jobs_dict("orca", orca_kwargs={"orcasimpleinput": "B3LYP def2-SVP"}),
        shared_db,
    )
    shared_db.add_structures([_labelled(periodic=True)])
    with pytest.raises(ValueError, match="periodic structures"):
        wf._check_database_against_evaluator()


@pytest.mark.unit
def test_level_of_theory_mismatch_warns(shared_db):
    wf = _workflow(_jobs_dict("qe"), shared_db)
    other = run_qe.level_of_theory({"qe_kwargs": {"system": {"input_dft": "pbesol"}}})
    shared_db.add_structures(
        [
            _labelled(periodic=True, level=other),
            # Labelled before the stamp existed: unknown, not a mismatch.
            _labelled(periodic=True),
        ],
        skip_duplicates=False,
    )
    records, handler = _records()
    active_learning_workflow.logger.addHandler(handler)
    try:
        wf._check_database_against_evaluator()
    finally:
        active_learning_workflow.logger.removeHandler(handler)
    mismatch = [
        r for r in records if getattr(r, "event", None) == "level_of_theory_mismatch"
    ]
    assert len(mismatch) == 1
    assert "differs in method" in mismatch[0].getMessage()
    assert mismatch[0].data["others"] == {other.identity_id(): 1}


@pytest.mark.unit
def test_matching_level_of_theory_does_not_warn(shared_db):
    wf = _workflow(_jobs_dict("qe"), shared_db)
    same = wf._evaluator_level_of_theory()
    shared_db.add_structures([_labelled(periodic=True, level=same)])
    records, handler = _records()
    active_learning_workflow.logger.addHandler(handler)
    try:
        wf._check_database_against_evaluator()
    finally:
        active_learning_workflow.logger.removeHandler(handler)
    assert not [
        r for r in records if getattr(r, "event", None) == "level_of_theory_mismatch"
    ]


@pytest.mark.unit
def test_summary_shows_the_level_of_theory(shared_db):
    wf = _workflow(
        _jobs_dict(
            "orca", orca_kwargs={"orcasimpleinput": "RI-MP2 def2-SVP def2-SVP/C"}
        ),
        shared_db,
    )
    records, handler = _records()
    active_learning_workflow.logger.addHandler(handler)
    try:
        wf.display_workflow_summary()
    finally:
        active_learning_workflow.logger.removeHandler(handler)
    text = "\n".join(r.getMessage() for r in records)
    assert "level of theory: CORRELATED RI-MP2" in text
