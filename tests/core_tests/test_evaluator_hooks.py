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


_AIMS_HPC = {
    **_HPC,
    "aims_path": "/opt/aims/aims.x",
    "aims_species_dir": "/opt/aims/species_defaults/defaults_2020",
    "pre_cmds": ["ulimit -s unlimited"],
    "node_info": dict(_ORCA_HPC["node_info"]),
}


def _aims_jobs_dict(**general) -> dict:
    jobs = _jobs_dict("qe")
    jobs["high_accuracy_evaluation"]["evaluator"] = "aims"
    jobs["high_accuracy_evaluation"]["hpc"] = dict(_AIMS_HPC)
    jobs["general"].update(general)
    return jobs


def _events(wf, name: str) -> list[logging.LogRecord]:
    records, handler = _records()
    active_learning_workflow.logger.addHandler(handler)
    try:
        wf._check_database_against_evaluator()
    finally:
        active_learning_workflow.logger.removeHandler(handler)
    return [r for r in records if getattr(r, "event", None) == name]


@pytest.mark.unit
def test_aims_kwargs_are_validated_at_construction(shared_db):
    jobs = _aims_jobs_dict()
    jobs["high_accuracy_evaluation"]["aims_kwargs"] = {"k_grid": [2, 2, 2]}
    with pytest.raises(ValueError, match="k_grid can't be set"):
        _workflow(jobs, shared_db)


@pytest.mark.unit
def test_labels_from_another_code_raise(shared_db):
    wf = _workflow(_aims_jobs_dict(), shared_db)
    shared_db.add_structures(
        [_labelled(periodic=True, level=run_qe.level_of_theory({}))]
    )
    with pytest.raises(ValueError, match=r"another code.*1 structure\(s\) from qe"):
        wf._check_database_against_evaluator()


@pytest.mark.unit
def test_labels_from_another_code_warn_with_the_opt_out(shared_db):
    wf = _workflow(_aims_jobs_dict(allow_mixed_level_of_theory=True), shared_db)
    shared_db.add_structures(
        [_labelled(periodic=True, level=run_qe.level_of_theory({}))]
    )
    assert len(_events(wf, "level_of_theory_mismatch")) == 1


@pytest.mark.unit
def test_allow_mixed_level_of_theory_must_be_a_bool(shared_db):
    with pytest.raises(ValueError, match="allow_mixed_level_of_theory"):
        _workflow(_aims_jobs_dict(allow_mixed_level_of_theory="yes"), shared_db)


@pytest.mark.unit
def test_unstamped_labels_warn_until_current_level_data_exists(shared_db):
    wf = _workflow(_aims_jobs_dict(), shared_db)
    shared_db.add_structures([_labelled(periodic=True)])
    assert len(_events(wf, "level_of_theory_unknown")) == 1
    current = _labelled(periodic=True, level=wf._evaluator_level_of_theory())
    current.positions[1, 2] = 0.8  # not a duplicate of the first
    shared_db.add_structures([current])
    assert _events(wf, "level_of_theory_unknown") == []


# --- few target structures: derived structures, projected check -----------


def _bulk(config_type: str, scale: float = 1.0) -> Atoms:
    from ase.build import bulk

    atoms = bulk("Pt", "fcc", a=3.92 * scale)
    atoms.info["config_type"] = config_type
    return atoms


def _target_workflow(shared_db, targets=("init_MP",)) -> CommitteeUncertaintyWorkflow:
    jobs = _jobs_dict("qe")
    jobs["high_accuracy_evaluation"]["max_vdw_volume_ratio"] = 10
    # Pt at the default 1 g/cm^3 raises at construction (every cell blown up).
    jobs["initialization"]["amorphous_kwargs"] = {"density_range": [15.0, 21.0]}
    jobs["general"]["elements"] = ["Pt"]
    jobs["general"]["dataset_kwargs"]["target_config_types"] = list(targets)
    return _workflow(jobs, shared_db)


@pytest.mark.unit
def test_derived_structures_fill_the_test_set(shared_db):
    """One MP structure (kept for training) plus its stretch/compress copies:
    the copies are test-eligible, so the test set isn't empty."""
    wf = _target_workflow(shared_db)
    structures = [_bulk("init_MP")] + [
        _bulk("init_stretch_compress", s) for s in (0.9, 0.95, 1.05, 1.1)
    ]
    train, test = wf._split_untagged(structures, wf.dataset_kwargs)
    assert test
    assert {a.info["config_type"] for a in test} <= {"init_stretch_compress"}
    assert "init_MP" in {a.info["config_type"] for a in train}


@pytest.mark.unit
def test_one_target_and_nothing_derived_raises_before_dft(shared_db):
    wf = _target_workflow(shared_db)
    with pytest.raises(ValueError, match="at most 1 structure"):
        wf._check_projected_test_set([_bulk("init_MP"), _bulk("init_dimer")])


@pytest.mark.unit
def test_few_target_structures_warn(shared_db):
    wf = _target_workflow(shared_db)
    records, handler = _records()
    active_learning_workflow.logger.addHandler(handler)
    try:
        wf._check_projected_test_set(
            [
                _bulk("init_MP"),
                _bulk("init_stretch_compress", 1.05),
                _bulk("init_rattle"),
            ]
        )
        enough = [_bulk("init_MP")] + [
            _bulk("init_stretch_compress", s) for s in (0.9, 0.95, 1.05, 1.1)
        ]
        wf._check_projected_test_set(enough)
    finally:
        active_learning_workflow.logger.removeHandler(handler)
    events = [
        r for r in records if getattr(r, "event", None) == "few_target_structures"
    ]
    assert len(events) == 1
    assert events[0].data["total"] == 3


@pytest.mark.unit
def test_blown_up_targets_are_not_counted(shared_db):
    """Amorphous cells the blown-up filter drops before DFT never reach the
    test set (the Pt run's 1 g/cm^3 cells)."""
    wf = _target_workflow(shared_db, targets=("init_amorphous",))
    sparse = [_bulk("init_amorphous", 3.0) for _ in range(5)]  # 27x the volume
    with pytest.raises(ValueError, match="at most 0 structure"):
        wf._check_projected_test_set(sparse)


@pytest.mark.unit
def test_restart_with_unmet_needs_reuses_results_without_duplicating(
    shared_db, tmp_path, monkeypatch
):
    """The Pt run: initialisation DFT finished, but its amorphous cells were
    all dropped (blown up), so every restart re-entered initialisation,
    reloaded the cached results and added the dimers etc. again
    (19 -> 36 -> 53). Now they are recognised by digest and the unmet need
    is reported once instead of regenerated."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    monkeypatch.chdir(tmp_path)
    wf = _target_workflow(shared_db)
    results = []
    for i in range(3):
        atoms = _labelled(periodic=True)
        atoms.positions[1, 2] = 0.7 + 0.1 * i
        atoms.info["dft_input_digest"] = f"digest{i}"
        # As the workers write it: the DFT results on the calculator.
        atoms.calc = SinglePointCalculator(
            atoms, energy=atoms.info["REF_energy"], forces=atoms.arrays["REF_forces"]
        )
        results.append(atoms)
    work = tmp_path / "results" / "initialization"
    work.mkdir(parents=True)
    write(work / "high_accuracy_eval_results.xyz", results, format="extxyz")
    (work / "high_accuracy_eval.done").write_text("done\n")
    shared_db.add_structures(results)
    size = shared_db.size

    records, handler = _records()
    active_learning_workflow.logger.addHandler(handler)
    try:
        for _ in range(2):  # two restarts
            wf._initialize_training_set("initialization")
    finally:
        active_learning_workflow.logger.removeHandler(handler)
    assert shared_db.size == size
    unmet = [
        r for r in records if getattr(r, "event", None) == "initialization_needs_unmet"
    ]
    assert unmet and "amorphous" in unmet[0].getMessage()


@pytest.mark.unit
@pytest.mark.parametrize(
    "need",
    [
        {"amorphous_cluster_override": 50},
        {"molecule_cluster_override": {"H8O4": 2}},
        {"crystal_override": {"Pt": ["bcc"]}},
    ],
)
def test_a_shortfall_in_one_block_alone_starts_initialisation(need):
    """An ORCA run whose DB already held its atoms, dimers and trimers
    never generated its clusters: _needs_anything ignored them."""
    from alomancy.core.active_learning_workflow import _describe_needs, _needs_anything

    nothing = {
        "isolated_atoms": [],
        "dimer_override": {},
        "trimer_override": {},
        "amorphous_override": 0,
        "mp_structures": False,
        "molecule_override": {},
        "box_override": {},
        "amorphous_cluster_override": 0,
        "molecule_cluster_override": {},
        "crystal_override": {},
    }
    assert not _needs_anything(nothing)
    assert _needs_anything({**nothing, **need})
    assert _describe_needs({**nothing, **need}) != "none"


@pytest.mark.unit
def test_database_is_checked_when_resuming_after_finished_loops(
    shared_db, tmp_path, monkeypatch
):
    """The check ran only inside initialisation, which a run with a finished
    loop skips: restarting a QE run as FHI-aims mixed energy scales."""
    monkeypatch.chdir(tmp_path)
    wf = _workflow(_aims_jobs_dict(), shared_db)
    shared_db.add_structures(
        [_labelled(periodic=True, level=run_qe.level_of_theory({}))]
    )
    done = tmp_path / "results" / "al_loop_0"
    done.mkdir(parents=True)
    (done / "loop.done").write_text("done\n")
    monkeypatch.setattr(wf, "pre_run_checks", lambda: None)
    monkeypatch.setattr(wf, "_write_run_config", lambda: None)
    with pytest.raises(ValueError, match="another code"):
        wf.prepare_run()


@pytest.mark.unit
def test_candidates_above_the_evaluator_cap_never_reach_selection(
    shared_db, tmp_path, monkeypatch
):
    """The cap used to apply only at the DFT step, after selection: every
    selected candidate could be over it, and the loop added nothing."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from alomancy.core.active_learning_workflow import LoopContext, TrainedModel
    from alomancy.registry import resolve as real_resolve

    monkeypatch.chdir(tmp_path)
    jobs = _jobs_dict(
        "orca",
        orca_kwargs={
            "orcasimpleinput": "RI-MP2 def2-SVP def2-SVP/C",
            "max_num_of_atoms_correlated": 6,
        },
    )
    jobs["initialization"]["amorphous_cluster_kwargs"] = {"num_of_atoms_per_cluster": 5}
    wf = _workflow(jobs, shared_db)
    small = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    big = Atoms("H8O4", positions=np.arange(36).reshape(12, 3) * 1.1)
    generator = MagicMock()
    generator.generate.return_value = [small, big]

    def fake_resolve(category, name):
        return (
            generator
            if category == "structure_generator"
            else real_resolve(category, name)
        )

    ctx = LoopContext(
        loop=0,
        base_name="al_loop_0",
        workdir=Path("results/al_loop_0"),
        train=[small],
        test=[],
        train_only=False,
        plots_dir=Path("results/plots"),
    )
    model = TrainedModel(
        fit_idx=0,
        seed=803,
        model_path="m.model",
        compiled_model_path="m.model",
        metrics={},
        fit_dir=Path("results/al_loop_0/training/fit_0"),
    )
    with patch.object(active_learning_workflow, "resolve", side_effect=fake_resolve):
        kept = wf.generate_candidates(ctx, model)
    assert [len(a) for a in kept] == [3]
