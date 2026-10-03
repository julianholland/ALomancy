"""Tests for core/novelty_selection_workflow.py -- picks the candidates least
like each other and the database's non-redundant structures.

Dimers make the descriptor geometry easy to control: a dimer's char_vec is
its bond length repeated, so two dimers are |d1 - d2| * sqrt(128) apart.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from alomancy.core.active_learning_workflow import LoopContext, TrainedModel
from alomancy.core.entry import ALomancy
from alomancy.core.novelty_selection_workflow import (
    NoveltySelectionWorkflow,
    select_novel_indices,
)
from alomancy.utils.remove_redundancy import _cached_descriptors, descriptors_for_atoms


@pytest.fixture
def jobs_dict(minimal_jobs_dict):
    minimal_jobs_dict["general"] = {
        "al_workflow": "novelty_selection",
        "elements": ["H"],
        "num_of_al_loops": 2,
        "plots": False,
        "dataset_kwargs": {"target_config_types": ["IsolatedAtom"], "test_ratio": 0.1},
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["num_of_models_in_committee"]
    minimal_jobs_dict["structure_generation"]["desired_num_of_structures"] = 2
    return minimal_jobs_dict


def _workflow(jobs_dict, shared_db) -> NoveltySelectionWorkflow:
    wf = ALomancy(jobs_dict).workflow
    assert isinstance(wf, NoveltySelectionWorkflow)
    wf.db = shared_db
    return wf


def _ctx(loop: int = 0) -> LoopContext:
    base_name = f"al_loop_{loop}"
    return LoopContext(
        loop=loop,
        base_name=base_name,
        workdir=Path("results", base_name),
        train=[],
        test=[],
        train_only=False,
        plots_dir=Path("results/current_plots", base_name),
    )


_MODEL = TrainedModel(
    fit_idx=0,
    seed=803,
    model_path="model_0.pt",
    compiled_model_path=None,
    metrics={},
    fit_dir=Path("results/al_loop_0/training/fit_0"),
)


def _dimer(d: float, tag: int | None = None, labelled: bool = False) -> Atoms:
    a = Atoms("H2", positions=[[0, 0, 0], [d, 0, 0]], cell=[20] * 3, pbc=True)
    if tag is not None:
        a.info["candidate"] = tag
    if labelled:
        a.info["config_type"] = "init_dimer"
        a.info["REF_energy"] = -1.0
        a.arrays["REF_forces"] = np.zeros((2, 3))
    return a


def _candidates(*bond_lengths: float) -> list[Atoms]:
    return [_dimer(d, tag=i) for i, d in enumerate(bond_lengths)]


def _store(db, *bond_lengths: float) -> None:
    db.add_structures(
        [_dimer(d, labelled=True) for d in bond_lengths],
        split="train",
        skip_duplicates=False,
    )


@pytest.mark.unit
def test_candidate_descriptors_match_cached_db_descriptors(shared_db):
    _store(shared_db, 1.3)
    containers = list(shared_db.partition.list_containers())
    cached = _cached_descriptors(shared_db, containers, [0], 128)
    np.testing.assert_array_equal(descriptors_for_atoms([_dimer(1.3)]), cached)


@pytest.mark.unit
def test_picks_most_distinct_candidates(tmp_path, jobs_dict, monkeypatch, shared_db):
    """A tight cluster loses to two isolated candidates."""
    monkeypatch.chdir(tmp_path)
    _store(shared_db, 5.0)
    wf = _workflow(jobs_dict, shared_db)

    cands = _candidates(1.0, 1.01, 1.02, 2.0, 3.0)
    with patch.object(wf, "generate_candidates", return_value=cands):
        selected = wf.select_novel(_ctx(0), _MODEL)

    assert [a.info["candidate"] for a in selected] == [3, 4]
    saved = read(
        "results/al_loop_0/structure_generation/novelty_selection_structures.xyz", ":"
    )
    assert [a.info["candidate"] for a in saved] == [3, 4]


@pytest.mark.unit
def test_candidate_duplicating_the_database_is_not_novel(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    """Candidate 0 is as far from the other candidates as they are from each
    other, but it repeats a database structure."""
    monkeypatch.chdir(tmp_path)
    _store(shared_db, 1.0, 8.0)
    wf = _workflow(jobs_dict, shared_db)

    with patch.object(
        wf, "generate_candidates", return_value=_candidates(1.0, 2.5, 4.0)
    ):
        selected = wf.select_novel(_ctx(0), _MODEL)

    assert [a.info["candidate"] for a in selected] == [1, 2]


@pytest.mark.unit
def test_reference_set_skips_redundant_structures(jobs_dict, shared_db):
    _store(shared_db, 1.0, 2.0, 3.0)
    shared_db.flag_as_duplicates([1])
    wf = _workflow(jobs_dict, shared_db)

    reference = wf.reference_descriptors()

    np.testing.assert_allclose(reference[:, 0], [1.0, 3.0])


@pytest.mark.unit
def test_takes_all_candidates_when_too_few(tmp_path, jobs_dict, monkeypatch, shared_db):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)

    with (
        patch.object(wf, "generate_candidates", return_value=_candidates(1.0, 2.0)),
        patch.object(wf, "reference_descriptors") as reference,
    ):
        selected = wf.select_novel(_ctx(0), _MODEL)

    assert len(selected) == 2
    reference.assert_not_called()


@pytest.mark.unit
def test_reloads_saved_selection_when_done(tmp_path, jobs_dict, monkeypatch, shared_db):
    monkeypatch.chdir(tmp_path)
    _store(shared_db, 5.0)
    wf = _workflow(jobs_dict, shared_db)
    with patch.object(
        wf, "generate_candidates", return_value=_candidates(1.0, 1.01, 2.0, 3.0)
    ):
        first = wf.select_novel(_ctx(0), _MODEL)

    resumed = _workflow(jobs_dict, shared_db)
    with patch.object(resumed, "generate_candidates") as gen:
        again = resumed.select_novel(_ctx(0), _MODEL)

    gen.assert_not_called()
    assert [a.info["candidate"] for a in again] == [a.info["candidate"] for a in first]


@pytest.mark.unit
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_never_selects_more_than_n(n):
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 0.01, (20, 4))
    candidates = np.array(
        [
            [0, 0, 0, 0.001],
            [5, 0, 0, 0],
            [5.01, 0, 0, 0],
            [5.02, 0, 0, 0],
            [0, 9, 0, 0],
            [0, 0, -9, 0],
        ]
    )

    chosen, _ = select_novel_indices(reference, candidates, n)

    assert 0 < len(chosen) <= n
    assert 0 not in chosen  # duplicates the reference cluster
    assert chosen == sorted(chosen)


@pytest.mark.unit
def test_new_structures_labelled_novelty_selection(
    tmp_path, jobs_dict, monkeypatch, shared_db
):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(jobs_dict, shared_db)

    with patch(
        "alomancy.core.active_learning_workflow._evaluator_orchestrate",
        return_value=[_dimer(1.0, labelled=True)],
    ):
        (result,) = wf.high_accuracy_evaluate(_ctx(1), _candidates(1.0))

    assert result.info["config_type"] == "novelty_selection"
    assert result.info["al_loop"] == 1
