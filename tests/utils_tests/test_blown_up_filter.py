"""Tests for the blown-up-cell check before DFT
(utils/clean_structures.filter_blown_up_structures): the helper itself,
its config validation, and its two placements -- the workflow's
generate_candidates and high_accuracy_evaluation's final safety net."""

import contextlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from alomancy.utils.clean_structures import (
    filter_blown_up_structures,
    resolve_max_vdw_volume_ratio,
    resolve_vdw_exempt_config_types,
    vdw_volume_ratio,
)


def _crystal():
    """Bulk Si: a healthy condensed cell, vdW volume ratio about 0.3."""
    a = bulk("Si", "diamond", a=5.43)
    a.info["config_type"] = "high_sd"
    return a


def _blown_up(scale=10.0):
    """The same cell with every lattice vector scaled by *scale*."""
    a = _crystal()
    a.set_cell(a.cell * scale, scale_atoms=True)
    return a


class _Events(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        if getattr(record, "event", None) == "blown_up_excluded":
            self.records.append(record)


@pytest.fixture
def events():
    handler = _Events()
    logger = logging.getLogger("alomancy.utils.clean_structures")
    logger.addHandler(handler)
    yield handler.records
    logger.removeHandler(handler)


@pytest.mark.unit
def test_vdw_volume_ratio_of_a_crystal_and_its_blown_up_copy():
    healthy = vdw_volume_ratio(_crystal())
    assert 0.1 < healthy < 1.5
    assert vdw_volume_ratio(_blown_up(2.0)) == pytest.approx(8 * healthy)


@pytest.mark.unit
def test_blown_up_cells_are_excluded_and_warned_about(events):
    healthy, blown = _crystal(), _blown_up()
    kept = filter_blown_up_structures([healthy, blown], 10.0)

    assert kept == [healthy]
    (record,) = events
    assert record.data["n"] == 1
    assert record.data["config_types"] == ["high_sd"]
    message = record.getMessage()
    assert "md_kwargs.pressure" in message
    assert "max_vdw_volume_ratio" in message


@pytest.mark.unit
def test_non_finite_cells_are_excluded():
    nan_cell = _crystal()
    nan_cell.cell[0, 0] = np.inf
    assert filter_blown_up_structures([nan_cell], 10.0) == []


@pytest.mark.unit
@pytest.mark.parametrize("config_type", ["IsolatedAtom", "init_dimer", "init_trimer"])
def test_gas_phase_structures_are_exempt(config_type, events):
    """A lone atom, dimer or trimer sits in vacuum by design, even when
    DFT hands it back marked periodic."""
    gas = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[20] * 3, pbc=True)
    gas.info["config_type"] = config_type
    assert filter_blown_up_structures([gas], 10.0) == [gas]
    assert events == []


@pytest.mark.unit
def test_listed_config_types_are_exempt(events):
    """A config type the user lists (e.g. molecules in a vacuum box) is
    kept; the same geometry under another config type is not."""
    listed, other = _blown_up(), _blown_up()
    listed.info["config_type"] = "molecule_in_box"
    kept = filter_blown_up_structures(
        [listed, other], 10.0, frozenset({"molecule_in_box"})
    )
    assert kept == [listed]
    (record,) = events
    assert record.data["config_types"] == ["high_sd"]
    assert "vdw_exempt_config_types" in record.getMessage()


@pytest.mark.unit
def test_resolve_exempt_config_types():
    assert resolve_vdw_exempt_config_types(None) == frozenset()
    assert resolve_vdw_exempt_config_types({}) == frozenset()
    assert resolve_vdw_exempt_config_types(
        {"vdw_exempt_config_types": ["a", "b"]}
    ) == frozenset({"a", "b"})


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["molecule_in_box", [1, 2], {"a": 1}])
def test_resolve_rejects_bad_exempt_config_types(bad):
    with pytest.raises(ValueError, match="vdw_exempt_config_types"):
        resolve_vdw_exempt_config_types({"vdw_exempt_config_types": bad})


@pytest.mark.unit
def test_non_periodic_structures_are_exempt():
    molecule = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[20] * 3)
    assert filter_blown_up_structures([molecule], 10.0) == [molecule]


@pytest.mark.unit
def test_none_turns_the_check_off(events):
    blown = _blown_up()
    assert filter_blown_up_structures([blown], None) == [blown]
    assert events == []


@pytest.mark.unit
@pytest.mark.parametrize("bad", [1, 0.5, "x", True, -3, float("nan"), float("inf")])
def test_resolve_rejects_a_bad_ratio(bad):
    with pytest.raises(ValueError, match="max_vdw_volume_ratio"):
        resolve_max_vdw_volume_ratio({"max_vdw_volume_ratio": bad})


@pytest.mark.unit
def test_resolve_default_and_off():
    assert resolve_max_vdw_volume_ratio({}) == 10.0
    assert resolve_max_vdw_volume_ratio(None) == 10.0
    assert resolve_max_vdw_volume_ratio({"max_vdw_volume_ratio": None}) is None
    assert resolve_max_vdw_volume_ratio({"max_vdw_volume_ratio": 4}) == 4.0


@pytest.mark.unit
def test_dft_safety_net_never_submits_a_blown_up_cell(tmp_path, monkeypatch):
    """high_accuracy_evaluation drops blown-up cells before submission,
    whatever produced them (here: no generator at all)."""
    from alomancy.high_accuracy_evaluation import high_accuracy_calc_interface as hae

    monkeypatch.chdir(tmp_path)
    submitted = {}

    def fake_submit(*, input_atoms_list, **_kwargs):
        submitted["atoms"] = list(input_atoms_list)

    healthy, blown = _crystal(), _blown_up()
    with (
        patch.object(hae, "ase_remote_submitter", side_effect=fake_submit),
        pytest.raises(RuntimeError),  # the fake writes no results
    ):
        hae.high_accuracy_evaluation(
            [healthy, blown],
            {"evaluator": "qe"},
            base_name="test_loop",
            name="high_accuracy_evaluation",
            hpc={"hpc_name": "h", "pre_cmds": [], "partitions": ["p"]},
            max_time="1H",
        )
    assert len(submitted["atoms"]) == 1
    assert submitted["atoms"][0].cell.volume == pytest.approx(healthy.cell.volume)


@pytest.mark.unit
def test_generate_candidates_drops_blown_up_cells(
    tmp_path, minimal_jobs_dict, monkeypatch, shared_db
):
    """Blown-up candidates never reach selection, so they can't use up a
    DFT slot; the workflow reads the limit from high_accuracy_evaluation."""
    from alomancy.core.active_learning_workflow import LoopContext, TrainedModel
    from alomancy.core.committee_uncertainty_workflow import (
        CommitteeUncertaintyWorkflow,
    )

    monkeypatch.chdir(tmp_path)
    minimal_jobs_dict["high_accuracy_evaluation"]["max_vdw_volume_ratio"] = 10
    general = minimal_jobs_dict.setdefault("general", {})
    general.setdefault(
        "dataset_kwargs", {"test_ratio": 0.1, "target_config_types": ["IsolatedAtom"]}
    )
    general.update({"num_of_al_loops": 1, "plots": False})
    wf = CommitteeUncertaintyWorkflow(jobs_dict=minimal_jobs_dict)
    wf.db = shared_db
    generator = MagicMock()
    healthy = _crystal()
    generator.generate.return_value = [healthy, _blown_up()]
    ctx = LoopContext(
        loop=0,
        base_name="al_loop_0",
        workdir=tmp_path / "results" / "al_loop_0",
        train=[_crystal()],
        test=[],
        train_only=False,
        plots_dir=tmp_path / "plots",
    )
    model = TrainedModel(
        fit_idx=0,
        seed=803,
        model_path="model_0.pt",
        compiled_model_path=None,
        metrics={},
        fit_dir=tmp_path / "fit_0",
    )

    with patch(
        "alomancy.core.active_learning_workflow.resolve", return_value=generator
    ):
        result = wf.generate_candidates(ctx, model)

    assert len(result) == 1


@pytest.mark.unit
def test_a_bad_ratio_fails_at_construction(minimal_jobs_dict, tmp_path, monkeypatch):
    from alomancy.core.committee_uncertainty_workflow import (
        CommitteeUncertaintyWorkflow,
    )

    monkeypatch.chdir(tmp_path)
    minimal_jobs_dict["high_accuracy_evaluation"]["max_vdw_volume_ratio"] = 0.5
    minimal_jobs_dict.setdefault("general", {}).setdefault(
        "dataset_kwargs", {"test_ratio": 0.1, "target_config_types": ["IsolatedAtom"]}
    )
    with pytest.raises(ValueError, match="max_vdw_volume_ratio"):
        CommitteeUncertaintyWorkflow(jobs_dict=minimal_jobs_dict)


def _finished_outputs(n):
    """*n* DFT outputs where a restart looks for them, each with a distinct
    cell length so the reused ones can be told apart."""
    from ase.io import write

    from alomancy.remote_submission.submitters import ASE_OUTPUT_PREFIX

    for i in range(n):
        out = Path(
            "results",
            "test_loop",
            "high_accuracy_evaluation",
            "batch_0",
            f"{ASE_OUTPUT_PREFIX}_{i}",
        )
        out.mkdir(parents=True)
        done = _crystal()
        done.info["REF_energy"] = -1.0
        write(out / "high_accuracy_evaluation.xyz", done, format="extxyz")


def _evaluate_with_restart(structures):
    from alomancy.high_accuracy_evaluation import high_accuracy_calc_interface as hae

    submitted = []

    def fake_submit(*, input_atoms_list, **_kwargs):
        submitted.extend(input_atoms_list)

    with (
        patch.object(hae, "ase_remote_submitter", side_effect=fake_submit),
        patch.object(hae, "refresh_dft_labels", side_effect=lambda a, _p: a),
        contextlib.suppress(RuntimeError),  # the fake writes no new results
    ):
        hae.high_accuracy_evaluation(
            structures,
            {"evaluator": "qe"},
            base_name="test_loop",
            name="high_accuracy_evaluation",
            hpc={"hpc_name": "h", "pre_cmds": [], "partitions": ["p"]},
            max_time="1H",
        )
    return submitted


def _ten_with_two_blown_up():
    """10 structures, the first two blown up, the rest tagged by index."""
    structures = [_blown_up(), _blown_up()]
    for i in range(8):
        a = _crystal()
        a.info["index"] = i
        structures.append(a)
    return structures


@pytest.mark.unit
def test_dft_restart_reuses_everything_when_all_filtered_structures_finished(
    tmp_path, monkeypatch
):
    """Two of ten are blown up, so eight were submitted; with all eight
    outputs present, a restart submits nothing."""
    monkeypatch.chdir(tmp_path)
    _finished_outputs(8)
    assert _evaluate_with_restart(_ten_with_two_blown_up()) == []


@pytest.mark.unit
def test_dft_restart_submits_exactly_the_unfinished_structures(tmp_path, monkeypatch):
    """With four of the eight filtered structures finished, the restart
    submits the last four, not a slice shifted by the excluded two."""
    monkeypatch.chdir(tmp_path)
    _finished_outputs(4)
    submitted = _evaluate_with_restart(_ten_with_two_blown_up())
    assert [a.info["index"] for a in submitted] == [4, 5, 6, 7]


@pytest.mark.unit
def test_bad_exempt_config_types_fail_at_construction(
    minimal_jobs_dict, tmp_path, monkeypatch
):
    from alomancy.core.committee_uncertainty_workflow import (
        CommitteeUncertaintyWorkflow,
    )

    monkeypatch.chdir(tmp_path)
    minimal_jobs_dict["high_accuracy_evaluation"]["vdw_exempt_config_types"] = (
        "molecule_in_box"
    )
    minimal_jobs_dict.setdefault("general", {}).setdefault(
        "dataset_kwargs", {"test_ratio": 0.1, "target_config_types": ["IsolatedAtom"]}
    )
    with pytest.raises(ValueError, match="vdw_exempt_config_types"):
        CommitteeUncertaintyWorkflow(jobs_dict=minimal_jobs_dict)
