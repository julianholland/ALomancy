"""Tests for high_accuracy_evaluation/high_accuracy_calc_interface.py --
the modular AL architecture's DFT-evaluator orchestrator entry point.

Extracted largely as-is from standard_active_learning.py's
high_accuracy_evaluation method (still covered by
TestHighAccuracyEvaluationCoverage in test_standard_active_learning.py --
these tests mirror that coverage against the new free-function interface).
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
    high_accuracy_evaluation,
    output_paths,
    read_existing_result,
)
from alomancy.remote_submission.submitters import ASE_OUTPUT_PREFIX

_MODULE = "alomancy.high_accuracy_evaluation.high_accuracy_calc_interface"


def _config():
    return {"evaluator": "qe"}


def _atoms(symbol="H"):
    a = Atoms(symbol, positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    a.info["REF_energy"] = -1.0
    a.calc = SinglePointCalculator(a, energy=-1.0, forces=np.zeros((len(a), 3)))
    return a


def _call(structures, tmp_path, **overrides):
    # name must equal "high_accuracy_evaluation" -- ase_remote_submitter/
    # this module both hardcode that literal as the results subdirectory
    # (see CLAUDE.md's note on this convention), matching production's
    # jobs_dict["high_accuracy_evaluation"]["name"] requirement.
    kwargs = {
        "structures": structures,
        "config": _config(),
        "base_name": "test_loop",
        "name": "high_accuracy_evaluation",
        "hpc": {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]},
        "max_time": "2H",
    }
    kwargs.update(overrides)
    return high_accuracy_evaluation(**kwargs)


@pytest.mark.unit
class TestOutputPathsAndReadExistingResult:
    def test_output_paths_is_the_consolidated_results_file(self, tmp_path):
        paths = output_paths({}, base_name="test_loop", name="high_accuracy_evaluation")
        assert paths == [Path("results/test_loop/high_accuracy_eval_results.xyz")]

    def test_read_existing_result_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No cached"):
            read_existing_result(
                {}, base_name="test_loop", name="high_accuracy_evaluation"
            )

    def test_read_existing_result_reads_the_sentinel(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sentinel = Path("results/test_loop/high_accuracy_eval_results.xyz")
        sentinel.parent.mkdir(parents=True)
        ase_write(str(sentinel), [_atoms(), _atoms()], format="extxyz")

        result = read_existing_result(
            {}, base_name="test_loop", name="high_accuracy_evaluation"
        )
        assert len(result) == 2


@pytest.mark.unit
class TestHighAccuracyEvaluation:
    def test_reuses_all_existing_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for i in range(2):
            d = Path(
                "results/test_loop/high_accuracy_evaluation/batch_0",
                f"{ASE_OUTPUT_PREFIX}_{i}",
            )
            d.mkdir(parents=True)
            ase_write(
                str(d / "high_accuracy_evaluation.xyz"), _atoms(), format="extxyz"
            )

        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            result = _call([_atoms(), _atoms()], tmp_path)

        mock_sub.assert_not_called()
        assert len(result) == 2

    def test_partial_existing_trims_structures_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = Path(
            "results/test_loop/high_accuracy_evaluation/batch_0",
            f"{ASE_OUTPUT_PREFIX}_0",
        )
        d.mkdir(parents=True)
        ase_write(str(d / "high_accuracy_evaluation.xyz"), _atoms(), format="extxyz")

        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            _call([_atoms() for _ in range(3)], tmp_path)

        mock_sub.assert_called_once()
        submitted = mock_sub.call_args.kwargs["input_atoms_list"]
        assert len(submitted) == 2

    def test_go_sp_share_one_submission(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        go_atom = _atoms()
        go_atom.info["needs_relaxation"] = True
        sp_atom = _atoms("O")

        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            _call([go_atom, sp_atom], tmp_path, allow_relaxation=True)

        assert mock_sub.call_count == 1
        call_kwargs = mock_sub.call_args.kwargs
        submitted_atoms = call_kwargs["input_atoms_list"]
        per_structure_function = call_kwargs["per_structure_function"]
        assert len(submitted_atoms) == 2
        assert submitted_atoms[0].info.get("needs_relaxation") is True
        assert "go" in per_structure_function[0].__name__
        assert submitted_atoms[1].info.get("needs_relaxation") is not True
        assert "sp" in per_structure_function[1].__name__

    def test_sp_only_batch_no_go(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sp_atoms = [_atoms() for _ in range(2)]

        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            _call(sp_atoms, tmp_path, allow_relaxation=True)

        assert mock_sub.call_count == 1
        per_structure_function = mock_sub.call_args.kwargs["per_structure_function"]
        assert all("sp" in fn.__name__ for fn in per_structure_function)

    def test_single_call_regardless_of_structure_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            _call([_atoms() for _ in range(20)], tmp_path)

        assert mock_sub.call_count == 1
        assert len(mock_sub.call_args.kwargs["input_atoms_list"]) == 20

    def test_result_collection_reads_completed_xyz_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for i in range(2):
            d = Path(
                "results/test_loop/high_accuracy_evaluation/batch_0",
                f"{ASE_OUTPUT_PREFIX}_{i}",
            )
            d.mkdir(parents=True)
            ase_write(
                str(d / "high_accuracy_evaluation.xyz"), _atoms(), format="extxyz"
            )

        with patch(f"{_MODULE}.ase_remote_submitter"):
            result = _call([_atoms() for _ in range(3)], tmp_path)

        assert len(result) == 2

    def test_phase_done_sentinel_skips_reentirely(self, tmp_path, monkeypatch):
        """Once high_accuracy_eval.done exists, the whole call short-circuits
        to reading the consolidated sentinel results file -- no submission,
        no batch-directory globbing."""
        monkeypatch.chdir(tmp_path)
        sentinel_results = Path("results/test_loop/high_accuracy_eval_results.xyz")
        sentinel_results.parent.mkdir(parents=True)
        ase_write(str(sentinel_results), [_atoms()], format="extxyz")
        Path("results/test_loop/high_accuracy_eval.done").write_text("done\n")

        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            result = _call([_atoms(), _atoms()], tmp_path)

        mock_sub.assert_not_called()
        assert len(result) == 1

    def test_go_max_time_overrides_max_time_when_any_go_present(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        go_atom = _atoms()
        go_atom.info["needs_relaxation"] = True

        with (
            patch(f"{_MODULE}.ase_remote_submitter"),
            patch(f"{_MODULE}.get_remote_info") as mock_get_remote_info,
        ):
            _call(
                [go_atom],
                tmp_path,
                config={"evaluator": "qe", "max_go_time": "8H"},
                allow_relaxation=True,
            )

        submitted_config = mock_get_remote_info.call_args.args[0]
        assert submitted_config["max_time"] == "8H"

    def test_bond_distance_filter_applied_before_submission(
        self, tmp_path, monkeypatch
    ):
        """A pathologically close pair (< 0.5 A) must never reach submission."""
        monkeypatch.chdir(tmp_path)
        bad = Atoms("H2", positions=[[0, 0, 0], [0.1, 0, 0]], cell=[5, 5, 5], pbc=True)
        bad.info["REF_energy"] = -1.0
        good = _atoms()

        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            _call([bad, good], tmp_path)

        submitted = mock_sub.call_args.kwargs["input_atoms_list"]
        assert len(submitted) == 1
