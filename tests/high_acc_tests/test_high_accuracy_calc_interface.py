"""Tests for high_accuracy_evaluation/high_accuracy_calc_interface.py --
the modular AL architecture's DFT-evaluator orchestrator entry point.

Originally extracted largely as-is from the now-removed
standard_active_learning.py's high_accuracy_evaluation method -- these
tests mirror that coverage (previously TestHighAccuracyEvaluationCoverage
in the now-deleted test_standard_active_learning.py) against this module's
free-function interface.
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
    initial_frames_path,
    output_paths,
    read_existing_result,
    read_initial_frames,
)
from alomancy.remote_submission.submitters import ASE_OUTPUT_PREFIX

_MODULE = "alomancy.high_accuracy_evaluation.high_accuracy_calc_interface"


def _config():
    # Test atoms sit in roomy boxes; the blown-up-cell check is off here.
    return {"evaluator": "qe", "max_vdw_volume_ratio": None}


def _atoms(symbol="H"):
    a = Atoms(symbol, positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    a.info["REF_energy"] = -1.0
    a.calc = SinglePointCalculator(a, energy=-1.0, forces=np.zeros((len(a), 3)))
    return a


def _fake_submit(*, base_name, input_atoms_list, batch, **_kwargs):
    """Stand-in for ase_remote_submitter that writes one completed result
    per submitted structure, where the real submitter's jobs would."""
    for i, atoms in enumerate(input_atoms_list):
        d = Path(
            f"results/{base_name}/high_accuracy_evaluation/batch_{batch}",
            f"{ASE_OUTPUT_PREFIX}_{i}",
        )
        d.mkdir(parents=True, exist_ok=True)
        ase_write(str(d / "high_accuracy_evaluation.xyz"), atoms, format="extxyz")


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
    @pytest.mark.unit
    def test_output_paths_is_the_consolidated_results_file(self, tmp_path):
        paths = output_paths({}, base_name="test_loop", name="high_accuracy_evaluation")
        assert paths == [Path("results/test_loop/high_accuracy_eval_results.xyz")]

    @pytest.mark.unit
    def test_read_existing_result_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No cached"):
            read_existing_result(
                {}, base_name="test_loop", name="high_accuracy_evaluation"
            )

    @pytest.mark.unit
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
    @pytest.mark.unit
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

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            result = _call([_atoms(), _atoms()], tmp_path)

        mock_sub.assert_not_called()
        assert len(result) == 2

    @pytest.mark.unit
    def test_qe_kwargs_translated_to_legacy_qe_input_kwargs(
        self, tmp_path, monkeypatch
    ):
        """run_sp_qe/run_go_qe (unchanged, shared with the old production
        path) still read "qe_input_kwargs" directly -- new-style config
        uses the standardized "qe_kwargs" name, translated here."""
        monkeypatch.chdir(tmp_path)
        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call(
                [_atoms()],
                tmp_path,
                config={
                    "evaluator": "qe",
                    "max_vdw_volume_ratio": None,
                    "qe_kwargs": {"system": {"input_dft": "pbe"}},
                },
            )

        job_dict = mock_sub.call_args.kwargs["function_kwargs"][
            "high_accuracy_eval_job_dict"
        ]
        assert job_dict["qe_input_kwargs"] == {"system": {"input_dft": "pbe"}}
        assert "qe_kwargs" not in job_dict

    @pytest.mark.unit
    def test_vasp_kwargs_translated_to_legacy_vasp_input_kwargs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call(
                [_atoms()],
                tmp_path,
                config={
                    "evaluator": "vasp",
                    "vasp_kwargs": {"encut": 520},
                    "max_vdw_volume_ratio": None,
                },
            )

        job_dict = mock_sub.call_args.kwargs["function_kwargs"][
            "high_accuracy_eval_job_dict"
        ]
        assert job_dict["vasp_input_kwargs"] == {"encut": 520}
        assert "vasp_kwargs" not in job_dict

    @pytest.mark.unit
    def test_partial_existing_trims_structures_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = Path(
            "results/test_loop/high_accuracy_evaluation/batch_0",
            f"{ASE_OUTPUT_PREFIX}_0",
        )
        d.mkdir(parents=True)
        ase_write(str(d / "high_accuracy_evaluation.xyz"), _atoms(), format="extxyz")

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call([_atoms() for _ in range(3)], tmp_path)

        mock_sub.assert_called_once()
        submitted = mock_sub.call_args.kwargs["input_atoms_list"]
        assert len(submitted) == 2

    @pytest.mark.unit
    def test_go_sp_share_one_submission(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        go_atom = _atoms()
        go_atom.info["needs_relaxation"] = True
        sp_atom = _atoms("O")

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
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

    @pytest.mark.unit
    def test_sp_only_batch_no_go(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sp_atoms = [_atoms() for _ in range(2)]

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call(sp_atoms, tmp_path, allow_relaxation=True)

        assert mock_sub.call_count == 1
        per_structure_function = mock_sub.call_args.kwargs["per_structure_function"]
        assert all("sp" in fn.__name__ for fn in per_structure_function)

    @pytest.mark.unit
    def test_single_call_regardless_of_structure_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call([_atoms() for _ in range(20)], tmp_path)

        assert mock_sub.call_count == 1
        assert len(mock_sub.call_args.kwargs["input_atoms_list"]) == 20

    @pytest.mark.unit
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

        # The one newly submitted job produces no output (failed); the two
        # earlier results are still collected and the phase completes.
        with patch(f"{_MODULE}.ase_remote_submitter"):
            result = _call([_atoms() for _ in range(3)], tmp_path)

        assert len(result) == 2
        assert Path("results/test_loop/high_accuracy_eval.done").exists()

    @pytest.mark.unit
    def test_all_jobs_failing_raises_without_marking_done(self, tmp_path, monkeypatch):
        """Regression: every DFT job failing (e.g. QE never printing forces)
        used to mark the phase done with zero structures, permanently."""
        monkeypatch.chdir(tmp_path)
        with (
            patch(f"{_MODULE}.ase_remote_submitter"),
            pytest.raises(RuntimeError, match="All 2 high-accuracy"),
        ):
            _call([_atoms(), _atoms()], tmp_path)

        assert not Path("results/test_loop/high_accuracy_eval.done").exists()
        assert not Path("results/test_loop/high_accuracy_eval_results.xyz").exists()

    @pytest.mark.unit
    def test_phase_done_sentinel_skips_reentirely(self, tmp_path, monkeypatch):
        """Once high_accuracy_eval.done exists, the whole call short-circuits
        to reading the consolidated sentinel results file -- no submission,
        no batch-directory globbing."""
        monkeypatch.chdir(tmp_path)
        sentinel_results = Path("results/test_loop/high_accuracy_eval_results.xyz")
        sentinel_results.parent.mkdir(parents=True)
        ase_write(str(sentinel_results), [_atoms()], format="extxyz")
        Path("results/test_loop/high_accuracy_eval.done").write_text("done\n")

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            result = _call([_atoms(), _atoms()], tmp_path)

        mock_sub.assert_not_called()
        assert len(result) == 1

    @pytest.mark.unit
    def test_empty_selection_completes_and_restarts(self, tmp_path, monkeypatch):
        """Regression: a loop that sends nothing to DFT (an all-redundant
        selection) marks the phase done with an empty results file; a
        restart must read that back as no structures, not crash on it."""
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.ase_remote_submitter") as mock_sub:
            assert _call([], tmp_path) == []
            assert _call([], tmp_path) == []

        mock_sub.assert_not_called()
        assert Path("results/test_loop/high_accuracy_eval.done").exists()
        assert (
            read_existing_result(
                {}, base_name="test_loop", name="high_accuracy_evaluation"
            )
            == []
        )

    @pytest.mark.unit
    def test_go_max_time_overrides_max_time_when_any_go_present(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        go_atom = _atoms()
        go_atom.info["needs_relaxation"] = True

        with (
            patch(f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit),
            patch(f"{_MODULE}.get_remote_info") as mock_get_remote_info,
        ):
            _call(
                [go_atom],
                tmp_path,
                config={
                    "evaluator": "qe",
                    "max_go_time": "8H",
                    "max_vdw_volume_ratio": None,
                },
                allow_relaxation=True,
            )

        submitted_config = mock_get_remote_info.call_args.args[0]
        assert submitted_config["max_time"] == "8H"

    @pytest.mark.unit
    def test_bond_distance_filter_applied_before_submission(
        self, tmp_path, monkeypatch
    ):
        """A pathologically close pair (< 0.5 A) must never reach submission."""
        monkeypatch.chdir(tmp_path)
        bad = Atoms("H2", positions=[[0, 0, 0], [0.1, 0, 0]], cell=[5, 5, 5], pbc=True)
        bad.info["REF_energy"] = -1.0
        good = _atoms()

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call([bad, good], tmp_path)

        submitted = mock_sub.call_args.kwargs["input_atoms_list"]
        assert len(submitted) == 1


def _fake_go_submit(*, base_name, input_atoms_list, batch, **_kwargs):
    """_fake_submit, plus a GO starting frame for every even job."""
    _fake_submit(base_name=base_name, input_atoms_list=input_atoms_list, batch=batch)
    for i, atoms in enumerate(input_atoms_list):
        if i % 2:
            continue
        d = Path(
            f"results/{base_name}/high_accuracy_evaluation/batch_{batch}",
            f"{ASE_OUTPUT_PREFIX}_{i}",
        )
        initial = atoms.copy()
        initial.info["geometry_initial"] = True
        initial.calc = SinglePointCalculator(
            initial, energy=-2.0, forces=np.ones((len(initial), 3))
        )
        ase_write(
            str(d / "high_accuracy_evaluation_initial.xyz"), initial, format="extxyz"
        )


@pytest.mark.unit
class TestInitialFrames:
    @pytest.mark.unit
    def test_go_starting_frames_are_collected_beside_the_results(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.ase_remote_submitter", side_effect=_fake_go_submit):
            result = _call([_atoms() for _ in range(3)], tmp_path)

        assert len(result) == 3
        assert not any(a.info.get("geometry_initial") for a in result)
        frames = read_initial_frames("test_loop")
        assert len(frames) == 2
        assert all(a.info["geometry_initial"] for a in frames)
        assert all(a.info["REF_energy"] == -2.0 for a in frames)
        assert initial_frames_path("test_loop").exists()

    @pytest.mark.unit
    def test_no_starting_frames_without_go_moves(self, tmp_path, monkeypatch):
        """SP jobs, unmoved GO jobs and jobs from before this change
        write none: an empty collection, not an error."""
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit):
            _call([_atoms() for _ in range(2)], tmp_path)
        assert read_initial_frames("test_loop") == []
        assert read_initial_frames("never_evaluated") == []


# --- Restart: finished results matched to structures by digest ---


def _distinct(i: int) -> Atoms:
    a = _atoms()
    a.positions += [0.1 * i, 0.0, 0.0]
    return a


def _write_finished(structures: list[Atoms], batch: int = 0) -> None:
    """Results of an earlier attempt, as the workers write them: the input's
    atoms.info (with its digest) carried through to <name>.xyz."""
    from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
        DFT_INPUT_DIGEST_KEY,
        dft_input_digest,
    )

    for i, atoms in enumerate(structures):
        done = atoms.copy()
        done.calc = SinglePointCalculator(
            done, energy=-1.0, forces=np.zeros((len(done), 3))
        )
        done.info[DFT_INPUT_DIGEST_KEY] = dft_input_digest(atoms)
        d = Path(
            f"results/test_loop/high_accuracy_evaluation/batch_{batch}",
            f"{ASE_OUTPUT_PREFIX}_{i}",
        )
        d.mkdir(parents=True, exist_ok=True)
        ase_write(str(d / "high_accuracy_evaluation.xyz"), done, format="extxyz")


def _submitted_x(mock_sub) -> list[float]:
    return [
        round(float(a.positions[0, 0]), 3)
        for a in mock_sub.call_args.kwargs["input_atoms_list"]
    ]


@pytest.mark.unit
class TestRestartByDigest:
    @pytest.mark.unit
    def test_unfinished_middle_structures_are_submitted_not_the_last(
        self, tmp_path, monkeypatch
    ):
        """Jobs finish in any order: with 0 and 3 done, 1 and 2 are left.
        The old positional rule would have resubmitted 2 and 3."""
        monkeypatch.chdir(tmp_path)
        structures = [_distinct(i) for i in range(4)]
        _write_finished([structures[0], structures[3]])

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            result = _call(structures, tmp_path)

        assert _submitted_x(mock_sub) == [0.1, 0.2]
        assert len(result) == 4

    @pytest.mark.unit
    def test_reordered_list_submits_only_the_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        structures = [_distinct(i) for i in range(4)]
        _write_finished([structures[0], structures[2]])

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call(structures[::-1], tmp_path)

        assert _submitted_x(mock_sub) == [0.3, 0.1]

    @pytest.mark.unit
    def test_all_finished_submits_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        structures = [_distinct(i) for i in range(3)]
        _write_finished(structures)

        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            result = _call(structures, tmp_path)

        mock_sub.assert_not_called()
        assert len(result) == 3

    @pytest.mark.unit
    def test_results_for_structures_no_longer_listed_warn_and_are_kept(
        self, tmp_path, monkeypatch
    ):
        """Settings changed between restarts: the old result is kept, and the
        warning says this isn't supported."""
        import logging

        from alomancy.high_accuracy_evaluation import high_accuracy_calc_interface

        monkeypatch.chdir(tmp_path)
        _write_finished([_distinct(9)])
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        high_accuracy_calc_interface.logger.addHandler(handler)
        try:
            with patch(
                f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
            ) as mock_sub:
                result = _call([_distinct(0), _distinct(1)], tmp_path)
        finally:
            high_accuracy_calc_interface.logger.removeHandler(handler)

        assert _submitted_x(mock_sub) == [0.0, 0.1]
        changed = [
            r for r in records if getattr(r, "event", None) == "dft_inputs_changed"
        ]
        assert len(changed) == 1
        assert "isn't supported" in changed[0].getMessage()
        assert len(result) == 3

    @pytest.mark.unit
    def test_submitted_structures_carry_their_digest(self, tmp_path, monkeypatch):
        from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
            DFT_INPUT_DIGEST_KEY,
        )

        monkeypatch.chdir(tmp_path)
        with patch(
            f"{_MODULE}.ase_remote_submitter", side_effect=_fake_submit
        ) as mock_sub:
            _call([_distinct(0)], tmp_path)

        submitted = mock_sub.call_args.kwargs["input_atoms_list"]
        assert submitted[0].info[DFT_INPUT_DIGEST_KEY]


@pytest.mark.unit
def test_digest_survives_an_extxyz_round_trip(tmp_path):
    """A restart compares structures held in memory on the first attempt
    with the same structures read back from file."""
    from ase.io import read

    from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
        dft_input_digest,
    )

    rng = np.random.default_rng(0)
    for k in range(50):
        atoms = Atoms(
            "H20",
            positions=(rng.random((20, 3)) - 0.3) * 20,
            cell=rng.random((3, 3)) * 10 + np.eye(3) * 5,
            pbc=True,
        )
        path = tmp_path / f"{k}.xyz"
        ase_write(str(path), atoms, format="extxyz")
        assert dft_input_digest(read(path)) == dft_input_digest(atoms)
