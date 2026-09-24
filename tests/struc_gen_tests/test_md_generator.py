"""Tests for the modular AL architecture's structure_generator entry
points added to structure_generation/md/md_wfl.py (generate, output_paths,
read_existing_result, _run_md_via_trainer) and the new calculator=
parameter on run_md itself.

run_md's own dynamics-loop behavior is still covered by
TestMolecularDynamics in test_structure_generation.py (calculator=None,
the default, preserves that behavior exactly, unchanged) -- these tests
cover only the new pieces.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from alomancy.structure_generation.md.md_wfl import (
    _run_md_via_trainer,
    generate,
    output_paths,
    read_existing_result,
    run_md,
)

_MODULE = "alomancy.structure_generation.md.md_wfl"


def _seed_atoms(n=5, config_type="init_amorphous"):
    out = []
    for _i in range(n):
        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
        a.info["config_type"] = config_type
        out.append(a)
    return out


@pytest.mark.unit
class TestRunMdCalculatorParameter:
    @patch(f"{_MODULE}.MACECalculator")
    @patch(f"{_MODULE}.Langevin")
    def test_default_none_still_builds_mace_calculator(
        self, mock_langevin_cls, mock_mace_calc_cls, tmp_path
    ):
        mock_mace_calc_cls.return_value = MagicMock()
        mock_langevin_cls.side_effect = RuntimeError("stop")

        initial_structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        initial_structure.info["job_id"] = 0
        with pytest.raises(RuntimeError, match="stop"):
            run_md(
                structure_generation_job_dict={
                    "name": "t",
                    "desired_number_of_structures": 1,
                },
                initial_structure=initial_structure,
                total_md_runs=1,
                out_dir=str(tmp_path),
                model_path=["model.pt"],
                steps=10,
            )
        mock_mace_calc_cls.assert_called_once_with(
            model_paths=["model.pt"], device="cuda", default_dtype="float64"
        )

    @patch(f"{_MODULE}.MACECalculator")
    @patch(f"{_MODULE}.Langevin")
    def test_explicit_calculator_bypasses_mace_calculator_construction(
        self, mock_langevin_cls, mock_mace_calc_cls, tmp_path
    ):
        mock_langevin_cls.side_effect = RuntimeError("stop")
        explicit_calc = MagicMock(name="explicit_calc")

        initial_structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        initial_structure.info["job_id"] = 0
        with pytest.raises(RuntimeError, match="stop"):
            run_md(
                structure_generation_job_dict={
                    "name": "t",
                    "desired_number_of_structures": 1,
                },
                initial_structure=initial_structure,
                total_md_runs=1,
                out_dir=str(tmp_path),
                model_path=["model.pt"],
                steps=10,
                calculator=explicit_calc,
            )
        mock_mace_calc_cls.assert_not_called()
        # atoms passed to Langevin carry the explicit calculator.
        passed_atoms = mock_langevin_cls.call_args.kwargs["atoms"]
        assert passed_atoms.calc is explicit_calc


@pytest.mark.unit
class TestRunMdViaTrainer:
    def test_resolves_trainer_and_delegates_to_run_md(self, tmp_path):
        fake_calc = MagicMock(name="fake_calc")
        fake_get_calculator = MagicMock(return_value=fake_calc)

        initial_structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        initial_structure.info["job_id"] = 0

        with (
            patch(f"{_MODULE}.resolve") as mock_resolve,
            patch(f"{_MODULE}.run_md") as mock_run_md,
        ):
            mock_resolve.return_value = MagicMock(get_calculator=fake_get_calculator)
            _run_md_via_trainer(
                structure_generation_job_dict={"name": "t"},
                initial_structure=initial_structure,
                total_md_runs=1,
                out_dir=str(tmp_path),
                model_path="model.pt",
                trainer="mace",
                trainer_config={"device": "cpu"},
                steps=10,
            )

        mock_resolve.assert_called_once_with("mlip_trainer", "mace")
        fake_get_calculator.assert_called_once_with("model.pt", {"device": "cpu"})
        assert mock_run_md.call_args.kwargs["calculator"] is fake_calc
        assert mock_run_md.call_args.kwargs["steps"] == 10


@pytest.mark.unit
class TestOutputPathsAndReadExistingResult:
    def test_output_paths_is_the_candidates_file(self):
        paths = output_paths({}, base_name="al_loop_0", name="md")
        assert paths == [
            Path("results/al_loop_0/structure_generation/md_generated_candidates.xyz")
        ]

    def test_read_existing_result_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No cached"):
            read_existing_result({}, base_name="al_loop_0", name="md")

    def test_read_existing_result_reads_candidates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = Path(
            "results/al_loop_0/structure_generation/md_generated_candidates.xyz"
        )
        path.parent.mkdir(parents=True)
        write(str(path), _seed_atoms(2), format="extxyz")
        result = read_existing_result({}, base_name="al_loop_0", name="md")
        assert len(result) == 2


@pytest.mark.unit
class TestGenerate:
    def test_selects_seeds_and_fans_out_one_job_per_seed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        seeds = _seed_atoms(10)

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            # Simulate each remote job writing its own trajectory file.
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(1), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info") as mock_get_remote_info,
        ):
            result = generate(
                seed_atoms=seeds,
                model_path="model.pt",
                config={
                    "md_kwargs": {
                        "structure_selection_kwargs": {
                            "max_number_of_concurrent_jobs": 3
                        }
                    }
                },
                base_name="al_loop_0",
                name="md",
                hpc={"hpc_name": "test"},
                max_time="1H",
            )

        assert mock_submit_n.call_count == 1
        job_configs = mock_submit_n.call_args.args[1]
        assert len(job_configs) == 3
        assert len(result) == 3
        assert mock_get_remote_info.call_args.args[0] == {
            "hpc": {"hpc_name": "test"},
            "name": "md",
            "max_time": "1H",
        }
        # run_md (unchanged, shared with the old production path) reads its
        # own "name" out of structure_generation_job_dict -- config itself
        # carries no "name" key (hardcoded by the skeleton, not user
        # config), so generate() must merge it in.
        assert (
            job_configs[0]["function_kwargs"]["structure_generation_job_dict"]["name"]
            == "md"
        )

    def test_md_kwargs_forwarded_to_run_md_excluding_selection_kwargs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(1), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            generate(
                seed_atoms=_seed_atoms(3),
                model_path="model.pt",
                config={
                    "md_kwargs": {
                        "steps": 2000,
                        "temperature": 1000,
                        "structure_selection_kwargs": {
                            "max_number_of_concurrent_jobs": 3
                        },
                    }
                },
                base_name="al_loop_0",
                name="md",
                hpc={},
                max_time="1H",
            )

        job_configs = mock_submit_n.call_args.args[1]
        function_kwargs = job_configs[0]["function_kwargs"]
        assert function_kwargs["steps"] == 2000
        assert function_kwargs["temperature"] == 1000
        # structure_selection_kwargs is consumed by select_diverse_seeds
        # inside generate() itself -- it must not also be forwarded on to
        # run_md, which has no such parameter.
        assert "structure_selection_kwargs" not in function_kwargs

    def test_reuses_cached_candidates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        candidates_path = Path(
            "results/al_loop_0/structure_generation/md_generated_candidates.xyz"
        )
        candidates_path.parent.mkdir(parents=True)
        write(str(candidates_path), _seed_atoms(4), format="extxyz")

        with patch(f"{_MODULE}.submit_n") as mock_submit_n:
            result = generate(
                seed_atoms=_seed_atoms(10),
                model_path="model.pt",
                config={},
                base_name="al_loop_0",
                name="md",
                hpc={},
                max_time="1H",
            )

        mock_submit_n.assert_not_called()
        assert len(result) == 4

    def test_partial_existing_only_submits_remaining_seeds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        md_dir = Path("results/al_loop_0/structure_generation")
        existing = md_dir / "md_output_0"
        existing.mkdir(parents=True)
        write(str(existing / "md.xyz"), _seed_atoms(1), format="extxyz")

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(1), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            generate(
                seed_atoms=_seed_atoms(10),
                model_path="model.pt",
                config={
                    "md_kwargs": {
                        "structure_selection_kwargs": {
                            "max_number_of_concurrent_jobs": 3
                        }
                    }
                },
                base_name="al_loop_0",
                name="md",
                hpc={},
                max_time="1H",
            )

        job_configs = mock_submit_n.call_args.args[1]
        assert len(job_configs) == 2  # 3 requested, 1 already existing
        # New jobs use directory indices continuing from the existing one.
        out_dirs = {jc["function_kwargs"]["out_dir"] for jc in job_configs}
        assert str(md_dir / "md_output_1") in out_dirs
        assert str(md_dir / "md_output_2") in out_dirs
