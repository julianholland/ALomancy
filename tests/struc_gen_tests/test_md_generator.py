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
from ase.io import read, write

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
                    "desired_num_of_structures": 1,
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
                    "desired_num_of_structures": 1,
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
            patch("alomancy.mlip.base.get_trainer") as mock_get_trainer,
            patch(f"{_MODULE}.run_md") as mock_run_md,
        ):
            mock_get_trainer.return_value = MagicMock(
                get_calculator=fake_get_calculator
            )
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

        mock_get_trainer.assert_called_once_with("mace", {"device": "cpu"})
        fake_get_calculator.assert_called_once_with("model.pt")
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
    def test_num_of_md_starts_defaults_to_ten(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        seeds = _seed_atoms(15)

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(2), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            generate(
                seed_atoms=seeds,
                model_path="model.pt",
                config={},
                base_name="al_loop_0",
                name="md",
                hpc={},
                max_time="1H",
            )

        job_configs = mock_submit_n.call_args.args[1]
        assert len(job_configs) == 10

    def test_selects_seeds_and_fans_out_one_job_per_seed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        seeds = _seed_atoms(10)

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            # Simulate each remote job writing its own trajectory file.
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(2), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info") as mock_get_remote_info,
        ):
            result = generate(
                seed_atoms=seeds,
                model_path="model.pt",
                config={
                    "md_kwargs": {"structure_selection_kwargs": {"num_of_md_starts": 3}}
                },
                base_name="al_loop_0",
                name="md",
                hpc={"hpc_name": "test"},
                max_time="1H",
            )

        assert mock_submit_n.call_count == 1
        job_configs = mock_submit_n.call_args.args[1]
        assert len(job_configs) == 3
        assert len(result) == 6  # 3 runs x 2 frames
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
                write(str(out_dir / "md.xyz"), _seed_atoms(2), format="extxyz")
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
                        "structure_selection_kwargs": {"num_of_md_starts": 3},
                        "trainer": "mace",
                        "trainer_config": {"device": "cpu"},
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
        assert function_kwargs["trainer"] == "mace"
        assert function_kwargs["trainer_config"] == {"device": "cpu"}
        # structure_selection_kwargs/trainer/trainer_config are consumed
        # inside generate() itself (the first for select_diverse_seeds, the
        # rest passed as their own explicit function_kwargs above) -- they
        # must not also be forwarded a second time via **md_kwargs, which
        # run_md has no matching parameters for.
        assert "structure_selection_kwargs" not in function_kwargs

    def test_trainer_defaults_to_mace_when_absent_from_md_kwargs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(2), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            generate(
                seed_atoms=_seed_atoms(3),
                model_path="model.pt",
                config={},
                base_name="al_loop_0",
                name="md",
                hpc={},
                max_time="1H",
            )

        job_configs = mock_submit_n.call_args.args[1]
        function_kwargs = job_configs[0]["function_kwargs"]
        assert function_kwargs["trainer"] == "mace"
        assert function_kwargs["trainer_config"] == {}
        # ALomancy's own defaults, not run_md's far-shorter built-in ones.
        assert function_kwargs["steps"] == 20000
        assert function_kwargs["temperature"] == 300
        assert function_kwargs["timestep_fs"] == 0.5

    def test_md_kwargs_defaults_overridden_by_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(2), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            generate(
                seed_atoms=_seed_atoms(3),
                model_path="model.pt",
                config={"md_kwargs": {"steps": 500, "temperature": 1200}},
                base_name="al_loop_0",
                name="md",
                hpc={},
                max_time="1H",
            )

        function_kwargs = mock_submit_n.call_args.args[1][0]["function_kwargs"]
        assert function_kwargs["steps"] == 500
        assert function_kwargs["temperature"] == 1200
        # Untouched default still applies for whatever wasn't overridden.
        assert function_kwargs["timestep_fs"] == 0.5

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
        write(str(existing / "md.xyz"), _seed_atoms(2), format="extxyz")

        def fake_submit_n(function, job_configs, remote_info, **kwargs):
            for jc in job_configs:
                out_dir = Path(jc["function_kwargs"]["out_dir"])
                out_dir.mkdir(parents=True)
                write(str(out_dir / "md.xyz"), _seed_atoms(2), format="extxyz")
            return [None] * len(job_configs)

        with (
            patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n) as mock_submit_n,
            patch(f"{_MODULE}.get_remote_info"),
        ):
            generate(
                seed_atoms=_seed_atoms(10),
                model_path="model.pt",
                config={
                    "md_kwargs": {"structure_selection_kwargs": {"num_of_md_starts": 3}}
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


def _distinct_seeds(n: int) -> list[Atoms]:
    """Eligible structures with different geometries (so "not used yet"
    can tell them apart)."""
    out = []
    for i in range(n):
        a = Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.7 + 0.05 * i]], cell=[10] * 3, pbc=True
        )
        a.info["config_type"] = "init_amorphous"
        a.info["source"] = i
        out.append(a)
    return out


def _md_generate(tmp_path, outcome, n_seeds=3, eligible=None):
    """Run generate() with a fake submit_n. `outcome(run_index, seed_atoms)`
    returns the number of frames that run writes (0 = no output at all).
    Returns (result, submitted run indices per submit_n call)."""
    calls: list[list[int]] = []

    def fake_submit_n(function, job_configs, remote_info, **kwargs):
        indices = []
        for jc in job_configs:
            out_dir = Path(jc["function_kwargs"]["out_dir"])
            index = int(out_dir.name.rsplit("_", 1)[1])
            indices.append(index)
            frames = outcome(index, jc["function_kwargs"]["initial_structure"])
            out_dir.mkdir(parents=True, exist_ok=True)
            if frames:
                write(str(out_dir / "md.xyz"), _seed_atoms(frames), format="extxyz")
        calls.append(indices)
        return [None] * len(job_configs)

    with (
        patch(f"{_MODULE}.submit_n", side_effect=fake_submit_n),
        patch(f"{_MODULE}.get_remote_info"),
    ):
        result = generate(
            seed_atoms=eligible if eligible is not None else _distinct_seeds(10),
            model_path="model.pt",
            config={
                "md_kwargs": {
                    "structure_selection_kwargs": {"num_of_md_starts": n_seeds}
                }
            },
            base_name="al_loop_0",
            name="md",
            hpc={},
            max_time="1H",
        )
    return result, calls


_MD_DIR = Path("results/al_loop_0/structure_generation")


@pytest.mark.unit
class TestGenerateFailedRuns:
    def test_partial_runs_kept_and_not_replaced(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result, calls = _md_generate(tmp_path, lambda i, a: 3 if i == 0 else 2)

        assert calls == [[0, 1, 2]]  # nothing replaced
        assert len(result) == 3 + 2 + 2

    def test_runs_that_never_stepped_are_replaced_once_with_unused_seeds(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        # run 1 records only its starting structure; run 2 writes nothing.
        result, calls = _md_generate(
            tmp_path, lambda i, a: {0: 2, 1: 1, 2: 0}.get(i, 2)
        )

        assert calls == [[0, 1, 2], [3, 4]]
        seeds = read(_MD_DIR / "md_seeds.xyz", ":")
        assert [a.info.get("replaces") for a in seeds[3:]] == [1, 2]
        original_sources = {a.info["source"] for a in seeds[:3]}
        assert not original_sources & {a.info["source"] for a in seeds[3:]}
        assert len({a.info["md_seed"] for a in seeds}) == 5
        assert len(result) == 2 + 2 + 2  # runs 0, 3 and 4

    def test_replacement_that_also_fails_is_not_replaced_again(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.chdir(tmp_path)
        result, calls = _md_generate(tmp_path, lambda i, a: 0 if i in (1, 3) else 2)

        assert calls == [[0, 1, 2], [3]]  # one replacement round only
        assert len(result) == 2 + 2  # runs 0 and 2

    def test_all_runs_failing_raises_and_caches_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="completed a single MD step"):
            _md_generate(tmp_path, lambda i, a: 1)

        assert not (_MD_DIR / "md_generated_candidates.xyz").exists()

    def test_restart_reuses_saved_seeds_and_submits_only_missing_runs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        # A previous attempt chose seeds and finished run 0 before stopping.
        seeds = _distinct_seeds(3)
        for i, a in enumerate(seeds):
            a.info["md_seed"] = 803 + i
        _MD_DIR.mkdir(parents=True)
        write(_MD_DIR / "md_seeds.xyz", seeds, format="extxyz")
        (_MD_DIR / "md_output_0").mkdir()
        write(_MD_DIR / "md_output_0/md.xyz", _seed_atoms(2), format="extxyz")
        submitted_sources = []

        def outcome(i, seed_atoms):
            submitted_sources.append(seed_atoms.info["source"])
            return 2

        _, calls = _md_generate(tmp_path, outcome)

        assert calls == [[1, 2]]
        assert submitted_sources == [1, 2]  # the saved seeds, not new ones
