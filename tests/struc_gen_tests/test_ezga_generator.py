"""Tests for the modular AL architecture's structure_generator entry
points added to structure_generation/ezga/generate_structures.py
(generate, output_paths, read_existing_result). run_ezga itself is
unchanged and untested here (heavy external dependency, already not
end-to-end tested elsewhere in this suite -- see test_ezga_generation.py's
existing coverage of its config-building helpers)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.io import write

from alomancy.structure_generation.ezga.generate_structures import (
    generate,
    output_paths,
    read_existing_result,
)

_MODULE = "alomancy.structure_generation.ezga.generate_structures"


def _atoms(n=2):
    out = []
    for _ in range(n):
        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
        out.append(a)
    return out


@pytest.mark.unit
class TestOutputPathsAndReadExistingResult:
    def test_output_paths_is_ezga_candidates_file(self):
        paths = output_paths({}, base_name="al_loop_0", name="structure_generation")
        assert paths == [
            Path("results/al_loop_0/structure_generation/ezga/ezga_candidates.xyz")
        ]

    def test_read_existing_result_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No cached"):
            read_existing_result({}, base_name="al_loop_0", name="structure_generation")

    def test_read_existing_result_reads_candidates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = Path("results/al_loop_0/structure_generation/ezga/ezga_candidates.xyz")
        path.parent.mkdir(parents=True)
        write(str(path), _atoms(3), format="extxyz")
        result = read_existing_result(
            {}, base_name="al_loop_0", name="structure_generation"
        )
        assert len(result) == 3


@pytest.mark.unit
class TestGenerate:
    def test_calls_run_ezga_with_full_population_and_model_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        seeds = _atoms(5)

        with patch(f"{_MODULE}.run_ezga") as mock_run_ezga:
            mock_run_ezga.return_value = _atoms(2)
            result = generate(
                seed_atoms=seeds,
                model_path="model.pt",
                config={"run_ezga_kwargs": {"max_generations": 4}},
                base_name="al_loop_0",
                name="structure_generation",
                hpc={},
                max_time="1H",
            )

        mock_run_ezga.assert_called_once()
        call_kwargs = mock_run_ezga.call_args.kwargs
        assert call_kwargs["initial_structures"] is seeds
        assert call_kwargs["model_path"] == "model.pt"
        assert call_kwargs["max_generations"] == 4
        assert len(result) == 2

    def test_reuses_cached_candidates_without_calling_run_ezga(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        path = Path("results/al_loop_0/structure_generation/ezga/ezga_candidates.xyz")
        path.parent.mkdir(parents=True)
        write(str(path), _atoms(4), format="extxyz")

        with patch(f"{_MODULE}.run_ezga") as mock_run_ezga:
            result = generate(
                seed_atoms=_atoms(5),
                model_path="model.pt",
                config={},
                base_name="al_loop_0",
                name="structure_generation",
                hpc={},
                max_time="1H",
            )

        mock_run_ezga.assert_not_called()
        assert len(result) == 4
