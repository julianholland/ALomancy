"""Tests for the EZGA entry points and their manifest-backed cache.

The expensive EZGA engine is mocked. The mock writes the candidate file,
while the real generate() writes and checks the cache manifest.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.io import read, write

from alomancy.mlip.base import CalculatorSpec
from alomancy.structure_generation.ezga.generate_structures import (
    generate,
    output_paths,
    read_existing_result,
)

_MODULE = "alomancy.structure_generation.ezga.generate_structures"
_SPEC = CalculatorSpec(trainer="sevennet", trainer_config={}, model_path="model.pth")
_OUTPUT_DIR = Path("results/al_loop_0/structure_generation/ezga")
_CANDIDATES_PATH = _OUTPUT_DIR / "ezga_new_candidates.xyz"
_MANIFEST_PATH = _OUTPUT_DIR / "ezga_cache_manifest.json"


@pytest.fixture(autouse=True)
def _isolated_workdir(tmp_path, monkeypatch):
    """Keep every test's generated files outside the repository and real runs."""
    monkeypatch.chdir(tmp_path)


def _atoms(n=2, *, start=1.0, symbols="H2"):
    """Make distinct dimers; use a different start for seeds and candidates."""
    return [
        Atoms(
            symbols,
            positions=[[0, 0, 0], [0, 0, start + 0.05 * i]],
            cell=[10, 10, 10],
            pbc=True,
        )
        for i in range(n)
    ]


def _generate_kwargs():
    return {
        "seed_atoms": _atoms(5),
        "calculator_spec": _SPEC,
        "config": {},
        "elements": ["H"],
        "base_name": "al_loop_0",
        "name": "structure_generation",
        "hpc": {},
        "max_time": "1H",
    }


def _generate_with_fake_ezga(kwargs, candidates=None):
    """Create the cache through the public entry point, not private helpers."""
    generated = _atoms(4, start=2.0) if candidates is None else candidates

    def fake_run_ezga(**run_kwargs):
        output_dir = Path(run_kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        write(
            output_dir / "ezga_new_candidates.xyz",
            generated,
            format="extxyz",
        )
        return generated

    with patch(f"{_MODULE}.run_ezga", side_effect=fake_run_ezga) as mock_run:
        result = generate(**kwargs)

    return result, mock_run


@pytest.mark.unit
class TestOutputPathsAndReadExistingResult:
    def test_output_paths_require_candidates_and_manifest(self):
        paths = output_paths({}, base_name="al_loop_0", name="structure_generation")

        assert paths == [_CANDIDATES_PATH, _MANIFEST_PATH]

    def test_read_existing_result_raises_when_missing(self):
        with pytest.raises(ValueError, match="cached EZGA result"):
            read_existing_result({}, base_name="al_loop_0", name="structure_generation")

    def test_read_existing_result_reads_complete_cache(self):
        expected, _ = _generate_with_fake_ezga(_generate_kwargs())

        result = read_existing_result(
            {}, base_name="al_loop_0", name="structure_generation"
        )

        assert len(result) == len(expected)
        for actual, original in zip(result, expected, strict=True):
            assert actual == original


@pytest.mark.unit
class TestGenerate:
    def test_calls_run_ezga_with_full_population_and_calculator_spec(self):
        kwargs = _generate_kwargs()
        kwargs["config"] = {"ezga_kwargs": {"max_generations": 4}}
        expected = _atoms(2, start=2.0)

        result, mock_run = _generate_with_fake_ezga(kwargs, candidates=expected)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["initial_structures"] is kwargs["seed_atoms"]
        assert call_kwargs["calculator_spec"] is _SPEC
        assert call_kwargs["elements"] == ["H"]
        assert call_kwargs["max_generations"] == 4
        assert call_kwargs["output_dir"] == _OUTPUT_DIR
        assert result is expected
        assert _CANDIDATES_PATH.is_file()
        assert _MANIFEST_PATH.is_file()

    def test_reuses_compatible_cache_without_calling_run_ezga(self):
        kwargs = _generate_kwargs()
        expected, _ = _generate_with_fake_ezga(kwargs)

        with patch(f"{_MODULE}.run_ezga") as mock_run:
            result = generate(**kwargs)

        mock_run.assert_not_called()
        assert len(result) == len(expected)
        for actual, original in zip(result, expected, strict=True):
            assert actual == original

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("elements", ["He"], id="elements"),
            pytest.param("seed_atoms", _atoms(5, start=1.5), id="seeds"),
            pytest.param(
                "calculator_spec",
                CalculatorSpec(
                    trainer="sevennet", trainer_config={}, model_path="other.pth"
                ),
                id="calculator",
            ),
            pytest.param(
                "config",
                {"ezga_kwargs": {"population_size": 3}},
                id="population-size",
            ),
            pytest.param("config", {"ezga_kwargs": {"min_atoms": 3}}, id="min-atoms"),
            pytest.param("config", {"ezga_kwargs": {"max_atoms": 40}}, id="max-atoms"),
        ],
    )
    def test_rejects_cache_when_runtime_inputs_change(self, field, value):
        kwargs = _generate_kwargs()
        _generate_with_fake_ezga(kwargs)
        kwargs[field] = value

        with (
            patch(f"{_MODULE}.run_ezga") as mock_run,
            pytest.raises(ValueError, match="incompatible"),
        ):
            generate(**kwargs)

        mock_run.assert_not_called()

    @pytest.mark.parametrize(
        "missing_path", [_CANDIDATES_PATH, _MANIFEST_PATH], ids=["xyz", "manifest"]
    )
    def test_rejects_incomplete_cache(self, missing_path):
        kwargs = _generate_kwargs()
        _generate_with_fake_ezga(kwargs)
        missing_path.unlink()

        with (
            patch(f"{_MODULE}.run_ezga") as mock_run,
            pytest.raises(ValueError, match="Incomplete EZGA cache"),
        ):
            generate(**kwargs)

        mock_run.assert_not_called()

    def test_rejects_elements_in_ezga_kwargs_even_with_cache(self):
        kwargs = _generate_kwargs()
        _generate_with_fake_ezga(kwargs)
        kwargs["config"] = {"ezga_kwargs": {"elements": ["H"]}}

        with (
            patch(f"{_MODULE}.run_ezga") as mock_run,
            pytest.raises(ValueError, match=r"Set elements in general\.elements"),
        ):
            generate(**kwargs)

        mock_run.assert_not_called()

    def test_rejects_unreadable_manifest(self):
        kwargs = _generate_kwargs()
        _generate_with_fake_ezga(kwargs)
        _MANIFEST_PATH.write_text("not valid JSON")

        with (
            patch(f"{_MODULE}.run_ezga") as mock_run,
            pytest.raises(ValueError, match="Cannot read EZGA cache manifest"),
        ):
            generate(**kwargs)

        mock_run.assert_not_called()

    @pytest.mark.parametrize(
        "change",
        ["geometry", "remove-frame", "add-frame", "byte"],
    )
    def test_rejects_modified_candidate_file(self, change):
        kwargs = _generate_kwargs()
        _generate_with_fake_ezga(kwargs)

        if change == "byte":
            contents = _CANDIDATES_PATH.read_bytes()
            _CANDIDATES_PATH.write_bytes(contents + b"\n")
        else:
            cached = read(_CANDIDATES_PATH, index=":", format="extxyz")
            candidates = [cached] if isinstance(cached, Atoms) else list(cached)

            if change == "geometry":
                candidates[0].positions[0, 0] += 0.1
            elif change == "remove-frame":
                candidates = candidates[:-1]
            else:
                candidates.append(_atoms(1, start=3.0)[0])

            write(_CANDIDATES_PATH, candidates, format="extxyz")

        with (
            patch(f"{_MODULE}.run_ezga") as mock_run,
            pytest.raises(ValueError, match="integrity check failed"),
        ):
            generate(**kwargs)

        mock_run.assert_not_called()

    def test_rejects_candidate_count_mismatch(self):
        kwargs = _generate_kwargs()
        _generate_with_fake_ezga(kwargs)

        manifest = json.loads(_MANIFEST_PATH.read_text())
        manifest["candidate_file"]["structure_count"] += 1
        _MANIFEST_PATH.write_text(json.dumps(manifest))

        with (
            patch(f"{_MODULE}.run_ezga") as mock_run,
            pytest.raises(ValueError, match="candidate count does not match"),
        ):
            generate(**kwargs)

        mock_run.assert_not_called()
