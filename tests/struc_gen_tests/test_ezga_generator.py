"""EZGA's structure_generator entry points: restart files, seed choice,
moving an unfinished search aside, and one real (tiny) EZGA search read
back from its database."""

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write

from alomancy.mlip.base import ALomancyTrainer, CalculatorSpec
from alomancy.structure_generation.ezga import generate_structures as ezga
from alomancy.structure_generation.ezga.generate_structures import (
    generate,
    output_paths,
    read_existing_result,
    validate_kwargs,
)

_ELEMENTS = ["Cu"]
_SG_DIR = Path("results/al_loop_0/structure_generation")


class _EMTTrainer(ALomancyTrainer):
    """Backend-free trainer: the "model" is ASE's EMT."""

    NAME = "emt_ezga"
    KWARGS_KEY = "emt_ezga_kwargs"

    def model_path(self, fit_dir):
        return Path(fit_dir) / "emt.model"

    def get_calculator(self, model_path, *, device=None):
        return EMT()

    def fit(self, *args, **kwargs):  # pragma: no cover -- never trained here
        raise NotImplementedError


@pytest.fixture
def emt_spec():
    from alomancy.registry import _REGISTRY, register

    register("mlip_trainer", "emt_ezga", __name__, trainer_class="_EMTTrainer")
    yield CalculatorSpec(trainer="emt_ezga", trainer_config={}, model_path="emt.model")
    _REGISTRY["mlip_trainer"].pop("emt_ezga", None)


def _cu_seeds(n):
    seeds = []
    for i in range(n):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.rattle(0.02, seed=i)
        seeds.append(atoms)
    return seeds


def _sg(to_generate=5, **ezga_kwargs):
    sg = {
        "generator": "ezga",
        "num_of_structures_to_generate": to_generate,
        "ezga_kwargs": {
            "max_generations": 2,
            "multiobjective": {"size": 4},
            "local": True,
            **ezga_kwargs,
        },
    }
    validate_kwargs(sg, elements=_ELEMENTS, seed=803)
    return sg


def _generate(seeds, spec, sg):
    return generate(
        seeds,
        spec,
        sg,
        base_name="al_loop_0",
        name="structure_generation",
        hpc={},
        max_time="1h",
        elements=_ELEMENTS,
        seed=803,
    )


@pytest.mark.unit
def test_output_paths_is_the_candidates_file():
    assert output_paths({}, base_name="al_loop_0", name="structure_generation") == [
        _SG_DIR / "ezga" / "ezga_candidates.xyz"
    ]


@pytest.mark.unit
def test_read_existing_result_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="No cached EZGA result"):
        read_existing_result({}, base_name="al_loop_0", name="structure_generation")


@pytest.mark.unit
def test_an_empty_candidates_file_reads_as_no_structures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _SG_DIR / "ezga" / "ezga_candidates.xyz"
    path.parent.mkdir(parents=True)
    path.write_text("")
    assert (
        read_existing_result({}, base_name="al_loop_0", name="structure_generation")
        == []
    )


@pytest.mark.unit
def test_cached_candidates_are_returned_without_a_search(
    tmp_path, monkeypatch, emt_spec
):
    monkeypatch.chdir(tmp_path)
    path = _SG_DIR / "ezga" / "ezga_candidates.xyz"
    path.parent.mkdir(parents=True)
    write(path, _cu_seeds(3), format="extxyz")
    monkeypatch.setattr(ezga, "run_ezga", lambda **_: pytest.fail("searched again"))
    assert len(_generate(_cu_seeds(2), emt_spec, _sg())) == 3


@pytest.mark.unit
def test_seeds_are_capped_and_never_padded(tmp_path, monkeypatch, emt_spec):
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run(*, seeds, output_dir, **_):
        calls["seeds"] = seeds
        raise RuntimeError("stop after seeding")

    monkeypatch.setattr(ezga, "run_ezga", fake_run)
    sg = _sg(structure_selection_kwargs={"num_of_seeds": 8})
    with pytest.raises(RuntimeError, match="stop after seeding"):
        _generate(_cu_seeds(3), emt_spec, sg)
    assert len(calls["seeds"]) == 3  # pool of 3 < num_of_seeds, no duplicates
    assert len(read(_SG_DIR / "ezga_seeds.xyz", ":")) == 3

    sg = _sg(structure_selection_kwargs={"num_of_seeds": 2})
    (_SG_DIR / "ezga_seeds.xyz").unlink()
    with pytest.raises(RuntimeError):
        _generate(_cu_seeds(5), emt_spec, sg)
    assert len(calls["seeds"]) == 2


@pytest.mark.unit
def test_no_seed_structures_raises(tmp_path, monkeypatch, emt_spec):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="no seed structures"):
        _generate([], emt_spec, _sg())


@pytest.mark.unit
def test_an_unfinished_search_is_moved_aside(tmp_path, monkeypatch, emt_spec):
    monkeypatch.chdir(tmp_path)
    stale = _SG_DIR / "ezga" / "db"
    stale.mkdir(parents=True)
    (stale / "marker").write_text("old")

    def fake_run(*, output_dir, **_):
        assert not Path(output_dir).exists()
        raise RuntimeError("stop")

    monkeypatch.setattr(ezga, "run_ezga", fake_run)
    with pytest.raises(RuntimeError, match="stop"):
        _generate(_cu_seeds(2), emt_spec, _sg())
    [moved] = list(_SG_DIR.glob("ezga.failed_*"))
    assert (moved / "db" / "marker").read_text() == "old"


@pytest.mark.integration
@pytest.mark.slow
def test_a_real_search_is_read_back_from_ezgas_database(
    tmp_path, monkeypatch, emt_spec
):
    """A real 2-generation, population-4 EZGA search with EMT: the
    candidates are the offspring in EZGA's database (no seeds), newest
    generations first, capped at num_of_structures_to_generate, with no
    DFT labels; a rerun returns the cache."""
    monkeypatch.chdir(tmp_path)
    seeds = _cu_seeds(2)
    # Short relaxations: fully relaxed, this tiny Cu cell falls back into the
    # same minima, which EZGA's duplicate filter drops.
    sg = _sg(to_generate=5, simulator={"calculator": {"steps_max": 5}})

    candidates = _generate(seeds, emt_spec, sg)

    assert (_SG_DIR / "ezga" / "db" / "db_index.sqlite").exists()
    assert len(candidates) == 5
    for atoms in candidates:
        assert "ezga_generation" in atoms.info
        assert not any(key.startswith("REF_") for key in atoms.info)
        assert "REF_forces" not in atoms.arrays
        assert atoms.info["config_type"] == "al_loop_0_structure_generation"
    all_generations = ezga.read_offspring(_SG_DIR / "ezga" / "db")
    newest = max(a.info["ezga_generation"] for a in all_generations)
    n_newest = sum(a.info["ezga_generation"] == newest for a in all_generations)
    assert sum(a.info["ezga_generation"] == newest for a in candidates) == min(
        n_newest, 5
    )

    cached = read(_SG_DIR / "ezga" / "ezga_candidates.xyz", ":")
    assert len(cached) == 5
    monkeypatch.setattr(ezga, "run_ezga", lambda **_: pytest.fail("searched again"))
    assert len(_generate(seeds, emt_spec, sg)) == 5
    assert isinstance(candidates[0], Atoms)


@pytest.mark.unit
def test_report_section_reads_the_ezga_summary():
    from alomancy.analysis.report.sections import ezga_section

    stats = {"events": {"data": {}}}
    assert ezga_section(stats, base_name="b", plots_dir=None, config={}) is None
    stats["events"]["data"]["ezga_summary"] = [
        {
            "seeds": 4,
            "generations": 3,
            "generated": 40,
            "candidates": 20,
            "oldest_generation_used": 1,
        }
    ]
    config = {"structure_generation": _sg()}
    section = ezga_section(stats, base_name="b", plots_dir=None, config=config)
    text = "\n".join(section.lines)
    assert "Seeds: 4, generations run: 3" in text
    assert "max_generations 2, population 4" in text
    assert "Structures generated: 40, taken as candidates: 20" in text
    assert "mutation_rattle" in text


@pytest.mark.integration
@pytest.mark.slow
def test_a_hise_search_is_read_from_the_last_stage_only(
    tmp_path, monkeypatch, emt_spec
):
    """Two HiSE stages, 1x1x1 then 2x1x1 with an MD burst (the EZGA wiki's
    pattern): the candidates are the last stage's own offspring, not the
    earlier stage's structures carried (tiled) into its database."""
    monkeypatch.chdir(tmp_path)
    sg = _sg(
        to_generate=4,
        hise={
            "supercells": [[1, 1, 1], [2, 1, 1]],
            "overrides": {
                "max_generations": [2, 1],
                "simulator.calculator": [{}, {"nvt_steps": 5, "T": [300, 600]}],
            },
        },
    )

    candidates = _generate(_cu_seeds(2), emt_spec, sg)

    ezga_dir = _SG_DIR / "ezga"
    first, last = sorted(p for p in ezga_dir.glob("supercell_*") if p.is_dir())
    assert last.name == "supercell_2_1_1"
    # The 4-atom seeds were tiled 2x1x1 (8 atoms) before the last stage
    # mutated them by at most an atom or two.
    assert min(len(a) for a in candidates) > 4
    last_stage = ezga.read_offspring(last)
    first_stage = ezga.read_offspring(first)
    assert len(candidates) == 4
    assert max(a.info["ezga_generation"] for a in last_stage) <= 1
    # The last stage's database also holds the first stage's structures,
    # with their own (older) generation numbers; none of them is read.
    from alomancy.database.global_database import read_storage

    assert len(read_storage(last)) > len(last_stage) + len(_cu_seeds(2)) - 1
    assert all(not any(k.startswith("REF_") for k in a.info) for a in candidates)
    assert first_stage
