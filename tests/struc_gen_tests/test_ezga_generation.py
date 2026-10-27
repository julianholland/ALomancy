"""EZGA settings: the defaults, the merge with the user's EZGA config, the
construction-time check (validate_kwargs) and the helpers EZGA calls."""

import logging

import numpy as np
import pytest
from ase import Atoms

from alomancy.structure_generation.ezga.generate_structures import (
    _bounded,
    _merge,
    _resolve,
    most_recent,
    objective_energy_per_atom,
    validate_kwargs,
)

_ELEMENTS = ["Cu", "O"]


class _Dataset:
    def get_all_energies(self):
        return np.array([-10.0, -30.0])

    def get_all_compositions(self, return_species=False):
        compositions = np.array([[2], [5]])
        if return_species:
            return compositions, ["Cu"]
        return compositions


def _sg(ezga_kwargs=None, to_generate=20, max_generations=10, **extra):
    """A small search (size 4) by default; max_generations=None leaves
    both settings to the defaults."""
    base = (
        {}
        if max_generations is None
        else {"max_generations": max_generations, "multiobjective": {"size": 4}}
    )
    return {
        "generator": "ezga",
        "num_of_structures_to_generate": to_generate,
        "ezga_kwargs": {**base, **(ezga_kwargs or {})},
        **extra,
    }


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def ezga_events():
    handler = _Capture()
    logger = logging.getLogger("alomancy.structure_generation.ezga.generate_structures")
    logger.addHandler(handler)
    yield handler.records
    logger.removeHandler(handler)


@pytest.mark.unit
def test_objective_energy_per_atom():
    objective = objective_energy_per_atom()

    np.testing.assert_allclose(objective(_Dataset()), [-5.0, -6.0])


@pytest.mark.unit
def test_objective_energy_per_atom_accepts_unevaluated_seeds():
    class DatasetWithNan(_Dataset):
        def get_all_energies(self):
            return np.array([np.nan, -30.0])

    objective = objective_energy_per_atom()

    np.testing.assert_allclose(objective(DatasetWithNan()), [0.0, -6.0])


@pytest.mark.unit
def test_merge_is_per_key_for_dicts_and_replaces_lists():
    defaults = {"a": {"x": 1, "y": {"z": 2}}, "l": [1, 2], "s": 1}
    merged = _merge(defaults, {"a": {"y": {"w": 3}}, "l": [9]})
    assert merged == {"a": {"x": 1, "y": {"z": 2, "w": 3}}, "l": [9], "s": 1}
    assert defaults["a"]["y"] == {"z": 2}  # not mutated


@pytest.mark.unit
def test_defaults_take_species_from_general_elements():
    resolved = _resolve(_sg(), _ELEMENTS)
    assert "Pd" not in str(resolved)
    rattle = resolved["mutation_funcs"][0]
    assert rattle["species"] == _ELEMENTS
    removes = [m for m in resolved["mutation_funcs"] if "remove" in m["type"]]
    assert {m.get("species") for m in removes if "bounded" in m["type"]} == {"Cu", "O"}
    assert all(m.get("slab", False) is False for m in resolved["mutation_funcs"])
    features = resolved["evaluator"]["features_funcs"][0]
    assert features["IDs"] == _ELEMENTS


@pytest.mark.unit
def test_atom_number_range_bounds_the_default_add_and_remove():
    resolved = _resolve(
        _sg(structure_selection_kwargs={"atom_number_range": [2, 12]}), _ELEMENTS
    )
    by_type = {m["type"].rsplit(".", 1)[-1]: m for m in resolved["mutation_funcs"]}
    assert by_type["bounded_mutation_add"]["max_atoms"] == 12
    assert by_type["bounded_mutation_remove"]["min_atoms"] == 2


@pytest.mark.unit
def test_user_sections_merge_over_the_defaults():
    resolved = _resolve(
        _sg({"simulator": {"calculator": {"steps_max": 50}}, "mutation_funcs": []}),
        _ELEMENTS,
    )
    assert resolved["simulator"]["calculator"] == {
        "fmax": 0.05,
        "steps_max": 50,
        "optimizer": "FIRE",
    }
    assert resolved["mutation_funcs"] == []
    assert resolved["structure_selection_kwargs"]["num_of_seeds"] == 8


@pytest.mark.unit
def test_validate_stores_the_resolved_config_and_is_idempotent():
    sg = _sg()
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    resolved = sg["ezga_kwargs"]
    assert resolved["variation"]["crossover_probability"] == 0.0
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    assert sg["ezga_kwargs"] == resolved


@pytest.mark.unit
@pytest.mark.parametrize(
    "ezga_kwargs, message",
    [
        ({"output_path": "x"}, "output_path"),
        ({"resume": True}, "resume"),
        ({"rng": 3}, "rng"),
        ({"population": {"db_path": "x"}}, "population.db_path"),
        ({"population": {"storage": "memory"}}, "population.storage"),
        ({"simulator": {"calculator": {"calculator": "mace"}}}, "calculator"),
        ({"simulator": {"calculator": {"type": "ase"}}}, "simulator.calculator.type"),
    ],
)
def test_owned_keys_raise(ezga_kwargs, message):
    with pytest.raises(ValueError, match="must not set") as err:
        validate_kwargs(_sg(ezga_kwargs), elements=_ELEMENTS, seed=1)
    assert message in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "key, new_home",
    [
        ("population_size", "multiobjective.size"),
        ("mutation_operators", "mutation_funcs"),
        ("min_atoms", "bounded_mutation_remove"),
        ("max_atoms", "bounded_mutation_add"),
    ],
)
def test_removed_keys_name_their_new_home(key, new_home):
    with pytest.raises(ValueError, match=new_home):
        validate_kwargs(_sg({key: 2}), elements=_ELEMENTS, seed=1)


@pytest.mark.unit
def test_a_typo_is_caught_by_ezgas_own_schema():
    with pytest.raises(Exception, match="max_generation"):
        validate_kwargs(_sg({"max_generation": 3}), elements=_ELEMENTS, seed=1)


@pytest.mark.unit
def test_an_unknown_factory_fails_at_construction():
    with pytest.raises(Exception, match="no_such_mutation"):
        validate_kwargs(
            _sg(
                {
                    "mutation_funcs": [
                        {"type": "ezga.variation.mutation.no_such_mutation"}
                    ]
                }
            ),
            elements=_ELEMENTS,
            seed=1,
        )


@pytest.mark.unit
def test_an_unknown_calculator_setting_raises():
    with pytest.raises(ValueError, match="fmaks"):
        validate_kwargs(
            _sg({"simulator": {"calculator": {"fmaks": 0.1}}}),
            elements=_ELEMENTS,
            seed=1,
        )


@pytest.mark.unit
def test_a_species_outside_general_elements_raises():
    mutation = {
        "type": "ezga.variation.mutation.mutation_rattle",
        "std": 0.1,
        "species": ["Pd"],
    }
    with pytest.raises(ValueError, match=r"\['Pd'\] not in general.elements"):
        validate_kwargs(_sg({"mutation_funcs": [mutation]}), elements=_ELEMENTS, seed=1)


@pytest.mark.unit
def test_defaults_follow_the_ezga_wiki_example():
    resolved = _resolve(_sg({}, max_generations=None), _ELEMENTS)
    assert resolved["multiobjective"] == {
        "size": 100,
        "composition_repulsion_weight": 1.0,
    }
    assert resolved["thermostat"] == {
        "initial_temperature": 0.05,
        "constant_temperature": True,
    }
    assert resolved["population"]["collision_factor"] == 0.6
    assert resolved["simulator"]["calculator"]["steps_max"] == 100
    assert resolved["simulator"]["calculator"]["fmax"] == 0.05


@pytest.mark.unit
@pytest.mark.parametrize("to_generate, generations", [(500, 10), (20, 5), (1000, 20)])
def test_max_generations_is_derived_from_the_candidate_pool(to_generate, generations):
    sg = {
        "generator": "ezga",
        "num_of_structures_to_generate": to_generate,
        "ezga_kwargs": {},
    }
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    assert sg["ezga_kwargs"]["max_generations"] == generations


@pytest.mark.unit
def test_derived_generations_use_the_users_size():
    sg = {
        "generator": "ezga",
        "num_of_structures_to_generate": 100,
        "ezga_kwargs": {"multiobjective": {"size": 10}},
    }
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    assert sg["ezga_kwargs"]["max_generations"] == 20


@pytest.mark.unit
def test_a_users_max_generations_wins():
    sg = _sg({"max_generations": 7, "multiobjective": {"size": 100}})
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    assert sg["ezga_kwargs"]["max_generations"] == 7


@pytest.mark.unit
def test_a_non_positive_size_raises():
    with pytest.raises(ValueError, match="must be positive integers"):
        validate_kwargs(
            _sg({"multiobjective": {"size": 0}}), elements=_ELEMENTS, seed=1
        )


@pytest.mark.unit
def test_default_mutations_add_remove_each_element_and_swap_every_pair():
    resolved = _resolve(_sg(), ["Cu", "O", "H"])
    by_type: dict = {}
    for m in resolved["mutation_funcs"]:
        by_type.setdefault(m["type"].rsplit(".", 1)[-1], []).append(m)
    assert [m["species"] for m in by_type["bounded_mutation_add"]] == [
        ["Cu"],
        ["O"],
        ["H"],
    ]
    assert [m["species"] for m in by_type["bounded_mutation_remove"]] == [
        "Cu",
        "O",
        "H",
    ]
    swaps = {(m["ID1"], m["ID2"]) for m in by_type["mutation_swap"]}
    assert swaps == {
        ("Cu", "O"),
        ("O", "Cu"),
        ("Cu", "H"),
        ("H", "Cu"),
        ("O", "H"),
        ("H", "O"),
    }
    assert "mutation_remove_add" not in by_type


@pytest.mark.unit
def test_one_element_gets_no_swaps():
    resolved = _resolve(_sg(), ["Cu"])
    assert not [m for m in resolved["mutation_funcs"] if "swap" in m["type"]]


def _hise(**overrides):
    return {"hise": {"supercells": [[1, 1, 1], [2, 1, 1]], "overrides": overrides}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "key", ["output_path", "population.db_path", "population.dataset_path", "rng"]
)
def test_owned_hise_overrides_raise(key):
    with pytest.raises(ValueError, match=f"hise.overrides.{key}"):
        validate_kwargs(_sg(_hise(**{key: ["a", "b"]})), elements=_ELEMENTS, seed=1)


@pytest.mark.unit
def test_a_stage_calculator_is_settings_not_a_model():
    stages = [{}, {"calculator": "mace"}]
    with pytest.raises(ValueError, match=r"simulator.calculator\[1\].calculator"):
        validate_kwargs(
            _sg(_hise(**{"simulator.calculator": stages})), elements=_ELEMENTS, seed=1
        )


@pytest.mark.unit
def test_an_unknown_stage_calculator_setting_raises():
    stages = [{}, {"nvt_step": 40}]
    with pytest.raises(ValueError, match="nvt_step"):
        validate_kwargs(
            _sg(_hise(**{"simulator.calculator": stages})), elements=_ELEMENTS, seed=1
        )


@pytest.mark.unit
def test_a_valid_hise_block_passes_ezgas_schema():
    stages = [{}, {"nvt_steps": 40, "T": [300, 600]}]
    sg = _sg(_hise(**{"simulator.calculator": stages, "max_generations": [5, 5]}))
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    assert sg["ezga_kwargs"]["hise"]["overrides"]["simulator.calculator"] == stages


@pytest.mark.unit
def test_the_count_check_uses_the_last_hise_stage():
    sg = _sg(_hise(max_generations=[50, 1]), to_generate=20)  # last stage: 1 x 4
    with pytest.raises(ValueError, match="= 4 structures, fewer than"):
        validate_kwargs(sg, elements=_ELEMENTS, seed=1)


@pytest.mark.unit
def test_too_few_structures_raises():
    sg = _sg({"max_generations": 2, "multiobjective": {"size": 4}}, to_generate=9)
    with pytest.raises(ValueError, match="= 8 structures, fewer than"):
        validate_kwargs(sg, elements=_ELEMENTS, seed=1)


@pytest.mark.unit
def test_under_twice_the_structures_warns(ezga_events):
    sg = _sg({"max_generations": 2, "multiobjective": {"size": 4}}, to_generate=5)
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    events = [r for r in ezga_events if getattr(r, "event", None)]
    assert [r.event for r in events] == ["ezga_few_structures_configured"]
    assert events[0].data == {"expected": 8, "to_generate": 5}


@pytest.mark.unit
def test_enough_structures_does_not_warn(ezga_events):
    validate_kwargs(_sg(to_generate=20), elements=_ELEMENTS, seed=1)
    assert not [r for r in ezga_events if getattr(r, "event", None)]


class _APM:
    def __init__(self, n):
        self.atomCount = n


class _Structure:
    def __init__(self, n):
        self.AtomPositionManager = _APM(n)


@pytest.mark.unit
def test_bounded_mutation_rejects_results_outside_the_limits():
    grow = _bounded(lambda s: _Structure(s.AtomPositionManager.atomCount + 1), None, 5)
    assert grow(_Structure(3)).AtomPositionManager.atomCount == 4
    assert grow(_Structure(5)) is None  # already at the limit
    shrink = _bounded(
        lambda s: _Structure(s.AtomPositionManager.atomCount - 1), 2, None
    )
    assert shrink(_Structure(3)).AtomPositionManager.atomCount == 2
    assert shrink(_Structure(2)) is None


@pytest.mark.unit
def test_bounded_mutation_passes_rng_and_keeps_tuple_results():
    seen = {}

    def mutation(structure, rng=None):
        seen["rng"] = rng
        return (_Structure(4), 0.0, None, None)

    rng = np.random.default_rng(0)
    result = _bounded(mutation, None, 10)(_Structure(3), rng=rng)
    assert seen["rng"] is rng
    assert isinstance(result, tuple)


def _offspring(generations):
    out = []
    for generation, n in generations.items():
        for i in range(n):
            atoms = Atoms("Cu")
            atoms.info["ezga_generation"] = generation
            atoms.info["i"] = i
            out.append(atoms)
    return out


@pytest.mark.unit
def test_most_recent_takes_newest_generations_whole():
    chosen = most_recent(_offspring({0: 4, 1: 4, 2: 4}), 6, seed=1)
    generations = [a.info["ezga_generation"] for a in chosen]
    assert generations.count(2) == 4  # newest generation in full
    assert generations.count(1) == 2  # only the oldest one needed is sampled
    assert 0 not in generations


@pytest.mark.unit
def test_most_recent_is_reproducible_and_returns_all_when_short():
    offspring = _offspring({0: 5, 1: 5})
    first = [a.info["i"] for a in most_recent(offspring, 7, seed=3)]
    assert first == [a.info["i"] for a in most_recent(offspring, 7, seed=3)]
    assert len(most_recent(offspring, 50, seed=3)) == 10


@pytest.mark.unit
def test_with_hise_the_seed_range_does_not_bound_the_default_mutations(ezga_events):
    sg = _sg(
        _hise(max_generations=[10, 10]),
        structure_selection_kwargs={"atom_number_range": [2, 20]},
    )
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    bounded = [m for m in sg["ezga_kwargs"]["mutation_funcs"] if "bounded" in m["type"]]
    assert bounded
    assert all(m.get("max_atoms") is None for m in bounded)
    assert all(m.get("min_atoms") is None for m in bounded)
    events = [r.event for r in ezga_events if getattr(r, "event", None)]
    assert events == ["ezga_atom_range_seeds_only"]


@pytest.mark.unit
def test_with_hise_and_own_mutations_there_is_no_range_warning(ezga_events):
    mutation = {"type": "ezga.variation.mutation.mutation_rattle", "std": 0.1}
    sg = _sg(
        {**_hise(max_generations=[10, 10]), "mutation_funcs": [mutation]},
        structure_selection_kwargs={"atom_number_range": [2, 20]},
    )
    validate_kwargs(sg, elements=_ELEMENTS, seed=1)
    assert not [r for r in ezga_events if getattr(r, "event", None)]
