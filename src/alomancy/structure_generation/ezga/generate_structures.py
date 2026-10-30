"""EZGA structure generator (``structure_generation.generator: "ezga"``).

``structure_generation.ezga_kwargs`` is EZGA's own config (the dict its YAML
holds), plus two ALomancy keys (``structure_selection_kwargs`` and
``local``). ALomancy fills in what it owns: the calculator (the loop's best
model, wrapped in EZGA's ``ase_calculator`` with the user's
``simulator.calculator`` settings), the seeds, the output and database
paths, ``resume`` and ``rng``. The search runs as one remote job on
``structure_generation.hpc`` (or in-process with ``local: true``), and the
candidates are read back from EZGA's database: the offspring of the most
recent generations, up to ``num_of_structures_to_generate``.
"""

import copy
import inspect
import itertools
import logging
import math
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from alomancy.mlip.base import CalculatorSpec
from alomancy.utils.clean_structures import drop_unwritable_info
from alomancy.utils.file_saving_and_parsing import read_xyz
from alomancy.utils.seed_selection import mark_structures_for_dft, select_diverse_seeds

logger = logging.getLogger(__name__)

_EZGA_DIRNAME = "ezga"
_CANDIDATES_FILENAME = "ezga_candidates.xyz"
_SEEDS_FILENAME = "ezga_seeds.xyz"
_INITIAL_POPULATION_FILENAME = "initial_population.xyz"
_DEFAULT_NUM_OF_SEEDS = 8
_MODULE = "alomancy.structure_generation.ezga.generate_structures"

# ezga_kwargs keys that are ALomancy's, popped before EZGA sees the config.
_ALOMANCY_KEYS = ("structure_selection_kwargs", "local")
_SELECTION_KEYS = {"num_of_seeds", "enforce_chemical_diversity"}

# EZGA settings ALomancy sets itself; a user value raises.
_OWNED_KEYS = {
    "output_path": "ALomancy writes each loop's search to its own directory",
    "resume": "every search starts fresh (ALomancy restarts a crashed search)",
    "rng": "EZGA is seeded from general.seed",
}
_OWNED_POPULATION_KEYS = {
    "dataset_path": "the seeds are ALomancy's training structures",
    "db_path": "ALomancy reads the candidates from this database",
    "db_ro_path": "ALomancy reads the candidates from this database",
    "storage": "the candidates are read from EZGA's database, so it must persist",
}
# ase_calculator's own arguments that aren't settings: the model is the
# loop's best model.
_OWNED_CALCULATOR_KEYS = {"calculator"}

# The 1.0.3 ezga_kwargs keys, with where each setting lives now.
_REMOVED_KEYS = {
    "population_size": "multiobjective.size",
    "mutation_operators": "mutation_funcs (EZGA's own list of mutation factories)",
    "min_atoms": (
        "the min_atoms argument of bounded_mutation_remove, or "
        "structure_generation.structure_selection_kwargs.atom_number_range"
    ),
    "max_atoms": (
        "the max_atoms argument of bounded_mutation_add, or "
        "structure_generation.structure_selection_kwargs.atom_number_range"
    ),
}

# Element-independent defaults, after the EZGA wiki's MLIP active-learning
# example (Examples/MLIP-Active-Learning): a wide population (100), a low,
# constant selection temperature, a looser collision check, composition
# niching and full relaxations. The mutations and the composition feature
# depend on general.elements and are added by _default_kwargs;
# max_generations is derived from num_of_structures_to_generate (_resolve).
_EZGA_KWARGS_DEFAULTS: dict[str, Any] = {
    "structure_selection_kwargs": {
        "num_of_seeds": _DEFAULT_NUM_OF_SEEDS,
        "enforce_chemical_diversity": False,
    },
    "local": False,
    "population": {"filter_duplicates": True, "collision_factor": 0.6},
    "thermostat": {"initial_temperature": 0.05, "constant_temperature": True},
    "multiobjective": {"size": 100, "composition_repulsion_weight": 1.0},
    "variation": {
        "initial_mutation_rate": 1.0,
        "min_mutation_rate": 1.0,
        "crossover_probability": 0.0,
    },
    "simulator": {
        "mode": "sampling",
        "calculator": {"fmax": 0.05, "steps_max": 100, "optimizer": "FIRE"},
    },
}
# Without max_generations, enough generations for twice
# num_of_structures_to_generate, and at least this many.
_MIN_DERIVED_GENERATIONS = 5

# Keys of mutation/feature factories that name chemical species.
_SPECIES_KEYS = {
    "species",
    "species_add",
    "species_remove",
    "bound",
    "molecule",
    "IDs",
    "ID1",
    "ID2",
}

_MOLECULE_KEYS = {
    "species",
    "species_add",
    "species_remove",
    "molecule",
}

# hise.overrides keys (dot paths) ALomancy owns, as at the top level.
_OWNED_OVERRIDE_KEYS = {
    "output_path",
    "resume",
    "rng",
    *(f"population.{key}" for key in _OWNED_POPULATION_KEYS),
}
_CALCULATOR_OVERRIDE = "simulator.calculator"


def objective_energy_per_atom(scale: float = 1.0) -> Callable[[Any], np.ndarray]:
    """Return an EZGA objective based on potential energy per atom."""

    def compute(dataset: Any) -> np.ndarray:
        energies = np.asarray(dataset.get_all_energies(), dtype=float)
        compositions, _ = dataset.get_all_compositions(return_species=True)
        atom_counts = np.asarray(compositions, dtype=float).sum(axis=1)

        if energies.shape[0] != atom_counts.shape[0]:
            raise ValueError(
                "EZGA energies and compositions contain different numbers "
                "of structures."
            )
        if np.any(atom_counts < 1):
            raise ValueError("Cannot compute energy per atom for an empty structure.")
        # Initial EZGA seeds have not been evaluated by the model yet and therefore
        # carry NaN energies.  Match EZGA objective_energy semantics by using a
        # neutral zero until the simulator populates their energies.
        clean_energies = np.nan_to_num(energies, nan=0.0)
        return scale * clean_energies / atom_counts

    return compute


def _bounded(
    mutation: Callable, min_atoms: int | None, max_atoms: int | None
) -> Callable[..., Any]:
    """Wrap an EZGA mutation so a result outside [min_atoms, max_atoms]
    (or an input already at the limit it would cross) is rejected (None,
    EZGA's no-op). Passes EZGA's rng through so the search stays seeded."""
    takes_rng = "rng" in inspect.signature(mutation).parameters

    def apply(structure: Any, rng: np.random.Generator | None = None) -> Any:
        n = structure.AtomPositionManager.atomCount
        if (max_atoms is not None and n >= max_atoms) or (
            min_atoms is not None and n <= min_atoms
        ):
            return None
        result = mutation(structure, rng=rng) if takes_rng else mutation(structure)
        candidate = result[0] if isinstance(result, tuple) else result
        if candidate is None:
            return None
        n_new = candidate.AtomPositionManager.atomCount
        if (max_atoms is not None and n_new > max_atoms) or (
            min_atoms is not None and n_new < min_atoms
        ):
            return None
        return result

    return apply


def bounded_mutation_add(
    species: list[str],
    max_atoms: int | None = None,
    bound: list[str] | None = None,
    collision_tolerance: float = 2.0,
    slab: bool = False,
) -> Callable[..., Any]:
    """EZGA's add mutation, never growing a structure past *max_atoms*."""
    from ezga.variation.mutation import mutation_add

    mutation = mutation_add(
        species=species, bound=bound, collision_tolerance=collision_tolerance, slab=slab
    )
    return _bounded(mutation, None, max_atoms)


def bounded_mutation_remove(
    species: str, min_atoms: int | None = None
) -> Callable[..., Any]:
    """EZGA's remove mutation, never shrinking a structure below *min_atoms*."""
    from ezga.variation.mutation import mutation_remove

    return _bounded(mutation_remove(species=species), min_atoms, None)


def bounded_mutation_remove_molecule(
    molecule: str,
    min_atoms: int | None = None,
) -> Callable[..., Any]:
    from collections import Counter

    from ezga.utils.chemistry import unit_stoichiometry
    from ezga.variation.mutation_adsorbates import identify_adsorbates

    target = Counter(unit_stoichiometry(molecule))

    def in_molecule(idx: int, structure: Any) -> bool:
        return structure.AtomPositionManager.atomLabelsList[idx] in target

    def mutation(structure: Any, rng: np.random.Generator | None = None) -> Any:
        rng = rng or np.random.default_rng()
        found = [
            fragment
            for fragment in identify_adsorbates(structure, [in_molecule])
            if Counter(fragment.labels) == target
        ]
        if not found:
            return None
        child = copy.deepcopy(structure)
        child.AtomPositionManager.remove_atom(found[rng.integers(len(found))].indices)
        return child

    return _bounded(mutation, min_atoms, None)


def structure_elements(elements: list[str], molecules: list[str] | None) -> list[str]:
    from ezga.utils.chemistry import unit_stoichiometry

    in_molecules = {elem for mol in molecules or [] for elem in unit_stoichiometry(mol)}
    return [elem for elem in elements if elem not in in_molecules]


def _default_kwargs(
    elements: list[str], atom_number_range: Any, molecules: list[str] | None = None
) -> dict[str, Any]:
    """The defaults, with every species-dependent setting from
    general.elements and the add/remove limits from the seed filter's
    atom_number_range (unbounded when it isn't set, or with HiSE: _resolve
    passes None)."""
    min_atoms = max_atoms = None
    framework = structure_elements(elements, molecules)
    if atom_number_range and tuple(atom_number_range) != (0, 0):
        min_atoms, max_atoms = (int(n) for n in atom_number_range)
    mutations: list[dict[str, Any]] = [
        {
            "type": "ezga.variation.mutation.mutation_rattle",
            "std": 0.05,
            "species": list(elements),
        },
        {"type": "ezga.variation.mutation.mutation_random_strain", "max_strain": 0.02},
    ]
    for element in framework:
        mutations += [
            {
                "type": f"{_MODULE}.bounded_mutation_add",
                "species": [element],
                "max_atoms": max_atoms,
            },
            {
                "type": f"{_MODULE}.bounded_mutation_remove",
                "species": element,
                "min_atoms": min_atoms,
            },
        ]
    mutations += [
        {"type": "ezga.variation.mutation.mutation_swap", "ID1": a, "ID2": b}
        for a, b in itertools.permutations(framework, 2)
    ]
    for molecule in molecules or []:
        mutations += [
            {
                "type": f"{_MODULE}.bounded_mutation_add",
                "species": [molecule],
                "max_atoms": max_atoms,
                "bound": framework,
                "slab": True,
            },
            {
                "type": f"{_MODULE}.bounded_mutation_remove_molecule",
                "molecule": molecule,
                "min_atoms": min_atoms,
            },
        ]
    return {
        **copy.deepcopy(_EZGA_KWARGS_DEFAULTS),
        "mutation_funcs": mutations,
        "evaluator": {
            "features_funcs": [
                {
                    "type": "ezga.evaluator.features.feature_composition_vector",
                    "IDs": list(elements),
                }
            ],
            "objectives_funcs": [{"type": f"{_MODULE}.objective_energy_per_atom"}],
        },
    }


def _merge(defaults: dict, user: dict) -> dict:
    """User settings over the defaults: dicts merge key by key at every
    level; anything else (lists such as mutation_funcs, scalars) replaces
    the default."""
    merged = copy.deepcopy(defaults)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _check_species(
    value: Any, elements: set[str], molecules: set[str], where: str
) -> None:
    """Raise on a species name outside general.elements anywhere in a
    mutation or evaluator entry."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _SPECIES_KEYS and item is not None:
                allowed = elements | molecules if key in _MOLECULE_KEYS else elements
                names = [item] if isinstance(item, str) else item
                if isinstance(names, list | tuple):
                    unknown = sorted({n for n in names if isinstance(n, str)} - allowed)
                    if unknown:
                        raise ValueError(
                            f"structure_generation.ezga_kwargs.{where}: {key} "
                            f"{unknown} not in general.elements {sorted(elements)}."
                        )
            _check_species(item, elements, molecules, where)
    elif isinstance(value, list | tuple):
        for item in value:
            _check_species(item, elements, molecules, where)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _resolve(
    sg_config: dict, elements: list[str], molecules: list[str] | None = None
) -> dict[str, Any]:
    """ezga_kwargs merged over the defaults, after rejecting removed and
    ALomancy-owned keys. Idempotent: resolving the result changes nothing."""
    user = sg_config.get("ezga_kwargs") or {}
    if not isinstance(user, dict):
        raise ValueError(
            "structure_generation.ezga_kwargs must be a mapping (EZGA's own "
            f"config), got {type(user).__name__}."
        )
    removed = [key for key in _REMOVED_KEYS if key in user]
    if removed:
        moves = "; ".join(f"{key} -> {_REMOVED_KEYS[key]}" for key in removed)
        raise ValueError(
            "structure_generation.ezga_kwargs is now EZGA's own config "
            f"(docs/deprecations.md). Removed keys: {moves}."
        )
    problems = [f"{key} ({why})" for key, why in _OWNED_KEYS.items() if key in user]
    population = user.get("population") or {}
    problems += [
        f"population.{key} ({why})"
        for key, why in _OWNED_POPULATION_KEYS.items()
        if key in population
    ]
    calculator = (user.get("simulator") or {}).get("calculator") or {}
    if not isinstance(calculator, dict):
        raise ValueError(
            "structure_generation.ezga_kwargs.simulator.calculator holds the "
            "settings of EZGA's ase_calculator (fmax, steps_max, nvt_steps, ...); "
            "the model is the loop's best model. Got "
            f"{type(calculator).__name__}."
        )
    problems += [
        f"simulator.calculator.{key} (the model is the loop's best model)"
        for key in sorted(
            set(calculator) & (_OWNED_CALCULATOR_KEYS | {"type", "factory"})
        )
    ]
    overrides = (user.get("hise") or {}).get("overrides") or {}
    problems += [
        f"hise.overrides.{key} (as at the top level)"
        for key in sorted(set(overrides) & _OWNED_OVERRIDE_KEYS)
    ]
    for stage, settings in enumerate(_stage_calculator_overrides(user) or []):
        problems += [
            f"hise.overrides.{_CALCULATOR_OVERRIDE}[{stage}].{key} (the model is "
            "the loop's best model)"
            for key in sorted(
                set(settings) & (_OWNED_CALCULATOR_KEYS | {"type", "factory"})
            )
        ]
    if problems:
        raise ValueError(
            "structure_generation.ezga_kwargs must not set: "
            + "; ".join(problems)
            + ". ALomancy sets these itself."
        )
    # The seed filter's range is sized for the seeds; HiSE tiles them into
    # larger cells, where it would refuse every add and remove.
    atom_number_range = (
        None
        if "hise" in user
        else (sg_config.get("structure_selection_kwargs") or {}).get(
            "atom_number_range"
        )
    )
    resolved = _merge(_default_kwargs(elements, atom_number_range, molecules), user)
    size: Any = (resolved.get("multiobjective") or {}).get("size")
    if "max_generations" not in resolved and _positive_int(size):
        to_generate = sg_config.get("num_of_structures_to_generate") or 0
        resolved["max_generations"] = max(
            _MIN_DERIVED_GENERATIONS, math.ceil(2 * to_generate / size)
        )
    return resolved


def _stage_calculator_overrides(ezga_kwargs: dict[str, Any]) -> list[dict] | None:
    """hise.overrides["simulator.calculator"]: one dict of ase_calculator
    settings per HiSE stage (None without one). Raises if it isn't a list
    of dicts."""
    overrides = (ezga_kwargs.get("hise") or {}).get("overrides") or {}
    if _CALCULATOR_OVERRIDE not in overrides:
        return None
    stages = overrides[_CALCULATOR_OVERRIDE]
    if not isinstance(stages, list) or not all(isinstance(s, dict) for s in stages):
        raise ValueError(
            "structure_generation.ezga_kwargs.hise.overrides."
            f"{_CALCULATOR_OVERRIDE} must be a list with one dict of "
            "ase_calculator settings per stage (the model is the loop's best "
            "model)."
        )
    return stages


def _last_stage_value(ezga_kwargs: dict[str, Any], dotted: str, default: Any) -> Any:
    """A setting as the last HiSE stage sees it: its hise.overrides entry
    if there is one, else *default* (the top-level value)."""
    overrides = (ezga_kwargs.get("hise") or {}).get("overrides") or {}
    values = overrides.get(dotted)
    return values[-1] if isinstance(values, list) and values else default


def _ezga_config(
    ezga_kwargs: dict[str, Any],
    *,
    output_dir: Path,
    make_calculator: Callable[[dict[str, Any]], Any],
    seed: int,
) -> dict[str, Any]:
    """EZGA's config dict: the user's settings plus everything ALomancy
    owns. *make_calculator* turns ase_calculator settings into the live
    object for simulator.calculator (and for each HiSE stage's calculator
    override, its settings over the top-level ones), so nothing has to
    round-trip through YAML."""
    config = {
        key: copy.deepcopy(value)
        for key, value in ezga_kwargs.items()
        if key not in _ALOMANCY_KEYS
    }
    config["output_path"] = str(output_dir)
    config["resume"] = False
    config["rng"] = int(seed)
    population = config.setdefault("population", {})
    population["dataset_path"] = str(output_dir / _INITIAL_POPULATION_FILENAME)
    population["db_path"] = str(output_dir / "db")
    population["db_ro_path"] = str(output_dir / "db_ro")
    population["storage"] = "composite"
    settings = _calculator_settings(ezga_kwargs)
    config.setdefault("simulator", {})["calculator"] = make_calculator(settings)
    stages = _stage_calculator_overrides(ezga_kwargs)
    if stages is not None:
        config["hise"]["overrides"][_CALCULATOR_OVERRIDE] = [
            make_calculator({**settings, **stage}) for stage in stages
        ]
    return config


def _calculator_settings(ezga_kwargs: dict[str, Any]) -> dict[str, Any]:
    return dict(ezga_kwargs.get("simulator", {}).get("calculator", {}))


def validate_kwargs(
    sg_config: dict,
    *,
    elements: list[str],
    seed: int,
    molecules: list[str] | None = None,
) -> None:
    """Construction-time check (registry hook): resolve ezga_kwargs and
    store the result back into *sg_config*, so the summary and generate()
    see the full config. EZGA's own schema (pydantic, unknown keys
    rejected) checks the config and every factory is imported, so a typo
    fails now rather than at the first loop's generation phase."""
    from ezga.factory import load_config
    from ezga.simulator.ase_calculator import ase_calculator

    user = sg_config.get("ezga_kwargs") or {}
    atom_number_range = (sg_config.get("structure_selection_kwargs") or {}).get(
        "atom_number_range"
    )
    if (
        isinstance(user, dict)
        and "hise" in user
        and "mutation_funcs" not in user
        and atom_number_range
        and tuple(atom_number_range) != (0, 0)
    ):
        logger.warning(
            "structure_generation.structure_selection_kwargs.atom_number_range "
            "%s only filters EZGA's seeds here: with HiSE the structures are "
            "tiled into larger cells, so the default add/remove mutations are "
            "unbounded. Write mutation_funcs to bound them per your cells.",
            list(atom_number_range),
            extra={
                "event": "ezga_atom_range_seeds_only",
                "data": {"atom_number_range": list(atom_number_range)},
            },
        )
    resolved = _resolve(sg_config, elements, molecules)
    _check_species(
        resolved.get("mutation_funcs"),
        set(elements),
        set(molecules or []),
        "mutation_funcs",
    )
    _check_species(
        resolved.get("evaluator"), set(elements), set(molecules or []), "evaluator"
    )

    selection = resolved["structure_selection_kwargs"]
    unknown = set(selection) - _SELECTION_KEYS
    if unknown:
        raise ValueError(
            "structure_generation.ezga_kwargs.structure_selection_kwargs: unknown "
            f"keys {sorted(unknown)} (known: {sorted(_SELECTION_KEYS)})."
        )
    if not _positive_int(selection["num_of_seeds"]):
        raise ValueError(
            "structure_generation.ezga_kwargs.structure_selection_kwargs."
            f"num_of_seeds must be a positive integer, got {selection['num_of_seeds']!r}."
        )
    if not isinstance(resolved["local"], bool):
        raise ValueError(
            "structure_generation.ezga_kwargs.local must be true or false, "
            f"got {resolved['local']!r}."
        )

    calculator_params = set(inspect.signature(ase_calculator).parameters)
    calculator_settings = {"simulator.calculator": _calculator_settings(resolved)}
    for stage, settings in enumerate(_stage_calculator_overrides(resolved) or []):
        calculator_settings[f"hise.overrides.{_CALCULATOR_OVERRIDE}[{stage}]"] = (
            settings
        )
    for where, settings in calculator_settings.items():
        unknown = set(settings) - calculator_params
        if unknown:
            raise ValueError(
                f"structure_generation.ezga_kwargs.{where}: unknown "
                f"ase_calculator settings {sorted(unknown)}."
            )

    # Only the last HiSE stage's database is read, so its settings count.
    generations: Any = _last_stage_value(
        resolved, "max_generations", resolved.get("max_generations")
    )
    size: Any = _last_stage_value(
        resolved,
        "multiobjective.size",
        (resolved.get("multiobjective") or {}).get("size"),
    )
    if not (_positive_int(generations) and _positive_int(size)):
        raise ValueError(
            "structure_generation.ezga_kwargs.max_generations and "
            "multiobjective.size must be positive integers (for HiSE, the "
            "last stage's): together they set how many structures EZGA makes "
            "(max_generations x size), which must cover structure_generation."
            f"num_of_structures_to_generate. Got max_generations={generations!r}, "
            f"multiobjective.size={size!r}."
        )
    to_generate = sg_config.get("num_of_structures_to_generate")
    expected = generations * size
    if to_generate is not None and expected < to_generate:
        raise ValueError(
            f"EZGA makes about max_generations x multiobjective.size = {expected} "
            "structures, fewer than structure_generation."
            f"num_of_structures_to_generate ({to_generate}). Raise either setting."
        )
    if to_generate is not None and expected < 2 * to_generate:
        logger.warning(
            "EZGA makes about max_generations x multiobjective.size = %d "
            "structures, less than twice structure_generation."
            "num_of_structures_to_generate (%d): duplicates and failed "
            "relaxations can leave it short of candidates.",
            expected,
            to_generate,
            extra={
                "event": "ezga_few_structures_configured",
                "data": {"expected": expected, "to_generate": to_generate},
            },
        )

    with tempfile.TemporaryDirectory() as tmp:
        # load_config mutates its input; the placeholder calculator is never
        # called.
        load_config(
            _ezga_config(
                resolved,
                output_dir=Path(tmp),
                make_calculator=lambda _settings: lambda **_: None,
                seed=seed,
            )
        )
    sg_config["ezga_kwargs"] = resolved


def run_ezga(
    seeds: list[Atoms],
    calculator_spec: CalculatorSpec,
    ezga_kwargs: dict[str, Any],
    output_dir: str,
    seed: int = 803,
) -> str:
    """Remote worker (module-level: ExPyRe pickles it by reference): run one
    EZGA search (or, with a ``hise`` block, EZGA's HiSE stages) from *seeds*
    with the trained model, built once on the node. Every structure EZGA
    evaluates is stored in a database: ``<output_dir>/db``, or for HiSE the
    last stage's directory. Returns that database's path relative to
    *output_dir*, which generate() reads after the job (a failed job
    returns None)."""
    from ezga.factory import build_default_engine, load_config
    from ezga.simulator.ase_calculator import ase_calculator

    if not seeds:
        raise ValueError("EZGA requires at least one seed structure.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out / _INITIAL_POPULATION_FILENAME, seeds, format="extxyz")
    model = calculator_spec.build()
    config = _ezga_config(
        ezga_kwargs,
        output_dir=out,
        make_calculator=lambda settings: ase_calculator(calculator=model, **settings),
        seed=seed,
    )
    if "hise" in config:
        from ezga.hise.manager import run_hise

        stage_dirs = list(run_hise(load_config(config)).values())
        return str(Path(stage_dirs[-1]).resolve().relative_to(out.resolve()))
    build_default_engine(load_config(config)).run()
    return "db"


def read_offspring(db_path: Path) -> list[Atoms]:
    """Every structure EZGA's latest run generated, from its database: the
    containers with a generation number (the seeds have none) and the
    newest run_uid. A HiSE stage's database also holds the earlier stages'
    structures, carried with their own run_uid and generation numbers.
    Geometry only, plus ezga_generation / ezga_operation /
    ezga_mutation_list (indices into mutation_funcs; mutation children
    only, a crossover child's list is its parent's) / ezga_seed_db_id (the
    global_db_id of the seed it descends from) / ezga_energy (the model's
    energy, not a label)."""
    from alomancy.database.global_database import atoms_from_container, read_storage

    generated = []
    for container in read_storage(db_path):
        metadata = dict(container.AtomPositionManager.metadata or {})
        if metadata.get("generation") is not None:
            generated.append((container, metadata))
    started: dict[Any, float] = {}
    for _, metadata in generated:
        run, created = metadata.get("run_uid"), float(metadata.get("created_at", 0))
        started[run] = min(created, started.get(run, created))
    latest = max(started, key=lambda run: started[run]) if started else None

    offspring = []
    for container, metadata in generated:
        if metadata.get("run_uid") != latest:
            continue
        atoms = atoms_from_container(container, labels=False)
        atoms.info["ezga_generation"] = int(metadata["generation"])
        if isinstance(metadata.get("operation"), str):
            atoms.info["ezga_operation"] = metadata["operation"]
        mutations = np.atleast_1d(metadata.get("mutation_list", []))
        if metadata.get("operation") == "mutation" and mutations.size:
            atoms.info["ezga_mutation_list"] = [int(i) for i in mutations]
        if metadata.get("global_db_id") is not None:
            atoms.info["ezga_seed_db_id"] = int(metadata["global_db_id"])
        energy = container.AtomPositionManager.energy
        if energy is not None and np.isfinite(energy):
            atoms.info["ezga_energy"] = float(energy)
        offspring.append(atoms)
    return offspring


def most_recent(offspring: list[Atoms], n: int, seed: int) -> list[Atoms]:
    """Up to *n* structures, newest generations first: whole generations
    are taken while they fit, and only the oldest one needed is sampled
    (reproducibly, from *seed*)."""
    by_generation: dict[int, list[Atoms]] = {}
    for atoms in offspring:
        by_generation.setdefault(atoms.info["ezga_generation"], []).append(atoms)
    rng = np.random.default_rng(seed)
    chosen: list[Atoms] = []
    for generation in sorted(by_generation, reverse=True):
        group = by_generation[generation]
        room = n - len(chosen)
        if room <= 0:
            break
        if len(group) <= room:
            chosen.extend(group)
        else:
            picks = sorted(rng.choice(len(group), room, replace=False))
            chosen.extend(group[i] for i in picks)
    return chosen


# ---------------------------------------------------------------------------
# structure_generator registry entry points
# ---------------------------------------------------------------------------


def _sg_dir(base_name: str) -> Path:
    return Path("results", base_name, "structure_generation")


def _candidates_path(base_name: str) -> Path:
    return _sg_dir(base_name) / _EZGA_DIRNAME / _CANDIDATES_FILENAME


def output_paths(config: dict, *, base_name: str, name: str) -> list[Path]:  # noqa: ARG001
    """Restart check: the candidates file generate() writes last."""
    return [_candidates_path(base_name)]


def read_existing_result(config: dict, *, base_name: str, name: str) -> list[Atoms]:  # noqa: ARG001
    path = _candidates_path(base_name)
    if not path.exists():
        raise ValueError(f"No cached EZGA result at {path}.")
    return list(read_xyz(path))


def _seeds(
    seed_atoms: list[Atoms], selection: dict, *, base_name: str, name: str, seed: int
) -> list[Atoms]:
    """At most num_of_seeds diverse seeds from the eligible structures,
    saved so a restart searches from the same ones."""
    path = _sg_dir(base_name) / _SEEDS_FILENAME
    if path.exists():
        return list(read_xyz(path))
    if not seed_atoms:
        raise ValueError(
            "EZGA has no seed structures: no training structure passed "
            "structure_generation.structure_selection_kwargs."
        )
    seeds = select_diverse_seeds(
        base_name=base_name,
        job_name=name,
        eligible_structures=seed_atoms,
        num_of_md_starts=min(selection["num_of_seeds"], len(seed_atoms)),
        enforce_chemical_diversity=selection["enforce_chemical_diversity"],
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, seeds, format="extxyz")
    return list(seeds)


def generate(
    seed_atoms: list[Atoms],
    calculator_spec: CalculatorSpec,
    config: dict,
    *,
    base_name: str,
    name: str,
    hpc: dict,
    max_time: str,
    elements: list[str],
    molecules: list[str] | None = None,
    seed: int = 803,
) -> list[Atoms]:
    """Run one EZGA search from diverse seeds as a remote job (or locally
    with ezga_kwargs.local), then take num_of_structures_to_generate
    candidates from its database, newest generations first."""
    candidates_path = _candidates_path(base_name)
    if candidates_path.exists():
        return list(read_xyz(candidates_path))

    ezga_kwargs = _resolve(config, elements, molecules)
    seeds = _seeds(
        seed_atoms,
        ezga_kwargs["structure_selection_kwargs"],
        base_name=base_name,
        name=name,
        seed=seed,
    )
    ezga_dir = candidates_path.parent
    if ezga_dir.exists():
        # A search that didn't finish: EZGA would append to its database.
        moved = ezga_dir.with_name(
            f"{_EZGA_DIRNAME}.failed_{datetime.now():%Y%m%d_%H%M%S}"
        )
        shutil.move(str(ezga_dir), str(moved))
        logger.warning(
            "An unfinished EZGA search for %s was moved to %s; starting a new one.",
            base_name,
            moved,
            extra={"event": "ezga_restarted", "data": {"moved_to": str(moved)}},
        )

    kwargs = {
        "seeds": seeds,
        "calculator_spec": calculator_spec,
        "ezga_kwargs": ezga_kwargs,
        "output_dir": str(ezga_dir),
        "seed": seed,
    }
    if ezga_kwargs["local"]:
        result = run_ezga(**kwargs)
    else:
        from alomancy.configs.remote_info import get_remote_info
        from alomancy.remote_submission.executor import submit_n

        remote_info = get_remote_info(
            {"hpc": hpc, "name": name, "max_time": max_time},
            input_files=[calculator_spec.model_path],
        )
        [result] = submit_n(
            run_ezga,
            [
                {
                    "function_kwargs": kwargs,
                    "output_files": [str(ezga_dir)],
                    "item": "ezga_search",
                }
            ],
            remote_info,
        )
        if result is None:
            raise RuntimeError(
                f"The EZGA search for {base_name} failed (see the job's logs). "
                "Nothing was cached, so a restart will retry."
            )

    db_path = ezga_dir / result
    offspring = read_offspring(db_path)
    if not offspring:
        raise RuntimeError(
            f"The EZGA search for {base_name} generated no structures ({db_path}). "
            "Nothing was cached, so a restart will retry."
        )
    to_generate = config.get("num_of_structures_to_generate", len(offspring))
    candidates = most_recent(offspring, to_generate, seed)
    mark_structures_for_dft(candidates, base_name, name)
    for atoms in candidates:
        drop_unwritable_info(atoms)
    generations = sorted({a.info["ezga_generation"] for a in offspring})
    logger.info(
        "EZGA for %s: %d seed(s), %d generation(s), %d structure(s) generated, "
        "%d taken as candidates.",
        base_name,
        len(seeds),
        len(generations),
        len(offspring),
        len(candidates),
        extra={
            "event": "ezga_summary",
            "data": {
                "seeds": len(seeds),
                "generations": len(generations),
                "generated": len(offspring),
                "candidates": len(candidates),
                "oldest_generation_used": min(
                    a.info["ezga_generation"] for a in candidates
                ),
            },
        },
    )
    write(candidates_path, candidates, format="extxyz")
    return candidates


def report_section(stats: dict, **kwargs: Any) -> Any:
    """Loop-report section for this module (see analysis/report/sections.py)."""
    from alomancy.analysis.report.sections import ezga_section

    return ezga_section(stats, **kwargs)
