import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write
from ezga.factory import build_default_engine, load_config

from alomancy.mlip.base import CalculatorSpec
from alomancy.utils.dataset_curation import geometry_digest

logger = logging.getLogger(__name__)
_CANDIDATE_FILENAME = "ezga_new_candidates.xyz"
_CACHE_MANIFEST_FILENAME = "ezga_cache_manifest.json"
_CACHE_SCHEMA_VERSION = 2

# EZGA relaxes candidates in the driver process, so on the CPU.
_EZGA_DEVICE = "cpu"
_CALCULATOR_SPEC_FILENAME = "calculator_spec.json"

# run_ezga's own signature defaults, mirrored here (rather than
# introspected via inspect.signature, which this codebase doesn't use
# elsewhere) so the modular structure_generator entry point (generate(),
# below) and the skeleton's pre-run config summary (committee_uncertainty_
# workflow.py's display_workflow_summary) have one obvious, named place to
# read ezga_kwargs' defaults from. Keep in sync with run_ezga's own
# defaults below if those ever change.
_EZGA_KWARGS_DEFAULTS: dict[str, Any] = {
    "max_generations": 2,
    "population_size": 2,
    "min_atoms": 2,
    "max_atoms": 41,
}


def validate_elements(elements: list[str]) -> list[str]:
    """Validate system elements and preserve their declared order."""
    if not isinstance(elements, list) or not elements:
        raise ValueError("elements must be a non-empty list of chemical symbols.")
    if any(
        not isinstance(e, str) or e not in atomic_numbers or e == "X" for e in elements
    ):
        raise ValueError(f"Invalid chemical elements: {elements}")
    if len(set(elements)) != len(elements):
        raise ValueError(f"Repeated chemical elements: {elements}")
    return list(elements)


def _cache_manifest(
    *,
    elements: list[str],
    seed_atoms: list[Atoms],
    calculator_spec: CalculatorSpec,
    ezga_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Describe the inputs that must match before reusing an EZGA cache."""
    effective_ezga_kwargs = {
        **_EZGA_KWARGS_DEFAULTS,
        **ezga_kwargs,
    }

    manifest = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "elements": list(elements),
        "seed_digests": sorted(geometry_digest(atoms) for atoms in seed_atoms),
        "calculator_spec": asdict(calculator_spec),
        "ezga_kwargs": effective_ezga_kwargs,
    }

    # Convert Path objects and any other JSON-compatible custom values
    # into a stable representation before writing or comparing manifests.
    return json.loads(
        json.dumps(
            manifest,
            sort_keys=True,
            default=str,
        )
    )


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_validated_candidate_file(
    candidates_path: Path,
    manifest: dict[str, Any],
) -> list[Atoms]:
    """Read cached candidates after checking file integrity and frame count."""
    metadata = manifest.get("candidate_file")

    if not isinstance(metadata, dict):
        raise ValueError("EZGA cache manifest has no candidate-file metadata.")

    expected_sha256 = metadata.get("sha256")
    expected_count = metadata.get("structure_count")

    if (
        not isinstance(expected_sha256, str)
        or not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ValueError("Invalid candidate-file metadata in EZGA cache manifest.")

    if _file_sha256(candidates_path) != expected_sha256:
        raise ValueError(
            f"EZGA candidate-file integrity check failed: {candidates_path}"
        )

    cached = read(candidates_path, index=":", format="extxyz")
    candidates = [cached] if isinstance(cached, Atoms) else list(cached)

    if len(candidates) != expected_count:
        raise ValueError(
            f"EZGA candidate count does not match the cache manifest: {candidates_path}"
        )

    return candidates


def filter_new_candidates(
    candidates: list[Atoms],
    seed_hashes: set[str],
    *,
    elements: list[str],
    min_atoms: int,
    max_atoms: int,
) -> tuple[list[Atoms], dict[str, int]]:
    """Exclude unchanged ordered seeds and repeated ordered geometries.

    This is not a symmetry-aware comparison or a global-database novelty test.
    """
    allowed = set(validate_elements(elements))
    seen = set(seed_hashes)
    kept: list[Atoms] = []
    summary = {
        "exported": len(candidates),
        "excluded_seeds": 0,
        "excluded_batch_duplicates": 0,
        "new_candidates": 0,
        "excluded_atom_count": 0,
        "excluded_elements": 0,
    }
    for atoms in candidates:
        if not min_atoms <= len(atoms) <= max_atoms:
            summary["excluded_atom_count"] += 1
            continue
        if set(atoms.get_chemical_symbols()) - allowed:
            summary["excluded_elements"] += 1
            continue

        key = geometry_digest(atoms)
        if key in seed_hashes:
            summary["excluded_seeds"] += 1
            continue
        if key in seen:
            summary["excluded_batch_duplicates"] += 1
            continue
        seen.add(key)
        candidate = atoms.copy()
        candidate.info["structure_generator"] = "ezga"
        kept.append(candidate)
    summary["new_candidates"] = len(kept)
    return kept, summary


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


def bounded_mutation_add(
    species: list[str],
    max_atoms: int,
    bound: list[str | int] | None = None,
    collision_tolerance: float = 2.0,
    slab: bool = False,
) -> Callable[[Any], Any | None]:
    """Build an EZGA add mutation that respects an upper atom-count limit."""
    from ezga.variation.mutation import mutation_add

    mutation = mutation_add(
        species=species,
        bound=bound,
        collision_tolerance=collision_tolerance,
        slab=slab,
    )

    def apply(structure: Any) -> Any | None:
        if structure.AtomPositionManager.atomCount >= max_atoms:
            return None

        result = mutation(structure)
        if result is None:
            return None

        candidate = result[0]
        if candidate is None:
            return None
        if candidate.AtomPositionManager.atomCount > max_atoms:
            return None

        return result

    return apply


def bounded_mutation_remove(
    species: str,
    min_atoms: int,
) -> Callable[[Any], Any | None]:
    """Build an EZGA remove mutation that respects a lower atom-count limit."""
    from ezga.variation.mutation import mutation_remove

    mutation = mutation_remove(species=species)

    def apply(structure: Any) -> Any | None:
        if structure.AtomPositionManager.atomCount <= min_atoms:
            return None

        result = mutation(structure)
        if result is None:
            return None

        candidate = result[0]
        if candidate is None:
            return None
        if candidate.AtomPositionManager.atomCount < min_atoms:
            return None

        return result

    return apply


def bounded_mutation_remove_add(
    species_add: list[str],
    species_remove: list[str],
    min_atoms: int,
    max_atoms: int,
    collision_tolerance: float = 2.0,
    slab: bool = False,
    **mutation_kwargs: Any,
) -> Callable[[Any], Any | None]:
    """Build an EZGA remove-add mutation respecting atom-count limits."""
    from ezga.variation.mutation import mutation_remove_add

    if "bound" in mutation_kwargs:
        raise ValueError(
            "EZGA remove_add.bound is not supported because mutation_remove_add "
            "manages its own anchor atoms internally."
        )

    mutation = mutation_remove_add(
        species_add=species_add,
        species_remove=species_remove,
        collision_tolerance=collision_tolerance,
        slab=slab,
        **mutation_kwargs,
    )

    def apply(structure: Any) -> Any | None:
        result = mutation(structure)
        if result is None:
            return None

        candidate = result[0]
        if candidate is None:
            return None

        atom_count = candidate.AtomPositionManager.atomCount
        if atom_count < min_atoms or atom_count > max_atoms:
            return None

        return result

    return apply


def build_mutation_configs(
    mutation_operators: dict[str, dict[str, Any]] | None,
    min_atoms: int,
    max_atoms: int,
    *,
    elements: list[str],
) -> list[dict[str, Any]]:
    """Translate ALomancy mutation settings into EZGA factory specifications."""
    elements = validate_elements(elements)
    defaults: dict[str, dict[str, Any]] = {
        "rattle": {"enabled": True, "std": 0.05, "species": list(elements)},
        "random_strain": {"enabled": True, "max_strain": 0.02},
        "add": {
            "enabled": True,
            "species": list(elements),
            "bound": list(elements),
            "collision_tolerance": 2.0,
            "slab": False,
        },
        "remove": {"enabled": True, "species": list(elements)},
        "remove_add": {
            "enabled": True,
            "species_add": list(elements),
            "species_remove": list(elements),
            "collision_tolerance": 2.0,
            "slab": False,
        },
    }
    supplied = mutation_operators or {}
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown EZGA mutation operators: {sorted(unknown)}")

    remove_add_options = supplied.get("remove_add", {})
    if "bound" in remove_add_options:
        raise ValueError(
            "EZGA remove_add.bound is not supported because mutation_remove_add "
            "manages its own anchor atoms internally."
        )

    settings = {
        name: {**default, **supplied.get(name, {})}
        for name, default in defaults.items()
    }
    for name, options in settings.items():
        if not options.get("enabled"):
            continue
        for key in ("species", "species_add", "species_remove", "bound"):
            value = options.get(key)
            if value is None:
                continue
            symbols = [value] if isinstance(value, str) else list(value)
            if key == "bound":
                invalid = any(
                    not (
                        (isinstance(item, str) and item in elements)
                        or (
                            isinstance(item, int)
                            and not isinstance(item, bool)
                            and item >= 0
                        )
                    )
                    for item in symbols
                )
            else:
                invalid = any(symbol not in elements for symbol in symbols)

            if not symbols or invalid:
                raise ValueError(
                    f"Invalid EZGA {name}.{key}: {symbols}. "
                    f"Allowed elements: {elements}. "
                    "bound also accepts non-negative atom indices."
                )

    mutations: list[dict[str, Any]] = []

    if settings["rattle"].pop("enabled"):
        mutations.append(
            {
                "type": "ezga.variation.mutation.mutation_rattle",
                **settings["rattle"],
            }
        )
    if settings["random_strain"].pop("enabled"):
        mutations.append(
            {
                "type": "ezga.variation.mutation.mutation_random_strain",
                **settings["random_strain"],
            }
        )
    if settings["add"].pop("enabled"):
        mutations.append(
            {
                "type": (
                    "alomancy.structure_generation.ezga.generate_structures."
                    "bounded_mutation_add"
                ),
                "max_atoms": max_atoms,
                **settings["add"],
            }
        )
    if settings["remove"].pop("enabled"):
        remove_settings = dict(settings["remove"])
        species = remove_settings.pop("species")
        symbols = [species] if isinstance(species, str) else list(species)

        # Keep bounded_mutation_remove's single-species interface. Each species
        # becomes one EZGA operator and therefore has its own selection weight.
        for symbol in symbols:
            mutations.append(
                {
                    "type": (
                        "alomancy.structure_generation.ezga.generate_structures."
                        "bounded_mutation_remove"
                    ),
                    "min_atoms": min_atoms,
                    "species": symbol,
                    **remove_settings,
                }
            )

    if settings["remove_add"].pop("enabled"):
        mutations.append(
            {
                "type": (
                    "alomancy.structure_generation.ezga.generate_structures."
                    "bounded_mutation_remove_add"
                ),
                "min_atoms": min_atoms,
                "max_atoms": max_atoms,
                **settings["remove_add"],
            }
        )

    if not mutations:
        raise ValueError("At least one EZGA mutation operator must be enabled.")
    return mutations


def trained_model_calculator(
    calculator_spec_path: str, device: str = _EZGA_DEVICE, **ezga_kwargs: Any
) -> Any:
    """EZGA calculator factory (referenced by string from the generated
    ezga_config.yaml): the trained model's own calculator, built by the
    trainer that trained it, wrapped in EZGA's generic ASE adapter.

    The spec is read from a JSON file because EZGA imports any config string
    containing ":" or a dotted name, so the trainer config can't be inlined."""
    from ezga.simulator.ase_calculator import ase_calculator

    spec = CalculatorSpec(**json.loads(Path(calculator_spec_path).read_text()))
    return ase_calculator(calculator=spec.build(device), device=device, **ezga_kwargs)


def build_ezga_config(
    dataset_path: Path,
    output_path: Path,
    calculator_spec_path: Path,
    max_generations: int = 2,
    population_size: int = 2,
    min_atoms: int = 2,
    max_atoms: int = 41,
    mutation_operators: dict[str, dict[str, Any]] | None = None,
    *,
    elements: list[str],
) -> dict:
    elements = validate_elements(elements)
    if max_generations < 1:
        raise ValueError("max_generations must be at least 1.")
    if population_size < 1:
        raise ValueError("population_size must be at least 1.")
    if min_atoms < 1:
        raise ValueError("min_atoms must be at least 1.")
    if max_atoms < min_atoms:
        raise ValueError("max_atoms must be greater than or equal to min_atoms.")

    return {
        "max_generations": max_generations,
        "resume": False,
        "output_path": str(output_path),
        "population": {
            "dataset_path": str(dataset_path),
            "db_path": str(output_path / "db"),
            "db_ro_path": str(output_path / "db_ro"),
            "filter_duplicates": True,
            "collision_factor": 0.80,
        },
        "multiobjective": {
            "size": population_size,
        },
        "variation": {
            "initial_mutation_rate": 1.0,
            "min_mutation_rate": 1.0,
            "crossover_probability": 0.0,
            "use_magnitude_scaling": False,
        },
        "mutation_funcs": build_mutation_configs(
            mutation_operators=mutation_operators,
            elements=elements,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        ),
        "crossover_funcs": [
            "ezga.variation.crossover.crossover_inplane_shuffle",
        ],
        "thermostat": {
            "initial_temperature": 1.0,
            "constant_temperature": True,
        },
        "evaluator": {
            "features_funcs": [
                {
                    "type": "ezga.evaluator.features.feature_composition_vector",
                    "IDs": list(elements),
                }
            ],
            "objectives_funcs": [
                {
                    "type": (
                        "alomancy.structure_generation.ezga."
                        "generate_structures.objective_energy_per_atom"
                    ),
                    "scale": 1.0,
                }
            ],
        },
        "simulator": {
            "mode": "sampling",
            "calculator": {
                "type": (
                    "alomancy.structure_generation.ezga."
                    "generate_structures.trained_model_calculator"
                ),
                "calculator_spec_path": str(calculator_spec_path),
                "device": _EZGA_DEVICE,
                "default_dtype": "float64",
                "nvt_steps": None,
                "fmax": 0.05,
                "steps_max": 5,
                "optimizer": "FIRE",
            },
        },
    }


def run_ezga(
    initial_structures: list[Atoms],
    calculator_spec: CalculatorSpec,
    output_dir: Path,
    max_generations: int = 2,
    population_size: int = 2,
    min_atoms: int = 2,
    max_atoms: int = 41,
    mutation_operators: dict[str, dict[str, Any]] | None = None,
    *,
    elements: list[str],
) -> list[Atoms]:

    elements = validate_elements(elements)
    if not initial_structures:
        raise ValueError("EZGA requires at least one initial structure.")
    allowed = set(elements)
    seed_species = {
        symbol
        for atoms in initial_structures
        for symbol in atoms.get_chemical_symbols()
    }
    if seed_species - allowed:
        raise ValueError(
            f"EZGA seeds contain unsupported elements: {sorted(seed_species - allowed)}"
        )
    seed_hashes = {geometry_digest(atoms) for atoms in initial_structures}

    invalid_sizes = [
        len(structure)
        for structure in initial_structures
        if not min_atoms <= len(structure) <= max_atoms
    ]
    if invalid_sizes:
        raise ValueError(
            "EZGA initial structures must contain between "
            f"{min_atoms} and {max_atoms} atoms; found invalid sizes "
            f"{sorted(set(invalid_sizes))}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Write ALomancy/ASE structures into an EZGA-readable EXTXYZ file
    # ------------------------------------------------------------------

    initial_population_path = output_dir / "initial_population.xyz"

    write(
        initial_population_path,
        initial_structures,
        format="extxyz",
    )

    # ------------------------------------------------------------------
    # 2. Build EZGA configuration
    # ------------------------------------------------------------------

    calculator_spec_path = output_dir / _CALCULATOR_SPEC_FILENAME
    calculator_spec_path.write_text(
        json.dumps(asdict(calculator_spec), indent=2, default=str)
    )
    config = build_ezga_config(
        elements=elements,
        dataset_path=initial_population_path,
        output_path=output_dir,
        calculator_spec_path=calculator_spec_path,
        max_generations=max_generations,
        population_size=population_size,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        mutation_operators=mutation_operators,
    )

    # ------------------------------------------------------------------
    # 3. Write YAML config
    #
    # EZGA's YAML loader performs the post-processing required to
    # materialize strings such as the calculator factory into real callables.
    # ------------------------------------------------------------------

    config_path = output_dir / "ezga_config.yaml"

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    # ------------------------------------------------------------------
    # 4. Build and run EZGA
    # ------------------------------------------------------------------

    cfg = load_config(config_path)

    engine = build_default_engine(cfg)

    population = engine.run()

    # Export the raw archive separately: it can contain the original seeds.
    raw_path = output_dir / "ezga_population.xyz"
    candidate_path = output_dir / _CANDIDATE_FILENAME
    if raw_path.exists():
        raw_path.unlink()
    population.export_structures(
        dataset=population.get_dataset("dataset"),
        file_path=str(raw_path),
    )
    exported = read(raw_path, index=":", format="extxyz")
    if isinstance(exported, Atoms):
        exported = [exported]
    candidates, summary = filter_new_candidates(
        list(exported),
        seed_hashes,
        elements=elements,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
    )
    (output_dir / "candidate_filter_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("EZGA candidate filtering: %s", summary)
    if not candidates:
        raise RuntimeError(
            "EZGA produced no new candidates after excluding seeds and duplicates."
        )
    # A restart must see a complete, filtered candidate file.
    temporary_path = output_dir / "ezga_new_candidates.tmp"
    write(temporary_path, candidates, format="extxyz")
    temporary_path.replace(candidate_path)
    return candidates


# ---------------------------------------------------------------------------
# Modular AL architecture: structure_generator registry entry points.
#
# EZGA runs entirely in the local driver process (no remote submission --
# see CLAUDE.md's note on this), so its orchestrator entry point is a thin
# wrapper: it uses the full eligible seed population directly (a
# population-based genetic search, unlike MD, needs no per-seed diversity
# selection via utils.seed_selection -- see the architecture plan's
# structure-generation decision). Its calculator is the trained model's
# own, built from the CalculatorSpec by trained_model_calculator.
# ---------------------------------------------------------------------------


def output_paths(
    config: dict,  # noqa: ARG001 -- common generator interface
    *,
    base_name: str,
    name: str,
) -> list[Path]:
    """Files required for a complete EZGA cached result."""
    output_dir = Path("results", base_name, name, "ezga")
    return [
        output_dir / _CANDIDATE_FILENAME,
        output_dir / _CACHE_MANIFEST_FILENAME,
    ]


def read_existing_result(
    config: dict,
    *,
    base_name: str,
    name: str,
) -> list[Atoms]:
    """Read a structurally complete EZGA cache.

    Full seed-, element-, and calculator-aware validation is performed by
    generate(), which has access to those runtime inputs.
    """
    ezga_kwargs = dict(config.get("ezga_kwargs", {}))
    if "elements" in ezga_kwargs:
        raise ValueError("Set elements in general.elements, not ezga_kwargs.")

    output_dir = Path("results", base_name, name, "ezga")
    candidates_path = output_dir / _CANDIDATE_FILENAME
    manifest_path = output_dir / _CACHE_MANIFEST_FILENAME

    if not candidates_path.exists() or not manifest_path.exists():
        raise ValueError(
            "No complete cached EZGA result at "
            f"{output_dir}; both {_CANDIDATE_FILENAME} and "
            f"{_CACHE_MANIFEST_FILENAME} are required."
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read EZGA cache manifest at {manifest_path}."
        ) from exc

    if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported EZGA cache manifest version at {manifest_path}.")

    effective_ezga_kwargs = {
        **_EZGA_KWARGS_DEFAULTS,
        **ezga_kwargs,
    }
    normalised_ezga_kwargs = json.loads(
        json.dumps(
            effective_ezga_kwargs,
            sort_keys=True,
            default=str,
        )
    )

    if manifest.get("ezga_kwargs") != normalised_ezga_kwargs:
        raise ValueError(
            "Existing EZGA cache was created with different ezga_kwargs: "
            f"{manifest_path}"
        )
    return _read_validated_candidate_file(candidates_path, manifest)


def generate(
    seed_atoms: list[Atoms],
    calculator_spec: CalculatorSpec,
    config: dict,
    *,
    elements: list[str],
    base_name: str,
    name: str,
    hpc: dict,  # noqa: ARG001 -- unused; common generator interface
    max_time: str,  # noqa: ARG001 -- unused; common generator interface
) -> list[Atoms]:
    """Run EZGA using the full eligible seed population.

    elements is supplied from general.elements. Generator-specific
    settings are supplied through config["ezga_kwargs"].
    """
    elements = validate_elements(elements)

    ezga_kwargs = dict(config.get("ezga_kwargs", {}))
    if "elements" in ezga_kwargs:
        raise ValueError("Set elements in general.elements, not ezga_kwargs.")

    expected_manifest = _cache_manifest(
        elements=elements,
        seed_atoms=seed_atoms,
        calculator_spec=calculator_spec,
        ezga_kwargs=ezga_kwargs,
    )

    output_dir = Path("results", base_name, name, "ezga")
    candidates_path = output_dir / _CANDIDATE_FILENAME
    manifest_path = output_dir / _CACHE_MANIFEST_FILENAME

    cache_files_exist = candidates_path.exists() or manifest_path.exists()

    if cache_files_exist:
        if not candidates_path.exists() or not manifest_path.exists():
            raise ValueError(
                "Incomplete EZGA cache: both "
                f"{candidates_path} and {manifest_path} are required."
            )

        try:
            cached_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read EZGA cache manifest at {manifest_path}."
            ) from exc

        cached_input_manifest = {
            key: value
            for key, value in cached_manifest.items()
            if key != "candidate_file"
        }

        if cached_input_manifest != expected_manifest:
            raise ValueError(
                "Existing EZGA cache is incompatible with the current "
                "elements, seeds, calculator, or ezga_kwargs. "
                f"Cache directory: {output_dir}"
            )

        cached_candidates = _read_validated_candidate_file(
            candidates_path,
            cached_manifest,
        )

        effective_kwargs = expected_manifest["ezga_kwargs"]
        seed_hashes = {geometry_digest(atoms) for atoms in seed_atoms}

        validated_candidates, summary = filter_new_candidates(
            cached_candidates,
            seed_hashes,
            elements=elements,
            min_atoms=int(effective_kwargs["min_atoms"]),
            max_atoms=int(effective_kwargs["max_atoms"]),
        )

        if len(validated_candidates) != len(cached_candidates):
            raise ValueError(
                "The EZGA cache contains seed structures or duplicate "
                f"candidates and cannot be reused: {candidates_path}"
            )

        if not validated_candidates:
            raise ValueError(
                f"The EZGA cache contains no usable candidates: {candidates_path}"
            )

        logger.info(
            "Reusing validated EZGA cache from %s: %s",
            candidates_path,
            summary,
        )
        return validated_candidates

    candidates = run_ezga(
        elements=elements,
        initial_structures=seed_atoms,
        calculator_spec=calculator_spec,
        output_dir=output_dir,
        **ezga_kwargs,
    )

    completed_manifest = {
        **expected_manifest,
        "candidate_file": {
            "sha256": _file_sha256(candidates_path),
            "structure_count": len(candidates),
        },
    }

    temporary_manifest_path = manifest_path.with_suffix(".tmp")
    temporary_manifest_path.write_text(
        json.dumps(
            completed_manifest,
            indent=2,
            sort_keys=True,
        )
    )
    temporary_manifest_path.replace(manifest_path)

    return candidates


def report_section(stats: dict, **kwargs: Any) -> Any:
    """Loop-report section for this module (see analysis/report/sections.py)."""
    from alomancy.analysis.report.sections import ezga_section

    return ezga_section(stats, **kwargs)
