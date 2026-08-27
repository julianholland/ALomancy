from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write
from ezga.factory import build_default_engine, load_config


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
        if not np.all(np.isfinite(energies)):
            raise ValueError("Cannot compute energy per atom from non-finite energies.")

        return scale * energies / atom_counts

    return compute


def bounded_mutation_add(
    species: list[str],
    max_atoms: int,
    bound: list[str] | None = None,
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
        candidate = mutation(structure)
        if (
            candidate is None
            or candidate.AtomPositionManager.atomCount > max_atoms
        ):
            return None
        return candidate

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
        candidate = mutation(structure)
        if (
            candidate is None
            or candidate.AtomPositionManager.atomCount < min_atoms
        ):
            return None
        return candidate

    return apply


def build_ezga_config(
    dataset_path: Path,
    output_path: Path,
    model_path: str,
    max_generations: int = 2,
    population_size: int = 2,
    min_atoms: int = 2,
    max_atoms: int = 41,
) -> dict:
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

        "mutation_funcs": [
            {
                "type": "ezga.variation.mutation.mutation_rattle",
                "std": 0.05,
                "species": ["Pd"],
            },
            {
                "type":
                    "ezga.variation.mutation.mutation_random_strain",
                "max_strain": 0.02,
            },
            {
                "type": (
                    "alomancy.structure_generation.ezga.generate_structures."
                    "bounded_mutation_add"
                ),
                "species": ["Pd"],
                "max_atoms": max_atoms,
                "bound": ["Pd"],
                "collision_tolerance": 2.0,
                "slab": True,
            },
            {
                "type": (
                    "alomancy.structure_generation.ezga.generate_structures."
                    "bounded_mutation_remove"
                ),
                "species": "Pd",
                "min_atoms": min_atoms,
            },
            {
                "type":
                    "ezga.variation.mutation.mutation_remove_add",
                "species_add": ["Pd"],
                "species_remove": ["Pd"],
                "bound": ["Pd"],
                "collision_tolerance": 2.0,
                "slab": True,
            },
        ],

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
                    "type":
                    "ezga.evaluator.features.feature_composition_vector",
                    "IDs": ["Pd"],
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
                "type":
                    "ezga.simulator.mace_calculator.mace_calculator",
                "calc_path": model_path,
                "device": "cpu",
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
    model_path: str,
    output_dir: Path,
    max_generations: int = 2,
    population_size: int = 2,
    min_atoms: int = 2,
    max_atoms: int = 41,
) -> list[Atoms]:

    if not initial_structures:
        raise ValueError("EZGA requires at least one initial structure.")
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

    config = build_ezga_config(
        dataset_path=initial_population_path,
        output_path=output_dir,
        model_path=model_path,
        max_generations=max_generations,
        population_size=population_size,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
    )

    # ------------------------------------------------------------------
    # 3. Write YAML config
    #
    # EZGA's YAML loader performs the post-processing required to
    # materialize strings such as the MACE calculator into real callables.
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

    # ------------------------------------------------------------------
    # 5. Export EZGA population
    # ------------------------------------------------------------------

    candidate_path = output_dir / "ezga_candidates.xyz"

    # EZGA export may append to an existing file.
    # Remove the old candidate file so we only return structures
    # belonging to the current run.
    if candidate_path.exists():
        candidate_path.unlink()

    population.export_structures(
        dataset=population.get_dataset("dataset"),
        file_path=str(candidate_path),
    )

    # ------------------------------------------------------------------
    # 6. Convert EZGA structures back to ASE
    # ------------------------------------------------------------------

    candidates = read(
        candidate_path,
        index=":",
        format="extxyz",
    )

    if isinstance(candidates, Atoms):
        candidates = [candidates]

    if len(candidates) == 0:
        raise RuntimeError("EZGA returned no candidate structures.")

    return list(candidates)
