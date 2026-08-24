from pathlib import Path

import yaml
from ase import Atoms
from ase.io import read, write
from ezga.factory import build_default_engine, load_config


def build_ezga_config(
    dataset_path: Path,
    output_path: Path,
    model_path: str,
) -> dict:
    return {
        "max_generations": 2,
        "resume": False,
        "output_path": str(output_path),

        "population": {
            "dataset_path": str(dataset_path),

            # Keep the EZGA database local to this run/output directory.
            # This avoids reusing population data from unrelated previous runs.
            "db_path": str(output_path / "db"),
            "db_ro_path": str(output_path / "db_ro"),

            "filter_duplicates": True,
            "collision_factor": 0.80,
        },

        "multiobjective": {
            "size": 2,
        },

        "variation": {
            "initial_mutation_rate": 1.0,
            "min_mutation_rate": 1.0,
            "crossover_probability": 0.0,
        },

        "mutation_funcs": [
            {
                "type": "ezga.variation.mutation.mutation_rattle",
                "std": 0.05,
                "species": ["Pd"],
            }
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
                    "type":
                    "ezga.evaluator.objective.objective_energy",
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
) -> list[Atoms]:

    if not initial_structures:
        raise ValueError("EZGA requires at least one initial structure.")

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