from pathlib import Path

from ase import Atoms
from ase.io import write
from ezga.core.config import GAConfig
from ezga.factory import build_default_engine


def build_ezga_config(
    dataset_path: Path,
    output_path: Path,
    model_path: str,
) -> dict:
    return {
        "max_generations": 10,
        "resume": False,
        "output_path": str(output_path),

        "population": {
            "dataset_path": str(dataset_path),
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

        "thermostat": {
            "initial_temperature": 1.0,
            "constant_temperature": True,
        },

        "evaluator": {
            "features_funcs": [
                {
                    "type":
                    "ezga.evaluator.features.feature_average_interatomic_distance",
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
                "device": "cuda",
                "default_dtype": "float64",
                "nvt_steps": None,
                "fmax": 0.05,
                "steps_max": 100,
                "optimizer": "FIRE",
            },
        },
    }


def run_ezga(
    initial_structures: list[Atoms],
    model_path: str,
    output_dir: Path,
) -> list[Atoms]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_population_path = output_dir / "initial_population.xyz"

    write(
        initial_population_path,
        initial_structures,
        format="extxyz",
    )

    config = build_ezga_config(
        dataset_path=initial_population_path,
        output_path=output_dir,
        model_path=model_path,
    )

    cfg = GAConfig(**config)

    print(type(cfg))
    
    engine = build_default_engine(cfg)
    
    print(type(engine))
    
    return []