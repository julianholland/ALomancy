import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.langevinbaoab import LangevinBAOAB
from ase.units import GPa, fs
from mace.calculators import MACECalculator
from tqdm import tqdm

from alomancy.configs.remote_info import get_remote_info
from alomancy.registry import resolve
from alomancy.remote_submission.executor import submit_n
from alomancy.utils.seed_selection import select_diverse_seeds

logger = logging.getLogger(__name__)


def run_md(
    structure_generation_job_dict: dict,
    initial_structure: Atoms,
    total_md_runs: int,
    out_dir,
    model_path,
    steps=100,
    temperature=300,
    timestep_fs: float = 0.5,
    friction: float = 0.002,
    ensemble: str = "nvt",
    pressure: float = 0.0,
    equilibration_steps: int = 0,
    equilibration_temperature: float = 300.0,
    calculator=None,
):
    """
    ensemble : {"nvt", "npt"}
        "nvt" runs fixed-cell Langevin dynamics. "npt" runs LangevinBAOAB
        with a barostat targeting `pressure` (GPa); the cell is free to
        fluctuate in shape and volume.
    pressure : float
        Target external pressure in GPa, only used when ensemble="npt".
        ASE's externalstress is the negative of pressure (positive pressure
        compresses), so it is derived here as -pressure * ase.units.GPa.
    equilibration_steps : int
        Number of initial fixed-cell NVT Langevin steps that are not written to
        the production trajectory. Zero disables equilibration.
    equilibration_temperature : float
        Temperature in K used during the initial NVT equilibration.
    calculator : optional
        A pre-built calculator to drive the dynamics with, instead of the
        default hardcoded MACECalculator(model_paths=model_path, ...). Used
        by the modular structure-generator module (structure_generation.md's
        registry entry), which builds it via the trainer's own
        get_calculator(model_path, config) -- see the architecture plan's
        generator/calculator-coupling decision. None (the default) preserves
        this function's original behavior exactly for every existing caller.
    """
    if ensemble.lower() not in ("nvt", "npt"):
        raise ValueError(f"Unknown ensemble {ensemble!r}; must be 'nvt' or 'npt'.")

    assert structure_generation_job_dict["desired_number_of_structures"] > 0, (
        "Number of structures must be greater than 0"
    )
    assert (
        steps
        > structure_generation_job_dict["desired_number_of_structures"] / total_md_runs
    ), (
        "Number of steps must be greater than the number of structures divided by the number of intended MD runs"
    )
    # further asserting needed here to avoid:
    # for i in range(steps // snapshot_interval):
    #                ~~~~~~^^~~~~~~~~~~~~~~~~~~
    # ZeroDivisionError: integer division or modulo by zero

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    atom_traj_list = []

    md_structure = initial_structure.copy()
    md_structure.calc = (
        calculator
        if calculator is not None
        else MACECalculator(
            model_paths=model_path,
            device="cuda",
            default_dtype="float64",
        )
    )

    # md_seed is set by select_initial_structures when a structure is reused
    # across more than one concurrent job (not enough selectable structures
    # to give every job a unique starting point). Seeding Langevin's rng
    # per-job makes duplicate starting structures diverge into different
    # trajectories instead of running identical MD; without it, falls back
    # to ASE's own default (numpy's global RNG).
    md_seed = md_structure.info.get("md_seed")
    rng = np.random.default_rng(md_seed) if md_seed is not None else None

    logfile = str(
        Path(
            out_dir,
            f"{structure_generation_job_dict['name']}_{md_structure.info['job_id']}.log",
        )
    )

    if equilibration_steps < 0:
        raise ValueError("equilibration_steps must be non-negative.")
    if equilibration_temperature <= 0:
        raise ValueError("equilibration_temperature must be positive.")

    if equilibration_steps:
        logger.debug(
            "Equilibrating MD run %s for %d NVT steps at %g K.",
            structure_generation_job_dict["name"],
            equilibration_steps,
            equilibration_temperature,
        )
        equilibration = Langevin(
            atoms=md_structure,
            timestep=timestep_fs * fs,
            temperature_K=equilibration_temperature,
            friction=friction,
            rng=rng,
            logfile=logfile,
        )
        equilibration.run(steps=equilibration_steps)

    logger.debug(
        "MD run %s: ensemble=%s%s.",
        structure_generation_job_dict["name"],
        ensemble.upper(),
        f", target pressure={pressure} GPa" if ensemble.lower() == "npt" else "",
    )

    dyn: Langevin | LangevinBAOAB
    if ensemble.lower() == "nvt":
        dyn = Langevin(
            atoms=md_structure,
            timestep=timestep_fs * fs,
            temperature_K=temperature,
            friction=friction,
            rng=rng,
            logfile=logfile,
        )
    else:  # "npt", validated above
        dyn = LangevinBAOAB(
            atoms=md_structure,
            timestep=timestep_fs * fs,
            temperature_K=temperature,
            # externalstress is the negative of pressure (see ASE docstring);
            # must not be None, or LangevinBAOAB never activates the
            # barostat and silently runs fixed-cell dynamics instead of NPT.
            externalstress=-pressure * GPa,
            hydrostatic=False,
            rng=rng,
            logfile=logfile,
        )
    snapshot_interval = (
        steps
        * total_md_runs
        // (structure_generation_job_dict["desired_number_of_structures"] * 10)
    )

    for _ in range(steps // snapshot_interval):
        # recording -- happens before any force check/dynamics step below,
        # so whatever triggers a stop (excessive forces, non-finite forces,
        # or an outright exception) still leaves the structure that
        # triggered it captured in the trajectory. That structure -- right
        # at the edge of what the committee can currently describe -- is
        # exactly the kind active learning most needs; silently discarding
        # it (e.g. by letting an exception propagate and kill the whole
        # remote job with nothing recovered) would throw away the most
        # informative candidate this run produced.
        write(
            str(Path(out_dir, f"{structure_generation_job_dict['name']}.xyz")),
            dyn.atoms.copy(),
            append=True,
        )
        atom_traj_list.append(dyn.atoms.copy())

        # force check -- also catches non-finite (NaN/Inf) forces, which a
        # bare ">" comparison silently lets through (e.g. `np.nan > 1000`
        # is False), letting an unstable run keep going and pollute the
        # trajectory with garbage structures instead of stopping cleanly.
        try:
            max_forces = np.max(np.abs(dyn.atoms.get_forces()), axis=0)
            unstable = not np.all(np.isfinite(max_forces)) or np.any(max_forces > 1000)
        except Exception:
            logger.warning(
                "Stopping MD run %s: force evaluation raised an exception "
                "(likely a numerically unstable structure, e.g. a gap in "
                "the committee's PES). The structure just recorded is "
                "retained.",
                structure_generation_job_dict["name"],
                exc_info=True,
            )
            break
        if unstable:
            logger.warning(
                "Stopping MD run %s due to unstable forces: %s",
                structure_generation_job_dict["name"],
                max_forces,
            )
            break

        # run -- also guarded: an exception raised mid-integration (e.g.
        # the integrator numerically diverging between recorded snapshots)
        # must not discard the structures already written above.
        try:
            dyn.run(steps=snapshot_interval)
        except Exception:
            logger.warning(
                "Stopping MD run %s: dynamics step raised an exception "
                "mid-integration (likely numerical instability). "
                "Structures recorded so far are retained.",
                structure_generation_job_dict["name"],
                exc_info=True,
            )
            break

    logger.debug(
        f"MD run {structure_generation_job_dict['name']} completed, {len(atom_traj_list)} structures generated."
    )


def flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0] * 3))


def std_deviation_of_forces(
    structure_forces_dict: dict[str, dict[str, dict[str, np.ndarray]]],
    md_dir,
) -> pd.DataFrame:
    """
    Calculate the standard deviation of forces for each structure in the dictionary.

    Parameters
    ----------
    structure_force_dict : dict
        A dictionary where keys are fit names and values are dictionaries with structure names as keys and forces as values.

        e.g.:
        {
            'base_mace': {
                'structure_0': {'forces': np.ndarray, 'energy': float},
                'structure_1': {'forces': np.ndarray, 'energy': float},
                ...
            },
            'fit_1': {
                ...
            },
        }

    Returns
    -------
    list
        A list of standard deviations of forces for each structure.
    """
    number_of_structures = len(structure_forces_dict["base_mace"])
    std_dev_array = np.zeros((number_of_structures, 3))
    for structure in range(number_of_structures):
        forces_array = np.concatenate(
            [
                structure_forces_dict[fit][f"structure_{structure}"]["forces"]
                for fit in structure_forces_dict
            ],
            axis=0,
        )
        std_dev_per_force_fragment = np.std(forces_array, axis=0)
        energy_array = np.array(
            [
                structure_forces_dict[fit][f"structure_{structure}"]["energy"]
                for fit in structure_forces_dict
            ]
        )
        std_dev_per_energy = np.std(energy_array)

        logger.debug(
            f"Structure {structure}, max std dev: {np.max(std_dev_per_force_fragment)}, mean std dev: {np.mean(std_dev_per_force_fragment)}, std dev of energy: {std_dev_per_energy}, energies: {energy_array}"
        )

        std_dev_array[structure, :] = np.array(
            [
                np.max(std_dev_per_force_fragment),
                np.mean(std_dev_per_force_fragment),
                std_dev_per_energy,
            ]
        )

    df = pd.DataFrame(
        std_dev_array, columns=["max_std_dev", "mean_std_dev", "std_dev_energy"]
    ).sort_values(by="max_std_dev", ascending=False)

    df.to_csv(str(Path(md_dir, "std_dev_forces.csv")), index=True)

    return df


def get_forces_for_all_maces(
    structure_list: list[Atoms],
    base_name: str,
    job_dict: dict[str, dict[str, str]],
    base_mlip: str,
    fits_to_use: list[int] | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Get forces for all MACE models specified in fits_to_use.
    """

    if fits_to_use is None:
        fits_to_use = [0]

    logger.info(
        "MACE evaluation: scoring %d structure(s) against the base model plus "
        "%d committee fit(s) %s to compute per-structure force std dev.",
        len(structure_list),
        len(fits_to_use),
        fits_to_use,
    )

    calc = MACECalculator(model_paths=base_mlip, device="cuda", default_dtype="float64")

    for atoms in structure_list:
        atoms.calc = calc
    structure_forces_dict = {
        "base_mlip": {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(structure_list[i].get_forces()),
                "energy": np.array(structure_list[i].get_potential_energy()),
            }
            for i in range(len(structure_list))
        }
    }
    logger.info(
        "MACE evaluation: base model done (%d structures).", len(structure_list)
    )

    for i in fits_to_use:
        calc = MACECalculator(
            model_paths=str(
                Path(
                    "results",
                    base_name,
                    f"{job_dict['mlip_committee']['name']}/fit_{i}/{job_dict['mlip_committee']['name']}_stagetwo.model",
                )
            ),
            device="cuda",
            default_dtype="float64",
        )

        for atoms in tqdm(structure_list):
            atoms.calc = calc

        structure_forces_dict[f"fit_{i}"] = {
            f"structure_{i}": {
                "forces": flatten_array_of_forces(structure_list[i].get_forces()),
                "energy": structure_list[i].get_potential_energy(),
            }
            for i in range(len(structure_list))
        }
        logger.info(
            "MACE evaluation: fit_%d done (%d/%d committee fit(s) complete).",
            i,
            fits_to_use.index(i) + 1,
            len(fits_to_use),
        )

    logger.info(
        "MACE evaluation complete: forces collected from %d model(s) for %d "
        "structure(s).",
        1 + len(fits_to_use),
        len(structure_list),
    )

    return structure_forces_dict


# ---------------------------------------------------------------------------
# Modular AL architecture: structure_generator registry entry points.
#
# generate() is a local orchestrator (see the architecture plan's
# remote-submission-shape decision): called once by the skeleton with the
# full eligible seed population, it selects a diversity-maximizing subset
# via utils.seed_selection (population-side selection is MD's own business,
# unlike EZGA which uses the population directly) and fans out one remote
# MD job per selected seed via submit_n. get_forces_for_all_maces/
# all_maces_remote_submitter above are committee-scoring concerns that move
# to the skeleton, not part of this module's new interface.
# ---------------------------------------------------------------------------


def _run_md_via_trainer(
    structure_generation_job_dict: dict,
    initial_structure: Atoms,
    total_md_runs: int,
    out_dir: str,
    model_path: str,
    trainer: str,
    trainer_config: dict,
    **run_md_kwargs: Any,
) -> None:
    """Per-seed remote worker for generate() below. Builds its calculator
    via the named trainer's own get_calculator(model_path, trainer_config),
    resolved through the shared registry -- never a hardcoded MACECalculator
    -- then delegates to run_md's unchanged dynamics loop. Must only run
    inside a remote job: a live calculator must never cross the ExPyRe
    boundary (see the architecture plan's generator/calculator-coupling
    decision).
    """
    entry = resolve("mlip_trainer", trainer)
    calc = entry.get_calculator(model_path, trainer_config)
    run_md(
        structure_generation_job_dict=structure_generation_job_dict,
        initial_structure=initial_structure,
        total_md_runs=total_md_runs,
        out_dir=out_dir,
        model_path=model_path,
        calculator=calc,
        **run_md_kwargs,
    )


def _candidates_path(base_name: str, name: str) -> Path:
    return Path(
        "results", base_name, "structure_generation", f"{name}_generated_candidates.xyz"
    )


def output_paths(config: dict, *, base_name: str, name: str) -> list[Path]:  # noqa: ARG001
    """Coarse restart check, mirroring the DFT evaluator's pattern: the one
    consolidated candidates file generate() writes once every selected seed
    has produced a trajectory. Fine-grained, per-seed reuse (n_existing) is
    already handled internally by generate() itself.
    """
    return [_candidates_path(base_name, name)]


def read_existing_result(config: dict, *, base_name: str, name: str) -> list[Atoms]:  # noqa: ARG001
    path = _candidates_path(base_name, name)
    if not path.exists():
        raise ValueError(f"No cached structure-generation result at {path}.")
    return list(read(path, ":", format="extxyz"))


def generate(
    seed_atoms: list[Atoms],
    model_path: str,
    config: dict,
    *,
    base_name: str,
    name: str,
    hpc: dict,
    max_time: str,
) -> list[Atoms]:
    """Local orchestrator: select a diversity-maximizing subset of
    seed_atoms (via utils.seed_selection.select_diverse_seeds), then fan
    out one remote MD job per selected seed through the generic submit_n
    mechanism -- mirroring today's md_remote_submitter, generalized to
    resolve its calculator through the trainer registry instead of a
    hardcoded MACECalculator.

    config carries only generator-specific settings (md_kwargs);
    name/hpc/max_time are explicit kwargs. md_kwargs holds run_md's own
    direct kwargs (steps, temperature, ensemble, pressure, ...) plus two
    nested keys: structure_selection_kwargs (select_diverse_seeds' own
    params -- max_number_of_concurrent_jobs, enforce_chemical_diversity,
    seed) and trainer/trainer_config (which trainer registry entry built
    the model this MD run's calculator should use). Both live under
    md_kwargs rather than the top structure_generation level since they're
    genuinely MD-specific -- EZGA never calls select_diverse_seeds or the
    trainer registry (it loads its model directly, always assuming MACE).
    structure_generation's own top-level structure_selection_kwargs is
    unrelated and generator-agnostic (filter_eligible_structures, called
    once by the skeleton before any generator dispatch).
    """
    candidates_path = _candidates_path(base_name, name)
    if candidates_path.exists():
        logger.info(
            "Structure generation already done for %s/%s, loading cached candidates.",
            base_name,
            name,
        )
        return list(read(candidates_path, ":", format="extxyz"))

    md_kwargs = dict(config.get("md_kwargs", {}))
    selection_kwargs = md_kwargs.pop("structure_selection_kwargs", {})
    trainer = md_kwargs.pop("trainer", "mace")
    trainer_config = md_kwargs.pop("trainer_config", {})
    selected = select_diverse_seeds(
        base_name=base_name,
        job_name=name,
        eligible_structures=seed_atoms,
        max_number_of_concurrent_jobs=selection_kwargs.get(
            "max_number_of_concurrent_jobs", 5
        ),
        enforce_chemical_diversity=selection_kwargs.get(
            "enforce_chemical_diversity", False
        ),
        seed=selection_kwargs.get("seed", 803),
    )

    md_dir = Path("results", base_name, "structure_generation")
    target_file = f"{name}.xyz"

    def find_target_files() -> list[Path]:
        return list(md_dir.glob(f"md_output_*/{target_file}"))

    target_file_list = find_target_files()
    n_existing = len(target_file_list)

    if n_existing < len(selected):
        remaining = selected[n_existing:]

        remote_info = get_remote_info(
            {"hpc": hpc, "name": name, "max_time": max_time},
            input_files=[str(model_path)],
        )

        # run_md (unchanged, shared with the old production path) still
        # reads its own "name" out of structure_generation_job_dict rather
        # than taking it as an explicit kwarg -- config itself no longer
        # carries "name" (it's hardcoded by the skeleton, not user config),
        # so it's merged in here rather than changing run_md.
        structure_generation_job_dict = {**config, "name": name}

        # output_files is set explicitly per job (keyed by the real
        # n_existing + i directory name), matching md_remote_submitter's
        # own reasoning: submit_n/submit_multiple_jobs derives its
        # positional job_id from index within job_configs (0..len-1), which
        # only matches n_existing + i when n_existing == 0.
        job_configs = [
            {
                "function_kwargs": {
                    "structure_generation_job_dict": structure_generation_job_dict,
                    "initial_structure": atoms,
                    "total_md_runs": len(selected),
                    "out_dir": str(md_dir / f"md_output_{n_existing + i}"),
                    "model_path": model_path,
                    "trainer": trainer,
                    "trainer_config": trainer_config,
                    **md_kwargs,
                },
                "output_files": [str(md_dir / f"md_output_{n_existing + i}")],
            }
            for i, atoms in enumerate(remaining)
        ]
        submit_n(_run_md_via_trainer, job_configs, remote_info)
        target_file_list = find_target_files()

    structure_list: list[Atoms] = []
    for path in target_file_list:
        structure_list.extend(read(path, ":", format="extxyz"))

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    write(candidates_path, structure_list, format="extxyz")
    return structure_list
