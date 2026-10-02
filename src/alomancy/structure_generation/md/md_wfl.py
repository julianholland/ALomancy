import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.langevinbaoab import LangevinBAOAB
from ase.units import GPa, fs
from mace.calculators import MACECalculator

from alomancy.configs.remote_info import get_remote_info
from alomancy.registry import resolve
from alomancy.remote_submission.executor import submit_n
from alomancy.utils.dataset_curation import geometry_digest
from alomancy.utils.seed_selection import mark_structures_for_dft, select_diverse_seeds

logger = logging.getLogger(__name__)

_DEFAULT_NUM_OF_MD_STARTS = 10

# Every run records its starting structure before any MD step, so a run
# with fewer frames than this completed no MD step at all.
_MIN_FRAMES = 2
# The chosen MD seeds (plus any replacements, tagged info["replaces"]) for
# a loop, so restarts resume the same runs.
_SEEDS_FILENAME = "md_seeds.xyz"

# ALomancy's own defaults for the modular structure_generator entry point
# (generate(), below) -- deliberately different from run_md's own built-in
# defaults (steps=100), which are far too short for real production MD.
# Mirrors md_kwargs' actual runtime shape (including the two nested keys
# generate() pops out before spreading the rest onto run_md's own kwargs)
# so it can also be used, as-is, to display the fully-resolved effective
# config (see active_learning_workflow.py's display_workflow_summary).
_MD_KWARGS_DEFAULTS: dict[str, Any] = {
    "steps": 20000,
    "temperature": 300,
    "timestep_fs": 0.5,
    "traj_interval": None,
    "trainer": "mace",
    "trainer_config": {},
    "structure_selection_kwargs": {
        "num_of_md_starts": _DEFAULT_NUM_OF_MD_STARTS,
        "enforce_chemical_diversity": False,
        "seed": 803,
    },
}


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
    traj_interval: int | None = None,
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
    traj_interval : int, optional
        When set, every ``traj_interval``-th production MD step is appended
        to ``out_dir/md_trajectory.xyz`` (extxyz), a full-resolution record
        alongside the subsampled candidate snapshots. ``None`` (default)
        writes no trajectory file. Equilibration steps are never written.
    """
    if ensemble.lower() not in ("nvt", "npt"):
        raise ValueError(f"Unknown ensemble {ensemble!r}; must be 'nvt' or 'npt'.")
    if traj_interval is not None and (
        isinstance(traj_interval, bool)
        or not isinstance(traj_interval, int)
        or traj_interval <= 0
    ):
        raise ValueError(
            f"md_kwargs.traj_interval must be a positive integer or null, got "
            f"{traj_interval!r}."
        )

    assert structure_generation_job_dict["desired_num_of_structures"] > 0, (
        "Number of structures must be greater than 0"
    )
    assert (
        steps
        > structure_generation_job_dict["desired_num_of_structures"] / total_md_runs
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
    if traj_interval is not None:
        traj_path = Path(out_dir, "md_trajectory.xyz")
        last_written: dict[str, int | None] = {"step": None}

        def _write_traj_frame() -> None:
            # ASE calls observers again at the start of every dyn.run(), so
            # the step closing one snapshot segment would otherwise be
            # written twice.
            if dyn.nsteps == last_written["step"]:
                return
            last_written["step"] = dyn.nsteps
            write(str(traj_path), dyn.atoms, format="extxyz", append=True)

        dyn.attach(_write_traj_frame, interval=traj_interval)

    snapshot_interval = (
        steps
        * total_md_runs
        // (structure_generation_job_dict["desired_num_of_structures"] * 10)
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


# ---------------------------------------------------------------------------
# Modular AL architecture: structure_generator registry entry points.
#
# generate() is a local orchestrator (see the architecture plan's
# remote-submission-shape decision): called once by the skeleton with the
# full eligible seed population, it selects a diversity-maximizing subset
# via utils.seed_selection (population-side selection is MD's own business,
# unlike EZGA which uses the population directly) and fans out one remote
# MD job per selected seed via submit_n. Committee scoring of the
# candidates lives in the skeleton (ActiveLearningWorkflow.predict), not
# in this module.
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
    direct kwargs (steps, temperature, ensemble, pressure, ...) -- defaults
    to _MD_KWARGS_DEFAULTS (steps=20000, temperature=300, timestep_fs=0.5)
    rather than run_md's own far-shorter defaults, merged with whatever the
    config overrides -- plus two nested keys: structure_selection_kwargs
    (select_diverse_seeds' own params -- num_of_md_starts,
    defaulting here to 10, enforce_chemical_diversity, seed) and
    trainer/trainer_config (which trainer registry entry built the model
    this MD run's calculator should use). Both live under md_kwargs rather
    than the top structure_generation level since they're genuinely
    MD-specific -- EZGA never calls select_diverse_seeds or the trainer
    registry (it loads its model directly, always assuming MACE).
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
    trainer = md_kwargs.pop("trainer", _MD_KWARGS_DEFAULTS["trainer"])
    trainer_config = md_kwargs.pop(
        "trainer_config", _MD_KWARGS_DEFAULTS["trainer_config"]
    )
    # Only run_md's own flat kwargs (steps, temperature, ...) get
    # ALomancy's defaults merged in here (deliberately different from
    # run_md's own steps=100 default, far too short for real production
    # MD) -- structure_selection_kwargs/trainer/trainer_config above
    # already have their own dedicated defaults, resolved separately, and
    # must not leak back in via this merge: **md_kwargs is spread after
    # the explicit "trainer"/"trainer_config" function_kwargs below, so a
    # leaked-back default would silently clobber an explicit user value.
    _run_md_kwargs_defaults = {
        k: v
        for k, v in _MD_KWARGS_DEFAULTS.items()
        if k not in ("structure_selection_kwargs", "trainer", "trainer_config")
    }
    md_kwargs = {**_run_md_kwargs_defaults, **md_kwargs}
    md_dir = Path("results", base_name, "structure_generation")
    seeds_path = md_dir / _SEEDS_FILENAME
    target_file = f"{name}.xyz"
    selection_seed = selection_kwargs.get("seed", 803)

    # The chosen seeds are saved so a restart resumes the same MD runs
    # (run i always lives in md_output_i) instead of drawing new ones.
    if seeds_path.exists():
        seeds = list(read(seeds_path, ":", format="extxyz"))
    else:
        seeds = select_diverse_seeds(
            base_name=base_name,
            job_name=name,
            eligible_structures=seed_atoms,
            num_of_md_starts=selection_kwargs.get(
                "num_of_md_starts", _DEFAULT_NUM_OF_MD_STARTS
            ),
            enforce_chemical_diversity=selection_kwargs.get(
                "enforce_chemical_diversity", False
            ),
            seed=selection_seed,
        )
        md_dir.mkdir(parents=True, exist_ok=True)
        write(seeds_path, seeds, format="extxyz")
    # Replacement seeds don't change the sampling stride: run_md spaces its
    # snapshots by the number of MD runs originally planned.
    total_md_runs = sum(1 for a in seeds if "replaces" not in a.info)

    def out_dir(i: int) -> Path:
        return md_dir / f"md_output_{i}"

    def n_frames(i: int) -> int:
        path = out_dir(i) / target_file
        if not path.exists():
            return 0
        try:
            return len(read(path, ":", format="extxyz"))
        except Exception:
            return 0

    remote_info = None

    def submit(indices: list[int]) -> None:
        nonlocal remote_info
        if remote_info is None:
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
        # output_files is set explicitly per job (keyed by the run's own
        # index), since submit_n derives its positional job_id from the
        # position within job_configs, not from the run index.
        job_configs = [
            {
                "function_kwargs": {
                    "structure_generation_job_dict": structure_generation_job_dict,
                    "initial_structure": seeds[i],
                    "total_md_runs": total_md_runs,
                    "out_dir": str(out_dir(i)),
                    "model_path": model_path,
                    "trainer": trainer,
                    "trainer_config": trainer_config,
                    **md_kwargs,
                },
                "output_files": [str(out_dir(i))],
            }
            for i in indices
        ]
        submit_n(_run_md_via_trainer, job_configs, remote_info)

    # First pass: every run with no output directory yet (a fresh start, or
    # runs never submitted before an interruption).
    unsubmitted = [i for i in range(len(seeds)) if not out_dir(i).exists()]
    if unsubmitted:
        submit(unsubmitted)

    # One replacement round: a run that recorded only its starting
    # structure (or nothing) never completed an MD step, so its seed is
    # replaced once by a structure not used yet. Runs that did step are
    # kept, however short -- their frames are real candidates.
    replaced = {a.info["replaces"] for a in seeds if "replaces" in a.info}
    not_started = [
        i
        for i in range(len(seeds))
        if n_frames(i) < _MIN_FRAMES
        and "replaces" not in seeds[i].info
        and i not in replaced
    ]
    if not_started:
        logger.warning(
            "MD run(s) %s for %s completed no MD step; replacing their seeds once.",
            not_started,
            base_name,
        )
        replacements = _replacement_seeds(
            seed_atoms, seeds, len(not_started), seed=selection_seed + len(seeds)
        )
        new_indices = []
        for failed_index, replacement in zip(not_started, replacements, strict=True):
            replacement.info["replaces"] = failed_index
            seeds.append(replacement)
            new_indices.append(len(seeds) - 1)
        mark_structures_for_dft(replacements, base_name, name)
        write(seeds_path, seeds, format="extxyz")
        submit(new_indices)

    completed = [i for i in range(len(seeds)) if n_frames(i) >= _MIN_FRAMES]
    unfinished = [
        i
        for i in range(len(seeds))
        if i not in completed and not any(a.info.get("replaces") == i for a in seeds)
    ]
    if unfinished:
        logger.warning(
            "MD run(s) %s for %s still completed no MD step after replacement; "
            "continuing with the %d run(s) that did.",
            unfinished,
            base_name,
            len(completed),
        )
    if not completed:
        raise RuntimeError(
            f"No MD run for {base_name} completed a single MD step (seeds and "
            f"replacements all failed). Check the MD logs in {md_dir}/md_output_*. "
            "Nothing was cached, so a restart will retry."
        )

    structure_list: list[Atoms] = []
    for i in completed:
        structure_list.extend(read(out_dir(i) / target_file, ":", format="extxyz"))

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    write(candidates_path, structure_list, format="extxyz")
    return structure_list


def _replacement_seeds(
    eligible: list[Atoms], used: list[Atoms], n: int, *, seed: int
) -> list[Atoms]:
    """*n* new MD seeds, preferring eligible structures not used yet (by
    geometry). If too few are unused, fills up by reusing structures; every
    replacement gets a fresh md_seed so repeats still diverge."""
    used_digests = {geometry_digest(a) for a in used}
    unused = [a for a in eligible if geometry_digest(a) not in used_digests]
    rng = np.random.default_rng(seed)
    if len(unused) >= n:
        picks = [unused[i] for i in rng.choice(len(unused), size=n, replace=False)]
    else:
        pool = unused or eligible
        picks = unused + [
            pool[i] for i in rng.choice(len(pool), size=n - len(unused), replace=True)
        ]
    next_md_seed = (
        max((int(a.info.get("md_seed", seed)) for a in used), default=seed) + 1
    )
    replacements = []
    for k, atoms in enumerate(picks):
        copy = atoms.copy()
        copy.info["md_seed"] = next_md_seed + k
        replacements.append(copy)
    return replacements
