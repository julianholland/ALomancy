"""DFT evaluator module: the shared high-accuracy-evaluation orchestrator
implementing the modular AL architecture's evaluator entry points.

Originally extracted largely as-is from
``core.standard_active_learning.ActiveLearningStandardMACE.high_accuracy_evaluation``
(that module has since been removed along with the rest of the pre-1.0
workflow it belonged to), unchanged in behavior -- it was already
skeleton-agnostic (batches structures across GO/SP dispatch, bond-distance
filtering, cell-wrapping, restart via existing-batch reuse).

The evaluator category has two kinds of entry points:
- ``high_accuracy_evaluation`` (this module): the orchestrator, calculator-
  agnostic, receiving ``name``/``hpc``/``max_time``/``base_name`` explicitly.
  There is exactly one of these regardless of which DFT backend is
  configured -- it resolves the calculator-specific ``sp``/``go`` worker
  pair itself via the shared registry (``registry.resolve("dft_evaluator",
  calculator)``).
- the low-level ``sp``/``go`` worker functions
  (``high_accuracy_evaluation.dft.run_qe``/``run_vasp``), worker-shaped and
  calculator-specific, registered under the "dft_evaluator" category.
  Unchanged: they still receive a single config dict with ``name``/``hpc``
  embedded, exactly as today -- this orchestrator reconstructs that shape
  internally before calling them, so nothing downstream of dispatch needed
  to change.

``output_paths``/``read_existing_result`` here are intentionally coarse
(mirroring the orchestrator's own phase-done sentinel), not per-structure:
unlike a module with no restart logic of its own, this orchestrator already
does its own fine-grained partial-batch reuse internally (see
``high_accuracy_evaluation`` below) -- the skeleton-level restart mechanism
only needs to gate whether to call it at all.
"""

import copy
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms
from ase.io import read, write

from alomancy.configs.remote_info import get_remote_info
from alomancy.high_accuracy_evaluation.level_of_theory import (
    LevelOfTheory,
    stamp_level_of_theory,
)
from alomancy.registry import resolve
from alomancy.remote_submission.submitters import (
    ASE_OUTPUT_PREFIX,
    ase_remote_submitter,
)
from alomancy.utils.clean_structures import (
    filter_blown_up_structures,
    filter_structures_by_atom_count,
    filter_structures_by_min_bond_distance,
    resolve_max_vdw_volume_ratio,
    resolve_vdw_exempt_config_types,
    wrap_structures_into_cell,
)
from alomancy.utils.dft_utils import INITIAL_FRAME_SUFFIX, refresh_dft_labels
from alomancy.utils.file_saving_and_parsing import read_xyz

logger = logging.getLogger(__name__)

_PHASE = "high_accuracy_eval"

# The shared, unchanged run_sp/run_go workers (run_qe.py/run_vasp.py, also
# used by the old production path) read their own calculator-specific
# kwargs under legacy key names ("qe_input_kwargs"/"vasp_input_kwargs")
# that predate this refactor and can't be renamed without touching those
# shared functions. New-style config instead uses the standardized
# "<evaluator>_kwargs" naming (matching training.mace_kwargs,
# structure_generation.md_kwargs/ezga_kwargs) -- translated to the legacy
# key here, at the one place both naming schemes have to meet.
_LEGACY_EVALUATOR_KWARGS_KEY = {
    "qe": "qe_input_kwargs",
    "vasp": "vasp_input_kwargs",
}


def _sentinel_results_path(base_name: str) -> Path:
    return Path("results", base_name, "high_accuracy_eval_results.xyz")


# atoms.info key: dft_input_digest of the structure as it was sent to DFT.
# The workers keep atoms.info, so each result names the input it came from,
# and a restart knows exactly which structures are finished.
DFT_INPUT_DIGEST_KEY = "dft_input_digest"


def dft_input_digest(atoms: Atoms) -> str:
    """Fingerprint of a structure's geometry that survives an extxyz round
    trip: coordinates are hashed as extxyz writes them ("%.8f"), and
    formatting a value read back from that gives the same string. (A
    rounded-float hash such as dataset_curation.geometry_digest changes
    for most structures after one write and read.) A restart compares
    structures held in memory on the first attempt with ones read from
    file, so this has to hold."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(atoms.numbers).tobytes())
    for array in (atoms.positions, atoms.cell.array):
        h.update(" ".join(f"{x:.8f}" for x in array.ravel()).encode())
    h.update(np.ascontiguousarray(atoms.pbc).tobytes())
    return h.hexdigest()


def _still_to_evaluate(structures: list[Atoms], found: list[Path]) -> list[Atoms]:
    """The *structures* (stamped with DFT_INPUT_DIGEST_KEY) with no result
    among the *found* output files of an earlier attempt.

    Matched by digest, not position: jobs finish in any order, and a job
    that failed in the middle must be retried, not the last one. Results
    from before the digest was recorded fall back to the old positional
    rule (the first len(found) structures are done). Results matching no
    current structure mean the list changed since that attempt (filter or
    DFT settings edited between restarts): they are kept, with a warning
    (event dft_inputs_changed)."""
    if not found:
        return list(structures)
    digests = [
        cast(Atoms, read(p, index=0, format="extxyz")).info.get(DFT_INPUT_DIGEST_KEY)
        for p in found
    ]
    if any(d is None for d in digests):
        return list(structures[len(found) :])
    done = set(digests)
    current = {a.info[DFT_INPUT_DIGEST_KEY] for a in structures}
    unmatched = done - current
    if unmatched:
        logger.warning(
            "%d DFT result(s) from the previous attempt match none of the "
            "structures now going to DFT: the structures changed since then, "
            "most likely because filter or DFT settings were edited between "
            "restarts. That isn't supported: the results come from different "
            "settings. They are kept; only structures without a result are "
            "submitted. To redo the phase with the new settings, delete %s.",
            len(unmatched),
            found[0].parent.parent.parent,
            extra={
                "event": "dft_inputs_changed",
                "data": {"unmatched": len(unmatched), "found": len(found)},
            },
        )
    return [a for a in structures if a.info[DFT_INPUT_DIGEST_KEY] not in done]


def check_periodicity(
    structures: list[Atoms],
    evaluator: str,
    supports_periodic: bool,
    supports_non_periodic: bool,
) -> None:
    """Raise ValueError, before anything is submitted, when *structures*
    hold periodic structures (any pbc True) and the evaluator has no
    periodic boundary conditions (ORCA), or non-periodic ones and it
    needs them. Counts per config_type in the message."""
    for wanted, supported, test in (
        ("periodic", supports_periodic, lambda a: bool(a.pbc.any())),
        ("non-periodic", supports_non_periodic, lambda a: not a.pbc.any()),
    ):
        if supported:
            continue
        found: dict[str, int] = {}
        for atoms in structures:
            if test(atoms):
                ct = str(atoms.info.get("config_type", "unknown"))
                found[ct] = found.get(ct, 0) + 1
        if found:
            raise ValueError(
                f"The {evaluator} evaluator can't take {wanted} structures, but "
                f"{sum(found.values())} of the {len(structures)} structures "
                f"going to it are {wanted} (config_type: "
                + ", ".join(f"{k} {v}" for k, v in sorted(found.items()))
                + "). "
                + (
                    "Generate non-periodic structures for it: in "
                    "initialization use amorphous_cluster_kwargs and "
                    "molecule_cluster_kwargs instead of amorphous_kwargs, "
                    "mp_kwargs and molecules_in_box_kwargs, and seed structure "
                    "generation with non-periodic structures only."
                    if wanted == "periodic"
                    else ""
                )
            )


def initial_frames_path(base_name: str) -> Path:
    """Where the starting frames of this call's geometry optimisations are
    collected (step 0 of each GO that moved its structure)."""
    return Path("results", base_name, "high_accuracy_eval_initial_frames.xyz")


def _collect_initial_frames(
    eval_dir: Path, base_name: str, name: str, level: LevelOfTheory
) -> None:
    """Gather every job's ``<name>_initial.xyz`` (written by ``_run_go``)
    into ``initial_frames_path``, stamped with *level*. Jobs run before
    this existed, and SP or unmoved GO jobs, have none."""
    frames = [
        refresh_dft_labels(read(p, format="extxyz"), str(p))
        for p in sorted(
            eval_dir.glob(
                f"batch_*/{ASE_OUTPUT_PREFIX}_*/{name}{INITIAL_FRAME_SUFFIX}.xyz"
            )
        )
    ]
    stamp_level_of_theory(frames, level)
    path = initial_frames_path(base_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")
    if frames:
        logger.info(
            "Kept the starting frame of %d geometry optimisation(s) in %s.",
            len(frames),
            path,
        )


def read_initial_frames(base_name: str) -> list[Atoms]:
    """The GO starting frames collected for *base_name*; empty when none
    were kept (or the evaluation predates them)."""
    path = initial_frames_path(base_name)
    if not path.exists():
        return []
    return [refresh_dft_labels(a, str(path)) for a in read_xyz(path)]


def _phase_done(base_name: str) -> bool:
    return Path("results", base_name, f"{_PHASE}.done").exists()


def _mark_phase_done(base_name: str) -> None:
    sentinel = Path("results", base_name, f"{_PHASE}.done")
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(datetime.now().isoformat() + "\n")


def output_paths(config: dict, *, base_name: str, name: str) -> list[Path]:  # noqa: ARG001
    """Coarse restart check: the one consolidated results file the
    orchestrator itself writes once every structure is done. Fine-grained,
    per-structure reuse is already handled internally (see
    ``high_accuracy_evaluation``'s existing-batch globbing) -- this only
    needs to answer "did the whole call already complete."
    """
    return [_sentinel_results_path(base_name)]


def read_existing_result(config: dict, *, base_name: str, name: str) -> list[Atoms]:  # noqa: ARG001
    """Reconstruct the cached result from the consolidated results file."""
    sentinel_results = _sentinel_results_path(base_name)
    if not sentinel_results.exists():
        raise ValueError(
            f"No cached high-accuracy-evaluation result at {sentinel_results}."
        )
    return [
        refresh_dft_labels(a, str(sentinel_results))
        # Empty when the loop sent nothing to DFT (an all-redundant
        # selection); still a completed phase.
        for a in read_xyz(sentinel_results)
    ]


def high_accuracy_evaluation(
    structures: list[Atoms],
    config: dict,
    *,
    base_name: str,
    name: str,
    hpc: dict,
    max_time: str,
    allow_relaxation: bool = False,
) -> list[Atoms]:
    """Orchestrate DFT evaluation of ``structures``.

    Batches remaining structures into one shared ``RemoteJobExecutor`` pool
    (via ``ase_remote_submitter``, which itself uses the generic ``submit_n``
    mechanism), dispatching GO or SP per structure depending on
    ``needs_relaxation`` when ``allow_relaxation`` is set. Reuses whatever a
    prior partial run already completed (globbing existing batch
    directories) rather than an all-or-nothing restart. ``config`` carries
    only evaluator-specific settings (``evaluator``, ``qe_kwargs``/
    ``vasp_kwargs``, ``fmax``, ``max_num_of_relax_steps``, ``max_go_time``);
    ``name``/``hpc``/``max_time`` are explicit kwargs, reassembled into the
    single config dict the calculator-specific ``sp``/``go`` workers still
    expect (unchanged from today) -- including translating
    ``qe_kwargs``/``vasp_kwargs`` to the legacy key names those workers
    still read directly (see ``_LEGACY_EVALUATOR_KWARGS_KEY``).
    """
    worker_config = {**config, "name": name, "hpc": hpc, "max_time": max_time}

    sentinel_results = _sentinel_results_path(base_name)
    if _phase_done(base_name):
        logger.info(
            "%s already done for %s, loading cached results.", _PHASE, base_name
        )
        return read_existing_result(config, base_name=base_name, name=name)

    evaluator = worker_config.get("evaluator", "qe")
    entry = resolve("dft_evaluator", evaluator)
    run_sp, run_go = entry.sp, entry.go
    level = entry.level_of_theory(config, hpc)

    legacy_kwargs_key = _LEGACY_EVALUATOR_KWARGS_KEY.get(evaluator)
    new_kwargs_key = f"{evaluator}_kwargs"
    if legacy_kwargs_key and new_kwargs_key in worker_config:
        worker_config[legacy_kwargs_key] = worker_config.pop(new_kwargs_key)

    logger.debug(
        "Starting high accuracy evaluation with %d structures (evaluator=%s).",
        len(structures),
        evaluator,
    )

    function_kwargs = {"high_accuracy_eval_job_dict": worker_config}

    # Final safety net before any DFT is attempted. Applied before the
    # restart check below, which matches the outputs already found to the
    # filtered structures by digest (_still_to_evaluate).
    structures = filter_structures_by_min_bond_distance(structures)
    structures = filter_blown_up_structures(
        structures,
        resolve_max_vdw_volume_ratio(config),
        resolve_vdw_exempt_config_types(config),
    )
    check_periodicity(
        structures, evaluator, entry.supports_periodic, entry.supports_non_periodic
    )
    structures = filter_structures_by_atom_count(
        structures, entry.max_num_of_atoms(config), evaluator
    )

    for atoms in structures:
        atoms.info[DFT_INPUT_DIGEST_KEY] = dft_input_digest(atoms)
    eval_dir = Path("results", base_name, "high_accuracy_evaluation")
    if eval_dir.exists():
        found_structures = list(
            eval_dir.glob(f"batch_*/{ASE_OUTPUT_PREFIX}_*/{name}.xyz")
        )
        pending = _still_to_evaluate(structures, found_structures)
        if found_structures and not pending:
            logger.info(
                "Found %d structures from previous high accuracy evaluation. "
                "Skipping remote submission and reusing these structures.",
                len(found_structures),
            )
            _collect_initial_frames(eval_dir, base_name, name, level)
            return stamp_level_of_theory(
                [
                    refresh_dft_labels(read(p, format="extxyz"), str(p))
                    for p in found_structures
                ],
                level,
            )
        elif found_structures:
            logger.info(
                "Found %d structures from previous high accuracy evaluation. "
                "These will be reused; the other %d will be submitted as new "
                "remote jobs.",
                len(found_structures),
                len(pending),
            )
        else:
            logger.info(
                "No previous results found. Submitting all %d structures.",
                len(structures),
            )
        structures = pending

    # MD trajectories can leave atoms drifted outside the periodic cell --
    # QE/VASP expect coordinates within the reference cell. Applied only to
    # structures that already survived the bond-distance filter.
    structures = wrap_structures_into_cell(structures)

    current_batches = sum(1 for _ in eval_dir.glob("batch_*"))

    logger.info(
        "Structures to process: %d (existing batch dirs: %d)",
        len(structures),
        current_batches,
    )

    if structures:
        if allow_relaxation:
            needs_go = any(
                atom.info.get("needs_relaxation") is True for atom in structures
            )
            submit_config = worker_config
            if needs_go:
                go_max_time = worker_config.get(
                    "max_go_time", worker_config["max_time"]
                )
                submit_config = copy.deepcopy(worker_config)
                submit_config["max_time"] = go_max_time

            per_structure_function = [
                run_go if atom.info.get("needs_relaxation") is True else run_sp
                for atom in structures
            ]
            n_go = sum(fn is run_go for fn in per_structure_function)
            logger.info(
                "Submitting batch %d: %d GO + %d SP structures (shared queue)",
                current_batches,
                n_go,
                len(structures) - n_go,
            )
            ase_remote_submitter(
                remote_info=get_remote_info(submit_config, input_files=[]),
                base_name=base_name,
                input_atoms_list=structures,
                per_structure_function=per_structure_function,
                batch=current_batches,
                function_kwargs=function_kwargs,
                max_output_dir_len=entry.max_output_dir_len,
            )
        else:
            logger.info(
                "Submitting batch %d (%d structures)", current_batches, len(structures)
            )
            ase_remote_submitter(
                remote_info=get_remote_info(worker_config, input_files=[]),
                base_name=base_name,
                input_atoms_list=structures,
                function=run_sp,
                batch=current_batches,
                function_kwargs=function_kwargs,
                max_output_dir_len=entry.max_output_dir_len,
            )

    high_accuracy_structures = []
    directory_list = list(eval_dir.glob(f"batch_*/{ASE_OUTPUT_PREFIX}_*"))
    for directory in directory_list:
        completed_file = Path(directory, f"{name}.xyz")
        if completed_file.exists():
            high_accuracy_structures.append(
                refresh_dft_labels(
                    read(completed_file, format="extxyz"), str(completed_file)
                )
            )

    # Jobs were submitted but the phase has no results at all: a systematic
    # problem (bad calculator input, broken remote install), not bad luck.
    # Marking the phase done would permanently record zero new structures.
    if structures and not high_accuracy_structures:
        raise RuntimeError(
            f"All {len(structures)} high-accuracy evaluation jobs in "
            f"{eval_dir / f'batch_{current_batches}'} failed and no earlier "
            f"results exist; not marking {_PHASE} done. Check the job errors "
            "above and the calculator output files in that directory, fix "
            "the cause, then rerun."
        )

    stamp_level_of_theory(high_accuracy_structures, level)
    sentinel_results.parent.mkdir(parents=True, exist_ok=True)
    write(sentinel_results, high_accuracy_structures, format="extxyz")
    _collect_initial_frames(eval_dir, base_name, name, level)
    _mark_phase_done(base_name)
    return high_accuracy_structures
