"""DFT evaluator module: the shared high-accuracy-evaluation orchestrator
implementing the modular AL architecture's evaluator entry points.

Extracted largely as-is from
``core.standard_active_learning.ActiveLearningStandardMACE.high_accuracy_evaluation``,
unchanged in behavior -- it was already skeleton-agnostic (batches
structures across GO/SP dispatch, bond-distance filtering, cell-wrapping,
restart via existing-batch reuse).

The evaluator category has two kinds of entry points:
- ``high_accuracy_evaluation`` (this module): the orchestrator, calculator-
  agnostic, receiving ``name``/``hpc``/``max_time``/``base_name`` explicitly.
  There is exactly one of these regardless of which DFT backend is
  configured -- it resolves the calculator-specific ``sp``/``go`` worker
  pair itself via the shared registry (``registry.resolve("dft_evaluator",
  calculator)``), the same way ``get_dft_functions`` already does today.
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
import logging
from datetime import datetime
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from alomancy.configs.remote_info import get_remote_info
from alomancy.registry import resolve
from alomancy.remote_submission.submitters import (
    ASE_OUTPUT_PREFIX,
    ase_remote_submitter,
)
from alomancy.utils.clean_structures import (
    filter_structures_by_min_bond_distance,
    wrap_structures_into_cell,
)
from alomancy.utils.dft_utils import refresh_dft_labels

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
        for a in read(sentinel_results, ":")
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
    start_index: int = 0,
) -> list[Atoms]:
    """Orchestrate DFT evaluation of ``structures``.

    Batches remaining structures into one shared ``RemoteJobExecutor`` pool
    (via ``ase_remote_submitter``, which itself uses the generic ``submit_n``
    mechanism), dispatching GO or SP per structure depending on
    ``needs_relaxation`` when ``allow_relaxation`` is set. Reuses whatever a
    prior partial run already completed (globbing existing batch
    directories) rather than an all-or-nothing restart. ``config`` carries
    only evaluator-specific settings (``evaluator``, ``qe_kwargs``/
    ``vasp_kwargs``, ``fmax``, ``relax_max_steps``, ``max_go_time``);
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

    eval_dir = Path("results", base_name, "high_accuracy_evaluation")
    if eval_dir.exists():
        found_structures = list(
            eval_dir.glob(f"batch_*/{ASE_OUTPUT_PREFIX}_*/{name}.xyz")
        )
        if len(found_structures) >= len(structures) + start_index:
            logger.info(
                "Found %d structures from previous high accuracy evaluation. "
                "Skipping remote submission and reusing these structures.",
                len(found_structures),
            )
            return [
                refresh_dft_labels(read(p, format="extxyz"), str(p))
                for p in found_structures
            ]
        elif found_structures:
            logger.info(
                "Found %d structures from previous high accuracy evaluation. "
                "These will be reused; the rest will be submitted as new remote jobs.",
                len(found_structures),
            )
            structures = structures[len(found_structures) + start_index :]
        else:
            logger.info(
                "No previous results found. Submitting all %d structures.",
                len(structures),
            )

    # Final safety net before any DFT is attempted.
    structures = filter_structures_by_min_bond_distance(structures)

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

    sentinel_results.parent.mkdir(parents=True, exist_ok=True)
    write(sentinel_results, high_accuracy_structures, format="extxyz")
    _mark_phase_done(base_name)
    return high_accuracy_structures
