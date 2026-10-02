"""ActiveLearningWorkflow: the generic parent of every AL workflow.

A concrete workflow (e.g. ``CommitteeUncertaintyWorkflow``,
``RandomSelectionWorkflow``) subclasses ``ActiveLearningWorkflow``,
declares its own settings as class attributes and writes a short ``run()``
that calls the parent's helpers in whatever order its strategy needs (see
docs/writing_a_workflow.md). Everything that is the same for every AL
strategy lives here:

- config loading and validation (``general``, ``general.dataset_kwargs``,
  ``general.start_from``, the train/test filters,
  ``high_accuracy_evaluation.force_ceiling``, renamed/removed-key errors);
- cold/warm start and resume (``prepare_run``), the loop itself
  (``iterate_loops``/``finish_loop``) and per-phase restart sentinels
  (the ``phase`` decorator);
- training N models on one shared train/valid/test split in one parallel
  remote batch (``train_models``), best-model tracking, prediction with a
  list of models (``predict``);
- candidate generation (``generate_candidates``), DFT evaluation
  (``high_accuracy_evaluate``) and adding labelled structures to the
  database (``add_to_dataset``).

Anything submitted to a remote machine (the trainer's ``train``,
``mlip/predict.predict_with_model``, the generator/evaluator workers) is a
module-level function, never a method: ExPyRe pickles functions by
reference and re-imports them on the remote node.

Config schema: a top-level `general` section holds `al_workflow` (the
registry name selecting which workflow class `build_workflow()` returns:
"committee_uncertainty" or "random_selection"), `elements`, run control
(`num_of_al_loops`, `start_loop`, `train_only`, `seed`, ...),
`dataset_kwargs` (how data is split: `test_ratio`, `target_config_types`,
`valid_fraction`, `valid_config_types`, `grouped_splits`,
`grouped_validation`, `fixed_test`) and, for workflows that have one, the
workflow's own `<al_workflow>_kwargs` block (e.g.
`committee_uncertainty_kwargs.num_of_models_in_committee`), matching the
`<dispatch_value>_kwargs` convention used everywhere else in this schema
(`mace_kwargs`, `md_kwargs`, ...). `mlip_committee` was renamed `training`
(usable by a non-committee workflow too) and has a `trainer` key
(defaults to "mace").

`general.elements` (list of atomic symbols, e.g. `["C", "O"]` -- not
atomic numbers) is the single shared source of element identity across
modules -- a direct sibling of `al_workflow`, not nested inside a
workflow's own kwargs block, since it's genuinely universal (any future
AL workflow would need it too, not just this one): passed as an explicit
`elements` kwarg to the initialiser (replacing the old `initialization.
creation_kwargs.elements`) and to the trainer (used there for an
E0s-coverage safety-net check, since MACE itself auto-detects the element
set from the training data but cannot infer E0s, a physical reference
value). `general.seed` is similarly universal, defaulting to 803 --
threaded down to the initialiser's `rattle_target_structures` step, and
used for every other random-selection point in the workflow (`self.seed`)
too.

`ActiveLearningWorkflow.__init__` takes only `jobs_dict` -- every
setting that used to be a separate Python constructor kwarg
(`num_of_al_loops`, `verbose`, `log_file`, `start_loop`, `plots`,
`seed`, `db_path`, `remove_redundancy`) now
lives as a direct child of `general` (see
`_GENERAL_KWARGS_DEFAULTS`), the same non-nested level as `al_workflow`/
`elements`, since none of these are specific to one AL workflow. Every key
falls back to its old constructor default. Where the run's first
structures come from is `general.start_from` (warm start from train/test
xyz files, a single xyz file, or a former run's database; cold start when
absent -- see `_parse_start_from` and docs/starting_a_run.md).
`general.train_filter`/`general.test_filter` exclude poor-quality DFT
structures from each split (utils/split_filter.py; the train filter's
`max_force` defaults to 100 eV/Angstrom, everything else is off), and
`high_accuracy_evaluation.force_ceiling` (default 100 eV/Angstrom, null =
single point) is how far AL-generated structures are relaxed before their
DFT labels are kept. The sole exception is `db`: a live `GlobalDatabase`
instance can't be a config value, so it's no longer constructor-settable
at all -- `self.db` is a lazily-constructed property (built from
`general.db_path` on first access), and a caller needing to inject an
already-built instance (mainly tests, to skip GlobalDatabase's real
construction cost) sets `wf.db = ...` after construction instead.

`structure_generation.method` is renamed `generator`, and
`high_accuracy_evaluation.calculator` is renamed `evaluator` -- matching
`training.trainer`'s existing naming pattern (each section names the
registry entry it dispatches to with a key matching what it selects).

No section takes a `name` key any more: each one's name is hardcoded to
match its own config section key (`_INITIALIZATION_NAME` etc., below) --
there is exactly one of each section per run, so a user-supplied name added
nothing but another place to typo (and `high_accuracy_evaluation`'s already
had to equal this literal anyway). Old shared functions that still read
`config["name"]` internally (`find_high_sd_structures`, `run_md`,
`check_quality_gate`, `select_best_committee_model`, the plotting
functions) get it merged into a shallow config copy at each call site
instead of being changed themselves.

`mace_kwargs.E0s` defaults to isolated-atom reference energies already
in the `GlobalDatabase` (`db.get_isolated_atom_energies()`, computed once
locally per training call and passed to the trainer as a plain dict) when
not set explicitly -- MACE cannot infer this physical value on its own, but
it's exactly what `IsolatedAtom` structures already generated/DFT-evaluated
by the initialiser provide. This is the first of what will grow into a
general config safety-net: mechanical per-module defaults, applied after
config overrides, catching conflicting/missing settings with a clear error
rather than a cryptic failure deep inside a remote job. Kept deliberately
inline per-module (not a centralized pre-flight registry entry point):
some checks would need to actually construct a calculator or invoke
software (GPU-bound MACE, QE/VASP binaries) that isn't available on the
local driver machine, so there's no way to validate everything before
remote submission without running it somewhere first anyway.

Per-module kwargs are named `<method>_kwargs` (`mace_kwargs`, `md_kwargs`,
`ezga_kwargs`, `qe_kwargs`, `vasp_kwargs`), matching the dispatch key each
section resolves against (`training.trainer`, `structure_generation.
generator`, `high_accuracy_evaluation.evaluator`). `qe_kwargs`/
`vasp_kwargs` are translated to the legacy `qe_input_kwargs`/
`vasp_input_kwargs` names at the evaluator orchestrator boundary (see
`high_accuracy_calc_interface.py`), since the shared, unchanged `run_sp`/
`run_go` workers still read those directly. Settings genuinely
generator-agnostic (`structure_generation.desired_num_of_structures`,
`structure_generation.structure_selection_kwargs` for the skeleton's own
`filter_eligible_structures` pre-filter, called once before any generator
dispatch) stay at the top `structure_generation` level rather than being
duplicated per-generator; MD-specific settings that would make no sense
for EZGA (`select_diverse_seeds`' own `structure_selection_kwargs` --
`num_of_md_starts`/`enforce_chemical_diversity`/`seed`) live
nested inside `md_kwargs` instead. `structure_generation.trainer`/
`trainer_config` (which trainer registry entry built the model MD's own
dynamics calculator should use) are the same story -- MD-only, since EZGA
loads its model directly rather than through the trainer registry -- so
they live nested inside `md_kwargs` too. `training.max_num_epochs`
likewise moves inside `mace_kwargs`: it's a MACE-specific training
control (not every trainer backend would necessarily have "epochs" at
all), kept at the top level only incidentally because MACE is the only
trainer today. Omitting `mace_kwargs.max_num_epochs` entirely resolves
dynamically (not a fixed number, and not MACE's own native default of
2048) -- the same as explicitly setting it to `"dynamic"`.

Other per-module defaults introduced alongside this: `structure_generation
.desired_num_of_structures` defaults to 50 when omitted (applied once
by the skeleton, so it's consistent regardless of which generator runs);
`md_kwargs` defaults to `steps=20000`/`temperature=300`/`timestep_fs=0.5`
(not `run_md`'s own far-shorter built-in defaults) and `md_kwargs.
structure_selection_kwargs.num_of_md_starts` defaults to 10;
`qe_kwargs`/`vasp_kwargs` need no explicit functional setting at all --
both `get_qe_input_data` and `get_vasp_input_kwargs` (old, shared,
unchanged) already default to PBE.

`initialization` is architecturally unlike the other three sections: it
has no dispatch key (trainer/generator/evaluator) because it isn't a
choice between interchangeable backends -- it's a single method that
always runs every structure-generating sub-task it's configured for, to
differing degrees. So its settings are namespaced per *structure type*
directly under `initialization` (no `creation_kwargs` wrapper -- nothing
else in this section needed the extra nesting level, unlike
training/structure_generation/high_accuracy_evaluation, which each hold
more than one kind of setting): `isolated_atom_kwargs`, `dimer_kwargs`,
`trimer_kwargs`, `amorphous_kwargs`, `mp_kwargs`,
`stretch_compress_targets_kwargs`, `rattle_target_structures` -- each
with its own `enabled` flag (default `True`, except `rattle_target_
structures` which defaults `False` as a new capability with no prior
behavior to preserve), so a sub-task can be toggled off without zeroing
out its count field. `stretch_compress_targets_kwargs`
(`max_lattice_deformation`, `num_of_stretch_compress_per_target`) and
`rattle_target_structures` (`rattle_standard_deviation`,
`num_of_rattled_per_target`) are each a sibling of `mp_kwargs`, not nested
inside it -- both apply to every structure this call generates whose
config_type is listed in `general.dataset_kwargs.target_config_types`, not just Materials Project ones. Designed to
extend cleanly as new sub-tasks are added (surfaces, interfaces): each
gets its own sibling `*_kwargs` namespace here and a matching branch in
`create_initialization_atoms_list`, without touching the others.
"""

import functools
import hashlib
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TypeVar, cast

import numpy as np
import polars as pl
from ase import Atoms
from ase.io import read, write

from alomancy.configs.hpc_profiles import format_table, hpc_profile_row
from alomancy.configs.remote_info import get_remote_info
from alomancy.database.global_database import (
    _DEFAULT_DEDUP_CONFIG_TYPES,
    GlobalDatabase,
)
from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
    high_accuracy_evaluation as _evaluator_orchestrate,
)
from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
    read_existing_result as _read_evaluated_structures,
)
from alomancy.mlip.evaluation import (
    best_fit_test_metrics,
    check_quality_gate,
    loop_metrics_frame,
    rank_committee,
)
from alomancy.mlip.mace.mace_wfl import read_mace_eval_predictions
from alomancy.mlip.predict import predict_with_model
from alomancy.registry import registered, resolve
from alomancy.remote_submission.executor import acquire_local_expyre_lock, submit_n
from alomancy.utils.clean_structures import (
    clean_structures,
    filter_structures_by_min_bond_distance,
)
from alomancy.utils.dataset_curation import (
    curate_database,
    geometry_digest,
    grouped_split,
    structure_domain,
    validate_policy,
)
from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled
from alomancy.utils.import_structures import (
    EXTERNAL_CONFIG_TYPE,
    file_sha256,
    normalize_metadata,
    read_structures,
)
from alomancy.utils.logging_config import setup_logging
from alomancy.utils.remote_ssh import (
    ensure_ssh_connectivity,
)
from alomancy.utils.remove_redundancy import remove_redundancy_from_partition
from alomancy.utils.seed_selection import filter_eligible_structures
from alomancy.utils.split_filter import (
    apply_split_filter,
    resolve_split_filter,
    validate_split_filter,
)
from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train
from alomancy.version import __version__, __version_tuple__

logger = logging.getLogger(__name__)

_PHASE_LABELS: dict[str, str] = {
    "initialization": "Initialisation",
    "training": "Model Trainer",
    "structure_generation": "Structure Generation",
    "high_accuracy_evaluation": "High-Accuracy Evaluation",
}

# Each module's "name" (used for result-directory naming, MACE run names,
# etc.) is hardcoded to match its config section key rather than being a
# separate config field -- there is exactly one of each section per run, so
# a user-supplied name added nothing but another place to typo (and
# high_accuracy_evaluation's already had to equal this literal anyway, see
# CLAUDE.md).
_INITIALIZATION_NAME = "initialization"
_TRAINING_NAME = "training"
_BEST_MODEL_DIR = "best_model"
_SUMMARY_HPC_COLUMNS = (
    "hpc_name",
    "alomancy_version",
    "gpu",
    "partitions",
    "ranks_per_node",
    "max_mem_per_node",
)
_BEST_MODEL_FILENAME = "ALomancy_best_model.model"
_STRUCTURE_GENERATION_NAME = "structure_generation"
_HIGH_ACCURACY_EVALUATION_NAME = "high_accuracy_evaluation"

# structure_generation.desired_num_of_structures is generator-agnostic
# (find_high_sd_structures' post-generation selection cap, and run_md's own
# trajectory-sampling stride -- both old/shared, both require this key with
# no default of their own). Defaulted once here, before generator dispatch,
# so the same value applies regardless of which generator module runs
# (EZGA doesn't read it today, but would get the same default too if a
# future version started to).
_DEFAULT_DESIRED_NUM_OF_STRUCTURES = 50

# general.dataset_kwargs: how the data is split, the same for every AL
# workflow. test_ratio and target_config_types have no default -- a
# silently guessed test/validation split policy is worse than a clear
# error at construction.
_DATASET_KWARGS_DEFAULTS: dict[str, Any] = {
    "valid_fraction": 0.05,
    "grouped_splits": False,
    "grouped_validation": False,
    "fixed_test": False,
}
_DATASET_KWARGS_REQUIRED = ("test_ratio", "target_config_types")
_DATASET_KWARGS_KEYS = frozenset(
    {*_DATASET_KWARGS_DEFAULTS, *_DATASET_KWARGS_REQUIRED, "valid_config_types"}
)

# general's own direct-child defaults, for every setting that used to be a
# workflow constructor kwarg -- these apply regardless of which AL
# workflow is chosen (matching elements/seed's precedent), so they're not
# nested inside a workflow's own kwargs block. The constructor
# now takes only jobs_dict; everything it used to accept as a Python kwarg
# is read from here instead. Where a run's starting data comes from is
# general.start_from (see _parse_start_from and docs/starting_a_run.md).
_GENERAL_KWARGS_DEFAULTS: dict[str, Any] = {
    "num_of_al_loops": 5,
    "verbose": 0,
    "log_file": "results/alomancy.log",
    "start_loop": 0,
    "plots": True,
    "seed": 803,
    "db_path": "results/global_database",
    "remove_redundancy": True,
    # Stop after the first loop's training (before any structure
    # generation) -- run control, honoured by every workflow's run().
    "train_only": False,
}

# high_accuracy_evaluation.force_ceiling default (eV/Angstrom): AL-generated
# structures are relaxed until their max force is at most this; null means
# single point. Formerly general.high_force_threshold.
_DEFAULT_FORCE_CEILING = 100.0

# Count-type keys renamed to the num_of_* convention. Old names are a hard
# error (checked in __init__), not silently ignored: a count quietly
# falling back to its default is exactly the kind of mistake that only
# shows up hours into a run. (section path, old key, new key)
_RENAMED_KEYS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("general",), "number_of_al_loops", "num_of_al_loops"),
    (
        ("general", "committee_uncertainty_kwargs"),
        "number_models_in_committee",
        "num_of_models_in_committee",
    ),
    (
        ("structure_generation",),
        "desired_number_of_structures",
        "desired_num_of_structures",
    ),
    (
        ("structure_generation", "structure_selection_kwargs"),
        "max_number_of_concurrent_jobs",
        "num_of_md_starts",
    ),
    (
        ("structure_generation", "md_kwargs", "structure_selection_kwargs"),
        "max_number_of_concurrent_jobs",
        "num_of_md_starts",
    ),
    (("high_accuracy_evaluation",), "relax_max_steps", "max_num_of_relax_steps"),
    (
        ("initialization", "dimer_kwargs"),
        "num_dimers_per_combo",
        "num_of_dimers_per_combo",
    ),
    (
        ("initialization", "trimer_kwargs"),
        "num_trimers_per_combo",
        "num_of_trimers_per_combo",
    ),
    (
        ("initialization", "amorphous_kwargs"),
        "num_amorphous",
        "num_of_amorphous_structures",
    ),
    (
        ("initialization", "amorphous_kwargs"),
        "amorphous_atom_number",
        "num_of_atoms_per_amorphous",
    ),
    (("initialization", "mp_kwargs"), "max_atom_number", "max_num_of_atoms"),
    (
        ("initialization", "stretch_compress_targets_kwargs"),
        "num_stretch_compress_per_target",
        "num_of_stretch_compress_per_target",
    ),
    (
        ("initialization", "rattle_target_structures"),
        "num_rattled_per_target",
        "num_of_rattled_per_target",
    ),
)


# Settings that moved to another section when the generic workflow
# parent was split out of the committee workflow: generic split settings
# now live in general.dataset_kwargs, run control directly under general.
# (old section path, old key, full new dotted path)
_MOVED_KEYS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    *(
        (
            ("general", "committee_uncertainty_kwargs"),
            key,
            f"general.dataset_kwargs.{key}",
        )
        for key in (
            "test_ratio",
            "target_config_types",
            "valid_fraction",
            "valid_config_types",
            "grouped_splits",
            "grouped_validation",
            "fixed_test",
        )
    ),
    (("general", "committee_uncertainty_kwargs"), "train_only", "general.train_only"),
)


def _section(jobs_dict: dict, path: tuple[str, ...]) -> Any:
    node: Any = jobs_dict
    for part in path:
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _find_renamed_keys(jobs_dict: dict) -> list[str]:
    """One "old -> new" entry per _RENAMED_KEYS or _MOVED_KEYS old key
    present in jobs_dict."""
    found = []
    for path, old, new in _RENAMED_KEYS:
        node = _section(jobs_dict, path)
        if isinstance(node, dict) and old in node:
            prefix = ".".join(path)
            found.append(f"{prefix}.{old} -> {prefix}.{new}")
    for path, old, new_path in _MOVED_KEYS:
        node = _section(jobs_dict, path)
        if isinstance(node, dict) and old in node:
            found.append(f"{'.'.join(path)}.{old} -> {new_path}")
    return found


# Start-up entry points folded into general.start_from. Old keys are a
# hard error naming the replacement. (section path, old key, replacement)
_REMOVED_KEYS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("general",), "initial_train_file_path", "general.start_from.train_xyz"),
    (("general",), "initial_test_file_path", "general.start_from.test_xyz"),
    (
        ("general",),
        "skip_initialization",
        "general.start_from (a warm start only generates the initialization "
        "structures the imported data doesn't already cover)",
    ),
    (("initialization",), "extra_datasets", "general.start_from.xyz"),
    (
        ("general",),
        "high_force_threshold",
        "high_accuracy_evaluation.force_ceiling (relaxing AL-generated "
        "structures before DFT) and general.train_filter.max_force (excluding "
        "high-force training structures)",
    ),
    (
        ("initialization",),
        "reset_extra_splits",
        "nothing (imported files always drop the old run's splits and flags)",
    ),
)


def _find_removed_keys(jobs_dict: dict) -> list[str]:
    """One "old -> replacement" entry per _REMOVED_KEYS key present."""
    found = []
    for path, old, replacement in _REMOVED_KEYS:
        node: Any = jobs_dict
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and old in node:
            found.append(f"{'.'.join(path)}.{old} -> {replacement}")
    return found


# general.start_from: where the run's first structures come from. Which
# source keys are present picks the mode (see docs/starting_a_run.md).
_START_FROM_KEYS = frozenset(
    {"train_xyz", "test_xyz", "xyz", "database", "metadata_map"}
)
START_MODE_SPLIT_FILES = "split_files"
START_MODE_SINGLE_XYZ = "single_xyz"
START_MODE_DATABASE = "database"
START_MODE_COLD = "cold"


def _parse_start_from(start_from: Any) -> tuple[str, dict]:
    """Validate general.start_from and return ``(mode, start_from)``.

    Modes: train_xyz + test_xyz -> split_files; xyz (a path or list of
    paths) -> single_xyz; database -> database; none -> cold. More than one
    source, or half of the train/test pair, raises ValueError.
    """
    if start_from is None:
        start_from = {}
    if not isinstance(start_from, dict):
        raise ValueError("general.start_from must be a mapping.")
    unknown = sorted(set(start_from) - _START_FROM_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown general.start_from key(s) {unknown}; allowed: "
            f"{sorted(_START_FROM_KEYS)}."
        )
    has_train, has_test = "train_xyz" in start_from, "test_xyz" in start_from
    if has_train != has_test:
        raise ValueError(
            "general.start_from.train_xyz and test_xyz must be given together "
            "(use start_from.xyz for a single file to be split)."
        )
    sources = [
        name
        for name, present in (
            ("train_xyz + test_xyz", has_train),
            ("xyz", "xyz" in start_from),
            ("database", "database" in start_from),
        )
        if present
    ]
    if len(sources) > 1:
        raise ValueError(
            f"general.start_from has more than one data source ({', '.join(sources)}); "
            "choose one."
        )
    if "metadata_map" in start_from and not (has_train or "xyz" in start_from):
        raise ValueError(
            "general.start_from.metadata_map only applies to xyz imports "
            "(train_xyz/test_xyz or xyz)."
        )
    if has_train:
        return START_MODE_SPLIT_FILES, start_from
    if "xyz" in start_from:
        xyz = start_from["xyz"]
        paths = [xyz] if isinstance(xyz, str) else xyz
        if not isinstance(paths, list) or not paths:
            raise ValueError(
                "general.start_from.xyz must be a path or a list of paths."
            )
        return START_MODE_SINGLE_XYZ, {**start_from, "xyz": paths}
    if "database" in start_from:
        return START_MODE_DATABASE, start_from
    return START_MODE_COLD, start_from


# general keys every workflow accepts; a workflow adds its own KWARGS_KEY.
_GENERAL_KNOWN_KEYS = set(_GENERAL_KWARGS_DEFAULTS) | {
    "al_workflow",
    "elements",
    "start_from",
    "train_filter",
    "test_filter",
    "dataset_kwargs",
}


def _needs_anything(needs: dict) -> bool:
    return bool(
        needs["isolated_atoms"]
        or needs["dimer_override"]
        or needs["trimer_override"]
        or needs["amorphous_override"] > 0
        or needs["mp_structures"]
    )


def _flatten_settings(
    d: dict, prefix: str = "", max_depth: int = 3
) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for key, value in d.items():
        if key in ("hpc", "name"):
            continue
        full_key = f"{prefix}{key}"
        if isinstance(value, dict) and max_depth > 0:
            items.extend(
                _flatten_settings(value, prefix=f"{full_key}.", max_depth=max_depth - 1)
            )
        else:
            items.append((full_key, value))
    return items


def _is_user_specified(raw_phase_dict: dict, dotted_key: str) -> bool:
    """Whether dotted_key (e.g. "mace_kwargs.max_num_epochs", as produced
    by _flatten_settings) was present verbatim in raw_phase_dict -- the
    config exactly as the user wrote it, before _resolve_effective_phase_
    dict merged in any per-module defaults for display. Nested dicts are
    walked key by key, matching get_qe_input_data's per-namelist merge:
    writing only qe_kwargs.system.input_dft leaves every other
    qe_kwargs.system.* key reported as a default. If the walk reaches a
    non-dict value before the key path ends, the user supplied that whole
    value, so every key beneath it counts as user-specified.
    """
    node: Any = raw_phase_dict
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            # Not a dict any more -> inside a value the user supplied
            # wholesale (see docstring), so every key beneath it counts as
            # user-specified. Still a dict but missing this key ->
            # genuinely not user-specified.
            return not isinstance(node, dict)
    return True


def _resolve_effective_phase_dict(phase: str, phase_dict: dict) -> dict:
    """Return phase_dict with its <method>_kwargs (or, for
    "initialization", each of its structure-type namespaces -- see
    initialiser_interface.py's module docstring; or, for "general",
    dataset_kwargs plus the selected workflow's own kwargs block -- see
    this module's own docstring)
    replaced by the fully defaults-merged version the corresponding module
    actually uses at runtime -- everything else in phase_dict passes
    through unchanged.

    Display-only (used by display_workflow_summary below): never mutates
    self.jobs_dict or feeds any module's real runtime call. Each module
    already independently merges its own defaults with its own user
    overrides at the point it actually reads config (trainer.py's
    mace_kwargs, md_wfl.py's md_kwargs, initialiser_interface.py's
    isolated_atom_kwargs/dimer_kwargs/..., ...) -- this reads each module's
    defaults back out via the shared registry (resolve(category, name).
    kwargs_defaults, or .resolve_effective_kwargs(...) for the two DFT
    evaluators, whose own merge is shallow-per-section rather than a flat
    dict) purely to mirror that same merge for the summary, so the two
    can't silently diverge beyond what's noted below.

    Known gaps, not attempted here: hpc/max_time defaults (see
    config_dictionaries.py) are resolved before the workflow object even
    exists, mutating jobs_dict in place -- by the time this runs, there's
    no way to tell whether a value already in jobs_dict was user-written
    or auto-filled, so those two keys are shown plain, never marked either
    way. structure_generation.structure_selection_kwargs (the top-level,
    generator-agnostic one feeding filter_eligible_structures, not
    generator_kwargs' own nested copy) is shown as given, un-defaulted.
    """
    effective = dict(phase_dict)
    if phase == "general":
        effective = {**_GENERAL_KWARGS_DEFAULTS, **effective}
        effective["dataset_kwargs"] = {
            **_DATASET_KWARGS_DEFAULTS,
            **effective.get("dataset_kwargs", {}),
        }
        al_workflow = effective.get("al_workflow", _DEFAULT_AL_WORKFLOW)
        try:
            workflow_cls = resolve("al_workflow", al_workflow).workflow_class
        except ValueError:
            workflow_cls = None  # reported by build_workflow, not here
        if workflow_cls is not None and workflow_cls.KWARGS_KEY:
            kwargs_key = workflow_cls.KWARGS_KEY
            effective[kwargs_key] = {
                **workflow_cls.KWARGS_DEFAULTS,
                **effective.get(kwargs_key, {}),
            }
    elif phase == "initialization":
        namespace_defaults = resolve("initialiser", "default").kwargs_defaults
        for namespace, defaults in namespace_defaults.items():
            effective[namespace] = {**defaults, **effective.get(namespace, {})}
    elif phase == "training":
        trainer_name = effective.get("trainer", "mace")
        kwargs_key = f"{trainer_name}_kwargs"
        defaults = resolve("mlip_trainer", trainer_name).kwargs_defaults
        merged = {**defaults, **effective.get(kwargs_key, {})}
        if trainer_name == "mace" and "E0s" not in merged:
            merged["E0s"] = "<resolved at train time from IsolatedAtom structures>"
        effective[kwargs_key] = merged
    elif phase == "structure_generation":
        generator = effective.get("generator", "md")
        kwargs_key = f"{generator}_kwargs"
        defaults = resolve("structure_generator", generator).kwargs_defaults
        effective[kwargs_key] = {**defaults, **effective.get(kwargs_key, {})}
        effective.setdefault(
            "desired_num_of_structures", _DEFAULT_DESIRED_NUM_OF_STRUCTURES
        )
    elif phase == "high_accuracy_evaluation":
        evaluator = effective.get("evaluator", "qe")
        kwargs_key = f"{evaluator}_kwargs"
        entry = resolve("dft_evaluator", evaluator)
        effective[kwargs_key] = entry.resolve_effective_kwargs(
            effective.get(kwargs_key, {})
        )
    return effective


def _collect_hpc_profiles(jobs_dict: dict) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for phase in _PHASE_LABELS:
        phase_dict = jobs_dict.get(phase)
        if not phase_dict:
            continue
        hpc = phase_dict.get("hpc")
        if not isinstance(hpc, dict):
            continue
        name = hpc.get("hpc_name", "<unnamed>")
        profiles.setdefault(name, hpc)
    return profiles


def _fetch_latest_pypi_version(
    package: str = "alomancy", timeout: float = 3.0
) -> str | None:
    if (
        os.getenv("ALOMANCY_TEST_MODE") == "1"
        or os.getenv("ALOMANCY_MOCK_EXTERNAL") == "1"
    ):
        return None
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{package}/json", timeout=timeout
        ) as response:
            data = json.loads(response.read())
        return data["info"]["version"]
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
    ) as exc:
        logger.debug("Could not check PyPI for latest %s version: %s", package, exc)
        return None


def _select_validation_split(
    all_training: list[Atoms],
    acceptable_configs: list[str],
    valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[Atoms], list[Atoms]]:
    """Carve the shared validation set from all_training. Only structures
    with config_type in acceptable_configs are eligible; the rest always
    stay in training. Returns (new_train_set, valid_set). Built ONCE per
    training call -- not re-derived per model -- and the same (train,
    valid) lists are then written to disk once and passed as file paths to
    every trainer.train() call.
    """
    eligible = [
        a for a in all_training if a.info.get("config_type") in acceptable_configs
    ]
    if not eligible:
        logger.warning(
            "No structures with config_type in %s found; skipping validation split.",
            acceptable_configs,
        )
        return all_training, []

    n_valid = int(np.floor(valid_fraction * len(eligible)))
    if n_valid == 0:
        logger.warning(
            "%.0f%% of %d eligible structure(s) rounds to 0; skipping validation split.",
            valid_fraction * 100,
            len(eligible),
        )
        return all_training, []

    chosen = rng.choice(len(eligible), size=n_valid, replace=False)
    valid_set = [eligible[i] for i in chosen]
    valid_ids = {id(a) for a in valid_set}
    new_train_set = [a for a in all_training if id(a) not in valid_ids]

    logger.info(
        "Validation split: %d valid, %d train (from %d total, %d eligible).",
        len(valid_set),
        len(new_train_set),
        len(all_training),
        len(eligible),
    )
    return new_train_set, valid_set


_DEFAULT_AL_WORKFLOW = "committee_uncertainty"


@dataclass
class TrainedModel:
    """One trained model: paths only (a live model never crosses the ExPyRe
    boundary). ``fit_idx`` is its index in the training call and names its
    directory (``results/<loop>/training/fit_<fit_idx>``)."""

    fit_idx: int
    seed: int
    model_path: str
    compiled_model_path: str | None
    metrics: dict
    fit_dir: Path


@dataclass
class LoopContext:
    """What one AL loop's steps need, built by ``iterate_loops``."""

    loop: int
    base_name: str
    workdir: Path
    train: list[Atoms]
    test: list[Atoms]
    train_only: bool
    plots_dir: Path


_StepT = TypeVar("_StepT", bound=Callable[..., Any])


def phase(
    name: str, load: Callable[..., Any] | None = None
) -> Callable[[_StepT], _StepT]:
    """Restart sentinel for one step of an AL loop.

    Wraps a workflow method whose first argument after ``self`` is a
    ``LoopContext``. If ``results/<loop>/<name>.done`` exists, the step is
    skipped and ``load(self, ctx, *args, **kwargs)`` returns its result
    from the files the step wrote; otherwise the step runs and the
    sentinel is written after it succeeds. ``load=None`` means a finished
    phase is simply re-run.

    Each phase runs at most once per loop: a second call with the same
    name raises, because both calls would share one sentinel. A step that
    genuinely runs twice per loop passes ``phase_name="..."`` to give the
    second call its own sentinel.
    """

    def decorator(fn: _StepT) -> _StepT:
        @functools.wraps(fn)
        def wrapper(
            self: "ActiveLearningWorkflow", ctx: LoopContext, *args: Any, **kwargs: Any
        ) -> Any:
            phase_name = kwargs.pop("phase_name", name)
            used = self._phases_run.setdefault(ctx.base_name, set())
            if phase_name in used:
                raise RuntimeError(
                    f"Phase {phase_name!r} already ran in {ctx.base_name}; a second "
                    "call would share its restart sentinel. Pass "
                    "phase_name='<unique name>' to the second call."
                )
            used.add(phase_name)
            if load is not None and self._phase_done(ctx.base_name, phase_name):
                logger.info(
                    "%s already done for %s; loading its results.",
                    phase_name,
                    ctx.base_name,
                )
                return load(self, ctx, *args, **kwargs)
            result = fn(self, ctx, *args, **kwargs)
            self._mark_phase_done(ctx.base_name, phase_name)
            return result

        return cast(_StepT, wrapper)

    return decorator


def _load_high_accuracy_results(
    self: "ActiveLearningWorkflow",
    ctx: LoopContext,
    *_args: Any,
    **_kwargs: Any,
) -> list[Atoms]:
    """high_accuracy_eval.done loader: the DFT-labelled structures the
    evaluator saved (high_accuracy_eval_results.xyz), relabelled exactly as
    the step does."""
    labelled = _read_evaluated_structures(
        self.jobs_dict["high_accuracy_evaluation"],
        base_name=ctx.base_name,
        name=_HIGH_ACCURACY_EVALUATION_NAME,
    )
    logger.info("High-accuracy evaluation completed for %d structures.", len(labelled))
    return self._label_new_structures(ctx, labelled)


def _load_trained_models(
    self: "ActiveLearningWorkflow",
    ctx: LoopContext,
    seeds: list[int],
    **_kwargs: Any,
) -> list[TrainedModel]:
    """train_mlip.done loader: re-read every fit that has a valid cached
    result (a fit lost after the sentinel was written is skipped)."""
    trainer_entry = resolve("mlip_trainer", self.training_config.get("trainer", "mace"))
    models = []
    for fit_idx, seed in enumerate(seeds):
        try:
            result = trainer_entry.read_existing_result(
                self.training_config,
                base_name=ctx.base_name,
                name=_TRAINING_NAME,
                fit_idx=fit_idx,
            )
        except (FileNotFoundError, ValueError):
            continue
        models.append(self._trained_model(ctx, fit_idx, seed, result))
    return models


class ActiveLearningWorkflow(ABC):
    """Generic AL workflow. Subclasses declare their settings as class
    attributes and implement run() from the helpers below (see
    docs/writing_a_workflow.md)."""

    #: Registry name (general.al_workflow).
    NAME: ClassVar[str] = ""
    #: general.<KWARGS_KEY> holds this workflow's own settings; None = none.
    KWARGS_KEY: ClassVar[str | None] = None
    #: Defaults for general.<KWARGS_KEY>; also its set of known keys.
    KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = {}
    #: config_type given to structures this workflow adds to the dataset.
    NEW_STRUCTURE_CONFIG_TYPE: ClassVar[str] = "al_generated"

    def __init__(self, jobs_dict: dict):
        self.jobs_dict = jobs_dict
        if jobs_dict.get("dataset_curation"):
            validate_policy(jobs_dict["dataset_curation"])

        general_config = jobs_dict.get("general", {})
        general_kwargs = {**_GENERAL_KWARGS_DEFAULTS, **general_config}

        # The config, not the class a caller happened to import, picks the
        # skeleton: constructing one class with a config naming another
        # would silently run the wrong algorithm.
        configured = general_config.get("al_workflow")
        if configured is not None and self.NAME and configured != self.NAME:
            raise ValueError(
                f"Config selects general.al_workflow={configured!r} but "
                f"{type(self).__name__} is {self.NAME!r}. Build the workflow "
                "from the config instead: `from alomancy import ALomancy; "
                "ALomancy(config).run()`."
            )

        # Every outdated key is reported in one error, so fixing a config
        # takes one pass rather than one restart per key.
        renamed = _find_renamed_keys(jobs_dict)
        removed = _find_removed_keys(jobs_dict)
        if renamed or removed:
            sections = []
            if renamed:
                sections.append(
                    "Config uses renamed or moved key(s); update them:\n  "
                    + "\n  ".join(renamed)
                )
            if removed:
                sections.append(
                    "Config uses removed key(s):\n  " + "\n  ".join(removed)
                )
            raise ValueError("\n".join(sections))
        self.start_mode, self.start_from = _parse_start_from(
            general_config.get("start_from")
        )
        # Checked here, not at first use: test_ratio is otherwise only read
        # after an AL loop's DFT has finished, hours into a run.
        dataset_config = general_config.get("dataset_kwargs") or {}
        missing = [k for k in _DATASET_KWARGS_REQUIRED if k not in dataset_config]
        if missing:
            raise ValueError(
                "general.dataset_kwargs is missing required "
                f"key(s) {missing} (e.g. test_ratio: 0.1, target_config_types: "
                "['init_MP', 'init_amorphous'])."
            )
        unknown_dataset = sorted(set(dataset_config) - _DATASET_KWARGS_KEYS)
        if unknown_dataset:
            raise ValueError(
                f"Unknown general.dataset_kwargs key(s) {unknown_dataset}; "
                f"allowed: {sorted(_DATASET_KWARGS_KEYS)}."
            )
        self.dataset_kwargs = {**_DATASET_KWARGS_DEFAULTS, **dataset_config}
        workflow_config = (
            general_config.get(self.KWARGS_KEY) or {} if self.KWARGS_KEY else {}
        )
        self.workflow_kwargs = {**self.KWARGS_DEFAULTS, **workflow_config}
        self.training_config = jobs_dict.get("training", {})
        self._phases_run: dict[str, set[str]] = {}
        self.num_of_al_loops = general_kwargs["num_of_al_loops"]
        self.verbose = general_kwargs["verbose"]
        self.start_loop = general_kwargs["start_loop"]
        self.plots = general_kwargs["plots"]
        self.seed = general_kwargs["seed"]
        self._db: GlobalDatabase | None = None
        self._db_path = general_kwargs["db_path"]
        self.remove_redundancy = general_kwargs["remove_redundancy"]
        self.train_only = bool(general_kwargs["train_only"])
        for filter_name in ("train_filter", "test_filter"):
            validate_split_filter(general_config.get(filter_name), filter_name)
        self.train_filter = resolve_split_filter(
            general_config.get("train_filter"), "train"
        )
        self.test_filter = resolve_split_filter(
            general_config.get("test_filter"), "test"
        )
        self.force_ceiling = jobs_dict.get("high_accuracy_evaluation", {}).get(
            "force_ceiling", _DEFAULT_FORCE_CEILING
        )
        if self.force_ceiling is not None and (
            isinstance(self.force_ceiling, bool)
            or not isinstance(self.force_ceiling, int | float)
            or not np.isfinite(self.force_ceiling)
            or self.force_ceiling <= 0
        ):
            raise ValueError(
                "high_accuracy_evaluation.force_ceiling must be a positive number "
                f"(eV/Angstrom) or null, got {self.force_ceiling!r}."
            )
        self.log_file = general_kwargs["log_file"]
        setup_logging(verbose=self.verbose, log_file=self.log_file)
        # After setup_logging so the warnings reach the console/log file.
        known = _GENERAL_KNOWN_KEYS | ({self.KWARGS_KEY} if self.KWARGS_KEY else set())
        unknown = sorted(set(general_config) - known)
        if unknown:
            logger.warning(
                "Ignoring unrecognised general key(s) %s -- check for typos. "
                "Known keys: %s",
                unknown,
                sorted(known),
            )
        unknown_workflow = sorted(set(workflow_config) - set(self.KWARGS_DEFAULTS))
        if unknown_workflow:
            logger.warning(
                "Ignoring unrecognised general.%s key(s) %s -- check for typos. "
                "Known keys: %s",
                self.KWARGS_KEY,
                unknown_workflow,
                sorted(self.KWARGS_DEFAULTS),
            )
        self.validate_settings()

    @property
    def db(self) -> GlobalDatabase:
        """Lazily constructed from general.db_path on first access -- never
        pays GlobalDatabase's real construction cost (observed ~1-3s) when
        a caller (mainly tests) overrides this with an already-built
        instance via the setter before ever reading it."""
        if self._db is None:
            self._db = GlobalDatabase(self._db_path)
        return self._db

    @db.setter
    def db(self, value: GlobalDatabase) -> None:
        self._db = value

    # -- Phase/loop bookkeeping --

    def _phase_done(self, base_name: str, phase: str) -> bool:
        return Path("results", base_name, f"{phase}.done").exists()

    def _mark_phase_done(self, base_name: str, phase: str) -> None:
        sentinel = Path("results", base_name, f"{phase}.done")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(datetime.now().isoformat() + "\n")
        logger.debug("Phase %s marked complete for %s.", phase, base_name)

    def _last_complete_loop(self) -> int:
        last = -1
        for loop in range(self.num_of_al_loops):
            if (Path("results", f"al_loop_{loop}") / "loop.done").exists():
                last = loop
            else:
                break
        return last

    def display_workflow_summary(self) -> None:
        lines: list[str] = [
            "",
            "=" * 70,
            f"ALomancy Workflow Summary (v{__version__})",
            "=" * 70,
        ]
        general_dict = self.jobs_dict.get("general")
        if general_dict is not None:
            lines.append("")
            lines.append("--- General ---")
            effective_general = _resolve_effective_phase_dict("general", general_dict)
            for key, value in _flatten_settings(effective_general):
                marker = (
                    "  [user-specified]"
                    if _is_user_specified(general_dict, key)
                    else ""
                )
                lines.append(f"  {key}: {value}{marker}")

        hpc_usage: dict[str, dict] = {}
        for phase, heading in _PHASE_LABELS.items():
            phase_dict = self.jobs_dict.get(phase)
            # Not `if not phase_dict`: an empty-but-present section (e.g.
            # "initialization: {}", now valid -- every one of its settings
            # defaults to enabled) must still show its resolved defaults,
            # not be silently skipped. Only a genuinely absent section key
            # (phase_dict is None) is skipped here.
            if phase_dict is None:
                continue
            lines.append("")
            lines.append(f"--- {heading} ({phase_dict.get('name', phase)}) ---")
            # Shows the fully-resolved effective config (per-module
            # defaults merged in, not just what's in self.jobs_dict) --
            # see _resolve_effective_phase_dict's docstring for exactly
            # what is and isn't covered. Values the user actually wrote
            # are marked explicitly; unmarked lines are values filled in
            # purely from a module's own defaults.
            effective_phase_dict = _resolve_effective_phase_dict(phase, phase_dict)
            for key, value in _flatten_settings(effective_phase_dict):
                # max_time is never marked either way (see
                # _resolve_effective_phase_dict's docstring): it's already
                # been defaulted-or-not by config_dictionaries.py, in
                # place, before this workflow object even existed, so
                # there's no way to tell here which one happened.
                marker = (
                    "  [user-specified]"
                    if key != "max_time" and _is_user_specified(phase_dict, key)
                    else ""
                )
                lines.append(f"  {key}: {value}{marker}")
            hpc = phase_dict.get("hpc")
            if hpc:
                name = (
                    hpc.get("hpc_name", "<unnamed>")
                    if isinstance(hpc, dict)
                    else str(hpc)
                )
                entry = hpc_usage.setdefault(
                    name,
                    {"profile": hpc if isinstance(hpc, dict) else {}, "phases": []},
                )
                entry["phases"].append(heading)

        lines.append("")
        lines.append("--- HPC Profiles ---")
        if hpc_usage:
            rows = []
            for name, entry in hpc_usage.items():
                # Same row builder as `alomancy list-hpc`, narrowed to the
                # columns that matter at run start.
                full = hpc_profile_row(name, entry["profile"], check_remote=True)
                rows.append(
                    {
                        **{k: full[k] for k in _SUMMARY_HPC_COLUMNS},
                        "job_types": "\n".join(entry["phases"]),
                    }
                )
            lines.append(format_table(rows))
        else:
            lines.append("  No HPC profiles configured.")
        lines.append("=" * 70)
        logger.info("\n".join(lines))

    def pre_run_checks(self) -> None:
        acquire_local_expyre_lock()
        self.display_workflow_summary()
        ensure_ssh_connectivity(_collect_hpc_profiles(self.jobs_dict))

        latest_version = _fetch_latest_pypi_version()
        if latest_version is None:
            return

        current_major, current_minor = __version_tuple__[0], __version_tuple__[1]
        try:
            latest_tuple = tuple(int(part) for part in latest_version.split(".")[:3])
        except ValueError:
            logger.debug(
                "Could not parse latest PyPI version %r; skipping version check.",
                latest_version,
            )
            return
        latest_major, latest_minor = latest_tuple[0], latest_tuple[1]

        if latest_major > current_major:
            raise RuntimeError(
                f"Installed alomancy version {__version__} is a major release "
                f"behind the latest available version {latest_version}. "
                "Breaking changes are likely — please upgrade "
                "(`pip install -U alomancy`) before running."
            )
        if latest_major == current_major and latest_minor > current_minor:
            logger.warning(
                "Installed alomancy version %s is a minor release behind the "
                "latest available version %s. Consider upgrading "
                "(`pip install -U alomancy`).",
                __version__,
                latest_version,
            )

    def _import_xyz(self, path: str, split: str | None = None) -> int:
        """Import one extxyz file into the global DB (start_from warm starts).

        Labels are normalized first (utils/import_structures.normalize_
        metadata, honouring start_from.metadata_map), which also drops the
        writing run's splits/flags. *split* tags every structure (train_xyz/
        test_xyz); None leaves them for the split rule. Idempotent: the
        file's sha256 is stored on each structure and a file already
        imported is skipped.
        """
        digest = file_sha256(path)
        if any(
            c.AtomPositionManager.metadata.get("source_dataset_sha256") == digest
            for c in self.db.partition.list_containers()
        ):
            logger.info("%s already imported (sha256=%s); skipping.", path, digest)
            return 0
        atoms_list = normalize_metadata(
            read_structures(path),
            self.start_from.get("metadata_map"),
            source=str(path),
        )
        for atoms in atoms_list:
            atoms.info["source_dataset_sha256"] = digest
            atoms.info.setdefault("domain", structure_domain(atoms))
            if split is not None:
                atoms.info["split"] = split
        added = int(self.db.add_structures(atoms_list, skip_duplicates=True))
        skipped = len(atoms_list) - added
        logger.info(
            "Imported %s: %d structure(s) added%s.",
            path,
            added,
            f", {skipped} duplicate(s) skipped" if skipped else "",
        )
        return added

    def _import_start_data(self) -> None:
        """Bring general.start_from's data into the global DB. Every mode
        then continues through the same DB-driven initialization."""
        if self.start_mode == START_MODE_SPLIT_FILES:
            self._import_xyz(self.start_from["train_xyz"], split="train")
            self._import_xyz(self.start_from["test_xyz"], split="test")
        elif self.start_mode == START_MODE_SINGLE_XYZ:
            for path in self.start_from["xyz"]:
                self._import_xyz(path)
        elif self.start_mode == START_MODE_DATABASE:
            self.db.import_from_database(self.start_from["database"])
        logger.info(
            "Start mode: %s (global DB now holds %d structures).",
            self.start_mode,
            self.db.size,
        )

    # -- Initialiser -> evaluator orchestration (decision 10) --

    def _initialize_training_set(
        self, base_name: str
    ) -> tuple[list[Atoms], list[Atoms]]:
        work_dir = Path("results", base_name)
        work_dir.mkdir(exist_ok=True, parents=True)
        init_config = self.jobs_dict["initialization"]
        general_config = self.jobs_dict.get("general", {})

        self._import_start_data()

        initialiser_entry = resolve("initialiser", "default")
        elements = general_config.get("elements")
        if not elements:
            raise ValueError(
                "general.elements is required (list of atomic symbols, e.g. "
                '["C", "O"]).'
            )

        if self.db.size > 0:
            logger.info(
                "Global DB has %d existing structures; reading those in first.",
                self.db.size,
            )

        needs = initialiser_entry.compute_needs(self.db, init_config, elements)

        if _needs_anything(needs):
            logger.info(
                "DB check: %d structure(s) already evaluated. Generating missing "
                "structures: %d isolated atoms, %d dimers, %d trimers, %d amorphous.",
                self.db.size,
                len(needs["isolated_atoms"]),
                sum(needs["dimer_override"].values()),
                sum(needs["trimer_override"].values()),
                needs["amorphous_override"],
            )

            generated_atoms_list = None
            if init_config.get("read_generated_file") is not None:
                generated_atoms_list = read_atoms_file_if_enabled(
                    True, work_dir / init_config["read_generated_file"]
                )
                if generated_atoms_list:
                    logger.info(
                        "Read %d pre-generated structures from file: %s",
                        len(generated_atoms_list),
                        init_config["read_generated_file"],
                    )

            if not generated_atoms_list:
                generated_atoms_list = initialiser_entry.generate(
                    init_config,
                    base_name=base_name,
                    name=_INITIALIZATION_NAME,
                    elements=elements,
                    hpc=init_config.get("hpc"),
                    max_time=init_config.get("max_time"),
                    needs=needs,
                    target_config_types=self.dataset_kwargs["target_config_types"],
                    seed=self.seed,
                )

            if not generated_atoms_list:
                raise ValueError(
                    "No structures were generated. Check initialization configuration."
                )

            high_accuracy_structures = _evaluator_orchestrate(
                generated_atoms_list,
                self.jobs_dict["high_accuracy_evaluation"],
                base_name=base_name,
                name=_HIGH_ACCURACY_EVALUATION_NAME,
                hpc=self.jobs_dict["high_accuracy_evaluation"]["hpc"],
                max_time=self.jobs_dict["high_accuracy_evaluation"]["max_time"],
                allow_relaxation=True,
                start_index=0,
            )

            if not high_accuracy_structures:
                raise ValueError(
                    "No high-accuracy structures returned. Check HPC configuration "
                    "and make sure remote jobs are running correctly."
                )

            logger.info(
                "config_type of first evaluated structure: %s",
                high_accuracy_structures[0].info.get("config_type"),
            )

            high_accuracy_structures = clean_structures(
                high_accuracy_structures,
                base_name,
                override_config_type=False,
                already_computed=True,
            )
            added = self.db.add_structures(
                high_accuracy_structures,
                skip_duplicates=True,
                config_types_to_dedup=_DEFAULT_DEDUP_CONFIG_TYPES,
            )
            logger.info("Added %d new structure(s) to the global database.", added)
        else:
            logger.info(
                "All initialization targets already met in global DB "
                "(%d structures). Skipping generation and DFT.",
                self.db.size,
            )

        all_evaluated = self.db.get_all_as_atoms()
        # Structures that already carry a split (train_xyz/test_xyz imports,
        # a former run's database) keep it; only untagged ones are split
        # here. Other tags (e.g. "diagnostic") stay out of both sets.
        pre_train = [a for a in all_evaluated if a.info.get("split") == "train"]
        pre_test = [a for a in all_evaluated if a.info.get("split") == "test"]
        untagged = [a for a in all_evaluated if not a.info.get("split")]
        if pre_train or pre_test:
            logger.info(
                "Keeping existing split tags: %d train / %d test; splitting %d "
                "untagged structure(s).",
                len(pre_train),
                len(pre_test),
                len(untagged),
            )
        new_train, new_test = self._split_untagged(untagged, self.dataset_kwargs)
        train_xyzs = pre_train + new_train
        test_xyzs = pre_test + new_test

        target_config_types = set(self.dataset_kwargs["target_config_types"])
        if (
            self.start_mode == START_MODE_SINGLE_XYZ
            and not test_xyzs
            and EXTERNAL_CONFIG_TYPE not in target_config_types
            and any(a.info.get("config_type") == EXTERNAL_CONFIG_TYPE for a in untagged)
        ):
            raise ValueError(
                "No test set could be formed from general.start_from.xyz: its "
                f"structures have no config_type and were labelled "
                f"{EXTERNAL_CONFIG_TYPE!r}, which is not in "
                "general.dataset_kwargs.target_config_types. "
                "Either give some structures a config_type (in the file, or map "
                "an existing key with general.start_from.metadata_map."
                "config_type) and list it in target_config_types, or add "
                f"{EXTERNAL_CONFIG_TYPE!r} to target_config_types explicitly."
            )

        write(work_dir / "train_set.xyz", train_xyzs, format="extxyz")
        write(work_dir / "test_set.xyz", test_xyzs, format="extxyz")

        config_types_in_train = {
            atoms.info["config_type"]
            for atoms in train_xyzs
            if "config_type" in atoms.info
        }
        logger.info("Config types in training set: %s", config_types_in_train)
        return train_xyzs, test_xyzs

    def _split_untagged(
        self, structures: list[Atoms], dataset_kwargs: dict
    ) -> tuple[list[Atoms], list[Atoms]]:
        """Split structures with no split tag: test_ratio of the
        target_config_types pool goes to test, everything else to train
        (grouped by split_group instead when grouped_splits is set)."""
        if not structures:
            return [], []
        if dataset_kwargs["grouped_splits"]:
            grouped: tuple[list[Atoms], list[Atoms]] = grouped_split(
                structures, dataset_kwargs["test_ratio"], self.seed
            )
            return grouped

        target_config_types = set(dataset_kwargs["target_config_types"])
        eligible_test_structures: list[Atoms] = []
        always_train_structures: list[Atoms] = []
        for atoms in structures:
            (
                eligible_test_structures
                if atoms.info.get("config_type") in target_config_types
                else always_train_structures
            ).append(atoms)

        if not eligible_test_structures:
            logger.warning(
                "No eligible test structures found for the specified "
                "target_config_types. All structures will be used for training."
            )
            return structures, []

        eligible_train, test_xyzs = split_atoms_list_into_test_and_train(
            eligible_test_structures,
            dataset_kwargs["test_ratio"],
            self.seed,
        )
        train_config_types = {a.info.get("config_type", "") for a in eligible_train}
        eligible_config_types = {
            a.info.get("config_type", "") for a in eligible_test_structures
        }
        missing_types = eligible_config_types - train_config_types
        if missing_types:
            for config_type in missing_types:
                idx = next(
                    i
                    for i, a in enumerate(test_xyzs)
                    if a.info.get("config_type", "") == config_type
                )
                eligible_train.append(test_xyzs.pop(idx))
            logger.warning(
                "Reserved one structure from each of %s for training to "
                "avoid entirely excluding these config_types from "
                "train_atoms_list.",
                sorted(missing_types),
            )
        return always_train_structures + eligible_train, test_xyzs

    def _apply_split_filters(self) -> None:
        """Re-flag train/test structures against general.train_filter /
        general.test_filter. Always runs (a disabled filter clears old
        flags), after redundancy removal and before the next loop reads the
        splits."""
        apply_split_filter(self.db, "train", self.train_filter)
        apply_split_filter(self.db, "test", self.test_filter)

    def _update_best_model(self, base_name: str, num_of_models: int) -> None:
        """Copy this loop's best model (chosen as for structure generation,
        see rank_committee) to results/best_model/ALomancy_best_model.model,
        replacing the previous loop's, with model_metadata.json alongside
        holding its errors per split and per config_type.

        The compiled model comes from the trainer's own read_existing_result
        (trainer-agnostic). Anything missing -- no evaluations, no compiled
        model -- logs a warning and leaves the previous best model in place;
        it never fails the AL loop.
        """
        training_config = self.jobs_dict["training"]
        committee_dir = Path("results", base_name, _TRAINING_NAME)
        fit_dirs = {
            i: committee_dir / f"fit_{i}"
            for i in range(num_of_models)
            if (committee_dir / f"fit_{i}" / "evaluation_metrics.json").exists()
        }
        if not fit_dirs:
            logger.warning(
                "No evaluated models in %s; best_model not updated.",
                committee_dir,
            )
            return
        try:
            best_fit, _, split_used = rank_committee(
                fit_dirs, label=f"best_model update for {base_name!r}"
            )
        except RuntimeError as exc:
            logger.warning("best_model not updated: %s", exc)
            return

        trainer_entry = resolve("mlip_trainer", training_config.get("trainer", "mace"))
        _, compiled_path, _ = trainer_entry.read_existing_result(
            training_config, base_name=base_name, name=_TRAINING_NAME, fit_idx=best_fit
        )
        if compiled_path is None:
            logger.warning(
                "fit_%d of %s has no compiled model; best_model not updated.",
                best_fit,
                base_name,
            )
            return

        best_dir = Path("results", _BEST_MODEL_DIR)
        best_dir.mkdir(parents=True, exist_ok=True)
        target = best_dir / _BEST_MODEL_FILENAME
        tmp = best_dir / f".{_BEST_MODEL_FILENAME}.tmp"
        shutil.copy2(compiled_path, tmp)
        os.replace(tmp, target)
        for stale in best_dir.glob("*.model"):
            if stale != target:
                stale.unlink()

        record = json.loads(
            (fit_dirs[best_fit] / "evaluation_metrics.json").read_text()
        )
        metric_keys = (
            "n_structures",
            "mae_e_per_atom",
            "rmse_e_per_atom",
            "mae_f",
            "rmse_f",
            "n_structures_with_stress",
            "mae_stress",
            "rmse_stress",
        )
        errors = {}
        for split, result in record.get("splits", {}).items():
            if not result.get("complete"):
                continue
            errors[split] = {k: result[k] for k in metric_keys if k in result}
            errors[split]["config_types"] = {
                config_type: {k: v for k, v in metrics.items() if k in metric_keys}
                for config_type, metrics in result.get("config_types", {}).items()
            }
        metadata = {
            "al_loop": int(base_name.rsplit("_", 1)[-1]),
            "fit_idx": best_fit,
            "selected_on_split": split_used,
            "source_model": str(compiled_path),
            "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "units": record.get("units", {}),
            "errors": errors,
        }
        metadata_path = best_dir / "model_metadata.json"
        tmp_metadata = best_dir / ".model_metadata.json.tmp"
        tmp_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(tmp_metadata, metadata_path)
        logger.info(
            "Best model for %s (fit_%d, chosen on %s) copied to %s.",
            base_name,
            best_fit,
            split_used,
            target,
        )

    def _cross_loop_metrics_dataframe(self, name: str) -> pl.DataFrame:
        """One row per AL loop (loop number in the "al_loop" column): the test-split
        mae_f/mae_e_per_atom of that loop's best committee member, chosen
        the same way as MD's base model (mlip/evaluation.py's
        best_fit_test_metrics). A loop whose evaluations are missing or
        inconsistent is skipped with a warning rather than failing the
        plot.
        """
        al_loop_dirs = sorted(
            Path("results").glob("al_loop_*"),
            key=lambda p: int(p.name.rsplit("_", 1)[1]),
        )
        rows = []
        loops = []
        for al_loop_dir in al_loop_dirs:
            try:
                row = best_fit_test_metrics(al_loop_dir / name)
            except (RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
                logger.warning(
                    "Skipping %s in the MAE-vs-loop metrics: %s", al_loop_dir.name, exc
                )
                continue
            if row is None:
                continue
            rows.append(row)
            loops.append(int(al_loop_dir.name.rsplit("_", 1)[1]))
        return loop_metrics_frame(loops, rows)

    def _store_predictions_and_cleanup(
        self, base_name: str, name: str, results: dict[int, tuple]
    ) -> None:
        """Store per-fit predictions in the GlobalDatabase and clean up
        local checkpoints/ directories -- local, post-sync, committee-shaped
        operations needing db/loop_idx/base_name, so skeleton-level rather
        than trainer-internal (decision 2/15)."""
        loop_idx = int(base_name.rsplit("_", 1)[-1]) if "al_loop_" in base_name else 0
        for fit_idx, (_model_path, compiled_model_path, _metrics) in results.items():
            fit_dir = Path("results", base_name, name, f"fit_{fit_idx}")
            preds = read_mace_eval_predictions(fit_dir)
            if preds:
                self.db.store_model_predictions(loop_idx, fit_idx, preds)
            if compiled_model_path is not None:
                checkpoints_dir = fit_dir / "checkpoints"
                if checkpoints_dir.exists():
                    shutil.rmtree(checkpoints_dir, ignore_errors=True)
                    logger.info(
                        "Removed local %s after successful fit.", checkpoints_dir
                    )

    # -- Helpers a workflow's run() calls --------------------------------------

    def seeds(self, n: int) -> list[int]:
        """``general.seed + i`` for i in range(n): the per-model seed
        mapping every run has used, so cached fits stay valid."""
        return [self.seed + i for i in range(n)]

    def prepare_run(self) -> int:
        """Everything before the first loop: curation-policy check, pre-run
        checks, then resume from the database or build the initial
        train/test set (cold/warm start, see general.start_from), followed
        by redundancy removal, the train/test filters and curation. Returns
        the first loop to run."""
        if self.jobs_dict.get("dataset_curation"):
            policy_path = Path("results/curation_policy.json")
            policy = json.dumps(
                self.jobs_dict["dataset_curation"], sort_keys=True, indent=2
            )
            if policy_path.exists() and policy_path.read_text() != policy:
                raise ValueError(
                    "Curation policy changed: use a new results directory to avoid stale checkpoints"
                )
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(policy)
        self.pre_run_checks()

        last_complete = self._last_complete_loop()
        if last_complete >= 0:
            effective_start = max(self.start_loop, last_complete + 1)
            logger.info(
                "Resuming from loop %d (%d train / %d test from DB).",
                effective_start,
                len(self.db.get_train_atoms()),
                len(self.db.get_test_atoms()),
            )
        else:
            train_xyzs, test_xyzs = self._initialize_training_set(_INITIALIZATION_NAME)
            n_tagged = self.db.update_splits_post_hoc(train_xyzs, test_xyzs)
            logger.info(
                "Initialized training set with %d structures; tagged %d in DB.",
                len(train_xyzs),
                n_tagged,
            )
            effective_start = self.start_loop

        self._curate_dataset()
        return int(effective_start)

    def _curate_dataset(self) -> None:
        """Redundancy removal, train/test quality filters and (if
        configured) dataset curation -- run before the first loop and at
        the end of every loop."""
        if self.remove_redundancy:
            remove_redundancy_from_partition(
                self.db,
                config_list=self.dataset_kwargs["target_config_types"]
                + [self.NEW_STRUCTURE_CONFIG_TYPE],
            )
        self._apply_split_filters()
        if self.jobs_dict.get("dataset_curation"):
            curate_database(self.db, self.jobs_dict["dataset_curation"])

    def iterate_loops(self, start: int) -> Iterator[LoopContext]:
        """Yield one LoopContext per loop from *start* to num_of_al_loops,
        reading the current train/test splits from the database and writing
        them to results/<loop>/train_set.xyz / test_set.xyz. A loop already
        marked done (loop.done) is skipped."""
        for loop in range(start, self.num_of_al_loops):
            base_name = f"al_loop_{loop}"
            if self._phase_done(base_name, "loop"):
                continue
            train_xyzs = self.db.get_train_atoms()
            test_xyzs = self.db.get_test_atoms()

            loop_plots_dir = Path("results", "current_plots", base_name)
            if self.plots:
                loop_plots_dir.mkdir(exist_ok=True, parents=True)
                from alomancy.analysis.bond_distance_plots import (
                    plot_training_bond_distances,
                )

                plot_training_bond_distances(base_name, self.db, loop_plots_dir)

            workdir = Path("results", base_name)
            try:
                workdir.mkdir(exist_ok=True, parents=True)
            except OSError as exc:
                logger.warning("Could not create directory %s: %s", workdir, exc)

            try:
                write(workdir / "train_set.xyz", train_xyzs, format="extxyz")
                write(workdir / "test_set.xyz", test_xyzs, format="extxyz")
            except OSError as exc:
                if "test" not in str(exc).lower():
                    raise
                logger.warning("Could not write files (test environment): %s", exc)

            logger.debug("Starting AL loop %d", loop)
            logger.debug("  Training set size: %d", len(train_xyzs))
            logger.debug("  Test set size: %d", len(test_xyzs))

            yield LoopContext(
                loop=loop,
                base_name=base_name,
                workdir=workdir,
                train=train_xyzs,
                test=test_xyzs,
                train_only=self.train_only,
                plots_dir=loop_plots_dir,
            )

    def train_models(
        self, ctx: LoopContext, seeds: list[int], *, min_successful: int = 1
    ) -> list[TrainedModel]:
        """Train one model per seed on this loop's shared train/valid/test
        split, as one parallel remote batch, then update
        results/best_model/. ``fit_idx`` is each seed's position, so
        ``seeds[i]`` trains ``training/fit_<i>``. Raises if fewer than
        ``min_successful`` models succeed."""
        models = self._train_models_phase(ctx, seeds, min_successful=min_successful)
        self._update_best_model(ctx.base_name, len(seeds))
        return models

    def _trained_model(
        self, ctx: LoopContext, fit_idx: int, seed: int, result: tuple
    ) -> TrainedModel:
        model_path, compiled_model_path, metrics = result
        return TrainedModel(
            fit_idx=fit_idx,
            seed=seed,
            model_path=str(model_path),
            compiled_model_path=compiled_model_path,
            metrics=metrics,
            fit_dir=Path("results", ctx.base_name, _TRAINING_NAME, f"fit_{fit_idx}"),
        )

    @phase("train_mlip", load=_load_trained_models)
    def _train_models_phase(
        self, ctx: LoopContext, seeds: list[int], *, min_successful: int = 1
    ) -> list[TrainedModel]:
        base_name = ctx.base_name
        training_config = self.training_config
        general_config = self.jobs_dict.get("general", {})
        dataset_kwargs = self.dataset_kwargs
        name = _TRAINING_NAME
        num_of_models = len(seeds)
        hpc = training_config["hpc"]
        max_time = training_config["max_time"]
        trainer_name = training_config.get("trainer", "mace")

        workdir = Path("results", base_name)

        trainer_entry = resolve("mlip_trainer", trainer_name)

        all_training = list(read(workdir / "train_set.xyz", ":", format="extxyz"))
        test_path = workdir / "test_set.xyz"

        valid_config_types = dataset_kwargs.get(
            "valid_config_types", dataset_kwargs.get("target_config_types", [])
        )
        acceptable_configs = [*valid_config_types, self.NEW_STRUCTURE_CONFIG_TYPE]
        valid_fraction = dataset_kwargs["valid_fraction"]
        rng = np.random.default_rng(self.seed)
        if dataset_kwargs["grouped_validation"]:
            new_train_set, valid_set = grouped_split(
                all_training, valid_fraction, self.seed
            )
        else:
            new_train_set, valid_set = _select_validation_split(
                all_training, acceptable_configs, valid_fraction, rng
            )

        train_path = workdir / "split_train.xyz"
        write(train_path, new_train_set, format="extxyz")
        valid_path_str: str | None = None
        if valid_set:
            valid_path = workdir / "split_valid.xyz"
            write(valid_path, valid_set, format="extxyz")
            valid_path_str = str(valid_path)

        results: dict[int, tuple] = {}
        for fit_idx in range(num_of_models):
            paths = trainer_entry.output_paths(
                training_config, base_name=base_name, name=name, fit_idx=fit_idx
            )
            if not all(p.exists() for p in paths):
                continue
            cached = self._collect_fit(trainer_entry, ctx, fit_idx)
            if cached is None:
                logger.warning(
                    "fit_%d's cached result failed validation; retraining.", fit_idx
                )
                continue
            results[fit_idx] = cached
            self._check_recorded_seed(ctx, fit_idx, seeds[fit_idx])

        missing = [i for i in range(num_of_models) if i not in results]
        if missing:
            logger.info(
                "train_mlip for %s: %d/%d fit(s) already cached; submitting %s.",
                base_name,
                num_of_models - len(missing),
                num_of_models,
                missing,
            )
            input_files = [str(train_path), str(test_path)]
            if valid_path_str:
                input_files.append(valid_path_str)
            remote_info = get_remote_info(
                {"hpc": hpc, "name": name, "max_time": max_time},
                input_files=input_files,
            )
            # Computed once, locally, and passed down as a plain dict
            # (not a live GlobalDatabase, which must never cross the ExPyRe
            # boundary) so the trainer can default its reference energies
            # (e.g. MACE's E0s) when the config doesn't set them.
            isolated_atom_e0s = self.db.get_isolated_atom_energies()

            def submit(fit_indices: list[int]) -> None:
                job_configs = [
                    {
                        "function_kwargs": {
                            "train_atoms_path": str(train_path),
                            "valid_atoms_path": valid_path_str,
                            "test_atoms_path": str(test_path),
                            "config": training_config,
                            "fit_seed": seeds[fit_idx],
                            "base_name": base_name,
                            "name": name,
                            "fit_idx": fit_idx,
                            "hpc": hpc,
                            "max_time": max_time,
                            "elements": general_config.get("elements"),
                            "isolated_atom_e0s": isolated_atom_e0s,
                        },
                        "output_files": [str(workdir / name / f"fit_{fit_idx}")],
                    }
                    for fit_idx in fit_indices
                ]
                submitted = submit_n(trainer_entry.train, job_configs, remote_info)
                for position, fit_idx in enumerate(fit_indices):
                    if submitted[position] is None:
                        continue
                    # A job that returned is only a success once its outputs
                    # (for MACE: the model and evaluation_metrics.json) are
                    # present and valid -- the same test a restart applies.
                    collected = self._collect_fit(trainer_entry, ctx, fit_idx)
                    if collected is None:
                        logger.warning(
                            "fit_%d of %s finished but was not evaluated "
                            "(outputs missing or invalid); counting it as failed.",
                            fit_idx,
                            base_name,
                        )
                        continue
                    results[fit_idx] = collected
                    self._record_seed(ctx, fit_idx, seeds[fit_idx])

            submit(missing)
            # One retry pass for every fit still missing: a single transient
            # failure (node crash, sync error) must not end the run.
            failed = [i for i in missing if i not in results]
            if failed:
                logger.warning(
                    "train_mlip for %s: fit(s) %s failed; retrying them once.",
                    base_name,
                    failed,
                )
                submit(failed)

        if len(results) < min_successful:
            raise RuntimeError(
                f"train_mlip for {base_name}: only {len(results)} trained model(s) "
                f"succeeded (out of {num_of_models} requested) — this workflow "
                f"needs at least {min_successful}. Check remote job logs for "
                "failures."
            )
        if len(results) < num_of_models:
            logger.warning(
                "train_mlip for %s: only %d/%d model(s) succeeded; proceeding with %d.",
                base_name,
                len(results),
                num_of_models,
                len(results),
            )

        self._store_predictions_and_cleanup(base_name, name, results)

        if training_config.get("quality_gate"):
            # check_quality_gate (mlip/evaluation.py) reads
            # committee["num_of_models_in_committee"] and committee["name"]
            # from whatever dict it's given; both are set by the caller
            # here (the number of models trained, and the hardcoded name).
            check_quality_gate(
                workdir,
                {
                    **training_config,
                    "num_of_models_in_committee": num_of_models,
                    "name": name,
                },
            )

        return [
            self._trained_model(ctx, fit_idx, seeds[fit_idx], results[fit_idx])
            for fit_idx in sorted(results)
        ]

    def _collect_fit(
        self, trainer_entry: Any, ctx: LoopContext, fit_idx: int
    ) -> tuple | None:
        """A fit's (model_path, compiled_model_path, metrics) if all its
        outputs exist and read back cleanly (for MACE: the model plus a
        checksum-verified evaluation_metrics.json), else None."""
        paths = trainer_entry.output_paths(
            self.training_config,
            base_name=ctx.base_name,
            name=_TRAINING_NAME,
            fit_idx=fit_idx,
        )
        if not all(p.exists() for p in paths):
            return None
        try:
            result: tuple = trainer_entry.read_existing_result(
                self.training_config,
                base_name=ctx.base_name,
                name=_TRAINING_NAME,
                fit_idx=fit_idx,
            )
        except (FileNotFoundError, ValueError):
            return None
        return result

    def _record_seed(self, ctx: LoopContext, fit_idx: int, seed: int) -> None:
        fit_dir = Path("results", ctx.base_name, _TRAINING_NAME, f"fit_{fit_idx}")
        fit_dir.mkdir(parents=True, exist_ok=True)
        (fit_dir / "fit_seed.json").write_text(json.dumps({"seed": seed}) + "\n")

    def _check_recorded_seed(self, ctx: LoopContext, fit_idx: int, seed: int) -> None:
        """A cached fit trained with a different seed (general.seed changed
        between restarts mid-training) is reused, but said so. Fits trained
        before seeds were recorded are trusted."""
        seed_file = Path(
            "results", ctx.base_name, _TRAINING_NAME, f"fit_{fit_idx}", "fit_seed.json"
        )
        if not seed_file.exists():
            return
        try:
            recorded = json.loads(seed_file.read_text()).get("seed")
        except (json.JSONDecodeError, OSError):
            return
        if recorded != seed:
            logger.warning(
                "Reusing cached fit_%d of %s trained with seed %s, but this "
                "run asks for seed %s (general.seed changed mid-training?). "
                "The models in this loop were trained with mixed seeds.",
                fit_idx,
                ctx.base_name,
                recorded,
                seed,
            )

    def best_model(self, models: list[TrainedModel]) -> TrainedModel:
        """The model with the lowest force MAE on the common validation
        split (test when no model has one) -- see rank_committee."""
        fit_dirs = {m.fit_idx: m.fit_dir for m in models}
        best_idx, _, split_used = rank_committee(fit_dirs, label="best_model")
        logger.info(
            "Best model: fit_%d (%s mae_f, of %d).", best_idx, split_used, len(models)
        )
        return next(m for m in models if m.fit_idx == best_idx)

    def generate_candidates(self, ctx: LoopContext, model: TrainedModel) -> list[Atoms]:
        """Run the configured structure generator (MD, EZGA, ...) with
        *model*, seeded from this loop's eligible training structures, and
        drop candidates with unphysically short bonds. The generator caches
        its own candidates file for restarts."""
        sg_config = self.jobs_dict["structure_generation"]
        sg_config.setdefault(
            "desired_num_of_structures", _DEFAULT_DESIRED_NUM_OF_STRUCTURES
        )
        selection_kwargs = sg_config.get("structure_selection_kwargs", {})
        eligible = filter_eligible_structures(
            ctx.train,
            chem_formula_list=selection_kwargs.get("chem_formula_list"),
            selectable_configs=selection_kwargs.get("selectable_configs"),
            atom_number_range=tuple(selection_kwargs.get("atom_number_range", (0, 0))),
        )
        generator_entry = resolve(
            "structure_generator", sg_config.get("generator", "md")
        )
        candidates = generator_entry.generate(
            eligible,
            model.model_path,
            sg_config,
            base_name=ctx.base_name,
            name=_STRUCTURE_GENERATION_NAME,
            hpc=sg_config["hpc"],
            max_time=sg_config["max_time"],
        )
        kept: list[Atoms] = filter_structures_by_min_bond_distance(candidates)
        return kept

    def predict(
        self,
        ctx: LoopContext,
        models: list[TrainedModel],
        structures: list[Atoms],
    ) -> list[dict]:
        """Energies/forces of *structures* from every model, as one parallel
        remote batch on the structure-generation HPC. Returns one
        ``{"forces": [...], "energies": [...]}`` per model (same order as
        *models*, each index-aligned with *structures*); raises if any
        model's prediction fails."""
        sg_config = self.jobs_dict["structure_generation"]
        trainer_name = self.training_config.get("trainer", "mace")
        remote_info = get_remote_info(
            {
                "hpc": sg_config["hpc"],
                "name": f"score_{_TRAINING_NAME}",
                "max_time": sg_config["max_time"],
            },
            input_files=[m.model_path for m in models],
        )
        job_configs = [
            {
                "function_kwargs": {
                    "structure_list": structures,
                    "model_path": m.model_path,
                    "trainer": trainer_name,
                    "trainer_config": self.training_config,
                }
            }
            for m in models
        ]
        results = submit_n(predict_with_model, job_configs, remote_info)
        for m, result in zip(models, results, strict=True):
            if result is None:
                raise RuntimeError(
                    f"Prediction failed for fit_{m.fit_idx} on {ctx.base_name}."
                )
        return list(results)

    # Same sentinel name as the evaluator module's own (_PHASE =
    # "high_accuracy_eval"), so runs written before this step used @phase
    # resume unchanged. The evaluator still checks and writes that file
    # itself too, because initialization calls it outside any loop.
    @phase("high_accuracy_eval", load=_load_high_accuracy_results)
    def high_accuracy_evaluate(
        self, ctx: LoopContext, structures: list[Atoms]
    ) -> list[Atoms]:
        """DFT-label AL-generated *structures*: each is relaxed until its
        max force is at most high_accuracy_evaluation.force_ceiling (null =
        single point), then labelled with this workflow's
        NEW_STRUCTURE_CONFIG_TYPE and the loop number."""
        for i, structure in enumerate(structures):
            structure.info["job_id"] = i
            # Set explicitly: a candidate can inherit needs_relaxation=True
            # from its MD seed (see CLAUDE.md).
            structure.info["needs_relaxation"] = self.force_ceiling is not None

        config = self.jobs_dict["high_accuracy_evaluation"]
        if self.force_ceiling is not None:
            config = {**config, "fmax": self.force_ceiling}
        labelled = _evaluator_orchestrate(
            structures,
            config,
            base_name=ctx.base_name,
            name=_HIGH_ACCURACY_EVALUATION_NAME,
            hpc=config["hpc"],
            max_time=config["max_time"],
            allow_relaxation=True,
            start_index=0,
        )
        logger.info(
            "High-accuracy evaluation completed for %d structures.", len(labelled)
        )
        return self._label_new_structures(ctx, labelled)

    def _label_new_structures(
        self, ctx: LoopContext, labelled: list[Atoms]
    ) -> list[Atoms]:
        """Tag freshly DFT-labelled structures with this workflow's
        NEW_STRUCTURE_CONFIG_TYPE and the loop number."""
        cleaned: list[Atoms] = clean_structures(
            labelled,
            config_type=self.NEW_STRUCTURE_CONFIG_TYPE,
            override_config_type=True,
            already_computed=True,
            extra_metadata={"al_loop": ctx.loop},
        )
        return cleaned

    def add_to_dataset(
        self,
        ctx: LoopContext,  # noqa: ARG002 -- uniform step signature
        structures: list[Atoms],
    ) -> None:
        """Add newly labelled structures to the database: split by
        dataset_kwargs.test_ratio, or with fixed_test all go to training
        except known held-out geometries (kept as "diagnostic")."""
        if self.dataset_kwargs["fixed_test"]:
            archive = self.db.get_all_as_atoms()
            known = {geometry_digest(a) for a in archive}
            held_groups = {
                a.info.get("split_group")
                for a in archive
                if a.info.get("split") == "test" and a.info.get("split_group")
            }
            new_train_data = []
            diagnostic = []
            for atoms in structures:
                key = geometry_digest(atoms)
                if key in known or atoms.info.get("split_group") in held_groups:
                    diagnostic.append(atoms)
                else:
                    new_train_data.append(atoms)
                    known.add(key)
            self.db.add_structures(
                diagnostic, split="diagnostic", skip_duplicates=False
            )
            new_test_data: list[Atoms] = []
        else:
            new_train_data, new_test_data = split_atoms_list_into_test_and_train(
                structures,
                test_fraction=self.dataset_kwargs["test_ratio"],
                seed=self.seed,
            )

        self.db.add_structures(new_train_data, split="train", skip_duplicates=False)
        self.db.add_structures(new_test_data, split="test", skip_duplicates=False)

    def finish_loop(self, ctx: LoopContext) -> None:
        """End of a loop: redundancy removal, train/test filters and
        curation on the grown database, then mark the loop done."""
        self._curate_dataset()
        self._mark_phase_done(ctx.base_name, "loop")
        logger.debug(
            "Completed AL loop %d, retraining with %d structures.",
            ctx.loop,
            len(ctx.train),
        )
        if self.plots and self.log_file is not None:
            from alomancy.analysis.timing_plots import timing_plots

            timing_plots(self.log_file, Path("results", "current_plots"))

    def validate_settings(self) -> None:  # noqa: B027 -- optional hook
        """Hook for workflow-specific config checks, called at the end of
        __init__ (after logging is set up). Raise ValueError on bad
        settings."""

    @abstractmethod
    def run(self) -> None:
        """Run the workflow: call the helpers above in this workflow's order."""


def build_workflow(jobs_dict: dict) -> ActiveLearningWorkflow:
    """Construct the workflow named by general.al_workflow (default
    "committee_uncertainty"), resolved through the registry's
    "al_workflow" category.

    Takes only jobs_dict -- every setting a workflow needs lives under
    jobs_dict["general"] (see ActiveLearningWorkflow.__init__ and
    _GENERAL_KWARGS_DEFAULTS). The one exception is db: a live
    GlobalDatabase instance can't be a config value, so a caller that needs
    to inject a pre-built one (mainly tests) sets `wf.db = ...` after
    construction instead -- the db property is lazy, so this never pays for
    the default GlobalDatabase(general.db_path) construction it replaces.
    """
    al_workflow = jobs_dict.get("general", {}).get("al_workflow", _DEFAULT_AL_WORKFLOW)
    if al_workflow not in registered("al_workflow"):
        raise ValueError(
            f"Unknown general.al_workflow {al_workflow!r}. "
            f"Available: {sorted(registered('al_workflow'))}"
        )
    workflow_cls: type[ActiveLearningWorkflow] = resolve(
        "al_workflow", al_workflow
    ).workflow_class
    return workflow_cls(jobs_dict=jobs_dict)
