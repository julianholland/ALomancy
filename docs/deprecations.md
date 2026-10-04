# Deprecations & Planned Breaking Changes

This page tracks deprecated features and breaking changes planned for future
ALomancy releases, so users can migrate ahead of time. See `CHANGELOG.md` for
changes that have already shipped.

## Shipped in 1.0.0

### `max_batch_size` job-dict key removed

- **Status**: removed in 1.0.0 (deprecated since v0.4.8)
- **What changed**: the `max_batch_size` key on `high_accuracy_evaluation`
  (and any other job-dict section) is no longer read at all. Concurrency is
  governed entirely by `max_num_of_concurrent_jobs`, which lives on the **HPC
  profile** (`~/.alomancy/hpc_config.yaml`), not a per-workflow-phase job
  dict, since it's a property of the HPC system/account, not of any one
  phase.
- **Migrate**: re-run `alomancy add-hpc` for each profile (or hand-edit
  `~/.alomancy/hpc_config.yaml`) to add `max_num_of_concurrent_jobs: <N>`
  under the relevant profile's `hpc:` dict, then delete `max_batch_size`
  from your job YAML -- it is now ignored rather than used as a fallback.
  Profiles written before the rename, which use `max_concurrent_jobs`, are
  still read, with a warning asking you to rename the key.

### `ActiveLearningStandardMACE` / `BaseActiveLearningWorkflow` removed

- **Status**: removed in 1.0.0
- **What changed**: `src/alomancy/core/standard_active_learning.py` and
  `src/alomancy/core/base_active_learning.py` have been deleted.
  Workflows now subclass `ActiveLearningWorkflow`
  (`core/active_learning_workflow.py`; `CommitteeUncertaintyWorkflow`,
  `RandomSelectionWorkflow`, `NoveltySelectionWorkflow`) and resolve their
  trainer/structure-generator/DFT-evaluator/initialiser from config via the
  shared module registry instead of a different Python subclass per
  backend combination. The config, through `general.al_workflow`, picks
  the workflow.
- **Migrate**: replace `ActiveLearningStandardMACE(...)` with
  `ALomancy("config.yaml").run()` (`from alomancy import ALomancy`) or
  `alomancy run config.yaml`; every former constructor kwarg is a
  `general` key in the config. The job-dict schema changed too: `workflow`
  is renamed `general` (holding `al_workflow`, `elements`, and a
  `committee_uncertainty_kwargs` dict for committee-specific settings like
  `num_of_models_in_committee`, formerly `size_of_committee`);
  `mlip_committee` is renamed `training` and
  gains a `trainer` key; `initialization.creation_kwargs` is flattened --
  its sub-namespaces (`isolated_atom_kwargs`, `dimer_kwargs`, etc.) are now
  direct children of `initialization`. See `examples/configs/` for
  complete, tested configs of every workflow, and `docs/examples.md` for
  each section's settings.

### Config keys renamed or moved

- **Status**: changed in 1.0.0
- **What changed**: an old key raises a `ValueError` at construction
  naming its replacement, so nothing is ignored without warning.
  - Counts follow one `num_of_*` convention (full table in `CHANGELOG.md`),
    e.g. `number_of_al_loops` -> `num_of_al_loops`.
  - `structure_generation.desired_num_of_structures` is split in two:
    `general.num_of_structures_per_loop` (how many structures the selector
    sends to DFT each loop) and `structure_generation.
    num_of_structures_to_generate` (the candidate pool, default 10x per
    loop; it must be at least the per-loop count).
  - The start-up keys (`initial_train_file_path`/`initial_test_file_path`,
    `initialization.extra_datasets`, `skip_initialization`) are replaced by
    `general.start_from` (see `starting_a_run.md`).
  - `general.high_force_threshold` is replaced by
    `high_accuracy_evaluation.force_ceiling` (relax AL structures before
    DFT) and `general.train_filter.max_force` (exclude high-force training
    structures).
  - Split settings (`test_ratio`, `target_config_types`, ...) moved from
    `general.committee_uncertainty_kwargs` to `general.dataset_kwargs`.
- **Migrate**: run your config once; the error lists every old key with its
  new name or location.
