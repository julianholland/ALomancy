# Deprecations & Planned Breaking Changes

This page tracks deprecated features and breaking changes planned for future
ALomancy releases, so users can migrate ahead of time. See `CHANGELOG.md` for
changes that have already shipped.

## Shipped in 1.0.0

### `max_batch_size` job-dict key removed

- **Status**: removed in 1.0.0 (deprecated since v0.4.8)
- **What changed**: the `max_batch_size` key on `high_accuracy_evaluation`
  (and any other job-dict section) is no longer read at all. Concurrency is
  governed entirely by `max_concurrent_jobs`, which lives on the **HPC
  profile** (`~/.alomancy/hpc_config.yaml`), not a per-workflow-phase job
  dict, since it's a property of the HPC system/account, not of any one
  phase.
- **Migrate**: re-run `alomancy add-hpc` for each profile (or hand-edit
  `~/.alomancy/hpc_config.yaml`) to add `max_concurrent_jobs: <N>` under the
  relevant profile's `hpc:` dict, then delete `max_batch_size` from your job
  YAML -- it is now silently ignored rather than used as a fallback.

### `ActiveLearningStandardMACE` / `BaseActiveLearningWorkflow` removed

- **Status**: removed in 1.0.0
- **What changed**: `src/alomancy/core/standard_active_learning.py` and
  `src/alomancy/core/base_active_learning.py` have been deleted.
  `CommitteeUncertaintyWorkflow` (`core/committee_uncertainty_workflow.py`,
  built via `build_workflow()`) is now the only workflow implementation --
  it does not subclass anything, resolving its trainer/structure-generator/
  DFT-evaluator/initialiser from config via the shared module registry
  instead of a different Python subclass per backend combination.
- **Migrate**: replace `ActiveLearningStandardMACE(...)` with
  `build_workflow(jobs_dict=..., ...)` (same constructor kwargs otherwise).
  The job-dict schema changed too: `workflow` is renamed `general` (holding
  `al_workflow`, `elements`, and a `committee_uncertainty_kwargs` dict for
  committee-specific settings like `number_models_in_committee`, renamed
  from `size_of_committee`); `mlip_committee` is renamed `training` and
  gains a `trainer` key; `initialization.creation_kwargs` is flattened --
  its sub-namespaces (`isolated_atom_kwargs`, `dimer_kwargs`, etc.) are now
  direct children of `initialization`. See `examples/basic_use/` and
  `examples/ezga_use/` for complete migrated configs, and
  `committee_uncertainty_workflow.py`'s module docstring for the full
  schema reference.
