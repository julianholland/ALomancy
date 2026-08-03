# Deprecations & Planned Breaking Changes

This page tracks deprecated features and breaking changes planned for future
ALomancy releases, so users can migrate ahead of time. See `CHANGELOG.md` for
changes that have already shipped.

## Targeting 1.0.0

### `max_batch_size` job-dict key removal

- **Status**: deprecated since v0.4.8 (still works via a fallback, logs a
  warning)
- **Removal target**: 1.0.0
- **What's changing**: the `max_batch_size` key on `high_accuracy_evaluation`
  (and any other job-dict section) will no longer be read at all.
- **Why**: `max_batch_size` used to control how many structures were grouped
  into one remote-submission chunk, and indirectly how much submission
  concurrency you got, with the workflow waiting for an entire chunk to
  finish before submitting the next one. Chunking has been removed entirely
  — all remaining structures are submitted in a single call (geometry
  optimisation and single-point structures share one submission queue,
  distinguished only by which function each job runs) — and concurrency is
  now governed by `max_concurrent_jobs`, which lives on the **HPC profile**
  (`~/.alomancy/hpc_config.yaml`), not a per-workflow-phase job dict, since
  it's a property of the HPC system/account, not of any one phase.
- **Migrate now**: re-run `alomancy add-hpc` for each profile (or hand-edit
  `~/.alomancy/hpc_config.yaml`) to add `max_concurrent_jobs: <N>` under the
  relevant profile's `hpc:` dict, then delete `max_batch_size` from your job
  YAML. Until 1.0.0, a `max_batch_size` left on a job dict is used as a
  fallback source for the concurrency cap — but *only* when the HPC profile
  doesn't already define `max_concurrent_jobs` — with a deprecation warning
  logged either way.
