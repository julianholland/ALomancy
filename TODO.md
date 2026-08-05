# TODO

## GlobalDatabase: auto-recover from mid-run corruption

If the sage_lib `Partition` backing `GlobalDatabase` is detected as corrupted
(HDF5/SQLite auto-repair resets it to 0 containers — observed in the wild via
`WARNING: Corrupted HDF5 file detected... Backing it up`), a resumed run
currently only self-heals in one narrow case: `initialize_training_set`'s fast
path re-seeds the DB from `initial_train_file_path`/`initial_test_file_path`
when `self.db.size == 0`, but only reaches that branch when
`_last_complete_loop() < 0` (i.e. before loop 0 has completed).

Once at least one AL loop has completed, `BaseActiveLearningWorkflow.run()`
(`core/base_active_learning.py:98-107`) resumes by calling
`self.db.get_train_atoms()` / `get_test_atoms()` directly with no fallback —
if the DB is empty/corrupted at that point, the run silently proceeds with 0
structures instead of detecting the corruption and reinitializing from the
most recent `results/al_loop_<N>/train_set.xyz` / `test_set.xyz` (which
`run()` already writes at the top of every loop iteration, `base_active_learning.py:157-166`).

Proposed fix: on resume, if `db.size == 0` but a prior loop's `train_set.xyz`/
`test_set.xyz` exist on disk, reload from the most recent one and re-seed the
DB (mirroring the existing fast-path logic), rather than silently continuing
with an empty train/test set.

## Parity plots never populate, even on fresh (post-v0.4.2) runs

`plot_dft_vs_model` consistently logs, on fresh runs:

```
No MACE eval prediction files found for al_loop_0 — parity plots will not be
available for this loop. Predictions are written during remote training from
alomancy v0.4.2 onwards.
```

This message implies missing prediction files is expected only for
pre-v0.4.2 runs, but it's firing on current runs too. Need to check whether
`_save_mace_eval_predictions` (`mlip/mace_wfl.py`) is actually writing
`train_pred.xyz`/`test_pred.xyz` in `fit_<i>/` on the remote node, and/or
whether `store_mlip_predictions`/`plot_dft_vs_model` are looking in the
right path/filename for them.
