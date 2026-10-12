# Remote submission

Everything that runs a Python function on an HPC node: `src/alomancy/remote_submission/` (`executor.py`, `submitters.py`), `configs/remote_info.py`, the ExPyRe isolation in `alomancy/__init__.py`, and the ssh helpers in `utils/remote_ssh.py`. This subsystem caused most of the production incidents so far; read this before changing it.

## Call path

```
workflow / module entry point
   │  submit_n(function, job_configs, remote_info)        executor.py
   │  ase_remote_submitter(...)  (DFT only, builds job_configs per structure)
   ▼
RemoteJobExecutor(remote_info).run_and_wait(function, job_configs)
   │  submit_multiple_jobs  → one expyre.func.ExPyRe per job_configs entry
   │  run_all_jobs_bounded  → ThreadPoolExecutor, one _run_single_job per job
   │  cleanup_jobs          → mark_processed + wipe stage dirs of finished jobs
   ▼
expyre.func.ExPyRe (third party): .start() = ssh/stage-in/sbatch; .get_results() = poll + rsync back
   ▼
Slurm on the configured host
```

- **`submit_n`** (`executor.py`) is the one generic primitive. Training (`ActiveLearningWorkflow._train_models_phase`), prediction (`ActiveLearningWorkflow.predict`) and MD (`structure_generation/md/md_wfl.py`) call it directly. It returns a list aligned with `job_configs`; a job that failed contributes `None` and never raises, so callers count failures themselves.
- **`ase_remote_submitter`** (`submitters.py`) is the only submitter left: one DFT job per structure for `high_accuracy_evaluation`. `per_structure_function` (same length as the structures, else `ValueError`) picks `run_go` or `run_sp` per structure, so GO and SP structures share one pool, distinguished by the job name.
- A `job_configs` entry is a dict with `function_kwargs` (required) and optionally `input_files`, `output_files`, `job_name`, `item`, and `function` (a per-job override of the shared function, used for the GO/SP mix).
- `item` names what the job computes (`fit_<i>`, `md_run_<slot>`, `structure_<job_id>`); every job event carries it as `data.item` so the loop report can drop failures whose retry succeeded (see `reports_and_plots.md`).

## `RemoteInfo` and `get_remote_info`

`get_remote_info(job_dict, input_files=None)` (`configs/remote_info.py`) builds one `RemoteInfo` per phase from `job_dict["hpc"]` (a profile from `~/.alomancy/hpc_config.yaml`), `job_dict["max_time"]` and `job_dict["name"]` (the section name, merged in by the caller). `sys_name` is an ExPyRe system name from the run's ExPyRe config, not defined in this repo.

Fields that matter:
- Passed to ExPyRe: `sys_name`, `job_name`, `resources`, `pre_cmds`, `post_cmds`, `env_vars`, `input_files`, `output_files`, `header_extra`, `exact_fit`, `partial_node`; `timeout`/`check_interval` go to `get_results`.
- `max_num_of_concurrent_jobs` caps the thread pool. It comes from the HPC profile (default 20, set by `alomancy add-hpc`), resolved by `_resolve_max_concurrent_jobs`; profiles with the pre-rename key `max_concurrent_jobs` are honoured with a warning. The job-dict-level `max_batch_size` fallback was removed in 1.0.0. Unrelated to MD's `num_of_md_starts`.
- `resubmit_killed_jobs` (default `False`): see "Resume and resubmit" below.
- `lock_timeout` defaults to `None`: a worker waits indefinitely for the per-host ssh lock. A version that derived it from `max_time` was reverted: the thread *holding* the lock while blocked on a password/OTP prompt was never bounded (expyre's `Popen(...).communicate()` has no timeout), so a timeout only made every queued thread fail loudly around the one that was still waiting. The real fix is `ensure_ssh_connectivity` (below).

## Scheduling: `run_all_jobs_bounded`

`ThreadPoolExecutor(max_workers=min(max_num_of_concurrent_jobs, n_jobs))`, one `_run_single_job` per job. Jobs beyond the cap queue, and the pool starts the next one the instant a worker frees up (success or failure). Results land index-aligned regardless of completion order. A non-positive cap falls back to one worker with a warning.

`_run_single_job`: acquire the per-host ssh lock → `job.start()` → release → `_get_results_with_resume` → log stdout/stderr at DEBUG → compute queue time from `_expyre_job_started`'s mtime (logged as `Job N queue_time=… s.`, parsed by the timing plot) → return `(index, result)`. It never lets an exception escape; its failure handler logs the full traceback at DEBUG in `results/alomancy.log`.

## Resume and resubmit

**Transient failures are resumed on the same job object, never treated as permanent** (`_get_results_with_resume`). A `get_results()` exception is retried while `job.status` is in `_RESUMABLE_JOB_STATUSES`:
- the ongoing statuses `created`/`submitted`/`started` (expyre raised from transport, e.g. an rsync "No space left on device", before updating the status for that poll);
- plus `succeeded`: expyre sets it *before* copying outputs back to `cwd`, so a local ENOSPC during the copy-out still reads `succeeded`; the remote side is done, so re-entering `get_results()` just retries the copy.

Transport exceptions get up to `_TRANSPORT_RETRY_LIMIT` (5) retries with exponential backoff from `_TRANSPORT_RETRY_BACKOFF_SECONDS`; `ExPyReTimeoutError` gets `_TIMEOUT_RETRY_LIMIT` (1). The two budgets are separate counters, so an interleaved sequence of both can't exhaust the wrong one. Once `job.status` is `failed`/`died` (expyre's only terminal statuses, set before it raises), the exception re-raises immediately.

**Died jobs** (`ExPyReJobDiedError`): `_salvage_partial_output` first copies whatever the dead job's registered outputs already hold from its stage dir (it never raises), then, only if `RemoteInfo.resubmit_killed_jobs` is `True`, one fresh resubmission with `job.start(force_rerun=True)`. Salvage must come first: `force_rerun` wipes only the *remote* stage dir and reuses the local one, so the replacement's synced output would overwrite anything not yet salvaged. Off by default because a scheduler-killed job (OOM, walltime, node failure) usually needs different resources, not an identical retry.

**`cleanup_jobs`** wipes the local and remote stage dirs of `succeeded` jobs (under the ssh lock) and calls `mark_processed()` only on terminal jobs. A job still ongoing after resume retries ran out is left alone: `processed` is excluded from expyre's restart reattachment, so marking it would orphan a job that may still finish.

## Thread and process safety

| Mechanism | Guards | Why |
|---|---|---|
| `_get_ssh_call_lock(sys_name)` / `_acquire_ssh_call_lock_or_raise` (per-host `threading.Lock`) | `job.start()` and every status sync | Too many concurrent ssh sessions to one host fall back to a fresh connection that hangs on an unattended password/OTP prompt |
| `_ensure_expyre_sync_serialized` (patches `ExPyRe.sync_remote_results_status`; per-host generation counter skips redundant syncs) | the squeue + rsync inside `get_results()` polling | a stuck `squeue` seen in production when many threads called it at once |
| `_ensure_expyre_db_thread_safe` (reopens `expyre.config.db` with `check_same_thread=False`, wraps `JobsDB._execute` in a lock via `_locked_execute`) | expyre's single module-level sqlite connection | expyre's job DB isn't thread-safe |
| `acquire_local_expyre_lock` (`fcntl.flock` on `local_stage_dir`) | one alomancy process per job-state directory | two runs sharing a `jobs.db` |

**`_locked_execute` must hold the lock until the rows are fetched**, not just while `execute()` runs. `execute()` returns a live, unfetched cursor and every expyre caller consumes it afterwards (`list(self._execute(...))`). Releasing early let another thread's `execute()` on the same connection corrupt the pending read and silently return zero rows for a row that exists: the real cause of a production `list(config.db.jobs(id=...))[0]` `IndexError` from one process polling ~20 MD jobs. `_locked_execute` materializes the cursor into a list inside the lock.

`acquire_local_expyre_lock` runs at the start of `pre_run_checks()`: an exclusive non-blocking `flock`, raising `RuntimeError` if another process holds it (a no-op if this process already does, or if `local_stage_dir` hasn't resolved). The OS releases an `flock` when the holder's descriptor closes, including on a crash or `kill -9`, so a dead run never blocks a restart. The lock file records `pid`/`host`/`started`. The module-global handle is set under `_expyre_root_lock_guard` (double-checked locking): `flock` is per open file description, so two threads racing could otherwise each open the file and the second would report a conflicting process.

## ssh connectivity before submission

`pre_run_checks()` calls `ensure_ssh_connectivity` (`utils/remote_ssh.py`): one bare `ssh <host> true` per distinct HPC host in the run, with no timeout and no captured output, so a password/OTP prompt reaches the real terminal while the person who launched the run is still there. With `ControlMaster`/`ControlPersist` this also opens the shared connection every later call reuses. It never raises; an unreachable host is a warning, and the real submission fails loudly later.

## Each run gets its own ExPyRe state

`alomancy/__init__.py`'s `_ensure_local_expyre_root` runs at import, before anything imports `expyre.config`, and copies ALomancy's canonical HPC systems config (`configs/global_config.py`'s `EXPYRE_CONFIG`, `~/.alomancy/expyre_config.json`, written by `alomancy add-hpc`) to `<rundir>/.expyre/config.json`, unless the directory already has a `.expyre`/`_expyre` or `EXPYRE_ROOT` is set. `_pin_local_expyre_root` then sets `EXPYRE_ROOT` to that directory (only if it has a `config.json`), so expyre reads it alone.

Without the pin, expyre walks from `cwd` up to `$HOME` merging every `.expyre` it finds and uses the deepest as `local_stage_dir` (where `jobs.db` and every stage dir live). The walk only stops at a directory equal to `$HOME`, so a run under `/work/home/u` with `$HOME=/home/u` walked to `/` and merged the shared `~/.expyre`, including a `local_stage_dir` that moved the run's job state into it (issue #82).

The copy is a one-time snapshot: editing `EXPYRE_CONFIG` later doesn't change a run whose `.expyre/` exists. `write_expyre_config` (`cli/add_hpc.py`) migrates once from `LEGACY_EXPYRE_CONFIG` (`~/.expyre/config.json`) when `EXPYRE_CONFIG` is missing, and raises `ValueError` on a corrupt legacy file rather than dropping systems. Every reader of the systems config must fall back the same way: `resolve_hpc_host` (`utils/remote_ssh.py`) tries `EXPYRE_CONFIG`, then `LEGACY_EXPYRE_CONFIG`. `alomancy nuke` (`cli/nuke.py`, `resolve_default_expyre_dir`) clears the run's own `.expyre`, falling back to `~/.expyre` only when none resolved.

## Remote deployment

ExPyRe pickles functions **by reference** (module path + name); the node imports the module at run time. After changing the signature or body of anything that runs remotely (`mlip/base.run_training`, a trainer's `fit`/`evaluate`, `md_wfl._run_md_with_spec`, `run_md`, `utils/dft_utils._run_go`/`_run_sp`, `run_qe.get_qe_input_data`, `mlip/predict.predict_with_model`), reinstall on the HPC (`alomancy upgrade-hpc`, or `pip install -e .` there) before restarting, or jobs import the old code and fail with a `TypeError`, typically hours after submission. **Anything submitted remotely stays a module-level function**, never a method of a workflow class. Guard config errors in remote code with `if`/`raise ValueError`, never `assert` (`python -O`, common on HPC, strips asserts).

## Slurm details

- **The nested `srun` requests `--mem=0`** (`_build_srun_command`, `utils/dft_utils.py`): "all memory already granted to the job". On Slurm setups that credit the batch script, as the job's first step, with the whole allocation, an explicit sub-amount (`--mem=<N>`) competes with that reservation and fails with `Unable to create step ... Memory required by task is not available`, however much was granted. The job-level ceiling belongs on the outer `#SBATCH --mem`.
- **`#SBATCH --mem` is per partition**: `add_hpc_wizard` (`cli/add_hpc.py`) asks for `max_mem` per partition and writes each partition's own `--mem=<N>M` into that partition's `header` in the ExPyRe config (`_with_partition_mem_headers`); ExPyRe adds a partition's header to the system-wide one. One blanket value under-requests on large-memory partitions or is rejected on small ones. `node_info.max_mem_per_node` in `hpc_config.yaml` is descriptive only.

## Tests

`tests/utils_tests/test_utilities.py` covers the executor with a real-object fake (`_FakeExPyReJob`, with a `get_results_side_effects` queue for transport-only vs `(exception, status)` terminal failures), real threads and wall-clock timing: resume, retry limits, terminal no-retry, both resubmit branches, salvage, cleanup, concurrency caps, the ssh lock and the sqlite thread safety (`test_add_then_immediate_jobs_lookup_never_empty_under_concurrency`). `tests/remote_submission_tests/test_submitters.py` covers `ase_remote_submitter`, mocking only `RemoteJobExecutor.run_and_wait`. Known untested paths, judged not worth it: the invalid-cap fallback, the outer worker exception guard, `job.clean()` raising during cleanup.

## History

- **No resume on transient failures** (fixed): any `get_results()` exception used to abandon the job. An rsync race with MACE's checkpoint rotation at epoch 6 lost `fit_0` permanently; the old backfill only retried below 3 fits and the old best-model fallback defaulted to `fit_0`, so MD then crashed staging a model that didn't exist. Fixed by resume (above), retrying every missing fit, and `rank_committee` raising instead of guessing (`core_workflow.md`).
- **Disk exhaustion** (fixed): stage dirs were never cleaned (297 dirs, 249 GB), causing `No space left on device` mid-sync; `cleanup_jobs` now wipes finished jobs and trainers clean local checkpoints (`mlip.md`).
- **Parity plots with one point** (fixed, two causes): a stale prediction file from a disk-full sync, and the TorchScript-compiled MACE model failing force evaluation (`Global alloc not supported yet`); evaluation now uses the uncompiled model (`mlip.md`).
