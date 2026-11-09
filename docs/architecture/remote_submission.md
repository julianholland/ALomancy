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
- `lock_timeout` defaults to `None`: a worker waits indefinitely for the per-host ssh lock. A version that derived it from `max_time` was reverted: the thread *holding* the lock while blocked on a password/OTP prompt was never bounded (expyre's `Popen(...).communicate()` has no timeout), so a timeout only made every queued thread fail loudly around the one that was still waiting. The real fix is the run's own ssh master, logged in at start, plus BatchMode for everything after (below).

## Scheduling: `run_all_jobs_bounded`

`ThreadPoolExecutor(max_workers=min(max_num_of_concurrent_jobs, n_jobs))`, one `_run_single_job` per job. Jobs beyond the cap queue, and the pool starts the next one the instant a worker frees up (success or failure). Results land index-aligned regardless of completion order. A non-positive cap falls back to one worker with a warning.

`_run_single_job`: acquire the per-host ssh lock → `job.start()` → release → `_get_results_with_resume` → log stdout/stderr at DEBUG → compute queue time from `_expyre_job_started`'s mtime (logged as `Job N queue_time=… s.`, parsed by the timing plot) → return `(index, result)`. It never lets an exception escape; its failure handler logs the full traceback at DEBUG in `results/alomancy.log`.

## Live run status

`progress.py` (`remote_submission/`) replaces expyre's own progress output. `_get_results_with_resume` passes `quiet=True`, so expyre no longer writes a status letter per poll per thread.

- **`TRACKER`** (`JobTracker`) holds the run and one batch per active `run_all_jobs_bounded`.
  - The run: `set_run_info` (`ActiveLearningWorkflow.panel_run_info()` at the end of `pre_run_checks`: name, directory, workflow, `num_of_al_loops`, elements and the Settings rows; selector text from `selector_description()`, which a workflow may override), `set_loop` (`iterate_loops`; the start time is the loop's first event in `results/events.jsonl`, `_loop_started_at`, so loop time survives a restart), `set_phase` (the `@phase` decorator), `set_best_model` (`_update_best_model`).
  - A batch: phase (`remote_info.job_name`), loop (from the jobs' `results/al_loop_N` output paths, `loop_from_paths`) and a row per job. `unregister` keeps a one-line summary in `finished_batches` for `FINISHED_KEEP` (5 min).
  - `_run_single_job` marks each job submitted (with its remote id), and done or failed.
  - The `_locked_sync` patch calls `TRACKER.synced(sys_name, _remote_statuses(sys_name))` after every successful sync. That records "last data" for the HPC and moves jobs between queued and running from expyre's DB. A failed sync calls `sync_failed` and re-raises unchanged; a coalesced (skipped) sync records nothing.
  - ssh: `auth_lost` (system → since, host) and `ssh_state` (`ok` / `login needed`), set by `mark_auth_lost`, `auth_restored`, `set_ssh_ok` and `synced`.
- **`ProgressMonitor`**: `ALomancy.run` (`core/entry.py`) holds one for the whole run (`defer_live=True, watch_ssh=True`). Each `run_all_jobs_bounded` enters its own, which nests and shares the outer one's panels. A bare executor run outside a workflow gets its own.
  - **On a terminal** (and unless `ALOMANCY_LIVE=0`, set by `alomancy run --no-live`), a rich `Live` view of two bordered panels. Everything sits in a sub-section headed by `section()`: a dim rule with leading dashes and the title in the brand violet.
    - "ALomancy" has Run Details, Settings and Best Model.
      - Settings is a Block / Information / HPC table, with bold violet headers over a thin dim rule (`box.SIMPLE_HEAD`). It is dropped below 80 columns.
      - Best Model shows the MAE on the split it was selected on.
    - "Current task" has Errors, Status and Options.
      - Errors (`error_lines`) is only shown when something is wrong: lost logins, failed jobs, failing syncs.
      - Status has the current phase, then each batch: `batch_header`, `batch_bar` and the per-HPC table with an `ssh` column. Finished batches follow as `✓` lines.
      - Options lists the keys, or `alomancy login` when there are none.
    - With `defer_live`, the panels first appear once `set_run_info` has run or a batch starts, so the start-of-run logins reach a plain terminal.
    - The colours come from `analysis/palette.py` (`_BAR_STYLE`: done brand violet, running cyan, queued amber, to submit `PENDING_COLOR`; failures rose, warnings amber, ok emerald), each summary label in its bar segment's colour. `palette.py` doesn't import matplotlib (`analysis/colors.py` re-exports it for plots).
    - The console log handler is pointed at rich's stdout proxy, so log lines print above the panels.
  - **Keys** (`_key_loop`, only when stdin is a terminal):
    - The thread puts the terminal in cbreak mode and `select`s on stdin. `handle_key` maps `L` / `1`–`9` to a login and `?` to help.
    - A login runs `executor.relogin` inside `_handoff`. That pauses the key thread (`_keys_idle`), stops `Live`, buffers the log handlers into a `StringIO`, restores the terminal mode, then undoes all of it and flushes the buffered lines.
    - `terminal_handoff()` gives the same to other callers (`pre_run_checks`' logins); it is a no-op without a monitor.
  - **Otherwise:** an INFO `Remote jobs: ...` summary line every 5 minutes while a batch runs or a login is lost, prefixed with loop, phase and the best model's `mae_f`.
  - **Always:** `results/remote_status.json`, written atomically every 5 s (with the `run`, `ssh` and `finished_batches` keys) and marked `finished` at the end. `alomancy status` renders it.
  - **ssh master watch** (`_watch_loop` → `executor.check_ssh_masters`, every `SSH_WATCH_INTERVAL` = 30 s):
    - For each host it checks the run's master (`ssh -O check`). A dead one is reopened with BatchMode, so key-auth hosts recover silently.
    - Otherwise `_note_ssh_auth_lost` raises the banner before any poll fails.
- "Last data" is flagged stale after 3 check intervals without a sync.
- `tests/conftest.py` sets `progress.STATUS_FILE = None`, so tests never write into the repository's `results/`. The file is only written when its directory exists.

## Resume and resubmit

**Transient failures are resumed on the same job object, never treated as permanent** (`_get_results_with_resume`). A `get_results()` exception is retried while `job.status` is in `_RESUMABLE_JOB_STATUSES`:
- the ongoing statuses `created`/`submitted`/`started` (expyre raised from transport, e.g. an rsync "No space left on device", before updating the status for that poll);
- plus `succeeded`: expyre sets it *before* copying outputs back to `cwd`, so a local ENOSPC during the copy-out still reads `succeeded`; the remote side is done, so re-entering `get_results()` just retries the copy.

Transport exceptions get up to `_TRANSPORT_RETRY_LIMIT` (5) retries with exponential backoff from `_TRANSPORT_RETRY_BACKOFF_SECONDS`; `ExPyReTimeoutError` gets `_TIMEOUT_RETRY_LIMIT` (1), but only once the job has started: a timeout while `job.status` is still `submitted` (expyre never saw `_expyre_job_started`) resumes without using an attempt (INFO event `job_still_queued`), so queue time never counts toward the wait limit. Raven and viper allow 8 running and 300 submitted jobs per user, so with a high `max_num_of_concurrent_jobs` most jobs wait in the queue; good_c's DFT jobs once all failed this way after 2 × 30 h of queueing. The two budgets are separate counters, so an interleaved sequence of both can't exhaust the wrong one. Once `job.status` is `failed`/`died` (expyre's only terminal statuses, set before it raises), the exception re-raises immediately.

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

## The run's own ssh connections

**alomancy owns one ssh master per HPC host** (`utils/ssh_master.py`).
- `pre_run_checks()` calls `start_run_masters` (inside `progress.terminal_handoff`) for every distinct host of the run's HPC profiles (`resolve_hpc_host`).
- Each master is `ssh -M -N -f -o ControlPath=<template> -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 <host>`, run without captured output or a timeout. A password/OTP prompt therefore reaches the real terminal while the person who launched the run is still there.
- A master that is still alive (a restart of the same run) is reused without a prompt. An unreachable host is a warning, never an exception.

**The socket path.**
- `control_path_template` is `~/.alomancy/cm/<run_tag>-%C`. `run_tag` is 6 hex characters of the resolved run directory, and `ssh` expands `%C` per host.
- `control_dir` falls back to `/tmp/alomancy-cm-<uid>` when the home path would push the socket past the ~104-character unix limit (the 40-character `%C` plus ssh's 17-character temporary suffix).
- The directory has permissions 0700. A stale socket from a dead master is removed before `-M`, which would otherwise run without becoming a master (`_socket_path` via `ssh -G`).

**Owning the master doesn't depend on the user's ssh config.**
- Before, multiplexing had to be set in `~/.ssh/config`. Without it the start-of-run login did nothing for later calls, and BatchMode then failed every call on an OTP-only host.
- `HostName`, `User`, `IdentityFile`, `ProxyJump` and the like still come from `~/.ssh/config`; only the `Control*` options are overridden. A jump host is logged in through the master command, which is interactive.
- `ALOMANCY_OWN_SSH_MASTER=0` (`own_master_enabled`) goes back to `ensure_ssh_connectivity` (`utils/remote_ssh.py`): one bare `ssh <host> true` per host, relying on the user's own ControlMaster.

**Lifetime.**
- `ALomancy.run` calls `MASTERS.close_all()` (`ssh -O exit`) when the workflow returns normally.
- A crash or Ctrl-C leaves the masters open, so the restart needs no login.
- `alomancy cancel` uses them for its own ssh (`run_ssh_command(control_path=...)`) and closes them at the end.
- Each run has its own masters, so two runs to one HPC log in once each and never close each other's connection.

**After that login, unattended ssh never prompts.**
- `use_noninteractive_ssh` (`utils/remote_ssh.py`, called next in `pre_run_checks`) routes ExPyRe's ssh through `alomancy-rsh` (`utils/rsh.py`, a console script whose `build_argv` gives `ssh -o BatchMode=yes -o ConnectTimeout=30 "$@"`).
- When `start_run_masters` has set `ALOMANCY_SSH_CONTROL_DIR` and `ALOMANCY_SSH_RUN_TAG` in the run's environment (which ExPyRe's subprocesses inherit), it also adds `-o ControlPath=<dir>/<tag>-%C -o ControlMaster=no`. Every call then goes through the run's master and never tries to become one. Under `ALOMANCY_TEST_MODE` the environment is left alone.
- ExPyRe takes the ssh command from two places, so both are switched: rsync copies read `$EXPYRE_RSH`, while each `System` and its scheduler (squeue, sbatch, scancel) keep the `remsh_cmd` from the systems config, which `add-hpc` writes as `"ssh"`. A user-set `EXPYRE_RSH` or a custom `remsh_cmd` is left alone.
- It has to be a single executable: ExPyRe builds `rsync -e <EXPYRE_RSH>` by joining strings and splitting on spaces, so `"ssh -o …"` would break every rsync.
- Once the master is gone, a call fails within a second instead of blocking. Without the wrapper, good_c and fine_tune_good_c once hung for good (2026-11-03): fhi-raccoon only accepts `keyboard-interactive`, its master died after a network blip, and the next rsync and squeue each sat on a password prompt the live panel hid.

**A lost ssh login waits rather than failing jobs.**
- `is_ssh_auth_failure` (`utils/rsh.py`) recognises ssh's own login errors (`Permission denied (<method>)`, `Host key verification failed`, but not a remote file's `Permission denied (13)`).
- Such a failure calls `_note_ssh_auth_lost` from three places: the sync wrapper (`_ensure_expyre_sync_serialized`), `_get_results_with_resume`, and `job.start()` in `_start_and_wait`.
  - It logs one WARNING per outage, event `ssh_auth_lost`, telling the user to press L or run `alomancy login`.
  - `progress.TRACKER` shows the same line under Errors and `login needed [n]` in the ssh column.
  - The caller then waits `_SSH_AUTH_WAIT_SECONDS` (60) and tries again without using an attempt, so a job is never abandoned over a login.
- `job.start()` is simply repeated: expyre sets `submitted` only after sbatch returned, so the job is still `created`. The wait happens outside the per-host ssh lock.
- A login through `relogin` (the L key) or the first successful sync afterwards logs `ssh_auth_restored` and clears the banner. `relogin` restores every system that shares the host (`MASTERS.systems_for`). It writes its "Logging in to …" line straight to the terminal, because the log handlers are buffered during the handoff.
- `alomancy login` (`cli/login.py`) reopens the same socket from another terminal, and the next sync clears the banner.
- `run_ssh_command`, used only by CLI commands someone is watching, stays interactive.

## Each run gets its own ExPyRe state

`alomancy/__init__.py`'s `_ensure_local_expyre_root` runs at import, before anything imports `expyre.config`, and copies ALomancy's canonical HPC systems config (`configs/global_config.py`'s `EXPYRE_CONFIG`, `~/.alomancy/expyre_config.json`, written by `alomancy add-hpc`) to `<rundir>/.expyre/config.json`, unless the directory already has a `.expyre`/`_expyre` or `EXPYRE_ROOT` is set. `_pin_local_expyre_root` then sets `EXPYRE_ROOT` to that directory (only if it has a `config.json`), so expyre reads it alone.

Without the pin, expyre walks from `cwd` up to `$HOME` merging every `.expyre` it finds and uses the deepest as `local_stage_dir` (where `jobs.db` and every stage dir live). The walk only stops at a directory equal to `$HOME`, so a run under `/work/home/u` with `$HOME=/home/u` walked to `/` and merged the shared `~/.expyre`, including a `local_stage_dir` that moved the run's job state into it (issue #82).

The copy is a one-time snapshot: editing `EXPYRE_CONFIG` later doesn't change a run whose `.expyre/` exists. `write_expyre_config` (`cli/add_hpc.py`) migrates once from `LEGACY_EXPYRE_CONFIG` (`~/.expyre/config.json`) when `EXPYRE_CONFIG` is missing, and raises `ValueError` on a corrupt legacy file rather than dropping systems. Every reader of the systems config must fall back the same way: `resolve_hpc_host` (`utils/remote_ssh.py`) tries `EXPYRE_CONFIG`, then `LEGACY_EXPYRE_CONFIG`. `alomancy nuke` (`cli/nuke.py`, `resolve_default_expyre_dir`) clears the run's own `.expyre`, falling back to `~/.expyre` only when none resolved.

## Remote run directory and its length

ExPyRe puts a job in `<rundir>/<rundir_extra>/run_<job name>_<44-char arghash>_<8-char mkdtemp suffix>/`, and a DFT job's VASP/QE runs in `results/<base_name>/high_accuracy_evaluation/batch_<b>/ase_output_<i>/` below that. VASP keeps paths in a ~256-character buffer: at 255 characters the filename is cut off and VASP fails with `forrtl: severe (30): open failure ... Is a directory`.

- **`rundir_extra`**: expyre always sets it to `$HOSTNAME-<full local run path, / as _>` (`unkownhost` when `$HOSTNAME` is unset). `_shorten_expyre_remote_rundirs` (`alomancy/__init__.py`) replaces it, at import, with `_run_rundir_extra`: a 6-hex-character hash of `socket.gethostname()` (`_host_tag`; the host the run lock records, set even where `$HOSTNAME` isn't), then the run path below `$HOME` (`3fa9c2-Simulations_alomancy_test_5`; `$HOME` is also tried resolved, since it is often a symlink to `/work/home/u`; a run outside `$HOME` keeps its full flattened path). Each run still has its own remote directory, and the hash keeps the same path on two machines apart. It keeps the long layout while the run's `jobs.db` has a job `submitted` or `started` (`_jobs_pinning_long_layout`): a resumed job finds its remote directory from the current `remote_rundir`, and only those still need one (`created` isn't staged, `succeeded` has its results locally, `failed`/`died`/`processed`/`cleaned` aren't resumed remotely). Their ids go in `_LONG_RUNDIR_JOB_IDS`, which the path-length error names, with `alomancy cancel` (it marks them `failed`) as the way out. `alomancy cancel` searches for both layouts when deleting the run's remote folder (`remote_short_run_dir`, with the host from the run lock via `run_host`, `cli/cancel.py`).
- **Job names**: `ase_remote_submitter` names DFT jobs with `dft_job_name` (`sp_vasp_0`, `go_qe_3`), without `run_` (ExPyRe adds its own) or the section name.
- **Check**: `check_remote_path_length` (`remote_submission/submitters.py`) raises above the evaluator's own limit, its registry entry point `max_output_dir_len` (`MAX_OUTPUT_DIR_LEN` in the evaluator module: `run_vasp` 240, for VASP's Fortran path buffer; `run_qe` `None`, no check, since QE has no such limit). `ase_remote_submitter` takes it as a required `max_output_dir_len` argument. `projected_remote_output_dir` gives the exact length before submission (hash and suffix have fixed lengths). The workflow checks an early estimate at construction (`_validate_remote_dft_path_length`: the last loop that runs DFT, `al_loop_{num_of_al_loops - 1}`, or `initialization`; `batch_9`; an index as wide as `num_of_structures_per_loop`, at least 3 digits), and `ase_remote_submitter` checks every job's exact path before submitting any.
- **No whitespace in the `alomancy-rsh` path**: `use_noninteractive_ssh` raises `ValueError` if it has any, since ExPyRe splits `remsh_cmd`/`EXPYRE_RSH` on whitespace and every background ssh and rsync would fail.

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
