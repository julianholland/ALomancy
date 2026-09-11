# Remote submission architecture: submitters, executor, and where it actually breaks

This document is a from-source audit of `src/alomancy/remote_submission/` (`submitters.py`, `executor.py`), its immediate dependency `src/alomancy/configs/remote_info.py`, and the one other module that submits jobs its own way (`src/alomancy/mlip/mace_wfl.py`). It exists because this subsystem has caused the majority of production incidents on the `yun_an` run: silent local-disk exhaustion, a corrupted parity-plot dataset, a permanently-lost committee model that crashed a live run, and (per an earlier deep review referenced in project history) several still-open architectural gaps around unbounded ssh hangs and non-retried transient failures.

Every claim below is anchored to a specific file:line or a specific test name — nothing here is inferred from memory or documentation comments alone.

> **Status (2026-08-15): all Tier 1/Tier 2 fixes from §9's strategy memo, and its full sequencing plan (steps 0–4, §9.3), have been implemented and are covered by regression tests.** The gaps and case studies below are kept as-written — they're the evidence trail that justified each fix — but are now historical: §6's case studies describe bugs that have since been fixed, and §8's gap list is annotated with each item's resolution. Read this document as "what was wrong and why" plus "what's now true instead," not as an open TODO list. Anything not explicitly marked fixed below is still open.

## 1. Architecture overview

```
core/standard_active_learning.py
   │  (train_mlip, generate_structures, high_accuracy_evaluation)
   ▼
remote_submission/submitters.py          ◄── one function per AL phase
   │  committee_remote_submitter()           (mlip_committee)
   │  md_remote_submitter()                  (structure_generation: MD)
   │  all_maces_remote_submitter()           (structure_generation: uncertainty scoring)
   │  ase_remote_submitter()                 (high_accuracy_evaluation: DFT SP/GO)
   ▼
remote_submission/executor.py :: RemoteJobExecutor
   │  submit_job() / submit_multiple_jobs()  → build one expyre.func.ExPyRe per structure/fit
   │  run_all_jobs_bounded()                 → ThreadPoolExecutor, one thread per in-flight job
   │  _run_single_job()                      → job.start() + job.get_results(), never raises
   │  cleanup_jobs()                         → mark_processed() + wipe stage dir for successes
   ▼
expyre.func.ExPyRe (third-party, not owned by this repo)
   │  .start()        → ssh mkdir/stage-in/sbatch
   │  .get_results()  → polls squeue + rsyncs results back, blocking, until done/timeout
   ▼
HPC scheduler (Slurm) on the configured host
```

`mlip/mace_wfl.py` also constructs an `ExPyRe` job directly (`_mace_fit_expyre_call`), bypassing `RemoteJobExecutor` entirely — see §5, this is dead code and not part of the real call path.

Everything in this subsystem exists to solve one problem: run a Python function on a remote HPC node via Slurm, for a list of independent inputs (committee fits, MD seeds, DFT structures), with bounded concurrency, without blocking the whole process on a single stuck job, and without losing partial work when a job dies.

## 2. `RemoteInfo` (`configs/remote_info.py`)

A plain data container passed into every submitter and into `RemoteJobExecutor`. Built once per AL phase by `get_remote_info(job_dict, input_files=None)` (`remote_info.py:188-225`), which reads `job_dict["hpc"]` (an HPC profile from `~/.alomancy/hpc_config.yaml`) and `job_dict["max_time"]`.

**Fields that are actually consumed downstream** (verified by grep + read of every consumer):
- `sys_name`, `job_name`, `resources`, `pre_cmds`, `post_cmds`, `env_vars`, `input_files`, `output_files`, `header_extra`, `exact_fit`, `partial_node` — all passed straight into `ExPyRe(...)`/`job.start(...)` in `executor.py:347-357` and `439-445`.
- `timeout`, `check_interval` — passed into `job.get_results(...)` (`executor.py:451-454`).
- `max_concurrent_jobs` — caps the `ThreadPoolExecutor` in `run_all_jobs_bounded` (`executor.py:557-565`).
- `lock_timeout` — bounds how long a worker thread waits for the per-host ssh lock (`executor.py:430-436`, `622-626`).

**Fields that are still dead** — accepted by the constructor, documented in the docstring, stored on `self`, and **never read by `RemoteJobExecutor` or any submitter**:
- `hash_ignore` (`remote_info.py:69-71`, `93`, `128`) — grep across `src/` finds zero readers. Not fixed; candidate for deletion per §9.3's explicit recommendation.
- `num_inputs_per_queued_job` (`remote_info.py:25-27`, `78`, `112`) — only referenced in `RemoteInfo.__str__` (`remote_info.py:132`); never passed to `ExPyRe(...)`. Not fixed; same deletion candidate.
- `ignore_failed_jobs` (`remote_info.py:48-49`, `89`, `124`) — the *only* reader anywhere in `src/` was the dead `_mace_fit_expyre_call` (§5), which is never called. Still unused by live code; docstring updated to say so explicitly rather than describing behavior that doesn't exist.

**FIXED — no longer dead:** `resubmit_killed_jobs` (`remote_info.py:50-54`) is now read by `RemoteJobExecutor._run_single_job`: when a job is declared definitively dead (`ExPyReJobDiedError`), and this flag is `True` (default `False`), one fresh replacement is submitted via `job.start(force_rerun=True)` instead of giving up immediately. See §3.3 and §9.2's "resume, not resubmit" distinction — this flag governs only the resubmit path, not the always-on transport-failure resume.

## 3. `RemoteJobExecutor` (`remote_submission/executor.py`)

### 3.1 `submit_job(function, function_kwargs, input_files=None, output_files=None, job_name=None, **expyre_kwargs) -> ExPyRe`
Builds and records (in `self.jobs`) one `ExPyRe` object. Falls back to `self.remote_info.input_files`/`.output_files` when the per-job lists are empty (`executor.py:342-345`) — **this fallback path has no direct test**; every existing test supplies explicit `output_files` per job.

### 3.2 `submit_multiple_jobs(function, job_configs, common_input_files=None, common_output_pattern=None, job_name_pattern=None) -> list[ExPyRe]`
Iterates `job_configs`, building one job per entry via `submit_job`. Each `config` dict is expected to have `"function_kwargs"` — accessed as a **bare subscript** (`config["function_kwargs"]`, `executor.py:396`), so a malformed config raises an uncaught `KeyError` deep inside submission, not a clear validation error at the call site. `common_output_pattern.format(job_id=i)` uses the config's **position** in the list as `job_id` — this is the exact mechanism that caused the historical `md_remote_submitter` restart bug (§4.2) and is structurally the same risk in `committee_remote_submitter`'s `fit_indices` path (§4.4).

### 3.3 `_run_single_job(index, job) -> tuple[int, Any]`
The core unit of work, run inside one `ThreadPoolExecutor` worker thread. Sequence: acquire per-host ssh lock → `job.start(...)` → release lock → `_get_results_with_resume(job, ...)` (§3.3.1) → log stdout/stderr at DEBUG → compute queue time from `_expyre_job_started`'s mtime → return `(index, result)`.

**FIXED — no retry gap.** Previously *any* exception from `get_results()` (ssh lock timeout, `get_results()` failure, a transient rsync error) was caught at the top level, logged, salvaged via `_salvage_partial_output`, and converted into `(index, None)` with zero distinction between "unrecoverably broken" and "five-second transient blip" (see §6.1's original case study). `_run_single_job` still never lets an exception propagate out of itself — that invariant is unchanged — but the exception is now handled in two layers before that outer catch-all is ever reached:

#### 3.3.1 `_get_results_with_resume(job, timeout, check_interval, job_display_index) -> tuple[Any, str, str]`
Module-level function (`executor.py`). Calls `job.get_results(...)` in a loop; on exception, checks `job.status`:
- If `job.status` is still in `_ONGOING_JOB_STATUSES` (`{"created", "submitted", "started"}`) — meaning expyre raised from transport (e.g. inside `sync_remote_results_status()`) *before* ever updating status for that poll — this is resumable: re-enter `get_results()` on the **same job object**, never resubmitting. `ExPyReTimeoutError` gets exactly 1 extra attempt; any other exception gets up to 5, with exponential backoff starting at 5s (`_TRANSPORT_RETRY_LIMIT`, `_TIMEOUT_RETRY_LIMIT`, `_TRANSPORT_RETRY_BACKOFF_SECONDS`).
- If `job.status` has already left that set (`'failed'`/`'died'` — expyre sets status *before* raising in both of its own terminal-failure paths), the exception re-raises immediately on the very first occurrence — identical to the pre-fix behavior for genuinely terminal failures.

This distinguishes "the remote Slurm job is still alive, this was a local/network hiccup" from "expyre has already concluded this job is done for," using `job.status` (not exception type) as the signal — see §9.2 for why this is more reliable than branching on exception class.

#### 3.3.2 Resubmit-on-death, gated behind `resubmit_killed_jobs`
`_run_single_job` wraps the `job.start()` + `_get_results_with_resume(...)` sequence in a local closure (`_start_and_wait(force_rerun)`), called once normally. If `_get_results_with_resume` raises `ExPyReJobDiedError` (the job reached `status == 'died'`, i.e. genuinely gone — not the resumable case above) and `remote_info.resubmit_killed_jobs` is `True`, it's called a second time with `force_rerun=True`, submitting one fresh replacement job. Off by default, per §9.2's reasoning: a scheduler-killed job (OOM, walltime, node failure) often needs different resources, not just an identical retry.

Both layers are tested with a real-object fake (`_FakeExPyReJob` in `test_utilities.py`, extended this session with a `get_results_side_effects` queue that can simulate either a transport-only exception — status unchanged — or a `(exception, status)` pair matching expyre's real "set status, then raise" terminal paths): `test_transport_failure_resumes_on_same_job_and_succeeds`, `test_transport_failure_gives_up_after_retry_limit`, `test_timeout_exception_only_gets_one_resume_attempt`, `test_terminal_failure_status_is_not_retried`, `test_died_job_resubmitted_when_resubmit_killed_jobs_enabled`, `test_died_job_not_resubmitted_by_default`.

### 3.4 `_salvage_partial_output(index, job)` (static method)
Best-effort recovery for a job that died/failed: reads the `_expyre_output_files` marker ExPyRe itself wrote at job construction, and for each registered output path calls `ExPyRe._copy(job.stage_dir, Path.cwd(), out_file)`. Never raises — a copy failure (nothing was ever written) is logged at DEBUG and ignored (`executor.py:531-537`).

Well-tested (`TestSalvagePartialOutput`, 4 tests, `test_utilities.py:555-654`): copies existing output, no-ops when the marker is absent, no-ops when the referenced output never materialized, and is exercised end-to-end via a real `_run_single_job` failure.

**Limitation this doesn't cover, and doesn't need to post-fix:** salvage only recovers whatever the remote side had *already written* at the moment the local exception fired — if a job dies 6 epochs into a 200-epoch training run, there's no model file yet to salvage. §6.2 describes an incident that hit exactly this. With §3.3's resume fix in place, a *transient* failure (the actual cause in §6.2) no longer reaches this fallback at all — the job resumes and finishes normally instead of being abandoned mid-training. Salvage's blind spot here is now only relevant for genuinely terminal deaths this early, not for the transient case that used to masquerade as one.

### 3.5 `run_all_jobs_bounded() -> list[Any]`
`ThreadPoolExecutor(max_workers=min(max_concurrent_jobs, len(self.jobs)))`, one `_run_single_job` task per job, collected via `as_completed` so results land index-aligned in a pre-sized `[None] * len(self.jobs)` list regardless of completion order. Falls back to `max_workers=1` with a warning if `max_concurrent_jobs` is `None`/`0`/negative (`executor.py:557-564`) — **this fallback has no direct test**.

There is also a second, outer exception guard (`executor.py:578-581`, `except Exception as exc: logger.error("Job %d worker raised unexpectedly")`) for the case where `_run_single_job` itself fails to catch something (e.g. `future.result()` raising `CancelledError`). This path is unreachable under normal operation since `_run_single_job` is designed to swallow everything — it has no test, and would be very hard to trigger deliberately.

Well-tested for its core scheduling guarantees: never exceeds concurrency cap, one failure doesn't block others, results stay index-aligned regardless of completion order, a finishing job frees its slot promptly for the next queued one.

### 3.6 Concurrency/locking machinery
Three separate serialization mechanisms, each solving a different observed production failure:

| Mechanism | Guards | Why | Evidence it was needed |
|---|---|---|---|
| `_get_ssh_call_lock(sys_name)` / `_acquire_ssh_call_lock_or_raise` (per-host `threading.Lock`) | `job.start()` and every `sync_remote_results_status()` call | Concurrent ssh sessions past the host's session cap silently fall back to a fresh, unauthenticated connection that hangs forever on an unattended password/OTP prompt | Documented production incident, module docstring `executor.py:117-135` |
| `_ensure_expyre_sync_serialized` (monkeypatches `ExPyRe.sync_remote_results_status`, with a per-host generation counter to skip redundant syncs) | The squeue-status + rsync call inside `get_results()`'s poll loop | A stuck `squeue` subprocess observed in production from many threads calling it simultaneously | `executor.py:196-226` |
| `_ensure_expyre_db_thread_safe` (reopens expyre's sqlite connection `check_same_thread=False`, wraps `_execute` to fully materialize the cursor before releasing the lock) | `expyre.config.db`, a single module-level sqlite connection | Two threads racing `execute()`/cursor-iteration on the same connection silently corrupted a still-pending read, producing a bare `IndexError` in production | `executor.py:256-309`, regression test `test_add_then_immediate_jobs_lookup_never_empty_under_concurrency` |

All three are well-covered by targeted concurrency tests using real `threading.Thread`s and wall-clock timing (not mocks) — this is the best-tested part of the subsystem.

### 3.7 `cleanup_jobs()`
Added earlier this session. For every job with `status == "succeeded"`, calls `job.clean(wipe=True)` (deletes local *and* remote stage directory) under the same per-host lock; failed/died jobs are left untouched.

**FIXED (§9.3 step 0, an issue Opus's review caught that the original audit missed):** `mark_processed()` is no longer called unconditionally. It's now skipped for any job still in `_ONGOING_JOB_STATUSES` after `_run_single_job` gives up on it (i.e. resume retries were exhausted but the job might still be genuinely running remotely) — `mark_processed()` sets `status='processed'`, which expyre's own `can_produce_results` group excludes, so marking an abandoned-but-possibly-alive job processed would have permanently prevented a future restart from reattaching to it, even if the underlying Slurm job went on to finish successfully on its own. `succeeded`/`failed`/`died` jobs (all terminal) are still marked processed exactly as before.

Tested: `test_cleanup_marks_terminal_status_jobs_processed` (succeeded + failed both get marked), `test_cleanup_skips_mark_processed_for_still_ongoing_job` (new, this fix), `test_cleanup_wipes_succeeded_job_stage_dir`, `test_cleanup_leaves_failed_job_stage_dir_untouched`. **Still not tested**: the `except Exception as exc: logger.debug(...)` path when `job.clean()` itself raises (e.g. the remote host is unreachable at cleanup time), and the ssh-lock-timeout path during cleanup specifically.

### 3.8 `run_and_wait(function, job_configs, **kwargs) -> list[Any]`
Trivial composition: `submit_multiple_jobs` → `run_all_jobs_bounded` → `cleanup_jobs` → return results. This is what every submitter actually calls; it has no dedicated unit test of its own, but every submitter-level test that mocks `RemoteJobExecutor.run_and_wait` (§4, now real tests rather than mocked-away callers) implicitly documents its contract (one call, returns index-aligned results).

## 4. `submitters.py` — one function per AL phase

All four functions share the same shape: build `job_configs` (a list of per-job dicts), construct a `RemoteJobExecutor`, call `run_and_wait`. None of them retry, backfill, or validate their own results beyond what's described below — result validation happens in the *caller* (`standard_active_learning.py`).

### 4.1 `ase_remote_submitter(remote_info, base_name, input_atoms_list, function=None, per_structure_function=None, batch=0, function_kwargs=None) -> None`
Used by `high_accuracy_evaluation` for DFT single-point/geometry-optimization jobs, one job per structure. `per_structure_function`, when given, must match `len(input_atoms_list)` (validated, raises `ValueError` — `submitters.py:42-48`) and lets GO/SP structures share one submission queue, distinguished by a `{fn.__name__}_{job_name}_{i}` job name.

**FIXED — direct test coverage added** (`tests/remote_submission_tests/test_submitters.py::TestAseRemoteSubmitter`, new file/directory this session): `test_per_structure_function_length_mismatch_raises` exercises the `ValueError`, `test_per_structure_function_picks_go_or_sp_per_structure` exercises the per-structure job-name construction (`{fn.__name__}_{job_name}_{i}`) and confirms each job config points at the correct function. `test_standard_active_learning.py` still only imports `ASE_OUTPUT_PREFIX` and mocks the function when testing its caller — that's fine, since the submitter's own logic is now tested directly instead.

### 4.2 `md_remote_submitter(remote_info, base_name, target_file, input_atoms_list, function=None, function_kwargs=None) -> list[str]`
Used by `generate_structures` to run one MD trajectory per seed structure. Idempotent: globs `md_output_*/{target_file}` first and skips/reuses however many already exist, only submitting jobs for the remainder.

This is the **best-tested submitter function** — `test_structure_generation.py:1477-1533` (`test_md_remote_submission_offsets_output_dirs_past_existing_runs`) is a real regression test for a genuine historical bug: `output_files` must be keyed by `n_existing + i` (the actual directory the remote job writes to), not by the job's position `i` in the *sliced* remaining-jobs list, or ExPyRe's stage-out glob fails even though the job succeeded. The docstring at `submitters.py:111-119` explains the bug precisely.

The *other* test of this function, `test_md_remote_submission` (`test_structure_generation.py:1443-1474`), is a mock-testing anti-pattern: it patches `md_remote_submitter` itself and asserts the mock returns what the test told it to return. It provides zero real coverage and is also missing `@pytest.mark.unit` (confirmed: no class-level marker on `TestMolecularDynamics`, and this specific method has none — it is silently excluded from `pytest -m unit` runs, violating this repo's own stated convention that every test method must carry the marker individually).

### 4.3 `all_maces_remote_submitter(remote_info, function=None, function_kwargs=None, job_name=None) -> dict`
Submits the single "evaluate every committee model against candidate structures" job. `job_name` defaults to `f"mace_eval_{remote_info.job_name}"` specifically to avoid colliding with the MD jobs submitted just before it in the same phase (this was the actual bug fixed in v0.6.0). Unwraps `executor.run_and_wait(...)[0]` — always safe since exactly one job is ever submitted here.

**FIXED — direct test coverage added** (`TestAllMacesRemoteSubmitter`, same new file): `test_job_name_defaults_to_mace_eval_prefixed` exercises the default naming and the single-job unwrap together (mocks only `RemoteJobExecutor.run_and_wait`, checks both the job config it received and the unwrapped return value); `test_explicit_job_name_overrides_default` covers the override path.

### 4.4 `committee_remote_submitter(remote_info, base_name, function, seed=803, size_of_committee=5, function_kwargs=None, fit_indices=None) -> None`
Submits one MACE training job per committee member. `fit_indices`, when given, retrains only specific member indices (used by `train_mlip`'s backfill logic) — critically, each job's `output_files` is keyed by its **own** `fit_idx` value (`f"fit_{i}"` for `i in indices`), *not* by its position in the `job_configs` list, specifically to stay correct for non-contiguous subsets like `fit_indices=[2, 4]` (documented at `submitters.py:200-209` as guarding against the exact same bug class §4.2 already hit once).

**FIXED — the highest-priority gap in the original audit, closed** (`TestCommitteeRemoteSubmitter`, same new file): `test_fit_indices_output_files_keyed_by_own_index_not_position` submits `fit_indices=[2, 4]` and asserts `output_files == ["fit_2"]`/`["fit_4"]` (and the corresponding `fit_idx`/`seed` in each job's `function_kwargs`) — the exact regression test the docstring's claim was missing, per §9.4's testing-strategy recommendation. `test_default_fit_indices_covers_full_committee_in_order` covers the default (no `fit_indices` given) path.

## 5. `mlip/mace_wfl.py::_mace_fit_expyre_call` — dead code

Defined at `mace_wfl.py:343-424`: builds an `ExPyRe` job with `function=_mace_fit_expyre_call` — **itself** — and calls `.start()`/`.get_results()` directly, entirely bypassing `RemoteJobExecutor` (no concurrency bounding, no ssh-lock serialization, no salvage, no cleanup).

Grep confirms this function is referenced exactly once in the entire codebase — inside its own body. It is never imported or called from `standard_active_learning.py` or anywhere else. The actual, live remote-training path is `committee_remote_submitter(..., function=mace_fit, ...)` (`mace_wfl.py:176-340`, called from `standard_active_learning.py:388` and `:443`) going through `RemoteJobExecutor` normally. `_mace_fit_expyre_call` is pure dead code that also happens to be the only live reader of `RemoteInfo.ignore_failed_jobs` (§2) — removing it would make that field dead too, which it effectively already is.

## 6. Case studies: how these gaps actually manifested (historical — both fixed, kept as the evidence trail)

### 6.1 The no-retry gap (§3.3) — the architectural root cause behind most incidents this run — **FIXED, §3.3.1**
`_run_single_job` treated *any* exception from `get_results()` — including the well-documented transient "stale file handle"/"file has vanished" rsync race between MACE's own checkpoint rotation and the periodic mid-training sync (already known well enough to have a targeted `FailedSubprocessWarning` filter in `alomancy/__init__.py`) — as a permanent, unretried job failure. `RemoteInfo.resubmit_killed_jobs` (§2) existed to describe the missing behavior and was wired nowhere. Now implemented as `_get_results_with_resume` — see §3.3.1 for the fix and §9.2 for the design reasoning (resume, not resubmit).

### 6.2 `fit_0`'s model permanently missing — **FIXED**
Traced directly: `fit_0`'s committee-training job hit the exact rsync race above at epoch 6 of 200, `_run_single_job` gave up permanently, `_salvage_partial_output` recovered only the sparse early-training scraps (no model file existed yet — §3.4's known limitation). `fit_1`/`fit_3`/`fit_4` completed normally. Because `train_mlip`'s backfill only retried when fewer than 3 fits succeeded (3 succeeded here), `fit_0` was never retried and stayed permanently broken for the loop. `select_best_committee_model` (`mlip/get_mace_eval_info.py`, outside this subsystem but downstream of it) then defaulted to `fit_0` with no existence check when no committee member had a readable test-metrics file, and `generate_structures` crashed trying to stage `fit_0`'s nonexistent model as an MD job input.

Three independent fixes now cover this chain: (1) §3.3.1's resume means a transient failure like this no longer abandons the job at all; (2) `train_mlip`'s backfill (`core/standard_active_learning.py`) now retrains every missing fit whenever any are missing, not just enough to reach 3 (the old `missing_fit_indices[: 3 - len(found_fit_indices)]` slice is gone); (3) `select_best_committee_model` (`mlip/get_mace_eval_info.py`) now only considers fits whose `{name}_stagetwo.model` file actually exists — both when scoring by metric and in the no-metrics fallback — and raises a clear `ValueError` rather than ever returning a nonexistent path. Regression tests: `TestSelectBestCommitteeModel::test_falls_back_to_lowest_indexed_fit_with_model_when_fit_0_has_none`, `test_raises_when_no_fit_has_a_model` (`tests/mlip_train_tests/test_mace_training.py`).

### 6.3 The disk-exhaustion → corrupted-parity-plot chain — **FIXED (two separate root causes, both addressed)**
`cleanup_jobs()` not existing until this session meant every job's full stage directory (mlip_committee checkpoints included) accumulated forever — 297 leftover directories, 249GB, causing repeated `No space left on device` during later jobs' mid-training syncs. One such disk-full event caused a job's `get_results()` to fail (§6.1's gap), and the local copy of that fit's prediction file ended up stale/corrupted rather than reflecting the run's actual successful remote output — degrading parity plots to a single trivial point with no error anywhere pointing at the real cause, until direct log/filesystem archaeology traced it back.

That symptom (parity plots showing one point) turned out to have a **second, independent root cause** discovered after the above was fixed and a fresh loop still showed the same one-point pattern: `_save_mace_eval_predictions` (`mlip/mace_wfl.py`) was evaluating the TorchScript-**compiled** model, whose force computation (`torch.autograd.grad` through the TorchScript interpreter) raised `RuntimeError: ... Global alloc not supported yet` for every multi-atom structure on this cluster's GPU/CUDA/PyTorch combination — only the trivial single-atom case ever succeeded, on every fit, every loop, for the run's entire history. Fixed by evaluating the **uncompiled** model instead (eager-mode execution doesn't go through the TorchScript interpreter's op dispatch at all); confirmed against the live `yun_an` database that only 1 of 1940 stored structures had ever received a `mace_energy_loop_*`/`mace_forces_loop_*` prediction before this fix.

Also fixed independently: `mlip_committee`'s *local* checkpoint directories (a related but distinct disk-usage source — see `cleanup_local_committee_checkpoints` in `mlip/mace_wfl.py`, called from `train_mlip`) accumulate via expyre's additive-only (`delete=False`) mid-training sync regardless of the remote-side cleanup added earlier this session, since that only ever touches the remote copy.

## 7. Test coverage matrix

| Function | Direct test? | Quality |
|---|---|---|
| `RemoteJobExecutor.submit_job` | Indirect only | Input/output-file fallback to `remote_info` defaults untested |
| `RemoteJobExecutor.submit_multiple_jobs` | Indirect only | Malformed-config `KeyError` path untested; "job config validation" tests are vacuous (test plain dicts, never call real code) |
| `RemoteJobExecutor._run_single_job` | Yes, extensively | Strong — real threads, real timing, failure isolation, stdout/stderr logging |
| `RemoteJobExecutor._salvage_partial_output` | Yes | Strong, 4 tests covering success/no-marker/no-material/integration |
| `RemoteJobExecutor.run_all_jobs_bounded` | Yes | Strong for scheduling guarantees; invalid-`max_concurrent_jobs` fallback and outer worker-exception guard untested |
| `RemoteJobExecutor.cleanup_jobs` | Yes (4 tests) | Terminal-status + ongoing-status-skip branches covered; exception-during-wipe path still untested |
| `RemoteJobExecutor._get_results_with_resume` | Yes (6 tests) | New this update — transport resume, retry-limit exhaustion, timeout's tighter bound, terminal-status no-retry, both resubmit branches |
| `RemoteJobExecutor.run_and_wait` | Indirect only | No dedicated test, but implicitly exercised everywhere |
| ssh-lock / db-thread-safety machinery (§3.6) | Yes, extensively | Best-tested part of the subsystem |
| `submitters.ase_remote_submitter` | **Yes (2 tests)** | Fixed — length-mismatch `ValueError` and per-structure job naming/function selection |
| `submitters.md_remote_submitter` | Yes (2 tests) | One real regression test (good); one vacuous mock-test missing `@pytest.mark.unit` (not yet cleaned up — still gap #6) |
| `submitters.all_maces_remote_submitter` | **Yes (2 tests)** | Fixed — job-name default and override |
| `submitters.committee_remote_submitter` | **Yes (2 tests)** | Fixed — the `fit_indices` output-path regression test this audit called for is now in place |
| `mace_wfl._mace_fit_expyre_call` | **None** | Still dead code; untestable-by-design self-recursion (deletion deliberately deferred, see §9.3) |

## 8. Summary of concrete gaps (with resolutions)

1. **~~No retry for transient remote failures.~~ FIXED.** `_run_single_job` now resumes on the same job for transport-only failures via `_get_results_with_resume` (§3.3.1); `RemoteInfo.resubmit_killed_jobs` now actually gates one resubmit attempt for definitively-died jobs (§2, §3.3.2). Was the root cause of the two largest production incidents this run.
2. **~~`committee_remote_submitter` has zero test coverage~~ FIXED.** `TestCommitteeRemoteSubmitter` (`tests/remote_submission_tests/test_submitters.py`) now includes the `fit_indices` position-vs-index regression test its docstring called for.
3. **~~`ase_remote_submitter` and `all_maces_remote_submitter` have zero test coverage~~ FIXED.** `TestAseRemoteSubmitter`, `TestAllMacesRemoteSubmitter` (same new file).
4. **Four `RemoteInfo` fields were dead.** `resubmit_killed_jobs` is now live (item 1). `hash_ignore`, `num_inputs_per_queued_job`, `ignore_failed_jobs` (read only by dead code) remain dead — deliberately not deleted yet, per §9.3's "don't remove `ignore_failed_jobs` until/unless it's wired up" guidance; `hash_ignore`/`num_inputs_per_queued_job` are still open deletion candidates.
5. **`mlip/mace_wfl.py::_mace_fit_expyre_call` is still dead, self-recursive code** that bypasses `RemoteJobExecutor`'s concurrency bounding, ssh-lock serialization, salvage, and cleanup entirely. Deliberately not removed — §9.3 recommended a cheap landmine guard (`raise NotImplementedError`) over a diff through the MLIP path for zero behavioral gain; that guard has not been added yet either. Still open.
6. **Vacuous/mock-testing-anti-pattern tests still present**: `test_remote_job_executor_initialization`, `test_job_config_validation`/`test_invalid_job_config`, `test_md_remote_submission`. §9.4 recommended deleting rather than fixing these; not yet done. Still open.
7. **`test_md_remote_submission` still missing `@pytest.mark.unit`.** Not yet fixed (tied to item 6 — deleting the test resolves this too). Still open.
8. **`submit_multiple_jobs`'s `config["function_kwargs"]` bare subscript** still raises an unhelpful `KeyError` rather than a clear validation error for a malformed job config. Explicitly triaged as skip-worthy in §9.4 ("untestable-or-trivial, none have ever fired"). Still open, low priority.
9. **~~`train_mlip`'s backfill threshold~~ FIXED.** Now retrains every missing fit whenever any are missing (`core/standard_active_learning.py`), not just enough to reach the floor of 3 — that floor is now only the post-retry hard-failure check, not the retry trigger.
10. **~~`select_best_committee_model`'s fit-0 fallback~~ FIXED.** Only considers fits with an existing model file, in both the scoring loop and the fallback; raises `ValueError` if genuinely none exist. See §6.2.

## 9. Opus review — strategy memo

The audit above was reviewed by Opus, which spot-checked the load-bearing claims directly against `expyre`'s source (not just this repo's usage of it) before writing the following. Its central correction: the right fix for gap 1 is **resume, not resubmit** — `expyre`'s `get_results()` is a safely re-enterable poll loop, and its exceptions are already distinguishable (transport-only vs. a definitively dead job vs. a deterministic remote failure) once you actually look, which changes the fix from a risky retry-with-resubmission design to a much smaller and safer one. It also found a real, unreported issue introduced by this session's own `cleanup_jobs` change (§9.3, item 0).

### 9.1 Prioritization

**Tier 1 — actually caused the incidents:**
- **Gap 1** (no retry/resume on transient failure) — both the fit_0 crash and the corrupted-parity-plot incident are the same event: `get_results()` raised from *transport*, not from the job itself.
- **Gap 9** (backfill capped at 3) — turned a single transient loss into a *permanent* one. With 3 fits present, `standard_active_learning.py`'s backfill never fires at all.
- **Gap 10** (fit-0 fallback, no existence check) — the cheapest fix on the list and the literal crash site.

These three are a chain, not independent bugs: fixing 9 and 10 alone would have downgraded last week's crash from "run dies" to "committee runs at 4/5 for one loop."

**Tier 2 — real risk, no incident yet:** Gap 2 (`committee_remote_submitter` untested, including the `fit_indices` index-vs-position guard) — this is the one gap that would fail *silently and wrongly* (overwriting a good model with a new one under the wrong index) rather than loudly, and the same bug class already bit `md_remote_submitter` once.

**Tier 3 — low-stakes cleanup:** gaps 3, 5, 6, 7, 8 (gap 4 partially reclassified below).

### 9.2 Resume, not resubmit

Reading `expyre/func.py` directly:
- `get_results()` is safe to re-enter. If it raises from transport, the job's status is still `submitted`/`started`, the Slurm job is still running, and calling `get_results()` again simply resumes polling.
- Its exceptions are already distinguishable: `ExPyReJobDiedError` → definitively dead (expyre already gives one extra poll before declaring this); a re-raised pickled remote exception → the *user function* failed deterministically, retrying is pointless; `ExPyReTimeoutError` → local wait expired, job status unknown-but-probably-alive; anything else (the `RuntimeError`s this run has actually hit) → local/transport only, remote job untouched.
- `expyre`'s own `subprocess_run` already retries 3×/5s internally — a failure reaching `_run_single_job` means a *sustained* problem (the 249GB disk-full window lasted minutes), not a momentary blip needing yet another retry layer on top.

So: branch on exception type in `_run_single_job` instead of collapsing everything to `return index, None`. Transport exceptions → re-enter `get_results()` on the *same* job, bounded backoff, no resubmission, no lost epochs. `ExPyReTimeoutError` → resume once more, then give up. `ExPyReJobDiedError` → salvage, then optionally `start(force_rerun=True)` gated behind `resubmit_killed_jobs` (default off). Remote user exception → salvage, fail permanently. `resubmit_killed_jobs` is a legitimate knob only for the "definitely dead" case — it would not by itself have saved fit_0.

### 9.3 Sequencing — **all 5 steps implemented**

**0. ✅ DONE.** `cleanup_jobs`'s `mark_processed()` no longer runs on jobs still in an ongoing status (§3.7).

**1. ✅ DONE.** Gap 10 guard (`select_best_committee_model` existence check) + gap 9 backfill-to-full-size (`train_mlip`) — §6.2, §8 items 9/10.

**2. ✅ DONE.** Resume-on-transport-failure in `_run_single_job` via `_get_results_with_resume` — §3.3.1. Lands inside `run_all_jobs_bounded`, before `cleanup_jobs` runs, as specified.

**3. ✅ DONE.** Regression test for `committee_remote_submitter(fit_indices=[2, 4])` — §4.4, §8 item 2.

**4. ✅ DONE.** `ExPyReJobDiedError` → `force_rerun`, gated on `resubmit_killed_jobs`, default off — §3.3.2.

**Explicitly not done, as recommended:** `resubmit_killed_jobs`/`ignore_failed_jobs` were not deleted (the former is now live; the latter deliberately left alone). `hash_ignore`/`num_inputs_per_queued_job` were **not yet deleted** either, despite being recommended for deletion — still open (§8 item 4). `_mace_fit_expyre_call` was **not** deleted and **not** given the recommended `NotImplementedError` landmine guard — still open (§8 item 5). No job-state-machine abstraction was built and no persistent retry-count store was added, matching the recommendation to avoid both.

### 9.4 Testing strategy — items 1–3 done, item 4 still open

The pattern held up: tests that mocked the boundary (the four submitters) found nothing and were exactly where the incidents lived; tests that ran real machinery against a fake remote (the ssh-lock/db-safety code) were incident-free. Of the four recommendations: (1) ✅ the real-object `_FakeExPyReJob` fixture in `test_utilities.py` was extended with a `get_results_side_effects` queue modeling realistic failure sequences — this is what made §3.3.1/§3.3.2's resume/resubmit design testable; (2) ✅ `committee_remote_submitter(fit_indices=[2, 4])` regression test added (§4.4); (3) ✅ `ase_remote_submitter`'s `per_structure_function` length-mismatch and job-naming tests added (§4.1); (4) **not done** — the vacuous tests (gap 6) were left in place rather than deleted, and still cost credibility in this same audit. Skipped as recommended: the unreachable outer worker guard, the `max_concurrent_jobs` fallback, and the `KeyError` path remain untested — still correctly judged not worth it.
