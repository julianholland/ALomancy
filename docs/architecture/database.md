# Global database

`src/alomancy/database/global_database.py`: `GlobalDatabase`, the persistent store of every DFT-evaluated structure (`REF_energy`/`REF_forces`), at `results/global_database/` by default (`general.db_path`).

## sage_lib storage

A sage_lib `Partition` with `storage="hybrid"` (HDF5 + SQLite). It persists `atoms.info`, **not** `atoms.arrays`.

- `REF_forces` round-trips natively via `AtomPositionManager.configure(total_force=...)` / `apm.forces`.
- `REF_stresses` has no native slot, so `_prepare_for_storage` writes it to `info["_REF_stresses"]` (`np.asarray(stress).tolist()`) and `_atoms_from_container` restores it (`np.array(meta.pop("_REF_stresses"))`).
- `_atoms_from_container` strips every `char_vec*` key (cached descriptors never reach `atoms.info` or train files) and calls `drop_unwritable_info` (below).
- `atoms_from_container(container, *, labels=True)` is the module-level converter (`GlobalDatabase._atoms_from_container` calls it). With `labels=False` it rebuilds geometry only: no `REF_*`, no calculator, no metadata. Use that for another program's storage whose energies aren't DFT labels (EZGA's). `read_storage(path)` opens a sage_lib hybrid storage that sits directly in `path` (EZGA's layout, not `GlobalDatabase`'s `hybrid_local/`) read-only, and returns its containers.

**A DB replaced under a live run is restored on the next write.** When another process opens the DB read-write while a run holds it, sage_lib's `HybridStorage` reads HDF5's lock error as corruption: it renames the live `db_objects.h5`/`db_index.sqlite` (and its -wal/-shm files) to `*.corrupted_<ts>` and creates empty ones. The run's handles still point at the renamed (often since deleted) files, so its writes succeed but never reach disk. Every write method is wrapped in `_guarded_write`:
- **Detect:** `db_replaced()` compares the (device, inode) of each path with the one recorded when it was opened (`_open`).
- **Restore:** `_recover_from_open_handles` commits, checkpoints SQLite (`wal_checkpoint(TRUNCATE)`), moves whatever is at the paths to `*.replaced_<ts>` (never deleted; SQLite sidecars become `<db>.replaced_<ts>-wal`/`-shm`, so the copy still opens with its WAL), copies the open files back through their descriptors (`_fds_by_inode` lists them from `/proc/self/fd` on Linux, `/dev/fd` on macOS; `_copy_fd` reads with `os.pread` so the open descriptor's position doesn't move), reopens, checks the structure count, and logs an ERROR (event `database_recovered`).
- **Failed write:** if it raises and the DB turns out to be replaced, it recovers and retries once, provided no structures were added.
- **Open files not found:** a `RuntimeError` saying how to copy them back by hand (`/proc/<pid>/fd` on Linux, `lsof -p <pid>` elsewhere).

This uses sage_lib internals (`partition._store`, its `_h5`, `_conn`, `h5_path`, `sqlite_path`), kept to those methods.

Construction is SQLite fsync-bound (3–6 s on btrfs, ~5 ms on tmpfs); tests share instances and use `clear()` (see the test notes in CLAUDE.md).

## Identity, dedup, counts

- Every structure carries `atoms.info["global_db_id"]`, a zero-based index assigned at insertion and stable across sessions; `assign_global_db_ids()` backfills older DBs.
- `add_structures()` dedups `IsolatedAtom` and `init_MP` by `(config_type, formula)`; everything else is always added.
- `count_all_by_config_type_and_formula()` is one O(N) scan returning `{config_type: {formula: count}}`; use it instead of repeated `count_by_config_type_and_formula()` calls.
- `get_train_atoms`/`get_test_atoms` skip quality-filtered structures by default (`exclude_quality_filtered=True`); `exclude_duplicates=True` skips redundancy-flagged ones. The training-set rule is `is_in_training_set(metadata)`, shared with redundancy removal.
- Metadata writes from outside the class go through `set_metadata_bulk` (split filters, curation), so they're guarded like the rest. `clear_duplicate_flags(config_types=None)` undoes redundancy flags; the workflow never calls it (each redundancy pass rewrites the flags of the types it deduplicates itself, `core_workflow.md`).

## Descriptors, predictions, imports

- `store_descriptors` caches redundancy descriptors under `char_vec_128` (`core_workflow.md`).
- `store_model_predictions` / `get_model_predictions(loop_idx, fit_idx, e0=None)`: per-loop, per-fit predictions under `model_energy_loop_{N}_fit_{i}` / `model_forces_loop_{N}_fit_{i}`; with `e0`, energies come back as formation energy per atom. `database/migrate_mace_prediction_keys.py` migrates the pre-1.0 `mace_*` keys and sentinel.
- `get_isolated_atom_energies()` builds `{symbol: REF_energy}` from the `IsolatedAtom` structures.
- `import_from_database`, `record_import`/`is_imported`/`imported_sources.json`: see `core_workflow.md` (start_from).

## Empty `info` values and extxyz

**Empty `atoms.info` values don't survive two extxyz round trips**: ASE writes `[]` as `"_JSON []"`, reads it back as an empty array, writes that as a bare `key=`, and reading a bare `key=` takes the next `key=value` as its value. `quality_filter_reasons=[]` swallowed `model_energy` in every `test_pred.xyz`, so test parity plots vanished. `utils/clean_structures.drop_unwritable_info` strips empty lists/tuples/arrays/strings; `_atoms_from_container` and `ALomancyTrainer.evaluate` call it, and **any new path that writes xyz from atoms not taken from the DB should too**. Older files are still readable: `mlip/base.read_prediction_file` (used by `read_predictions` and `mlip_plots._parse_eval_xyz`) moves a swallowed `model_energy` back (`recover_swallowed_model_energy`, event `predictions_recovered`).

All file I/O is extxyz via ASE's `read`/`write`.
