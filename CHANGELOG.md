# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] - 2026-07-29

### Fixed
- **MD-generated structures could be incorrectly routed to geometry optimisation**: `atoms.info["needs_relaxation"]` set on amorphous initialization structures was never cleared after their first relaxation+DFT pass, and survived every subsequent `.copy()` — into the GlobalDatabase, onto MD seed structures selected from the training set, and through every MD trajectory frame. If such a structure was later picked as an MD seed, all high-SD structures generated from that trajectory inherited `needs_relaxation=True` and were sent to GO instead of single-point evaluation in `high_accuracy_evaluation`. `generate_structures` (`core/standard_active_learning.py`) now explicitly forces `needs_relaxation=False` on every returned high-SD structure, on both the fresh-pipeline path and the cached-restart path.

## [0.4.2] - 2026-07-28

### Fixed
- **`store_mlip_predictions` no longer runs local CPU inference**: previously fell back to loading `MACECalculator` and evaluating all train+test structures locally, causing a multi-hour hang when resuming at a later AL loop. Now reads per-structure predictions from `train_pred.xyz` / `test_pred.xyz` files written by `mace_fit` on the remote GPU node. If those files do not exist (runs predating v0.4.2), writes the sentinel and logs a clear message instead of hanging.
- **`plot_dft_vs_model` no longer runs local inference as fallback**: follows the same DB → eval-file → skip chain. Blank subplots show "No predictions available" for loops that predate v0.4.2.

### Added
- **Post-training eval file writing in `mace_fit`** (`mlip/mace_wfl.py`): after `mace_run_train` finishes on the remote GPU node, the stagetwo compiled model evaluates both the training and test sets and writes `train_pred.xyz` / `test_pred.xyz` in the fit directory. These files carry `mace_energy` and `mace_forces` keys and are synced back by ExPyRe alongside the model files. `store_mlip_predictions` then reads them locally with no model loading required. **Note:** the remote machine must have the updated alomancy package installed (`pip install alomancy==0.4.2` or `pip install -e .` there) for the new eval step to run.

## [0.4.1] - 2026-07-27

### Added
- **`alomancy add-hpc` wizard** (`cli/add_hpc.py`): interactive terminal wizard that configures two files in one go — `~/.expyre/config.json` (ExPyRe scheduler entry) and `~/.alomancy/hpc_config.yaml` (ALomancy HPC profile). Wizard flow: reads available SSH aliases from `~/.ssh/config` and presents a numbered pick-list; collects Slurm partition details (name, cores, max time, memory); builds the correct CPU or GPU SBATCH header automatically; collects the venv activation command and optional `TRITON_CACHE_DIR` for GPU nodes; optionally collects DFT paths (`pwx_path`/`pp_path` for QE, `vasp_path`/`pp_path` for VASP); optionally runs `pip install alomancy` on the remote over SSH. Existing entries in both files are preserved — the wizard only adds/overwrites the named entry.
- **Global HPC config** (`configs/global_config.py`): `~/.alomancy/hpc_config.yaml` stores named HPC profiles. `_load_global_hpc_config()` reads it; path constants (`ALOMANCY_HPC_CONFIG`, `EXPYRE_CONFIG`) are importable for reuse.
- **HPC string resolution in `load_dictionaries`**: if a job section's `hpc:` value is a string (e.g. `hpc: 'raven_gpu'`), it is looked up in `~/.alomancy/hpc_config.yaml` and replaced with the full profile dict before the `jobs_dict` is returned. Dict values pass through unchanged for backwards compatibility. Raises `ValueError` with a helpful message if the named profile is not found.
- **Best committee model selection** (`mlip/get_mace_eval_info.py`): `select_best_committee_model(base_name, mlip_committee_job_dict, seed, metric="mae_f")` reads each committee member's `*_test.txt` metrics file and returns `(fit_idx, stagetwo_model_path)` for the member with the lowest test-set force MAE. `generate_structures` now uses this model as the MD base; the remaining committee members form the uncertainty comparison set. Handles both JSON-lines (newer MACE) and Python list-of-tuples (older MACE) file formats. Falls back to `fit_0` when no test metrics are readable.
- **`skip_initialization`** parameter on `BaseActiveLearningWorkflow` (default `False`): when `True` and no AL loops have completed yet, loads train/test sets directly from the GlobalDatabase and begins the AL loop without running `initialize_training_set`. Use this when the DB is already populated from a prior run and you want to resume from loop 0 without regenerating structures.
- **`remove_redundancy`** parameter on `BaseActiveLearningWorkflow` (default `True`): calls `remove_redundancy_from_partition` after initialization and after each AL loop to prune geometrically similar structures from the DB before the next training round.
- **`high_force_threshold`** parameter on `BaseActiveLearningWorkflow` (default `100.0` eV/Å): calls `remove_high_force_structures_from_partition` after initialization and after each AL loop to flag and exclude structures with unphysically large atomic forces.
- **`global_db_id`** integer tag on every structure in `GlobalDatabase`: zero-based positional index assigned at insertion time and stable across sessions (structures are never deleted from the DB). Propagated automatically into `atoms.info` by all query methods (`get_train_atoms`, `get_test_atoms`, `get_all_as_atoms`, `get_structures_by_config_type`). `assign_global_db_ids()` backfills existing DBs that predate this feature.
- **`eligible_config_types`** parameter on `split_atoms_list_into_test_and_train`: restricts which config_types are eligible to enter the test split; structures with other config_types always go to train unconditionally (e.g. `IsolatedAtom` stays in train so MACE can read E0s).

### Fixed
- **MAE plot energy/force swap**: `mae_al_loop_plot` was displaying the energy and force error lines inverted. Root cause: `get_mace_eval_info` was collecting `mae_e` (total energy per structure, ~0.2 eV) which has similar magnitude to `mae_f` (~0.2 eV/Å). Fixed throughout `get_mace_eval_info.py` and `plotting.py` by switching to `mae_e_per_atom` (~0.017 eV/atom), which now also matches the scale used in `mlip_plots.py` training-curve plots.
- `None | str` type annotation in `test_train_manager.py` corrected to `str | None` (ruff RUF036).

## [0.2.0] - 2026-07-13

### Added
- **Per-fit validation set carving**: `mace_fit` now carves a per-committee-member held-out validation set from the training data before calling MACE. Eligible structures are those with `config_type` in `mlip_committee.valid_config_types` (defaults to `initialization.test_config_types`) or `"high_sd"`. Fraction is controlled by `mlip_committee.valid_fraction` (default 0.05). When no eligible structures exist or the fraction rounds to zero, MACE is pointed directly at the shared `train_set.xyz` without writing a redundant copy. Each member uses seed `seed + fit_idx` for reproducible but distinct splits. The helper `_select_validation_split` is module-level for testability.
- **Log-scale MAE plots**: `mae_al_loop_plot` now renders the y-axis on a log scale, making error improvements across orders of magnitude visible across many AL loops.
- **Pluggable DFT backend**: `high_accuracy_evaluation` now supports multiple calculators via a `calculator` config key (default `"qe"`). The registry in `high_accuracy_evaluation/dft/__init__.py` resolves `(run_sp_fn, run_go_fn)` at runtime with lazy imports, so installing only one calculator does not break the other.
- **VASP backend** (`high_accuracy_evaluation/dft/run_vasp.py`): `run_sp_vasp` / `run_go_vasp` mirror the QE interface exactly. `create_vasp_calc_object` builds a `Vasp` calculator with sensible INCAR defaults (PBE, ENCUT=500, non-self-consistent geometry relaxation via BFGS). Monkhorst-Pack k-points derived from the cell via the shared `generate_kpts` utility.
- **Shared DFT utilities** (`utils/dft_utils.py`): `generate_kpts`, `_build_srun_command`, `_run_sp`, and `_run_go` extracted from `run_qe.py` and shared by both backends. Neither backend duplicates this logic.
- **Mismatched-kwargs warning**: `warn_mismatched_kwargs(calculator, job_dict)` logs a WARNING when config keys belonging to a different calculator (e.g. `vasp_input_kwargs` in a QE run) are present, so misconfigured YAMLs are caught early.
- **`high_sd` config_type for AL loop structures**: Structures returned by `high_accuracy_evaluation` are now tagged `config_type="high_sd"` with `atoms.info["al_loop"] = <loop>` metadata, replacing the previous `al_loop_N` per-loop types. This gives a stable, queryable label across all loops.
- **GlobalDatabase**: Persistent sage_lib Partition (hybrid HDF5+SQLite) storing all DFT-evaluated structures across AL loops. Deduplication by (config_type, formula) prevents double-adding IsolatedAtom and init_MP entries from multiple datasets. REF_forces arrays serialised into atoms.info for round-trip storage.
- **DB-aware initialization**: `compute_initialization_needs()` queries the GlobalDatabase to determine what structures still need to be generated before starting DFT. Replaces the old `d_t_s_a_ratio` + `target_non_mp_structures_to_add` API with explicit per-type counts: `num_dimers_per_combo`, `num_trimers_per_combo`, `num_amorphous`, `num_stretch_compress_per_mp`.
- **Structured logging**: All 108 `print()` calls replaced with Python `logging` module. `setup_logging(verbose, log_file)` called once at workflow construction. verbose=0 silences console; verbose=1 shows INFO progress; verbose=2 shows DEBUG per-job detail. File handler always captures DEBUG regardless of verbose level. ExPyRe job stdout/stderr captured at DEBUG.
- **Extra dataset seeding**: Extra datasets are now seeded into the GlobalDatabase *before* `initialize_training_set` is called, so `compute_initialization_needs` accounts for them and avoids regenerating already-provided structures.
- `log_file` parameter on `BaseActiveLearningWorkflow` (default: `"results/alomancy.log"`)
- `db_path` parameter on `BaseActiveLearningWorkflow` (default: `"results/global_database"`)
- Warning if structure generation could contain single-atom structures
- `"high_sd"` is automatically appended to `selectable_configs` in `select_initial_structures` so that structures from previous AL loops are always eligible for seeding MD, even when `selectable_configs` is explicitly set.

### Changed
- `train_mlip` abstract method and `ActiveLearningStandardMACE` implementation now receive the full `job_dict` (all four top-level keys) instead of `mlip_committee_job_dict`. `mace_fit` likewise takes `job_dict` so it can read `initialization.test_config_types` as the default validation config types.
- `ase_remote_submitter` replaces `qe_remote_submitter`; output directories use `ase_output_` prefix instead of `qe_output_`. The interface is otherwise unchanged.
- `run_qe.py` imports `generate_kpts` and shared runner helpers from `utils/dft_utils.py` instead of defining them locally.
- AL loop structures now carry `config_type="high_sd"` (stable across loops) rather than `config_type="al_loop_N"` (loop-specific). The originating loop is preserved in `atoms.info["al_loop"]`.
- Initialization configuration now uses per-type counts instead of ratio-based approach.
- `initialize_training_set` now has a DB-aware path that only generates missing structures.
- Logging replaces all bare print() calls; no external API change.

### Fixed
- `select_initial_structures` no longer mutates `Atoms` objects in `train_atoms_list`; selected structures are `.copy()`-d before `mark_structures_for_dft` writes to them. Previously, marking corrupted `config_type` on shared references, causing the `selectable_configs` filter to find zero matches on subsequent AL loops.
- Batch numbering collision between GO (geometry-optimisation) and SP (single-point) DFT jobs.

### Dependencies
- Added: sage-lib (HDF5+SQLite storage backend for GlobalDatabase)
- Added: polars (Polars DataFrame used in structure generation std-dev calculations)


## [0.1.1] - 2025-08-14

### Added
- Initial changelog documentation
- Optional extra dictionaries to control function behaviour inside core funcitons
- CI/CD for precommit hooks
- read the docs documentation

### Changed
- Improved package documentation and examples
- Improved testing
- Folded eval MLIP into the mlip_committee function
- mlip_committee now returns a pd.Dataframe of mae_e and mae_f for each loop
- ruff formatting for everything


## [0.1.0] - 2025-08-13

### Added
- Initial release of ALomancy package
- Standard MACE active learning workflow (`ActiveLearningStandardMACE`)
- Support for remote job execution via ExPyRe
- Structure generation using molecular dynamics
- MLIP committee training and evaluation
- High-accuracy DFT evaluation pipeline with Quantum Espresso
- Configuration management for HPC systems
- Example workflows and configuration files

### Features
- **Core Workflows**
  - Base active learning framework
  - MACE-specific implementation
  - Configurable loop iteration

- **Structure Generation**
  - Initial structure selection
  - Molecular dynamics simulations
  - High standard deviation structure identification

- **Machine Learning**
  - MACE model training and committee evaluation
  - Uncertainty quantification
  - Model performance metrics

- **Remote Execution**
  - HPC job submission and monitoring
  - Queue system integration
  - Automatic result collection

- **Configuration**
  - YAML-based job configuration
  - HPC system definitions
  - Flexible parameter management

[Unreleased]: https://github.com/julianholland/ALomancy/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/julianholland/ALomancy/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/julianholland/ALomancy/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/julianholland/ALomancy/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/julianholland/ALomancy/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/julianholland/ALomancy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/julianholland/ALomancy/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/julianholland/ALomancy/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/julianholland/ALomancy/releases/tag/v0.1.0
