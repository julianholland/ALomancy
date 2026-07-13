# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/your-username/alomancy/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/your-username/alomancy/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/your-username/alomancy/releases/tag/v0.1.0
