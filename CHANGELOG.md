# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **GlobalDatabase**: Persistent sage_lib Partition (hybrid HDF5+SQLite) storing all DFT-evaluated structures across AL loops. Deduplication by (config_type, formula) prevents double-adding IsolatedAtom and init_MP entries from multiple datasets. REF_forces arrays serialised into atoms.info for round-trip storage.
- **DB-aware initialization**: `compute_initialization_needs()` queries the GlobalDatabase to determine what structures still need to be generated before starting DFT. Replaces the old `d_t_s_a_ratio` + `target_non_mp_structures_to_add` API with explicit per-type counts: `num_dimers_per_combo`, `num_trimers_per_combo`, `num_amorphous`, `num_stretch_compress_per_mp`.
- **Structured logging**: All 108 `print()` calls replaced with Python `logging` module. `setup_logging(verbose, log_file)` called once at workflow construction. verbose=0 silences console; verbose=1 shows INFO progress; verbose=2 shows DEBUG per-job detail. File handler always captures DEBUG regardless of verbose level. ExPyRe job stdout/stderr captured at DEBUG.
- **Extra dataset seeding**: Extra datasets are now seeded into the GlobalDatabase *before* `initialize_training_set` is called, so `compute_initialization_needs` accounts for them and avoids regenerating already-provided structures.
- `log_file` parameter on `BaseActiveLearningWorkflow` (default: `"results/alomancy.log"`)
- `db_path` parameter on `BaseActiveLearningWorkflow` (default: `"results/global_database"`)
- Warning if structure generation could contain single-atom structures

### Changed
- Initialization configuration now uses per-type counts instead of ratio-based approach
- `initialize_training_set` now has a DB-aware path that only generates missing structures
- Logging replaces all bare print() calls; no external API change

### Fixed
- Batch numbering collision between GO (geometry-optimisation) and SP (single-point) QE jobs

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
