# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode (required before anything else)
pip install -e ".[dev]"

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/alomancy

# Run all tests
pytest

# Run a single test file
pytest tests/core_tests/test_base_active_learning.py

# Run by marker (unit/integration/slow/requires_external/requires_gpu/requires_hpc)
pytest -m unit
pytest -m "not slow and not requires_external"

# Run with coverage
pytest --cov=alomancy --cov-report=html

# Run in CI mode (env vars disable real external calls)
ALOMANCY_TEST_MODE=1 ALOMANCY_MOCK_EXTERNAL=1 pytest tests/
```

Pre-commit hooks run `ruff` (with `--fix`) and unit tests on every commit. Install with `pre-commit install`.

## Architecture

ALomancy implements active learning (AL) workflows for training machine-learned interatomic potentials (MLIPs). The core loop is: **train MLIP committee → MD structure generation → uncertainty-based selection → DFT evaluation → extend training set → repeat**.

### Core layer (`src/alomancy/core/`)

`BaseActiveLearningWorkflow` (abstract) owns the AL loop in its `run()` method. It calls four abstract methods that concrete subclasses must implement:

- `initialize_training_set(base_name)` — produce initial `(train_xyzs, test_xyzs)` atom lists
- `train_mlip(base_name, job_dict)` → `pd.DataFrame` of evaluation metrics
- `generate_structures(base_name, job_dict, train_atoms_list)` → `list[Atoms]` of high-uncertainty candidates
- `high_accuracy_evaluation(base_name, job_dict, structures)` → `list[Atoms]` with DFT results

`ActiveLearningStandardMACE` (in `standard_active_learning.py`) is the production implementation: MACE committee for the MLIP, ASE MD for structure generation, Quantum Espresso (QE) for DFT.

All results land under `results/<base_name>/` with a fixed subdirectory layout. The workflow has **idempotency logic** throughout: each step checks for existing output files and skips remote submission if they exist, enabling restart after failures.

### Configuration (`src/alomancy/configs/`)

`load_dictionaries(config_path)` reads a YAML file into a `jobs_dict` passed to the workflow constructor. Three top-level keys are required: `initialization`, `mlip_committee`, `structure_generation`, `high_accuracy_evaluation`. Each carries a `name`, `max_time`, and `hpc` sub-dict (`hpc_name`, `partitions`, `pre_cmds`).

`RemoteInfo` and `get_remote_info()` convert a job sub-dict into an Expyre `RemoteInfo` object. The `sys_name` field maps to an Expyre system name (defined in the user's Expyre config, not in this repo).

### Remote execution (`src/alomancy/utils/remote_job_executor.py`)

`RemoteJobExecutor` wraps Expyre's `ExPyRe`. Call pattern: `submit_multiple_jobs → start_all_jobs → wait_for_all_jobs → cleanup_jobs`. The convenience method `run_and_wait` does all four steps. **Note**: `wait_for_all_jobs` is intentionally called twice in `run_and_wait` to ensure results sync locally from the remote.

### Module responsibilities

| Module | Purpose |
|---|---|
| `database/global_database.py` | `GlobalDatabase` — persistent sage_lib `Partition` (hybrid HDF5+SQLite) storing all post-DFT structures; handles dedup by `(config_type, formula)` and forces round-trip via `atoms.info['_REF_forces']` |
| `initialize/initialization_structure_list.py` | `create_initialization_atoms_list()` generates initial structures (dimers, trimers, amorphous, MP); `compute_initialization_needs()` queries the DB and returns per-combo counts of what's still missing |
| `initialize/mp_interface.py` | Fetches structures from the Materials Project API (`mp_api`) |
| `mlip/committee_remote_submitter.py` | Submits N MACE training jobs (committee); each fit lands in `results/<base>/mlip_committee/fit_<i>/` |
| `mlip/mace_wfl.py` | Wraps the `mace_fit` CLI call for a single committee member |
| `mlip/get_mace_eval_info.py` | Reads trained MACE model error metrics into a DataFrame |
| `structure_generation/md/md_remote_submitter.py` | Submits MD runs (one per input structure) and force-evaluation jobs across all committee members |
| `structure_generation/find_high_sd_structures.py` | Selects structures with highest force standard-deviation across the committee (uncertainty metric); uses Polars DataFrames |
| `structure_generation/select_initial_structures.py` | Picks seed structures from training data for MD |
| `high_accuracy_evaluation/dft/qe_remote_submitter.py` | Batches structures and submits QE jobs; supports both single-point (`run_sp_qe`) and geometry-optimisation (`run_go_qe`) |
| `utils/clean_structures.py` | Validates/cleans ASE Atoms objects after DFT (sets `config_type`, removes bad structures) |
| `utils/test_train_manager.py` | Splits atom lists into train/test and merges extra datasets |
| `analysis/plotting.py` | Plots MAE vs AL loop number |

### Key conventions

- All file I/O uses `extxyz` format via ASE's `read`/`write`.
- Energy and force labels stored in `atoms.info["REF_energy"]` and `atoms.arrays["REF_forces"]`.
- Structures that need geometry optimisation carry `atoms.info["needs_relaxation"] = True`.
- `config_type` in `atoms.info` tracks provenance (e.g. `"al_loop_0"`, `"initialization"`).
- The `seed` parameter (default `803`) is used everywhere randomness appears for reproducibility.
- `verbose` is an int: `0` = silent, `>0` = progress prints.
- The global DB lives at `results/global_database/` (configurable via `db_path` in `BaseActiveLearningWorkflow.__init__`). Only DFT-evaluated structures (with `REF_energy`/`REF_forces`) are stored in it.
- Initialization config uses individual counts (`num_dimers_per_combo`, `num_trimers_per_combo`, `num_amorphous`, `num_stretch_compress_per_mp`) rather than the old `d_t_s_a_ratio` + `target_non_mp_structures_to_add`.
- `IsolatedAtom` and `init_MP` config_types are deduplicated by `(config_type, formula)` in `GlobalDatabase.add_structures()`; other config_types (dimers, trimers, amorphous, AL loop structures) are always added without exact dedup.
