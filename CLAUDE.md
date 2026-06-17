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

# Run all tests (fast — skips coverage overhead)
/home/jholl/.venvs/al_install/bin/pytest --no-cov

# Run only unit tests (no HPC/MACE/QE/MP required)
/home/jholl/.venvs/al_install/bin/pytest -m unit --no-cov

# Run a single test file
/home/jholl/.venvs/al_install/bin/pytest tests/core_tests/test_base_active_learning.py --no-cov

# Run by marker
pytest -m "not requires_external"

# Run with coverage (slow — pyproject.toml sets --cov-fail-under=80)
pytest --cov=alomancy --cov-report=html

# Run in CI mode (env vars disable real external calls)
ALOMANCY_TEST_MODE=1 ALOMANCY_MOCK_EXTERNAL=1 pytest tests/
```

Pre-commit hooks run `ruff` (with `--fix`) and unit tests on every commit. Install with `pre-commit install`.

## Logging

All output goes through Python's `logging` module. No bare `print()` calls exist in `src/`.

```python
# Single configuration point — called in BaseActiveLearningWorkflow.__init__
from alomancy.utils.logging_config import setup_logging
setup_logging(verbose=1, log_file="results/my_run.log")
```

| `verbose` | Console | File (`results/alomancy.log`) |
|---|---|---|
| 0 (default) | WARNING+ only | DEBUG always |
| 1 | INFO (step progress) | DEBUG always |
| 2+ | DEBUG (per-job detail) | DEBUG always |

The file handler always writes DEBUG so every run produces a complete timestamped record. ExPyRe job stdout/stderr are captured at DEBUG level.

Each module declares `logger = logging.getLogger(__name__)` — no per-module setup needed. The `"expyre"` logger is also routed through the same handlers.

**Testing logs:** pytest's `caplog` does not intercept alomancy logs because `propagate=False` is set on the root alomancy logger. Attach a `logging.Handler` directly to `logging.getLogger("alomancy")` in tests that need to assert on log output (see `test_seed_logs_message` in `test_base_active_learning.py`).

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
- Energy and force labels stored in `atoms.info["REF_energy"]` and `atoms.arrays["REF_forces"]`. **Never use bare `"energy"` as an info key** — ASE moves it to the calculator on extxyz read, losing it from `atoms.info`.
- Structures that need geometry optimisation carry `atoms.info["needs_relaxation"] = True`.
- `config_type` in `atoms.info` tracks provenance (e.g. `"al_loop_0"`, `"initialization"`).
- The `seed` parameter (default `803`) is used everywhere randomness appears for reproducibility.
- `verbose` is an int: `0` = silent, `>0` = progress prints.
- The global DB lives at `results/global_database/` (configurable via `db_path` in `BaseActiveLearningWorkflow.__init__`). Only DFT-evaluated structures (with `REF_energy`/`REF_forces`) are stored in it.
- Initialization config uses individual counts (`num_dimers_per_combo`, `num_trimers_per_combo`, `num_amorphous`, `num_stretch_compress_per_mp`) rather than the old `d_t_s_a_ratio` + `target_non_mp_structures_to_add`.
- `IsolatedAtom` and `init_MP` config_types are deduplicated by `(config_type, formula)` in `GlobalDatabase.add_structures()`; other config_types (dimers, trimers, amorphous, AL loop structures) are always added without exact dedup.
- In `run()`, `extra_datasets` are seeded into the DB **before** `initialize_training_set` is called, so `compute_initialization_needs` accounts for them and avoids regenerating already-provided structures.
- `initialize_training_set` has two paths: (1) fast path — if `initial_train_file_path` and `initial_test_file_path` exist on disk, load them directly; (2) DB path — call `compute_initialization_needs`, generate only missing structures, run DFT, build train/test from `db.get_all_as_atoms()`.
- GO (geometry-optimisation) and SP (single-point) QE batches use different numbering ranges to avoid directory collision: GO uses `[current_batches, total_batches)`, SP uses `[total_batches, ...)`.

### sage_lib / GlobalDatabase internals

`sage_lib.Partition` (hybrid HDF5+SQLite) only persists `atoms.info`, **not** `atoms.arrays`. `REF_forces` must be serialised into info before storage (`a.info["_REF_forces"] = forces.tolist()`) and restored on retrieval (`np.array(meta.pop("_REF_forces"))`). This is handled transparently by `GlobalDatabase._prepare_for_storage` / `_atoms_from_container`.

`count_all_by_config_type_and_formula()` does a single O(N) scan over all containers and returns `{config_type: {formula: count}}`. Use it instead of calling `count_by_config_type_and_formula()` multiple times.

### Test suite

Tests live in `tests/` with subdirectories mirroring the source layout. New directories added:
- `tests/database_tests/` — GlobalDatabase dedup and round-trip tests
- `tests/initialize_tests/` — `compute_initialization_needs` delta logic

All tests are marked `@pytest.mark.unit` (run without any external services). The suite tests real code paths — no "mock-testing" (asserting that a mock returns what you told it to return).

Key test patterns:
- `GlobalDatabase` tests use real sage_lib via `GlobalDatabase(str(tmp_path / "db"))` for genuine round-trip verification.
- `compute_initialization_needs` tests use a `MagicMock` DB that returns controlled `count_all_by_config_type_and_formula()` dicts.
- `mace` is patched at the top of `test_standard_active_learning.py` via `sys.modules.setdefault(...)` before the module is imported (avoids GPU dependency at collection time).
- `wfl` is not installed — never patch or import it in tests.
