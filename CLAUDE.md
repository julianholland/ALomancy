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

`load_dictionaries(config_path)` reads a YAML file into a `jobs_dict` passed to the workflow constructor. Four top-level keys are required: `initialization`, `mlip_committee`, `structure_generation`, `high_accuracy_evaluation`. Each carries a `name`, `max_time`, and `hpc` value.

**HPC string resolution**: `hpc:` may be either a dict (full inline profile, backwards-compatible) or a string name (e.g. `hpc: 'raven_gpu'`). When it is a string, `load_dictionaries` looks it up in `~/.alomancy/hpc_config.yaml` (loaded by `global_config._load_global_hpc_config()`) and substitutes the full dict in-place. Raises `ValueError` if the named profile is not found — message directs the user to `alomancy add-hpc`.

`global_config.py` holds path constants (`ALOMANCY_HPC_CONFIG = ~/.alomancy/hpc_config.yaml`, `EXPYRE_CONFIG = ~/.expyre/config.json`) and `_load_global_hpc_config()`.

`RemoteInfo` and `get_remote_info()` convert a job sub-dict into an Expyre `RemoteInfo` object. The `sys_name` field maps to an Expyre system name (defined in the user's Expyre config, not in this repo).

### Remote execution (`src/alomancy/utils/remote_job_executor.py`)

`RemoteJobExecutor` wraps Expyre's `ExPyRe`. Call pattern: `submit_multiple_jobs → start_all_jobs → wait_for_all_jobs → cleanup_jobs`. The convenience method `run_and_wait` does all four steps. **Note**: `wait_for_all_jobs` is intentionally called twice in `run_and_wait` to ensure results sync locally from the remote.

**PyTorch triton requirement**: Recent PyTorch versions ship native triton-backed GPU ops via `torch._native` (confirmed present in torch 2.13). Triton JIT-compiles CUDA extensions at runtime, which requires `python3-dev` (Python headers) on the compute node. HPC compute nodes typically do not have these headers. Fix: set `TRITON_CACHE_DIR` to a persistent path in the `mlip_committee` `pre_cmds` and pre-warm the cache once in an **interactive GPU job** (not the login node — the login node driver may be too old). `mace_wfl.py` logs a WARNING at import time if `torch._native` is detected and `TRITON_CACHE_DIR` is not set. Pre-warm command (run inside an interactive GPU job): `TRITON_CACHE_DIR=/fhi/home/jholl/.triton_cache python -m mace.cli.run_train --help`

**Remote deployment**: ExPyRe serializes functions **by reference** (module path + function name). The remote machine imports the module to resolve the function at run time. When changing the signature or body of any function submitted remotely (e.g. `mace_fit`), you must reinstall the updated package on the remote machine (`pip install -e .` there) before restarting, or jobs will import the old code and fail with a `TypeError` at the call site — typically hours after submission.

**Nested `srun` must not request its own `--mem`**: `_build_srun_command` (`utils/dft_utils.py`) launches the DFT/MLIP binary via `srun` from inside the job's own batch script, and requests `--mem=0` — Slurm's documented sentinel for "use all memory already granted to the job." Do not change this to an explicit sub-amount (e.g. `--mem={max_mem_per_node}`): on Slurm configs where the running batch script is accounted as the job's first step and credited with the job's *entire* memory allocation, a nested step's own explicit request competes with that reservation and fails immediately with `"Unable to create step ... Memory required by task is not available"` — regardless of how much memory the job was granted or how small the request is. The job-level memory ceiling belongs on the outer `#SBATCH --mem` line (see next paragraph), not on the inner `srun`.

**HPC wizard memory headers are per-partition**: `add_hpc_wizard()` (`cli/add_hpc.py`) collects `max_mem` per partition and bakes each partition's own `#SBATCH --mem=<N>M` line into that partition's `header` entry in `~/.expyre/config.json` (via `_with_partition_mem_headers`), not into the system-wide header. ExPyRe applies a partition's `header` list in addition to the system-wide header for any job that lands on it. A single blanket `--mem` shared across all of a system's partitions is wrong whenever those partitions have different node memory sizes — it under-requests on large-memory partitions or gets jobs rejected outright on small ones. `node_info.max_mem_per_node` in `~/.alomancy/hpc_config.yaml` is descriptive metadata only; it does not drive any `#SBATCH`/`srun` `--mem` value.

### Module responsibilities

| Module | Purpose |
|---|---|
| `database/global_database.py` | `GlobalDatabase` — persistent sage_lib `Partition` (hybrid HDF5+SQLite) storing all post-DFT structures; handles dedup by `(config_type, formula)` and forces round-trip via `atoms.info['_REF_forces']` |
| `initialize/initialization_structure_list.py` | `create_initialization_atoms_list()` generates initial structures (dimers, trimers, amorphous, MP); `compute_initialization_needs()` queries the DB and returns per-combo counts of what's still missing |
| `initialize/mp_interface.py` | Fetches structures from the Materials Project API (`mp_api`); `retrieve_mp_material_docs` requests only `fields=["material_id", "structure"]` from `mpr.materials.summary.search` rather than the client's `all_fields=True` default — the live MP API and the installed `mp_api` client version can drift out of sync on unrequested sub-model schemas (e.g. `bandstructure`), and a missing field there raises a pydantic `ValidationError` that kills the whole fetch even though `docs_to_atoms` only ever reads `doc.structure` |
| `mlip/committee_remote_submitter.py` | Submits N MACE training jobs (committee); each fit lands in `results/<base>/mlip_committee/fit_<i>/` |
| `mlip/mace_wfl.py` | Wraps the `mace_fit` CLI call for a single committee member; after training completes, `_save_mace_eval_predictions` evaluates the stagetwo model on both train and test sets and writes `train_pred.xyz` / `test_pred.xyz` in the fit directory on the remote node |
| `mlip/get_mace_eval_info.py` | `get_mace_eval_info` reads trained model metrics into a DataFrame; `select_best_committee_model` picks the fit with lowest test-set `mae_f` and returns `(fit_idx, model_path)` |
| `structure_generation/md/md_remote_submitter.py` | Submits MD runs (one per input structure) and force-evaluation jobs across all committee members |
| `structure_generation/find_high_sd_structures.py` | Selects structures with highest force standard-deviation across the committee (uncertainty metric); uses Polars DataFrames |
| `structure_generation/select_initial_structures.py` | Picks seed structures from training data for MD; reuses structures with distinct `atoms.info["md_seed"]` values when `max_number_of_concurrent_jobs` exceeds the number of selectable structures, rather than erroring |
| `high_accuracy_evaluation/dft/qe_remote_submitter.py` | Batches structures and submits QE jobs; supports both single-point (`run_sp_qe`) and geometry-optimisation (`run_go_qe`) |
| `utils/clean_structures.py` | Validates/cleans ASE Atoms objects after DFT (sets `config_type`, removes bad structures) |
| `utils/test_train_manager.py` | Splits atom lists into train/test and merges extra datasets |
| `analysis/plotting.py` | Plots MAE vs AL loop number |
| `configs/global_config.py` | Path constants (`ALOMANCY_HPC_CONFIG`, `EXPYRE_CONFIG`) and `_load_global_hpc_config()` — reads `~/.alomancy/hpc_config.yaml` |
| `cli/add_hpc.py` | `add_hpc_wizard()` interactive setup wizard; pure builders `build_expyre_entry`, `build_alomancy_profile`, `write_expyre_config`, `write_alomancy_hpc_config`, `run_remote_install` |
| `cli/nuke.py` | `nuke_expyre_results()` deletes local ExPyRe job state (job cache, unsynced stage dirs) under `~/.expyre`, preserving `config.json`; prompts for confirmation |

### Key conventions

- All file I/O uses `extxyz` format via ASE's `read`/`write`.
- Energy and force labels stored in `atoms.info["REF_energy"]` and `atoms.arrays["REF_forces"]`. **Never use bare `"energy"` as an info key** — ASE moves it to the calculator on extxyz read, losing it from `atoms.info`.
- Structures that need geometry optimisation carry `atoms.info["needs_relaxation"] = True`. **This flag is never cleared automatically** — it survives `atoms.copy()` (and `clean_structures`'s `structure.info.copy()`) through GlobalDatabase storage and back out again. If a `needs_relaxation=True` structure is later reused as an MD seed, every MD trajectory frame derived from it inherits the flag too. Any code path that generates structures which must never be relaxed (e.g. `generate_structures` in `standard_active_learning.py`) must explicitly set `needs_relaxation=False` on its output rather than relying on the key being absent.
- `config_type` in `atoms.info` tracks provenance (e.g. `"initialization"`, `"init_dimer"`, `"high_sd"`). Structures selected by the AL loop uncertainty criterion get `config_type="high_sd"` (set by `clean_structures` in `base_active_learning.run()`); the per-loop index is stored separately in `atoms.info["al_loop"]` via `extra_metadata`. Do not use `"al_loop_N"` as a config_type — the loop number is not embedded in the type string.
- The `seed` parameter (default `803`) is used everywhere randomness appears for reproducibility.
- `select_initial_structures` (`structure_generation/select_initial_structures.py`) reuses structures with replacement when `max_number_of_concurrent_jobs` exceeds the number of structures passing the `chem_formula_list`/`selectable_configs`/`atom_number_range` filters — it no longer raises. Each of the `max_number_of_concurrent_jobs` returned atoms gets a distinct `atoms.info["md_seed"] = seed + i`, which `run_md` (`structure_generation/md/md_wfl.py`) reads to seed Langevin's `rng`; without this, two concurrent MD jobs started from the same duplicated structure would (depending on process-level RNG state) risk producing identical or under-diverged trajectories. A `ValueError` is only raised when the filtered pool is empty (nothing to reuse from at all).
- `verbose` is an int: `0` = silent, `>0` = progress prints.
- The global DB lives at `results/global_database/` (configurable via `db_path` in `BaseActiveLearningWorkflow.__init__`). Only DFT-evaluated structures (with `REF_energy`/`REF_forces`) are stored in it.
- Every structure in `GlobalDatabase` carries `atoms.info["global_db_id"]`: a zero-based integer index assigned at insertion time, stable across sessions. Call `db.assign_global_db_ids()` to backfill existing DBs.
- `BaseActiveLearningWorkflow` accepts `remove_redundancy=True` and `high_force_threshold=100.0` — these run after initialization and after each AL loop to prune similar structures and exclude unphysical forces from the DB before the next training round.
- `skip_initialization=True` on `BaseActiveLearningWorkflow`: when no AL loops have completed, loads train/test from the DB and starts at `start_loop` without calling `initialize_training_set`. Useful when the DB is pre-populated.
- `generate_structures` uses `select_best_committee_model` (lowest test-set `mae_f` from `fit_*/results/*_test.txt`) as the MD base model; all other fits form the uncertainty comparison committee. `fits_to_use` excludes the best fit index to avoid double-counting in the std-dev computation.
- Energy MAE reported as `mae_e_per_atom` (eV/atom) everywhere — `get_mace_eval_info`, `plotting.py`, and `mlip_plots.py` all use this key. Never use `mae_e` (total energy per structure) in plots; it has similar magnitude to `mae_f` and causes visual confusion.
- Initialization config uses individual counts (`num_dimers_per_combo`, `num_trimers_per_combo`, `num_amorphous`, `num_stretch_compress_per_mp`) rather than the old `d_t_s_a_ratio` + `target_non_mp_structures_to_add`.
- `creation_kwargs.max_atom_number` and `creation_kwargs.amorphous_atom_number` are separate knobs (both default to `20`): `max_atom_number` caps atom count for Materials Project fetches (`atoms_list_from_mp`'s `max_num_atoms`), `amorphous_atom_number` sets the target atom count per generated amorphous cell (`create_amorphous_atoms_list`'s `atom_number`). They used to be the same value (`max_atom_number` drove both); changing one no longer affects the other.
- `creation_kwargs.mp_max_energy_above_hull` (default `0.1` eV) sets the energy-above-hull cutoff for Materials Project fetches (`atoms_list_from_mp`'s `max_energy_above_hull`), previously hardcoded. Only affects MP structures, not amorphous/dimer/trimer generation.
- `IsolatedAtom` and `init_MP` config_types are deduplicated by `(config_type, formula)` in `GlobalDatabase.add_structures()`; other config_types (dimers, trimers, amorphous, AL loop structures) are always added without exact dedup.
- In `initialize_training_set`, the DB is checked **first** (`compute_initialization_needs` against current DB state). `extra_datasets` are seeded only if the DB is still missing some initialization targets, then needs are re-checked before any structure generation. An already-populated DB is always honoured before consulting extra datasets.
- `initialize_training_set` has two paths: (1) fast path — if `initial_train_file_path` and `initial_test_file_path` exist on disk, load them directly; (2) DB path — call `compute_initialization_needs`, generate only missing structures, run DFT, build train/test from `db.get_all_as_atoms()`.
- `jobs_dict["high_accuracy_evaluation"]["name"]` must equal `"high_accuracy_evaluation"`. `ase_remote_submitter` always writes output to the hardcoded path `results/<base>/high_accuracy_evaluation/batch_N/`; the result-collection glob in `standard_active_learning.high_accuracy_evaluation` uses `output_name` (i.e. the `name` field) as the subdirectory. If they diverge the glob finds nothing, an empty sentinel is written, and the phase is permanently marked done with zero structures.
- Guard user config errors in functions that run remotely with `if`/`raise ValueError(...)`, not `assert`. Python's `-O` flag (common in HPC module environments) strips all asserts silently, deferring config failures to the remote job hours after submission.
- **VASP pseudopotentials**: ASE's `Vasp` calculator reads POTCAR locations only from the `VASP_PP_PATH` environment variable — there is no constructor kwarg for it. `create_vasp_calc_object` (`high_accuracy_evaluation/dft/run_vasp.py`) sets `os.environ["VASP_PP_PATH"] = hpc["pp_path"]` before building the calculator whenever `pp_path` is present in the HPC profile. `pp_path` must point at the directory *containing* `potpaw_PBE`/`potpaw_LDA`. Elements without a `pseudo_dict` override fall back to the default (unsuffixed) POTCAR under that path; if the library doesn't carry a given element, VASP fails with `No pseudopotential for <element>!`.
- **VASP requires full periodicity**: ASE's `Vasp` calculator rejects any `pbc` other than `[True, True, True]` — VASP has no non-periodic mode. Non-periodic structures (isolated atoms, dimers, trimers from initialization) are padded with vacuum and forced to `pbc=True` by `_ensure_fully_periodic` inside `create_vasp_calc_object` before the calculator is built. QE's `Espresso` calculator handles non-periodic cells natively and needs no equivalent step.
- GO (geometry-optimisation) and SP (single-point) QE batches use different numbering ranges to avoid directory collision. `current_batches` = count of existing batch dirs (the directory-name offset). `n_new_batches` = new GO batches needed. GO dirs are numbered `[current_batches, current_batches + n_new_batches)`; SP dirs start at `current_batches + n_new_batches`. The loop index `i` (0-based) is used to slice the trimmed structures list; `batch_num = current_batches + i` is used for the directory name only.
- **MACE prediction files**: after `mace_fit` finishes training on the remote GPU, `_save_mace_eval_predictions` writes `train_pred.xyz` and `test_pred.xyz` in `results/<base>/<name>/fit_<i>/`. These files carry `mace_energy` (float, eV) and `mace_forces` (array, eV/Å) keys alongside the standard DFT keys. `store_mlip_predictions` (in `standard_active_learning.py`) reads these files locally and stores predictions in the GlobalDB using keys `mace_energy_loop_{N}_fit_{i}` / `mace_forces_loop_{N}_fit_{i}`. `plot_dft_vs_model` reads from DB first, then falls back to these files, then skips with a blank subplot — **never runs local MACECalculator inference**. Loops trained before v0.4.2 will not have the eval files; the sentinel (`mace_predictions.done`) is written and skipped gracefully.
- **Formation-energy parity plots**: `plot_dft_vs_model` plots *formation energy per atom* (`(E - sum(E0[symbol] for symbol in structure)) / n_atoms`), not raw per-atom energy, whenever a `GlobalDatabase` is available and has `IsolatedAtom` structures — `GlobalDatabase.get_isolated_atom_energies()` builds the `{symbol: REF_energy}` dict once per loop from those, and it is threaded into both `get_mace_predictions(..., e0=...)` (DB tier) and `_parse_eval_xyz(..., e0=...)` (xyz-fallback tier). The same E0 shift is subtracted from both DFT and model energy for a given structure, so it cancels exactly in the reported `mae_e_per_atom`/MAE annotation — it only changes the *shape/spread* of the parity scatter, not the training-curve MAE plots (`mae_al_loop_plot`, `plot_training_curves`), which read `mae_e_per_atom` straight from MACE's own eval output and are untouched by this. Falls back silently to raw per-atom energy (logging one WARNING) when no `GlobalDatabase`/no `IsolatedAtom` entries are available, or when a structure contains an element missing from the E0 dict — this preserves old behavior for pre-formation-energy runs and never mixes formation- and raw-energy points within one figure. `alomancy replot` opens the `GlobalDatabase` at `results_dir/global_database` (matching `GlobalDatabase.__init__`'s default `db_path`) if it exists, so replotted parity plots match the live AL loop's; older results directories without a `global_database/` fall back gracefully.

### sage_lib / GlobalDatabase internals

`sage_lib.Partition` (hybrid HDF5+SQLite) only persists `atoms.info`, **not** `atoms.arrays`. `REF_forces` must be serialised into info before storage (`a.info["_REF_forces"] = forces.tolist()`) and restored on retrieval (`np.array(meta.pop("_REF_forces"))`). This is handled transparently by `GlobalDatabase._prepare_for_storage` / `_atoms_from_container`.

`count_all_by_config_type_and_formula()` does a single O(N) scan over all containers and returns `{config_type: {formula: count}}`. Use it instead of calling `count_by_config_type_and_formula()` multiple times.

### Test suite

Tests live in `tests/` with subdirectories mirroring the source layout. New directories added:
- `tests/database_tests/` — GlobalDatabase dedup and round-trip tests
- `tests/initialize_tests/` — `compute_initialization_needs` delta logic

All tests are marked `@pytest.mark.unit` **on each individual test method** (run without any external services). Do not apply the mark only at the class level — pytest's mark inheritance is configuration-dependent and `pytest -m unit` may miss class-level-only marks. The suite tests real code paths — no "mock-testing" (asserting that a mock returns what you told it to return).

Key test patterns:
- `GlobalDatabase` tests use real sage_lib via `GlobalDatabase(str(tmp_path / "db"))` for genuine round-trip verification.
- `compute_initialization_needs` tests use a `MagicMock` DB that returns controlled `count_all_by_config_type_and_formula()` dicts.
- `mace` is patched at the top of `test_standard_active_learning.py` via `sys.modules.setdefault(...)` before the module is imported (avoids GPU dependency at collection time).
- `wfl` is not installed — never patch or import it in tests.
- `select_best_committee_model` tests (`TestSelectBestCommitteeModel` in `tests/mlip_train_tests/test_mace_training.py`) write fake `*_test.txt` files under `tmp_path` and use `monkeypatch.chdir` — no real MACE models needed. Cover both JSON-lines and Python list-of-tuples formats.
- `_write_train_txt` in `TestGetMaceEvalInfo` writes `mae_e_per_atom` (not `mae_e`) to match what `get_mace_eval_info` filters for.
