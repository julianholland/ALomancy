# Structure generation

`src/alomancy/structure_generation/`: candidate generation (MD or EZGA) and committee-uncertainty selection. Generators are registered under `structure_generator` (`structure_generation.generator`, default `"md"`).

## Generators use the trained MLIP

`generate_candidates` passes the generator a `CalculatorSpec` (`mlip.md`), not a model path: plain data that pickles into remote jobs, since a live (often GPU-resident) calculator must not cross the ExPyRe boundary. `spec.build(device=None)` calls the trainer's `get_calculator` where the calculator is needed: on the node for MD (`md_wfl._run_md_with_spec`), in the driver on the CPU for EZGA. `run_md` takes the calculator as a required argument and has no MACE fallback. The removed `md_kwargs.trainer`/`trainer_config` (default `"mace"`) made a SevenNet run hand its `.pth` to MACE (`'dict' object has no attribute 'to'` in every MD job); they're in `_REMOVED_KEYS`. `tests/core_tests/test_active_learning_workflow.py::test_md_explores_with_the_calculator_of_the_trained_mlip` guards this. MD and EZGA changes need a remote reinstall.

## Counts: generation vs selection

`structure_generation.num_of_structures_to_generate` is the candidate pool (MD: `snapshot_interval = steps × runs // to_generate`); `general.num_of_structures_per_loop` (default 50) is what every selector sends to DFT. `_resolve_structure_counts` (`ActiveLearningWorkflow.__init__`) defaults to_generate to 10 × per_loop, raises if to_generate < per_loop, and warns below 2 × per_loop (`few_candidates_configured`); `generate_candidates` warns when fewer than 2 × per_loop candidates survive the bond filter (`few_candidates_generated`). The old `desired_num_of_structures` is in `_MOVED_KEYS`.

## Bond filter

**No structure with a pairwise distance below 0.5 Å is scored or sent to DFT**: `filter_structures_by_min_bond_distance` (`utils/clean_structures.py`) runs in `generate_candidates`, before committee scoring, and again in the DFT orchestrator as a final net (which also covers initialization). `get_all_distances(mic=True)` is safe for every structure; single atoms pass.

## MD (`structure_generation/md/md_wfl.py`)

Settings live in `structure_generation.md_kwargs` (schema from `run_md`'s signature); seed selection in `md_kwargs.structure_selection_kwargs` (`num_of_md_starts`, `enforce_chemical_diversity`, `seed`); the generic seed filter (`chem_formula_list`, `selectable_configs`, `atom_number_range`) in `structure_generation.structure_selection_kwargs`.

- **Ensembles**: `ensemble: "nvt"` (default, fixed-cell `ase.md.langevin.Langevin`) or `"npt"` (`ase.md.langevinbaoab.LangevinBAOAB`, variable cell, targeting `pressure` in GPa). Anything else raises `ValueError`. **`externalstress` must never be `None` for NPT**: ASE only activates the barostat when it isn't `None`, so `None` silently runs fixed-cell dynamics with the costlier integrator. `run_md` passes `-pressure * ase.units.GPa`, so even `pressure=0.0` keeps the barostat on.
- **Seeds are reproducible, saved, and replaced once if their run never starts** (`generate`). `select_diverse_seeds` (`utils/seed_selection.py`) draws from `np.random.default_rng(seed)` (formulas sorted first), and the seeds are written to `structure_generation/md_seeds.xyz`: run `i` is always `md_output_i` with `seeds[i]`, and only runs with no output directory are submitted. A trajectory with fewer than `_MIN_FRAMES` (2) frames did no MD step and gets **one** replacement: an unused structure (by `geometry_digest`; a used one with a fresh `md_seed` if none are left), appended to `md_seeds.xyz` with `info["replaces"] = i` and run as a new index (sharing the original's `md_run_<slot>` item). Runs that stepped are kept however short; `run_md` appends frames as it goes and catches force/integration failures. The candidates cache (`<name>_generated_candidates.xyz`) is written only after the replacement round, from runs with ≥ 2 frames; if none have any, `generate` raises `RuntimeError` without writing the cache, so a restart retries. Events: `md_no_steps`, `md_unfinished`, `md_summary`.
- `select_initial_structures` (`structure_generation/select_initial_structures.py`) reuses structures with replacement when `num_of_md_starts` exceeds the filtered pool, giving each a distinct `info["md_seed"] = seed + i` that seeds Langevin's `rng` (so duplicated seeds don't produce identical trajectories); only an empty pool raises.
- Reloading `structure_generation_input_structures.xyz` reads with `index=":"` (a bare `read()` returns only the last frame) and regenerates if the count is short of `num_of_md_starts`, so a half-written file isn't trusted.
- **`md_kwargs.traj_interval`** (default off) writes `md_output_i/md_trajectory.xyz`, every n-th production step, via a `dyn.attach` observer that skips steps already written (ASE re-calls observers at each `dyn.run()`, and the snapshot loop runs in segments). Validated with `if`/`raise ValueError`.

## EZGA (`structure_generation/ezga/generate_structures.py`)

`generator: "ezga"` replaces candidate generation only; the bond filter, committee scoring and selection still apply to its output.

- `run_ezga` runs **in the local driver process** (no remote submission) and relaxes candidates on the CPU with the trained model's calculator: `trained_model_calculator`, named in the generated `ezga_config.yaml`, reads the `CalculatorSpec` from `calculator_spec.json` (EZGA's YAML loader imports any string containing `:` or a dotted name, so the config can't be inlined) and wraps the calculator in EZGA's `ase_calculator`. Expect a real performance cost with a large population or many generations.
- It uses the single best model (the `rank_committee` pick, as MD does).
- `structure_generation.ezga_kwargs`: `max_generations`/`population_size` (default 2 each), `min_atoms`/`max_atoms` (default 2/41; every seed must be in range or `run_ezga` raises before calling EZGA), `mutation_operators` (override any of `rattle`, `random_strain`, `add`, `remove`, `remove_add`; an unknown name or disabling all of them raises).
- `bounded_mutation_add`/`bounded_mutation_remove` clamp EZGA's `add`/`remove` to `[min_atoms, max_atoms]` (returning `None`, a no-op mutation). `objective_energy_per_atom` treats unevaluated seeds' NaN energies as 0.0, as EZGA does.
- `build_ezga_config`/`build_mutation_configs` translate the settings into EZGA's YAML. Output in `results/<base>/structure_generation/ezga/`: `initial_population.xyz`, `ezga_config.yaml`, `ezga_candidates.xyz`, EZGA's population database (`db`/`db_ro`).

## Committee uncertainty selection

`find_high_sd_structures` (`structure_generation/find_high_sd_structures.py`, polars) selects the candidates with the largest force standard deviation across the committee and saves `std_dev_forces.csv`. It takes the count as `num_of_structures`; when fewer candidates survive than requested it warns (`fewer_candidates`) and selects them all. `select_indices(ranked, n)` replaces the plain top-n cut; the committee workflow passes `non_redundant_selector` (`core_workflow.md`, redundancy) so picks redundancy removal would flag are replaced by the next candidates; if it keeps fewer than requested (possibly none) that logs `selection_redundant_shortfall`. The selection is recorded in `std_dev_forces.csv`'s `selected` column (the report's selection cut). `high_sd_structures.xyz` is always written, empty for an empty selection, and read back through `read_selection` (`utils/file_saving_and_parsing.read_xyz`, which reads an empty file as no structures; the DFT phase's cached results use it too). Only an empty list or a count ≤ 0 raises.
