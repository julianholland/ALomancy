# EZGA interface review (for 1.0.4)

> **Historical.** This reviews the 1.0.3 EZGA interface, which 1.0.4 replaced:
> `ezga_kwargs` is now EZGA's own config, EZGA runs as a remote job, and the
> keys and functions named below (`population_size`, `mutation_operators`,
> `min_atoms`/`max_atoms`, `build_ezga_config`, `trained_model_calculator`,
> `calculator_spec.json`, `ezga_config.yaml`) no longer exist. For the current
> behaviour see `docs/ezga.md` and `docs/architecture/structure_generation.md`.

State of `structure_generation.generator: "ezga"` at v1.0.3 (ezga-lib 1.2.32 installed,
`ezga-lib>=1.2.0` required). Code: `src/alomancy/structure_generation/ezga/generate_structures.py`.

## How it works now

1. **The workflow calls the generator.** `ActiveLearningWorkflow.generate_candidates`
   filters this loop's training structures with
   `structure_generation.structure_selection_kwargs` (`filter_eligible_structures`). It
   then calls the registered `structure_generator` "ezga" entry point, `generate()`, with:
   - **every** eligible structure as seeds (MD instead picks `num_of_md_starts` diverse
     seeds);
   - the best model's `CalculatorSpec`;
   - the `structure_generation` section.
2. **`generate()` is a thin local wrapper.**
   - It returns the cached `ezga_candidates.xyz` if one exists.
   - Otherwise it calls `run_ezga(**ezga_kwargs)`.
   - `hpc` and `max_time` are accepted but unused: EZGA runs **in the driver process, on
     the CPU**.
3. **`run_ezga`:**
   1. Checks every seed has `min_atoms ≤ n ≤ max_atoms`, and raises otherwise.
   2. Writes `initial_population.xyz` and `calculator_spec.json`.
   3. Builds a full EZGA config dict (`build_ezga_config`) and dumps it to
      `ezga_config.yaml`.
   4. Reloads it with `ezga.factory.load_config`; the YAML round trip is how EZGA turns
      dotted strings into callables.
   5. Runs `build_default_engine(cfg).run()`.
   6. Exports the population's `"dataset"` partition to `ezga_candidates.xyz` and returns
      it.
4. **The EZGA run itself** (fixed in `build_ezga_config`; only five values are user
   settings):

   | Part | What it does |
   |---|---|
   | Population | Seeds plus EZGA's own duplicate filter (`collision_factor: 0.8`) |
   | Variation | Mutation rate 1.0, crossover probability 0.0 (a crossover function is listed but never used), constant thermostat at 1.0 |
   | Mutations (`build_mutation_configs`) | rattle, random_strain, add, remove, remove_add; add/remove wrapped by `bounded_mutation_add` / `bounded_mutation_remove` to respect the atom-count limits |
   | Objective | `objective_energy_per_atom`: minimise the model's energy per atom (NaN for unevaluated seeds → 0) |
   | Features | composition vector |
   | Simulator | "sampling" mode, the trained model through EZGA's `ase_calculator`, FIRE relaxation with `fmax: 0.05` and `steps_max: 5` |

5. **After EZGA:**
   - the bond-length filter;
   - committee prediction on the HPC;
   - committee-uncertainty selection;
   - DFT.

   These run exactly as for MD.
6. **User settings** (`structure_generation.ezga_kwargs`):

   | Key | Default |
   |---|---|
   | `max_generations` | 2 |
   | `population_size` | 2 |
   | `min_atoms` | 2 |
   | `max_atoms` | 41 |
   | `mutation_operators` | per-operator overrides |

   The defaults live twice: in `_EZGA_KWARGS_DEFAULTS` and in `run_ezga`'s signature.
7. **Restart and report:**
   - **Restart:** "done" means `ezga_candidates.xyz` exists (`output_paths`,
     `read_existing_result`).
   - **Report:** `ezga_section` lists generations, population and the number of
     candidates.
8. **Tests:**
   - unit tests of the objective, the config builder, the atom limits and the calculator
     factory (`tests/struc_gen_tests/test_ezga_generation.py`);
   - wrapper tests that mock `run_ezga` (`test_ezga_generator.py`);
   - **no test runs a real (tiny) EZGA search.**

## Weaknesses

### Correctness: these break or silently mislead real runs
1. **Palladium is hard-coded.** The default mutation species are `["Pd"]` (rattle, add,
   remove, remove_add, `bound`), and the composition feature's `IDs` are `["Pd"]`.
   - For any other system, add/remove try to place or remove Pd. The feature vector is
     meaningless unless the user overrides every operator. Nothing reads
     `general.elements`.
   - `slab: True` is also the default for add/remove_add, which is wrong for bulk and
     amorphous cells.
2. **The search goes the wrong way for active learning.** The objective minimises model
   energy per atom, so EZGA drives towards low-energy structures *according to the
   current model*. Two problems follow:
   - those are where the model is most confident;
   - they are where a bad model has spurious minima.

   AL wants structures where the committee disagrees, or novel structures. Committee
   selection afterwards can only pick from what EZGA chose to keep.
3. **The candidate pool size isn't controlled.**
   - `num_of_structures_to_generate` (validated, and used by MD) is ignored by EZGA.
   - The pool is whatever ends up in the exported `"dataset"` partition. With the
     defaults (2 generations, population 2) that is a handful of structures, so
     `few_candidates_generated` fires every loop.
   - Whether `"dataset"` holds every evaluated structure or only the survivors needs
     checking against ezga-lib; either way it isn't tied to the AL setting.
4. **Every eligible training structure is a seed, and all of them are relaxed on the
   driver's CPU.**
   - By loop 20 the training set can hold thousands of structures. EZGA evaluates every
     seed with the model (5 FIRE steps each) in the login/driver process, with no
     walltime and no GPU.
   - A run can stall for hours, block the driver, and is invisible to ExPyRe's
     retry/timeout machinery.
5. **One seed out of range kills the loop.** A single eligible seed outside
   `[min_atoms, max_atoms]` makes `run_ezga` raise, and so does the default
   `max_atoms: 41` with any larger structure. The seed filter
   (`structure_selection_kwargs.atom_number_range`) and these limits are separate
   settings, so nothing makes them agree.
6. **Restart after a crash isn't clean.** "Done" is only the final candidates file, so a
   crash mid-search restarts EZGA from scratch (`resume: False`). It does so on top of the
   previous run's `db` / `db_ro` directories, and whether EZGA reads or appends to those
   is unverified. An empty `ezga_candidates.xyz` is unreadable by ASE (the same bug class
   fixed in 1.0.3 for DFT results).
7. **The report section is broken** (already in v1.0.2, in TODO):
   - it builds its path from `structure_generation.name` (no longer read);
   - it reads `run_ezga_kwargs` instead of `ezga_kwargs`, so it always prints "default".
8. **Settings aren't checked at construction.** There is no `kwargs_schema`, and only
   some values are checked, inside `build_ezga_config`, at the first loop's generation
   phase. The checks don't cover a typo in a key, a wrong type, or an unknown species in
   a mutation (TODO 1.1).

### Design: limits on what EZGA can do here
9. **Almost nothing is configurable.** The whole EZGA config is fixed except five values:
   - crossover is off;
   - the thermostat and the mutation-rate schedule are fixed;
   - the simulator's relaxation (`fmax`, `steps_max`, optimiser, NVT steps) is fixed;
   - the objective, the features and the duplicate filter are fixed.

   Users can't use EZGA's own objectives, among them `objective_information_novelty`,
   `objective_anomality`, `objective_formation_energy` and hull distance.
10. **Only one model drives the search.** EZGA only ever sees the best model, so it can't
    use committee disagreement as an objective, even though the committee is trained
    anyway.
11. **It runs locally only.** It never goes through `submit_n`, unlike every other
    phase, so there is:
    - no HPC, GPU or walltime;
    - no use of the `structure_generation.hpc` profile the user must still set.
12. **The config is built twice.** ALomancy writes a dict to YAML and reads it back
    because EZGA's loader resolves dotted strings. That's why `calculator_spec.json`
    exists. It's fragile: any ALomancy value containing `:` or a dot-path would be
    imported as code.
13. **Most of it is untested.** No integration test runs EZGA end to end, so an
    ezga-lib update (the pin is `>=1.2.0`, open-ended) can break the generator without
    a failing test.

## Suggested improvements for 1.0.4

Ordered by value per effort. Items 1–5 are fixes; 6–9 make EZGA a useful AL generator.

1. **Derive species from `general.elements`.**
   - Default every mutation's `species` / `bound` / `species_add` / `species_remove` and
     the composition feature `IDs` to `general.elements`.
   - Default `slab: false`.
   - Reject a species outside `general.elements`.
   - Effort: small.
2. **Validate at construction.**
   - Register a `kwargs_schema` for `ezga_kwargs`, as MD has.
   - Check `min_atoms ≤ max_atoms`, the operator names, the species and positive
     integers when the workflow is built.
   - Make `_EZGA_KWARGS_DEFAULTS` the single source (`run_ezga` reads it; no duplicated
     signature defaults).
   - Effort: small.
3. **Filter seeds, don't crash on them.**
   - Drop seeds outside `[min_atoms, max_atoms]` with one warning and an event.
   - Raise only if none remain.
   - Cap the seed count: pick `num_of_seeds` diverse seeds with the existing
     `select_diverse_seeds` (as MD does, default e.g. 2 × `population_size`) instead of
     the whole training set.
   - Effort: small. This removes the hours-long CPU seeding.
4. **Honour `num_of_structures_to_generate`.**
   - Export every structure EZGA evaluated across generations, not only the final
     population (check which partition holds them in ezga-lib 1.2.x).
   - If that's still more than `num_of_structures_to_generate`, keep a diverse subset;
     if it's fewer, log `few_candidates_generated` as now.
   - Raise the defaults to something that can feed a loop (e.g. `population_size: 32`,
     `max_generations: 10`).
   - Effort: small–medium.
5. **Restart and report fixes.**
   - Wipe or move aside `db`/`db_ro` before a fresh search (or use `resume: true` if
     ezga-lib supports it reliably).
   - Read the candidates file with `utils/file_saving_and_parsing.read_xyz`.
   - Fix `ezga_section`: use the fixed `structure_generation` directory and
     `ezga_kwargs`, and add the seeds used, the candidates exported and the generations
     run.
   - Effort: small.
6. **An AL objective.** Make the objective a setting, `ezga_kwargs.objective`:
   - `energy_per_atom` (today's);
   - `committee_uncertainty`: maximise the committee's force std dev, by passing all
     committee `CalculatorSpec`s and a calculator that returns the mean energy plus the
     disagreement as a property the objective reads;
   - `novelty`: EZGA's `objective_information_novelty` against the training-set
     descriptors.

   Committee uncertainty matches what the selector does afterwards, so the search and
   the selection pull the same way. It needs the generator interface to accept several
   models (today `generate()` gets one `CalculatorSpec`). Effort: medium. This is the
   change that makes EZGA worth using for AL.
7. **Expose the useful EZGA settings, keeping the safe defaults:**
   - `crossover_probability` (enable the listed crossover);
   - `mutation_rate` / `min_mutation_rate`;
   - `relax: {fmax, steps_max, optimizer}`;
   - `collision_factor`;
   - an escape hatch `ezga_overrides:` merged into the generated config last, for
     anything else (documented as unsupported).

   Effort: small–medium.
8. **Run it remotely.**
   - Run the whole `run_ezga` as one `submit_n` job on `structure_generation.hpc`, like
     an MD job.
   - It is already a module-level function, and its arguments are plain data (atoms, a
     `CalculatorSpec`, kwargs).
   - That gives GPU relaxation, a walltime, retries and a free driver.
   - Keep a `local: true` option for small tests.
   - Effort: medium. **Remote reinstall needed** in the changelog.
9. **Tests and pinning.**
   - Add one `slow`/integration test that runs a real 2-generation, population-4 EZGA on
     a tiny C system with `EMTTrainer` (the backend-free trainer already used for MLIP
     tests).
   - Bound the dependency, `ezga-lib>=1.2,<1.3`, until that test runs in CI.
   - Effort: small.

### Suggested 1.0.4 scope
**Items 1–5 and 9.** They make the current search correct for any element set, robust on
real training-set sizes, restartable, reported and tested, without changing what EZGA
optimises. Then, as a follow-up minor version (1.1), **6–8**: the AL objective, the
exposed settings and remote execution. Those change the generator interface (several
models, remote submission) and deserve their own release notes.

### Open questions before starting
- Which ezga-lib partition holds every evaluated structure (the "dataset" or
  "generation" partition), and does `resume` work against an existing `db`?
- Is EZGA's own duplicate filter still wanted when ALomancy's redundancy removal and the
  bond filter run afterwards?
- Should `committee_uncertainty` drive EZGA through the mean model plus disagreement, or
  through a multi-objective set-up (low energy and high disagreement) using EZGA's
  `multiobjective` block?
