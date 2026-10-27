# Exploring with EZGA and HiSE

This walkthrough sets up a full active-learning run that explores with
EZGA (`ezga-lib`, a genetic algorithm) instead of
molecular dynamics. It uses EZGA's hierarchical supercell escalation (HiSE)
and finishes each search with a short MD burst, following the EZGA wiki's
*Active Learning for MLIPs* example. The complete config is
`examples/configs/committee_mace_ezga_hise.yaml`; every block below comes
from it.

## What EZGA does in a loop

Each AL loop trains a committee, generates candidates, scores them and sends
the most uncertain to DFT. With `generator: "ezga"`, only the generation step
changes:

1. **Seeds.** ALomancy picks `num_of_seeds` diverse structures from the
   training set and saves them to `ezga_seeds.xyz`.
2. **Search.** One job on `structure_generation.hpc` runs EZGA with the loop's
   best model. EZGA mutates the population, relaxes every new structure with
   the model and keeps the best by its objective (energy per atom by
   default), generation after generation. EZGA drops structures it has
   already seen.
3. **HiSE stages.** With a `hise:` block, the search runs in stages. Each
   stage starts from the previous one's population, tiled into the next
   supercell. Here the stages are a wide search in the training cells, the
   same tiled 2×1×1, then two generations tiled 2×2×1 with a short MD burst.
4. **Candidates.** ALomancy reads EZGA's database from the last stage. It
   takes only the structures that stage generated, newest generations first,
   up to `num_of_structures_to_generate`.
5. **As usual from here.** The bond filter, committee scoring, selection
   and DFT run as they do after MD.

## Before you start

- **HPC profiles.** You need a GPU profile for training and EZGA, and a CPU
  profile for DFT (`alomancy add-hpc`; see `examples.md`). EZGA relaxes every
  structure with the model, so give it a GPU.
- **HPC install.** EZGA runs on the HPC, so the HPC install must be the same
  ALomancy version as your machine. After upgrading, run
  `alomancy upgrade-hpc`.
- **EZGA version.** ALomancy pins `ezga-lib>=1.2,<1.3`. It is installed with
  ALomancy, locally and by `alomancy upgrade-hpc`.

## The config

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["Cu", "O"]
  num_of_al_loops: 10
  num_of_structures_per_loop: 25     # selected for DFT each loop
  verbose: 1
  dataset_kwargs:
    target_config_types: ["init_MP", "init_amorphous"]
    test_ratio: 0.1
  committee_uncertainty_kwargs:
    num_of_models_in_committee: 3    # minimum 3

initialization:
  max_time: "2H"
  hpc: "my_cpu_hpc"
  amorphous_kwargs:
    num_of_amorphous_structures: 50
    num_of_atoms_per_amorphous: 10   # small seeds: 2x2x1 makes them 40 atoms
    densities_list: [5.0, 7.0]

training:
  trainer: "mace"
  max_time: "5H"
  hpc: "my_gpu_hpc"
  mace_kwargs:
    max_num_epochs: "dynamic"        # resolved from the training-set size

structure_generation:
  generator: "ezga"
  num_of_structures_to_generate: 250 # candidates the committee scores per loop
  max_time: "12H"                    # walltime of the one EZGA job
  hpc: "my_gpu_hpc"                  # EZGA relaxes with the model: use a GPU
  structure_selection_kwargs:        # which training structures may seed EZGA
    selectable_configs: ["init_MP", "init_amorphous"]  # (+ earlier loops' picks)
  ezga_kwargs:                       # EZGA's own config, plus ALomancy's two keys
    structure_selection_kwargs:
      num_of_seeds: 8                # diverse training structures to start from
    hise:
      supercells: [[1, 1, 1], [2, 1, 1], [2, 2, 1]]  # a new cell each stage
      overrides:                     # one value per stage
        max_generations: [20, 10, 2]
        multiobjective.size: [100, 100, 250]
        simulator.calculator:        # ase_calculator settings, over the defaults
          - {}
          - {}
          - {nvt_steps: 40, T: [300, 600]}

high_accuracy_evaluation:
  evaluator: "qe"
  max_time: "30m"
  max_go_time: "4H"
  force_ceiling: 100.0
  hpc: "my_cpu_hpc"
```

Run it with `alomancy run committee_mace_ezga_hise.yaml`. The rest of this
page explains the `structure_generation` section; the other sections are as
in `examples.md`.

## The `structure_generation` section

### Seeds

`structure_generation.structure_selection_kwargs` filters which training
structures may seed any generator: `selectable_configs`, `chem_formula_list`
and `atom_number_range`. Structures picked in earlier loops are always
eligible. Without `selectable_configs`, isolated atoms, dimers and trimers
from initialization could become seeds.

`ezga_kwargs.structure_selection_kwargs.num_of_seeds` (default 8) caps how
many of them start the search. They are chosen for diversity with
`general.seed`, so a restart picks the same ones. Set
`enforce_chemical_diversity: true` to prefer distinct formulas.

### `ezga_kwargs` is EZGA's own config

Anything EZGA's YAML file accepts goes under `ezga_kwargs`, with EZGA's own
names: `max_generations`, `multiobjective`, `population`, `thermostat`,
`variation`, `mutation_funcs`, `crossover_funcs`, `evaluator`, `simulator`,
`hise`, `convergence` and so on. EZGA checks the config when the workflow is
built, so a typo or an unknown mutation fails at startup, not hours into the
run.

ALomancy adds two keys of its own, removed before EZGA sees the config:
`structure_selection_kwargs` (above) and `local`. `local: true` runs EZGA
in the driver process instead of as a job, for tests and small runs.

ALomancy sets these itself, and setting them raises an error:

| Setting | Why |
|---|---|
| `output_path` | each loop's search has its own directory |
| `resume` | every search starts fresh; ALomancy restarts a crashed one |
| `rng` | seeded from `general.seed` |
| `population.dataset_path` | the seeds come from the training set |
| `population.db_path`, `db_ro_path`, `storage` | ALomancy reads the candidates from EZGA's database |
| the model in `simulator.calculator` | always the loop's best model |

### The calculator

`simulator.calculator` holds settings for EZGA's `ase_calculator`, which
ALomancy wraps around the loop's best model on the HPC node:
- `fmax` and `steps_max` for the relaxation;
- `nvt_steps`, `T` and `md_timestep_fs` for MD;
- `constant_volume` to relax the cell too.

Here it is left at the defaults (`fmax` 0.05, `steps_max` 100, FIRE).
ALomancy rejects any setting `ase_calculator` doesn't have.

### Defaults

Leave a section out and ALomancy fills it in. The defaults follow the EZGA
wiki's active-learning example:

| Setting | Default |
|---|---|
| `multiobjective.size` | 100, with `composition_repulsion_weight` 1.0 |
| `max_generations` | enough for twice `num_of_structures_to_generate` (at least 5) |
| `thermostat` | `initial_temperature` 0.05, constant |
| `population` | `filter_duplicates` on, `collision_factor` 0.6 |
| `variation` | mutation rate 1.0, no crossover |
| `simulator.calculator` | `fmax` 0.05, `steps_max` 100, FIRE |
| `mutation_funcs` | rattle and strain; an add and a remove per element in `general.elements`; a swap for every pair of elements |
| `evaluator` | composition feature over `general.elements`; energy per atom as the objective |

Your values are merged over these. A dict merges key by key, so
`simulator: {calculator: {steps_max: 20}}` keeps `fmax`. A list replaces the
default, so `mutation_funcs` is either all yours or all the defaults. If
`structure_selection_kwargs.atom_number_range` is set, it also bounds the
default add and remove mutations.

### HiSE: supercell escalation and the MD burst

`hise.supercells` lists one supercell per stage, each larger than the last.
Each stage starts from the previous one's structures, tiled into its cell. `hise.overrides` sets a
value per stage, one list entry per stage:

| Stage | Supercell | Generations | Population | Calculator |
|---|---|---|---|---|
| 1 | 1×1×1 | 20 | 100 | relax |
| 2 | 2×1×1 | 10 | 100 | relax |
| 3 | 2×2×1 | 2 | 250 | 40 MD steps at 300–600 K, then relax |

- **Why stages.** The first stage explores compositions cheaply in small
  cells. The second refines in larger cells. The last, in the largest cell,
  adds thermal distortions, which gives the committee force-rich structures
  to choose from.
- **Per-stage calculators.** `simulator.calculator` in `overrides` takes one
  dict per stage. It is merged over the top-level `simulator.calculator`
  settings and wrapped around the model like it. `T: [300, 600]` spreads the
  MD temperature over that range across the population.
- **Candidates come from the last stage only.** Its database also holds the
  earlier stages' structures, but ALomancy reads only the ones that stage
  generated.

### How many structures EZGA makes

Only the last stage is read, so its `max_generations × multiobjective.size`
must cover `num_of_structures_to_generate`. Here that is 2 × 250 = 500
for 250 candidates.
- **Below 1×** the run won't start.
- **Below 2×** it warns (`ezga_few_structures_configured`).

EZGA often makes fewer structures than that product. It drops duplicates,
and full relaxations of small cells often fall back into minima it has
already found, so keep the margin. If a loop still gets fewer than twice
`num_of_structures_per_loop` candidates, it logs `few_candidates_generated`.

When there are more than `num_of_structures_to_generate`, ALomancy takes the
newest generations whole. Only the oldest generation it needs is sampled
(seeded), so the candidates come from as late in the search as possible.

### Cost

**Model evaluations.** A loop's search is about Σ generations × population
model evaluations: 20 × 100 + 10 × 100 + 2 × 250 = 3,500, each up to 100
relaxation steps. Set `max_time` to cover it. Run a loop with `local: true`
on a small case first if you aren't sure.

**DFT cost.** Supercell escalation multiplies atom counts, and DFT cost with
them. The candidates come from the 2×2×1 stage, so they have four times
their seed's atoms. That is why the amorphous structures here are 10 atoms
(`num_of_atoms_per_amorphous`, default 20): they become 40-atom DFT
structures. Choose `supercells`, and the seed sizes, for the cells you want
to pay DFT for. Note that `atom_number_range` also caps the default add mutation
in every stage, so leave it off when stages tile beyond it.

## What you get

For loop `N`, in `results/al_loop_N/structure_generation/`:

```
ezga_seeds.xyz               the seeds (reused on restart)
ezga/
  initial_population.xyz     the seeds as EZGA read them
  supercell_1_1_1/           stage 1: EZGA's database, generation/, workflow.log
  supercell_2_1_1/           stage 2
  supercell_2_2_1/           stage 3
  ezga_candidates.xyz        the candidates, written last
```

Each candidate carries `ezga_generation` and `ezga_operation` (the mutation
that made it). It also carries `ezga_energy`, the model's energy and not a
DFT label. The committee's scores and picks follow as after MD.

The loop report has a **Structure generation: EZGA** section:
- the seeds, generations run, structures generated and candidates taken;
- the oldest generation used;
- the main settings.

The events table counts `ezga_summary`, and `ezga_restarted` if a search had
to start over.

## Restarts

`ezga_candidates.xyz` marks the search as done, so a restart after it reuses
the candidates.

If the run stopped mid-search, the next start moves the unfinished `ezga/`
directory to `ezga.failed_<date>` and runs the search again from the same
seeds. EZGA adds to an existing database rather than replacing it, so
starting over in place would mix two searches.

If the job fails, or the search generates nothing, the loop stops with an
error and nothing is cached. Restarting retries the search.

## Common errors at startup

| Message | Fix |
|---|---|
| `ezga_kwargs must not set: ...` | Remove the named setting; ALomancy sets it (see the table above). |
| `ezga_kwargs is now EZGA's own config ... Removed keys` | A 1.0.3 key (`population_size`, `mutation_operators`, `min_atoms`, `max_atoms`); the message names its new place (`deprecations.md`). |
| `EZGA makes about ... structures, fewer than ... num_of_structures_to_generate` | Raise the last stage's `max_generations` or `multiobjective.size`. |
| `... not in general.elements` | A mutation or feature names a species the run doesn't have. |
| `unknown ase_calculator settings` | A typo in `simulator.calculator` or a stage's calculator override. |
| A pydantic error from EZGA | A key EZGA doesn't know, or a wrong type, in its own config. |
