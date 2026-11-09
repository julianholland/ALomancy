# Examples

## Setting Up an HPC System with `alomancy add-hpc`

Before writing a run config, register your HPC system with the interactive wizard:

```bash
alomancy add-hpc
```

The wizard configures two files:
- `~/.expyre/config.json` — ExPyRe scheduler config (Slurm headers, partitions, scratch directory)
- `~/.alomancy/hpc_config.yaml` — ALomancy profile (venv path, DFT binary paths, node info)

> **Prerequisite:** Add the HPC to `~/.ssh/config` first so it is reachable by alias, e.g.:
> ```
> Host raven
>     HostName raven.mpcdf.mpg.de
>     User jholl
> ```
> Verify with: `ssh raven hostname`

### Example walkthrough

```
=== ALomancy HPC Setup Wizard ===

Configures two files:
  /home/jholl/.expyre/config.json
  /home/jholl/.alomancy/hpc_config.yaml

Before continuing, make sure this HPC is reachable by SSH alias ...

--- ExPyRe System (scheduler config) ---
System name in ~/.expyre/config.json (e.g. 'raven_gpu'): raven_gpu

Available SSH hosts from ~/.ssh/config:
  1) raven
  2) draco
  3) raccoon
  Enter a number to select, or type a hostname directly.
SSH host: 1                          # picks 'raven'

GPU system? [y/N]: y

Scratch/run directory — ExPyRe will create job subdirectories here.
Use a fast scratch filesystem, not your home directory. All results
are automatically synced back to your local machine after each job.
Scratch directory path on remote, e.g. /ptmp/user/alomancy_scratch: /ptmp/jholl/alomancy_scratch

Module/setup commands — press Enter after each command.
Enter on a blank line to finish.
Examples: 'module purge'  'module load python/3.11'  'export OMP_NUM_THREADS=1'
  > module purge
  > module load cuda/12.2 python/3.11
  > export OMP_NUM_THREADS=1
  >

Partitions (at least one required):
  Partition name (e.g. 'general'): gpubig
  Cores per node for 'gpubig': 18
  Max time for 'gpubig' [24:00:00]:
  Max memory for 'gpubig', e.g. 240GB: 120GB
  Add another partition? [y/N]: n

GPU SBATCH options (press Enter to skip each):
  Constraint string, e.g. gpu:
  Gres string, e.g. gpu:a100:1: gpu:a100:1

--- ALomancy HPC Profile ---
  (This name goes in your run YAML: hpc: '<profile_name>')
Profile name [raven_gpu]:
Venv activation command, e.g. source /u/user/.venvs/alomancy/bin/activate: source /u/jholl/.venvs/alomancy/bin/activate
TRITON_CACHE_DIR path for GPU PyTorch JIT cache, or Enter to skip: /u/jholl/.triton_cache

Which partition(s) will this profile use? (comma-separated) [gpubig]:
Ranks per node (usually = cores per node) [18]:
Max memory per node [120GB]:

Concurrency — how many ALomancy jobs should run on this HPC at once?
ExPyRe/Slurm still queue everything submitted; this caps how many are
started (occupying a queue slot) at the same time — the next queued job
starts the instant a running one finishes.
Number of concurrent jobs you wish to have running on this hpc from alomancy [20]:

DFT code on this system? Options: qe / vasp / none
DFT code [none]:

--- Writing config files ---
  /home/jholl/.expyre/config.json  ← added 'raven_gpu'
  /home/jholl/.alomancy/hpc_config.yaml  ← added 'raven_gpu'

--- Remote Installation ---
Install alomancy on this system now? [y/N]: y
Python executable path on remote [/u/jholl/.venvs/alomancy/bin/python]:
  Running: ssh raven '/u/jholl/.venvs/alomancy/bin/python -m pip install alomancy' …
  Done.

Setup complete! Use 'raven_gpu' in your run YAML:
  training:
    hpc: 'raven_gpu'
  structure_generation:
    hpc: 'raven_gpu'
  high_accuracy_evaluation:
    hpc: 'raven_gpu'
```

The wizard writes a profile like this to `~/.alomancy/hpc_config.yaml`:

```yaml
raven_gpu:
  hpc_name: raven_gpu
  gpu: true
  pre_cmds:
    - source /u/jholl/.venvs/alomancy/bin/activate
    - export TRITON_CACHE_DIR=/u/jholl/.triton_cache
  partitions:
    - gpubig
  node_info:
    ranks_per_system: 18
    ranks_per_node: 18
    threads_per_rank: 1
    max_mem_per_node: 120GB
  max_num_of_concurrent_jobs: 20
```

And this entry to `~/.expyre/config.json`:

```json
{
  "systems": {
    "raven_gpu": {
      "host": "raven",
      "remsh_cmd": "ssh",
      "scheduler": "slurm",
      "header": [
        "#SBATCH --no-requeue",
        "#SBATCH --nodes={num_nodes}",
        "#SBATCH --cpus-per-task={num_cores}",
        "#SBATCH --gres=gpu:a100:1"
      ],
      "commands": [
        "module purge",
        "module load cuda/12.2 python/3.11",
        "export OMP_NUM_THREADS=1"
      ],
      "partitions": {
        "gpubig": {"num_cores": 18, "max_time": "24:00:00", "max_mem": "120GB"}
      },
      "rundir": "/ptmp/jholl/scratch"
    }
  }
}
```

Run `alomancy add-hpc` again to add more profiles (e.g. a separate CPU profile for DFT).
Existing entries in both files are preserved.

> **DFT paths and `pseudo_dict`:** The wizard writes flat keys (`pwx_path`, `pp_path`,
> `vasp_path`) directly into the profile. The `pseudo_dict` (element → UPF/POTCAR mapping)
> must be added manually to `~/.alomancy/hpc_config.yaml` after the wizard finishes,
> because it is per-element and changes between projects.

---

## Basic Usage

Here's a simple example of running an active learning workflow:

```python
from alomancy import ALomancy

# The config picks everything: the AL skeleton (general.al_workflow) and
# every module (training.trainer, structure_generation.generator,
# high_accuracy_evaluation.evaluator). Every run setting (start_from,
# num_of_al_loops, verbose, ...) lives under the YAML's `general:` section.
ALomancy("standard_config.yaml").run()
```

## Configuration File

The configuration YAML file defines all the stages of the active learning workflow. Here's a complete example:

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["H", "O"]   # atomic symbols, not atomic numbers
  num_of_structures_per_loop: 50   # structures selected for DFT each loop
  dataset_kwargs:
    target_config_types:
      - "IsolatedAtom"
      - "init_dimer"
    test_ratio: 0.1
  committee_uncertainty_kwargs:
    num_of_models_in_committee: 5

initialization:
  max_time: "2H"
  mp_kwargs:
    max_num_of_atoms: 20
    mp_max_energy_above_hull: 0.1
  dimer_kwargs:
    num_of_dimers_per_combo: 10
  trimer_kwargs:
    num_of_trimers_per_combo: 5
  amorphous_kwargs:
    num_of_amorphous_structures: 300
    num_of_atoms_per_amorphous: 20
  stretch_compress_targets_kwargs:
    num_of_stretch_compress_per_target: 5
  hpc: 'my_hpc'

training:
  trainer: "mace"
  max_time: "5H"
  hpc: 'my_gpu_hpc'

structure_generation:
  generator: "md"   # "md" (default) or "ezga"
  num_of_structures_to_generate: 500   # candidate pool; default 10x per loop
  max_time: "10H"
  hpc: 'my_gpu_hpc'

high_accuracy_evaluation:
  evaluator: "qe"   # "qe" (default) or "vasp"
  max_time: "30m"
  hpc: 'my_cpu_hpc'   # concurrency is set on the HPC profile, see max_num_of_concurrent_jobs above
```

### Configuration Key Descriptions

- **general**: Settings shared across the whole workflow. `al_workflow` selects which AL skeleton `ALomancy(config)` builds (`"committee_uncertainty"`, the default, `"random_selection"` or `"novelty_selection"`). `elements` (atomic symbols, e.g. `["C", "O"]`) is the single shared source of element identity. `num_of_structures_per_loop` (default 50) is how many structures the selector sends to DFT each loop. `dataset_kwargs` sets the train/test split for every workflow: `target_config_types` (which config types count toward the split) and `test_ratio` (the fraction that goes to test). `remove_redundancy` (default true) flags near-duplicate training structures; `redundancy_kwargs` tunes how its tolerance is chosen: `tolerance` (set it directly), `max_flagged_fraction` (default 0.5: if the automatic choice would flag more, nothing is flagged), `min_plateau_width` and `probe_steps_per_structure`. Settings specific to one AL skeleton go in its own block, e.g. `committee_uncertainty_kwargs.num_of_models_in_committee` (how many committee members are trained in parallel).

- **initialization**: Generates initial training and test sets. Supports Materials Project structures, dimers, trimers, amorphous structures (`density_range: [min, max]`), molecules, and stretched/compressed and rattled target structures — each namespaced under its own `*_kwargs` (`mp_kwargs`, `dimer_kwargs`, `trimer_kwargs`, `amorphous_kwargs`, `stretch_compress_targets_kwargs`, `isolated_atom_kwargs`), each with its own `enabled` flag (default `true`). Molecules are two more blocks, off by default: `molecule_kwargs` (single molecules in vacuum) and `molecules_in_box_kwargs` (periodic boxes of molecules at a density range), each naming its own molecules (below). `read_generated_file` uses a ready-made extxyz file instead of generating. Every setting is shown, commented, in [Initialization in Full](#initialization-in-full).

  ```yaml
  general:
    elements: ["C", "H", "O"]
    dataset_kwargs:
      # Stretch/compress and rattle the molecules too (bond-length scans).
      target_config_types: ["init_MP", "init_amorphous", "init_molecule"]
      test_ratio: 0.1
  initialization:
    molecule_kwargs:
      enabled: true
      molecules: ["CO", "smiles-O", "smiles-CCO"]   # SAGE formula or SMILES
    molecules_in_box_kwargs:
      enabled: true
      box_compositions:
        - molecules: {"smiles-O": 2, "smiles-CO": 1}   # water:methanol 2:1
          density_range: [0.8, 1.1]                    # g/cm^3
          num_of_molecules: 24
          num_of_structures: 4                         # random densities in range
  ```

- **training**: Trains the interatomic potential(s) the workflow needs (a committee for `committee_uncertainty`, one model for `random_selection` and `novelty_selection`). `trainer` selects the registered `mlip_trainer` backend: `"mace"` (default) or `"sevennet"`. Backend-specific settings go under `mace_kwargs` / `sevennet_kwargs` (SevenNet's own `model`/`train`/`data` input sections; requires `sevenn` on the training nodes). Complete, tested configs for each workflow and backend are in `examples/configs/`.

- **structure_generation**: Generates candidate structures for labeling. `num_of_structures_to_generate` (default 10 × `general.num_of_structures_per_loop`) is the size of the candidate pool; it must be at least `num_of_structures_per_loop` (error) and should be at least twice it (warning). `generator` selects the registered `structure_generator` backend (`"md"`, the default, or `"ezga"` for genetic-algorithm search, configured as in [Exploring with EZGA and HiSE](ezga.md)); uncertainty is measured as force standard deviation across the committee. MD parameters (`steps`, `temperature`, `timestep_fs`, `friction`, `ensemble`, `pressure`) go under `md_kwargs`:

  ```yaml
  structure_generation:
    generator: "md"
    num_of_structures_to_generate: 500   # candidate pool; default 10x per loop
    max_time: "10H"
    md_kwargs:
      steps: 20000
      temperature: 1200
      timestep_fs: 0.5
      friction: 0.002
      ensemble: "npt"   # "nvt" (default, fixed cell) or "npt" (variable cell)
      pressure: 0.0      # GPa; only used when ensemble is "npt"
    hpc: 'my_gpu_hpc'
  ```

  `md_kwargs.snapshot_oversample` (default 10) makes each MD run record that many times more snapshots than candidates are needed. Every snapshot stays in `md_output_i/structure_generation.xyz` as a viewable trajectory (each frame's MD step in `md_frame`). The candidates are the physical frames (no seed frame, no bond under 0.5 Å), spread evenly over each run, up to `num_of_structures_to_generate`; a run that goes unstable early still contributes the frames it recorded before that. Set it to 1 for the old one-snapshot-per-candidate behaviour.

  `md_kwargs.max_volume_ratio` (default 3; `null` turns it off) stops a run's frames from becoming candidates once its volume per atom passes that multiple of its starting structure's. Under NPT, a model whose stress can't hold the cell together lets it grow without bound, and DFT on such cells runs out of memory or fails. When the filter cuts a run, the warning `md_volume_expanded` suggests a small positive `pressure` (e.g. 0.1 GPa) to keep the cell compact, or `ensemble: "nvt"`.

  `mlip_relaxation` (off by default) relaxes every generated candidate with the loop's best model before selection, so the committee (or random/novelty selection) chooses among relaxed structures, the ones DFT starts from, instead of raw MD snapshots. A candidate whose relaxation fails or ends with a bond shorter than 0.5 Å is dropped (warning `mlip_relax_dropped`); if the whole relaxation job fails, the unrelaxed candidates are used (warning `mlip_relax_fallback`). DFT still relaxes anything whose forces exceed `high_accuracy_evaluation.force_ceiling`. EZGA candidates are already relaxed by EZGA, so this is meant for MD. It adds a remote function: run `alomancy upgrade-hpc` before switching it on.

  ```yaml
  structure_generation:
    mlip_relaxation:
      fmax: 0.05          # eV/Å (default 0.05)
      steps: 200          # optimiser step budget (default 200)
      optimizer: "FIRE"   # "FIRE" (default), "BFGS" or "LBFGS"
  ```

  `ensemble: "nvt"` runs fixed-cell Langevin dynamics (the default). `ensemble: "npt"` runs ASE's `LangevinBAOAB` integrator with a barostat targeting `pressure` (GPa), letting the cell shape and volume fluctuate — useful when candidate structures should sample compressed/expanded states rather than just the seed cell's fixed volume.

- **high_accuracy_evaluation**: Performs high-accuracy DFT evaluation on selected structures. The `evaluator` key selects the registered `dft_evaluator` backend: `"qe"` (Quantum Espresso, default), `"vasp"` or `"orca"` (molecules only, see [Using ORCA](#using-orca-as-the-evaluator)). Every label records its level of theory, and ALomancy warns at start-up (`level_of_theory_mismatch`) if the database already holds labels from a different one. Submission concurrency (how many jobs run at once) is controlled by `max_num_of_concurrent_jobs` on the HPC profile (`~/.alomancy/hpc_config.yaml`, default 20) — a property of the HPC system/account, not a per-workflow-phase setting; see [Deprecations](deprecations.md) for the removed `max_batch_size` fallback. Settings for the evaluator you didn't select (e.g. `qe_kwargs` while `evaluator: "vasp"`) are reported at start-up and ignored. `max_vdw_volume_ratio` (default 10; `null` turns it off) keeps blown-up cells away from DFT: a periodic structure whose volume is more than that many times its atoms' van der Waals volume is dropped, both from the candidates before selection and from anything else about to go to DFT, with the warning `blown_up_excluded`. Healthy condensed cells sit near 1 and molecular crystals up to about 10; cells far above run DFT out of memory or fail. Isolated atoms, dimers and trimers are exempt, as are the config types listed in `vdw_exempt_config_types` (e.g. `["molecules_in_vacuum"]`): tag structures that are meant to have vacuum, such as molecules in a box, with their own `config_type` and list it there. If genuine low-density crystals are dropped, raise `max_vdw_volume_ratio`.

## Initialization in Full

On a cold start (no `general.start_from`), the `initialization` section builds the starting training set before loop 0. Each block below makes one kind of structure and has its own `enabled` flag. Everything it makes is labelled by DFT through `high_accuracy_evaluation`, then split into train and test by `general.dataset_kwargs`. On a restart, only structures still missing from the database are made again. The block shows every setting with example values; where one differs from the default, its comment gives the default. An empty `initialization: {}` runs every default; the two molecule blocks are off unless you turn them on and name their molecules.

```yaml
initialization:
  # Initialization builds the starting training set before the first AL loop.
  # Every block below is a separate kind of structure; each has
  # `enabled` (default true; false for rattle and the two molecule blocks)
  # to turn it off without zeroing its count. All structures are labelled by DFT through the
  # high_accuracy_evaluation section (its evaluator and hpc), then split into
  # train/test using general.dataset_kwargs. On a restart, only the
  # structures still missing from the database are regenerated.

  # Generation runs on the local machine (no remote job), so these two are
  # accepted for symmetry with the other sections but not used for anything.
  max_time: "2H"
  hpc: "my_hpc"

  # Optional: skip generation and use your own structures instead (an extxyz
  # file, path relative to results/initialization/). If the file is missing or
  # empty, ALomancy falls back to generating them as configured below.
  read_generated_file: null

  # One isolated atom per element in general.elements, in a box (3x the vdW
  # radius). Their DFT energies are the per-element reference energies used
  # by the trainer and for formation energies in the plots and filters,
  # so keep this on.
  isolated_atom_kwargs:
    enabled: true

  # Two-atom structures for every element pair, at evenly spaced separations
  # from 0.2x to 1.0x the sum of the vdW radii. These teach the model
  # short-range repulsion (default 10 per pair).
  dimer_kwargs:
    enabled: true
    num_of_dimers_per_combo: 20

  # Three-atom structures for every element triple, on a grid of the two
  # bond lengths over the same range as the dimers (default 5 per triple).
  trimer_kwargs:
    enabled: true
    num_of_trimers_per_combo: 10

  # Random amorphous cells: atoms placed at random at a given density, then
  # pushed apart with a Lennard-Jones relaxation so no two overlap.
  amorphous_kwargs:
    enabled: true
    num_of_amorphous_structures: 100   # total
    num_of_atoms_per_amorphous: 20     # atoms per cell
    density_range: [1.75, 2.25]        # g/cm^3, [min, max]; each cell's density
                                       # random within it (default [1.0, 1.0])
    # Fixed compositions: a list of cells, each a list of exactly
    # num_of_atoms_per_amorphous symbols, e.g. [["Li", "Li", ..., "S"], ...];
    # repeated or sampled to reach the total. null -> random mixes of
    # general.elements.
    composition_list: null
    # Placement, compositions and densities are seeded by general.seed.

  # Known crystals from the Materials Project that contain only
  # general.elements. Needs the MP_API_KEY environment variable to be set.
  mp_kwargs:
    enabled: true
    max_num_of_atoms: 50               # skip larger structures (default 20)
    mp_max_energy_above_hull: 0.1      # eV/atom; 0 keeps only stable phases

  # Single molecules in vacuum (config_type init_molecule; off by default).
  # One structure per molecule at its equilibrium geometry, in a cubic cell
  # of its largest extent plus `vacuum`. A molecule is a SAGE formula ("CO",
  # "H2O") or "smiles-<SMILES>" ("smiles-CCO"); a formula SAGE has no preset
  # for (CH4) is written as SMILES ("smiles-C"). Every element must be in
  # general.elements (the example needs C and O). For bond-length scans,
  # list init_molecule in general.dataset_kwargs.target_config_types: the
  # stretch/compress and rattle blocks below then distort the molecules too.
  molecule_kwargs:
    enabled: true
    molecules: ["CO", "CO2", "O2"]     # required when enabled
    vacuum: 10.0                       # Å around the molecule

  # Periodic boxes of molecules (config_type init_molecules_in_box; off by
  # default), e.g. liquids and mixtures. Each box_compositions entry gives the
  # molecules and their ratio, and makes num_of_structures boxes, each at a
  # random density within its density_range and sized so it is exact.
  # (Like the molecule_kwargs example, these need C and O in general.elements;
  # with H as well, water:methanol 2:1 is {"smiles-O": 2, "smiles-CO": 1}.)
  # The molecules are given random rotations and positions (seeded with
  # general.seed) and kept min_distance apart; they are relaxed by DFT, like
  # the amorphous cells.
  molecules_in_box_kwargs:
    enabled: true
    box_compositions:                  # required when enabled
      - molecules: {"CO": 2, "CO2": 1}   # CO:CO2 2:1
        density_range: [0.8, 1.1]      # g/cm^3, [min, max]
        num_of_molecules: 24           # per box, split by the ratio
        num_of_structures: 4           # 4 boxes, random densities in 0.8-1.1
      - molecules: {"CO": 1}           # pure CO
        density_range: [0.5, 0.5]
        num_of_molecules: 16
        num_of_structures: 1
    min_distance: 1.5                  # Å between atoms of different molecules
    max_attempts_per_molecule: 1000    # placements tried before giving up

  # Isotropic strains of every "target" structure generated here (those whose
  # config_type is in general.dataset_kwargs.target_config_types, e.g.
  # init_MP, init_amorphous, init_molecule). Covers the equation of state
  # around each one; on a molecule in vacuum it scales the bonds.
  stretch_compress_targets_kwargs:
    enabled: true
    num_of_stretch_compress_per_target: 5  # evenly spaced scalings
    max_lattice_deformation: 0.2           # +/-20 % on the lattice vectors

  # Randomly displaced copies of the same target structures (off by default).
  # Adds off-equilibrium forces around each starting geometry.
  rattle_target_structures:
    enabled: false
    num_of_rattled_per_target: 5
    rattle_standard_deviation: 0.05    # Å; no default, required when enabled
```

- Each `composition_list` entry must have exactly `num_of_atoms_per_amorphous` symbols.
- `amorphous_kwargs.densities_list` is deprecated: use `density_range` (see [Deprecations](deprecations.md)).
- Densities are in g/cm³, so the same number is very sparse for heavy elements: Pt at the default 1 g/cm³ fills 14× its atoms' van der Waals volume, and `high_accuracy_evaluation.max_vdw_volume_ratio` (default 10) would drop every such cell before DFT. ALomancy warns about this when the workflow is built (`init_density_too_low`), and about densities packing atoms tighter than any solid (`init_density_too_high`), naming the g/cm³ window to use. Random amorphous mixes are checked as an even mix of `general.elements`.
- Molecule names, every `density_range` and the box settings are checked when the workflow is built, so a bad name fails at startup, not after DFT has started. A box too small for its largest molecule, or one that can't be packed in `max_attempts_per_molecule` tries, raises naming the composition and density: lower the density or `min_distance`, or raise `num_of_molecules`.
- On a restart, molecules and boxes are counted per chemical formula, so only the missing ones are made again.
- `general.dataset_kwargs.target_config_types` decides which structures are stretched, compressed and rattled.
- Dimers and trimers are far from equilibrium, so they are usually listed in `general.train_filter.exempt_config_types` (see [Filtering the Training and Test Sets](#filtering-the-training-and-test-sets)).

## Using VASP as the DFT Backend

To switch from Quantum Espresso to VASP, set `evaluator: "vasp"` in the `high_accuracy_evaluation` block and replace the QE-specific HPC keys:

```yaml
high_accuracy_evaluation:
  evaluator: "vasp"
  max_time: "30m"
  vasp_kwargs:                # INCAR overrides (optional)
    encut: 600
    ediff: 1.0e-7
  hpc:
    hpc_name: "raven"
    vasp_path: "/path/to/vasp_std"
    pp_path: "/path/to/potpaw_PBE"  # sets VASP_PP_PATH before each calculation
    pseudo_dict:              # element → POTCAR suffix
      H: ""
      O: "_GW"
    node_info:
      ranks_per_system: 72
      ranks_per_node: 36
      threads_per_rank: 1
      max_mem_per_node: "90G"
    max_num_of_concurrent_jobs: 20   # jobs started at once on this HPC (default 20)
    partitions: ["cpu"]
    pre_cmds: ["module load vasp"]
```

> **`pp_path` and `VASP_PP_PATH`:** ASE's `Vasp` calculator locates POTCAR files via
> the `VASP_PP_PATH` environment variable, not a constructor argument. `create_vasp_calc_object`
> sets `os.environ["VASP_PP_PATH"] = hpc["pp_path"]` before building the calculator, so
> `pp_path` must point at the directory *containing* `potpaw_PBE` (or `potpaw_LDA`), matching
> what `module load vasp` would otherwise export. Elements without a `pseudo_dict` override
> fall back to the default (unsuffixed) POTCAR under that path — if that directory doesn't
> carry a given element (e.g. `Pd`), VASP fails with `No pseudopotential for <element>!`.

QE configs work unchanged — `evaluator: "qe"` is the default and can be omitted.

## Using ORCA as the Evaluator

ORCA is a molecular code: it has no periodic boundary conditions, and besides DFT it runs wavefunction methods such as MP2 and coupled cluster. Set `evaluator: "orca"` and give the ORCA keyword line in `orca_kwargs`:

```yaml
high_accuracy_evaluation:
  evaluator: "orca"
  max_time: "2H"
  orca_kwargs:
    orcasimpleinput: "RI-MP2 def2-TZVP def2-TZVP/C TightSCF"
    orcablocks: "%scf maxiter 300 end"   # extra % blocks (optional)
    max_num_of_atoms_correlated: 20      # default; null = no cap
  hpc:
    hpc_name: "raven"
    orca_path: "/path/to/orca/orca"      # absolute path, required
    node_info:
      ranks_per_system: 16
      ranks_per_node: 72
      threads_per_rank: 1
      max_mem_per_node: "240G"
    partitions: ["cpu"]
    pre_cmds: ["module load openmpi/4.1"]  # the OpenMPI ORCA was built with

initialization:
  # With ORCA, the periodic blocks (amorphous_kwargs, mp_kwargs,
  # molecules_in_box_kwargs) are off by default and these two are on.
  amorphous_cluster_kwargs:
    num_of_amorphous_clusters: 20
    num_of_atoms_per_cluster: 6
    density_range: [0.5, 1.0]
  molecule_cluster_kwargs:
    cluster_compositions:
      - molecules: {"CO": 1}
        density_range: [0.5, 1.0]
        num_of_molecules: 3
        num_of_structures: 5
```

- **Only non-periodic structures.** Anything with `pbc` true on any axis raises before submission.
  - Initialisation therefore defaults to non-periodic structures: isolated atoms, dimers, trimers, molecules, amorphous clusters (`amorphous_cluster_kwargs`) and clusters of molecules (`molecule_cluster_kwargs`, built only when you give `cluster_compositions`).
  - Enabling `amorphous_kwargs`, `mp_kwargs` or `molecules_in_box_kwargs` yourself raises, naming the replacement.
  - Starting from a database that holds periodic structures raises too.
- **Parallelism goes in the ORCA input.** ALomancy writes `%pal nprocs` (`node_info.ranks_per_system`) and `%maxcore` (75% of `max_mem_per_node` split between the processes).
  - To use other values, set `orca_kwargs.nprocs` / `orca_kwargs.maxcore` (MB), or write a `%pal`/`%maxcore` block or a `PALn` keyword yourself. Your value wins, but must fit one node of the profile, or the run stops at start-up.
  - ORCA is started by `orca_path` without srun and starts its own `mpirun`, so load its OpenMPI in `pre_cmds`.
- **Post-HF methods are expensive.** For correlated and multireference methods (MP2, DLPNO-CCSD(T), double hybrids, CASSCF/NEVPT2), structures with more than `max_num_of_atoms_correlated` atoms are dropped before submission (`too_many_atoms_for_method`).
  - Initialisation blocks configured to make larger structures stop the run at start-up.
  - Coupled-cluster and multireference forces may need numerical gradients (about 6 × atoms energy calculations each); ALomancy warns about this at start-up.
- `EnGrad` (forces) and `SCFConvergenceForced` are added to the keyword line when missing.
- The spin multiplicity is the lowest the electron count allows, the ground state for isolated atoms (H 2, O 3), or `atoms.info["spin_multiplicity"]`; the charge is `orca_kwargs.charge` (default 0) or `atoms.info["charge"]`.

## Adding a New Backend

Users never pick a workflow class in Python — `ALomancy(config)` builds the skeleton named by `general.al_workflow` (new skeletons: see `writing_a_workflow.md`), and each pluggable category (`mlip_trainer`, `structure_generator`, `dft_evaluator`, `initialiser`) is a module registered against a name in `src/alomancy/registry.py`, resolved lazily from config at runtime (`training.trainer`, `structure_generation.generator`, `high_accuracy_evaluation.evaluator`).

Adding a new backend means writing a new module that implements the category's expected entry points — typically `output_paths(config, *, base_name, name, ...)` and `read_existing_result(config, *, base_name, name, ...)` for the skeleton's restart mechanism, plus the category-specific worker function (`train`, `generate`, or `sp`/`go`) — and registering it in `registry.py`. See any existing module under `mlip/` (e.g. `mlip/mace/trainer.py`), `structure_generation/` (`structure_generation/md/md_wfl.py`, `structure_generation/ezga/generate_structures.py`), or `high_accuracy_evaluation/dft/` (`run_qe.py`, `run_vasp.py`) and `high_accuracy_evaluation/orca/` (`run_orca.py`) for the pattern to follow. A new evaluator's checklist (periodicity, parallel model, level of theory, atom cap) is in `docs/architecture/high_accuracy_evaluation.md`.

## Starting From Existing Data

External or earlier datasets are imported with `general.start_from`
instead of being generated from scratch. For example, a single xyz file:

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["H", "O"]
  start_from:
    xyz:
      - "path/to/external_structures.xyz"
      - "path/to/another_dataset.xyz"
  dataset_kwargs:
    test_ratio: 0.1
    target_config_types: ["liquid"]
```

Structures don't need ALomancy's own keys: `config_type`, `REF_energy`
and `REF_forces` are mapped from common equivalents (or named explicitly
with `start_from.metadata_map`). See [Starting a run](starting_a_run.md)
for all start modes and the full rules.

## Filtering the Training and Test Sets

Poor-quality DFT structures, such as near-collision MD frames with huge
forces, can be kept out of either split without deleting them from the
database. `general.train_filter` and `general.test_filter` take the same keys:

```yaml
general:
  train_filter:
    max_force: 100.0                  # eV/Å; the default. null disables it
    formation_energy_per_atom: null   # off by default
    exempt_config_types: ["init_dimer", "init_trimer", "init_stretch_compress"]
  test_filter:                        # off by default
    max_force: 20.0
    formation_energy_per_atom: [null, 1.0]   # eV/atom; null bound = open
```

- `max_force` excludes structures whose largest per-atom force is at or above the value.
- `formation_energy_per_atom` keeps only structures inside `[min, max]` (inclusive), measured against the database's `IsolatedAtom` energies: `(E − Σ E_isolated) / n_atoms`, the same definition as the parity plots. Structures containing an element without an `IsolatedAtom` energy are not energy-filtered, and a warning says so.
- `exempt_config_types` lists config_types the whole filter skips (both criteria), for example the initialization's dimers and stretched/compressed crystals, whose large forces are the short-range data MD needs. Exempt structures are counted in the filter's log line, and removing an exemption filters them again in the next loop.

Excluded structures are flagged (`is_quality_filtered`, with the reason in `quality_filter_reasons`) and left out of `train_set.xyz`/`test_set.xyz`, training and test metrics. The flags are recomputed every loop, so loosening or removing a filter brings them back.

Separately, `high_accuracy_evaluation.force_ceiling` (default `100.0` eV/Å) controls how far AL-generated structures are relaxed before their DFT labels are kept: each is relaxed until its largest force is at most the ceiling. `null` means a single-point calculation with no relaxation. Initialization structures are not affected.

## Model Predictions in the GlobalDatabase

After each loop's training, every model is evaluated on its training, validation and test sets on the remote node, before the job returns, whichever trainer is configured. The predictions are written to `train_pred.xyz` / `valid_pred.xyz` / `test_pred.xyz` in each fit directory (`results/al_loop_N/training/fit_i/`) and synced back with the model files.

ALomancy then reads those files locally and stores the predicted energies and forces in the GlobalDatabase (`store_model_predictions`), with no model loading or local GPU needed. Parity plots read from the database first, fall back to the prediction files, and never run local inference.

The stored keys follow the pattern:
```
model_energy_loop_{N}_fit_{i}   # predicted energy (eV, raw)
model_forces_loop_{N}_fit_{i}   # predicted forces ([[fx,fy,fz], ...], eV/Å)
```

You can retrieve predictions programmatically:

```python
from alomancy.database.global_database import GlobalDatabase

db = GlobalDatabase("results/global_database")

# Returns {"train": (e_dft, e_pred, f_dft, f_pred), "test": (...)}
# where each element is a numpy array; e values are per-atom (eV/atom)
preds = db.get_model_predictions(loop_idx=0, fit_idx=0)

e_dft, e_pred, f_dft, f_pred = preds["train"]
print(f"Train energy MAE: {abs(e_dft - e_pred).mean():.4f} eV/atom")
```
