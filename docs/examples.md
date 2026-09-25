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
  max_concurrent_jobs: 20
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
from alomancy.configs.config_dictionaries import load_dictionaries
from alomancy.core.committee_uncertainty_workflow import build_workflow

# Load configuration from YAML file
jobs_dict = load_dictionaries("standard_config.yaml")

# Create and run the workflow
workflow = build_workflow(
    jobs_dict=jobs_dict,
    initial_train_file_path="results/initialization/train_set.xyz",
    initial_test_file_path="results/initialization/test_set.xyz",
    number_of_al_loops=5,
    verbose=1,  # 0=silent, 1=INFO progress, 2=DEBUG
    log_file="results/alomancy.log",  # file always captures DEBUG
    db_path="results/global_database",
)

workflow.run()
```

## Configuration File

The configuration YAML file defines all the stages of the active learning workflow. Here's a complete example:

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["H", "O"]   # atomic symbols, not atomic numbers
  committee_uncertainty_kwargs:
    number_models_in_committee: 5
    target_config_types:
      - "IsolatedAtom"
      - "init_dimer"
    test_ratio: 0.1

initialization:
  name: "initialization"
  max_time: "2H"
  mp_kwargs:
    max_atom_number: 20
    mp_max_energy_above_hull: 0.1
  dimer_kwargs:
    num_dimers_per_combo: 10
  trimer_kwargs:
    num_trimers_per_combo: 5
  amorphous_kwargs:
    num_amorphous: 300
    amorphous_atom_number: 20
  stretch_compress_targets_kwargs:
    num_stretch_compress_per_mp: 5
  hpc: 'my_hpc'

training:
  name: "training"
  trainer: "mace"
  max_time: "5H"
  hpc: 'my_gpu_hpc'

structure_generation:
  name: "structure_generation"
  generator: "md"   # "md" (default) or "ezga"
  desired_number_of_structures: 50
  max_time: "10H"
  hpc: 'my_gpu_hpc'

high_accuracy_evaluation:
  name: "high_accuracy_evaluation"
  evaluator: "qe"   # "qe" (default) or "vasp"
  max_time: "30m"
  hpc: 'my_cpu_hpc'   # concurrency is set on the HPC profile, see max_concurrent_jobs above
```

### Configuration Key Descriptions

- **general**: Settings shared across the whole workflow. `al_workflow` selects which AL skeleton `build_workflow()` returns (currently only `"committee_uncertainty"`). `elements` (atomic symbols, e.g. `["C", "O"]`) is the single shared source of element identity. `committee_uncertainty_kwargs` holds everything specific to this AL skeleton: `number_models_in_committee` (how many committee members are trained in parallel), `target_config_types` (which config types count toward the train/test split), and `test_ratio` (the split between test and training data).

- **initialization**: Generates initial training and test sets. Supports Materials Project structures, dimers, trimers, amorphous structures, and stretched/compressed MP structures — each namespaced under its own `*_kwargs` (`mp_kwargs`, `dimer_kwargs`, `trimer_kwargs`, `amorphous_kwargs`, `stretch_compress_targets_kwargs`, `isolated_atom_kwargs`), each with its own `enabled` flag (default `true`).

- **training**: Trains an ensemble (committee) of interatomic potentials. `trainer` selects the registered `mlip_trainer` backend (currently only `"mace"`); backend-specific settings go under `mace_kwargs`.

- **structure_generation**: Generates candidate structures for labeling. `generator` selects the registered `structure_generator` backend (`"md"`, the default, or `"ezga"` for genetic-algorithm search); uncertainty is measured as force standard deviation across the committee. MD parameters (`steps`, `temperature`, `timestep_fs`, `friction`, `ensemble`, `pressure`) go under `md_kwargs`:

  ```yaml
  structure_generation:
    name: "structure_generation"
    generator: "md"
    desired_number_of_structures: 50
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

  `ensemble: "nvt"` runs fixed-cell Langevin dynamics (the default). `ensemble: "npt"` runs ASE's `LangevinBAOAB` integrator with a barostat targeting `pressure` (GPa), letting the cell shape and volume fluctuate — useful when candidate structures should sample compressed/expanded states rather than just the seed cell's fixed volume.

- **high_accuracy_evaluation**: Performs high-accuracy DFT evaluation on selected structures. The `evaluator` key selects the registered `dft_evaluator` backend: `"qe"` (Quantum Espresso, default) or `"vasp"`. Submission concurrency (how many jobs run at once) is controlled by `max_concurrent_jobs` on the HPC profile (`~/.alomancy/hpc_config.yaml`, default 20) — a property of the HPC system/account, not a per-workflow-phase setting; see [Deprecations](deprecations.md) for the removed `max_batch_size` fallback. If QE-specific keys (e.g. `pwx_path`) appear in a VASP config or vice versa, a warning is logged and the mismatched keys are ignored.

## Using VASP as the DFT Backend

To switch from Quantum Espresso to VASP, set `evaluator: "vasp"` in the `high_accuracy_evaluation` block and replace the QE-specific HPC keys:

```yaml
high_accuracy_evaluation:
  name: "high_accuracy_evaluation"
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
    max_concurrent_jobs: 20   # jobs started at once on this HPC (default 20)
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

## Adding a New Backend

There's no subclassing to extend ALomancy — `CommitteeUncertaintyWorkflow` (built via `build_workflow()`) is the only workflow implementation, and each pluggable category (`mlip_trainer`, `structure_generator`, `dft_evaluator`, `initialiser`) is a module registered against a name in `src/alomancy/registry.py`, resolved lazily from config at runtime (`training.trainer`, `structure_generation.generator`, `high_accuracy_evaluation.evaluator`).

Adding a new backend means writing a new module that implements the category's expected entry points — typically `output_paths(config, *, base_name, name, ...)` and `read_existing_result(config, *, base_name, name, ...)` for the skeleton's restart mechanism, plus the category-specific worker function (`train`, `generate`, or `sp`/`go`) — and registering it in `registry.py`. See any existing module under `mlip/` (e.g. `mlip/mace/trainer.py`), `structure_generation/` (`structure_generation/md/md_wfl.py`, `structure_generation/ezga/generate_structures.py`), or `high_accuracy_evaluation/dft/` (`run_qe.py`, `run_vasp.py`) for the pattern to follow.

## Extra Datasets

The initialization configuration can include external datasets via an `extra_datasets` parameter. These structures are seeded into the GlobalDatabase before initialization runs. This is useful for incorporating reference data (e.g., from literature or previous computations) without regenerating isolated atoms.

Example configuration:

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["H", "O"]
  committee_uncertainty_kwargs:
    test_ratio: 0.1
    # ... other options ...

initialization:
  name: "initialization"
  max_time: "2H"
  extra_datasets:
    - "path/to/external_structures.xyz"
    - "path/to/another_dataset.xyz"
```

Structures in extra datasets should have:
- `atoms.info["REF_energy"]` (float) — DFT energy
- `atoms.arrays["REF_forces"]` (array, shape N×3) — DFT forces
- `atoms.info["config_type"]` (str) — origin label (e.g. `"IsolatedAtom"`, `"external_data"`)

## MACE Committee Predictions in the GlobalDatabase

After each AL loop's MACE committee training finishes on the remote GPU node, ALomancy evaluates every committee model on the training and test sets **before returning from the remote job** and saves the per-structure predictions to `train_pred.xyz` / `test_pred.xyz` inside each fit directory. These files are synced back to your local machine by ExPyRe alongside the model files.

The skeleton then reads those files locally and stores the predicted energies and forces in the GlobalDatabase via `store_model_predictions` — no model loading or local GPU required. Parity plots (`plot_dft_vs_model`) read from the DB first, then fall back to the eval xyz files, and never run local inference.

> **Resuming from before v0.4.2**: loops trained without the post-training eval step will not have `train_pred.xyz` / `test_pred.xyz`. The sentinel `mace_predictions.done` is written and those loops are skipped gracefully; parity plots show "No predictions available" for them. Only future loops (trained with the updated remote package) will have predictions stored.

The stored metadata keys follow the pattern (generalized to `model_*` since the trainer backend is no longer assumed to be MACE — see the module registry):
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

Predictions are guarded by a `results/al_loop_{N}/mace_predictions.done` sentinel so they are not recomputed on restart. If you need to regenerate predictions (e.g. after installing a different model), delete that sentinel file.
