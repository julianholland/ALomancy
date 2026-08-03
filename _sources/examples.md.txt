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
  mlip_committee:
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
from alomancy.core.standard_active_learning import ActiveLearningStandardMACE

# Load configuration from YAML file
jobs_dict = load_dictionaries("standard_config.yaml")

# Create and run the workflow
workflow = ActiveLearningStandardMACE(
    initial_train_file_path="results/initialization/train_set.xyz",
    initial_test_file_path="results/initialization/test_set.xyz",
    jobs_dict=jobs_dict,
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
initialization:
  name: "initialization"
  max_time: "2H"
  test_to_train_ratio: 0.1
  test_config_types:
    - "IsolatedAtom"
    - "init_dimer"
  creation_kwargs:
    elements: ["H", "O"]
    mp_structures: true
    single_atoms: true
    num_dimers_per_combo: 10
    num_trimers_per_combo: 5
    num_amorphous: 300
    num_stretch_compress_per_mp: 5
    max_atom_number: 20
    amorphous_atom_number: 20
    mp_max_energy_above_hull: 0.1
  hpc: 'my_hpc'

mlip_committee:
  name: "mlip_committee"
  size_of_committee: 5
  max_time: "5H"
  hpc: 'my_gpu_hpc'

structure_generation:
  name: "structure_generation"
  desired_number_of_structures: 50
  max_time: "10H"
  hpc: 'my_gpu_hpc'

high_accuracy_evaluation:
  name: "high_accuracy_evaluation"
  calculator: "qe"   # "qe" (default) or "vasp"
  max_time: "30m"
  hpc: 'my_cpu_hpc'   # concurrency is set on the HPC profile, see max_concurrent_jobs above
```

### Configuration Key Descriptions

- **initialization**: Generates initial training and test sets. Supports Materials Project structures, dimers, trimers, amorphous structures, and stretched/compressed MP structures. The `test_to_train_ratio` determines the split between test and training data.

- **mlip_committee**: Trains an ensemble (committee) of MACE interatomic potentials. The `size_of_committee` parameter determines how many committee members are trained in parallel.

- **structure_generation**: Uses MD to generate candidate structures for labeling. Uncertainty is measured as force standard deviation across the committee.

- **high_accuracy_evaluation**: Performs high-accuracy DFT evaluation on selected structures. The `calculator` key selects the backend: `"qe"` (Quantum Espresso, default) or `"vasp"`. Submission concurrency (how many jobs run at once) is controlled by `max_concurrent_jobs` on the HPC profile (`~/.alomancy/hpc_config.yaml`, default 20) — not a per-workflow-phase setting, since it's a property of the HPC system/account. The older job-dict-level `max_batch_size` key is deprecated and will be removed in 1.0.0; see [Deprecations](deprecations.md). If QE-specific keys (e.g. `pwx_path`) appear in a VASP config or vice versa, a warning is logged and the mismatched keys are ignored.

## Using VASP as the DFT Backend

To switch from Quantum Espresso to VASP, set `calculator: vasp` in the `high_accuracy_evaluation` block and replace the QE-specific HPC keys:

```yaml
high_accuracy_evaluation:
  name: "high_accuracy_evaluation"
  calculator: "vasp"
  max_time: "30m"
  vasp_input_kwargs:          # INCAR overrides (optional)
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

QE configs work unchanged — `calculator: qe` is the default and can be omitted.

## Custom Workflows

You can create custom active learning workflows by extending `BaseActiveLearningWorkflow`. You must implement four abstract methods:

```python
from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
import pandas as pd
from ase.atoms import Atoms


class MyCustomWorkflow(BaseActiveLearningWorkflow):
    def initialize_training_set(
        self, base_name, **kwargs
    ) -> tuple[list[Atoms], list[Atoms]]:
        """
        Generate initial training and test sets.

        Args:
            base_name: Name used for output directories
            **kwargs: Additional configuration parameters

        Returns:
            Tuple of (train_atoms_list, test_atoms_list)
        """
        # Your custom initialization logic
        train_atoms = []  # Load or generate training structures
        test_atoms = []  # Load or generate test structures
        return train_atoms, test_atoms

    def train_mlip(self, base_name, job_dict, **kwargs) -> pd.DataFrame:
        """
        Train the machine-learned interatomic potential (MLIP).

        Args:
            base_name: Name used for output directories
            job_dict: Configuration dictionary for this job
            **kwargs: Additional parameters

        Returns:
            DataFrame with training metrics (MAE, RMSE, etc.)
        """
        # Your custom MLIP training logic
        metrics = pd.DataFrame(
            {
                "train_mae": [0.01],
                "test_mae": [0.02],
            }
        )
        return metrics

    def generate_structures(
        self, base_name, job_dict, train_data, **kwargs
    ) -> list[Atoms]:
        """
        Generate candidate structures for labeling based on uncertainty.

        Args:
            base_name: Name used for output directories
            job_dict: Configuration dictionary for this job
            train_data: Current training set (list[Atoms])
            **kwargs: Additional parameters

        Returns:
            List of candidate Atoms objects
        """
        # Your custom structure generation logic
        candidates = []  # Generate structures using MD or other methods
        return candidates

    def high_accuracy_evaluation(
        self, base_name, job_dict, structures, **kwargs
    ) -> list[Atoms]:
        """
        Perform high-accuracy evaluation (e.g., DFT) on selected structures.

        Args:
            base_name: Name used for output directories
            job_dict: Configuration dictionary for this job
            structures: List of Atoms objects to evaluate
            **kwargs: Additional parameters

        Returns:
            List of Atoms objects with energy and force labels in:
            - atoms.info["REF_energy"]
            - atoms.arrays["REF_forces"]
        """
        # Your custom high-accuracy evaluation logic
        evaluated = []  # Run DFT and attach results
        return evaluated
```

### Method Signatures and Responsibilities

- **initialize_training_set**: Called once at the start. Should return initial train/test splits. Results are written to `results/<base_name>/initialization/`.

- **train_mlip**: Called once per AL loop. Should train your MLIP on the current training set and return a DataFrame with performance metrics.

- **generate_structures**: Called once per AL loop. Should use MD, Monte Carlo, or other methods to generate high-uncertainty candidates from the current committee.

- **high_accuracy_evaluation**: Called once per AL loop. Should evaluate structures with DFT (or equivalent high-accuracy method) and attach energies and forces to the Atoms objects.

## Extra Datasets

The initialization configuration can include external datasets via an `extra_datasets` parameter. These structures are seeded into the GlobalDatabase before initialization runs. This is useful for incorporating reference data (e.g., from literature or previous computations) without regenerating isolated atoms.

Example configuration:

```yaml
initialization:
  name: "initialization"
  max_time: "2H"
  test_to_train_ratio: 0.1
  extra_datasets:
    - "path/to/external_structures.xyz"
    - "path/to/another_dataset.xyz"
  creation_kwargs:
    elements: ["H", "O"]
    # ... other options ...
```

Structures in extra datasets should have:
- `atoms.info["REF_energy"]` (float) — DFT energy
- `atoms.arrays["REF_forces"]` (array, shape N×3) — DFT forces
- `atoms.info["config_type"]` (str) — origin label (e.g. `"IsolatedAtom"`, `"external_data"`)

## MACE Committee Predictions in the GlobalDatabase

After each AL loop's MACE committee training finishes on the remote GPU node, ALomancy evaluates every committee model on the training and test sets **before returning from the remote job** and saves the per-structure predictions to `train_pred.xyz` / `test_pred.xyz` inside each fit directory. These files are synced back to your local machine by ExPyRe alongside the model files.

`store_mlip_predictions` then reads those files locally and stores the predicted energies and forces in the GlobalDatabase — no model loading or local GPU required. Parity plots (`plot_dft_vs_model`) read from the DB first, then fall back to the eval xyz files, and never run local inference.

> **Resuming from before v0.4.2**: loops trained without the post-training eval step will not have `train_pred.xyz` / `test_pred.xyz`. The sentinel `mace_predictions.done` is written and those loops are skipped gracefully; parity plots show "No predictions available" for them. Only future loops (trained with the updated remote package) will have predictions stored.

The stored metadata keys follow the pattern:
```
mace_energy_loop_{N}_fit_{i}   # predicted energy (eV, raw)
mace_forces_loop_{N}_fit_{i}   # predicted forces ([[fx,fy,fz], ...], eV/Å)
```

You can retrieve predictions programmatically:

```python
from alomancy.database.global_database import GlobalDatabase

db = GlobalDatabase("results/global_database")

# Returns {"train": (e_dft, e_pred, f_dft, f_pred), "test": (...)}
# where each element is a numpy array; e values are per-atom (eV/atom)
preds = db.get_mace_predictions(loop_idx=0, fit_idx=0)

e_dft, e_pred, f_dft, f_pred = preds["train"]
print(f"Train energy MAE: {abs(e_dft - e_pred).mean():.4f} eV/atom")
```

Predictions are guarded by a `results/al_loop_{N}/mace_predictions.done` sentinel so they are not recomputed on restart. If you need to regenerate predictions (e.g. after installing a different model), delete that sentinel file.
