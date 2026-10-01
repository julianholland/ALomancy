# Quick Start

## Basic Active Learning Workflow

```python
from alomancy import ALomancy

# The config picks everything: the AL skeleton (general.al_workflow) and
# every module (training.trainer, structure_generation.generator,
# high_accuracy_evaluation.evaluator). Every run setting (start_from,
# num_of_al_loops, verbose, ...) lives under the YAML's `general:` section.
ALomancy("standard_config.yaml").run()
```

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["C", "O"]
  # Optional warm start -- omit for a cold start. See starting_a_run.md.
  # start_from:
  #   train_xyz: "my_train.xyz"
  #   test_xyz: "my_test.xyz"
  num_of_al_loops: 5
  verbose: 1  # 0=silent, 1=INFO, 2=DEBUG
  log_file: "results/alomancy.log"  # debug logs always written here
  db_path: "results/global_database"
  dataset_kwargs:
    target_config_types: ["IsolatedAtom"]
    test_ratio: 0.1
  committee_uncertainty_kwargs:
    num_of_models_in_committee: 5
```

## HPC Setup

Before writing a config file, run the interactive wizard to register your HPC system:

```bash
alomancy add-hpc
```

The wizard reads your `~/.ssh/config` and lets you pick a host alias from a numbered list.
It writes two files:
- `~/.expyre/config.json` — ExPyRe scheduler entry (Slurm headers, partitions, scratch dir)
- `~/.alomancy/hpc_config.yaml` — ALomancy profile (venv activation, DFT paths, node info)

Once registered, refer to profiles by name in your run YAML (see below).

## Configuration

Create a `standard_config.yaml` file with the required top-level keys.  
The `hpc:` value is the profile name written by `alomancy add-hpc`:

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["C", "O"]   # atomic symbols, not atomic numbers
  dataset_kwargs:
    target_config_types: ["IsolatedAtom"]
    test_ratio: 0.1
  committee_uncertainty_kwargs:
    num_of_models_in_committee: 5

initialization:
  name: "initialization"
  max_time: "4:00:00"
  hpc: "raven_cpu"       # profile name from ~/.alomancy/hpc_config.yaml

training:
  name: "mace_training"
  trainer: "mace"        # selects the registered mlip_trainer backend
  max_time: "12:00:00"
  hpc: "raven_gpu"

structure_generation:
  name: "md_generation"
  generator: "md"        # "md" (default) or "ezga"
  max_time: "8:00:00"
  hpc: "raven_gpu"

high_accuracy_evaluation:
  name: "high_accuracy_evaluation"
  evaluator: "qe"        # "qe" (default) or "vasp"
  max_time: "24:00:00"
  hpc: "raven_cpu"
```

`load_dictionaries` resolves each string `hpc:` value from `~/.alomancy/hpc_config.yaml`
automatically — no manual merging needed in your run script.

See the [examples](examples.md) for more detailed configurations.

## Initialization Behavior

`general.start_from` decides where the first structures come from: a
train/test pair of xyz files, a single xyz file, a former ALomancy
database, or nothing (a cold start). Every mode then:

1. Imports the start data into the global database
2. Generates only the initialization structures still missing (isolated atoms, dimers, trimers, amorphous, Materials Project)
3. Evaluates them with DFT
4. Builds train/test splits from the database

See [Starting a run](starting_a_run.md) for each mode, label handling for
foreign xyz files, and the split rules.

## Verbosity Levels

- `verbose=0`: Silent mode (no progress output)
- `verbose=1`: INFO level (progress and high-level updates)
- `verbose=2`: DEBUG level (detailed progress and intermediate steps)

All debug-level logs are always written to `log_file` regardless of `verbose` setting.
