# Quick Start

## Basic Active Learning Workflow

```python
from alomancy.configs.config_dictionaries import load_dictionaries
from alomancy.core.standard_active_learning import ActiveLearningStandardMACE

# Load configuration from YAML file
jobs_dict = load_dictionaries("standard_config.yaml")

# Initialize the active learning workflow
workflow = ActiveLearningStandardMACE(
    initial_train_file_path="results/initialization/train_set.xyz",
    initial_test_file_path="results/initialization/test_set.xyz",
    jobs_dict=jobs_dict,
    number_of_al_loops=5,
    verbose=1,                        # 0=silent, 1=INFO, 2=DEBUG
    log_file="results/alomancy.log",  # debug logs always written here
    db_path="results/global_database",
)

# Run the active learning workflow
workflow.run()
```

## Configuration

Create a `standard_config.yaml` file with the required top-level keys:

```yaml
initialization:
  name: "initialization"
  max_time: "4:00:00"
  hpc:
    hpc_name: "local"
    partitions: []
    pre_cmds: []

mlip_committee:
  name: "mace_training"
  max_time: "12:00:00"
  hpc:
    hpc_name: "local"
    partitions: []
    pre_cmds: []

structure_generation:
  name: "md_generation"
  max_time: "8:00:00"
  hpc:
    hpc_name: "local"
    partitions: []
    pre_cmds: []

high_accuracy_evaluation:
  name: "dft_evaluation"
  max_time: "24:00:00"
  hpc:
    hpc_name: "local"
    partitions: []
    pre_cmds: []
```

See the [examples](examples.md) for more detailed configurations.

## Initialization Behavior

The workflow handles initialization in two ways:

- **Fast path**: If `initial_train_file_path` and `initial_test_file_path` already exist, the workflow loads them directly and begins the AL loops.
- **Full path**: If either file is missing, the workflow automatically:
  1. Checks the global database for existing structures
  2. Generates missing structures via ASE MD (dimers, trimers, amorphous, Materials Project)
  3. Evaluates them with DFT (Quantum Espresso)
  4. Builds train/test splits from the database

## Verbosity Levels

- `verbose=0`: Silent mode (no progress output)
- `verbose=1`: INFO level (progress and high-level updates)
- `verbose=2`: DEBUG level (detailed progress and intermediate steps)

All debug-level logs are always written to `log_file` regardless of `verbose` setting.
