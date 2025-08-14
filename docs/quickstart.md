# Quick Start

## Basic Active Learning Workflow

```python
from alomancy.core import StandardActiveLearningWorkflow



# Initialize workflow
workflow = StandardActiveLearningWorkflow(
    initial_train_file_path="train_set.xyz",
    initial_test_file_path="test_set.xyz",
    config_file_path="config.yaml",
    number_of_al_loops=5,
    verbose=1
)

# Run the active learning workflow
workflow.run()
```

## Configuration

Create a `config.yaml` file:

```yaml
mlip_committee:
  name: "mace_training"
  size_of_committee: 4

structure_generation:
  name: "md_generation"
  number_of_concurrent_jobs: 8
  desired_number_of_structures: 100

high_accuracy_evaluation:
  name: "dft_evaluation"
  max_time: "48:00:00"
```

See the [examples](examples.md) for more detailed configurations.
