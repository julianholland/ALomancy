# Examples

## Basic Usage

Here's a simple example of running an active learning workflow:

```python
from alomancy.core import StandardActiveLearningWorkflow

workflow = StandardActiveLearningWorkflow(
    initial_train_file_path="data/train.xyz",
    initial_test_file_path="data/test.xyz",
    config_file_path="config.yaml",
    number_of_al_loops=3
)

workflow.run()
```

## Custom Workflows

You can extend the base workflow for custom implementations:

```python
from alomancy.core import BaseActiveLearningWorkflow

class MyCustomWorkflow(BaseActiveLearningWorkflow):
    def train_mlip(self, base_name, mlip_committee_job_dict, **kwargs):
        # Custom training logic
        pass
```

See the `examples/` directory in the repository for complete working examples.
