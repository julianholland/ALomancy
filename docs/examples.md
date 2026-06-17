# Examples

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
    verbose=1,                        # 0=silent, 1=INFO progress, 2=DEBUG
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
  max_time: "30m"
  max_batch_size: 20
  hpc: 'my_cpu_hpc'
```

### Configuration Key Descriptions

- **initialization**: Generates initial training and test sets. Supports Materials Project structures, dimers, trimers, amorphous structures, and stretched/compressed MP structures. The `test_to_train_ratio` determines the split between test and training data.

- **mlip_committee**: Trains an ensemble (committee) of MACE interatomic potentials. The `size_of_committee` parameter determines how many committee members are trained in parallel.

- **structure_generation**: Uses MD to generate candidate structures for labeling. Uncertainty is measured as force standard deviation across the committee.

- **high_accuracy_evaluation**: Performs high-accuracy DFT evaluation (via Quantum Espresso) on selected structures. The `max_batch_size` controls how many structures are included in each QE batch submission.

## Custom Workflows

You can create custom active learning workflows by extending `BaseActiveLearningWorkflow`. You must implement four abstract methods:

```python
from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
import pandas as pd
from ase.atoms import Atoms

class MyCustomWorkflow(BaseActiveLearningWorkflow):
    def initialize_training_set(self, base_name, **kwargs) -> tuple[list[Atoms], list[Atoms]]:
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
        test_atoms = []   # Load or generate test structures
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
        metrics = pd.DataFrame({
            'train_mae': [0.01],
            'test_mae': [0.02],
        })
        return metrics

    def generate_structures(self, base_name, job_dict, train_data, **kwargs) -> list[Atoms]:
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

    def high_accuracy_evaluation(self, base_name, job_dict, structures, **kwargs) -> list[Atoms]:
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
