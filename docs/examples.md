# Examples

## Basic Usage

```python
from alomancy.core import StandardActiveLearningWorkflow
from pathlib import Path

# a bare minimum settings dict for running the StandardActiveLearningWorkflow module
# this can also be loaded in as a .yaml file (c.f. examples)

your_hpc_dict = {
  'hpc_name' : 'your_ssh_hpc_name',
  'gpu': True, # the presence of a gpu in your hpc
  'pre_cmds': ["command to enable correct python environment on your hpc"], # e.g. `conda activate alomancy` or `source ~/.venvs/alomancy/bin/activate`
  'partitions': ["hpc_partition_name"],
  'node_info': {
    'ranks_per_system': 72, # number of mpi parallelization to use per system (can be more or less than number per core )
    'ranks_per_node': 72, # number of mpi parallelizations per node
    'threads_per_rank': 1, # number of omp threads per mpi paralellization
    'max_mem_per_node': "60GB", # memore of your hpcs node
  },
  'high_accuracy_executable_path': "/path/to/your/quantum/espresso/bin/pw.x",
  'pp_path': "/path/to/your/pseudo/potentials/directory",
  'pseudo_dict': {
    'C': "name_of_carbon_pp.UPF",
    'Na': "name_of_sodium_pp.UPF", # change to elements used in your training set
  },
}

jobs_dict= {
  'mlip_committee': {
    'name': 'mace_committee',
    'size_of_committee': 3,
    'max_time': '5H',
    'mace_fit_kwargs': {
      'E0s': {
        6: -241.94038776317848,
        11: -1296.5877903540002,
      }
      'atomic_numbers': [6, 11],
      'energy_key': "REF_energy",
      'forces_key': "REF_forces",
    },
    'hpc' : your_hpc_dict
  },
  'structure_generation': {
    'name': 'md_1200_generation',
    'desired_number_of_structures': 50,
    'max_time': "10H",
    'structure_selection_kwargs':,
      'max_number_of_concurrent_jobs': 5,
      'chem_formula_list': None,
      'atom_number_range': [0, 21],
      'enforce_chemical_diversity': True,
    'run_md_kwargs':,
      'steps': 20000,
      'temperature': 1200,
      'timestep_fs': 0.5,
      'friction': 0.002,
    'hpc': your_hpc_dict,
  }

  },
  'high_accuracy_evaluation': {
    'name': 'qe_dft',
    'max_time': "30m",
    'qe_input_kwargs':,
      'system':,
        'input_dft': "pbe",
    'hpc': your_hpc_dict,
} # the hpc can/should be changed to whatever is the most appropriate hpc for each step of the workflow

# Initialize workflow
workflow = StandardActiveLearningWorkflow(
    initial_train_file_path="train_set.xyz",
    initial_test_file_path="test_set.xyz",
    jobs_dict=jobs_dict,
    number_of_al_loops=5,
    verbose=1
)

# Run the active learning workflow
workflow.run()
```

## Custom Workflows

Extend the base class for specialized workflows:

```python
from alomancy.core import BaseActiveLearningWorkflow
from ase import Atoms
import pandas as pd

class CustomActiveLearningWorkflow(BaseActiveLearningWorkflow):

    def train_mlip(self, base_name: str, mlip_committee_job_dict: dict, **kwargs) -> pd.Dataframe:
        """Custom MLIP training implementation"""
        # Your custom training logic here
        return "path/to/trained/model.pt"

    def generate_structures(self, base_name: str, job_dict: dict,
                          train_data: list[Atoms], **kwargs) -> list[Atoms]:
        """Custom structure generation"""
        # Your structure generation logic here
        return generated_structures

    def high_accuracy_evaluation(self, base_name: str,
                               high_accuracy_eval_job_dict: dict,
                               structures: list[Atoms], **kwargs) -> list[Atoms]:
        """Custom high-accuracy evaluation"""
        # Your high-accuracy calculation logic here
        return evaluated_structures
```

Or just change a single function by importing the `StandardActiveLearningWorkflow` module

```python
from alomancy.core import StandardActiveLearningWorkflow

class SingleChangeActiveLearningWorkflow(StandardActiveLearningWorkflow):
  def generate_structures(self, base_name: str, job_dict: dict,
                          train_data: list[Atoms], **kwargs) -> list(Atoms):
      """new structure generation function while retaining StandardActiveLearningWorkflow's mlip training and high accuracy evaluation functionality"""
      # New structure generation logic
      return generated_structures

```

See the `examples/` directory in the repository for complete working examples.
