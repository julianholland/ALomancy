from typing import Dict, Any
from yaml import safe_load
from pathlib import Path

def load_dictionaries(config_path: Path) -> Dict[str, Any]:
    """
    Set the information for the HPCs and jobs used in the active learning workflow.

    three jobs are required
    mlip_committee: advised to use a GPU based HPC
    structure_generation: advised to use a GPU based HPC
    high_accuracy_evaluation: advised to use a CPU based HPC

    for each job, the following information is required:
    - name: the name of the job, used to identify the job in the workflow
    - max_time: the maximum time allowed for the job to run
    - hpc: a dictionary containing information about the HPCs available for the job
        - pre_cmds: commands to run before starting the job, e.g. activating a virtual environment
        - partitions: the partition used for the job on the HPC

        
    for the mlip_committee job, the following information is required:
    - size_of_committee: the number of models in the committee
    - epochs: the number of epochs to train the models for

    for the structure_generation job, the following information is required:
    - number_of_concurrent_jobs: the number of concurrent jobs to run
    
    for the high_accuracy_evaluation job, the following information is required in the hpc dictionary:
    - node_info: a dictionary containing information about the nodes available for the job
        - ranks_per_system: the number of ranks per system
        - ranks_per_node: the number of ranks per node
        - threads_per_rank: the number of threads per rank
        - max_mem_per_node: the maximum memory per node
    - pwx_path: the path to the Quantum Espresso pw.x executable
    - pp_path: the path to the pseudopotentials directory
    - pseudo_dict: a dictionary containing the pseudopotentials used for each element
    Returns
    -------
    dict
        A dictionary containing the HPC and job information.
    """

    JOB_DICT = safe_load(open(config_path, "r"))

    return JOB_DICT

if __name__ == "__main__":
    config_path = 'standard_config.yaml'
    print(load_dictionaries(config_path))