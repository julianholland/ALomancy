# from expyre.func import ExPyRe
from pathlib import Path

from yaml import safe_load

from alomancy.core.standard_active_learning import ActiveLearningStandardMACE

# load jobs_dict from a YAML files
with open("input_files/standard_config.yaml") as f:
    config = safe_load(f)

with open("input_files/hpc_config.yaml") as f:
    hpc_config = safe_load(f)


# assign hpcs to the jobs_dict
config["mlip_committee"]["hpc"] = hpc_config[config["mlip_committee"]["hpc"]]
config["structure_generation"]["hpc"] = hpc_config[
    config["structure_generation"]["hpc"]
]
config["high_accuracy_evaluation"]["hpc"] = hpc_config[
    config["high_accuracy_evaluation"]["hpc"]
]

print("Using config:")
print(config)
al_workflow = ActiveLearningStandardMACE(
    initial_train_file_path=Path("input_files/C_Na_amorphous_5255_train.xyz"),
    initial_test_file_path=Path("input_files/C_Na_amorphous_583_test.xyz"),
    jobs_dict=config,
    number_of_al_loops=25,
    verbose=1,
    start_loop=0,
)

al_workflow.run()
