# Demonstrates EZGA (genetic-algorithm) structure generation as an
# alternative to the MD-based candidate generation shown in
# examples/basic_use/basic_use.py. See input_files/standard_config.yaml's
# structure_generation section for the ezga_kwargs this exercises.
from pathlib import Path

from yaml import safe_load

from alomancy.core.committee_uncertainty_workflow import build_workflow

# load jobs_dict from a YAML files
with open("input_files/standard_config.yaml") as f:
    config = safe_load(f)

with open("input_files/hpc_config.yaml") as f:
    hpc_config = safe_load(f)


# assign hpcs to the jobs_dict (a self-contained local hpc_config.yaml here
# rather than load_dictionaries()'s ~/.alomancy/hpc_config.yaml string
# resolution, so this example runs from a fresh checkout without requiring
# `alomancy add-hpc` to have been run first)
config["initialization"]["hpc"] = hpc_config[config["initialization"]["hpc"]]
config["training"]["hpc"] = hpc_config[config["training"]["hpc"]]
config["structure_generation"]["hpc"] = hpc_config[
    config["structure_generation"]["hpc"]
]
config["high_accuracy_evaluation"]["hpc"] = hpc_config[
    config["high_accuracy_evaluation"]["hpc"]
]

print("Using config:")
print(config)
al_workflow = build_workflow(
    jobs_dict=config,
    initial_train_file_path=Path("input_files/C_Na_amorphous_5255_train.xyz"),
    initial_test_file_path=Path("input_files/C_Na_amorphous_583_test.xyz"),
    number_of_al_loops=25,
    verbose=1,
    start_loop=0,
)

al_workflow.run()
