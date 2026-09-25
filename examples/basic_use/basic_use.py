# from expyre.func import ExPyRe
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
al_workflow = build_workflow(jobs_dict=config)

al_workflow.run()
