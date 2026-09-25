"""Minimal single-element (Carbon) run using the CommitteeUncertaintyWorkflow
skeleton, dispatched via build_workflow() + the shared module registry --
see src/alomancy/core/committee_uncertainty_workflow.py.

build_workflow() reads config["general"]["al_workflow"] to pick the
workflow class, and that class in turn resolves its trainer/structure-
generator/DFT-evaluator/initialiser from config via registry.resolve(...)
rather than Python subclassing -- there is no other workflow class or
config schema in this codebase (examples/basic_use/ and examples/ezga_use/
use the same build_workflow() entry point, just with more elaborate
configs).

initial_train_file_path/initial_test_file_path below point at files that
don't exist yet -- that's expected for a first run: the skeleton falls
through to the DB-driven path (initialization's own structure-type
settings, e.g. dimer_kwargs/amorphous_kwargs/mp_kwargs, plus
general.elements) and generates + DFT-evaluates a bootstrap dataset
itself. Point them at existing xyz files instead to skip that and start
straight from a pre-built training set.
"""

from pathlib import Path

from alomancy.configs.config_dictionaries import load_dictionaries
from alomancy.core.committee_uncertainty_workflow import build_workflow

config = load_dictionaries(Path("config.yaml"))

al_workflow = build_workflow(
    jobs_dict=config,
    initial_train_file_path="input_files/carbon_train.xyz",
    initial_test_file_path="input_files/carbon_test.xyz",
    number_of_al_loops=5,
    verbose=1,
    start_loop=0,
)

al_workflow.run()
