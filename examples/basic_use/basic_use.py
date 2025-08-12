# from expyre.func import ExPyRe
from pathlib import Path

from alomancy.core.standard_active_learning import ActiveLearningStandardMACE

al_workflow = ActiveLearningStandardMACE(
    initial_train_file_path=Path("input_files/C_Na_amorphous_5255_train.xyz"),
    initial_test_file_path=Path("input_files/C_Na_amorphous_583_test.xyz"),
    config_file_path=Path("input_files/standard_config.yaml"),
    number_of_al_loops=25,
    verbose=1,
    start_loop=0,
)

al_workflow.run()
