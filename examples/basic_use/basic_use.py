# from expyre.func import ExPyRe
from pathlib import Path

from alomancy.core.standard_active_learning import ActiveLearningStandardMACE

al_workflow = ActiveLearningStandardMACE(
    initial_train_file_path=Path(
        "structure_data/ac_all_33_2025_07_11_ftrim_100_train_set.xyz"
    ),
    initial_test_file_path=Path(
        "structure_data/ac_all_33_2025_07_11_ftrim_100_test_set.xyz"
    ),
    number_of_al_loops=25,
    verbose=1,
    start_loop=0,
)

al_workflow.run()
