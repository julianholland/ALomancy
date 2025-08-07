# from expyre.func import ExPyRe
from alomancy.core.standard_active_learning import ActiveLearningStandardMACE
from pathlib import Path



al_workflow=ActiveLearningStandardMACE(
    initial_train_file_path=Path('../results/al_loop_0/train_set.xyz'),
    initial_test_file_path=Path('../results/al_loop_0/test_set.xyz'),
    number_of_al_loops=5,
    verbose=1,
    start_loop=0,
)

al_workflow.run()