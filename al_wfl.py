from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize.remoteinfo import RemoteInfo
from expyre.resources import Resources
from mace_wfl import create_mace_committee
from md_wfl import get_structures_for_dft
# from 


RI_DICT = {
    "raven_gpu": {
        "pre_cmds": ["source /u/jholl/venvs/wfl/bin/activate"],
        "partitions": ["gpu"],
    },
    "raven": {
        "pre_cmds": ["source /u/jholl/venvs/wfl/bin/activate"],
        "partitions": ["general"],
    },
    "fhi-raccoon": {
        "pre_cmds": ["source /home/jholl/venvs/mace/bin/activate"],
        "partitions": ["gpubig"],
    },
    "nersc": {
        "pre_cmds": ["source /global/homes/j/joh1e19/.venvs/wfl/bin/activate"],
        "partitions": ["regular"],
    },
}

def get_remote_info(name, hpc="raven_gpu", input_files=[]):
    """
    Returns a RemoteInfo object for running MACE fits on a GPU cluster.
    """

    print(f'HPC: {hpc}')
    return RemoteInfo(
        sys_name=hpc,
        job_name=name,
        num_inputs_per_queued_job=1,
        timeout=36000 * 3,
        input_files=input_files,
        pre_cmds=RI_DICT[hpc]["pre_cmds"],
        resources=Resources(
            max_time="30m",
            num_nodes=1,
            partitions=RI_DICT[hpc]["partitions"],
        ),
    )

# def active_learn_mace(number_of_al_loops=5):
#     for 
#     create_mace_committee(5)
#     get_structures_for_dft()
#     perform_dft_calculations()
#     add_dft_to_training_data()

