from ase.io import read, write
from wfl.autoparallelize.remoteinfo import RemoteInfo
from expyre.resources import Resources
from mace_wfl import create_mace_committee
# from md_wfl import select_md_structures
# from generate_structures import get_structures_for_dft
# from dft_wfl import perform_dft_calculations, add_dft_to_training_data

from pathlib import Path
import shutil


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

    print(f"HPC: {hpc}")
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


def active_learn_mace(number_of_al_loops=5):
    al_loop = 0
    train_xyzs = read(Path("mace_general/c_all_33_2025_07_11_ftrim_100_train_set.xyz"), ":")[:100]
    test_xyzs = read(Path("mace_general/c_all_33_2025_07_11_ftrim_100_test_set.xyz"), ":")[:20]

    for al_loop in range(number_of_al_loops):
        # 0. set up
        base_name = f"al_loop_{al_loop}"
        Path.mkdir(Path(f"results/{base_name}"), exist_ok=True, parents=True)
        train_file = str(Path(f"results/{base_name}/train_set.xyz"))
        test_file = str(Path(f"results/{base_name}/test_set.xyz"))
        write(
            train_file,
            train_xyzs,
            format="extxyz",
        )
        write(
            test_file,
            test_xyzs,
            format="extxyz",
        )

        # 1. create MACE committee
        create_mace_committee(
            job_name="mace_committee",
            base_file_name=base_name,
            seed=803,
            remote_info=get_remote_info(f"mace_com_{base_name}", hpc="fhi-raccoon"),
            size_of_committee=2,
        )

        # # 2. select structures from train set to perform MD on
        # initial_atoms = select_md_structures(
        #     number_of_mds=5,
        #     name=f"md_{base_name}",
        #     chem_formula="C10Na",
        #     test_set_xyz="mace_general/ac_all_33_2025_07_11_ftrim_100_test_set.xyz",
        # )

        # # 3. select high standard deviation structures from MD
        # dft_strucutres = get_structures_for_dft(
        #     md_name=f"md_{base_name}",
        #     initial_atoms=initial_atoms,
        #     remote_info=get_remote_info(f"md_{base_name}", hpc="fhi-raccoon"),
        #     read_md=False,
        #     number_of_structures=50,
        #     verbose=0,
        #     save_xyz=True,
        # )

        # # 4. perform DFT calculations on selected structures
        # dft_structures=perform_dft_calculations(
        #     name=f"dft_{base_name}",
        #     atoms_list=dft_strucutres,
        #     remote_info=get_remote_info(f"dft_{base_name}", hpc="raven"),
        #     hpc="raven",
        # )

        # # 5. add DFT results to training data
        # add_dft_to_training_data(dft_structures)

        al_loop += 1
        print(f"Active learning loop {al_loop} completed.")


if __name__ == "__main__":
    active_learn_mace(number_of_al_loops=3)
