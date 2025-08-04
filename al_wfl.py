from ase.io import read, write
from ase import Atoms
from wfl.autoparallelize.remoteinfo import RemoteInfo
from expyre.resources import Resources
from mace_wfl import create_mace_committee
from md_wfl import select_md_structures
from generate_structures import get_structures_for_dft
from dft_wfl import perform_qe_calculations_per_cell  #
from test_train_manager import add_new_training_data

from pathlib import Path


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
        "partitions": ["gpusmall"],
    },
    "nersc": {
        "pre_cmds": ["source /global/homes/j/joh1e19/.venvs/wfl/bin/activate"],
        "partitions": ["regular"],
    },
}

JOB_DICT = {
    "mace_committee": {
        "name": "mace_committee",
        "max_time": "5H",
    },
    "md_run": {"name": "md_run", "max_time": "10H"},
    "dft_run": {"name": "dft_run", "max_time": "30m"},
}


def get_remote_info(hpc: str, job: str, input_files: list[str] = []):
    """
    Returns a RemoteInfo object for running MACE fits on a GPU cluster.
    """

    print(f"HPC: {hpc}, Job: {JOB_DICT[job]['name']}")
    return RemoteInfo(
        sys_name=hpc,
        job_name=JOB_DICT[job]["name"],
        num_inputs_per_queued_job=1,
        timeout=36000 * 3,
        input_files=input_files,
        pre_cmds=RI_DICT[hpc]["pre_cmds"],
        resources=Resources(
            max_time=JOB_DICT[job]["max_time"],
            num_nodes=1,
            partitions=RI_DICT[hpc]["partitions"],
        ),
    )


def active_learn_mace(
    initial_train_file_path, initial_test_file_path, number_of_al_loops: int = 5, verbose: int = 0, start_loop: int = 0, committee_size: int = 5, md_runs: int = 5):
    al_loop = start_loop

    train_xyzs = [
        atoms
        for atoms in read(Path(initial_train_file_path), ":")
        if isinstance(atoms, Atoms)
    ]

    test_xyzs = [
        atoms
        for atoms in read(Path(initial_test_file_path), ":")
        if isinstance(atoms, Atoms)
    ]

    assert len(train_xyzs) > 1, "more than one training structure required."
    assert len(test_xyzs) > 1, "more than one test structure required."

    for al_loop in range(number_of_al_loops):
        # 0. set up
        base_name = f"al_loop_{al_loop}"
        loop_dir = Path(f"results/{base_name}")
        Path.mkdir(loop_dir, exist_ok=True, parents=True)

        train_file = str(Path(loop_dir, "train_set.xyz"))
        test_file = str(Path(loop_dir, "test_set.xyz"))
        # if isinstance(train_xyzs[0], Atoms):
        write(
            train_file,
            train_xyzs,  # pyright: ignore[reportArgumentType]
            format="extxyz",
        )

        write(
            test_file,
            test_xyzs,
            format="extxyz",
        )

        # 1. create MACE committee
        create_mace_committee(
            base_name=base_name,
            job_name=JOB_DICT["mace_committee"]["name"],
            seed=803,
            remote_info=get_remote_info(
                hpc="fhi-raccoon", job="mace_committee", input_files=[]
            ),
            size_of_committee=committee_size,
            epochs=60,
        )

        # 2. select structures from train set to perform MD on
        md_input_structures = select_md_structures(
            base_name=base_name,
            job_name=JOB_DICT["md_run"]["name"],
            number_of_mds=md_runs,
            chem_formula_list=[],
            atom_number_range=(9,21),
            enforce_chemical_diversity=True,
            train_xyzs=train_xyzs, # type: ignore
            verbose=verbose,
        ) 

        Path.mkdir(Path(loop_dir, "MD"), exist_ok=True, parents=True)
        write(
            Path(loop_dir, "MD", "md_input_structures.xyz"),
            md_input_structures,
            format="extxyz",
        )

        # 3. select high standard deviation structures from MD
        dft_input_structures = get_structures_for_dft(
            base_name=base_name,
            job_dict=JOB_DICT,
            initial_atoms=md_input_structures,
            remote_info=get_remote_info(
                # hpc="fhi-raccoon", job="md_run", input_files=[]
                hpc="raven_gpu", job="md_run", input_files=[]
            ),
            number_of_structures=50,
            verbose=verbose,
            temperature=1200.0,
            steps=1000,
            timestep_fs=0.5,
            base_mace=str(
                Path(
                    loop_dir,
                    "MACE",
                    "fit_0",
                    f"{JOB_DICT['mace_committee']['name']}_stagetwo.model",
                )
            ),
        )
        Path.mkdir(Path(loop_dir, "DFT"), exist_ok=True, parents=True)
        write(
            Path(loop_dir, "DFT", "dft_input_structures.xyz"),
            dft_input_structures,
            format="extxyz",
        )

        # 4. perform DFT calculations on selected structures
        dft_structures = perform_qe_calculations_per_cell(
            base_name=base_name,
            job_name=JOB_DICT["dft_run"]["name"],
            atoms_list=dft_input_structures,
            remote_info=get_remote_info(hpc="raven", job="dft_run", input_files=[]),
            hpc="raven",
            verbose=verbose
        )
        print(dft_structures)

        # 5. add DFT results to training data
        train_xyzs = list(
            add_new_training_data(
                base_name=base_name, job_dict=JOB_DICT, train_xyzs=train_xyzs # type: ignore
            )
        )  

        al_loop += 1
        print(f"Active learning loop {al_loop} completed. New training set size: {len(train_xyzs)}")


if __name__ == "__main__":
    train_data_dir= Path("mace_general")
    active_learn_mace(
        initial_test_file_path=Path(
            train_data_dir,
            "ac_all_33_2025_07_31_ftrim_10_grpspread_01_test_set.xyz"
        ),
        initial_train_file_path=Path(
            train_data_dir,
            "ac_all_33_2025_07_31_ftrim_10_grpspread_01_train_set.xyz"
        ),
        number_of_al_loops=50,
        verbose=1,
        start_loop=0,
        committee_size=3,
        md_runs=2,
    )
