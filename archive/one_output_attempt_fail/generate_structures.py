from ase.io import read, write
from ase.io.trajectory import Trajectory
import numpy as np
from wfl.autoparallelize import AutoparaInfo
from wfl.autoparallelize.base import autoparallelize, autoparallelize_docstring
from pathlib import Path

from wfl.configset import OutputSpec
from al_wfl import get_remote_info
from md_wfl import primary_run, get_forces_for_all_maces, std_deviation_of_forces

# calc=MACECalculator(model_paths=[f'MACE_model/fit_{i}/test_stagetwo_compiled.model' for i in range(2)],device='cpu')

def get_structures_for_dft(
    md_name,
    initial_atoms,
    read_md=False,
    number_of_structures=50,
    verbose=0,
    save_xyz=True,
) -> list:
    """
    Select structures for DFT calculations based on the standard deviation of forces.

    Parameters
    ----------
    md_name : str
        Name of the MD run.
    read_md : bool, optional
        Whether to read the MD file directly, by default False. If False, it will run
    number_of_structures : int, optional
        Number of structures to select, by default 50.

    Returns
    -------
    list
        List of structures selected for DFT calculations.
    """

    traj_files = sorted(Path("md_trajs").glob(f"{md_name}*.xyz"))

    def autopara_md(*args, **kwargs):
        return autoparallelize(primary_run, *args, **kwargs)

    if not (read_md and len(traj_files) > 0):
        primary_run_args = {
            "out_dir": str(Path("md_trajs")),
            "steps": 200,
            "temperature": 1200,
            'device': 'cuda',
            'base_mace': 0,
            'md_name': md_name,
        }
        remote_info=get_remote_info(
            "mace_md",
            "fhi-raccoon",
        )
        remote_info.input_files=['md_wfl.py', 'al_wfl.py', 'MACE_model/fit_0/test_stagetwo.model']

        structure_list=list(autopara_md(
            inputs=initial_atoms,
            outputs=OutputSpec(f'{md_name}.xyz'),
            autopara_info=AutoparaInfo(
                remote_info=remote_info,
                ),
            **primary_run_args,
        ))
    
    
    
    # traj_list = [Trajectory(traj_file, "r") for traj_file in traj_files]
    structure_list = read(f'{md_name}.xyz', ":")
    print(structure_list)

    if verbose > 0:
        print(len(structure_list), "trajectories found in", traj_files)

    structure_forces_dict = get_forces_for_all_maces(
        structure_list, base_mace=0, fits_to_use=[1, 2, 3, 4]
    )
    std_dev_df = std_deviation_of_forces(structure_forces_dict)
    index_list = list(std_dev_df[:number_of_structures].index)

    if verbose > 0:
        print(std_dev_df[:number_of_structures])
        print(f"total mean: {std_dev_df['mean_std_dev'].mean()}")

    if save_xyz:
        write(
            "high_sd_structures.xyz",
            [structure_list[i] for i in index_list],
            format="extxyz",
        )

    return [structure_list[i] for i in index_list]


if __name__ == "__main__":
    al_loop=0
    c10Na1_structures = [
        s
        for s in read("mace_general/ac_all_33_2025_07_11_ftrim_100_test_set.xyz", ":")
        if s.get_chemical_formula() == "C10Na"
    ]
    initial_atoms = [
        c10Na1_structures[x]
        for x in np.random.choice(
            np.array(range(len(c10Na1_structures))), 5, replace=False
        )
    ]
    for i, atoms in enumerate(initial_atoms):
        atoms.info['job_id'] = i
        atoms.info['config_type'] = f'md_al_loop_{al_loop}_{i}'

    get_structures_for_dft(
        "test", initial_atoms=initial_atoms, read_md=False, number_of_structures=20
    )
    # dft_structure_list=get_structures_for_dft('md', read_md=True, number_of_structures=10, verbose=1)
