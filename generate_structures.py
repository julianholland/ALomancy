from ase.io import read, write
from wfl.autoparallelize import AutoparaInfo
from wfl.autoparallelize.base import autoparallelize
from pathlib import Path

from wfl.configset import OutputSpec
from md_wfl import (
    primary_run,
    get_forces_for_all_maces,
    std_deviation_of_forces,
    select_md_structures,
)

# calc=MACECalculator(model_paths=[f'MACE_model/fit_{i}/test_stagetwo_compiled.model' for i in range(2)],device='cpu')


def get_structures_for_dft(
    md_name,
    initial_atoms,
    remote_info=None,
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
    md_dir = Path("md_trajs")

    traj_files = sorted(md_dir.glob(f"{md_name}*.xyz"))

    def autopara_md(*args, **kwargs):
        return autoparallelize(primary_run, *args, **kwargs)

    if not (read_md and len(traj_files) > 0):
        primary_run_args = {
            "out_dir": str(md_dir),
            "steps": 200,
            "temperature": 1200,
            "device": "cuda",
            "base_mace": 0,
            "md_name": md_name,
        }

        remote_info.input_files = [
            "md_wfl.py",
            "al_wfl.py",
            "MACE_model/fit_0/test_stagetwo.model",
        ]
        remote_info.output_files = [str(Path(md_dir, f"{md_name}_*.xyz"))]
        structure_list = list(
            autopara_md(
                inputs=initial_atoms,
                outputs=OutputSpec(),
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                ),
                **primary_run_args,
            )
        )

    # traj_list = [Trajectory(traj_file, "r") for traj_file in traj_files]
    structure_list = []
    for i in range(len(initial_atoms)):
        structures = read(Path(md_dir, f"{md_name}_{i}.xyz"), ":")
        structure_list.extend(structures)

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
    al_loop = 0
    name = "test"
    get_structures_for_dft(
        name,
        initial_atoms=select_md_structures(name=name),
        remote_info=None,
        read_md=True,
        number_of_structures=10,
    )
    # dft_structure_list=get_structures_for_dft('md', read_md=True, number_of_structures=10, verbose=1)
