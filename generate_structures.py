from re import S
from ase.io import read, write
from wfl.autoparallelize import AutoparaInfo, RemoteInfo
from wfl.autoparallelize.base import autoparallelize
from pathlib import Path

from wfl.configset import OutputSpec, ConfigSet
from md_wfl import (
    primary_run,
    get_forces_for_all_maces,
    std_deviation_of_forces,
)
import pandas as pd
from ase import Atoms
# calc=MACECalculator(model_paths=[f'MACE_model/fit_{i}/test_stagetwo_compiled.model' for i in range(2)],device='cpu')


def get_structures_for_dft(
    base_name: str,
    job_dict: dict,
    initial_atoms: list[Atoms],
    base_mace: str,
    remote_info: RemoteInfo,
    number_of_structures: int = 50,
    verbose: int = 0,
    save_xyz: bool = True,
    read_sd_csv: bool = True,
    temperature: float = 300.0,
    steps: int = 100,
    timestep_fs: float = 0.5,
) -> list[Atoms]:
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

    assert number_of_structures < steps * len(initial_atoms), (
        "Number of structures must be less than the number of steps times the number of initial atoms"
    )

    md_dir = Path("results", base_name, "MD/md_trajs")
    md_dir.mkdir(exist_ok=True, parents=True)
    traj_files = sorted(md_dir.glob(f"{job_dict['md_run']['name']}*.xyz"))

    def autopara_md(*args, **kwargs):
        return autoparallelize(primary_run, *args, **kwargs)

    if not (len(traj_files) > 0):
        primary_run_args = {
            "out_dir": str(md_dir),
            "device": "cuda",
            "base_mace": base_mace,
            "md_name": job_dict["md_run"]["name"],
            "steps": steps,
            "temperature": temperature,
            "number_of_structures": number_of_structures,
            "timestep_fs": timestep_fs,
            "verbose": verbose,
        }

        remote_info.input_files = ["md_wfl.py", "al_wfl.py", base_mace]
        remote_info.output_files = [
            str(Path(md_dir, f"{job_dict['md_run']['name']}_*.xyz"))
        ]
        remote_info.check_interval = 10
        
        autopara_md(
            inputs=initial_atoms,
            outputs=OutputSpec(),
            autopara_info=AutoparaInfo(
                remote_info=remote_info,
            ),
            **primary_run_args,
        )  # type: ignore

    structure_list = []
    for i in range(len(initial_atoms)):
        structures = read(Path(md_dir, f"{job_dict['md_run']['name']}_{i}.xyz"), ":", format="extxyz")
        structure_list.extend(structures)

    if verbose > 0:
        print(len(structure_list), "structures found from trajectory files.")

    if read_sd_csv and Path.exists(Path(md_dir, "std_dev_forces.csv")):
        std_dev_df = pd.read_csv(Path(md_dir, "std_dev_forces.csv"), index_col=0)

    else:
        def find_fits_to_use():
            path_list = list(
                Path.glob(
                    Path("results", base_name, "MACE"),
                    f"fit_*/{job_dict['mace_committee']['name']}_stagetwo.model",
                )
            )
            return [int(Path(p).parent.name.split("_")[-1]) for p in path_list]

        structure_forces_dict = get_forces_for_all_maces(
            structure_list=[s.copy() for s in structure_list],
            base_name=base_name,
            job_dict=job_dict,
            base_mace=base_mace,
            fits_to_use=find_fits_to_use(),
        )

        std_dev_df = std_deviation_of_forces(structure_forces_dict, md_dir)


    index_list = list(std_dev_df[:number_of_structures].index)
    
    high_sd_structures = [
        structure_list[i] for i in index_list
    ]
    
    if verbose > 0:
        print(f"Selected {len(high_sd_structures)} structures for DFT calculations based on standard deviation of forces.")
        print(std_dev_df[:number_of_structures])
        print(f"total mean: {std_dev_df['mean_std_dev'].mean()}")

    if save_xyz:
        for structure in high_sd_structures:
            write(
                str(Path(md_dir.parent, "high_sd_structures.xyz")),
                structure,
                append=True,
            )

    return [structure_list[i] for i in index_list]


# if __name__ == "__main__":
#     al_loop = 0
#     name = "test"
#     get_structures_for_dft(
#         name,
#         initial_atoms=select_md_structures(name=name),
#         remote_info=None,
#         read_md=True,
#         number_of_structures=10,
#     )
# dft_structure_list=get_structures_for_dft('md', read_md=True, number_of_structures=10, verbose=1)
