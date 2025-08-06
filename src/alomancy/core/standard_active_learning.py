from ase import Atoms
from pathlib import Path

from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
from alomancy.configs.remote_info import get_remote_info
from alomancy.mlip.committee_remote_submitter import committee_remote_submitter
from alomancy.mlip.mace_wfl import mace_fit


from typing import List, Optional, Dict, Any

# def active_learn_mace(
#     initial_train_file_path,
#     initial_test_file_path,
#     number_of_al_loops: int = 5,
#     verbose: int = 0,
#     start_loop: int = 0,
#     target_force_error: float = 0.05,  # eV
#     committee_size: int = 5,
#     md_runs: int = 5,
# ):
class ActiveLearningStandardMACE(BaseActiveLearningWorkflow):
    """
    AL Technique: Committee
    MLIP: MACE
    Structure Generation: MD
    High-Accuracy Evaluation: QE (DFT)
    """

    def train_mlip(self, base_name: str, mlip_committee_job_dict: dict, train_data: List[Atoms], **kwargs) -> Optional[str]:
        workdir = Path("results", base_name)
        
        committee_remote_submitter(
            remote_info=get_remote_info(mlip_committee_job_dict,
                                        input_files=[str(Path(workdir, "train_set.xyz")),
                                        str(Path(workdir, "test_set.xyz")),
                                        ],),
            base_name=base_name,
            seed=803,
            size_of_committee=2,
            function=mace_fit,
            function_kwargs={"epochs": 10, "mlip_committee_job_dict": mlip_committee_job_dict, "workdir_str": str(workdir)},
        )

#         # 2. evaluate model
#         evaluation_df = mace_al_loop_average_error(base_name, plot=False)
#         if self.verbose > 0:
#             print(
#                 f"AL Loop {al_loop}, MAE (energy): {evaluation_df['mae_e'].iloc[al_loop]}, MAE (forces): {evaluation_df['mae_f'].iloc[al_loop]}"
#             )
#         if evaluation_df["mae_f"].iloc[al_loop] < target_force_error:
#             print(f"AL Loop {al_loop} reached target force error.")
#             break

#         # 3. select structures from train set to perform MD on
#         md_input_structures = select_md_structures(
#             base_name=base_name,
#             job_name=JOB_DICT["md_run"]["name"],
#             number_of_mds=md_runs,
#             chem_formula_list=[],
#             atom_number_range=(9, 21),
#             enforce_chemical_diversity=True,
#             train_xyzs=train_xyzs,  # type: ignore
#             verbose=verbose,
#         )

#         Path.mkdir(Path(loop_dir, "MD"), exist_ok=True, parents=True)
#         write(
#             Path(loop_dir, "MD", "md_input_structures.xyz"),
#             md_input_structures,
#             format="extxyz",
#         )

#         # 4. select high standard deviation structures from MD
#         dft_input_structures = get_structures_for_dft(
#             base_name=base_name,
#             job_dict=JOB_DICT,
#             initial_atoms=md_input_structures,
#             remote_info=get_remote_info(
#                 # hpc="fhi-raccoon", job="md_run", input_files=[]
#                 hpc="raven_gpu",
#                 job="md_run",
#                 input_files=[],
#             ),
#             number_of_structures=50,
#             verbose=verbose,
#             temperature=1200.0,
#             steps=1000,
#             timestep_fs=0.5,
#             base_mace=str(
#                 Path(
#                     loop_dir,
#                     "MACE",
#                     "fit_0",
#                     f"{JOB_DICT['mace_committee']['name']}_stagetwo.model",
#                 )
#             ),
#         )
#         Path.mkdir(Path(loop_dir, "DFT"), exist_ok=True, parents=True)
#         write(
#             Path(loop_dir, "DFT", "dft_input_structures.xyz"),
#             dft_input_structures,
#             format="extxyz",
#         )

#         # 5. perform DFT calculations on selected structures
#         dft_structures = perform_qe_calculations_per_cell(
#             base_name=base_name,
#             job_name=JOB_DICT["dft_run"]["name"],
#             atoms_list=dft_input_structures,
#             remote_info=get_remote_info(hpc="raven", job="dft_run", input_files=[]),
#             hpc="raven",
#             verbose=verbose,
#         )
#         print(dft_structures)

#         # 6. add DFT results to training data
#         train_xyzs = list(
#             add_new_training_data(
#                 base_name=base_name,
#                 job_dict=JOB_DICT,
#                 train_xyzs=train_xyzs,  # type: ignore
#             )
#         )

#         al_loop += 1
#         print(
#             f"Active learning loop {al_loop} completed. New training set size: {len(train_xyzs)}"
#         )


# if __name__ == "__main__":
#     train_data_dir = Path("mace_general")
#     active_learn_mace(
#         initial_test_file_path=Path(
#             train_data_dir, "ac_all_33_2025_07_31_ftrim_10_grpspread_01_test_set.xyz"
#         ),
#         initial_train_file_path=Path(
#             train_data_dir, "ac_all_33_2025_07_31_ftrim_10_grpspread_01_train_set.xyz"
#         ),
#         number_of_al_loops=50,
#         verbose=1,
#         target_force_error=0.05,
#         start_loop=0,
#         committee_size=3,
#         md_runs=2,
#     )
