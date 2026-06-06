from pathlib import Path

import numpy as np
import os
import pandas as pd
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from alomancy.configs.remote_info import get_remote_info
from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
from alomancy.high_accuracy_evaluation.dft.qe_remote_submitter import (
    qe_remote_submitter,
)
from alomancy.high_accuracy_evaluation.dft.run_qe import run_go_qe, run_sp_qe
from alomancy.initialize.initialization_structure_list import (
    create_initialization_atoms_list,
)
from alomancy.mlip.committee_remote_submitter import committee_remote_submitter
from alomancy.mlip.get_mace_eval_info import (
    get_mace_eval_info,
)
from alomancy.mlip.mace_wfl import mace_fit
from alomancy.structure_generation.find_high_sd_structures import (
    find_high_sd_structures,
)
from alomancy.structure_generation.md.md_remote_submitter import (
    all_maces_remote_submitter,
    md_remote_submitter,
)
from alomancy.structure_generation.md.md_wfl import get_forces_for_all_maces, run_md
from alomancy.structure_generation.select_initial_structures import (
    select_initial_structures,
)
from alomancy.utils.file_saving_and_parsing import (
    read_atoms_file_if_enabled,
)
from alomancy.utils.test_train_manager import (
    extend_test_and_train_sets_with_extra_dataset,
    split_atoms_list_into_test_and_train,
)
from alomancy.utils.clean_structures import clean_structures


class ActiveLearningStandardMACE(BaseActiveLearningWorkflow):
    """
    AL Technique: Committee
    MLIP: MACE
    Structure Generation: MD
    High-Accuracy Evaluation: Quantum Espresso (DFT)
    """

    def initialize_training_set(
        self, base_name: str, **kwargs
    ) -> tuple[list[Atoms], list[Atoms]]:

        work_dir = Path("results", base_name)
        Path.mkdir(work_dir, exist_ok=True, parents=True)

        init_job_dict = self.jobs_dict["initialization"]
        train_xyzs, test_xyzs = [], []

        # check if test and train files already exist
        if Path.exists(Path(self.initial_train_file_path)) and Path.exists(
            Path(self.initial_test_file_path)
        ):
            train_xyzs, test_xyzs = self.load_initial_train_test_sets()
            print(
                "Initial train and test sets loaded from files:",
                self.initial_train_file_path,
                self.initial_test_file_path,
            )
            write(
                Path(work_dir, Path(self.initial_train_file_path).name),
                train_xyzs,
                format="extxyz",
            )
            write(
                Path(work_dir, Path(self.initial_test_file_path).name),
                test_xyzs,
                format="extxyz",
            )
            return train_xyzs, test_xyzs

        # check if generated dataset structures should be read from a file
        generated_atoms_list = None
        if init_job_dict["read_generated_file"] is not None:
            generated_atoms_list = read_atoms_file_if_enabled(
                True, Path(work_dir, init_job_dict["read_generated_file"])
            )
            print(
                "Read generated structures from file:",
                init_job_dict["read_generated_file"],
            )

        print("Generated atoms list:", generated_atoms_list)
        if generated_atoms_list is None:
            generated_atoms_list = create_initialization_atoms_list(
                work_dir=str(work_dir), **init_job_dict["creation_kwargs"]
            )

        if generated_atoms_list is None or len(generated_atoms_list) == 0:
            raise ValueError(
                "No generated structures found. Please check the configuration for the initialization step."
            )

        sp_atoms_list = [
            atom
            for atom in generated_atoms_list
            if "needs_relaxation" not in atom.info
            or atom.info["needs_relaxation"] is False
        ]

        go_atoms_list = [
            atom
            for atom in generated_atoms_list
            if "needs_relaxation" in atom.info and atom.info["needs_relaxation"] is True
        ]

        print(
            f"Created {len(sp_atoms_list)} structures that do not need relaxation and {len(go_atoms_list)} structures that do need relaxation."
        )

        function_kwargs = {
            "high_accuracy_eval_job_dict": self.jobs_dict["high_accuracy_evaluation"],
        }

        qe_remote_submitter(
            remote_info=get_remote_info(
                job_dict=self.jobs_dict["high_accuracy_evaluation"],
                input_files=[],
            ),
            base_name=base_name,
            input_atoms_list=sp_atoms_list,
            function=run_sp_qe,
            function_kwargs=function_kwargs,
        )

        qe_remote_submitter(
            remote_info=get_remote_info(
                self.jobs_dict["high_accuracy_evaluation"], input_files=[]
            ),
            base_name=base_name,
            input_atoms_list=go_atoms_list,
            function=run_go_qe,
            function_kwargs=function_kwargs,
        )

        high_accuracy_structure_paths = list(
            Path("results", base_name).glob(
                f"qe_output_*/{self.jobs_dict['high_accuracy_evaluation']['name']}.xyz"
            )
        )

        print(
            len(high_accuracy_structure_paths), "high accuracy structure files found."
        )

        if len(high_accuracy_structure_paths) == 0:
            raise ValueError(
                "No high accuracy structures found. Please check the configuration for the high accuracy evaluation step and make sure the remote jobs are running correctly."
            )

        high_accuracy_structures = []
        for path in high_accuracy_structure_paths:
            structure = read(path, format="extxyz")
            high_accuracy_structures.append(structure)

        high_accuracy_structures = clean_structures(
            high_accuracy_structures,
            base_name,
            self.jobs_dict["high_accuracy_evaluation"],
            already_computed=True,
        )

        test_structure_count = int(
            len(high_accuracy_structures) * init_job_dict["test_to_train_ratio"]
        )

        elegible_test_structures = [
            atoms
            for atoms in high_accuracy_structures
            if "config_type" in atoms.info
            and atoms.info["config_type"] in init_job_dict["test_config_types"]
        ]
        print(test_structure_count, len(elegible_test_structures))
        if test_structure_count > len(elegible_test_structures):
            print(
                f"WARNING: Not enough elegible structures found for the test set based on the specified test_config_types. Found {len(elegible_test_structures)} elegible structures but need {test_structure_count} for the test set based on the specified test_to_train_ratio. Consider adjusting the test_to_train_ratio or the test_config_types in the configuration. For now, all elegible structures will be added to the test set and the rest will be added to the training set."
            )
            test_structure_count = len(elegible_test_structures)

        if len(elegible_test_structures) == 0:
            print(
                f"WARNING: No elegible structures found for the test set based on the specified test_config_types. Found {len(elegible_test_structures)} elegible structures. Consider adjusting the test_to_train_ratio or the test_config_types in the configuration. For now, all structures will be added to the training set and the test set will be empty."
            )
            new_test_xyzs = []
            new_train_xyzs = high_accuracy_structures
        else:
            new_train_xyzs, new_test_xyzs = split_atoms_list_into_test_and_train(
                high_accuracy_structures,
                test_structure_count / len(elegible_test_structures),
                self.seed,
            )

        train_xyzs.extend(new_train_xyzs)
        test_xyzs.extend(new_test_xyzs)

        write(
            Path(work_dir, Path(self.initial_train_file_path).name),
            train_xyzs,
            format="extxyz",
        )
        write(
            Path(work_dir, Path(self.initial_test_file_path).name),
            test_xyzs,
            format="extxyz",
        )

        config_types_in_train_set = set(
            atoms.info["config_type"]
            for atoms in train_xyzs
            if "config_type" in atoms.info
        )
        print(f"Config types in training set: {config_types_in_train_set}")

        return train_xyzs, test_xyzs

    def train_mlip(self, base_name: str, mlip_committee_job_dict: dict) -> pd.DataFrame:
        workdir = Path("results", base_name)

        if "mace_fit_kwargs" not in mlip_committee_job_dict:
            mlip_committee_job_dict["mace_fit_kwargs"] = {}
        print("here when reading fit dirs:", os.getcwd())
        if (
            len(
                list(
                    Path(f"results/{base_name}").glob(
                        f"{mlip_committee_job_dict['name']}/fit_*/{mlip_committee_job_dict['name']}_stagetwo_compiled.model"
                    )
                )
            )
            < mlip_committee_job_dict["size_of_committee"]
        ):
            committee_remote_submitter(
                remote_info=get_remote_info(
                    mlip_committee_job_dict,
                    input_files=[
                        str(Path(workdir, "train_set.xyz")),
                        str(Path(workdir, "test_set.xyz")),
                    ],
                ),
                base_name=base_name,
                target_file=f"{mlip_committee_job_dict['name']}_stagetwo_compiled.model",
                seed=803,
                size_of_committee=mlip_committee_job_dict["size_of_committee"],
                function=mace_fit,
                function_kwargs={
                    "mlip_committee_job_dict": mlip_committee_job_dict,
                    "workdir_str": str(workdir),
                },
            )

        mae_avg_results = get_mace_eval_info(
            mlip_committee_job_dict=mlip_committee_job_dict
        )

        return mae_avg_results

    def generate_structures(
        self, base_name: str, job_dict: dict, train_atoms_list: list[Atoms]
    ) -> list[Atoms]:
        if "structure_selection_kwargs" not in job_dict["structure_generation"]:
            job_dict["structure_generation"]["structure_selection_kwargs"] = {}

        operating_dir = Path(
            "results", base_name, job_dict["structure_generation"]["name"]
        )

        # skip step if high SD structures already exist from a previous run
        if Path(operating_dir, "high_sd_structures.xyz").exists():
            high_sd_structures = read(
                Path(operating_dir, "high_sd_structures.xyz"), ':', format="extxyz"
            )
            if isinstance(high_sd_structures, Atoms):
                high_sd_structures = [high_sd_structures]

            print(
                f"{len(high_sd_structures)} High SD structures loaded from file: {Path(operating_dir, 'high_sd_structures.xyz')}"
            )

            return high_sd_structures

        if Path(
            operating_dir,
            f"{job_dict['structure_generation']['name']}_input_structures.xyz",
        ).exists():
            input_structures = read(
                Path(
                    operating_dir,
                    f"{job_dict['structure_generation']['name']}_input_structures.xyz",
                ),
                format="extxyz",
            )
            print(
                f"Input structures for structure generation step loaded from file: {Path(operating_dir, f'{job_dict['structure_generation']['name']}_input_structures.xyz')}"
            )

        else:
            input_structures = select_initial_structures(
                base_name=base_name,
                structure_generation_job_dict=job_dict["structure_generation"],
                train_atoms_list=train_atoms_list,  # type: ignore
                verbose=self.verbose,
                **job_dict["structure_generation"]["structure_selection_kwargs"],
            )

        if isinstance(input_structures, Atoms):
            input_structures = [input_structures]

        print(
            f"{len(input_structures)} structures selected for structure generation step."
        )
        Path.mkdir(
            Path(operating_dir),
            exist_ok=True,
            parents=True,
        )
        write(
            Path(
                operating_dir,
                f"{job_dict['structure_generation']['name']}_input_structures.xyz",
            ),
            input_structures,
            format="extxyz",
        )
        base_mace_model_path = str(
            Path(
                "results",
                base_name,
                job_dict["mlip_committee"]["name"],
                "fit_0",
                f"{job_dict['mlip_committee']['name']}_stagetwo.model",
            )
        )

        if "run_md_kwargs" not in job_dict["structure_generation"]:
            job_dict["structure_generation"]["run_md_kwargs"] = {}

        function_kwargs = {
            "structure_generation_job_dict": job_dict["structure_generation"],
            "total_md_runs": len(input_structures),
            "model_path": [
                base_mace_model_path
            ],  # need to pass model path to preserve consistent dtype
            "verbose": self.verbose,
            **job_dict["structure_generation"]["run_md_kwargs"],
        }

        md_trajectory_paths = md_remote_submitter(
            remote_info=get_remote_info(
                job_dict["structure_generation"], input_files=[base_mace_model_path]
            ),
            base_name=base_name,
            target_file=f"{job_dict['structure_generation']['name']}.xyz",
            input_atoms_list=input_structures,
            function=run_md,
            function_kwargs=function_kwargs,
        )

        structure_list = []
        for md_trajectory_path in md_trajectory_paths:
            structures = read(md_trajectory_path, ":", format="extxyz")
            structure_list.extend(structures)

        if self.verbose > 0:
            print(len(structure_list), "structures found from trajectory files.")

        model_paths_list = list(
            Path.glob(
                Path("results", base_name, job_dict["mlip_committee"]["name"]),
                f"fit_*/{job_dict['mlip_committee']['name']}_stagetwo.model",
            )
        )

        structure_forces_dict = all_maces_remote_submitter(
            remote_info=get_remote_info(
                job_dict["structure_generation"],
                input_files=[str(m) for m in model_paths_list],
            ),
            function=get_forces_for_all_maces,
            function_kwargs={
                "structure_list": structure_list,
                "base_name": base_name,
                "job_dict": job_dict,
                "base_mlip": base_mace_model_path,
                "fits_to_use": list(
                    range(1, job_dict["mlip_committee"]["size_of_committee"])
                ),
            },
        )

        high_sd_structures = find_high_sd_structures(
            structure_list=structure_list,
            base_name=base_name,
            job_dict=job_dict,
            structure_forces_dict=structure_forces_dict,
            verbose=self.verbose,
        )

        # Assign job IDs to high SD structures
        for i in range(len(high_sd_structures)):
            high_sd_structures[i].info["job_id"] = i

        return high_sd_structures

    def high_accuracy_evaluation(
        self,
        base_name: str,
        high_accuracy_eval_job_dict: dict,
        structures: list[Atoms],
    ) -> list[Atoms]:
        
        
        print("Starting high accuracy evaluation with", len(structures), "structures.")

        function_kwargs = {
            "high_accuracy_eval_job_dict": high_accuracy_eval_job_dict,
        }
        if Path("results", base_name, "high_accuracy_evaluation").exists():
            found_structures = list(
                Path("results", base_name, "high_accuracy_evaluation").glob(
                    f"qe_output_*/{high_accuracy_eval_job_dict['name']}.xyz"
                )
            )
            if len(found_structures) > 0:
                print(
                    f"Found {len(found_structures)} structures from previous high accuracy evaluation. Skipping remote submission and reusing these structures for high accuracy evaluation results."
                )
                high_accuracy_structures = []
                for path in found_structures:
                    structure = read(path, format="extxyz")
                    high_accuracy_structures.append(structure)

                return high_accuracy_structures
        
        qe_remote_submitter(
            remote_info=get_remote_info(high_accuracy_eval_job_dict, input_files=[]),
            base_name=base_name,
            input_atoms_list=structures,
            function=run_sp_qe,
            function_kwargs=function_kwargs,
        )

        high_accuracy_structures = []
        collected_output_files = list(
            Path.glob(
                Path("results", base_name, "qe_output_*"),
                f"{self.jobs_dict['high_accuracy_evaluation']['name']}.xyz",
            )
        )
        for path in collected_output_files:
            structure = read(path, format="extxyz")
            high_accuracy_structures.append(structure)

        return high_accuracy_structures
