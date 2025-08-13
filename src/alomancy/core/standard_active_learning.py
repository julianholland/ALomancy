from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from alomancy.configs.remote_info import get_remote_info
from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
from alomancy.high_accuracy_evaluation.dft.qe_remote_submitter import (
    qe_remote_submitter,
)
from alomancy.high_accuracy_evaluation.dft.run_qe import run_qe
from alomancy.mlip.committee_remote_submitter import committee_remote_submitter
from alomancy.mlip.get_mace_eval_info import (
    get_mace_eval_info,
)
from alomancy.mlip.mace_wfl import mace_fit
from alomancy.structure_generation.find_high_sd_structures import (
    find_high_sd_structures,
)
from alomancy.structure_generation.md.md_remote_submitter import md_remote_submitter
from alomancy.structure_generation.md.md_wfl import run_md
from alomancy.structure_generation.select_initial_structures import (
    select_initial_structures,
)


class ActiveLearningStandardMACE(BaseActiveLearningWorkflow):
    """
    AL Technique: Committee
    MLIP: MACE
    Structure Generation: MD
    High-Accuracy Evaluation: Quantum Espresso (DFT)
    """

    def train_mlip(self, base_name: str, mlip_committee_job_dict: dict) -> pd.DataFrame:
        workdir = Path("results", base_name)

        if "mace_fit_kwargs" not in mlip_committee_job_dict:
            mlip_committee_job_dict["mace_fit_kwargs"] = {}

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

        input_structures = select_initial_structures(
            base_name=base_name,
            structure_generation_job_dict=job_dict["structure_generation"],
            train_atoms_list=train_atoms_list,  # type: ignore
            verbose=self.verbose,
            **job_dict["structure_generation"]["structure_selection_kwargs"],
        )

        Path.mkdir(
            Path("results", base_name, job_dict["structure_generation"]["name"]),
            exist_ok=True,
            parents=True,
        )
        write(
            Path(
                "results",
                base_name,
                job_dict["structure_generation"]["name"],
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

        list_of_other_calculators = [
            MACECalculator(
                model_paths=[mace_model_path],
                device="cpu",
                default_dtype="float64",
            )
            for mace_model_path in model_paths_list
            if str(mace_model_path) != base_mace_model_path
        ]
        high_sd_structures = find_high_sd_structures(
            structure_list=structure_list,
            base_name=base_name,
            job_dict=job_dict,
            list_of_other_calculators=list_of_other_calculators,
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
        function_kwargs = {
            "high_accuracy_eval_job_dict": high_accuracy_eval_job_dict,
        }

        high_accuracy_structure_paths = qe_remote_submitter(
            remote_info=get_remote_info(high_accuracy_eval_job_dict, input_files=[]),
            base_name=base_name,
            target_file=f"{high_accuracy_eval_job_dict['name']}.xyz",
            input_atoms_list=structures,
            function=run_qe,
            function_kwargs=function_kwargs,
        )

        high_accuracy_structures = []
        for path in high_accuracy_structure_paths:
            structure = read(path, format="extxyz")
            high_accuracy_structures.append(structure)

        return high_accuracy_structures
