from pathlib import Path

from ase import Atoms, read, write
from mace.calculators import MACECalculator

from alomancy.core.standard_active_learning import ActiveLearningStandardMACE
from alomancy.structure_generation.find_high_sd_structures import (
    find_high_sd_structures,
)
from alomancy.structure_generation.mc.fbmc import run_fbmc
from alomancy.structure_generation.select_initial_structures import (
    select_initial_structures,
)
from alomancy.utils.remote_atoms_list_submitter import remote_atoms_list_submitter
from alomancy.utils.remote_info import get_remote_info


class ActiveLearningForceBiasMCMACE(ActiveLearningStandardMACE):
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

        md_trajectory_paths = remote_atoms_list_submitter(
            remote_info=get_remote_info(
                job_dict["structure_generation"], input_files=[base_mace_model_path]
            ),
            base_name=base_name,
            specific_job_dict=job_dict["structure_generation"],
            target_file=f"{job_dict['structure_generation']['name']}.xyz",
            input_atoms_list=input_structures,
            function=run_fbmc,
            function_kwargs=function_kwargs,
            verbose=self.verbose,
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
