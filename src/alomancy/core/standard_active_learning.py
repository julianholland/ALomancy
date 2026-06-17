import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write

from alomancy.configs.remote_info import get_remote_info
from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
from alomancy.database.global_database import _DEFAULT_DEDUP_CONFIG_TYPES
from alomancy.high_accuracy_evaluation.dft.qe_remote_submitter import (
    qe_remote_submitter,
)
from alomancy.high_accuracy_evaluation.dft.run_qe import run_go_qe, run_sp_qe
from alomancy.initialize.initialization_structure_list import (
    compute_initialization_needs,
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
from alomancy.utils.clean_structures import clean_structures
from alomancy.utils.file_saving_and_parsing import (
    read_atoms_file_if_enabled,
)
from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train

logger = logging.getLogger(__name__)


class ActiveLearningStandardMACE(BaseActiveLearningWorkflow):
    """
    AL Technique: Committee
    MLIP: MACE
    Structure Generation: MD
    High-Accuracy Evaluation: Quantum Espresso (DFT)
    """

    def initialize_training_set(
        self, base_name: str, **_kwargs
    ) -> tuple[list[Atoms], list[Atoms]]:
        """
        Build the initial train/test sets.

        Priority order:
        1. If initial_train_file_path and initial_test_file_path already exist
           on disk, load them directly (backward-compat fast path).
        2. Otherwise, consult the global DB to determine what still needs to
           be generated (compute_initialization_needs), generate only the
           missing structures, run DFT, and add results to the DB.
        3. Build train/test sets from the DB contents.
        """
        work_dir = Path("results", base_name)
        Path.mkdir(work_dir, exist_ok=True, parents=True)

        init_job_dict = self.jobs_dict["initialization"]

        # --- Fast path: pre-existing xyz files -------------------------
        if Path(self.initial_train_file_path).exists() and Path(
            self.initial_test_file_path
        ).exists():
            train_xyzs, test_xyzs = self.load_initial_train_test_sets()
            logger.info(
                "Initial train and test sets loaded from files: %s, %s",
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

        # --- DB-aware path --------------------------------------------
        creation_kwargs = init_job_dict["creation_kwargs"]

        # Extract defaults once — used by both compute_initialization_needs
        # and create_initialization_atoms_list to avoid silent default skew.
        num_dimers_per_combo = creation_kwargs.get("num_dimers_per_combo", 10)
        num_trimers_per_combo = creation_kwargs.get("num_trimers_per_combo", 5)
        num_amorphous = creation_kwargs.get("num_amorphous", 100)
        num_stretch_compress_per_mp = creation_kwargs.get("num_stretch_compress_per_mp", 5)

        # Determine what still needs to be generated
        needs = compute_initialization_needs(
            db=self.db,
            elements=creation_kwargs["elements"],
            single_atoms=creation_kwargs.get("single_atoms", True),
            mp_structures=creation_kwargs.get("mp_structures", True),
            num_dimers_per_combo=num_dimers_per_combo,
            num_trimers_per_combo=num_trimers_per_combo,
            num_amorphous=num_amorphous,
        )

        anything_needed = (
            needs["isolated_atoms"]
            or needs["dimer_override"]
            or needs["trimer_override"]
            or needs["amorphous_override"] > 0
            or needs["mp_structures"]
        )

        if anything_needed:
            logger.info(
                "DB check: %d structure(s) already evaluated. "
                "Generating missing structures: "
                "%d isolated atoms, "
                "%d dimers, "
                "%d trimers, "
                "%d amorphous.",
                self.db.size,
                len(needs['isolated_atoms']),
                sum(needs['dimer_override'].values()),
                sum(needs['trimer_override'].values()),
                needs['amorphous_override']
            )

            # Check if structures were already generated but not yet DFT-evaluated
            generated_atoms_list = None
            if init_job_dict.get("read_generated_file") is not None:
                generated_atoms_list = read_atoms_file_if_enabled(
                    True,
                    Path(work_dir, init_job_dict["read_generated_file"]),
                )
                if generated_atoms_list:
                    logger.info(
                        "Read %d pre-generated structures from file: %s",
                        len(generated_atoms_list),
                        init_job_dict['read_generated_file']
                    )

            if not generated_atoms_list:
                generated_atoms_list = create_initialization_atoms_list(
                    work_dir=str(work_dir),
                    elements=creation_kwargs["elements"],
                    mp_structures=needs["mp_structures"],
                    single_atoms=bool(needs["isolated_atoms"]),
                    num_dimers_per_combo=num_dimers_per_combo,
                    num_trimers_per_combo=num_trimers_per_combo,
                    num_amorphous=num_amorphous,
                    num_stretch_compress_per_mp=num_stretch_compress_per_mp,
                    densities_list=creation_kwargs.get("densities_list"),
                    deform_xyz=creation_kwargs.get("deform_xyz", False),
                    max_deformation=creation_kwargs.get("max_deformation", 0.2),
                    max_atom_number=creation_kwargs.get("max_atom_number", 20),
                    composition_list=creation_kwargs.get("composition_list"),
                    seed=creation_kwargs.get("seed", self.seed),
                    isolated_atoms_override=needs["isolated_atoms"] or None,
                    dimer_override=needs["dimer_override"] or None,
                    trimer_override=needs["trimer_override"] or None,
                    amorphous_override=needs["amorphous_override"] or None,
                )

            if not generated_atoms_list:
                raise ValueError(
                    "No structures were generated. Check initialization configuration."
                )

            high_accuracy_structures = self.high_accuracy_evaluation(
                base_name=base_name,
                high_accuracy_eval_job_dict=self.jobs_dict["high_accuracy_evaluation"],
                structures=generated_atoms_list,
                allow_relaxation=True,
                start_index=0,
            )

            if not high_accuracy_structures:
                raise ValueError(
                    "No high-accuracy structures returned. Check HPC configuration "
                    "and make sure remote jobs are running correctly."
                )

            logger.info(
                "config_type of first evaluated structure: %s",
                high_accuracy_structures[0].info.get('config_type')
            )

            high_accuracy_structures = clean_structures(
                high_accuracy_structures,
                base_name,
                override_config_type=False,
                already_computed=True,
            )

            # Add newly evaluated structures to the global DB
            added = self.db.add_structures(
                high_accuracy_structures,
                skip_duplicates=True,
                config_types_to_dedup=_DEFAULT_DEDUP_CONFIG_TYPES,
            )
            logger.info("Added %d new structure(s) to the global database.", added)
        else:
            logger.info(
                "All initialization targets already met in global DB "
                "(%d structures). Skipping generation and DFT.",
                self.db.size
            )

        # --- Build train/test from DB contents -----------------------
        all_evaluated = self.db.get_all_as_atoms()

        eligible_test_structures = [
            atoms
            for atoms in all_evaluated
            if atoms.info.get("config_type") in init_job_dict["test_config_types"]
        ]

        test_structure_count = int(
            len(all_evaluated) * init_job_dict["test_to_train_ratio"]
        )

        if test_structure_count > len(eligible_test_structures):
            logger.warning(
                "Not enough eligible structures for the test set. "
                "Found %d, needed %d. "
                "All eligible structures will go to the test set.",
                len(eligible_test_structures),
                test_structure_count
            )
            test_structure_count = len(eligible_test_structures)

        if not eligible_test_structures:
            logger.warning(
                "No eligible test structures found for the specified "
                "test_config_types. All structures will be used for training."
            )
            train_xyzs = all_evaluated
            test_xyzs = []
        else:
            train_xyzs, test_xyzs = split_atoms_list_into_test_and_train(
                all_evaluated,
                test_structure_count / len(eligible_test_structures),
                self.seed,
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

        config_types_in_train = {
            atoms.info["config_type"]
            for atoms in train_xyzs
            if "config_type" in atoms.info
        }
        logger.info("Config types in training set: %s", config_types_in_train)

        return train_xyzs, test_xyzs

    def train_mlip(self, base_name: str, mlip_committee_job_dict: dict) -> pd.DataFrame:
        workdir = Path("results", base_name)

        if "mace_fit_kwargs" not in mlip_committee_job_dict:
            mlip_committee_job_dict["mace_fit_kwargs"] = {}
        logger.debug("Working directory: %s", os.getcwd())
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
                Path(operating_dir, "high_sd_structures.xyz"), ":", format="extxyz"
            )
            if isinstance(high_sd_structures, Atoms):
                high_sd_structures = [high_sd_structures]

            logger.info(
                "%d High SD structures loaded from file: %s",
                len(high_sd_structures),
                Path(operating_dir, 'high_sd_structures.xyz')
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
            logger.info(
                "Input structures for structure generation step loaded from file: %s",
                Path(operating_dir, f"{job_dict['structure_generation']['name']}_input_structures.xyz")
            )

        else:
            input_structures = select_initial_structures(
                base_name=base_name,
                structure_generation_job_dict=job_dict["structure_generation"],
                train_atoms_list=train_atoms_list,  # type: ignore
                **job_dict["structure_generation"]["structure_selection_kwargs"],
            )

        if isinstance(input_structures, Atoms):
            input_structures = [input_structures]

        logger.info(
            "%d structures selected for structure generation step.",
            len(input_structures)
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

        logger.debug("%d structures found from trajectory files.", len(structure_list))

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
        allow_relaxation: bool = False,
        start_index: int = 0,
    ) -> list[Atoms]:

        logger.debug("Starting high accuracy evaluation with %d structures.", len(structures))

        function_kwargs = {
            "high_accuracy_eval_job_dict": high_accuracy_eval_job_dict,
        }

        if Path("results", base_name, "high_accuracy_evaluation").exists():
            found_structures = list(
                Path("results", base_name, "high_accuracy_evaluation").glob(
                    f"batch_*/qe_output_*/{high_accuracy_eval_job_dict['name']}.xyz"
                )
            )
            if len(found_structures) >= len(structures) + start_index:
                logger.info(
                    "Found %d structures from previous high accuracy evaluation. "
                    "Skipping remote submission and reusing these structures.",
                    len(found_structures)
                )

                atoms_list=[read(p, format="extxyz") for p in found_structures]
                return atoms_list

            elif len(found_structures) > 0:
                logger.info(
                    "Found %d structures from previous high accuracy evaluation. "
                    "These will be reused; the rest will be submitted as new remote jobs.",
                    len(found_structures)
                )
                structures = structures[len(found_structures) + start_index:]
            else:
                logger.info(
                    "No previous results found. Submitting all %d structures.",
                    len(structures)
                )

        total_batches = int(
            np.ceil(len(structures) / high_accuracy_eval_job_dict["max_batch_size"])
        )

        logger.info(
            "Total structures: %d, max batch size: %d, total batches: %d",
            len(structures),
            high_accuracy_eval_job_dict['max_batch_size'],
            total_batches
        )
        current_batches = len(
            list(Path("results", base_name, "high_accuracy_evaluation").glob("batch_*"))
        )
        logger.debug("Found %d existing batch directories.", current_batches)

        # GO batches use indices [current_batches, total_batches).
        # SP batches (when allow_relaxation=True) use [total_batches, ...) to avoid collision.
        sp_batch_num = total_batches
        for batch_num in range(current_batches, total_batches):
            batch_start = batch_num * high_accuracy_eval_job_dict["max_batch_size"]
            batch_end = min(
                (batch_num + 1) * high_accuracy_eval_job_dict["max_batch_size"],
                len(structures),
            )
            logger.info(
                "Submitting batch %d/%d (structures %d-%d)",
                batch_num,
                total_batches,
                batch_start,
                batch_end - 1
            )
            batch_structures: list[Atoms] = structures[batch_start:batch_end]

            if allow_relaxation:
                batch_structures_to_relax = [
                    atom for atom in batch_structures
                    if atom.info.get("needs_relaxation") is True
                ]
                single_point_batch_structures = [
                    atom for atom in batch_structures
                    if atom.info.get("needs_relaxation") is not True
                ]
                logger.debug(
                    "%d GO structures, %d SP structures.",
                    len(batch_structures_to_relax),
                    len(single_point_batch_structures)
                )

                go_max_time = high_accuracy_eval_job_dict.get(
                    "go_max_time", high_accuracy_eval_job_dict["max_time"]
                )
                go_high_accuracy_eval_job_dict = copy.deepcopy(high_accuracy_eval_job_dict)
                go_high_accuracy_eval_job_dict["max_time"] = go_max_time

                if batch_structures_to_relax:
                    qe_remote_submitter(
                        remote_info=get_remote_info(
                            go_high_accuracy_eval_job_dict, input_files=[]
                        ),
                        base_name=base_name,
                        input_atoms_list=batch_structures_to_relax,
                        function=run_go_qe,
                        batch=batch_num,
                        function_kwargs=function_kwargs,
                    )

                if single_point_batch_structures:
                    qe_remote_submitter(
                        remote_info=get_remote_info(
                            high_accuracy_eval_job_dict, input_files=[]
                        ),
                        base_name=base_name,
                        input_atoms_list=single_point_batch_structures,
                        function=run_sp_qe,
                        batch=sp_batch_num,
                        function_kwargs=function_kwargs,
                    )
                    sp_batch_num += 1
            else:
                qe_remote_submitter(
                    remote_info=get_remote_info(
                        high_accuracy_eval_job_dict, input_files=[]
                    ),
                    base_name=base_name,
                    input_atoms_list=batch_structures,
                    function=run_sp_qe,
                    batch=batch_num,
                    function_kwargs=function_kwargs,
                )

        high_accuracy_structures = []
        output_name = self.jobs_dict["high_accuracy_evaluation"]["name"]
        directory_list = list(
            Path("results", base_name).glob(f"{output_name}/batch_*/qe_output_*")
        )
        for directory in directory_list:
            completed_file = Path(directory, f"{output_name}.xyz")
            structure = None
            if completed_file.exists():
                structure = read(completed_file, format="extxyz")
            if structure is not None:
                high_accuracy_structures.append(structure)

        return high_accuracy_structures
