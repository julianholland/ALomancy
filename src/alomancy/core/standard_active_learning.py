import copy
import logging
import os
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import read, write

from alomancy.configs.remote_info import get_remote_info
from alomancy.core.base_active_learning import BaseActiveLearningWorkflow
from alomancy.database.global_database import _DEFAULT_DEDUP_CONFIG_TYPES
from alomancy.high_accuracy_evaluation.dft import (
    get_dft_functions,
    warn_mismatched_kwargs,
)
from alomancy.initialize.initialization_structure_list import (
    compute_initialization_needs,
    create_initialization_atoms_list,
)
from alomancy.mlip.get_mace_eval_info import (
    get_mace_eval_info,
    select_best_committee_model,
)
from alomancy.mlip.mace_wfl import mace_fit
from alomancy.remote_submission import (
    all_maces_remote_submitter,
    ase_remote_submitter,
    committee_remote_submitter,
    md_remote_submitter,
)
from alomancy.remote_submission.submitters import ASE_OUTPUT_PREFIX
from alomancy.structure_generation.find_high_sd_structures import (
    find_high_sd_structures,
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


def _needs_anything(needs: dict) -> bool:
    return bool(
        needs["isolated_atoms"]
        or needs["dimer_override"]
        or needs["trimer_override"]
        or needs["amorphous_override"] > 0
        or needs["mp_structures"]
    )


def _read_mace_eval_predictions(fit_dir: Path) -> dict[int, dict]:
    """Read per-structure MACE predictions from train_pred.xyz / test_pred.xyz.

    These files are written by mace_fit on the remote GPU node immediately after
    training, so predictions are available locally without re-running inference.
    Returns {global_db_id: {"energy": float, "forces": list}} or empty dict.
    """
    preds: dict[int, dict] = {}
    for tag in ("train", "test"):
        xyz = fit_dir / f"{tag}_pred.xyz"
        if not xyz.exists():
            continue
        try:
            atoms_list = list(read(xyz, ":", format="extxyz"))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", xyz, exc)
            continue
        for atoms in atoms_list:
            gid = atoms.info.get("global_db_id")
            if gid is None or "mace_energy" not in atoms.info:
                continue
            forces = atoms.arrays.get("mace_forces")
            preds[int(gid)] = {
                "energy": float(atoms.info["mace_energy"]),
                "forces": forces.tolist() if forces is not None else [],
            }
    return preds


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
        if (
            Path(self.initial_train_file_path).exists()
            and Path(self.initial_test_file_path).exists()
        ):
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
            # Seed DB with split tags if empty so restart can reconstruct
            # train/test from the DB instead of accumulated xyz files.
            if self.db.size == 0:
                self.db.add_structures(train_xyzs, split="train", skip_duplicates=True)
                self.db.add_structures(test_xyzs, split="test", skip_duplicates=True)
            return train_xyzs, test_xyzs

        # --- DB-aware path --------------------------------------------
        creation_kwargs = init_job_dict["creation_kwargs"]

        # Extract defaults once — used by both compute_initialization_needs
        # and create_initialization_atoms_list to avoid silent default skew.
        num_dimers_per_combo = creation_kwargs.get("num_dimers_per_combo", 10)
        num_trimers_per_combo = creation_kwargs.get("num_trimers_per_combo", 5)
        num_amorphous = creation_kwargs.get("num_amorphous", 100)
        num_stretch_compress_per_mp = creation_kwargs.get(
            "num_stretch_compress_per_mp", 5
        )

        # Check DB first — existing structures take priority over extra_datasets
        if self.db.size > 0:
            logger.info(
                "Global DB has %d existing structures; reading those in first.",
                self.db.size,
            )

        _needs_kwargs = {
            "db": self.db,
            "elements": creation_kwargs["elements"],
            "_single_atoms": creation_kwargs.get("single_atoms", True),
            "mp_structures": creation_kwargs.get("mp_structures", True),
            "num_dimers_per_combo": num_dimers_per_combo,
            "num_trimers_per_combo": num_trimers_per_combo,
            "num_amorphous": num_amorphous,
        }
        needs = compute_initialization_needs(**_needs_kwargs)

        # Seed extra_datasets only if DB is still missing some initialization targets
        extra_datasets = init_job_dict.get("extra_datasets") or []
        if extra_datasets and _needs_anything(needs):
            for ed in extra_datasets:
                self._seed_db_from_extra_dataset(ed)
            needs = compute_initialization_needs(**_needs_kwargs)

        anything_needed = _needs_anything(needs)

        if anything_needed:
            logger.info(
                "DB check: %d structure(s) already evaluated. "
                "Generating missing structures: "
                "%d isolated atoms, "
                "%d dimers, "
                "%d trimers, "
                "%d amorphous.",
                self.db.size,
                len(needs["isolated_atoms"]),
                sum(needs["dimer_override"].values()),
                sum(needs["trimer_override"].values()),
                needs["amorphous_override"],
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
                        init_job_dict["read_generated_file"],
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
                    amorphous_atom_number=creation_kwargs.get(
                        "amorphous_atom_number", 20
                    ),
                    mp_max_energy_above_hull=creation_kwargs.get(
                        "mp_max_energy_above_hull", 0.1
                    ),
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
                high_accuracy_structures[0].info.get("config_type"),
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
                self.db.size,
            )

        # --- Build train/test from DB contents -----------------------
        all_evaluated = self.db.get_all_as_atoms()

        test_config_types = set(init_job_dict["test_config_types"])
        eligible_test_structures: list[Atoms] = []
        always_train_structures: list[Atoms] = []
        for atoms in all_evaluated:
            (
                eligible_test_structures
                if atoms.info.get("config_type") in test_config_types
                else always_train_structures
            ).append(atoms)

        if not eligible_test_structures:
            logger.warning(
                "No eligible test structures found for the specified "
                "test_config_types. All structures will be used for training."
            )
            train_xyzs = all_evaluated
            test_xyzs = []
        else:
            # test_to_train_ratio applies only within the test_config_types
            # pool, not against the whole DB. Dimers/trimers/stretch_compress/
            # IsolatedAtom (always_train_structures) never count toward this
            # ratio's denominator — with a small test_config_types pool and a
            # much larger always-train pool, computing the quota against
            # len(all_evaluated) could exceed the entire eligible pool,
            # routing 100% of it to test and leaving train_xyzs with zero
            # representatives of that config_type (e.g. init_amorphous),
            # permanently once update_splits_post_hoc tags the DB.
            eligible_train, test_xyzs = split_atoms_list_into_test_and_train(
                eligible_test_structures,
                init_job_dict["test_to_train_ratio"],
                self.seed,
            )

            # Guarantee every eligible config_type keeps at least one
            # representative in train_xyzs, as a backstop against an unlucky
            # shuffle leaving a low-count config_type entirely in test.
            train_config_types = {a.info.get("config_type", "") for a in eligible_train}
            eligible_config_types = {
                a.info.get("config_type", "") for a in eligible_test_structures
            }
            missing_types = eligible_config_types - train_config_types
            if missing_types:
                for config_type in missing_types:
                    idx = next(
                        i
                        for i, a in enumerate(test_xyzs)
                        if a.info.get("config_type", "") == config_type
                    )
                    eligible_train.append(test_xyzs.pop(idx))
                logger.warning(
                    "Reserved one structure from each of %s for training "
                    "to avoid entirely excluding these config_types from "
                    "train_atoms_list.",
                    sorted(missing_types),
                )

            # IsolatedAtom and other ineligible types always go to training so
            # MACE can read E0s for every element from the training file.
            train_xyzs = always_train_structures + eligible_train

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

    def train_mlip(self, base_name: str, job_dict: dict) -> pd.DataFrame:
        mlip_committee_job_dict = job_dict["mlip_committee"]
        if self._phase_done(base_name, "train_mlip"):
            logger.info("train_mlip already done for %s, reloading metrics.", base_name)
            return get_mace_eval_info(mlip_committee_job_dict=mlip_committee_job_dict)

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
                seed=803,
                size_of_committee=mlip_committee_job_dict["size_of_committee"],
                function=mace_fit,
                function_kwargs={
                    "job_dict": job_dict,
                    "workdir_str": str(workdir),
                },
            )

        model_glob = (
            f"{mlip_committee_job_dict['name']}/fit_*/"
            f"{mlip_committee_job_dict['name']}_stagetwo_compiled.model"
        )
        configured_size = mlip_committee_job_dict["size_of_committee"]

        def _found_fit_indices() -> set[int]:
            return {
                int(p.parent.name.removeprefix("fit_"))
                for p in Path(f"results/{base_name}").glob(model_glob)
            }

        found_fit_indices = _found_fit_indices()

        if len(found_fit_indices) < configured_size:
            logger.warning(
                "train_mlip for %s: expected %d trained models but found %d. "
                "Check remote job logs for failures.",
                base_name,
                configured_size,
                len(found_fit_indices),
            )
            if len(found_fit_indices) < 3:
                # 3 is the minimum committee size for a reasonable force
                # std-dev across members; below that, uncertainty-based
                # structure selection has too few members to be meaningful.
                missing_fit_indices = sorted(
                    set(range(configured_size)) - found_fit_indices
                )[: 3 - len(found_fit_indices)]
                logger.info(
                    "train_mlip for %s: only found %d trained model(s). "
                    "Retraining missing fit(s) %s to reach the minimum "
                    "committee size of 3.",
                    base_name,
                    len(found_fit_indices),
                    missing_fit_indices,
                )
                committee_remote_submitter(
                    remote_info=get_remote_info(
                        mlip_committee_job_dict,
                        input_files=[
                            str(Path(workdir, "train_set.xyz")),
                            str(Path(workdir, "test_set.xyz")),
                        ],
                    ),
                    base_name=base_name,
                    seed=803,
                    function=mace_fit,
                    function_kwargs={
                        "job_dict": job_dict,
                        "workdir_str": str(workdir),
                    },
                    fit_indices=missing_fit_indices,
                )
                found_fit_indices = _found_fit_indices()

                if len(found_fit_indices) < 3:
                    raise RuntimeError(
                        f"train_mlip for {base_name}: still only "
                        f"{len(found_fit_indices)} trained model(s) after "
                        f"retrying the missing committee member(s) "
                        f"{missing_fit_indices} — need at least 3 for a "
                        "usable committee std-dev. Check remote job logs "
                        "for failures."
                    )

        mae_avg_results = get_mace_eval_info(
            mlip_committee_job_dict=mlip_committee_job_dict
        )

        self._mark_phase_done(base_name, "train_mlip")
        return mae_avg_results

    def store_mlip_predictions(
        self, loop_idx: int, base_name: str, job_dict: dict
    ) -> None:
        done_file = Path(f"results/{base_name}/mace_predictions.done")
        if done_file.exists():
            logger.info("MACE predictions already stored for %s, skipping.", base_name)
            return

        mlip_job_dict = job_dict["mlip_committee"]
        name = mlip_job_dict["name"]
        size = mlip_job_dict.get("size_of_committee", 1)
        workdir = Path(f"results/{base_name}")

        any_stored = False
        for fit_idx in range(size):
            fit_dir = workdir / name / f"fit_{fit_idx}"
            preds = _read_mace_eval_predictions(fit_dir)
            if preds:
                self.db.store_mace_predictions(loop_idx, fit_idx, preds)
                logger.info(
                    "Stored MACE predictions from eval files: loop %d fit %d (%d structures).",
                    loop_idx,
                    fit_idx,
                    len(preds),
                )
                any_stored = True
            else:
                logger.debug(
                    "No eval prediction files found for fit_%d in %s.", fit_idx, fit_dir
                )

        if not any_stored:
            logger.info(
                "No MACE eval prediction files found for %s — parity plots will not be "
                "available for this loop. Predictions are written during remote training "
                "from alomancy v0.4.2 onwards.",
                base_name,
            )

        done_file.touch()

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

            # MD-generated structures must never be routed to geometry optimisation
            # in high_accuracy_evaluation, even if a stale needs_relaxation=True was
            # inherited by the on-disk copy from an earlier run.
            for structure in high_sd_structures:
                structure.info["needs_relaxation"] = False

            logger.info(
                "%d High SD structures loaded from file: %s",
                len(high_sd_structures),
                Path(operating_dir, "high_sd_structures.xyz"),
            )

            self._mark_phase_done(base_name, "generate_structures")
            return high_sd_structures

        input_structures_path = Path(
            operating_dir,
            f"{job_dict['structure_generation']['name']}_input_structures.xyz",
        )
        expected_count = job_dict["structure_generation"][
            "structure_selection_kwargs"
        ].get("max_number_of_concurrent_jobs", 5)

        if input_structures_path.exists():
            input_structures = read(input_structures_path, ":", format="extxyz")
            if len(input_structures) < expected_count:
                # A prior run's write() can leave a partial file behind (e.g. a
                # full local disk truncating it mid-write, as happened on
                # 2026-08-04 — see TODO.md). Trusting it silently starves
                # find_high_sd_structures of candidates far downstream with a
                # confusing error. Regenerate instead of reusing a short file.
                logger.warning(
                    "Input structures file %s has only %d structure(s), expected "
                    "%d — treating it as incomplete/corrupted and regenerating.",
                    input_structures_path,
                    len(input_structures),
                    expected_count,
                )
                input_structures = select_initial_structures(
                    base_name=base_name,
                    structure_generation_job_dict=job_dict["structure_generation"],
                    train_atoms_list=train_atoms_list,  # type: ignore
                    seed=self.seed,
                    **job_dict["structure_generation"]["structure_selection_kwargs"],
                )
            else:
                logger.info(
                    "Input structures for structure generation step loaded "
                    "from file: %s",
                    input_structures_path,
                )

        else:
            input_structures = select_initial_structures(
                base_name=base_name,
                structure_generation_job_dict=job_dict["structure_generation"],
                train_atoms_list=train_atoms_list,  # type: ignore
                seed=self.seed,
                **job_dict["structure_generation"]["structure_selection_kwargs"],
            )

        if isinstance(input_structures, Atoms):
            input_structures = [input_structures]

        logger.info(
            "%d structures selected for structure generation step.",
            len(input_structures),
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
        best_fit_idx, best_model_path = select_best_committee_model(
            base_name,
            job_dict["mlip_committee"],
            seed=self.seed,
        )
        base_mace_model_path = str(best_model_path)
        committee_size = job_dict["mlip_committee"]["size_of_committee"]
        fits_to_use = [i for i in range(committee_size) if i != best_fit_idx]

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
                "fits_to_use": fits_to_use,
            },
        )

        high_sd_structures = find_high_sd_structures(
            structure_list=structure_list,
            base_name=base_name,
            job_dict=job_dict,
            structure_forces_dict=structure_forces_dict,
        )

        # Assign job IDs to high SD structures. Also force needs_relaxation=False:
        # MD seeds can inherit needs_relaxation=True from their source structure (e.g.
        # amorphous init structures) via .copy(), and that flag survives the MD
        # trajectory unless cleared here. MD-generated structures must never be
        # routed to geometry optimisation in high_accuracy_evaluation.
        for i in range(len(high_sd_structures)):
            high_sd_structures[i].info["job_id"] = i
            high_sd_structures[i].info["needs_relaxation"] = False

        self._mark_phase_done(base_name, "generate_structures")
        return high_sd_structures

    def high_accuracy_evaluation(
        self,
        base_name: str,
        high_accuracy_eval_job_dict: dict,
        structures: list[Atoms],
        allow_relaxation: bool = False,
        start_index: int = 0,
    ) -> list[Atoms]:
        sentinel_results = Path("results", base_name, "high_accuracy_eval_results.xyz")
        if self._phase_done(base_name, "high_accuracy_eval"):
            logger.info(
                "high_accuracy_eval already done for %s, loading cached results.",
                base_name,
            )
            return list(read(sentinel_results, ":"))

        calculator = high_accuracy_eval_job_dict.get("calculator", "qe")
        warn_mismatched_kwargs(calculator, high_accuracy_eval_job_dict)
        run_sp, run_go = get_dft_functions(calculator)

        logger.debug(
            "Starting high accuracy evaluation with %d structures (calculator=%s).",
            len(structures),
            calculator,
        )

        function_kwargs = {
            "high_accuracy_eval_job_dict": high_accuracy_eval_job_dict,
        }

        if Path("results", base_name, "high_accuracy_evaluation").exists():
            found_structures = list(
                Path("results", base_name, "high_accuracy_evaluation").glob(
                    f"batch_*/{ASE_OUTPUT_PREFIX}_*/{high_accuracy_eval_job_dict['name']}.xyz"
                )
            )
            if len(found_structures) >= len(structures) + start_index:
                logger.info(
                    "Found %d structures from previous high accuracy evaluation. "
                    "Skipping remote submission and reusing these structures.",
                    len(found_structures),
                )

                atoms_list = [read(p, format="extxyz") for p in found_structures]
                return atoms_list

            elif len(found_structures) > 0:
                logger.info(
                    "Found %d structures from previous high accuracy evaluation. "
                    "These will be reused; the rest will be submitted as new remote jobs.",
                    len(found_structures),
                )
                structures = structures[len(found_structures) + start_index :]
            else:
                logger.info(
                    "No previous results found. Submitting all %d structures.",
                    len(structures),
                )

        current_batches = sum(
            1
            for _ in Path("results", base_name, "high_accuracy_evaluation").glob(
                "batch_*"
            )
        )

        logger.info(
            "Structures to process: %d (existing batch dirs: %d)",
            len(structures),
            current_batches,
        )

        # All remaining structures are submitted in one call, sharing one
        # RemoteJobExecutor pool bounded by max_concurrent_jobs -- no more
        # chunking by max_batch_size, and GO/SP structures share the same
        # queue (distinguished only by which function each job runs and a
        # go_/sp_ job-name prefix) instead of running as separate batches.
        if structures:
            if allow_relaxation:
                needs_go = any(
                    atom.info.get("needs_relaxation") is True for atom in structures
                )
                submit_job_dict = high_accuracy_eval_job_dict
                if needs_go:
                    go_max_time = high_accuracy_eval_job_dict.get(
                        "max_go_time", high_accuracy_eval_job_dict["max_time"]
                    )
                    submit_job_dict = copy.deepcopy(high_accuracy_eval_job_dict)
                    submit_job_dict["max_time"] = go_max_time

                per_structure_function = [
                    run_go if atom.info.get("needs_relaxation") is True else run_sp
                    for atom in structures
                ]
                n_go = sum(fn is run_go for fn in per_structure_function)
                logger.info(
                    "Submitting batch %d: %d GO + %d SP structures (shared queue)",
                    current_batches,
                    n_go,
                    len(structures) - n_go,
                )
                ase_remote_submitter(
                    remote_info=get_remote_info(submit_job_dict, input_files=[]),
                    base_name=base_name,
                    input_atoms_list=structures,
                    per_structure_function=per_structure_function,
                    batch=current_batches,
                    function_kwargs=function_kwargs,
                )
            else:
                logger.info(
                    "Submitting batch %d (%d structures)",
                    current_batches,
                    len(structures),
                )
                ase_remote_submitter(
                    remote_info=get_remote_info(
                        high_accuracy_eval_job_dict, input_files=[]
                    ),
                    base_name=base_name,
                    input_atoms_list=structures,
                    function=run_sp,
                    batch=current_batches,
                    function_kwargs=function_kwargs,
                )

        high_accuracy_structures = []
        output_name = self.jobs_dict["high_accuracy_evaluation"]["name"]
        directory_list = list(
            Path("results", base_name).glob(
                f"{output_name}/batch_*/{ASE_OUTPUT_PREFIX}_*"
            )
        )
        for directory in directory_list:
            completed_file = Path(directory, f"{output_name}.xyz")
            structure = None
            if completed_file.exists():
                structure = read(completed_file, format="extxyz")
            if structure is not None:
                high_accuracy_structures.append(structure)

        sentinel_results.parent.mkdir(parents=True, exist_ok=True)
        write(sentinel_results, high_accuracy_structures, format="extxyz")
        self._mark_phase_done(base_name, "high_accuracy_eval")
        return high_accuracy_structures
