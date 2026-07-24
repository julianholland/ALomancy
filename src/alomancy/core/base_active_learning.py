import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import read, write

from alomancy.analysis.plotting import mae_al_loop_plot
from alomancy.database.global_database import GlobalDatabase
from alomancy.utils.clean_structures import clean_structures
from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled
from alomancy.utils.logging_config import setup_logging
from alomancy.utils.remove_high_force_structures import (
    remove_high_force_structures_from_partition,
)
from alomancy.utils.remove_redundancy import remove_redundancy_from_partition
from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train

logger = logging.getLogger(__name__)


class BaseActiveLearningWorkflow(ABC):
    """
    Abstract base class for active learning workflows.

    This class provides the core AL loop structure while requiring
    subclasses to implement the specific methods for structure generation,
    high-accuracy evaluation, MLIP training, and evaluation.

    Subclasses must implement the following abstract methods:
    - `initialize_training_set`
    - `high_accuracy_evaluation`
    - `train_mlip`
    - `generate_structures`
    """

    def __init__(
        self,
        initial_train_file_path: str,
        initial_test_file_path: str,
        jobs_dict: dict,
        number_of_al_loops: int = 5,
        verbose: int = 0,
        log_file: str | None = "results/alomancy.log",
        start_loop: int = 0,
        plots: bool = True,
        seed: int = 803,
        db_path: str = "results/global_database",
        remove_redundancy: bool = True,
        high_force_threshold: float | None = 100.0,
        skip_initialization: bool = False,
    ):
        self.initial_train_file_path = Path(initial_train_file_path)
        self.initial_test_file_path = Path(initial_test_file_path)
        self.jobs_dict = jobs_dict
        self.number_of_al_loops = number_of_al_loops
        self.verbose = verbose
        self.start_loop = start_loop
        self.plots = plots
        self.seed = seed
        self.db = GlobalDatabase(db_path)
        self.remove_redundancy = remove_redundancy
        self.high_force_threshold = high_force_threshold
        self.skip_initialization = skip_initialization
        self.log_file = log_file
        setup_logging(verbose=verbose, log_file=log_file)

    def _phase_done(self, base_name: str, phase: str) -> bool:
        return Path("results", base_name, f"{phase}.done").exists()

    def _mark_phase_done(self, base_name: str, phase: str) -> None:
        sentinel = Path("results", base_name, f"{phase}.done")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(datetime.now().isoformat() + "\n")
        logger.debug("Phase %s marked complete for %s.", phase, base_name)

    def _last_complete_loop(self) -> int:
        """Return index of last consecutive completed loop (has loop.done), or -1."""
        last = -1
        for loop in range(self.number_of_al_loops):
            if (Path("results", f"al_loop_{loop}") / "loop.done").exists():
                last = loop
            else:
                break
        return last

    def run(self, **kwargs) -> None:
        """
        Run the active learning workflow.

        This method defines the core AL loop and calls the abstract methods
        that must be implemented by subclasses.
        """
        last_complete = self._last_complete_loop()

        if last_complete >= 0:
            train_xyzs = self.db.get_train_atoms()
            test_xyzs = self.db.get_test_atoms()
            effective_start = max(self.start_loop, last_complete + 1)
            logger.info(
                "Resuming from loop %d (%d train / %d test from DB).",
                effective_start,
                len(train_xyzs),
                len(test_xyzs),
            )
        elif self.skip_initialization:
            train_xyzs = self.db.get_train_atoms()
            test_xyzs = self.db.get_test_atoms()
            effective_start = self.start_loop
            logger.info(
                "skip_initialization=True: loading %d train / %d test from DB, "
                "starting at loop %d.",
                len(train_xyzs),
                len(test_xyzs),
                effective_start,
            )
        else:
            train_xyzs, test_xyzs = self.initialize_training_set(
                "initialization", **kwargs
            )
            n_tagged = self.db.update_splits_post_hoc(train_xyzs, test_xyzs)
            logger.info(
                "Initialized training set with %d structures; tagged %d in DB.",
                len(train_xyzs),
                n_tagged,
            )
            effective_start = self.start_loop

        if self.remove_redundancy:
            remove_redundancy_from_partition(
                self.db,
                config_list=self.jobs_dict["initialization"]["test_config_types"]
                + ["high_sd"],
            )

        if self.high_force_threshold is not None:
            remove_high_force_structures_from_partition(
                self.db, force_threshold=self.high_force_threshold
            )

        for loop in range(effective_start, self.number_of_al_loops):
            # Derive current train/test from DB at the start of each iteration
            # so any redundancy flags from the previous loop are reflected.
            train_xyzs = self.db.get_train_atoms()
            test_xyzs = self.db.get_test_atoms()

            base_name = f"al_loop_{loop}"
            workdir = Path(f"results/{base_name}")

            try:
                workdir.mkdir(exist_ok=True, parents=True)
            except OSError as e:
                logger.warning("Could not create directory %s: %s", workdir, e)

            train_file = Path(workdir, "train_set.xyz")
            test_file = Path(workdir, "test_set.xyz")

            try:
                write(train_file, train_xyzs, format="extxyz")
                write(test_file, test_xyzs, format="extxyz")
            except OSError as e:
                if "test" not in str(e).lower():
                    raise
                logger.warning("Could not write files (test environment): %s", e)

            logger.debug("Starting AL loop %d", loop)
            logger.debug("  Training set size: %d", len(train_xyzs))
            logger.debug("  Test set size: %d", len(test_xyzs))

            evaluation_results = self.train_mlip(base_name, self.jobs_dict, **kwargs)

            logger.debug("AL Loop %d evaluation results:\n%s", loop, evaluation_results)

            if self.plots:
                plots_dir = Path("results", "current_plots")
                plots_dir.mkdir(exist_ok=True, parents=True)
                mae_al_loop_plot(
                    evaluation_results,
                    self.jobs_dict["mlip_committee"],
                    directory=plots_dir,
                )
                from alomancy.analysis.mlip_plots import (
                    plot_dft_vs_model,
                    plot_training_curves,
                )

                plot_training_curves(
                    base_name, self.jobs_dict["mlip_committee"], self.seed, plots_dir
                )
                plot_dft_vs_model(
                    base_name, self.jobs_dict["mlip_committee"], self.seed, plots_dir
                )

            generated_structures = self.generate_structures(
                base_name, self.jobs_dict, train_xyzs, **kwargs
            )

            new_training_data = self.high_accuracy_evaluation(
                base_name,
                self.jobs_dict["high_accuracy_evaluation"],
                generated_structures,
                **kwargs,
            )
            logger.info(
                "High-accuracy evaluation completed for %d structures.",
                len(new_training_data),
            )

            new_training_data = clean_structures(
                new_training_data,
                config_type="high_sd",
                override_config_type=True,
                already_computed=True,
                extra_metadata={"al_loop": loop},
            )

            new_train_data, new_test_data = split_atoms_list_into_test_and_train(
                new_training_data,
                test_fraction=self.jobs_dict["initialization"]["test_to_train_ratio"],
                seed=self.seed,
            )

            # Add AL loop structures to DB with split tags; DB is the restart source.
            self.db.add_structures(new_train_data, split="train", skip_duplicates=False)
            self.db.add_structures(new_test_data, split="test", skip_duplicates=False)

            if self.remove_redundancy:
                remove_redundancy_from_partition(
                    self.db,
                    config_list=self.jobs_dict["initialization"]["test_config_types"]
                    + ["high_sd"],
                )

            if self.high_force_threshold is not None:
                remove_high_force_structures_from_partition(
                    self.db,
                    force_threshold=self.high_force_threshold,
                )

            self._mark_phase_done(base_name, "loop")

            logger.debug(
                "Completed AL loop %d, retraining with %d structures.",
                loop,
                len(train_xyzs),
            )

            if self.plots and self.log_file is not None:
                from alomancy.analysis.timing_plots import timing_plots

                timing_plots(self.log_file, Path("results", "current_plots"))

    def _seed_db_from_extra_dataset(self, extra_dataset: str) -> None:
        """
        Read an extra dataset file and add its structures to the global DB.

        Called before initialize_training_set so compute_initialization_needs
        can account for already-provided structures when deciding what still
        needs to be generated.

        IsolatedAtom and init_MP are deduplicated by (config_type, formula).
        All other config_types (dimers, trimers, amorphous, etc.) are added
        without exact dedup — they are counted by compute_initialization_needs
        and the existing count reduces the generation target accordingly.
        """
        all_atoms: list[Atoms] = read(extra_dataset, ":", format="extxyz")
        if isinstance(all_atoms, Atoms):
            all_atoms = [all_atoms]

        added = self.db.add_structures(all_atoms, skip_duplicates=True)
        skipped = len(all_atoms) - added
        msg = f"Seeded DB from {extra_dataset}: {added} structure(s) added"
        if skipped:
            msg += f", {skipped} duplicate(s) skipped"
        logger.info("%s.", msg)

    def load_initial_train_test_sets(
        self,
        dummy_run: bool = False,
    ) -> tuple[list[Atoms], list[Atoms]]:
        train_xyzs = read_atoms_file_if_enabled(True, self.initial_train_file_path)
        test_xyzs = read_atoms_file_if_enabled(True, self.initial_test_file_path)

        if train_xyzs is None or test_xyzs is None:
            raise FileNotFoundError(
                "Initial training or test file not found. Please provide valid file paths."
            )

        if len(train_xyzs) <= 1:
            logger.warning(
                "Only %d structure(s) found in the training set. "
                "More than one structure is recommended to start active learning. "
                "Consider adding more structures to %s.",
                len(train_xyzs),
                self.initial_train_file_path,
            )
        if len(test_xyzs) <= 1:
            logger.warning(
                "Only %d structure(s) found in the test set. "
                "More than one structure is recommended to start active learning. "
                "Consider adding more structures to %s.",
                len(test_xyzs),
                self.initial_test_file_path,
            )

        if dummy_run:
            train_xyzs = train_xyzs[:500]
            test_xyzs = test_xyzs[:200]

        return train_xyzs, test_xyzs

    def process_structure(self, structure: Atoms) -> Atoms:
        new_structure = structure.copy()
        new_structure.info["REF_energy"] = structure.get_potential_energy()
        new_structure.arrays["REF_forces"] = structure.get_forces()
        return new_structure

    @abstractmethod
    def initialize_training_set(
        self, base_name: str, **kwargs
    ) -> tuple[list[Atoms], list[Atoms]]:
        """
        Initialize the training and test sets.

        Returns
        -------
        Tuple[List[Atoms], List[Atoms]]
            Initial training and test structures.
        """
        pass

    @abstractmethod
    def high_accuracy_evaluation(
        self,
        base_name: str,
        high_accuracy_eval_job_dict: dict,
        structures: list[Atoms],
        **kwargs,
    ) -> list[Atoms]:
        """
        Run high-accuracy calculations on selected structures.

        Returns
        -------
        List[Atoms]
            Structures with high-accuracy results (energy, forces, etc.)
        """
        pass

    @abstractmethod
    def train_mlip(self, base_name: str, job_dict: dict, **kwargs) -> pd.DataFrame:
        """
        Train machine learning interatomic potential.

        Returns
        -------
        pd.DataFrame
            Evaluation metrics (MAE, RMSE, etc.) for the trained committee.
        """
        pass

    @abstractmethod
    def generate_structures(
        self,
        base_name: str,
        structure_generation_job_dict: dict,
        train_data: list[Atoms],
        **kwargs,
    ) -> list[Atoms]:
        """
        Generate structures for active learning selection.

        Returns
        -------
        List[Atoms]
            Generated structures for high-accuracy evaluation.
        """
        pass
