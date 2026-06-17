from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import read, write

from alomancy.analysis.plotting import mae_al_loop_plot
from alomancy.database.global_database import GlobalDatabase, _DEFAULT_DEDUP_CONFIG_TYPES
from alomancy.initialize.initialization_structure_list import (
    create_initialization_atoms_list,
)
from alomancy.utils.clean_structures import clean_structures
from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled
from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train


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
        start_loop: int = 0,
        plots: bool = True,
        seed: int = 803,
        db_path: str = "results/global_database",
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

    def run(self, **kwargs) -> None:
        """
        Run the active learning workflow.

        This method defines the core AL loop and calls the abstract methods
        that must be implemented by subclasses.
        """
        train_xyzs, test_xyzs = self.initialize_training_set("initialization", **kwargs)

        extra_datasets = self.jobs_dict["initialization"].get("extra_datasets") or []
        for extra_dataset in extra_datasets:
            train_xyzs, test_xyzs = self._add_extra_dataset(
                extra_dataset=extra_dataset,
                train_xyzs=train_xyzs,
                test_xyzs=test_xyzs,
            )

        print(f"Initialized training set with {len(train_xyzs)} structures.")

        for loop in range(self.start_loop, self.number_of_al_loops):
            base_name = f"al_loop_{loop}"
            workdir = Path(f"results/{base_name}")

            try:
                workdir.mkdir(exist_ok=True, parents=True)
            except OSError as e:
                print(f"Warning: Could not create directory {workdir}: {e}")

            train_file = Path(workdir, "train_set.xyz")
            test_file = Path(workdir, "test_set.xyz")

            try:
                write(train_file, train_xyzs, format="extxyz")
                write(test_file, test_xyzs, format="extxyz")
            except OSError as e:
                if "test" not in str(e).lower():
                    raise
                print(f"Warning: Could not write files (test environment): {e}")

            if self.verbose > 0:
                print(f"Starting AL loop {loop}")
                print(f"  Training set size: {len(train_xyzs)}")
                print(f"  Test set size: {len(test_xyzs)}")

            evaluation_results = self.train_mlip(
                base_name, self.jobs_dict["mlip_committee"], **kwargs
            )

            if self.verbose > 0:
                print(f"AL Loop {loop} evaluation results: \n{evaluation_results}")

            if self.plots:
                mae_al_loop_plot(evaluation_results, self.jobs_dict["mlip_committee"])

            generated_structures = self.generate_structures(
                base_name, self.jobs_dict, train_xyzs, **kwargs
            )

            new_training_data = self.high_accuracy_evaluation(
                base_name,
                self.jobs_dict["high_accuracy_evaluation"],
                generated_structures,
                **kwargs,
            )
            print(
                f"High-accuracy evaluation completed for {len(new_training_data)} structures."
            )

            new_training_data = clean_structures(
                new_training_data,
                config_type=f"al_loop_{loop}",
                override_config_type=True,
                already_computed=True,
            )

            # Add evaluated AL loop structures to the global DB
            self.db.add_structures(new_training_data, skip_duplicates=False)

            new_train_data, new_test_data = split_atoms_list_into_test_and_train(
                new_training_data,
                test_fraction=self.jobs_dict["initialization"]["test_to_train_ratio"],
                seed=self.seed,
            )

            train_xyzs += new_train_data
            test_xyzs += new_test_data

            if self.verbose > 0:
                print(
                    f"Completed AL loop {loop}, retraining with {len(train_xyzs)} structures."
                )

    def _add_extra_dataset(
        self,
        extra_dataset: str,
        train_xyzs: list[Atoms],
        test_xyzs: list[Atoms],
    ) -> tuple[list[Atoms], list[Atoms]]:
        """
        Load an extra dataset and add only structures not already in the DB.

        IsolatedAtom and init_MP structures are deduped by (config_type, formula);
        all other config_types are always added.  After dedup, the new structures
        are appended to the in-memory train/test lists via the existing helper.
        """
        all_atoms: list[Atoms] = read(extra_dataset, ":", format="extxyz")
        if isinstance(all_atoms, Atoms):
            all_atoms = [all_atoms]

        existing_keys = self.db._get_config_type_formula_set()
        dedup_config_types = _DEFAULT_DEDUP_CONFIG_TYPES

        new_atoms: list[Atoms] = []
        for atoms in all_atoms:
            ct = atoms.info.get("config_type", "")
            formula = atoms.get_chemical_formula()
            key = (ct, formula)
            if ct in dedup_config_types and key in existing_keys:
                if self.verbose > 0:
                    print(
                        f"Skipping duplicate {ct} structure ({formula}) "
                        f"already in global database."
                    )
                continue
            new_atoms.append(atoms)
            existing_keys.add(key)

        if not new_atoms:
            print(
                f"Extra dataset {extra_dataset}: all structures already in global DB, skipping."
            )
            return train_xyzs, test_xyzs

        skipped = len(all_atoms) - len(new_atoms)
        if skipped:
            print(
                f"Extra dataset {extra_dataset}: skipped {skipped} duplicate structure(s); "
                f"adding {len(new_atoms)} new structure(s)."
            )

        # Add the genuinely new structures to the DB
        self.db.add_structures(new_atoms, skip_duplicates=False)

        # Extend in-memory train/test sets directly from new_atoms — do NOT
        # re-read the file, which would re-introduce the structures we just deduped.
        fall_back_config_type = f"extra_dataset_{Path(extra_dataset).name}"
        cleaned = clean_structures(
            new_atoms,
            fall_back_config_type,
            override_config_type=False,
            already_computed=True,
        )
        isolated = [a for a in cleaned if a.info.get("config_type") == "IsolatedAtom"]
        eligible = [a for a in cleaned if a.info.get("config_type") != "IsolatedAtom"]
        new_train, new_test = split_atoms_list_into_test_and_train(
            eligible,
            self.jobs_dict["initialization"]["test_to_train_ratio"],
            self.seed,
        )
        train_xyzs.extend(new_train + isolated)
        test_xyzs.extend(new_test)
        print(
            f"Added {len(new_train)} structures from {extra_dataset} to training set "
            f"and {len(new_test)} to test set."
        )
        return train_xyzs, test_xyzs

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
            print(
                f"WARNING: Only {len(train_xyzs)} structure(s) found in the training set. "
                f"More than one structure is recommended to start active learning. "
                f"Consider adding more structures to {self.initial_train_file_path}."
            )
        if len(test_xyzs) <= 1:
            print(
                f"WARNING: Only {len(test_xyzs)} structure(s) found in the test set. "
                f"More than one structure is recommended to start active learning. "
                f"Consider adding more structures to {self.initial_test_file_path}."
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
    def train_mlip(
        self, base_name: str, mlip_committee_job_dict: dict, **kwargs
    ) -> pd.DataFrame:
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
