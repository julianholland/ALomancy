from abc import ABC, abstractmethod
from multiprocessing import dummy
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from ase import Atoms
from ase.io import read, write
from alomancy.configs.config_dictionaries import load_dictionaries
from alomancy.utils.test_train_manager import add_new_training_data


class BaseActiveLearningWorkflow(ABC):
    """
    Abstract base class for active learning workflows.
    
    This class provides the core AL loop structure while requiring
    subclasses to implement the specific methods for structure generation,
    high-accuracy evaluation, MLIP training, and evaluation.
    """
    
    def __init__(
        self,
        initial_train_file_path: str,
        initial_test_file_path: str,
        number_of_al_loops: int = 5,
        verbose: int = 0,
        start_loop: int = 0,
    ):
        self.initial_train_file = Path(initial_train_file_path)
        self.initial_test_file = Path(initial_test_file_path)
        self.number_of_al_loops = number_of_al_loops
        self.verbose = verbose
        self.start_loop = start_loop
        self.jobs_dict = load_dictionaries()

    def run(self, **kwargs) -> None:
        """
        Run the active learning workflow.
        
        This method defines the core AL loop and calls the abstract methods
        that must be implemented by subclasses.
        """
        
        def load_initial_train_test_sets(dummy_run: bool = False) -> Tuple[List[Atoms], List[Atoms]]:
            train_xyzs = [
                atoms
                for atoms in read(self.initial_train_file, ":")
                if isinstance(atoms, Atoms)
            ]
            test_xyzs = [
                atoms
                for atoms in read(self.initial_test_file, ":")
                if isinstance(atoms, Atoms)
            ]

            assert len(train_xyzs) > 1, "More than one training structure required."
            assert len(test_xyzs) > 1, "More than one test structure required."

            if dummy_run:
                train_xyzs = train_xyzs[:500]
                test_xyzs = test_xyzs[:200]

            return train_xyzs, test_xyzs

        train_xyzs, test_xyzs = load_initial_train_test_sets(dummy_run=True)

        for loop in range(self.start_loop, self.number_of_al_loops):
            base_name = f"al_loop_{loop}"
            workdir = Path(f"results/{base_name}")
            workdir.mkdir(exist_ok=True, parents=True)

            train_file = str(workdir / "train_set.xyz")
            test_file = str(workdir / "test_set.xyz")

            # Write current training and test sets
            write(train_file, train_xyzs, format="extxyz")
            write(test_file, test_xyzs, format="extxyz")

            if self.verbose > 0:
                print(f"Starting AL loop {loop}")
                print(f"  Training set size: {len(train_xyzs)}")
                print(f"  Test set size: {len(test_xyzs)}")

            # Core AL loop steps - these methods must be implemented by subclasses
            self.train_mlip(base_name, self.jobs_dict['mlip_committee'], train_xyzs, **kwargs)
            
            # evaluation_results = self.evaluate_mlip(base_name, self.jobs_dict['mace_committee'], test_xyzs, **kwargs)
            # if self.verbose > 0:
            #     print(f"AL Loop {loop} evaluation results: {evaluation_results}")

            # generated_structures = self.generate_structures(base_name, self.jobs_dict['generate_structures'], train_xyzs, **kwargs)

            # new_training_data = self.high_accuracy_evaluation(
            #     base_name, self.jobs_dict['high_accuracy_evaluation'], generated_structures, **kwargs
            # )

            # train_xyzs = add_new_training_data(
            #     base_name, self.jobs_dict['high_accuracy_evaluation'], train_xyzs, new_training_data
            # )

            # if self.verbose > 0:
            #     print(f"Completed AL loop {loop}, retraining with {len(train_xyzs)} structures.")


    # @abstractmethod
    # def generate_structures(self, base_name: str, structure_generation_job_dict: dict, train_data: List[Atoms], **kwargs) -> List[Atoms]:
    #     """
    #     Generate structures for active learning selection.
        
    #     Parameters
    #     ----------
    #     base_name : str
    #         Base name for this AL loop
    #     structure_generation_job_dict : dict
    #         Dictionary containing job name and HPC parameters for structure generation
    #     train_data : List[Atoms]
    #         Current training data
    #     **kwargs
    #         Additional keyword arguments
            
    #     Returns
    #     -------
    #     List[Atoms]
    #         Generated structures for high-accuracy evaluation
    #     """
    #     pass

    # @abstractmethod
    # def high_accuracy_evaluation(
    #     self, 
    #     base_name: str, 
    #     high_accuracy_eval_job_dict: dict,
    #     structures: List[Atoms], 
    #     **kwargs
    # ) -> List[Atoms]:
    #     """
    #     Run high-accuracy calculations on selected structures.
        
    #     Parameters
    #     ----------
    #     base_name : str
    #         Base name for this AL loop
    #     high_accuracy_eval_job_dict : dict
    #         Dictionary containing job name and HPC parameters for high-accuracy evaluation
    #     structures : List[Atoms]
    #         Structures to evaluate with high-accuracy method
    #     **kwargs
    #         Additional keyword arguments
            
    #     Returns
    #     -------
    #     List[Atoms]
    #         Structures with high-accuracy results (energy, forces, etc.)
    #     """
    #     pass

    @abstractmethod
    def train_mlip(self, base_name: str, mlip_committee_job_dict: dict, train_data: List[Atoms], **kwargs) -> Optional[str]:
        """
        Train machine learning interatomic potential.
        
        Parameters
        ----------
        base_name : str
            Base name for this AL loop
        mlip_committee_job_dict : dict
            Dictionary containing job name and HPC parameters for MLIP training
        train_data : List[Atoms]
            Training data for MLIP
        **kwargs
            Additional keyword arguments
            
        Returns
        -------
        Optional[str]
            Path to trained model file, if applicable
        """
        pass

    # @abstractmethod
    # def evaluate_mlip(self, base_name: str, mlip_committee_job_dict: dict, test_data: List[Atoms], **kwargs) -> Dict[str, Any]:
    #     """
    #     Evaluate MLIP model on test data.
        
    #     Parameters
    #     ----------
    #     base_name : str
    #         Base name for this AL loop
    #     mlip_committee_job_dict : dict
    #         Dictionary containing job name and HPC parameters for MLIP evaluation
    #     test_data : List[Atoms]
    #         Test data for evaluation
    #     **kwargs
    #         Additional keyword arguments
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Evaluation metrics (RMSE, MAE, etc.)
    #     """
    #     pass

    