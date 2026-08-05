import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
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
from alomancy.version import __version__, __version_tuple__

logger = logging.getLogger(__name__)

_PHASE_LABELS: dict[str, str] = {
    "initialization": "Initialisation",
    "mlip_committee": "MLIP Committee Trainer",
    "structure_generation": "Structure Generation",
    "high_accuracy_evaluation": "High-Accuracy Evaluation",
}


def _flatten_settings(
    d: dict, prefix: str = "", max_depth: int = 2
) -> list[tuple[str, object]]:
    """Flatten a job-dict phase into (dotted.key, value) pairs.

    Skips `hpc` (reported separately in the HPC summary) and `name` (used as
    the section heading). Nested dicts (e.g. `creation_kwargs`,
    `mace_fit_kwargs`, `qe_input_kwargs`) are walked up to `max_depth` levels
    so calculator/model-specific settings surface without hardcoding any
    particular module's key names here.
    """
    items: list[tuple[str, object]] = []
    for key, value in d.items():
        if key in ("hpc", "name"):
            continue
        full_key = f"{prefix}{key}"
        if isinstance(value, dict) and max_depth > 0:
            items.extend(
                _flatten_settings(value, prefix=f"{full_key}.", max_depth=max_depth - 1)
            )
        else:
            items.append((full_key, value))
    return items


def _fetch_latest_pypi_version(
    package: str = "alomancy", timeout: float = 3.0
) -> str | None:
    """Best-effort lookup of the latest published version on PyPI.

    Returns None (rather than raising) on any network/parse failure — HPC
    compute nodes commonly have no outbound internet access, and a version
    check must never block a run over that. Also skipped outright under
    ALOMANCY_TEST_MODE/ALOMANCY_MOCK_EXTERNAL (set autouse for the whole
    test suite, see tests/conftest.py) so tests never make a real PyPI call.
    """
    if (
        os.getenv("ALOMANCY_TEST_MODE") == "1"
        or os.getenv("ALOMANCY_MOCK_EXTERNAL") == "1"
    ):
        return None
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{package}/json", timeout=timeout
        ) as response:
            data = json.loads(response.read())
        return data["info"]["version"]
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
    ) as exc:
        logger.debug("Could not check PyPI for latest %s version: %s", package, exc)
        return None


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

    def display_workflow_summary(self) -> None:
        """
        Display a summary of the active learning workflow configuration:
        one section per configured phase (name + major settings, flattened
        from its job dict) and a table of the HPC profiles in use and which
        job types run on each.
        """
        lines: list[str] = [
            "",
            "=" * 70,
            f"ALomancy Workflow Summary (v{__version__})",
            "=" * 70,
        ]

        hpc_usage: dict[str, dict] = {}

        for phase, heading in _PHASE_LABELS.items():
            phase_dict = self.jobs_dict.get(phase)
            if not phase_dict:
                continue

            lines.append("")
            lines.append(f"--- {heading} ({phase_dict.get('name', phase)}) ---")
            for key, value in _flatten_settings(phase_dict):
                lines.append(f"  {key}: {value}")

            hpc = phase_dict.get("hpc")
            if hpc:
                name = (
                    hpc.get("hpc_name", "<unnamed>")
                    if isinstance(hpc, dict)
                    else str(hpc)
                )
                entry = hpc_usage.setdefault(
                    name,
                    {"profile": hpc if isinstance(hpc, dict) else {}, "phases": []},
                )
                entry["phases"].append(heading)

        lines.append("")
        lines.append("--- HPC Profiles ---")
        if hpc_usage:
            rows = []
            for name, entry in hpc_usage.items():
                profile = entry["profile"]
                node_info = profile.get("node_info", {})
                rows.append(
                    {
                        "hpc_name": name,
                        "alomancy_version": __version__,
                        "gpu": profile.get("gpu", "?"),
                        "partitions": ", ".join(profile.get("partitions", []) or [])
                        or "?",
                        "ranks_per_node": node_info.get("ranks_per_node", "?"),
                        "max_mem_per_node": node_info.get("max_mem_per_node", "?"),
                        "job_types": "\n".join(entry["phases"]),
                    }
                )
            with pl.Config(
                fmt_str_lengths=200, tbl_width_chars=200, tbl_hide_dataframe_shape=True
            ):
                lines.append(str(pl.DataFrame(rows)))
        else:
            lines.append("  No HPC profiles configured.")

        lines.append("=" * 70)
        logger.info("\n".join(lines))

    def pre_run_checks(self) -> None:
        """
        Display the workflow summary, then run pre-flight checks:
        currently just the installed-vs-latest-published alomancy version.
        Warns if behind by a minor release; raises if behind by a major
        release (breaking changes are likely). Silently skipped if PyPI
        can't be reached (e.g. no internet on an HPC compute node).
        """
        self.display_workflow_summary()

        latest_version = _fetch_latest_pypi_version()
        if latest_version is None:
            return

        current_major, current_minor = __version_tuple__[0], __version_tuple__[1]
        try:
            latest_tuple = tuple(int(part) for part in latest_version.split(".")[:3])
        except ValueError:
            logger.debug(
                "Could not parse latest PyPI version %r; skipping version check.",
                latest_version,
            )
            return
        latest_major, latest_minor = latest_tuple[0], latest_tuple[1]

        if latest_major > current_major:
            raise RuntimeError(
                f"Installed alomancy version {__version__} is a major release "
                f"behind the latest available version {latest_version}. "
                "Breaking changes are likely — please upgrade "
                "(`pip install -U alomancy`) before running."
            )
        if latest_major == current_major and latest_minor > current_minor:
            logger.warning(
                "Installed alomancy version %s is a minor release behind the "
                "latest available version %s. Consider upgrading "
                "(`pip install -U alomancy`).",
                __version__,
                latest_version,
            )

    def run(self, **kwargs) -> None:
        """
        Run the active learning workflow.

        This method defines the core AL loop and calls the abstract methods
        that must be implemented by subclasses.
        """
        self.pre_run_checks()

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
            self.store_mlip_predictions(loop, base_name, self.jobs_dict)

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
                    base_name,
                    self.jobs_dict["mlip_committee"],
                    self.seed,
                    plots_dir,
                    db=self.db,
                    loop_idx=loop,
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

    def store_mlip_predictions(  # noqa: B027
        self, loop_idx: int, base_name: str, job_dict: dict
    ) -> None:
        """Store per-structure MLIP predictions in the DB after training.

        Default is a no-op. Override in subclasses that support prediction storage.
        """

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
