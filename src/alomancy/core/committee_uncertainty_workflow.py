"""CommitteeUncertaintyWorkflow: the concrete AL skeleton for the modular
architecture (see TODO.md's Refactor section / the associated plan).

Replaces the abstract BaseActiveLearningWorkflow + its ActiveLearningStandardMACE
subclass (both still in place, unchanged, and still what production runs
use until this skeleton is explicitly cut over to). Subclassing goes away
entirely: this is one concrete class that resolves its trainer/structure
generator/DFT evaluator/initialiser from config via the shared registry,
instead of a different Python subclass per MACE+QE+MD-vs-MACE+VASP+EZGA
combination.

Owns:
- The shared submit_n-driven committee training loop (N calls to the
  trainer's train(), aggregated, quality-gated, backfilled, hard-failing
  under 3 successful fits).
- The one-time shared train/valid/test split per loop, written to disk
  once and passed to every fit as file paths.
- The partial-aware restart mechanism around each module call
  (output_paths/read_existing_result).
- Committee force-std-dev scoring (resolving each member's calculator via
  the trainer registry, not a hardcoded MACECalculator) and post-generation
  high-uncertainty selection.
- Storing predictions in the GlobalDatabase and local checkpoint cleanup --
  local, post-sync, committee-shaped operations needing db/loop_idx/
  base_name, so skeleton-level rather than trainer-internal.
- Orchestrating initialiser -> evaluator (bootstrap structures need DFT
  labels before they can seed training) and calling the evaluator on
  AL-loop-generated structures identically.

Config schema (breaking, see the plan's "Config schema changes" section):
split-building parameters that used to live in `initialization`/
`mlip_committee` now live in `workflow` (test_config_types,
test_to_train_ratio, grouped_splits, valid_fraction, valid_config_types,
grouped_validation), alongside the existing `train_only`/`fixed_test`
flags and the new `skeleton` key. `mlip_committee` gains a `trainer` key
(defaults to "mace" if absent, for configs written before this existed).
"""

import hashlib
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from ase import Atoms
from ase.io import read, write

from alomancy.analysis.plotting import mae_al_loop_plot
from alomancy.configs.remote_info import get_remote_info
from alomancy.core.standard_active_learning import _read_mace_eval_predictions
from alomancy.database.global_database import (
    _DEFAULT_DEDUP_CONFIG_TYPES,
    GlobalDatabase,
)
from alomancy.high_accuracy_evaluation.high_accuracy_calc_interface import (
    high_accuracy_evaluation as _evaluator_orchestrate,
)
from alomancy.mlip.evaluation import check_quality_gate, read_evaluation
from alomancy.mlip.mace.get_mace_eval_info import select_best_committee_model
from alomancy.registry import resolve
from alomancy.remote_submission.executor import acquire_local_expyre_lock, submit_n
from alomancy.structure_generation.find_high_sd_structures import (
    find_high_sd_structures,
)
from alomancy.utils.clean_structures import (
    clean_structures,
    filter_structures_by_min_bond_distance,
)
from alomancy.utils.dataset_curation import (
    curate_database,
    geometry_digest,
    grouped_split,
    structure_domain,
    validate_policy,
)
from alomancy.utils.file_saving_and_parsing import read_atoms_file_if_enabled
from alomancy.utils.logging_config import setup_logging
from alomancy.utils.remote_ssh import (
    ensure_ssh_connectivity,
    get_alomancy_version_for_profile,
)
from alomancy.utils.remove_high_force_structures import (
    remove_high_force_structures_from_partition,
)
from alomancy.utils.remove_redundancy import remove_redundancy_from_partition
from alomancy.utils.seed_selection import filter_eligible_structures
from alomancy.utils.test_train_manager import split_atoms_list_into_test_and_train
from alomancy.version import __version__, __version_tuple__

logger = logging.getLogger(__name__)

_PHASE_LABELS: dict[str, str] = {
    "initialization": "Initialisation",
    "mlip_committee": "MLIP Committee Trainer",
    "structure_generation": "Structure Generation",
    "high_accuracy_evaluation": "High-Accuracy Evaluation",
}


def _needs_anything(needs: dict) -> bool:
    return bool(
        needs["isolated_atoms"]
        or needs["dimer_override"]
        or needs["trimer_override"]
        or needs["amorphous_override"] > 0
        or needs["mp_structures"]
    )


def _flatten_settings(
    d: dict, prefix: str = "", max_depth: int = 2
) -> list[tuple[str, object]]:
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


def _collect_hpc_profiles(jobs_dict: dict) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for phase in _PHASE_LABELS:
        phase_dict = jobs_dict.get(phase)
        if not phase_dict:
            continue
        hpc = phase_dict.get("hpc")
        if not isinstance(hpc, dict):
            continue
        name = hpc.get("hpc_name", "<unnamed>")
        profiles.setdefault(name, hpc)
    return profiles


def _fetch_latest_pypi_version(
    package: str = "alomancy", timeout: float = 3.0
) -> str | None:
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


def _flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0] * 3))


def _score_structures_with_member(
    structure_list: list[Atoms],
    model_path: str,
    trainer: str,
    trainer_config: dict,
) -> dict:
    """Remote worker: evaluate structure_list's forces/energy against ONE
    committee member's model, resolving the calculator via the trainer
    registry -- never a hardcoded MACECalculator -- see the architecture
    plan's committee-scoring decision. Returns
    {"forces": [...], "energies": [...]}, index-aligned with structure_list.
    """
    entry = resolve("mlip_trainer", trainer)
    calc = entry.get_calculator(model_path, trainer_config)
    forces = []
    energies = []
    for atoms in structure_list:
        atoms.calc = calc
        forces.append(_flatten_array_of_forces(atoms.get_forces()))
        energies.append(np.array(atoms.get_potential_energy()))
    return {"forces": forces, "energies": energies}


def _select_validation_split(
    all_training: list[Atoms],
    acceptable_configs: list[str],
    valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[Atoms], list[Atoms]]:
    """Carve the shared validation set from all_training. Only structures
    with config_type in acceptable_configs are eligible; the rest always
    stay in training. Returns (new_train_set, valid_set). Built ONCE per
    loop by the skeleton (decision 6) -- not re-derived per committee
    member -- and the same (train, valid) lists are then written to disk
    once and passed as file paths to every trainer.train() call.
    """
    eligible = [
        a for a in all_training if a.info.get("config_type") in acceptable_configs
    ]
    if not eligible:
        logger.warning(
            "No structures with config_type in %s found; skipping validation split.",
            acceptable_configs,
        )
        return all_training, []

    n_valid = int(np.floor(valid_fraction * len(eligible)))
    if n_valid == 0:
        logger.warning(
            "%.0f%% of %d eligible structure(s) rounds to 0; skipping validation split.",
            valid_fraction * 100,
            len(eligible),
        )
        return all_training, []

    chosen = rng.choice(len(eligible), size=n_valid, replace=False)
    valid_set = [eligible[i] for i in chosen]
    valid_ids = {id(a) for a in valid_set}
    new_train_set = [a for a in all_training if id(a) not in valid_ids]

    logger.info(
        "Validation split: %d valid, %d train (from %d total, %d eligible).",
        len(valid_set),
        len(new_train_set),
        len(all_training),
        len(eligible),
    )
    return new_train_set, valid_set


class CommitteeUncertaintyWorkflow:
    """Concrete AL skeleton: committee-uncertainty selection, resolving its
    trainer/structure-generator/DFT-evaluator/initialiser from config via
    the shared registry."""

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
        db: GlobalDatabase | None = None,
    ):
        self.initial_train_file_path = Path(initial_train_file_path)
        self.initial_test_file_path = Path(initial_test_file_path)
        self.jobs_dict = jobs_dict
        if jobs_dict.get("dataset_curation"):
            validate_policy(jobs_dict["dataset_curation"])
        self.number_of_al_loops = number_of_al_loops
        self.verbose = verbose
        self.start_loop = start_loop
        self.plots = plots
        self.seed = seed
        self.db = db if db is not None else GlobalDatabase(db_path)
        self.remove_redundancy = remove_redundancy
        self.high_force_threshold = high_force_threshold
        self.skip_initialization = skip_initialization
        self.log_file = log_file
        setup_logging(verbose=verbose, log_file=log_file)

    # -- Phase/loop bookkeeping (unchanged from BaseActiveLearningWorkflow) --

    def _phase_done(self, base_name: str, phase: str) -> bool:
        return Path("results", base_name, f"{phase}.done").exists()

    def _mark_phase_done(self, base_name: str, phase: str) -> None:
        sentinel = Path("results", base_name, f"{phase}.done")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(datetime.now().isoformat() + "\n")
        logger.debug("Phase %s marked complete for %s.", phase, base_name)

    def _last_complete_loop(self) -> int:
        last = -1
        for loop in range(self.number_of_al_loops):
            if (Path("results", f"al_loop_{loop}") / "loop.done").exists():
                last = loop
            else:
                break
        return last

    def display_workflow_summary(self) -> None:
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
                        "alomancy_version": get_alomancy_version_for_profile(profile)
                        or "?",
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
        acquire_local_expyre_lock()
        self.display_workflow_summary()
        ensure_ssh_connectivity(_collect_hpc_profiles(self.jobs_dict))

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

    def _seed_db_from_extra_dataset(self, extra_dataset: str) -> None:
        all_atoms: list[Atoms] = read(extra_dataset, ":", format="extxyz")
        if isinstance(all_atoms, Atoms):
            all_atoms = [all_atoms]

        digest = hashlib.sha256(Path(extra_dataset).read_bytes()).hexdigest()
        existing = {
            a.info.get("source_dataset_sha256") for a in self.db.get_all_as_atoms()
        }
        if digest in existing:
            logger.info(
                "Extra dataset %s already imported (sha256=%s)", extra_dataset, digest
            )
            return
        reset_splits = self.jobs_dict["initialization"].get("reset_extra_splits", False)
        for atoms in all_atoms:
            atoms.info["source_dataset_sha256"] = digest
            atoms.info.setdefault("domain", structure_domain(atoms))
            if reset_splits:
                for key in ("split", "global_db_id", "is_duplicate", "is_high_force"):
                    atoms.info.pop(key, None)
                for key in list(atoms.info):
                    if key.startswith(("model_", "mace_")):
                        del atoms.info[key]
        added = self.db.add_structures(all_atoms, skip_duplicates=True)
        skipped = len(all_atoms) - added
        msg = f"Seeded DB from {extra_dataset}: {added} structure(s) added"
        if skipped:
            msg += f", {skipped} duplicate(s) skipped"
        logger.info("%s.", msg)

    def load_initial_train_test_sets(
        self, dummy_run: bool = False
    ) -> tuple[list[Atoms], list[Atoms]]:
        train_xyzs = read_atoms_file_if_enabled(True, self.initial_train_file_path)
        test_xyzs = read_atoms_file_if_enabled(True, self.initial_test_file_path)
        if train_xyzs is None or test_xyzs is None:
            raise FileNotFoundError(
                "Initial training or test file not found. Please provide valid file paths."
            )
        if dummy_run:
            train_xyzs = train_xyzs[:500]
            test_xyzs = test_xyzs[:200]
        return train_xyzs, test_xyzs

    # -- Initialiser -> evaluator orchestration (decision 10) --

    def _initialize_training_set(
        self, base_name: str
    ) -> tuple[list[Atoms], list[Atoms]]:
        work_dir = Path("results", base_name)
        work_dir.mkdir(exist_ok=True, parents=True)
        init_config = self.jobs_dict["initialization"]
        workflow_config = self.jobs_dict.get("workflow", {})

        if (
            self.initial_train_file_path.exists()
            and self.initial_test_file_path.exists()
        ):
            train_xyzs, test_xyzs = self.load_initial_train_test_sets()
            logger.info(
                "Initial train and test sets loaded from files: %s, %s",
                self.initial_train_file_path,
                self.initial_test_file_path,
            )
            write(
                work_dir / self.initial_train_file_path.name,
                train_xyzs,
                format="extxyz",
            )
            write(
                work_dir / self.initial_test_file_path.name, test_xyzs, format="extxyz"
            )
            if self.db.size == 0:
                self.db.add_structures(train_xyzs, split="train", skip_duplicates=True)
                self.db.add_structures(test_xyzs, split="test", skip_duplicates=True)
            return train_xyzs, test_xyzs

        initialiser_entry = resolve("initialiser", "default")

        if self.db.size > 0:
            logger.info(
                "Global DB has %d existing structures; reading those in first.",
                self.db.size,
            )

        needs = initialiser_entry.compute_needs(self.db, init_config)

        extra_datasets = init_config.get("extra_datasets") or []
        if extra_datasets:
            for extra_dataset in extra_datasets:
                self._seed_db_from_extra_dataset(extra_dataset)
            needs = initialiser_entry.compute_needs(self.db, init_config)

        if _needs_anything(needs):
            logger.info(
                "DB check: %d structure(s) already evaluated. Generating missing "
                "structures: %d isolated atoms, %d dimers, %d trimers, %d amorphous.",
                self.db.size,
                len(needs["isolated_atoms"]),
                sum(needs["dimer_override"].values()),
                sum(needs["trimer_override"].values()),
                needs["amorphous_override"],
            )

            generated_atoms_list = None
            if init_config.get("read_generated_file") is not None:
                generated_atoms_list = read_atoms_file_if_enabled(
                    True, work_dir / init_config["read_generated_file"]
                )
                if generated_atoms_list:
                    logger.info(
                        "Read %d pre-generated structures from file: %s",
                        len(generated_atoms_list),
                        init_config["read_generated_file"],
                    )

            if not generated_atoms_list:
                generated_atoms_list = initialiser_entry.generate(
                    init_config,
                    base_name=base_name,
                    name=init_config["name"],
                    hpc=init_config["hpc"],
                    max_time=init_config["max_time"],
                    needs=needs,
                )

            if not generated_atoms_list:
                raise ValueError(
                    "No structures were generated. Check initialization configuration."
                )

            high_accuracy_structures = _evaluator_orchestrate(
                generated_atoms_list,
                self.jobs_dict["high_accuracy_evaluation"],
                base_name=base_name,
                name=self.jobs_dict["high_accuracy_evaluation"]["name"],
                hpc=self.jobs_dict["high_accuracy_evaluation"]["hpc"],
                max_time=self.jobs_dict["high_accuracy_evaluation"]["max_time"],
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

        all_evaluated = self.db.get_all_as_atoms()

        if workflow_config.get("grouped_splits", False):
            train_xyzs, test_xyzs = grouped_split(
                all_evaluated, workflow_config["test_to_train_ratio"], self.seed
            )
        else:
            test_config_types = set(workflow_config["test_config_types"])
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
                eligible_train, test_xyzs = split_atoms_list_into_test_and_train(
                    eligible_test_structures,
                    workflow_config["test_to_train_ratio"],
                    self.seed,
                )
                train_config_types = {
                    a.info.get("config_type", "") for a in eligible_train
                }
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
                        "Reserved one structure from each of %s for training to "
                        "avoid entirely excluding these config_types from "
                        "train_atoms_list.",
                        sorted(missing_types),
                    )
                train_xyzs = always_train_structures + eligible_train

        write(work_dir / self.initial_train_file_path.name, train_xyzs, format="extxyz")
        write(work_dir / self.initial_test_file_path.name, test_xyzs, format="extxyz")

        config_types_in_train = {
            atoms.info["config_type"]
            for atoms in train_xyzs
            if "config_type" in atoms.info
        }
        logger.info("Config types in training set: %s", config_types_in_train)
        return train_xyzs, test_xyzs

    # -- Committee training (decisions 2, 3, 6, 7) --

    def _train_mlip(self, base_name: str) -> pd.DataFrame:
        committee_config = self.jobs_dict["mlip_committee"]
        name = committee_config["name"]
        committee_size = committee_config["size_of_committee"]
        hpc = committee_config["hpc"]
        max_time = committee_config["max_time"]
        trainer_name = committee_config.get("trainer", "mace")
        workflow_config = self.jobs_dict.get("workflow", {})

        workdir = Path("results", base_name)

        if self._phase_done(base_name, "train_mlip"):
            logger.info("train_mlip already done for %s, reloading metrics.", base_name)
            return self._cross_loop_metrics_dataframe(name)

        trainer_entry = resolve("mlip_trainer", trainer_name)

        all_training = list(read(workdir / "train_set.xyz", ":", format="extxyz"))
        test_path = workdir / "test_set.xyz"

        valid_config_types = workflow_config.get(
            "valid_config_types", workflow_config.get("test_config_types", [])
        )
        acceptable_configs = [*valid_config_types, "high_sd"]
        valid_fraction = workflow_config.get("valid_fraction", 0.05)
        rng = np.random.default_rng(self.seed)
        if workflow_config.get("grouped_validation", False):
            new_train_set, valid_set = grouped_split(
                all_training, valid_fraction, self.seed
            )
        else:
            new_train_set, valid_set = _select_validation_split(
                all_training, acceptable_configs, valid_fraction, rng
            )

        train_path = workdir / "split_train.xyz"
        write(train_path, new_train_set, format="extxyz")
        valid_path_str: str | None = None
        if valid_set:
            valid_path = workdir / "split_valid.xyz"
            write(valid_path, valid_set, format="extxyz")
            valid_path_str = str(valid_path)

        results: dict[int, tuple] = {}
        missing: list[int] = []
        for fit_idx in range(committee_size):
            paths = trainer_entry.output_paths(
                committee_config, base_name=base_name, name=name, fit_idx=fit_idx
            )
            if all(p.exists() for p in paths):
                try:
                    results[fit_idx] = trainer_entry.read_existing_result(
                        committee_config,
                        base_name=base_name,
                        name=name,
                        fit_idx=fit_idx,
                    )
                    continue
                except ValueError:
                    logger.warning(
                        "fit_%d's cached result failed validation; retraining.", fit_idx
                    )
            missing.append(fit_idx)

        if missing:
            logger.info(
                "train_mlip for %s: %d/%d fit(s) already cached; submitting %s.",
                base_name,
                committee_size - len(missing),
                committee_size,
                missing,
            )
            input_files = [str(train_path), str(test_path)]
            if valid_path_str:
                input_files.append(valid_path_str)
            remote_info = get_remote_info(
                {"hpc": hpc, "name": name, "max_time": max_time},
                input_files=input_files,
            )
            job_configs = [
                {
                    "function_kwargs": {
                        "train_atoms_path": str(train_path),
                        "valid_atoms_path": valid_path_str,
                        "test_atoms_path": str(test_path),
                        "config": committee_config,
                        "fit_seed": self.seed + fit_idx,
                        "base_name": base_name,
                        "name": name,
                        "fit_idx": fit_idx,
                        "hpc": hpc,
                        "max_time": max_time,
                    },
                    "output_files": [str(workdir / name / f"fit_{fit_idx}")],
                }
                for fit_idx in missing
            ]
            submitted = submit_n(trainer_entry.train, job_configs, remote_info)
            for position, fit_idx in enumerate(missing):
                if submitted[position] is not None:
                    results[fit_idx] = submitted[position]

        if len(results) < 3:
            raise RuntimeError(
                f"train_mlip for {base_name}: only {len(results)} trained model(s) "
                f"succeeded (out of {committee_size} requested) — need at least 3 "
                "for a usable committee std-dev. Check remote job logs for failures."
            )
        if len(results) < committee_size:
            logger.warning(
                "train_mlip for %s: only %d/%d committee member(s) succeeded; "
                "proceeding with %d.",
                base_name,
                len(results),
                committee_size,
                len(results),
            )

        self._store_predictions_and_cleanup(base_name, name, results)

        if committee_config.get("quality_gate"):
            check_quality_gate(workdir, committee_config)

        self._mark_phase_done(base_name, "train_mlip")
        return self._cross_loop_metrics_dataframe(name)

    def _cross_loop_metrics_dataframe(self, name: str) -> pd.DataFrame:
        """Skeleton-level replacement for mlip.mace.get_mace_eval_info's
        cross-loop DataFrame aggregation, generalized to read the
        trainer-agnostic evaluation_metrics.json schema directly (via
        read_evaluation) rather than through a MACE-specific function.
        One row per AL loop (in loop order), aggregating each loop's "test"
        split across committee members -- matches what mae_al_loop_plot/
        plot_training_curves already expect (decision 19: those stay
        unchanged, out of scope for this refactor).
        """
        al_loop_dirs = sorted(
            Path("results").glob("al_loop_*"),
            key=lambda p: int(p.name.rsplit("_", 1)[1]),
        )
        rows = []
        for al_loop_dir in al_loop_dirs:
            metric_files = sorted(
                (al_loop_dir / name).glob("fit_*/evaluation_metrics.json")
            )
            if not metric_files:
                continue
            records = []
            for metric_file in metric_files:
                try:
                    record, _ = read_evaluation(metric_file.parent, "test")
                    records.append(record)
                except (FileNotFoundError, KeyError, ValueError):
                    continue
            if not records:
                continue
            row = {
                key: float(np.mean([r[key] for r in records]))
                for key in ("mae_f", "mae_e_per_atom")
            }
            row.update(
                {
                    f"{key}_std_dev": float(np.std([r[key] for r in records]))
                    for key in ("mae_f", "mae_e_per_atom")
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _store_predictions_and_cleanup(
        self, base_name: str, name: str, results: dict[int, tuple]
    ) -> None:
        """Store per-fit predictions in the GlobalDatabase and clean up
        local checkpoints/ directories -- local, post-sync, committee-shaped
        operations needing db/loop_idx/base_name, so skeleton-level rather
        than trainer-internal (decision 2/15)."""
        loop_idx = int(base_name.rsplit("_", 1)[-1]) if "al_loop_" in base_name else 0
        for fit_idx, (_model_path, compiled_model_path, _metrics) in results.items():
            fit_dir = Path("results", base_name, name, f"fit_{fit_idx}")
            preds = _read_mace_eval_predictions(fit_dir)
            if preds:
                self.db.store_model_predictions(loop_idx, fit_idx, preds)
            if compiled_model_path is not None:
                checkpoints_dir = fit_dir / "checkpoints"
                if checkpoints_dir.exists():
                    shutil.rmtree(checkpoints_dir, ignore_errors=True)
                    logger.info(
                        "Removed local %s after successful fit.", checkpoints_dir
                    )

    def store_mlip_predictions(
        self, loop_idx: int, base_name: str, job_dict: dict
    ) -> None:
        """No-op: prediction storage now happens inside _train_mlip (via
        _store_predictions_and_cleanup), since the trainer's returned
        results are only available there. Kept as a method (matching
        BaseActiveLearningWorkflow's call site in run()) for interface
        parity; run() below does not call it separately."""

    # -- Structure generation: eligibility filtering, generate, committee scoring --

    def _score_committee(
        self,
        structure_list: list[Atoms],
        base_name: str,
        best_fit_idx: int,
        fits_to_use: list[int],
        trainer_name: str,
        committee_config: dict,
        name: str,
        sg_hpc: dict,
        sg_max_time: str,
    ) -> dict:
        trainer_entry = resolve("mlip_trainer", trainer_name)
        order = [best_fit_idx, *fits_to_use]
        member_paths: dict[int, str] = {}
        for fit_idx in order:
            model_path, _, _ = trainer_entry.read_existing_result(
                committee_config, base_name=base_name, name=name, fit_idx=fit_idx
            )
            member_paths[fit_idx] = model_path

        remote_info = get_remote_info(
            {"hpc": sg_hpc, "name": f"score_{name}", "max_time": sg_max_time},
            input_files=list(member_paths.values()),
        )
        job_configs = [
            {
                "function_kwargs": {
                    "structure_list": structure_list,
                    "model_path": member_paths[fit_idx],
                    "trainer": trainer_name,
                    "trainer_config": committee_config,
                }
            }
            for fit_idx in order
        ]
        results = submit_n(_score_structures_with_member, job_configs, remote_info)

        structure_forces_dict: dict = {}
        for position, fit_idx in enumerate(order):
            result = results[position]
            if result is None:
                raise RuntimeError(
                    f"Committee scoring failed for fit_{fit_idx} on {base_name}."
                )
            label = "base_mlip" if fit_idx == best_fit_idx else f"fit_{fit_idx}"
            structure_forces_dict[label] = {
                f"structure_{i}": {
                    "forces": result["forces"][i],
                    "energy": result["energies"][i],
                }
                for i in range(len(structure_list))
            }
        return structure_forces_dict

    def _generate_structures(
        self, base_name: str, train_atoms_list: list[Atoms]
    ) -> list[Atoms]:
        sg_config = self.jobs_dict["structure_generation"]
        name = sg_config["name"]
        method = sg_config.get("method", "md")
        hpc = sg_config["hpc"]
        max_time = sg_config["max_time"]

        operating_dir = Path("results", base_name, name)

        high_sd_path = operating_dir / "high_sd_structures.xyz"
        if high_sd_path.exists():
            high_sd_structures = list(read(high_sd_path, ":", format="extxyz"))
            for structure in high_sd_structures:
                structure.info["needs_relaxation"] = (
                    self.high_force_threshold is not None
                )
            logger.info(
                "%d High SD structures loaded from file: %s",
                len(high_sd_structures),
                high_sd_path,
            )
            self._mark_phase_done(base_name, "generate_structures")
            return high_sd_structures

        selection_kwargs = sg_config.get("structure_selection_kwargs", {})
        eligible = filter_eligible_structures(
            train_atoms_list,
            chem_formula_list=selection_kwargs.get("chem_formula_list"),
            selectable_configs=selection_kwargs.get("selectable_configs"),
            atom_number_range=tuple(selection_kwargs.get("atom_number_range", (0, 0))),
        )

        committee_config = self.jobs_dict["mlip_committee"]
        best_fit_idx, best_model_path = select_best_committee_model(
            base_name, committee_config, seed=self.seed
        )
        committee_size = committee_config["size_of_committee"]
        fits_to_use = [i for i in range(committee_size) if i != best_fit_idx]
        trainer_name = committee_config.get("trainer", "mace")

        generator_entry = resolve("structure_generator", method)
        structure_list = generator_entry.generate(
            eligible,
            str(best_model_path),
            sg_config,
            base_name=base_name,
            name=name,
            hpc=hpc,
            max_time=max_time,
        )

        structure_list = filter_structures_by_min_bond_distance(structure_list)

        logger.info(
            "Structure generation: evaluating %d candidate structure(s) against "
            "the full %d-member committee to select the most uncertain ones.",
            len(structure_list),
            committee_size,
        )
        structure_forces_dict = self._score_committee(
            structure_list,
            base_name,
            best_fit_idx,
            fits_to_use,
            trainer_name,
            committee_config,
            committee_config["name"],
            hpc,
            max_time,
        )

        high_sd_structures = find_high_sd_structures(
            structure_list=structure_list,
            base_name=base_name,
            job_dict=self.jobs_dict,
            structure_forces_dict=structure_forces_dict,
        )

        for i, structure in enumerate(high_sd_structures):
            structure.info["job_id"] = i
            structure.info["needs_relaxation"] = self.high_force_threshold is not None

        self._mark_phase_done(base_name, "generate_structures")
        return high_sd_structures

    # -- run() --

    def run(self) -> None:
        if self.jobs_dict.get("dataset_curation"):
            policy_path = Path("results/curation_policy.json")
            policy = json.dumps(
                self.jobs_dict["dataset_curation"], sort_keys=True, indent=2
            )
            if policy_path.exists() and policy_path.read_text() != policy:
                raise ValueError(
                    "Curation policy changed: use a new results directory to avoid stale checkpoints"
                )
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(policy)
        self.pre_run_checks()

        last_complete = self._last_complete_loop()
        workflow_config = self.jobs_dict.get("workflow", {})

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
            train_xyzs, test_xyzs = self._initialize_training_set("initialization")
            n_tagged = self.db.update_splits_post_hoc(train_xyzs, test_xyzs)
            logger.info(
                "Initialized training set with %d structures; tagged %d in DB.",
                len(train_xyzs),
                n_tagged,
            )
            effective_start = self.start_loop

        if self.remove_redundancy:
            remove_redundancy_from_partition(
                self.db, config_list=workflow_config["test_config_types"] + ["high_sd"]
            )
        if self.high_force_threshold is not None:
            remove_high_force_structures_from_partition(
                self.db, force_threshold=self.high_force_threshold
            )
        if self.jobs_dict.get("dataset_curation"):
            curate_database(self.db, self.jobs_dict["dataset_curation"])

        for loop in range(effective_start, self.number_of_al_loops):
            base_name = f"al_loop_{loop}"
            train_xyzs = self.db.get_train_atoms()
            test_xyzs = self.db.get_test_atoms()

            loop_plots_dir = Path("results", "current_plots", base_name)
            if self.plots:
                loop_plots_dir.mkdir(exist_ok=True, parents=True)
                from alomancy.analysis.bond_distance_plots import (
                    plot_training_bond_distances,
                )

                plot_training_bond_distances(base_name, self.db, loop_plots_dir)

            workdir = Path(f"results/{base_name}")
            try:
                workdir.mkdir(exist_ok=True, parents=True)
            except OSError as exc:
                logger.warning("Could not create directory %s: %s", workdir, exc)

            train_file = workdir / "train_set.xyz"
            test_file = workdir / "test_set.xyz"
            try:
                write(train_file, train_xyzs, format="extxyz")
                write(test_file, test_xyzs, format="extxyz")
            except OSError as exc:
                if "test" not in str(exc).lower():
                    raise
                logger.warning("Could not write files (test environment): %s", exc)

            logger.debug("Starting AL loop %d", loop)
            logger.debug("  Training set size: %d", len(train_xyzs))
            logger.debug("  Test set size: %d", len(test_xyzs))

            evaluation_results = self._train_mlip(base_name)
            logger.debug("AL Loop %d evaluation results:\n%s", loop, evaluation_results)

            if self.plots:
                mae_al_loop_plot(
                    evaluation_results,
                    self.jobs_dict["mlip_committee"],
                    directory=loop_plots_dir,
                )
                from alomancy.analysis.mlip_plots import (
                    plot_dft_vs_model,
                    plot_training_curves,
                )

                plot_training_curves(
                    base_name,
                    self.jobs_dict["mlip_committee"],
                    self.seed,
                    loop_plots_dir,
                )
                plot_dft_vs_model(
                    base_name,
                    self.jobs_dict["mlip_committee"],
                    self.seed,
                    loop_plots_dir,
                    db=self.db,
                    loop_idx=loop,
                )

            if workflow_config.get("train_only", False):
                logger.info(
                    "Initial committee training complete; train_only stops before generation."
                )
                return

            generated_structures = self._generate_structures(base_name, train_xyzs)

            high_accuracy_eval_config = self.jobs_dict["high_accuracy_evaluation"]
            if self.high_force_threshold is not None:
                high_accuracy_eval_config = {
                    **high_accuracy_eval_config,
                    "fmax": self.high_force_threshold,
                }

            new_training_data = _evaluator_orchestrate(
                generated_structures,
                high_accuracy_eval_config,
                base_name=base_name,
                name=high_accuracy_eval_config["name"],
                hpc=high_accuracy_eval_config["hpc"],
                max_time=high_accuracy_eval_config["max_time"],
                allow_relaxation=True,
                start_index=0,
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

            if workflow_config.get("fixed_test", False):
                archive = self.db.get_all_as_atoms()
                known = {geometry_digest(a) for a in archive}
                held_groups = {
                    a.info.get("split_group")
                    for a in archive
                    if a.info.get("split") == "test" and a.info.get("split_group")
                }
                new_train_data = []
                diagnostic = []
                for atoms in new_training_data:
                    key = geometry_digest(atoms)
                    if key in known or atoms.info.get("split_group") in held_groups:
                        diagnostic.append(atoms)
                    else:
                        new_train_data.append(atoms)
                        known.add(key)
                self.db.add_structures(
                    diagnostic, split="diagnostic", skip_duplicates=False
                )
                new_test_data = []
            else:
                new_train_data, new_test_data = split_atoms_list_into_test_and_train(
                    new_training_data,
                    test_fraction=workflow_config["test_to_train_ratio"],
                    seed=self.seed,
                )

            self.db.add_structures(new_train_data, split="train", skip_duplicates=False)
            self.db.add_structures(new_test_data, split="test", skip_duplicates=False)

            if self.remove_redundancy:
                remove_redundancy_from_partition(
                    self.db,
                    config_list=workflow_config["test_config_types"] + ["high_sd"],
                )
            if self.high_force_threshold is not None:
                remove_high_force_structures_from_partition(
                    self.db, force_threshold=self.high_force_threshold
                )
            if self.jobs_dict.get("dataset_curation"):
                curate_database(self.db, self.jobs_dict["dataset_curation"])

            self._mark_phase_done(base_name, "loop")
            logger.debug(
                "Completed AL loop %d, retraining with %d structures.",
                loop,
                len(train_xyzs),
            )

            if self.plots and self.log_file is not None:
                from alomancy.analysis.timing_plots import timing_plots

                timing_plots(self.log_file, Path("results", "current_plots"))


def build_workflow(jobs_dict: dict, **init_kwargs: Any) -> CommitteeUncertaintyWorkflow:
    """Factory dispatching on workflow.skeleton (inside the existing
    `workflow` config section). Currently the only registered skeleton is
    "committee_uncertainty"; a future FurthestPointSamplingWorkflow would
    add its own name here."""
    skeleton = jobs_dict.get("workflow", {}).get("skeleton", "committee_uncertainty")
    if skeleton != "committee_uncertainty":
        raise ValueError(
            f"Unknown workflow.skeleton {skeleton!r}. Available: ['committee_uncertainty']"
        )
    return CommitteeUncertaintyWorkflow(jobs_dict=jobs_dict, **init_kwargs)
