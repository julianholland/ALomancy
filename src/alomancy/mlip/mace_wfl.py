import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from expyre import ExPyRe
from mace import tools
from mace.cli.run_train import run

from alomancy.configs.remote_info import RemoteInfo

logger = logging.getLogger(__name__)

if (
    importlib.util.find_spec("torch._native") is not None
    and "TRITON_CACHE_DIR" not in os.environ
):
    logger.warning(
        "This PyTorch version uses triton for native GPU ops (torch._native). "
        "On HPC nodes without python3-dev headers, triton kernel compilation will "
        "fail at training time. Set TRITON_CACHE_DIR to a persistent path and "
        "pre-warm the cache in an interactive GPU job. See CLAUDE.md for details."
    )


def _save_mace_eval_predictions(name: str, train_filename: str) -> None:
    """Evaluate the trained stagetwo model on train and test sets; write predictions.

    Called from inside mace_fit while os.chdir'd into mlip_dir. Writes
    train_pred.xyz and test_pred.xyz in the current directory with mace_energy
    and mace_forces keys so store_mlip_predictions can read them locally without
    re-running inference.
    """
    model_path = Path(f"{name}_stagetwo_compiled.model")
    if not model_path.exists():
        logger.warning("Stagetwo compiled model not found; skipping eval predictions.")
        return

    try:
        from mace.calculators import MACECalculator

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        calc = MACECalculator(
            model_paths=[str(model_path.resolve())],
            device=device,
            default_dtype="float64",
        )
    except Exception as exc:
        logger.warning("Failed to load MACECalculator for post-training eval: %s", exc)
        return

    for tag, xyz_path in [("train", train_filename), ("test", "../../test_set.xyz")]:
        try:
            atoms_list = list(read(xyz_path, ":", format="extxyz"))
        except Exception as exc:
            logger.debug("Could not read %s for eval predictions: %s", xyz_path, exc)
            continue

        out = []
        n_ok = 0
        n_failed = 0
        for atoms in atoms_list:
            a = atoms.copy()
            a.calc = calc
            try:
                a.info["mace_energy"] = float(a.get_potential_energy())
                a.arrays["mace_forces"] = a.get_forces()
                n_ok += 1
            except Exception as exc:
                n_failed += 1
                # Only the first failure gets a full traceback -- this loop
                # runs once per structure (hundreds to thousands per AL
                # loop) and a repeated failure mode logs the same
                # traceback every time; the summary line below still
                # reports the total count. Was previously logger.debug,
                # which silently produced nothing at all: this function
                # runs in a freshly spawned remote process where
                # setup_logging() is never called, so there was no handler
                # for DEBUG-level records, and RemoteJobExecutor discarded
                # a successful job's stdout/stderr entirely -- meaning a
                # near-total prediction failure (e.g. 1 success out of
                # 1405 structures, previously observed) was completely
                # invisible, silently degrading parity plots to a single
                # trivial point (the one structure whose prediction
                # happened to succeed) with no error anywhere.
                if n_failed == 1:
                    logger.warning(
                        "Prediction failed for structure %d/%d (config_type=%s): %s",
                        n_ok + n_failed,
                        len(atoms_list),
                        atoms.info.get("config_type"),
                        exc,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "Prediction failed for structure %d/%d: %s",
                        n_ok + n_failed,
                        len(atoms_list),
                        exc,
                    )
            out.append(a)

        if n_failed:
            logger.warning(
                "%s predictions: %d succeeded, %d failed out of %d structures.",
                tag,
                n_ok,
                n_failed,
                len(atoms_list),
            )

        try:
            write(f"{tag}_pred.xyz", out, format="extxyz")
            logger.info("Saved %d %s prediction(s) to %s_pred.xyz.", len(out), tag, tag)
        except Exception as exc:
            logger.warning("Failed to write %s_pred.xyz: %s", tag, exc)


def _select_validation_split(
    all_training: list[Atoms],
    acceptable_configs: list[str],
    valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[Atoms], list[Atoms]]:
    """Carve a per-fit validation set from all_training.

    Only structures with config_type in acceptable_configs are eligible;
    the rest always stay in training. Returns (new_train_set, valid_set).
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


def mace_fit(
    job_dict: dict,
    seed: int,
    workdir_str: str,
    fit_idx: int = 0,
    _mace_fit_cmd: str = "mace_run_train",
):
    """
    Minimal MACE model fitting function.

    Parameters
    ----------
    job_dict : dict
        Full jobs dictionary (mlip_committee and initialization sub-dicts are used).
    seed : int
        Base random seed; each committee member uses seed + fit_idx.
    workdir_str : str
        Path to the AL loop working directory (contains train_set.xyz / test_set.xyz).
    fit_idx : int, optional
        Committee member index (default 0).
    """
    mlip_committee_job_dict = job_dict["mlip_committee"]
    workdir = Path(workdir_str)
    mlip_dir = Path(workdir, mlip_committee_job_dict["name"], f"fit_{fit_idx}")
    logger.info("Creating MLIP directory: %s", mlip_dir)
    mlip_dir.mkdir(exist_ok=True, parents=True)

    assert "seed" not in mlip_committee_job_dict["mace_fit_kwargs"], (
        "Seed should not be in mace_fit_kwargs, it is passed separately."
    )
    assert "energy_key" in mlip_committee_job_dict["mace_fit_kwargs"], (
        "energy_key must be specified in mace_fit_kwargs. This corresponds to the energy key in the training set. using 'energy' is not recommended."
    )
    assert "forces_key" in mlip_committee_job_dict["mace_fit_kwargs"], (
        "forces_key must be specified in mace_fit_kwargs. This corresponds to the forces key in the training set. using 'forces' is not recommended."
    )

    epochs = (
        80
        if mlip_committee_job_dict["max_num_epochs"] is None
        else mlip_committee_job_dict["max_num_epochs"]
    )

    # Read training data and carve per-fit validation set before chdir
    training_file = Path(workdir, "train_set.xyz")
    if not training_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {training_file}. "
            "Ensure train_set.xyz has been written to the working directory before fitting."
        )
    all_training = list(read(training_file, ":", format="extxyz"))
    logger.info(
        "Read %d training structures from %s.", len(all_training), training_file
    )

    # valid_config_types can be overridden in mlip_committee config; defaults to
    # initialization test_config_types so validation covers the same structure classes
    valid_config_types = mlip_committee_job_dict.get(
        "valid_config_types", job_dict["initialization"]["test_config_types"]
    )
    acceptable_configs = [*valid_config_types, "high_sd"]
    valid_fraction = mlip_committee_job_dict.get("valid_fraction", 0.05)
    fit_seed = seed + fit_idx
    rng = np.random.default_rng(fit_seed)
    new_train_set, valid_set = _select_validation_split(
        all_training, acceptable_configs, valid_fraction=valid_fraction, rng=rng
    )

    # When a split occurred, write per-fit train/valid files into mlip_dir (accessible
    # after chdir). When no split, point directly at the original train_set.xyz.
    valid_filename = f"valid_set_{fit_idx}.xyz"
    if valid_set:
        train_filename = f"train_set_{fit_idx}.xyz"
        write(mlip_dir / train_filename, new_train_set, format="extxyz")
        write(mlip_dir / valid_filename, valid_set, format="extxyz")
        logger.debug(
            "Wrote split: %d train to %s, %d valid to %s.",
            len(new_train_set),
            train_filename,
            len(valid_set),
            valid_filename,
        )
    else:
        train_filename = "../../train_set.xyz"

    # default MACE fit parameters
    # These can be overridden by the job_dict passed to the function
    mace_fit_params = {
        # Relative paths — MACE runs from inside mlip_dir after os.chdir below
        "train_file": train_filename,
        "test_file": "../../test_set.xyz",
        "model": "MACE",
        "correlation": 3,
        "device": "cuda",
        "ema": None,
        "energy_weight": 1,
        "forces_weight": 10,
        "error_table": "PerAtomMAE",
        "eval_interval": 1,
        "max_L": 2,
        "max_num_epochs": epochs,
        "name": mlip_committee_job_dict["name"],
        "num_channels": 128,
        "num_interactions": 2,
        "patience": 30,
        "r_max": 5.0,
        "restart_latest": None,
        "save_cpu": None,
        "scheduler_patience": 15,
        "start_swa": int(np.floor(epochs * 0.8)),
        "swa": None,
        "batch_size": 16,
        "valid_batch_size": 16,
        "distributed": None,
        "seed": fit_seed,
        **mlip_committee_job_dict["mace_fit_kwargs"],
    }
    if valid_set:
        mace_fit_params["valid_file"] = valid_filename

    logger.debug("MACE fit parameters:")
    for key, value in mace_fit_params.items():
        logger.debug("  %s: %s", key, value)

    parser = tools.build_default_arg_parser()
    args = parser.parse_args(["--name", mace_fit_params["name"]])  # seed defaults
    for key, value in mace_fit_params.items():
        setattr(args, key, value)

    orig_dir = os.getcwd()
    try:
        os.chdir(mlip_dir)
        run(args)
        _save_mace_eval_predictions(mlip_committee_job_dict["name"], train_filename)
    finally:
        os.chdir(orig_dir)


def _mace_fit_expyre_call(
    train_atoms_path: str,
    test_atoms_path: str,
    remote_info: RemoteInfo,
    mace_name: str,
    mace_fit_params: dict,
    mace_fit_cmd="mace_run_train",
    run_dir: Path = Path("mace_fit"),
):

    # fill in some params from standard function arguments
    mace_fit_params["name"] = mace_name
    mace_fit_params["energy_key"] = "REF_energy"
    mace_fit_params["forces_key"] = "REF_forces"
    if "compute_stress" in mace_fit_params:
        mace_fit_params["stress_key"] = "REF_stress"

    input_files = remote_info.input_files.copy()
    output_files = [*remote_info.output_files, str(run_dir)]

    # set number of threads in queued job, only if user hasn't set them
    if not any(
        var.split("=")[0] == "WFL_MACE_FIT_OMP_NUM_THREADS"
        for var in remote_info.env_vars
    ):
        remote_info.env_vars.append(
            "WFL_MACE_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE"
        )
    if not any(
        var.split("=")[0] == "WFL_NUM_PYTHON_SUBPROCESSES"
        for var in remote_info.env_vars
    ):
        remote_info.env_vars.append(
            "WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE"
        )

    remote_func_kwargs = {
        "train_atoms_path": train_atoms_path,
        "test_atoms_path": test_atoms_path,
        "remote_info": remote_info,
        "mace_name": mace_name,
        "mace_fit_params": mace_fit_params,
        "mace_fit_cmd": mace_fit_cmd,
        "run_dir": run_dir,
    }

    xpr = ExPyRe(
        name=remote_info.job_name,
        pre_run_commands=remote_info.pre_cmds,
        post_run_commands=remote_info.post_cmds,
        env_vars=remote_info.env_vars,
        input_files=input_files,
        output_files=output_files,
        function=_mace_fit_expyre_call,
        kwargs=remote_func_kwargs,
    )

    xpr.start(
        resources=remote_info.resources,
        system_name=remote_info.sys_name,
        header_extra=remote_info.header_extra,
        exact_fit=remote_info.exact_fit,
        partial_node=remote_info.partial_node,
    )

    results, stdout, stderr = xpr.get_results(
        timeout=remote_info.timeout, check_interval=remote_info.check_interval
    )
    if stdout is not None:
        sys.stdout.write(stdout)
    if stderr is not None:
        sys.stderr.write(stderr)

    # no outputs to rename since everything should be in run_dir
    xpr.mark_processed()

    if results is None and not remote_info.ignore_failed_jobs:
        raise RuntimeError(
            f"Remote job failed with stdout: {stdout} and stderr: {stderr}"
        )
    else:
        return results
