import sys
from pathlib import Path

import numpy as np
from expyre import ExPyRe

from alomancy.configs.remote_info import RemoteInfo, get_remote_info
from argparse import Namespace
from mace import tools
from mace.cli.run_train import run

# def mace_fit(seed: int, 
#              mlip_committee_job_dict: dict, 
#              workdir_str: str, 
#              fit_idx: int = 0):
    
#     workdir = Path(workdir_str)
#     mlip_dir = Path(workdir, mlip_committee_job_dict["name"])
#     print(f"Creating MLIP directory: {mlip_dir}")
#     mlip_dir.mkdir(exist_ok=True, parents=True)

#     assert "seed" not in mlip_committee_job_dict["mace_fit_kwargs"], (
#         "Seed should not be in mace_fit_kwargs, it is passed separately."
#     )
#     assert "energy_key" in mlip_committee_job_dict["mace_fit_kwargs"], (
#         "energy_key must be specified in mace_fit_kwargs. This corresponds to the energy key in the training set. using 'energy' is not recommended."
#     )
#     assert "forces_key" in mlip_committee_job_dict["mace_fit_kwargs"], (
#         "forces_key must be specified in mace_fit_kwargs. This corresponds to the forces key in the training set. using 'forces' is not recommended."
#     )

#     if mlip_committee_job_dict["max_num_epochs"] is None:
#         epochs = 80
#     else:
#         epochs = mlip_committee_job_dict["max_num_epochs"]

#     # Read MACE fit parameters
#     training_file = Path(workdir, "train_set.xyz")
#     test_file = Path(workdir, "test_set.xyz")

#     # default MACE fit parameters
#     # These can be overridden by the job_dict passed to the function
#     mace_fit_params = {
#         "train_file": str(training_file),
#         "test_file": str(test_file),
#         "model": "MACE",
#         "correlation": 3,
#         "device": "cuda",
#         "ema": None,
#         "energy_weight": 1,
#         "forces_weight": 10,
#         "error_table": "PerAtomMAE",
#         "eval_interval": 1,
#         "max_L": 2,
#         "max_num_epochs": epochs,
#         "name": mlip_committee_job_dict["name"],
#         "num_channels": 128,
#         "num_interactions": 2,
#         "patience": 30,
#         "r_max": 5.0,
#         "restart_latest": None,
#         "save_cpu": None,
#         "scheduler_patience": 15,
#         "start_swa": int(np.floor(epochs * 0.8)),
#         "swa": None,
#         "batch_size": 16,
#         "valid_batch_size": 16,
#         "distributed": None,
#         "seed": seed + fit_idx,
#         **mlip_committee_job_dict["mace_fit_kwargs"],
#     }
#     print("MACE fit parameters:")
#     for key, value in mace_fit_params.items():
#         print(f"  {key}: {value}")
    
#     parser = tools.build_default_arg_parser()
#     args = parser.parse_args(["--name", mace_fit_params["name"]])  # seed defaults
#     for key, value in mace_fit_params.items():
#         setattr(args, key, value)

#     # run(args)
#     # subprocess.run(
#     #     ['mace_run_train',]
#     # )

#     # _mace_fit_expyre_call(
#     #     train_atoms_path=str(training_file),
#     #     test_atoms_path=str(test_file),
#     #     remote_info=get_remote_info(mlip_committee_job_dict, input_files=[str(training_file), str(test_file)]),
#     #     mace_name=mlip_committee_job_dict["name"],
#     #     mace_fit_params=mace_fit_params,
#     #     run_dir=Path(mlip_dir, f"fit_{fit_idx}")
#     # )

import os
import subprocess
import shutil
from pathlib import Path


def mace_fit(mlip_committee_job_dict: dict, seed: int, workdir_str: str, fit_idx: int = 0, mace_fit_cmd: str = 'mace_run_train'):
    """
    Minimal MACE model fitting function.

    Parameters
    ----------
    fitting_configs_path : str or Path
        Path to the training data file (XYZ or similar).
    mace_name : str
        Name/label for the MACE model.
    mace_fit_params : dict
        Hyperparameters passed as CLI flags to mace_run_train.
    mace_fit_cmd : str, optional
        Path/command for mace_run_train. Auto-detected if None.
    run_dir : str, optional
        Directory to run fitting in. Created if it doesn't exist.
    """
    workdir = Path(workdir_str)
    mlip_dir = Path(workdir, mlip_committee_job_dict["name"], f"fit_{fit_idx}")
    print(f"Creating MLIP directory: {mlip_dir}")
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

    if mlip_committee_job_dict["max_num_epochs"] is None:
        epochs = 80
    else:
        epochs = mlip_committee_job_dict["max_num_epochs"]

    # Read MACE fit parameters
    training_file = Path("../../train_set.xyz")
    test_file = Path("../../test_set.xyz")

    # default MACE fit parameters
    # These can be overridden by the job_dict passed to the function
    mace_fit_params = {
        "train_file": str(training_file),
        "test_file": str(test_file),
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
        **mlip_committee_job_dict["mace_fit_kwargs"],
    }

    mace_fit_params["seed"] = seed + fit_idx
    # mace_fit_params["results_dir"] = str(mlip_dir)
    print("MACE fit parameters:")
    for key, value in mace_fit_params.items():
        print(f"  {key}: {value}")

    
    # Resolve the fitting command
    # if mace_fit_cmd is None:
    #     mace_fit_cmd = os.environ.get("WFL_MACE_FIT_COMMAND") or shutil.which("mace_run_train")
    #     if mace_fit_cmd is None:
    #         raise RuntimeError("mace_run_train not found. Set WFL_MACE_FIT_COMMAND or add it to PATH.")


    parser = tools.build_default_arg_parser()
    args = parser.parse_args(["--name", mace_fit_params["name"]])  # seed defaults
    for key, value in mace_fit_params.items():
        setattr(args, key, value)

    orig_dir = os.getcwd()
    try:
        os.chdir(mlip_dir)
        run(args)
    finally:
        os.chdir(orig_dir)
    # # Build CLI command string
    # for key, val in mace_fit_params.items():
    #     if isinstance(val, (int, float)):
    #         mace_fit_cmd += f" --{key}={val}"
    #     elif val is None:
    #         mace_fit_cmd += f" --{key}"
    #     else:
    #         mace_fit_cmd += f" --{key}='{val}'"

    # orig_dir = os.getcwd()
    # try:
    #     os.chdir(mlip_dir)
    #     subprocess.run(mace_fit_cmd, shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     raise RuntimeError(f"MACE fitting failed with exit code {e.returncode}") from e
    # finally:
    #     os.chdir(orig_dir)

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
