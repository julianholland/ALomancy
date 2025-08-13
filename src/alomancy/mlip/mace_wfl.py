from pathlib import Path

import numpy as np
from wfl.configset import ConfigSet
from wfl.fit.mace import fit


def mace_fit(seed, mlip_committee_job_dict, workdir_str, fit_idx=0):
    workdir = Path(workdir_str)
    mlip_dir = Path(workdir, mlip_committee_job_dict["name"])
    print(f"Creating MLIP directory: {mlip_dir}")
    mlip_dir.mkdir(exist_ok=True, parents=True)

    assert (
        "seed" not in mlip_committee_job_dict["mace_fit_kwargs"]
    ), "Seed should not be in mace_fit_kwargs, it is passed separately."
    assert (
        "energy_key" in mlip_committee_job_dict["mace_fit_kwargs"]
    ), "energy_key must be specified in mace_fit_kwargs. This corresponds to the energy key in the training set. using 'energy' is not recommended."
    assert (
        "forces_key" in mlip_committee_job_dict["mace_fit_kwargs"]
    ), "forces_key must be specified in mace_fit_kwargs. This corresponds to the forces key in the training set. using 'forces' is not recommended."

    if mlip_committee_job_dict["max_num_epochs"] is None:
        epochs = 80
    else:
        epochs = mlip_committee_job_dict["max_num_epochs"]

    # Read MACE fit parameters
    training_file = Path(workdir, "train_set.xyz")
    test_file = Path(workdir, "test_set.xyz")

    # default MACE fit parameters
    # These can be overridden by the job_dict passed to the function
    mace_fit_params = {
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
        "seed": seed + fit_idx,
        **mlip_committee_job_dict["mace_fit_kwargs"],
    }

    fit(
        fitting_configs=ConfigSet(str(training_file)),
        mace_name=mlip_committee_job_dict["name"],
        mace_fit_params=mace_fit_params,
        mace_fit_cmd="mace_run_train",  # f'python {str(Path(mace_file_dir, "run_train.py"))}',
        # remote_info=remote_info,
        run_dir=str(Path(mlip_dir, f"fit_{fit_idx}")),
        prev_checkpoint_file=None,
        test_configs=ConfigSet(str(test_file)),
        dry_run=False,
        wait_for_results=True,
    )
