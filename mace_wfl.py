from wfl.configset import ConfigSet
from wfl.fit.mace import fit
from pathlib import Path
import os
import time


def train_mace(remote_info, base_file_name, job_name, seed=803, fit_idx=0):
    workdir = Path("results", base_file_name)

    # Make directory named "MACE" and save model there
    mace_dir = Path(workdir, "MACE_model")
    print(f"Creating MACE directory: {mace_dir}")
    mace_dir.mkdir(exist_ok=True, parents=True)

    # Read MACE fit parameters
    training_file = Path(workdir, f"{base_file_name}_train_set.xyz")
    test_file = Path(workdir, f"{base_file_name}_test_set.xyz")

    # any name
    mace_name = f"{job_name}_fit_{fit_idx}"

    # MACE fit parameters can be initiated as following by reading YAML file.
    # Or one can also directly make dictionary
    mace_fit_params = {
        "E0s": {6: -241.94038776317848, 11: -1296.5877903540002},
        "model": "MACE",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "atomic_numbers": [6, 11],
        "correlation": 3,
        "device": "cuda",
        "ema": None,
        "energy_weight": 1,
        "forces_weight": 10,
        "error_table": "PerAtomMAE",
        "eval_interval": 1,
        "max_L": 2,
        "max_num_epochs": 10,
        "name": mace_name,
        "num_channels": 128,
        "num_interactions": 2,
        "patience": 30,
        "r_max": 5.0,
        "restart_latest": None,
        "save_cpu": None,
        "scheduler_patience": 15,
        "start_swa": 6,
        "swa": None,
        "batch_size": 16,
        "valid_batch_size": 16,
        "distributed": None,
        "seed": seed,
    }

    fit(
        fitting_configs=ConfigSet(str(training_file)),
        mace_name=mace_name,
        mace_fit_params=mace_fit_params,
        mace_fit_cmd="mace_run_train",  # f'python {str(Path(mace_file_dir, "run_train.py"))}',
        remote_info=remote_info,
        run_dir=mace_dir,
        prev_checkpoint_file=None,
        test_configs=ConfigSet(str(test_file)),
        dry_run=False,
        wait_for_results=False,
    )


def make_job_list(base_file_name, job_name, size_of_committee=5):
    dir_list = [
        f"results/{base_file_name}/MACE_model/{job_name}_fit_{i}" for i in range(size_of_committee)
    ]
    completed_mask = [
        os.path.exists(Path(dir, f"{job_name}_stagetwo_compiled.model")) for dir in dir_list
    ]
    return [i for i in range(size_of_committee) if not completed_mask[i]]


def create_mace_committee(remote_info, base_file_name, job_name, seed=803, size_of_committee=5):
    """
    Create a MACE committee by training multiple MACE models with different seeds.
    """
    job_list = make_job_list(base_file_name, job_name, size_of_committee)

    for fit_idx in job_list:
        train_mace(
            remote_info, base_file_name, job_name=job_name, seed=seed + fit_idx, fit_idx=fit_idx
        )

    while len(job_list) > 0:
        print(job_list)
        time.sleep(5)
        job_list = make_job_list(base_file_name, job_name, size_of_committee)
