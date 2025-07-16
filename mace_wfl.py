from wfl.configset import ConfigSet
from wfl.fit.mace import fit
from pathlib import Path
import yaml
import os
import time



def train_mace(remote_info, base_file_name, seed=803, fit_idx=0):
    workdir = Path(os.getcwd())
    training_dir = Path(workdir, "mace_general")

    # Make directory named "MACE" and save model there
    mace_dir = Path(workdir, "MACE_model")
    mace_dir.mkdir(exist_ok=True)

    # mace_file_dir = Path('/u/jholl/mace_general')
    # Read MACE fit parameters
    training_file = Path(training_dir, f"{base_file_name}_train_set.xyz")
    # validation and test set file can be also defined.
    # validation = "validation.xyz"
    test_file = Path(training_dir, f"{base_file_name}_test.xyz")

    # any name
    mace_name = f"{base_file_name}_fit_{fit_idx}"

    # MACE fit parameters can be initiated as following by reading YAML file.
    # Or one can also directly make dictionary
    # train_mace(training_file, mace_name=mace_name, mace_fit_params=mace_params, run_dir=f"MACE/fit_{fit_idx}")

    mace_params = yaml.safe_load(Path(training_dir, "mace_train_2.yaml").read_text())
    mace_params["max_num_epochs"] = 60  # * (fit_idx + 1)
    mace_params["start_swa"] = 40  # + 30 * fit_idx
    mace_params["seed"] = seed
    # if RI_DICT[hpc]["partitions"][0] == "gpubig":
    #     mace_params["distributed"] = 'null'

    # prev_checkpoints = glob.glob(f"{workdir}/MACE/fit_{fit_idx-1}/checkpoints/*_swa.pt")

    # # Some dirty way to get checkpoint file with larger epoch in case there are more than one
    # if len(prev_checkpoints) > 0:
    #     p = re.compile("epoch-[0-9]*_swa.pt")
    #     prev_checkpoint = sorted(prev_checkpoints, key=lambda x: int(p.search(x).group()[6:].split("_")[0]))[-1].split("/")[-1]
    #     prev_checkpoint = f"MACE/fit_{fit_idx-1}/checkpoints/{prev_checkpoint}"

    #     remote_info_gpu= get_remote_info([prev_checkpoint])

    # else:

    fit(
        fitting_configs=ConfigSet(str(training_file)),
        mace_name=mace_name,
        mace_fit_params=mace_params,
        mace_fit_cmd="mace_run_train",  # f'python {str(Path(mace_file_dir, "run_train.py"))}',
        remote_info=remote_info,
        run_dir=f"MACE_model/fit_{fit_idx}",
        prev_checkpoint_file=None,
        test_configs=ConfigSet(str(test_file)),
        dry_run=False,
        wait_for_results=False,
    )


def make_job_list(size_of_committee=5):

    dir_list=[f'MACE_model/fit_{i}' for i in range(size_of_committee)]
    completed_mask= [os.path.exists(Path(dir, "test_stagetwo_compiled.model")) for dir in dir_list]
    return [i for i in range(size_of_committee) if not completed_mask[i]]


def create_mace_committee(remote_info, base_file_name, size_of_committee=5):
    """
    Create a MACE committee by training multiple MACE models with different seeds.
    """
    job_list = make_job_list(size_of_committee)
    
    while len(job_list) > 0:
        print(job_list)
        for fit_idx in job_list:
            train_mace(remote_info, base_file_name, seed=803+fit_idx, fit_idx=fit_idx)
        time.sleep(30)
        job_list=make_job_list(size_of_committee)



if __name__ == "__main__":
    create_mace_committee(5)
