from wfl.fit.mace import fit
from expyre.func import ExPyRe
import numpy as np




def run_example_remote(mace_files, job_dict, remote_info):
    def mace_fit(seed, epochs, mace_name):
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
        "max_num_epochs": epochs,
        "name": mace_name,
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
        "seed": seed,
    }

        fit(fitting_configs=ConfigSet(str(training_file)),
        mace_name=mace_name,
        mace_fit_params=mace_fit_params,
        mace_fit_cmd="mace_run_train",  # f'python {str(Path(mace_file_dir, "run_train.py"))}',
        # remote_info=remote_info,
        run_dir=str(Path(mace_dir,f"fit_{fit_idx}")),
        prev_checkpoint_file=None,
        test_configs=ConfigSet(str(test_file)),
        dry_run=False,
        wait_for_results=True,)

    xprs = []

    for i in models:
        remote_info.input_files = [mace_files] #what to send remote
        remote_info.output_files = ['./mace_model_1'] #what to get back

        xprs.append(ExPyRe(name=remote_info.job_name,
                           pre_run_commands=remote_info.pre_cmds,
                           post_run_commands=remote_info.post_cmds,
                           env_vars=remote_info.env_vars,
                           input_files=remote_info.input_files,
                           output_files=remote_info.output_files,
                           function=mace_fit, kwargs={'mace': mace_model}))
#start models
    for xpr in xprs:
        xpr.start(resources=remote_info.resources,
                  system_name=remote_info.sys_name,
                  header_extra=remote_info.header_extra,
                  exact_fit=remote_info.exact_fit,
                  partial_node=remote_info.partial_node)

    # gather results
    for xpr in xprs:
        try:
            results, stdout, stderr = xpr.get_results(timeout=remote_info.timeout,
                                                      check_interval=remote_info.check_interval)
        except Exception as exc:
            print("stdout", "-" * 30)
            print(stdout)
            print("stderr", "-" * 30)
            print(stderr)

    # mark as processed in jobs db in case of restarts
    for xpr in xprs:
        xpr.mark_processed()

    return None