from pathlib import Path

import numpy as np
import pandas as pd


def get_mace_eval_info(
    mlip_committee_job_dict: dict,
) -> pd.DataFrame:
    """
    Recover final results from train.txt files in MACE AL loop directories.
    """

    al_loop_dirs = list(Path.glob(Path("results"), "al_loop_*"))
    all_avg_results = []
    for al_loop_dir in al_loop_dirs:
        results_files = list(
            Path.glob(
                Path(al_loop_dir, mlip_committee_job_dict["name"]),
                "fit_*/results/*train.txt",
            )
        )
        if not results_files:
            continue
        results = []
        for results_file in results_files:
            with open(results_file) as file:
                data_line = file.readlines()[-1]
                result = dict(eval(data_line))
                results.append(result)

        avg_result = {
            key: np.mean([np.float32(result[key]) for result in results])
            for key in results[0]
            if key in ["mae_f", "mae_e"]
        }
        all_avg_results.append(avg_result)
    return pd.DataFrame(all_avg_results)
