import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
            if key in ["mae_f", "mae_e_per_atom"]
        }
        std_dev_results = {
            key: np.std([np.float32(result[key]) for result in results])
            for key in results[0]
            if key in ["mae_f", "mae_e_per_atom"]
        }
        avg_result.update(
            {f"{key}_std_dev": std_dev_results[key] for key in std_dev_results}
        )
        all_avg_results.append(avg_result)
    return pd.DataFrame(all_avg_results)


def _read_last_metric_record(txt_path: Path) -> dict | None:
    """Read the last parseable key-value record from a MACE metrics file.

    Handles JSON-lines format (newer MACE) and Python list-of-tuples format
    (older MACE).
    """
    last_record: dict | None = None
    with txt_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    last_record = record
                continue
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                record = dict(eval(line))
                last_record = record
            except Exception:
                pass
    return last_record


def select_best_committee_model(
    base_name: str,
    mlip_committee_job_dict: dict,
    seed: int,
    metric: str = "mae_f",
) -> tuple[int, Path]:
    """
    Select the committee member with the lowest test-set force MAE.

    Reads the ``*_test.txt`` metrics file written by MACE at the end of each
    training run, picks the fit with the lowest value of *metric* on the held-
    out test set, and returns ``(best_fit_index, stagetwo_model_path)``.

    Falls back to ``(0, fit_0_path)`` if test metrics cannot be read for any
    committee member.
    """
    name = mlip_committee_job_dict["name"]
    n_fits = mlip_committee_job_dict["size_of_committee"]
    committee_dir = Path("results", base_name, name)

    best_fit = 0
    best_score = float("inf")

    for i in range(n_fits):
        fit_dir = committee_dir / f"fit_{i}"
        results_dir = fit_dir / "results"
        fit_seed = seed + i

        txt_path = results_dir / f"{name}_run-{fit_seed}_test.txt"
        if not txt_path.exists():
            candidates = sorted(results_dir.glob("*_test.txt"))
            if not candidates:
                logger.warning("No test metrics file found for fit_%d — skipping.", i)
                continue
            txt_path = candidates[0]
            logger.debug("Using test metrics file: %s", txt_path)

        record = _read_last_metric_record(txt_path)
        if record is None:
            logger.warning("No parseable records in %s — skipping fit_%d.", txt_path, i)
            continue

        score = record.get(metric)
        if score is None:
            logger.warning(
                "Metric %r not in test file for fit_%d — skipping.", metric, i
            )
            continue

        score = float(score)
        logger.debug("fit_%d test %s = %.6f", i, metric, score)
        if score < best_score:
            best_score = score
            best_fit = i

    if best_score == float("inf"):
        logger.warning(
            "Could not read test %r for any committee member; defaulting to fit_0.",
            metric,
        )
    else:
        logger.info(
            "Best committee member: fit_%d (test %s = %.6f).",
            best_fit,
            metric,
            best_score,
        )

    model_path = committee_dir / f"fit_{best_fit}" / f"{name}_stagetwo.model"
    return best_fit, model_path
