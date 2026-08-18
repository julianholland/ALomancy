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

    Only considers fits whose stagetwo model file actually exists on disk --
    a fit can be missing readable test metrics while ALSO genuinely having
    no model at all, e.g. a committee member whose training job was
    abandoned after a sustained remote-communication failure (see
    remote_submission/executor.py's _get_results_with_resume) or never
    retried after train_mlip's backfill. Falls back to the lowest-indexed
    fit that HAS a model on disk if no committee member has readable test
    metrics -- previously this defaulted to fit_0 unconditionally, which
    crashed a downstream consumer (structure_generation trying to stage a
    nonexistent model file as an MD job input) the one time fit_0 itself was
    the fit with no model. Raises ValueError if literally none of the
    committee's fits have a model file (train_mlip guarantees at least 3
    before calling this, so this should only fire if that invariant is ever
    broken).
    """
    name = mlip_committee_job_dict["name"]
    n_fits = mlip_committee_job_dict["size_of_committee"]
    committee_dir = Path("results", base_name, name)

    def _model_path(i: int) -> Path:
        return committee_dir / f"fit_{i}" / f"{name}_stagetwo.model"

    fits_with_model = [i for i in range(n_fits) if _model_path(i).exists()]
    if not fits_with_model:
        raise ValueError(
            f"select_best_committee_model for {base_name!r}: none of the "
            f"{n_fits} committee fit(s) have a {name}_stagetwo.model file "
            "on disk. Check remote job logs for failures."
        )

    best_fit: int | None = None
    best_score = float("inf")

    for i in fits_with_model:
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

    if best_fit is None:
        best_fit = fits_with_model[0]
        logger.warning(
            "Could not read test %r for any committee member; defaulting to "
            "fit_%d (lowest-indexed fit with a model on disk).",
            metric,
            best_fit,
        )
    else:
        logger.info(
            "Best committee member: fit_%d (test %s = %.6f).",
            best_fit,
            metric,
            best_score,
        )

    return best_fit, _model_path(best_fit)
