import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alomancy.mlip.evaluation import read_evaluation

logger = logging.getLogger(__name__)


def get_mace_eval_info(
    mlip_committee_job_dict: dict,
) -> pd.DataFrame:
    """
    Read final test metrics; explicitly identify legacy validation-only logs.
    """

    al_loop_dirs = sorted(
        Path("results").glob("al_loop_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    )
    all_avg_results = []
    for al_loop_dir in al_loop_dirs:
        metric_files = sorted(
            (al_loop_dir / mlip_committee_job_dict["name"]).glob(
                "fit_*/evaluation_metrics.json"
            )
        )
        if metric_files:
            expected = mlip_committee_job_dict.get(
                "size_of_committee", len(metric_files)
            )
            expected_dirs = {f"fit_{i}" for i in range(expected)}
            if {p.parent.name for p in metric_files} != expected_dirs:
                raise RuntimeError(
                    "Missing checkpoint evaluations for committee members"
                )
            records = [read_evaluation(p.parent, "test")[0] for p in metric_files]
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
            row["metric_source"] = "checkpoint_test"
            all_avg_results.append(row)
            continue
        if mlip_committee_job_dict.get("require_checkpoint_metrics", False):
            raise RuntimeError(
                f"{al_loop_dir}: checkpoint evaluations are required; training logs are insufficient"
            )
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
                result = dict(ast.literal_eval(data_line))
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
        avg_result["metric_source"] = "legacy_training_validation"
        logger.warning(
            "%s: using legacy training-time validation metrics, not final test metrics",
            al_loop_dir,
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
                record = dict(ast.literal_eval(line))
                last_record = record
            except Exception:
                pass
    return last_record


def select_best_committee_model(
    base_name: str,
    mlip_committee_job_dict: dict,
    seed: int,  # noqa: ARG001 -- unused now that selection is checkpoint-based, kept for call-site compatibility
    metric: str = "mae_f",
) -> tuple[int, Path]:
    """
    Select the committee member with the lowest common-validation error.

    Prefers the shared ``"valid"`` checkpoint split (``mlip/evaluation.py``'s
    ``save_evaluation``/``read_evaluation``, written by
    ``_save_mace_eval_predictions`` right after training) when every
    committee member has one. Falls back to the held-out ``"test"`` split
    when NO fit has a ``"valid"`` entry at all -- ``mace_fit``'s own
    ``_select_validation_split`` legitimately skips carving a validation
    split (just a warning, not a failure) whenever the eligible pool is too
    small, in which case every fit is missing "valid" uniformly, and the
    only sensible remaining common metric across the committee is "test"
    (always attempted regardless of pool size). If fits DISAGREE on having
    "valid" (some do, some don't), that indicates a genuine per-fit
    evaluation failure rather than a normal small-pool run, and this raises.

    Picks the fit with the lowest value of *metric* on whichever split was
    used and returns ``(best_fit_index, stagetwo_model_path)``.

    ``seed`` is no longer used by this function (the old *_test.txt-based
    legacy path derived a per-fit seed from it) -- kept as a required
    parameter only so existing call sites (``standard_active_learning.py``,
    the checkpoint-evaluation test suite) don't need to change.
    """
    name = mlip_committee_job_dict["name"]
    n_fits = mlip_committee_job_dict["size_of_committee"]
    committee_dir = Path("results", base_name, name)

    def _try_read(fit_dir: Path, split: str) -> tuple[float, Path] | None:
        try:
            metrics, model_path = read_evaluation(fit_dir, split)
        except (FileNotFoundError, KeyError, ValueError):
            return None
        return float(metrics[metric]), model_path

    valid_scores: dict[int, tuple[float, Path]] = {}
    test_scores: dict[int, tuple[float, Path]] = {}
    for i in range(n_fits):
        fit_dir = committee_dir / f"fit_{i}"
        valid_result = _try_read(fit_dir, "valid")
        if valid_result is not None:
            valid_scores[i] = valid_result
        test_result = _try_read(fit_dir, "test")
        if test_result is not None:
            test_scores[i] = test_result

    if valid_scores:
        if len(valid_scores) < n_fits:
            raise RuntimeError(
                f"select_best_committee_model for {base_name!r}: "
                f"{n_fits - len(valid_scores)} of {n_fits} committee fit(s) "
                "are missing complete checkpoint validation on the 'valid' "
                "split while others have it. Check remote job logs for "
                "evaluation failures."
            )
        scores, split_used = valid_scores, "valid"
    else:
        logger.info(
            "No committee member has a 'valid' checkpoint evaluation "
            "(expected when the eligible pool is too small for a "
            "validation split) — falling back to the 'test' split."
        )
        if len(test_scores) < n_fits:
            raise RuntimeError(
                f"select_best_committee_model for {base_name!r}: "
                f"{n_fits - len(test_scores)} of {n_fits} committee fit(s) "
                "are missing complete checkpoint validation (neither "
                "'valid' nor 'test' evaluation is available). Check remote "
                "job logs for evaluation failures."
            )
        scores, split_used = test_scores, "test"

    best_fit = min(scores, key=lambda i: scores[i][0])
    logger.info(
        "Best committee member: fit_%d (%s %s = %.6f).",
        best_fit,
        split_used,
        metric,
        scores[best_fit][0],
    )
    return best_fit, scores[best_fit][1]
