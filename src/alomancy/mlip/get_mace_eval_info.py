import ast
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
    Read final test metrics; explicitly identify legacy validation-only logs.
    """

    al_loop_dirs = sorted(
        Path("results").glob("al_loop_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    )
    all_avg_results = []
    for al_loop_dir in al_loop_dirs:
        from alomancy.mlip.evaluation import read_evaluation

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
    seed: int,
    metric: str = "mae_f",
) -> tuple[int, Path]:
    """Select the exported checkpoint with the lowest common-validation MAE.

    No silent fit_0 fallback and no test-set selection. Old runs must be
    re-evaluated on a common validation split before resuming exploration.
    """
    from alomancy.mlip.evaluation import read_evaluation

    logger.debug("Committee selection for training seed %d", seed)
    name = mlip_committee_job_dict["name"]
    directory = Path("results", base_name, name)
    candidates = []
    identities = set()
    for i in range(mlip_committee_job_dict["size_of_committee"]):
        try:
            record, model = read_evaluation(directory / f"fit_{i}", "valid")
            score = float(record[metric])
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"Cannot select committee: fit_{i} needs a complete checkpoint validation evaluation"
            ) from exc
        if not np.isfinite(score):
            raise RuntimeError(f"Non-finite validation {metric} in fit_{i}")
        identities.add(record["data_id"])
        candidates.append((score, i, model))
    if not candidates or len(identities) != 1:
        raise RuntimeError("Committee selection requires a common validation dataset")
    score, best_fit, model = min(candidates)
    logger.info("Selected fit_%d by validation %s=%.6g", best_fit, metric, score)
    return best_fit, model
