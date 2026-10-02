import ast
import json
import logging
from pathlib import Path

import polars as pl

from alomancy.mlip.evaluation import (
    best_fit_test_metrics,
    loop_metrics_frame,
    rank_committee,
)

logger = logging.getLogger(__name__)


def get_mace_eval_info(
    mlip_committee_job_dict: dict,
) -> pl.DataFrame:
    """One row per AL loop (loop number in the "al_loop" column) with the best committee
    member's metrics -- the same member MD uses as its base model.

    Loops with checkpoint evaluations (``evaluation_metrics.json``) report
    the best fit's final ``"test"`` metrics (``metric_source=
    "checkpoint_test"``, see ``mlip/evaluation.py``'s
    ``best_fit_test_metrics``). Older loops with only MACE's ``*train.txt``
    logs fall back to the fit with the lowest logged ``mae_f``
    (``metric_source="legacy_training_validation"``, logged as a warning).
    """
    name = mlip_committee_job_dict["name"]
    al_loop_dirs = sorted(
        Path("results").glob("al_loop_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    )
    rows = []
    loops = []
    for al_loop_dir in al_loop_dirs:
        committee_dir = al_loop_dir / name
        metric_files = sorted(committee_dir.glob("fit_*/evaluation_metrics.json"))
        if metric_files:
            expected = mlip_committee_job_dict.get(
                "num_of_models_in_committee", len(metric_files)
            )
            expected_dirs = {f"fit_{i}" for i in range(expected)}
            if {p.parent.name for p in metric_files} != expected_dirs:
                raise RuntimeError(
                    "Missing checkpoint evaluations for committee members"
                )
            row = best_fit_test_metrics(committee_dir)
            if row is None:
                continue
            row["metric_source"] = "checkpoint_test"
        else:
            if mlip_committee_job_dict.get("require_checkpoint_metrics", False):
                raise RuntimeError(
                    f"{al_loop_dir}: checkpoint evaluations are required; training logs are insufficient"
                )
            legacy = []
            for results_file in sorted(committee_dir.glob("fit_*/results/*train.txt")):
                with open(results_file) as file:
                    result = dict(ast.literal_eval(file.readlines()[-1]))
                fit_idx = int(results_file.parent.parent.name.rsplit("_", 1)[1])
                legacy.append((fit_idx, result))
            if not legacy:
                continue
            best_fit, best = min(legacy, key=lambda item: float(item[1]["mae_f"]))
            row = {
                "mae_f": float(best["mae_f"]),
                "mae_e_per_atom": float(best["mae_e_per_atom"]),
                "best_fit_idx": best_fit,
                "metric_source": "legacy_training_validation",
            }
            logger.warning(
                "%s: using legacy training-time validation metrics, not final test metrics",
                al_loop_dir,
            )
        rows.append(row)
        loops.append(int(al_loop_dir.name.rsplit("_", 1)[1]))
    return loop_metrics_frame(loops, rows)


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
    parameter only so existing call sites (the checkpoint-evaluation test
    suite) don't need to change. The workflows themselves now pick their
    best model with ``mlip/evaluation.rank_committee`` directly.
    """
    name = mlip_committee_job_dict["name"]
    n_fits = mlip_committee_job_dict["num_of_models_in_committee"]
    committee_dir = Path("results", base_name, name)
    fit_dirs = {i: committee_dir / f"fit_{i}" for i in range(n_fits)}

    best_fit, model_path, split_used = rank_committee(
        fit_dirs, metric=metric, label=f"select_best_committee_model for {base_name!r}"
    )
    if split_used == "test":
        logger.info(
            "No committee member has a 'valid' checkpoint evaluation "
            "(expected when the eligible pool is too small for a "
            "validation split) — fell back to the 'test' split."
        )
    logger.info("Best committee member: fit_%d (%s %s).", best_fit, split_used, metric)
    return best_fit, model_path
