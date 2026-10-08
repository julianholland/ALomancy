"""Collect one AL loop's statistics into a plain, JSON-safe dict (saved as
report.json). Triggers, the trends table and ``alomancy report`` all read
these dicts, so everything here is a number, string, list or dict.

Sources: the GlobalDatabase (composition, redundancy, quality filters), the
loop's high_accuracy_eval_results.xyz (per-structure DFT data), each fit's
evaluation_metrics.json (model errors), alomancy.log (phase timings, via
parse_timing_log) and events.jsonl (warnings and coded events, see
utils/logging_config.JsonlEventHandler).
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from alomancy.analysis.report.current_events import current_events
from alomancy.utils.logging_config import EVENTS_FILENAME, read_events

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def load_dft_results(base_name: str) -> list[Atoms]:
    """This loop's DFT-labelled structures ([] if the loop has none yet)."""
    path = Path("results", base_name, "high_accuracy_eval_results.xyz")
    if not path.exists():
        return []
    try:
        return list(read(path, ":", format="extxyz"))
    except Exception as exc:
        logger.warning("Could not read %s for the loop report: %s", path, exc)
        return []


def _summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _max_force(atoms: Atoms) -> float | None:
    forces = atoms.arrays.get("REF_forces")
    if forces is None:
        return None
    return float(np.linalg.norm(np.asarray(forces), axis=1).max())


REDUNDANCY_PROBE_FILENAME = "redundancy_probe.json"


def read_redundancy_probe(base_name: str) -> dict[str, Any] | None:
    """The loop's saved redundancy tolerance probe (utils/remove_redundancy
    .probe_redundancy_tolerance), or None if this loop didn't save one."""
    path = Path("results", base_name, REDUNDANCY_PROBE_FILENAME)
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _redundancy_probe_stats(base_name: str) -> dict[str, Any] | None:
    """The probe's outcome, without the sweep itself (that only feeds the
    plot)."""
    record = read_redundancy_probe(base_name)
    if record is None:
        return None
    return {
        key: record.get(key)
        for key in (
            "n_structures",
            "n_flagged",
            "chosen_tolerance",
            "outcome",
            "plateaus",
            "descriptor",
        )
    }


def _model_stats(base_name: str) -> dict[str, Any] | None:
    """The loop's best fit (rank_committee, as for MD) and its per-split
    errors from evaluation_metrics.json."""
    from alomancy.mlip.evaluation import rank_committee

    committee_dir = Path("results", base_name, "training")
    fit_dirs = {
        int(p.parent.name.rsplit("_", 1)[1]): p.parent
        for p in committee_dir.glob("fit_*/evaluation_metrics.json")
    }
    if not fit_dirs:
        return None
    n_fits = len(list(committee_dir.glob("fit_*")))
    try:
        best_fit, _, split_used = rank_committee(fit_dirs, label=base_name)
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.warning("Loop report: no best model for %s: %s", base_name, exc)
        return {"n_fits": n_fits, "n_fits_evaluated": len(fit_dirs)}
    record = json.loads((fit_dirs[best_fit] / "evaluation_metrics.json").read_text())
    errors = {}
    for split, result in record.get("splits", {}).items():
        if result.get("complete"):
            errors[split] = {
                k: result[k]
                for k in ("n_structures", "mae_e_per_atom", "mae_f", "mae_stress")
                if k in result
            }
    return {
        "best_fit_idx": best_fit,
        "selected_on_split": split_used,
        "n_fits": n_fits,
        "n_fits_evaluated": len(fit_dirs),
        "errors": errors,
    }


def _dataset_stats(db: Any, loop: int) -> dict[str, Any]:
    """Composition by config_type and status, plus what redundancy removal
    and the quality filters took out. Flags are the database's current
    ones; structures from later loops (al_loop > loop) are left out so a
    regenerated report for an old loop counts only what existed then."""
    composition: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    new: Counter = Counter()
    new_reasons: Counter = Counter()
    flagged_total = 0
    for container in db.partition.list_containers():
        meta = container.AtomPositionManager.metadata
        al_loop = meta.get("al_loop")
        if al_loop is not None and int(al_loop) > loop:
            continue
        if meta.get("is_duplicate", False):
            status = "redundant"
            flagged_total += 1
        elif meta.get("is_quality_filtered", False):
            status = "quality_filtered"
            for reason in meta.get("quality_filter_reasons") or []:
                reasons[reason] += 1
                if al_loop is not None and int(al_loop) == loop:
                    new_reasons[reason] += 1
        else:
            status = meta.get("split") or "unsplit"
        composition[meta.get("config_type", "unknown")][status] += 1
        if al_loop is not None and int(al_loop) == loop:
            new[status] += 1
    totals: Counter = Counter()
    for counts in composition.values():
        totals.update(counts)
    return {
        "composition": {k: dict(v) for k, v in sorted(composition.items())},
        "totals": dict(totals),
        "quality_filter_reasons": dict(reasons),
        "redundant_total": flagged_total,
        "new_this_loop": dict(new),
        "new_quality_filter_reasons": dict(new_reasons),
    }


def _dft_stats(base_name: str, events: dict, force_ceiling: float | None) -> dict:
    results = load_dft_results(base_name)
    summaries = events["data"].get("dft_summary") or []
    submitted = summaries[-1].get("submitted") if summaries else None
    go = [a for a in results if "geometry_converged" in a.info]
    steps = [int(a.info["geometry_steps"]) for a in go if "geometry_steps" in a.info]
    budgets = [
        int(a.info["geometry_max_steps"]) for a in go if "geometry_max_steps" in a.info
    ]
    by_type: dict[str, list[float]] = defaultdict(list)
    for a in results:
        if "dft_wall_time_s" in a.info:
            kind = "go" if "geometry_converged" in a.info else "sp"
            by_type[kind].append(float(a.info["dft_wall_time_s"]))
    forces = [f for f in (_max_force(a) for a in results) if f is not None]
    return {
        "submitted": submitted,
        "returned": len(results),
        "n_go": len(go),
        "n_sp": len(results) - len(go),
        "go_not_converged": sum(1 for a in go if not a.info["geometry_converged"]),
        "go_steps": _summary(steps),
        "go_step_budget": max(budgets) if budgets else None,
        "wall_time_s": {k: _summary(v) for k, v in by_type.items()},
        "max_force": _summary(forces),
        "force_ceiling": force_ceiling,
    }


def _events_for_loop(log_file: str | None, loop: int) -> dict[str, Any]:
    events_path = (
        Path(log_file).with_name(EVENTS_FILENAME)
        if log_file
        else Path("results", EVENTS_FILENAME)
    )
    by_event: Counter = Counter()
    by_level: Counter = Counter()
    data: dict[str, list] = defaultdict(list)
    uncoded: Counter = Counter()
    # Only what still stands: a step's latest attempt, minus failures whose
    # retry succeeded (current_events.py).
    log_path = Path(log_file) if log_file else Path("results", "alomancy.log")
    for event in current_events(read_events(events_path), loop, log_path):
        by_level[event.get("level", "?")] += 1
        code = event.get("event")
        if code:
            by_event[code] += 1
            if event.get("data") is not None:
                data[code].append(event["data"])
        elif event.get("level") in ("WARNING", "ERROR", "CRITICAL"):
            uncoded[event.get("message", "")[:200]] += 1
    return {
        "by_event": dict(by_event),
        "by_level": dict(by_level),
        "data": dict(data),
        "uncoded_warnings": [
            {"message": m, "count": c} for m, c in uncoded.most_common(10)
        ],
    }


def _timing_for_loop(log_file: str | None, loop: int) -> dict[str, Any] | None:
    if not log_file or not Path(log_file).exists():
        return None
    from alomancy.analysis.timing_plots import parse_timing_log

    df = parse_timing_log(log_file)
    if df.is_empty():
        return None
    rows = df.filter(df["loop"] == loop)
    if rows.is_empty():
        return None
    row = rows.row(0, named=True)
    return {
        k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()
    }


def collect_loop_stats(workflow: Any, loop: int) -> dict[str, Any]:
    """Everything the report needs for AL loop *loop* of *workflow*."""
    base_name = f"al_loop_{loop}"
    jobs = workflow.jobs_dict
    events = _events_for_loop(workflow.log_file, loop)
    return {
        "schema_version": SCHEMA_VERSION,
        "loop": loop,
        "base_name": base_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "workflow": workflow.NAME,
        "seed": workflow.seed,
        "modules": {
            "trainer": jobs.get("training", {}).get("trainer", "mace"),
            "generator": jobs.get("structure_generation", {}).get("generator", "md"),
            "evaluator": jobs.get("high_accuracy_evaluation", {}).get(
                "evaluator", "qe"
            ),
        },
        "num_of_structures_per_loop": workflow.num_of_structures_per_loop,
        "num_of_structures_to_generate": jobs.get("structure_generation", {}).get(
            "num_of_structures_to_generate"
        ),
        "train_filter": workflow.train_filter,
        "model": _model_stats(base_name),
        "dataset": _dataset_stats(workflow.db, loop),
        "redundancy_probe": _redundancy_probe_stats(base_name),
        "dft": _dft_stats(base_name, events, workflow.force_ceiling),
        "timing": _timing_for_loop(workflow.log_file, loop),
        "events": events,
    }


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, default=str) + "\n")


def load_all_stats(reports_dir: Path) -> list[dict]:
    """Every saved report.json under *reports_dir*, ordered by loop."""
    stats = []
    for path in reports_dir.glob("al_loop_*/report.json"):
        try:
            stats.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable %s: %s", path, exc)
    return sorted(stats, key=lambda s: s.get("loop", 0))
