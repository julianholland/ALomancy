"""Triggers: what the loop report measures to decide whether to suggest
anything. One small function per trigger, registered by name.

A trigger takes a loop's stats dict (analysis/report/stats.py) and returns
None when it doesn't apply this loop (e.g. no geometry optimisation ran),
or a dict of named values with at least ``value`` -- the number compared
against the ``threshold`` in suggestions.yaml. Every other key is
available to the suggestion text as ``{name}``.

Adding a trigger: write a function here decorated with
``@trigger("name")`` and add a ``name:`` entry to suggestions.yaml
(threshold, severity, suggestion). Wording and thresholds live only in the
YAML so they can be tuned without touching this file.
"""

from collections.abc import Callable
from typing import Any

TriggerFn = Callable[[dict], dict[str, Any] | None]

TRIGGERS: dict[str, TriggerFn] = {}


def trigger(name: str) -> Callable[[TriggerFn], TriggerFn]:
    def register(fn: TriggerFn) -> TriggerFn:
        TRIGGERS[name] = fn
        return fn

    return register


def _event_totals(
    stats: dict, code: str, n_key: str = "n", total_key: str = "total"
) -> tuple[int, int]:
    rows = stats["events"]["data"].get(code) or []
    return (
        sum(int(r.get(n_key) or 0) for r in rows),
        sum(int(r.get(total_key) or 0) for r in rows),
    )


def _fraction(n: int, total: int, **extra: Any) -> dict[str, Any] | None:
    if total <= 0:
        return None
    return {"value": n / total, "n": n, "total": total, **extra}


@trigger("go_not_converged")
def go_not_converged(stats: dict) -> dict | None:
    """Fraction of geometry optimisations that stopped at the step budget."""
    dft = stats["dft"]
    return _fraction(
        dft["go_not_converged"],
        dft["n_go"],
        max_steps=dft.get("go_step_budget") or "?",
        force_ceiling=dft.get("force_ceiling"),
    )


@trigger("go_steps_near_budget")
def go_steps_near_budget(stats: dict) -> dict | None:
    """Mean geometry-optimisation steps as a fraction of the step budget."""
    dft = stats["dft"]
    steps, budget = dft.get("go_steps"), dft.get("go_step_budget")
    if not steps or not budget:
        return None
    return {
        "value": steps["mean"] / budget,
        "mean_steps": steps["mean"],
        "max_steps": budget,
    }


@trigger("new_structures_force_filtered")
def new_structures_force_filtered(stats: dict) -> dict | None:
    """Fraction of this loop's new structures excluded by the train
    filter's max_force."""
    dataset = stats["dataset"]
    n = int(dataset.get("new_quality_filter_reasons", {}).get("high_force", 0))
    total = sum(dataset.get("new_this_loop", {}).values())
    return _fraction(
        n,
        total,
        max_force=(stats.get("train_filter") or {}).get("max_force"),
        force_ceiling=stats["dft"].get("force_ceiling"),
    )


@trigger("short_bond_excluded")
def short_bond_excluded(stats: dict) -> dict | None:
    """Fraction of structures dropped for a bond shorter than 0.5 Å."""
    return _fraction(*_event_totals(stats, "short_bond_excluded"))


@trigger("md_runs_no_steps")
def md_runs_no_steps(stats: dict) -> dict | None:
    """Fraction of MD runs that completed no MD step."""
    return _fraction(*_event_totals(stats, "md_no_steps"))


@trigger("remote_jobs_failed")
def remote_jobs_failed(stats: dict) -> dict | None:
    """Fraction of remote jobs (training, MD, DFT) that failed or died."""
    events = stats["events"]
    failed = events["by_event"].get("job_failed", 0)
    died = events["by_event"].get("job_died", 0)
    started = sum(
        int(r.get("n") or 0) for r in events["data"].get("jobs_started") or []
    )
    return _fraction(failed + died, started, failed=failed, died=died)


@trigger("fewer_candidates")
def fewer_candidates(stats: dict) -> dict | None:
    """Shortfall of candidates against general.num_of_structures_per_loop."""
    rows = stats["events"]["data"].get("fewer_candidates") or []
    if not rows:
        return None
    row = rows[-1]
    desired = int(row.get("desired") or 0)
    if desired <= 0:
        return None
    n = int(row.get("n") or 0)
    return {"value": 1 - n / desired, "n": n, "desired": desired}


@trigger("few_candidates_generated")
def few_candidates_generated(stats: dict) -> dict | None:
    """Shortfall of generated candidates against twice general.
    num_of_structures_per_loop (the selector's minimum useful pool)."""
    rows = stats["events"]["data"].get("few_candidates_generated") or []
    if not rows:
        return None
    row = rows[-1]
    per_loop = int(row.get("per_loop") or 0)
    if per_loop <= 0:
        return None
    n = int(row.get("n") or 0)
    return {"value": 1 - n / (2 * per_loop), "n": n, "per_loop": per_loop}


@trigger("redundancy_removed_new")
def redundancy_removed_new(stats: dict) -> dict | None:
    """Fraction of this loop's new structures flagged as redundant."""
    new = stats["dataset"].get("new_this_loop", {})
    return _fraction(int(new.get("redundant", 0)), sum(new.values()))


@trigger("queue_dominated")
def queue_dominated(stats: dict) -> dict | None:
    """Largest share of a phase's wall-clock time spent queued."""
    timing = stats.get("timing")
    if not timing:
        return None
    worst = None
    for phase in ("training_plots", "generate_structures", "high_accuracy_evaluation"):
        total, queued = timing.get(f"{phase}_s"), timing.get(f"{phase}_queue_s")
        if total and queued is not None and total > 0:
            share = min(queued / total, 1.0)
            if worst is None or share > worst["value"]:
                worst = {"value": share, "phase": phase, "queue_hours": queued / 3600}
    return worst


@trigger("fit_retries")
def fit_retries(stats: dict) -> dict | None:
    """Fraction of model fits that had to be retried."""
    return _fraction(*_event_totals(stats, "fit_retry"))


@trigger("overfitting")
def overfitting(stats: dict) -> dict | None:
    """Best model's test force MAE relative to its training force MAE."""
    errors = (stats.get("model") or {}).get("errors") or {}
    train = (errors.get("train") or {}).get("mae_f")
    test = (errors.get("test") or {}).get("mae_f")
    if not train or test is None:
        return None
    return {"value": test / train, "train_mae_f": train, "test_mae_f": test}


@trigger("dft_partial_failure")
def dft_partial_failure(stats: dict) -> dict | None:
    """Fraction of structures sent to DFT that came back without a result."""
    dft = stats["dft"]
    submitted = dft.get("submitted")
    if not submitted:
        return None
    missing = max(int(submitted) - int(dft["returned"]), 0)
    return _fraction(missing, int(submitted))
