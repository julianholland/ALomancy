"""Per-loop AL report: Markdown plus PNGs under results/reports/.

``write_loop_report(workflow, loop)`` writes
``results/reports/al_loop_N/{report.md, report.json, plots/}`` and
refreshes ``results/reports/latest.md``. Called from
ActiveLearningWorkflow.finish_loop (general.report) and from
``alomancy report``. See docs/reports.md.

Parts:
- stats.py:     numbers for the loop (saved as report.json)
- plots.py:     the report's own figures
- sections.py:  module-specific sections (``report_section`` entry points)
                and workflow sections (``ActiveLearningWorkflow.report_sections``)
- triggers.py + suggestions.yaml + rules.py: issues and suggested fixes
- render.py:    Markdown
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alomancy.analysis.report.render import relink_for_latest, render_markdown
from alomancy.analysis.report.rules import evaluate, load_suggestions
from alomancy.analysis.report.sections import Section
from alomancy.analysis.report.stats import (
    collect_loop_stats,
    load_all_stats,
    save_stats,
)

__all__ = ["REPORTS_DIR", "Section", "write_loop_report"]

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("results", "reports")

# Generator/evaluator modules expose report_section as a registry entry
# point; the trainer is a class (mlip/base.ALomancyTrainer) and has it as a
# method.
_MODULE_CATEGORIES = (
    ("structure_generator", "generator"),
    ("dft_evaluator", "evaluator"),
)


def _attempt(what: str, fn: Callable[[], Any]) -> Any:
    """Run one report part; a failure is logged and skipped, never raised."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("Loop report: %s failed: %s", what, exc, exc_info=True)
        return None


def _core_plots(workflow: Any, stats: dict, plots_dir: Path) -> dict[str, Path]:
    from alomancy.analysis.plotting import mae_al_loop_plot
    from alomancy.analysis.report import plots
    from alomancy.analysis.timing_plots import timing_plots

    made: dict[str, Path] = {}

    def mae() -> None:
        frame = workflow._cross_loop_metrics_dataframe("training")
        if not frame.is_empty():
            # A regenerated report for an old loop shows history up to it.
            frame = frame.filter(frame["al_loop"] <= stats["loop"])
        if frame.is_empty():
            return
        mae_al_loop_plot(frame, {"name": "training"}, directory=plots_dir)
        made["mae"] = plots_dir / "training_al_loop_mae_plot.png"

    def timing() -> None:
        if workflow.log_file and Path(workflow.log_file).exists():
            timing_plots(workflow.log_file, plots_dir)
            if (plots_dir / "timing_combined.png").exists():
                made["timing"] = plots_dir / "timing_combined.png"

    def parity() -> None:
        from alomancy.analysis.mlip_plots import parity_predictions

        model = stats.get("model") or {}
        fit_idx = model.get("best_fit_idx")
        if fit_idx is None:
            return
        e0 = workflow.db.get_isolated_atom_energies() or None
        fit_dir = Path("results", stats["base_name"], "training", f"fit_{fit_idx}")
        predictions = parity_predictions(
            workflow.db, stats["loop"], fit_idx, fit_dir, e0=e0
        )
        # Read by render_markdown, to say when the test set is missing.
        model["parity_splits"] = sorted(predictions)
        if not predictions:
            return
        path = plots_dir / "best_model_parity.png"
        plots.parity(
            predictions,
            path,
            title=f"Best model (fit_{fit_idx}) [{stats['base_name']}]",
            energy_label="formation energy" if e0 else "energy",
        )
        made["parity"] = path

    def redundancy() -> None:
        from alomancy.analysis.report.stats import read_redundancy_probe

        record = read_redundancy_probe(stats["base_name"])
        if not record or not record.get("tolerances"):
            return
        path = plots_dir / "redundancy_tolerance_probe.png"
        plots.redundancy_probe(
            record,
            path,
            title=f"Redundancy tolerance probe [{stats['base_name']}]",
        )
        made["redundancy"] = path

    def composition() -> None:
        if not stats["dataset"]["composition"]:
            return
        path = plots_dir / "training_set_composition.png"
        plots.composition(
            stats["dataset"]["composition"],
            path,
            title=f"Dataset by config_type [{stats['base_name']}]",
        )
        made["composition"] = path

    for name, fn in (
        ("MAE plot", mae),
        ("timing plot", timing),
        ("best-model parity plot", parity),
        ("redundancy probe plot", redundancy),
        ("composition plot", composition),
    ):
        _attempt(name, fn)
    return made


def _sections(workflow: Any, stats: dict, plots_dir: Path | None) -> list[Section]:
    from alomancy.mlip.base import get_trainer
    from alomancy.registry import resolve

    sections: list[Section] = []
    trainer_section = _attempt(
        f"{stats['modules']['trainer']} report section",
        lambda: get_trainer(
            stats["modules"]["trainer"], workflow.jobs_dict.get("training", {})
        ).report_section(
            stats,
            base_name=stats["base_name"],
            plots_dir=plots_dir,
            config=workflow.jobs_dict,
        ),
    )
    if trainer_section is not None:
        sections.append(trainer_section)
    for category, key in _MODULE_CATEGORIES:
        name = stats["modules"][key]
        entry = _attempt(
            f"resolving {category} {name!r}", lambda c=category, n=name: resolve(c, n)
        )
        if entry is None or not hasattr(entry, "report_section"):
            continue
        section = _attempt(
            f"{name} report section",
            lambda e=entry: e.report_section(
                stats,
                base_name=stats["base_name"],
                plots_dir=plots_dir,
                config=workflow.jobs_dict,
            ),
        )
        if section is not None:
            sections.append(section)
    own = _attempt(
        f"{workflow.NAME} report sections",
        lambda: workflow.report_sections(stats, plots_dir),
    )
    sections.extend(own or [])
    return sections


def write_loop_report(workflow: Any, loop: int) -> Path:
    """Write AL loop *loop*'s report for *workflow*; returns report.md's path."""
    base_name = f"al_loop_{loop}"
    loop_dir = REPORTS_DIR / base_name
    plots_dir = loop_dir / "plots"
    stats = collect_loop_stats(workflow, loop)
    save_stats(stats, loop_dir / "report.json")

    all_stats = [s for s in load_all_stats(REPORTS_DIR) if s.get("loop", 0) <= loop]
    previous = next((s for s in reversed(all_stats) if s.get("loop", 0) < loop), None)

    if workflow.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        core_plots = _core_plots(workflow, stats, plots_dir)
    else:
        core_plots = {}
    sections = _sections(workflow, stats, plots_dir if workflow.plots else None)
    findings, notes = evaluate(stats, load_suggestions(workflow.report_suggestions))

    markdown = render_markdown(
        stats,
        previous=previous,
        all_stats=all_stats,
        core_plots=core_plots,
        sections=sections,
        findings=findings,
        notes=notes,
    )
    report = loop_dir / "report.md"
    report.write_text(markdown, encoding="utf-8")
    latest_loop = max(
        (s.get("loop", 0) for s in load_all_stats(REPORTS_DIR)), default=loop
    )
    if loop >= latest_loop:
        (REPORTS_DIR / "latest.md").write_text(
            relink_for_latest(markdown, base_name), encoding="utf-8"
        )
    logger.info(
        "Loop report written to %s (%d issue(s) flagged).", report, len(findings)
    )
    return report
