"""Render a loop's stats, sections and findings as Markdown. Plot paths are
written relative to the report file (``plots/<name>.png``)."""

from pathlib import Path

from alomancy.analysis.report.plots import STATUS_COLORS, STATUS_LABELS
from alomancy.analysis.report.rules import Finding
from alomancy.analysis.report.sections import Section, _fmt

_ICONS = {"error": "✖", "warning": "⚠", "info": "💡"}

# Short descriptions for the event table; unknown codes show as-is.
EVENT_DESCRIPTIONS = {
    "jobs_started": "remote jobs started",
    "job_failed": "remote job failed",
    "job_died": "remote job died (killed / lost)",
    "job_resumed": "transient remote error, polling resumed",
    "job_resubmitted": "died job resubmitted",
    "md_no_steps": "MD runs replaced (no MD step)",
    "md_unfinished": "MD runs still failing after replacement",
    "md_summary": "MD summary",
    "short_bond_excluded": "structures dropped for a short bond",
    "quality_filtered": "train/test quality filter passes",
    "redundancy_flagged": "redundancy removal passes",
    "fit_retry": "model fits retried",
    "fit_missing": "model fits missing after retry",
    "fewer_candidates": "fewer candidates than requested",
    "predictions_unreadable": "prediction files with unreadable structures",
    "few_candidates_generated": "fewer than 2x per-loop candidates generated",
    "few_candidates_configured": "to-generate below 2x per-loop (config)",
    "novelty_inexact": "novelty tolerance not exact",
    "novelty_selected": "novelty selection",
    "dft_summary": "DFT batch summary",
}


def _mae_f(stats: dict, split: str = "test") -> float | None:
    errors = (stats.get("model") or {}).get("errors") or {}
    return (errors.get(split) or {}).get("mae_f")


def _mae_e_mev(stats: dict, split: str = "test") -> float | None:
    errors = (stats.get("model") or {}).get("errors") or {}
    value = (errors.get(split) or {}).get("mae_e_per_atom")
    return None if value is None else value * 1000


def _change(now: float | None, before: float | None) -> str:
    if now is None or before is None:
        return ""
    if before == 0:
        return f"{now - before:+.3g}"
    return f"{(now - before) / abs(before):+.0%}"


def _warnings(stats: dict) -> int:
    levels = stats["events"]["by_level"]
    return sum(levels.get(k, 0) for k in ("WARNING", "ERROR", "CRITICAL"))


def _image(path: Path, alt: str | None = None) -> str:
    alt = alt or path.stem.replace("_", " ").capitalize()
    return f"![{alt}](plots/{path.name})"


def _headline(stats: dict, previous: dict | None) -> list[str]:
    totals = stats["dataset"]["totals"]
    rows = [
        (
            "Best model test force MAE (eV/Å)",
            _mae_f(stats),
            _mae_f(previous) if previous else None,
        ),
        (
            "Best model test energy MAE (meV/atom)",
            _mae_e_mev(stats),
            _mae_e_mev(previous) if previous else None,
        ),
        (
            "Training structures",
            totals.get("train"),
            (previous or {}).get("dataset", {}).get("totals", {}).get("train"),
        ),
        (
            "Test structures",
            totals.get("test"),
            (previous or {}).get("dataset", {}).get("totals", {}).get("test"),
        ),
        (
            "New structures this loop",
            sum(stats["dataset"]["new_this_loop"].values()),
            sum((previous or {}).get("dataset", {}).get("new_this_loop", {}).values())
            if previous
            else None,
        ),
        (
            "Warnings / errors logged",
            _warnings(stats),
            _warnings(previous) if previous else None,
        ),
    ]
    prev_name = f"loop {previous['loop']}" if previous else "previous"
    lines = [
        f"| | {prev_name} | loop {stats['loop']} | change |",
        "|---|---:|---:|---:|",
    ]
    for label, now, before in rows:
        lines.append(
            f"| {label} | {_fmt(before)} | {_fmt(now)} | {_change(now, before)} |"
        )
    model = stats.get("model") or {}
    if model.get("best_fit_idx") is not None:
        lines.append("")
        lines.append(
            f"Best model: fit_{model['best_fit_idx']} of {model.get('n_fits')} "
            f"(chosen on the `{model.get('selected_on_split')}` split)."
        )
    return lines


def _trends(all_stats: list[dict]) -> list[str]:
    lines = [
        "| loop | test F-MAE (eV/Å) | train | new | DFT returned | GO converged | new redundant | warnings |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_stats:
        dft = s["dft"]
        go = (
            f"{(dft['n_go'] - dft['go_not_converged']) / dft['n_go']:.0%}"
            if dft.get("n_go")
            else "n/a"
        )
        submitted = dft.get("submitted")
        returned = (
            f"{dft['returned']}/{submitted}" if submitted else str(dft["returned"])
        )
        new = s["dataset"]["new_this_loop"]
        lines.append(
            f"| {s['loop']} | {_fmt(_mae_f(s))} | {s['dataset']['totals'].get('train', 0)} "
            f"| {sum(new.values())} | {returned} | {go} | {new.get('redundant', 0)} "
            f"| {_warnings(s)} |"
        )
    return lines


def _dataset(stats: dict) -> list[str]:
    dataset = stats["dataset"]
    statuses = [s for s in STATUS_COLORS if s in dataset["totals"]]
    lines = [
        "| config_type | "
        + " | ".join(STATUS_LABELS[s] for s in statuses)
        + " | total |",
        "|---|" + "---:|" * (len(statuses) + 1),
    ]
    for config_type, counts in dataset["composition"].items():
        cells = " | ".join(str(counts.get(s, 0)) for s in statuses)
        lines.append(f"| {config_type} | {cells} | {sum(counts.values())} |")
    total_cells = " | ".join(f"**{dataset['totals'].get(s, 0)}**" for s in statuses)
    lines.append(f"| **all** | {total_cells} | **{sum(dataset['totals'].values())}** |")
    lines.append("")
    new = dataset["new_this_loop"]
    lines.append(
        f"- Redundancy removal: {dataset['redundant_total']} structure(s) flagged in "
        f"total, {new.get('redundant', 0)} of this loop's {sum(new.values())} new."
    )
    if dataset["quality_filter_reasons"]:
        reasons = ", ".join(
            f"{k}: {v}" for k, v in dataset["quality_filter_reasons"].items()
        )
        lines.append(f"- Quality filters: {reasons}.")
    return lines


def _dft(stats: dict) -> list[str]:
    dft = stats["dft"]
    if not dft["returned"] and not dft.get("submitted"):
        return ["No DFT this loop."]
    submitted = dft.get("submitted")
    lines = [
        f"- Structures: {dft['returned']} returned"
        + (f" of {submitted} submitted" if submitted else "")
        + f" ({dft['n_go']} relaxed, {dft['n_sp']} single point)."
    ]
    if dft["n_go"]:
        steps = dft.get("go_steps")
        lines.append(
            f"- Geometry optimisation: {dft['n_go'] - dft['go_not_converged']}/{dft['n_go']} "
            f"reached the force ceiling ({_fmt(dft.get('force_ceiling'))} eV/Å)"
            + (
                f"; steps mean {steps['mean']:.1f}, median {steps['median']:.0f}, "
                f"max {steps['max']:.0f} of {dft.get('go_step_budget')} allowed."
                if steps
                else "; step counts not recorded."
            )
        )
    for kind, summary in (dft.get("wall_time_s") or {}).items():
        if summary:
            lines.append(
                f"- Average DFT time per {'relaxation' if kind == 'go' else 'single point'}: "
                f"{summary['mean'] / 60:.1f} min (median {summary['median'] / 60:.1f}, "
                f"max {summary['max'] / 60:.1f}; {summary['n']} "
                f"structure{'s' if summary['n'] != 1 else ''})."
            )
    if not dft.get("wall_time_s"):
        lines.append("- Per-structure DFT time not recorded for this loop.")
    if dft.get("max_force"):
        mf = dft["max_force"]
        lines.append(
            f"- Largest per-atom force after DFT: median {mf['median']:.2f}, "
            f"max {mf['max']:.2f} eV/Å."
        )
    timing = stats.get("timing")
    if timing and timing.get("high_accuracy_evaluation_s"):
        queued = timing.get("high_accuracy_evaluation_queue_s")
        lines.append(
            f"- DFT phase wall-clock: {timing['high_accuracy_evaluation_s'] / 3600:.1f} h"
            + (f" (mean queue time {queued / 60:.0f} min)" if queued else "")
            + "."
        )
    return lines


def _events(stats: dict) -> list[str]:
    events = stats["events"]
    if not events["by_event"] and not events["uncoded_warnings"]:
        return ["No warnings or coded events logged for this loop."]
    lines = []
    if events["by_event"]:
        lines += ["| event | meaning | count |", "|---|---|---:|"]
        for code, count in sorted(events["by_event"].items()):
            lines.append(f"| `{code}` | {EVENT_DESCRIPTIONS.get(code, '')} | {count} |")
    if events["uncoded_warnings"]:
        lines += ["", "Other warnings (most frequent first):", ""]
        for item in events["uncoded_warnings"]:
            lines.append(f"- ({item['count']}x) {item['message']}")
    return lines


def _findings(findings: list[Finding], notes: list[str]) -> list[str]:
    if not findings:
        lines = ["No issues detected."]
    else:
        lines = [
            f"- {_ICONS[f.severity]} **{f.trigger}**: {f.message}" for f in findings
        ]
    if notes:
        lines += ["", "<details><summary>Suggestions file notes</summary>", ""]
        lines += [f"- {n}" for n in notes]
        lines += ["", "</details>"]
    return lines


def _status(stats: dict, findings: list[Finding]) -> str:
    counts = {s: sum(1 for f in findings if f.severity == s) for s in _ICONS}
    parts = [
        f"{_ICONS[s]} {n} {s}{'s' if n != 1 else ''}" for s, n in counts.items() if n
    ]
    return (
        "**Status:** "
        + (" · ".join(parts) if parts else "✓ no issues")
        + (f" · {_warnings(stats)} warning(s) logged")
    )


def render_markdown(
    stats: dict,
    *,
    previous: dict | None,
    all_stats: list[dict],
    core_plots: dict[str, Path],
    sections: list[Section],
    findings: list[Finding],
    notes: list[str],
) -> str:
    modules = stats["modules"]
    out: list[str] = [
        f"# ALomancy report: {stats['base_name']}",
        "",
        f"Workflow `{stats['workflow']}` · trainer `{modules['trainer']}` · "
        f"generator `{modules['generator']}` · evaluator `{modules['evaluator']}` · "
        f"generated {stats['generated']}",
        "",
        _status(stats, findings),
        "",
        "## Headline",
        "",
        *_headline(stats, previous),
        "",
        "## Issues & suggestions",
        "",
        *_findings(findings, notes),
        "",
        "## Trends",
        "",
        *_trends(all_stats),
        "",
    ]
    for key, alt in (
        ("mae", "Best-model MAE per loop"),
        ("timing", "Time per phase and loop"),
    ):
        if key in core_plots:
            out += [_image(core_plots[key], alt), ""]
    out += ["## Best model", ""]
    if "parity" in core_plots:
        out += [_image(core_plots["parity"], "Best model parity"), ""]
    else:
        out += ["No predictions stored for this loop's best model.", ""]
    out += ["## Training set", "", *_dataset(stats), ""]
    if "composition" in core_plots:
        out += [_image(core_plots["composition"], "Training set composition"), ""]
    out += ["## DFT", "", *_dft(stats), ""]
    for section in sections:
        out += [f"## {section.title}", ""]
        if section.lines:
            out += [*section.lines, ""]
        for plot in section.plots:
            out += [_image(plot), ""]
    out += ["## Warnings and events", "", *_events(stats), ""]
    return "\n".join(out)


def relink_for_latest(markdown: str, base_name: str) -> str:
    """The same report with plot links pointing at <base_name>/plots/, for
    results/reports/latest.md."""
    return markdown.replace("](plots/", f"]({base_name}/plots/")
