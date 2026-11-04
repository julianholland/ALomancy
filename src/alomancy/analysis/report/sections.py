"""The Section type every report part returns, plus the module-specific
sections registered as ``report_section`` entry points (see registry.py).

A module opts in by defining ``report_section(stats, *, base_name,
plots_dir, config)`` (a thin wrapper that lazily imports from here, so
remote-only modules never import matplotlib) and registering it. Each
returns a Section or None when it has nothing to show for this loop.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """One block of the report: a heading, Markdown lines and plot files."""

    title: str
    lines: list[str] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)


def _fmt(value: Any, spec: str = ".3g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return format(value, spec) if isinstance(value, int | float) else str(value)


def md_section(
    stats: dict, *, base_name: str, plots_dir: Path | None, config: dict
) -> Section | None:
    """MD generator: runs, replacements and candidate frames per run."""
    summaries = stats["events"]["data"].get("md_summary") or []
    if not summaries:
        return None
    summary = summaries[-1]
    frames = summary.get("frames_per_run") or []
    md_kwargs = config.get("structure_generation", {}).get("md_kwargs", {})
    section = Section(
        "Structure generation: MD",
        [
            f"- Runs: {summary.get('runs')} "
            f"({summary.get('completed')} completed, "
            f"{summary.get('replaced', 0)} seed(s) replaced)",
            f"- Candidate frames: {summary.get('candidates')} "
            f"(per completed run: median {_fmt(float(np.median(frames)) if frames else None)}, "
            f"min {min(frames) if frames else 'n/a'})"
            if "valid_per_run" not in summary
            else f"- Frames: {sum(frames)} recorded, "
            f"{sum(summary['valid_per_run'])} physical (no bond < 0.5 Å, seed "
            f"frames left out), {summary.get('candidates')} picked as candidates, "
            "spread over each run",
            f"- Ensemble {md_kwargs.get('ensemble', 'nvt')}, "
            f"T = {md_kwargs.get('temperature', 'default')} K, "
            f"timestep {md_kwargs.get('timestep_fs', 'default')} fs",
        ],
    )
    if frames and plots_dir is not None:
        from alomancy.analysis.report.plots import bar_per_item

        path = plots_dir / "md_frames_per_run.png"
        bar_per_item(
            frames,
            path,
            title=f"Candidate frames per completed MD run [{base_name}]",
            xlabel="Completed MD run",
            ylabel="Frames",
        )
        section.plots.append(path)
    if plots_dir is not None:
        temperature_plot = _md_temperature_plot(base_name, plots_dir, md_kwargs)
        if temperature_plot is not None:
            section.plots.append(temperature_plot)
    return section


def _md_temperature_plot(
    base_name: str, plots_dir: Path, md_kwargs: dict
) -> Path | None:
    """Temperature against step for every MD run, one panel each, from the
    runs' own ASE MD logs (``md_output_<i>/*.log``: one
    row per step, ``T[K]`` last). Runs without a log are skipped."""
    import inspect

    from alomancy.analysis.report.plots import md_runs
    from alomancy.structure_generation.md.md_wfl import _MD_KWARGS_DEFAULTS, run_md

    settings = {**_MD_KWARGS_DEFAULTS, **md_kwargs}
    equilibration = int(settings.get("equilibration_steps") or 0)
    md_dir = Path("results", base_name, "structure_generation")
    temperatures, completed, labels = [], [], []
    for run_dir in sorted(
        md_dir.glob("md_output_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    ):
        logs = sorted(run_dir.glob("*.log"))
        if not logs:
            continue
        try:
            temps = np.atleast_1d(np.loadtxt(logs[0], skiprows=1, usecols=-1))
        except (ValueError, OSError) as exc:
            logger.debug("Could not read MD log %s: %s", logs[0], exc)
            continue
        if not len(temps):
            continue
        temperatures.append(temps)
        labels.append(f"run {run_dir.name.rsplit('_', 1)[1]}")
        # Equilibration (if any) and production each log their step 0.
        logged_before = equilibration + 1 if equilibration else 0
        completed.append(max(0, len(temps) - 1 - logged_before))
    if not temperatures:
        return None
    path = plots_dir / "md_temperature.png"
    md_runs(
        temperatures,
        completed,
        path,
        run_labels=labels,
        title=f"MD temperature per run [{base_name}]",
        target_temperature=float(settings["temperature"]),
        requested_steps=int(settings["steps"]),
        equilibration_steps=equilibration,
        equilibration_temperature=float(
            settings.get(
                "equilibration_temperature",
                inspect.signature(run_md)
                .parameters["equilibration_temperature"]
                .default,
            )
        ),
    )
    return path


def ezga_section(
    stats: dict,
    *,
    base_name: str,
    plots_dir: Path | None,
    config: dict,
) -> Section | None:
    """EZGA generator: seeds, generations, structures generated and the
    candidates taken (ezga_summary), and the main search settings; with
    stats["ezga"], plots of each generation and of the structures selected
    for DFT by seed type, with the operators that made them."""
    summaries = stats["events"]["data"].get("ezga_summary") or []
    if not summaries:
        return None
    summary = summaries[-1]
    ezga_kwargs = config.get("structure_generation", {}).get("ezga_kwargs", {})
    size = (ezga_kwargs.get("multiobjective") or {}).get("size", "n/a")
    mutations = [
        str(m.get("type") or m.get("factory") or m).rsplit(".", 1)[-1]
        if isinstance(m, dict)
        else str(m).rsplit(".", 1)[-1]
        for m in ezga_kwargs.get("mutation_funcs") or []
    ]
    section = Section(
        "Structure generation: EZGA",
        [
            f"- Seeds: {summary.get('seeds')}, generations run: "
            f"{summary.get('generations')} "
            f"(max_generations {ezga_kwargs.get('max_generations', 'n/a')}, "
            f"population {size})",
            f"- Structures generated: {summary.get('generated')}, taken as "
            f"candidates: {summary.get('candidates')} (newest first, back to "
            f"generation {summary.get('oldest_generation_used')})",
            f"- Mutations: {', '.join(mutations) or 'none'}",
        ],
    )
    if plots_dir is not None and stats.get("ezga"):
        section.plots += _ezga_plots(stats["ezga"], base_name, plots_dir)
    return section


def _ezga_plots(ezga: dict, base_name: str, plots_dir: Path) -> list[Path]:
    """Two stacked bar charts: per generation, the structures generated by
    operator with the duplicates EZGA filtered out on top; per seed
    config_type, the structures selected for DFT by operator. Each column's
    total of structures (generated, selected) is written above it."""
    from alomancy.analysis.report.plots import stacked_counts

    # Excluded data and operators without a name in greys.
    greys = {"Duplicates filtered": "#6B7280", "mutation": "#9CA3AF"}

    # Every configured operator (a zero is worth seeing), plus any other
    # name the counts carry ("mutation" for structures without a recorded
    # mutation_list, "mut_N" for an index past the configured list).
    found = {
        name
        for group in [*ezga["by_generation"].values(), *ezga["by_seed_type"].values()]
        for name in group["operators"]
    }
    operators = [*ezga["operator_names"]]
    operators += sorted(found - set(operators))

    def series(groups: dict) -> dict:
        rows = list(groups.values())
        counts = {op: [g["operators"].get(op, 0) for g in rows] for op in operators}
        # On top of the column. Older report.json files have no duplicates;
        # nor do loops whose EZGA logger files are gone.
        if any(g.get("duplicates") is not None for g in rows):
            counts["Duplicates filtered"] = [g.get("duplicates") or 0 for g in rows]
        return counts

    made = []
    generations = ezga["by_generation"]
    path = plots_dir / "ezga_generations.png"
    stacked_counts(
        list(generations),
        series(generations),
        path,
        title=f"EZGA generations: structures generated by operator [{base_name}]",
        xlabel="Generation",
        colors=greys,
        totals=[g["generated"] for g in generations.values()],
    )
    made.append(path)
    seed_types = ezga["by_seed_type"]
    if seed_types:
        path = plots_dir / "ezga_selected_by_seed_type.png"
        stacked_counts(
            list(seed_types),
            series(seed_types),
            path,
            title=f"Structures selected for DFT by seed type [{base_name}]",
            xlabel="Seed config_type",
            colors=greys,
            totals=[g["selected"] for g in seed_types.values()],
        )
        made.append(path)
    # Also next to the loop's other plots.
    current = Path("results", "current_plots", base_name)
    current.mkdir(parents=True, exist_ok=True)
    for path in made:
        shutil.copy2(path, current / path.name)
    return made


def dft_section(
    stats: dict,  # noqa: ARG001 -- uniform report_section signature
    *,
    base_name: str,
    plots_dir: Path | None,
    config: dict,
) -> Section | None:
    """DFT evaluator (QE/VASP): geometry-optimisation steps and wall time."""
    from alomancy.analysis.report.plots import histogram
    from alomancy.analysis.report.stats import load_dft_results

    results = load_dft_results(base_name)
    if not results:
        return None
    evaluator = config.get("high_accuracy_evaluation", {}).get("evaluator", "qe")
    section = Section(f"High-accuracy evaluation: {evaluator.upper()}")
    steps = [a.info["geometry_steps"] for a in results if "geometry_steps" in a.info]
    times = [
        a.info["dft_wall_time_s"] / 60 for a in results if "dft_wall_time_s" in a.info
    ]
    if not steps and not times:
        section.lines.append(
            "- No per-structure step counts or wall times recorded (results "
            "from before this was added, or an evaluator that doesn't record them)."
        )
        return section
    if plots_dir is None:
        return None
    if steps:
        budget = max(
            int(a.info.get("geometry_max_steps", 0))
            for a in results
            if "geometry_steps" in a.info
        )
        path = plots_dir / "dft_geometry_steps.png"
        histogram(
            steps,
            path,
            title=f"Geometry-optimisation steps [{base_name}]",
            xlabel="BFGS steps",
            marker=budget or None,
            marker_label=f"step budget ({budget})",
        )
        section.plots.append(path)
    if times:
        path = plots_dir / "dft_wall_time.png"
        histogram(
            times,
            path,
            title=f"DFT wall time per structure [{base_name}]",
            xlabel="Minutes",
        )
        section.plots.append(path)
    return section


def trainer_section(
    stats: dict,
    *,
    trainer: Any,
    base_name: str,
    plots_dir: Path | None,
    config: dict,  # noqa: ARG001 -- uniform report_section signature
) -> Section | None:
    """Any trainer: the best fit's training length and validation curve,
    from ``trainer.training_history`` (mlip/base.py)."""
    from alomancy.analysis.mlip_plots import _curve_rows
    from alomancy.analysis.report.plots import training_curve

    model = stats.get("model") or {}
    fit_idx = model.get("best_fit_idx")
    if fit_idx is None:
        return None
    fit_dir = Path("results", base_name, trainer.name, f"fit_{fit_idx}")
    seed = (stats.get("seed") or 0) + int(fit_idx)
    history = trainer.training_history(fit_dir, seed)
    if history is None or history.frame.is_empty():
        return None
    section = Section(f"MLIP training: {trainer.NAME}")
    line = (
        f"- Best fit (fit_{fit_idx}) trained for "
        f"{int(history.frame['epoch'].max())} epochs"
    )
    if history.stage_two_epoch is not None:
        line += f", stage two from epoch {history.stage_two_epoch}"
    if history.selected_epoch is not None:
        line += f"; weights kept from epoch {history.selected_epoch}"
    section.lines.append(line + ".")
    df = _curve_rows(history.frame)
    if plots_dir is not None and "mae_f" in df.columns:
        path = plots_dir / "best_model_training_curve.png"
        training_curve(df, path, title=f"Best model (fit_{fit_idx}) [{base_name}]")
        section.plots.append(path)
    return section


def committee_section(
    stats: dict, *, base_name: str, plots_dir: Path | None
) -> Section | None:
    """Committee uncertainty: force std-dev of every candidate, with the
    cut that the selected structures sit above (the lowest selected one;
    redundant picks skipped before DFT can leave gaps above it)."""
    import polars as pl

    from alomancy.analysis.report.plots import selection_rank_plot

    path = Path("results", base_name, "structure_generation", "std_dev_forces.csv")
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if "max_std_dev" not in df.columns or df.is_empty():
        return None
    df = df.sort("max_std_dev", descending=True)
    values = df["max_std_dev"].to_list()
    if "selected" in df.columns:
        mask = df["selected"].fill_null(False).to_list()
    else:  # written before the selection was recorded: assume the top N
        n = (stats.get("dft") or {}).get("submitted") or 0
        mask = [i < n for i in range(len(values))]
    chosen = [v for v, keep in zip(values, mask, strict=True) if keep]
    selected = len(chosen)
    cut = min(chosen) if chosen else None
    share = 100 * selected / len(values)
    section = Section(
        "Selection: committee uncertainty",
        [
            f"- {len(values)} candidates scored; {selected} selected "
            f"({share:.0f} %; max force std dev ≥ {_fmt(cut)} eV/Å)."
        ],
    )
    if plots_dir is not None:
        out = plots_dir / "committee_force_std_dev.png"
        selection_rank_plot(
            values,
            mask,
            out,
            title=f"Committee force std dev of candidates [{base_name}]",
            ylabel="Max per-atom force std dev (eV/Å)",
            cut=cut,
            unit="eV/Å",
        )
        section.plots.append(out)
    return section


def novelty_section(stats: dict) -> Section | None:
    """Novelty selection: the tolerance found and the reference-set size."""
    selections = stats["events"]["data"].get("novelty_selected") or []
    if not selections:
        return None
    data = selections[-1]
    return Section(
        "Selection: novelty",
        [
            f"- {data.get('selected')} of {data.get('candidates')} candidates "
            f"selected against {data.get('reference')} reference structure(s), "
            f"tolerance {_fmt(data.get('tolerance'))} (descriptor distance)."
        ],
    )
