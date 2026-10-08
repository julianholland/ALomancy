"""The Section type every report part returns, plus the module-specific
sections registered as ``report_section`` entry points (see registry.py).

A module opts in by defining ``report_section(stats, *, base_name,
plots_dir, config)`` (a thin wrapper that lazily imports from here, so
remote-only modules never import matplotlib) and registering it. Each
returns a Section or None when it has nothing to show for this loop.
"""

import logging
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
            f"min {min(frames) if frames else 'n/a'})",
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
    from alomancy.analysis.report.plots import md_runs
    from alomancy.structure_generation.md.md_wfl import _MD_KWARGS_DEFAULTS

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
    )
    return path


def ezga_section(
    stats: dict,  # noqa: ARG001 -- uniform signature
    *,
    base_name: str,
    plots_dir: Path | None,  # noqa: ARG001
    config: dict,
) -> Section | None:
    """EZGA generator: generations, population and candidates returned."""
    from ase.io import read

    sg = config.get("structure_generation", {})
    path = Path("results", base_name, sg.get("name", "structure_generation"), "ezga")
    candidates = path / "ezga_candidates.xyz"
    if not candidates.exists():
        return None
    kwargs = sg.get("run_ezga_kwargs", {})
    n = len(read(candidates, ":", format="extxyz"))
    return Section(
        "Structure generation: EZGA",
        [
            f"- Generations: {kwargs.get('max_generations', 'default')}, "
            f"population: {kwargs.get('population_size', 'default')}",
            f"- Candidates returned: {n}",
        ],
    )


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
    cut that the selected structures sit above."""
    import polars as pl

    from alomancy.analysis.report.plots import histogram

    path = Path("results", base_name, "structure_generation", "std_dev_forces.csv")
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if "max_std_dev" not in df.columns or df.is_empty():
        return None
    selected = (stats.get("dft") or {}).get("submitted") or 0
    values = df["max_std_dev"].sort(descending=True).to_list()
    cut = values[min(selected, len(values)) - 1] if selected else None
    section = Section(
        "Selection: committee uncertainty",
        [
            f"- {len(values)} candidates scored; {selected} selected "
            f"(max force std dev ≥ {_fmt(cut)} eV/Å)."
        ],
    )
    if plots_dir is not None:
        out = plots_dir / "committee_force_std_dev.png"
        histogram(
            values,
            out,
            title=f"Committee force std dev of candidates [{base_name}]",
            xlabel="Max per-atom force std dev (eV/Å)",
            marker=cut,
            marker_label="selection cut" if cut is not None else None,
            log_y=True,
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
