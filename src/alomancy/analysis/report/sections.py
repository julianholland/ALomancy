"""The Section type every report part returns, plus the module-specific
sections registered as ``report_section`` entry points (see registry.py).

A module opts in by defining ``report_section(stats, *, base_name,
plots_dir, config)`` (a thin wrapper that lazily imports from here, so
remote-only modules never import matplotlib) and registering it. Each
returns a Section or None when it has nothing to show for this loop.
"""

import json
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
    return section


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


def mace_section(
    stats: dict,
    *,
    base_name: str,
    plots_dir: Path | None,
    config: dict,  # noqa: ARG001 -- uniform report_section signature
) -> Section | None:
    """MACE trainer: the best fit's resolved epochs and training curve."""
    from alomancy.analysis.mlip_plots import _parse_training_jsonl
    from alomancy.analysis.report.plots import training_curve

    model = stats.get("model") or {}
    fit_idx = model.get("best_fit_idx")
    if fit_idx is None:
        return None
    fit_dir = Path("results", base_name, "training", f"fit_{fit_idx}")
    section = Section("MLIP training: MACE")
    epochs_file = fit_dir / "resolved_mace_epochs.json"
    if epochs_file.exists():
        resolved = json.loads(epochs_file.read_text())
        section.lines.append(
            f"- Best fit (fit_{fit_idx}) trained for up to "
            f"{resolved.get('max_num_epochs')} epochs, stage two from epoch "
            f"{resolved.get('start_swa')}."
        )
    seed = (stats.get("seed") or 0) + int(fit_idx)
    df = _parse_training_jsonl(fit_dir, "training", seed) if plots_dir else None
    if df is not None and "mae_f" in df.columns:
        path = plots_dir / "best_model_training_curve.png"
        training_curve(df, path, title=f"Best model (fit_{fit_idx}) [{base_name}]")
        section.plots.append(path)
    return section if section.lines or section.plots else None


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
