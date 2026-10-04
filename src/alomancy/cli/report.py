"""`alomancy report`: (re)write loop reports for an existing results
directory without running anything (see analysis/report)."""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def completed_loops(results_dir: Path) -> list[int]:
    """Loop numbers with a loop.done sentinel, in order."""
    return sorted(
        int(p.parent.name.rsplit("_", 1)[1])
        for p in results_dir.glob("al_loop_*/loop.done")
    )


def write_reports(
    results_dir: Path,
    *,
    loops: list[int] | None = None,
    config_path: Path | None = None,
) -> list[Path]:
    """Write reports for *loops* (default: the latest completed loop) of
    the run in *results_dir*. The run's config comes from *config_path*,
    else results/run_config.yaml (saved by every run since reports were
    added). Returns the report paths."""
    from alomancy.analysis.report import write_loop_report
    from alomancy.core.entry import ALomancy

    results_dir = results_dir.resolve()
    config_path = (config_path or results_dir / "run_config.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}. Runs from before reports were "
            "added didn't save one; pass the run's YAML with --config."
        )
    done = completed_loops(results_dir)
    if loops is None:
        if not done:
            raise RuntimeError(f"No completed AL loops in {results_dir}.")
        loops = [done[-1]]

    # Paths inside a run are relative to the directory holding results/.
    os.chdir(results_dir.parent)
    jobs_dict = (
        yaml.safe_load(config_path.read_text())
        if config_path.name == "run_config.yaml"
        else None
    )
    workflow = ALomancy(jobs_dict if jobs_dict is not None else config_path).workflow
    written = []
    for loop in loops:
        if loop not in done:
            logger.warning(
                "al_loop_%d is not complete; its report may be partial.", loop
            )
        written.append(write_loop_report(workflow, loop))
    return written
