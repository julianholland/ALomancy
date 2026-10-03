import json
import logging
import os
import re
from pathlib import Path

import yaml

from alomancy.analysis.mlip_plots import plot_dft_vs_model, plot_training_curves
from alomancy.analysis.plotting import mae_al_loop_plot
from alomancy.analysis.timing_plots import timing_plots
from alomancy.database.global_database import GlobalDatabase
from alomancy.mlip.base import ALomancyTrainer, get_trainer
from alomancy.mlip.evaluation import metrics_by_loop

logger = logging.getLogger(__name__)


def detect_committee_info(results_dir: Path) -> tuple[str, int, int]:
    """Infer (name, num_of_models_in_committee, seed) from the results directory layout.

    Searches ``results_dir/al_loop_*/`` for the first subdirectory that
    contains a ``fit_0/`` child. The seed is read from ``fit_0/fit_seed.json``
    (recorded by the workflow), else parsed from the filename of the
    ``*_run-{N}_train.txt`` metrics file written by MACE.
    """
    for loop_dir in sorted(results_dir.glob("al_loop_*")):
        if not loop_dir.is_dir():
            continue
        for candidate in sorted(loop_dir.iterdir()):
            if not candidate.is_dir():
                continue
            if (candidate / "fit_0").is_dir():
                name = candidate.name
                n_fits = sum(1 for _ in candidate.glob("fit_*") if _.is_dir())

                seed = 803  # project-wide default fallback
                seed_file = candidate / "fit_0" / "fit_seed.json"
                if seed_file.exists():
                    try:
                        return (
                            name,
                            n_fits,
                            int(json.loads(seed_file.read_text())["seed"]),
                        )
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                txt_files = list(
                    (candidate / "fit_0" / "results").glob("*_run-*_train.txt")
                )
                if txt_files:
                    m = re.search(r"_run-(\d+)_", txt_files[0].name)
                    if m:
                        seed = int(m.group(1))

                return name, n_fits, seed

    raise RuntimeError(
        f"Could not detect mlip_committee directory under {results_dir}. "
        "Expected a subdirectory of an al_loop_* dir that contains fit_0/."
    )


def _trainer_for(results_dir: Path, name: str) -> ALomancyTrainer:
    """The run's trainer, from results/run_config.yaml's ``training``
    section when the run saved one, else MACE (runs from before the
    config was saved were all MACE)."""
    training: dict = {}
    config_path = results_dir / "run_config.yaml"
    if config_path.exists():
        try:
            training = (yaml.safe_load(config_path.read_text()) or {}).get(
                "training"
            ) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Could not read %s: %s", config_path, exc)
    return get_trainer(training.get("trainer", "mace"), training, name)


def replot_results(results_dir: Path, no_parity: bool = False) -> None:
    """Regenerate all plots from an existing alomancy results directory.

    Detects committee name, size, and seed from the directory layout, then
    calls each plotting function in the same order as the AL workflow.

    Parameters
    ----------
    results_dir:
        Path to the ``results/`` directory produced by the workflow.
    no_parity:
        When True, skip ``plot_dft_vs_model`` (which loads MACE models and
        runs forward passes — can take tens of minutes per loop).
    """
    # All plotting functions resolve paths relative to CWD.
    os.chdir(results_dir.parent)

    name, n_fits, seed = detect_committee_info(results_dir)
    logger.info("Detected committee: name=%r, size=%d, seed=%d", name, n_fits, seed)

    job_dict = {"name": name, "num_of_models_in_committee": n_fits}
    trainer = _trainer_for(results_dir, name)
    plots_dir = results_dir / "current_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Reuse the GlobalDatabase's IsolatedAtom energies for formation-energy parity
    # plots, matching the live AL loop. Only open it if it already exists — the
    # constructor unconditionally mkdir's, and we don't want to create an empty
    # DB for runs that predate GlobalDatabase support.
    db_path = results_dir / "global_database"
    db = GlobalDatabase(str(db_path)) if db_path.exists() else None
    if db is None:
        logger.info(
            "No GlobalDatabase found at %s — parity plots will use raw per-atom "
            "energy (older run, or run predates GlobalDatabase support).",
            db_path,
        )

    # Loops with at least one evaluated fit (any trainer).
    def _has_evaluated_fit(loop_dir: Path) -> bool:
        return bool(next((loop_dir / name).glob("fit_*/evaluation_metrics.json"), None))

    # Numeric order: sorting by name would put al_loop_9 after al_loop_14.
    loops = sorted(
        (
            d
            for d in results_dir.glob("al_loop_*")
            if d.is_dir() and _has_evaluated_fit(d)
        ),
        key=lambda p: int(p.name.rsplit("_", 1)[1]),
    )

    if not loops:
        logger.warning(
            "No completed loops found under %s — nothing to plot.", results_dir
        )
        return

    for loop_dir in loops:
        base_name = loop_dir.name
        m = re.fullmatch(r"al_loop_(\d+)", base_name)
        loop_idx = int(m.group(1)) if m else None
        logger.info("Plotting loop %s …", base_name)
        # Per-loop plots land in their own subdirectory, matching the live
        # AL loop's results/current_plots/<base_name>/ layout.
        loop_plots_dir = plots_dir / base_name
        loop_plots_dir.mkdir(exist_ok=True, parents=True)
        plot_training_curves(base_name, job_dict, seed, loop_plots_dir, trainer)
        if not no_parity:
            plot_dft_vs_model(
                base_name, job_dict, seed, loop_plots_dir, db=db, loop_idx=loop_idx
            )

    # Cross-loop MAE summary (reads all al_loop_*/... train.txt files). This
    # mirrors the live run's last-loop mae_al_loop_plot call (which always
    # shows the fullest cumulative history at that point), so it lands in the
    # last plotted loop's subdirectory rather than flat in plots_dir.
    df = metrics_by_loop(name, strict=True, expected_fits=n_fits)
    if not df.is_empty():
        last_loop_plots_dir = plots_dir / loops[-1].name
        last_loop_plots_dir.mkdir(exist_ok=True, parents=True)
        mae_al_loop_plot(df, job_dict, directory=last_loop_plots_dir)
    else:
        logger.warning(
            "No evaluation_metrics.json found for any loop — MAE loop plot skipped."
        )

    # Timing plots (purely log-based, no MACE needed)
    log_file = results_dir / "alomancy.log"
    if log_file.exists():
        timing_plots(log_file, plots_dir)
    else:
        logger.info("No alomancy.log found at %s — timing plots skipped.", log_file)

    logger.info("Replot complete. Output in %s", plots_dir)
