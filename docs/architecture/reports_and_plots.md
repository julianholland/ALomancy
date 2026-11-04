# Logging, events, reports and plots

`src/alomancy/utils/logging_config.py`, `src/alomancy/analysis/` (plots, `report/`). User docs: `docs/reports.md`.

## Logging

All output goes through `logging`; there are no bare `print()` calls in `src/`. `setup_logging(verbose, log_file)` is the single configuration point, called during workflow construction. Each module uses `logger = logging.getLogger(__name__)`; the `"expyre"` logger is routed through the same handlers.

| `verbose` | Console | File (`results/alomancy.log`) |
|---|---|---|
| 0 (default) | WARNING+ | DEBUG always |
| 1 | INFO (step progress) | DEBUG always |
| 2+ | DEBUG (per-job detail) | DEBUG always |

The file always gets DEBUG, so every run has a complete timestamped record; ExPyRe job stdout/stderr go there at DEBUG.

**Testing logs**: `caplog` doesn't see alomancy logs (`propagate=False` on the `alomancy` logger). Attach a `logging.Handler` to `logging.getLogger("alomancy")` (or the module's own logger, for construction-time warnings) directly; see `_capture` in `TestDisplayWorkflowSummary` (`tests/core_tests/test_active_learning_workflow.py`).

**Keep message text unchanged where `analysis/timing_plots` parses it**: the `@phase` decorator's "Phase %s started for %s." and `_mark_phase_done`'s "Phase %s marked complete for %s." (phase boundaries for every workflow), "Starting AL loop %d", "Training set size: %d", "Completed AL loop %d, ...", "High-accuracy evaluation completed for %d structures." and the executor's "Job %d queue_time=%.1f s.". `tests/analysis_tests/test_timing_plots.py` parses a log the real decorator wrote, so a renamed message fails a test.

## Events: `results/events.jsonl`

`JsonlEventHandler` (`utils/logging_config.py`) writes every WARNING+ record, plus any record with `extra={"event": "<code>", "data": {...}}`, as a JSON line next to the log file. Each line carries:
- the AL loop (`set_current_loop`, set by `iterate_loops`; a module global, not a ContextVar, because `RemoteJobExecutor` logs from worker threads);
- `attempt` (`ATTEMPT_ID`, one per process);
- `phase` (`set_current_phase`, set by `@phase`, which also logs `phase_started`/`phase_completed`).

Job events carry `data.item` (the job config's `item`: `fit_<i>`, `md_run_<slot>` (a replacement seed shares its original's slot), `structure_<job_id>`); the executor logs `job_succeeded` per finished job; retry/summary events (`fit_retry`, `fit_missing`, `fit_not_evaluated`, `md_no_steps`, `md_unfinished`) list `data.items`.

The report counts events instead of parsing log text, so **give a new warning an event code when the report should count it**, and a new retry path should give its jobs an `item` and name the items in its retry event.

`analysis/report/current_events.current_events` decides what the report shows: per step, only the latest attempt that ran it; then failure/retry events whose items all ended in `job_succeeded` are dropped; `phase_*`/`job_succeeded` are never shown. Older untagged events get attempt and step from `alomancy.log` (attempt = the "ALomancy Workflow Summary" header; step = the first not-yet-done step completing after the event).

## Loop reports (`analysis/report/`)

`finish_loop` calls `write_loop_report(self, loop)` (`general.report`, default true; a report failure never fails the loop), writing `results/reports/al_loop_N/{report.md, report.json, plots/}` and `results/reports/latest.md`.

- `stats.py` builds a JSON-safe stats dict from the DB, `high_accuracy_eval_results.xyz` (`geometry_steps`, `geometry_max_steps`, `dft_wall_time_s`), each fit's `evaluation_metrics.json`, `alomancy.log` timings and `events.jsonl`.
- Issues: `triggers.py` (one `@trigger("name")` function per measurement, returning `None` or `{"value": ..., ...}`) + `suggestions.yaml` (package data: `threshold`/`severity`/`suggestion` per trigger; `general.report_suggestions` overrides entries key by key) + `rules.py` (never raises over a bad entry; a test enforces that triggers and entries match).
- Module sections: an optional `report_section` registry entry point (a thin lazy wrapper in the module, the implementation in `report/sections.py`; `plots_dir=None` when `general.plots` is off). Workflow sections: `ActiveLearningWorkflow.report_sections(stats, plots_dir)`.
- `prepare_run` saves `results/run_config.yaml`, so `alomancy report [--loop N|--all] [--config]` (`cli/report.py`) rebuilds reports offline.

## Plots

**No plotting path may be interactive**: `alomancy/__init__.py` calls `matplotlib.use("Agg")` before any submodule imports `pyplot`, because a `Figure` created under an interactive backend (TkAgg whenever `DISPLAY` is set) and garbage-collected on a `RemoteJobExecutor` worker thread crashes the process with SIGABRT. Every plot function creates, saves and closes its figure (`plt.close(fig)`); never `plt.show()`. `analysis/plotting.py`'s `Plot` has no `show()`; `Plot.save()` closes its figure.

**Layout**: per-loop plots go to `results/current_plots/<base_name>/` (filenames still `_al_loop_N`-suffixed). The directory is chosen at the call site (the workflow's loop and `cli/replot.py`); plot functions just take a `plots_dir`. `plot_training_curves` also writes each fit's per-epoch numbers to `results/current_plots/<base_name>/metrics/<base_name>_fit_<i>_training_metrics.csv`. Run-level plots (the timing plot) stay flat in `results/current_plots/`. `alomancy results --replot` writes its cross-loop MAE plot into the last loop's directory.

- **Timing** (`analysis/timing_plots.py`): one figure, `timing_combined.png`: stacked bars of wall time per phase per loop (training, generation, DFT, post-process), each split into queued and running, plus a training-set-size line on a `twinx()` axis. Both axes' handles go into one `ax.legend(h1 + h2, l1 + l2, ...)` (a bare `ax.legend()` drops the twin's line). Bars sit at real loop numbers with `MaxNLocator(integer=True, nbins=12)`; fixed `figsize=(10, 6)`. Writing it deletes the old `timing_total.png`/`timing_phases.png` (`_STALE_TIMING_PLOTS`). Phase boundaries come from the `@phase` markers (training ends at `train_mlip marked complete` when there's no generation start), with the pre-1.0 messages as fallbacks; a restarted loop keeps its latest attempt. Each phase averages only the queue times of jobs that finished inside it, and `_phase_segments` caps the queued segment at the phase length (the training bar was hidden by the DFT jobs' queue time before 1.0.3).
- **MAE vs loop** (`mae_al_loop_plot`, `analysis/plotting.py`): the best model per loop (`metrics_by_loop`, the `rank_committee` pick's **test** `mae_f`/`mae_e_per_atom`), one row per loop with `al_loop` as the x axis, so a skipped loop shows as a gap. Live runs skip a loop with inconsistent evaluations (warning); `replot` raises.
- **Parity** (`analysis/mlip_plots.py`): `parity_predictions` is the single source of parity data for plots and the report: the database per split, then each missing split from its `{split}_pred.xyz`. `plot_dft_vs_model` never runs local inference. Energies are **formation energy per atom** (`(E − Σ E0[symbol]) / n`, E0 from `GlobalDatabase.get_isolated_atom_energies()`), subtracted from both DFT and model so the MAE is unchanged; it falls back to raw per-atom energy (one warning) without a DB, without `IsolatedAtom` entries, or when an element lacks an E0, never mixing the two in one figure. `alomancy results --replot` opens `results_dir/global_database` when present. When any structure has both `REF_stresses` and a model stress (models trained with `compute_stress`), the parity figures get a third **stress** column (Voigt components, GPa; `mlip.evaluation.stress_pair_gpa`): parity tuples are `(e_ref, e_pred, f_ref, f_pred, s_ref, s_pred)`, read with `mlip_plots.stress_pairs` (empty stress for 4-tuples), from `model_stress_loop_<l>_fit_<f>` in the database (`store_model_predictions`, from `read_predictions`) or `model_stress` in `{split}_pred.xyz`. The lowest-MAE energy, force and stress subplots each get a gold `★` (`_add_best_star`, `gid="best_star"`), train and test figures independently. The report adds `best_model_percent_error.png` (`analysis/report/plots.percent_error`): signed energy % and per-atom force % (`|ΔF|/|F|`, log-log) against the DFT value; energy on a symlog y. `percent_errors` computes the points for both the figure and the report's median line, leaving out |E| < `PERCENT_ENERGY_FLOOR` (0.05 eV/atom) and |F| < `PERCENT_FORCE_FLOOR` (0.1 eV/Å).
- **Training curves**: `plot_training_curves(..., trainer)` reads `training_history` only; the Stage Two marker comes from the first successfully collected fit, never assuming `fit_0`.
- **Bond distances** (`analysis/bond_distance_plots.py`): one histogram per element pair of the training set, before training each loop, as an MD-instability diagnostic. The title is left-aligned and the logo sized in inches so neither overlaps the panels.

Tabular data is polars, never pandas (not a dependency). Polars has no index: what pandas used as one is an explicit column (`al_loop`, `epoch`, `loop`, `structure_index`). Use `.is_empty()`, `.to_list()`, `df.row(i, named=True)`, and `.cast(pl.Float64).to_numpy()` for nullable numeric columns.
