# Loop reports

At the end of every AL loop ALomancy writes a Markdown report on how the
run is going:

```
results/reports/
├── latest.md                 # the newest loop's report (links into al_loop_N/)
├── al_loop_0/
│   ├── report.md
│   ├── report.json           # the numbers behind the report
│   └── plots/*.png
└── al_loop_1/ ...
```

Open `latest.md` in any Markdown viewer (VS Code preview, GitHub, GitLab).
The report never stops a run: if part of it fails, that part is skipped
and a warning is logged.

## What's in a report

| Section | Contents |
|---|---|
| Status | Count of issues found, by severity, and warnings logged this loop |
| Headline | Best model's test force/energy MAE, train/test size, new structures, warnings: this loop vs the previous one |
| Issues & suggestions | Each problem detected, with what to change in the config (see below) |
| Trends | One row per loop: test force MAE, train size, new structures, DFT returned/submitted, GO convergence rate, new structures found redundant, warnings. Plus the MAE-vs-loop and timing plots |
| Best model | Energy and force parity of this loop's best model (train and test) |
| Training set | Structures per `config_type`: train, test, excluded as redundant, excluded by the quality filters (with reasons) |
| DFT | Returned vs submitted, single points vs relaxations, how many relaxations reached the force ceiling, BFGS steps vs the step budget, average DFT time per structure, largest force after DFT, DFT phase wall-clock and queue time |
| Module sections | From the configured trainer, generator and evaluator, e.g. MD frames per run, the best fit's training curve, GO-step and DFT-time histograms |
| Workflow section | e.g. the committee's force std-dev distribution with the selection cut, or the novelty tolerance |
| Warnings and events | Count per event code, and the most frequent other warnings |

Per-structure DFT step counts and wall time (`geometry_steps`,
`dft_wall_time_s` in `atoms.info`) are recorded by the remote DFT runners
from this version on. Run `alomancy upgrade-hpc` after upgrading; loops
evaluated before that show "not recorded".

## Settings

```yaml
general:
  report: true                 # default; false turns reports off
  report_suggestions: null     # optional YAML overriding suggestions.yaml
  plots: true                  # false: reports are written without figures
```

## Rebuilding reports

```bash
alomancy report                 # the latest completed loop
alomancy report --loop 3
alomancy report --all
alomancy report --config my_run.yaml   # runs from before reports existed
```

Run it from the directory holding `results/` (or pass `--results-dir`).
The run's config is read from `results/run_config.yaml`, which every run
now saves at start-up.

## Issues and suggestions: editing them

Detection is split in two so the advice can be tuned without touching
code:

- **Triggers** (`src/alomancy/analysis/report/triggers.py`) measure
  something about the loop, e.g. the fraction of geometry optimisations
  that hit the step budget. They change rarely.
- **Suggestions** (`src/alomancy/analysis/report/suggestions.yaml`) say
  when a trigger counts as a problem and what to do about it:

```yaml
go_not_converged:
  threshold: 0.10          # fires when the measured value is >= this
  severity: warning        # info | warning | error
  suggestion: >-
    {n}/{total} geometry optimisations ({value:.0%}) stopped at the step
    budget ({max_steps} steps). Raise
    high_accuracy_evaluation.max_num_of_relax_steps ...
```

`{value}` and every other number the trigger reports can be used in the
text, with Python format specs (`{value:.0%}`, `{mean_steps:.0f}`).

**Per-project wording without editing the package:** write a YAML with
just the entries you want to change and point `general.report_suggestions`
at it. Its entries replace the packaged ones one key at a time, so this is
enough to raise one threshold:

```yaml
go_not_converged:
  threshold: 0.25
```

Mistakes never break the report: an entry naming an unknown trigger, a
trigger with no entry, or a `{name}` the trigger doesn't provide are listed
under "Suggestions file notes" in the report and logged.

### Current triggers

| Trigger | Measures |
|---|---|
| `go_not_converged` | geometry optimisations that stopped at the step budget / all GO |
| `go_steps_near_budget` | mean BFGS steps / step budget |
| `new_structures_force_filtered` | new structures excluded by `train_filter.max_force` / new structures |
| `short_bond_excluded` | structures dropped for a bond < 0.5 Å / structures checked |
| `md_runs_no_steps` | MD runs that never took a step / runs |
| `remote_jobs_failed` | remote jobs failed or died / jobs started |
| `fewer_candidates` | shortfall of candidates against `desired_num_of_structures` |
| `redundancy_removed_new` | this loop's new structures flagged redundant / new |
| `queue_dominated` | largest share of a phase's time spent queued |
| `fit_retries` | model fits retried / fits |
| `overfitting` | best model's test force MAE / train force MAE |
| `dft_partial_failure` | structures sent to DFT with no result / submitted |

### Adding a trigger

1. Add a function to `triggers.py`:

   ```python
   @trigger("my_trigger")
   def my_trigger(stats: dict) -> dict | None:
       ...  # read stats (see report.json for its layout)
       return {"value": fraction, "n": n, "total": total}  # or None if not applicable
   ```

2. Add a `my_trigger:` entry to `suggestions.yaml`.

A test checks that every trigger has an entry and every entry a trigger.

## Where the numbers come from

- **The database**: composition, redundancy (`is_duplicate`), quality
  filters (`is_quality_filtered`, `quality_filter_reasons`). Flags are the
  database's current ones, so a regenerated report for an old loop counts
  structures from that loop and earlier with today's flags.
- **`results/<loop>/high_accuracy_eval_results.xyz`**: per-structure DFT
  data.
- **Each fit's `evaluation_metrics.json`**: model errors; the best fit is
  chosen as for MD.
- **`results/alomancy.log`**: phase timings.
- **`results/events.jsonl`**: warnings and coded events, written next to
  the log. Each line is one JSON object (`time`, `level`, `logger`,
  `message`, `event`, `loop`, `data`). Every WARNING or above is recorded;
  INFO records only when they carry an event code.

To make a new situation countable, log it with an event code:

```python
logger.warning(
    "Something went wrong for %s.", base_name,
    extra={"event": "my_event", "data": {"n": n, "total": total}},
)
```

## Module sections

A trainer, generator or evaluator adds its own section by defining
`report_section(stats, *, base_name, plots_dir, config)` returning an
`alomancy.analysis.report.Section` (title, Markdown lines, plot paths) or
`None`, and registering it as the `report_section` entry point in
`registry.py`. `plots_dir` is `None` when plots are off. A workflow adds
sections by overriding `ActiveLearningWorkflow.report_sections(stats,
plots_dir)`.
