# TODO

Checklist by milestone; the reasons and design notes are in [ROADMAP.md](ROADMAP.md).

## 1.0.2 — patch
- [ ] Merge #82: isolated per-run `.expyre`, e3nn/sevenn pins, optional `initialization`
- [ ] Timings plot: only DFT plus queue time visible (queue window, uncapped queue bar, dead fallback regex)
- [ ] Bond-distribution title overlaps the subplot titles
- [ ] Don't include the executable in `alomancy list-hpc`
- [ ] Coverage badge (#24): codecov `files:`, fail visibly
- [ ] Remove dead code and stale docs (unused `RemoteInfo` fields, old `dft/__init__` registry, `remote_submission_architecture.md`)

## 1.1 — production reliability and paper plots
- [ ] DFT failure triage: classify, retry the retryable, report DFT yield per loop
- [ ] Retry failed committee prediction jobs once, with a minimum
- [ ] Validate EZGA settings at construction; EZGA `kwargs_schema`
- [ ] Pin a tested `expyre-wfl` range; test the patched expyre attributes exist
- [ ] Plotting overhaul: each plot at the right level (run-level like timings at the top, plug-in plots with their plug-in); `plots: true` calls every relevant module's plotting; `plot_loop` in the base class
- [ ] Plot the distribution of committee uncertainty
- [ ] Add stresses to the parity plots when present

## 1.2 — module refactor, users, dependencies, DFT setup
- [ ] Extend the MLIP abstract-class structure to structure generation and high-accuracy evaluation; one superclass for all base modules (old functions kept as shims)
- [ ] Optional imports per module (mace core; sevennet, ezga extras)
- [ ] HPC/local version mismatch: warn on minor, raise on major, record in the report (e.g. "raven is on 1.0.1, the submitter is on 1.0.2; consider running `alomancy upgrade-hpc`")
- [ ] Catch ExPyRe/wfl output and replace it with a structure status line
- [ ] Sort out the pseudopotential setup: central repository (SSSP default, lowercase-element names, named set + per-element overrides)
- [ ] Manual check that VASP converged
- [ ] Example configs for VASP, EZGA, NPT MD; EZGA docs page (#44)
- [ ] Installation docs: dependency stack and pins
- [ ] `docs/deprecations.md`: legacy shims with removal planned for 2.0
- [ ] mypy and docs build gate CI; stop filtering all warnings in tests

## 1.3 — scientific scope
- [ ] FHI-aims interface (or a better generic ASE interface)
- [ ] Surface construction: different terminations, target symmetry vs preserved stoichiometry, cuts
- [ ] Element-swap initialization: take known crystals and swap elements
- [ ] Force-bias workflow (#22)

## 2.0
- [ ] External validation beyond energy and forces (PDF, XRD), outside the loop
- [ ] torch-sim integration for MLIP training/running
- [ ] Remove legacy shims

## Ideas
- Furthest-point sampling (#43) as a novelty-selection variant
- `alomancy init` / HPC wizard (#48), probably an external tool

## Done
- [x] Combined timings plot: integer x ticks, thinned; old timing plots removed
- [x] MAE-vs-loop plot: best model per loop, outside the loop folders
- [x] Gold star on the lowest-MAE energy and force parity subplots
- [x] Test parity plots restored (1.0)
- [x] MD trajectories saved
- [x] Redundancy descriptors cached in the database
- [x] `results/best_model/` holds the latest loop's best model
- [x] `alomancy list-hpc`
- [x] `general.start_from`: train/test xyz, single xyz, former database, cold start
- [x] Abstract `ActiveLearningWorkflow` parent; children only define `run()`
- [x] Per-loop report with stats, plots, warnings and suggestions (1.0)
- [x] Config picks the workflow; `ALomancy(config)` entry point (1.0)
- [x] pandas replaced by polars (1.0)
- [x] SevenNet interface (1.0)
- [x] Novelty-based selection (1.0)
