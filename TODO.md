# TODO

Checklist by milestone; the reasons and design notes are in [ROADMAP.md](ROADMAP.md).

## 1.0.3 — patch
- [x] Timings plot: only DFT plus queue time visible (queue window, uncapped queue bar, missing generation start)
- [x] Bond-distribution title overlaps the subplot titles
- [x] Don't include the executable in `alomancy list-hpc`
- [x] Coverage badge (#24): codecov `files:`, fail visibly
- [x] Remove dead code (unused `RemoteInfo` fields, old `dft/__init__` registry, dead submitters)
- [x] README links go to the Read the Docs documentation, not the repo files
- [x] Outdated examples: `name:` keys (now warned about), MACE-only predictions section generalised, docs config blocks tested
- [x] Split CLAUDE.md into `docs/architecture/` per subsystem (replaces `remote_submission_architecture.md`), with a drift test
- [x] A `name:` key in a config section warns that it is ignored
- [x] Restore a database that sage_lib replaced under a live run on the next write (`database_recovered`)
- [x] Redundancy removal: deduplicate-lib >= 0.1.3 (keep one per group), finer sweep with a minimum plateau width and a flagged-fraction cap, `general.redundancy_kwargs`, training-set-only pool, filters and curation before redundancy
- [x] Replace redundant committee picks before DFT; an empty or short selection still writes `high_sd_structures.xyz` and logs `fewer_candidates`
- [x] `GlobalDatabase.clear_duplicate_flags`
- [x] `exempt_config_types` for `train_filter` / `test_filter`
- [x] Best-model relative (percentage) error plot in the report
- [x] `start_from`: any number of databases and xyz files, each key a path or a list, combined
- [x] Redundancy removal: a repeated pass changes nothing (flagged structures stay in the pool, flags rewritten each pass); the pre-DFT check reuses the loop's tolerance
- [x] 1.0.3 review fixes: empty-selection restart crash, novelty selection near database rows, relative energy error without formation energies, recovered WAL naming, `selection_redundant_shortfall` event, finite `tolerance`, committee selection cut, executable in the startup HPC table, guarded `clear()`/migration, per-test marks

## 1.1 — production reliability and paper plots
- [ ] DFT failure triage: classify, retry the retryable, report DFT yield per loop
- [ ] Retry failed committee prediction jobs once, with a minimum
- [ ] Prediction jobs ending with "remote status done but no _succeeded or _error" (committee_mace_qe, loop 3): find the cause
- [ ] Read-only database opens for readers (`alomancy results --replot`, `start_from.database` source, migration): a read-write open of a live run's DB makes sage_lib swap in empty files; an import from a live run records "0 imported" for good. A read-only open of a held DB fails with a lock error, so readers must catch it
- [ ] Redundancy plateau rule: on a pool without duplicates it still flags a few distinct structures (lowest non-all-kept plateau), and on a highly redundant pool the cap vetoes the plateau instead of picking a more persistent one
- [ ] Redundancy descriptor: `char_vec_128` has no usable plateau on amorphous data (the cap hits); try a per-atom-normalised RDF or SOAP
- [ ] Upstream sage_lib: don't treat an HDF5 lock error as corruption (it renames the live files)
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
- [ ] Example configs for VASP, NPT MD (EZGA done in 1.0.4: `docs/ezga.md`)
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
- [x] Isolated per-run `.expyre`, e3nn/sevenn pins, optional `initialization` (#82, 1.0.2)
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
