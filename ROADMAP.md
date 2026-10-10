# ALomancy roadmap

Current release: **v1.0.1**. Priorities, in order: reliable production runs, usability for
other groups, new scientific scope. The granular checklist is in [TODO.md](TODO.md).

## 1.0.2 — patch (now)

Goal: ship the fixes already on the `82-local-expyre-failing` branch plus quick wins.

- **Merge #82.** Isolated per-run `.expyre`; `e3nn==0.4.4` / `sevenn<0.11` pins (e3nn 0.6
  broke MACE stage two, see yun_an loop 36); a missing `initialization` section skips
  initialization.
- **Timings plot.** The training queue window includes the DFT queue, the queue bar is not
  capped at its phase length, and the generation-start fallback regex matches nothing,
  so loop 35 of yun_an shows only DFT and queue time.
- **Bond-distribution plot.** The title overlaps the subplot titles and the logo.
- **`alomancy list-hpc`.** Drop the executable column.
- **Coverage badge (#24).** codecov-action v5 needs `files:`; uploads currently fail silently.
- **Dead code and stale docs.** Unused `RemoteInfo` fields, the old `dft/__init__`
  registry, and the sections of `docs/remote_submission_architecture.md` that still
  describe deleted modules.

## 1.1 — production reliability and paper plots (target ~2026-11-10)

Goal: long HPC runs that don't need babysitting, and the figures the paper needs.

- **DFT failure triage.** yun_an logged 711 QE `srun` failures, and DFT yield swung
  between 1 and 38 of 50 structures per loop. Classify failures (srun, SCF, walltime,
  missing forces), retry the retryable ones, report DFT yield per loop.
- **Retry prediction jobs.** One failed committee prediction job after MD currently
  raises with no retry. Retry once, with a minimum, as training already does.
- **Validate EZGA settings at construction.** EZGA kwargs are only checked when EZGA
  runs, hours in; register a `kwargs_schema` so typos are caught at startup too.
- **Guard expyre internals.** We patch expyre internals but accept any
  `expyre-wfl>=0.1.0`. Pin a tested range and test that the patched attributes still exist.
- **Plotting overhaul, first.** Plotting is wired in five places today. One dispatcher
  shared by `general.plots`, the report and `alomancy replot`; run-level plots separate
  from module plots; `plot_loop` moved to the base class so random and novelty runs get
  training plots too.
- **Then, inside the new structure:**
  - Committee uncertainty distribution plot (the data is already saved as `std_dev_forces.csv`).
  - Stresses on parity plots: store `model_stress` in the database, add an optional
    third column. Needed for the NPT runs.

## 1.2 — module refactor, users, dependencies, DFT setup

Goal: a cleaner plugin architecture, and an install other groups can get working.

- **Module refactor, first.** Abstract base classes for generators and evaluators, like
  `ALomancyTrainer`, under one shared superclass for all base modules (trainer,
  generator, evaluator, initialiser). The registry keeps accepting the old function-style
  entries as shims until 2.0. The pseudopotential repository and the VASP check below
  are built on the new evaluator class.
- **Optional per-backend extras.** MACE stays a core dependency; `alomancy[sevennet]` (which
  lifts the `<0.11` pin in its own environment) and an EZGA extra. `configs/schema.py`
  stops importing every trainer; a missing extra gives a clear error at startup.
- **HPC/local version mismatch.** Warn on a minor mismatch, raise on a major one, log a
  coded event so the report suggests `alomancy upgrade-hpc`.
- **ExPyRe output to status line.** Silence expyre's progress characters; log a periodic
  queued / running / done / failed line per phase.
- **Central pseudopotential repository** (design below).
- **VASP convergence check.** Detect a hit NELM limit from OSZICAR/OUTCAR.
- **Example configs and docs.** Examples for VASP, EZGA and NPT MD; an EZGA docs page (#44).
- **Installation docs.** The dependency stack: why each pin exists, the triton cache, when
  the remote needs a reinstall.
- **Deprecation schedule.** List every legacy shim in `docs/deprecations.md` with removal
  planned for 2.0.
- **CI.** mypy and the docs build gate CI (clear the existing backlog first); stop
  filtering every warning in the test suite, so upcoming torch/e3nn deprecations show.

### Pseudopotential repository: design

- One repository per HPC, set up by `alomancy add-hpc`, which downloads SSSP by default.
  The HPC profile records its path.
- Files are named by lowercase element, e.g. `c_<SSSP file name>`.
- The run config picks a set and can override single elements:
  ```yaml
  pseudopotentials:
    set: sssp_efficiency
    overrides:
      c: c_custom.upf
  ```
  A pre-run check confirms every element is covered and the file exists, instead of a
  `KeyError` on the node.
- VASP: by default the user's own licensed POTCARs, as today. A user who points at
  alternatives gets the central directory with the same naming.

## 1.3 — scientific scope

- **FHI-aims, or a generic ASE evaluator**, built on the 1.2 evaluator class.
- **Surface construction.** Different terminations, target symmetry vs preserved
  stoichiometry, cuts.
- **Element-swap initialization.** Take known crystals and swap elements.
- **Force-bias workflow (#22).** Rewritten as an `ActiveLearningWorkflow` subclass; the old
  branch predates 1.0.

## 2.0 — breaking changes and long-term

- **External validation** beyond energy and forces (PDF, XRD), run outside the loop.
- **torch-sim integration** for MLIP training and running.
- **Remove legacy shims.** Legacy expyre config path, `max_concurrent_jobs`, old QE/VASP
  kwargs names, `is_high_force`, function-style generator/evaluator registration.

## Ideas (no milestone)

- **Furthest-point sampling (#43)** as a variant of novelty selection.
- **`alomancy init` and the HPC wizard (#48)** will likely move to an external tool.
