# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It is the summary and index; the detail for each subsystem is in `docs/architecture/` (table below). Read the matching file before changing a subsystem.

## Commands

```bash
# Install in development mode (required before anything else): alomancy
# editable + the `dev` dependency group (pytest, ruff, mypy, pre-commit), from uv.lock
uv sync
# After editing dependencies in pyproject.toml: re-lock and commit uv.lock
# (CI runs `uv sync --locked`)
uv lock

# Build sdist + wheel into dist/ (hatchling + hatch-vcs; version from git tags)
uv build

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/alomancy

# Local / pre-commit tier: every unit test except `slow`, in parallel (~20 s)
/home/jholl/.venvs/al_install/bin/pytest -m "unit and not slow" -n auto

# Full suite, slow tests included (~20-30 s with -n auto)
/home/jholl/.venvs/al_install/bin/pytest -n auto

# Run a single test file
/home/jholl/.venvs/al_install/bin/pytest tests/core_tests/test_committee_uncertainty_workflow.py

# What CI runs (coverage is never on by default; CI passes it explicitly)
pytest -n auto --cov=alomancy --cov-report=xml --cov-fail-under=80 tests/
```

Test tiers: every test carries `@pytest.mark.unit` or `@pytest.mark.integration` (`tests/conftest.py` fails collection otherwise, because `-m unit` silently skips an unmarked test). `slow` marks real training or anything over ~2 s even on tmpfs; it is left out locally and runs in CI. `tests/conftest.py` puts pytest's temp dir on `/dev/shm` when usable: `GlobalDatabase` construction is SQLite fsync-bound (3–6 s on a btrfs `/tmp`, 5 ms on tmpfs), which took the suite from 278 s to 18 s. `ALOMANCY_TEST_DISK_TMP=1` or `--basetemp` opts out. The pytest config lives only in `pyproject.toml`. CI runs the full suite on one Linux job for pull requests and the full OS/Python matrix on pushes and releases.

Pre-commit hooks run `ruff` (with `--fix`) and the fast test tier on every commit. Install with `pre-commit install`.

## Architecture

ALomancy runs active learning (AL) for machine-learned interatomic potentials (MLIPs): **train model(s) → generate candidates (MD or EZGA) → select (committee uncertainty, random, novelty) → DFT → extend the training set → repeat**. `ALomancy(config)` (`core/entry.py`) builds the skeleton named by `general.al_workflow`; every module (trainer, generator, evaluator, initialiser) is picked by config through `registry.py`. Results land in `results/<base_name>/` with restart sentinels, so a restarted run redoes only what's missing. All structures live in the `GlobalDatabase` at `results/global_database/`.

| Working on | Read |
|---|---|
| `core/` (skeletons, `@phase`, start_from, splits, curation, retries, ranking, best model) | `docs/architecture/core_workflow.md` |
| `configs/` (loading, HPC profiles, unread-key check, registry) | `docs/architecture/configuration.md` |
| `remote_submission/`, `configs/remote_info.py`, ExPyRe isolation, ssh, Slurm | `docs/architecture/remote_submission.md` |
| `mlip/` (trainer base, MACE, SevenNet, evaluation, predictions) | `docs/architecture/mlip.md` |
| `structure_generation/` (MD, EZGA, seeds, uncertainty selection) | `docs/architecture/structure_generation.md` |
| `high_accuracy_evaluation/`, `utils/dft_utils.py` (QE, VASP, GO/SP) | `docs/architecture/high_accuracy_evaluation.md` |
| `database/` (sage_lib storage, ids, predictions, extxyz pitfalls) | `docs/architecture/database.md` |
| `initialize/` | `docs/architecture/initialization.md` |
| logging, `events.jsonl`, `analysis/` (reports, plots) | `docs/architecture/reports_and_plots.md` |
| `cli/` | `docs/architecture/cli.md` |

User-facing docs (published on Read the Docs): `docs/quickstart.md`, `docs/starting_a_run.md`, `docs/examples.md`, `docs/reports.md`, `docs/writing_a_workflow.md`, `docs/deprecations.md`.

When a change alters what one of these files says, update it in the same change. `tests/test_architecture_docs.py` checks that every source path and `path.name` reference in them (and in this file) still exists.

## Rules that apply everywhere

- **Remote code**: anything submitted remotely is a module-level function (ExPyRe pickles by reference), never a method. After changing a remote function's signature or body, the HPC install must be upgraded (`alomancy upgrade-hpc`) before restarting, or jobs fail with a `TypeError` hours later; say so in the changelog ("Remote reinstall needed"). Guard config errors in remote code with `if`/`raise ValueError`, never `assert` (`python -O` strips asserts).
- **Validate config at construction**, not at first use: anything read only after a loop's DFT finishes would otherwise fail hours in. A new config key must be added to its section's schema (`configs/schema.py` / the module's `kwargs_schema`), or the example-config tests fail on the warning.
- **Labels**: `atoms.info["REF_energy"]`, `atoms.arrays["REF_forces"]`, `atoms.info["REF_stresses"]`. Never a bare `"energy"` info key (ASE moves it to the calculator on extxyz read). Model predictions use generic `model_*` keys. All files are extxyz via ASE.
- **Empty `atoms.info` values** break extxyz round trips; call `utils/clean_structures.drop_unwritable_info` before writing xyz from atoms that didn't come from the DB (`docs/architecture/database.md`).
- **`needs_relaxation` is never cleared automatically**; a path with a fixed relaxation policy sets it explicitly on its output.
- **Backend-neutral names**: shared code says `isolated_atom_reference_energies`, `model_energy`, `trainer`, never a backend's own term (MACE `E0s`).
- **Polars, never pandas** (pandas isn't a dependency). **No interactive plotting**: create, save, `plt.close(fig)`; never `plt.show()`.
- **Logging only**, no `print()` in `src/`; give a warning the report should count an event code (`docs/architecture/reports_and_plots.md`). Don't change message text the timing plot parses.
- **Provenance**: `config_type` says where a structure came from; the loop number goes in `atoms.info["al_loop"]`, never in the config_type. `general.seed` (default 803) seeds every random choice.
- **Dependencies**: mace-torch pins `e3nn==0.4.4`, so `sevenn` stays `<0.11` (`docs/architecture/mlip.md`).

## Test conventions

Tests live in `tests/`, mirroring the source layout. Every test function carries `@pytest.mark.unit` (or `integration`) itself, not only through its class: mark inheritance is configuration-dependent and `-m unit` may miss class-level marks. Tests run real code paths; no mock-testing (asserting a mock returns what it was told to).

- `GlobalDatabase` tests use real sage_lib (`GlobalDatabase(str(tmp_path / "db"))`); shared instances are reset with `clear()`.
- `compute_initialization_needs` tests use a `MagicMock` DB returning controlled `count_all_by_config_type_and_formula()` dicts.
- `mace` is a real dev dependency, imported directly. `wfl` is not installed; never import or patch it.
- The generic trainer path is tested backend-free with a tiny `EMTTrainer` (`tests/mlip_train_tests/test_base.py`).
- `caplog` doesn't see alomancy logs (`propagate=False`): attach a `logging.Handler` to the logger directly.
- `tests/test_example_configs.py` builds every example config, the README config and its snippets, and every config block in `docs/examples.md` and `docs/starting_a_run.md`, failing on any config warning.
