<div align="center">

<img src="docs/_static/alomancy_logo.png" alt="ALomancy" width="200"/>

# ALomancy

**Modular active learning for machine-learned interatomic potentials**

[![PyPI version](https://badge.fury.io/py/alomancy.svg)](https://badge.fury.io/py/alomancy)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/julianholland/ALomancy/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/julianholland/ALomancy/actions)
[![codecov](https://codecov.io/gh/julianholland/ALomancy/branch/master/graph/badge.svg)](https://codecov.io/gh/julianholland/ALomancy)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://readthedocs.org/projects/alomancy/badge/?version=latest)](https://alomancy.readthedocs.io/en/latest/index.html)

[Quick start](#quick-start) • [Swap anything](#swap-any-stage-with-one-line) • [What's available](#whats-available) • [Documentation](#documentation) • [Development](#development)

</div>

---

ALomancy runs the whole active-learning loop for a machine-learned
interatomic potential (MLIP): train a model, explore new structures with it,
pick the ones worth labelling, label them with DFT on your cluster, and
repeat. You describe the run in **one YAML file**. Every stage is a module
chosen by name, so changing the MLIP, the exploration method, the DFT code
or the selection strategy is a one-line edit, not new code.

```python
from alomancy import ALomancy

ALomancy("config.yaml").run()
```

or, from the shell, `alomancy run config.yaml`.

## Quick start

**1. Install**

```bash
pip install alomancy
```

**2. Tell ALomancy about your clusters.** An interactive wizard records each
HPC system (scheduler, partitions, memory, environment setup, how many jobs to
run at once) once per machine, and installs ALomancy on it:

```bash
alomancy add-hpc
alomancy list-hpc        # check what's configured
```

**3. Write a config and run it.** This is a complete cold-start run: ALomancy
builds and DFT-labels its own starting dataset (isolated atoms, dimers,
trimers, amorphous cells, Materials Project structures), then iterates. Any
setting you leave out takes its default.

```yaml
general:
  al_workflow: "committee_uncertainty"
  elements: ["C", "O"]
  num_of_al_loops: 10
  num_of_structures_per_loop: 50     # selected for DFT each loop
  verbose: 1
  dataset_kwargs:
    target_config_types: ["init_MP", "init_amorphous"]
    test_ratio: 0.1
  committee_uncertainty_kwargs:
    num_of_models_in_committee: 3
  train_filter:
    max_force: 100.0                 # eV/A; leave very-high-force structures out of training

initialization:
  max_time: "2H"
  hpc: "my_cpu_hpc"                  # a profile name from `alomancy add-hpc`
  amorphous_kwargs:
    num_of_amorphous_structures: 50
    densities_list: [1.0, 2.0]

training:
  trainer: "mace"
  max_time: "5H"
  hpc: "my_gpu_hpc"
  mace_kwargs:
    max_num_epochs: "dynamic"        # scaled to the training-set size

structure_generation:
  generator: "md"
  num_of_structures_to_generate: 500 # candidate pool to choose the 50 from
  max_time: "10H"
  hpc: "my_gpu_hpc"
  md_kwargs:
    steps: 20000
    temperature: 800
    structure_selection_kwargs:
      num_of_md_starts: 10

high_accuracy_evaluation:
  evaluator: "qe"
  max_time: "30m"
  max_go_time: "4H"
  force_ceiling: 100.0               # relax far-from-equilibrium structures before labelling
  hpc: "my_cpu_hpc"
```

```bash
alomancy run config.yaml
```

Each loop trains a committee of three MACE models, runs MD with the best one,
sends the 50 structures the committee disagrees on most to Quantum ESPRESSO,
adds them to the training set and starts again. Stop it at any point and run
the same command again: it picks up where it left off, down to the step
within a loop.

More complete, tested configs are in [`examples/configs/`](examples/configs/).

## Swap any stage with one line

Every change below is an edit to the config above. Nothing else changes.

**Use a different MLIP**, e.g. SevenNet instead of MACE:

```yaml
training:
  trainer: "sevennet"
```

**Explore differently**: a genetic-algorithm search (EZGA) instead of MD...

```yaml
structure_generation:
  generator: "ezga"
  ezga_kwargs:
    max_generations: 10
    population_size: 20
```

...or keep MD but run it at constant pressure:

```yaml
structure_generation:
  md_kwargs:
    ensemble: "npt"
    pressure: 1.0                    # GPa
```

**Use a different DFT code**, e.g. VASP:

```yaml
high_accuracy_evaluation:
  evaluator: "vasp"
```

**Change how structures are chosen.** Random selection is a baseline to
measure the others against; novelty selection picks the candidates least like
each other and like the data you already have. Both train a single model, so
remove `committee_uncertainty_kwargs`:

```yaml
general:
  al_workflow: "novelty_selection"   # or "random_selection"
```

**Start from data you already have** instead of a cold start. Imported labels
are normalised automatically, and only the initial structures your data
doesn't already cover are generated:

```yaml
general:
  start_from:
    xyz: "my_dft_data.xyz"           # or train_xyz + test_xyz, or a former run's database
```

**Set the DFT budget**: how many structures to label per loop, and how big a
candidate pool to choose them from:

```yaml
general:
  num_of_structures_per_loop: 20
structure_generation:
  num_of_structures_to_generate: 400
```

## What's available

| Stage | Config key | Options |
|---|---|---|
| Active-learning strategy | `general.al_workflow` | `committee_uncertainty` (default), `random_selection`, `novelty_selection` |
| MLIP trainer | `training.trainer` | `mace` (default), `sevennet` |
| Structure generation | `structure_generation.generator` | `md` (default), `ezga` |
| DFT labelling | `high_accuracy_evaluation.evaluator` | `qe` (default), `vasp` |
| Starting data | `general.start_from` | none (cold start), `xyz`, `train_xyz` + `test_xyz`, `database` |
| Initial structures | `initialization.<type>_kwargs` | isolated atoms, dimers, trimers, amorphous cells, Materials Project structures, stretched/compressed and rattled copies of target structures |

**Active-learning strategies**

- **`committee_uncertainty`**: trains a committee of models (3 by default)
  and selects the candidates with the largest disagreement in predicted
  forces.
- **`random_selection`**: trains one model and selects candidates uniformly
  at random, seeded for reproducibility. A baseline for comparing the other
  strategies.
- **`novelty_selection`**: trains one model and selects the candidates
  furthest, in a structural descriptor space, from each other and from every
  structure already in the database.

**Modules**

- **MACE**: committee training with stage-two/SWA models, energy-only or
  energy+stress loss, and an epoch count that scales with the dataset.
- **SevenNet**: SevenNet's own `model`/`train`/`data` settings, with the
  same evaluation, best-model selection and plots as MACE.
- **MD**: Langevin NVT or Langevin NPT (variable cell) with reproducible
  per-run seeds. Optional full-resolution trajectories.
- **EZGA**: genetic-algorithm structure search with configurable mutations
  (rattle, strain, add, remove).
- **Quantum ESPRESSO / VASP**: single points, or geometry optimisation below
  a force ceiling. Your settings are merged into sensible PBE defaults.

## What you get from every run

- **A report per loop** (`results/reports/al_loop_N/report.md`): error
  trends, dataset composition, DFT statistics, per-module sections and a list
  of **issues with suggested fixes**. Rebuild reports any time with
  `alomancy report`.
- **Plots**: parity plots per committee member, error per loop, training
  curves and a timing breakdown of every phase.
- **The current best model**, always at
  `results/best_model/ALomancy_best_model.model`, with its metadata and
  errors.
- **A single database** of every DFT-labelled structure, with
  near-duplicates and low-quality structures flagged rather than deleted, so
  any filter can be loosened later.
- **Restarts that cost nothing**: every step records when it's done, so a
  crashed or cancelled run resumes without redoing training, MD or DFT.

## Built for HPC

ALomancy submits every remote step (training, MD, DFT) through
[ExPyRe](https://github.com/libAtoms/ExPyRe). Concurrency is capped per HPC
profile, and transient connection or copy failures are retried instead of
losing the job. Each run keeps its own job state, so several runs can share a
machine. Structures that are far from equilibrium are relaxed until their
largest force is below `force_ceiling` before they are labelled, so that DFT
time isn't spent on structures the training filter would throw away.

## Command line

| Command | What it does |
|---|---|
| `alomancy run config.yaml` | Run the workflow described by a config |
| `alomancy report [--loop N \| --all]` | Rebuild the per-loop reports from a results directory |
| `alomancy results --replot` | Regenerate every plot from a results directory |
| `alomancy add-hpc` | Interactive wizard to add an HPC system |
| `alomancy list-hpc [--check-remote]` | List configured HPC systems (and their installed ALomancy version) |
| `alomancy upgrade-hpc` | Upgrade ALomancy on one or more HPC systems |
| `alomancy nuke` | Delete local ExPyRe job state |

## Extending ALomancy

Modules are registered by name in
[`src/alomancy/registry.py`](src/alomancy/registry.py). A new MLIP is an
`ALomancyTrainer` subclass that implements `fit`, `model_path` and
`get_calculator`; the base class handles evaluation, restarts, plots and
best-model selection:

```python
register("mlip_trainer", "my_mlip", "my_package.trainer", trainer_class="MyTrainer")
```

after which `trainer: "my_mlip"` works in any config. A new active-learning
strategy is a short `ActiveLearningWorkflow` subclass whose `run()` puts
the shared steps in order. See
[Writing a workflow](docs/writing_a_workflow.md).

## Documentation

- [Quick start](docs/quickstart.md) and [installation](docs/installation.md)
- [Configuration reference and examples](docs/examples.md), plus tested configs in [`examples/configs/`](examples/configs/)
- [Starting a run](docs/starting_a_run.md): cold start, warm start, continuing from a database
- [Loop reports](docs/reports.md)
- [Writing a workflow](docs/writing_a_workflow.md)
- [Remote submission architecture](docs/remote_submission_architecture.md)
- [Deprecations and migrating to 1.0](docs/deprecations.md)
- Full documentation: [alomancy.readthedocs.io](https://alomancy.readthedocs.io/en/latest/index.html)

## Development

```bash
git clone https://github.com/julianholland/ALomancy.git
cd ALomancy
uv sync                                     # alomancy (editable) + dev tools
pre-commit install

uv run pytest -m "unit and not slow" -n auto   # fast tier, ~20 s
uv run pytest -n auto                          # full suite
uv run ruff check src tests && uv run ruff format src tests
```

See the [contributing guide](docs/contributing.md).

## Citation

If you use ALomancy in your research, please cite:

```bibtex
@software{alomancy,
  title   = {ALomancy: Modular Active Learning Workflows for Modern Computational Chemistry},
  author  = {Julian Holland},
  year    = {2025},
  url     = {https://github.com/julianholland/ALomancy},
  version = {1.0.0}
}
```

## License

MIT.

## Acknowledgments

The Fritz Haber Institute of the Max Planck Society.

## Support

- Documentation: [alomancy.readthedocs.io](https://alomancy.readthedocs.io)
- Bug reports and questions: [GitHub Issues](https://github.com/julianholland/ALomancy/issues)
- Email: holland@fhi.mpg.de
