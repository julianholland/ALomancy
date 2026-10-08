# Writing a Workflow

An ALomancy *workflow* decides how each active-learning loop chooses which
new structures to label with DFT. Two ship with ALomancy:

| `general.al_workflow` | Class | Strategy |
|---|---|---|
| `committee_uncertainty` (default) | `CommitteeUncertaintyWorkflow` | Train a committee; keep the candidates the committee disagrees on most (force standard deviation). |
| `random_selection` | `RandomSelectionWorkflow` | Train one model; keep candidates chosen uniformly at random. A baseline for comparing strategies. |

Every workflow is a subclass of `ActiveLearningWorkflow`
(`src/alomancy/core/active_learning_workflow.py`). The parent does
everything that is the same for every strategy. A new workflow only has to
declare its settings and write a short `run()`.

## What the parent provides

| Helper | What it does |
|---|---|
| `prepare_run()` | Everything before the first loop: config and HPC checks, then either resume from the database or build the initial dataset (see [Starting a run](starting_a_run.md)), then redundancy removal and the train/test filters. Returns the first loop to run. |
| `iterate_loops(start)` | Yields a `LoopContext` for each loop still to run: `loop`, `base_name` (`al_loop_N`), `workdir`, the current `train`/`test` structures, `train_only`, `plots_dir`. Writes `train_set.xyz`/`test_set.xyz`. |
| `train_models(ctx, seeds, *, min_successful=1)` | Trains one model per seed on this loop's shared train/validation/test split, **all in one parallel remote batch**. Returns a list of `TrainedModel` (`fit_idx`, `seed`, `model_path`, `compiled_model_path`, `metrics`, `fit_dir`), updates `results/best_model/`, and raises if fewer than `min_successful` succeed. |
| `seeds(n)` | `[general.seed, general.seed + 1, ...]`, the standard per-model seeds. |
| `best_model(models)` | The model with the lowest force error on the shared validation split (test split if there is none). |
| `generate_candidates(ctx, model)` | Runs the configured structure generator (MD, EZGA, ...) with `model`'s own calculator, starting from this loop's eligible training structures. Drops unphysical candidates. |
| `predict(ctx, models, structures)` | Energies and forces of `structures` from every model, as one parallel remote batch. One `{"forces": [...], "energies": [...]}` per model. |
| `high_accuracy_evaluate(ctx, structures)` | DFT-labels the chosen structures (relaxed up to `high_accuracy_evaluation.force_ceiling`), tagging them with your `NEW_STRUCTURE_CONFIG_TYPE` and the loop number. |
| `add_to_dataset(ctx, structures)` | Adds labelled structures to the database, split by `dataset_kwargs.test_ratio` (or `fixed_test` rules). |
| `finish_loop(ctx)` | Redundancy removal, train/test filters and curation on the grown dataset; marks the loop done. |

## What a workflow declares

```python
from typing import Any, ClassVar

from alomancy.core.active_learning_workflow import (
    ActiveLearningWorkflow,
    LoopContext,
    TrainedModel,
    phase,
)


class MyWorkflow(ActiveLearningWorkflow):
    NAME = "my_workflow"  # general.al_workflow value
    KWARGS_KEY = "my_workflow_kwargs"  # general.<key>; None if no settings
    KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = {"num_to_pick": 20}
    NEW_STRUCTURE_CONFIG_TYPE = "my_selection"  # config_type of structures it adds

    def validate_settings(self) -> None:  # optional
        if self.workflow_kwargs["num_to_pick"] < 1:
            raise ValueError("general.my_workflow_kwargs.num_to_pick must be >= 1")

    def run(self) -> None:  # the only required method
        for ctx in self.iterate_loops(self.prepare_run()):
            (model,) = self.train_models(ctx, self.seeds(1))
            if ctx.train_only:
                return
            selected = self.select(ctx, model)
            self.add_to_dataset(ctx, self.high_accuracy_evaluate(ctx, selected))
            self.finish_loop(ctx)

    @phase("generate_structures", load=load_my_selection)
    def select(self, ctx: LoopContext, model: TrainedModel) -> list:
        candidates = self.generate_candidates(ctx, model)
        ...  # choose, save to a file load_my_selection can read back
```

`self.workflow_kwargs` holds `KWARGS_DEFAULTS` merged with the user's
`general.<KWARGS_KEY>` block. Unknown keys in that block log a warning.
Generic settings live elsewhere and are available on `self`:
`self.dataset_kwargs`, `self.training_config`, `self.seed`,
`self.force_ceiling`, `self.db`.

Register the class so `general.al_workflow` can select it, in
`src/alomancy/registry.py`:

```python
register(
    "al_workflow",
    "my_workflow",
    "alomancy.core.my_workflow",
    workflow_class="MyWorkflow",
)
```

Users then select it from the config alone, with `al_workflow: my_workflow`
under `general`, and run it with `ALomancy("config.yaml").run()` (or
`alomancy run config.yaml`). They never import `MyWorkflow`. Constructing
`MyWorkflow` directly with a config that names a different `al_workflow`
raises `ValueError`.

`RandomSelectionWorkflow` (`src/alomancy/core/random_selection_workflow.py`)
is a complete example in under 100 lines.

## Restarts: the `phase` decorator

A run can stop at any point (a crashed job, a full disk, a queue limit)
and is restarted by running it again. Each step inside a loop that is
expensive to redo should be a `@phase(name, load=...)` method:

- When the step finishes, `results/<al_loop_N>/<name>.done` is written.
- On a restart, if that file exists, the step is skipped, and
  `load(self, ctx, *args, **kwargs)` returns its result from whatever the
  step saved. **Your step must save its result to a file its loader can
  read**; the decorator does not store return values itself.
- Each phase name may run once per loop. Calling the same phase twice in a
  loop raises, because both calls would share one sentinel. Pass
  `phase_name="..."` to give a second call its own.

The parent's steps are already phases: `train_models` (sentinel
`train_mlip`) and `high_accuracy_evaluate` (sentinel `high_accuracy_eval`,
reloading `high_accuracy_eval_results.xyz`). `generate_candidates` is
cached by the generator itself. Wrap your own *selection* step, as above,
typically with the name `generate_structures`. Together with `loop.done`
(written by `finish_loop`), that gives every loop four restart points:
after training, after selection, after DFT, and at the end.

Within `train_models`, individual models are also cached. If a loop dies
after three of five models finished, a restart only trains the other two.
Each model's seed is recorded (`training/fit_i/fit_seed.json`). If the seed
it was trained with differs from the one requested (e.g. `general.seed`
changed mid-training), the model is reused and a warning is logged.

## Rules that are easy to break

- **Remote functions must be module-level.** Anything submitted to an HPC
  node (a trainer's `train`, `mlip/predict.predict_with_model`, generator
  and DFT workers) is pickled *by reference* by ExPyRe and re-imported on
  the remote node. Never submit a method, a lambda or a nested function,
  and reinstall ALomancy on every HPC host after moving or renaming one.
- **Keep results file names stable.** Restarts find finished work by file
  name, so renaming a saved file or a phase makes old runs redo that step.
- **Set `NEW_STRUCTURE_CONFIG_TYPE`.** It labels the structures your
  workflow adds, and it's used for redundancy removal and validation-split
  eligibility. Reusing another workflow's label (e.g. `high_sd`) mixes the
  two in those steps.
- **Plots are committee-shaped.** The per-loop training plots were written
  for a committee. `CommitteeUncertaintyWorkflow.plot_loop` calls them; a
  new workflow can call the same functions or skip plotting. Plotting will
  be generalised separately.

## Adding an MLIP trainer

Trainers are separate from workflows: any workflow trains through
`train_models`, which runs whichever trainer `training.trainer` names. A
trainer subclasses `ALomancyTrainer` (`src/alomancy/mlip/base.py`) and
implements three methods; evaluation, restart checks, isolated-atom
energies and clean-up are inherited.

```python
# src/alomancy/mlip/sevennet/trainer.py
from pathlib import Path

from alomancy.mlip.base import ALomancyTrainer
from alomancy.utils.training_schedule import resolve_epochs


class SevenNetTrainer(ALomancyTrainer):
    NAME = "sevennet"  # training.trainer value
    KWARGS_KEY = "sevennet_kwargs"  # its settings: training.sevennet_kwargs
    KWARGS_DEFAULTS = {"epoch": "dynamic", "batch_size": 8}
    # The backend's name for per-element isolated-atom energies. ALomancy
    # fills it from the database's IsolatedAtom energies unless the user
    # sets it; None if the backend doesn't use them.
    ISOLATED_ATOM_ENERGIES_KWARG = "elemwise_reference_energies"

    def model_path(self, fit_dir: Path) -> Path:
        return fit_dir / "checkpoint_best.pth"

    def get_calculator(self, model_path, *, device=None):
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator(str(model_path), device=device or "auto")

    def fit(
        self,
        train_path,
        valid_path,
        test_path,
        seed,
        fit_dir,
        *,
        isolated_atom_energies,
    ):
        epochs = resolve_epochs(
            self.kwargs["epoch"], self.kwargs["batch_size"], n_structures
        )
        ...  # run SevenNet in fit_dir with self.kwargs, epochs and seed
        return self.model_path(fit_dir)  # or None if no model was produced
```

Then register it in `src/alomancy/registry.py`:

```python
register(
    "mlip_trainer",
    "sevennet",
    "alomancy.mlip.sevennet.trainer",
    trainer_class="SevenNetTrainer",
)
```

- **`train` is the same for every backend:** split paths and a seed in, the
  model's path (or `None`) out. Metrics are read back from the
  `evaluation_metrics.json` that `evaluate()` writes, never returned.
- **Override only what differs:** `format_isolated_atom_energies` (MACE
  converts the dict to its own string), `deployable_model_path` (what
  `results/best_model/` gets; MACE uses its compiled model),
  `cleanup_paths` (files to delete after a fit), `report_section` (the
  loop report).
- **`get_calculator` is how every other part of ALomancy uses your model**:
  evaluation, committee prediction and structure generation. Generators get
  a `CalculatorSpec` (trainer name, config, model path) and call
  `spec.build()` on the HPC node (MD) or with `device="cpu"` in the driver
  (EZGA), so MD and EZGA work with any registered trainer without changes.
- Training runs remotely through the module-level `run_training`, which
  builds the trainer on the HPC node from its registry name, so changes
  to a trainer need `alomancy upgrade-hpc` before the next run.
