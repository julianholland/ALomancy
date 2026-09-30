"""CommitteeUncertaintyWorkflow: train a committee, pick the structures it
disagrees on most, label them with DFT, repeat.

Each loop: train ``num_of_models_in_committee`` models on the same data
(different seeds) → run the structure generator (MD/EZGA) with the best
one → predict every candidate with every model → keep the candidates with
the highest force standard deviation across the committee
(``find_high_sd_structures``) → DFT → add to the dataset.

Everything generic (config, cold/warm start, resume, training, prediction,
DFT, dataset handling, restart sentinels) lives in
``ActiveLearningWorkflow`` (core/active_learning_workflow.py); this module
only declares the committee's settings, its loop order and its selection
rule. See docs/writing_a_workflow.md.

Settings: ``general.committee_uncertainty_kwargs.num_of_models_in_committee``
(default 3, the minimum for a usable force std-dev).
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

from ase import Atoms
from ase.io import read

from alomancy.analysis.plotting import mae_al_loop_plot
from alomancy.core.active_learning_workflow import (
    _STRUCTURE_GENERATION_NAME,
    _TRAINING_NAME,
    ActiveLearningWorkflow,
    LoopContext,
    TrainedModel,
    build_workflow,
    phase,
)
from alomancy.structure_generation.find_high_sd_structures import (
    find_high_sd_structures,
)

__all__ = ["CommitteeUncertaintyWorkflow", "build_workflow"]

logger = logging.getLogger(__name__)

# A force standard deviation needs at least this many models.
_MIN_COMMITTEE_SIZE = 3

_HIGH_SD_FILENAME = "high_sd_structures.xyz"


def _load_high_sd(
    _self: "CommitteeUncertaintyWorkflow",
    ctx: LoopContext,
    *_args: object,
    **_kwargs: object,
) -> list[Atoms]:
    """generate_structures.done loader: the selection find_high_sd_structures
    wrote last time."""
    path = Path("results", ctx.base_name, _STRUCTURE_GENERATION_NAME, _HIGH_SD_FILENAME)
    structures = list(read(path, ":", format="extxyz"))
    logger.info("%d High SD structures loaded from file: %s", len(structures), path)
    return structures


class CommitteeUncertaintyWorkflow(ActiveLearningWorkflow):
    """Committee force-std-dev active learning (see module docstring)."""

    NAME = "committee_uncertainty"
    KWARGS_KEY = "committee_uncertainty_kwargs"
    KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = {
        "num_of_models_in_committee": _MIN_COMMITTEE_SIZE
    }
    NEW_STRUCTURE_CONFIG_TYPE = "high_sd"

    def validate_settings(self) -> None:
        size = self.workflow_kwargs["num_of_models_in_committee"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < _MIN_COMMITTEE_SIZE
        ):
            raise ValueError(
                "general.committee_uncertainty_kwargs.num_of_models_in_committee "
                f"must be an integer >= {_MIN_COMMITTEE_SIZE} (a force standard "
                f"deviation needs at least {_MIN_COMMITTEE_SIZE} models), got {size!r}."
            )

    def run(self) -> None:
        committee_size = self.workflow_kwargs["num_of_models_in_committee"]
        for ctx in self.iterate_loops(self.prepare_run()):
            models = self.train_models(
                ctx, self.seeds(committee_size), min_successful=_MIN_COMMITTEE_SIZE
            )
            self.plot_loop(ctx)
            if ctx.train_only:
                logger.info(
                    "Initial committee training complete; train_only stops before generation."
                )
                return
            selected = self.select_uncertain(ctx, models)
            self.add_to_dataset(ctx, self.high_accuracy_evaluate(ctx, selected))
            self.finish_loop(ctx)

    @phase("generate_structures", load=_load_high_sd)
    def select_uncertain(
        self, ctx: LoopContext, models: list[TrainedModel]
    ) -> list[Atoms]:
        """Generate candidates with the best model, predict them with the
        whole committee and keep the most uncertain (highest force std-dev).
        find_high_sd_structures writes structure_generation/
        high_sd_structures.xyz, which is what a restart reloads."""
        best = self.best_model(models)
        candidates = self.generate_candidates(ctx, best)
        logger.info(
            "Structure generation: evaluating %d candidate structure(s) against "
            "the full %d-member committee to select the most uncertain ones.",
            len(candidates),
            len(models),
        )
        # The best model is listed first, as "base_mlip" -- the label
        # find_high_sd_structures expects for the model that drove the MD.
        ordered = [best, *(m for m in models if m.fit_idx != best.fit_idx)]
        predictions = self.predict(ctx, ordered, candidates)
        structure_forces_dict = {
            ("base_mlip" if model is best else f"fit_{model.fit_idx}"): {
                f"structure_{i}": {
                    "forces": prediction["forces"][i],
                    "energy": prediction["energies"][i],
                }
                for i in range(len(candidates))
            }
            for model, prediction in zip(ordered, predictions, strict=True)
        }
        # find_high_sd_structures reads structure_generation["name"] from the
        # job dict it's given; the name is hardcoded, not config, so it's
        # merged into a shallow copy here.
        sg_config = self.jobs_dict["structure_generation"]
        selected: list[Atoms] = find_high_sd_structures(
            structure_list=candidates,
            base_name=ctx.base_name,
            job_dict={
                **self.jobs_dict,
                "structure_generation": {
                    **sg_config,
                    "name": _STRUCTURE_GENERATION_NAME,
                },
            },
            structure_forces_dict=structure_forces_dict,
        )
        return selected

    def plot_loop(self, ctx: LoopContext) -> None:
        """This loop's training plots (unchanged from before the workflow
        split; plotting will be generalised separately)."""
        if not self.plots:
            return
        # The plotting functions read mlip_committee_job_dict["name"] and
        # the committee size; both are merged into one dict here.
        training_config_with_name = {
            **self.workflow_kwargs,
            **self.training_config,
            "name": _TRAINING_NAME,
        }
        mae_al_loop_plot(
            self._cross_loop_metrics_dataframe(_TRAINING_NAME),
            training_config_with_name,
            directory=ctx.plots_dir,
        )
        from alomancy.analysis.mlip_plots import (
            plot_dft_vs_model,
            plot_training_curves,
        )

        plot_training_curves(
            ctx.base_name, training_config_with_name, self.seed, ctx.plots_dir
        )
        plot_dft_vs_model(
            ctx.base_name,
            training_config_with_name,
            self.seed,
            ctx.plots_dir,
            db=self.db,
            loop_idx=ctx.loop,
        )
