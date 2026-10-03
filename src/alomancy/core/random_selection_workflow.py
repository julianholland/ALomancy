"""RandomSelectionWorkflow: train one model, pick generated candidates at
random, label them with DFT, repeat.

A baseline to compare uncertainty-driven workflows against: it spends the
same DFT budget per loop (``structure_generation.desired_num_of_structures``)
but chooses which candidates to label uniformly at random. Also the
smallest complete example of an ``ActiveLearningWorkflow`` child -- see
docs/writing_a_workflow.md.

No settings of its own (no ``general.random_selection_kwargs``). The random
choice is seeded by ``general.seed + loop``, so a run is reproducible.
"""

import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from alomancy.core.active_learning_workflow import (
    _STRUCTURE_GENERATION_NAME,
    ActiveLearningWorkflow,
    LoopContext,
    TrainedModel,
    phase,
)

logger = logging.getLogger(__name__)

_SELECTED_FILENAME = "random_selection_structures.xyz"


def _selected_path(ctx: LoopContext) -> Path:
    return Path(
        "results", ctx.base_name, _STRUCTURE_GENERATION_NAME, _SELECTED_FILENAME
    )


def _load_selected(
    _self: "RandomSelectionWorkflow",
    ctx: LoopContext,
    *_args: object,
    **_kwargs: object,
) -> list[Atoms]:
    """generate_structures.done loader: last time's random selection."""
    return list(read(_selected_path(ctx), ":", format="extxyz"))


class RandomSelectionWorkflow(ActiveLearningWorkflow):
    """Single-model, random-selection baseline (see module docstring)."""

    NAME = "random_selection"
    NEW_STRUCTURE_CONFIG_TYPE = "random_selection"

    def run(self) -> None:
        for ctx in self.iterate_loops(self.prepare_run()):
            (model,) = self.train_models(ctx, self.seeds(1))
            if ctx.train_only:
                logger.info(
                    "Initial training complete; train_only stops before generation."
                )
                return
            selected = self.select_random(ctx, model)
            self.add_to_dataset(ctx, self.high_accuracy_evaluate(ctx, selected))
            self.finish_loop(ctx)

    @phase("generate_structures", load=_load_selected)
    def select_random(self, ctx: LoopContext, model: TrainedModel) -> list[Atoms]:
        """Generate candidates with *model* and keep
        desired_num_of_structures of them, chosen uniformly at random."""
        candidates = self.generate_candidates(ctx, model)
        wanted = self.jobs_dict["structure_generation"]["desired_num_of_structures"]
        n = min(wanted, len(candidates))
        if n < wanted:
            logger.warning(
                "Only %d candidate(s) generated for %s; selecting all of them "
                "(desired_num_of_structures=%d).",
                len(candidates),
                ctx.base_name,
                wanted,
                extra={
                    "event": "fewer_candidates",
                    "data": {"n": len(candidates), "desired": wanted},
                },
            )
        rng = np.random.default_rng(self.seed + ctx.loop)
        chosen = sorted(rng.choice(len(candidates), size=n, replace=False))
        selected = [candidates[i] for i in chosen]
        path = _selected_path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, selected, format="extxyz")
        logger.info(
            "Selected %d of %d candidate(s) at random for DFT.", n, len(candidates)
        )
        return selected
