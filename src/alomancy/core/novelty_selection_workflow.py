"""NoveltySelectionWorkflow: train one model, keep the generated candidates
least like anything already known, label them with DFT, repeat.

Each loop: train one model → generate candidates with it → describe every
candidate with redundancy removal's global descriptor (``char_vec_128``) →
build one distance matrix over the database's non-redundant structures
plus the candidates → binary-search (deduplicate_lib's
``binary_search_tolerance``) the largest tolerance at which at most
``general.num_of_structures_per_loop`` candidates have no
neighbour, candidate or database, closer than it → DFT those candidates →
add to the dataset.

The database rows are reference points only: a candidate close to existing
data is not novel even if it is far from the other candidates, but no
database structure is ever selected.

No settings of its own (no ``general.novelty_selection_kwargs``).
"""

import contextlib
import io
import logging
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)

from alomancy.core.active_learning_workflow import (
    _STRUCTURE_GENERATION_NAME,
    ActiveLearningWorkflow,
    LoopContext,
    TrainedModel,
    phase,
)
from alomancy.utils.remove_redundancy import _cached_descriptors, descriptors_for_atoms

logger = logging.getLogger(__name__)

_SELECTED_FILENAME = "novelty_selection_structures.xyz"
# Same descriptor length as redundancy removal, so cached DB vectors are reused.
_DESCRIPTOR_DIMENSIONS = 128


def _selected_path(ctx: LoopContext) -> Path:
    return Path(
        "results", ctx.base_name, _STRUCTURE_GENERATION_NAME, _SELECTED_FILENAME
    )


def _load_selected(
    _self: "NoveltySelectionWorkflow",
    ctx: LoopContext,
    *_args: object,
    **_kwargs: object,
) -> list[Atoms]:
    """generate_structures.done loader: last time's novelty selection."""
    return list(read(_selected_path(ctx), ":", format="extxyz"))


class _CandidateDistanceMatrix(DistanceMatrix):
    """DistanceMatrix whose unique count covers only the candidate rows
    (``first_candidate`` onwards), so deduplicate_lib's binary search
    targets the number of novel candidates rather than all unique rows.
    Reference rows still count as neighbours."""

    def __init__(self, dataset_array: np.ndarray, first_candidate: int) -> None:
        super().__init__(
            dataset_array=dataset_array, max_vector_array_size=len(dataset_array)
        )
        self.first_candidate = first_candidate

    def get_dataset_unique_structures(self) -> int:
        super().get_dataset_unique_structures()
        return int(self.unique_vector_indices[self.first_candidate :].sum())


def select_novel_indices(
    reference: np.ndarray, candidates: np.ndarray, n: int
) -> tuple[list[int], float]:
    """Indices of at most *n* candidate descriptors with no neighbour
    (another candidate or a *reference* row) within the largest tolerance
    that leaves at most *n* such candidates; returns (indices, tolerance).

    Reference rows go first: DistanceMatrix always marks row 0 unique, so
    with any reference data that quirk lands on a row that is dropped.
    """
    if len(candidates) <= n:
        return list(range(len(candidates))), 0.0
    if len(reference) == 0:
        logger.warning(
            "No reference structures in the database; novelty is judged among "
            "the candidates only, and candidate 0 is always kept."
        )
    first = len(reference)
    dataset = np.vstack([reference, candidates]) if first else candidates
    dda = _CandidateDistanceMatrix(dataset, first_candidate=first)
    search = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        tolerance_dataset_array=dataset,
        perturbations_per_vector=1,
    )
    # deduplicate_lib prints every search step and warns when no tolerance
    # hits the target exactly; route both through logging.
    steps = io.StringIO()
    with (
        warnings.catch_warnings(record=True) as caught,
        contextlib.redirect_stdout(steps),
    ):
        warnings.simplefilter("always")
        tolerance = search.binary_search_tolerance(
            target_unique_vectors=n, find_largest_tolerance_for_target=True
        )
    logger.debug("Novelty tolerance search:\n%s", steps.getvalue().rstrip())
    for w in caught:
        logger.warning(
            "Novelty tolerance search: %s",
            w.message,
            extra={"event": "novelty_inexact"},
        )

    dda.tolerance = tolerance
    dda.get_dataset_unique_structures()
    chosen = [int(i) - first for i in dda.get_unique_vector_indices() if i >= first]
    if len(chosen) > n:
        # No tolerance gave exactly n: keep the n most isolated.
        matrix = dda.get_filled_distance_matrix().copy()
        np.fill_diagonal(matrix, np.inf)
        nearest = matrix.min(axis=1)
        chosen = sorted(chosen, key=lambda i: -nearest[first + i])[:n]
    return sorted(chosen), float(tolerance)


class NoveltySelectionWorkflow(ActiveLearningWorkflow):
    """Single-model workflow selecting the most novel candidates (see module
    docstring)."""

    NAME = "novelty_selection"
    NEW_STRUCTURE_CONFIG_TYPE = "novelty_selection"

    def run(self) -> None:
        for ctx in self.iterate_loops(self.prepare_run()):
            (model,) = self.train_models(ctx, self.seeds(1))
            if ctx.train_only:
                logger.info(
                    "Initial training complete; train_only stops before generation."
                )
                return
            selected = self.select_novel(ctx, model)
            self.add_to_dataset(ctx, self.high_accuracy_evaluate(ctx, selected))
            self.finish_loop(ctx)

    def report_sections(
        self,
        stats: dict,
        plots_dir: Path | None,  # noqa: ARG002 -- text only
    ) -> list:
        """Loop report: the tolerance found and the reference-set size."""
        from alomancy.analysis.report.sections import novelty_section

        section = novelty_section(stats)
        return [section] if section else []

    def reference_descriptors(self) -> np.ndarray:
        """Descriptors of every database structure not flagged redundant
        (is_duplicate), read from the DB's cache and computed only where
        missing."""
        containers = list(self.db.partition.list_containers())
        indices = [
            i
            for i, c in enumerate(containers)
            if not c.AtomPositionManager.metadata.get("is_duplicate", False)
        ]
        if not indices:
            return np.empty((0, _DESCRIPTOR_DIMENSIONS))
        return _cached_descriptors(self.db, containers, indices, _DESCRIPTOR_DIMENSIONS)

    @phase("generate_structures", load=_load_selected)
    def select_novel(self, ctx: LoopContext, model: TrainedModel) -> list[Atoms]:
        """Generate candidates with *model* and keep up to
        general.num_of_structures_per_loop of them, chosen for highest novelty."""
        candidates = self.generate_candidates(ctx, model)
        wanted = self.num_of_structures_per_loop
        if len(candidates) <= wanted:
            logger.warning(
                "Only %d candidate(s) generated for %s; selecting all of them "
                "(general.num_of_structures_per_loop=%d).",
                len(candidates),
                ctx.base_name,
                wanted,
                extra={
                    "event": "fewer_candidates",
                    "data": {"n": len(candidates), "desired": wanted},
                },
            )
            chosen, tolerance = list(range(len(candidates))), 0.0
            reference_count = 0
        else:
            reference = self.reference_descriptors()
            reference_count = len(reference)
            chosen, tolerance = select_novel_indices(
                reference,
                descriptors_for_atoms(candidates, _DESCRIPTOR_DIMENSIONS),
                wanted,
            )

        selected = [candidates[i] for i in chosen]
        path = _selected_path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, selected, format="extxyz")
        logger.info(
            "Selected %d of %d candidate(s) for DFT by novelty against %d "
            "reference structure(s) (tolerance=%.4f).",
            len(selected),
            len(candidates),
            reference_count,
            tolerance,
            extra={
                "event": "novelty_selected",
                "data": {
                    "selected": len(selected),
                    "candidates": len(candidates),
                    "reference": reference_count,
                    "tolerance": tolerance,
                },
            },
        )
        return selected
