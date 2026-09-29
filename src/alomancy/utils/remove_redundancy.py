"""Flag near-duplicate structures in the training split of the GlobalDatabase."""

import logging
import warnings
from typing import Any

import numpy as np

from alomancy.global_descriptor.atomic_distance_descriptor import make_char_vec

logger = logging.getLogger(__name__)


def descriptor_key(dimensions: int) -> str:
    """Container-metadata key for a cached descriptor. Dimension-specific, so
    changing ``dimensions`` never reuses vectors of the wrong length."""
    return f"char_vec_{dimensions}"


def _cached_descriptors(
    db: Any, all_containers: list, global_indices: list[int], dimensions: int
) -> np.ndarray:
    """Descriptor per container, reading the copy cached in the DB and
    computing (and persisting) only the ones not stored yet. Structures in
    the DB never change, so a stored descriptor is always valid."""
    key = descriptor_key(dimensions)
    vectors: list[list[float]] = []
    new: dict[int, list[float]] = {}
    for i in global_indices:
        apm = all_containers[i].AtomPositionManager
        vector = apm.metadata.get(key)
        if vector is None:
            vector = make_char_vec(apm, dimensions=dimensions).tolist()
            new[i] = vector
        vectors.append(vector)
    logger.info(
        "Redundancy descriptors: %d reused from the DB, %d computed.",
        len(global_indices) - len(new),
        len(new),
    )
    if new:
        db.store_descriptors(new, key)
    return np.array(vectors)


def remove_redundancy_from_partition(
    db, config_list: list, tolerance: float = 0.01, dimensions: int = 128
) -> None:
    """Flag near-duplicate structures in the training split of *db*.

    Only structures whose config_type is in config_list are subject to dedup.
    Near-duplicates receive is_duplicate=True in the DB metadata — they are
    never deleted (the DB is a full DFT archive) but are excluded from
    get_train_atoms(exclude_duplicates=True) and therefore from train XYZ files.

    Structures NOT in config_list keep whatever is_duplicate state they already have.

    Args:
        db: GlobalDatabase instance.
        config_list: config_types subject to redundancy removal
            (e.g. ["init_amorphous", "high_sd"]).
        tolerance: Euclidean distance threshold in descriptor space. Pairs closer
            than this are considered duplicates; the later-encountered one is flagged.
        dimensions: descriptor length. Descriptors are cached in the DB
            (``descriptor_key(dimensions)``) and only computed for structures
            that don't have one yet.
    """
    from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
        DistanceMatrix,
    )
    from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
        NaturalTolerancePlateauProbe,
    )

    all_containers = list(db.partition.list_containers())
    train_global_indices = [
        i
        for i, c in enumerate(all_containers)
        if c.AtomPositionManager.metadata.get("split") == "train"
    ]
    if not train_global_indices:
        logger.warning("No train-split structures in DB — skipping redundancy removal.")
        return

    dedup_global_indices = [
        i
        for i in train_global_indices
        if all_containers[i].AtomPositionManager.metadata.get("config_type")
        in config_list
    ]
    if not dedup_global_indices:
        logger.info(
            "No train structures matching config_list %s — nothing to dedup.",
            config_list,
        )
        return

    descriptor_array = _cached_descriptors(
        db, all_containers, dedup_global_indices, dimensions
    )

    dm_dda = DistanceMatrix(
        dataset_array=descriptor_array,
        max_vector_array_size=len(descriptor_array),
    )

    # calculate_tolerance can fail two distinct ways when the data doesn't
    # support a confident plateau: it raises ValueError outright when there
    # are too few probe steps to even attempt gradient detection (small
    # config_list-matching subsets), or it emits a "No plateaus found"
    # warning and returns an arbitrary fallback tolerance (midpoint of the
    # all-same/all-different bounds) when the probe ran but found nothing
    # stable. Neither case should flag any structure as a duplicate --
    # skip redundancy removal for this call rather than crashing or
    # applying a tolerance with no real relationship to the data.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            tolerance = NaturalTolerancePlateauProbe(
                duplicate_detection_algorithm_object=dm_dda,
                tolerance_dataset_array=descriptor_array,
                probe_steps=len(descriptor_array),
                probe_buffer_fraction=0.01,
            ).calculate_tolerance(condition="minimum")
        except ValueError as exc:
            logger.info(
                "Not enough structures (%d) to probe a natural tolerance "
                "plateau (%s) — skipping redundancy removal for this call.",
                len(descriptor_array),
                exc,
            )
            return

    if any("No plateaus found" in str(w.message) for w in caught):
        logger.info(
            "No tolerance plateau isolated among %d structures — skipping "
            "redundancy removal for this call rather than using an "
            "arbitrary fallback tolerance.",
            len(descriptor_array),
        )
        return

    dm_dda.tolerance = tolerance
    dm_dda.get_dataset_unique_structures()
    unique_local = set(map(int, dm_dda.get_unique_vector_indices()))

    duplicate_global = [
        dedup_global_indices[j]
        for j in range(len(dedup_global_indices))
        if j not in unique_local
    ]

    logger.info(
        "Redundancy removal: %d/%d structures flagged as duplicates (tolerance=%.4f).",
        len(duplicate_global),
        len(dedup_global_indices),
        tolerance,
    )
    if duplicate_global:
        db.flag_as_duplicates(duplicate_global)
