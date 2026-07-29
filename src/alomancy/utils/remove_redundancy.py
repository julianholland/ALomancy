"""Flag near-duplicate structures in the training split of the GlobalDatabase."""

import logging

import numpy as np

from alomancy.global_descriptor.atomic_distance_descriptor import (
    assign_descriptor_to_all_partition,
)

logger = logging.getLogger(__name__)


def remove_redundancy_from_partition(
    db, config_list: list, tolerance: float = 0.01
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
    """
    from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
        DistanceMatrix,
    )

    train_partition = db.get_split_partition("train")
    if len(train_partition) == 0:
        logger.warning("No train-split structures in DB — skipping redundancy removal.")
        return

    metadata = list(train_partition.get_metadata("config_type"))
    config_indices = [i for i, ct in enumerate(metadata) if ct in config_list]

    if not config_indices:
        logger.info(
            "No train structures matching config_list %s — nothing to dedup.",
            config_list,
        )
        return

    subset_p = train_partition.export_subset(
        config_indices,
        new_path=None,
        new_storage="memory",
        batch_size=500,
        verbose=False,
    )
    logger.info(
        "Assigning descriptors to %d train structures for redundancy removal.",
        len(subset_p),
    )
    assign_descriptor_to_all_partition(subset_p, dimensions=128)

    descriptor_array = np.array(list(subset_p.get_metadata("char_vec")))
    dm_dda = DistanceMatrix(
        tolerance=tolerance,
        dataset_array=descriptor_array,
        max_vector_array_size=len(descriptor_array),
    )
    dm_dda.get_dataset_unique_structures()
    unique_local = set(map(int, dm_dda.get_unique_vector_indices()))

    # Map local subset indices → positional indices in the global DB partition
    all_containers = list(db.partition.list_containers())
    train_global_indices = [
        i
        for i, c in enumerate(all_containers)
        if c.AtomPositionManager.metadata.get("split") == "train"
    ]
    dedup_global_indices = [train_global_indices[j] for j in config_indices]
    duplicate_global = [
        dedup_global_indices[j]
        for j in range(len(config_indices))
        if j not in unique_local
    ]

    logger.info(
        "Redundancy removal: %d/%d structures flagged as duplicates (tolerance=%.4f).",
        len(duplicate_global),
        len(config_indices),
        tolerance,
    )
    if duplicate_global:
        db.flag_as_duplicates(duplicate_global)
