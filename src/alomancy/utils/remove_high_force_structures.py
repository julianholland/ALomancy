"""Flag high-force structures in the training split of the GlobalDatabase."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_high_force_structures_from_partition(
    db, force_threshold: float = 100.0
) -> None:
    """Flag high-force structures in the training split of *db*.

    High-force structures receive is_high_force=True in the DB metadata — they
    are never deleted (the DB is a full DFT archive) but are excluded from
    get_train_atoms(exclude_high_force=True) and therefore from train XYZ files.

    Args:
        db: GlobalDatabase instance.
        force_threshold: Maximum allowed force magnitude (eV/Å). Structures
            whose maximum per-atom force component exceeds this are flagged.
    """
    train_partition = db.get_split_partition("train")
    if len(train_partition) == 0:
        logger.warning(
            "No train-split structures in DB — skipping high-force structure removal."
        )
        return

    # Collect local (partition-positional) indices of high-force structures.
    # Forces are stored in AtomPositionManager, not in metadata.
    high_force_local: list[int] = []
    for i, container in enumerate(train_partition.list_containers()):
        forces = container.AtomPositionManager.forces
        if forces is not None and np.max(np.abs(forces)) >= force_threshold:
            high_force_local.append(i)

    # Map local indices → positional indices in the global DB partition.
    all_containers = list(db.partition.list_containers())
    train_global_indices = [
        j
        for j, c in enumerate(all_containers)
        if c.AtomPositionManager.metadata.get("split") == "train"
    ]
    high_force_global = [train_global_indices[i] for i in high_force_local]

    logger.info(
        "High-force structure removal: %d/%d structures flagged (threshold=%.4f eV/Å).",
        len(high_force_global),
        len(train_global_indices),
        force_threshold,
    )
    if high_force_global:
        db.flag_as_high_force(high_force_global)
