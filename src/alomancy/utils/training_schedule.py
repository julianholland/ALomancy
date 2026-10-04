"""Training-length rules shared by every MLIP trainer.

The number of epochs is set so a model sees roughly the same number of
samples whatever the training-set size: small early-loop sets get more
epochs, large late-loop sets fewer. Any trainer whose backend has an epoch
setting calls ``resolve_epochs`` (MACE: ``mace_kwargs.max_num_epochs``).
"""

import logging
import math

logger = logging.getLogger(__name__)

DYNAMIC_EPOCHS_TARGET_SAMPLES = 200_000
DYNAMIC_EPOCHS_FLOOR = 20
DYNAMIC_EPOCHS_CAP = 300


def dynamic_epochs(
    batch_size: int,
    n_structures: int,
    *,
    target_samples: int = DYNAMIC_EPOCHS_TARGET_SAMPLES,
    floor: int = DYNAMIC_EPOCHS_FLOOR,
    cap: int = DYNAMIC_EPOCHS_CAP,
) -> int:
    """epochs = ceil(target_samples * batch_size / n_structures), clamped to
    [floor, cap] with a warning when clamped.

    The cap prevents an absurd epoch count for a small early-loop training
    set; the floor keeps a data-rich late-loop set from being cut to a
    near-zero second training stage (MACE starts it at 80% of the epochs,
    so a floor of 20 leaves at least 4).
    """
    if n_structures <= 0:
        raise ValueError(f"n_structures must be positive, got {n_structures}.")
    raw = math.ceil(target_samples * batch_size / n_structures)
    epochs = max(floor, min(cap, raw))
    if epochs != raw:
        logger.warning(
            "Dynamic epoch formula produced %d epochs (batch_size=%d, "
            "n_structures=%d); clamped to %d.",
            raw,
            batch_size,
            n_structures,
            epochs,
        )
    return epochs


def resolve_epochs(
    configured: int | str | None, batch_size: int, n_structures: int
) -> int:
    """The epoch count to train for: ``None`` or ``"dynamic"`` (the default
    behaviour) gives ``dynamic_epochs``; a positive integer passes through."""
    if configured is None or configured == "dynamic":
        epochs = dynamic_epochs(batch_size, n_structures)
        logger.info(
            "Dynamic max_num_epochs resolved to %d (batch_size=%d, n_structures=%d).",
            epochs,
            batch_size,
            n_structures,
        )
        return epochs
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured <= 0
    ):
        raise ValueError(
            f"max_num_epochs must be a positive integer, 'dynamic' or null, got "
            f"{configured!r}."
        )
    return configured
