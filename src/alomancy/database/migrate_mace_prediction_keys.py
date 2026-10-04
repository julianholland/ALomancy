"""One-time migration for the mace_* -> model_* prediction-key rename
(see the module registry / architecture plan for why: the trainer backend
is no longer assumed to be MACE). Run once, after upgrading, against an
existing results tree and its GlobalDatabase.

Two independent pieces:
- GlobalDatabase.migrate_mace_prediction_keys() (database/global_database.py)
  renames the persisted mace_energy_loop_*/mace_forces_loop_* metadata keys.
- rename_legacy_prediction_sentinels() below renames each AL loop's
  mace_predictions.done sentinel file to model_predictions.done, so a
  resumed run doesn't think prediction storage still needs (re-)doing for
  an already-migrated loop -- store_mlip_predictions now only checks for
  model_predictions.done.

migrate() runs both. Idempotent -- safe to run more than once.
"""

import logging
from pathlib import Path

from alomancy.database.global_database import GlobalDatabase

logger = logging.getLogger(__name__)

_LEGACY_SENTINEL_NAME = "mace_predictions.done"
_CURRENT_SENTINEL_NAME = "model_predictions.done"


def rename_legacy_prediction_sentinels(results_dir: str | Path = "results") -> int:
    """Rename every results/<base_name>/mace_predictions.done sentinel to
    model_predictions.done. Skips (with a warning, not an error) any AL
    loop that already has both files -- leaves the current one alone
    rather than guessing which is authoritative.

    Returns the number of sentinels renamed.
    """
    renamed = 0
    for legacy in Path(results_dir).glob(f"*/{_LEGACY_SENTINEL_NAME}"):
        current = legacy.with_name(_CURRENT_SENTINEL_NAME)
        if current.exists():
            logger.warning(
                "%s already exists alongside %s -- leaving both in place.",
                current,
                legacy,
            )
            continue
        legacy.rename(current)
        renamed += 1
        logger.info("Renamed %s -> %s.", legacy, current)
    logger.info("rename_legacy_prediction_sentinels: renamed %d sentinel(s).", renamed)
    return renamed


def migrate(
    results_dir: str | Path = "results",
    db_path: str | Path = "results/global_database",
) -> dict[str, int]:
    """Run the full mace_* -> model_* prediction-key migration: the
    GlobalDatabase metadata-key rename, plus the per-loop sentinel rename.

    Returns {"containers_updated": ..., "sentinels_renamed": ...}.
    """
    db = GlobalDatabase(str(db_path))
    containers_updated = db.migrate_mace_prediction_keys()
    sentinels_renamed = rename_legacy_prediction_sentinels(results_dir)
    return {
        "containers_updated": containers_updated,
        "sentinels_renamed": sentinels_renamed,
    }
