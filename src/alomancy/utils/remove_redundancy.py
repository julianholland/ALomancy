"""Flag near-duplicate structures in the training split of the GlobalDatabase."""

import json
import logging
import os
from pathlib import Path
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


def descriptors_for_atoms(atoms_list: list, dimensions: int = 128) -> np.ndarray:
    """Descriptor per ASE structure not stored in the DB (e.g. AL candidates).

    Builds each structure's AtomPositionManager the same way
    GlobalDatabase._prepare_for_storage does, so the vectors are directly
    comparable with the cached ``descriptor_key(dimensions)`` ones. Nothing
    is cached: the structures aren't in the DB yet.
    """
    from sage_lib.single_run.SingleRun import SingleRun

    vectors = []
    for atoms in atoms_list:
        run = SingleRun()
        run.AtomPositionManager.configure(
            atomPositions=atoms.positions,
            atomLabels=atoms.symbols,
            latticeVectors=atoms.cell,
        )
        vectors.append(make_char_vec(run.AtomPositionManager, dimensions=dimensions))
    return np.array(vectors)


# general.redundancy_kwargs. tolerance: a fixed duplicate tolerance, which
# skips the automatic choice. Otherwise the tolerance is the start of the
# lowest plateau of the sweep at least min_plateau_width (a fraction of the
# swept range) wide, and nothing is flagged when that would flag more than
# max_flagged_fraction of the pool.
REDUNDANCY_KWARGS_DEFAULTS: dict[str, Any] = {
    "tolerance": None,
    "probe_steps_per_structure": 10,
    "min_plateau_width": 0.0025,
    "max_flagged_fraction": 0.5,
}
_MIN_PROBE_STEPS = 20
_MAX_PROBE_STEPS = 2000
# A plateau: the kept count unchanged over at least this many sweep steps.
_PLATEAU_MIN_STEPS = 3
_MIN_STRUCTURES_TO_PROBE = 3


def validate_redundancy_kwargs(cfg: Any) -> dict[str, Any]:
    """general.redundancy_kwargs merged over its defaults; raises
    ValueError on an unknown key or a bad value (checked at construction)."""
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("general.redundancy_kwargs must be a mapping.")
    unknown = sorted(set(cfg) - set(REDUNDANCY_KWARGS_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown general.redundancy_kwargs key(s) {unknown}; "
            f"allowed: {sorted(REDUNDANCY_KWARGS_DEFAULTS)}."
        )
    settings = {**REDUNDANCY_KWARGS_DEFAULTS, **cfg}
    tolerance = settings["tolerance"]
    if tolerance is not None and (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance < 0
    ):
        raise ValueError(
            "general.redundancy_kwargs.tolerance must be null or a number >= 0."
        )
    steps = settings["probe_steps_per_structure"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(
            "general.redundancy_kwargs.probe_steps_per_structure must be an "
            "integer >= 1."
        )
    width = settings["min_plateau_width"]
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not (0 <= width < 1)
    ):
        raise ValueError(
            "general.redundancy_kwargs.min_plateau_width must be a fraction of "
            "the swept tolerance range in [0, 1)."
        )
    cap = settings["max_flagged_fraction"]
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not (0 < cap <= 1):
        raise ValueError(
            "general.redundancy_kwargs.max_flagged_fraction must be in (0, 1]."
        )
    return settings


def _distance_matrix(descriptor_array: np.ndarray):
    from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
        DistanceMatrix,
    )

    dda = DistanceMatrix(
        dataset_array=descriptor_array, max_vector_array_size=len(descriptor_array)
    )
    dda.pre_dda_processing()
    return dda


def tolerance_sweep(descriptor_array: np.ndarray, steps: int) -> dict[float, int]:
    """{tolerance: structures kept} for *steps* tolerances from 0 to the
    largest distance from the first structure, where one is kept.

    Counts are DistanceMatrix's (deduplicate_lib >= 0.1.3): structures in
    order, each kept unless within the tolerance of one already kept -- the
    same rule the flagging uses. Starts at 0, so exact duplicates (distance
    0) are flagged from the first step above it.
    """
    dda = _distance_matrix(descriptor_array)
    distances = dda.get_filled_distance_matrix()
    top = float(distances[0].max()) * (1 + 1e-9) if len(distances) > 1 else 0.0
    sweep: dict[float, int] = {}
    for tolerance in np.linspace(0.0, top, steps):
        dda.tolerance = float(tolerance)
        sweep[float(tolerance)] = int(dda.get_dataset_unique_structures())
    return sweep


def find_plateaus(sweep: dict[float, int]) -> list[tuple[float, float, int]]:
    """(start, end, kept) for every stretch of the sweep where the kept
    count stays the same over at least _PLATEAU_MIN_STEPS steps."""
    tolerances = sorted(sweep)
    plateaus = []
    start = 0
    for i in range(1, len(tolerances) + 1):
        if i == len(tolerances) or sweep[tolerances[i]] != sweep[tolerances[start]]:
            if i - 1 - start >= _PLATEAU_MIN_STEPS:
                plateaus.append(
                    (tolerances[start], tolerances[i - 1], sweep[tolerances[start]])
                )
            start = i
    return plateaus


def choose_tolerance(
    sweep: dict[float, int], n_structures: int, settings: dict[str, Any]
) -> tuple[float | None, str, list[tuple[float, float, int]]]:
    """(tolerance or None, outcome, plateaus) for a sweep.

    Plateaus where every structure is kept say nothing about duplicates and
    are skipped, as are plateaus narrower than min_plateau_width of the
    swept range. The start of the lowest remaining plateau is the
    tolerance, unless it would flag more than max_flagged_fraction of the
    structures (outcome "cap_exceeded"). Outcome "no_plateau" when none
    qualifies, "applied" otherwise.
    """
    plateaus = find_plateaus(sweep)
    swept = max(sweep) - min(sweep) if sweep else 0.0
    candidates = [
        p
        for p in plateaus
        if p[2] < n_structures and p[1] - p[0] >= settings["min_plateau_width"] * swept
    ]
    if not candidates:
        return None, "no_plateau", plateaus
    start, _end, kept = min(candidates, key=lambda p: p[0])
    if (n_structures - kept) / n_structures > settings["max_flagged_fraction"]:
        return None, "cap_exceeded", plateaus
    return start, "applied", plateaus


def probe_redundancy_tolerance(
    descriptor_array: np.ndarray,
    dimensions: int = 128,
    settings: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any] | None]:
    """The duplicate tolerance for *descriptor_array* and the record of the
    probe behind it (see tolerance_sweep and choose_tolerance; with
    settings["tolerance"] set, that value, outcome "fixed").

    Returns (None, record) when no tolerance qualifies -- nothing should be
    flagged then -- and (None, None) when there are too few structures to
    probe.
    """
    settings = validate_redundancy_kwargs(settings)
    n = len(descriptor_array)
    if n < _MIN_STRUCTURES_TO_PROBE:
        logger.info("Not enough structures (%d) to probe a redundancy tolerance.", n)
        return None, None
    steps = int(
        np.clip(
            settings["probe_steps_per_structure"] * n,
            _MIN_PROBE_STEPS,
            _MAX_PROBE_STEPS,
        )
    )
    sweep = tolerance_sweep(descriptor_array, steps)
    if settings["tolerance"] is not None:
        tolerance: float | None = float(settings["tolerance"])
        outcome = "fixed"
        plateaus = find_plateaus(sweep)
    else:
        tolerance, outcome, plateaus = choose_tolerance(sweep, n, settings)
    tolerances = sorted(sweep)
    record = {
        "tolerances": tolerances,
        "unique_counts": [sweep[t] for t in tolerances],
        "plateaus": [[float(a), float(b)] for a, b, _ in plateaus],
        "chosen_tolerance": tolerance,
        "outcome": outcome,
        "n_structures": n,
        "settings": settings,
        "descriptor": {
            "key": descriptor_key(dimensions),
            "dimensions": dimensions,
            "metric": "euclidean",
        },
    }
    return tolerance, record


def _write_probe_record(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)


def _pool_indices(all_containers: list, config_list: list[str]) -> list[int]:
    """Positions of the structures redundancy removal compares: in the
    training set (global_database.is_in_training_set, so never one already
    flagged, quality-filtered or rejected by curation) and of a listed
    config_type."""
    from alomancy.database.global_database import is_in_training_set

    return [
        i
        for i, c in enumerate(all_containers)
        if is_in_training_set(c.AtomPositionManager.metadata)
        and c.AtomPositionManager.metadata.get("config_type") in config_list
    ]


def remove_redundancy_from_partition(
    db,
    config_list: list,
    dimensions: int = 128,
    probe_path: Path | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    """Flag near-duplicate structures in the training set of *db*.

    Compared: the structures in the training set whose config_type is in
    config_list (_pool_indices); anything already out of the training set
    never counts as an original. In DB order, each is kept unless it is
    within the tolerance of one already kept, so every group of
    near-duplicates keeps its earliest member. Flagged structures get
    is_duplicate=True -- never deleted (the DB is a full DFT archive), but
    excluded from get_train_atoms() and so from the train xyz files.
    Flags are never cleared here (GlobalDatabase.clear_duplicate_flags).

    Args:
        db: GlobalDatabase instance.
        config_list: config_types subject to redundancy removal
            (e.g. ["init_amorphous", "high_sd"]).
        dimensions: descriptor length. Descriptors are cached in the DB
            (``descriptor_key(dimensions)``) and only computed for structures
            that don't have one yet.
        probe_path: where to save the tolerance probe (JSON: the sweep of
            structures kept against tolerance, the plateaus, the chosen
            tolerance, the outcome and how many were flagged), for the loop
            report.
        settings: general.redundancy_kwargs (probe_redundancy_tolerance).
    """
    all_containers = list(db.partition.list_containers())
    if not any(
        c.AtomPositionManager.metadata.get("split") == "train" for c in all_containers
    ):
        logger.warning("No train-split structures in DB — skipping redundancy removal.")
        return

    pool = _pool_indices(all_containers, config_list)
    if not pool:
        logger.info(
            "No training structures matching config_list %s — nothing to dedup.",
            config_list,
        )
        return

    descriptor_array = _cached_descriptors(db, all_containers, pool, dimensions)
    tolerance, record = probe_redundancy_tolerance(
        descriptor_array, dimensions, settings
    )
    if tolerance is None:
        if record is not None:
            if record["outcome"] == "cap_exceeded":
                cap = record["settings"]["max_flagged_fraction"]
                logger.warning(
                    "Redundancy removal: the lowest tolerance plateau among %d "
                    "structures would flag more than %.0f%% of them "
                    "(general.redundancy_kwargs.max_flagged_fraction); nothing "
                    "flagged. Set general.redundancy_kwargs.tolerance to choose "
                    "one, or remove_redundancy: false.",
                    len(pool),
                    100 * cap,
                    extra={
                        "event": "redundancy_cap_hit",
                        "data": {"n": len(pool), "max_flagged_fraction": cap},
                    },
                )
            else:
                logger.info(
                    "No tolerance plateau isolated among %d structures -- "
                    "skipping redundancy removal for this call rather than "
                    "using an arbitrary tolerance.",
                    len(pool),
                )
            if probe_path is not None:
                _write_probe_record(probe_path, {**record, "n_flagged": 0})
        return

    dda = _distance_matrix(descriptor_array)
    dda.tolerance = tolerance
    dda.get_dataset_unique_structures()
    kept = set(map(int, dda.get_unique_vector_indices()))
    duplicate_global = [pool[j] for j in range(len(pool)) if j not in kept]

    logger.info(
        "Redundancy removal: %d/%d structures flagged as duplicates (tolerance=%.4f).",
        len(duplicate_global),
        len(pool),
        tolerance,
        extra={
            "event": "redundancy_flagged",
            "data": {
                "n": len(duplicate_global),
                "total": len(pool),
                "tolerance": float(tolerance),
            },
        },
    )
    if probe_path is not None and record is not None:
        _write_probe_record(probe_path, {**record, "n_flagged": len(duplicate_global)})
    if duplicate_global:
        db.flag_as_duplicates(duplicate_global)


def training_reference(db, config_list: list[str], dimensions: int = 128) -> np.ndarray:
    """Descriptors of the structures redundancy removal compares against
    (_pool_indices), for checking candidates before they reach the DB."""
    all_containers = list(db.partition.list_containers())
    pool = _pool_indices(all_containers, config_list)
    if not pool:
        return np.empty((0, dimensions))
    return _cached_descriptors(db, all_containers, pool, dimensions)


def select_non_redundant(
    ranked: np.ndarray, reference: np.ndarray, tolerance: float, n: int
) -> list[int]:
    """Positions in *ranked* (candidate descriptors, best first) of the
    first *n* that are not within *tolerance* of a *reference* descriptor
    or of a candidate already accepted -- what end-of-loop redundancy
    removal would flag. Fewer than *n* if the candidates run out."""
    accepted: list[int] = []
    for i, vector in enumerate(ranked):
        if len(accepted) == n:
            break
        if len(reference) and np.linalg.norm(reference - vector, axis=1).min() < (
            tolerance
        ):
            continue
        if (
            accepted
            and np.linalg.norm(ranked[accepted] - vector, axis=1).min() < tolerance
        ):
            continue
        accepted.append(i)
    return accepted
