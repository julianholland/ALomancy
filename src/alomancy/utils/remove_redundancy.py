"""Flag near-duplicate structures in the training split of the GlobalDatabase."""

import contextlib
import io
import json
import logging
import os
import warnings
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


def probe_redundancy_tolerance(
    descriptor_array: np.ndarray, dimensions: int = 128
) -> tuple[float | None, dict[str, Any] | None]:
    """The duplicate tolerance for *descriptor_array* from deduplicate_lib's
    NaturalTolerancePlateauProbe, and the record of the probe behind it.

    The probe sweeps the tolerance (Euclidean distance in descriptor space
    below which two structures are duplicates) between "everything is one
    structure" and "everything is unique", counts the unique structures at
    each step, and finds plateaus: stretches where that count barely
    changes. The start of the lowest plateau is the tolerance. Returns
    (None, record) when no plateau is found -- nothing should be flagged
    then -- and (None, None) when there are too few structures to probe.
    """
    from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
        DistanceMatrix,
    )
    from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
        NaturalTolerancePlateauProbe,
    )

    class _RecordingProbe(NaturalTolerancePlateauProbe):
        """Keeps the full sweep and the plateaus it found (the library's own
        plateau_data drops the last points)."""

        sweep: dict[float, int]
        plateaus: list[tuple[float, float, int]]

        def tolerance_probe(self, *args: Any, **kwargs: Any) -> dict:
            self.sweep = super().tolerance_probe(*args, **kwargs)
            return self.sweep

        def find_plateaus(self, *args: Any, **kwargs: Any) -> list:
            self.plateaus = super().find_plateaus(*args, **kwargs)
            return self.plateaus

    dda = DistanceMatrix(
        dataset_array=descriptor_array, max_vector_array_size=len(descriptor_array)
    )
    probe = _RecordingProbe(
        duplicate_detection_algorithm_object=dda,
        tolerance_dataset_array=descriptor_array,
        probe_steps=len(descriptor_array),
        probe_buffer_fraction=0.01,
    )
    # calculate_tolerance fails two ways when the data can't support a
    # plateau: ValueError (too few probe steps for a gradient) and a "No
    # plateaus found" warning with an arbitrary midpoint fallback. The
    # library also prints its whole plateau log to stdout.
    with (
        warnings.catch_warnings(record=True) as caught,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        warnings.simplefilter("always")
        try:
            tolerance: float | None = probe.calculate_tolerance(condition="minimum")
        except ValueError as exc:
            logger.info(
                "Not enough structures (%d) to probe a natural tolerance plateau (%s).",
                len(descriptor_array),
                exc,
            )
            return None, None
    found = not any("No plateaus found" in str(w.message) for w in caught)
    if not found:
        tolerance = None
    sweep = getattr(probe, "sweep", {})
    tolerances = sorted(sweep)
    record = {
        "tolerances": [float(t) for t in tolerances],
        "unique_counts": [int(sweep[t]) for t in tolerances],
        "plateaus": [
            [float(a), float(b)] for a, b, _ in getattr(probe, "plateaus", [])
        ],
        "chosen_tolerance": None if tolerance is None else float(tolerance),
        "outcome": "applied" if found else "no_plateau",
        "n_structures": len(descriptor_array),
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


def remove_redundancy_from_partition(
    db,
    config_list: list,
    tolerance: float = 0.01,
    dimensions: int = 128,
    probe_path: Path | None = None,
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
        probe_path: where to save the tolerance probe (JSON: the sweep of
            unique structures against tolerance, the plateaus, the chosen
            tolerance and how many were flagged), for the loop report.
    """
    from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
        DistanceMatrix,
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

    tolerance, record = probe_redundancy_tolerance(descriptor_array, dimensions)
    if tolerance is None:
        # Too few structures, or no stable plateau: flag nothing rather
        # than apply a tolerance with no real relationship to the data.
        if record is not None:
            logger.info(
                "No tolerance plateau isolated among %d structures -- skipping "
                "redundancy removal for this call rather than using an "
                "arbitrary fallback tolerance.",
                len(descriptor_array),
            )
            if probe_path is not None:
                _write_probe_record(probe_path, {**record, "n_flagged": 0})
        return

    dm_dda = DistanceMatrix(
        dataset_array=descriptor_array,
        max_vector_array_size=len(descriptor_array),
    )
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
        extra={
            "event": "redundancy_flagged",
            "data": {
                "n": len(duplicate_global),
                "total": len(dedup_global_indices),
                "tolerance": float(tolerance),
            },
        },
    )
    if probe_path is not None and record is not None:
        _write_probe_record(probe_path, {**record, "n_flagged": len(duplicate_global)})
    if duplicate_global:
        db.flag_as_duplicates(duplicate_global)
