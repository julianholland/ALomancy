"""Quality filters for the train and test splits of the GlobalDatabase.

Configured by ``general.train_filter`` / ``general.test_filter`` (same
schema). A structure failing its split's filter gets
``is_quality_filtered=True`` plus ``quality_filter_reasons`` in DB metadata:
it is never deleted (the DB is a full DFT archive) but is left out of
``get_train_atoms()``/``get_test_atoms()`` and therefore out of the xyz
files, training and test metrics. Flags are rewritten on every pass, so
loosening or switching off a filter brings excluded structures back.
"""

import logging
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FILTER_KEYS = frozenset({"max_force", "formation_energy_per_atom"})

# train keeps the protection the old general.high_force_threshold gave by
# default (flag max force >= 100 eV/Angstrom); test is entirely opt-in.
TRAIN_FILTER_DEFAULTS: dict[str, Any] = {
    "max_force": 100.0,
    "formation_energy_per_atom": None,
}
TEST_FILTER_DEFAULTS: dict[str, Any] = {
    "max_force": None,
    "formation_energy_per_atom": None,
}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def validate_split_filter(cfg: Any, name: str) -> None:
    """Raise ValueError for a malformed ``general.<name>`` block. ``None``
    (block absent) is valid."""
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        raise ValueError(f"general.{name} must be a mapping.")
    unknown = sorted(set(cfg) - FILTER_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown general.{name} key(s) {unknown}; allowed: {sorted(FILTER_KEYS)}."
        )
    max_force = cfg.get("max_force")
    if max_force is not None and not (_is_finite_number(max_force) and max_force > 0):
        raise ValueError(
            f"general.{name}.max_force must be a positive number (eV/Angstrom) "
            f"or null, got {max_force!r}."
        )
    window = cfg.get("formation_energy_per_atom")
    if window is None:
        return
    if not isinstance(window, list | tuple) or len(window) != 2:
        raise ValueError(
            f"general.{name}.formation_energy_per_atom must be [min, max] "
            f"(eV/atom; either may be null) or null, got {window!r}."
        )
    lo, hi = window
    for bound in (lo, hi):
        if bound is not None and not _is_finite_number(bound):
            raise ValueError(
                f"general.{name}.formation_energy_per_atom bounds must be finite "
                f"numbers or null, got {window!r}."
            )
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(
            f"general.{name}.formation_energy_per_atom minimum {lo} is above "
            f"its maximum {hi}."
        )


def resolve_split_filter(cfg: dict | None, split: str) -> dict:
    """The user's block merged over that split's defaults."""
    defaults = TRAIN_FILTER_DEFAULTS if split == "train" else TEST_FILTER_DEFAULTS
    return {**defaults, **(cfg or {})}


def _force_reason(forces: Any, max_force: float) -> str | None:
    arr = None if forces is None else np.asarray(forces)
    if (
        arr is None
        or arr.ndim != 2
        or arr.shape[1] != 3
        or arr.size == 0
        or not np.isfinite(arr).all()
    ):
        return "invalid_forces"
    if np.linalg.norm(arr, axis=1).max() >= max_force:
        return "high_force"
    return None


def _scalar_energy(energy: Any) -> float | None:
    """sage_lib stores energies as one-element arrays; None if missing or
    non-finite."""
    if energy is None:
        return None
    arr = np.asarray(energy, dtype=float).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def apply_split_filter(db: Any, split: str, cfg: dict | None) -> dict[str, int]:
    """Flag the structures of *split* that fail *cfg* (already resolved
    against its defaults, see ``resolve_split_filter``).

    - ``max_force`` (eV/Angstrom): max per-atom force norm >= value, or
      missing/non-finite forces, fails.
    - ``formation_energy_per_atom`` ([min, max] eV/atom, null bound = open):
      ``(E - sum(E0[element])) / n_atoms`` against the DB's IsolatedAtom
      energies (the parity plots' definition); outside the window fails.
      Structures containing an element with no IsolatedAtom energy are not
      energy-filtered (one warning per call).

    Every container of *split* has its flags rewritten, so a disabled
    filter (all criteria null) clears earlier flags. Returns excluded
    counts per reason plus ``"total"``/``"excluded"``.
    """
    cfg = cfg or {}
    max_force = cfg.get("max_force")
    window = cfg.get("formation_energy_per_atom")
    e0: dict[str, float] = {}
    if window is not None:
        for element, value in db.get_isolated_atom_energies().items():
            scalar = _scalar_energy(value)
            if scalar is not None:
                e0[element] = scalar

    updates: dict[int, dict] = {}
    reasons_count: Counter[str] = Counter()
    missing_e0: Counter[str] = Counter()
    n_missing_e0 = 0
    for i, container in enumerate(db.partition.list_containers()):
        apm = container.AtomPositionManager
        if apm.metadata.get("split") != split:
            continue
        reasons: list[str] = []
        meta: dict[str, Any] = {}
        if max_force is not None:
            reason = _force_reason(apm.forces, max_force)
            if reason:
                reasons.append(reason)
        if window is not None:
            symbols = [str(s) for s in apm.atomLabelsList]
            absent = {s for s in symbols if s not in e0}
            energy = _scalar_energy(apm.energy)
            if absent:
                n_missing_e0 += 1
                missing_e0.update(absent)
            elif energy is not None and symbols:
                ef = (energy - sum(e0[s] for s in symbols)) / len(symbols)
                meta["REF_formation_energy_per_atom_e0"] = ef
                lo, hi = window
                if (lo is not None and ef < lo) or (hi is not None and ef > hi):
                    reasons.append("formation_energy")
        reasons_count.update(reasons)
        updates[i] = {
            **meta,
            "is_quality_filtered": bool(reasons),
            "quality_filter_reasons": reasons,
        }

    if updates:
        db.partition.set_metadata_bulk(updates, use_indices=True)
    if n_missing_e0:
        logger.warning(
            "%s filter: %d structure(s) contain element(s) %s with no "
            "IsolatedAtom energy in the DB; their formation energy is not "
            "filtered.",
            split,
            n_missing_e0,
            sorted(missing_e0),
        )
    n_excluded = sum(1 for u in updates.values() if u["is_quality_filtered"])
    logger.info(
        "%s filter: %d/%d structure(s) excluded %s (max_force=%s, "
        "formation_energy_per_atom=%s).",
        split,
        n_excluded,
        len(updates),
        dict(reasons_count),
        max_force,
        window,
        extra={
            "event": "quality_filtered",
            "data": {
                "split": split,
                "n": n_excluded,
                "total": len(updates),
                "by_reason": dict(reasons_count),
            },
        },
    )
    return {**reasons_count, "total": len(updates), "excluded": n_excluded}
