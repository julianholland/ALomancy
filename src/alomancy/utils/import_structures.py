"""Bring structures from outside this run into ALomancy's labelling scheme.

Used by the workflow's warm-start modes (``general.start_from``, see
``docs/starting_a_run.md``). Files written by ALomancy already carry
``config_type``/``REF_energy``/``REF_forces`` and pass through unchanged;
foreign files get their labels mapped from equivalent keys (or the
attached calculator) before anything is imported or sent to DFT.
"""

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

logger = logging.getLogger(__name__)

EXTERNAL_CONFIG_TYPE = "external"

# Auto-detected equivalents, tried in order after the canonical key and any
# user-supplied metadata_map entry.
CONFIG_TYPE_ALIASES = ("type", "label", "config", "config_type_name")
ENERGY_ALIASES = ("energy", "dft_energy", "total_energy")
FORCES_ALIASES = ("forces", "dft_forces")
STRESS_ALIASES = ("stress", "dft_stress")

METADATA_MAP_KEYS = frozenset({"config_type", "energy", "forces", "stress"})

# Operational metadata that belongs to the run that wrote a file, never to
# the run importing it: splits and flags are recomputed here, DB ids are
# reassigned, and per-loop model predictions refer to someone else's models.
_OPERATIONAL_INFO_KEYS = (
    "split",
    "global_db_id",
    "is_duplicate",
    "is_high_force",
    "is_quality_filtered",
    "quality_filter_reasons",
)
_OPERATIONAL_INFO_PREFIXES = ("model_", "mace_")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_structures(path: str | Path) -> list[Atoms]:
    """All frames of an extxyz file (a bare ase.io.read returns only the last)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    frames = read(path, ":", format="extxyz")
    return [frames] if isinstance(frames, Atoms) else list(frames)


def filter_by_elements(
    atoms_list: Iterable[Atoms], elements: Iterable[str], *, source: str = ""
) -> list[Atoms]:
    """Keep only structures made entirely of *elements*.

    Imported data may contain species the run doesn't model (e.g. Na frames
    in a carbon run); those are dropped before they reach the DB, so
    redundancy removal and the train/test filters never see them. Logs one
    WARNING (event ``start_from_elements_excluded``) when anything is dropped.
    """
    allowed = set(elements)
    kept: list[Atoms] = []
    foreign: set[str] = set()
    n_excluded = 0
    for atoms in atoms_list:
        extra = set(atoms.get_chemical_symbols()) - allowed
        if extra:
            foreign |= extra
            n_excluded += 1
        else:
            kept.append(atoms)
    if n_excluded:
        logger.warning(
            "%s: excluded %d structure(s) containing element(s) %s not in "
            "general.elements %s.",
            source or "import",
            n_excluded,
            sorted(foreign),
            sorted(allowed),
            extra={
                "event": "start_from_elements_excluded",
                "data": {
                    "source": source,
                    "excluded": n_excluded,
                    "foreign_elements": sorted(foreign),
                    "elements": sorted(allowed),
                },
            },
        )
    return kept


def strip_operational_metadata(atoms: Atoms) -> None:
    for key in _OPERATIONAL_INFO_KEYS:
        atoms.info.pop(key, None)
    for key in list(atoms.info):
        if key.startswith(_OPERATIONAL_INFO_PREFIXES):
            del atoms.info[key]


def _calc_value(atoms: Atoms, getter: str) -> Any:
    if atoms.calc is None:
        return None
    try:
        return getattr(atoms, getter)()
    except Exception:
        return None


def _first_info(atoms: Atoms, keys: Iterable[str | None]) -> tuple[str | None, Any]:
    for key in keys:
        if key and key in atoms.info and atoms.info[key] is not None:
            return key, atoms.info[key]
    return None, None


def _first_array(atoms: Atoms, keys: Iterable[str | None]) -> tuple[str | None, Any]:
    for key in keys:
        if key and key in atoms.arrays:
            return key, atoms.arrays[key]
    return None, None


def normalize_metadata(
    atoms_list: list[Atoms], metadata_map: dict | None = None, *, source: str = ""
) -> list[Atoms]:
    """Return copies of *atoms_list* carrying ALomancy's canonical labels.

    For each label the lookup order is: the canonical key already set, then
    the ``metadata_map`` key, then the auto-detected aliases, then the
    attached calculator (energy/forces/stress only):

    - ``config_type`` -> falls back to ``"external"`` when nothing matches.
    - ``REF_energy`` / ``REF_forces`` -> required; any structure left without
      one raises ``ValueError`` (before any import or DFT) naming the keys
      tried and suggesting ``metadata_map``.
    - ``REF_stresses`` -> optional, never an error.

    Operational metadata from the run that wrote the file (splits, DB ids,
    duplicate/high-force flags, model predictions) is removed. Logs one
    summary line saying where each label came from.
    """
    metadata_map = dict(metadata_map or {})
    unknown = set(metadata_map) - METADATA_MAP_KEYS
    if unknown:
        raise ValueError(
            f"Unknown metadata_map key(s) {sorted(unknown)}; allowed: "
            f"{sorted(METADATA_MAP_KEYS)}."
        )

    config_keys = (metadata_map.get("config_type"), *CONFIG_TYPE_ALIASES)
    energy_keys = (metadata_map.get("energy"), *ENERGY_ALIASES)
    forces_keys = (metadata_map.get("forces"), *FORCES_ALIASES)
    stress_keys = (metadata_map.get("stress"), *STRESS_ALIASES)

    sources: dict[str, Counter] = {
        label: Counter() for label in ("config_type", "energy", "forces", "stress")
    }
    missing_energy = missing_forces = 0
    normalized = []
    for original in atoms_list:
        atoms = original.copy()
        atoms.calc = original.calc
        strip_operational_metadata(atoms)

        if atoms.info.get("config_type"):
            sources["config_type"]["config_type"] += 1
        else:
            key, value = _first_info(atoms, config_keys)
            atoms.info["config_type"] = (
                str(value) if key is not None else EXTERNAL_CONFIG_TYPE
            )
            sources["config_type"][key or EXTERNAL_CONFIG_TYPE] += 1

        if atoms.info.get("REF_energy") is not None:
            sources["energy"]["REF_energy"] += 1
        else:
            key, value = _first_info(atoms, energy_keys)
            if key is None:
                key, value = "calculator", _calc_value(atoms, "get_potential_energy")
            if value is None:
                missing_energy += 1
            else:
                atoms.info["REF_energy"] = float(value)
                sources["energy"][key] += 1

        if "REF_forces" in atoms.arrays:
            sources["forces"]["REF_forces"] += 1
        else:
            key, value = _first_array(atoms, forces_keys)
            if key is None:
                key, value = "calculator", _calc_value(atoms, "get_forces")
            if value is None:
                missing_forces += 1
            else:
                atoms.arrays["REF_forces"] = np.asarray(value, dtype=float)
                sources["forces"][key] += 1

        if atoms.info.get("REF_stresses") is None:
            key, value = _first_info(atoms, stress_keys)
            if key is None:
                key, value = "calculator", _calc_value(atoms, "get_stress")
            if value is not None:
                atoms.info["REF_stresses"] = np.asarray(value, dtype=float)
                sources["stress"][key] += 1
        else:
            sources["stress"]["REF_stresses"] += 1

        # Labels now live in info/arrays; a stale calculator would otherwise
        # be preferred by later get_potential_energy() calls.
        atoms.calc = None
        normalized.append(atoms)

    where = f" in {source}" if source else ""
    if missing_energy or missing_forces:
        raise ValueError(
            f"Could not find DFT labels for {missing_energy} structure(s) "
            f"(energy) / {missing_forces} structure(s) (forces){where}. Tried "
            f"energy keys REF_energy, {', '.join(k for k in energy_keys if k)}, "
            f"calculator; force keys REF_forces, "
            f"{', '.join(k for k in forces_keys if k)}, calculator. Name the "
            "right keys with general.start_from.metadata_map (e.g. "
            "{energy: my_energy_key, forces: my_forces_key})."
        )

    logger.info(
        "Imported %d structure(s)%s; label sources: %s",
        len(normalized),
        where,
        "; ".join(
            f"{label} <- {dict(counts)}" for label, counts in sources.items() if counts
        ),
    )
    n_external = sources["config_type"][EXTERNAL_CONFIG_TYPE]
    if n_external:
        logger.warning(
            "%d structure(s)%s had no config_type (or equivalent key) and were "
            "labelled %r.",
            n_external,
            where,
            EXTERNAL_CONFIG_TYPE,
        )
    return normalized
