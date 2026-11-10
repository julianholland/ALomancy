import logging
import math
import re

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

# Per-fit trainer predictions the DB stores on a container
# (GlobalDatabase.store_model_predictions). They describe that container's
# geometry only, so a structure derived from it (an MD frame started from a
# DB seed) must not inherit them.
MODEL_PREDICTION_KEY = re.compile(r"^model_(energy|forces|stress)_loop_\d+_fit_\d+$")


def drop_unwritable_info(atoms: Atoms) -> Atoms:
    """Remove ``atoms.info`` entries extxyz can't round-trip -- empty
    lists, tuples, arrays and strings -- in place, and return *atoms*.

    ASE writes ``[]`` as ``key="_JSON []"`` but reads that back as an empty
    numpy array, which it then writes as a bare ``key=``. Reading a bare
    ``key=`` takes the next ``key=value`` in the header as the value, so
    whichever key follows is silently lost. That is how
    ``quality_filter_reasons=[]`` swallowed ``model_energy`` in every
    ``test_pred.xyz`` (its second write) and so emptied the test parity
    plots.
    """
    for key in [k for k, v in atoms.info.items() if _is_empty_value(v)]:
        del atoms.info[key]
    return atoms


def _is_empty_value(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, list | tuple | str) and len(value) == 0


def recover_swallowed_model_energy(atoms: Atoms) -> bool:
    """Undo the empty-value bug in prediction files written before
    drop_unwritable_info existed (ALomancy < 1.0.2): a bare
    ``quality_filter_reasons=`` swallowed the following ``model_energy=<x>``,
    which ASE then read as ``info["quality_filter_reasons"] ==
    "model_energy=<x>"``. Moves the value back into ``info["model_energy"]``
    (dropping the damaged key) and returns True if it did; the value is
    exact, so nothing needs re-evaluating."""
    if "model_energy" in atoms.info:
        return False
    for key, value in list(atoms.info.items()):
        if isinstance(value, str) and value.startswith("model_energy="):
            try:
                atoms.info["model_energy"] = float(value.split("=", 1)[1])
            except ValueError:
                return False
            del atoms.info[key]
            return True
    return False


def _computed_stress(structure: Atoms, label_source: str) -> np.ndarray | None:
    """The stress label for *structure*: from the calculator's results only
    (never ``get_stress()``, which on a file-based calculator such as QE's
    reruns the whole calculation when stress wasn't computed), or, for
    ``label_source="reference"``, an existing ``REF_stresses`` first. With
    ``"calculator"`` an ``info`` value is ignored: it may be stale, copied
    from the structure an MD frame started from."""
    if label_source == "reference" and "REF_stresses" in structure.info:
        return np.asarray(structure.info["REF_stresses"], dtype=float)
    results = getattr(structure.calc, "results", None) or {}
    stress = results.get("stress")
    if stress is None:
        return None
    stress = np.asarray(stress, dtype=float)
    return stress if np.isfinite(stress).all() else None


def clean_structures(
    structures: list[Atoms],
    config_type: str,
    override_config_type: bool = False,
    already_computed: bool = True,
    extra_metadata: dict | None = None,
    label_source: str = "reference",
) -> list[Atoms]:
    """
    adds DFT results to copy of structures info dictionary.
    """
    if label_source not in {"reference", "calculator"}:
        raise ValueError("label_source must be reference or calculator")
    cleaned_structures = []
    n_no_stress = 0
    for structure in structures:
        # copy structure with just the right information
        structure_copy = Atoms(
            symbols=structure.get_chemical_symbols(),
            positions=structure.get_positions(),
            cell=structure.get_cell(),
            pbc=structure.get_pbc(),
        )
        # Start from the original info, minus per-fit model predictions,
        # which belong to the geometry they were computed on (an MD seed's
        # are not this frame's).
        structure_copy.info = {
            key: value
            for key, value in structure.info.items()
            if not MODEL_PREDICTION_KEY.match(key)
        }

        structure_copy.info.update(
            extra_metadata or {}
        )  # add any extra metadata if provided

        if already_computed:
            if (
                label_source == "calculator"
                or "REF_energy" not in structure.info
                or "REF_forces" not in structure.arrays
            ):
                try:
                    energy = structure.get_potential_energy()
                    forces = structure.get_forces(apply_constraint=False)
                except Exception as e:
                    raise ValueError(
                        "Structure is marked as already_computed but is missing REF_energy or REF_forces, and they could not be computed. Original error: "
                        + str(e)
                    ) from e
            else:
                energy = structure.info["REF_energy"]
                forces = structure.arrays["REF_forces"]

            energy = float(energy)
            forces = np.asarray(forces, dtype=float)
            if not len(structure) or not np.isfinite(energy):
                raise ValueError("DFT energy must be finite and structure nonempty")
            if forces.shape != (len(structure), 3) or not np.isfinite(forces).all():
                raise ValueError("DFT forces must be finite with shape (N, 3)")
            structure_copy.info["REF_energy"] = energy
            structure_copy.arrays["REF_forces"] = forces

            stress = _computed_stress(structure, label_source)
            if stress is not None:
                structure_copy.info["REF_stresses"] = stress
            else:
                # Stress is not always physically meaningful (e.g.
                # non-periodic structures) or available (e.g. a calculator
                # that didn't compute it) -- unlike REF_energy/REF_forces
                # this is never fatal.
                structure_copy.info.pop("REF_stresses", None)
                n_no_stress += 1

        if override_config_type or "config_type" not in structure.info:
            logger.debug("Setting config_type to '%s'.", config_type)
            structure_copy.info["config_type"] = config_type

        cleaned_structures.append(structure_copy)

    if n_no_stress:
        logger.info(
            "No stress available for %d/%d structure(s) (non-periodic, or "
            "calculator did not report it).",
            n_no_stress,
            len(structures),
        )

    return cleaned_structures


def filter_structures_by_min_bond_distance(
    structures: list[Atoms], min_distance: float = 0.5
) -> list[Atoms]:
    """Exclude structures with any pairwise atomic distance below min_distance (Å).

    Guards against submitting an unphysical/exploded structure (e.g. an MD
    instability, or a badly-generated dimer/trimer/stretch-compress
    structure) to expensive DFT. `get_all_distances(mic=True)` is safe to
    call unconditionally: it uses the minimum-image convention for
    periodic structures and falls back to plain distances for non-periodic
    ones, correctly covering every config_type this codebase produces.
    Single-atom structures always pass through unfiltered — there is no
    pairwise distance to check, and `np.triu_indices(1, k=1)` correctly
    returns an empty index set rather than raising.
    """
    filtered = []
    n_excluded = 0
    for structure in structures:
        if len(structure) < 2:
            filtered.append(structure)
            continue
        distance_matrix = structure.get_all_distances(mic=True)
        upper = distance_matrix[np.triu_indices(len(structure), k=1)]
        if upper.min() >= min_distance:
            filtered.append(structure)
        else:
            n_excluded += 1

    if n_excluded:
        logger.warning(
            "Excluded %d/%d structure(s) with a bond shorter than %.2f Å "
            "from DFT submission.",
            n_excluded,
            len(structures),
            min_distance,
            extra={
                "event": "short_bond_excluded",
                "data": {"n": n_excluded, "total": len(structures)},
            },
        )

    return filtered


# Default for high_accuracy_evaluation.max_vdw_volume_ratio. Healthy
# condensed cells sit near 1; molecular crystals (P4, Li2S8) reach ~10; the
# blown-up NPT cells that ran DFT out of memory were 30 and far above.
DEFAULT_MAX_VDW_VOLUME_RATIO = 10.0
# Single atoms, dimers and trimers sit in a box of vacuum by design (and
# come back from DFT marked periodic), so they are never "blown up".
GAS_PHASE_CONFIG_TYPES = frozenset({"IsolatedAtom", "init_dimer", "init_trimer"})


def _vdw_volume(numbers: np.ndarray) -> float:
    """Summed van der Waals sphere volume of atoms with these atomic
    numbers (ase.data.vdw_radii; 1.5 x the covalent radius where ASE has no
    vdW radius)."""
    from ase.data import covalent_radii, vdw_radii

    numbers = np.asarray(numbers, dtype=int)
    radii = np.array([vdw_radii[z] for z in numbers], dtype=float)
    missing = ~np.isfinite(radii)
    radii[missing] = 1.5 * covalent_radii[numbers[missing]]
    return float(np.sum(4.0 / 3.0 * np.pi * radii**3))


def vdw_volume_ratio(atoms: Atoms) -> float:
    """Cell volume over the summed van der Waals sphere volumes of its
    atoms (_vdw_volume). Non-finite for a degenerate or non-finite cell."""
    with np.errstate(all="ignore"):
        volume = abs(float(np.linalg.det(atoms.cell.array)))
        return volume / _vdw_volume(atoms.numbers)


# 1 amu/Å^3 = 1.66054 g/cm^3, so a mass M (amu) at rho g/cm^3 fills
# M / (rho * 0.6022) Å^3.
_G_CM3_PER_AMU_A3 = 1.66054


def vdw_volume_ratio_at_density(symbols: list[str], density_g_cm3: float) -> float:
    """vdw_volume_ratio a cell of *symbols* would have at a mass density of
    *density_g_cm3*: what filter_blown_up_structures will measure, known
    before the cell exists."""
    from ase.data import atomic_masses, atomic_numbers

    numbers = np.array([atomic_numbers[s] for s in symbols])
    mass = float(np.sum(atomic_masses[numbers]))
    volume = mass * _G_CM3_PER_AMU_A3 / density_g_cm3
    return volume / _vdw_volume(numbers)


def density_for_vdw_volume_ratio(symbols: list[str], ratio: float) -> float:
    """The mass density (g/cm^3) at which a cell of *symbols* has
    vdw_volume_ratio *ratio* (the inverse of vdw_volume_ratio_at_density)."""
    return vdw_volume_ratio_at_density(symbols, 1.0) / ratio


def is_blown_up(
    structure: Atoms,
    max_ratio: float | None,
    exempt_config_types: frozenset[str] = frozenset(),
) -> bool:
    """Whether filter_blown_up_structures would exclude *structure*: a
    periodic cell above *max_ratio* x its atoms' vdW volume, or with a
    non-finite ratio (see there for the exemptions)."""
    if max_ratio is None:
        return False
    checked = (
        len(structure) > 0
        and structure.pbc.all()
        and structure.info.get("config_type")
        not in GAS_PHASE_CONFIG_TYPES | exempt_config_types
    )
    ratio = vdw_volume_ratio(structure) if checked else 0.0
    return not (np.isfinite(ratio) and ratio <= max_ratio)


def filter_blown_up_structures(
    structures: list[Atoms],
    max_ratio: float | None = DEFAULT_MAX_VDW_VOLUME_RATIO,
    exempt_config_types: frozenset[str] = frozenset(),
) -> list[Atoms]:
    """Exclude periodic cells whose volume is more than *max_ratio* x their
    atoms' van der Waals volume (vdw_volume_ratio), or non-finite: DFT on
    such cells runs out of memory, times out or fails outright. None turns
    the check off. Non-periodic structures, GAS_PHASE_CONFIG_TYPES and
    *exempt_config_types* (e.g. molecules in vacuum) are never excluded."""
    if max_ratio is None:
        return list(structures)
    exempt = GAS_PHASE_CONFIG_TYPES | exempt_config_types
    kept, excluded = [], []
    for structure in structures:
        checked = (
            len(structure) > 0
            and structure.pbc.all()
            and structure.info.get("config_type") not in exempt
        )
        ratio = vdw_volume_ratio(structure) if checked else 0.0
        if np.isfinite(ratio) and ratio <= max_ratio:
            kept.append(structure)
        else:
            excluded.append((structure, ratio))
    if excluded:
        ratios = [r for _, r in excluded]
        largest = max((r for r in ratios if np.isfinite(r)), default=float("inf"))
        config_types = sorted(
            {str(s.info.get("config_type", "unknown")) for s, _ in excluded}
        )
        logger.warning(
            "Excluded %d/%d structure(s) from DFT whose volume is above %g x "
            "their atoms' van der Waals volume (largest %.3g x; config_type %s): "
            "DFT on such cells runs out of memory or fails. If they came from "
            "NPT MD, add a small positive structure_generation.md_kwargs."
            "pressure (e.g. 0.1 GPa); if they are genuine low-density crystals, "
            "raise high_accuracy_evaluation.max_vdw_volume_ratio; if they are "
            "meant to have vacuum (e.g. molecules in a box), list their "
            "config_type in high_accuracy_evaluation.vdw_exempt_config_types.",
            len(excluded),
            len(structures),
            max_ratio,
            largest,
            ", ".join(config_types),
            extra={
                "event": "blown_up_excluded",
                "data": {
                    "n": len(excluded),
                    "total": len(structures),
                    "max_ratio": max_ratio,
                    "config_types": config_types,
                },
            },
        )
    return kept


def filter_structures_by_atom_count(
    structures: list[Atoms], max_atoms: int | None, evaluator: str = ""
) -> list[Atoms]:
    """Exclude structures with more than *max_atoms* atoms (None: keep
    all), the evaluator's size cap (registry entry point max_num_of_atoms,
    e.g. ORCA with a correlated method). Warning too_many_atoms_for_method."""
    if max_atoms is None:
        return list(structures)
    kept = [s for s in structures if len(s) <= max_atoms]
    excluded = [s for s in structures if len(s) > max_atoms]
    if excluded:
        config_types = sorted(
            {str(s.info.get("config_type", "unknown")) for s in excluded}
        )
        logger.warning(
            "Excluded %d/%d structure(s) from %s evaluation with more than %d "
            "atoms (largest %d; config_type %s): the configured method's cost "
            "grows steeply with size. Raise high_accuracy_evaluation."
            "<evaluator>_kwargs.max_num_of_atoms_correlated to evaluate them "
            "anyway, or generate smaller structures.",
            len(excluded),
            len(structures),
            evaluator or "high-accuracy",
            max_atoms,
            max(len(s) for s in excluded),
            ", ".join(config_types),
            extra={
                "event": "too_many_atoms_for_method",
                "data": {
                    "n": len(excluded),
                    "total": len(structures),
                    "max_atoms": max_atoms,
                    "config_types": config_types,
                },
            },
        )
    return kept


def resolve_ratio(value: object, key: str) -> float | None:
    """A config ratio limit: a finite number above 1, or None (check off)."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 1
    ):
        raise ValueError(
            f"{key} must be a finite number above 1, or null to turn it off; "
            f"got {value!r}."
        )
    return float(value)


def resolve_max_vdw_volume_ratio(hae_config: dict | None) -> float | None:
    """high_accuracy_evaluation.max_vdw_volume_ratio (default
    DEFAULT_MAX_VDW_VOLUME_RATIO), validated by resolve_ratio."""
    return resolve_ratio(
        (hae_config or {}).get("max_vdw_volume_ratio", DEFAULT_MAX_VDW_VOLUME_RATIO),
        "high_accuracy_evaluation.max_vdw_volume_ratio",
    )


def resolve_vdw_exempt_config_types(hae_config: dict | None) -> frozenset[str]:
    """high_accuracy_evaluation.vdw_exempt_config_types: config types the
    blown-up check skips, on top of GAS_PHASE_CONFIG_TYPES."""
    value = (hae_config or {}).get("vdw_exempt_config_types") or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(
            "high_accuracy_evaluation.vdw_exempt_config_types must be a list "
            f"of config_type names; got {value!r}."
        )
    return frozenset(value)


def wrap_structures_into_cell(structures: list[Atoms]) -> list[Atoms]:
    """Wrap each structure's atoms back inside its periodic cell, in place.

    MD trajectories routinely let atoms drift outside the reference cell
    (a normal artifact of unwrapped-coordinate propagation, distinct from
    the unphysical short-bond-distance case handled by
    filter_structures_by_min_bond_distance) — DFT codes (QE/VASP) generally
    expect coordinates within the cell, so this runs as the last step
    before DFT submission, on structures that have already survived that
    bond-distance filter.

    `Atoms.wrap()` mutates the structure in place and returns None; it is
    called unconditionally here rather than gated on `atoms.pbc.any()`
    because it is a safe no-op on non-periodic axes/structures (confirmed
    empirically against ASE 3.28: positions on a non-periodic axis, and on
    a structure with `pbc=False` and no cell at all, are left unchanged).
    `atoms.info`, `atoms.arrays` (including any non-standard extra arrays),
    and any already-attached `atoms.calc` all survive `wrap()` unchanged —
    confirmed empirically, since this is the property this function must
    not violate.
    """
    for structure in structures:
        structure.wrap()
    return structures
