"""Initialiser module: wraps create_initialization_atoms_list/
compute_initialization_needs for the modular AL architecture.

Produces unlabeled structures only -- no DFT evaluation, no DB writes, no
train/test split. It does not depend on the evaluator category (see the
architecture plan's initialiser decision): the skeleton is what calls the
DFT evaluator on this module's output afterward, exactly as it already
does for AL-loop-generated structures. This also means importing
general.start_from's data, the needs computation against the DB,
clean_structures, db.add_structures,
and the initial train/test split all live in the skeleton's own
_initialize_training_set (active_learning_workflow.py), not here.

Self-contained config surface: reads only initialiser-specific settings
directly off the `initialization` section (plus `read_generated_file`,
read directly by the skeleton, not by this
module), never reaching into another module's section. Three exceptions,
all passed as explicit keyword arguments by the skeleton rather than read
from `config` (matching how `name`/`hpc`/`max_time` already are):
`elements` (general.elements, a list of atomic symbols e.g. ["C", "O"]);
`target_config_types` (general.committee_uncertainty_kwargs.
target_config_types -- which of this call's freshly-generated structures
count as "target" structures for stretch_compress_targets_kwargs/
rattle_target_structures below, reusing the same list already used
elsewhere to pick out the scientifically-significant structures from
purely-auxiliary ones like dimers/trimers); and `seed` (general.seed,
defaulting to 803 when absent -- currently only used to seed
rattle_target_structures).

Bootstrap generation is structurally a different problem from the AL-loop
structure_generator abstraction (no trained model, no uncertainty-based
selection) -- generate() here does not take seed_atoms/model_path.

Unlike training/structure_generation/high_accuracy_evaluation (each a
choice between interchangeable backends, dispatched via trainer/generator/
evaluator), initialisation is structurally a single method that always
runs every structure-generating sub-task it's configured for, to differing
degrees depending on settings -- there's no dispatch key here. So its
settings are namespaced per *structure type* rather than per backend,
directly under `initialization` (no `creation_kwargs` wrapper -- nothing
else in this module's config surface needed one, so it was just an extra
level of nesting): `isolated_atom_kwargs`, `dimer_kwargs`, `trimer_kwargs`,
`amorphous_kwargs`, `mp_kwargs`, `stretch_compress_targets_kwargs` --
each holding only the settings that sub-task actually reads, mirroring
the <method>_kwargs convention used elsewhere, and each with its own
`enabled` flag (default `True`) to toggle that sub-task on/off without
having to zero out its count field. This is deliberately designed to
extend cleanly: a future sub-task (surfaces, rattled structures,
interfaces) adds its own `surface_kwargs`/`rattle_kwargs`/
`interface_kwargs` sibling here and a matching branch in
create_initialization_atoms_list, without touching any other namespace.

`stretch_compress_targets_kwargs` (`max_lattice_deformation`,
`num_of_stretch_compress_per_target`) and `rattle_target_structures`
(`rattle_standard_deviation`, `num_of_rattled_per_target`) are each a
sibling of `mp_kwargs`, not nested inside it -- both apply to *every*
freshly-generated structure this call produces whose config_type is
listed in `target_config_types` (see above), not just Materials Project
ones, so e.g. setting `target_config_types: ["init_MP", "init_amorphous"]`
makes amorphous structures subject to both too. `rattle_target_structures
.enabled` defaults to `False` (unlike every other namespace here) since
it's a new capability with no prior default behavior to preserve, and
`rattle_standard_deviation` has no default at all -- a silently guessed
perturbation magnitude is worse than a clear KeyError once rattle is
turned on, matching how `test_ratio`/`target_config_types` are handled
committee-workflow-side.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase import Atoms
from ase.io import read

from alomancy.initialize.initialization_structure_list import (
    compute_initialization_needs,
    create_initialization_atoms_list,
)

if TYPE_CHECKING:
    from alomancy.database.global_database import GlobalDatabase

logger = logging.getLogger(__name__)

# Named, reusable defaults, one entry per structure-type namespace --
# merged with user overrides in _resolve_kwargs below, and also used by
# the skeleton's pre-run config summary (active_learning_workflow.
# py's display_workflow_summary) to show the fully-resolved effective
# config, not just what the user wrote. Mirrors create_initialization_
# atoms_list's (old, shared) own parameter defaults.
_CREATION_KWARGS_DEFAULTS: dict[str, dict[str, Any]] = {
    "isolated_atom_kwargs": {"enabled": True},
    "dimer_kwargs": {"enabled": True, "num_of_dimers_per_combo": 10},
    "trimer_kwargs": {"enabled": True, "num_of_trimers_per_combo": 5},
    "amorphous_kwargs": {
        "enabled": True,
        "num_of_amorphous_structures": 100,
        "num_of_atoms_per_amorphous": 20,
        "density_range": [1.0, 1.0],
        "composition_list": None,
    },
    "mp_kwargs": {
        "enabled": True,
        "max_num_of_atoms": 20,
        "mp_max_energy_above_hull": 0.1,
    },
    "stretch_compress_targets_kwargs": {
        "enabled": True,
        "num_of_stretch_compress_per_target": 5,
        "max_lattice_deformation": 0.2,
    },
    "rattle_target_structures": {
        "enabled": False,
        "num_of_rattled_per_target": 5,
    },
    # Off by default: enabling needs the molecules named (no default list).
    "molecule_kwargs": {
        "enabled": False,
        "molecules": None,
        "vacuum": 10.0,
    },
    "molecules_in_box_kwargs": {
        "enabled": False,
        "box_compositions": None,
        "min_distance": 1.5,
        "max_attempts_per_molecule": 1000,
    },
}

# The keys of one molecules_in_box_kwargs.box_compositions entry.
_BOX_COMPOSITION_KEYS = {
    "molecules",
    "density_range",
    "num_of_molecules",
    "num_of_structures",
}


# The keys each structure-type block may hold (configs/schema.py): its
# defaults, plus settings that deliberately have none.
_CREATION_KWARGS_SCHEMA: dict[str, list[str]] = {
    namespace: list(defaults)
    for namespace, defaults in _CREATION_KWARGS_DEFAULTS.items()
}
_CREATION_KWARGS_SCHEMA["rattle_target_structures"].append("rattle_standard_deviation")
# Deprecated (docs/deprecations.md): read as density_range [min, max].
_CREATION_KWARGS_SCHEMA["amorphous_kwargs"].append("densities_list")
# Deprecated (docs/deprecations.md): ignored, general.seed seeds the cells.
_CREATION_KWARGS_SCHEMA["amorphous_kwargs"].append("seed")


def _resolve_kwargs(config: dict, namespace: str) -> dict:
    """config[namespace] (a direct child of the initialization section --
    see module docstring), with _CREATION_KWARGS_DEFAULTS[namespace] filled
    in underneath for any key the user didn't override. The deprecated
    amorphous_kwargs.densities_list is read as density_range [min, max]
    (validate_kwargs warns about it)."""
    resolved = {
        **_CREATION_KWARGS_DEFAULTS[namespace],
        **config.get(namespace, {}),
    }
    user = config.get(namespace) or {}
    if (
        namespace == "amorphous_kwargs"
        and user.get("densities_list")
        and "density_range" not in user
    ):
        densities = [float(d) for d in user["densities_list"]]
        resolved["density_range"] = [min(densities), max(densities)]
    resolved.pop("densities_list", None)
    if namespace == "amorphous_kwargs":
        resolved.pop("seed", None)  # deprecated: general.seed is used
    return resolved


def _is_density_range(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(
            isinstance(v, int | float) and not isinstance(v, bool) and v > 0
            for v in value
        )
        and value[0] <= value[1]
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_kwargs(
    config: dict,
    *,
    elements: list[str],
    max_vdw_volume_ratio: float | None = None,
    vdw_exempt_config_types: frozenset[str] = frozenset(),
) -> None:
    """Construction-time check (registry hook) of the initialization
    blocks that would otherwise fail only once initialisation runs: the
    molecule blocks (named explicitly when enabled, every molecule built
    by SAGE from general.elements), and every density_range. Warns about
    the deprecated amorphous_kwargs.densities_list and amorphous_kwargs.seed,
    and about densities
    the blown-up filter would drop or that pack atoms tighter than any
    solid (_warn_unusual_densities; *max_vdw_volume_ratio* and
    *vdw_exempt_config_types* are high_accuracy_evaluation's, None skips
    the low-density check)."""
    from alomancy.utils.molecules import validate_molecules

    amorphous = config.get("amorphous_kwargs") or {}
    if "densities_list" in amorphous:
        logger.warning(
            "initialization.amorphous_kwargs.densities_list is deprecated: "
            "use density_range [min, max] (structures evenly spaced over it). "
            "Reading %s as density_range %s.",
            amorphous["densities_list"],
            _resolve_kwargs(config, "amorphous_kwargs")["density_range"],
            extra={
                "event": "config_key_deprecated",
                "data": {"key": "initialization.amorphous_kwargs.densities_list"},
            },
        )
    if "seed" in amorphous:
        logger.warning(
            "initialization.amorphous_kwargs.seed is deprecated and ignored: "
            "the amorphous cells are seeded by general.seed, like every other "
            "random choice. Remove it (set general.seed to change the cells).",
            extra={
                "event": "config_key_deprecated",
                "data": {"key": "initialization.amorphous_kwargs.seed"},
            },
        )
    density_range = _resolve_kwargs(config, "amorphous_kwargs")["density_range"]
    if not _is_density_range(density_range):
        raise ValueError(
            "initialization.amorphous_kwargs.density_range must be [min, max] "
            f"g/cm^3 with 0 < min <= max, got {density_range!r}."
        )

    molecule_kwargs = _resolve_kwargs(config, "molecule_kwargs")
    if molecule_kwargs["enabled"]:
        where = "initialization.molecule_kwargs.molecules"
        if not molecule_kwargs["molecules"]:
            raise ValueError(
                f"{where} is required when molecule_kwargs is enabled, e.g. "
                "['CO', 'smiles-CCO']."
            )
        validate_molecules(molecule_kwargs["molecules"], elements, where)

    box_kwargs = _resolve_kwargs(config, "molecules_in_box_kwargs")
    if box_kwargs["enabled"]:
        _validate_box_compositions(box_kwargs, elements)
    _warn_unusual_densities(
        config, elements, max_vdw_volume_ratio, vdw_exempt_config_types
    )


def _validate_box_compositions(box_kwargs: dict, elements: list[str]) -> None:
    from alomancy.utils.molecules import validate_molecules

    where = "initialization.molecules_in_box_kwargs.box_compositions"
    compositions = box_kwargs["box_compositions"]
    if not compositions or not isinstance(compositions, list):
        raise ValueError(
            f"{where} is required when molecules_in_box_kwargs is enabled: a "
            "list of {molecules: {name: ratio}, density_range: [min, max], "
            "num_of_molecules, num_of_structures}."
        )
    for i, composition in enumerate(compositions):
        at = f"{where}[{i}]"
        if not isinstance(composition, dict):
            raise ValueError(f"{at} must be a mapping, got {composition!r}.")
        unknown = set(composition) - _BOX_COMPOSITION_KEYS
        missing = _BOX_COMPOSITION_KEYS - set(composition)
        if unknown or missing:
            raise ValueError(
                f"{at}: unknown keys {sorted(unknown)}, missing keys "
                f"{sorted(missing)} (expected {sorted(_BOX_COMPOSITION_KEYS)})."
            )
        ratio = composition["molecules"]
        if (
            not isinstance(ratio, dict)
            or not ratio
            or not all(
                isinstance(v, int | float) and not isinstance(v, bool) and v > 0
                for v in ratio.values()
            )
        ):
            raise ValueError(
                f"{at}.molecules must map molecule names to positive ratios, "
                f"e.g. {{'H2O': 2, 'smiles-CO': 1}}; got {ratio!r}."
            )
        validate_molecules(list(ratio), elements, f"{at}.molecules")
        if not _is_density_range(composition["density_range"]):
            raise ValueError(
                f"{at}.density_range must be [min, max] g/cm^3 with "
                f"0 < min <= max, got {composition['density_range']!r}."
            )
        if not _positive_int(composition["num_of_molecules"]) or (
            composition["num_of_molecules"] < len(ratio)
        ):
            raise ValueError(
                f"{at}.num_of_molecules must be an integer of at least the "
                f"number of molecules ({len(ratio)}), got "
                f"{composition['num_of_molecules']!r}."
            )
        if not _positive_int(composition["num_of_structures"]):
            raise ValueError(
                f"{at}.num_of_structures must be a positive integer, got "
                f"{composition['num_of_structures']!r}."
            )
    for key in ("min_distance", "max_attempts_per_molecule"):
        value = box_kwargs[key]
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"initialization.molecules_in_box_kwargs.{key} must be positive, "
                f"got {value!r}."
            )


# Below this vdW volume ratio atoms are packed tighter than in any solid
# (diamond is 0.28, close-packed metals 0.6-0.7): the cell is a mistake.
_MIN_PLAUSIBLE_VDW_VOLUME_RATIO = 0.3


def _density_compositions(
    config: dict, elements: list[str]
) -> list[tuple[str, str, list[str], list[float]]]:
    """(block, label, symbols, density_range) for every composition an
    enabled amorphous or molecules-in-box block makes. Amorphous cells
    without a composition_list are random mixes of general.elements,
    checked as their typical cell, an even mix (one of each element):
    checking each pure element instead would flag every hydrogen config,
    since a pure-H cell at any solid's density is far too dense."""
    from alomancy.initialize.molecules import box_composition_label, molecule_counts
    from alomancy.utils.molecules import molecule_symbols

    found = []
    amorphous = _resolve_kwargs(config, "amorphous_kwargs")
    if amorphous["enabled"]:
        density_range = list(amorphous["density_range"])
        if amorphous["composition_list"]:
            for cell in amorphous["composition_list"]:
                symbols = [str(s) for s in cell]
                try:
                    label = Atoms(symbols).get_chemical_formula()
                except (KeyError, ValueError):
                    continue  # a bad composition_list fails at generation
                found.append(("amorphous_kwargs", label, symbols, density_range))
        elif elements:
            label = (
                str(elements[0])
                if len(elements) == 1
                else f"random mixes of {', '.join(elements)} (as an even mix)"
            )
            found.append(("amorphous_kwargs", label, list(elements), density_range))
    boxes = _resolve_kwargs(config, "molecules_in_box_kwargs")
    if boxes["enabled"]:
        for composition in boxes["box_compositions"]:
            counts = molecule_counts(
                composition["molecules"], composition["num_of_molecules"]
            )
            symbols = [
                symbol
                for name, count in counts.items()
                for symbol, n in molecule_symbols(name).items()
                for _ in range(n * count)
            ]
            found.append(
                (
                    "molecules_in_box_kwargs",
                    box_composition_label(composition["molecules"]),
                    symbols,
                    list(composition["density_range"]),
                )
            )
    return found


def _warn_unusual_densities(
    config: dict,
    elements: list[str],
    max_vdw_volume_ratio: float | None,
    vdw_exempt_config_types: frozenset[str],
) -> None:
    """Warn, per block, about compositions whose density_range (g/cm^3)
    gives cells the blown-up filter would drop before DFT (vdW volume
    ratio above *max_vdw_volume_ratio*, e.g. Pt at the default 1 g/cm^3:
    14x), or packs atoms tighter than any solid (below
    _MIN_PLAUSIBLE_VDW_VOLUME_RATIO). Each names the g/cm^3 window
    between the two for that composition."""
    from alomancy.utils.clean_structures import (
        density_for_vdw_volume_ratio,
        vdw_volume_ratio_at_density,
    )

    config_types = {
        "amorphous_kwargs": "init_amorphous",
        "molecules_in_box_kwargs": "init_molecules_in_box",
    }
    upper_ratio = max_vdw_volume_ratio
    low: dict[str, list[str]] = {}
    high: dict[str, list[str]] = {}
    for block, label, symbols, (lo, hi) in _density_compositions(config, elements):
        sparsest = vdw_volume_ratio_at_density(symbols, lo)
        densest = vdw_volume_ratio_at_density(symbols, hi)
        window_top = density_for_vdw_volume_ratio(
            symbols, _MIN_PLAUSIBLE_VDW_VOLUME_RATIO
        )
        checked_low = (
            upper_ratio is not None
            and config_types[block] not in vdw_exempt_config_types
        )
        if checked_low and sparsest > upper_ratio:
            window_bottom = density_for_vdw_volume_ratio(symbols, upper_ratio)
            low.setdefault(block, []).append(
                f"{label} at {lo:g} g/cm^3 ({sparsest:.3g}x its vdW volume; keep "
                f"it between {window_bottom:.3g} and {window_top:.3g} g/cm^3)"
            )
        if densest < _MIN_PLAUSIBLE_VDW_VOLUME_RATIO:
            high.setdefault(block, []).append(
                f"{label} at {hi:g} g/cm^3 ({densest:.3g}x its vdW volume; keep "
                f"it below {window_top:.3g} g/cm^3)"
            )
    for block, items in low.items():
        logger.warning(
            "initialization.%s.density_range is too low for %s: those cells "
            "are above high_accuracy_evaluation.max_vdw_volume_ratio (%g) and "
            "will be dropped before DFT as blown up. Raise the density (bulk "
            "solids sit near 0.3-1x their vdW volume), or list %s in "
            "high_accuracy_evaluation.vdw_exempt_config_types if they are "
            "meant to be that sparse.",
            block,
            "; ".join(items),
            upper_ratio,
            config_types[block],
            extra={
                "event": "init_density_too_low",
                "data": {"block": block, "compositions": items},
            },
        )
    for block, items in high.items():
        logger.warning(
            "initialization.%s.density_range is too high for %s: the atoms "
            "would be packed tighter than in any solid (diamond is 0.28x its "
            "vdW volume), so the cells are unphysical.",
            block,
            "; ".join(items),
            extra={
                "event": "init_density_too_high",
                "data": {"block": block, "compositions": items},
            },
        )


def compute_needs(
    db: "GlobalDatabase",
    config: dict,
    elements: list[str],
) -> dict:
    """Compare DB contents against initialization targets (config's
    structure-type namespaces, plus the shared workflow.elements list) and
    return what still needs to be generated -- see
    compute_initialization_needs for the returned dict's shape.
    """
    isolated_atom_kwargs = _resolve_kwargs(config, "isolated_atom_kwargs")
    dimer_kwargs = _resolve_kwargs(config, "dimer_kwargs")
    trimer_kwargs = _resolve_kwargs(config, "trimer_kwargs")
    amorphous_kwargs = _resolve_kwargs(config, "amorphous_kwargs")
    mp_kwargs = _resolve_kwargs(config, "mp_kwargs")
    molecule_kwargs = _resolve_kwargs(config, "molecule_kwargs")
    box_kwargs = _resolve_kwargs(config, "molecules_in_box_kwargs")
    return compute_initialization_needs(
        db=db,
        elements=elements,
        molecules=molecule_kwargs["molecules"] if molecule_kwargs["enabled"] else None,
        box_compositions=(
            box_kwargs["box_compositions"] if box_kwargs["enabled"] else None
        ),
        _single_atoms=isolated_atom_kwargs["enabled"],
        mp_structures=mp_kwargs["enabled"],
        num_of_dimers_per_combo=(
            dimer_kwargs["num_of_dimers_per_combo"] if dimer_kwargs["enabled"] else 0
        ),
        num_of_trimers_per_combo=(
            trimer_kwargs["num_of_trimers_per_combo"] if trimer_kwargs["enabled"] else 0
        ),
        num_of_amorphous_structures=(
            amorphous_kwargs["num_of_amorphous_structures"]
            if amorphous_kwargs["enabled"]
            else 0
        ),
    )


def _generated_structures_path(base_name: str) -> Path:
    return Path("results", base_name, "initialization_structures_generated.xyz")


def output_paths(config: dict, *, base_name: str, name: str) -> list[Path]:  # noqa: ARG001
    """Restart check: the file create_initialization_atoms_list already
    writes unconditionally."""
    return [_generated_structures_path(base_name)]


def read_existing_result(config: dict, *, base_name: str, name: str) -> list[Atoms]:  # noqa: ARG001
    path = _generated_structures_path(base_name)
    if not path.exists():
        raise ValueError(f"No cached initialiser result at {path}.")
    return list(read(path, ":", format="extxyz"))


def generate(
    config: dict,
    *,
    base_name: str,
    name: str,  # noqa: ARG001 -- unused (no remote submission); uniform across module categories
    elements: list[str],
    hpc: dict | None = None,  # noqa: ARG001 -- unused (no remote submission); uniform across module categories
    max_time: str | None = None,  # noqa: ARG001 -- unused (no remote submission); uniform across module categories
    needs: dict | None = None,
    target_config_types: list[str] | None = None,
    seed: int = 803,
) -> list[Atoms]:
    """Generate bootstrap structures (dimers/trimers/amorphous/MP/isolated
    atoms), reading targets from config's structure-type namespaces plus
    the shared general.elements list.

    When `needs` is given (from compute_needs), only the missing subset
    (relative to what's already in the DB) is generated -- enabling
    idempotent restarts. When None, the full target set is generated.
    hpc/max_time are accepted (and optional, defaulting to None) for
    interface uniformity across module categories but unused: initialisation
    runs entirely in the local driver process (the Materials Project fetch
    is a plain network call, not an ExPyRe remote job).
    """
    work_dir = Path("results", base_name)
    work_dir.mkdir(exist_ok=True, parents=True)
    isolated_atom_kwargs = _resolve_kwargs(config, "isolated_atom_kwargs")
    dimer_kwargs = _resolve_kwargs(config, "dimer_kwargs")
    trimer_kwargs = _resolve_kwargs(config, "trimer_kwargs")
    amorphous_kwargs = _resolve_kwargs(config, "amorphous_kwargs")
    mp_kwargs = _resolve_kwargs(config, "mp_kwargs")
    stretch_compress_targets_kwargs = _resolve_kwargs(
        config, "stretch_compress_targets_kwargs"
    )
    rattle_kwargs = _resolve_kwargs(config, "rattle_target_structures")
    molecule_kwargs = _resolve_kwargs(config, "molecule_kwargs")
    box_kwargs = _resolve_kwargs(config, "molecules_in_box_kwargs")

    return create_initialization_atoms_list(
        work_dir=str(work_dir),
        elements=elements,
        mp_structures=(
            needs["mp_structures"] if needs is not None else mp_kwargs["enabled"]
        ),
        single_atoms=(
            bool(needs["isolated_atoms"])
            if needs is not None
            else isolated_atom_kwargs["enabled"]
        ),
        num_of_dimers_per_combo=(
            dimer_kwargs["num_of_dimers_per_combo"] if dimer_kwargs["enabled"] else 0
        ),
        num_of_trimers_per_combo=(
            trimer_kwargs["num_of_trimers_per_combo"] if trimer_kwargs["enabled"] else 0
        ),
        num_of_amorphous_structures=(
            amorphous_kwargs["num_of_amorphous_structures"]
            if amorphous_kwargs["enabled"]
            else 0
        ),
        num_of_stretch_compress_per_target=(
            stretch_compress_targets_kwargs["num_of_stretch_compress_per_target"]
            if stretch_compress_targets_kwargs["enabled"]
            else 0
        ),
        density_range=amorphous_kwargs["density_range"],
        max_lattice_deformation=stretch_compress_targets_kwargs[
            "max_lattice_deformation"
        ],
        max_num_of_atoms=mp_kwargs["max_num_of_atoms"],
        num_of_atoms_per_amorphous=amorphous_kwargs["num_of_atoms_per_amorphous"],
        mp_max_energy_above_hull=mp_kwargs["mp_max_energy_above_hull"],
        composition_list=amorphous_kwargs["composition_list"],
        seed=seed,
        target_config_types=target_config_types,
        num_of_rattled_per_target=(
            rattle_kwargs["num_of_rattled_per_target"]
            if rattle_kwargs["enabled"]
            else 0
        ),
        rattle_standard_deviation=(
            rattle_kwargs["rattle_standard_deviation"]
            if rattle_kwargs["enabled"]
            else None
        ),
        rattle_seed=seed,
        isolated_atoms_override=(
            (needs["isolated_atoms"] or None) if needs is not None else None
        ),
        # An empty/zero override means "none still needed", not "no
        # override": `or None` here regenerated every dimer, trimer and
        # amorphous cell whenever that kind was already complete.
        dimer_override=needs["dimer_override"] if needs is not None else None,
        trimer_override=needs["trimer_override"] if needs is not None else None,
        amorphous_override=needs["amorphous_override"] if needs is not None else None,
        molecules=molecule_kwargs["molecules"] if molecule_kwargs["enabled"] else None,
        molecule_vacuum=molecule_kwargs["vacuum"],
        box_compositions=(
            box_kwargs["box_compositions"] if box_kwargs["enabled"] else None
        ),
        box_min_distance=box_kwargs["min_distance"],
        box_max_attempts_per_molecule=box_kwargs["max_attempts_per_molecule"],
        molecule_override=needs.get("molecule_override") if needs is not None else None,
        box_override=needs.get("box_override") if needs is not None else None,
    )
