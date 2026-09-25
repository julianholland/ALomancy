"""Initialiser module: wraps create_initialization_atoms_list/
compute_initialization_needs for the modular AL architecture.

Produces unlabeled structures only -- no DFT evaluation, no DB writes, no
train/test split. It does not depend on the evaluator category (see the
architecture plan's initialiser decision): the skeleton is what calls the
DFT evaluator on this module's output afterward, exactly as it already
does for AL-loop-generated structures. This also means the on-disk
fast-path check (pre-existing initial_train/test files), the DB-first/
extra-datasets-second needs computation, clean_structures, db.add_structures,
and the initial train/test split all live in the skeleton's own
_initialize_training_set (committee_uncertainty_workflow.py), not here.

Self-contained config surface: reads only initialiser-specific settings
directly off the `initialization` section (plus `extra_datasets`/
`read_generated_file`, read directly by the skeleton, not by this
module), never reaching into another module's section. The one exception
is `elements` (workflow.elements, a list of atomic symbols e.g.
["C", "O"]), passed as an explicit keyword argument by the skeleton --
shared element identity, not initialiser-specific config, so it lives in
`workflow` rather than here (matching how `name`/`hpc`/`max_time` are
already explicit kwargs rather than config-dict reads).

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

`stretch_compress_targets_kwargs` (`deform_xyz`, `max_deformation`,
`num_stretch_compress_per_mp`) is a sibling of `mp_kwargs`, not nested
inside it -- but create_initialization_atoms_list (old, shared,
unchanged) only ever generates stretch/compress structures *from*
MP-fetched ones, so its own `enabled=True` still produces nothing
whenever `mp_kwargs.enabled` is `False` or the MP fetch itself returns no
structures. That coupling is a hard constraint of the shared function,
not something enforced or hidden here -- `stretch_compress_targets_kwargs
.enabled` only ever narrows what `mp_kwargs.enabled` already allows.
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
# the skeleton's pre-run config summary (committee_uncertainty_workflow.
# py's display_workflow_summary) to show the fully-resolved effective
# config, not just what the user wrote. Mirrors create_initialization_
# atoms_list's (old, shared) own parameter defaults.
_CREATION_KWARGS_DEFAULTS: dict[str, dict[str, Any]] = {
    "isolated_atom_kwargs": {"enabled": True},
    "dimer_kwargs": {"enabled": True, "num_dimers_per_combo": 10},
    "trimer_kwargs": {"enabled": True, "num_trimers_per_combo": 5},
    "amorphous_kwargs": {
        "enabled": True,
        "num_amorphous": 100,
        "amorphous_atom_number": 20,
        "densities_list": None,
        "composition_list": None,
        "seed": 803,
    },
    "mp_kwargs": {
        "enabled": True,
        "max_atom_number": 20,
        "mp_max_energy_above_hull": 0.1,
    },
    "stretch_compress_targets_kwargs": {
        "enabled": True,
        "num_stretch_compress_per_mp": 5,
        "deform_xyz": False,
        "max_deformation": 0.2,
    },
}


def _resolve_kwargs(config: dict, namespace: str) -> dict:
    """config[namespace] (a direct child of the initialization section --
    see module docstring), with _CREATION_KWARGS_DEFAULTS[namespace] filled
    in underneath for any key the user didn't override."""
    return {
        **_CREATION_KWARGS_DEFAULTS[namespace],
        **config.get(namespace, {}),
    }


def compute_needs(db: "GlobalDatabase", config: dict, elements: list[str]) -> dict:
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
    return compute_initialization_needs(
        db=db,
        elements=elements,
        _single_atoms=isolated_atom_kwargs["enabled"],
        mp_structures=mp_kwargs["enabled"],
        num_dimers_per_combo=(
            dimer_kwargs["num_dimers_per_combo"] if dimer_kwargs["enabled"] else 0
        ),
        num_trimers_per_combo=(
            trimer_kwargs["num_trimers_per_combo"] if trimer_kwargs["enabled"] else 0
        ),
        num_amorphous=(
            amorphous_kwargs["num_amorphous"] if amorphous_kwargs["enabled"] else 0
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
) -> list[Atoms]:
    """Generate bootstrap structures (dimers/trimers/amorphous/MP/isolated
    atoms), reading targets from config's structure-type namespaces plus
    the shared workflow.elements list.

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
        num_dimers_per_combo=(
            dimer_kwargs["num_dimers_per_combo"] if dimer_kwargs["enabled"] else 0
        ),
        num_trimers_per_combo=(
            trimer_kwargs["num_trimers_per_combo"] if trimer_kwargs["enabled"] else 0
        ),
        num_amorphous=(
            amorphous_kwargs["num_amorphous"] if amorphous_kwargs["enabled"] else 0
        ),
        # Note: create_initialization_atoms_list (old, shared) only ever
        # generates stretch/compress structures from MP-fetched ones -- so
        # this is 0 whenever mp_kwargs.enabled is False too, regardless of
        # stretch_compress_targets_kwargs.enabled; that dependency is a
        # hard constraint of the shared function, not something this
        # module can decouple.
        num_stretch_compress_per_mp=(
            stretch_compress_targets_kwargs["num_stretch_compress_per_mp"]
            if stretch_compress_targets_kwargs["enabled"]
            else 0
        ),
        densities_list=amorphous_kwargs["densities_list"],
        deform_xyz=stretch_compress_targets_kwargs["deform_xyz"],
        max_deformation=stretch_compress_targets_kwargs["max_deformation"],
        max_atom_number=mp_kwargs["max_atom_number"],
        amorphous_atom_number=amorphous_kwargs["amorphous_atom_number"],
        mp_max_energy_above_hull=mp_kwargs["mp_max_energy_above_hull"],
        composition_list=amorphous_kwargs["composition_list"],
        seed=amorphous_kwargs["seed"],
        isolated_atoms_override=(
            (needs["isolated_atoms"] or None) if needs is not None else None
        ),
        dimer_override=(needs["dimer_override"] or None) if needs is not None else None,
        trimer_override=(needs["trimer_override"] or None)
        if needs is not None
        else None,
        amorphous_override=(
            (needs["amorphous_override"] or None) if needs is not None else None
        ),
    )
