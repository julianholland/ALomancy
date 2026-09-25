import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import write

from alomancy.initialize.amorphize import create_amorphous_atoms_list
from alomancy.initialize.mp_interface import atoms_list_from_mp
from alomancy.initialize.rattle import create_rattle_atoms_list
from alomancy.initialize.singles_dimers_trimers import (
    create_dimer_atoms_list,
    create_single_atoms_list,
    create_trimer_atoms_list,
)
from alomancy.initialize.stretch_and_compress import create_stretch_compress_atoms_list

if TYPE_CHECKING:
    from alomancy.database.global_database import GlobalDatabase

logger = logging.getLogger(__name__)


def compute_initialization_needs(
    db: "GlobalDatabase",
    elements: list[str],
    _single_atoms: bool,
    mp_structures: bool,
    num_dimers_per_combo: int,
    num_trimers_per_combo: int,
    num_amorphous: int,
) -> dict:
    """
    Compare DB contents against initialization targets and return what still
    needs to be generated.

    Dimers and trimers are checked per element-combination; amorphous by total
    count; IsolatedAtoms per element; MP structures once only.

    Returns a dict with keys:
      'isolated_atoms'      list[str]        elements not yet in the DB
      'dimer_override'      dict[str, int]   {formula: count_still_needed}
      'trimer_override'     dict[str, int]
      'amorphous_override'  int              total still needed
      'mp_structures'       bool             True iff MP fetch is still needed
    """
    needs: dict = {}

    # Single pass over the DB — avoids four separate O(N) container scans.
    all_counts = db.count_all_by_config_type_and_formula()
    isolated_counts = all_counts.get("IsolatedAtom", {})
    dimer_counts = all_counts.get("init_dimer", {})
    trimer_counts = all_counts.get("init_trimer", {})

    # IsolatedAtom: one per element
    needs["isolated_atoms"] = [el for el in elements if el not in isolated_counts]

    # Dimers: per element-combination
    dimer_override: dict[str, int] = {}
    for combo in itertools.combinations_with_replacement(elements, 2):
        formula = Atoms(list(combo)).get_chemical_formula()
        have = dimer_counts.get(formula, 0)
        still_need = max(0, num_dimers_per_combo - have)
        if still_need:
            dimer_override[formula] = still_need
    needs["dimer_override"] = dimer_override

    # Trimers: per element-combination
    trimer_override: dict[str, int] = {}
    for combo in itertools.combinations_with_replacement(elements, 3):
        formula = Atoms(list(combo)).get_chemical_formula()
        have = trimer_counts.get(formula, 0)
        still_need = max(0, num_trimers_per_combo - have)
        if still_need:
            trimer_override[formula] = still_need
    needs["trimer_override"] = trimer_override

    # Amorphous: total count (sum across all formulas for this config_type)
    have_amorphous = sum(all_counts.get("init_amorphous", {}).values())
    needs["amorphous_override"] = max(0, num_amorphous - have_amorphous)

    # MP structures: fetch once (skip if any init_MP already in DB)
    needs["mp_structures"] = mp_structures and "init_MP" not in all_counts

    return needs


def create_initialization_atoms_list(
    work_dir: str,
    elements: list[str],
    mp_structures: bool = True,
    single_atoms: bool = True,
    num_dimers_per_combo: int = 10,
    num_trimers_per_combo: int = 5,
    num_amorphous: int = 100,
    num_stretch_compress_per_target: int = 5,
    densities_list: list[float] | None = None,
    max_lattice_deformation: float = 0.2,
    max_atom_number: int = 20,
    amorphous_atom_number: int = 20,
    composition_list: list[list[str]] | None = None,
    seed: int = 803,
    mp_max_energy_above_hull: float = 0.1,
    target_config_types: list[str] | None = None,
    num_rattled_per_target: int = 0,
    rattle_standard_deviation: float | None = None,
    rattle_seed: int = 803,
    # Override kwargs supplied by compute_initialization_needs to skip
    # already-completed subsets.  When None, the full target is used.
    isolated_atoms_override: list[str] | None = None,
    dimer_override: dict[str, int] | None = None,
    trimer_override: dict[str, int] | None = None,
    amorphous_override: int | None = None,
) -> list[Atoms]:
    """
    Generate structures for the initialization phase.

    When called without override kwargs (first run), generates the full set
    defined by num_*_per_combo / num_amorphous targets.

    When override kwargs are supplied (from compute_initialization_needs),
    only the missing subset is generated — enabling idempotent restarts.

    Parameters
    ----------
    num_dimers_per_combo
        Number of dimer structures to generate per element combination.
    num_trimers_per_combo
        Number of trimer structures to generate per element combination.
    num_amorphous
        Total number of amorphous structures to generate.
    num_stretch_compress_per_target
        Number of stretched/compressed variants per target structure (see
        target_config_types).
    max_atom_number
        Maximum atom count for structures fetched from the Materials Project
        (passed to `atoms_list_from_mp` as `max_num_atoms`). Independent of
        amorphous_atom_number — does not affect amorphous cell size.
    mp_max_energy_above_hull
        Maximum energy above hull for structures fetched from the Materials
        Project (passed to `atoms_list_from_mp` as `max_energy_above_hull`).
    amorphous_atom_number
        Target atom count per generated amorphous cell (passed to
        `create_amorphous_atoms_list` as `atom_number`). Independent of
        max_atom_number — does not affect the MP fetch cap.
    target_config_types
        config_type values (e.g. "init_MP", "init_amorphous") identifying
        which of this call's freshly-generated structures count as "target"
        structures -- every one of them (not just Materials Project ones)
        is fed through both stretch/compress and rattle. Structures already
        resident in the DB from an earlier run are not re-derived from here
        (matches this function's existing restart behavior: only the
        subset generated in this call is ever subject to these two
        transforms).
    num_rattled_per_target
        Number of independently-rattled copies per target structure.
    rattle_standard_deviation
        Standard deviation (Angstrom) for ase.Atoms.rattle, applied to each
        target structure. Only read when num_rattled_per_target > 0.
    rattle_seed
        Base seed for rattle (copy i of a given target uses rattle_seed +
        i); independent of `seed` above, which is amorphous generation's
        own seed.
    isolated_atoms_override
        If provided, only generate isolated atoms for these elements.
    dimer_override
        If provided, {formula: count} of dimers still needed per combo.
    trimer_override
        If provided, {formula: count} of trimers still needed per combo.
    amorphous_override
        If provided, generate this many amorphous structures instead of
        num_amorphous.
    """
    assert len(elements) > 0, "At least one element must be specified."

    # --- MP structures -------------------------------------------------
    mp_atoms_list: list[Atoms] = []
    if mp_structures:
        mp_atoms_list = atoms_list_from_mp(
            elements=elements,
            max_energy_above_hull=mp_max_energy_above_hull,
            max_num_atoms=max_atom_number,
            relax_structures=True,
        )
        logger.info(
            "Retrieved %d structures from Materials Project.", len(mp_atoms_list)
        )

    # --- Single / isolated atoms ---------------------------------------
    elements_for_singles = (
        isolated_atoms_override if isolated_atoms_override is not None else elements
    )
    single_atoms_list: list[Atoms] = []
    if single_atoms and elements_for_singles:
        for el in elements_for_singles:
            single_atoms_list.extend(create_single_atoms_list(element=el))

    # --- Dimers --------------------------------------------------------
    dimer_atoms_list: list[Atoms] = []
    all_dimer_combos = list(itertools.combinations_with_replacement(elements, 2))

    if dimer_override is not None:
        # Only generate combos that still need more structures
        combos_to_generate = {
            combo: dimer_override.get(Atoms(list(combo)).get_chemical_formula(), 0)
            for combo in all_dimer_combos
        }
    else:
        combos_to_generate = dict.fromkeys(all_dimer_combos, num_dimers_per_combo)

    for combo, count in combos_to_generate.items():
        if count > 0:
            dimer_atoms_list.extend(
                create_dimer_atoms_list(
                    element_a=combo[0], element_b=combo[1], num_dimers=count
                )
            )
    logger.info(
        "Created %d dimer structures across %d combos.",
        len(dimer_atoms_list),
        sum(c > 0 for c in combos_to_generate.values()),
    )

    # --- Trimers -------------------------------------------------------
    trimer_atoms_list: list[Atoms] = []
    all_trimer_combos = list(itertools.combinations_with_replacement(elements, 3))

    if trimer_override is not None:
        trimer_combos_to_generate = {
            combo: trimer_override.get(Atoms(list(combo)).get_chemical_formula(), 0)
            for combo in all_trimer_combos
        }
    else:
        trimer_combos_to_generate = dict.fromkeys(
            all_trimer_combos, num_trimers_per_combo
        )

    for combo, count in trimer_combos_to_generate.items():
        if count > 0:
            trimer_atoms_list.extend(
                create_trimer_atoms_list(
                    element_a=combo[0],
                    element_b=combo[1],
                    element_c=combo[2],
                    num_trimers=count,
                )
            )
    logger.info(
        "Created %d trimer structures across %d combos.",
        len(trimer_atoms_list),
        sum(1 for c in trimer_combos_to_generate.values() if c > 0),
    )

    # --- Amorphous -----------------------------------------------------
    amorphous_target = (
        amorphous_override if amorphous_override is not None else num_amorphous
    )
    amorphous_atoms_list: list[Atoms] = []
    if densities_list is None:
        densities_list = [1.0]
    for density in densities_list:
        logger.debug("Creating amorphous structures with density %.2f g/cm^3.", density)
        per_density = int(np.floor(amorphous_target / len(densities_list)))
        if per_density == 0:
            continue
        amorphous_atoms_list.extend(
            create_amorphous_atoms_list(
                elements=elements,
                atom_number=amorphous_atom_number,
                density=density,
                num_structures=per_density,
                seed=seed,
                composition_list=composition_list,
            )
        )
    logger.info("Created %d amorphous structures.", len(amorphous_atoms_list))

    # --- Assemble the base pool -----------------------------------------
    # Built before stretch/compress and rattle below, since both draw from
    # whichever of these freshly-generated structures count as "target"
    # structures (target_config_types), not just Materials Project ones.
    base_atoms_list: list[Atoms] = []
    if single_atoms:
        base_atoms_list.extend(single_atoms_list)
    if mp_structures:
        base_atoms_list.extend(mp_atoms_list)
    base_atoms_list.extend(dimer_atoms_list)
    base_atoms_list.extend(trimer_atoms_list)
    base_atoms_list.extend(amorphous_atoms_list)

    # --- Stretch/compress + rattle on target structures -----------------
    target_config_type_set = set(target_config_types or [])
    target_atoms_list = [
        a
        for a in base_atoms_list
        if a.info.get("config_type") in target_config_type_set
    ]

    stretch_compress_atoms_list: list[Atoms] = []
    rattle_atoms_list: list[Atoms] = []
    for target in target_atoms_list:
        stretch_compress_atoms_list.extend(
            create_stretch_compress_atoms_list(
                atoms=target,
                max_lattice_deformation=max_lattice_deformation,
                num_structures=num_stretch_compress_per_target,
            )
        )
        rattle_atoms_list.extend(
            create_rattle_atoms_list(
                atoms=target,
                rattle_standard_deviation=rattle_standard_deviation or 0.0,
                num_structures=num_rattled_per_target,
                seed=rattle_seed,
            )
        )
    logger.info(
        "Created %d stretch/compress and %d rattle structures from %d target "
        "structures.",
        len(stretch_compress_atoms_list),
        len(rattle_atoms_list),
        len(target_atoms_list),
    )

    # --- Assemble --------------------------------------------------------
    total_atoms_list = list(base_atoms_list)
    total_atoms_list.extend(stretch_compress_atoms_list)
    total_atoms_list.extend(rattle_atoms_list)

    logger.info(
        "Created %d total structures for initialization.", len(total_atoms_list)
    )
    out_path = Path(work_dir, "initialization_structures_generated.xyz")
    write(out_path, total_atoms_list)
    logger.info("Saved initialization structures to %s.", out_path)
    return total_atoms_list
