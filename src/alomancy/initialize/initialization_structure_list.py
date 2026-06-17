import itertools
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import write

from alomancy.initialize.amorphize import create_amorphous_atoms_list
from alomancy.initialize.mp_interface import atoms_list_from_mp
from alomancy.initialize.singles_dimers_trimers import (
    create_dimer_atoms_list,
    create_single_atoms_list,
    create_trimer_atoms_list,
)
from alomancy.initialize.stretch_and_compress import create_stretch_compress_atoms_list

if TYPE_CHECKING:
    from alomancy.database.global_database import GlobalDatabase


def compute_initialization_needs(
    db: "GlobalDatabase",
    elements: list[str],
    single_atoms: bool,
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
    num_stretch_compress_per_mp: int = 5,
    densities_list: list[float] | None = None,
    deform_xyz: bool | list[bool] = False,
    max_deformation: float = 0.2,
    max_atom_number: int = 20,
    composition_list: list[list[str]] | None = None,
    seed: int = 803,
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
    num_stretch_compress_per_mp
        Number of stretched/compressed variants per MP structure.
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
            max_energy_above_hull=0.1,
            max_num_atoms=max_atom_number,
            relax_structures=True,
        )
        print(f"Retrieved {len(mp_atoms_list)} structures from Materials Project.")

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
        combos_to_generate = {combo: num_dimers_per_combo for combo in all_dimer_combos}

    for combo, count in combos_to_generate.items():
        if count > 0:
            dimer_atoms_list.extend(
                create_dimer_atoms_list(
                    element_a=combo[0], element_b=combo[1], num_dimers=count
                )
            )
    print(
        f"Created {len(dimer_atoms_list)} dimer structures "
        f"across {sum(c > 0 for c in combos_to_generate.values())} combos."
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
        trimer_combos_to_generate = {
            combo: num_trimers_per_combo for combo in all_trimer_combos
        }

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
    print(
        f"Created {len(trimer_atoms_list)} trimer structures "
        f"across {sum(1 for c in trimer_combos_to_generate.values() if c > 0)} combos."
    )

    # --- Amorphous -----------------------------------------------------
    amorphous_target = (
        amorphous_override if amorphous_override is not None else num_amorphous
    )
    amorphous_atoms_list: list[Atoms] = []
    if densities_list is None:
        densities_list = [1.0]
    for density in densities_list:
        print(f"Creating amorphous structures with density {density} g/cm^3.")
        per_density = int(np.floor(amorphous_target / len(densities_list)) or 1)
        amorphous_atoms_list.extend(
            create_amorphous_atoms_list(
                elements=elements,
                atom_number=max_atom_number,
                density=density,
                num_structures=per_density,
                seed=seed,
                composition_list=composition_list,
            )
        )
    print(f"Created {len(amorphous_atoms_list)} amorphous structures.")

    # --- Stretch / compress MP structures ------------------------------
    stretch_compress_atoms_list: list[Atoms] = []
    if mp_structures and len(mp_atoms_list) > 0:
        for mp in mp_atoms_list:
            stretch_compress_atoms_list.extend(
                create_stretch_compress_atoms_list(
                    atoms=mp,
                    deform_xyz=deform_xyz,
                    max_deformation=max_deformation,
                    num_structures=num_stretch_compress_per_mp,
                )
            )
    print(
        f"Created {len(stretch_compress_atoms_list)} stretch/compress structures "
        f"from {len(mp_atoms_list)} MP structures."
    )

    # --- Assemble ------------------------------------------------------
    total_atoms_list: list[Atoms] = []
    if single_atoms:
        total_atoms_list.extend(single_atoms_list)
    if mp_structures:
        total_atoms_list.extend(mp_atoms_list)
    total_atoms_list.extend(dimer_atoms_list)
    total_atoms_list.extend(trimer_atoms_list)
    total_atoms_list.extend(amorphous_atoms_list)
    total_atoms_list.extend(stretch_compress_atoms_list)

    print(f"Created {len(total_atoms_list)} total structures for initialization.")
    out_path = Path(work_dir, "initialization_structures_generated.xyz")
    write(out_path, total_atoms_list)
    print(f"Saved initialization structures to {out_path}.")
    return total_atoms_list
