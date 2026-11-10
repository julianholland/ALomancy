import itertools
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ase import Atoms
from ase.io import write

from alomancy.initialize.amorphize import (
    create_amorphous_atoms_list,
    create_amorphous_cluster_atoms_list,
)
from alomancy.initialize.crystals import create_crystal_atoms_list
from alomancy.initialize.molecules import (
    create_molecule_atoms_list,
    create_molecule_cluster_atoms_list,
    create_molecules_in_box_atoms_list,
    molecule_counts,
)
from alomancy.initialize.mp_interface import atoms_list_from_mp
from alomancy.initialize.rattle import create_rattle_atoms_list
from alomancy.initialize.singles_dimers_trimers import (
    create_dimer_atoms_list,
    create_single_atoms_list,
    create_trimer_atoms_list,
)
from alomancy.initialize.stretch_and_compress import create_stretch_compress_atoms_list
from alomancy.utils.molecules import build_molecule

if TYPE_CHECKING:
    from alomancy.database.global_database import GlobalDatabase

logger = logging.getLogger(__name__)


def compute_initialization_needs(
    db: "GlobalDatabase",
    elements: list[str],
    _single_atoms: bool,
    mp_structures: bool,
    num_of_dimers_per_combo: int,
    num_of_trimers_per_combo: int,
    num_of_amorphous_structures: int,
    molecules: list[str] | None = None,
    box_compositions: list[dict] | None = None,
    num_of_amorphous_clusters: int = 0,
    cluster_compositions: list[dict] | None = None,
    crystal_targets: dict[str, list[str]] | None = None,
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
      'molecule_override'   dict[str, int]   {formula: count_still_needed}
      'box_override'        dict[str, int]   {formula: count_still_needed}
      'amorphous_cluster_override'  int      total still needed
      'molecule_cluster_override'   dict[str, int] {formula: count_still_needed}

    *molecules* (molecule_kwargs.molecules; one structure each) and
    *box_compositions* (molecules_in_box_kwargs.box_compositions;
    num_of_structures each) are counted per chemical formula, so two
    names or compositions with the same formula share one target.
    *num_of_amorphous_clusters* (amorphous_cluster_kwargs) is counted in
    total like the amorphous cells, *cluster_compositions*
    (molecule_cluster_kwargs) per formula like the boxes.
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
        still_need = max(0, num_of_dimers_per_combo - have)
        if still_need:
            dimer_override[formula] = still_need
    needs["dimer_override"] = dimer_override

    # Trimers: per element-combination
    trimer_override: dict[str, int] = {}
    for combo in itertools.combinations_with_replacement(elements, 3):
        formula = Atoms(list(combo)).get_chemical_formula()
        have = trimer_counts.get(formula, 0)
        still_need = max(0, num_of_trimers_per_combo - have)
        if still_need:
            trimer_override[formula] = still_need
    needs["trimer_override"] = trimer_override

    # Amorphous: total count (sum across all formulas for this config_type)
    have_amorphous = sum(all_counts.get("init_amorphous", {}).values())
    needs["amorphous_override"] = max(0, num_of_amorphous_structures - have_amorphous)

    # MP structures: fetch once (skip if any init_MP already in DB)
    needs["mp_structures"] = mp_structures and "init_MP" not in all_counts

    needs["molecule_override"] = _still_needed(
        _molecule_targets(molecules or []), all_counts.get("init_molecule", {})
    )
    needs["box_override"] = _still_needed(
        _box_targets(box_compositions or []),
        all_counts.get("init_molecules_in_box", {}),
    )
    have_clusters = sum(all_counts.get("init_amorphous_cluster", {}).values())
    needs["amorphous_cluster_override"] = max(
        0, num_of_amorphous_clusters - have_clusters
    )
    needs["molecule_cluster_override"] = _still_needed(
        _box_targets(cluster_compositions or []),
        all_counts.get("init_molecule_cluster", {}),
    )
    # Crystals: formula can't tell fcc from bcc (both "Pt"), so they are
    # counted by (formula, crystal_structure); {element: [missing]}.
    crystal_override: dict[str, list[str]] = {}
    if crystal_targets:
        counts = db.count_by_info("init_crystal", "crystal_structure")
        have = (
            {(re.sub(r"\d", "", f), str(v)) for (f, v) in counts}
            if isinstance(counts, dict)
            else set()
        )
        for element, structures in crystal_targets.items():
            missing = [s for s in structures if (element, s) not in have]
            if missing:
                crystal_override[element] = missing
    needs["crystal_override"] = crystal_override

    return needs


def _molecule_formula(name: str) -> str:
    return str(build_molecule(name).get_chemical_formula())


def _box_formula(composition: dict) -> str:
    counts = molecule_counts(composition["molecules"], composition["num_of_molecules"])
    symbols = [
        symbol
        for name, count in counts.items()
        for symbol in build_molecule(name).get_chemical_symbols() * count
    ]
    return str(Atoms(symbols).get_chemical_formula())


def _molecule_targets(molecules: list[str]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for name in molecules:
        formula = _molecule_formula(name)
        targets[formula] = targets.get(formula, 0) + 1
    return targets


def _box_targets(box_compositions: list[dict]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for composition in box_compositions:
        formula = _box_formula(composition)
        targets[formula] = targets.get(formula, 0) + composition["num_of_structures"]
    return targets


def _still_needed(targets: dict[str, int], have: dict[str, int]) -> dict[str, int]:
    return {
        formula: target - have.get(formula, 0)
        for formula, target in targets.items()
        if target > have.get(formula, 0)
    }


def create_initialization_atoms_list(
    work_dir: str,
    elements: list[str],
    mp_structures: bool = True,
    single_atoms: bool = True,
    num_of_dimers_per_combo: int = 10,
    num_of_trimers_per_combo: int = 5,
    num_of_amorphous_structures: int = 100,
    num_of_stretch_compress_per_target: int = 5,
    density_range: list[float] | None = None,
    max_lattice_deformation: float = 0.2,
    max_num_of_atoms: int = 20,
    num_of_atoms_per_amorphous: int = 20,
    composition_list: list[list[str]] | None = None,
    seed: int = 803,
    mp_max_energy_above_hull: float = 0.1,
    target_config_types: list[str] | None = None,
    num_of_rattled_per_target: int = 0,
    rattle_standard_deviation: float | None = None,
    rattle_seed: int = 803,
    dimer_distance_range: list[float] | None = None,
    trimer_distance_range: list[float] | None = None,
    crystal_targets: dict[str, list[str]] | None = None,
    crystal_lattice_constants: dict | None = None,
    crystal_volume_per_atom: dict | None = None,
    crystal_override: dict[str, list[str]] | None = None,
    # Override kwargs supplied by compute_initialization_needs to skip
    # already-completed subsets.  When None, the full target is used.
    isolated_atoms_override: list[str] | None = None,
    dimer_override: dict[str, int] | None = None,
    trimer_override: dict[str, int] | None = None,
    amorphous_override: int | None = None,
    molecules: list[str] | None = None,
    molecule_vacuum: float = 10.0,
    box_compositions: list[dict] | None = None,
    box_min_distance: float = 1.5,
    box_max_attempts_per_molecule: int = 1000,
    molecule_override: dict[str, int] | None = None,
    box_override: dict[str, int] | None = None,
    num_of_amorphous_clusters: int = 0,
    num_of_atoms_per_cluster: int = 10,
    cluster_density_range: list[float] | None = None,
    cluster_composition_list: list[list[str]] | None = None,
    cluster_vacuum: float = 10.0,
    amorphous_cluster_override: int | None = None,
    cluster_compositions: list[dict] | None = None,
    cluster_min_distance: float = 1.5,
    cluster_max_attempts_per_molecule: int = 1000,
    molecule_cluster_vacuum: float = 10.0,
    molecule_cluster_override: dict[str, int] | None = None,
) -> list[Atoms]:
    """
    Generate structures for the initialization phase.

    When called without override kwargs (first run), generates the full set
    defined by num_*_per_combo / num_of_amorphous_structures targets.

    When override kwargs are supplied (from compute_initialization_needs),
    only the missing subset is generated — enabling idempotent restarts.

    Parameters
    ----------
    num_of_dimers_per_combo
        Number of dimer structures to generate per element combination.
    num_of_trimers_per_combo
        Number of trimer structures to generate per element combination.
    num_of_amorphous_structures
        Total number of amorphous structures to generate.
    num_of_stretch_compress_per_target
        Number of stretched/compressed variants per target structure (see
        target_config_types).
    max_num_of_atoms
        Maximum atom count for structures fetched from the Materials Project
        (passed to `atoms_list_from_mp` as `max_num_atoms`). Independent of
        num_of_atoms_per_amorphous — does not affect amorphous cell size.
    mp_max_energy_above_hull
        Maximum energy above hull for structures fetched from the Materials
        Project (passed to `atoms_list_from_mp` as `max_energy_above_hull`).
    num_of_atoms_per_amorphous
        Target atom count per generated amorphous cell (passed to
        `create_amorphous_atoms_list` as `atom_number`). Independent of
        max_num_of_atoms — does not affect the MP fetch cap.
    target_config_types
        config_type values (e.g. "init_MP", "init_amorphous") identifying
        which of this call's freshly-generated structures count as "target"
        structures -- every one of them (not just Materials Project ones)
        is fed through both stretch/compress and rattle. Structures already
        resident in the DB from an earlier run are not re-derived from here
        (matches this function's existing restart behavior: only the
        subset generated in this call is ever subject to these two
        transforms).
    num_of_rattled_per_target
        Number of independently-rattled copies per target structure.
    rattle_standard_deviation
        Standard deviation (Angstrom) for ase.Atoms.rattle, applied to each
        target structure. Only read when num_of_rattled_per_target > 0.
        None: 3% of each structure's nearest-neighbour distance. A 1-atom
        cell is rattled as a 2x2x2 supercell (initialize/rattle.py).
    seed
        general.seed: seeds the amorphous cells and the molecule boxes.
    rattle_seed
        Base seed for rattle (copy i of a given target uses rattle_seed +
        i); generate() passes general.seed here too.
    isolated_atoms_override
        If provided, only generate isolated atoms for these elements.
    dimer_override
        If provided, {formula: count} of dimers still needed per combo.
    trimer_override
        If provided, {formula: count} of trimers still needed per combo.
    amorphous_override
        If provided, generate this many amorphous structures instead of
        num_of_amorphous_structures.
    density_range
        [min, max] g/cm^3: each amorphous structure's density is drawn
        uniformly at random from the range (default [1.0, 1.0]).
    molecules
        Molecule names (formula or ``smiles-...``) to make one
        equilibrium structure of each, in vacuum (init_molecule).
    box_compositions
        Periodic boxes of molecules (init_molecules_in_box), each entry
        {molecules: {name: ratio}, density_range: [min, max],
        num_of_molecules, num_of_structures}; see initialize/molecules.py.
    molecule_override / box_override
        If provided, {formula: count} still needed (compute_initialization_needs).
    num_of_amorphous_clusters, num_of_atoms_per_cluster, cluster_density_range,
    cluster_composition_list, cluster_vacuum, amorphous_cluster_override
        Non-periodic random clusters (init_amorphous_cluster,
        amorphous_cluster_kwargs); see create_amorphous_cluster_atoms_list.
    cluster_compositions, cluster_min_distance,
    cluster_max_attempts_per_molecule, molecule_cluster_vacuum,
    molecule_cluster_override
        Non-periodic clusters of molecules (init_molecule_cluster,
        molecule_cluster_kwargs), entries as for box_compositions.
    """
    assert len(elements) > 0, "At least one element must be specified."

    # --- MP structures -------------------------------------------------
    mp_atoms_list: list[Atoms] = []
    if mp_structures:
        mp_atoms_list = atoms_list_from_mp(
            elements=elements,
            max_energy_above_hull=mp_max_energy_above_hull,
            max_num_atoms=max_num_of_atoms,
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
        combos_to_generate = dict.fromkeys(all_dimer_combos, num_of_dimers_per_combo)

    for combo, count in combos_to_generate.items():
        if count > 0:
            dimer_atoms_list.extend(
                create_dimer_atoms_list(
                    element_a=combo[0],
                    element_b=combo[1],
                    num_dimers=count,
                    distance_range=dimer_distance_range,
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
            all_trimer_combos, num_of_trimers_per_combo
        )

    for combo, count in trimer_combos_to_generate.items():
        if count > 0:
            trimer_atoms_list.extend(
                create_trimer_atoms_list(
                    element_a=combo[0],
                    element_b=combo[1],
                    element_c=combo[2],
                    num_trimers=count,
                    distance_range=trimer_distance_range,
                )
            )
    logger.info(
        "Created %d trimer structures across %d combos.",
        len(trimer_atoms_list),
        sum(1 for c in trimer_combos_to_generate.values() if c > 0),
    )

    # --- Amorphous -----------------------------------------------------
    amorphous_target = (
        amorphous_override
        if amorphous_override is not None
        else num_of_amorphous_structures
    )
    amorphous_atoms_list: list[Atoms] = []
    if amorphous_target > 0:
        amorphous_atoms_list = create_amorphous_atoms_list(
            elements=elements,
            atom_number=num_of_atoms_per_amorphous,
            density_range=list(density_range or [1.0, 1.0]),
            num_structures=amorphous_target,
            seed=seed,
            composition_list=composition_list,
            # How many the DB already holds: new cells on every restart.
            attempt=max(0, num_of_amorphous_structures - amorphous_target),
        )
    logger.info("Created %d amorphous structures.", len(amorphous_atoms_list))

    # --- Molecules in vacuum --------------------------------------------
    molecule_atoms_list: list[Atoms] = []
    remaining = dict(molecule_override) if molecule_override is not None else None
    for name in molecules or []:
        if remaining is not None:
            formula = _molecule_formula(name)
            if remaining.get(formula, 0) <= 0:
                continue
            remaining[formula] -= 1
        molecule_atoms_list.extend(
            create_molecule_atoms_list(name=name, vacuum=molecule_vacuum)
        )
    logger.info("Created %d molecules in vacuum.", len(molecule_atoms_list))

    # --- Molecules in a box ---------------------------------------------
    box_atoms_list: list[Atoms] = []
    remaining = dict(box_override) if box_override is not None else None
    for index, composition in enumerate(box_compositions or []):
        num = composition["num_of_structures"]
        still = num
        if remaining is not None:
            formula = _box_formula(composition)
            still = min(num, remaining.get(formula, 0))
            if still <= 0:
                continue
            remaining[formula] -= still
        box_atoms_list.extend(
            create_molecules_in_box_atoms_list(
                molecules_ratio=composition["molecules"],
                density_range=composition["density_range"],
                num_of_molecules=composition["num_of_molecules"],
                num_of_structures=still,
                min_distance=box_min_distance,
                max_attempts=box_max_attempts_per_molecule,
                seed=seed + index,
                # How many the DB already holds: new boxes on every restart.
                attempt=num - still,
            )
        )
    logger.info("Created %d boxes of molecules.", len(box_atoms_list))

    # --- Amorphous clusters (non-periodic) -------------------------------
    cluster_target = (
        amorphous_cluster_override
        if amorphous_cluster_override is not None
        else num_of_amorphous_clusters
    )
    amorphous_cluster_atoms_list = create_amorphous_cluster_atoms_list(
        elements=elements,
        atom_number=num_of_atoms_per_cluster,
        density_range=list(cluster_density_range or [1.0, 1.0]),
        num_structures=cluster_target,
        seed=seed,
        composition_list=cluster_composition_list,
        vacuum=cluster_vacuum,
        # How many the DB already holds: new clusters on every restart.
        attempt=max(0, num_of_amorphous_clusters - cluster_target),
    )
    if amorphous_cluster_atoms_list:
        logger.info("Created %d amorphous clusters.", len(amorphous_cluster_atoms_list))

    # --- Clusters of molecules (non-periodic) ----------------------------
    molecule_cluster_atoms_list: list[Atoms] = []
    remaining = (
        dict(molecule_cluster_override)
        if molecule_cluster_override is not None
        else None
    )
    for index, composition in enumerate(cluster_compositions or []):
        num = composition["num_of_structures"]
        still = num
        if remaining is not None:
            formula = _box_formula(composition)
            still = min(num, remaining.get(formula, 0))
            if still <= 0:
                continue
            remaining[formula] -= still
        molecule_cluster_atoms_list.extend(
            create_molecule_cluster_atoms_list(
                molecules_ratio=composition["molecules"],
                density_range=composition["density_range"],
                num_of_molecules=composition["num_of_molecules"],
                num_of_structures=still,
                min_distance=cluster_min_distance,
                max_attempts=cluster_max_attempts_per_molecule,
                seed=seed + index,
                vacuum=molecule_cluster_vacuum,
                attempt=num - still,
            )
        )
    if molecule_cluster_atoms_list:
        logger.info(
            "Created %d clusters of molecules.", len(molecule_cluster_atoms_list)
        )

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
    base_atoms_list.extend(molecule_atoms_list)
    base_atoms_list.extend(box_atoms_list)
    base_atoms_list.extend(amorphous_cluster_atoms_list)
    base_atoms_list.extend(molecule_cluster_atoms_list)

    # --- Crystal structures (crystal_structures_kwargs) -----------------
    crystal_atoms_list: list[Atoms] = []
    if crystal_targets:
        crystal_atoms_list = create_crystal_atoms_list(
            list(crystal_targets),
            [],
            lattice_constants=crystal_lattice_constants,
            volume_per_atom=crystal_volume_per_atom,
            override=crystal_override
            if crystal_override is not None
            else crystal_targets,
        )
    if crystal_atoms_list:
        logger.info("Created %d crystal structures.", len(crystal_atoms_list))
    base_atoms_list.extend(crystal_atoms_list)

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
                num_structures=num_of_stretch_compress_per_target,
            )
        )
        rattle_atoms_list.extend(
            create_rattle_atoms_list(
                atoms=target,
                rattle_standard_deviation=rattle_standard_deviation,
                num_structures=num_of_rattled_per_target,
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
