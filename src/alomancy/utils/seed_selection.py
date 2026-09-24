"""Two responsibilities extracted from
``structure_generation.select_initial_structures.select_initial_structures``
(still in place unchanged, still what the current production
``generate_structures`` orchestration calls) for the modular AL
architecture:

1. ``filter_eligible_structures`` -- eligibility filtering (config_type
   inclusion, atom_number_range, chem_formula_list). Called once, by the
   skeleton, before any generator's ``generate()`` -- producing the
   population every generator receives (see the architecture plan's
   structure-generation decision).
2. ``select_diverse_seeds`` -- diversity-maximizing selection. Called
   internally by generators (MD today) that can only run from a bounded,
   individually-submitted set of seeds; picks a subset from an
   already-eligible population, assigns per-job diversity metadata
   (``md_seed``) and provenance tagging (``config_type``/``job_id``), and
   handles reuse-with-replacement when the requested concurrency exceeds
   the pool size. EZGA does not call this -- it uses the eligible
   population directly (population-based genetic search).
"""

import logging
import warnings

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def filter_eligible_structures(
    structures: list[Atoms],
    chem_formula_list: list[str] | None = None,
    selectable_configs: list[str] | None = None,
    atom_number_range: tuple[int, int] = (0, 0),
) -> list[Atoms]:
    """Filter ``structures`` down to the eligible population for structure
    generation. Raises ValueError if nothing survives filtering.

    ``selectable_configs``, when given, implicitly includes "high_sd" --
    generated candidates from a prior AL loop are always eligible seeds for
    the next one, matching today's select_initial_structures behavior.
    """
    if selectable_configs is not None:
        selectable_configs = [*selectable_configs, "high_sd"]

    logger.debug(
        "available config_types: %s",
        {s.info.get("config_type") for s in structures},
    )
    logger.info("Filtering eligible structures from %d available.", len(structures))

    if atom_number_range != (0, 0):
        if atom_number_range[0] > atom_number_range[1]:
            raise ValueError(
                "atom_number_range must be a tuple of two integers where the "
                "first is less than or equal to the second"
            )
        if atom_number_range[0] < 2:
            warnings.warn(
                f"atom_number_range minimum value is {atom_number_range[0]}, "
                "which allows single-atom structures. This can lead to "
                "problems with some structure generators like MD simulations, "
                "as single atoms cannot form proper molecular dynamics "
                "trajectories. Consider setting the minimum to 2 or higher.",
                UserWarning,
                stacklevel=2,
            )

    if chem_formula_list is None:
        chem_formula_list = []

    if atom_number_range != (0, 0) and chem_formula_list:
        eligible = [
            s
            for s in structures
            if s.get_chemical_formula() in chem_formula_list
            and atom_number_range[0] <= len(s) <= atom_number_range[1]
        ]
    elif atom_number_range != (0, 0):
        eligible = [
            s
            for s in structures
            if atom_number_range[0] <= len(s) <= atom_number_range[1]
        ]
    elif chem_formula_list:
        eligible = [
            s for s in structures if s.get_chemical_formula() in chem_formula_list
        ]
    else:
        eligible = list(structures)

    if selectable_configs is not None:
        eligible = [
            s for s in eligible if s.info.get("config_type") in selectable_configs
        ]

    if not eligible:
        raise ValueError(
            "No structures available to select from after filtering "
            f"(chem_formula_list={chem_formula_list}, selectable_configs="
            f"{selectable_configs}, atom_number_range={atom_number_range})."
        )

    logger.info("Eligible structures after filtering: %d", len(eligible))
    return eligible


def _assign_md_seeds(atoms_list: list[Atoms], seed: int) -> None:
    """Give each selected structure a distinct atoms.info['md_seed'].

    Needed because the same source structure can be selected more than once
    (when the requested concurrency exceeds the number of eligible
    structures) -- a generator seeds its stochastic dynamics from this value
    so duplicate starting structures still diverge into different
    trajectories instead of running identical MD.
    """
    for i, atoms in enumerate(atoms_list):
        atoms.info["md_seed"] = seed + i


def mark_structures_for_dft(
    atoms_list: list[Atoms], base_name: str, job_name: str
) -> None:
    for atoms in atoms_list:
        atoms.info["job_id"] = atoms.info.get("job_id", -1)
        atoms.info["config_type"] = f"{base_name}_{job_name}"


def select_diverse_seeds(
    base_name: str,
    job_name: str,
    eligible_structures: list[Atoms],
    max_number_of_concurrent_jobs: int = 5,
    enforce_chemical_diversity: bool = False,
    seed: int = 803,
) -> list[Atoms]:
    """Pick ``max_number_of_concurrent_jobs`` seeds from an already-eligible
    population (see filter_eligible_structures), maximizing configurational
    diversity when requested. Reuses structures with distinct md_seed
    values when the pool is smaller than the requested concurrency, rather
    than erroring.
    """
    reuse = len(eligible_structures) < max_number_of_concurrent_jobs
    if reuse:
        logger.warning(
            "Only %d structures available for %d concurrent structure_generation "
            "jobs; reusing structures to fill the requested concurrency. Each "
            "reused structure is assigned a distinct MD seed (atoms.info['md_seed']) "
            "so its duplicate runs diverge into different trajectories.",
            len(eligible_structures),
            max_number_of_concurrent_jobs,
        )

    if not enforce_chemical_diversity:
        selected = [
            eligible_structures[x].copy()
            for x in np.random.choice(
                np.array(range(len(eligible_structures))),
                max_number_of_concurrent_jobs,
                replace=reuse,
            )
        ]
        _assign_md_seeds(selected, seed)
        mark_structures_for_dft(selected, base_name, job_name)
        return selected

    # Ensure chemical diversity by selecting unique chemical formulas. If
    # there are fewer unique formulas than max_number_of_concurrent_jobs,
    # select all and pad with random repeats.
    unique_chemical_formulas = {s.get_chemical_formula() for s in eligible_structures}
    if len(unique_chemical_formulas) <= max_number_of_concurrent_jobs:
        list_of_formulas = list(unique_chemical_formulas)
        extra_formulas = [
            np.random.choice(list(unique_chemical_formulas), replace=False)
            for _ in range(max_number_of_concurrent_jobs - len(list_of_formulas))
        ]
        list_of_formulas.extend(extra_formulas)
    else:
        # Select formulas with probability inversely proportional to their
        # frequency in the dataset to promote diversity.
        all_chemical_formulas = [s.get_chemical_formula() for s in eligible_structures]
        formula_counts = {
            formula: all_chemical_formulas.count(formula)
            for formula in set(all_chemical_formulas)
        }
        formula_probabilities = {
            formula: 1 / count for formula, count in formula_counts.items()
        }
        list_of_formulas = list(
            np.random.choice(
                list(unique_chemical_formulas),
                max_number_of_concurrent_jobs,
                replace=False,
                p=[
                    formula_probabilities[formula] / sum(formula_probabilities.values())
                    for formula in unique_chemical_formulas
                ],
            )
        )

    selected = []
    for chemical_formula in list_of_formulas:
        formula_structures = [
            s
            for s in eligible_structures
            if s.get_chemical_formula() == chemical_formula
        ]
        chosen_idx = np.random.choice(np.array(range(len(formula_structures))))
        selected.append(formula_structures[chosen_idx].copy())

    _assign_md_seeds(selected, seed)
    mark_structures_for_dft(selected, base_name, job_name)

    logger.debug(
        "Structures selected: %s", [a.get_chemical_formula() for a in selected]
    )
    return selected
