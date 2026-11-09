"""Molecules for initialisation: single molecules in vacuum
(``molecule_kwargs``, config_type ``init_molecule``), periodic boxes of
molecules at a density (``molecules_in_box_kwargs``, config_type
``init_molecules_in_box``), and their non-periodic counterpart, clusters of
molecules in vacuum (``molecule_cluster_kwargs``, config_type
``init_molecule_cluster``) for evaluators without periodic boundary
conditions (ORCA).

Molecules are named as SAGE builds them (a formula such as ``"CO"`` or
``"smiles-<SMILES>"``, see utils/molecules.py). Only the equilibrium
geometry is made here; distorted copies come from the shared
stretch/compress and rattle blocks when the config_type is listed in
general.dataset_kwargs.target_config_types (stretch/compress scales the
atoms with the cell, so on a molecule in vacuum it scales the bonds).

Boxes are packed here rather than with SAGE's add_solvent, which can't be
seeded: each molecule gets a random rotation and position from a numpy rng
seeded with general.seed and is kept when no atom comes closer than
``min_distance`` to another molecule's (minimum image). No force-field
cleanup afterwards: it would distort the molecules.
"""

import logging

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

from alomancy.utils.molecules import build_molecule

logger = logging.getLogger(__name__)

# 1 amu / Å^3 = 1.66054 g/cm^3: a box of total mass M (amu) at density rho
# (g/cm^3) has volume M / (rho * _AMU_PER_A3_TO_G_PER_CM3) Å^3.
_AMU_PER_A3_TO_G_PER_CM3 = 0.6022


def box_composition_label(molecules_ratio: dict[str, float]) -> str:
    """``{"H2O": 2, "smiles-CO": 1}`` -> ``"H2O:2,smiles-CO:1"``."""
    return ",".join(f"{name}:{ratio:g}" for name, ratio in molecules_ratio.items())


def molecule_counts(
    molecules_ratio: dict[str, float], num_of_molecules: int
) -> dict[str, int]:
    """Integer counts summing to *num_of_molecules*, as close to the ratio
    as possible (largest remainder), every molecule at least once."""
    names = list(molecules_ratio)
    if num_of_molecules < len(names):
        raise ValueError(
            f"num_of_molecules {num_of_molecules} is fewer than the "
            f"{len(names)} molecules in {box_composition_label(molecules_ratio)}."
        )
    total = sum(molecules_ratio.values())
    exact = {n: num_of_molecules * molecules_ratio[n] / total for n in names}
    counts = {n: max(1, int(np.floor(exact[n]))) for n in names}
    # Remove the surplus the minimum of one may have added, smallest
    # shortfall first; then hand out what is left, largest remainder first.
    while sum(counts.values()) > num_of_molecules:
        name = max(
            (n for n in names if counts[n] > 1), key=lambda n: counts[n] - exact[n]
        )
        counts[name] -= 1
    by_remainder = sorted(names, key=lambda n: exact[n] - counts[n], reverse=True)
    for i in range(num_of_molecules - sum(counts.values())):
        counts[by_remainder[i % len(names)]] += 1
    return counts


def create_molecule_atoms_list(name: str, vacuum: float) -> list[Atoms]:
    """The molecule *name* at its equilibrium geometry (SAGE's), centred in
    a cubic cell of its largest extent plus *vacuum*, non-periodic."""
    atoms = build_molecule(name)
    extent = float(np.ptp(atoms.positions, axis=0).max())
    side = extent + vacuum
    atoms.cell = [side] * 3
    atoms.pbc = False
    atoms.center()
    atoms.info["config_type"] = "init_molecule"
    atoms.info["molecule"] = name
    atoms.info["needs_relaxation"] = False
    return [atoms]


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A rotation matrix uniform over SO(3) (from a random unit quaternion)."""
    w, x, y, z = (q := rng.normal(size=4)) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pack_box(
    templates: list[Atoms],
    side: float,
    min_distance: float,
    max_attempts: int,
    rng: np.random.Generator,
    label: str,
    density: float,
) -> Atoms:
    cell = np.eye(3) * side
    box = Atoms(cell=cell, pbc=True)
    for template in templates:
        for _ in range(max_attempts):
            positions = template.positions @ _random_rotation(rng).T
            positions += rng.random(3) * side
            if len(box) == 0:
                break
            _, distances = get_distances(positions, box.positions, cell=cell, pbc=True)
            if distances.min() >= min_distance:
                break
        else:
            raise ValueError(
                f"initialization.molecules_in_box_kwargs: could not place every "
                f"molecule of {label} at {density:g} g/cm^3 ({len(box)} atoms placed "
                f"after {max_attempts} attempts for the next one). Lower the "
                "density or min_distance, or raise max_attempts_per_molecule."
            )
        box += Atoms(template.get_chemical_symbols(), positions=positions)
    box.wrap()
    return box


def _pack_sphere(
    templates: list[Atoms],
    radius: float,
    min_distance: float,
    max_attempts: int,
    rng: np.random.Generator,
    label: str,
    density: float,
) -> Atoms:
    """Place each template, randomly rotated, with its centre uniform in a
    sphere of *radius*, keeping it when no atom comes within
    *min_distance* of an atom already placed (no periodic images)."""
    cluster = Atoms(pbc=False)
    for template in templates:
        centred = template.positions - template.positions.mean(axis=0)
        for _ in range(max_attempts):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            offset = direction * radius * rng.random() ** (1 / 3)
            positions = centred @ _random_rotation(rng).T + offset
            if len(cluster) == 0:
                break
            _, distances = get_distances(positions, cluster.positions)
            if distances.min() >= min_distance:
                break
        else:
            raise ValueError(
                f"initialization.molecule_cluster_kwargs: could not place every "
                f"molecule of {label} at {density:g} g/cm^3 ({len(cluster)} atoms "
                f"placed after {max_attempts} attempts for the next one). Lower "
                "the density or min_distance, or raise max_attempts_per_molecule."
            )
        cluster += Atoms(template.get_chemical_symbols(), positions=positions)
    return cluster


def create_molecule_cluster_atoms_list(
    molecules_ratio: dict[str, float],
    density_range: list[float],
    num_of_molecules: int,
    num_of_structures: int,
    min_distance: float,
    max_attempts: int,
    seed: int,
    vacuum: float = 10.0,
    attempt: int = 0,
) -> list[Atoms]:
    """*num_of_structures* non-periodic clusters of *num_of_molecules*
    molecules in the ratio *molecules_ratio* (config_type
    ``init_molecule_cluster``), for evaluators without periodic boundary
    conditions (ORCA): the non-periodic counterpart of
    create_molecules_in_box_atoms_list. Each cluster's density, drawn
    uniformly from *density_range*, sets the radius of the sphere the
    molecule centres are placed in; the cluster is centred in a cubic cell
    of its extent plus *vacuum*, ``pbc`` False. Cluster i is seeded
    (seed, attempt, i), as the boxes are."""
    counts = molecule_counts(molecules_ratio, num_of_molecules)
    label = box_composition_label(molecules_ratio)
    molecules = {name: build_molecule(name) for name in counts}
    mass = sum(molecules[n].get_masses().sum() * c for n, c in counts.items())
    templates = [molecules[n] for n, c in counts.items() for _ in range(c)]

    clusters = []
    for index in range(num_of_structures):
        rng = np.random.default_rng([seed, attempt, index])
        density = float(rng.uniform(density_range[0], density_range[1]))
        volume = mass / (density * _AMU_PER_A3_TO_G_PER_CM3)
        radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
        order = rng.permutation(len(templates))
        cluster = _pack_sphere(
            [templates[i] for i in order],
            radius,
            min_distance,
            max_attempts,
            rng,
            label,
            density,
        )
        extent = float(np.ptp(cluster.positions, axis=0).max())
        cluster.cell = [extent + vacuum] * 3
        cluster.center()
        cluster.info["config_type"] = "init_molecule_cluster"
        cluster.info["cluster_composition"] = label
        cluster.info["density"] = density
        cluster.info["needs_relaxation"] = True
        clusters.append(cluster)
    logger.debug("Packed %d clusters of %s.", len(clusters), label)
    return clusters


def create_molecules_in_box_atoms_list(
    molecules_ratio: dict[str, float],
    density_range: list[float],
    num_of_molecules: int,
    num_of_structures: int,
    min_distance: float,
    max_attempts: int,
    seed: int,
    attempt: int = 0,
) -> list[Atoms]:
    """*num_of_structures* periodic cubic boxes of *num_of_molecules*
    molecules in the ratio *molecules_ratio* (name -> relative amount), each
    at a density drawn uniformly at random from *density_range* [min, max]
    g/cm^3 and sized so that density is exact.

    *attempt* is how many boxes of this composition the database already
    holds: box i uses its own rng seeded (seed, attempt, i), so a restart
    that replaces boxes lost to DFT failures draws new ones instead of
    repeating earlier ones, while the same database state always gives the
    same boxes."""
    counts = molecule_counts(molecules_ratio, num_of_molecules)
    label = box_composition_label(molecules_ratio)
    molecules = {name: build_molecule(name) for name in counts}
    mass = sum(molecules[n].get_masses().sum() * c for n, c in counts.items())
    largest = max(float(np.ptp(m.positions, axis=0).max()) for m in molecules.values())
    templates = [molecules[n] for n, c in counts.items() for _ in range(c)]

    boxes = []
    for index in range(num_of_structures):
        # One rng per box, so a box doesn't depend on which others are made.
        rng = np.random.default_rng([seed, attempt, index])
        density = float(rng.uniform(density_range[0], density_range[1]))
        side = (mass / (density * _AMU_PER_A3_TO_G_PER_CM3)) ** (1 / 3)
        if largest + min_distance > side:
            raise ValueError(
                f"initialization.molecules_in_box_kwargs: a {side:.2f} Å box for "
                f"{label} at {density:g} g/cm^3 is too small for a molecule "
                f"{largest:.2f} Å across; raise num_of_molecules."
            )
        order = rng.permutation(len(templates))
        box = _pack_box(
            [templates[i] for i in order],
            side,
            min_distance,
            max_attempts,
            rng,
            label,
            float(density),
        )
        box.info["config_type"] = "init_molecules_in_box"
        box.info["box_composition"] = label
        box.info["density"] = float(density)
        box.info["needs_relaxation"] = True
        boxes.append(box)
    logger.debug("Packed %d boxes of %s.", len(boxes), label)
    return boxes
