import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from ase.io import write

# Dimer and trimer separations, as multiples of the sum of the two atoms'
# covalent radii (Pt-Pt: 1.36 to 5.44 Angstrom around the 2.3 Angstrom
# bond). The former 0.2-1.0 x the vdW radii started at 0.70 Angstrom for
# Pt: overlapping PAW spheres, forces above 100 eV/Angstrom and failed SCFs.
DEFAULT_DISTANCE_RANGE = (0.5, 2.0)

# Smallest isolated-atom box edge (Angstrom). 3 x the vdW radius alone gave
# about 5 Angstrom: in a code that is always periodic (QE), the atom then
# interacts with its images.
MIN_ISOLATED_ATOM_BOX = 15.0


def _separation_range(
    elements: list[str], distance_range: tuple[float, float] | list[float] | None
) -> tuple[float, float]:
    lo, hi = distance_range or DEFAULT_DISTANCE_RANGE
    radii = [covalent_radii[atomic_numbers[e]] for e in elements]
    # Pairs: the two largest radii bound the separations of a trimer too.
    pair = sum(sorted(radii)[-2:]) if len(radii) > 1 else 2 * radii[0]
    return float(lo) * pair, float(hi) * pair


def create_single_atoms_list(element: str) -> list:
    """Create a single atoms object in a list."""
    edge = max(vdw_radii[atomic_numbers[element]] * 3, MIN_ISOLATED_ATOM_BOX)
    cell = [edge] * 3
    atom = Atoms(element, positions=[(0, 0, 0)])
    atom.cell = cell
    atom.info["config_type"] = "IsolatedAtom"
    atom.info["needs_relaxation"] = False
    return [atom]


def create_dimer_atoms_list(
    element_a: str,
    element_b: str,
    num_dimers: int,
    distance_range: tuple[float, float] | list[float] | None = None,
) -> list:
    """Create a list of atoms objects of two atoms at various distances:
    *distance_range* (default DEFAULT_DISTANCE_RANGE) times the sum of
    their covalent radii."""
    dimer_atoms = []
    distance_range = _separation_range([element_a, element_b], distance_range)
    cell = [distance_range[1] * 3] * 3  # Ensure enough space
    if num_dimers > 0:
        for d in np.linspace(distance_range[0], distance_range[1], num_dimers):
            a1 = Atoms(element_a, positions=[(0, 0, 0)])
            a2 = Atoms(element_b, positions=[(0, 0, d)])
            dimer = a1 + a2
            dimer.cell = cell
            dimer.info["config_type"] = "init_dimer"
            dimer.info["distance"] = f"{d:.3f}"
            dimer.info["needs_relaxation"] = False
            dimer_atoms.append(dimer)
    return dimer_atoms


def create_trimer_atoms_list(
    element_a: str,
    element_b: str,
    element_c: str,
    num_trimers: int,
    distance_range: tuple[float, float] | list[float] | None = None,
) -> list:
    """Create a list of atoms objects of three atoms at various distances:
    *distance_range* (default DEFAULT_DISTANCE_RANGE) times the largest
    covalent-radius pair sum among them."""
    trimer_atoms = []
    distance_range = _separation_range(
        [element_a, element_b, element_c], distance_range
    )
    cell = [distance_range[1] * 3] * 3  # Ensure enough space
    if num_trimers > 1:
        for dx in np.linspace(
            distance_range[0],
            distance_range[1],
            int(np.floor(np.sqrt(num_trimers - 2))),
        ):
            for dy in np.linspace(
                distance_range[0],
                distance_range[1],
                int(np.floor(np.sqrt(num_trimers - 2))),
            ):
                a1 = Atoms(element_a, positions=[(0, 0, 0)])
                a2 = Atoms(element_b, positions=[(0, dy, 2 * dx)])
                a3 = Atoms(element_c, positions=[(0, 2 * dy, dx)])
                trimer = a1 + a2 + a3
                trimer.cell = cell
                trimer.info["config_type"] = "init_trimer"
                trimer.info["deformation"] = f"{dx:.3f}_{dy:.3f}"
                trimer.info["needs_relaxation"] = False
                trimer_atoms.append(trimer)

        # add some linear trimers if we don't have enough
        for dx in np.linspace(
            distance_range[0], distance_range[1], num_trimers - len(trimer_atoms)
        ):
            a1 = Atoms(element_a, positions=[(0, 0, 0)])
            a2 = Atoms(element_b, positions=[(0, 0, dx)])
            a3 = Atoms(element_c, positions=[(0, 0, 2 * dx)])
            trimer = a1 + a2 + a3
            trimer.cell = cell
            trimer.info["config_type"] = "init_trimer"
            trimer.info["deformation"] = f"{dx:.3f}_0.000"
            trimer.info["needs_relaxation"] = False
            trimer_atoms.append(trimer)

    return trimer_atoms


if __name__ == "__main__":
    dimers = create_dimer_atoms_list("H", "O", 50)
    write("dimers.xyz", dimers)
    trimers = create_trimer_atoms_list("H", "O", "H", 50)
    write("trimers.xyz", trimers)
