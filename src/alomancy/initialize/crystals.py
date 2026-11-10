"""Common single-element crystal structures (initialization block
``crystal_structures_kwargs``, config_type ``init_crystal``).

Every chosen element is built in each of CRYSTAL_STRUCTURES as a
primitive cell, so the model sees the competing phases of an element
(bcc and hcp Pt, open structures), not only the ground state Materials
Project returns.

Lattice constants (first match wins, per element and structure):
1. ``lattice_constants[element][structure]``: a (or ``{a, c}`` for hcp and
   omega), in Angstrom;
2. ``volume_per_atom[element]``: Angstrom^3 per atom, for every structure
   of that element without its own lattice constant;
3. ASE's reference crystal (``ase.data.reference_states``): its volume per
   atom. The element's own reference structure therefore comes out at the
   reference lattice constant (fcc Pt: 3.92 Angstrom); every other
   structure is scaled to the same volume per atom (bcc Pt: 3.11 Angstrom).

hcp uses the element's reference c/a when its reference crystal is hcp,
else the ideal sqrt(8/3); omega uses c/a 0.62 (Ti, Zr omega).
"""

import logging
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_numbers, reference_states

logger = logging.getLogger(__name__)

CONFIG_TYPE = "init_crystal"

CRYSTAL_STRUCTURES = ("fcc", "bcc", "hcp", "sc", "diamond", "a15", "omega")

# Metallic elements (the default element list: the metals among
# general.elements). Alkali, alkaline-earth, transition, lanthanide and
# actinide metals, and the post-transition metals Al, Ga, In, Sn, Tl, Pb,
# Bi, Po. Metalloids (B, Si, Ge, As, Sb, Te) are not included.
METALS = frozenset(
    [
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "Fr",
        "Be",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Ra",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Al",
        "Ga",
        "In",
        "Sn",
        "Tl",
        "Pb",
        "Bi",
        "Po",
    ]
)

IDEAL_HCP_C_OVER_A = float(np.sqrt(8.0 / 3.0))
OMEGA_C_OVER_A = 0.62

# Reference crystals ase.build.bulk can't build (no atomic basis): atoms in
# the reference cell, to get its volume per atom from a, b/a, c/a.
_REFERENCE_ATOMS_PER_CELL = {"Mn": 58, "Ga": 8, "Pa": 2}

# A15 (beta-W): 2a and 6c sites of the cubic cell, fractional.
_A15_POSITIONS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.25, 0.0, 0.5),
    (0.75, 0.0, 0.5),
    (0.5, 0.25, 0.0),
    (0.5, 0.75, 0.0),
    (0.0, 0.5, 0.25),
    (0.0, 0.5, 0.75),
]
# omega: hexagonal, 3 atoms, fractional.
_OMEGA_POSITIONS = [(0.0, 0.0, 0.0), (1 / 3, 2 / 3, 0.5), (2 / 3, 1 / 3, 0.5)]


def reference_volume_per_atom(element: str) -> float | None:
    """The volume per atom (Angstrom^3) of *element*'s ASE reference crystal,
    or None when ASE has no usable reference data (Pm, Fr, Ra, Pu, Am...)."""
    try:
        atoms = bulk(element)
        return float(atoms.get_volume() / len(atoms))
    except Exception:  # ASE raises various errors for missing data
        pass
    ref = reference_states[atomic_numbers[element]] or {}
    n = _REFERENCE_ATOMS_PER_CELL.get(element)
    if n is None or "a" not in ref:
        return None
    a = float(ref["a"])
    volume = a**3 * float(ref.get("b/a", 1.0)) * float(ref.get("c/a", 1.0))
    return volume / n


def _hcp_c_over_a(element: str) -> float:
    ref = reference_states[atomic_numbers[element]] or {}
    if ref.get("symmetry") == "hcp" and "c/a" in ref:
        return float(ref["c/a"])
    return IDEAL_HCP_C_OVER_A


def _unit_cell(element: str, structure: str, a: float, c: float | None) -> Atoms:
    """*structure* of *element* with lattice constant *a* (and *c* for the
    hexagonal ones), primitive."""
    if structure in ("fcc", "bcc", "sc", "diamond"):
        return bulk(element, structure, a=a)
    if structure == "hcp":
        return bulk(element, "hcp", a=a, c=c)
    if structure == "a15":
        return Atoms(
            [element] * 8,
            scaled_positions=_A15_POSITIONS,
            cell=[a, a, a],
            pbc=True,
        )
    if structure == "omega":
        assert c is not None
        cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
        return Atoms(
            [element] * 3, scaled_positions=_OMEGA_POSITIONS, cell=cell, pbc=True
        )
    raise ValueError(
        f"Unknown crystal structure {structure!r}; expected one of "
        f"{list(CRYSTAL_STRUCTURES)}."
    )


def _c_over_a(element: str, structure: str) -> float | None:
    if structure == "hcp":
        return _hcp_c_over_a(element)
    if structure == "omega":
        return OMEGA_C_OVER_A
    return None


def _user_lattice(value: Any) -> tuple[float, float | None]:
    """(a, c) from a user lattice value: a number, or {a, c} for hcp/omega
    (c defaults to the structure's c/a times a)."""
    if isinstance(value, dict):
        a = value.get("a")
        c = value.get("c")
    else:
        a, c = value, None
    return float(a), (None if c is None else float(c))


def build_crystal(
    element: str,
    structure: str,
    lattice: Any = None,
    volume_per_atom: float | None = None,
) -> Atoms:
    """*structure* of *element* as a primitive cell (see the module
    docstring for where its size comes from). info: config_type
    init_crystal, crystal_structure, lattice_source (user, equal_volume
    or reference), needs_relaxation False (always a single point)."""
    c_over_a = _c_over_a(element, structure)
    if lattice is not None:
        a, c = _user_lattice(lattice)
        if c is None and c_over_a is not None:
            c = a * c_over_a
        atoms = _unit_cell(element, structure, a, c)
        source = "user"
    else:
        target = volume_per_atom
        source = "equal_volume"
        if target is None:
            target = reference_volume_per_atom(element)
            if target is None:
                raise ValueError(
                    f"ASE has no reference crystal for {element}: set "
                    "initialization.crystal_structures_kwargs.volume_per_atom"
                    f".{element} or lattice_constants.{element}."
                )
            ref = reference_states[atomic_numbers[element]] or {}
            if ref.get("symmetry") == structure:
                source = "reference"
        else:
            source = "user"
        unit = _unit_cell(
            element, structure, 1.0, None if c_over_a is None else c_over_a
        )
        scale = (target * len(unit) / unit.get_volume()) ** (1.0 / 3.0)
        atoms = _unit_cell(
            element,
            structure,
            scale,
            None if c_over_a is None else scale * c_over_a,
        )
    atoms.info["config_type"] = CONFIG_TYPE
    atoms.info["crystal_structure"] = structure
    atoms.info["lattice_source"] = source
    atoms.info["needs_relaxation"] = False
    return atoms


def default_elements(elements: list[str]) -> list[str]:
    """The metals among *elements* (METALS), in their order."""
    return [e for e in elements if e in METALS]


def lattice_for(
    element: str, structure: str, lattice_constants: dict, volume_per_atom: dict
) -> tuple[Any, float | None]:
    """(user lattice or None, user volume per atom or None) for one element
    and structure, by the documented precedence."""
    per_element = (lattice_constants or {}).get(element) or {}
    return per_element.get(structure), (volume_per_atom or {}).get(element)


def create_crystal_atoms_list(
    elements: list[str],
    structures: list[str],
    lattice_constants: dict | None = None,
    volume_per_atom: dict | None = None,
    override: dict[str, list[str]] | None = None,
) -> list[Atoms]:
    """Every structure of every element; with *override* ({element:
    [structures]}), only those (the ones still missing on a restart)."""
    built = []
    for element in elements:
        wanted = override.get(element, []) if override is not None else structures
        for structure in wanted:
            lattice, volume = lattice_for(
                element, structure, lattice_constants or {}, volume_per_atom or {}
            )
            built.append(build_crystal(element, structure, lattice, volume))
    return built


def describe_lattices(
    elements: list[str],
    structures: list[str],
    lattice_constants: dict | None = None,
    volume_per_atom: dict | None = None,
) -> list[str]:
    """One line per element: each structure's a (Angstrom) and where it came
    from, for the start-up log."""
    lines = []
    for element in elements:
        parts = []
        for structure in structures:
            lattice, volume = lattice_for(
                element, structure, lattice_constants or {}, volume_per_atom or {}
            )
            atoms = build_crystal(element, structure, lattice, volume)
            a = _conventional_a(atoms, structure)
            v = atoms.get_volume() / len(atoms)
            parts.append(
                f"{structure} a={a:.3f} A ({atoms.info['lattice_source']}, "
                f"{v:.2f} A^3/atom)"
            )
        lines.append(f"{element}: " + ", ".join(parts))
    return lines


def _conventional_a(atoms: Atoms, structure: str) -> float:
    """The conventional lattice constant of a primitive cell built here."""
    length = float(np.linalg.norm(atoms.cell[0]))
    if structure in ("fcc", "diamond"):
        return length * np.sqrt(2.0)
    if structure == "bcc":
        return length * 2.0 / np.sqrt(3.0)
    return length
