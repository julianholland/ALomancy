"""Molecules named by formula or SMILES, built with SAGE.

A name is anything SAGE's ``AtomPosition.build`` accepts: a formula from
its compound tables (``"CO"``, ``"H2O"``) or ``"smiles-<SMILES>"``
(``"smiles-CCO"``, built by SAGE's own SMILES parser, hydrogens added and
the geometry relaxed). EZGA builds the species of its add mutations the
same way, so a name that builds here is one EZGA can add.
"""

from collections import Counter
from functools import cache
from typing import Any

from ase import Atoms


@cache
def _build(name: str) -> Atoms:
    from sage_lib.partition.Partition import Partition

    partition = Partition()
    partition.add_empty()
    manager = partition[0].AtomPositionManager
    try:
        manager.build(name)
    except Exception as exc:
        # SAGE's own message is its whole builder help text; keep the
        # first line only.
        first_line = next((ln for ln in str(exc).splitlines() if ln.strip()), "")
        raise ValueError(
            f"SAGE cannot build molecule {name!r} ({first_line.strip()}). Name a "
            "molecule by SMILES, e.g. 'smiles-C' for CH4, or by one of SAGE's "
            "preset formulas ('CO', 'H2O', ...)."
        ) from exc
    symbols = [str(label) for label in manager.atomLabelsList]
    if len(symbols) < 2:
        raise ValueError(
            f"Molecule {name!r} builds {symbols}, not a molecule (names are "
            "case-sensitive: 'co' is cobalt; write SMILES as 'smiles-CCO')."
        )
    atoms = Atoms(symbols=symbols, positions=manager.atomPositions)
    atoms.positions -= atoms.get_center_of_mass()
    return atoms


def build_molecule(name: str) -> Atoms:
    """The molecule *name* (formula or ``smiles-...``) as an ASE Atoms
    centred on its centre of mass, no cell. A fresh copy on every call."""
    atoms: Atoms = _build(name).copy()
    return atoms


def molecule_symbols(name: str) -> Counter:
    """Element -> count in one molecule *name*."""
    return Counter(_build(name).get_chemical_symbols())


def validate_molecules(names: Any, elements: list[str], where: str) -> list[str]:
    """Check *names* is a list of molecule names that SAGE builds, every
    element of which is in general.elements; return it. *where* is the
    config key, for the error message."""
    if not isinstance(names, list | tuple) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError(
            f"{where} must be a list of molecule names, e.g. ['CO', 'smiles-CCO']; "
            f"got {names!r}."
        )
    allowed = set(elements)
    for name in names:
        try:
            symbols = molecule_symbols(name)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        missing = sorted(set(symbols) - allowed)
        if missing:
            raise ValueError(
                f"{where}: {name!r} contains {missing}, which are not in "
                f"general.elements {sorted(allowed)}."
            )
    return list(names)
