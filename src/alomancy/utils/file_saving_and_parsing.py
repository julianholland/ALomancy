from pathlib import Path

from ase import Atoms
from ase.io import read


def read_xyz(path: str | Path) -> list[Atoms]:
    """Every structure in an extxyz file ALomancy wrote; an empty file (as
    ase.io.write leaves for an empty list) holds no structures. ASE's own
    read raises on it."""
    path = Path(path)
    if path.stat().st_size == 0:
        return []
    return list(read(path, ":", format="extxyz"))


def read_atoms_file_if_enabled(
    read_file: bool, file_path: str | Path
) -> list[Atoms] | None:
    """Return file contents only when reading is enabled and the file exists."""
    if not read_file:
        return None

    path = Path(file_path)
    if not path.exists():
        return None

    if path.stat().st_size == 0:
        return []

    atoms_list = [atoms for atoms in read(path, ":") if isinstance(atoms, Atoms)]

    return atoms_list
