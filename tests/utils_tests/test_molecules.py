"""utils/molecules.py: molecules named by formula or SMILES, built by SAGE."""

import numpy as np
import pytest

from alomancy.utils.molecules import (
    build_molecule,
    molecule_symbols,
    validate_molecules,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, formula",
    [
        ("CO", "CO"),
        ("H2O", "H2O"),
        ("smiles-O", "H2O"),
        ("smiles-CCO", "C2H6O"),
        ("smiles-CO", "CH4O"),
    ],
)
def test_formula_and_smiles_names_build(name, formula):
    assert build_molecule(name).get_chemical_formula() == formula


@pytest.mark.unit
def test_built_molecule_is_centred_and_cell_free():
    atoms = build_molecule("smiles-CCO")
    assert np.allclose(atoms.get_center_of_mass(), 0.0, atol=1e-8)
    assert not atoms.pbc.any()
    assert atoms.cell.rank == 0


@pytest.mark.unit
def test_each_call_returns_an_independent_copy():
    first = build_molecule("CO")
    first.positions += 5.0
    assert np.allclose(build_molecule("CO").get_center_of_mass(), 0.0, atol=1e-8)


@pytest.mark.unit
def test_lower_case_formula_is_a_single_atom_and_raises_with_the_hint():
    with pytest.raises(ValueError, match="'co' is cobalt"):
        build_molecule("co")


@pytest.mark.unit
def test_unknown_name_raises_pointing_to_smiles():
    with pytest.raises(ValueError, match="smiles-C"):
        build_molecule("CH4")


@pytest.mark.unit
def test_molecule_symbols_counts_hydrogens_from_smiles():
    assert molecule_symbols("smiles-CCO") == {"C": 2, "H": 6, "O": 1}


@pytest.mark.unit
def test_validate_accepts_molecules_of_general_elements():
    names = validate_molecules(["CO", "smiles-O"], ["C", "H", "O"], "where")
    assert names == ["CO", "smiles-O"]


@pytest.mark.unit
def test_validate_rejects_an_element_outside_general_elements():
    with pytest.raises(ValueError, match=r"where: 'smiles-CCO' contains \['C'\]"):
        validate_molecules(["smiles-CCO"], ["H", "O"], "where")


@pytest.mark.unit
def test_validate_rejects_a_non_list():
    with pytest.raises(ValueError, match="must be a list of molecule names"):
        validate_molecules("CO", ["C", "O"], "where")


_AROMATIC_ELEMENTS = ["C", "H", "B", "O", "Br"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, formula",
    [("smiles-OB(O)c1ccccc1", "C6H7BO2"), ("smiles-Brc1ccccc1", "C6H5Br")],
    ids=["phenylboronic_acid", "bromobenzene"],
)
def test_aromatic_smiles_build_one_sound_molecule(name, formula):
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components

    assert validate_molecules([name], _AROMATIC_ELEMENTS, "where") == [name]
    atoms = build_molecule(name)
    assert atoms.get_chemical_formula() == formula
    neighbours = NeighborList(
        natural_cutoffs(atoms, mult=1.2), self_interaction=False, bothways=True
    )
    neighbours.update(atoms)
    n_fragments, _ = connected_components(neighbours.get_connectivity_matrix())
    assert n_fragments == 1
    distances = atoms.get_all_distances()
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 0.9
