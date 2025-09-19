from ase import Atoms
from ase

def create_sets(
    size_of_test_set: int,
    density_range: tuple[float, float],
    atom_types: list[str],
    atom_range: int,
    singl_atom_marker: str = ""
) -> tuple[list[Atoms], list[Atoms]]:
    """ creates a test and train set of structures for a given density range, 
    atom types and max number of atoms """

    train_set = []
    def generate_single_atom(atom_type: str) -> Atoms:
        return Atoms(atom_type, positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=[1, 1, 1])
    
    for atom_type in atom_types:
        single_atom = generate_single_atom(atom_type)

    

    return train_set, test_set