"""Remote worker: predict energies/forces of structures with one trained model.

Submitted by ``ActiveLearningWorkflow.predict`` once per model, all models
in one parallel batch. Must stay a module-level function: ExPyRe pickles
functions by reference and re-imports this module on the remote node, so
moving or renaming it needs a reinstall on every HPC host.
"""

import numpy as np
from ase import Atoms

from alomancy.mlip.base import get_trainer


def _flatten_array_of_forces(forces: np.ndarray) -> np.ndarray:
    return np.reshape(forces, (1, forces.shape[0] * 3))


def predict_with_model(
    structure_list: list[Atoms],
    model_path: str,
    trainer: str,
    trainer_config: dict,
) -> dict:
    """Evaluate every structure with ONE model, building its calculator via
    the trainer registry (never a hardcoded MACECalculator). Returns
    ``{"forces": [...], "energies": [...]}``, index-aligned with
    structure_list; forces are flattened to shape (1, 3 * n_atoms)."""
    calc = get_trainer(trainer, trainer_config).get_calculator(model_path)
    forces = []
    energies = []
    for atoms in structure_list:
        atoms.calc = calc
        forces.append(_flatten_array_of_forces(atoms.get_forces()))
        energies.append(np.array(atoms.get_potential_energy()))
    return {"forces": forces, "energies": energies}
