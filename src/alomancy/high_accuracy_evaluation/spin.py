"""Spin polarisation of high-accuracy labels: ``high_accuracy_evaluation.
spin_polarisation``, one setting for every evaluator.

- ``"essential"`` (default): isolated atoms, dimers and trimers
  (``ESSENTIAL_CONFIG_TYPES``) are spin-polarised; every other structure is
  left as before (spin-polarised only when it carries initial magnetic
  moments, e.g. a magnetic Materials Project cell).
- ``"all"``: every structure is spin-polarised.
- ``"none"``: no structure is (initial moments are cleared).

Why the essentials: a free atom is open-shell (carbon is a triplet), so a
spin-restricted isolated atom sits ~1 eV above its ground state. That error
lands in the E0s and from there in every structure's energy per atom; the
dimers and trimers that dissociate towards it need the same treatment to
reach 2 x E0 and 3 x E0. MACE's fine-tuning guidance (Tompa et al. 2026,
arXiv:2606.12704) asks for spin-polarised isolated-atom E0s.

An isolated atom's moment is fixed to its Hund's-rule ground state
(``hund_moment``); dimers and trimers start ferromagnetic and relax freely.
How each code is told: run_qe, run_vasp, run_aims, run_orca.
"""

from typing import Any

import numpy as np
from ase import Atoms
from ase.data import ground_state_magnetic_moments

SPIN_POLARISATION_KEY = "spin_polarisation"
SPIN_POLARISATION_MODES = ("all", "essential", "none")
DEFAULT_SPIN_POLARISATION = "essential"
ESSENTIAL_CONFIG_TYPES = ("IsolatedAtom", "init_dimer", "init_trimer")


def resolve_spin_polarisation(config: dict | None) -> str:
    """high_accuracy_evaluation.spin_polarisation, defaulted and checked."""
    mode = (config or {}).get(SPIN_POLARISATION_KEY, DEFAULT_SPIN_POLARISATION)
    if mode not in SPIN_POLARISATION_MODES:
        raise ValueError(
            f"high_accuracy_evaluation.{SPIN_POLARISATION_KEY} must be one of "
            f"{list(SPIN_POLARISATION_MODES)}, got {mode!r}."
        )
    return str(mode)


def spin_polarised(atoms: Atoms, mode: str) -> bool | None:
    """True: spin-polarise *atoms*; False: don't; None ("essential", a
    structure outside ESSENTIAL_CONFIG_TYPES): as its initial moments say."""
    if mode == "all":
        return True
    if mode == "none":
        return False
    if atoms.info.get("config_type") in ESSENTIAL_CONFIG_TYPES:
        return True
    return None


def is_isolated_atom(atoms: Atoms) -> bool:
    return len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"


def hund_moment(atoms: Atoms) -> float | None:
    """The ground-state (Hund's rule) moment, in Bohr magnetons, of a
    single neutral atom (C 2, H 1, O 2, N 3); None for anything else."""
    if len(atoms) != 1 or atoms.info.get("charge", 0) != 0:
        return None
    return float(ground_state_magnetic_moments[atoms.numbers[0]])


def starting_moments(atoms: Atoms) -> np.ndarray:
    """Initial moments (Bohr magnetons) for a spin-polarised calculation:
    the structure's own if it has any, else each atom's Hund's-rule moment
    (a ferromagnetic start the calculation relaxes from)."""
    own = atoms.get_initial_magnetic_moments()
    if np.any(own):
        return np.asarray(own, dtype=float)
    return np.array(
        [ground_state_magnetic_moments[z] for z in atoms.numbers], dtype=float
    )


def spin_treatment_label(mode: str) -> str:
    """LevelOfTheory.spin_treatment for a mode (an identity field, so
    labels made under different modes never share an id)."""
    return {"none": "restricted", "all": "collinear", "essential": "essential"}[mode]


def reject_owned_keys(
    settings: dict[str, Any] | None, owned: tuple[str, ...], where: str
) -> None:
    """Raise if an evaluator's own settings set a spin keyword that
    spin_polarisation now decides (case-insensitive; QE's indexed
    starting_magnetization(i) counts as starting_magnetization)."""
    for key in settings or {}:
        base = str(key).lower().split("(")[0]
        if base in owned:
            raise ValueError(
                f"{where}.{key} is set by high_accuracy_evaluation."
                f"{SPIN_POLARISATION_KEY} ({'/'.join(SPIN_POLARISATION_MODES)}, "
                f"default {DEFAULT_SPIN_POLARISATION!r}); remove it."
            )
