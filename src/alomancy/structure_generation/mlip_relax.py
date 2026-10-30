"""Relax the generated candidates with the loop's best model before they are
scored and selected (``structure_generation.mlip_relaxation``).

``relax_with_model`` is a remote worker submitted by
``ActiveLearningWorkflow._mlip_relax`` (from ``generate_candidates``). Must stay a module-level
function: ExPyRe pickles functions by reference and re-imports this module
on the remote node, so moving or renaming it needs a reinstall on every
HPC host (``alomancy upgrade-hpc``).
"""

import logging
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS

from alomancy.mlip.base import get_trainer

logger = logging.getLogger(__name__)

OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}
MLIP_RELAXATION_DEFAULTS: dict[str, Any] = {
    "fmax": 0.05,
    "steps": 200,
    "optimizer": "FIRE",
}


def resolve_mlip_relaxation(cfg: Any) -> dict[str, Any] | None:
    """``structure_generation.mlip_relaxation`` with defaults filled in, or
    None when it is absent/null (off). Raises ValueError for a malformed
    block, at construction rather than after a loop's selection."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(
            "structure_generation.mlip_relaxation must be a mapping "
            "(e.g. {fmax: 0.05, steps: 200, optimizer: FIRE}) or null."
        )
    unknown = sorted(set(cfg) - set(MLIP_RELAXATION_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown structure_generation.mlip_relaxation key(s) {unknown}; "
            f"allowed: {sorted(MLIP_RELAXATION_DEFAULTS)}."
        )
    resolved = {**MLIP_RELAXATION_DEFAULTS, **cfg}
    fmax, steps = resolved["fmax"], resolved["steps"]
    if (
        isinstance(fmax, bool)
        or not isinstance(fmax, int | float)
        or not np.isfinite(fmax)
        or fmax <= 0
    ):
        raise ValueError(
            f"structure_generation.mlip_relaxation.fmax must be a positive "
            f"number (eV/Å), got {fmax!r}."
        )
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(
            f"structure_generation.mlip_relaxation.steps must be a positive "
            f"integer, got {steps!r}."
        )
    if resolved["optimizer"] not in OPTIMIZERS:
        raise ValueError(
            f"structure_generation.mlip_relaxation.optimizer must be one of "
            f"{sorted(OPTIMIZERS)}, got {resolved['optimizer']!r}."
        )
    return resolved


def relax_with_model(
    structure_list: list[Atoms],
    model_path: str,
    trainer: str,
    trainer_config: dict,
    fmax: float,
    steps: int,
    optimizer: str,
) -> list[Atoms | None]:
    """Relax each structure with ONE model (calculator built through the
    trainer registry). Returns relaxed copies, index-aligned with
    structure_list and tagged ``mlip_relaxed``/``mlip_relax_steps``/
    ``mlip_relax_converged``; None where the relaxation raised or left
    non-finite positions, so the driver can fall back to the original."""
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}.")
    calc = get_trainer(trainer, trainer_config).get_calculator(model_path)
    relaxed: list[Atoms | None] = []
    for i, atoms in enumerate(structure_list):
        candidate = atoms.copy()
        candidate.calc = calc
        opt = OPTIMIZERS[optimizer](candidate, logfile=None)
        try:
            converged = bool(opt.run(fmax=fmax, steps=steps))
        except Exception as exc:  # one bad structure must not lose the rest
            logger.warning("MLIP relaxation of structure %d failed: %s", i, exc)
            relaxed.append(None)
            continue
        # Never ship the model back with the result.
        candidate.calc = None
        if not np.isfinite(candidate.positions).all():
            relaxed.append(None)
            continue
        candidate.info["mlip_relaxed"] = True
        candidate.info["mlip_relax_steps"] = int(opt.nsteps)
        candidate.info["mlip_relax_converged"] = converged
        relaxed.append(candidate)
    return relaxed
