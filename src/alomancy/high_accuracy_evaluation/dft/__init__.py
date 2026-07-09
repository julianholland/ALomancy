import importlib
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_CALCULATOR_REGISTRY: dict[str, dict] = {
    "qe": {
        "module": "alomancy.high_accuracy_evaluation.dft.run_qe",
        "sp": "run_sp_qe",
        "go": "run_go_qe",
        "keys": ["qe_input_kwargs", "pwx_path", "pp_path"],
    },
    "vasp": {
        "module": "alomancy.high_accuracy_evaluation.dft.run_vasp",
        "sp": "run_sp_vasp",
        "go": "run_go_vasp",
        "keys": ["vasp_input_kwargs", "vasp_path"],
    },
}


def get_dft_functions(calculator: str) -> tuple[Callable, Callable]:
    """Return (run_sp_fn, run_go_fn) for the requested calculator.

    Lazy imports keep optional heavy dependencies from breaking installs
    that only have one calculator available.
    """
    if calculator not in _CALCULATOR_REGISTRY:
        raise ValueError(
            f"Unknown calculator {calculator!r}. Available: {list(_CALCULATOR_REGISTRY)}"
        )
    entry = _CALCULATOR_REGISTRY[calculator]
    mod = importlib.import_module(entry["module"])
    return getattr(mod, entry["sp"]), getattr(mod, entry["go"])


def warn_mismatched_kwargs(calculator: str, job_dict: dict) -> None:
    """Warn when kwargs that belong to a different calculator are present in the config."""
    hpc = job_dict.get("hpc", {})
    for other_calc, entry in _CALCULATOR_REGISTRY.items():
        if other_calc == calculator:
            continue
        for key in entry["keys"]:
            if key in job_dict or key in hpc:
                logger.warning(
                    "Config key %r looks like a %s setting but calculator=%r — it will be ignored.",
                    key,
                    other_calc,
                    calculator,
                )
