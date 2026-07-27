import logging
from pathlib import Path
from typing import Any

from yaml import safe_load

from alomancy.configs.global_config import _load_global_hpc_config

logger = logging.getLogger(__name__)

_JOB_SECTIONS = (
    "initialization",
    "mlip_committee",
    "structure_generation",
    "high_accuracy_evaluation",
)


def load_dictionaries(config_path: Path) -> dict[str, Any]:
    """Load a run YAML config and resolve any HPC string references.

    If a job section's ``hpc:`` value is a string, it is looked up in
    ``~/.alomancy/hpc_config.yaml`` and replaced with the full profile dict.
    Dict values are passed through unchanged (backwards compatible).

    Raises
    ------
    ValueError
        If a string HPC name is not found in the global config.
    """
    with open(config_path) as f:
        jobs_dict: dict[str, Any] = safe_load(f)

    hpc_config = _load_global_hpc_config()
    for section in _JOB_SECTIONS:
        if section not in jobs_dict:
            continue
        hpc_ref = jobs_dict[section].get("hpc")
        if isinstance(hpc_ref, str):
            if hpc_ref not in hpc_config:
                raise ValueError(
                    f"HPC profile '{hpc_ref}' referenced in '{section}.hpc' was not "
                    f"found in ~/.alomancy/hpc_config.yaml. "
                    f"Run 'alomancy add-hpc' to add it."
                )
            jobs_dict[section]["hpc"] = hpc_config[hpc_ref]

    return jobs_dict


if __name__ == "__main__":
    logger.info("%s", load_dictionaries(Path("standard_config.yaml")))
