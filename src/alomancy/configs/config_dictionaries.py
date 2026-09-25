import logging
from pathlib import Path
from typing import Any

from yaml import safe_load

from alomancy.configs.global_config import _load_global_hpc_config

logger = logging.getLogger(__name__)

_JOB_SECTIONS = (
    "initialization",
    "mlip_committee",
    "training",
    "structure_generation",
    "high_accuracy_evaluation",
)

_DEFAULT_MAX_TIME = "24:00:00"


def load_dictionaries(config_path: Path) -> dict[str, Any]:
    """Load a run YAML config and resolve any HPC string references.

    If a job section's ``hpc:`` value is a string, it is looked up in
    ``~/.alomancy/hpc_config.yaml`` and replaced with the full profile dict.
    Dict values are passed through unchanged (backwards compatible).

    If a section's ``max_time`` is missing (or falsy), it defaults to that
    section's resolved HPC profile's own ``default_max_time`` if the profile
    defines one, else the hardcoded fallback ``"24:00:00"``. A section with
    no ``hpc`` at all (e.g. ``initialization``, which runs locally) gets no
    ``max_time`` default either -- there is nothing for it to bound.

    An empty section (``"initialization:"`` with nothing indented under it)
    is normalized to ``{}`` rather than left as YAML's ``None`` -- every
    top-level key in the loaded YAML, not just the four processed here, so
    e.g. an empty ``workflow:`` section doesn't crash downstream either.

    Raises
    ------
    ValueError
        If a string HPC name is not found in the global config.
    """
    with open(config_path) as f:
        jobs_dict: dict[str, Any] = safe_load(f)

    # An empty section (e.g. "initialization:" with nothing indented under
    # it -- entirely valid now that every section's own settings are
    # optional/defaulted) parses from YAML as None, not {}. Every section
    # reader downstream (including this function's own section_dict.get
    # below) assumes a dict -- normalize once here rather than requiring
    # every section, in every module, to guard against None individually.
    for key, value in jobs_dict.items():
        if value is None:
            jobs_dict[key] = {}

    hpc_config = _load_global_hpc_config()
    for section in _JOB_SECTIONS:
        if section not in jobs_dict:
            continue
        section_dict = jobs_dict[section]
        hpc_ref = section_dict.get("hpc")
        if isinstance(hpc_ref, str):
            if hpc_ref not in hpc_config:
                raise ValueError(
                    f"HPC profile '{hpc_ref}' referenced in '{section}.hpc' was not "
                    f"found in ~/.alomancy/hpc_config.yaml. "
                    f"Run 'alomancy add-hpc' to add it."
                )
            section_dict["hpc"] = hpc_config[hpc_ref].copy()

        resolved_hpc = section_dict.get("hpc")
        if isinstance(resolved_hpc, dict) and not section_dict.get("max_time"):
            section_dict["max_time"] = resolved_hpc.get(
                "default_max_time", _DEFAULT_MAX_TIME
            )

    return jobs_dict


if __name__ == "__main__":
    logger.info("%s", load_dictionaries(Path("standard_config.yaml")))
