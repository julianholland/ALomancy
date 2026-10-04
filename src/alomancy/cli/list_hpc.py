"""``alomancy list-hpc``: summarize the HPC profiles in ~/.alomancy/hpc_config.yaml."""

from alomancy.configs.global_config import (
    ALOMANCY_HPC_CONFIG,
    _load_global_hpc_config,
)
from alomancy.configs.hpc_profiles import format_table, hpc_profile_row


def list_hpc(check_remote: bool = False) -> str:
    """Return the HPC profile table (or a pointer to ``alomancy add-hpc``
    when none are configured). ``check_remote`` adds each host's installed
    alomancy version via ssh."""
    profiles = _load_global_hpc_config()
    if not profiles:
        return (
            f"No HPC profiles found in {ALOMANCY_HPC_CONFIG}. "
            "Run 'alomancy add-hpc' to add one."
        )
    rows = [
        hpc_profile_row(name, profile, check_remote=check_remote)
        for name, profile in profiles.items()
    ]
    return f"{len(rows)} HPC profile(s) in {ALOMANCY_HPC_CONFIG}:\n{format_table(rows)}"
