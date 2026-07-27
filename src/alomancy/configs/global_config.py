from pathlib import Path

from yaml import safe_load

ALOMANCY_DIR = Path.home() / ".alomancy"
ALOMANCY_HPC_CONFIG = ALOMANCY_DIR / "hpc_config.yaml"
EXPYRE_CONFIG = Path.home() / ".expyre" / "config.json"


def _load_global_hpc_config() -> dict:
    """Load ~/.alomancy/hpc_config.yaml, returning {} if the file does not exist."""
    if not ALOMANCY_HPC_CONFIG.exists():
        return {}
    with open(ALOMANCY_HPC_CONFIG) as f:
        return safe_load(f) or {}
