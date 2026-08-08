from pathlib import Path

from yaml import safe_load

ALOMANCY_DIR = Path.home() / ".alomancy"
ALOMANCY_HPC_CONFIG = ALOMANCY_DIR / "hpc_config.yaml"

# Canonical, alomancy-managed master copy of ExPyRe's systems config.
# `alomancy add-hpc` writes here (write_expyre_config in cli/add_hpc.py);
# alomancy/__init__.py's _ensure_local_expyre_root copies this file into
# every new run's own <rundir>/.expyre/config.json at import time, so each
# run gets an isolated jobs.db/job-stage-dir tree (see that module's
# docstring) instead of every alomancy process on this $HOME sharing one
# "~/.expyre/config.json" and one jobs.db. Kept under ~/.alomancy rather
# than ~/.expyre since ~/.expyre is no longer meant to hold live shared
# config -- only a per-run job-state directory (or, for installs that
# predate this split, the LEGACY_EXPYRE_CONFIG fallback below).
EXPYRE_CONFIG = ALOMANCY_DIR / "expyre_config.json"

# Pre-isolation location ExPyRe itself still reads by default (its own
# config.py walks up to "~/.expyre" looking for "config.json"). Used as a
# one-time migration source by write_expyre_config the first time it's
# asked to write EXPYRE_CONFIG and finds it doesn't exist yet, and as a
# fallback copy source by _ensure_local_expyre_root for installs that
# haven't re-run `alomancy add-hpc` since upgrading.
LEGACY_EXPYRE_CONFIG = Path.home() / ".expyre" / "config.json"


def _load_global_hpc_config() -> dict:
    """Load ~/.alomancy/hpc_config.yaml, returning {} if the file does not exist."""
    if not ALOMANCY_HPC_CONFIG.exists():
        return {}
    with open(ALOMANCY_HPC_CONFIG) as f:
        data = safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{ALOMANCY_HPC_CONFIG} must contain a YAML mapping, "
            f"but got {type(data).__name__}. Check the file for formatting errors."
        )
    return data
