"""Tabular summaries of HPC profiles from ``~/.alomancy/hpc_config.yaml``.

Shared by ``alomancy list-hpc`` and the workflow's pre-run summary
(``CommitteeUncertaintyWorkflow.display_workflow_summary``), so the two
tables can't drift apart.
"""

from typing import Any

import polars as pl

from alomancy.utils.remote_ssh import get_alomancy_version_for_profile, resolve_hpc_host

# Profile keys that point at a DFT/MLIP executable, in lookup order.
_EXECUTABLE_KEYS = ("pwx_path", "vasp_path", "high_accuracy_executable_path")


def _concurrency_cap(profile: dict) -> str:
    if profile.get("max_num_of_concurrent_jobs") is not None:
        return str(profile["max_num_of_concurrent_jobs"])
    if profile.get("max_concurrent_jobs") is not None:
        return f"{profile['max_concurrent_jobs']} (old key max_concurrent_jobs)"
    return "20 (default)"


def hpc_profile_row(
    name: str, profile: dict, *, check_remote: bool = False
) -> dict[str, Any]:
    """One summary row for an HPC profile. ``check_remote`` adds the
    alomancy version installed on the host, looked up over ssh (slow, and
    may prompt for a password)."""
    node_info = profile.get("node_info", {}) or {}
    executable = next((profile[k] for k in _EXECUTABLE_KEYS if profile.get(k)), "?")
    row: dict[str, Any] = {
        "hpc_name": name,
        "ssh_host": resolve_hpc_host(profile.get("hpc_name", name)) or "?",
        "gpu": str(profile.get("gpu", "?")),
        "partitions": ", ".join(profile.get("partitions", []) or []) or "?",
        "ranks_per_node": str(node_info.get("ranks_per_node", "?")),
        "ranks_per_system": str(node_info.get("ranks_per_system", "?")),
        "threads_per_rank": str(node_info.get("threads_per_rank", "?")),
        "max_mem_per_node": str(node_info.get("max_mem_per_node", "?")),
        "max_num_of_concurrent_jobs": _concurrency_cap(profile),
        "default_max_time": str(profile.get("default_max_time", "?")),
        "executable": str(executable),
    }
    if check_remote:
        row["alomancy_version"] = get_alomancy_version_for_profile(profile) or "?"
    return row


def format_table(rows: list[dict[str, Any]]) -> str:
    # tbl_cols=-1: never elide columns with "…" -- every column is the point.
    with pl.Config(
        fmt_str_lengths=200,
        tbl_width_chars=250,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
    ):
        return str(pl.DataFrame(rows))
