from alomancy.configs.global_config import ALOMANCY_HPC_CONFIG, _load_global_hpc_config
from alomancy.utils.remote_ssh import (
    derive_python_from_venv,
    resolve_hpc_host,
    run_ssh_command,
)

# pip installs (PyPI resolution + download, possibly over a throttled
# cluster network) need much longer than a quick version-check ssh call.
UPGRADE_TIMEOUT_SECONDS = 300.0

# ---------------------------------------------------------------------------
# Pure data builders (testable without mocking input())
# ---------------------------------------------------------------------------


def _parse_hpc_selection(raw: str, available: list[str]) -> list[str]:
    """Parse a user's profile selection against the numbered `available` list.

    "all" (any case) selects every profile. Otherwise `raw` must be
    comma-separated 1-based indices into `available`; invalid/out-of-range/
    non-numeric tokens raise ValueError with a message naming the bad token,
    so callers can catch it and re-prompt (matches add_hpc.py's
    _prompt_int retry-on-ValueError convention). Duplicate selections are
    dropped, preserving first-seen order.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("No selection entered.")
    if raw.lower() == "all":
        return list(available)

    selected: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token.isdigit():
            raise ValueError(f"Invalid selection '{token}': not a number.")
        idx = int(token) - 1
        if not (0 <= idx < len(available)):
            raise ValueError(
                f"Invalid selection '{token}': must be between 1 and {len(available)}."
            )
        name = available[idx]
        if name not in selected:
            selected.append(name)
    return selected


def _upgrade_one_host(
    profile_name: str, profile: dict, timeout: float
) -> tuple[str, bool, str]:
    """Upgrade alomancy on one HPC profile. Returns (name, success, reason).

    `reason` is empty on success, otherwise a short human-readable
    explanation (host/python-path resolution failure, or ssh stderr/stdout).
    Never raises -- callers loop over multiple hosts and must not let one
    bad profile abort the rest of the batch.
    """
    host = resolve_hpc_host(profile.get("hpc_name", ""))
    if not host:
        return (
            profile_name,
            False,
            "could not resolve ssh host from ~/.alomancy/expyre_config.json "
            "or ~/.expyre/config.json",
        )

    pre_cmds = profile.get("pre_cmds") or []
    python_path = derive_python_from_venv(pre_cmds[0]) if pre_cmds else None
    if not python_path:
        return profile_name, False, "could not derive python path from pre_cmds"

    success, stdout, stderr = run_ssh_command(
        host, f"{python_path} -m pip install --upgrade alomancy", timeout=timeout
    )
    if success:
        return profile_name, True, ""
    return profile_name, False, stderr.strip() or stdout.strip() or "unknown error"


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------


def _yes_no(msg: str, default: bool = False) -> bool:
    yn = "[Y/n]" if default else "[y/N]"
    answer = input(f"{msg} {yn}: ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------


def upgrade_hpc_wizard() -> None:
    """Interactive terminal wizard to upgrade alomancy on configured HPC systems."""
    config = _load_global_hpc_config()
    if not config:
        print(
            f"No HPC profiles configured in {ALOMANCY_HPC_CONFIG}. "
            "Run 'alomancy add-hpc' first."
        )
        return

    names = list(config.keys())
    print("\n=== ALomancy HPC Upgrade ===\n")
    print("Configured HPC profiles:")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {name}")
    print("  Enter comma-separated numbers, or 'all'.")

    selected: list[str] = []
    while not selected:
        raw = input("Select profile(s) to upgrade: ").strip()
        try:
            selected = _parse_hpc_selection(raw, names)
        except ValueError as exc:
            print(f"  {exc}")

    print(f"\nAbout to run 'pip install --upgrade alomancy' on: {', '.join(selected)}")
    if not _yes_no("Proceed?", default=False):
        print("Aborted.")
        return

    print()
    results: list[tuple[str, bool, str]] = []
    for name in selected:
        print(f"Upgrading '{name}'...")
        result = _upgrade_one_host(name, config[name], timeout=UPGRADE_TIMEOUT_SECONDS)
        results.append(result)
        _, success, reason = result
        print("  OK" if success else f"  FAILED: {reason}")

    ok = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print(f"\n{len(ok)} succeeded, {len(failed)} failed.")
    for name, _success, reason in failed:
        print(f"  - {name}: {reason}")
