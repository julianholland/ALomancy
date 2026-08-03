import json
import re
import subprocess
from pathlib import Path

from expyre.units import mem_to_kB
from yaml import safe_dump, safe_load

from alomancy.configs.global_config import ALOMANCY_HPC_CONFIG, EXPYRE_CONFIG

# ---------------------------------------------------------------------------
# Pure data builders (testable without mocking input())
# ---------------------------------------------------------------------------

_CPU_HEADER = [
    "#SBATCH --no-requeue",
    "#SBATCH --nodes={num_nodes}",
    "#SBATCH --ntasks-per-node={num_cores}",
]

_GPU_HEADER_BASE = [
    "#SBATCH --no-requeue",
    "#SBATCH --nodes={num_nodes}",
    "#SBATCH --cpus-per-task={num_cores}",
]


def mem_str_to_mb(mem_str: str | None) -> int | None:
    """Convert a human memory string (e.g. '60GB') to whole MB.

    Returns None if mem_str is empty or unparseable, so callers can skip
    adding a --mem line rather than crash.
    """
    if not mem_str:
        return None
    try:
        mem_kb = mem_to_kB(mem_str)
    except ValueError:
        return None
    if mem_kb is None:
        return None
    return mem_kb // 1024


def _with_partition_mem_headers(partitions: dict[str, dict]) -> dict[str, dict]:
    """Return a copy of partitions, each carrying its own #SBATCH --mem line.

    A system can offer partitions with very different node memory sizes
    (e.g. a 12GB GPU partition and a 240GB CPU partition). ExPyRe applies a
    partition's own "header" entries in addition to the system-wide header
    for any job landing on it (see System.submit's per-partition header_extra
    lookup), so scoping --mem here means each job gets the right ceiling for
    the partition it actually uses — unlike a single blanket --mem baked
    into the system-wide header from just one partition's max_mem, which
    either under-requests on bigger-memory partitions or gets jobs rejected
    outright on smaller ones.
    """
    result: dict[str, dict] = {}
    for name, spec in partitions.items():
        spec = dict(spec)
        mem_mb = mem_str_to_mb(spec.get("max_mem"))
        if mem_mb is not None:
            spec_header = list(spec.get("header", []))
            if not any("--mem" in line for line in spec_header):
                spec_header.append(f"#SBATCH --mem={mem_mb}M")
            spec["header"] = spec_header
        result[name] = spec
    return result


def build_expyre_entry(
    host: str,
    gpu: bool,
    partitions: dict[str, dict],
    commands: list[str],
    rundir: str,
    gpu_constraint: str | None = None,
    gpu_gres: str | None = None,
) -> dict:
    """Build the dict for one ExPyRe system entry.

    Each partition gets its own #SBATCH --mem line (see
    _with_partition_mem_headers), sized from that partition's own max_mem,
    rather than one blanket value shared across every partition the system
    offers. Without it, jobs get whatever memory default the cluster applies
    to a job with no explicit --mem, which is not necessarily enough for
    what the DFT/MLIP binary launched inside the job actually needs. Note
    that the nested `srun` command which launches that binary
    (`_build_srun_command` in dft_utils.py) deliberately requests `--mem=0`
    ("use everything the job already has") rather than a specific
    sub-amount: on some Slurm configs, the running batch script counts as
    the job's first step and is credited with the *entire* job memory
    allocation, so a nested step asking for its own explicit amount
    competes with that reservation and fails immediately with "Unable to
    create step ... Memory required by task is not available" — regardless
    of how much the job was granted. The partition-level --mem here exists
    to set a real ceiling on the job as a whole, not to be matched exactly
    by anything inside it.
    """
    if gpu:
        header = list(_GPU_HEADER_BASE)
        if gpu_constraint:
            header.append(f"#SBATCH --constraint='{gpu_constraint}'")
        if gpu_gres:
            header.append(f"#SBATCH --gres={gpu_gres}")
    else:
        header = list(_CPU_HEADER)

    return {
        "host": host,
        "remsh_cmd": "ssh",
        "scheduler": "slurm",
        "header": header,
        "commands": commands,
        "partitions": _with_partition_mem_headers(partitions),
        "rundir": rundir,
    }


def build_alomancy_profile(
    expyre_sys_name: str,
    gpu: bool,
    partitions: list[str],
    venv_cmd: str,
    node_info: dict,
    triton_cache: str | None = None,
    dft_code: str | None = None,
    dft_paths: dict | None = None,
    max_concurrent_jobs: int = 20,
) -> dict:
    """Build the dict for one ALomancy HPC profile entry."""
    pre_cmds = [venv_cmd]
    if gpu and triton_cache:
        pre_cmds.append(f"export TRITON_CACHE_DIR={triton_cache}")

    profile: dict = {
        "hpc_name": expyre_sys_name,
        "gpu": gpu,
        "pre_cmds": pre_cmds,
        "partitions": partitions,
        "node_info": node_info,
        "max_concurrent_jobs": max_concurrent_jobs,
    }

    if dft_paths:
        if dft_code == "qe":
            if "pwx_path" in dft_paths:
                profile["pwx_path"] = dft_paths["pwx_path"]
            if "pp_path" in dft_paths:
                profile["pp_path"] = dft_paths["pp_path"]
        elif dft_code == "vasp":
            if "vasp_path" in dft_paths:
                profile["vasp_path"] = dft_paths["vasp_path"]
            if "pp_path" in dft_paths:
                profile["pp_path"] = dft_paths["pp_path"]

    return profile


def write_expyre_config(
    system_name: str, entry: dict, path: Path = EXPYRE_CONFIG
) -> None:
    """Add or overwrite a system entry in the ExPyRe config JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path) as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not valid JSON. Fix or delete it before running the wizard.\n"
                f"Parse error: {exc}"
            ) from exc
    else:
        config = {"systems": {}}
    config.setdefault("systems", {})[system_name] = entry
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def write_alomancy_hpc_config(
    profile_name: str, entry: dict, path: Path = ALOMANCY_HPC_CONFIG
) -> None:
    """Add or overwrite a profile entry in ~/.alomancy/hpc_config.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path) as f:
            config = safe_load(f) or {}
    else:
        config = {}
    config[profile_name] = entry
    with open(path, "w") as f:
        safe_dump(config, f, default_flow_style=False, sort_keys=False)


def run_remote_install(host: str, python_path: str) -> None:
    """Run ``pip install alomancy`` on the remote system over SSH."""
    subprocess.run(
        ["ssh", host, f"{python_path} -m pip install alomancy"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------


def _prompt(msg: str, default: str = "") -> str:
    if default:
        answer = input(f"{msg} [{default}]: ").strip()
        return answer if answer else default
    return input(f"{msg}: ").strip()


def _prompt_int(msg: str, default: int | None = None) -> int:
    default_str = str(default) if default is not None else ""
    while True:
        raw = _prompt(msg, default=default_str)
        try:
            return int(raw)
        except ValueError:
            print(f"  Please enter a whole number (got '{raw}').")


def _yes_no(msg: str, default: bool = False) -> bool:
    yn = "[Y/n]" if default else "[y/N]"
    answer = input(f"{msg} {yn}: ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def _read_ssh_hosts() -> list[str]:
    """Return Host aliases from ~/.ssh/config, excluding wildcards."""
    ssh_config = Path.home() / ".ssh" / "config"
    if not ssh_config.exists():
        return []
    hosts: list[str] = []
    with open(ssh_config) as f:
        for line in f:
            stripped = line.strip()
            if stripped.lower().startswith("host "):
                for alias in stripped[5:].split():
                    if "*" not in alias and "?" not in alias:
                        hosts.append(alias)
    return hosts


def _derive_python_from_venv(venv_cmd: str) -> str | None:
    """Extract the python executable path from a venv activation command.

    Handles: source /path/to/venv/bin/activate → /path/to/venv/bin/python
    """
    m = re.search(r"source\s+(.+)/bin/activate", venv_cmd)
    if m:
        return f"{m.group(1)}/bin/python"
    return None


def _pick_ssh_host() -> str:
    """Show SSH aliases from ~/.ssh/config and let the user pick or type one."""
    hosts = _read_ssh_hosts()
    if hosts:
        print("\nAvailable SSH hosts from ~/.ssh/config:")
        for i, h in enumerate(hosts, 1):
            print(f"  {i}) {h}")
        print("  Enter a number to select, or type a hostname directly.")
        while True:
            raw = input("SSH host: ").strip()
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(hosts):
                    return hosts[idx]
                print(
                    f"  Please enter a number between 1 and {len(hosts)}, or a hostname."
                )
            elif raw:
                return raw
    return _prompt("SSH host alias (e.g. 'raven')")


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------


def add_hpc_wizard() -> None:
    """Interactive terminal wizard to add an HPC system to ALomancy."""
    print("\n=== ALomancy HPC Setup Wizard ===\n")
    print("Configures two files:")
    print(f"  {EXPYRE_CONFIG}")
    print(f"  {ALOMANCY_HPC_CONFIG}")
    print()
    print(
        "Before continuing, make sure this HPC is reachable by SSH alias.\n"
        "Add it to ~/.ssh/config (Linux/macOS) if you haven't already, e.g.:\n"
        "\n"
        "  Host raven\n"
        "      HostName raven.mpcdf.mpg.de\n"
        "      User jholl\n"
        "\n"
        "Then verify with: ssh <alias> hostname"
    )
    print()

    # --- ExPyRe system ---
    print("--- ExPyRe System (scheduler config) ---")
    system_name = _prompt(
        "System name in ~/.expyre/config.json (e.g. 'raven_gpu')"
    ).strip()
    while not system_name:
        system_name = _prompt("System name cannot be empty. Try again").strip()

    ssh_host = _pick_ssh_host()
    while not ssh_host:
        ssh_host = _pick_ssh_host()

    gpu = _yes_no("GPU system?", default=False)
    print(
        "\nScratch/run directory — ExPyRe will create job subdirectories here.\n"
        "Use a fast scratch filesystem, not your home directory. All results\n"
        "are automatically synced back to your local machine after each job."
    )
    rundir = _prompt(
        "Scratch directory path on remote, e.g. /ptmp/user/alomancy_scratch"
    )

    print(
        "\nModule/setup commands — press Enter after each command.\n"
        "Enter on a blank line to finish.\n"
        "Examples: 'module purge'  'module load python/3.11'  "
        "'export OMP_NUM_THREADS=1'"
    )
    commands: list[str] = []
    while True:
        cmd = input("  > ").strip()
        if not cmd:
            break
        commands.append(cmd)

    print("\nPartitions (at least one required):")
    partitions: dict[str, dict] = {}
    while True:
        pname = _prompt("  Partition name (e.g. 'general')").strip()
        if not pname:
            print("  Partition name cannot be empty.")
            continue
        cores = _prompt_int(f"  Cores per node for '{pname}'")
        max_time = _prompt(f"  Max time for '{pname}'", default="24:00:00")
        max_mem = _prompt(f"  Max memory for '{pname}', e.g. 240GB")
        if max_mem and mem_str_to_mb(max_mem) is None:
            print(
                f"  Warning: could not parse '{max_mem}' as a memory size "
                "(expected e.g. '240GB'); this partition's job header will "
                f"get no #SBATCH --mem line. Add one manually in "
                f"{EXPYRE_CONFIG} under partitions.{pname}.header, or jobs "
                "may get whatever memory default the cluster applies."
            )
        partitions[pname] = {
            "num_cores": cores,
            "max_time": max_time,
            "max_mem": max_mem,
        }
        if not _yes_no("  Add another partition?", default=False):
            break

    # Ranks per job ALomancy will request, and node_info.max_mem_per_node for
    # the ALomancy profile below. The latter is descriptive only — it no
    # longer drives any #SBATCH/srun --mem line (see _with_partition_mem_headers
    # for the header source, and _build_srun_command's --mem=0 for why the
    # nested srun deliberately doesn't request a specific sub-amount).
    first_part = next(iter(partitions.values())) if partitions else {}
    default_ranks = first_part.get("num_cores")
    default_mem = str(first_part.get("max_mem", ""))
    ranks_per_node = _prompt_int(
        "Ranks per node (usually = cores per node)", default=default_ranks
    )
    max_mem_node = _prompt("Max memory per node", default=default_mem)
    node_info = {
        "ranks_per_system": ranks_per_node,
        "ranks_per_node": ranks_per_node,
        "threads_per_rank": 1,
        "max_mem_per_node": max_mem_node,
    }

    gpu_constraint: str | None = None
    gpu_gres: str | None = None
    if gpu:
        print("\nGPU SBATCH options (press Enter to skip each):")
        c = _prompt("  Constraint string, e.g. gpu").strip()
        gpu_constraint = c or None
        g = _prompt("  Gres string, e.g. gpu:a100:1").strip()
        gpu_gres = g or None

    expyre_entry = build_expyre_entry(
        host=ssh_host,
        gpu=gpu,
        partitions=partitions,
        commands=commands,
        rundir=rundir,
        gpu_constraint=gpu_constraint,
        gpu_gres=gpu_gres,
    )

    # --- ALomancy profile ---
    print("\n--- ALomancy HPC Profile ---")
    print("  (This name goes in your run YAML: hpc: '<profile_name>')")
    profile_name = _prompt("Profile name", default=system_name)
    venv_cmd = _prompt(
        "Venv activation command, e.g. source /u/user/.venvs/alomancy/bin/activate"
    )

    triton_cache: str | None = None
    if gpu:
        tc = _prompt(
            "TRITON_CACHE_DIR path for GPU PyTorch JIT cache, or Enter to skip"
        ).strip()
        triton_cache = tc or None

    default_partitions = ",".join(partitions.keys())
    partitions_str = _prompt(
        "Which partition(s) will this profile use? (comma-separated)",
        default=default_partitions,
    )
    profile_partitions = [p.strip() for p in partitions_str.split(",") if p.strip()]

    print(
        "\nConcurrency — how many ALomancy jobs should run on this HPC at once? "
        "ExPyRe/Slurm still queue everything submitted; this caps how many are "
        "started (occupying a queue slot) at the same time — the next queued job "
        "starts the instant a running one finishes."
    )
    max_concurrent_jobs = _prompt_int(
        "Number of concurrent jobs you wish to have running on this hpc from alomancy",
        default=20,
    )

    print("\nDFT code on this system? Options: qe / vasp / none")
    dft_code_raw = _prompt("DFT code", default="none").strip().lower()
    dft_code: str | None = dft_code_raw if dft_code_raw in ("qe", "vasp") else None
    dft_paths: dict = {}
    if dft_code == "qe":
        dft_paths["pwx_path"] = _prompt("  QE pw.x executable path")
        dft_paths["pp_path"] = _prompt("  Pseudopotentials directory (pp_path)")
        print(
            "  Note: add pseudo_dict element→UPF mappings manually"
            f" in {ALOMANCY_HPC_CONFIG}"
        )
    elif dft_code == "vasp":
        dft_paths["vasp_path"] = _prompt("  VASP executable path")
        dft_paths["pp_path"] = _prompt(
            "  POTCAR directory (pp_path, the parent of potpaw_PBE/potpaw_LDA;"
            " sets VASP_PP_PATH)"
        )
        print(
            "  Note: add pseudo_dict element→POTCAR-suffix overrides manually"
            f" in {ALOMANCY_HPC_CONFIG}"
        )

    alomancy_profile = build_alomancy_profile(
        expyre_sys_name=system_name,
        gpu=gpu,
        partitions=profile_partitions,
        venv_cmd=venv_cmd,
        node_info=node_info,
        triton_cache=triton_cache,
        dft_code=dft_code,
        dft_paths=dft_paths if dft_paths else None,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    # --- Write files (before remote install so a failed install doesn't lose answers) ---
    print("\n--- Writing config files ---")
    write_expyre_config(system_name, expyre_entry)
    print(f"  {EXPYRE_CONFIG}  ← added '{system_name}'")
    write_alomancy_hpc_config(profile_name, alomancy_profile)
    print(f"  {ALOMANCY_HPC_CONFIG}  ← added '{profile_name}'")

    # --- Remote install ---
    print("\n--- Remote Installation ---")
    do_install = _yes_no("Install alomancy on this system now?", default=False)
    if do_install:
        derived_python = _derive_python_from_venv(venv_cmd) or ""
        python_path = _prompt(
            "Python executable path on remote",
            default=derived_python,
        ).strip()
        if python_path:
            print(
                f"  Running: ssh {ssh_host} '{python_path} -m pip install alomancy' …"
            )
            try:
                run_remote_install(ssh_host, python_path)
                print("  Done.")
            except Exception as exc:
                print(f"  Remote install failed: {exc}")
                print(
                    "  Please install manually by running:\n"
                    f"    ssh {ssh_host} '{python_path} -m pip install alomancy'\n"
                    "  Config files are already written — continuing setup."
                )

    print(f"\nSetup complete! Use '{profile_name}' in your run YAML:")
    print("  mlip_committee:")
    print(f"    hpc: '{profile_name}'")
    print("  structure_generation:")
    print(f"    hpc: '{profile_name}'")
    print("  high_accuracy_evaluation:")
    print(f"    hpc: '{profile_name}'")
    print()
