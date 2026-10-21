# Command-line interface

`src/alomancy/cli/`, dispatched by `cli/main.py` (`alomancy <command>`).

| Command | Module | What it does |
|---|---|---|
| `run config.yaml` | `cli/main.py` | `ALomancy(config).run()` |
| `add-hpc` | `cli/add_hpc.py` | Interactive wizard `add_hpc_wizard()`; pure builders `build_expyre_entry`, `build_alomancy_profile`, `write_expyre_config`, `write_alomancy_hpc_config`, `run_remote_install`. Writes the ExPyRe systems config (`EXPYRE_CONFIG`) and the profile in `~/.alomancy/hpc_config.yaml`; per-partition `#SBATCH --mem` headers (`remote_submission.md`) |
| `list-hpc [--check-remote]` | `cli/list_hpc.py` | One table of every profile: ssh host, gpu, partitions, ranks/threads, memory, concurrency cap, default max time (no executable column). `--check-remote` adds each host's installed alomancy version over ssh |
| `upgrade-hpc` | `cli/upgrade_hpc.py` | `upgrade_hpc_wizard()`: pick profiles (one, several or `all`; parsed by `_parse_hpc_selection`), then `pip install --upgrade alomancy` over ssh on each (`_upgrade_one_host`: resolve host → derive the venv's python → install). Reports per host and never aborts the batch over one bad profile |
| `nuke` | `cli/nuke.py` | `nuke_expyre_results()` deletes local ExPyRe job state (job DB, unsynced stage dirs) under the run's `.expyre` (`resolve_default_expyre_dir`), keeping `config.json`; asks for confirmation |
| `report [--loop N\|--all] [--config]` | `cli/report.py` | Rebuilds loop reports offline from `results/run_config.yaml` |
| `results --replot` | `cli/replot.py` | Redraws every loop's plots in the same layout as a live run; strict about inconsistent evaluations |

Shared ssh helpers (`utils/remote_ssh.py`): `resolve_hpc_host` (a system's ssh host from `EXPYRE_CONFIG`, falling back to `LEGACY_EXPYRE_CONFIG`), `run_ssh_command` (bounded timeout, never raises), `derive_python_from_venv`, `get_alomancy_version_for_profile`, `ensure_ssh_connectivity`.
