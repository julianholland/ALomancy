# Example configs

Complete configs for the main ways to run ALomancy. Each one is checked by
`tests/test_example_configs.py`: it must build its workflow and resolve
every module's settings without warnings, so these stay valid as the code
changes.

| Config | Workflow | Trainer | Generator | Evaluator | Start |
|---|---|---|---|---|---|
| `committee_mace_cold_start.yaml` | committee uncertainty | MACE | MD | QE | cold start |
| `committee_sevennet.yaml` | committee uncertainty | SevenNet | MD (NPT) | QE | `start_from.xyz` |
| `random_selection_baseline.yaml` | random selection (baseline) | MACE | MD | QE | cold start |
| `novelty_selection_from_database.yaml` | novelty selection | MACE | EZGA | VASP | `start_from.database` |

The `hpc` values name profiles in `~/.alomancy/hpc_config.yaml`; create
yours with `alomancy add-hpc` and change the names to match. Then:

```bash
alomancy run committee_mace_cold_start.yaml
```

Any setting not shown uses its default; `docs/examples.md` describes every
section, and `docs/starting_a_run.md` the `start_from` modes.
