# Starting a Run

Every ALomancy run needs an initial set of DFT-labelled structures before the
first committee can be trained. `general.start_from` says where those come
from. There are four start modes, and the mode is picked automatically from
which keys you give.

## Choosing a start mode

| Keys under `general.start_from` | Mode | What happens |
|---|---|---|
| `train_xyz` + `test_xyz` | Pre-split files | Both files are imported. Every structure in `train_xyz` is training data and every structure in `test_xyz` is test data, exactly as given. |
| `xyz` (one path or a list) | Single file | The file(s) are imported, then split into train/test by `test_ratio` (see [Train/test split](#train-test-split)). |
| `database` | Former ALomancy database | A **copy** of a previous run's `global_database` is imported, keeping its `config_type`, train/test split and duplicate flags. The old database is never modified. |
| *(no `start_from`)* | Cold start | Nothing is imported. ALomancy generates and DFT-evaluates the whole initial dataset itself. |

When to use each:

- **Pre-split files:** you already have a train/test split you want to keep, e.g. from a paper or from another ALomancy run's `results/initialization/train_set.xyz` and `test_set.xyz`.
- **Single file:** you have one pile of DFT data and want ALomancy to hold out a test set.
- **Former database:** you are starting a new run (new settings, new results directory) on top of everything an earlier run computed.
- **Cold start:** you have no data yet.

Giving more than one source (e.g. `xyz` and `database`), or only one of `train_xyz`/`test_xyz`, is an error at start-up.

## Examples

Pre-split files (ALomancy-written, or any extxyz with DFT labels):

```yaml
general:
  elements: ["C"]
  start_from:
    train_xyz: "data/train.xyz"
    test_xyz: "data/test.xyz"
  committee_uncertainty_kwargs:
    test_ratio: 0.1
    target_config_types: ["init_MP", "init_amorphous"]
```

Single file from another code, whose keys don't match ALomancy's:

```yaml
general:
  elements: ["C"]
  start_from:
    xyz: "data/carbon_dft.xyz"        # or a list of files
    metadata_map:                    # only needed for non-standard keys
      config_type: "phase"            # info key holding the structure label
      energy: "energy_pbe"            # info key holding the DFT energy (eV)
      forces: "forces_pbe"            # per-atom array holding DFT forces (eV/Å)
  committee_uncertainty_kwargs:
    test_ratio: 0.1
    target_config_types: ["liquid", "amorphous"]   # values of the "phase" key
```

Former ALomancy database:

```yaml
general:
  elements: ["C"]
  start_from:
    database: "../previous_run/results/global_database"
  committee_uncertainty_kwargs:
    test_ratio: 0.1
    target_config_types: ["init_MP", "init_amorphous"]
```

Cold start: leave `start_from` out.

```yaml
general:
  elements: ["C"]
  committee_uncertainty_kwargs:
    test_ratio: 0.1
    target_config_types: ["init_MP", "init_amorphous"]
```

## What every mode does next

After the import, all four modes follow the same path:

1. **Import into this run's database** (`results/global_database`). Imports are idempotent. Each xyz file's SHA-256 (and a database's path) is recorded on its structures, so restarting a run never imports the same data twice.
2. **Fill the gaps.** ALomancy counts what the database already holds against the `initialization` targets (isolated atoms, dimers, trimers, amorphous, Materials Project, ...). It generates and DFT-evaluates **only what is missing**. Imported data counts towards those targets. To skip a structure type entirely, set its `enabled: false` in the `initialization` section.
3. **Split into train/test** (next section) and write `results/initialization/train_set.xyz` / `test_set.xyz`.
4. **Start the AL loop.**

Once any AL loop has finished, a restart resumes from the database. `start_from` is not consulted again, and neither is the initialization step.

## Label handling for xyz files

ALomancy stores DFT labels as `atoms.info["REF_energy"]`, `atoms.arrays["REF_forces"]`, optionally `atoms.info["REF_stresses"]`, and a provenance label `atoms.info["config_type"]`. Files that already use these keys pass through unchanged. For other files, each label is looked up in this order, and the first match wins:

| Label | 1. Already set | 2. `metadata_map` | 3. Auto-detected keys | 4. Calculator | If nothing matches |
|---|---|---|---|---|---|
| `config_type` | `config_type` | `metadata_map.config_type` | `type`, `label`, `config`, `config_type_name` | — | set to `"external"` |
| `REF_energy` | `REF_energy` | `metadata_map.energy` | `energy`, `dft_energy`, `total_energy` | energy stored with the structure | **error** |
| `REF_forces` | `REF_forces` | `metadata_map.forces` | `forces`, `dft_forces` | forces stored with the structure | **error** |
| `REF_stresses` | `REF_stresses` | `metadata_map.stress` | `stress`, `dft_stress` | stress stored with the structure | left out (stress is optional) |

"Calculator" covers the common case where ASE's extxyz reader has already moved plain `energy`/`forces` keys into a single-point calculator.

Labels are checked **before** anything is imported or sent to DFT. One summary line in the log says which key each label came from and how many structures became `"external"`.

Metadata that belongs to the run that wrote the file is dropped on import: train/test `split` tags, `global_db_id`, duplicate and quality-filter flags, and per-loop model predictions (`model_*`, `mace_*`). This run recomputes all of these. Provenance keys such as `al_loop` are kept. Pre-split files then get their split from which file they came from.

## Train/test split

| Mode | Split |
|---|---|
| Pre-split files | As given: `train_xyz` → train, `test_xyz` → test. |
| Former database | The imported database's own `split` tags are kept (its quality-filter flags are recomputed by this run's `train_filter`/`test_filter`). |
| Single file | `test_ratio` of the structures whose `config_type` is in `target_config_types` go to test; everything else trains. At least one structure of each targeted type stays in training. |
| Cold start | Same rule as a single file. |

Structures generated to fill gaps have no split yet, so they follow the single-file rule in every mode.

**Single file with no usable `config_type`.** If none of a single file's structures has a `config_type` (or an equivalent key), they all become `"external"`. `"external"` is not a target type unless you list it, so no test set could be formed and ALomancy stops with an error. Fix it in one of two ways:

- Give some structures a `config_type` (in the file, or point `start_from.metadata_map.config_type` at an existing key) and list those types in `target_config_types`.
- Add `"external"` to `target_config_types` to hold out a random `test_ratio` of the whole file.

## Migrating from the removed keys

The old start-up keys now stop the run with an error that names the replacement:

| Old key | Replacement |
|---|---|
| `general.initial_train_file_path` | `general.start_from.train_xyz` |
| `general.initial_test_file_path` | `general.start_from.test_xyz` |
| `initialization.extra_datasets` | `general.start_from.xyz` (a list is allowed) |
| `general.skip_initialization` | Not needed. A warm start only generates what the imported data doesn't cover; use `enabled: false` on a structure type to skip it. |
| `initialization.reset_extra_splits` | Not needed. Imported files always drop the old run's splits and flags. |

Two behaviors differ from the old fast path. `initial_train_file_path`/`initial_test_file_path` used to be loaded as-is, with no gap filling. The same files under `train_xyz`/`test_xyz` now also get missing initialization structures generated. Set `enabled: false` on the structure types you don't want.

## Common errors and fixes

| Error message (start) | Cause | Fix |
|---|---|---|
| `general.start_from has more than one data source` | e.g. both `xyz` and `database` given | Keep one source. |
| `general.start_from.train_xyz and test_xyz must be given together` | Only one of the pair | Add the other, or use `xyz` for a single file to be split. |
| `Unknown general.start_from key(s)` | Typo | Allowed keys: `train_xyz`, `test_xyz`, `xyz`, `database`, `metadata_map`. |
| `general.start_from.metadata_map only applies to xyz imports` | `metadata_map` with `database` or a cold start | Remove `metadata_map`; databases already use ALomancy's keys. |
| `Could not find DFT labels for N structure(s)` | Energy or forces under an unrecognised key | Name the keys with `start_from.metadata_map.energy` / `.forces`. |
| `No test set could be formed from general.start_from.xyz` | Single file, all structures `"external"` | See [Train/test split](#train-test-split). |
| `No ALomancy database at ...` | `database` path wrong | Point at the `global_database` directory itself, e.g. `../run/results/global_database`. |
| `... is this run's own database` | `database` is the current run's `db_path` | Point at the *former* run's database, or just restart the current run. |
| `Config uses removed key(s)` | Old keys | See [Migrating from the removed keys](#migrating-from-the-removed-keys). |
