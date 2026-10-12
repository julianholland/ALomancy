# Configuration

`src/alomancy/configs/`: loading a run config, HPC profiles, and the check that reports config keys the run won't read. Settings for a specific module are described with that module.

## Loading: `load_dictionaries`

`load_dictionaries(config_path)` (`configs/config_dictionaries.py`) reads the YAML into the `jobs_dict` the workflow is built from. Sections: `general`, `initialization` (optional when `general.start_from` is set), `training`, `structure_generation`, `high_accuracy_evaluation`. An empty section (`initialization:` with nothing under it) parses as `None` and is normalized to `{}`.

- **HPC string resolution**: `hpc:` is either an inline profile dict or a profile name (`hpc: 'raven_gpu'`). A name is looked up in `~/.alomancy/hpc_config.yaml` (`global_config._load_global_hpc_config()`) and replaced by the full dict; an unknown name raises `ValueError` pointing at `alomancy add-hpc`. A section with an HPC profile and no `max_time` gets the profile's `default_max_time`.
- Section names are fixed to their keys; a `name:` in the config is ignored with a warning (event `config_key_ignored`).

`configs/global_config.py` holds the paths: `ALOMANCY_HPC_CONFIG` (`~/.alomancy/hpc_config.yaml`), `EXPYRE_CONFIG` (`~/.alomancy/expyre_config.json`, the HPC systems config written by `alomancy add-hpc`) and `LEGACY_EXPYRE_CONFIG` (`~/.expyre/config.json`, its pre-isolation location; see `remote_submission.md`).

`configs/hpc_profiles.py` (`hpc_profile_row`, `format_table`) builds the HPC profile table shared by `alomancy list-hpc` and the run's startup summary. `list-hpc` drops the `executable` column; the startup table keeps it.

## Validation at construction, not at first use

Anything that would otherwise fail hours into a run is checked when the workflow is built (`ActiveLearningWorkflow.__init__`): required `general.dataset_kwargs` keys, renamed/moved/removed keys (`_RENAMED_KEYS`, `_MOVED_KEYS`, `_REMOVED_KEYS` in `core/active_learning_workflow.py`, each raising with the replacement), `start_from`, split filters, committee size, the generation vs selection counts. `test_ratio` would otherwise first be read after a loop's DFT finished.

## Unread keys: `configs/schema.py`

`ActiveLearningWorkflow._warn_about_unread_keys` calls `check_config_keys`. The schema is the tree of keys each section reads **for the selected modules**, built from existing definitions: `_GENERAL_KNOWN_KEYS`, `_DATASET_KWARGS_KEYS`, each workflow's `KWARGS_DEFAULTS`, and each module's registered `kwargs_schema` (MD's comes from `run_md`'s signature, the initialiser's from its defaults). A key not read where it's written is:

- **misplaced** if the schema reads it elsewhere: one WARNING each naming the closest location (event `config_key_misplaced`), e.g. `num_of_md_starts` in the generic `structure_generation.structure_selection_kwargs` belongs in `md_kwargs.structure_selection_kwargs`;
- **unused** if it's another registered module's settings block (`mace_kwargs` while `trainer: sevennet`; event `config_key_unused`);
- **ignored** if it's an old key no longer read (a phase's `name`; event `config_key_ignored`);
- else **unrecognised** (one combined WARNING, event `config_key_unrecognised`).

External packages' settings (`mace_kwargs`, `sevennet_kwargs`, `ezga_kwargs`, `qe_kwargs`, `vasp_kwargs`) and pass-through blocks (`hpc`, filters, `start_from`, `quality_gate`, `dataset_curation`) are `OPEN`: too many keys to list, or validated elsewhere.

A module whose settings ALomancy owns should register a `kwargs_schema`, and a new key a section reads must be added to its schema. `tests/test_example_configs.py` builds every example config, the README config and its one-line edits, and every config block in `docs/examples.md` and `docs/starting_a_run.md`, failing on any config warning.

## Registry

`registry.py` maps names to modules per category (`al_workflow`, `mlip_trainer`, `structure_generator`, `dft_evaluator`, `initialiser`). `register()` never imports; `resolve()` imports lazily on first use. Config keys pick the entry: `general.al_workflow`, `training.trainer`, `structure_generation.generator`, `high_accuracy_evaluation.evaluator`.
