# Initialization

`src/alomancy/initialize/`: the default initialiser (registered under `initialiser`), which generates the starting structures a run is missing.

## Flow

`compute_needs(db, init_config, elements)` (`initialize/initialiser_interface.py`, calling `initialize/initialization_structure_list.py`'s `compute_initialization_needs`) compares what the DB already holds (`count_all_by_config_type_and_formula()`) with the configured targets and returns per-combo counts of what's missing: isolated atoms, dimers, trimers, amorphous, Materials Project. `generate(...)` (`create_initialization_atoms_list`; per type `initialize/singles_dimers_trimers.py`, `initialize/amorphize.py`, `initialize/stretch_and_compress.py`, `initialize/rattle.py`, `initialize/mp_interface.py`) builds only those; `_initialize_training_set` (`core_workflow.md`) sends them to DFT. Without an `initialization` section nothing is generated.

## Settings

One block per structure type in the `initialization` section, each with an `enabled` flag (default true); defaults in `_CREATION_KWARGS_DEFAULTS` (`initialize/initialiser_interface.py`), which is also the registered `kwargs_schema`:

- `isolated_atom_kwargs`
- `dimer_kwargs.num_of_dimers_per_combo` (default 10), `trimer_kwargs.num_of_trimers_per_combo` (default 5)
- `amorphous_kwargs`: `num_of_amorphous_structures` (default 100), `num_of_atoms_per_amorphous` (default 20), `densities_list`, `composition_list`, `seed`
- `mp_kwargs`: `max_num_of_atoms` (default 20, caps MP fetches), `mp_max_energy_above_hull` (default 0.1 eV)
- `stretch_compress_targets_kwargs`: `num_of_stretch_compress_per_target` (default 5), `max_lattice_deformation` (default 0.2)
- `rattle_target_structures` (off by default): `num_of_rattled_per_target`, `rattle_standard_deviation`

`num_of_atoms_per_amorphous` and `mp_kwargs.max_num_of_atoms` are independent (they used to be one value). The old `d_t_s_a_ratio`/`target_non_mp_structures_to_add` are gone.

Initialization structures keep their own relaxation policy: `needs_relaxation` per structure type, `fmax` 0.05. Deliberately compressed dimers/trimers aren't capped by `force_ceiling`.

## Materials Project (`initialize/mp_interface.py`)

`retrieve_mp_material_docs` requests only `fields=["material_id", "structure"]` from `mpr.materials.summary.search`, not the client's `all_fields=True` default: the live API and the installed `mp_api` client drift on unrequested sub-models (e.g. `bandstructure`), and a missing field raised a pydantic `ValidationError` that killed the whole fetch, though `docs_to_atoms` only reads `doc.structure`.
