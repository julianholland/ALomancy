# Initialization

`src/alomancy/initialize/`: the default initialiser (registered under `initialiser`), which generates the starting structures a run is missing.

## Flow

`compute_needs(db, init_config, elements)` (`initialize/initialiser_interface.py`, calling `initialize/initialization_structure_list.py`'s `compute_initialization_needs`) compares what the DB already holds (`count_all_by_config_type_and_formula()`) with the configured targets and returns per-combo counts of what's missing: isolated atoms, dimers, trimers, amorphous, Materials Project, molecules, boxes of molecules. `generate(...)` (`create_initialization_atoms_list`; per type `initialize/singles_dimers_trimers.py`, `initialize/amorphize.py`, `initialize/stretch_and_compress.py`, `initialize/rattle.py`, `initialize/mp_interface.py`, `initialize/molecules.py`) builds only those; `_initialize_training_set` (`core_workflow.md`) sends them to DFT. Without an `initialization` section nothing is generated.

## Settings

One block per structure type in the `initialization` section, each with an `enabled` flag (default true); defaults in `_CREATION_KWARGS_DEFAULTS` (`initialize/initialiser_interface.py`), which is also the registered `kwargs_schema`:

- `isolated_atom_kwargs`
- `dimer_kwargs.num_of_dimers_per_combo` (default 10), `trimer_kwargs.num_of_trimers_per_combo` (default 5)
- `amorphous_kwargs`: `num_of_amorphous_structures` (default 100), `num_of_atoms_per_amorphous` (default 20), `density_range` (`[min, max]` g/cm³, default `[1.0, 1.0]`; each structure's density drawn uniformly at random from the range), `composition_list`. Seeded by `general.seed`, like the molecule boxes. `densities_list` is deprecated (`docs/deprecations.md`): it warns (`config_key_deprecated`) and is read as `[min, max]`. `seed` is deprecated too: it warns and is ignored (it used to seed the amorphous cells and, through the same argument, the molecule boxes).
- `mp_kwargs`: `max_num_of_atoms` (default 20, caps MP fetches), `mp_max_energy_above_hull` (default 0.1 eV)
- `stretch_compress_targets_kwargs`: `num_of_stretch_compress_per_target` (default 5), `max_lattice_deformation` (default 0.2)
- `rattle_target_structures` (off by default): `num_of_rattled_per_target`, `rattle_standard_deviation`
- `molecule_kwargs` (off by default, config_type `init_molecule`): `molecules` (required when enabled), `vacuum` (default 10 Å)
- `molecules_in_box_kwargs` (off by default, config_type `init_molecules_in_box`): `box_compositions` (required when enabled), `min_distance` (default 1.5 Å), `max_attempts_per_molecule` (default 1000)

`validate_kwargs` (registry hook, called from `ActiveLearningWorkflow._validate_initialiser_kwargs` at construction) checks the molecule blocks and every `density_range`, so a bad name fails at startup, not after other structures are already generated. It also warns (`_warn_unusual_densities`, events `init_density_too_low`/`init_density_too_high`) when a `density_range` gives cells above the run's `max_vdw_volume_ratio` (the blown-up filter would drop them before DFT; skipped when that is null or the config type is in `vdw_exempt_config_types`, both passed in by the workflow) or below 0.3× their vdW volume (`vdw_volume_ratio_at_density`, `utils/clean_structures.py`). Random amorphous mixes are checked as an even mix of `general.elements`: per pure element, every H config would warn.

`num_of_atoms_per_amorphous` and `mp_kwargs.max_num_of_atoms` are independent (they used to be one value). The old `d_t_s_a_ratio`/`target_non_mp_structures_to_add` are gone.

Initialization structures keep their own relaxation policy: `needs_relaxation` per structure type, `fmax` 0.05. Deliberately compressed dimers/trimers aren't capped by `force_ceiling`.

## Molecules (`initialize/molecules.py`)

Molecules are named the way SAGE's `AtomPosition.build` takes them, through `utils/molecules.py` (`build_molecule`, `molecule_symbols`, `validate_molecules`): a SAGE preset formula (`"CO"`, `"H2O"`) or `"smiles-<SMILES>"` (`"smiles-CCO"`; SAGE's own SMILES parser, hydrogens added, geometry relaxed, no RDKit). Formulas outside SAGE's presets (`"CH4"`) don't build; write them as SMILES (`"smiles-C"`). Every element must be in `general.elements`. There is no shared molecule list: each block names its molecules, and nothing falls back to another section.

- **In vacuum** (`molecule_kwargs`): one structure per molecule, SAGE's equilibrium geometry centred in a cubic cell of its largest extent plus `vacuum`, `pbc` false, `needs_relaxation` false. Distorted copies come from the shared blocks: list `init_molecule` in `general.dataset_kwargs.target_config_types` and `stretch_compress_targets_kwargs` scales the bonds (it scales the atoms with the cell), `rattle_target_structures` rattles them. That list also sets the train/test pool, and every target type shares one `max_lattice_deformation`.
- **In a box** (`molecules_in_box_kwargs.box_compositions`): each entry is `{molecules: {name: ratio}, density_range: [min, max], num_of_molecules, num_of_structures}`. Integer counts come from the ratio by largest remainder (each molecule at least once); `num_of_structures` cubic periodic boxes, each at a density drawn uniformly at random from the range and sized so that density is exact. `needs_relaxation` true, as for amorphous. `info`: `box_composition` (`"H2O:2,smiles-CO:1"`), `density`.
- **Packing is ALomancy's own.** SAGE's `MoleculeCluster_builder.add_solvent` does the same job but can't be seeded (it never passes `rng_seed` on). Here box *i* of composition *j* uses `np.random.default_rng([general.seed + j, attempt, i])` (its density too): each molecule a uniform random rotation and position, kept when no atom is within `min_distance` of another molecule's under the minimum image; failing `max_attempts_per_molecule` raises naming the composition and density. No force-field cleanup afterwards (LJ, as for amorphous, would distort the molecules).
- **Restarts** count both per chemical formula (`molecule_override`, `box_override`), so names or compositions with one formula share a target. Each kind seeds its rng with `attempt`, the number the database already holds (amorphous: `create_amorphous_atoms_list(..., attempt=)`, rng `[seed, attempt]`; boxes: `create_molecules_in_box_atoms_list(..., attempt=)`), so a restart that replaces structures lost to DFT failures draws new ones (new positions, compositions and densities) rather than copies of ones already labelled, while the same database state always gives the same structures (which the DFT restart's `dft_input_digest` matching relies on: a crash before the results reach the database regenerates the identical set). `generate` passes the overrides through as given: an empty dict or 0 means "none still needed" (it used to become `None`, "no override", and regenerate every dimer, trimer or amorphous cell of a kind that was already complete).
- `utils/dataset_curation.structure_domain` maps them to the domains `molecule` and `molecules_in_box`, before the size rules (CO is not a dimer).

## Materials Project (`initialize/mp_interface.py`)

`retrieve_mp_material_docs` requests only `fields=["material_id", "structure"]` from `mpr.materials.summary.search`, not the client's `all_fields=True` default: the live API and the installed `mp_api` client drift on unrequested sub-models (e.g. `bandstructure`), and a missing field raised a pydantic `ValidationError` that killed the whole fetch, though `docs_to_atoms` only reads `doc.structure`.
