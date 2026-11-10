# Initialization

`src/alomancy/initialize/`: the default initialiser (registered under `initialiser`), which generates the starting structures a run is missing.

## Flow

`compute_needs(db, init_config, elements)` (`initialize/initialiser_interface.py`, calling `initialize/initialization_structure_list.py`'s `compute_initialization_needs`) compares what the DB already holds (`count_all_by_config_type_and_formula()`) with the configured targets and returns per-combo counts of what's missing: isolated atoms, dimers, trimers, amorphous, Materials Project, molecules, boxes of molecules. `generate(...)` (`create_initialization_atoms_list`; per type `initialize/singles_dimers_trimers.py`, `initialize/amorphize.py`, `initialize/stretch_and_compress.py`, `initialize/rattle.py`, `initialize/mp_interface.py`, `initialize/molecules.py`) builds only those; `_initialize_training_set` (`core_workflow.md`) sends them to DFT. Without an `initialization` section nothing is generated.

## Settings

One block per structure type in the `initialization` section, each with an `enabled` flag (default true); defaults in `_CREATION_KWARGS_DEFAULTS` (`initialize/initialiser_interface.py`), which is also the registered `kwargs_schema`:

- `isolated_atom_kwargs`
- `dimer_kwargs.num_of_dimers_per_combo` (default 10), `trimer_kwargs.num_of_trimers_per_combo` (default 5); both take `distance_range` (default `[0.5, 2.0]`), the separations as multiples of the covalent radii sum (`DEFAULT_DISTANCE_RANGE` in `initialize/singles_dimers_trimers.py`; for a trimer, of the largest pair). The former 0.2–1.0 × the vdW radii started at 0.70 Å for Pt-Pt: overlapping PAW spheres, forces above 100 eV/Å, failed SCFs. Isolated atoms sit in a cube of at least 15 Å (`MIN_ISOLATED_ATOM_BOX`): 3 × the vdW radius alone (about 5 Å) left a QE atom interacting with its images.
- `amorphous_kwargs`: `num_of_amorphous_structures` (default 100), `num_of_atoms_per_amorphous` (default 20), `density_range` (`[min, max]` g/cm³, default `[1.0, 1.0]`; each structure's density drawn uniformly at random from the range), `composition_list`. Seeded by `general.seed`, like the molecule boxes. `densities_list` is deprecated (`docs/deprecations.md`): it warns (`config_key_deprecated`) and is read as `[min, max]`. `seed` is deprecated too: it warns and is ignored (it used to seed the amorphous cells and, through the same argument, the molecule boxes).
- `mp_kwargs`: `max_num_of_atoms` (default 20, caps MP fetches), `mp_max_energy_above_hull` (default 0.1 eV). The query takes cells from 1 site up (`num_sites=(1, max)`): a lower bound of 2 left out 1-atom primitive cells, the ground state of fcc metals.
- `stretch_compress_targets_kwargs`: `num_of_stretch_compress_per_target` (default 5), `max_lattice_deformation` (default 0.2)
- `rattle_target_structures` (off by default): `num_of_rattled_per_target`, `rattle_standard_deviation` (default `DEFAULT_RATTLE_FRACTION`, 3 %, of each structure's nearest-neighbour distance, periodic images included: about 0.08 Å for Pt). `create_rattle_atoms_list` (`initialize/rattle.py`) rattles a 1-atom periodic cell as a 2×2×2 supercell (info `supercell`, `parent_config_type`), and skips a single non-periodic atom: moving the only atom is a pure translation, a duplicate label.
- `molecule_kwargs` (off by default, config_type `init_molecule`): `molecules` (required when enabled), `vacuum` (default 10 Å)
- `molecules_in_box_kwargs` (off by default, config_type `init_molecules_in_box`): `box_compositions` (required when enabled), `min_distance` (default 1.5 Å), `max_attempts_per_molecule` (default 1000)
- `amorphous_cluster_kwargs` (config_type `init_amorphous_cluster`, domain `cluster`): `num_of_amorphous_clusters` (default 50), `num_of_atoms_per_cluster` (default 10), `density_range` (default `[1.0, 1.0]`), `composition_list`, `vacuum` (default 10 Å)
- `molecule_cluster_kwargs` (config_type `init_molecule_cluster`, domain `molecule_cluster`): `cluster_compositions` (entries as for `box_compositions`), `min_distance` (default 1.5 Å), `max_attempts_per_molecule` (default 1000), `vacuum` (default 10 Å)
- `crystal_structures_kwargs` (**on** by default; config_type `init_crystal`, domain `bulk`): `elements` (default None: `default_elements`, the `METALS` among `general.elements`), `structures` (default all of `CRYSTAL_STRUCTURES`), `lattice_constants`, `volume_per_atom`. See "Crystal structures" below.

## Crystal structures (`initialize/crystals.py`)

The block matters most when few bulk structures are available, e.g. a cold start for one metal where Materials Project returns only the ground state. A worked single-element example is in `docs/examples.md` ("Example: a Single Bulk Metal from a Cold Start").

Each chosen element in every structure of `CRYSTAL_STRUCTURES` (fcc, bcc, hcp, sc, diamond, A15/β-W, ω) as a primitive cell (1, 1, 2, 1, 2, 8, 3 atoms), so the model sees an element's competing phases, not only the ground state MP returns.
- **Sizes** (`build_crystal`), first match wins:
  1. `lattice_constants[element][structure]` (Å; hcp and ω also `{a, c}`);
  2. `volume_per_atom[element]`;
  3. `reference_volume_per_atom(element)`: the volume per atom of `ase.build.bulk(element)`, or for Mn, Ga and Pa (no atomic basis in ASE) of the reference cell with its known atoms per cell.

  Without user values the element's own reference structure reproduces the reference lattice constant (fcc Pt 3.92 Å, 15.06 Å³/atom), and every other structure is scaled to that volume per atom (bcc Pt 3.11 Å, hcp 2.77 Å). hcp c/a is the reference's when the reference crystal is hcp (Ti 1.588), else √(8/3); ω uses 0.62. `info["lattice_source"]` records `reference`, `equal_volume` or `user`.
- **Validation** (`_validate_crystal_structures`, at construction) raises on unknown structures, elements outside `general.elements`, malformed or non-positive values, and elements with no reference data and no user value. It logs one line per element with each lattice and its source.
- **Single points** (`needs_relaxation` False): a fixed-cell relaxation leaves a symmetric crystal unchanged. Not test targets unless `init_crystal` is in `target_config_types` (then stretch/compress and rattle apply, 1-atom cells rattled as 2×2×2 supercells).
- **Always exempt from the train filter** (`utils/split_filter.py` `ALWAYS_EXEMPT_FROM_TRAIN_FILTER`, merged into the user's `exempt_config_types`): the unstable phases are far above the ground state by design.
- **Restarts**: counted per `(formula, crystal_structure)` (`GlobalDatabase.count_by_info`; formula alone can't tell fcc from bcc), `needs["crystal_override"] = {element: [missing structures]}`. Initialisation only runs again while no AL loop has finished, and not at all once its DFT is done, so the default-on block doesn't change runs already past it.
- Periodic only: off by default without periodic boundary conditions (`_NON_PERIODIC_DEFAULTS`), and enabling it explicitly there raises (`_PERIODIC_ONLY_BLOCKS`, no replacement).
- `structure_domain` checks the bulk config_types (`init_MP`, `init_crystal`, ...) before its 1-atom rule, so a 1-atom primitive cell is `bulk`, not `isolated_atom`.

## Checks that stop a run before initialisation DFT

- **A block that can never be met** raises in `validate_kwargs` (`_warn_unusual_densities`): when every composition of an enabled `amorphous_kwargs` or `molecules_in_box_kwargs` is blown up (above `max_vdw_volume_ratio`) even at the densest end of its `density_range`. Such a need could never be met, so every restart used to re-enter initialisation. A partly sparse range only warns.
- **Too few test-eligible structures**: `ActiveLearningWorkflow._check_projected_test_set` runs after generation, before submission. It counts the structures that can go to the test set: test-tagged and eligible ones already in the DB, plus generated ones of the eligible types (`test_eligible_config_types`: `target_config_types` and `TARGET_DERIVED_CONFIG_TYPES`, i.e. `init_stretch_compress` and `init_rattle`) that the blown-up filter won't drop (`utils/clean_structures.py` `is_blown_up`, the same rule as the orchestrator). Below 2 it raises; below 5 it warns `few_target_structures`. Skipped with `grouped_splits`.
- **Initialisation DFT already done, needs still unmet** (e.g. cells dropped as blown up, failed jobs): `_initialize_training_set` doesn't regenerate them, since the orchestrator would return its cached results anyway (`evaluation_done`). It reloads those results and warns `initialization_needs_unmet`, naming the needs and how to retry (delete `results/initialization/high_accuracy_eval.done`). `add_structures` skips results whose digest is already stored (`database.md`), so the reload adds nothing twice.

## Evaluators without periodic boundary conditions

The defaults depend on the evaluator (`kwargs_defaults(evaluator_supports_periodic)`, registry entry point `kwargs_defaults_for`). `compute_needs`, `generate` and `validate_kwargs` take `evaluator_supports_periodic`, which the workflow passes from the evaluator's registry entry point `supports_periodic`. With an evaluator that has none (ORCA), `_NON_PERIODIC_DEFAULTS` applies:
- `amorphous_kwargs`, `mp_kwargs` and `molecules_in_box_kwargs` are off;
- `amorphous_cluster_kwargs` and `molecule_cluster_kwargs` are on.

A block whose `enabled` the user wrote is never switched. `validate_kwargs`:
- raises when the user enabled a periodic block, naming its replacement (`_PERIODIC_ONLY_BLOCKS`: amorphous → amorphous clusters, boxes → molecule clusters, MP has none);
- warns `config_key_unused` when a periodic block has settings but no `enabled`;
- logs which defaults changed.

A default-on `molecule_cluster_kwargs` without `cluster_compositions` makes nothing (info line). Explicitly enabled without them, it raises: no molecule list is shared with another block.

With a size cap (`max_num_of_atoms`, from the evaluator's entry point of that name: ORCA with a correlated method), `validate_kwargs` raises when an enabled block is configured to make larger structures (`_structure_sizes`: amorphous cells, clusters, molecules, boxes, molecule clusters). Every one of them would be dropped before evaluation.

- **Amorphous clusters** (`create_amorphous_cluster_atoms_list`, `initialize/amorphize.py`): atoms uniform in a sphere whose radius comes from a density drawn from `density_range`, the same LJ cleanup as the periodic cells, centred in a cube of extent + `vacuum`, `pbc` False, `needs_relaxation` true. Random compositions draw each atom from `general.elements`. rng `[seed, attempt, 1]` (the trailing 1 keeps it apart from the periodic cells' stream); restarts count the total, like amorphous.
- **Molecule clusters** (`create_molecule_cluster_atoms_list`, `initialize/molecules.py`): the box recipe (`molecule_counts`, random rotations, `min_distance` between molecules) with molecule centres uniform in a sphere sized by the density (`_pack_sphere`, no periodic images), centred in a cube of extent + `vacuum`, `pbc` False. rng `[seed + j, attempt, i]`; restarts count per formula (`molecule_cluster_override`), like boxes.

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
