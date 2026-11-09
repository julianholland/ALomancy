# High-accuracy evaluation (reference labels)

`src/alomancy/high_accuracy_evaluation/`: the orchestrator (`high_accuracy_calc_interface.py`), the QE/VASP backends (`dft/run_qe.py`, `dft/run_vasp.py`) and the ORCA backend (`orca/run_orca.py`), all registered under `dft_evaluator`, the level-of-theory record (`level_of_theory.py`), plus the shared remote workers in `utils/dft_utils.py`.

The category is still called `dft_evaluator` and the per-structure keys still say `dft_*` (`dft_input_digest`, `dft_wall_time_s`, `dft_converged`): restart files depend on those names. ORCA's wavefunction methods (MP2, coupled cluster) aren't DFT; read "DFT" in these names as "reference evaluator".

## Orchestrator: `high_accuracy_calc_interface.high_accuracy_evaluation`

Calculator-agnostic; resolves the backend's `sp`/`go` workers through `registry.resolve("dft_evaluator", evaluator)`. Before submission:
1. `filter_structures_by_min_bond_distance` drops anything with a pair closer than 0.5 Å (also covers initialization, which has no upstream filter);
2. `filter_blown_up_structures` drops periodic cells whose volume is above `high_accuracy_evaluation.max_vdw_volume_ratio` (default 10, null = off; `resolve_max_vdw_volume_ratio`, validated at construction) × the summed van der Waals sphere volumes of their atoms (`vdw_volume_ratio`; 1.5 × the covalent radius where ASE has no vdW radius), or non-finite. Non-periodic structures, `GAS_PHASE_CONFIG_TYPES` (IsolatedAtom, init_dimer, init_trimer: vacuum boxes by design, returned periodic by DFT) and the config types listed in `high_accuracy_evaluation.vdw_exempt_config_types` (`resolve_vdw_exempt_config_types`, for structures tagged as meant to have vacuum, e.g. molecules in a box) are exempt; the exemption is by config type, not geometry, since a geometric test would also pass the blown-up cells. Warning `blown_up_excluded`, suggesting a small positive MD pressure, a higher ratio for genuine low-density crystals, or exempting the config type. Both filters run **before** the restart check, which matches by digest (below), so a structure they exclude never shifts what counts as done. The ratio limits here and in MD (`md_kwargs.max_volume_ratio`) share `resolve_ratio` (a finite number above 1, or null). Why: in yun_an loop 45 such cells ran QE out of memory, timed it out or broke its symmetry analysis (`nsym == 0`); healthy cells sit near 1, molecular crystals (P4, Li2S8) up to about 10, the failed ones at 30 and far above;
3. `check_periodicity` raises `ValueError` before anything is submitted when the evaluator can't take the structures' periodicity (registry entry points `supports_periodic` / `supports_non_periodic`: ORCA has no periodic boundary conditions, so any `pbc` True raises), with counts per `config_type` and a pointer to the cluster initialisation blocks;
4. `filter_structures_by_atom_count` (`utils/clean_structures.py`) drops structures above the evaluator's size cap (entry point `max_num_of_atoms(config)`: ORCA returns `orca_kwargs.max_num_of_atoms_correlated`, default 20, for correlated and multireference methods only; QE/VASP return None), warning `too_many_atoms_for_method`. Like the other filters it runs before the restart check;
5. `wrap_structures_into_cell` (`utils/clean_structures.py`) wraps the survivors into their cell (MD drifts atoms out; QE/VASP expect positions in the reference cell). `Atoms.wrap()` is called unconditionally: it mutates in place, keeps `info`/`arrays`/the calculator, and is a no-op on non-periodic axes.

**GO and SP share one batch**: one `ase_remote_submitter` call per batch dir `batch_N` (`N` = number of existing `batch_*` dirs, so a restart never collides with a partial earlier run). `per_structure_function` picks `run_go` or `run_sp` from each structure's `needs_relaxation`; job names carry a `go_`/`sp_` prefix. If any structure needs relaxation, the shared walltime comes from `max_go_time` instead of `max_time` (an SP job finishing early under a long walltime is harmless; a GO job under an SP walltime is not).

**A restart matches results to structures by digest, not position.** Before submission each structure gets `atoms.info["dft_input_digest"]` (`dft_input_digest`: sha256 of the numbers, `pbc`, and the positions and cell formatted `%.8f`, as extxyz writes them, so the digest of a structure held in memory equals that of the same structure read back from file; `dataset_curation.geometry_digest` rounds floats and changes for most structures after one round trip). The workers keep `atoms.info`, so every `<name>.xyz` names its input. `_still_to_evaluate` submits only the structures with no result: jobs finish in any order, and the old rule (`structures[len(found):]`) resubmitted finished structures and skipped unfinished ones whenever a middle job was still running when the run stopped. Results matching no current structure (filter or DFT settings edited between restarts, which isn't supported) are kept, with warning `dft_inputs_changed`. Results without a digest (written before this) fall back to the positional rule.

Results are collected from `results/<base>/high_accuracy_evaluation/batch_*/`. The section name is fixed to `high_accuracy_evaluation`, so the submit path and the collection glob always agree (a user-set name that differed used to make the glob find nothing and mark the phase done with zero structures).

**Every result carries its level of theory** (`stamp_level_of_theory`, §"Level of theory" below): the results, the consolidated `high_accuracy_eval_results.xyz` and the GO starting frames. Stamped on the driver side, so changing it needs no remote reinstall.

**It never marks itself done with zero results after submitting jobs**: if structures were submitted and no results exist at all (this batch or earlier), it raises `RuntimeError` instead of writing `high_accuracy_eval.done` and an empty `high_accuracy_eval_results.xyz`; a partial failure completes normally.

## Workers (`utils/dft_utils.py`, run remotely)

- `_run_go` reads `fmax` (`job_dict.get("fmax", 0.05)`; AL loops inject `force_ceiling`, see `core_workflow.md`) and `max_num_of_relax_steps` (default 200) for its BFGS budget. Not converging within the budget isn't a failure: it warns and keeps the best structure found with `atoms.info["geometry_converged"] = False`; `train_filter.max_force` is the backstop.
- **`_run_go` keeps the starting frame.** Step 0 of the BFGS is a full DFT calculation of the unrelaxed geometry. `_run_go` computes it before `opt.run` (BFGS reuses the cached result, so it costs nothing) and `_snapshot` copies it from `calc.results` only: asking the live calculator for a property it didn't compute (QE stress with `tstress: false`) would rerun pw.x. If the GO moved the structure (`opt.nsteps > 0`), it is written to `<out_dir>/<name>_initial.xyz` with `geometry_step = 0` and `geometry_initial = True`, and without the run statistics below (so the report doesn't count it twice); the final frame gets `geometry_step = opt.nsteps`. The orchestrator collects these files into `results/<base>/high_accuracy_eval_initial_frames.xyz` (`_collect_initial_frames`, read back by `read_initial_frames`); results from before this change have none. `add_to_dataset` adds them to the split of their final frame (matched by `job_id`), initialization adds them with its structures, and the train/test filters decide whether they are used (most exceed the default `train_filter.max_force`, since GO only starts above `force_ceiling`). `*opt.traj` still holds every frame (energy and forces only); nothing reads it.
- Both workers record `geometry_steps`, `geometry_max_steps`, `geometry_fmax_target` and `dft_wall_time_s` in `atoms.info` for the loop report.
- `refresh_dft_labels` rejects a calculator reporting `converged is False`.
- `_build_srun_command` launches the binary with `srun --mem=0` (see `remote_submission.md`).

## QE (`dft/run_qe.py`)

- **`qe_kwargs` merges into QE's defaults per namelist** (`get_qe_input_data`), key by key. A whole-dict merge used to let `qe_kwargs: {control: {tstress: false}}` drop `control.tprnfor`/`calculation`; QE then printed no forces and every DFT job of a production loop failed with `forces not present in this calculation`. Runs remotely: changes need a reinstall.
- `scf_must_converge=True` by default; `tstress: True` by default (stress harvesting).
- Pseudopotentials come from the HPC profile (`pp_path`, `pseudo_dict`); a missing key fails as a `KeyError` on the node (roadmap: a central pseudopotential repository with a pre-run check).
- QE's `Espresso` handles non-periodic cells natively.

## VASP (`dft/run_vasp.py`)

- ASE's `Vasp` reads POTCAR locations only from `VASP_PP_PATH`; `create_vasp_calc_object` sets `os.environ["VASP_PP_PATH"] = hpc["pp_path"]` before building the calculator. `pp_path` must contain `potpaw_PBE`/`potpaw_LDA`. Elements without a `pseudo_dict` override use the unsuffixed POTCAR; a missing element fails with `No pseudopotential for <element>!`.
- **VASP needs full periodicity**: `_ensure_fully_periodic` pads non-periodic structures (isolated atoms, dimers, trimers) with vacuum and sets `pbc=True` before building the calculator.

## ORCA (`orca/run_orca.py`)

A molecular code: no periodic boundary conditions (the orchestrator's `check_periodicity`, and `create_orca_calc_object` again on the node), DFT plus wavefunction methods. Settings in `orca_kwargs` (keys `ORCA_KWARGS_DEFAULTS`; anything else raises at construction): `orcasimpleinput` (required, the `!` line), `orcablocks` (extra `%` blocks), `charge` (default 0), `nprocs`, `maxcore` (MB per process), `max_num_of_atoms_correlated` (default 20, null = no cap). Runs remotely: changes need a reinstall.

- **Parallelism is in the input**, the "in-input" model (below): ORCA must be started by its absolute path (`hpc.orca_path`, no srun) and starts its own `mpirun`, so OpenMPI comes from the profile's `pre_cmds`. `build_orca_input` writes exactly one `%pal nprocs` and one `%maxcore`. Their values come from `resolve_parallel_resources`: the user's if set (`orca_kwargs.nprocs`/`maxcore`, a `%pal`/`%maxcore` block in `orcablocks`, or a `PALn` keyword; the same setting given twice with different values raises), else the profile's. A user value wins over the profile but raises when the job can't hold it.
- `EnGrad` (forces, read by ASE from `orca.engrad`) and `SCFConvergenceForced` (an unconverged SCF aborts instead of labelling) are added when missing.
- **Charge and multiplicity per structure** (`structure_charge_and_multiplicity`). Charge: `atoms.info["charge"]`, else `orca_kwargs.charge`. Multiplicity: `atoms.info["spin_multiplicity"]`; else, for a neutral single atom, its ground state (`ase.data.ground_state_magnetic_moments` + 1: H 2, O 3, N 4); else the lowest the electron count allows. An impossible combination raises. ORCA's default `mult=1` would abort on every odd-electron atom.
- **Method classes** (`classify_orca_method`, from the keywords): `semi_empirical` (xTB), `mean_field` (HF, DFT, `-3c` composites), `correlated` (MP2/CC families with `RI-`/`DLPNO-`/`SCS-`/`F12` affixes, and double hybrids, which cost about as much as MP2), `multireference` (CASSCF, NEVPT2, MRCI; also from a `%casscf`/`%mrci` block). The atom cap applies to the last two. `validate_kwargs` warns `orca_method_unrecognised` when no method keyword is known (no cap applied), and `numerical_gradients` for CC and multireference methods, whose forces may need finite differences (about 6N energies per structure; check the manual for the ORCA version in use).
- `run_go_orca` uses the shared ASE-BFGS `_run_go`, like QE/VASP.

## Parallel resources (`utils/dft_utils.py`)

`resolve_parallel_resources(node_info, *, nprocs=None, mem_per_proc_mb=None)` is where a backend turns the profile's `node_info` (and the user's overrides) into a `ParallelResources` (`nprocs`, `mem_per_proc_mb`, `threads_per_proc`). Defaults: `ranks_per_system` processes, 75% of `max_mem_per_node` (`parse_memory_mb`: `"120GB"`, `"90G"`, MB as a number) split between them. An override raises when it can't fit: jobs always run on one node (`configs/remote_info.py`), so at most `ranks_per_node × threads_per_rank` processes, and together within `max_mem_per_node`.

Backends follow one of two models; the ASE calculator stays the stock one either way:
- **launcher**: resources go in the command. QE and VASP through `_build_srun_command` (unchanged); FHI-aims, CASTEP, CP2K (`mpiexec … cp2k_shell`) and DFTB+ would too.
- **in-input**: resources go in the input file and the binary starts bare. ORCA (`%pal`, `%maxcore`); Gaussian would use `%nprocshared`/`%mem`.

## Level of theory and adding an evaluator

`level_of_theory.py`'s `LevelOfTheory` records how a label was made, in fields that fit any electronic-structure code ASE drives. Each backend's entry point `level_of_theory(config, hpc)` fills it from its *effective* settings. A field it can't determine is None (unspecified), never a guess. The orchestrator stores it as `atoms.info["REF_level_of_theory"]` (canonical JSON) and `atoms.info["REF_level_of_theory_id"]` (12 hex characters of the sha256 of the identity fields). The run summary shows `label()`.

Identity fields (everything but the precision row) decide which labels can share a training set. Precision fields are recorded but don't change the id. Before initialisation, `ActiveLearningWorkflow._check_database_against_evaluator` makes one pass over the database. It warns `level_of_theory_mismatch` for ids that differ from the evaluator's, naming the differing fields. Precision-only differences and unstamped (older or imported) structures are logged at info. With a non-periodic evaluator it raises when the database holds periodic structures, which would seed periodic candidates.

Where each code's settings go (QE, VASP, ORCA implemented; the other columns are a starting point taken from their ASE calculators, to check against each code's manual when it is added):

| field | QE | VASP | ORCA | FHI-aims | CASTEP | CP2K | Gaussian | DFTB+ |
|---|---|---|---|---|---|---|---|---|
| `method_class` | dft (hf for `input_dft: hf`) | dft | from keywords | dft / hf | dft | dft / hf | from route | semi_empirical |
| `method` | `input_dft` (else from the PPs: None) | `xc`, else `LHFCALC`+`HFSCREEN` (HSE06/HSE03/PBE0), `METAGGA`, `GGA` tag, else the POTCAR set | method keyword | `xc` | `xc_functional` | `xc` | `method` | DFTB2 / DFTB3 (`ThirdOrderFull`) |
| `rung` | `functional_rung` | `functional_rung` | `functional_rung` | `functional_rung` | `functional_rung` | `functional_rung` | `functional_rung` | None |
| `dispersion` | `vdw_corr` + `dftd3_version`; nonlocal `input_dft` | `IVDW`, `LUSE_VDW` | `D3BJ`/`D3ZERO`/`D4` keywords, `-D3BJ` suffix, built into `-3c` | `many_body_dispersion`, `vdw_correction_hirshfeld` | `sedc_apply`/`sedc_scheme` | `&VDW_POTENTIAL` | `EmpiricalDispersion` | `Dispersion` block |
| `hubbard_u` | `Hubbard_*` / `lda_plus_u` | `LDAU*` | {} | `plus_u` | `hubbard_u` block | `&DFT_PLUS_U` | — | — |
| `basis_type` | plane_wave | plane_wave | gaussian | numeric_atom_centred | plane_wave | gaussian_plane_wave | gaussian | slater_koster |
| `basis` | None | None | basis keyword | species defaults (`light`/`tight`/…) + `tier` | None | `basis_set` | `basis` | SK set (`3ob-3-1`) |
| `cutoff_eV` / `density_cutoff_eV` | `ecutwfc` / `ecutrho` (Ry → eV) | `ENCUT` / None | None | None | `cut_off_energy` (or `basis_precision`) | None / `cutoff` | None | None |
| `core_treatment` | from the PP file names (paw/ultrasoft/norm_conserving) | paw | all_electron; ecp with def2 (Z > 36) | all_electron | from the pspots (usually ultrasoft, on the fly) | from `pseudo_potential` (GTH: norm_conserving) | all_electron / ecp | in_parametrisation |
| `pseudopotentials` | `pseudo_dict` | `pp` set + `pseudo_dict` (`setups`) | `{"Z>36": "def2-ECP"}` with def2 | None | pspot names | `pseudo_potential` | ECP names | None |
| `relativity` | in_pseudopotential, spin_orbit (`lspinorb`) | in_pseudopotential, spin_orbit (`LSORBIT`) | `ZORA`/`DKH`/`X2C`, else none | `relativistic` (atomic_ZORA) | in_pseudopotential, `spin_orbit_coupling` | in_pseudopotential | `DKH`/`X2C` | none |
| `spin_treatment` | `nspin`, `noncolin` | `ISPIN`, `LNONCOLLINEAR` | per_structure (by multiplicity), or `UKS`/`RKS`/`ROKS` | `spin` | `spin_polarized` | `uks` | from route / multiplicity | `SpinPolarisation` |
| `solvation` | none | VASPsol (`LSOL`) | `CPCM(...)`/`SMD(...)` | MPE | none | SCCS | `SCRF` | none |
| `approximations` | [] | [] | RI/COSX keywords, aux bases, PNO thresholds, unknown keywords | — | — | `&QS` method | density fitting | — |
| *precision* `k_spacing`, `smearing`, `scf_threshold`, `numerical_settings` | `KSPACING`, `degauss`/`smearing`, `conv_thr` | `KSPACING`, `SIGMA`/`ISMEAR`, `EDIFF`, `PREC`/`LREAL` | —, —, `TightSCF`…, grid | `k_grid`, `occupation_type`, `sc_accuracy_*` | `kpoints_mp_spacing`, `smearing_width`, `elec_energy_tol` | `&KPOINTS`, `&SMEAR`, `EPS_SCF` | —, —, `SCF=Tight`, `Int=` grid | `KPointsAndWeights`, `Filling`, `SCCTolerance` |

`functional_rung` knows the common functionals by name (LDA through double hybrids, several spellings each) and ignores a trailing dispersion suffix; extend `_FUNCTIONALS` as needed.

**Adding an evaluator** (a module under `high_accuracy_evaluation/`, registered in `registry.py` under `dft_evaluator`):
1. `sp`/`go` workers: module-level functions wrapping `_run_sp`/`_run_go` with a `create_<code>_calc_object(atoms, job_dict, out_dir)`;
2. constants `SUPPORTS_PERIODIC`, `SUPPORTS_NON_PERIODIC`, `MAX_OUTPUT_DIR_LEN`;
3. the parallel model: launcher (`_build_srun_command`) or in-input (`resolve_parallel_resources`), with user overrides that raise when they can't fit;
4. `max_num_of_atoms(config)` (None for no cap), `level_of_theory(config, hpc)` from the table above, `resolve_effective_kwargs`, `report_section`, and optionally `validate_kwargs(config, hpc)` for construction-time checks;
5. its HPC profile key in `configs/hpc_profiles.py` (`_EXECUTABLE_KEYS`) and `alomancy add-hpc`;
6. the changelog: "Remote reinstall needed".

## After DFT: `clean_structures` (`utils/clean_structures.py`)

Sets `config_type` and `al_loop` and removes bad structures. **Stress is harvested by default** into `atoms.info["REF_stresses"]` (Voigt 6-vector, `Atoms.get_stress()`'s shape, what MACE's `stress_key` reader expects) whenever available; capture failures (e.g. non-periodic structures) are tallied into one summary line and never raise. Training on stress is a separate opt-in (`mlip.md`).

## Labels

Energies and forces are `atoms.info["REF_energy"]` and `atoms.arrays["REF_forces"]`. **Never use a bare `"energy"` info key**: ASE moves it to the calculator on extxyz read and it disappears from `info`.
