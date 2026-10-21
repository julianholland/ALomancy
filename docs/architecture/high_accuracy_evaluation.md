# High-accuracy evaluation (DFT)

`src/alomancy/high_accuracy_evaluation/`: the DFT orchestrator (`high_accuracy_calc_interface.py`) and the QE/VASP backends (`dft/run_qe.py`, `dft/run_vasp.py`, registered under `dft_evaluator`), plus the shared remote workers in `utils/dft_utils.py`.

## Orchestrator: `high_accuracy_calc_interface.high_accuracy_evaluation`

Calculator-agnostic; resolves the backend's `sp`/`go` workers through `registry.resolve("dft_evaluator", evaluator)`. Before submission:
1. `filter_structures_by_min_bond_distance` drops anything with a pair closer than 0.5 Å (also covers initialization, which has no upstream filter);
2. `wrap_structures_into_cell` (`utils/clean_structures.py`) wraps the survivors into their cell (MD drifts atoms out; QE/VASP expect positions in the reference cell). `Atoms.wrap()` is called unconditionally: it mutates in place, keeps `info`/`arrays`/the calculator, and is a no-op on non-periodic axes.

**GO and SP share one batch**: one `ase_remote_submitter` call per batch dir `batch_N` (`N` = number of existing `batch_*` dirs, so a restart never collides with a partial earlier run). `per_structure_function` picks `run_go` or `run_sp` from each structure's `needs_relaxation`; job names carry a `go_`/`sp_` prefix. If any structure needs relaxation, the shared walltime comes from `max_go_time` instead of `max_time` (an SP job finishing early under a long walltime is harmless; a GO job under an SP walltime is not).

Results are collected from `results/<base>/high_accuracy_evaluation/batch_*/`. The section name is fixed to `high_accuracy_evaluation`, so the submit path and the collection glob always agree (a user-set name that differed used to make the glob find nothing and mark the phase done with zero structures).

**It never marks itself done with zero results after submitting jobs**: if structures were submitted and no results exist at all (this batch or earlier), it raises `RuntimeError` instead of writing `high_accuracy_eval.done` and an empty `high_accuracy_eval_results.xyz`; a partial failure completes normally.

## Workers (`utils/dft_utils.py`, run remotely)

- `_run_go` reads `fmax` (`job_dict.get("fmax", 0.05)`; AL loops inject `force_ceiling`, see `core_workflow.md`) and `max_num_of_relax_steps` (default 200) for its BFGS budget. Not converging within the budget isn't a failure: it warns and keeps the best structure found with `atoms.info["geometry_converged"] = False`; `train_filter.max_force` is the backstop.
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

## After DFT: `clean_structures` (`utils/clean_structures.py`)

Sets `config_type` and `al_loop` and removes bad structures. **Stress is harvested by default** into `atoms.info["REF_stresses"]` (Voigt 6-vector, `Atoms.get_stress()`'s shape, what MACE's `stress_key` reader expects) whenever available; capture failures (e.g. non-periodic structures) are tallied into one summary line and never raise. Training on stress is a separate opt-in (`mlip.md`).

## Labels

Energies and forces are `atoms.info["REF_energy"]` and `atoms.arrays["REF_forces"]`. **Never use a bare `"energy"` info key**: ASE moves it to the calculator on extxyz read and it disappears from `info`.
