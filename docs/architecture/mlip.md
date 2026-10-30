# MLIP training and evaluation

`src/alomancy/mlip/`: the trainer base class, the MACE and SevenNet backends, checkpoint evaluation and prediction. Ranking and the best-model copy are in `core_workflow.md`.

## `ALomancyTrainer` (`mlip/base.py`)

A backend implements only `fit(train, valid, test, seed, fit_dir, *, isolated_atom_energies) -> model path or None`, `model_path(fit_dir)` and `get_calculator(model_path, *, device=None)`. The base supplies:

- `train(train_path, valid_path, test_path, seed, *, fit_dir, elements, isolated_atom_energies) -> str | None`: fit, then `evaluate`, then `cleanup`.
- `resolve_isolated_atom_energies`: the backend setting named by `ISOLATED_ATOM_ENERGIES_KWARG` (MACE `E0s`, SevenNet `isolated_atom_reference_energies`) wins, else the DB's `IsolatedAtom` energies, else `ValueError` when `elements` is set. Dicts are coverage-checked, then `format_isolated_atom_energies`'d. Shared code calls these energies `isolated_atom_reference_energies`, never a backend's own term; `isolated_atom_reference_energies_by_z` validates symbol/Z keys for any backend.
- `evaluate`: writes `{split}_pred.xyz` (`model_energy`, `model_forces`, `model_stress`; `read_predictions` passes `model_stress` on to the database as `model_stress_loop_<l>_fit_<f>` for the stress parity panel) and a checksummed `evaluation_metrics.json`, through `get_calculator`.
- `output_paths`/`read_existing_result` (restart; metrics are never returned by `train`, always re-read checksum-verified), `deployable_model_path` (what `results/best_model/` gets), `cleanup_paths`, `report_section`, `effective_kwargs`.
- `training_history(fit_dir, seed) -> TrainingHistory | None`: the fit's per-epoch errors as one backend-independent table (`epoch`, `split`, `loss`, `mae_e_per_atom`, `mae_f`) plus optional `stage_two_epoch`/`selected_epoch` markers. `plot_training_curves(..., trainer)`, the report's `trainer_section` and `alomancy results --replot` read curves only through it, never a backend's own log files.

The remote worker is the module-level `run_training(trainer, config, train/valid/test paths, seed, *, fit_dir, fit_name, elements, isolated_atom_energies)` (ExPyRe pickles by reference; never a method). `get_trainer(name, config)` builds a registered trainer (`register("mlip_trainer", name, module, trainer_class=...)`). `read_predictions(fit_dir)` reads the prediction files for DB storage.

**`CalculatorSpec`** (`mlip/base.py`: trainer name, training config, model path; built by `ActiveLearningWorkflow.calculator_spec`) is how generators get the trained model: plain data that pickles into remote jobs, where `spec.build(device=None)` calls the trainer's `get_calculator`. See `structure_generation.md`.

`utils/training_schedule.py`: `dynamic_epochs`/`resolve_epochs`, the backend-independent epoch rule `ceil(200000 × batch_size / n)` clamped to [20, 300]; `None`/`"dynamic"` → the rule, a positive int → as is.

## MACE (`mlip/mace/trainer.py`)

`MaceTrainer`: MACE arguments from `_MACE_KWARGS_DEFAULTS` + `training.mace_kwargs` only.

- **Epochs**: `mace_kwargs.max_num_epochs` is `"dynamic"` (default) or a positive int, resolved by `resolve_epochs(configured, batch_size, n_train + n_valid)` (validation counted, so counts match earlier versions). `start_swa = floor(0.8 × epochs)`. Both are written to `resolved_mace_epochs.json`; `training_history` returns its `start_swa` as `stage_two_epoch` (None for older fits: no Stage 2 line, never guessed).
- **`E0s` reach MACE as a string keyed by atomic number** (`_mace_e0s_arg`). MACE's parser (`mace.tools.scripts_utils.get_atomic_energies`) calls `E0s.lower()`/`ast.literal_eval` and indexes by Z, so a dict (from the DB as `{symbol: eV}`, or written with symbol or Z keys) becomes `"{6: -155.1, ...}"` after the coverage check. Strings (`"average"`, a `.json` path) pass through. A test feeds the converted value to MACE's real parser.
- **Stress training is opt-in** (`mace_kwargs.compute_stress`, default false): `_apply_compute_stress_defaults` sets `stress_key="REF_stresses"` and `loss="stress"` with `setdefault`, never overriding an explicit choice. A bare `stress_key` has no effect: MACE only uses stress with `loss` `"stress"`/`"huber"`/`"universal"`. Kept separate from stress harvesting (default on, `high_accuracy_evaluation.md`) so a loss change never applies silently to a running production run.
- **Evaluation uses the uncompiled `{name}_stagetwo.model`** (`model_path`), never the TorchScript-compiled one: compiled force evaluation raised `RuntimeError: ... Global alloc not supported yet` for every multi-atom structure on one GPU/CUDA/PyTorch combination, silently reducing every parity plot to one point. MACE writes the uncompiled model unconditionally but the compiled one inside a bare `except Exception: pass`, so restart checks (`output_paths`) gate on the uncompiled model; `deployable_model_path` = `{name}_stagetwo_compiled.model` or None.
- `cleanup_paths`: `checkpoints/` plus the never-read stage-one `<name>.model`/`<name>_compiled.model` (~88 MB each), once the compiled stage-two model exists (a fit drops from ~384 MB to ~210 MB).
- **mace-torch pins e3nn 0.4.4.** e3nn ≥ 0.5 stores a `torch.jit.script` function on `SphericalHarmonics`, which `AveragedModel`'s `deepcopy` at Stage Two can't copy (`PickleError: ScriptFunction cannot be pickled`; yun_an loop 36, after `sevenn 0.13` pulled in e3nn 0.6). `pyproject.toml` pins `e3nn==0.4.4` and `sevenn<0.11`. e3nn 0.4.4 can't load its constants under torch ≥ 2.6 without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, which both trainers set.
- **triton**: PyTorch builds with `torch._native` JIT-compile triton kernels, which needs `python3-dev` headers that HPC compute nodes usually lack. Set `TRITON_CACHE_DIR` to a persistent path in the training section's `pre_cmds` and pre-warm it once in an interactive GPU job (not the login node, whose driver may be too old): `TRITON_CACHE_DIR=/fhi/home/jholl/.triton_cache python -m mace.cli.run_train --help`. The trainer warns at import when `torch._native` exists and `TRITON_CACHE_DIR` isn't set.

## SevenNet (`mlip/sevennet/trainer.py`)

`SevenNetTrainer` (`training.trainer: sevennet`). We read the data and build the configs; the epoch loop is SevenNet's own `processing_epoch_v2` (what `sevenn train` runs since 0.9.6; `train_v1` can't take our recorder), given our `Trainer`, loaders (`trainset`/`validset`) and `ErrorRecorder`, with `per_epoch=epochs` and `Logger().switch_file(fit_dir/"log.sevenn")` (the Logger is a process-wide singleton).

- `sevennet_kwargs` holds SevenNet's `input.yaml` sections `model`/`train`/`data` (SevenNet key names, each deep-merged over `_SEVENNET_KWARGS_DEFAULTS`) plus optional `isolated_atom_reference_energies`. `data.shift` (those energies as a Z-indexed list), `model.chemical_species` and `train.random_seed` (the fit seed) are derived and raise if set. `train.epoch` defaults to `"dynamic"`; `is_train_stress` is opt-in.
- `_read_graphs` builds graphs in memory from `REF_energy`/`REF_forces`/`REF_stresses` (`_sevenn_stress`: ASE Voigt → SevenNet order, sign flipped; missing stress → NaN, ignored by `_FiniteOnly` in the stress loss and the recorded TotalLoss). `IsolatedAtom` structures stay in training: unlike MACE, SevenNet doesn't pin a lone atom's energy to the shift.
- After the loop `_keep_one_model` saves `checkpoint_best.pth` (else `checkpoint_<epochs>.pth`) as a slim `training_sevennet.pth` (`model_state_dict`/`config`/`epoch`, ~0.5 MB) and deletes every `checkpoint_*.pth` (also in `cleanup_paths`, for copies ExPyRe synced back). Fit files: `training_sevennet.pth`, `lc.csv` (per-epoch log read by `training_history`), `log.sevenn`, `training_sevennet_fit.json`. Stale `lc*.csv`/checkpoints are removed before training so a retry doesn't write `lc0.csv`.
- Remote nodes need `sevenn` installed.

## Evaluation records (`mlip/evaluation.py`)

- `prediction_metrics`: per-split metrics from predicted vs reference atoms, broken down per domain (`domains`) and per `config_type` (`config_types`), with stress errors (`mae_stress`/`rmse_stress`, eV/Å³) wherever a structure has both `REF_stresses` and `model_stress` (stress never makes a split incomplete), plus a `data_id` hash of the split (schema_version 2).
- `save_evaluation`/`read_evaluation`: a fit's `evaluation_metrics.json`, schema-versioned, with the model's SHA-256 so metrics for a since-changed model are rejected.
- `rank_committee`, `best_fit_test_metrics`, `metrics_by_loop(name, *, strict, expected_fits)` (the MAE-vs-loop table: live run `strict=False` skips and warns, `alomancy results --replot` uses `strict=True` and raises), `check_quality_gate`.
- Energy MAE is `mae_e_per_atom` (eV/atom) everywhere; never plot `mae_e` (per structure), which is similar in magnitude to `mae_f` and confusing.

## Predictions and local cleanup

- After training, `evaluate` (on the node) writes `train_pred.xyz`/`valid_pred.xyz`/`test_pred.xyz` in `results/<base>/training/fit_<i>/` with generic `model_energy`/`model_forces` keys. `_store_predictions_and_cleanup` (`core/active_learning_workflow.py`) reads them (`read_predictions`) and stores them in the DB (`store_model_predictions`, keys `model_energy_loop_{N}_fit_{i}`/`model_forces_loop_{N}_fit_{i}`).
- **Checkpoint cleanup also runs locally**: ExPyRe's mid-training sync pulls the remote stage dir back with `delete=False` (additive), so every checkpoint MACE ever wrote accumulated locally (5–7 GB per fit). `_store_predictions_and_cleanup` calls `trainer.cleanup(fit_dir)` on each local fit.
- `mlip/predict.predict_with_model` is the remote prediction worker used by `ActiveLearningWorkflow.predict`.

## Tests

The generic path is tested backend-free with a tiny `EMTTrainer` (`tests/mlip_train_tests/test_base.py`, and end to end in `tests/core_tests/test_active_learning_workflow.py`). `mace` is a real dev dependency, imported directly in tests. Ranking tests (`TestRankCommittee` in `tests/mlip_train_tests/test_mace_training.py`, plus `tests/mlip_train_tests/test_checkpoint_evaluation.py`) build real `evaluation_metrics.json` files with `save_evaluation(prediction_metrics(...))` and a small placeholder model file.
