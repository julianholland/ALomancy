# Config file advice

Notes from checking real run configs: what went wrong, why, and the pairings between
keys that the schema can't check. This started with yun_an
(`~/alomancy_runs/yun_an/standard_config.yaml`).

## General rules of thumb

### Pairings the validator doesn't check

- **`force_ceiling` < `train_filter.max_force`.** Both measure the largest per-atom
  force norm.
  - A ceiling above the filter pays for DFT on structures the filter then throws away.
  - A ceiling far below it (0.05) fully relaxes every AL structure. That is about
    15× the DFT cost, and it moves the structures to minima, away from the
    high-uncertainty geometry the committee picked.
  - Good value: just below the filter, e.g. 4.5 with `max_force: 5`.
- **`max_go_time` ≥ `max_num_of_relax_steps` (default 200) × seconds per BFGS step.**
  - On raven for 20-atom Li–B–I–P–S cells, a step takes 90–170 s. Relaxing from MD
    frames to 0.05 eV/Å takes about 100–130 steps.
  - A job that hits the Slurm limit is lost completely, partial relaxation included.
- **`md_kwargs.steps × timestep_fs` must fit in `structure_generation.max_time`.**
  - A run cut short by the time limit gives fewer candidates than
    `num_of_structures_to_generate`.
  - Snapshots are taken every `steps × num_of_md_starts // num_of_structures_to_generate`
    steps, so the frames a run never reaches are simply missing.
  - NPT is much slower per step than NVT.
- **`ensemble: npt` needs a model that has learned stress.**
  - The barostat is driven by the committee's predicted stress. With
    `mace_kwargs.compute_stress` off, that stress is untrained.
  - Harvesting stress (`qe_kwargs.control.tstress: true`) only records it; it doesn't
    train it.
  - Collect stress for a few loops first, then turn on `compute_stress`, then switch
    to NPT.

### Units and settings that are easy to get wrong

- **`md_kwargs.pressure` is in GPa.** 1 atm is about 0.0001 GPa; 1.01 is about
  10,000 atm.
- **`md_kwargs.friction` is in ASE inverse-time units, not 1/fs.**
  - 0.002 is a thermostat time of about 5 ps. ASE's own examples use 0.01/fs, which
    is 0.1 here.
  - MD starts from 0 K (no velocities are drawn), so a weak thermostat spends the
    first few ps heating up.
  - **Under `ensemble: npt` `friction` is ignored.** `LangevinBAOAB` uses its own
    default `T_tau` = 50 × timestep.
- **Filter config_type names must match real config types.**
  - Nothing checks `selectable_configs`, `exempt_config_types` or
    `target_config_types` against the database. A typo silently matches nothing.
  - The committee workflow tags its picks `high_sd`; initialization types are
    `init_*` and `IsolatedAtom`.
  - `selectable_configs` always includes `high_sd`, whether listed or not.

### Settings that have no effect

- **QE's own relaxation settings are unused.** `control.nstep`, `etot_conv_thr` and
  `forc_conv_thr` do nothing: relaxations are ASE BFGS driving QE single points. The
  target is `force_ceiling` and the step cap is `max_num_of_relax_steps`.
- **Section `name:` keys are ignored.** Each section's name is fixed; leaving them in
  gives a warning.

### Test filter

- A test structure that fails `test_filter` stays in the test split. It is lost to
  both test and training.
- A strict test filter therefore throws away most AL test picks, and the test set
  stops growing.

## yun_an: what went wrong in loops 40–41 (October 2026)

The config was edited on Oct 17 between loops 39 and 40
(`standard_config.yaml.bak` → `standard_config.yaml`):

| Key | Before | After |
|---|---|---|
| `force_ceiling` | 10 | 0.05 |
| `max_go_time` | 10H | 3H (back to 10H by Oct 21) |
| `train_filter.max_force` | 10 | 5 |
| `ensemble` | nvt | npt, `pressure: 1.01` |
| `num_of_md_starts` | 20 | 25 |
| `tstress` | false | true |

### What happened

- **Loop 40 DFT: 72 of 81 jobs failed.**
  - With a 0.05 ceiling, a relaxation needs about 100 BFGS steps (median 98, max
    133).
  - The median job took 1.8 h and the longest 2.9 h. In loops 37–39 it was 6–8 steps
    and about 0.25 h.
  - Most jobs hit the 3H limit and their partial relaxations were discarded.
  - 10 failed in QE itself:
    - 7 "fft order too large", likely from cells blown up by the NPT barostat;
    - 3 "nsym == 0".
  - Only 9 structures came back.
- **Loop 40 MD:** 24 of 25 runs hit the 2H limit after 0.4–4 ps of the planned 6 ps.
  This gave 148 frames, against a target of 500 and below the 200 (2 × per-loop)
  warning level. 45% of the frames had short bonds.
- **The training set shrank from 1972 to 1696.** With `max_force` cut from 10 to 5,
  initialization dimers, trimers and stretch/compress structures were dropped (116
  dimer, 62 trimer, 398 stretch_compress), plus 328 `high_sd`.
- **Loops 37–39 had wasted DFT.** With `force_ceiling` 10, their picks came back with
  a median max force of about 7 eV/Å. Under the new `max_force: 5` most of them are
  now filtered out.
- **Loop 41 crashed on staging.**
  - rsync reported "No space left on device" on the fhi-raccoon `/fhi` share.
  - The retry then failed with "remote job rundir already exists", because of the
    partial rsync.
  - If this happens again, clear the `run_training_*` dirs under
    `~/alomancy_expyre_gpu/unkownhost-_work_home_jholl_alomancy_runs_yun_an/`.
- **The test set barely grows.** `test_filter.max_force: 5` excludes 80–100% of each
  loop's AL test picks.

## yun_an: check of the current file (Oct 21, before resubmitting)

The file builds without errors. The only warning is about the ignored `name:` keys.

### Fix before submitting

1. **`force_ceiling: 0.05` → about 4.5.**
   - At 0.05 each loop costs about 100 structures × 2 h of DFT and returns relaxed
     structures.
   - With `max_go_time: 10H` the jobs should now finish, so it no longer fails. It is
     just expensive and less informative.
2. **`pressure: 1.01` is 1.01 GPa.** If 1 atm was meant, use `0.0`, which keeps the
   barostat on at about 0 GPa.
3. **NPT with untrained stress.**
   - The committee has never been trained on stress, and almost no stored structure
     has `REF_stresses` (tstress was false until loop 40).
   - Suggestion: go back to `ensemble: nvt` for now and keep `tstress: true` so
     stress is collected.
   - Later, add `mace_kwargs.compute_stress: true` and return to NPT.
   - Before turning it on, check how MACE weights structures without stress.
4. **MD will hit `max_time: "2H"` again.** 20000 steps × 0.3 fs = 6 ps did not fit
   under NPT. Either raise `max_time` (about 6H) or cut `steps` (about 6000). NVT on
   its own may fit in 2H; loops up to 39 did.
5. **The run venv still has `deduplicate-lib` 0.1.2.** The redundancy rework needs
   0.1.3:
   ```bash
   uv pip install --python /home/jholl/.venvs/al_install/bin/python "deduplicate-lib>=0.1.3"
   ```

### Worth deciding

6. **`exempt_config_types: ["init_dimer", "init_trimer"]` leaves out
   `init_stretch_compress`.**
   - At `max_force: 5`, 398 of its 788 structures are still dropped. They are the
     large-strain data.
   - Add it if those should be kept. All three exempt gave a 2274-structure training
     set on a copy of the database, against 1698 with none.
7. **`selectable_configs: ["init_amorphous", "amorphous"]`: no structure has
   `config_type` "amorphous".**
   - The database has `high_sd`, `init_*` and `IsolatedAtom`, so that entry matches
     nothing. It does no harm.
   - AL picks already seed MD: `seed_selection.py` always adds `"high_sd"` to
     `selectable_configs`. Seeds come from the 87 `init_amorphous` structures plus every
     `high_sd` pick.
8. **`friction: 0.002` is a weak thermostat** (about 5 ps, see above). It only matters
   under NVT. 0.02 gives about 0.5 ps.
9. **`test_filter`** (`max_force: 5`, formation energy up to 2.5) keeps throwing away
   most AL test picks.
   - Raising it to match what the model should be judged on (e.g. `max_force: 10`)
     lets the test set grow.
   - The training filter can stay at 5.
10. **QE "nsym == 0" failures:** `system.nosym: true` probably avoids them. AL
    amorphous cells have no symmetry to exploit anyway. Not verified.
11. **`electron_maxstep: 999`:** a non-converging SCF can use up a whole walltime.
    About 200 is enough with `local-TF` mixing.

### Cosmetic

- Remove the four section `name:` keys (they give a warning).
- Remove the stale "E0s" comment.
- `nstep`, `etot_conv_thr` and `forc_conv_thr` are inert (see above).
- `energy_key` and `forces_key` are already ALomancy's defaults.

### Not needed

- No HPC reinstall is needed for the unreleased changes: no remote function changed
  since 1.0.2.
