# Formation-energy curation and initial MACE validation

The optional `dataset_curation` policy annotates labelled structures and flags
eligibility across both training and test splits. It preserves the source DFT
energies and forces, the isolated-atom MACE reference, and rejected data.

For each structure it records `REF_formation_energy = REF_energy - sum(N_Z * mu_Z)`,
`REF_formation_energy_per_atom`, `REF_max_force`, `formation_reference_id`,
`domain`, `is_training_eligible`, `filter_reasons`, and a policy fingerprint.
`REF_max_force` is the maximum **atomic vector norm**, in eV/Angstrom. Formation
windows use eV/atom so different system sizes can be compared. The force cutoff
is exclusive; the energy-window endpoints are inclusive. Unknown domains can
be rejected explicitly. Non-finite labels and explicitly unconverged DFT are
always rejected. Absence of convergence metadata does not prove convergence.

Example:

```yaml
dataset_curation:
  reference:
    id: Pd_FCC_reference_version
    status: verified  # use estimated unless the reference has been verified
    chemical_potentials: {Pd: -5.0}  # example only; supply your DFT value
  require_known_domain: true
  default: {max_force: 10.0}
  domains:
    bulk: {formation_energy_per_atom: [-0.5, 2.0]}
    surface: {formation_energy_per_atom: [-0.5, 2.0]}
    dimer: {formation_energy_per_atom: [-0.5, 6.0]}
    trimer: {formation_energy_per_atom: [-0.5, 6.0]}
    cluster: {formation_energy_per_atom: [-0.5, 6.0]}
    isolated_atom: {max_force: 0.05}
```

These example limits are initial choices, not universal physical criteria.
For pure Pd a fixed reference shift leaves the ordering of E/N unchanged.
Filtering can remove unsuitable training examples; shifting energies alone
cannot reduce prediction errors or establish dynamical stability.

## Offline preparation

```bash
python -m alomancy.utils.recover_dft_labels source.xyz old_run/results recovery_v1
python -m alomancy.utils.prepare_dataset recovery_v1/recovered.xyz init.yaml prepared_v1
```

Recovery is intended for the historical `high_sd` label-inheritance defect. It
matches ordered geometries to raw DFT outputs, checks completed/converged OUTCARs,
and recovers energy and forces together. Missing or conflicting matches go to
quarantine. Other labels are retained with an explicit audit limitation.
Provenance includes source hashes and frame indices. Parentage is traced to seed
groups where available. The input file is never rewritten.

Preparation accepts an explicit reference. If it is null, it estimates Pd mu
from the minimum E/N of low-force FCC-derived `bulk_rattle` frames. This narrow
classifier checks the FCC primitive-supercell geometry; it does not assume
`init_MP` means FCC. The estimate is labelled `estimated`, with
`equilibrium_verified: false`. A relaxed FCC equation of state and matching DFT
protocol are needed for an equilibrium reference. ASE documents the
[energy-volume fitting procedure](https://ase-lib.org/examples_generated/tutorials/bulk.html).

Outputs are the annotated full archive, selected and rejected subsets,
train/test files, a per-frame CSV, reference JSON, manifest with file hashes,
and `resolved_init.yaml` pointing at the selected dataset. Existing prepared
versions cannot be overwritten by the CLI. Changing the reference or thresholds
requires a new preparation version and a new results directory; cached models
must not be silently reused with a changed policy.

## Training and evaluation

Set `initialization.grouped_splits: true` and
`mlip_committee.grouped_validation: true`. Identical ordered geometries and
shared `split_group` ancestry stay together; the splitter is stratified by
domain and keeps isolated atoms in training. It does not detect arbitrary
symmetry-equivalent geometries. `reset_extra_splits` removes obsolete operational
split flags on import. `workflow.fixed_test` prevents later acquisition from
expanding the held-out test, and routes known held-out descendants to diagnostics.

`workflow.train_only: true` stops after the initial committee and validation
gate, before structure generation. Exported checkpoints are evaluated on fit,
common validation, and held-out test. `evaluation_metrics.json` records per-domain
metrics, units, dataset identity and checkpoint SHA256. Committee selection uses
validation force MAE and refuses missing or inconsistent validation. With
`require_checkpoint_metrics: true`, historical training-log metrics cannot be
silently presented as final-checkpoint test results. Optional `quality_gate`
limits must pass for every member and required domain before exploration.

Energy MAE is in eV/atom; force MAE is over Cartesian components in eV/Angstrom.
The per-component error metric is different from the atomic-norm curation cutoff.
DFT result boundaries explicitly replace inherited REF labels from the current
DFT calculator. Imported REF labels keep precedence elsewhere. ASE `energy`
remains the current target convention; available `free_energy` is recorded
separately. VASP's [smearing documentation](https://vasp.at/wiki/Smearing_technique)
explains the distinction between extrapolated energies and force-consistent
free energies; protocol/energy conventions must be checked before combining
historical and new datasets.

A low validation MAE is not a stability certificate. Run physical checks such as
a dimer curve, FCC E(V), and short bulk/surface dynamics, then longer checks over
the intended domain. Specialization follows successful general-model validation.

ExPyRe imports remote functions by reference. Deploy this same ALomancy source
revision to the training/DFT environments before submission (see `CLAUDE.md`).
Updating only the local checkout leaves remote jobs using old implementations.
MD/EZGA code does not need modification for this preparation stage.
