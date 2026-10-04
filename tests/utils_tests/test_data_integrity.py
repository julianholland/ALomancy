"""Regression checks for DFT labels, domain curation and split integrity."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from alomancy.database.global_database import GlobalDatabase
from alomancy.utils.clean_structures import clean_structures
from alomancy.utils.dataset_curation import (
    annotate_structure,
    curate_database,
    geometry_digest,
    grouped_split,
    validate_policy,
)
from alomancy.utils.dft_utils import _run_sp


def labelled(positions=None, energy=-8.0, forces=None, **info):
    a = Atoms(
        "Pd2", positions=positions or [[0, 0, 0], [2.5, 0, 0]], cell=[20] * 3, pbc=True
    )
    a.info.update(REF_energy=energy, config_type="init_dimer", **info)
    a.set_array(
        "REF_forces",
        np.zeros((2, 3)) if forces is None else np.asarray(forces, dtype=float),
    )
    return a


@pytest.mark.unit
def test_new_dft_replaces_stale_labels_and_survives_disk(tmp_path):
    a = labelled(energy=1234, forces=[[99, 0, 0], [-99, 0, 0]])
    a.calc = EMT()
    expected_e, expected_f = a.get_potential_energy(), a.get_forces()
    _run_sp(a, str(tmp_path), {"name": "fresh"}, lambda *_: EMT())
    loaded = read(tmp_path / "fresh.xyz")
    result = clean_structures([loaded], "high_sd")[0]
    assert result.info["REF_energy"] == pytest.approx(expected_e)
    np.testing.assert_allclose(result.arrays["REF_forces"], expected_f)


@pytest.mark.unit
def test_import_does_not_replace_dft_with_surrogate_results():
    a = labelled()
    a.calc = SinglePointCalculator(a, energy=100, forces=np.ones((2, 3)))
    assert clean_structures([a], "dimer")[0].info["REF_energy"] == -8
    assert (
        clean_structures([a], "dimer", label_source="calculator")[0].info["REF_energy"]
        == 100
    )


@pytest.mark.unit
def test_invalid_labels_fail_explicitly():
    a = labelled(energy=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        clean_structures([a], "dimer")


@pytest.mark.unit
def test_formation_annotation_preserves_targets():
    a = labelled()
    policy = {
        "reference": {
            "id": "fcc-test",
            "status": "verified",
            "chemical_potentials": {"Pd": -5},
        },
        "domains": {"dimer": {"formation_energy_per_atom": [0, 2], "max_force": 10}},
    }
    validate_policy(policy)
    meta = annotate_structure(a, policy)
    assert meta["REF_formation_energy"] == 2
    assert meta["REF_formation_energy_per_atom"] == 1
    assert meta["is_training_eligible"]
    assert a.info["REF_energy"] == -8


@pytest.mark.unit
def test_missing_mu_and_unverified_reference_rejected():
    with pytest.raises(ValueError, match="verified"):
        validate_policy(
            {
                "reference": {
                    "id": "x",
                    "status": "provisional",
                    "chemical_potentials": {"Pd": -5},
                }
            }
        )
    with pytest.raises(ValueError, match="Missing chemical"):
        annotate_structure(
            labelled(), {"reference": {"id": "x", "chemical_potentials": {"H": -1}}}
        )


@pytest.mark.unit
def test_force_norm_filters_all_splits_and_flags_reset(tmp_path):
    db = GlobalDatabase(str(tmp_path / "db"))
    for split in ["train", "test"]:
        db.add_structures(
            [labelled(forces=[[6, 8, 0], [0, 0, 0]])],
            split=split,
            skip_duplicates=False,
        )
    curate_database(db, {"default": {"max_force": 9}})
    assert len(db.get_train_atoms()) == len(db.get_test_atoms()) == 0
    assert len(db.get_all_as_atoms()) == 2
    curate_database(db, {"default": {"max_force": 11}})
    assert len(db.get_train_atoms()) == len(db.get_test_atoms()) == 1


@pytest.mark.unit
def test_same_geometry_and_parent_never_cross_split():
    atoms = [
        labelled(
            positions=[[0, 0, 0], [2 + i * 0.01, 0, 0]], split_group=f"parent-{i // 2}"
        )
        for i in range(20)
    ]
    duplicate = atoms[0].copy()
    duplicate.info["split_group"] = "parent-9"
    atoms.append(duplicate)
    train, test = grouped_split(atoms, 0.2, 803)
    assert train and test and len(train) + len(test) == len(atoms)
    assert {geometry_digest(a) for a in train}.isdisjoint(
        geometry_digest(a) for a in test
    )
    assert {a.info["split_group"] for a in train}.isdisjoint(
        a.info["split_group"] for a in test
    )
    path = sorted(geometry_digest(a) for a in test)
    assert path == sorted(geometry_digest(a) for a in grouped_split(atoms, 0.2, 803)[1])


@pytest.mark.unit
def test_reference_atom_remains_in_training():
    atoms = [Atoms("Pd", info={"config_type": "IsolatedAtom"})]
    train, test = grouped_split(atoms, 0.5, 803)
    assert train == atoms and not test


@pytest.mark.unit
def test_reference_selection_excludes_hcp_and_retains_isolated_e0():
    from ase.build import bulk

    from alomancy.utils.prepare_dataset import estimate_pd_reference

    isolated = Atoms("Pd", info={"config_type": "IsolatedAtom", "REF_energy": -1.5})
    fcc = bulk("Pd", "fcc", a=3.9).repeat((2, 2, 2))
    fcc.info.update(config_type="bulk_rattle", REF_energy=-5.0 * len(fcc))
    fcc.set_array("REF_forces", np.zeros((len(fcc), 3)))
    hcp = bulk("Pd", "hcp", a=2.75, c=4.5)
    hcp.info.update(config_type="init_MP", REF_energy=-6.0 * len(hcp))
    hcp.set_array("REF_forces", np.zeros((len(hcp), 3)))
    reference = estimate_pd_reference([isolated, hcp, fcc])
    assert reference["chemical_potentials"]["Pd"] == -5.0
    assert reference["status"] == "estimated"
    assert reference["equilibrium_verified"] is False
    assert reference["preselection_isolated_atom_energy"] == -1.5


@pytest.mark.unit
def test_recovery_checks_convergence_and_replaces_both_labels(tmp_path):
    from ase.io import write

    from alomancy.utils.recover_dft_labels import recover_dataset

    a = labelled(energy=123.0, forces=[[99.0, 0, 0], [-99.0, 0, 0]])
    a.info["config_type"] = "high_sd"
    source = tmp_path / "source.xyz"
    write(source, [a])
    a.calc = SinglePointCalculator(a, energy=-8.0, forces=np.zeros((2, 3)))
    raw = tmp_path / "raw/al_loop_0/high_accuracy_evaluation/batch_0/ase_output_0"
    raw.mkdir(parents=True)
    write(raw / "high_accuracy_evaluation.xyz", a)
    (raw / "OUTCAR").write_text(
        "aborting loop because EDIFF is reached\nGeneral timing and accounting\n"
    )
    summary = recover_dataset(source, tmp_path / "raw", tmp_path / "recovered")
    assert summary["recovered_high_sd"] == 1
    recovered = read(tmp_path / "recovered/recovered.xyz")
    assert recovered.info["REF_energy"] == -8.0
    np.testing.assert_array_equal(recovered.arrays["REF_forces"], 0)
    (raw / "OUTCAR").write_text(
        "EDIFF was not reached\nGeneral timing and accounting\n"
    )
    summary = recover_dataset(source, tmp_path / "raw", tmp_path / "quarantined")
    assert summary["quarantined"] == 1


@pytest.mark.unit
def test_curation_keeps_small_periodic_bulk_in_bulk_domain():
    from alomancy.utils.dataset_curation import structure_domain

    a = labelled()
    a.info["config_type"] = "init_stretch_compress"
    assert structure_domain(a) == "bulk"


@pytest.mark.unit
def test_init_rattle_classified_as_bulk_domain():
    from alomancy.utils.dataset_curation import structure_domain

    a = labelled()
    a.info["config_type"] = "init_rattle"
    assert structure_domain(a) == "bulk"


@pytest.mark.unit
def test_explicit_source_domain_recovers_unknown_annotation():
    from alomancy.utils.dataset_curation import structure_domain

    a = labelled(domain="unknown")
    a.info.update(config_type="high_sd")
    a.info["structure type"] = "bulk"
    assert structure_domain(a) == "bulk"


@pytest.mark.unit
def test_high_sd_with_no_explicit_domain_falls_back_to_config_type():
    """An AL-loop "high_sd" candidate with >3 atoms and no explicit
    domain/structure_type override has no dedicated classification branch
    -- it must resolve to its own config_type ("high_sd"), never the
    literal string "unknown", so require_known_domain doesn't silently
    exclude every AL-loop structure from training (see
    test_high_sd_domain_is_not_flagged_unknown_under_require_known_domain
    below for the end-to-end consequence)."""
    from alomancy.utils.dataset_curation import structure_domain

    a = Atoms(
        "Pd4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[20] * 3,
        pbc=True,
    )
    a.info["config_type"] = "high_sd"
    assert structure_domain(a) == "high_sd"


@pytest.mark.unit
def test_high_sd_domain_is_not_flagged_unknown_under_require_known_domain():
    a = Atoms(
        "Pd4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[20] * 3,
        pbc=True,
    )
    a.info.update(REF_energy=-8.0, config_type="high_sd")
    a.set_array("REF_forces", np.zeros((4, 3)))
    metadata = annotate_structure(a, {"require_known_domain": True})
    assert metadata["domain"] == "high_sd"
    assert "unknown_domain" not in metadata["filter_reasons"]


@pytest.mark.unit
def test_missing_config_type_still_resolves_to_unknown():
    from alomancy.utils.dataset_curation import structure_domain

    a = Atoms(
        "Pd4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[20] * 3,
        pbc=True,
    )
    assert structure_domain(a) == "unknown"


@pytest.mark.unit
def test_full_preparation_annotates_filters_and_preserves_reference(tmp_path):
    import json

    import yaml
    from ase.build import bulk
    from ase.io import write

    from alomancy.utils.prepare_dataset import prepare_dataset

    configurations = []
    for i in range(10):
        a = bulk("Pd", "fcc", a=3.9 + i * 0.001).repeat((2, 2, 2))
        a.info.update(config_type="bulk_rattle", REF_energy=-40.0 + i * 0.01)
        a.set_array("REF_forces", np.zeros((len(a), 3)))
        configurations.append(a)
        configurations.append(
            labelled(positions=[[0, 0, 0], [2.3 + i * 0.01, 0, 0]], energy=-4.0)
        )
    isolated = Atoms("Pd", info={"config_type": "IsolatedAtom", "REF_energy": -1.5})
    isolated.set_array("REF_forces", np.zeros((1, 3)))
    configurations.extend([isolated, labelled(energy=100.0)])
    source = tmp_path / "source.xyz"
    write(source, configurations)
    config = tmp_path / "init.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "initialization": {"test_to_train_ratio": 0.2},
                "dataset_curation": {
                    "default": {"max_force": 10.0},
                    "domains": {
                        "bulk": {"formation_energy_per_atom": [-0.1, 2.0]},
                        "dimer": {"formation_energy_per_atom": [-0.1, 6.0]},
                    },
                },
            }
        )
    )
    output = tmp_path / "prepared"
    manifest = prepare_dataset(source, config, output)
    assert manifest["selected"] == 21 and manifest["rejected"] == 1
    assert manifest["reference"]["chemical_potentials"]["Pd"] == -5.0
    selected = read(output / "selected.xyz", ":")
    for a in selected:
        assert a.info["REF_formation_energy"] == pytest.approx(
            a.info["REF_energy"] + len(a) * 5.0
        )
    assert next(a.info["REF_energy"] for a in selected if len(a) == 1) == -1.5
    assert json.loads((output / "manifest.json").read_text())["output_sha256"][
        "selected.xyz"
    ]
