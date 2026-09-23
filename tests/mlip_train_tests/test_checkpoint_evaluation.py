import numpy as np
import pytest
from ase import Atoms

from alomancy.mlip.evaluation import (
    prediction_metrics,
    read_evaluation,
    save_evaluation,
)
from alomancy.mlip.mace.get_mace_eval_info import select_best_committee_model


def predicted(error):
    a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a.info.update(
        REF_energy=-8.0, mace_energy=-8.0 + 2 * error, config_type="init_dimer"
    )
    a.set_array("REF_forces", np.zeros((2, 3)))
    a.set_array("mace_forces", np.ones((2, 3)) * error)
    return a


@pytest.mark.unit
def test_selection_uses_common_validation_not_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i, error in enumerate([0.3, 0.1, 0.2]):
        fit = tmp_path / "results/al_loop_0/committee" / f"fit_{i}"
        fit.mkdir(parents=True)
        model = fit / "committee_stagetwo.model"
        model.write_bytes(b"checkpoint")
        save_evaluation(
            fit,
            model,
            {
                "valid": prediction_metrics([predicted(error)]),
                "test": prediction_metrics([predicted(1 - error)]),
            },
        )
    best, _ = select_best_committee_model(
        "al_loop_0", {"name": "committee", "size_of_committee": 3}, 803
    )
    assert best == 1


@pytest.mark.unit
def test_refuses_missing_validation_instead_of_fit_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="complete checkpoint validation"):
        select_best_committee_model(
            "al_loop_0", {"name": "c", "size_of_committee": 3}, 803
        )


@pytest.mark.unit
def test_falls_back_to_test_when_no_fit_has_a_validation_split(tmp_path, monkeypatch):
    """mace_fit's own _select_validation_split legitimately skips carving a
    validation split (logs a warning, doesn't fail) whenever the eligible
    pool is too small -- every fit is then uniformly missing 'valid'. This
    must not be treated as an evaluation failure: fall back to the 'test'
    split, which mace_fit always attempts regardless of pool size."""
    monkeypatch.chdir(tmp_path)
    for i, error in enumerate([0.3, 0.1, 0.2]):
        fit = tmp_path / "results/al_loop_0/committee" / f"fit_{i}"
        fit.mkdir(parents=True)
        model = fit / "committee_stagetwo.model"
        model.write_bytes(b"checkpoint")
        save_evaluation(fit, model, {"test": prediction_metrics([predicted(error)])})
    best, _ = select_best_committee_model(
        "al_loop_0", {"name": "committee", "size_of_committee": 3}, 803
    )
    assert best == 1


@pytest.mark.unit
def test_refuses_when_fits_disagree_on_having_a_validation_split(tmp_path, monkeypatch):
    """Some fits having 'valid' while others don't (rather than uniformly
    none) indicates a genuine per-fit evaluation failure, not a normal
    small-pool run -- this must still raise, not silently fall back."""
    monkeypatch.chdir(tmp_path)
    for i, error in enumerate([0.3, 0.1, 0.2]):
        fit = tmp_path / "results/al_loop_0/committee" / f"fit_{i}"
        fit.mkdir(parents=True)
        model = fit / "committee_stagetwo.model"
        model.write_bytes(b"checkpoint")
        splits = {"test": prediction_metrics([predicted(error)])}
        if i != 0:
            splits["valid"] = prediction_metrics([predicted(error)])
        save_evaluation(fit, model, splits)
    with pytest.raises(RuntimeError, match="complete checkpoint validation"):
        select_best_committee_model(
            "al_loop_0", {"name": "committee", "size_of_committee": 3}, 803
        )


@pytest.mark.unit
def test_invalid_prediction_is_not_silently_omitted():
    a = predicted(0.1)
    del a.info["mace_energy"]
    with pytest.raises(ValueError, match="Missing or invalid"):
        prediction_metrics([predicted(0.1), a])


@pytest.mark.unit
def test_checkpoint_fingerprint_prevents_stale_metrics(tmp_path):
    model = tmp_path / "c.model"
    model.write_bytes(b"old")
    save_evaluation(tmp_path, model, {"valid": prediction_metrics([predicted(0.1)])})
    model.write_bytes(b"new")
    with pytest.raises(ValueError, match="changed"):
        read_evaluation(tmp_path, "valid")


@pytest.mark.unit
def test_metrics_report_per_atom_and_component_units():
    m = prediction_metrics([predicted(0.2)])
    assert m["mae_e_per_atom"] == pytest.approx(0.2)
    assert m["mae_f"] == pytest.approx(0.2)
    assert m["domains"]["dimer"]["n_structures"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("error,passes", [(0.01, True), (0.2, False)])
def test_quality_gate_enforces_domain_limits(tmp_path, error, passes):
    from alomancy.mlip.evaluation import check_quality_gate

    committee = {
        "name": "c",
        "size_of_committee": 2,
        "quality_gate": {"domains": {"dimer": {"mae_f": 0.1}}},
    }
    for i in range(2):
        fit = tmp_path / "c" / f"fit_{i}"
        fit.mkdir(parents=True)
        model = fit / "c.model"
        model.write_bytes(b"checkpoint")
        save_evaluation(fit, model, {"valid": prediction_metrics([predicted(error)])})
    if passes:
        check_quality_gate(tmp_path, committee)
    else:
        with pytest.raises(RuntimeError, match="exceeds"):
            check_quality_gate(tmp_path, committee)


@pytest.mark.unit
def test_quality_gate_rejects_different_validation_sets(tmp_path):
    from alomancy.mlip.evaluation import check_quality_gate

    committee = {
        "name": "c",
        "size_of_committee": 2,
        "quality_gate": {"domains": {"dimer": {"mae_f": 0.1}}},
    }
    for i in range(2):
        fit = tmp_path / "c" / f"fit_{i}"
        fit.mkdir(parents=True)
        model = fit / "c.model"
        model.write_bytes(b"checkpoint")
        atoms = predicted(0.01)
        atoms.positions[1, 0] += i
        save_evaluation(fit, model, {"valid": prediction_metrics([atoms])})
    with pytest.raises(RuntimeError, match="same validation set"):
        check_quality_gate(tmp_path, committee)


@pytest.mark.unit
def test_train_only_recognizes_evaluated_stage_one_checkpoint(tmp_path, monkeypatch):
    from unittest.mock import patch

    from alomancy.core.standard_active_learning import ActiveLearningStandardMACE

    monkeypatch.chdir(tmp_path)
    committee = {
        "name": "c",
        "size_of_committee": 3,
        "require_checkpoint_metrics": True,
    }
    for i in range(3):
        fit = tmp_path / "results/al_loop_0/c" / f"fit_{i}"
        fit.mkdir(parents=True)
        model = fit / "c.model"
        model.write_bytes(b"stage one checkpoint")
        metrics = prediction_metrics([predicted(0.01)])
        save_evaluation(fit, model, {"valid": metrics, "test": metrics})
    workflow = ActiveLearningStandardMACE(
        "train.xyz", "test.xyz", {"mlip_committee": committee}, plots=False
    )
    with patch(
        "alomancy.core.standard_active_learning.committee_remote_submitter"
    ) as submit:
        metrics = workflow.train_mlip("al_loop_0", workflow.jobs_dict)
    submit.assert_not_called()
    assert metrics.iloc[0]["metric_source"] == "checkpoint_test"
