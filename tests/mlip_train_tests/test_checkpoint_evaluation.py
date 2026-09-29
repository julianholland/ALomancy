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
        REF_energy=-8.0, model_energy=-8.0 + 2 * error, config_type="init_dimer"
    )
    a.set_array("REF_forces", np.zeros((2, 3)))
    a.set_array("model_forces", np.ones((2, 3)) * error)
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
        "al_loop_0", {"name": "committee", "num_of_models_in_committee": 3}, 803
    )
    assert best == 1


@pytest.mark.unit
def test_refuses_missing_validation_instead_of_fit_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="complete checkpoint validation"):
        select_best_committee_model(
            "al_loop_0", {"name": "c", "num_of_models_in_committee": 3}, 803
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
        "al_loop_0", {"name": "committee", "num_of_models_in_committee": 3}, 803
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
            "al_loop_0", {"name": "committee", "num_of_models_in_committee": 3}, 803
        )


@pytest.mark.unit
def test_invalid_prediction_is_not_silently_omitted():
    a = predicted(0.1)
    del a.info["model_energy"]
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
        "num_of_models_in_committee": 2,
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
        "num_of_models_in_committee": 2,
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


# Restart-recognizes-evaluated-checkpoint coverage for the current skeleton
# now lives in test_committee_uncertainty_workflow.py's TestTrainMlip.
# test_recognizes_real_checkpoint_evaluation_on_restart (ported from the
# now-removed standard_active_learning.py/ActiveLearningStandardMACE.
# train_mlip, which this module previously tested directly).


def predicted_bulk(error, stress_error=None, config_type="init_MP"):
    a = Atoms("Pd2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4], pbc=True)
    a.info.update(
        REF_energy=-8.0, model_energy=-8.0 + 2 * error, config_type=config_type
    )
    a.set_array("REF_forces", np.zeros((2, 3)))
    a.set_array("model_forces", np.ones((2, 3)) * error)
    if stress_error is not None:
        a.info["REF_stresses"] = np.zeros(6)
        a.info["model_stress"] = np.full(6, stress_error)
    return a


@pytest.mark.unit
def test_metrics_broken_down_by_config_type():
    metrics = prediction_metrics(
        [
            predicted_bulk(0.1, config_type="init_MP"),
            predicted_bulk(0.3, config_type="high_sd"),
            predicted_bulk(0.5, config_type="high_sd"),
        ]
    )
    assert set(metrics["config_types"]) == {"init_MP", "high_sd"}
    assert metrics["config_types"]["init_MP"]["mae_f"] == pytest.approx(0.1)
    assert metrics["config_types"]["high_sd"]["mae_f"] == pytest.approx(0.4)
    assert metrics["config_types"]["high_sd"]["n_structures"] == 2


@pytest.mark.unit
def test_stress_errors_use_only_structures_with_both_stresses():
    metrics = prediction_metrics(
        [
            predicted_bulk(0.1, stress_error=0.02),
            predicted_bulk(0.1, stress_error=-0.04),
            predicted_bulk(0.1),  # no stress: excluded, not an error
        ]
    )
    assert metrics["n_structures"] == 3
    assert metrics["n_structures_with_stress"] == 2
    assert metrics["mae_stress"] == pytest.approx(0.03)
    assert metrics["rmse_stress"] == pytest.approx(np.sqrt((0.02**2 + 0.04**2) / 2))


@pytest.mark.unit
def test_full_3x3_model_stress_accepted():
    a = predicted_bulk(0.1)
    a.info["REF_stresses"] = np.zeros(6)
    a.info["model_stress"] = np.eye(3) * 0.01
    metrics = prediction_metrics([a])
    # Voigt: three diagonal components of 0.01, three shear of 0.
    assert metrics["mae_stress"] == pytest.approx(0.005)


@pytest.mark.unit
def test_no_stress_keys_without_stress_data():
    metrics = prediction_metrics([predicted_bulk(0.1)])
    assert "mae_stress" not in metrics
    assert "mae_stress" not in metrics["config_types"]["init_MP"]
