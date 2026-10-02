"""Tests for mlip/base.py's ALomancyTrainer, run through a tiny backend
(EMTTrainer: fit() writes a placeholder model file, get_calculator()
returns ASE's EMT) so the generic train/evaluate/restart path runs for real
without MACE."""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read, write

from alomancy.mlip import base
from alomancy.mlip.base import (
    ALomancyTrainer,
    get_trainer,
    read_predictions,
    run_training,
)
from alomancy.registry import _REGISTRY, register


class EMTTrainer(ALomancyTrainer):
    NAME = "emt_test"
    KWARGS_KEY = "emt_test_kwargs"
    KWARGS_DEFAULTS: ClassVar[dict] = {"steps": 1}
    ISOLATED_ATOM_ENERGIES_KWARG = "reference"

    fit_calls: ClassVar[list] = []

    def model_path(self, fit_dir: Path) -> Path:
        return Path(fit_dir) / f"{self.name}.model"

    def get_calculator(self, model_path):
        return EMT()

    def cleanup_paths(self, fit_dir: Path) -> list[Path]:
        return [Path(fit_dir) / "scratch"]

    def fit(
        self,
        train_path,
        valid_path,
        test_path,
        seed,
        fit_dir,
        *,
        isolated_atom_energies,
    ):
        type(self).fit_calls.append(
            {"seed": seed, "valid": valid_path, "energies": isolated_atom_energies}
        )
        (fit_dir / "scratch").mkdir(exist_ok=True)
        if self.kwargs.get("fail"):
            return None
        model = self.model_path(fit_dir)
        model.write_bytes(f"model seed {seed}".encode())
        return model


@pytest.fixture
def emt_registered():
    register("mlip_trainer", "emt_test", __name__, trainer_class="EMTTrainer")
    EMTTrainer.fit_calls = []
    yield
    del _REGISTRY["mlip_trainer"]["emt_test"]


def _cu(n: int, gid0: int = 0) -> list[Atoms]:
    out = []
    for i in range(n):
        a = Atoms(
            "Cu2", positions=[[0, 0, 0], [2.3 + 0.05 * i, 0, 0]], cell=[8] * 3, pbc=True
        )
        a.info.update(config_type="init_dimer", REF_energy=0.5, global_db_id=gid0 + i)
        a.arrays["REF_forces"] = np.zeros((2, 3))
        out.append(a)
    return out


@pytest.fixture
def splits(tmp_path):
    paths = {}
    for name, n, gid0 in (("train", 3, 0), ("valid", 1, 10), ("test", 2, 20)):
        paths[name] = tmp_path / f"{name}.xyz"
        write(paths[name], _cu(n, gid0), format="extxyz")
    return paths


@pytest.mark.unit
def test_train_fits_evaluates_and_cleans_up(tmp_path, splits):
    trainer = EMTTrainer({"emt_test_kwargs": {}}, name="training")
    fit_dir = tmp_path / "fit_0"

    model = trainer.train(
        splits["train"], splits["valid"], splits["test"], 7, fit_dir=fit_dir
    )

    assert model == str(fit_dir.resolve() / "training.model")
    assert EMTTrainer.fit_calls[-1]["seed"] == 7
    for split in ("train", "valid", "test"):
        predicted = read(fit_dir / f"{split}_pred.xyz", ":")
        assert all("model_energy" in a.info for a in predicted)
        assert all(a.arrays["model_forces"].shape == (2, 3) for a in predicted)
    record = json.loads((fit_dir / "evaluation_metrics.json").read_text())
    assert record["checkpoint"] == "training.model"
    assert set(record["splits"]) == {"train", "valid", "test"}
    assert not (fit_dir / "scratch").exists()

    model_path, deployable, metrics = trainer.read_existing_result(fit_dir)
    assert model_path == str(fit_dir / "training.model")
    assert deployable == model_path  # default: the model itself
    assert set(metrics) == {"train", "valid", "test"}
    assert trainer.output_paths(fit_dir) == [
        fit_dir / "training.model",
        fit_dir / "evaluation_metrics.json",
    ]


@pytest.mark.unit
def test_train_returns_none_without_a_model(tmp_path, splits):
    trainer = EMTTrainer({"emt_test_kwargs": {"fail": True}})
    fit_dir = tmp_path / "fit_0"
    assert (
        trainer.train(splits["train"], None, splits["test"], 1, fit_dir=fit_dir) is None
    )
    assert not (fit_dir / "evaluation_metrics.json").exists()


@pytest.mark.unit
def test_train_raises_on_missing_split_file(tmp_path, splits):
    with pytest.raises(FileNotFoundError):
        EMTTrainer({}).train(
            tmp_path / "nope.xyz", None, splits["test"], 1, fit_dir=tmp_path / "f"
        )


@pytest.mark.unit
def test_kwargs_merge_defaults_and_effective_kwargs():
    trainer = EMTTrainer({"emt_test_kwargs": {"extra": 2}})
    assert trainer.kwargs == {"steps": 1, "extra": 2}
    assert trainer.effective_kwargs()["reference"].startswith("<resolved at train time")
    explicit = EMTTrainer({"emt_test_kwargs": {"reference": {"Cu": -1.0}}})
    assert explicit.effective_kwargs()["reference"] == {"Cu": -1.0}


# ---------------------------------------------------------------------------
# Isolated-atom energies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveIsolatedAtomEnergies:
    def test_database_energies_used_when_unset(self):
        assert EMTTrainer({}).resolve_isolated_atom_energies(["Cu"], {"Cu": -3.0}) == {
            "Cu": -3.0
        }

    def test_explicit_setting_wins(self):
        trainer = EMTTrainer({"emt_test_kwargs": {"reference": {"Cu": -1.0}}})
        assert trainer.resolve_isolated_atom_energies(["Cu"], {"Cu": -3.0}) == {
            "Cu": -1.0
        }

    def test_string_passes_through_unchecked(self):
        trainer = EMTTrainer({"emt_test_kwargs": {"reference": "average"}})
        assert trainer.resolve_isolated_atom_energies(["Cu", "O"], None) == "average"

    def test_atomic_number_keys_cover_elements(self):
        trainer = EMTTrainer({"emt_test_kwargs": {"reference": {29: -1.0}}})
        assert trainer.resolve_isolated_atom_energies(["Cu"], None) == {29: -1.0}

    def test_missing_element_raises(self):
        with pytest.raises(ValueError, match=r"emt_test_kwargs\.reference.*\['O'\]"):
            EMTTrainer({}).resolve_isolated_atom_energies(["Cu", "O"], {"Cu": -3.0})

    def test_no_source_with_elements_raises(self):
        with pytest.raises(ValueError, match="IsolatedAtom"):
            EMTTrainer({}).resolve_isolated_atom_energies(["Cu"], {})

    def test_no_source_and_no_elements_gives_none(self):
        assert EMTTrainer({}).resolve_isolated_atom_energies(None, None) is None

    def test_backend_without_the_setting_skips_everything(self, monkeypatch):
        monkeypatch.setattr(EMTTrainer, "ISOLATED_ATOM_ENERGIES_KWARG", None)
        assert EMTTrainer({}).resolve_isolated_atom_energies(["Cu"], {}) is None
        assert "reference" not in EMTTrainer({}).effective_kwargs()

    def test_resolved_energies_reach_fit(self, tmp_path, splits):
        EMTTrainer({}).train(
            splits["train"],
            None,
            splits["test"],
            1,
            fit_dir=tmp_path / "f",
            elements=["Cu"],
            isolated_atom_energies={"Cu": -3.0},
        )
        assert EMTTrainer.fit_calls[-1]["energies"] == {"Cu": -3.0}


# ---------------------------------------------------------------------------
# Remote worker and helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_training_builds_trainer_by_registry_name(tmp_path, splits, emt_registered):
    fit_dir = tmp_path / "fit_3"
    model = run_training(
        "emt_test",
        {"emt_test_kwargs": {}},
        str(splits["train"]),
        None,
        str(splits["test"]),
        42,
        fit_dir=str(fit_dir),
        fit_name="committee",
    )
    assert model == str(fit_dir.resolve() / "committee.model")
    assert EMTTrainer.fit_calls[-1]["seed"] == 42
    assert isinstance(get_trainer("emt_test", {}), EMTTrainer)


@pytest.mark.unit
def test_run_training_is_module_level_for_expyre():
    """ExPyRe pickles remote functions by reference."""
    assert base.run_training.__qualname__ == "run_training"


@pytest.mark.unit
def test_read_predictions_maps_global_db_ids(tmp_path, splits):
    fit_dir = tmp_path / "fit_0"
    EMTTrainer({}).train(splits["train"], None, splits["test"], 1, fit_dir=fit_dir)

    preds = read_predictions(fit_dir)

    assert set(preds) == {0, 1, 2, 20, 21}
    assert isinstance(preds[0]["energy"], float)
    assert np.asarray(preds[0]["forces"]).shape == (2, 3)
    assert read_predictions(tmp_path / "missing") == {}


# ---------------------------------------------------------------------------
# Evaluation failure reporting (regression: a near-total per-structure
# prediction failure, 1 of 1405 structures succeeding, was once invisible
# and silently reduced every parity plot to a single point)
# ---------------------------------------------------------------------------


def _collect_alomancy_logs():
    """setup_logging sets propagate=False on "alomancy", so caplog can't see
    these records; attach a handler directly."""
    import logging

    al_logger = logging.getLogger("alomancy")
    al_logger.setLevel(logging.DEBUG)
    records: list = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector(level=logging.DEBUG)
    al_logger.addHandler(handler)
    return al_logger, handler, records


@pytest.mark.unit
def test_first_prediction_failure_warns_with_traceback_rest_are_debug(
    tmp_path, splits, monkeypatch
):
    import logging

    calls = {"n": 0}
    real = Atoms.get_potential_energy

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError(f"boom {calls['n']}")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Atoms, "get_potential_energy", flaky)
    al_logger, handler, records = _collect_alomancy_logs()
    try:
        EMTTrainer({}).train(
            splits["train"], None, splits["test"], 1, fit_dir=tmp_path / "f"
        )
    finally:
        al_logger.removeHandler(handler)

    failures = [
        r for r in records if "Prediction failed for structure" in r.getMessage()
    ]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG]
    assert failures[0].exc_info is not None
    summary = [r for r in records if "succeeded" in r.getMessage()]
    assert "1 succeeded, 2 failed out of 3 structures" in summary[0].getMessage()


@pytest.mark.unit
def test_no_failure_logs_when_every_prediction_succeeds(tmp_path, splits):
    al_logger, handler, records = _collect_alomancy_logs()
    try:
        EMTTrainer({}).train(
            splits["train"], None, splits["test"], 1, fit_dir=tmp_path / "f"
        )
    finally:
        al_logger.removeHandler(handler)
    assert not [r for r in records if "Prediction failed" in r.getMessage()]


@pytest.mark.unit
def test_prediction_files_carry_no_calculator_results(tmp_path, splits):
    """Only the explicit model_* fields are written: ASE would otherwise
    serialise the calculator's results, some not per-atom, which breaks
    extxyz writing for real MACE models."""
    fit_dir = tmp_path / "f"
    EMTTrainer({}).train(splits["train"], None, splits["test"], 1, fit_dir=fit_dir)
    for atoms in read(fit_dir / "train_pred.xyz", ":"):
        assert atoms.calc is None or "energy" not in (atoms.calc.results or {})
        assert "model_energy" in atoms.info
