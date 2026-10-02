"""Tests for mlip/mace/trainer.py's MaceTrainer -- the MACE backend of
ALomancyTrainer (mlip/base.py). The generic parts (train/evaluate/restart
checks/isolated-atom energies) are tested backend-free in test_base.py;
here they run through MACE's own fit(), with MACE's run() patched."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alomancy.mlip.evaluation import prediction_metrics, save_evaluation
from alomancy.mlip.mace.trainer import (
    _MACE_KWARGS_DEFAULTS,
    MaceTrainer,
    _apply_compute_stress_defaults,
    _mace_e0s_arg,
    _write_resolved_mace_epochs,
)

_FIT_DIR = Path("results/al_loop_0/committee/fit_0")


def _write_structures(path: Path, n: int, config_type: str = "init_dimer") -> None:
    structures = []
    for i in range(n):
        a = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        a.info["config_type"] = f"{config_type}_{i}"
        a.info["REF_energy"] = 1.0
        a.arrays["REF_forces"] = np.zeros((1, 3))
        structures.append(a)
    write(str(path), structures, format="extxyz")


def _trainer(mace_kwargs: dict | None = None, **config) -> MaceTrainer:
    return MaceTrainer({"mace_kwargs": mace_kwargs or {}, **config}, name="committee")


@pytest.mark.unit
class TestWriteResolvedMaceEpochs:
    def test_writes_max_num_epochs_and_start_swa(self, tmp_path):
        _write_resolved_mace_epochs(
            tmp_path, {"max_num_epochs": 160, "start_swa": 128, "other": "ignored"}
        )
        payload = json.loads((tmp_path / "resolved_mace_epochs.json").read_text())
        assert payload == {"max_num_epochs": 160, "start_swa": 128}


@pytest.mark.unit
class TestApplyComputeStressDefaults:
    def test_noop_when_compute_stress_false(self):
        params = {"loss": "weighted"}
        _apply_compute_stress_defaults(params, False)
        assert params == {"loss": "weighted"}

    def test_sets_stress_key_and_loss_when_enabled(self):
        params = {}
        _apply_compute_stress_defaults(params, True)
        assert params["stress_key"] == "REF_stresses"
        assert params["loss"] == "stress"

    def test_does_not_override_explicit_loss(self):
        params = {"loss": "huber"}
        _apply_compute_stress_defaults(params, True)
        assert params["loss"] == "huber"


@pytest.mark.unit
class TestModelFiles:
    def test_output_paths_are_uncompiled_model_and_metrics(self):
        assert _trainer().output_paths(_FIT_DIR) == [
            _FIT_DIR / "committee_stagetwo.model",
            _FIT_DIR / "evaluation_metrics.json",
        ]

    def test_output_paths_never_include_compiled_model(self):
        """MACE swallows a failed compile, so gating restart on the compiled
        model would make a genuinely finished fit look incomplete forever."""
        assert not any("compiled" in str(p) for p in _trainer().output_paths(_FIT_DIR))

    def test_deployable_model_is_compiled_or_none(self, tmp_path):
        trainer = _trainer()
        assert trainer.deployable_model_path(tmp_path) is None
        (tmp_path / "committee_stagetwo_compiled.model").write_bytes(b"compiled")
        assert trainer.deployable_model_path(tmp_path) == (
            tmp_path / "committee_stagetwo_compiled.model"
        )

    def test_checkpoints_cleaned_only_once_compiled_model_exists(self, tmp_path):
        trainer = _trainer()
        (tmp_path / "checkpoints").mkdir()
        trainer.cleanup(tmp_path)
        assert (tmp_path / "checkpoints").exists()
        (tmp_path / "committee_stagetwo_compiled.model").write_bytes(b"compiled")
        trainer.cleanup(tmp_path)
        assert not (tmp_path / "checkpoints").exists()


def _predicted(error: float) -> Atoms:
    a = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a.info.update(
        REF_energy=-8.0, model_energy=-8.0 + 2 * error, config_type="init_dimer"
    )
    a.set_array("REF_forces", np.zeros((2, 3)))
    a.set_array("model_forces", np.ones((2, 3)) * error)
    return a


@pytest.mark.unit
class TestReadExistingResult:
    def _write_fit(self, fit_dir: Path, splits: dict) -> None:
        fit_dir.mkdir(parents=True)
        model = fit_dir / "committee_stagetwo.model"
        model.write_bytes(b"checkpoint")
        save_evaluation(fit_dir, model, splits)

    def test_reconstructs_model_path_and_metrics(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fit(
            _FIT_DIR,
            {
                "train": prediction_metrics([_predicted(0.1)]),
                "test": prediction_metrics([_predicted(0.2)]),
            },
        )

        model_path, compiled_path, metrics = _trainer().read_existing_result(_FIT_DIR)

        assert model_path == str(_FIT_DIR / "committee_stagetwo.model")
        assert compiled_path is None  # never written in this test
        assert set(metrics) == {"train", "test"}

    def test_includes_compiled_path_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fit(_FIT_DIR, {"valid": prediction_metrics([_predicted(0.1)])})
        (_FIT_DIR / "committee_stagetwo_compiled.model").write_bytes(b"compiled")

        _, compiled_path, _ = _trainer().read_existing_result(_FIT_DIR)
        assert compiled_path == str(_FIT_DIR / "committee_stagetwo_compiled.model")

    def test_raises_when_nothing_valid_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No valid checkpoint-verified"):
            _trainer().read_existing_result(_FIT_DIR)

    def test_raises_when_checkpoint_changed_since_evaluation(
        self, tmp_path, monkeypatch
    ):
        """A stale/corrupted on-disk state must raise, not be silently
        treated as a valid cached result (read_evaluation's checksum)."""
        monkeypatch.chdir(tmp_path)
        self._write_fit(_FIT_DIR, {"test": prediction_metrics([_predicted(0.1)])})
        (_FIT_DIR / "committee_stagetwo.model").write_bytes(b"changed after eval")

        with pytest.raises(ValueError, match="No valid checkpoint-verified"):
            _trainer().read_existing_result(_FIT_DIR)


@pytest.mark.unit
class TestGetCalculator:
    def test_builds_mace_calculator_with_given_device_and_dtype(self):
        with patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_cls:
            mock_cls.return_value = MagicMock()
            MaceTrainer({"device": "cpu", "default_dtype": "float32"}).get_calculator(
                "model.pt"
            )
        mock_cls.assert_called_once_with(
            model_paths=["model.pt"], device="cpu", default_dtype="float32"
        )

    def test_defaults_dtype_to_float64(self):
        with patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_cls:
            mock_cls.return_value = MagicMock()
            MaceTrainer({"device": "cpu"}).get_calculator("model.pt")
        assert mock_cls.call_args.kwargs["default_dtype"] == "float64"


@pytest.mark.unit
class TestTrain:
    def _run_train(
        self, tmp_path, monkeypatch, mace_kwargs=None, compile_model=False, **kwargs
    ):
        monkeypatch.chdir(tmp_path)
        train_path = tmp_path / "train.xyz"
        test_path = tmp_path / "test.xyz"
        _write_structures(train_path, 4)
        _write_structures(test_path, 2)
        if mace_kwargs is None:
            mace_kwargs = {"max_num_epochs": 40}
        expected_cwd = _FIT_DIR.resolve()

        def fake_run(args):
            # MACE's run() writes into the current directory, which fit() has
            # chdir'd to.
            assert Path.cwd() == expected_cwd
            (Path.cwd() / "committee_stagetwo.model").write_bytes(b"checkpoint")
            if compile_model:
                (Path.cwd() / "committee_stagetwo_compiled.model").write_bytes(b"c")

        with (
            patch("alomancy.mlip.mace.trainer.run", side_effect=fake_run) as mock_run,
            patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_calc_cls,
            patch("alomancy.mlip.mace.trainer.tools") as mock_tools,
        ):
            mock_calc_cls.return_value = MagicMock()
            # A real, empty Namespace so unset attributes genuinely don't exist.
            mock_tools.build_default_arg_parser.return_value.parse_args.return_value = (
                argparse.Namespace()
            )
            monkeypatch.setattr(Atoms, "get_potential_energy", lambda self: 1.0)
            monkeypatch.setattr(
                Atoms, "get_forces", lambda self: np.zeros((1, 3)), raising=False
            )
            result = _trainer(mace_kwargs).train(
                str(train_path),
                kwargs.pop("valid_atoms_path", None),
                str(test_path),
                803,
                fit_dir=_FIT_DIR,
                **kwargs,
            )
        return result, mock_run, mock_calc_cls

    def test_returns_model_path_and_evaluates(self, tmp_path, monkeypatch):
        result, mock_run, _ = self._run_train(tmp_path, monkeypatch)
        assert result == str(_FIT_DIR.resolve() / "committee_stagetwo.model")
        assert mock_run.call_count == 1
        metrics = json.loads((_FIT_DIR / "evaluation_metrics.json").read_text())
        assert set(metrics["splits"]) == {"train", "test"}
        assert (_FIT_DIR / "train_pred.xyz").exists()

    def test_returns_none_and_skips_evaluation_without_a_model(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _write_structures(tmp_path / "train.xyz", 2)
        _write_structures(tmp_path / "test.xyz", 1)
        with (
            patch("alomancy.mlip.mace.trainer.run"),
            patch("alomancy.mlip.mace.trainer.tools") as mock_tools,
        ):
            mock_tools.build_default_arg_parser.return_value.parse_args.return_value = (
                argparse.Namespace()
            )
            result = _trainer({"max_num_epochs": 5}).train(
                "train.xyz", None, "test.xyz", 803, fit_dir=_FIT_DIR
            )
        assert result is None
        assert not (_FIT_DIR / "evaluation_metrics.json").exists()

    def test_evaluates_the_uncompiled_model_even_when_compiled_exists(
        self, tmp_path, monkeypatch
    ):
        """Parity plots and metrics come from the uncompiled model: the
        compiled model's forces failed in production."""
        _, _, mock_calc_cls = self._run_train(tmp_path, monkeypatch, compile_model=True)
        (model_paths,) = mock_calc_cls.call_args.kwargs["model_paths"]
        assert model_paths.endswith("committee_stagetwo.model")
        record = json.loads((_FIT_DIR / "evaluation_metrics.json").read_text())
        assert record["checkpoint"] == "committee_stagetwo.model"

    def test_mace_args_come_from_class_defaults_plus_user_kwargs(
        self, tmp_path, monkeypatch
    ):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            mace_kwargs={"num_channels": 64, "max_num_epochs": 10},
        )
        args = mock_run.call_args.args[0]
        assert args.num_channels == 64  # user override reaches MACE
        assert args.r_max == _MACE_KWARGS_DEFAULTS["r_max"]  # unset -> class default
        assert args.energy_key == "REF_energy"
        assert args.forces_key == "REF_forces"
        assert args.seed == 803
        assert (args.max_num_epochs, args.start_swa) == (10, 8)
        assert not hasattr(args, "compute_stress")

    def test_compute_stress_read_from_mace_kwargs(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, mace_kwargs={"compute_stress": True}
        )
        args = mock_run.call_args.args[0]
        assert args.stress_key == "REF_stresses"
        assert args.loss == "stress"
        assert not hasattr(args, "compute_stress")

    def test_raises_when_e0s_missing_element(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="E0s"):
            self._run_train(
                tmp_path,
                monkeypatch,
                mace_kwargs={"E0s": {"H": -1.0}},
                elements=["H", "O"],
            )

    def test_e0s_covering_all_elements_does_not_raise(self, tmp_path, monkeypatch):
        self._run_train(
            tmp_path,
            monkeypatch,
            mace_kwargs={"E0s": {"H": -1.0, "O": -2.0}},
            elements=["H", "O"],
        )

    def test_defaults_e0s_from_isolated_atom_energies_when_unset(
        self, tmp_path, monkeypatch
    ):
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, elements=["H"], isolated_atom_energies={"H": -13.6}
        )
        assert mock_run.call_args.args[0].E0s == "{1: -13.6}"

    def test_explicit_e0s_wins_over_isolated_atom_default(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            mace_kwargs={"E0s": {"H": -1.0}},
            elements=["H"],
            isolated_atom_energies={"H": -13.6},
        )
        assert mock_run.call_args.args[0].E0s == "{1: -1.0}"

    def test_raises_when_no_e0s_and_no_isolated_atom_energies(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(ValueError, match="E0s"):
            self._run_train(
                tmp_path, monkeypatch, elements=["H"], isolated_atom_energies={}
            )

    def test_no_e0s_passed_when_unset_and_elements_not_given(
        self, tmp_path, monkeypatch
    ):
        """No elements (a direct caller): nothing to check or pass, so MACE
        falls back to IsolatedAtom frames in the training file."""
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, isolated_atom_energies={}
        )
        assert not hasattr(mock_run.call_args.args[0], "E0s")

    def test_string_e0s_passed_through(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, mace_kwargs={"E0s": "average"}, elements=["H"]
        )
        assert mock_run.call_args.args[0].E0s == "average"

    def test_raises_when_seed_in_mace_kwargs(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="seed"):
            self._run_train(tmp_path, monkeypatch, mace_kwargs={"seed": 1})

    def test_no_valid_file_passed_when_valid_atoms_path_is_none(
        self, tmp_path, monkeypatch
    ):
        _, mock_run, _ = self._run_train(tmp_path, monkeypatch)
        assert not hasattr(mock_run.call_args.args[0], "valid_file")

    def test_valid_file_passed_when_valid_atoms_path_given(self, tmp_path, monkeypatch):
        valid_path = tmp_path / "valid.xyz"
        _write_structures(valid_path, 2)
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, valid_atoms_path=str(valid_path)
        )
        assert mock_run.call_args.args[0].valid_file == str(valid_path.resolve())

    def test_max_num_epochs_defaults_to_dynamic_when_unset(self, tmp_path, monkeypatch):
        """Unset -> dynamic (4 structures pin to the cap, 300), never MACE's
        own 2048 or an old fixed fallback."""
        _, mock_run, _ = self._run_train(tmp_path, monkeypatch, mace_kwargs={})
        assert mock_run.call_args.args[0].max_num_epochs == 300

    def test_dynamic_epochs_uses_train_plus_valid_count(self, tmp_path, monkeypatch):
        """19000 train + 1000 valid -> ceil(3_200_000/20000) = 160; train
        alone would give 169, so a regression to the train-only count shows."""
        monkeypatch.chdir(tmp_path)
        _write_structures(tmp_path / "train.xyz", 19000)
        _write_structures(tmp_path / "valid.xyz", 1000)
        _write_structures(tmp_path / "test.xyz", 1)
        captured = {}

        def fake_run(args):
            captured["max_num_epochs"] = args.max_num_epochs

        with (
            patch("alomancy.mlip.mace.trainer.run", side_effect=fake_run),
            patch("alomancy.mlip.mace.trainer.tools") as mock_tools,
        ):
            mock_tools.build_default_arg_parser.return_value.parse_args.return_value = (
                argparse.Namespace()
            )
            _trainer({"batch_size": 16, "max_num_epochs": "dynamic"}).train(
                "train.xyz", "valid.xyz", "test.xyz", 803, fit_dir=_FIT_DIR
            )
        assert captured["max_num_epochs"] == 160


class TestMaceE0sArg:
    """E0s reach MACE as the string its own parser accepts: a dict literal
    keyed by atomic number (mace.tools.scripts_utils.get_atomic_energies
    calls E0s.lower() and ast.literal_eval(E0s))."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "e0s",
        [
            {"C": -155.1, "H": -13.6},
            {6: -155.1, 1: -13.6},
            {"6": -155.1, "1": -13.6},
        ],
    )
    def test_accepted_by_mace_parser(self, e0s):
        from mace.tools.scripts_utils import get_atomic_energies

        parsed = get_atomic_energies(_mace_e0s_arg(e0s), None, None)
        assert parsed == {6: -155.1, 1: -13.6}

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [{"Xx": -1.0}, {0: -1.0}, {"C": float("nan")}])
    def test_rejects_bad_entries(self, bad):
        with pytest.raises(ValueError, match="E0s"):
            _mace_e0s_arg(bad)
