"""Tests for mlip/mace/trainer.py's MaceTrainer -- the MACE backend of
ALomancyTrainer (mlip/base.py). The generic parts (train/evaluate/restart
checks/isolated-atom energies) are tested backend-free in test_base.py;
here they run through MACE's own fit(), with MACE's run() patched."""

import argparse
import json
from pathlib import Path
from typing import Any, ClassVar
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
    @pytest.mark.unit
    def test_writes_max_num_epochs_and_start_swa(self, tmp_path):
        _write_resolved_mace_epochs(
            tmp_path, {"max_num_epochs": 160, "start_swa": 128, "other": "ignored"}
        )
        payload = json.loads((tmp_path / "resolved_mace_epochs.json").read_text())
        assert payload == {"max_num_epochs": 160, "start_swa": 128}


@pytest.mark.unit
class TestApplyComputeStressDefaults:
    @pytest.mark.unit
    def test_noop_when_compute_stress_false(self):
        params = {"loss": "weighted"}
        _apply_compute_stress_defaults(params, False)
        assert params == {"loss": "weighted"}

    @pytest.mark.unit
    def test_sets_stress_key_and_loss_when_enabled(self):
        params = {}
        _apply_compute_stress_defaults(params, True)
        assert params["stress_key"] == "REF_stresses"
        assert params["loss"] == "stress"

    @pytest.mark.unit
    def test_does_not_override_explicit_loss(self):
        params = {"loss": "huber"}
        _apply_compute_stress_defaults(params, True)
        assert params["loss"] == "huber"


@pytest.mark.unit
class TestModelFiles:
    @pytest.mark.unit
    def test_output_paths_are_uncompiled_model_and_metrics(self):
        assert _trainer().output_paths(_FIT_DIR) == [
            _FIT_DIR / "committee_stagetwo.model",
            _FIT_DIR / "evaluation_metrics.json",
        ]

    @pytest.mark.unit
    def test_output_paths_never_include_compiled_model(self):
        """MACE swallows a failed compile, so gating restart on the compiled
        model would make a genuinely finished fit look incomplete forever."""
        assert not any("compiled" in str(p) for p in _trainer().output_paths(_FIT_DIR))

    @pytest.mark.unit
    def test_deployable_model_is_compiled_or_none(self, tmp_path):
        trainer = _trainer()
        assert trainer.deployable_model_path(tmp_path) is None
        (tmp_path / "committee_stagetwo_compiled.model").write_bytes(b"compiled")
        assert trainer.deployable_model_path(tmp_path) == (
            tmp_path / "committee_stagetwo_compiled.model"
        )

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_includes_compiled_path_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_fit(_FIT_DIR, {"valid": prediction_metrics([_predicted(0.1)])})
        (_FIT_DIR / "committee_stagetwo_compiled.model").write_bytes(b"compiled")

        _, compiled_path, _ = _trainer().read_existing_result(_FIT_DIR)
        assert compiled_path == str(_FIT_DIR / "committee_stagetwo_compiled.model")

    @pytest.mark.unit
    def test_raises_when_nothing_valid_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No valid checkpoint-verified"):
            _trainer().read_existing_result(_FIT_DIR)

    @pytest.mark.unit
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
    @pytest.mark.unit
    def test_builds_mace_calculator_with_given_device_and_dtype(self):
        with patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_cls:
            mock_cls.return_value = MagicMock()
            MaceTrainer({"device": "cpu", "default_dtype": "float32"}).get_calculator(
                "model.pt"
            )
        mock_cls.assert_called_once_with(
            model_paths=["model.pt"], device="cpu", default_dtype="float32"
        )

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_returns_model_path_and_evaluates(self, tmp_path, monkeypatch):
        result, mock_run, _ = self._run_train(tmp_path, monkeypatch)
        assert result == str(_FIT_DIR.resolve() / "committee_stagetwo.model")
        assert mock_run.call_count == 1
        metrics = json.loads((_FIT_DIR / "evaluation_metrics.json").read_text())
        assert set(metrics["splits"]) == {"train", "test"}
        assert (_FIT_DIR / "train_pred.xyz").exists()

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_compute_stress_read_from_mace_kwargs(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, mace_kwargs={"compute_stress": True}
        )
        args = mock_run.call_args.args[0]
        assert args.stress_key == "REF_stresses"
        assert args.loss == "stress"
        assert not hasattr(args, "compute_stress")

    @pytest.mark.unit
    def test_raises_when_e0s_missing_element(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="E0s"):
            self._run_train(
                tmp_path,
                monkeypatch,
                mace_kwargs={"E0s": {"H": -1.0}},
                elements=["H", "O"],
            )

    @pytest.mark.unit
    def test_e0s_covering_all_elements_does_not_raise(self, tmp_path, monkeypatch):
        self._run_train(
            tmp_path,
            monkeypatch,
            mace_kwargs={"E0s": {"H": -1.0, "O": -2.0}},
            elements=["H", "O"],
        )

    @pytest.mark.unit
    def test_defaults_e0s_from_isolated_atom_energies_when_unset(
        self, tmp_path, monkeypatch
    ):
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, elements=["H"], isolated_atom_energies={"H": -13.6}
        )
        assert mock_run.call_args.args[0].E0s == "{1: -13.6}"

    @pytest.mark.unit
    def test_explicit_e0s_wins_over_isolated_atom_default(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            mace_kwargs={"E0s": {"H": -1.0}},
            elements=["H"],
            isolated_atom_energies={"H": -13.6},
        )
        assert mock_run.call_args.args[0].E0s == "{1: -1.0}"

    @pytest.mark.unit
    def test_raises_when_no_e0s_and_no_isolated_atom_energies(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(ValueError, match="E0s"):
            self._run_train(
                tmp_path, monkeypatch, elements=["H"], isolated_atom_energies={}
            )

    @pytest.mark.unit
    def test_no_e0s_passed_when_unset_and_elements_not_given(
        self, tmp_path, monkeypatch
    ):
        """No elements (a direct caller): nothing to check or pass, so MACE
        falls back to IsolatedAtom frames in the training file."""
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, isolated_atom_energies={}
        )
        assert not hasattr(mock_run.call_args.args[0], "E0s")

    @pytest.mark.unit
    def test_string_e0s_passed_through(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, mace_kwargs={"E0s": "average"}, elements=["H"]
        )
        assert mock_run.call_args.args[0].E0s == "average"

    @pytest.mark.unit
    def test_raises_when_seed_in_mace_kwargs(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="seed"):
            self._run_train(tmp_path, monkeypatch, mace_kwargs={"seed": 1})

    @pytest.mark.unit
    def test_no_valid_file_passed_when_valid_atoms_path_is_none(
        self, tmp_path, monkeypatch
    ):
        _, mock_run, _ = self._run_train(tmp_path, monkeypatch)
        assert not hasattr(mock_run.call_args.args[0], "valid_file")

    @pytest.mark.unit
    def test_valid_file_passed_when_valid_atoms_path_given(self, tmp_path, monkeypatch):
        valid_path = tmp_path / "valid.xyz"
        _write_structures(valid_path, 2)
        _, mock_run, _ = self._run_train(
            tmp_path, monkeypatch, valid_atoms_path=str(valid_path)
        )
        assert mock_run.call_args.args[0].valid_file == str(valid_path.resolve())

    @pytest.mark.unit
    def test_max_num_epochs_defaults_to_dynamic_when_unset(self, tmp_path, monkeypatch):
        """Unset -> dynamic (4 structures pin to the cap, 300), never MACE's
        own 2048 or an old fixed fallback."""
        _, mock_run, _ = self._run_train(tmp_path, monkeypatch, mace_kwargs={})
        assert mock_run.call_args.args[0].max_num_epochs == 300

    @pytest.mark.unit
    def test_dynamic_epochs_uses_train_plus_valid_count(self, tmp_path, monkeypatch):
        """950 train + 50 valid -> ceil(200_000/1000) = 200; train alone
        would give 211, so a regression to the train-only count shows."""
        monkeypatch.chdir(tmp_path)
        _write_structures(tmp_path / "train.xyz", 950)
        _write_structures(tmp_path / "valid.xyz", 50)
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
            _trainer({"batch_size": 1, "max_num_epochs": "dynamic"}).train(
                "train.xyz", "valid.xyz", "test.xyz", 803, fit_dir=_FIT_DIR
            )
        assert captured["max_num_epochs"] == 200


class TestMultiheadFineTuning:
    """Multihead (replay) fine-tuning: general.elements become MACE's
    atomic_numbers, and the replay data MACE adds to every epoch counts
    toward the dynamic epoch size (MACE 0.3.16's arithmetic)."""

    @staticmethod
    def _fit(tmp_path, monkeypatch, mace_kwargs, *, n_train, n_valid=0, **kw):
        monkeypatch.chdir(tmp_path)
        _write_structures(tmp_path / "train.xyz", n_train)
        _write_structures(tmp_path / "test.xyz", 1)
        valid = None
        if n_valid:
            _write_structures(tmp_path / "valid.xyz", n_valid)
            valid = "valid.xyz"
        captured = {}

        def fake_run(args):
            captured.update(vars(args))

        with (
            patch("alomancy.mlip.mace.trainer.run", side_effect=fake_run),
            patch("alomancy.mlip.mace.trainer.tools") as mock_tools,
        ):
            mock_tools.build_default_arg_parser.return_value.parse_args.return_value = (
                argparse.Namespace()
            )
            if "elements" in kw:  # train() requires E0s for every element
                kw.setdefault(
                    "isolated_atom_energies", dict.fromkeys(kw["elements"], -1.0)
                )
            _trainer(mace_kwargs).train(
                "train.xyz", valid, "test.xyz", 803, fit_dir=_FIT_DIR, **kw
            )
        sidecar = tmp_path / _FIT_DIR / "resolved_mace_epochs.json"
        return captured, json.loads(sidecar.read_text())

    _MH: ClassVar[dict[str, str]] = {
        "foundation_model": "/models/mace-mpa-0-medium.model",
        "pt_train_file": "mp",
    }

    @pytest.mark.unit
    def test_elements_become_atomic_numbers(self, tmp_path, monkeypatch):
        args, _ = self._fit(
            tmp_path, monkeypatch, dict(self._MH), n_train=4, elements=["H", "C"]
        )
        assert args["atomic_numbers"] == "[1, 6]"

    @pytest.mark.unit
    def test_replay_and_mace_copies_set_the_epochs(self, tmp_path, monkeypatch):
        """num_samples_pt 2000 -> replay train split 2000 - 200 = 1800; 100
        train structures are 0.0556 of it, under 0.1, so MACE adds
        int(0.1 / 0.0556) = 1 copy: 1800 + 200 + 5 valid = 2005 structures
        per epoch -> ceil(200_000 * 4 / 2005) = 399, clamped to 300; with
        batch_size 1, ceil(200_000 / 2005) = 100 (train alone would give
        1905)."""
        kwargs = {**self._MH, "num_samples_pt": 2000, "batch_size": 1}
        args, sidecar = self._fit(
            tmp_path, monkeypatch, kwargs, n_train=100, n_valid=5, elements=["H"]
        )
        assert args["max_num_epochs"] == 100
        assert sidecar["replay_train_structures"] == 1800
        assert sidecar["fine_tuning_copies"] == 1
        assert sidecar["epoch_train_structures"] == 2000

    @pytest.mark.unit
    def test_no_copies_when_the_data_is_large_enough(self, tmp_path, monkeypatch):
        kwargs = {**self._MH, "num_samples_pt": 1000, "batch_size": 1}
        _, sidecar = self._fit(
            tmp_path, monkeypatch, kwargs, n_train=100, elements=["H"]
        )
        # 100 / 900 = 0.111 >= 0.1: MACE adds nothing.
        assert sidecar["fine_tuning_copies"] == 0
        assert sidecar["epoch_train_structures"] == 1000

    @pytest.mark.unit
    def test_a_replay_file_is_counted_whole(self, tmp_path, monkeypatch):
        _write_structures(tmp_path / "replay.xyz", 30)
        kwargs = {
            "foundation_model": "/models/f.model",
            "pt_train_file": str(tmp_path / "replay.xyz"),
            "batch_size": 1,
        }
        _, sidecar = self._fit(
            tmp_path, monkeypatch, kwargs, n_train=10, elements=["H"]
        )
        assert sidecar["replay_structures"] == 30
        assert sidecar["replay_train_structures"] == 27

    @pytest.mark.unit
    def test_an_mp_foundation_model_replays_without_pt_train_file(
        self, tmp_path, monkeypatch
    ):
        """Once multihead is asked for: naive is the default (TestNaive...)."""
        args, sidecar = self._fit(
            tmp_path,
            monkeypatch,
            {"foundation_model": "medium", "multiheads_finetuning": True},
            n_train=4,
            elements=["H"],
        )
        assert args["atomic_numbers"] == "[1]"
        assert sidecar["replay_structures"] == 10000

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "mace_kwargs",
        [
            {},
            {"foundation_model": "/models/f.model"},
            {**_MH, "multiheads_finetuning": False},
            {**_MH, "multiheads_finetuning": "False"},
        ],
    )
    def test_single_head_fits_are_unchanged(self, tmp_path, monkeypatch, mace_kwargs):
        args, sidecar = self._fit(
            tmp_path,
            monkeypatch,
            {**mace_kwargs, "batch_size": 1},
            n_train=950,
            n_valid=50,
            elements=["H"],
        )
        assert "atomic_numbers" not in args
        assert args["max_num_epochs"] == 200
        assert set(sidecar) == {"max_num_epochs", "start_swa"}

    @pytest.mark.unit
    def test_a_non_mp_foundation_without_replay_is_warned_about(
        self, tmp_path, monkeypatch, caplog
    ):
        with caplog.at_level("WARNING", logger="alomancy.mlip.mace.trainer"):
            self._fit(
                tmp_path,
                monkeypatch,
                {"foundation_model": "/models/f.model", "multiheads_finetuning": True},
                n_train=4,
                elements=["H"],
            )
        assert any(
            getattr(r, "event", None) == "multihead_finetuning_disabled"
            for r in caplog.records
        )

    @pytest.mark.unit
    def test_a_naive_fine_tune_is_not_warned_about(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="alomancy.mlip.mace.trainer"):
            self._fit(
                tmp_path,
                monkeypatch,
                {"foundation_model": "/models/f.model"},
                n_train=4,
                elements=["H"],
            )
        assert not any(
            getattr(r, "event", None) == "multihead_finetuning_disabled"
            for r in caplog.records
        )

    @pytest.mark.unit
    def test_a_user_atomic_numbers_wins_and_a_gap_is_warned(
        self, tmp_path, monkeypatch, caplog
    ):
        kwargs = {**self._MH, "atomic_numbers": "[1]"}
        with caplog.at_level("WARNING", logger="alomancy.mlip.mace.trainer"):
            args, _ = self._fit(
                tmp_path, monkeypatch, kwargs, n_train=4, elements=["H", "C"]
            )
        assert args["atomic_numbers"] == "[1]"
        assert "misses [6]" in caplog.text

    @pytest.mark.unit
    def test_explicit_epochs_pass_through(self, tmp_path, monkeypatch):
        args, _ = self._fit(
            tmp_path,
            monkeypatch,
            {**self._MH, "max_num_epochs": 25},
            n_train=4,
            elements=["H"],
        )
        assert args["max_num_epochs"] == 25


class TestNaiveFineTuningDefaults:
    """A foundation model fine-tuned without replay gets the naive settings
    of Tompa et al. 2026 (arXiv:2606.12704), with constant loss weights and
    learning rate through stage two; from-scratch runs are untouched."""

    _NAIVE: ClassVar[dict[str, Any]] = {
        "multiheads_finetuning": False,
        "lr": 1e-3,
        "ema": True,
        "ema_decay": 0.999,
        "clip_grad": 1.0,
        "weight_decay": 0.0,
        "energy_weight": 10,
        "forces_weight": 10,
    }

    @staticmethod
    def _args(tmp_path, monkeypatch, mace_kwargs):
        args, _ = TestMultiheadFineTuning._fit(
            tmp_path, monkeypatch, mace_kwargs, n_train=4, elements=["H"]
        )
        return args

    @pytest.mark.unit
    def test_a_foundation_model_gets_the_naive_settings(self, tmp_path, monkeypatch):
        args = self._args(tmp_path, monkeypatch, {"foundation_model": "/m.model"})
        for key, value in self._NAIVE.items():
            assert args[key] == value, key
        assert args["swa_energy_weight"] == 10
        assert args["swa_forces_weight"] == 10
        assert args["swa_stress_weight"] == 1.0
        assert args["swa_lr"] == 1e-3
        assert "atomic_numbers" not in args

    @pytest.mark.unit
    def test_user_values_win_and_stage_two_follows_them(self, tmp_path, monkeypatch):
        kwargs = {
            "foundation_model": "/m.model",
            "lr": 5e-4,
            "forces_weight": 50,
            "swa_energy_weight": 99,
        }
        args = self._args(tmp_path, monkeypatch, kwargs)
        assert args["lr"] == 5e-4
        assert args["swa_lr"] == 5e-4
        assert args["swa_forces_weight"] == 50
        assert args["swa_energy_weight"] == 99  # set explicitly: kept
        assert args["ema"] is True

    @pytest.mark.unit
    def test_from_scratch_runs_are_unchanged(self, tmp_path, monkeypatch):
        args = self._args(tmp_path, monkeypatch, {})
        assert args["ema"] is None
        assert args["energy_weight"] == 1
        assert args["forces_weight"] == 10
        for key in ("lr", "clip_grad", "weight_decay", "multiheads_finetuning"):
            assert key not in args
        assert not any(key.startswith("swa_") for key in args)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "opt_in", [{"multiheads_finetuning": True}, {"pt_train_file": "mp"}]
    )
    def test_multihead_opt_in_skips_the_naive_settings(
        self, tmp_path, monkeypatch, opt_in
    ):
        args = self._args(
            tmp_path, monkeypatch, {"foundation_model": "medium", **opt_in}
        )
        assert "lr" not in args  # MACE sets its own multihead lr/EMA
        assert args["ema"] is None
        assert not any(key.startswith("swa_") for key in args)

    @pytest.mark.unit
    def test_effective_kwargs_show_the_naive_settings(self):
        effective = _trainer({"foundation_model": "/m.model"}).effective_kwargs()
        for key, value in self._NAIVE.items():
            assert effective[key] == value, key
        assert "lr" not in _trainer({}).effective_kwargs()


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
