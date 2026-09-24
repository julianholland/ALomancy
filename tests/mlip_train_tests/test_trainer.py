"""Tests for mlip/mace/trainer.py -- the MACE backend implementing the
modular AL architecture's trainer entry points (train, get_calculator,
output_paths, read_existing_result).

Unlike mace_wfl.mace_fit (still covered by test_mace_training.py, and
still what production train_mlip calls until the skeleton lands), train()
has zero committee awareness and receives an already-built, explicit
train/valid/test split as file paths rather than deriving one itself."""

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
    _apply_compute_stress_defaults,
    _compute_dynamic_epochs,
    _evaluate_and_save_predictions,
    _write_resolved_mace_epochs,
    get_calculator,
    output_paths,
    read_existing_result,
    train,
)


def _write_structures(path: Path, n: int, config_type: str = "init_dimer") -> None:
    structures = []
    for i in range(n):
        a = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        a.info["config_type"] = f"{config_type}_{i}"
        a.info["REF_energy"] = 1.0
        a.arrays["REF_forces"] = np.zeros((1, 3))
        structures.append(a)
    write(str(path), structures, format="extxyz")


@pytest.mark.unit
class TestComputeDynamicEpochs:
    def test_typical_mid_run_value(self):
        assert _compute_dynamic_epochs(batch_size=16, n_training_structures=4000) == 300

    def test_large_training_set_hits_floor(self):
        assert (
            _compute_dynamic_epochs(batch_size=16, n_training_structures=1_000_000)
            == 20
        )

    def test_uncapped_value_between_floor_and_cap(self):
        assert (
            _compute_dynamic_epochs(batch_size=16, n_training_structures=20_000) == 160
        )

    def test_raises_on_zero_training_structures(self):
        with pytest.raises(ValueError):
            _compute_dynamic_epochs(batch_size=16, n_training_structures=0)


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
class TestOutputPaths:
    def test_returns_model_and_metrics_paths(self):
        paths = output_paths({}, base_name="al_loop_0", name="committee", fit_idx=2)
        assert paths == [
            Path("results/al_loop_0/committee/fit_2/committee_stagetwo.model"),
            Path("results/al_loop_0/committee/fit_2/evaluation_metrics.json"),
        ]

    def test_does_not_include_compiled_model(self):
        """The compiled model is deliberately excluded -- MACE silently
        swallows a failed compile, so gating restart on it would make a
        genuinely-successful fit look incomplete forever."""
        paths = output_paths({}, base_name="al_loop_0", name="committee", fit_idx=0)
        assert not any("compiled" in str(p) for p in paths)


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
        fit_dir = Path("results/al_loop_0/committee/fit_0")
        self._write_fit(
            fit_dir,
            {
                "train": prediction_metrics([_predicted(0.1)]),
                "test": prediction_metrics([_predicted(0.2)]),
            },
        )

        model_path, compiled_path, metrics = read_existing_result(
            {}, base_name="al_loop_0", name="committee", fit_idx=0
        )

        assert model_path == str(fit_dir / "committee_stagetwo.model")
        assert compiled_path is None  # never written in this test
        assert set(metrics) == {"train", "test"}

    def test_includes_compiled_path_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fit_dir = Path("results/al_loop_0/committee/fit_0")
        self._write_fit(fit_dir, {"valid": prediction_metrics([_predicted(0.1)])})
        (fit_dir / "committee_stagetwo_compiled.model").write_bytes(b"compiled")

        _, compiled_path, _ = read_existing_result(
            {}, base_name="al_loop_0", name="committee", fit_idx=0
        )
        assert compiled_path == str(fit_dir / "committee_stagetwo_compiled.model")

    def test_raises_when_nothing_valid_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No valid checkpoint-verified"):
            read_existing_result({}, base_name="al_loop_0", name="committee", fit_idx=0)

    def test_raises_when_checkpoint_changed_since_evaluation(
        self, tmp_path, monkeypatch
    ):
        """A stale/corrupted on-disk state must raise, not be silently
        treated as a valid cached result (goes through read_evaluation's
        checksum check, same as production)."""
        monkeypatch.chdir(tmp_path)
        fit_dir = Path("results/al_loop_0/committee/fit_0")
        self._write_fit(fit_dir, {"test": prediction_metrics([_predicted(0.1)])})
        (fit_dir / "committee_stagetwo.model").write_bytes(b"changed after eval")

        with pytest.raises(ValueError, match="No valid checkpoint-verified"):
            read_existing_result({}, base_name="al_loop_0", name="committee", fit_idx=0)


@pytest.mark.unit
class TestGetCalculator:
    def test_builds_mace_calculator_with_given_device_and_dtype(self):
        with patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_calculator("model.pt", {"device": "cpu", "default_dtype": "float32"})
        mock_cls.assert_called_once_with(
            model_paths=["model.pt"], device="cpu", default_dtype="float32"
        )

    def test_defaults_dtype_to_float64(self):
        with patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_calculator("model.pt", {"device": "cpu"})
        assert mock_cls.call_args.kwargs["default_dtype"] == "float64"


@pytest.mark.unit
class TestEvaluateAndSavePredictions:
    def test_writes_pred_files_and_returns_metrics_for_every_split(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        model_path = Path("committee_stagetwo.model")
        model_path.write_bytes(b"checkpoint")
        _write_structures(Path("train.xyz"), 2)
        _write_structures(Path("test.xyz"), 1)

        monkeypatch.setattr(Atoms, "get_potential_energy", lambda self: 1.0)
        monkeypatch.setattr(
            Atoms, "get_forces", lambda self: np.zeros((1, 3)), raising=False
        )
        with patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_cls:
            mock_cls.return_value = MagicMock()
            metrics = _evaluate_and_save_predictions(
                model_path,
                {"train": Path("train.xyz"), "test": Path("test.xyz")},
            )

        assert set(metrics) == {"train", "test"}
        assert Path("train_pred.xyz").exists()
        assert Path("test_pred.xyz").exists()
        assert Path("evaluation_metrics.json").exists()

    def test_missing_model_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        metrics = _evaluate_and_save_predictions(
            Path("does_not_exist.model"), {"train": Path("train.xyz")}
        )
        assert metrics == {}


def _fake_run_writes_model(name: str, fit_dir: Path) -> None:
    """Simulate MACE's run() side effect: it writes the uncompiled stagetwo
    model into the current directory (train()'s worker has already
    chdir'd into fit_dir by the time this runs)."""
    (fit_dir / f"{name}_stagetwo.model").write_bytes(b"checkpoint")


@pytest.mark.unit
class TestTrain:
    def _run_train(self, tmp_path, monkeypatch, **overrides):
        monkeypatch.chdir(tmp_path)
        train_path = tmp_path / "train.xyz"
        test_path = tmp_path / "test.xyz"
        _write_structures(train_path, 4)
        _write_structures(test_path, 2)

        config = {
            "mace_kwargs": {"energy_key": "REF_energy", "forces_key": "REF_forces"},
            "max_num_epochs": 40,
        }
        config.update(overrides.pop("config_overrides", {}))

        kwargs = {
            "train_atoms_path": str(train_path),
            "valid_atoms_path": None,
            "test_atoms_path": str(test_path),
            "config": config,
            "fit_seed": 803,
            "base_name": "al_loop_0",
            "name": "committee",
            "fit_idx": 0,
            "hpc": {},
            "max_time": "1:00:00",
        }
        kwargs.update(overrides)

        fit_dir = Path(
            "results", kwargs["base_name"], kwargs["name"], f"fit_{kwargs['fit_idx']}"
        )
        # Resolve eagerly, while cwd is still tmp_path -- fake_run below runs
        # AFTER train() has already chdir'd into fit_dir, so resolving lazily
        # there would resolve the relative path against itself.
        expected_cwd = fit_dir.resolve()

        def fake_run(args):
            _fake_run_writes_model(kwargs["name"], Path.cwd())
            assert Path.cwd() == expected_cwd

        with (
            patch("alomancy.mlip.mace.trainer.run", side_effect=fake_run) as mock_run,
            patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_calc_cls,
            patch("alomancy.mlip.mace.trainer.tools") as mock_tools,
        ):
            mock_calc_cls.return_value = MagicMock()
            # A real, empty argparse.Namespace -- not a MagicMock -- so
            # hasattr()/getattr() on `args` behave like real argparse
            # (unset attributes genuinely don't exist) regardless of
            # whether some other test file's module-level
            # sys.modules.setdefault("mace", MagicMock()) has already run
            # in this pytest session (a real, session-wide ordering hazard:
            # whichever test file is collected first determines whether
            # `mace` resolves to the real package or a permanent mock for
            # every test after it -- this makes TestTrain's own tests
            # independent of that).
            mock_tools.build_default_arg_parser.return_value.parse_args.return_value = (
                argparse.Namespace()
            )
            monkeypatch.setattr(Atoms, "get_potential_energy", lambda self: 1.0)
            monkeypatch.setattr(
                Atoms, "get_forces", lambda self: np.zeros((1, 3)), raising=False
            )
            result = train(**kwargs)
        return result, mock_run, fit_dir

    def test_defaults_energy_key_when_missing(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            config_overrides={"mace_kwargs": {"forces_key": "REF_forces"}},
        )
        args = mock_run.call_args.args[0]
        assert args.energy_key == "REF_energy"

    def test_defaults_forces_key_when_missing(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            config_overrides={"mace_kwargs": {"energy_key": "REF_energy"}},
        )
        args = mock_run.call_args.args[0]
        assert args.forces_key == "REF_forces"

    def test_raises_when_e0s_missing_element(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="E0s"):
            self._run_train(
                tmp_path,
                monkeypatch,
                elements=["H", "O"],
                config_overrides={
                    "mace_kwargs": {
                        "energy_key": "REF_energy",
                        "forces_key": "REF_forces",
                        "E0s": {"H": -1.0},
                    }
                },
            )

    def test_e0s_covering_all_elements_does_not_raise(self, tmp_path, monkeypatch):
        self._run_train(
            tmp_path,
            monkeypatch,
            elements=["H", "O"],
            config_overrides={
                "mace_kwargs": {
                    "energy_key": "REF_energy",
                    "forces_key": "REF_forces",
                    "E0s": {"H": -1.0, "O": -2.0},
                }
            },
        )

    def test_defaults_e0s_from_isolated_atom_energies_when_unset(
        self, tmp_path, monkeypatch
    ):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            elements=["H"],
            isolated_atom_e0s={"H": -13.6},
        )
        args = mock_run.call_args.args[0]
        assert args.E0s == {"H": -13.6}

    def test_explicit_e0s_wins_over_isolated_atom_default(self, tmp_path, monkeypatch):
        _, mock_run, _ = self._run_train(
            tmp_path,
            monkeypatch,
            elements=["H"],
            isolated_atom_e0s={"H": -13.6},
            config_overrides={
                "mace_kwargs": {
                    "energy_key": "REF_energy",
                    "forces_key": "REF_forces",
                    "E0s": {"H": -1.0},
                }
            },
        )
        args = mock_run.call_args.args[0]
        assert args.E0s == {"H": -1.0}

    def test_raises_when_no_e0s_and_no_isolated_atom_energies(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(ValueError, match="E0s"):
            self._run_train(
                tmp_path,
                monkeypatch,
                elements=["H"],
                isolated_atom_e0s={},
            )

    def test_no_raise_when_e0s_missing_and_elements_not_given(
        self, tmp_path, monkeypatch
    ):
        """elements is None (e.g. an older/direct caller) -> the safety net
        is skipped entirely, matching pre-existing behavior where E0s was
        always optional."""
        self._run_train(tmp_path, monkeypatch, isolated_atom_e0s={})

    def test_raises_when_seed_in_mace_kwargs(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="seed"):
            self._run_train(
                tmp_path,
                monkeypatch,
                config_overrides={
                    "mace_kwargs": {
                        "energy_key": "REF_energy",
                        "forces_key": "REF_forces",
                        "seed": 1,
                    }
                },
            )

    def test_returns_model_path_and_metrics(self, tmp_path, monkeypatch):
        (result_model, result_compiled, result_metrics), mock_run, fit_dir = (
            self._run_train(tmp_path, monkeypatch)
        )
        assert result_model == str(fit_dir / "committee_stagetwo.model")
        assert result_compiled is None
        assert "train" in result_metrics
        assert "test" in result_metrics
        assert mock_run.call_count == 1

    def test_no_valid_file_passed_to_mace_when_valid_atoms_path_is_none(
        self, tmp_path, monkeypatch
    ):
        _result, mock_run, _fit_dir = self._run_train(tmp_path, monkeypatch)
        args = mock_run.call_args.args[0]
        assert not hasattr(args, "valid_file") or args.valid_file is None

    def test_valid_file_passed_when_valid_atoms_path_given(self, tmp_path, monkeypatch):
        # Write the valid.xyz file into the *eventual* tmp_path cwd before
        # _run_train chdir's there and creates train.xyz/test.xyz alongside it.
        valid_path = tmp_path / "valid.xyz"
        _write_structures(valid_path, 2)

        _result, mock_run, _fit_dir = self._run_train(
            tmp_path, monkeypatch, valid_atoms_path=str(valid_path)
        )
        args = mock_run.call_args.args[0]
        assert args.valid_file == str(valid_path.resolve())

    def test_dynamic_epochs_uses_train_plus_valid_count(self, tmp_path, monkeypatch):
        """Picks counts in the "uncapped" epoch range so "train+valid
        combined" (the correct, pre-carve-out pool size) and "train alone"
        give different, neither-clamped answers -- a regression to
        len(train_atoms) alone would be caught rather than both landing on
        the same clamp value."""
        monkeypatch.chdir(tmp_path)
        train_path = tmp_path / "train.xyz"
        valid_path = tmp_path / "valid.xyz"
        test_path = tmp_path / "test.xyz"
        # 19000 train + 1000 valid = 20000 total -> ceil(3_200_000/20000) = 160
        # 19000 train alone -> ceil(3_200_000/19000) = 169 (would be picked
        # up wrongly if the formula used only len(train_atoms)).
        _write_structures(train_path, 19000)
        _write_structures(valid_path, 1000)
        _write_structures(test_path, 1)

        captured_args = {}

        def fake_run(args):
            captured_args["max_num_epochs"] = args.max_num_epochs
            (Path.cwd() / "committee_stagetwo.model").write_bytes(b"checkpoint")

        with (
            patch("alomancy.mlip.mace.trainer.run", side_effect=fake_run),
            patch("alomancy.mlip.mace.trainer.MACECalculator") as mock_calc_cls,
            patch("alomancy.mlip.mace.trainer.tools") as mock_tools,
        ):
            mock_calc_cls.return_value = MagicMock()
            mock_tools.build_default_arg_parser.return_value.parse_args.return_value = (
                argparse.Namespace()
            )
            monkeypatch.setattr(Atoms, "get_potential_energy", lambda self: 1.0)
            monkeypatch.setattr(
                Atoms, "get_forces", lambda self: np.zeros((1, 3)), raising=False
            )
            train(
                train_atoms_path=str(train_path),
                valid_atoms_path=str(valid_path),
                test_atoms_path=str(test_path),
                config={
                    "mace_kwargs": {
                        "energy_key": "REF_energy",
                        "forces_key": "REF_forces",
                        "batch_size": 16,
                    },
                    "max_num_epochs": "dynamic",
                },
                fit_seed=803,
                base_name="al_loop_0",
                name="committee",
                fit_idx=0,
                hpc={},
                max_time="1:00:00",
            )

        assert captured_args["max_num_epochs"] == 160
