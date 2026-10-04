"""Tests for mlip/sevennet/trainer.py: label wiring (energy, forces, stress
sign/order), the stress-gap masking, config handling, the reference-energy
shift, and a real end-to-end SevenNet fit on CPU with a tiny model.

Tests that build graphs or train need sevenn installed and skip otherwise;
config handling runs without it (sevenn is only imported inside fit)."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from alomancy.mlip.base import get_trainer
from alomancy.mlip.sevennet.trainer import (
    _SEVENNET_KWARGS_DEFAULTS,
    SevenNetTrainer,
    _sevenn_stress,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label(atoms: Atoms, config_type: str) -> Atoms:
    """EMT labels under our keys; stress only for periodic cells."""
    atoms.calc = EMT()
    atoms.info["REF_energy"] = atoms.get_potential_energy()
    atoms.arrays["REF_forces"] = atoms.get_forces()
    if atoms.pbc.all():
        atoms.info["REF_stresses"] = atoms.get_stress()
    atoms.calc = None
    atoms.info["config_type"] = config_type
    return atoms


def _cell(i: int) -> Atoms:
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))
    a.symbols[[0, 3]] = "Al"
    a.rattle(0.1, seed=i)
    return _label(a, "init_amorphous")


def _split(n_cells: int, seed0: int = 0) -> list[Atoms]:
    """Periodic cells with stress, plus a non-periodic dimer (no stress) and
    an isolated atom."""
    return [
        *(_cell(seed0 + i) for i in range(n_cells)),
        _label(Atoms("CuAl", positions=[[0, 0, 0], [0, 0, 2.5]]), "init_dimer"),
        _label(Atoms("Cu", positions=[[0, 0, 0]]), "IsolatedAtom"),
    ]


_TINY = {
    "model": {"channel": 4, "lmax": 1, "num_convolution_layer": 1},
    "train": {"epoch": 2, "device": "cpu", "is_train_stress": True},
}


# ---------------------------------------------------------------------------
# Label wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sevenn_stress_reorders_and_flips_sign():
    # ASE Voigt order: xx yy zz yz xz xy
    ase_voigt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # SevenNet order: xx yy zz xy yz zx, opposite sign
    assert _sevenn_stress(ase_voigt).tolist() == [-1.0, -2.0, -3.0, -6.0, -4.0, -5.0]


@pytest.mark.unit
def test_read_graphs_labels_and_missing_stress(tmp_path):
    pytest.importorskip("sevenn")
    import sevenn._keys as KEY

    from alomancy.mlip.sevennet.trainer import _read_graphs

    atoms = _split(1)
    write(tmp_path / "s.xyz", atoms)
    graphs = _read_graphs(tmp_path / "s.xyz", cutoff=5.0)

    # IsolatedAtom is kept: SevenNet doesn't pin a lone atom to the shift.
    assert len(graphs) == 3
    cell, dimer, isolated = graphs
    assert float(cell[KEY.ENERGY]) == pytest.approx(atoms[0].info["REF_energy"])
    np.testing.assert_allclose(
        cell[KEY.FORCE].numpy(), atoms[0].arrays["REF_forces"], atol=1e-6
    )
    np.testing.assert_allclose(
        cell[KEY.STRESS].numpy().ravel(),
        _sevenn_stress(atoms[0].info["REF_stresses"]),
        atol=1e-6,
    )
    assert np.isnan(dimer[KEY.STRESS].numpy()).all()
    assert isolated[KEY.EDGE_IDX].shape[1] == 0


@pytest.mark.unit
def test_read_graphs_rejects_missing_forces(tmp_path):
    pytest.importorskip("sevenn")
    from alomancy.mlip.sevennet.trainer import _read_graphs

    a = _cell(0)
    del a.arrays["REF_forces"]
    write(tmp_path / "s.xyz", a)
    with pytest.raises(ValueError, match="REF_energy/REF_forces"):
        _read_graphs(tmp_path / "s.xyz", cutoff=5.0)


# ---------------------------------------------------------------------------
# Stress-gap masking
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_finite_only_ignores_nan_labels():
    torch = pytest.importorskip("torch")
    from alomancy.mlip.sevennet.trainer import _FiniteOnly

    mse = torch.nn.MSELoss()
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ref = torch.tensor([1.5, float("nan"), 2.0, float("nan")])

    loss = _FiniteOnly(mse)(pred, ref)
    assert float(loss) == pytest.approx(float(mse(pred[[0, 2]], ref[[0, 2]])))

    all_nan = torch.full((4,), float("nan"))
    assert float(_FiniteOnly(mse)(pred, all_nan)) == 0.0
    # As an error metric, a batch with no stress isn't counted at all.
    assert _FiniteOnly(mse, skip_empty=True)(all_nan, pred).numel() == 0


# ---------------------------------------------------------------------------
# Config handling (no sevenn needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sections_merge_over_defaults():
    trainer = SevenNetTrainer({"sevennet_kwargs": {"model": {"channel": 64}}})
    assert trainer.kwargs["model"]["channel"] == 64
    assert trainer.kwargs["model"]["lmax"] == _SEVENNET_KWARGS_DEFAULTS["model"]["lmax"]
    assert trainer.kwargs["train"] == _SEVENNET_KWARGS_DEFAULTS["train"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"modle": {}}, "Unknown sevennet_kwargs"),
        ({"train": {"random_seed": 1}}, "train.random_seed"),
        ({"data": {"shift": "per_atom_energy_mean"}}, "data.shift"),
        ({"model": {"chemical_species": "auto"}}, "model.chemical_species"),
    ],
)
def test_rejected_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SevenNetTrainer({"sevennet_kwargs": kwargs})


@pytest.mark.unit
def test_reference_energies_resolve_to_atomic_numbers():
    trainer = SevenNetTrainer({})
    resolved = trainer.resolve_isolated_atom_energies(
        ["Cu", "Al"], {"Cu": -0.1, "Al": -0.2}
    )
    assert resolved == {29: -0.1, 13: -0.2}

    override = SevenNetTrainer(
        {
            "sevennet_kwargs": {
                "isolated_atom_reference_energies": {"Cu": -1.0, 13: -2.0}
            }
        }
    )
    assert override.resolve_isolated_atom_energies(["Cu", "Al"], None) == {
        29: -1.0,
        13: -2.0,
    }


@pytest.mark.unit
def test_registered_as_sevennet():
    assert isinstance(get_trainer("sevennet", {}), SevenNetTrainer)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _write_splits(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for tag, n, seed0 in (("train", 6, 0), ("valid", 2, 100), ("test", 2, 200)):
        paths[tag] = tmp_path / f"{tag}.xyz"
        write(paths[tag], _split(n, seed0))
    return paths


@pytest.mark.unit
@pytest.mark.slow  # real SevenNet training
def test_end_to_end_fit_with_valid(tmp_path):
    pytest.importorskip("sevenn")
    import torch

    paths = _write_splits(tmp_path)
    trainer = get_trainer("sevennet", {"sevennet_kwargs": _TINY})
    fit_dir = tmp_path / "fit_0"

    model = trainer.train(
        paths["train"],
        paths["valid"],
        paths["test"],
        803,
        fit_dir=fit_dir,
        elements=["Cu", "Al"],
        isolated_atom_energies={"Cu": -0.1, "Al": -0.2},
    )

    assert model == str(fit_dir / "training_sevennet.pth")
    _, _, metrics = trainer.read_existing_result(fit_dir)
    assert set(metrics) == {"train", "valid", "test"}
    assert np.isfinite(metrics["valid"]["mae_f"])
    for tag in ("train", "valid", "test"):
        assert (fit_dir / f"{tag}_pred.xyz").exists()

    history = trainer.training_history(fit_dir, 803)
    assert history.frame["epoch"].to_list() == [1, 1, 2, 2]
    assert history.frame["split"].to_list() == ["train", "valid", "train", "valid"]
    assert {"loss", "mae_e_per_atom", "mae_f"} <= set(history.frame.columns)
    # Stress was trained with a stress-less dimer in every split: no NaN.
    assert history.frame["loss"].is_finite().all()
    assert history.selected_epoch in (1, 2)
    assert history.stage_two_epoch is None

    # One slim model left: SevenNet's checkpoints are gone, and the model
    # holds no optimizer state (evaluation above already loaded it).
    assert not list(fit_dir.glob("checkpoint_*.pth"))
    saved = torch.load(fit_dir / "training_sevennet.pth", weights_only=False)
    assert set(saved) == {"model_state_dict", "config", "epoch"}
    assert saved["epoch"] == history.selected_epoch
    assert (fit_dir / "lc.csv").exists()
    assert (fit_dir / "log.sevenn").stat().st_size > 0

    info = json.loads((fit_dir / "training_sevennet_fit.json").read_text())
    assert info["epoch"] == 2
    assert info["chemical_species"] == ["Al", "Cu"]
    assert info["shift"] == {"Cu": -0.1, "Al": -0.2}


@pytest.mark.unit
@pytest.mark.slow  # real SevenNet training
def test_end_to_end_without_valid_keeps_last_epoch(tmp_path):
    pytest.importorskip("sevenn")
    paths = _write_splits(tmp_path)
    trainer = get_trainer("sevennet", {"sevennet_kwargs": _TINY})
    fit_dir = tmp_path / "fit_0"

    trainer.train(
        paths["train"],
        None,
        paths["test"],
        803,
        fit_dir=fit_dir,
        elements=["Cu", "Al"],
        isolated_atom_energies={"Cu": -0.1, "Al": -0.2},
    )

    history = trainer.training_history(fit_dir, 803)
    assert set(history.frame["split"].to_list()) == {"train"}
    assert history.selected_epoch == 2
    assert not list(fit_dir.glob("checkpoint_*.pth"))

    # A retry into the same directory starts clean: lc.csv, never lc0.csv.
    trainer.train(
        paths["train"],
        None,
        paths["test"],
        803,
        fit_dir=fit_dir,
        elements=["Cu", "Al"],
        isolated_atom_energies={"Cu": -0.1, "Al": -0.2},
    )
    assert sorted(p.name for p in fit_dir.glob("lc*.csv")) == ["lc.csv"]
    assert trainer.training_history(fit_dir, 803).frame.height == 2


@pytest.mark.unit
def test_cleanup_removes_synced_back_checkpoints(tmp_path):
    """ExPyRe's additive sync can bring back checkpoints the remote node
    already deleted; the local clean-up pass removes them, keeping the model."""
    for name in ("checkpoint_0.pth", "checkpoint_best.pth", "checkpoint_20.pth"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "training_sevennet.pth").write_bytes(b"model")

    SevenNetTrainer({}).cleanup(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["training_sevennet.pth"]


@pytest.mark.unit
def test_report_section_from_training_history(tmp_path, monkeypatch):
    """The generic trainer_section reads the best fit's history through the
    trainer: training length, kept epoch and a validation curve."""
    monkeypatch.chdir(tmp_path)
    fit_dir = tmp_path / "results/al_loop_0/training/fit_1"
    fit_dir.mkdir(parents=True)
    (fit_dir / "lc.csv").write_text(
        "epoch,lr,trainset_Energy_MAE,trainset_Force_MAE,trainset_TotalLoss,"
        "validset_Energy_MAE,validset_Force_MAE,validset_TotalLoss\n"
        "1,0.005,0.2,0.5,1.0,0.3,0.6,1.2\n"
        "2,0.005,0.1,0.4,0.8,0.2,0.5,0.9\n"
    )
    (fit_dir / "training_sevennet_fit.json").write_text('{"selected_epoch": 2}')
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    section = SevenNetTrainer({}).report_section(
        {"model": {"best_fit_idx": 1}, "seed": 803},
        base_name="al_loop_0",
        plots_dir=plots_dir,
        config={},
    )

    assert section.title == "MLIP training: sevennet"
    assert "trained for 2 epochs" in section.lines[0]
    assert "weights kept from epoch 2" in section.lines[0]
    assert (plots_dir / "best_model_training_curve.png").exists()
