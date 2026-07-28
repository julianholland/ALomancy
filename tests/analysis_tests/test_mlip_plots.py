"""Unit tests for mlip_plots helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import write

# ---------------------------------------------------------------------------
# _get_stage_two_epoch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_stage_two_epoch_default():
    from alomancy.analysis.mlip_plots import _get_stage_two_epoch

    result = _get_stage_two_epoch({"max_num_epochs": 80, "mace_fit_kwargs": {}})
    assert result == 64


@pytest.mark.unit
def test_get_stage_two_epoch_explicit_start_swa():
    from alomancy.analysis.mlip_plots import _get_stage_two_epoch

    result = _get_stage_two_epoch({"mace_fit_kwargs": {"start_swa": 100}})
    assert result == 100


@pytest.mark.unit
def test_get_stage_two_epoch_top_level_max_epochs():
    from alomancy.analysis.mlip_plots import _get_stage_two_epoch

    result = _get_stage_two_epoch({"max_num_epochs": 200, "mace_fit_kwargs": {}})
    assert result == 160


@pytest.mark.unit
def test_get_stage_two_epoch_max_epochs_in_kwargs():
    from alomancy.analysis.mlip_plots import _get_stage_two_epoch

    # max_num_epochs inside mace_fit_kwargs, not at top level
    result = _get_stage_two_epoch({"mace_fit_kwargs": {"max_num_epochs": 100}})
    assert result == 80


@pytest.mark.unit
def test_get_stage_two_epoch_default_80_when_none():
    from alomancy.analysis.mlip_plots import _get_stage_two_epoch

    # Neither max_num_epochs nor start_swa anywhere → default 80 → 64
    result = _get_stage_two_epoch({"mace_fit_kwargs": {}})
    assert result == 64


# ---------------------------------------------------------------------------
# _parse_training_jsonl
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


@pytest.mark.unit
def test_parse_training_jsonl_filters_mode(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_training_jsonl

    fit_dir = tmp_path / "fit_0"
    txt = fit_dir / "results" / "mymodel_run-803_train.txt"
    records = [
        {
            "epoch": None,
            "mode": "eval",
            "loss": 1.0,
            "mae_e": 0.5,
            "mae_f": 0.4,
            "mae_e_per_atom": 0.05,
        },
        {
            "epoch": 0,
            "mode": "opt",
            "loss": 0.9,
            "mae_e": 0.45,
            "mae_f": 0.38,
            "mae_e_per_atom": 0.045,
        },
        {
            "epoch": 0,
            "mode": "eval",
            "loss": 0.9,
            "mae_e": 0.45,
            "mae_f": 0.38,
            "mae_e_per_atom": 0.045,
        },
        {
            "epoch": 1,
            "mode": "eval",
            "loss": 0.8,
            "mae_e": 0.40,
            "mae_f": 0.35,
            "mae_e_per_atom": 0.040,
        },
    ]
    _write_jsonl(txt, records)

    df = _parse_training_jsonl(fit_dir, "mymodel", 803)

    assert df is not None
    # null-epoch row and opt-mode row must be excluded
    assert len(df) == 2
    assert set(df.index.tolist()) == {0, 1}


@pytest.mark.unit
def test_parse_training_jsonl_dataframe_columns(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_training_jsonl

    fit_dir = tmp_path / "fit_0"
    txt = fit_dir / "results" / "mymodel_run-803_train.txt"
    records = [
        {
            "epoch": 0,
            "mode": "eval",
            "loss": 0.9,
            "mae_e": 0.45,
            "mae_f": 0.38,
            "mae_e_per_atom": 0.045,
        },
        {
            "epoch": 1,
            "mode": "eval",
            "loss": 0.8,
            "mae_e": 0.40,
            "mae_f": 0.35,
            "mae_e_per_atom": 0.040,
        },
    ]
    _write_jsonl(txt, records)

    df = _parse_training_jsonl(fit_dir, "mymodel", 803)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    for col in ("loss", "mae_e", "mae_f"):
        assert col in df.columns


@pytest.mark.unit
def test_parse_training_jsonl_missing_file(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_training_jsonl

    fit_dir = tmp_path / "fit_0"
    df = _parse_training_jsonl(fit_dir, "mymodel", 803)
    assert df is None


# ---------------------------------------------------------------------------
# _parse_used_epoch
# ---------------------------------------------------------------------------


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_parse_used_epoch_found(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_used_epoch

    fit_dir = tmp_path / "fit_0"
    log = fit_dir / "logs" / "mymodel_run-803.log"
    _write_log(
        log,
        [
            "2026-07-13 21:35:00.000 INFO: Starting training",
            "2026-07-13 21:35:55.248 INFO: Loaded Stage two model from epoch 74 for evaluation",
            "2026-07-13 21:36:00.000 INFO: Done",
        ],
    )

    epoch = _parse_used_epoch(fit_dir, "mymodel", 803)
    assert epoch == 74


@pytest.mark.unit
def test_parse_used_epoch_not_found(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_used_epoch

    fit_dir = tmp_path / "fit_0"
    log = fit_dir / "logs" / "mymodel_run-803.log"
    _write_log(
        log,
        [
            "2026-07-13 21:35:00.000 INFO: Starting training",
            "2026-07-13 21:36:00.000 INFO: Done",
        ],
    )

    epoch = _parse_used_epoch(fit_dir, "mymodel", 803)
    assert epoch is None


@pytest.mark.unit
def test_parse_used_epoch_missing_file(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_used_epoch

    fit_dir = tmp_path / "fit_0"
    epoch = _parse_used_epoch(fit_dir, "mymodel", 803)
    assert epoch is None


# ---------------------------------------------------------------------------
# glob fallbacks (seed mismatch)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_training_jsonl_glob_fallback(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_training_jsonl

    fit_dir = tmp_path / "fit_0"
    results_dir = fit_dir / "results"
    results_dir.mkdir(parents=True)
    txt = results_dir / "mymodel_run-999_train.txt"
    txt.write_text(
        json.dumps(
            {
                "epoch": 0,
                "mode": "eval",
                "loss": 0.5,
                "mae_e": 0.1,
                "mae_f": 0.2,
                "mae_e_per_atom": 0.05,
            }
        )
        + "\n"
    )
    # Requested seed 803, file has seed 999 — glob fallback should find it.
    df = _parse_training_jsonl(fit_dir, "mymodel", 803)
    assert df is not None
    assert len(df) == 1


@pytest.mark.unit
def test_parse_used_epoch_glob_fallback(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_used_epoch

    fit_dir = tmp_path / "fit_0"
    logs_dir = fit_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "mymodel_run-999.log").write_text(
        "INFO: Loaded Stage two model from epoch 42 for evaluation\n"
    )
    epoch = _parse_used_epoch(fit_dir, "mymodel", 803)
    assert epoch == 42


# ---------------------------------------------------------------------------
# _parse_eval_xyz
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_eval_xyz_missing_file(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_eval_xyz

    result = _parse_eval_xyz(tmp_path / "nonexistent.xyz")
    assert result is None


@pytest.mark.unit
def test_parse_eval_xyz_skips_missing_keys(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_eval_xyz

    # Structure has REF_energy but no mace_energy — should be skipped
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.info["REF_energy"] = -1.0
    xyz_path = tmp_path / "pred.xyz"
    write(str(xyz_path), [atoms], format="extxyz")

    result = _parse_eval_xyz(xyz_path)
    assert result is None


@pytest.mark.unit
def test_parse_eval_xyz_returns_per_atom_energy(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_eval_xyz

    atoms = Atoms("S2", positions=[[0, 0, 0], [0, 0, 2.0]], cell=[10, 10, 10], pbc=True)
    atoms.info["REF_energy"] = -4.0
    atoms.info["mace_energy"] = -3.8
    atoms.arrays["REF_forces"] = np.zeros((2, 3))
    atoms.arrays["mace_forces"] = np.ones((2, 3)) * 0.01
    xyz_path = tmp_path / "pred.xyz"
    write(str(xyz_path), [atoms], format="extxyz")

    e_dft, e_pred, f_dft, f_pred = _parse_eval_xyz(xyz_path)

    assert len(e_dft) == 1
    assert e_dft[0] == pytest.approx(-2.0)  # -4.0 / 2 atoms
    assert e_pred[0] == pytest.approx(-1.9)  # -3.8 / 2 atoms
    assert len(f_dft) == 6  # 2 atoms x 3 components
    assert len(f_pred) == 6


@pytest.mark.unit
def test_parse_eval_xyz_no_forces_still_returns_energy(tmp_path):
    from alomancy.analysis.mlip_plots import _parse_eval_xyz

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.info["REF_energy"] = -1.0
    atoms.info["mace_energy"] = -1.05
    # deliberately no mace_forces / REF_forces
    xyz_path = tmp_path / "pred.xyz"
    write(str(xyz_path), [atoms], format="extxyz")

    result = _parse_eval_xyz(xyz_path)
    assert result is not None
    e_dft, _e_pred, f_dft, _f_pred = result
    assert len(e_dft) == 1
    assert e_dft[0] == pytest.approx(-1.0)
    assert len(f_dft) == 0


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------


def _write_fit_data(base_dir: Path, name: str, seed: int, n_epochs: int = 10) -> None:
    fit_dir = base_dir / f"results/demo/{name}/fit_0"
    results_dir = fit_dir / "results"
    logs_dir = fit_dir / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    txt = results_dir / f"{name}_run-{seed}_train.txt"
    with txt.open("w") as fh:
        for ep in range(n_epochs):
            hf = json.dumps(
                {
                    "epoch": ep,
                    "mode": "eval",
                    "loss": 1.0 - 0.05 * ep,
                    "mae_e": 0.5 - 0.02 * ep,
                    "mae_f": 0.3 - 0.01 * ep,
                    "mae_e_per_atom": 0.05 - 0.002 * ep,
                }
            )
            fh.write(hf + "\n")

    (logs_dir / f"{name}_run-{seed}.log").write_text(
        f"INFO: Loaded Stage two model from epoch {n_epochs - 2} for evaluation\n"
    )


@pytest.mark.unit
def test_plot_training_curves_creates_files(tmp_path, monkeypatch):
    from alomancy.analysis.mlip_plots import plot_training_curves

    monkeypatch.chdir(tmp_path)
    _write_fit_data(tmp_path, "mlip_committee", seed=803)

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    job_dict = {
        "name": "mlip_committee",
        "size_of_committee": 1,
        "max_num_epochs": 10,
        "mace_fit_kwargs": {},
    }
    plot_training_curves("demo", job_dict, 803, plots_dir)

    assert (plots_dir / "training_mae_demo.png").exists()
    assert (plots_dir / "training_loss_demo.png").exists()


@pytest.mark.unit
def test_plot_training_curves_no_data_no_output(tmp_path, monkeypatch):
    from alomancy.analysis.mlip_plots import plot_training_curves

    monkeypatch.chdir(tmp_path)
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    job_dict = {
        "name": "mlip_committee",
        "size_of_committee": 2,
        "max_num_epochs": 80,
        "mace_fit_kwargs": {},
    }
    plot_training_curves("empty_loop", job_dict, 803, plots_dir)
    assert not list(plots_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# _draw_parity_figure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_draw_parity_figure_all_missing_creates_file(tmp_path):
    from alomancy.analysis.mlip_plots import _draw_parity_figure

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    _draw_parity_figure(
        results_per_fit=[None, None],
        n_fits=2,
        name="mlip_committee",
        seed=803,
        set_label="Test",
        base_name="test_loop",
        plots_dir=plots_dir,
        file_suffix="test",
    )
    assert (plots_dir / "fit_parity_test_test_loop.png").exists()


@pytest.mark.unit
def test_draw_parity_figure_with_data_creates_file(tmp_path):
    from alomancy.analysis.mlip_plots import _draw_parity_figure

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    rng = np.random.default_rng(0)
    result = (
        rng.random(20),  # e_dft
        rng.random(20),  # e_pred
        rng.random(60),  # f_dft
        rng.random(60),  # f_pred
    )
    _draw_parity_figure(
        results_per_fit=[result],
        n_fits=1,
        name="mlip_committee",
        seed=803,
        set_label="Train",
        base_name="test_loop",
        plots_dir=plots_dir,
        file_suffix="train",
    )
    assert (plots_dir / "fit_parity_train_test_loop.png").exists()
