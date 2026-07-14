"""Unit tests for mlip_plots helpers."""

import json
from pathlib import Path

import pandas as pd
import pytest

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
