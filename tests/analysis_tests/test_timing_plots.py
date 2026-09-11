"""Unit tests for timing_plots helpers."""

import math
from pathlib import Path
from unittest import mock

import pytest


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    log = tmp_path / "alomancy.log"
    log.write_text("\n".join(lines) + "\n")
    return log


def _loop0_lines(
    *,
    start="2026-07-23 21:19:32",
    n_train=15207,
    gen_start="2026-07-24 04:16:27",
    gen_end="2026-07-24 05:46:49",
    dft_end="2026-07-24 05:57:39",
    loop_end="2026-07-24 05:58:44",
) -> list[str]:
    return [
        f"{start} [DEBUG   ] alomancy.core.base_active_learning: Starting AL loop 0",
        f"{start} [DEBUG   ] alomancy.core.base_active_learning:   Training set size: {n_train}",
        f"{gen_start} [INFO    ] alomancy.core.standard_active_learning: 20 structures selected for structure generation step.",
        f"{gen_end} [INFO    ] alomancy.structure_generation.find_high_sd_structures: Selected 200 structures for DFT calculations based on force std dev.",
        f"{dft_end} [INFO    ] alomancy.core.base_active_learning: High-accuracy evaluation completed for 195 structures.",
        f"{loop_end} [DEBUG   ] alomancy.core.base_active_learning: Completed AL loop 0, retraining with 15207 structures.",
    ]


@pytest.mark.unit
def test_parse_single_loop(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    log = _write_log(tmp_path, _loop0_lines())
    df = parse_timing_log(log)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["loop"] == 0
    assert row["n_train"] == 15207
    # total: 2026-07-23 21:19:32 → 2026-07-24 05:58:44 = 8h 39m 12s = 31152 s
    assert abs(row["total_s"] - 31152) < 2
    # training+plots: 21:19:32 → 04:16:27 = 6h 56m 55s = 25015 s
    assert abs(row["training_plots_s"] - 25015) < 2
    # gen: 04:16:27 → 05:46:49 = 1h 30m 22s = 5422 s
    assert abs(row["generate_structures_s"] - 5422) < 2
    # dft: 05:46:49 → 05:57:39 = 650 s
    assert abs(row["high_accuracy_evaluation_s"] - 650) < 2
    # postprocess: 05:57:39 → 05:58:44 = 65 s
    assert abs(row["postprocess_s"] - 65) < 2


@pytest.mark.unit
def test_parse_multiple_loops(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    loop1 = [
        "2026-07-24 05:59:06 [DEBUG   ] alomancy.core.base_active_learning: Starting AL loop 1",
        "2026-07-24 05:59:06 [DEBUG   ] alomancy.core.base_active_learning:   Training set size: 15382",
        "2026-07-24 13:38:45 [INFO    ] alomancy.core.standard_active_learning: 20 structures selected for structure generation step.",
        "2026-07-24 14:30:00 [INFO    ] alomancy.structure_generation.find_high_sd_structures: Selected 200 structures for DFT calculations based on force std dev.",
        "2026-07-24 14:45:00 [INFO    ] alomancy.core.base_active_learning: High-accuracy evaluation completed for 190 structures.",
        "2026-07-24 14:46:00 [DEBUG   ] alomancy.core.base_active_learning: Completed AL loop 1, retraining with 15382 structures.",
    ]
    log = _write_log(tmp_path, _loop0_lines() + loop1)
    df = parse_timing_log(log)

    assert len(df) == 2
    assert list(df["loop"]) == [0, 1]
    assert df.iloc[1]["n_train"] == 15382


@pytest.mark.unit
def test_last_write_wins_on_restart(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    # First failed attempt at loop 0 (incomplete — no gen_start/loop_end)
    first_attempt = [
        "2026-07-23 20:00:00 [DEBUG   ] alomancy.core.base_active_learning: Starting AL loop 0",
        "2026-07-23 20:00:00 [DEBUG   ] alomancy.core.base_active_learning:   Training set size: 9999",
    ]
    log = _write_log(tmp_path, first_attempt + _loop0_lines())
    df = parse_timing_log(log)

    assert len(df) == 1
    # Second occurrence wins → n_train from second run
    assert df.iloc[0]["n_train"] == 15207


@pytest.mark.unit
def test_missing_phase_gives_nan(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    lines = [
        "2026-07-23 21:19:32 [DEBUG   ] alomancy.core.base_active_learning: Starting AL loop 0",
        "2026-07-23 21:19:32 [DEBUG   ] alomancy.core.base_active_learning:   Training set size: 15207",
        # no gen_start, gen_end, dft_end
        "2026-07-24 05:58:44 [DEBUG   ] alomancy.core.base_active_learning: Completed AL loop 0, retraining with 15207 structures.",
    ]
    log = _write_log(tmp_path, lines)
    df = parse_timing_log(log)

    assert len(df) == 1
    assert math.isnan(df.iloc[0]["generate_structures_s"])
    assert math.isnan(df.iloc[0]["high_accuracy_evaluation_s"])
    assert not math.isnan(df.iloc[0]["total_s"])


@pytest.mark.unit
def test_queue_time_parsed(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    # Queue messages for training phase (before gen_start)
    lines = _loop0_lines()
    # Insert queue messages before gen_start timestamp
    queue_lines = [
        "2026-07-24 01:40:55 [INFO    ] alomancy.remote_submission.executor: Job 1 completed successfully.",
        "2026-07-24 01:40:55 [INFO    ] alomancy.remote_submission.executor: Job 1 queue_time=1200.0 s.",
        "2026-07-24 03:17:11 [INFO    ] alomancy.remote_submission.executor: Job 2 completed successfully.",
        "2026-07-24 03:17:11 [INFO    ] alomancy.remote_submission.executor: Job 2 queue_time=1800.0 s.",
    ]
    all_lines = [lines[0], lines[1], *queue_lines, *lines[2:]]
    log = _write_log(tmp_path, all_lines)
    df = parse_timing_log(log)

    assert len(df) == 1
    q = df.iloc[0]["training_plots_queue_s"]
    assert abs(q - 1500.0) < 1e-6  # mean of 1200 and 1800


@pytest.mark.unit
def test_queue_time_nan_when_absent(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    log = _write_log(tmp_path, _loop0_lines())
    df = parse_timing_log(log)

    assert math.isnan(df.iloc[0]["training_plots_queue_s"])
    assert math.isnan(df.iloc[0]["generate_structures_queue_s"])
    assert math.isnan(df.iloc[0]["high_accuracy_evaluation_queue_s"])


@pytest.mark.unit
def test_empty_log_returns_empty_df(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    log = _write_log(tmp_path, ["no timing lines here"])
    df = parse_timing_log(log)

    assert df.empty


@pytest.mark.unit
def test_missing_file_returns_empty_df(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    df = parse_timing_log(tmp_path / "nonexistent.log")

    assert df.empty


@pytest.mark.unit
def test_timing_plots_saves_one_combined_file(tmp_path):
    """timing_plots now produces a single combined figure, not two."""
    from alomancy.analysis.timing_plots import timing_plots

    log = _write_log(tmp_path, _loop0_lines())
    plots_dir = tmp_path / "plots"

    timing_plots(log, plots_dir)

    pngs = list(plots_dir.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].name == "timing_combined.png"


@pytest.mark.unit
def test_timing_plots_no_op_on_empty_df(tmp_path):
    from alomancy.analysis.timing_plots import timing_plots

    log = _write_log(tmp_path, ["no timing lines"])
    plots_dir = tmp_path / "plots"

    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("matplotlib.figure.Figure.savefig") as mock_fig_save,
    ):
        timing_plots(log, plots_dir)
        assert mock_save.call_count == 0
        assert mock_fig_save.call_count == 0


@pytest.mark.unit
def test_timing_plots_legend_has_both_phase_and_training_size_entries(tmp_path):
    """twinx() legends must be combined manually or entries get silently
    dropped -- assert both the phase-timing series and the training-set-size
    line actually show up in the one legend."""
    import matplotlib.pyplot as plt

    from alomancy.analysis.timing_plots import timing_plots

    log = _write_log(tmp_path, _loop0_lines())
    plots_dir = tmp_path / "plots"

    captured = {}
    real_close = plt.close

    def fake_close(fig):
        # Two axes: primary (bars) and twinx (line) -- legend lives on ax.
        captured["legend_labels"] = [
            t.get_text() for t in fig.axes[0].get_legend().texts
        ]
        real_close(fig)

    with mock.patch("matplotlib.pyplot.close", side_effect=fake_close):
        timing_plots(log, plots_dir)

    labels = captured["legend_labels"]
    assert "Training structures" in labels
    assert any(
        label in labels
        for label in ("Training + plots", "Structure gen", "DFT", "Post-process")
    )


def _n_loop_lines(n: int) -> list[str]:
    """Build log lines for n consecutive, independently-numbered AL loops."""
    lines: list[str] = []
    for i in range(n):
        day = 23 + i
        lines.extend(
            [
                f"2026-07-{day:02d} 00:00:00 [DEBUG   ] alomancy.core.base_active_learning: Starting AL loop {i}",
                f"2026-07-{day:02d} 00:00:00 [DEBUG   ] alomancy.core.base_active_learning:   Training set size: {15000 + i}",
                f"2026-07-{day:02d} 01:00:00 [INFO    ] alomancy.core.standard_active_learning: 20 structures selected for structure generation step.",
                f"2026-07-{day:02d} 02:00:00 [INFO    ] alomancy.structure_generation.find_high_sd_structures: Selected 200 structures for DFT calculations based on force std dev.",
                f"2026-07-{day:02d} 03:00:00 [INFO    ] alomancy.core.base_active_learning: High-accuracy evaluation completed for 195 structures.",
                f"2026-07-{day:02d} 04:00:00 [DEBUG   ] alomancy.core.base_active_learning: Completed AL loop {i}, retraining with {15000 + i} structures.",
            ]
        )
    return lines


@pytest.mark.unit
def test_timing_plots_fixed_width_regardless_of_loop_count(tmp_path):
    """Figure width must be constant whether the run has 2 loops or 5 --
    the old per-plot figsize scaled with len(loops), growing unbounded."""
    import matplotlib.pyplot as plt

    from alomancy.analysis.timing_plots import timing_plots

    widths: dict[int, float] = {}
    for n_loops in (2, 5):
        run_dir = tmp_path / f"run_{n_loops}"
        run_dir.mkdir()
        log = _write_log(run_dir, _n_loop_lines(n_loops))
        plots_dir = run_dir / "plots"

        captured = {}
        real_close = plt.close

        def fake_close(fig, _captured=captured, _real_close=real_close):
            _captured["size"] = fig.get_size_inches()
            _real_close(fig)

        with mock.patch("matplotlib.pyplot.close", side_effect=fake_close):
            timing_plots(log, plots_dir)

        widths[n_loops] = captured["size"][0]

    assert widths[2] == widths[5]
