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
    row = df.row(0, named=True)
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
    assert df["n_train"][1] == 15382


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
    assert df["n_train"][0] == 15207


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
    assert math.isnan(df["generate_structures_s"][0])
    assert math.isnan(df["high_accuracy_evaluation_s"][0])
    assert not math.isnan(df["total_s"][0])


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
    q = df["training_plots_queue_s"][0]
    assert abs(q - 1500.0) < 1e-6  # mean of 1200 and 1800


@pytest.mark.unit
def test_queue_time_nan_when_absent(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    log = _write_log(tmp_path, _loop0_lines())
    df = parse_timing_log(log)

    assert math.isnan(df["training_plots_queue_s"][0])
    assert math.isnan(df["generate_structures_queue_s"][0])
    assert math.isnan(df["high_accuracy_evaluation_queue_s"][0])


@pytest.mark.unit
def test_empty_log_returns_empty_df(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    log = _write_log(tmp_path, ["no timing lines here"])
    df = parse_timing_log(log)

    assert df.is_empty()


@pytest.mark.unit
def test_missing_file_returns_empty_df(tmp_path):
    from alomancy.analysis.timing_plots import parse_timing_log

    df = parse_timing_log(tmp_path / "nonexistent.log")

    assert df.is_empty()


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


@pytest.mark.unit
def test_timing_plots_x_ticks_are_plain_loop_numbers(tmp_path):
    """Tick labels are bare integers (no "Loop N"), thinned out on long runs."""
    from datetime import datetime, timedelta

    import matplotlib.pyplot as plt

    from alomancy.analysis.timing_plots import timing_plots

    lines: list[str] = []
    t0 = datetime(2026, 7, 1)
    for i in range(30):
        day = t0 + timedelta(days=i)
        for hours, msg in (
            (0, f"alomancy.core.x: Starting AL loop {i}"),
            (0, f"alomancy.core.x:   Training set size: {1000 + i}"),
            (
                1,
                "alomancy.core.x: 20 structures selected for structure generation step.",
            ),
            (
                2,
                "alomancy.x: Selected 200 structures for DFT calculations based on force std dev.",
            ),
            (
                3,
                "alomancy.core.x: High-accuracy evaluation completed for 195 structures.",
            ),
            (
                4,
                f"alomancy.core.x: Completed AL loop {i}, retraining with {1000 + i} structures.",
            ),
        ):
            ts = (day + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{ts} [INFO    ] {msg}")
    log = _write_log(tmp_path, lines)
    captured = {}
    real_close = plt.close

    def fake_close(fig):
        ax = fig.axes[0]
        fig.canvas.draw()
        lo, hi = ax.get_xlim()
        captured["ticks"] = [t for t in ax.get_xticks() if lo <= t <= hi]
        captured["labels"] = [t.get_text() for t in ax.get_xticklabels()]
        real_close(fig)

    with mock.patch("matplotlib.pyplot.close", side_effect=fake_close):
        timing_plots(log, tmp_path / "plots")

    assert all(float(t).is_integer() for t in captured["ticks"])
    assert len(captured["ticks"]) <= 13
    assert not any("Loop" in label for label in captured["labels"])


@pytest.mark.unit
def test_timing_plots_removes_superseded_timing_files(tmp_path):
    from alomancy.analysis.timing_plots import timing_plots

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    for stale in ("timing_total.png", "timing_phases.png"):
        (plots_dir / stale).write_bytes(b"old")
    (plots_dir / "unrelated.png").write_bytes(b"keep")

    timing_plots(_write_log(tmp_path, _loop0_lines()), plots_dir)

    assert sorted(p.name for p in plots_dir.glob("*.png")) == [
        "timing_combined.png",
        "unrelated.png",
    ]


def _phase_marker_lines(*, with_std_dev_line: bool) -> list[str]:
    """One loop as the @phase decorator logs it (core/active_learning_
    workflow.py), for any workflow."""
    mod = "alomancy.core.active_learning_workflow"
    lines = [
        f"2026-10-04 10:00:00 [DEBUG   ] {mod}: Starting AL loop 0",
        f"2026-10-04 10:00:00 [DEBUG   ] {mod}:   Training set size: 100",
        f"2026-10-04 10:00:01 [DEBUG   ] {mod}: Phase train_mlip started for al_loop_0.",
        f"2026-10-04 11:00:00 [DEBUG   ] {mod}: Phase train_mlip marked complete for al_loop_0.",
        f"2026-10-04 11:10:00 [DEBUG   ] {mod}: Phase generate_structures started for al_loop_0.",
    ]
    if with_std_dev_line:
        lines.append(
            "2026-10-04 11:59:00 [INFO    ] alomancy.structure_generation."
            "find_high_sd_structures: Selected 5 structures for DFT calculations "
            "based on force std dev."
        )
    return [
        *lines,
        f"2026-10-04 12:00:00 [DEBUG   ] {mod}: Phase generate_structures marked complete for al_loop_0.",
        f"2026-10-04 12:00:01 [DEBUG   ] {mod}: Phase high_accuracy_eval started for al_loop_0.",
        f"2026-10-04 13:00:00 [DEBUG   ] {mod}: Phase high_accuracy_eval marked complete for al_loop_0.",
        f"2026-10-04 13:05:00 [DEBUG   ] {mod}: Completed AL loop 0, retraining with 105 structures.",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("with_std_dev_line", [False, True])
def test_parse_phase_markers(tmp_path, with_std_dev_line):
    """Phase markers give every segment, with or without the committee
    selector's own "force std dev" line (random/novelty never log it)."""
    from alomancy.analysis.timing_plots import parse_timing_log

    log = _write_log(tmp_path, _phase_marker_lines(with_std_dev_line=with_std_dev_line))
    row = parse_timing_log(log).row(0, named=True)

    assert abs(row["training_plots_s"] - 4200) < 2  # 10:00:00 -> 11:10:00
    # The first generation-end line wins: 11:59:00 or 12:00:00.
    expected_gen, expected_dft = (2940, 3660) if with_std_dev_line else (3000, 3600)
    assert abs(row["generate_structures_s"] - expected_gen) < 2
    assert abs(row["high_accuracy_evaluation_s"] - expected_dft) < 2
    assert abs(row["postprocess_s"] - 300) < 2


@pytest.mark.unit
def test_parses_the_log_the_phase_decorator_writes(tmp_path, monkeypatch):
    """Producer/parser link: the boundaries come from what @phase really
    logs, so renaming those messages fails here instead of silently
    blanking the timing plot (as happened when the old "structures selected
    for structure generation step" message disappeared)."""
    import logging

    from alomancy.analysis.timing_plots import parse_timing_log
    from alomancy.core.active_learning_workflow import (
        ActiveLearningWorkflow,
        LoopContext,
        phase,
    )
    from alomancy.utils.logging_config import setup_logging

    class _Steps(ActiveLearningWorkflow):
        NAME = "timing_test"

        def run(self) -> None:  # pragma: no cover - not used
            pass

        @phase("train_mlip")
        def train(self, ctx):
            return None

        @phase("generate_structures")
        def generate(self, ctx):
            return None

        @phase("high_accuracy_eval")
        def dft(self, ctx):
            return None

    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "results" / "alomancy.log"
    setup_logging(verbose=0, log_file=str(log_file))
    loop_logger = logging.getLogger("alomancy.core.active_learning_workflow")
    steps = object.__new__(_Steps)  # only the phase bookkeeping is needed
    steps._phases_run = {}
    ctx = LoopContext(
        loop=0,
        base_name="al_loop_0",
        workdir=tmp_path / "results" / "al_loop_0",
        train=[],
        test=[],
        train_only=False,
        plots_dir=None,
    )

    loop_logger.debug("Starting AL loop %d", 0)
    steps.train(ctx)
    steps.generate(ctx)
    steps.dft(ctx)
    loop_logger.debug("Completed AL loop %d, retraining with %d structures.", 0, 1)
    for handler in logging.getLogger("alomancy").handlers:
        handler.flush()

    row = parse_timing_log(log_file).row(0, named=True)
    for column in (
        "training_plots_s",
        "generate_structures_s",
        "high_accuracy_evaluation_s",
        "postprocess_s",
    ):
        assert math.isfinite(row[column]), column
