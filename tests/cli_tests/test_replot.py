"""Unit tests for alomancy.cli.replot."""

from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_results_tree(
    tmp_path: Path,
    *,
    committee_name: str = "mlip_committee",
    n_fits: int = 2,
    seed: int = 803,
    n_loops: int = 1,
    include_log: bool = True,
) -> Path:
    """Build a minimal fake results/ directory tree."""
    results = tmp_path / "results"
    results.mkdir()

    for loop_i in range(n_loops):
        loop_dir = results / f"al_loop_{loop_i}"
        committee = loop_dir / committee_name
        for fit_i in range(n_fits):
            fit_results = committee / f"fit_{fit_i}" / "results"
            fit_results.mkdir(parents=True)
            (fit_results / f"{committee_name}_run-{seed + fit_i}_train.txt").write_text(
                '{"mae_f": 0.1, "mae_e_per_atom": 0.01}\n'
            )
        (loop_dir / "train_set.xyz").write_text("")
        (loop_dir / "test_set.xyz").write_text("")

    if include_log:
        (results / "alomancy.log").write_text("no timing lines\n")

    return results


# ---------------------------------------------------------------------------
# detect_committee_info
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_name(tmp_path):
    from alomancy.cli.replot import detect_committee_info

    results = _make_results_tree(tmp_path, committee_name="mlip_committee")
    name, _, _ = detect_committee_info(results)
    assert name == "mlip_committee"


@pytest.mark.unit
def test_detect_n_fits(tmp_path):
    from alomancy.cli.replot import detect_committee_info

    results = _make_results_tree(tmp_path, n_fits=3)
    _, n_fits, _ = detect_committee_info(results)
    assert n_fits == 3


@pytest.mark.unit
def test_detect_seed(tmp_path):
    from alomancy.cli.replot import detect_committee_info

    results = _make_results_tree(tmp_path, seed=900)
    _, _, seed = detect_committee_info(results)
    assert seed == 900


@pytest.mark.unit
def test_detect_seed_fallback(tmp_path):
    from alomancy.cli.replot import detect_committee_info

    results = _make_results_tree(tmp_path)
    # Rename the train.txt so it has no _run-N_ pattern
    for f in results.rglob("*_run-*_train.txt"):
        f.rename(f.parent / "no_seed_train.txt")

    _, _, seed = detect_committee_info(results)
    assert seed == 803  # project default fallback


@pytest.mark.unit
def test_detect_raises_if_no_committee(tmp_path):
    from alomancy.cli.replot import detect_committee_info

    results = tmp_path / "results"
    results.mkdir()
    with pytest.raises(RuntimeError, match="Could not detect mlip_committee"):
        detect_committee_info(results)


# ---------------------------------------------------------------------------
# replot_results
# ---------------------------------------------------------------------------


def _mock_targets():
    return [
        "alomancy.cli.replot.plot_training_curves",
        "alomancy.cli.replot.plot_dft_vs_model",
        "alomancy.cli.replot.metrics_by_loop",
        "alomancy.cli.replot.mae_al_loop_plot",
        "alomancy.cli.replot.timing_plots",
    ]


@pytest.mark.unit
def test_replot_calls_plot_functions(tmp_path):
    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path, n_loops=2)

    import polars as pl

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves") as m_train,
        mock.patch("alomancy.cli.replot.plot_dft_vs_model") as m_parity,
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot") as m_mae,
        mock.patch("alomancy.cli.replot.timing_plots") as m_timing,
        mock.patch("os.chdir"),
    ):
        replot_results(results)

    assert m_train.call_count == 2  # once per loop
    assert m_parity.call_count == 2
    assert m_mae.call_count == 1
    assert m_timing.call_count == 1


@pytest.mark.unit
def test_replot_passes_db_and_loop_idx_when_global_database_exists(tmp_path):
    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path, n_loops=2)
    (results / "global_database").mkdir()

    import polars as pl

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves"),
        mock.patch("alomancy.cli.replot.plot_dft_vs_model") as m_parity,
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot"),
        mock.patch("alomancy.cli.replot.timing_plots"),
        mock.patch("os.chdir"),
        mock.patch("alomancy.cli.replot.GlobalDatabase") as m_db_cls,
    ):
        db_instance = m_db_cls.return_value
        replot_results(results)

    m_db_cls.assert_called_once_with(str(results / "global_database"))
    assert m_parity.call_count == 2
    loop_idxs = {call.kwargs["loop_idx"] for call in m_parity.call_args_list}
    assert loop_idxs == {0, 1}
    for call in m_parity.call_args_list:
        assert call.kwargs["db"] is db_instance


@pytest.mark.unit
def test_replot_no_global_database_dir_falls_back(tmp_path):
    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path)
    # Deliberately do not create results/global_database.

    import polars as pl

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves"),
        mock.patch("alomancy.cli.replot.plot_dft_vs_model") as m_parity,
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot"),
        mock.patch("alomancy.cli.replot.timing_plots"),
        mock.patch("os.chdir"),
        mock.patch("alomancy.cli.replot.GlobalDatabase") as m_db_cls,
    ):
        replot_results(results)

    m_db_cls.assert_not_called()
    for call in m_parity.call_args_list:
        assert call.kwargs["db"] is None
        assert call.kwargs["loop_idx"] == 0


@pytest.mark.unit
def test_replot_no_parity(tmp_path):
    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path)

    import polars as pl

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves"),
        mock.patch("alomancy.cli.replot.plot_dft_vs_model") as m_parity,
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot"),
        mock.patch("alomancy.cli.replot.timing_plots"),
        mock.patch("os.chdir"),
    ):
        replot_results(results, no_parity=True)

    m_parity.assert_not_called()


@pytest.mark.unit
def test_replot_skips_loop_without_train_txt(tmp_path):
    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path, n_loops=2)
    # Remove train.txt from loop_1 → should only plot loop_0
    for f in (results / "al_loop_1").rglob("*_train.txt"):
        f.unlink()

    import polars as pl

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves") as m_train,
        mock.patch("alomancy.cli.replot.plot_dft_vs_model"),
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot"),
        mock.patch("alomancy.cli.replot.timing_plots"),
        mock.patch("os.chdir"),
    ):
        replot_results(results)

    assert m_train.call_count == 1
    assert m_train.call_args[0][0] == "al_loop_0"


@pytest.mark.unit
def test_replot_skips_timing_when_no_log(tmp_path):
    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path, include_log=False)

    import polars as pl

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves"),
        mock.patch("alomancy.cli.replot.plot_dft_vs_model"),
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot"),
        mock.patch("alomancy.cli.replot.timing_plots") as m_timing,
        mock.patch("os.chdir"),
    ):
        replot_results(results)

    m_timing.assert_not_called()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_entrypoint_replot(tmp_path):
    from alomancy.cli.main import main

    results = _make_results_tree(tmp_path)

    with (
        mock.patch(
            "sys.argv",
            [
                "alomancy",
                "results",
                "--replot",
                "--no-parity",
                "--results-dir",
                str(results),
            ],
        ),
        mock.patch("alomancy.cli.replot.replot_results") as m_replot,
    ):
        main()

    m_replot.assert_called_once_with(results.resolve(), no_parity=True)


@pytest.mark.unit
def test_replot_orders_loops_numerically(tmp_path):
    """al_loop_10 is the last loop, not al_loop_9 (name order): per-loop
    plots run in loop order and the cross-loop MAE plot lands in the last
    loop's directory."""
    import polars as pl

    from alomancy.cli.replot import replot_results

    results = _make_results_tree(tmp_path, n_loops=11)

    with (
        mock.patch("alomancy.cli.replot.plot_training_curves") as m_train,
        mock.patch("alomancy.cli.replot.plot_dft_vs_model"),
        mock.patch(
            "alomancy.cli.replot.metrics_by_loop",
            return_value=pl.DataFrame([{"al_loop": 0, "mae_f": 0.1}]),
        ),
        mock.patch("alomancy.cli.replot.mae_al_loop_plot") as m_mae,
        mock.patch("alomancy.cli.replot.timing_plots"),
        mock.patch("os.chdir"),
    ):
        replot_results(results)

    plotted = [c.args[0] for c in m_train.call_args_list]
    assert plotted == [f"al_loop_{i}" for i in range(11)]
    assert m_mae.call_args.kwargs["directory"].name == "al_loop_10"
