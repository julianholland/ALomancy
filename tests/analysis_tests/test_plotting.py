"""Tests for plotting module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_df():
    return pd.DataFrame({"mae_e": [0.1, 0.05, 0.03], "mae_f": [0.2, 0.1, 0.06]})


@pytest.mark.unit
class TestPlotConstructor:
    def test_attributes_set(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        df = _make_df()
        p = Plot(
            data=df, title="My Test", xlabel="X", ylabel="Y", directory=str(tmp_path)
        )
        assert p.title == "My Test"
        assert p.xlabel == "X"
        assert p.ylabel == "Y"
        assert p.directory == str(tmp_path)

    def test_filename_contains_title(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        p = Plot(
            data=_make_df(),
            title="AL Loop MAE",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        assert "al_loop_mae" in p.filename

    def test_filename_ends_with_png(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        p = Plot(
            data=_make_df(),
            title="Test",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        assert p.filename.endswith("_plot.png")


@pytest.mark.unit
class TestPlotCreate:
    @patch("alomancy.analysis.plotting.plt")
    def test_create_with_dataframe(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot

        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        p = Plot(
            data=_make_df(),
            title="Test",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.create()
        mock_plt.subplots.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_create_with_dict(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot

        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        data = {"series_a": [1.0, 0.5, 0.2], "series_b": [2.0, 1.0, 0.5]}
        p = Plot(
            data=data,
            title="Dict Plot",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.create()
        mock_plt.subplots.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_create_with_list(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot

        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        data = [1.0, 0.8, 0.5]
        p = Plot(
            data=data,
            title="List Plot",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.create()
        mock_plt.subplots.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_create_sets_labels_and_grid(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot

        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        p = Plot(
            data=_make_df(),
            title="Test",
            xlabel="epoch",
            ylabel="MAE",
            directory=str(tmp_path),
        )
        p.create()
        mock_ax.set_xlabel.assert_called_with("epoch")
        mock_ax.set_ylabel.assert_called_with("MAE")
        mock_ax.set_title.assert_called_with("Test")
        mock_ax.grid.assert_called_with(True)


@pytest.mark.unit
class TestPlotFindData:
    def test_finds_column_from_dataframe(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        df = _make_df()
        p = Plot(data=df, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        result = p.find_data("mae_e")
        assert list(result) == list(df["mae_e"])


@pytest.mark.unit
class TestPlotSave:
    @patch("alomancy.analysis.plotting.plt")
    def test_save_calls_savefig(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot

        p = Plot(
            data=_make_df(),
            title="Test",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.save()
        mock_plt.savefig.assert_called_once_with(p.filename)


@pytest.mark.unit
class TestPlotNoInteractive:
    def test_no_show_method(self, tmp_path):
        """Plotting must never open an interactive window -- create an image and close."""
        from alomancy.analysis.plotting import Plot

        assert not hasattr(Plot, "show")

    @patch("alomancy.analysis.plotting.plt")
    def test_save_closes_figure_after_create(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot

        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        p = Plot(
            data=_make_df(),
            title="Test",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.create()
        p.save()
        mock_plt.savefig.assert_called_once_with(p.filename)
        mock_plt.close.assert_called_once_with(mock_fig)


@pytest.mark.unit
class TestPlotClear:
    def test_clear_dataframe(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        p = Plot(
            data=_make_df(),
            title="Test",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.clear()
        assert len(p.data) == 0
        assert list(p.data.columns) == ["mae_e", "mae_f"]

    def test_clear_dict(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        p = Plot(
            data=data, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path)
        )
        p.clear()
        assert p.data == {"a": [], "b": []}

    def test_clear_list(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        p = Plot(
            data=[1, 2, 3],
            title="Test",
            xlabel="X",
            ylabel="Y",
            directory=str(tmp_path),
        )
        p.clear()
        assert p.data == []


@pytest.mark.unit
class TestPlotUpdate:
    def test_update_dataframe(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        df1 = pd.DataFrame({"a": [1.0]})
        df2 = pd.DataFrame({"a": [2.0]})
        p = Plot(
            data=df1, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path)
        )
        p.update(df2)
        assert len(p.data) == 2

    def test_update_dict(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        data = {"a": [1.0]}
        p = Plot(
            data=data, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path)
        )
        p.update({"a": [2.0, 3.0]})
        assert p.data["a"] == [1.0, 2.0, 3.0]

    def test_update_list(self, tmp_path):
        from alomancy.analysis.plotting import Plot

        p = Plot(
            data=[1.0], title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path)
        )
        p.update([2.0, 3.0])
        assert p.data == [1.0, 2.0, 3.0]


@pytest.mark.unit
class TestMaeAlLoopPlot:
    @patch("alomancy.analysis.plotting.plt")
    def test_mae_al_loop_plot_runs(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import mae_al_loop_plot

        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        df = _make_df()
        mae_al_loop_plot(
            all_avg_results=df,
            mlip_committee_job_dict={"name": "test_committee"},
            directory=tmp_path,
        )
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_uses_errorbar_when_std_columns_present(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import mae_al_loop_plot

        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        df = pd.DataFrame(
            {
                "mae_e_per_atom": [0.1, 0.05],
                "mae_f": [0.3, 0.15],
                "mae_e_per_atom_std_dev": [0.01, 0.005],
                "mae_f_std_dev": [0.03, 0.015],
            }
        )
        mae_al_loop_plot(df, {"name": "test"}, directory=tmp_path)
        assert mock_ax.errorbar.called
        assert not mock_ax.plot.called

    @patch("alomancy.analysis.plotting.plt")
    def test_falls_back_to_plot_without_std_columns(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import mae_al_loop_plot

        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        df = pd.DataFrame({"mae_e_per_atom": [0.1, 0.05], "mae_f": [0.3, 0.15]})
        mae_al_loop_plot(df, {"name": "test"}, directory=tmp_path)
        assert not mock_ax.errorbar.called
        assert mock_ax.plot.called

    @patch("alomancy.analysis.plotting.plt")
    def test_legend_labels_include_units(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import mae_al_loop_plot

        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        df = pd.DataFrame({"mae_e_per_atom": [0.1], "mae_f": [0.3]})
        mae_al_loop_plot(df, {"name": "test"}, directory=tmp_path)
        labels = [call.kwargs.get("label", "") for call in mock_ax.plot.call_args_list]
        assert any("eV/atom" in lbl for lbl in labels)
        assert any("eV/Å" in lbl for lbl in labels)
