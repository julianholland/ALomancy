"""Tests for plotting module."""

from unittest.mock import patch

import pandas as pd
import pytest


def _make_df():
    return pd.DataFrame({"mae_e": [0.1, 0.05, 0.03], "mae_f": [0.2, 0.1, 0.06]})


@pytest.mark.unit
class TestPlotConstructor:
    def test_attributes_set(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        df = _make_df()
        p = Plot(data=df, title="My Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        assert p.title == "My Test"
        assert p.xlabel == "X"
        assert p.ylabel == "Y"
        assert p.directory == str(tmp_path)

    def test_filename_contains_title(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=_make_df(), title="AL Loop MAE", xlabel="X", ylabel="Y", directory=str(tmp_path))
        assert "al_loop_mae" in p.filename

    def test_filename_ends_with_png(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=_make_df(), title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        assert p.filename.endswith("_plot.png")


@pytest.mark.unit
class TestPlotCreate:
    @patch("alomancy.analysis.plotting.plt")
    def test_create_with_dataframe(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=_make_df(), title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.create()
        mock_plt.figure.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_create_with_dict(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot
        data = {"series_a": [1.0, 0.5, 0.2], "series_b": [2.0, 1.0, 0.5]}
        p = Plot(data=data, title="Dict Plot", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.create()
        mock_plt.figure.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_create_with_list(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot
        data = [1.0, 0.8, 0.5]
        p = Plot(data=data, title="List Plot", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.create()
        mock_plt.figure.assert_called_once()

    @patch("alomancy.analysis.plotting.plt")
    def test_create_sets_labels_and_grid(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=_make_df(), title="Test", xlabel="epoch", ylabel="MAE", directory=str(tmp_path))
        p.create()
        mock_plt.xlabel.assert_called_with("epoch")
        mock_plt.ylabel.assert_called_with("MAE")
        mock_plt.title.assert_called_with("Test")
        mock_plt.grid.assert_called_with(True)


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
        p = Plot(data=_make_df(), title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.save()
        mock_plt.savefig.assert_called_once_with(p.filename)


@pytest.mark.unit
class TestPlotShow:
    @patch("alomancy.analysis.plotting.plt")
    def test_show_calls_plt_show(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=_make_df(), title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.show()
        mock_plt.show.assert_called_once()


@pytest.mark.unit
class TestPlotClear:
    def test_clear_dataframe(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=_make_df(), title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.clear()
        assert len(p.data) == 0
        assert list(p.data.columns) == ["mae_e", "mae_f"]

    def test_clear_dict(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        p = Plot(data=data, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.clear()
        assert p.data == {"a": [], "b": []}

    def test_clear_list(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=[1, 2, 3], title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.clear()
        assert p.data == []


@pytest.mark.unit
class TestPlotUpdate:
    def test_update_dataframe(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        df1 = pd.DataFrame({"a": [1.0]})
        df2 = pd.DataFrame({"a": [2.0]})
        p = Plot(data=df1, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.update(df2)
        assert len(p.data) == 2

    def test_update_dict(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        data = {"a": [1.0]}
        p = Plot(data=data, title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.update({"a": [2.0, 3.0]})
        assert p.data["a"] == [1.0, 2.0, 3.0]

    def test_update_list(self, tmp_path):
        from alomancy.analysis.plotting import Plot
        p = Plot(data=[1.0], title="Test", xlabel="X", ylabel="Y", directory=str(tmp_path))
        p.update([2.0, 3.0])
        assert p.data == [1.0, 2.0, 3.0]


@pytest.mark.unit
class TestMaeAlLoopPlot:
    @patch("alomancy.analysis.plotting.plt")
    def test_mae_al_loop_plot_runs(self, mock_plt, tmp_path):
        from alomancy.analysis.plotting import mae_al_loop_plot
        df = _make_df()
        mae_al_loop_plot(
            all_avg_results=df,
            mlip_committee_job_dict={"name": "test_committee"},
            directory=tmp_path,
        )
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once()
