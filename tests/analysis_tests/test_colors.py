"""Unit tests for the ALomancy colour scheme module."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest


@pytest.mark.unit
def test_palette_starts_with_brand_purple():
    from alomancy.analysis.colors import ALOMANCY_PURPLE, PALETTE

    assert PALETTE[0] == ALOMANCY_PURPLE


@pytest.mark.unit
def test_palette_has_ten_entries():
    from alomancy.analysis.colors import PALETTE

    assert len(PALETTE) == 10


@pytest.mark.unit
def test_setup_alomancy_style_sets_grid_color():
    from alomancy.analysis.colors import GRID_COLOR, setup_alomancy_style

    setup_alomancy_style()
    assert plt.rcParams["grid.color"] == GRID_COLOR


@pytest.mark.unit
def test_setup_alomancy_style_removes_top_spine():
    from alomancy.analysis.colors import setup_alomancy_style

    setup_alomancy_style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


@pytest.mark.unit
def test_setup_alomancy_style_sets_prop_cycle():
    from alomancy.analysis.colors import PALETTE, setup_alomancy_style

    setup_alomancy_style()
    cycle_colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    assert cycle_colors == PALETTE


@pytest.mark.unit
def test_add_logo_watermark_silent_when_logo_missing(tmp_path):
    from alomancy.analysis import colors

    fig, _ = plt.subplots()
    with patch.object(colors, "_LOGO_PATH", tmp_path / "no_logo.png"):
        colors.add_logo_watermark(fig)  # must not raise
    plt.close(fig)


@pytest.mark.unit
def test_add_logo_watermark_adds_inset_axes():
    from alomancy.analysis.colors import add_logo_watermark

    fig, _ = plt.subplots()
    n_before = len(fig.axes)
    add_logo_watermark(fig)
    assert len(fig.axes) == n_before + 1
    plt.close(fig)


@pytest.mark.unit
def test_add_logo_watermark_inset_has_no_axis_lines():
    from alomancy.analysis.colors import add_logo_watermark

    fig, _ = plt.subplots()
    add_logo_watermark(fig)
    wm_ax = fig.axes[-1]
    assert not wm_ax.axison
    plt.close(fig)


@pytest.mark.unit
def test_add_logo_watermark_high_zorder():
    from alomancy.analysis.colors import add_logo_watermark

    fig, main_ax = plt.subplots()
    add_logo_watermark(fig)
    wm_ax = fig.axes[-1]
    assert wm_ax.get_zorder() > main_ax.get_zorder()
    plt.close(fig)
