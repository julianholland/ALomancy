"""Unit tests for the report's percentage-error figure and its numbers
(analysis/report/plots.py: percent_errors, percent_error_medians,
percent_error)."""

import numpy as np
import pytest

from alomancy.analysis.report.plots import (
    PERCENT_ENERGY_FLOOR,
    PERCENT_FORCE_FLOOR,
    percent_error,
    percent_error_medians,
    percent_errors,
)


def _split(e_ref, e_pred, f_ref, f_pred) -> tuple:
    return (
        np.asarray(e_ref, dtype=float),
        np.asarray(e_pred, dtype=float),
        np.asarray(f_ref, dtype=float).reshape(-1),
        np.asarray(f_pred, dtype=float).reshape(-1),
    )


@pytest.mark.unit
def test_energy_percent_is_signed_relative_to_the_dft_magnitude():
    errors = percent_errors({"train": _split([-2.0, 1.0], [-1.0, 1.1], [], [])})

    x, pct, n_below = errors["train"]["energy"]
    np.testing.assert_allclose(x, [-2.0, 1.0])
    np.testing.assert_allclose(pct, [50.0, 10.0])
    assert n_below == 0


@pytest.mark.unit
def test_force_percent_is_per_atom_vector_error_over_magnitude():
    errors = percent_errors(
        {"test": _split([], [], [[3.0, 4.0, 0.0]], [[3.0, 4.0, 1.0]])}
    )

    x, pct, n_below = errors["test"]["force"]
    np.testing.assert_allclose(x, [5.0])
    np.testing.assert_allclose(pct, [20.0])
    assert n_below == 0


@pytest.mark.unit
def test_points_below_the_floors_are_left_out_and_counted():
    small_e = PERCENT_ENERGY_FLOOR / 2
    small_f = PERCENT_FORCE_FLOOR / 2
    errors = percent_errors(
        {
            "train": _split(
                [small_e, -1.0],
                [small_e * 3, -1.0],
                [[small_f, 0, 0], [1.0, 0, 0]],
                [[1.0, 0, 0], [1.0, 0, 0]],
            )
        }
    )

    _x, e_pct, e_below = errors["train"]["energy"]
    _x, f_pct, f_below = errors["train"]["force"]
    np.testing.assert_allclose(e_pct, [0.0])
    np.testing.assert_allclose(f_pct, [0.0])
    assert (e_below, f_below) == (1, 1)


@pytest.mark.unit
def test_forces_that_are_not_whole_atoms_are_skipped():
    errors = percent_errors({"train": _split([-1.0], [-1.0], [1.0, 2.0], [1.0, 2.0])})

    assert "force" not in errors["train"]
    assert "energy" in errors["train"]


@pytest.mark.unit
def test_medians_cover_only_quantities_with_points():
    medians = percent_error_medians(
        {
            "train": _split([-1.0, -2.0, -4.0], [-1.1, -1.0, -4.0], [], []),
            "test": _split([0.0], [1.0], [], []),
        }
    )

    assert medians["train"] == {"energy": pytest.approx(10.0)}
    assert medians["test"] == {}


@pytest.mark.unit
def test_percent_error_writes_a_figure(tmp_path):
    path = tmp_path / "percent.png"
    predictions = {
        "train": _split([-1.0, -2.0], [-1.1, -2.1], [[1, 0, 0]], [[1.2, 0, 0]]),
        "test": _split([-1.5], [-1.4], [[0, 2, 0]], [[0, 2.5, 0]]),
    }

    percent_error(predictions, path, title="t")

    assert path.exists()


@pytest.mark.unit
def test_percent_error_writes_nothing_when_every_point_is_below_the_floor(tmp_path):
    path = tmp_path / "percent.png"

    percent_error(
        {"train": _split([0.0], [0.1], [[0, 0, 0]], [[1, 0, 0]])},
        path,
        title="t",
    )

    assert not path.exists()


@pytest.mark.unit
def test_percent_error_with_energies_only(tmp_path):
    """No atom above the force floor: the energy panel is still drawn."""
    path = tmp_path / "percent.png"

    percent_error(
        {"train": _split([-1.0], [-1.1], [[0, 0, 0]], [[0.01, 0, 0]])},
        path,
        title="t",
    )

    assert path.exists()


@pytest.mark.unit
def test_total_energies_give_no_energy_percentage(tmp_path):
    """Without formation energies the energy error is left out (relative to
    a total energy every error looks like 0%); forces still count."""
    path = tmp_path / "percent.png"
    predictions = {"train": _split([-150.0], [-150.1], [[0, 2, 0]], [[0, 2.5, 0]])}

    medians = percent_error(predictions, path, title="t", formation_energy=False)

    assert path.exists()
    assert medians == {"train": {"force": pytest.approx(25.0)}}
    assert "energy" not in percent_errors(predictions, formation_energy=False)["train"]


@pytest.mark.unit
def test_committee_cut_is_the_lowest_selected_candidate(tmp_path, monkeypatch):
    """Redundant picks skipped before DFT leave gaps in the ranking: the
    reported cut is the lowest std dev actually selected, not the N-th."""
    import polars as pl

    from alomancy.analysis.report.sections import committee_section

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "results/al_loop_0/structure_generation/std_dev_forces.csv"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "structure_index": [0, 1, 2, 3],
            "max_std_dev": [5.0, 4.0, 3.0, 2.0],
            "selected": [True, False, True, False],
        }
    ).write_csv(path)

    section = committee_section(
        {"dft": {"submitted": 2}}, base_name="al_loop_0", plots_dir=None
    )

    assert "2 selected (max force std dev ≥ 3" in section.lines[0]


def _md_axes(monkeypatch, tmp_path, temperatures, target=300.0) -> object:
    """Draw md_runs and return its axes (captured before the figure is saved
    and closed)."""
    from alomancy.analysis.report import plots

    captured = {}

    def keep(fig, path):
        captured["ax"] = fig.axes[0]

    monkeypatch.setattr(plots, "_save", keep)
    plots.md_runs(
        [np.asarray(t, dtype=float) for t in temperatures],
        [len(t) for t in temperatures],
        tmp_path / "md.png",
        run_labels=[f"run {i}" for i in range(len(temperatures))],
        title="t",
        target_temperature=target,
        requested_steps=3,
    )
    return captured["ax"]


@pytest.mark.unit
def test_md_temperature_axis_stops_at_three_times_the_target(monkeypatch, tmp_path):
    ax = _md_axes(monkeypatch, tmp_path, [[290, 310, 5000], [300, 305, 295]])
    assert ax.get_ylim() == (0.0, 900.0)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "run 0 (peaks at 5,000 K)" in labels
    assert "run 1" in labels


@pytest.mark.unit
def test_md_temperature_axis_is_free_without_a_target(monkeypatch, tmp_path):
    ax = _md_axes(monkeypatch, tmp_path, [[0, 10, 5000]], target=0.0)
    assert ax.get_ylim()[1] > 5000


@pytest.mark.unit
def test_md_temperature_cap_covers_a_hotter_equilibration(monkeypatch, tmp_path):
    """Melt-quench: equilibration at 2000 K, production at 300 K. The cap
    follows the hotter phase, so healthy runs aren't clipped or flagged."""
    from alomancy.analysis.report import plots

    captured = {}
    monkeypatch.setattr(
        plots, "_save", lambda fig, path: captured.update(ax=fig.axes[0])
    )
    plots.md_runs(
        [np.array([2000.0, 2050.0, 300.0, 310.0])],
        [1],
        tmp_path / "md.png",
        run_labels=["run 0"],
        title="t",
        target_temperature=300.0,
        requested_steps=1,
        equilibration_steps=2,
        equilibration_temperature=2000.0,
    )
    ax = captured["ax"]
    assert ax.get_ylim() == (0.0, 6000.0)
    assert ax.get_legend().get_texts()[0].get_text() == "run 0"
