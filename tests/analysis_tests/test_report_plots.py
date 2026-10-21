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

    percent_error(predictions, path, title="t", energy_label="formation energy")

    assert path.exists()


@pytest.mark.unit
def test_percent_error_writes_nothing_when_every_point_is_below_the_floor(tmp_path):
    path = tmp_path / "percent.png"

    percent_error(
        {"train": _split([0.0], [0.1], [[0, 0, 0]], [[1, 0, 0]])},
        path,
        title="t",
        energy_label="energy",
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
        energy_label="energy",
    )

    assert path.exists()
