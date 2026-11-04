"""Unit tests for the report's EZGA figures: the stacked bar chart
(analysis/report/plots.py: stacked_counts), the EZGA section that draws it
per generation and per seed config_type (sections.py: ezga_section), and
the counts behind its bars (stats.py: _operator_counts, _ezga_duplicates)."""

import json

import pytest
from ase import Atoms
from matplotlib.colors import to_hex

from alomancy.analysis.colors import PALETTE
from alomancy.analysis.report import plots
from alomancy.analysis.report.sections import ezga_section
from alomancy.analysis.report.stats import _ezga_duplicates, _operator_counts

NAMES = ["mutation_rattle", "mutation_random_strain", "bounded_mutation_add"]


def _atoms(operation: str | None, mutations: list[int] | None = None) -> Atoms:
    atoms = Atoms("Pd")
    if operation is not None:
        atoms.info["ezga_operation"] = operation
    if mutations is not None:
        atoms.info["ezga_mutation_list"] = mutations
    return atoms


def _stacked_axes(monkeypatch, tmp_path, series, **kwargs) -> object:
    """Draw stacked_counts and return its axes (captured before the figure
    is saved and closed)."""
    captured = {}
    monkeypatch.setattr(
        plots, "_save", lambda fig, path: captured.update(ax=fig.axes[0])
    )
    groups = [str(i) for i in range(len(next(iter(series.values()))))]
    plots.stacked_counts(
        groups, series, tmp_path / "stacked.png", title="t", xlabel="x", **kwargs
    )
    return captured["ax"]


# ---------------------------------------------------------------------------
# _operator_counts / _ezga_duplicates: what goes in the bars
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mutation_indices_are_counted_by_their_configured_name():
    counts = _operator_counts(
        [_atoms("mutation", [0]), _atoms("mutation", [2]), _atoms("mutation", [0, 1])],
        NAMES,
    )
    assert counts == {
        "mutation_rattle": 2,
        "bounded_mutation_add": 1,
        "mutation_random_strain": 1,
    }


@pytest.mark.unit
def test_crossover_is_counted_apart_whatever_list_it_carries():
    """A crossover child's mutation list is its parent's: not applied."""
    assert _operator_counts([_atoms("crossover", [1])], NAMES) == {"crossover": 1}


@pytest.mark.unit
def test_structures_without_a_mutation_list_count_as_mutation():
    """Structures from before ALomancy recorded the list: no name to give."""
    counts = _operator_counts([_atoms("mutation"), _atoms("mutation", [])], NAMES)
    assert counts == {"mutation": 2}


@pytest.mark.unit
def test_an_index_past_the_configured_list_keeps_its_number():
    assert _operator_counts([_atoms("mutation", [7])], NAMES) == {"mut_7": 1}


@pytest.mark.unit
def test_structures_without_an_operation_are_not_counted():
    assert _operator_counts([_atoms(None, [0])], NAMES) == {}


@pytest.mark.unit
def test_duplicates_per_generation_from_ezga_running_totals(tmp_path):
    """EZGA's logger files hold running totals (mutation and crossover hash
    collisions, per operator): each generation gets its difference."""
    totals = {0: ([0, 0], [0]), 1: ([2, 1], [0]), 2: ([3, 4], [1])}
    for generation, (mutation, crossover) in totals.items():
        record = {
            "generation": generation,
            "variation_stats": {
                "mutation_hashcolition_counts": mutation,
                "crossover_hashcolition_counts": crossover,
            },
        }
        path = tmp_path / f"gen_{generation:06d}_full.json"
        path.write_text(json.dumps(record))

    assert _ezga_duplicates(tmp_path) == {0: 0, 1: 3, 2: 5}


@pytest.mark.unit
def test_no_logger_files_give_no_duplicates(tmp_path):
    assert _ezga_duplicates(tmp_path) == {}


# ---------------------------------------------------------------------------
# stacked_counts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stacked_counts_writes_a_figure(tmp_path):
    path = tmp_path / "stacked.png"

    plots.stacked_counts(
        ["0", "1"], {"a": [1, 2], "b": [3, 0]}, path, title="t", xlabel="x"
    )

    assert path.exists()


@pytest.mark.unit
def test_series_are_stacked_in_order_on_one_column(monkeypatch, tmp_path):
    ax = _stacked_axes(monkeypatch, tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 0]})

    # One bar container per series, in order; each starts where the
    # previous one ends.
    bottoms = [[bar.get_y() for bar in c.patches] for c in ax.containers]
    heights = [[bar.get_height() for bar in c.patches] for c in ax.containers]
    assert heights == [[1, 2], [3, 4], [5, 0]]
    assert bottoms == [[0, 0], [1, 2], [4, 6]]
    # Same x for every series: one column per group.
    xs = {round(bar.get_x(), 6) for c in ax.containers for bar in c.patches[:1]}
    assert len(xs) == 1


@pytest.mark.unit
def test_totals_are_written_above_the_columns(monkeypatch, tmp_path):
    ax = _stacked_axes(
        monkeypatch, tmp_path, {"a": [1, 2], "dup": [1, 0]}, totals=[1, 2]
    )

    assert [t.get_text() for t in ax.texts] == ["1", "2"]
    # Above the whole column, duplicates included.
    assert [t.xy[1] for t in ax.texts] == [2, 2]


@pytest.mark.unit
def test_fixed_colours_and_the_palette_in_order(monkeypatch, tmp_path):
    """A fixed colour doesn't use up a palette slot, so the same operators
    keep the same colours with or without the duplicates series."""
    ax = _stacked_axes(
        monkeypatch,
        tmp_path,
        {"a": [1], "Duplicates filtered": [1], "b": [1]},
        colors={"Duplicates filtered": "#6b7280"},
    )

    colours = [to_hex(c.patches[0].get_facecolor()) for c in ax.containers]
    assert colours == [PALETTE[0].lower(), "#6b7280", PALETTE[1].lower()]


@pytest.mark.unit
def test_white_background_no_grid_legend_inside(monkeypatch, tmp_path):
    ax = _stacked_axes(monkeypatch, tmp_path, {"a": [1, 2], "b": [3, 4]})

    assert to_hex(ax.get_facecolor()) == "#ffffff"
    assert to_hex(ax.figure.get_facecolor()) == "#ffffff"
    gridlines = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
    assert not any(line.get_visible() for line in gridlines)
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]
    # Inside the axes (the top right corner is the watermark's).
    box = legend.get_window_extent(ax.figure.canvas.get_renderer())
    axes_box = ax.get_window_extent(ax.figure.canvas.get_renderer())
    assert axes_box.contains(box.x0, box.y0)
    assert axes_box.contains(box.x1, box.y1)
    assert box.x0 < axes_box.x0 + 0.5 * axes_box.width


@pytest.mark.unit
def test_a_single_series_has_no_legend(monkeypatch, tmp_path):
    ax = _stacked_axes(monkeypatch, tmp_path, {"a": [1, 2]})
    assert ax.get_legend() is None


@pytest.mark.unit
def test_the_axis_leaves_room_for_totals_and_legend(monkeypatch, tmp_path):
    ax = _stacked_axes(monkeypatch, tmp_path, {"a": [10, 20], "b": [5, 20]})
    assert ax.get_ylim()[1] == pytest.approx(1.35 * 40)


# ---------------------------------------------------------------------------
# ezga_section: the two figures of a loop
# ---------------------------------------------------------------------------


def _ezga_stats(with_duplicates: bool = True) -> dict:
    generations = {
        "0": {"generated": 3, "duplicates": 1, "operators": {"mutation_rattle": 3}},
        "1": {
            "generated": 2,
            "duplicates": 2,
            "operators": {"bounded_mutation_add": 1, "mutation": 1},
        },
    }
    if not with_duplicates:
        for generation in generations.values():
            del generation["duplicates"]
    return {
        "events": {
            "data": {
                "ezga_summary": [
                    {
                        "seeds": 2,
                        "generations": 2,
                        "generated": 5,
                        "candidates": 5,
                        "oldest_generation_used": 0,
                    }
                ]
            }
        },
        "ezga": {
            "operator_names": [*NAMES, "crossover"],
            "by_generation": generations,
            "by_seed_type": {
                "surface100_rattle": {
                    "selected": 1,
                    "operators": {"mutation_rattle": 1},
                },
                "surface111_rattle": {
                    "selected": 2,
                    "operators": {"bounded_mutation_add": 2},
                },
            },
        },
    }


_CONFIG = {"structure_generation": {"ezga_kwargs": {}}}


def _captured_charts(monkeypatch) -> list[dict]:
    """Replace stacked_counts with a recorder of its arguments (writing an
    empty file in place of the figure, for the copy to current_plots)."""
    calls = []

    def record(groups, series, path, **kwargs):
        calls.append({"groups": groups, "series": series, "path": path, **kwargs})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    monkeypatch.setattr(plots, "stacked_counts", record)
    return calls


@pytest.mark.unit
def test_ezga_section_draws_both_figures_and_copies_them_to_current_plots(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    section = ezga_section(
        _ezga_stats(),
        base_name="al_loop_0",
        plots_dir=tmp_path / "plots",
        config=_CONFIG,
    )

    names = ["ezga_generations.png", "ezga_selected_by_seed_type.png"]
    assert [p.name for p in section.plots] == names
    for name in names:
        assert (tmp_path / "plots" / name).exists()
        assert (tmp_path / "results/current_plots/al_loop_0" / name).exists()


@pytest.mark.unit
def test_generation_columns_operators_then_duplicates_on_top(monkeypatch, tmp_path):
    """Every configured operator (zeros included) in config order, any other
    name after them, the duplicates last (top of the column); the number
    above a column is the structures generated."""
    monkeypatch.chdir(tmp_path)
    calls = _captured_charts(monkeypatch)

    ezga_section(
        _ezga_stats(), base_name="al_loop_0", plots_dir=tmp_path, config=_CONFIG
    )

    generations = calls[0]
    assert generations["groups"] == ["0", "1"]
    assert list(generations["series"]) == [
        *NAMES,
        "crossover",
        "mutation",
        "Duplicates filtered",
    ]
    assert generations["series"]["mutation_rattle"] == [3, 0]
    assert generations["series"]["crossover"] == [0, 0]
    assert generations["series"]["Duplicates filtered"] == [1, 2]
    assert generations["totals"] == [3, 2]


@pytest.mark.unit
def test_seed_type_columns_are_the_selected_structures(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = _captured_charts(monkeypatch)

    ezga_section(
        _ezga_stats(), base_name="al_loop_0", plots_dir=tmp_path, config=_CONFIG
    )

    seed_types = calls[1]
    assert seed_types["groups"] == ["surface100_rattle", "surface111_rattle"]
    assert seed_types["series"]["bounded_mutation_add"] == [0, 2]
    assert seed_types["totals"] == [1, 2]
    # Same operators, in the same order, as the generations figure (so the
    # same colours); no duplicates, which only a generation has.
    operators = [s for s in calls[0]["series"] if s != "Duplicates filtered"]
    assert list(seed_types["series"]) == operators


@pytest.mark.unit
def test_no_duplicates_series_without_duplicate_counts(monkeypatch, tmp_path):
    """Loops whose EZGA logger files are gone (or older report.json files)."""
    monkeypatch.chdir(tmp_path)
    calls = _captured_charts(monkeypatch)

    ezga_section(
        _ezga_stats(with_duplicates=False),
        base_name="al_loop_0",
        plots_dir=tmp_path,
        config=_CONFIG,
    )

    assert "Duplicates filtered" not in calls[0]["series"]


@pytest.mark.unit
def test_no_seed_type_figure_without_selected_structures(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = _captured_charts(monkeypatch)
    stats = _ezga_stats()
    stats["ezga"]["by_seed_type"] = {}

    section = ezga_section(
        stats, base_name="al_loop_0", plots_dir=tmp_path, config=_CONFIG
    )

    assert len(calls) == 1
    assert [p.name for p in section.plots] == ["ezga_generations.png"]


@pytest.mark.unit
def test_ezga_section_without_ezga_stats_or_plots_has_text_only(monkeypatch, tmp_path):
    """An older report.json has no "ezga" key; plots off gives no plots_dir."""
    monkeypatch.chdir(tmp_path)
    calls = _captured_charts(monkeypatch)
    old = _ezga_stats()
    del old["ezga"]

    for stats, plots_dir in ((old, tmp_path), (_ezga_stats(), None)):
        section = ezga_section(
            stats, base_name="al_loop_0", plots_dir=plots_dir, config=_CONFIG
        )
        assert section.plots == []
        assert section.lines

    assert calls == []
    assert not (tmp_path / "results").exists()


# ---------------------------------------------------------------------------
# A mutation ALomancy doesn't know yet
# ---------------------------------------------------------------------------


def _offspring(generation: int, energy: float, mutations: list[int]) -> Atoms:
    atoms = Atoms("Pd", cell=[3, 3, 3], pbc=True)
    atoms.info.update(
        ezga_generation=generation,
        ezga_operation="mutation",
        ezga_mutation_list=mutations,
        ezga_energy=energy,
        ezga_seed_db_id=7,
    )
    return atoms


@pytest.mark.unit
def test_a_new_mutation_gets_its_own_series_without_code_changes(monkeypatch, tmp_path):
    """Any mutation added to mutation_funcs (here one ALomancy has never
    heard of) is counted under its own name and drawn as its own series in
    both figures: nothing in the report lists the mutations it knows."""
    from ase.io import write

    from alomancy.analysis.report.stats import _ezga_generation_stats
    from alomancy.structure_generation.ezga import generate_structures

    monkeypatch.chdir(tmp_path)
    sg = tmp_path / "results/al_loop_0/structure_generation"
    (sg / "ezga/db").mkdir(parents=True)
    offspring = [
        _offspring(0, -1.0, [0]),
        _offspring(0, -2.0, [2]),
        _offspring(1, -3.0, [2]),
        _offspring(1, -4.0, [1]),
    ]
    monkeypatch.setattr(generate_structures, "read_offspring", lambda path: offspring)
    write(sg / "high_sd_structures.xyz", offspring[1:3], format="extxyz")
    mutation_funcs = [
        {"type": "ezga.variation.mutation.mutation_rattle"},
        {"type": "ezga.variation.mutation.mutation_random_strain"},
        {"type": "some.future.module.mutation_brand_new", "species": ["Pd"]},
    ]

    ezga = _ezga_generation_stats("al_loop_0", mutation_funcs, {7: "surface111_rattle"})
    calls = _captured_charts(monkeypatch)
    ezga_section(
        {**_ezga_stats(), "ezga": ezga},
        base_name="al_loop_0",
        plots_dir=tmp_path,
        config=_CONFIG,
    )

    generations, seed_types = calls
    assert "mutation_brand_new" in ezga["operator_names"]
    assert generations["series"]["mutation_brand_new"] == [1, 1]
    assert generations["totals"] == [2, 2]
    assert seed_types["series"]["mutation_brand_new"] == [2]
    assert seed_types["groups"] == ["surface111_rattle"]
