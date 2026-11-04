"""Tests for analysis/report/structures_gif.py: the rocking GIF of the
structures a loop evaluated with DFT."""

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import write
from PIL import Image

from alomancy.analysis.report.structures_gif import (
    _for_display,
    added_structures_gif,
    rock_angles,
    structures_gif,
)


def _total_ms(path: Path) -> int:
    """Total playing time; PIL merges identical consecutive frames into
    one longer one, so the frame count alone can be lower."""
    image = Image.open(path)
    total = 0
    for i in range(image.n_frames):
        image.seek(i)
        total += image.info["duration"]
    return total


@pytest.mark.unit
def test_rock_swings_out_and_back_without_a_jump():
    angles = rock_angles(8)
    assert angles[0] == 0
    assert max(angles) == pytest.approx(20.0)
    assert min(angles) == pytest.approx(-20.0)
    # The loop closes: the step after the last frame is back at 0.
    assert abs(angles[-1]) < 20.0


@pytest.mark.unit
def test_gif_has_every_frame_of_every_structure(tmp_path):
    structures = [bulk("Si", "diamond", a=5.43), bulk("Cu", "fcc", a=3.6)]
    path = structures_gif(
        structures, tmp_path / "s.gif", max_structures=5, frames_per_structure=3
    )
    assert path is not None
    assert _total_ms(path) == 2 * 3 * 120
    assert Image.open(path).info.get("loop") == 0


@pytest.mark.unit
def test_gif_samples_at_most_max_structures(tmp_path):
    structures = [bulk("Cu", "fcc", a=3.6 + 0.01 * i) for i in range(5)]
    path = structures_gif(
        structures, tmp_path / "s.gif", max_structures=2, frames_per_structure=2
    )
    assert _total_ms(path) == 2 * 2 * 120


@pytest.mark.unit
def test_small_cells_are_tiled_and_large_ones_shown_alone():
    assert len(_for_display(bulk("Cu", "fcc", a=3.6))) > 1
    big = bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3))
    assert len(_for_display(big)) == len(big)
    molecule = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    assert len(_for_display(molecule)) == 2


@pytest.mark.unit
def test_added_structures_gif_reads_the_loops_dft_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "plots" / "added.gif"
    assert added_structures_gif("al_loop_0", out) is None

    results = Path("results/al_loop_0/high_accuracy_eval_results.xyz")
    results.parent.mkdir(parents=True)
    write(results, [bulk("Si", "diamond", a=5.43)], format="extxyz")
    from alomancy.analysis.report import structures_gif as module

    monkeypatch.setattr(module, "FRAMES_PER_STRUCTURE", 2)
    assert added_structures_gif("al_loop_0", out) == out
    assert out.exists()


@pytest.mark.unit
def test_added_structures_gif_is_reused_until_the_results_change(tmp_path, monkeypatch):
    """Each report write redraws the GIF only if the loop's DFT results
    are newer than it."""
    import os

    from alomancy.analysis.report import structures_gif as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "FRAMES_PER_STRUCTURE", 2)
    results = Path("results/al_loop_0/high_accuracy_eval_results.xyz")
    results.parent.mkdir(parents=True)
    write(results, [bulk("Si", "diamond", a=5.43)], format="extxyz")
    out = tmp_path / "plots" / "added.gif"
    added_structures_gif("al_loop_0", out)

    drawn = out.stat().st_mtime
    os.utime(out, (drawn + 10, drawn + 10))
    calls = []
    monkeypatch.setattr(module, "structures_gif", lambda *a, **k: calls.append(a))
    assert added_structures_gif("al_loop_0", out) == out
    assert calls == []

    os.utime(results, (drawn + 20, drawn + 20))
    added_structures_gif("al_loop_0", out)
    assert len(calls) == 1
