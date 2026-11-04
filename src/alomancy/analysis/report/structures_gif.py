"""A looping GIF of structures for the loop report: each one rocks gently
back and forth so it can be seen from several angles.

Drawn headlessly with ASE's matplotlib renderer (``plot_atoms``) and saved
with PIL (installed with matplotlib), so it works wherever the run's driver
runs, without a display. sage_lib's renderer needs an OpenGL window, see
TODO.md.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
from PIL import Image

from alomancy.analysis.palette import ALOMANCY_PURPLE

# Structures shown, evenly spaced through the file.
MAX_STRUCTURES = 6
# Frames per structure: one full rock, out to +ROCK_DEGREES, back through
# -ROCK_DEGREES and home again, so the loop has no jump.
FRAMES_PER_STRUCTURE = 10
ROCK_DEGREES = 20.0
# Cells smaller than this are tiled (to about this many atoms per tile), so
# a tiny crystal reads as a crystal rather than two lone atoms; larger
# cells are shown alone, with their cell outline, as tiling them hides
# their structure in a blob of atoms.
_MIN_DISPLAY_ATOMS = 10
_FRAME_SECONDS = 0.12
_SIZE_PX = 360


def _for_display(atoms: Atoms) -> Atoms:
    """*atoms* wrapped into its cell, tiled when it is small and periodic."""
    atoms = atoms.copy()
    atoms.calc = None
    if not atoms.pbc.all() or atoms.cell.rank < 3:
        return atoms
    atoms.wrap()
    if len(atoms) >= _MIN_DISPLAY_ATOMS:
        return atoms
    reps = min(4, math.ceil((_MIN_DISPLAY_ATOMS / len(atoms)) ** (1 / 3)))
    return atoms.repeat((reps, reps, reps))


def rock_angles(n_frames: int = FRAMES_PER_STRUCTURE) -> list[float]:
    """Rotation about the vertical axis for each frame: one sine period."""
    return [
        ROCK_DEGREES * math.sin(2 * math.pi * k / n_frames) for k in range(n_frames)
    ]


def _frame(atoms: Atoms, caption: str, angle: float) -> Image.Image:
    dpi = 100
    fig, ax = plt.subplots(figsize=(_SIZE_PX / dpi, _SIZE_PX / dpi), dpi=dpi)
    try:
        plot_atoms(
            atoms, ax, rotation=f"{angle:.1f}y,-20x", radii=0.6, show_unit_cell=2
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(caption, fontsize=11, color=ALOMANCY_PURPLE)
        # Fixed margins: tight_layout measures every patch (the cell
        # outline alone is hundreds), several times slower.
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        fig.patch.set_facecolor("white")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
    finally:
        plt.close(fig)
    return Image.fromarray(rgba[..., :3].copy())


def structures_gif(
    structures: list[Atoms],
    path: Path,
    *,
    max_structures: int | None = None,
    frames_per_structure: int | None = None,
) -> Path | None:
    """Write a looping GIF of up to *max_structures* of *structures*
    (evenly spaced; default MAX_STRUCTURES), each rocking through
    *frames_per_structure* frames (default FRAMES_PER_STRUCTURE) with its
    formula and position as the caption. None if there are no
    structures."""
    if not structures:
        return None
    max_structures = max_structures or MAX_STRUCTURES
    frames_per_structure = frames_per_structure or FRAMES_PER_STRUCTURE
    picks = list(
        dict.fromkeys(
            np.linspace(0, len(structures) - 1, min(max_structures, len(structures)))
            .round()
            .astype(int)
            .tolist()
        )
    )
    frames = []
    for position, index in enumerate(picks, start=1):
        atoms = structures[index]
        caption = (
            f"{atoms.get_chemical_formula(mode='hill')}  ({position}/{len(picks)})"
        )
        shown = _for_display(atoms)
        frames += [
            _frame(shown, caption, angle) for angle in rock_angles(frames_per_structure)
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(_FRAME_SECONDS * 1000),
        loop=0,
    )
    return path


def added_structures_gif(base_name: str, path: Path) -> Path | None:
    """The GIF of the structures this loop evaluated with DFT: the final
    geometries in ``results/<base_name>/high_accuracy_eval_results.xyz``,
    before curation. None if the loop has none; an existing GIF newer than
    that file is reused rather than redrawn."""
    results = Path("results", base_name, "high_accuracy_eval_results.xyz")
    if not results.exists() or results.stat().st_size == 0:
        return None
    if path.exists() and path.stat().st_mtime >= results.stat().st_mtime:
        return path
    return structures_gif(list(read(results, ":", format="extxyz")), path)
