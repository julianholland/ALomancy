import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from alomancy.analysis.colors import PALETTE, add_logo_watermark, setup_alomancy_style

logger = logging.getLogger(__name__)


def _pairwise_distances_by_element_pair(
    atoms_list: list[Atoms], max_distance: float
) -> dict[tuple[str, str], list[float]]:
    """Bucket every pairwise atomic distance (up to max_distance) by sorted
    element-pair, across every structure in atoms_list.

    get_all_distances(mic=True) is safe across periodic and non-periodic
    structures alike -- same building block filter_structures_by_min_bond_
    distance (utils/clean_structures.py) already uses. Structures with
    fewer than 2 atoms are skipped -- no pairwise distance to bucket.
    """
    distances_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for atoms in atoms_list:
        if len(atoms) < 2:
            continue
        symbols = atoms.get_chemical_symbols()
        dm = atoms.get_all_distances(mic=True)
        i_idx, j_idx = np.triu_indices(len(atoms), k=1)
        for i, j, d in zip(i_idx, j_idx, dm[i_idx, j_idx], strict=True):
            if d > max_distance:
                continue
            pair = tuple(sorted((symbols[i], symbols[j])))
            distances_by_pair[pair].append(float(d))
    return distances_by_pair


def plot_training_bond_distances(
    base_name: str,
    db: Any,
    plots_dir: Path,
    max_distance: float = 5.0,
    min_bond_distance_ref: float = 0.5,
) -> None:
    """Plot a per-element-pair histogram of pairwise interatomic distances
    across the current training set, one panel per element pair.

    Reads db.get_train_atoms(exclude_duplicates=True, exclude_high_force=True)
    directly -- train-split structures only, with anything flagged by
    remove_redundancy_from_partition or remove_high_force_structures_from_
    partition already excluded, matching what the committee actually trains
    on rather than everything the DB happens to hold.

    Intended to run at the start of every AL loop, before train_mlip, as a
    diagnostic for MD-instability incidents (see filter_structures_by_min_
    bond_distance): if MD is collapsing into unphysically short-range
    configurations, this plot shows whether the training set genuinely has
    coverage of that short-range/repulsive-wall regime for the relevant
    element pairs -- distinguishing "the model never saw this data" from
    "the model saw it but the flagging pipeline is filtering it out before
    training," which look identical from the MD-collapse symptom alone.
    """
    atoms_list = db.get_train_atoms(exclude_duplicates=True, exclude_high_force=True)
    if not atoms_list:
        logger.warning(
            "No training structures available -- skipping bond-distance plot."
        )
        return

    distances_by_pair = _pairwise_distances_by_element_pair(atoms_list, max_distance)
    if not distances_by_pair:
        logger.warning(
            "No pairwise distances found within %.2f Å -- skipping bond-distance plot.",
            max_distance,
        )
        return

    setup_alomancy_style()
    pairs = sorted(distances_by_pair)
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False
    )

    for idx, pair in enumerate(pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        d = distances_by_pair[pair]
        color = PALETTE[idx % len(PALETTE)]
        ax.hist(d, bins=40, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(
            min_bond_distance_ref,
            color="#E11D48",
            linestyle="--",
            linewidth=1.2,
            label=f"{min_bond_distance_ref:.2f} Å DFT filter",
        )
        ax.set_title(f"{pair[0]}-{pair[1]} (n={len(d)})", fontsize=10)
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    # Blank any unused grid cells (n_pairs need not fill n_rows * n_cols).
    for idx in range(n_pairs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"Training-set pairwise bond distances [{base_name}]")
    fig.tight_layout()
    add_logo_watermark(fig)
    path = plots_dir / f"train_bond_distances_{base_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved training bond-distance plot to %s", path)
