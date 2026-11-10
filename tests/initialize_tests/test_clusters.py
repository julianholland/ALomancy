"""Non-periodic initialisation (amorphous_cluster_kwargs,
molecule_cluster_kwargs) and the evaluator-dependent block defaults
(initialize/initialiser_interface.py), for evaluators without periodic
boundary conditions such as ORCA."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from alomancy.initialize import initialiser_interface
from alomancy.initialize.amorphize import create_amorphous_cluster_atoms_list
from alomancy.initialize.initialiser_interface import (
    compute_needs,
    generate,
    kwargs_defaults,
    validate_kwargs,
)
from alomancy.initialize.molecules import create_molecule_cluster_atoms_list
from alomancy.utils.dataset_curation import structure_domain

_WATER_CLUSTER = {
    "molecules": {"H2O": 1},
    "density_range": [0.5, 1.0],
    "num_of_molecules": 4,
    "num_of_structures": 2,
}


def _db(all_counts=None):
    db = MagicMock()
    db.count_all_by_config_type_and_formula.return_value = all_counts or {}
    return db


def _min_interatomic(atoms):
    d = atoms.get_all_distances()
    return float(d[~np.eye(len(atoms), dtype=bool)].min())


@pytest.mark.unit
def test_amorphous_clusters_are_non_periodic_and_seeded():
    clusters = create_amorphous_cluster_atoms_list(
        ["C", "H"], atom_number=8, density_range=[0.5, 1.0], num_structures=3
    )
    again = create_amorphous_cluster_atoms_list(
        ["C", "H"], atom_number=8, density_range=[0.5, 1.0], num_structures=3
    )
    assert len(clusters) == 3
    for a, b in zip(clusters, again, strict=True):
        assert not a.pbc.any()
        assert len(a) == 8
        assert a.info["config_type"] == "init_amorphous_cluster"
        assert a.info["needs_relaxation"] is True
        assert np.allclose(a.positions, b.positions)
        assert structure_domain(a) == "cluster"
    restart = create_amorphous_cluster_atoms_list(
        ["C", "H"], atom_number=8, density_range=[0.5, 1.0], num_structures=1, attempt=3
    )
    assert not np.allclose(restart[0].positions, clusters[0].positions)


@pytest.mark.unit
def test_molecule_clusters_are_non_periodic_and_keep_their_distance():
    clusters = create_molecule_cluster_atoms_list(
        molecules_ratio={"H2O": 1},
        density_range=[0.5, 1.0],
        num_of_molecules=4,
        num_of_structures=2,
        min_distance=1.5,
        max_attempts=1000,
        seed=803,
    )
    assert len(clusters) == 2
    for cluster in clusters:
        assert not cluster.pbc.any()
        assert cluster.get_chemical_formula() == "H8O4"
        assert cluster.info["config_type"] == "init_molecule_cluster"
        assert structure_domain(cluster) == "molecule_cluster"
        # Atoms of different molecules (each 3 atoms, placed in order)
        # never come closer than min_distance.
        for i in range(4):
            for j in range(i + 1, 4):
                a = cluster.positions[3 * i : 3 * i + 3]
                b = cluster.positions[3 * j : 3 * j + 3]
                gaps = np.linalg.norm(a[:, None] - b[None], axis=-1)
                assert gaps.min() >= 1.5 - 1e-9


@pytest.mark.unit
def test_non_periodic_defaults():
    defaults = kwargs_defaults(evaluator_supports_periodic=False)
    assert not defaults["amorphous_kwargs"]["enabled"]
    assert not defaults["mp_kwargs"]["enabled"]
    assert not defaults["molecules_in_box_kwargs"]["enabled"]
    assert defaults["amorphous_cluster_kwargs"]["enabled"]
    assert defaults["molecule_cluster_kwargs"]["enabled"]
    periodic = kwargs_defaults()
    assert periodic["amorphous_kwargs"]["enabled"]
    assert not periodic["amorphous_cluster_kwargs"]["enabled"]


@pytest.mark.unit
def test_needs_without_periodic_boundary_conditions():
    needs = compute_needs(_db(), {}, ["H", "O"], evaluator_supports_periodic=False)
    assert needs["amorphous_override"] == 0
    assert needs["mp_structures"] is False
    assert needs["amorphous_cluster_override"] == 50
    # Default-on, but nothing to build without cluster_compositions.
    assert needs["molecule_cluster_override"] == {}
    needs = compute_needs(
        _db({"init_molecule_cluster": {"H8O4": 1}}),
        {"molecule_cluster_kwargs": {"cluster_compositions": [_WATER_CLUSTER]}},
        ["H", "O"],
        evaluator_supports_periodic=False,
    )
    assert needs["molecule_cluster_override"] == {"H8O4": 1}


@pytest.mark.unit
def test_generate_makes_only_non_periodic_structures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "amorphous_cluster_kwargs": {
            "num_of_amorphous_clusters": 2,
            "num_of_atoms_per_cluster": 5,
        },
        "molecule_cluster_kwargs": {"cluster_compositions": [_WATER_CLUSTER]},
    }
    structures = generate(
        config,
        base_name="initialization",
        name="initialization",
        elements=["H", "O"],
        evaluator_supports_periodic=False,
    )
    config_types = {a.info["config_type"] for a in structures}
    assert "init_amorphous" not in config_types
    assert {"init_amorphous_cluster", "init_molecule_cluster"} <= config_types
    assert not any(a.pbc.any() for a in structures)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("block", "replacement"),
    [
        ("amorphous_kwargs", "amorphous_cluster_kwargs"),
        ("molecules_in_box_kwargs", "molecule_cluster_kwargs"),
        ("mp_kwargs", "no non-periodic equivalent"),
    ],
)
def test_explicitly_enabled_periodic_blocks_raise(block, replacement):
    with pytest.raises(ValueError, match=replacement):
        validate_kwargs(
            {block: {"enabled": True}},
            elements=["H", "O"],
            evaluator_supports_periodic=False,
        )


@pytest.mark.unit
def test_default_on_molecule_clusters_without_compositions_are_skipped():
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = initialiser_interface.logger
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        validate_kwargs({}, elements=["H", "O"], evaluator_supports_periodic=False)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    messages = " ".join(r.getMessage() for r in records)
    assert "amorphous_kwargs off" in messages
    assert "no clusters of molecules are made" in messages
    with pytest.raises(ValueError, match="cluster_compositions is required"):
        validate_kwargs(
            {"molecule_cluster_kwargs": {"enabled": True}},
            elements=["H", "O"],
            evaluator_supports_periodic=False,
        )


@pytest.mark.unit
def test_structures_larger_than_the_evaluator_cap_raise():
    with pytest.raises(ValueError, match="amorphous_cluster_kwargs clusters"):
        validate_kwargs(
            {"amorphous_cluster_kwargs": {"num_of_atoms_per_cluster": 25}},
            elements=["H", "O"],
            evaluator_supports_periodic=False,
            max_num_of_atoms=20,
        )
    with pytest.raises(ValueError, match="H2O:1 \\(12 atoms\\)"):
        validate_kwargs(
            {"molecule_cluster_kwargs": {"cluster_compositions": [_WATER_CLUSTER]}},
            elements=["H", "O"],
            evaluator_supports_periodic=False,
            max_num_of_atoms=10,
        )
    # Within the cap: fine.
    validate_kwargs(
        {"molecule_cluster_kwargs": {"cluster_compositions": [_WATER_CLUSTER]}},
        elements=["H", "O"],
        evaluator_supports_periodic=False,
        max_num_of_atoms=20,
    )
