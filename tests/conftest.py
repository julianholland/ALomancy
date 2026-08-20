"""Test Configuration and Fixtures for ALomancy test suite."""

import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

# ---------------------------------------------------------------------------
# Utility: build test Atoms objects
# ---------------------------------------------------------------------------


def make_atoms(
    symbols: list,
    config_type=None,
    ref_energy=None,
    ref_forces=None,
    needs_relaxation=False,
    cell=10.0,
):
    n = len(symbols)
    positions = np.eye(n, 3) * 2.0
    atoms = Atoms(symbols=symbols, positions=positions, cell=[cell] * 3, pbc=True)
    if config_type is not None:
        atoms.info["config_type"] = config_type
    if ref_energy is not None:
        atoms.info["REF_energy"] = ref_energy
    if ref_forces is not None:
        atoms.arrays["REF_forces"] = np.array(ref_forces)
    if needs_relaxation:
        atoms.info["needs_relaxation"] = True
    return atoms


# ---------------------------------------------------------------------------
# Common Atoms fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def h_atom():
    return make_atoms(
        ["H"],
        config_type="IsolatedAtom",
        ref_energy=-13.6,
        ref_forces=[[0.0, 0.0, 0.0]],
    )


@pytest.fixture
def o_atom():
    return make_atoms(
        ["O"],
        config_type="IsolatedAtom",
        ref_energy=-432.0,
        ref_forces=[[0.0, 0.0, 0.0]],
    )


@pytest.fixture
def h2_dimer():
    return make_atoms(
        ["H", "H"],
        config_type="init_dimer",
        ref_energy=-31.0,
        ref_forces=[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
    )


@pytest.fixture
def h2o_mol():
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.info["config_type"] = "al_loop_0"
    atoms.info["REF_energy"] = -76.0
    atoms.arrays["REF_forces"] = np.zeros((3, 3))
    return atoms


@pytest.fixture
def sample_atoms(h2o_mol):
    return h2o_mol


@pytest.fixture
def sample_atoms_list(h2o_mol):
    result = []
    for i in range(5):
        a = h2o_mol.copy()
        a.positions += np.random.default_rng(i).random((3, 3)) * 0.1
        result.append(a)
    return result


# ---------------------------------------------------------------------------
# File fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_xyz_file(tmp_path, h2o_mol, h_atom):
    p = tmp_path / "test_structures.xyz"
    write(str(p), [h_atom, h2o_mol], format="extxyz")
    return p


# ---------------------------------------------------------------------------
# Minimal jobs_dict
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_jobs_dict():
    return {
        "initialization": {
            "name": "initialization",
            "max_time": "1H",
            "extra_datasets": [],
            "test_to_train_ratio": 0.1,
            "test_config_types": ["IsolatedAtom"],
            "creation_kwargs": {"elements": ["H", "O"]},
            "hpc": {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]},
        },
        "mlip_committee": {
            "name": "mlip_committee",
            "size_of_committee": 3,
            "max_time": "1H",
            "hpc": {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]},
        },
        "structure_generation": {
            "name": "structure_generation",
            "desired_number_of_structures": 5,
            "max_time": "30m",
            "hpc": {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]},
        },
        "high_accuracy_evaluation": {
            "name": "high_accuracy_evaluation",
            "max_time": "10m",
            "max_batch_size": 10,
            "hpc": {"hpc_name": "test-hpc", "pre_cmds": [], "partitions": ["test"]},
        },
    }


@pytest.fixture
def mock_job_dict(minimal_jobs_dict):
    return minimal_jobs_dict


# ---------------------------------------------------------------------------
# Environment setup (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ALOMANCY_TEST_MODE", "1")
    monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "1")
    monkeypatch.setenv("ALOMANCY_TEST_DATA_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def skip_if_no_external():
    if os.getenv("ALOMANCY_MOCK_EXTERNAL", "1") == "1":
        pytest.skip("External dependencies not available in test environment")


@pytest.fixture
def skip_if_no_gpu():
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")


# ---------------------------------------------------------------------------
# Shared GlobalDatabase instance (performance)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _session_global_database(tmp_path_factory):
    """One real GlobalDatabase, constructed once for the whole test session.

    Constructing a GlobalDatabase bootstraps real HDF5+SQLite files on disk
    (sage_lib's Partition(storage="hybrid")) and costs ~1-3s even for a
    brand-new, empty DB -- confirmed by direct benchmarking. Dozens of tests
    across tests/core_tests/ each used to pay this cost individually via
    their own db_path=str(tmp_path / "db"), and were the single biggest
    contributor to this repo's `pytest -m unit` runtime. `shared_db` (below)
    clears this same instance before every test instead, which costs
    ~0.002s -- see GlobalDatabase.clear()'s docstring for the full
    benchmark comparison and why it's safe (global_db_id numbering is
    derived fresh from partition.N at add-time, so it isn't affected by
    reusing an instance across tests).
    """
    from alomancy.database.global_database import GlobalDatabase

    db_path = tmp_path_factory.mktemp("shared_global_db")
    return GlobalDatabase(str(db_path))


@pytest.fixture
def shared_db(_session_global_database):
    """A GlobalDatabase reused across the whole test session but cleared to
    empty before every individual test -- functionally identical to a fresh
    GlobalDatabase(db_path) for any test that doesn't care about the exact
    filesystem path, without paying that construction's ~1-3s cost per test.

    Safe under both plain `pytest` (sequential within a process) and
    pytest-xdist (distributes whole test files/classes to separate worker
    processes, each with its own conftest session fixtures -- never splits
    one test file's tests across workers in a way that would race on this
    instance). Not suitable for a test that specifically asserts on the
    DB's on-disk path/location, or that needs to inspect the raw contents
    of a `db_path` directory it controls -- those tests should keep
    constructing their own GlobalDatabase(str(tmp_path / "db")) directly.
    """
    _session_global_database.clear()
    return _session_global_database
