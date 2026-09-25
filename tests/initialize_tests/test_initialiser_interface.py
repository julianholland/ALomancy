"""Tests for initialize/initialiser_interface.py -- the modular AL
architecture's initialiser entry points (compute_needs, generate,
output_paths, read_existing_result).

compute_initialization_needs/create_initialization_atoms_list themselves
are unchanged and already covered by tests/initialize_tests/
test_initialization.py and test_create_initialization.py -- these tests
cover only this module's config -> kwargs wiring and restart entry points.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from alomancy.initialize.initialiser_interface import (
    compute_needs,
    generate,
    output_paths,
    read_existing_result,
)

_MODULE = "alomancy.initialize.initialiser_interface"


def _mock_db(all_counts):
    db = MagicMock()
    db.count_all_by_config_type_and_formula.return_value = all_counts
    return db


def _atoms(n=2):
    return [
        Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True) for _ in range(n)
    ]


@pytest.mark.unit
class TestComputeNeeds:
    def test_reads_targets_from_creation_kwargs(self):
        db = _mock_db({})
        config = {
            "creation_kwargs": {
                "dimer_kwargs": {"num_dimers_per_combo": 3},
                "trimer_kwargs": {"num_trimers_per_combo": 2},
                "amorphous_kwargs": {"num_amorphous": 10},
            }
        }
        needs = compute_needs(db, config, ["H", "O"])
        assert needs["isolated_atoms"] == ["H", "O"]
        assert needs["amorphous_override"] == 10

    def test_uses_defaults_when_not_specified(self):
        db = _mock_db({})
        config = {"creation_kwargs": {}}
        needs = compute_needs(db, config, ["H"])
        # Defaults match compute_initialization_needs' own defaults, applied
        # here since config doesn't override them.
        assert needs["amorphous_override"] == 100

    def test_nothing_needed_when_db_already_satisfies_targets(self):
        db = _mock_db(
            {
                "IsolatedAtom": {"H": 1},
                "init_amorphous": {"H100": 100},
            }
        )
        config = {"creation_kwargs": {"amorphous_kwargs": {"num_amorphous": 100}}}
        needs = compute_needs(db, config, ["H"])
        assert needs["isolated_atoms"] == []
        assert needs["amorphous_override"] == 0

    def test_disabled_sub_tasks_need_nothing_regardless_of_configured_count(self):
        db = _mock_db({})
        config = {
            "creation_kwargs": {
                "dimer_kwargs": {"enabled": False, "num_dimers_per_combo": 5},
                "trimer_kwargs": {"enabled": False, "num_trimers_per_combo": 5},
                "amorphous_kwargs": {"enabled": False, "num_amorphous": 100},
            }
        }
        needs = compute_needs(db, config, ["H", "O"])
        assert needs["dimer_override"] == {}
        assert needs["trimer_override"] == {}
        assert needs["amorphous_override"] == 0


@pytest.mark.unit
class TestOutputPathsAndReadExistingResult:
    def test_output_paths_is_the_generated_structures_file(self):
        paths = output_paths({}, base_name="al_loop_0", name="initialization")
        assert paths == [
            Path("results/al_loop_0/initialization_structures_generated.xyz")
        ]

    def test_read_existing_result_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No cached"):
            read_existing_result({}, base_name="al_loop_0", name="initialization")

    def test_read_existing_result_reads_generated_structures(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        path = Path("results/al_loop_0/initialization_structures_generated.xyz")
        path.parent.mkdir(parents=True)
        write(str(path), _atoms(3), format="extxyz")
        result = read_existing_result({}, base_name="al_loop_0", name="initialization")
        assert len(result) == 3


@pytest.mark.unit
class TestGenerate:
    def _config(self, **creation_overrides):
        creation_kwargs: dict = {}
        creation_kwargs.update(creation_overrides)
        return {"creation_kwargs": creation_kwargs}

    def test_full_generation_when_no_needs_given(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = _atoms(5)
            result = generate(
                self._config(),
                base_name="al_loop_0",
                name="initialization",
                elements=["H", "O"],
                hpc={},
                max_time="1H",
            )

        assert len(result) == 5
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["elements"] == ["H", "O"]
        assert call_kwargs["isolated_atoms_override"] is None
        assert call_kwargs["dimer_override"] is None
        assert call_kwargs["mp_structures"] is True  # config default, no needs given

    def test_enabled_flags_read_from_their_own_namespaces(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            isolated_atom_kwargs={"enabled": False},
            mp_kwargs={"enabled": False},
            dimer_kwargs={"enabled": False, "num_dimers_per_combo": 5},
            trimer_kwargs={"enabled": False, "num_trimers_per_combo": 3},
            amorphous_kwargs={"enabled": False, "num_amorphous": 20},
        )
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                config,
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["single_atoms"] is False
        assert call_kwargs["mp_structures"] is False
        # Disabled sub-tasks force their count to 0 rather than passing
        # their configured value through -- create_initialization_atoms_list
        # has no separate "enabled" concept of its own for these.
        assert call_kwargs["num_dimers_per_combo"] == 0
        assert call_kwargs["num_trimers_per_combo"] == 0
        assert call_kwargs["num_amorphous"] == 0

    def test_enabled_defaults_to_true_for_dimer_trimer_amorphous(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            dimer_kwargs={"num_dimers_per_combo": 5},
            trimer_kwargs={"num_trimers_per_combo": 3},
            amorphous_kwargs={"num_amorphous": 20},
        )
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                config,
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["num_dimers_per_combo"] == 5
        assert call_kwargs["num_trimers_per_combo"] == 3
        assert call_kwargs["num_amorphous"] == 20

    def test_stretch_compress_targets_kwargs_is_a_top_level_namespace(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            mp_kwargs={"max_atom_number": 30},
            stretch_compress_targets_kwargs={
                "num_stretch_compress_per_mp": 7,
                "deform_xyz": True,
                "max_deformation": 0.5,
            },
        )
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                config,
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_atom_number"] == 30
        assert call_kwargs["num_stretch_compress_per_mp"] == 7
        assert call_kwargs["deform_xyz"] is True
        assert call_kwargs["max_deformation"] == 0.5

    def test_stretch_compress_targets_disabled_forces_count_to_zero(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            stretch_compress_targets_kwargs={
                "enabled": False,
                "num_stretch_compress_per_mp": 7,
            },
        )
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                config,
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
            )

        assert mock_create.call_args.kwargs["num_stretch_compress_per_mp"] == 0

    def test_partial_generation_when_needs_given(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        needs = {
            "isolated_atoms": ["H"],
            "dimer_override": {"H2": 3},
            "trimer_override": {},
            "amorphous_override": 0,
            "mp_structures": False,
        }
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = _atoms(1)
            generate(
                self._config(),
                base_name="al_loop_0",
                name="initialization",
                elements=["H", "O"],
                hpc={},
                max_time="1H",
                needs=needs,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["isolated_atoms_override"] == ["H"]
        assert call_kwargs["dimer_override"] == {"H2": 3}
        assert call_kwargs["trimer_override"] is None  # empty dict -> None
        assert call_kwargs["amorphous_override"] is None  # 0 -> None
        assert call_kwargs["mp_structures"] is False
        assert call_kwargs["single_atoms"] is True  # bool(["H"])

    def test_creates_work_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                self._config(),
                base_name="al_loop_0",
                name="initialization",
                elements=["H", "O"],
                hpc={},
                max_time="1H",
            )
        assert Path("results/al_loop_0").is_dir()
