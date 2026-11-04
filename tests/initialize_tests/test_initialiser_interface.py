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

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alomancy.initialize.initialiser_interface import (
    _resolve_kwargs,
    compute_needs,
    generate,
    output_paths,
    read_existing_result,
    validate_kwargs,
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
    @pytest.mark.unit
    def test_reads_targets_from_structure_type_namespaces(self):
        db = _mock_db({})
        config = {
            "dimer_kwargs": {"num_of_dimers_per_combo": 3},
            "trimer_kwargs": {"num_of_trimers_per_combo": 2},
            "amorphous_kwargs": {"num_of_amorphous_structures": 10},
        }
        needs = compute_needs(db, config, ["H", "O"])
        assert needs["isolated_atoms"] == ["H", "O"]
        assert needs["amorphous_override"] == 10

    @pytest.mark.unit
    def test_uses_defaults_when_not_specified(self):
        db = _mock_db({})
        needs = compute_needs(db, {}, ["H"])
        # Defaults match compute_initialization_needs' own defaults, applied
        # here since config doesn't override them.
        assert needs["amorphous_override"] == 100

    @pytest.mark.unit
    def test_nothing_needed_when_db_already_satisfies_targets(self):
        db = _mock_db(
            {
                "IsolatedAtom": {"H": 1},
                "init_amorphous": {"H100": 100},
            }
        )
        config = {"amorphous_kwargs": {"num_of_amorphous_structures": 100}}
        needs = compute_needs(db, config, ["H"])
        assert needs["isolated_atoms"] == []
        assert needs["amorphous_override"] == 0

    @pytest.mark.unit
    def test_disabled_sub_tasks_need_nothing_regardless_of_configured_count(self):
        db = _mock_db({})
        config = {
            "dimer_kwargs": {"enabled": False, "num_of_dimers_per_combo": 5},
            "trimer_kwargs": {"enabled": False, "num_of_trimers_per_combo": 5},
            "amorphous_kwargs": {"enabled": False, "num_of_amorphous_structures": 100},
        }
        needs = compute_needs(db, config, ["H", "O"])
        assert needs["dimer_override"] == {}
        assert needs["trimer_override"] == {}
        assert needs["amorphous_override"] == 0


@pytest.mark.unit
class TestOutputPathsAndReadExistingResult:
    @pytest.mark.unit
    def test_output_paths_is_the_generated_structures_file(self):
        paths = output_paths({}, base_name="al_loop_0", name="initialization")
        assert paths == [
            Path("results/al_loop_0/initialization_structures_generated.xyz")
        ]

    @pytest.mark.unit
    def test_read_existing_result_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No cached"):
            read_existing_result({}, base_name="al_loop_0", name="initialization")

    @pytest.mark.unit
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
    def _config(self, **overrides):
        return dict(overrides)

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_enabled_flags_read_from_their_own_namespaces(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            isolated_atom_kwargs={"enabled": False},
            mp_kwargs={"enabled": False},
            dimer_kwargs={"enabled": False, "num_of_dimers_per_combo": 5},
            trimer_kwargs={"enabled": False, "num_of_trimers_per_combo": 3},
            amorphous_kwargs={"enabled": False, "num_of_amorphous_structures": 20},
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
        assert call_kwargs["num_of_dimers_per_combo"] == 0
        assert call_kwargs["num_of_trimers_per_combo"] == 0
        assert call_kwargs["num_of_amorphous_structures"] == 0

    @pytest.mark.unit
    def test_enabled_defaults_to_true_for_dimer_trimer_amorphous(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            dimer_kwargs={"num_of_dimers_per_combo": 5},
            trimer_kwargs={"num_of_trimers_per_combo": 3},
            amorphous_kwargs={"num_of_amorphous_structures": 20},
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
        assert call_kwargs["num_of_dimers_per_combo"] == 5
        assert call_kwargs["num_of_trimers_per_combo"] == 3
        assert call_kwargs["num_of_amorphous_structures"] == 20

    @pytest.mark.unit
    def test_stretch_compress_targets_kwargs_is_a_top_level_namespace(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            mp_kwargs={"max_num_of_atoms": 30},
            stretch_compress_targets_kwargs={
                "num_of_stretch_compress_per_target": 7,
                "max_lattice_deformation": 0.5,
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
        assert call_kwargs["max_num_of_atoms"] == 30
        assert call_kwargs["num_of_stretch_compress_per_target"] == 7
        assert call_kwargs["max_lattice_deformation"] == 0.5

    @pytest.mark.unit
    def test_stretch_compress_targets_disabled_forces_count_to_zero(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            stretch_compress_targets_kwargs={
                "enabled": False,
                "num_of_stretch_compress_per_target": 7,
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

        assert mock_create.call_args.kwargs["num_of_stretch_compress_per_target"] == 0

    @pytest.mark.unit
    def test_target_config_types_and_seed_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                self._config(),
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
                target_config_types=["init_MP", "init_amorphous"],
                seed=42,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["target_config_types"] == ["init_MP", "init_amorphous"]
        assert call_kwargs["rattle_seed"] == 42

    @pytest.mark.unit
    def test_rattle_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(f"{_MODULE}.create_initialization_atoms_list") as mock_create:
            mock_create.return_value = []
            generate(
                self._config(),
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["num_of_rattled_per_target"] == 0
        assert call_kwargs["rattle_standard_deviation"] is None

    @pytest.mark.unit
    def test_rattle_enabled_reads_its_own_namespace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = self._config(
            rattle_target_structures={
                "enabled": True,
                "num_of_rattled_per_target": 9,
                "rattle_standard_deviation": 0.03,
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
        assert call_kwargs["num_of_rattled_per_target"] == 9
        assert call_kwargs["rattle_standard_deviation"] == 0.03

    @pytest.mark.unit
    def test_rattle_enabled_without_standard_deviation_raises(
        self, tmp_path, monkeypatch
    ):
        """rattle_standard_deviation has no default -- turning rattle on
        without setting it must raise a clear error, not silently guess."""
        monkeypatch.chdir(tmp_path)
        config = self._config(
            rattle_target_structures={"enabled": True},
        )
        with (
            patch(f"{_MODULE}.create_initialization_atoms_list"),
            pytest.raises(KeyError, match="rattle_standard_deviation"),
        ):
            generate(
                config,
                base_name="al_loop_0",
                name="initialization",
                elements=["H"],
                hpc={},
                max_time="1H",
            )

    @pytest.mark.unit
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
        # Empty/zero means "none still needed", not "no override" (None
        # would regenerate the full set).
        assert call_kwargs["trimer_override"] == {}
        assert call_kwargs["amorphous_override"] == 0
        assert call_kwargs["mp_structures"] is False
        assert call_kwargs["single_atoms"] is True  # bool(["H"])

    @pytest.mark.unit
    def test_complete_kinds_are_not_regenerated(self, tmp_path, monkeypatch):
        """Dimers and trimers complete, 3 amorphous cells missing: only those
        3 are made (an empty override used to regenerate every dimer and
        trimer, sending them to DFT again)."""
        from collections import Counter

        monkeypatch.chdir(tmp_path)
        needs = {
            "isolated_atoms": [],
            "dimer_override": {},
            "trimer_override": {},
            "amorphous_override": 3,
            "mp_structures": False,
            "molecule_override": {},
            "box_override": {},
        }
        out = generate(
            {
                "mp_kwargs": {"enabled": False},
                "stretch_compress_targets_kwargs": {"enabled": False},
                "amorphous_kwargs": {"num_of_atoms_per_amorphous": 4},
            },
            base_name="initialization",
            name="initialization",
            elements=["C", "O"],
            needs=needs,
            target_config_types=[],
        )
        assert Counter(a.info["config_type"] for a in out) == {"init_amorphous": 3}

    @pytest.mark.unit
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


# --- validate_kwargs: molecule blocks and density ranges -------------------

_CHO = ["C", "H", "O"]
_BOX = {
    "molecules": {"H2O": 2, "smiles-CO": 1},
    "density_range": [0.8, 1.1],
    "num_of_molecules": 12,
    "num_of_structures": 2,
}


@pytest.mark.unit
def test_validate_accepts_the_defaults():
    validate_kwargs({}, elements=_CHO)


@pytest.mark.unit
def test_enabled_molecule_kwargs_without_molecules_raises():
    with pytest.raises(ValueError, match=r"molecule_kwargs\.molecules is required"):
        validate_kwargs({"molecule_kwargs": {"enabled": True}}, elements=_CHO)


@pytest.mark.unit
def test_molecule_kwargs_molecule_outside_general_elements_raises():
    config = {"molecule_kwargs": {"enabled": True, "molecules": ["smiles-CCO"]}}
    with pytest.raises(ValueError, match=r"contains \['C'\]"):
        validate_kwargs(config, elements=["H", "O"])


@pytest.mark.unit
def test_enabled_box_kwargs_without_compositions_raises():
    with pytest.raises(ValueError, match="box_compositions is required"):
        validate_kwargs({"molecules_in_box_kwargs": {"enabled": True}}, elements=_CHO)


@pytest.mark.unit
def test_valid_box_composition_passes():
    config = {"molecules_in_box_kwargs": {"enabled": True, "box_compositions": [_BOX]}}
    validate_kwargs(config, elements=_CHO)


@pytest.mark.unit
@pytest.mark.parametrize(
    "change, message",
    [
        ({"density_range": [1.2, 0.8]}, "density_range must be"),
        ({"density_range": [0.0, 1.0]}, "density_range must be"),
        ({"density_range": 1.0}, "density_range must be"),
        ({"molecules": {"H2O": 0}}, "positive ratios"),
        ({"num_of_molecules": 1}, "at least the number of molecules"),
        ({"num_of_structures": 0}, "num_of_structures must be"),
        ({"densities": [1.0]}, "unknown keys"),
    ],
)
def test_bad_box_composition_raises(change, message):
    composition = {**_BOX, **change}
    config = {
        "molecules_in_box_kwargs": {"enabled": True, "box_compositions": [composition]}
    }
    with pytest.raises(ValueError, match=message):
        validate_kwargs(config, elements=_CHO)


@pytest.mark.unit
def test_bad_amorphous_density_range_raises():
    with pytest.raises(ValueError, match=r"amorphous_kwargs\.density_range"):
        validate_kwargs({"amorphous_kwargs": {"density_range": [2, 1]}}, elements=_CHO)


@pytest.mark.unit
def test_densities_list_is_deprecated_and_read_as_its_range():
    import logging

    records = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    module_logger = logging.getLogger(_MODULE)
    module_logger.addHandler(handler)
    config = {"amorphous_kwargs": {"densities_list": [1.2, 0.8, 1.0]}}
    try:
        validate_kwargs(config, elements=_CHO)
    finally:
        module_logger.removeHandler(handler)
    assert [r.event for r in records] == ["config_key_deprecated"]
    resolved = _resolve_kwargs(config, "amorphous_kwargs")
    assert resolved["density_range"] == [0.8, 1.2]
    assert "densities_list" not in resolved


@pytest.mark.unit
def test_generate_passes_molecule_blocks_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "isolated_atom_kwargs": {"enabled": False},
        "dimer_kwargs": {"enabled": False},
        "trimer_kwargs": {"enabled": False},
        "amorphous_kwargs": {"enabled": False},
        "mp_kwargs": {"enabled": False},
        "molecule_kwargs": {"enabled": True, "molecules": ["CO"]},
        "molecules_in_box_kwargs": {"enabled": True, "box_compositions": [_BOX]},
    }
    atoms = generate(config, base_name="init", name="initialization", elements=_CHO)
    types = [a.info["config_type"] for a in atoms]
    assert types.count("init_molecule") == 1
    assert types.count("init_molecules_in_box") == 2


# --- Density warnings (vdW volume ratio, the blown-up filter's yardstick) ---


def _density_warnings(config, elements, **kwargs):
    import logging

    from alomancy.initialize import initialiser_interface

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    initialiser_interface.logger.addHandler(handler)
    try:
        validate_kwargs(config, elements=elements, **kwargs)
    finally:
        initialiser_interface.logger.removeHandler(handler)
    return {
        r.event: r.getMessage()  # type: ignore[attr-defined]
        for r in records
        if getattr(r, "event", "").startswith("init_density")
    }


@pytest.mark.unit
def test_default_density_warns_too_low_for_pt():
    """1 g/cm^3 Pt is 14x its vdW volume: the blown-up filter (10) would
    drop every amorphous cell before DFT."""
    warnings = _density_warnings({}, ["Pt"], max_vdw_volume_ratio=10.0)
    assert set(warnings) == {"init_density_too_low"}
    message = warnings["init_density_too_low"]
    assert "Pt at 1 g/cm^3" in message
    assert "init_amorphous" in message


@pytest.mark.unit
def test_bulk_like_density_is_quiet():
    config = {"amorphous_kwargs": {"density_range": [15.0, 21.0]}}
    assert _density_warnings(config, ["Pt"], max_vdw_volume_ratio=10.0) == {}


@pytest.mark.unit
def test_denser_than_any_solid_warns_too_high():
    config = {"amorphous_kwargs": {"density_range": [3.0, 10.0]}}
    warnings = _density_warnings(config, ["C"], max_vdw_volume_ratio=10.0)
    assert set(warnings) == {"init_density_too_high"}
    assert "C at 10 g/cm^3" in warnings["init_density_too_high"]


@pytest.mark.unit
def test_hydrogen_random_mixes_are_checked_as_an_even_mix():
    """A pure-H cell at 1 g/cm^3 is far too dense, but random mixes of H and
    O are checked as their typical cell, so H configs don't always warn."""
    assert _density_warnings({}, ["H", "O"], max_vdw_volume_ratio=10.0) == {}


@pytest.mark.unit
def test_sparse_molecule_box_warns_too_low():
    config = {
        "amorphous_kwargs": {"enabled": False},
        "molecules_in_box_kwargs": {
            "enabled": True,
            "box_compositions": [
                {
                    "molecules": {"CO": 1},
                    "density_range": [0.05, 0.05],
                    "num_of_molecules": 8,
                    "num_of_structures": 1,
                }
            ],
        },
    }
    warnings = _density_warnings(config, ["C", "O"], max_vdw_volume_ratio=10.0)
    assert set(warnings) == {"init_density_too_low"}
    assert "CO:1 at 0.05 g/cm^3" in warnings["init_density_too_low"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_vdw_volume_ratio": None},
        {
            "max_vdw_volume_ratio": 10.0,
            "vdw_exempt_config_types": frozenset({"init_amorphous"}),
        },
        {},
    ],
)
def test_no_low_warning_when_the_filter_wont_drop_them(kwargs):
    assert _density_warnings({}, ["Pt"], **kwargs) == {}


@pytest.mark.unit
def test_amorphous_seed_is_deprecated_and_ignored(tmp_path, monkeypatch):
    """general.seed seeds the amorphous cells: amorphous_kwargs.seed warns
    and changes nothing."""
    import logging

    from alomancy.initialize import initialiser_interface

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    initialiser_interface.logger.addHandler(handler)
    try:
        validate_kwargs({"amorphous_kwargs": {"seed": 1}}, elements=["C", "O"])
    finally:
        initialiser_interface.logger.removeHandler(handler)
    assert any(
        getattr(r, "event", None) == "config_key_deprecated"
        and "amorphous_kwargs.seed" in r.getMessage()
        for r in records
    )

    monkeypatch.chdir(tmp_path)
    base = {
        "mp_kwargs": {"enabled": False},
        "dimer_kwargs": {"enabled": False},
        "trimer_kwargs": {"enabled": False},
        "isolated_atom_kwargs": {"enabled": False},
        "stretch_compress_targets_kwargs": {"enabled": False},
    }

    def cells(amorphous, seed):
        return generate(
            {
                **base,
                "amorphous_kwargs": {
                    "num_of_amorphous_structures": 2,
                    "num_of_atoms_per_amorphous": 4,
                    "density_range": [1.0, 2.0],
                    **amorphous,
                },
            },
            base_name="initialization",
            name="initialization",
            elements=["C", "O"],
            target_config_types=[],
            seed=seed,
        )

    default = cells({}, 803)
    assert all(
        np.allclose(a.positions, b.positions)
        for a, b in zip(default, cells({"seed": 1}, 803), strict=True)
    )
    assert not np.allclose(cells({}, 7)[0].positions, default[0].positions)
