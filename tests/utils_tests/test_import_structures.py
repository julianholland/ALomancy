"""Tests for utils/import_structures.py -- label normalization for warm
starts from xyz files (general.start_from)."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from alomancy.utils.import_structures import (
    EXTERNAL_CONFIG_TYPE,
    normalize_metadata,
    read_structures,
)


def _bare(i: int = 0) -> Atoms:
    return Atoms(
        "H2", positions=[[0, 0, 0], [0.7 + 0.1 * i, 0, 0]], cell=[5] * 3, pbc=True
    )


@pytest.mark.unit
def test_alomancy_labels_pass_through_unchanged():
    a = _bare()
    a.info.update(config_type="init_dimer", REF_energy=-2.0)
    a.arrays["REF_forces"] = np.ones((2, 3))

    (out,) = normalize_metadata([a])

    assert out.info["config_type"] == "init_dimer"
    assert out.info["REF_energy"] == -2.0
    np.testing.assert_array_equal(out.arrays["REF_forces"], np.ones((2, 3)))


@pytest.mark.unit
def test_aliases_auto_detected():
    a = _bare()
    a.info.update(type="bulk", dft_energy=-3.0, dft_stress=[0.1] * 6)
    a.arrays["dft_forces"] = np.full((2, 3), 0.5)

    (out,) = normalize_metadata([a])

    assert out.info["config_type"] == "bulk"
    assert out.info["REF_energy"] == -3.0
    np.testing.assert_array_equal(out.arrays["REF_forces"], np.full((2, 3), 0.5))
    np.testing.assert_allclose(out.info["REF_stresses"], [0.1] * 6)


@pytest.mark.unit
def test_metadata_map_takes_precedence_over_aliases():
    a = _bare()
    a.info.update(type="wrong", phase="liquid", energy_pbe=-4.0, energy=-99.0)
    a.arrays["f_pbe"] = np.zeros((2, 3))

    (out,) = normalize_metadata(
        [a], {"config_type": "phase", "energy": "energy_pbe", "forces": "f_pbe"}
    )

    assert out.info["config_type"] == "liquid"
    assert out.info["REF_energy"] == -4.0


@pytest.mark.unit
def test_calculator_labels_used_when_no_keys(tmp_path):
    """extxyz moves bare energy/forces keys into a calculator on read."""
    a = _bare()
    a.calc = SinglePointCalculator(
        a, energy=-5.0, forces=np.full((2, 3), 0.2), stress=np.zeros(6)
    )
    write(tmp_path / "calc.xyz", [a], format="extxyz")

    (out,) = normalize_metadata(read_structures(tmp_path / "calc.xyz"))

    assert out.info["REF_energy"] == pytest.approx(-5.0)
    np.testing.assert_allclose(out.arrays["REF_forces"], np.full((2, 3), 0.2))
    assert out.calc is None


@pytest.mark.unit
def test_missing_config_type_becomes_external():
    a = _bare()
    a.info["REF_energy"] = -1.0
    a.arrays["REF_forces"] = np.zeros((2, 3))

    (out,) = normalize_metadata([a])

    assert out.info["config_type"] == EXTERNAL_CONFIG_TYPE


@pytest.mark.unit
def test_unresolvable_energy_raises_before_import():
    labelled = _bare(0)
    labelled.info["REF_energy"] = -1.0
    labelled.arrays["REF_forces"] = np.zeros((2, 3))
    unlabelled = _bare(1)

    with pytest.raises(ValueError, match="metadata_map") as exc_info:
        normalize_metadata([labelled, unlabelled], source="data.xyz")
    assert "1 structure(s) (energy)" in str(exc_info.value)
    assert "data.xyz" in str(exc_info.value)


@pytest.mark.unit
def test_missing_stress_is_not_an_error():
    a = _bare()
    a.info["REF_energy"] = -1.0
    a.arrays["REF_forces"] = np.zeros((2, 3))

    (out,) = normalize_metadata([a])

    assert "REF_stresses" not in out.info


@pytest.mark.unit
def test_operational_metadata_from_writing_run_is_dropped():
    a = _bare()
    a.info.update(
        REF_energy=-1.0,
        config_type="high_sd",
        split="test",
        global_db_id=7,
        is_duplicate=True,
        model_energy_loop_0_fit_0=-1.1,
        al_loop=3,
    )
    a.arrays["REF_forces"] = np.zeros((2, 3))

    (out,) = normalize_metadata([a])

    for key in ("split", "global_db_id", "is_duplicate", "model_energy_loop_0_fit_0"):
        assert key not in out.info
    assert out.info["al_loop"] == 3  # provenance is kept


@pytest.mark.unit
def test_unknown_metadata_map_key_raises():
    with pytest.raises(ValueError, match="Unknown metadata_map"):
        normalize_metadata([_bare()], {"enrgy": "e"})


@pytest.mark.unit
def test_input_structures_not_mutated():
    a = _bare()
    a.info.update(dft_energy=-3.0, split="train")
    a.arrays["dft_forces"] = np.zeros((2, 3))

    normalize_metadata([a])

    assert "REF_energy" not in a.info
    assert a.info["split"] == "train"


@pytest.mark.unit
def test_read_structures_reads_every_frame(tmp_path):
    write(tmp_path / "many.xyz", [_bare(i) for i in range(3)], format="extxyz")
    assert len(read_structures(tmp_path / "many.xyz")) == 3


@pytest.mark.unit
def test_read_structures_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_structures(tmp_path / "nope.xyz")
