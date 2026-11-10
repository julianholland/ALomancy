"""Tests for the ORCA backend (high_accuracy_evaluation/orca/run_orca.py).

No ORCA binary is needed: the calculator is built and its input written
with ASE's own OrcaTemplate, which is what a real job would run.
"""

import logging
import re

import pytest
from ase import Atoms
from ase.build import molecule

from alomancy.high_accuracy_evaluation.orca import run_orca
from alomancy.high_accuracy_evaluation.orca.run_orca import (
    build_orca_input,
    classify_orca_method,
    create_orca_calc_object,
    level_of_theory,
    max_num_of_atoms,
    parse_orca_simple_input,
    structure_charge_and_multiplicity,
    validate_kwargs,
)

NODE_INFO = {
    "ranks_per_system": 8,
    "ranks_per_node": 16,
    "threads_per_rank": 1,
    "max_mem_per_node": "64GB",
}
HPC = {
    "hpc_name": "test-hpc",
    "orca_path": "/opt/orca/orca",
    "node_info": NODE_INFO,
}


def _job(orca_kwargs: dict) -> dict:
    return {"name": "high_accuracy_evaluation", "hpc": HPC, "orca_kwargs": orca_kwargs}


def _written_input(atoms: Atoms, orca_kwargs: dict, tmp_path) -> str:
    calc = create_orca_calc_object(atoms, _job(orca_kwargs), str(tmp_path))
    calc.template.write_input(
        profile=calc.profile,
        directory=tmp_path,
        atoms=atoms,
        parameters=calc.parameters,
        properties=["energy", "forces"],
    )
    return (tmp_path / "orca.inp").read_text()


@pytest.mark.unit
def test_input_has_resources_from_the_hpc_profile(tmp_path):
    text = _written_input(
        molecule("H2O"), {"orcasimpleinput": "B3LYP def2-SVP"}, tmp_path
    )
    assert text.count("%pal") == 1
    assert "%pal nprocs 8 end" in text
    # 75% of 64 GB over 8 processes.
    assert "%maxcore 6144" in text
    assert "EnGrad" in text and "SCFConvergenceForced" in text
    assert re.search(r"\*xyz 0 1", text)


@pytest.mark.unit
@pytest.mark.parametrize(
    "orca_kwargs",
    [
        {"orcasimpleinput": "B3LYP def2-SVP", "nprocs": 4},
        {"orcasimpleinput": "B3LYP def2-SVP", "orcablocks": "%pal nprocs 4 end"},
        {"orcasimpleinput": "B3LYP def2-SVP PAL4"},
        # The same value in two places is fine.
        {"orcasimpleinput": "B3LYP def2-SVP PAL4", "nprocs": 4},
    ],
)
def test_user_nprocs_overrides_the_profile(orca_kwargs):
    built = build_orca_input(orca_kwargs, NODE_INFO)
    assert built["orcablocks"].count("%pal") == 1
    assert "%pal nprocs 4 end" in built["orcablocks"]
    assert "PAL4" not in built["orcasimpleinput"]
    assert "%maxcore 12288" in built["orcablocks"]


@pytest.mark.unit
def test_user_maxcore_and_other_blocks_are_kept():
    built = build_orca_input(
        {
            "orcasimpleinput": "B3LYP def2-SVP",
            "orcablocks": "%maxcore 2000\n%scf maxiter 300 end",
        },
        NODE_INFO,
    )
    assert built["orcablocks"].count("%maxcore") == 1
    assert "%maxcore 2000" in built["orcablocks"]
    assert "%scf maxiter 300 end" in built["orcablocks"]


@pytest.mark.unit
def test_conflicting_nprocs_raise():
    with pytest.raises(ValueError, match="more than once"):
        build_orca_input(
            {"orcasimpleinput": "B3LYP def2-SVP PAL4", "nprocs": 8}, NODE_INFO
        )


@pytest.mark.unit
def test_nprocs_larger_than_the_node_raise():
    with pytest.raises(ValueError, match="doesn't fit the job"):
        build_orca_input({"orcasimpleinput": "B3LYP def2-SVP", "nprocs": 32}, NODE_INFO)


@pytest.mark.unit
def test_memory_beyond_the_node_raises():
    with pytest.raises(ValueError, match="more than the node"):
        build_orca_input(
            {"orcasimpleinput": "B3LYP def2-SVP", "maxcore": 20000}, NODE_INFO
        )


@pytest.mark.unit
def test_periodic_structures_raise(tmp_path):
    atoms = molecule("H2O")
    atoms.cell = [10, 10, 10]
    atoms.pbc = True
    with pytest.raises(ValueError, match="no periodic boundary conditions"):
        create_orca_calc_object(
            atoms, _job({"orcasimpleinput": "B3LYP def2-SVP"}), str(tmp_path)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("atoms", "expected"),
    [
        (Atoms("H"), (0, 2)),
        (Atoms("O"), (0, 3)),
        (Atoms("N"), (0, 4)),
        (Atoms("He"), (0, 1)),
        (molecule("H2O"), (0, 1)),
        (molecule("CH3"), (0, 2)),
    ],
)
def test_charge_and_multiplicity(atoms, expected):
    assert structure_charge_and_multiplicity(atoms) == expected


@pytest.mark.unit
def test_multiplicity_and_charge_from_info():
    atoms = molecule("O2")
    atoms.info["spin_multiplicity"] = 3
    assert structure_charge_and_multiplicity(atoms) == (0, 3)
    atoms = molecule("H2O")
    atoms.info["charge"] = 1
    assert structure_charge_and_multiplicity(atoms) == (1, 2)


@pytest.mark.unit
def test_impossible_multiplicity_raises():
    atoms = molecule("H2O")
    atoms.info["spin_multiplicity"] = 2
    with pytest.raises(ValueError, match="impossible"):
        structure_charge_and_multiplicity(atoms)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("simpleinput", "blocks", "expected"),
    [
        ("B3LYP def2-TZVP", "", "mean_field"),
        ("r2SCAN-3c", "", "mean_field"),
        ("HF def2-SVP", "", "mean_field"),
        ("PBE0 D3BJ def2-SVP RIJCOSX def2/J", "", "mean_field"),
        ("RI-MP2 def2-TZVP def2-TZVP/C", "", "correlated"),
        ("DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C TightPNO", "", "correlated"),
        ("B2PLYP def2-TZVP", "", "correlated"),
        ("CASSCF(6,6) def2-SVP", "", "multireference"),
        ("def2-SVP", "%casscf nel 6 norb 6 end", "multireference"),
        ("GFN2-xTB", "", "semi_empirical"),
        ("M11 def2-SVP", "", None),
    ],
)
def test_classify_orca_method(simpleinput, blocks, expected):
    assert classify_orca_method(simpleinput, blocks) == expected


@pytest.mark.unit
def test_parse_sorts_the_keywords():
    parsed = parse_orca_simple_input(
        "! PBE0 D3BJ def2-TZVP RIJCOSX def2/J ZORA CPCM(water) TightSCF DefGrid3"
    )
    assert parsed.method == "PBE0" and parsed.rung == "hybrid"
    assert parsed.dispersion == "D3(BJ)"
    assert parsed.basis == "def2-TZVP"
    assert parsed.relativity == "scalar_ZORA"
    assert parsed.solvation == "CPCM(water)"
    assert parsed.approximations == ["def2/j", "rijcosx"]
    assert parsed.scf_threshold == "TightSCF" and parsed.grid == "DefGrid3"


@pytest.mark.unit
def test_atom_cap_applies_to_correlated_methods_only():
    correlated = {"orca_kwargs": {"orcasimpleinput": "RI-MP2 def2-TZVP def2-TZVP/C"}}
    assert max_num_of_atoms(correlated) == 20
    correlated["orca_kwargs"]["max_num_of_atoms_correlated"] = None
    assert max_num_of_atoms(correlated) is None
    dft = {"orca_kwargs": {"orcasimpleinput": "B3LYP def2-TZVP"}}
    assert max_num_of_atoms(dft) is None


@pytest.mark.unit
def test_level_of_theory():
    level = level_of_theory(
        {"orca_kwargs": {"orcasimpleinput": "DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C"}}
    )
    assert level.code == "orca"
    assert level.method_class == "correlated"
    assert level.method == "DLPNO-CCSD(T)"
    assert level.basis_type == "gaussian" and level.basis == "cc-pVTZ"
    assert level.core_treatment == "all_electron"
    # def2 bases bring ECPs beyond Kr.
    assert (
        level_of_theory(
            {"orca_kwargs": {"orcasimpleinput": "B3LYP def2-SVP"}}
        ).core_treatment
        == "ecp"
    )


@pytest.mark.unit
def test_validate_kwargs_rejects_unknown_keys_and_missing_path():
    with pytest.raises(ValueError, match="unknown keys"):
        validate_kwargs({"orca_kwargs": {"orcasimpleinput": "HF", "pal": 4}}, HPC)
    with pytest.raises(ValueError, match="no orca_path"):
        validate_kwargs(
            {"orca_kwargs": {"orcasimpleinput": "HF def2-SVP"}},
            {**HPC, "orca_path": None},
        )
    with pytest.raises(ValueError, match="orcasimpleinput is required"):
        validate_kwargs({"orca_kwargs": {}}, HPC)


@pytest.mark.unit
def test_validate_kwargs_warns_for_numerical_gradients_and_unknown_methods():
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    run_orca.logger.addHandler(handler)
    try:
        validate_kwargs(
            {"orca_kwargs": {"orcasimpleinput": "DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C"}},
            HPC,
        )
        validate_kwargs({"orca_kwargs": {"orcasimpleinput": "M11 def2-SVP"}}, HPC)
    finally:
        run_orca.logger.removeHandler(handler)
    events = [getattr(r, "event", None) for r in records]
    assert "numerical_gradients" in events
    assert "orca_method_unrecognised" in events


@pytest.mark.unit
def test_charge_and_multiplicity_survive_an_extxyz_round_trip(tmp_path):
    """extxyz reads info integers back as np.int64, which an isinstance(int)
    check rejected: every charged or open-shell structure from start_from,
    the DB or a restart failed on the node."""
    from ase.io import read, write

    atoms = molecule("H2O")
    atoms.info["charge"] = -1
    atoms.info["spin_multiplicity"] = 2
    write(tmp_path / "a.xyz", atoms, format="extxyz")
    back = read(tmp_path / "a.xyz", format="extxyz")
    assert not isinstance(back.info["charge"], int)  # numpy, as in a real run
    assert structure_charge_and_multiplicity(back) == (-1, 2)
    atoms.info["charge"] = True
    with pytest.raises(ValueError, match="integer charge"):
        structure_charge_and_multiplicity(atoms)
