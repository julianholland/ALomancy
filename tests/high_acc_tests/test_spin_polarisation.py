"""high_accuracy_evaluation.spin_polarisation: "essential" (default),
"all", "none", for every evaluator (high_accuracy_evaluation/spin.py)."""

import io

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io.espresso import write_espresso_in

from alomancy.high_accuracy_evaluation import spin
from alomancy.high_accuracy_evaluation.dft import run_aims, run_qe, run_vasp
from alomancy.high_accuracy_evaluation.orca import run_orca


def _atom() -> Atoms:
    return Atoms("C", cell=[10, 11, 12], info={"config_type": "IsolatedAtom"})


def _dimer() -> Atoms:
    return Atoms(
        "C2",
        positions=[[0, 0, 0], [0, 0, 1.6]],
        cell=[10, 11, 12],
        info={"config_type": "init_dimer"},
    )


def _bulk(magmom: float = 0.0) -> Atoms:
    atoms = bulk("C", "diamond", a=3.57)
    atoms.info["config_type"] = "init_MP"
    if magmom:
        atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    return atoms


def _qe_spin_lines(atoms: Atoms, mode: str) -> dict[str, str]:
    input_data = run_qe.get_qe_input_data("scf", {})
    run_qe.apply_spin_polarisation(atoms, input_data, mode)
    out = io.StringIO()
    write_espresso_in(
        out,
        atoms,
        input_data=input_data,
        pseudopotentials={"C": "C.UPF"},
        kpts=(1, 1, 1),
    )
    lines = {}
    for line in out.getvalue().splitlines():
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            lines[key] = value
    return lines


@pytest.mark.unit
class TestPolicy:
    def test_default_is_essential(self):
        assert spin.resolve_spin_polarisation({}) == "essential"
        assert spin.resolve_spin_polarisation(None) == "essential"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            spin.resolve_spin_polarisation({"spin_polarisation": "some"})

    @pytest.mark.parametrize(
        ("atoms", "mode", "expected"),
        [
            (_atom(), "essential", True),
            (_dimer(), "essential", True),
            (_bulk(), "essential", None),
            (_bulk(), "all", True),
            (_atom(), "none", False),
        ],
    )
    def test_which_structures(self, atoms, mode, expected):
        assert spin.spin_polarised(atoms, mode) is expected

    def test_hund_moments(self):
        assert spin.hund_moment(_atom()) == 2.0
        assert spin.hund_moment(Atoms("H")) == 1.0
        assert spin.hund_moment(_dimer()) is None

    def test_own_moments_win_over_hund(self):
        assert np.allclose(spin.starting_moments(_bulk(magmom=0.7)), 0.7)
        assert np.allclose(spin.starting_moments(_dimer()), [2.0, 2.0])


@pytest.mark.unit
class TestQE:
    def test_essential_fixes_the_atom_and_frees_the_dimer(self):
        atom = _qe_spin_lines(_atom(), "essential")
        assert atom["nspin"] == "2"
        assert atom["tot_magnetization"] == "2.0"
        assert atom["starting_magnetization(1)"] == "0.5"
        dimer = _qe_spin_lines(_dimer(), "essential")
        assert dimer["nspin"] == "2"
        assert "tot_magnetization" not in dimer
        assert dimer["starting_magnetization(1)"] == "0.5"

    def test_essential_leaves_bulk_restricted(self):
        assert _qe_spin_lines(_bulk(), "essential")["nspin"] == "1"

    def test_none_clears_moments(self):
        """A magnetic start would otherwise make ASE switch nspin to 2."""
        lines = _qe_spin_lines(_bulk(magmom=1.0), "none")
        assert lines["nspin"] == "1"
        assert not any(k.startswith("starting_magnetization") for k in lines)

    def test_all_polarises_bulk(self):
        assert _qe_spin_lines(_bulk(), "all")["nspin"] == "2"

    def test_starting_magnetization_stays_a_fraction(self):
        for atoms in (_atom(), _dimer(), _bulk()):
            lines = _qe_spin_lines(atoms, "all")
            for key, value in lines.items():
                if key.startswith("starting_magnetization"):
                    assert -1.0 <= float(value) <= 1.0

    @pytest.mark.parametrize(
        "key", ["nspin", "starting_magnetization(1)", "tot_magnetization"]
    )
    def test_spin_keywords_are_owned(self, key):
        with pytest.raises(ValueError, match="spin_polarisation"):
            run_qe.validate_kwargs({"qe_kwargs": {"system": {key: 2}}}, None)

    def test_level_of_theory_follows_the_mode(self):
        ids = {
            mode: run_qe.level_of_theory({"spin_polarisation": mode})
            for mode in spin.SPIN_POLARISATION_MODES
        }
        assert ids["none"].spin_treatment == "restricted"
        assert ids["all"].spin_treatment == "collinear"
        assert ids["essential"].spin_treatment == "essential"
        assert len({level.identity_id() for level in ids.values()}) == 3


@pytest.mark.unit
class TestVASP:
    def test_essential_atom_gets_nupdown(self):
        atoms, kwargs = _atom(), {}
        run_vasp.apply_spin_polarisation(atoms, kwargs, "essential")
        assert kwargs == {"ispin": 2, "nupdown": 2.0}
        assert np.allclose(atoms.get_initial_magnetic_moments(), [2.0])

    def test_dimer_and_bulk(self):
        dimer, kwargs = _dimer(), {}
        run_vasp.apply_spin_polarisation(dimer, kwargs, "essential")
        assert kwargs == {"ispin": 2}
        bulk_kwargs: dict = {}
        run_vasp.apply_spin_polarisation(_bulk(), bulk_kwargs, "essential")
        assert bulk_kwargs == {}
        none_kwargs: dict = {}
        run_vasp.apply_spin_polarisation(_atom(), none_kwargs, "none")
        assert none_kwargs == {"ispin": 1}

    @pytest.mark.parametrize("key", ["ISPIN", "magmom", "nupdown"])
    def test_spin_keywords_are_owned(self, key):
        with pytest.raises(ValueError, match="spin_polarisation"):
            run_vasp.validate_kwargs({"vasp_kwargs": {key: 2}}, None)


@pytest.mark.unit
class TestAims:
    def test_essential_atom_gets_a_fixed_moment(self):
        atoms, params = _atom(), {}
        run_aims.apply_spin_polarisation(atoms, params, "essential")
        assert params == {"spin": "collinear", "fixed_spin_moment": 2.0}

    def test_bulk_follows_its_own_moments(self):
        params: dict = {}
        run_aims.apply_spin_polarisation(_bulk(), params, "essential")
        assert params == {"spin": "none"}
        magnetic: dict = {}
        run_aims.apply_spin_polarisation(_bulk(magmom=1.0), magnetic, "essential")
        assert magnetic == {"spin": "collinear"}

    @pytest.mark.parametrize("key", ["spin", "default_initial_moment"])
    def test_spin_keywords_are_rejected(self, key):
        with pytest.raises(ValueError, match="spin_polarisation"):
            run_aims.validate_kwargs({"aims_kwargs": {key: "collinear"}}, None)


@pytest.mark.unit
class TestOrca:
    def test_none_gives_atoms_the_lowest_multiplicity(self):
        atom = Atoms("C")
        assert run_orca.structure_charge_and_multiplicity(atom) == (0, 3)
        assert run_orca.structure_charge_and_multiplicity(atom, hund_atoms=False) == (
            0,
            1,
        )
