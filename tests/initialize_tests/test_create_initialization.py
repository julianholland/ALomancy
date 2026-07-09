"""Tests for create_initialization_atoms_list."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
from ase import Atoms


def _make_atoms(symbol="H"):
    a = Atoms(symbol)
    a.cell = [5, 5, 5]
    a.pbc = True
    return a


def _all_patches(
    mp_return=None,
    single_return=None,
    dimer_return=None,
    trimer_return=None,
    amorphous_return=None,
    sc_return=None,
):
    """Return a list of (target, kwargs) pairs for patching all sub-generators."""
    return [
        (
            "alomancy.initialize.initialization_structure_list.atoms_list_from_mp",
            {"return_value": mp_return if mp_return is not None else []},
        ),
        (
            "alomancy.initialize.initialization_structure_list.create_single_atoms_list",
            {
                "return_value": single_return
                if single_return is not None
                else [_make_atoms()]
            },
        ),
        (
            "alomancy.initialize.initialization_structure_list.create_dimer_atoms_list",
            {
                "return_value": dimer_return
                if dimer_return is not None
                else [_make_atoms("H2")]
            },
        ),
        (
            "alomancy.initialize.initialization_structure_list.create_trimer_atoms_list",
            {
                "return_value": trimer_return
                if trimer_return is not None
                else [_make_atoms("H3")]
            },
        ),
        (
            "alomancy.initialize.initialization_structure_list.create_amorphous_atoms_list",
            {
                "return_value": amorphous_return
                if amorphous_return is not None
                else [_make_atoms()]
            },
        ),
        (
            "alomancy.initialize.initialization_structure_list.create_stretch_compress_atoms_list",
            {"return_value": sc_return if sc_return is not None else []},
        ),
    ]


def _enter_patches(stack, patches):
    """Enter all patches via ExitStack and return list of mocks."""
    return [stack.enter_context(patch(target, **kw)) for target, kw in patches]


@pytest.mark.unit
class TestCreateInitializationAtomsList:
    def test_empty_elements_raises(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with pytest.raises(AssertionError):
            create_initialization_atoms_list(work_dir=str(tmp_path), elements=[])

    def test_skips_mp_when_mp_structures_false(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(stack, _all_patches())
            mock_mp = mocks[0]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                num_dimers_per_combo=1,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        mock_mp.assert_not_called()

    def test_calls_mp_when_mp_structures_true(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        mp_atom = _make_atoms("NaCl")
        mp_atom.info["config_type"] = "init_MP"

        with ExitStack() as stack:
            mocks = _enter_patches(stack, _all_patches(mp_return=[mp_atom]))
            mock_mp = mocks[0]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=True,
                single_atoms=False,
                num_dimers_per_combo=1,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        mock_mp.assert_called_once()

    def test_skips_singles_when_single_atoms_false(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(stack, _all_patches())
            mock_single = mocks[1]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                num_dimers_per_combo=1,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        mock_single.assert_not_called()

    def test_isolated_atoms_override_restricts_elements(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(
                stack, _all_patches(single_return=[_make_atoms("O")])
            )
            mock_single = mocks[1]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H", "O"],
                mp_structures=False,
                single_atoms=True,
                isolated_atoms_override=["O"],
                num_dimers_per_combo=1,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        elements_called = [c.kwargs["element"] for c in mock_single.call_args_list]
        assert elements_called == ["O"]

    def test_no_singles_when_override_empty(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(stack, _all_patches())
            mock_single = mocks[1]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=True,
                isolated_atoms_override=[],
                num_dimers_per_combo=1,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        mock_single.assert_not_called()

    def test_dimer_override_zero_skips_combo(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(stack, _all_patches(dimer_return=[]))
            mock_dimer = mocks[2]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                dimer_override={"H2": 0},
                num_dimers_per_combo=5,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        mock_dimer.assert_not_called()

    def test_dimer_override_count_passed_to_generator(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(
                stack, _all_patches(dimer_return=[_make_atoms("H2")])
            )
            mock_dimer = mocks[2]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                dimer_override={"H2": 3},
                num_dimers_per_combo=10,
                num_trimers_per_combo=1,
                num_amorphous=1,
            )

        assert mock_dimer.call_count == 1
        assert mock_dimer.call_args.kwargs["num_dimers"] == 3

    def test_trimer_override_zero_skips_combo(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(stack, _all_patches(trimer_return=[]))
            mock_trimer = mocks[3]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                num_dimers_per_combo=0,
                trimer_override={"H3": 0},
                num_trimers_per_combo=5,
                num_amorphous=1,
            )

        mock_trimer.assert_not_called()

    def test_amorphous_override_controls_count(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(
                stack,
                _all_patches(amorphous_return=[_make_atoms()]),
            )
            mock_amorphous = mocks[4]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                num_dimers_per_combo=0,
                num_trimers_per_combo=0,
                num_amorphous=100,
                amorphous_override=20,
            )

        assert mock_amorphous.call_count == 1
        assert mock_amorphous.call_args.kwargs["num_structures"] == 20

    def test_writes_output_file(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            _enter_patches(
                stack,
                _all_patches(
                    dimer_return=[_make_atoms("H2")],
                    trimer_return=[],
                    amorphous_return=[_make_atoms()],
                ),
            )
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                num_dimers_per_combo=1,
                num_trimers_per_combo=0,
                num_amorphous=1,
            )

        assert (tmp_path / "initialization_structures_generated.xyz").exists()

    def test_stretch_compress_called_per_mp_structure(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        mp1 = _make_atoms("NaCl")
        mp2 = _make_atoms("MgO")

        with ExitStack() as stack:
            mocks = _enter_patches(
                stack,
                _all_patches(
                    mp_return=[mp1, mp2],
                    single_return=[],
                    dimer_return=[],
                    trimer_return=[],
                    amorphous_return=[],
                    sc_return=[_make_atoms()],
                ),
            )
            mock_sc = mocks[5]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["Na", "Cl"],
                mp_structures=True,
                single_atoms=False,
                num_dimers_per_combo=0,
                num_trimers_per_combo=0,
                num_amorphous=0,
                num_stretch_compress_per_mp=3,
            )

        assert mock_sc.call_count == 2
        assert mock_sc.call_args_list[0].kwargs["num_structures"] == 3

    def test_stretch_compress_skipped_when_mp_false(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(
                stack,
                _all_patches(
                    single_return=[],
                    dimer_return=[],
                    trimer_return=[],
                    amorphous_return=[],
                ),
            )
            mock_sc = mocks[5]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                num_dimers_per_combo=0,
                num_trimers_per_combo=0,
                num_amorphous=0,
            )

        mock_sc.assert_not_called()

    def test_result_includes_all_enabled_components(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        single = _make_atoms("H")
        dimer = _make_atoms("H2")
        amorphous = _make_atoms("H3")

        with ExitStack() as stack:
            _enter_patches(
                stack,
                _all_patches(
                    single_return=[single],
                    dimer_return=[dimer],
                    trimer_return=[],
                    amorphous_return=[amorphous],
                ),
            )
            result = create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=True,
                num_dimers_per_combo=1,
                num_trimers_per_combo=0,
                num_amorphous=1,
            )

        # single + dimer + amorphous = 3
        assert len(result) == 3

    def test_multiple_densities_calls_amorphous_per_density(self, tmp_path):
        from alomancy.initialize.initialization_structure_list import (
            create_initialization_atoms_list,
        )

        with ExitStack() as stack:
            mocks = _enter_patches(
                stack,
                _all_patches(
                    single_return=[],
                    dimer_return=[],
                    trimer_return=[],
                    amorphous_return=[],
                ),
            )
            mock_amorphous = mocks[4]
            create_initialization_atoms_list(
                work_dir=str(tmp_path),
                elements=["H"],
                mp_structures=False,
                single_atoms=False,
                num_dimers_per_combo=0,
                num_trimers_per_combo=0,
                num_amorphous=10,
                densities_list=[0.8, 1.2],
            )

        assert mock_amorphous.call_count == 2
        densities_used = [c.kwargs["density"] for c in mock_amorphous.call_args_list]
        assert 0.8 in densities_used
        assert 1.2 in densities_used
