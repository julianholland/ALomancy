"""Tests for utils/seed_selection.py -- the two responsibilities split out
of structure_generation.select_initial_structures.select_initial_structures
(still in place unchanged, still what current production
generate_structures calls) for the modular AL architecture: eligibility
filtering (called once by the skeleton before any generator) and
diversity-maximizing selection (called internally by generators that can
only run from a bounded set of individually-submitted seeds, e.g. MD)."""

import numpy as np
import pytest
from ase import Atoms

from alomancy.utils.seed_selection import (
    filter_eligible_structures,
    mark_structures_for_dft,
    select_diverse_seeds,
)


def _make_atoms(symbols, config_type="train"):
    n = len(symbols)
    atoms = Atoms(
        symbols=symbols, positions=np.eye(n, 3) * 2, cell=[10, 10, 10], pbc=True
    )
    atoms.info["config_type"] = config_type
    return atoms


@pytest.mark.unit
class TestFilterEligibleStructures:
    def test_formula_filter(self):
        structures = [_make_atoms(["H", "H"]) for _ in range(5)] + [
            _make_atoms(["O", "O"]) for _ in range(5)
        ]
        result = filter_eligible_structures(structures, chem_formula_list=["H2"])
        assert len(result) == 5
        assert all(a.get_chemical_formula() == "H2" for a in result)

    def test_atom_number_range_filter(self):
        structures = (
            [_make_atoms(["H"]) for _ in range(5)]
            + [_make_atoms(["H", "O"]) for _ in range(10)]
            + [_make_atoms(["H", "O", "O"]) for _ in range(5)]
        )
        result = filter_eligible_structures(structures, atom_number_range=(2, 2))
        assert len(result) == 10
        assert all(len(a) == 2 for a in result)

    def test_config_type_filter(self):
        structures = [
            _make_atoms(["H", "H"], config_type="al_loop_0") for _ in range(10)
        ] + [_make_atoms(["H", "H"], config_type="init_dimer") for _ in range(5)]
        result = filter_eligible_structures(
            structures, selectable_configs=["al_loop_0"]
        )
        assert len(result) == 10
        assert all(a.info["config_type"] == "al_loop_0" for a in result)

    def test_high_sd_auto_included_when_selectable_configs_set(self):
        init_structures = [
            _make_atoms(["H", "H"], config_type="init_amorphous") for _ in range(5)
        ]
        high_sd_structures = [
            _make_atoms(["H", "H"], config_type="high_sd") for _ in range(5)
        ]
        result = filter_eligible_structures(
            init_structures + high_sd_structures,
            selectable_configs=["init_amorphous"],
        )
        assert len(result) == 10

    def test_caller_selectable_configs_list_not_mutated(self):
        structures = [_make_atoms(["H", "H"], config_type="init_amorphous")]
        caller_list = ["init_amorphous"]
        filter_eligible_structures(structures, selectable_configs=caller_list)
        assert caller_list == ["init_amorphous"]

    def test_raises_when_nothing_eligible(self):
        structures = [_make_atoms(["H", "H"])]
        with pytest.raises(ValueError, match="No structures available"):
            filter_eligible_structures(structures, chem_formula_list=["O2"])

    def test_no_filters_returns_everything(self):
        structures = [_make_atoms(["H", "H"]) for _ in range(3)]
        result = filter_eligible_structures(structures)
        assert len(result) == 3

    def test_warns_on_atom_number_range_allowing_single_atoms(self):
        structures = [_make_atoms(["H"]) for _ in range(3)]
        with pytest.warns(UserWarning, match="single-atom"):
            filter_eligible_structures(structures, atom_number_range=(0, 5))

    def test_raises_on_inverted_atom_number_range(self):
        structures = [_make_atoms(["H", "H"])]
        with pytest.raises(ValueError, match="less than or equal"):
            filter_eligible_structures(structures, atom_number_range=(5, 2))


@pytest.mark.unit
class TestSelectDiverseSeeds:
    def test_returns_requested_count(self):
        structures = [_make_atoms(["H", "H"]) for _ in range(10)]
        result = select_diverse_seeds(
            base_name="test",
            job_name="md",
            eligible_structures=structures,
            max_number_of_concurrent_jobs=3,
        )
        assert len(result) == 3

    def test_reuses_with_replacement_when_pool_smaller_than_requested(self):
        structures = [_make_atoms(["H", "H"]) for _ in range(2)]
        result = select_diverse_seeds(
            base_name="test",
            job_name="md",
            eligible_structures=structures,
            max_number_of_concurrent_jobs=5,
        )
        assert len(result) == 5

    def test_assigns_distinct_md_seeds(self):
        structures = [_make_atoms(["H", "H"]) for _ in range(5)]
        result = select_diverse_seeds(
            base_name="test",
            job_name="md",
            eligible_structures=structures,
            max_number_of_concurrent_jobs=5,
            seed=803,
        )
        md_seeds = [a.info["md_seed"] for a in result]
        assert md_seeds == [803, 804, 805, 806, 807]

    def test_marks_config_type_and_job_id(self):
        structures = [
            _make_atoms(["H", "H"], config_type="al_loop_0") for _ in range(3)
        ]
        result = select_diverse_seeds(
            base_name="al_loop_1",
            job_name="md",
            eligible_structures=structures,
            max_number_of_concurrent_jobs=3,
        )
        assert all(a.info["config_type"] == "al_loop_1_md" for a in result)
        assert all("job_id" in a.info for a in result)

    def test_originals_not_mutated(self):
        structures = [
            _make_atoms(["H", "H"], config_type="al_loop_0") for _ in range(3)
        ]
        select_diverse_seeds(
            base_name="al_loop_1",
            job_name="md",
            eligible_structures=structures,
            max_number_of_concurrent_jobs=3,
        )
        assert all(a.info["config_type"] == "al_loop_0" for a in structures)

    def test_chemical_diversity_selects_unique_formulas(self):
        structures = [_make_atoms(["H", "H"]) for _ in range(5)] + [
            _make_atoms(["O", "O"]) for _ in range(5)
        ]
        result = select_diverse_seeds(
            base_name="test",
            job_name="md",
            eligible_structures=structures,
            max_number_of_concurrent_jobs=2,
            enforce_chemical_diversity=True,
        )
        assert {a.get_chemical_formula() for a in result} == {"H2", "O2"}


@pytest.mark.unit
class TestMarkStructuresForDft:
    def test_sets_config_type_and_job_id(self):
        structures = [_make_atoms(["H"]) for _ in range(2)]
        mark_structures_for_dft(structures, "al_loop_0", "md")
        assert all(a.info["config_type"] == "al_loop_0_md" for a in structures)
        assert all("job_id" in a.info for a in structures)
