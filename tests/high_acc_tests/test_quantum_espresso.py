"""
Tests for Quantum Espresso utility functions.

This module tests pure utility functions (find_optimal_npool, generate_kpts, get_qe_input_data)
that require no external dependencies or QE installation.
"""

import pytest
import numpy as np
from alomancy.high_accuracy_evaluation.dft.run_qe import (
    find_optimal_npool, generate_kpts, get_qe_input_data)


class TestFindOptimalNpool:
    @pytest.mark.unit
    def test_returns_1_when_no_valid_candidate(self):
        # With min_ranks_per_pool=4 and ranks_per_system=2, no valid npool
        result = find_optimal_npool(ranks_per_system=2, total_kpoints=10, min_ranks_per_pool=4)
        assert result == 1

    @pytest.mark.unit
    def test_valid_npool_divides_ranks(self):
        # npool must divide ranks_per_system evenly
        result = find_optimal_npool(ranks_per_system=16, total_kpoints=4)
        assert 16 % result == 0

    @pytest.mark.unit
    def test_npool_does_not_exceed_total_kpoints(self):
        result = find_optimal_npool(ranks_per_system=16, total_kpoints=4)
        assert result <= 4

    @pytest.mark.unit
    def test_kpoints_divisible_npool_preferred(self):
        # npool that evenly divides total_kpoints gets score +3
        result = find_optimal_npool(ranks_per_system=16, total_kpoints=8)
        assert 8 % result == 0  # preferred npool divides kpoints

    @pytest.mark.unit
    def test_ranks_per_node_aligns_pools(self):
        # With ranks_per_node matching, pools_per_node is integer -> score +2
        result = find_optimal_npool(ranks_per_system=16, total_kpoints=8, ranks_per_node=16)
        assert isinstance(result, int)
        assert result >= 1

    @pytest.mark.unit
    def test_min_ranks_per_pool_respected(self):
        result = find_optimal_npool(ranks_per_system=8, total_kpoints=8, min_ranks_per_pool=4)
        # ranks_per_pool = ranks_per_system / npool >= min_ranks_per_pool
        assert 8 // result >= 4


class TestGenerateKpts:
    @pytest.mark.unit
    def test_cubic_cell_kpoints(self):
        cell = np.eye(3) * 5.0  # 5 Angstrom cubic cell
        kpts = generate_kpts(cell, periodic_3d=True, kspacing=0.1)
        assert kpts.shape == (3,)
        assert all(k >= 1 for k in kpts)

    @pytest.mark.unit
    def test_larger_cell_fewer_kpoints(self):
        cell_small = np.eye(3) * 3.0
        cell_large = np.eye(3) * 10.0
        k_small = generate_kpts(cell_small, kspacing=0.1)
        k_large = generate_kpts(cell_large, kspacing=0.1)
        # Larger real-space cell -> smaller k-space -> fewer kpoints
        assert all(k_large[i] <= k_small[i] for i in range(3))

    @pytest.mark.unit
    def test_non_periodic_z_2d(self):
        cell = np.eye(3) * 5.0
        kpts = generate_kpts(cell, periodic_3d=False, kspacing=0.1)
        assert kpts[2] == 1  # z direction: 1 kpoint for 2D

    @pytest.mark.unit
    def test_kspacing_effect(self):
        cell = np.eye(3) * 5.0
        kpts_dense = generate_kpts(cell, kspacing=0.05)
        kpts_sparse = generate_kpts(cell, kspacing=0.2)
        # Smaller kspacing -> more kpoints
        assert all(kpts_dense[i] >= kpts_sparse[i] for i in range(3))


class TestGetQeInputData:
    @pytest.mark.unit
    def test_returns_all_required_sections(self):
        result = get_qe_input_data("scf", {})
        assert "control" in result
        assert "system" in result
        assert "electrons" in result
        assert "ions" in result
        assert "cell" in result

    @pytest.mark.unit
    def test_calculation_type_set_in_control(self):
        result = get_qe_input_data("vc-relax", {})
        assert result["control"]["calculation"] == "vc-relax"

    @pytest.mark.unit
    def test_sp_calculation_type(self):
        result = get_qe_input_data("scf", {})
        assert result["control"]["calculation"] == "scf"

    @pytest.mark.unit
    def test_extra_kwargs_merged(self):
        extra = {"system": {"ecutwfc": 80.0}}
        result = get_qe_input_data("scf", extra)
        # The extra system dict should override the default ecutwfc
        assert result["system"]["ecutwfc"] == 80.0

    @pytest.mark.unit
    def test_default_ecutwfc(self):
        result = get_qe_input_data("scf", {})
        assert result["system"]["ecutwfc"] == 40.0

    @pytest.mark.unit
    def test_default_conv_thr(self):
        result = get_qe_input_data("scf", {})
        assert result["electrons"]["conv_thr"] == pytest.approx(1.0e-12)
