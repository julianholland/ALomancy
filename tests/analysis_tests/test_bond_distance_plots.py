"""Unit tests for bond_distance_plots helpers."""

from unittest.mock import MagicMock

import pytest
from ase import Atoms


class TestPairwiseDistancesByElementPair:
    @pytest.mark.unit
    def test_buckets_by_sorted_element_pair(self):
        from alomancy.analysis.bond_distance_plots import (
            _pairwise_distances_by_element_pair,
        )

        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        result = _pairwise_distances_by_element_pair([atoms], max_distance=5.0)

        assert set(result.keys()) == {("H", "O"), ("H", "H")}
        assert len(result[("H", "O")]) == 2
        assert len(result[("H", "H")]) == 1

    @pytest.mark.unit
    def test_excludes_distances_beyond_max_distance(self):
        from alomancy.analysis.bond_distance_plots import (
            _pairwise_distances_by_element_pair,
        )

        atoms = Atoms(
            symbols=["H", "H"],
            positions=[[0, 0, 0], [3.0, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        result = _pairwise_distances_by_element_pair([atoms], max_distance=1.0)

        assert result == {}

    @pytest.mark.unit
    def test_single_atom_structure_contributes_nothing(self):
        from alomancy.analysis.bond_distance_plots import (
            _pairwise_distances_by_element_pair,
        )

        atoms = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        result = _pairwise_distances_by_element_pair([atoms], max_distance=5.0)

        assert result == {}

    @pytest.mark.unit
    def test_accumulates_across_multiple_structures(self):
        from alomancy.analysis.bond_distance_plots import (
            _pairwise_distances_by_element_pair,
        )

        a1 = Atoms(
            symbols=["H", "H"], positions=[[0, 0, 0], [1.0, 0, 0]], cell=[10, 10, 10]
        )
        a2 = Atoms(
            symbols=["H", "H"], positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10, 10, 10]
        )
        result = _pairwise_distances_by_element_pair([a1, a2], max_distance=5.0)

        assert sorted(result[("H", "H")]) == [1.0, 1.5]

    @pytest.mark.unit
    def test_non_periodic_structure_supported(self):
        """Dimers/trimers from initialization are pbc=False -- must not raise."""
        from alomancy.analysis.bond_distance_plots import (
            _pairwise_distances_by_element_pair,
        )

        atoms = Atoms(symbols=["C", "O"], positions=[[0, 0, 0], [1.2, 0, 0]])
        result = _pairwise_distances_by_element_pair([atoms], max_distance=5.0)

        assert result[("C", "O")] == pytest.approx([1.2])


class TestPlotTrainingBondDistances:
    @pytest.mark.unit
    def test_creates_plot_file(self, tmp_path):
        from alomancy.analysis.bond_distance_plots import plot_training_bond_distances

        atoms_list = [
            Atoms(
                symbols=["H", "H"],
                positions=[[0, 0, 0], [1.0, 0, 0]],
                cell=[10, 10, 10],
            )
            for _ in range(5)
        ]
        db = MagicMock()
        db.get_train_atoms.return_value = atoms_list

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        plot_training_bond_distances("demo", db, plots_dir)

        db.get_train_atoms.assert_called_once_with(
            exclude_duplicates=True, exclude_high_force=True
        )
        assert (plots_dir / "train_bond_distances_demo.png").exists()

    @pytest.mark.unit
    def test_no_training_structures_skips_gracefully(self, tmp_path):
        from alomancy.analysis.bond_distance_plots import plot_training_bond_distances

        db = MagicMock()
        db.get_train_atoms.return_value = []

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        plot_training_bond_distances("demo", db, plots_dir)

        assert not (plots_dir / "train_bond_distances_demo.png").exists()

    @pytest.mark.unit
    def test_no_distances_within_range_skips_gracefully(self, tmp_path):
        """Only single-atom structures in the DB -- nothing to plot."""
        from alomancy.analysis.bond_distance_plots import plot_training_bond_distances

        db = MagicMock()
        db.get_train_atoms.return_value = [
            Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[5, 5, 5])
        ]

        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        plot_training_bond_distances("demo", db, plots_dir)

        assert not (plots_dir / "train_bond_distances_demo.png").exists()
