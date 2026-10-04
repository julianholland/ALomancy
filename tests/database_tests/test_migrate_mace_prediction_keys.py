"""Tests for database/migrate_mace_prediction_keys.py's
rename_legacy_prediction_sentinels() and the combined migrate() entry
point. The GlobalDatabase metadata-key half of the migration is covered by
TestMigrateMacePredictionKeys in test_global_database.py; these tests
cover the per-loop sentinel-file rename and the two pieces wired together.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alomancy.database.global_database import GlobalDatabase
from alomancy.database.migrate_mace_prediction_keys import (
    migrate,
    rename_legacy_prediction_sentinels,
)


def _make_dimer(ref_energy: float = -1.0) -> Atoms:
    """Minimal, storable (has a cell) test Atoms object -- matching
    test_global_database.py's make_atoms helper, copied here per this
    test suite's existing convention of not cross-importing between test
    modules."""
    atoms = Atoms(
        symbols=["H", "H"], positions=np.eye(2, 3) * 2.0, cell=[10.0] * 3, pbc=True
    )
    atoms.info["config_type"] = "init_dimer"
    atoms.info["REF_energy"] = ref_energy
    atoms.arrays["REF_forces"] = np.zeros((2, 3))
    return atoms


@pytest.mark.unit
class TestRenameLegacyPredictionSentinels:
    def test_renames_legacy_sentinel(self, tmp_path):
        loop_dir = tmp_path / "results" / "al_loop_0"
        loop_dir.mkdir(parents=True)
        (loop_dir / "mace_predictions.done").touch()

        renamed = rename_legacy_prediction_sentinels(tmp_path / "results")

        assert renamed == 1
        assert not (loop_dir / "mace_predictions.done").exists()
        assert (loop_dir / "model_predictions.done").exists()

    def test_renames_across_multiple_loops(self, tmp_path):
        for i in range(3):
            loop_dir = tmp_path / "results" / f"al_loop_{i}"
            loop_dir.mkdir(parents=True)
            (loop_dir / "mace_predictions.done").touch()

        renamed = rename_legacy_prediction_sentinels(tmp_path / "results")

        assert renamed == 3
        for i in range(3):
            loop_dir = tmp_path / "results" / f"al_loop_{i}"
            assert (loop_dir / "model_predictions.done").exists()

    def test_leaves_both_in_place_when_current_already_exists(self, tmp_path):
        """Never overwrite an already-migrated (or freshly created) current
        sentinel with a stale legacy one -- just leave both and warn."""
        loop_dir = tmp_path / "results" / "al_loop_0"
        loop_dir.mkdir(parents=True)
        (loop_dir / "mace_predictions.done").touch()
        (loop_dir / "model_predictions.done").touch()

        renamed = rename_legacy_prediction_sentinels(tmp_path / "results")

        assert renamed == 0
        assert (loop_dir / "mace_predictions.done").exists()
        assert (loop_dir / "model_predictions.done").exists()

    def test_no_op_when_nothing_legacy_present(self, tmp_path):
        (tmp_path / "results").mkdir()
        assert rename_legacy_prediction_sentinels(tmp_path / "results") == 0

    def test_idempotent_second_call_renames_nothing(self, tmp_path):
        loop_dir = tmp_path / "results" / "al_loop_0"
        loop_dir.mkdir(parents=True)
        (loop_dir / "mace_predictions.done").touch()

        rename_legacy_prediction_sentinels(tmp_path / "results")
        assert rename_legacy_prediction_sentinels(tmp_path / "results") == 0


@pytest.mark.unit
class TestMigrate:
    def test_runs_both_pieces_and_reports_counts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        atoms = _make_dimer()

        db = GlobalDatabase("results/global_database")
        db.add_structures([atoms], split="train", skip_duplicates=False)
        db.assign_global_db_ids()
        db.partition.set_metadata_bulk(
            {0: {"mace_energy_loop_0_fit_0": -1.5}}, use_indices=True
        )

        loop_dir = Path("results/al_loop_0")
        loop_dir.mkdir(parents=True)
        (loop_dir / "mace_predictions.done").touch()

        result = migrate(results_dir="results", db_path="results/global_database")

        assert result == {"containers_updated": 1, "sentinels_renamed": 1}
        assert (loop_dir / "model_predictions.done").exists()
