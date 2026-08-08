"""Tests for alomancy/__init__.py's per-run ExPyRe isolation setup."""

import json

import pytest

from alomancy import _seed_local_expyre_root


@pytest.mark.unit
class TestSeedLocalExpyreRoot:
    def test_copies_canonical_master_into_run_local_expyre(self, tmp_path, monkeypatch):
        import alomancy as alomancy_module

        master = tmp_path / "master" / "expyre_config.json"
        master.parent.mkdir()
        master.write_text(json.dumps({"systems": {"raven": {"host": "raven"}}}))
        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", master)

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _seed_local_expyre_root(run_dir)

        copied = run_dir / ".expyre" / "config.json"
        assert copied.exists()
        assert json.loads(copied.read_text()) == {
            "systems": {"raven": {"host": "raven"}}
        }

    def test_falls_back_to_legacy_path_when_canonical_missing(
        self, tmp_path, monkeypatch
    ):
        import alomancy as alomancy_module

        missing_master = tmp_path / "does_not_exist" / "expyre_config.json"
        legacy = tmp_path / "legacy" / "config.json"
        legacy.parent.mkdir()
        legacy.write_text(json.dumps({"systems": {"old": {"host": "old"}}}))
        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", missing_master)
        monkeypatch.setattr(alomancy_module, "LEGACY_EXPYRE_CONFIG", legacy)

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _seed_local_expyre_root(run_dir)

        copied = run_dir / ".expyre" / "config.json"
        assert json.loads(copied.read_text()) == {"systems": {"old": {"host": "old"}}}

    def test_noop_when_neither_config_source_exists(self, tmp_path, monkeypatch):
        import alomancy as alomancy_module

        monkeypatch.setattr(
            alomancy_module, "EXPYRE_CONFIG", tmp_path / "nope" / "expyre_config.json"
        )
        monkeypatch.setattr(
            alomancy_module, "LEGACY_EXPYRE_CONFIG", tmp_path / "nope2" / "config.json"
        )

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _seed_local_expyre_root(run_dir)

        assert not (run_dir / ".expyre").exists()

    def test_noop_when_run_local_expyre_already_exists(self, tmp_path, monkeypatch):
        """Idempotent: never overwrites an existing run-local .expyre, so a
        rerun (or a user's deliberately customized one) is left alone."""
        import alomancy as alomancy_module

        master = tmp_path / "expyre_config.json"
        master.write_text(json.dumps({"systems": {"new": {"host": "new"}}}))
        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", master)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        existing = run_dir / ".expyre"
        existing.mkdir()
        (existing / "config.json").write_text(
            json.dumps({"systems": {"custom": {"host": "custom"}}})
        )

        _seed_local_expyre_root(run_dir)

        assert json.loads((existing / "config.json").read_text()) == {
            "systems": {"custom": {"host": "custom"}}
        }

    def test_noop_when_run_local_underscore_expyre_already_exists(
        self, tmp_path, monkeypatch
    ):
        import alomancy as alomancy_module

        master = tmp_path / "expyre_config.json"
        master.write_text(json.dumps({"systems": {}}))
        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", master)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "_expyre").mkdir()

        _seed_local_expyre_root(run_dir)

        assert not (run_dir / ".expyre").exists()

    def test_survives_copy_failure(self, tmp_path, monkeypatch):
        """A best-effort setup step must never raise -- package import
        cannot hard-fail over a transient filesystem issue."""
        import shutil

        import alomancy as alomancy_module

        master = tmp_path / "expyre_config.json"
        master.write_text(json.dumps({"systems": {}}))
        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", master)

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(shutil, "copyfile", _boom)

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _seed_local_expyre_root(run_dir)  # must not raise
