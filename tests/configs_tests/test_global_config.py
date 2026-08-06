"""Tests for alomancy.configs.global_config."""

import pytest


@pytest.mark.unit
class TestExpyreConfigConstants:
    def test_expyre_config_lives_under_alomancy_dir(self):
        from alomancy.configs.global_config import ALOMANCY_DIR, EXPYRE_CONFIG

        assert EXPYRE_CONFIG.parent == ALOMANCY_DIR
        assert EXPYRE_CONFIG.name == "expyre_config.json"

    def test_legacy_expyre_config_is_the_pre_isolation_path(self):
        from pathlib import Path

        from alomancy.configs.global_config import LEGACY_EXPYRE_CONFIG

        assert Path.home() / ".expyre" / "config.json" == LEGACY_EXPYRE_CONFIG
