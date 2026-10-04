"""Tests for utils/training_schedule.py: the shared epoch rule."""

import pytest

from alomancy.utils.training_schedule import dynamic_epochs, resolve_epochs


@pytest.mark.unit
class TestDynamicEpochs:
    def test_typical_mid_run_value_is_capped(self):
        assert dynamic_epochs(batch_size=16, n_structures=4000) == 300

    def test_large_training_set_hits_floor(self):
        assert dynamic_epochs(batch_size=16, n_structures=1_000_000) == 20

    def test_uncapped_value_between_floor_and_cap(self):
        assert dynamic_epochs(batch_size=16, n_structures=20_000) == 160

    def test_batch_size_scales_epochs(self):
        assert dynamic_epochs(batch_size=8, n_structures=20_000) == 80

    def test_raises_on_zero_structures(self):
        with pytest.raises(ValueError):
            dynamic_epochs(batch_size=16, n_structures=0)

    def test_custom_bounds(self):
        assert dynamic_epochs(16, 4000, cap=1000) == 800


@pytest.mark.unit
class TestResolveEpochs:
    @pytest.mark.parametrize("configured", [None, "dynamic"])
    def test_unset_or_dynamic_uses_the_rule(self, configured):
        assert resolve_epochs(configured, 16, 20_000) == 160

    def test_integer_passes_through(self):
        assert resolve_epochs(55, 16, 20_000) == 55

    @pytest.mark.parametrize("bad", [0, -3, 2.5, "fast", True])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError, match="max_num_epochs"):
            resolve_epochs(bad, 16, 100)
