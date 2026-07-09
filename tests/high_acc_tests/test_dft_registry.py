"""Tests for the DFT calculator registry and mismatch warnings."""

import logging

import pytest

from alomancy.high_accuracy_evaluation.dft import (
    get_dft_functions,
    warn_mismatched_kwargs,
)


class TestGetDftFunctions:
    @pytest.mark.unit
    def test_qe_returns_correct_callables(self):
        sp, go = get_dft_functions("qe")
        assert callable(sp)
        assert callable(go)
        assert sp.__name__ == "run_sp_qe"
        assert go.__name__ == "run_go_qe"

    @pytest.mark.unit
    def test_vasp_returns_correct_callables(self):
        sp, go = get_dft_functions("vasp")
        assert callable(sp)
        assert callable(go)
        assert sp.__name__ == "run_sp_vasp"
        assert go.__name__ == "run_go_vasp"

    @pytest.mark.unit
    def test_unknown_calculator_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown calculator"):
            get_dft_functions("notabackend")

    @pytest.mark.unit
    def test_sp_and_go_are_different_functions(self):
        sp, go = get_dft_functions("qe")
        assert sp is not go


class TestWarnMismatchedKwargs:
    @pytest.mark.unit
    def test_vasp_kwargs_with_qe_calculator_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alomancy"):
            warn_mismatched_kwargs("qe", {"vasp_input_kwargs": {"encut": 500}})
        assert any("vasp_input_kwargs" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_qe_kwargs_with_vasp_calculator_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alomancy"):
            warn_mismatched_kwargs("vasp", {"qe_input_kwargs": {"ecutwfc": 40}})
        assert any("qe_input_kwargs" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_matching_kwargs_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alomancy"):
            warn_mismatched_kwargs("qe", {"qe_input_kwargs": {"ecutwfc": 40}})
        assert not any("looks like a" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_hpc_vasp_key_with_qe_calculator_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alomancy"):
            warn_mismatched_kwargs("qe", {"hpc": {"vasp_path": "/path/to/vasp"}})
        assert any("vasp_path" in r.message for r in caplog.records)
