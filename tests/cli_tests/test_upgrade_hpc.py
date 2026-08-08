"""Tests for alomancy.cli.upgrade_hpc."""

from unittest import mock

import pytest


class TestParseHpcSelection:
    @pytest.mark.unit
    def test_all_selects_everything(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        assert _parse_hpc_selection("all", ["raven", "viper"]) == ["raven", "viper"]

    @pytest.mark.unit
    def test_all_case_insensitive(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        assert _parse_hpc_selection("ALL", ["raven", "viper"]) == ["raven", "viper"]

    @pytest.mark.unit
    def test_single_index(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        assert _parse_hpc_selection("2", ["raven", "viper"]) == ["viper"]

    @pytest.mark.unit
    def test_multiple_indices_with_whitespace(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        assert _parse_hpc_selection("1, 3", ["raven", "viper", "raccoon"]) == [
            "raven",
            "raccoon",
        ]

    @pytest.mark.unit
    def test_duplicate_indices_deduplicated_preserving_order(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        assert _parse_hpc_selection("2,1,2", ["raven", "viper"]) == ["viper", "raven"]

    @pytest.mark.unit
    def test_out_of_range_raises(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        with pytest.raises(ValueError, match="between 1 and 2"):
            _parse_hpc_selection("5", ["raven", "viper"])

    @pytest.mark.unit
    def test_non_numeric_raises(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        with pytest.raises(ValueError, match="not a number"):
            _parse_hpc_selection("raven", ["raven", "viper"])

    @pytest.mark.unit
    def test_empty_string_raises(self):
        from alomancy.cli.upgrade_hpc import _parse_hpc_selection

        with pytest.raises(ValueError, match="No selection"):
            _parse_hpc_selection("", ["raven", "viper"])


class TestUpgradeOneHost:
    @pytest.mark.unit
    def test_host_resolution_failure(self):
        from alomancy.cli.upgrade_hpc import _upgrade_one_host

        with mock.patch("alomancy.cli.upgrade_hpc.resolve_hpc_host", return_value=None):
            name, success, reason = _upgrade_one_host(
                "raven",
                {"hpc_name": "raven", "pre_cmds": ["source /v/bin/activate"]},
                timeout=5.0,
            )

        assert name == "raven"
        assert success is False
        assert "ssh host" in reason

    @pytest.mark.unit
    def test_python_path_derivation_failure(self):
        from alomancy.cli.upgrade_hpc import _upgrade_one_host

        with mock.patch(
            "alomancy.cli.upgrade_hpc.resolve_hpc_host", return_value="raven-alias"
        ):
            name, success, reason = _upgrade_one_host(
                "raven",
                {"hpc_name": "raven", "pre_cmds": ["module load python"]},
                timeout=5.0,
            )

        assert name == "raven"
        assert success is False
        assert "python path" in reason

    @pytest.mark.unit
    def test_ssh_success(self):
        from alomancy.cli.upgrade_hpc import _upgrade_one_host

        with (
            mock.patch(
                "alomancy.cli.upgrade_hpc.resolve_hpc_host", return_value="raven-alias"
            ),
            mock.patch(
                "alomancy.cli.upgrade_hpc.derive_python_from_venv",
                return_value="/v/bin/python",
            ),
            mock.patch(
                "alomancy.cli.upgrade_hpc.run_ssh_command",
                return_value=(True, "Successfully installed alomancy\n", ""),
            ) as mock_run,
        ):
            name, success, reason = _upgrade_one_host(
                "raven",
                {"hpc_name": "raven", "pre_cmds": ["source /v/bin/activate"]},
                timeout=42.0,
            )

        assert (name, success, reason) == ("raven", True, "")
        args, kwargs = mock_run.call_args
        assert args[0] == "raven-alias"
        assert "pip install --upgrade alomancy" in args[1]
        assert kwargs["timeout"] == 42.0

    @pytest.mark.unit
    def test_ssh_failure(self):
        from alomancy.cli.upgrade_hpc import _upgrade_one_host

        with (
            mock.patch(
                "alomancy.cli.upgrade_hpc.resolve_hpc_host", return_value="raven-alias"
            ),
            mock.patch(
                "alomancy.cli.upgrade_hpc.derive_python_from_venv",
                return_value="/v/bin/python",
            ),
            mock.patch(
                "alomancy.cli.upgrade_hpc.run_ssh_command",
                return_value=(False, "", "permission denied"),
            ),
        ):
            name, success, reason = _upgrade_one_host(
                "raven",
                {"hpc_name": "raven", "pre_cmds": ["source /v/bin/activate"]},
                timeout=5.0,
            )

        assert (name, success) == ("raven", False)
        assert reason == "permission denied"


class TestUpgradeHpcWizard:
    @pytest.mark.unit
    def test_no_profiles_configured(self, monkeypatch, capsys):
        from alomancy.cli.upgrade_hpc import upgrade_hpc_wizard

        with mock.patch(
            "alomancy.cli.upgrade_hpc._load_global_hpc_config", return_value={}
        ):
            upgrade_hpc_wizard()

        assert "alomancy add-hpc" in capsys.readouterr().out

    @pytest.mark.unit
    def test_happy_path_single_profile(self, monkeypatch, capsys):
        from alomancy.cli.upgrade_hpc import upgrade_hpc_wizard

        responses = iter(["1", "y"])
        monkeypatch.setattr("builtins.input", lambda *_a: next(responses))

        with (
            mock.patch(
                "alomancy.cli.upgrade_hpc._load_global_hpc_config",
                return_value={"raven": {"hpc_name": "raven", "pre_cmds": []}},
            ),
            mock.patch(
                "alomancy.cli.upgrade_hpc._upgrade_one_host",
                return_value=("raven", True, ""),
            ) as mock_upgrade,
        ):
            upgrade_hpc_wizard()

        mock_upgrade.assert_called_once()
        assert "1 succeeded, 0 failed" in capsys.readouterr().out

    @pytest.mark.unit
    def test_declining_confirmation_aborts(self, monkeypatch, capsys):
        from alomancy.cli.upgrade_hpc import upgrade_hpc_wizard

        responses = iter(["all", "n"])
        monkeypatch.setattr("builtins.input", lambda *_a: next(responses))

        with (
            mock.patch(
                "alomancy.cli.upgrade_hpc._load_global_hpc_config",
                return_value={"raven": {"hpc_name": "raven", "pre_cmds": []}},
            ),
            mock.patch("alomancy.cli.upgrade_hpc._upgrade_one_host") as mock_upgrade,
        ):
            upgrade_hpc_wizard()

        mock_upgrade.assert_not_called()
        assert "Aborted" in capsys.readouterr().out


@pytest.mark.unit
def test_cli_entrypoint_upgrade_hpc():
    from alomancy.cli.main import main

    with (
        mock.patch("sys.argv", ["alomancy", "upgrade-hpc"]),
        mock.patch("alomancy.cli.upgrade_hpc.upgrade_hpc_wizard") as m_upgrade,
    ):
        main()

    m_upgrade.assert_called_once_with()
