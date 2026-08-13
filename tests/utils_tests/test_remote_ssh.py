"""Tests for alomancy.utils.remote_ssh."""

import json
import logging
import subprocess
from unittest.mock import patch

import pytest


class TestRunSshCommand:
    @pytest.mark.unit
    def test_success(self):
        from alomancy.utils.remote_ssh import run_ssh_command

        completed = subprocess.CompletedProcess(
            args=["ssh", "h", "cmd"], returncode=0, stdout="ok\n", stderr=""
        )
        with patch("subprocess.run", return_value=completed) as mock_run:
            success, stdout, stderr = run_ssh_command("h", "cmd", timeout=5.0)

        assert (success, stdout, stderr) == (True, "ok\n", "")
        args, kwargs = mock_run.call_args
        assert args[0] == ["ssh", "h", "cmd"]
        assert kwargs["timeout"] == 5.0

    @pytest.mark.unit
    def test_nonzero_exit(self):
        from alomancy.utils.remote_ssh import run_ssh_command

        completed = subprocess.CompletedProcess(
            args=["ssh", "h", "cmd"], returncode=1, stdout="", stderr="boom\n"
        )
        with patch("subprocess.run", return_value=completed):
            success, _stdout, stderr = run_ssh_command("h", "cmd", timeout=5.0)

        assert success is False
        assert stderr == "boom\n"

    @pytest.mark.unit
    def test_timeout_expired(self):
        from alomancy.utils.remote_ssh import run_ssh_command

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=5.0),
        ):
            success, stdout, stderr = run_ssh_command("h", "cmd", timeout=5.0)

        assert success is False
        assert stdout == ""
        assert "timed out" in stderr.lower() or "5" in stderr

    @pytest.mark.unit
    def test_os_error(self):
        from alomancy.utils.remote_ssh import run_ssh_command

        with patch("subprocess.run", side_effect=OSError("no ssh binary")):
            success, _stdout, stderr = run_ssh_command("h", "cmd", timeout=5.0)

        assert success is False
        assert "no ssh binary" in stderr


class TestDerivePythonFromVenv:
    @pytest.mark.unit
    def test_matches_activate_command(self):
        from alomancy.utils.remote_ssh import derive_python_from_venv

        result = derive_python_from_venv("source /u/user/.venvs/alomancy/bin/activate")
        assert result == "/u/user/.venvs/alomancy/bin/python"

    @pytest.mark.unit
    def test_non_matching_returns_none(self):
        from alomancy.utils.remote_ssh import derive_python_from_venv

        assert derive_python_from_venv("module load python/3.11") is None

    @pytest.mark.unit
    def test_empty_string_returns_none(self):
        from alomancy.utils.remote_ssh import derive_python_from_venv

        assert derive_python_from_venv("") is None


class TestResolveHpcHost:
    @pytest.mark.unit
    def test_valid_lookup(self, tmp_path):
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"systems": {"raven": {"host": "raven-alias"}}}))
        assert resolve_hpc_host("raven", expyre_config_path=cfg) == "raven-alias"

    @pytest.mark.unit
    def test_missing_system_returns_none(self, tmp_path):
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"systems": {"raven": {"host": "raven-alias"}}}))
        assert resolve_hpc_host("nonexistent", expyre_config_path=cfg) is None

    @pytest.mark.unit
    def test_missing_systems_key_returns_none(self, tmp_path):
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({}))
        legacy = tmp_path / "legacy_does_not_exist.json"
        assert (
            resolve_hpc_host(
                "raven", expyre_config_path=cfg, legacy_expyre_config_path=legacy
            )
            is None
        )

    @pytest.mark.unit
    def test_malformed_json_returns_none(self, tmp_path):
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "config.json"
        cfg.write_text("{not valid json")
        legacy = tmp_path / "legacy_does_not_exist.json"
        assert (
            resolve_hpc_host(
                "raven", expyre_config_path=cfg, legacy_expyre_config_path=legacy
            )
            is None
        )

    @pytest.mark.unit
    def test_missing_file_returns_none(self, tmp_path):
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "does_not_exist.json"
        legacy = tmp_path / "legacy_does_not_exist.json"
        assert (
            resolve_hpc_host(
                "raven", expyre_config_path=cfg, legacy_expyre_config_path=legacy
            )
            is None
        )

    @pytest.mark.unit
    def test_falls_back_to_legacy_path_when_canonical_missing(self, tmp_path):
        """Canonical file doesn't exist yet (e.g. `alomancy add-hpc` hasn't
        been re-run since upgrading), but the pre-isolation legacy file
        does -- its systems must still resolve, not be silently dropped."""
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "does_not_exist.json"
        legacy = tmp_path / "legacy_config.json"
        legacy.write_text(json.dumps({"systems": {"raven": {"host": "raven-alias"}}}))
        assert (
            resolve_hpc_host(
                "raven", expyre_config_path=cfg, legacy_expyre_config_path=legacy
            )
            == "raven-alias"
        )

    @pytest.mark.unit
    def test_ignores_legacy_path_once_canonical_has_an_answer(self, tmp_path):
        """Once the canonical file exists and has the system, the legacy
        file (which may hold stale data) is never consulted."""
        from alomancy.utils.remote_ssh import resolve_hpc_host

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"systems": {"raven": {"host": "current-alias"}}}))
        legacy = tmp_path / "legacy_config.json"
        legacy.write_text(json.dumps({"systems": {"raven": {"host": "stale-alias"}}}))
        assert (
            resolve_hpc_host(
                "raven", expyre_config_path=cfg, legacy_expyre_config_path=legacy
            )
            == "current-alias"
        )


class TestGetRemoteAlomancyVersion:
    @pytest.mark.unit
    def test_success(self):
        from alomancy.utils.remote_ssh import get_remote_alomancy_version

        with patch(
            "alomancy.utils.remote_ssh.run_ssh_command",
            return_value=(True, "0.5.2\n", ""),
        ):
            assert get_remote_alomancy_version("h", "/venv/bin/python") == "0.5.2"

    @pytest.mark.unit
    def test_ssh_failure_returns_none(self):
        from alomancy.utils.remote_ssh import get_remote_alomancy_version

        with patch(
            "alomancy.utils.remote_ssh.run_ssh_command",
            return_value=(False, "", "ModuleNotFoundError"),
        ):
            assert get_remote_alomancy_version("h", "/venv/bin/python") is None

    @pytest.mark.unit
    def test_success_but_empty_stdout_returns_none(self):
        from alomancy.utils.remote_ssh import get_remote_alomancy_version

        with patch(
            "alomancy.utils.remote_ssh.run_ssh_command",
            return_value=(True, "", ""),
        ):
            assert get_remote_alomancy_version("h", "/venv/bin/python") is None


class TestGetAlomancyVersionForProfile:
    @pytest.mark.unit
    def test_skipped_under_test_mode(self, monkeypatch):
        """Default autouse env (ALOMANCY_TEST_MODE=1) must short-circuit to
        None without calling any of the real ssh-invoking helpers."""
        from alomancy.utils.remote_ssh import get_alomancy_version_for_profile

        with patch("alomancy.utils.remote_ssh.resolve_hpc_host") as mock_resolve:
            result = get_alomancy_version_for_profile(
                {"hpc_name": "raven", "pre_cmds": ["source /v/bin/activate"]}
            )

        assert result is None
        mock_resolve.assert_not_called()

    @pytest.mark.unit
    def test_full_chain_when_test_mode_disabled(self, monkeypatch):
        from alomancy.utils.remote_ssh import get_alomancy_version_for_profile

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        profile = {
            "hpc_name": "raven",
            "pre_cmds": ["source /u/user/.venvs/alomancy/bin/activate"],
        }
        with (
            patch(
                "alomancy.utils.remote_ssh.resolve_hpc_host", return_value="raven-alias"
            ),
            patch(
                "alomancy.utils.remote_ssh.get_remote_alomancy_version",
                return_value="0.5.2",
            ) as mock_get_version,
        ):
            result = get_alomancy_version_for_profile(profile)

        assert result == "0.5.2"
        mock_get_version.assert_called_once_with(
            "raven-alias", "/u/user/.venvs/alomancy/bin/python"
        )

    @pytest.mark.unit
    def test_missing_host_returns_none(self, monkeypatch):
        from alomancy.utils.remote_ssh import get_alomancy_version_for_profile

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        with patch("alomancy.utils.remote_ssh.resolve_hpc_host", return_value=None):
            result = get_alomancy_version_for_profile(
                {"hpc_name": "raven", "pre_cmds": ["source /v/bin/activate"]}
            )
        assert result is None

    @pytest.mark.unit
    def test_missing_pre_cmds_returns_none(self, monkeypatch):
        from alomancy.utils.remote_ssh import get_alomancy_version_for_profile

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        with patch(
            "alomancy.utils.remote_ssh.resolve_hpc_host", return_value="raven-alias"
        ):
            result = get_alomancy_version_for_profile(
                {"hpc_name": "raven", "pre_cmds": []}
            )
        assert result is None


class TestEnsureSshConnectivity:
    """ensure_ssh_connectivity is the pre_run_checks() step that
    authenticates every HPC host up front, before any remote submission
    begins, so a password/OTP prompt has a chance to be answered while
    whoever launched the run is presumably still at the terminal."""

    @pytest.mark.unit
    def test_skipped_under_test_mode(self):
        """Default autouse env (ALOMANCY_TEST_MODE=1) must short-circuit
        without calling any real ssh-invoking helper -- tests must never
        shell out to ssh or block on a password prompt."""
        from alomancy.utils.remote_ssh import ensure_ssh_connectivity

        with patch("subprocess.run") as mock_run:
            ensure_ssh_connectivity({"raven": {"hpc_name": "raven"}})

        mock_run.assert_not_called()

    @pytest.mark.unit
    def test_calls_ssh_true_per_distinct_host_unbounded_uncaptured(self, monkeypatch):
        """The actual subprocess call must have no timeout= and no
        capture_output=, unlike every other ssh call in this module --
        this one is deliberately allowed to block on, and print, an
        interactive password/OTP prompt on the real terminal."""
        from alomancy.utils.remote_ssh import ensure_ssh_connectivity

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        completed = subprocess.CompletedProcess(args=["ssh"], returncode=0)
        with (
            patch(
                "alomancy.utils.remote_ssh.resolve_hpc_host",
                return_value="raven-alias",
            ),
            patch("subprocess.run", return_value=completed) as mock_run,
        ):
            ensure_ssh_connectivity({"raven": {"hpc_name": "raven"}})

        mock_run.assert_called_once_with(["ssh", "raven-alias", "true"])

    @pytest.mark.unit
    def test_deduplicates_profiles_resolving_to_the_same_host(self, monkeypatch):
        from alomancy.utils.remote_ssh import ensure_ssh_connectivity

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        completed = subprocess.CompletedProcess(args=["ssh"], returncode=0)
        with (
            patch(
                "alomancy.utils.remote_ssh.resolve_hpc_host",
                return_value="shared-alias",
            ),
            patch("subprocess.run", return_value=completed) as mock_run,
        ):
            ensure_ssh_connectivity(
                {
                    "profile_a": {"hpc_name": "profile_a"},
                    "profile_b": {"hpc_name": "profile_b"},
                }
            )

        mock_run.assert_called_once()

    @pytest.mark.unit
    def test_unresolvable_host_skipped_without_raising(self, monkeypatch):
        from alomancy.utils.remote_ssh import ensure_ssh_connectivity

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        with (
            patch("alomancy.utils.remote_ssh.resolve_hpc_host", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            ensure_ssh_connectivity({"raven": {"hpc_name": "raven"}})  # must not raise

        mock_run.assert_not_called()

    @pytest.mark.unit
    def test_nonzero_exit_logged_but_does_not_raise(self, monkeypatch, caplog):
        from alomancy.utils.remote_ssh import ensure_ssh_connectivity

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        completed = subprocess.CompletedProcess(args=["ssh"], returncode=255)
        with (
            patch(
                "alomancy.utils.remote_ssh.resolve_hpc_host",
                return_value="raven-alias",
            ),
            patch("subprocess.run", return_value=completed),
            caplog.at_level(logging.WARNING, logger="alomancy.utils.remote_ssh"),
        ):
            ensure_ssh_connectivity({"raven": {"hpc_name": "raven"}})  # must not raise

        assert any("255" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_subprocess_error_logged_but_does_not_raise(self, monkeypatch, caplog):
        from alomancy.utils.remote_ssh import ensure_ssh_connectivity

        monkeypatch.setenv("ALOMANCY_TEST_MODE", "0")
        monkeypatch.setenv("ALOMANCY_MOCK_EXTERNAL", "0")

        with (
            patch(
                "alomancy.utils.remote_ssh.resolve_hpc_host",
                return_value="raven-alias",
            ),
            patch("subprocess.run", side_effect=OSError("no ssh binary")),
            caplog.at_level(logging.WARNING, logger="alomancy.utils.remote_ssh"),
        ):
            ensure_ssh_connectivity({"raven": {"hpc_name": "raven"}})  # must not raise

        assert any("no ssh binary" in r.message for r in caplog.records)
