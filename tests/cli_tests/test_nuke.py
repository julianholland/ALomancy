"""Unit tests for alomancy.cli.nuke."""

from pathlib import Path
from unittest import mock

import pytest

from alomancy.cli.nuke import nuke_expyre_results


def _make_expyre_dir(tmp_path: Path) -> Path:
    expyre_dir = tmp_path / ".expyre"
    expyre_dir.mkdir()
    (expyre_dir / "config.json").write_text('{"systems": {}}')
    (expyre_dir / "jobs.db").write_text("stale job cache")
    stage_dir = expyre_dir / "run_high_accuracy_evaluation_abc123_xyz"
    stage_dir.mkdir()
    (stage_dir / "_expyre_task_in.pckl").write_text("data")
    return expyre_dir


@pytest.mark.unit
class TestNukeExpyreResults:
    def test_deletes_everything_except_config_on_yes(self, tmp_path, monkeypatch):
        expyre_dir = _make_expyre_dir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda: "y")

        nuke_expyre_results(expyre_dir)

        assert (expyre_dir / "config.json").exists()
        assert not (expyre_dir / "jobs.db").exists()
        assert not (expyre_dir / "run_high_accuracy_evaluation_abc123_xyz").exists()

    def test_accepts_uppercase_yes(self, tmp_path, monkeypatch):
        expyre_dir = _make_expyre_dir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda: "Y")

        nuke_expyre_results(expyre_dir)

        assert not (expyre_dir / "jobs.db").exists()

    def test_aborts_and_deletes_nothing_on_no(self, tmp_path, monkeypatch):
        expyre_dir = _make_expyre_dir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda: "n")

        nuke_expyre_results(expyre_dir)

        assert (expyre_dir / "config.json").exists()
        assert (expyre_dir / "jobs.db").exists()
        assert (expyre_dir / "run_high_accuracy_evaluation_abc123_xyz").exists()

    def test_aborts_on_arbitrary_input(self, tmp_path, monkeypatch):
        expyre_dir = _make_expyre_dir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda: "")

        nuke_expyre_results(expyre_dir)

        assert (expyre_dir / "jobs.db").exists()

    def test_missing_directory_is_a_no_op_and_never_prompts(
        self, tmp_path, monkeypatch
    ):
        missing_dir = tmp_path / "does_not_exist"

        def _fail_if_called():
            raise AssertionError("input() must not be called for a missing directory")

        monkeypatch.setattr("builtins.input", lambda: _fail_if_called())

        nuke_expyre_results(missing_dir)

    def test_deletes_symlinks_without_following_them(self, tmp_path, monkeypatch):
        expyre_dir = _make_expyre_dir(tmp_path)
        target = tmp_path / "outside_target.txt"
        target.write_text("must survive")
        (expyre_dir / "link_to_outside").symlink_to(target)
        monkeypatch.setattr("builtins.input", lambda: "y")

        nuke_expyre_results(expyre_dir)

        assert not (expyre_dir / "link_to_outside").exists()
        assert target.exists()


@pytest.mark.unit
def test_cli_entrypoint_nuke(tmp_path):
    from alomancy.cli.main import main

    expyre_dir = tmp_path / ".expyre"
    expyre_dir.mkdir()

    with (
        mock.patch(
            "sys.argv",
            ["alomancy", "nuke", "--expyre-dir", str(expyre_dir)],
        ),
        mock.patch("alomancy.cli.nuke.nuke_expyre_results") as m_nuke,
    ):
        main()

    m_nuke.assert_called_once_with(expyre_dir.resolve())


@pytest.mark.unit
def test_cli_entrypoint_nuke_default_dir(tmp_path, monkeypatch):
    from alomancy.cli.main import main

    monkeypatch.setenv("HOME", str(tmp_path))
    fake_home_expyre = tmp_path / ".expyre"

    with (
        mock.patch("sys.argv", ["alomancy", "nuke"]),
        mock.patch("alomancy.cli.nuke.nuke_expyre_results") as m_nuke,
    ):
        main()

    m_nuke.assert_called_once_with(fake_home_expyre.resolve())
