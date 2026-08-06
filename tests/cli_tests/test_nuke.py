"""Unit tests for alomancy.cli.nuke."""

from pathlib import Path
from unittest import mock

import pytest

from alomancy.cli.nuke import nuke_expyre_results, resolve_default_expyre_dir


@pytest.mark.unit
class TestResolveDefaultExpyreDir:
    def test_uses_resolved_local_stage_dir_when_available(self, tmp_path, monkeypatch):
        from expyre import config as expyre_config

        run_local = tmp_path / "run" / ".expyre"
        monkeypatch.setattr(expyre_config, "local_stage_dir", run_local)

        assert resolve_default_expyre_dir() == run_local

    def test_falls_back_to_home_expyre_when_unresolved(self, tmp_path, monkeypatch):
        from expyre import config as expyre_config

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(expyre_config, "local_stage_dir", None)

        assert resolve_default_expyre_dir() == tmp_path / ".expyre"


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

    def test_none_default_resolves_via_resolve_default_expyre_dir(
        self, tmp_path, monkeypatch
    ):
        expyre_dir = _make_expyre_dir(tmp_path)
        monkeypatch.setattr(
            "alomancy.cli.nuke.resolve_default_expyre_dir", lambda: expyre_dir
        )
        monkeypatch.setattr("builtins.input", lambda: "y")

        nuke_expyre_results()

        assert not (expyre_dir / "jobs.db").exists()


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
def test_cli_entrypoint_nuke_default_dir_falls_back_to_home(tmp_path, monkeypatch):
    """With no local_stage_dir resolved (e.g. no HPC configured yet), the
    default target is still the legacy ~/.expyre."""
    from expyre import config as expyre_config

    from alomancy.cli.main import main

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(expyre_config, "local_stage_dir", None)
    fake_home_expyre = tmp_path / ".expyre"

    with (
        mock.patch("sys.argv", ["alomancy", "nuke"]),
        mock.patch("alomancy.cli.nuke.nuke_expyre_results") as m_nuke,
    ):
        main()

    m_nuke.assert_called_once_with(fake_home_expyre.resolve())


@pytest.mark.unit
def test_cli_entrypoint_nuke_default_dir_uses_resolved_local_stage_dir(
    tmp_path, monkeypatch
):
    """When expyre has resolved a (run-local) local_stage_dir, nuke targets
    that instead of ~/.expyre, so it only ever clears this run's own job
    state -- not every other alomancy run sharing the same home directory."""
    from expyre import config as expyre_config

    from alomancy.cli.main import main

    run_local_expyre = tmp_path / "some_run" / ".expyre"
    run_local_expyre.mkdir(parents=True)
    monkeypatch.setattr(expyre_config, "local_stage_dir", run_local_expyre)

    with (
        mock.patch("sys.argv", ["alomancy", "nuke"]),
        mock.patch("alomancy.cli.nuke.nuke_expyre_results") as m_nuke,
    ):
        main()

    m_nuke.assert_called_once_with(run_local_expyre.resolve())
