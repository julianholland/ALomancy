"""Tests for alomancy/__init__.py's per-run ExPyRe isolation setup."""

import json
import os
import sys
import warnings

import pytest

from alomancy import _pin_local_expyre_root, _seed_local_expyre_root


@pytest.mark.unit
class TestSeedLocalExpyreRoot:
    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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


def _write_config(directory, systems, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"systems": systems, **extra}))


def _no_partition_system(host):
    return {"host": host, "partitions": None, "scheduler": "slurm"}


class TestPinLocalExpyreRoot:
    """The run's ExPyRe directory is the only one expyre reads: EXPYRE_ROOT
    points at it, so parent ".expyre" configs are never merged in."""

    @pytest.fixture(autouse=True)
    def _no_expyre_root(self, monkeypatch):
        monkeypatch.delenv("EXPYRE_ROOT", raising=False)

    @pytest.mark.unit
    def test_parent_expyre_config_is_not_merged(self, tmp_path, monkeypatch):
        from expyre.config import _get_config

        import alomancy as alomancy_module

        # A shared parent .expyre with an extra system and its own
        # local_stage_dir: the walk would merge both into the run.
        _write_config(
            tmp_path / ".expyre",
            {"shared_only": _no_partition_system("shared")},
            local_stage_dir=str(tmp_path / ".expyre"),
        )
        master = tmp_path / "master" / "expyre_config.json"
        master.parent.mkdir()
        master.write_text(
            json.dumps({"systems": {"raven": _no_partition_system("raven")}})
        )
        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", master)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _pin_local_expyre_root(run_dir)

        assert os.environ["EXPYRE_ROOT"] == str(run_dir / ".expyre")
        stage_dir, config = _get_config(os.environ["EXPYRE_ROOT"])
        assert stage_dir == run_dir / ".expyre"
        assert set(config["systems"]) == {"raven"}
        assert "local_stage_dir" not in config

    @pytest.mark.unit
    @pytest.mark.parametrize("name", [".expyre", "_expyre"])
    def test_existing_run_expyre_is_pinned(self, tmp_path, name):
        run_dir = tmp_path / "run"
        _write_config(run_dir / name, {"raven": _no_partition_system("raven")})

        _pin_local_expyre_root(run_dir)

        assert os.environ["EXPYRE_ROOT"] == str(run_dir / name)

    @pytest.mark.unit
    def test_stub_without_config_is_not_pinned(self, tmp_path):
        """A hand-made .expyre with no config.json relies on merging the
        parents' systems; pinning it would leave it with none."""
        run_dir = tmp_path / "run"
        (run_dir / ".expyre").mkdir(parents=True)

        _pin_local_expyre_root(run_dir)

        assert "EXPYRE_ROOT" not in os.environ

    @pytest.mark.unit
    def test_nothing_to_pin_without_a_config_source(self, tmp_path, monkeypatch):
        import alomancy as alomancy_module

        monkeypatch.setattr(alomancy_module, "EXPYRE_CONFIG", tmp_path / "nope.json")
        monkeypatch.setattr(
            alomancy_module, "LEGACY_EXPYRE_CONFIG", tmp_path / "no.json"
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _pin_local_expyre_root(run_dir)

        assert "EXPYRE_ROOT" not in os.environ

    @pytest.mark.unit
    def test_user_set_expyre_root_is_left_alone(self, tmp_path, monkeypatch):
        """_ensure_local_expyre_root (the import-time entry point) never
        touches an EXPYRE_ROOT the user set."""
        import alomancy as alomancy_module

        monkeypatch.setenv("EXPYRE_ROOT", "/user/choice")
        monkeypatch.chdir(tmp_path)
        # Past the pytest guard; monkeypatch puts the module back.
        monkeypatch.delitem(sys.modules, "pytest")

        alomancy_module._ensure_local_expyre_root()

        assert os.environ["EXPYRE_ROOT"] == "/user/choice"
        assert not (tmp_path / ".expyre").exists()


@pytest.mark.unit
class TestRsyncRetryWarningFilter:
    """`alomancy.__init__._register_rsync_retry_warning_filter` silences
    expyre's per-retry-attempt FailedSubprocessWarning specifically for the
    known, harmless race between mid-training rsync (get_remotes(), polled
    every check_interval while an mlip_committee job is still running) and
    MACE's checkpoint housekeeping (CheckpointIO.save deletes the previous
    checkpoint right after writing the new one, which a mid-sync rsync can
    catch mid-delete). See that function's docstring for the full
    mechanism. It's called once at import time; each test re-applies it
    inside its own warnings.catch_warnings() block since pytest's warnings
    plugin resets the filter list to "always" around every test."""

    @pytest.mark.unit
    def test_rsync_retry_warning_is_silenced(self):
        from expyre.subprocess import FailedSubprocessWarning

        from alomancy import _register_rsync_retry_warning_filter

        with warnings.catch_warnings(record=True) as caught:
            _register_rsync_retry_warning_filter()
            warnings.warn(
                'Succeeded to run "bash -c rsync -e ssh -a host:/remote/dir '
                '/local/.expyre" on attempt 1 after failure(s), trying again',
                category=FailedSubprocessWarning,
                stacklevel=2,
            )
            assert caught == []

    @pytest.mark.unit
    def test_final_giveup_warning_still_surfaces(self):
        """The terminal 'giving up' warning (which precedes expyre raising a
        real RuntimeError) must never be silenced -- only the in-progress
        retry chatter is."""
        from expyre.subprocess import FailedSubprocessWarning

        from alomancy import _register_rsync_retry_warning_filter

        with warnings.catch_warnings(record=True) as caught:
            _register_rsync_retry_warning_filter()
            warnings.warn(
                'Failed to run "bash -c rsync -e ssh -a host:/remote/dir '
                '/local/.expyre" on attempt 2 for the last time, giving up.\n'
                "STDERR\nrsync: stale file handle",
                category=FailedSubprocessWarning,
                stacklevel=2,
            )
            assert len(caught) == 1

    @pytest.mark.unit
    def test_unrelated_subprocess_retry_warning_still_surfaces(self):
        """Only rsync retry chatter is filtered -- an unrelated transient
        subprocess retry (e.g. ssh during job submission/status polling)
        must stay visible, since it isn't the known checkpoint-race noise."""
        from expyre.subprocess import FailedSubprocessWarning

        from alomancy import _register_rsync_retry_warning_filter

        with warnings.catch_warnings(record=True) as caught:
            _register_rsync_retry_warning_filter()
            warnings.warn(
                'Failed to run "ssh headnode squeue" on attempt 0, trying '
                "again.\nSTDERR\nconnection reset",
                category=FailedSubprocessWarning,
                stacklevel=2,
            )
            assert len(caught) == 1


# --- Shortened ExPyRe remote rundir ("<host>-<cwd>" -> run path below $HOME) ---


def _expyre_system(local_stage_dir, hostname="unkownhost"):
    """A real expyre System built the way expyre/config.py's init builds it."""
    from expyre.system import System

    run_dir = local_stage_dir.parent
    extra = hostname + "-" + str(run_dir).replace("/", "_")
    return System(
        host="raven",
        partitions={},
        scheduler="slurm",
        rundir="/raven/ptmp/advr/alomancy_scratch",
        rundir_extra=extra,
    )


class TestShortenRundirExtra:
    @pytest.mark.unit
    def test_short_extra_is_path_below_home(self, tmp_path):
        from alomancy import _short_rundir_extra

        home = tmp_path / "home"
        assert (
            _short_rundir_extra(home / "Simulations" / "alomancy" / "test_5", home)
            == "Simulations_alomancy_test_5"
        )

    @pytest.mark.unit
    def test_short_extra_follows_symlinked_home(self, tmp_path):
        """$HOME=/home/u -> /work/home/u while the run path is the real one."""
        from alomancy import _short_rundir_extra

        real_home = tmp_path / "work" / "home" / "u"
        (real_home / "sims").mkdir(parents=True)
        link_home = tmp_path / "home_u"
        link_home.symlink_to(real_home)
        assert _short_rundir_extra(real_home / "sims", link_home) == "sims"

    @pytest.mark.unit
    def test_short_extra_outside_home_keeps_full_path(self, tmp_path):
        from pathlib import Path

        from alomancy import _short_rundir_extra

        assert (
            _short_rundir_extra(Path("/scratch/u/run1"), tmp_path / "home")
            == "scratch_u_run1"
        )

    @pytest.mark.unit
    def test_rewrites_advr_rundir(self, tmp_path):
        from pathlib import Path

        from alomancy import _host_tag, _shorten_rundir_extra

        stage = Path("/work/home/advr/Simulations/alomancy/test_5/.expyre")
        system = _expyre_system(stage)
        assert system.remote_rundir == (
            "/raven/ptmp/advr/alomancy_scratch/"
            "unkownhost-_work_home_advr_Simulations_alomancy_test_5"
        )

        changed = _shorten_rundir_extra(
            {"raven_cpu": system},
            stage,
            "unkownhost",
            Path("/work/home/advr"),
            "ravenlogin01",
        )

        assert changed
        tag = _host_tag("ravenlogin01")
        assert len(tag) == 6
        assert system.remote_rundir == (
            f"/raven/ptmp/advr/alomancy_scratch/{tag}-Simulations_alomancy_test_5"
        )

    @pytest.mark.unit
    def test_host_tag_keeps_two_machines_apart(self):
        from alomancy import _host_tag

        assert _host_tag("laptop") != _host_tag("workstation")
        assert _host_tag("laptop") == _host_tag("laptop")
        assert len(_host_tag("a" * 200)) <= 8

    @pytest.mark.unit
    def test_unexpected_rundir_left_alone(self, tmp_path):
        from expyre.system import System

        from alomancy import _shorten_rundir_extra

        system = System(
            host="raven", partitions={}, scheduler="slurm", rundir="/scratch/al"
        )
        stage = tmp_path / "run" / ".expyre"
        assert not _shorten_rundir_extra(
            {"r": system}, stage, "unkownhost", tmp_path, "h"
        )
        assert system.remote_rundir == "/scratch/al"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("status", "pins"),
        [
            ("submitted", True),
            ("started", True),
            ("created", False),
            ("succeeded", False),
            ("failed", False),
            ("died", False),
            ("processed", False),
        ],
    )
    def test_only_live_remote_jobs_pin_the_long_layout(self, tmp_path, status, pins):
        """Only submitted/started jobs need their remote directory on resume.
        A failed one (what `alomancy cancel` sets), or one orphaned in
        'created', must not keep the run on the long layout for good."""
        from expyre.jobsdb import JobsDB

        from alomancy import _jobs_pinning_long_layout

        db = JobsDB(tmp_path / "jobs.db")
        db.add("job_0", name="sp_vasp_0", from_dir=str(tmp_path / "x"), status=status)
        db.add(
            "job_1", name="sp_vasp_1", from_dir=str(tmp_path / "y"), status="processed"
        )

        assert _jobs_pinning_long_layout(db) == (("job_0",) if pins else ())
        assert _jobs_pinning_long_layout(None) == ()


@pytest.mark.unit
def test_path_error_names_the_jobs_pinning_the_long_layout(monkeypatch):
    import alomancy
    from alomancy.remote_submission.submitters import check_remote_path_length

    long_path = "/" + "x" * 300
    monkeypatch.setattr(alomancy, "_LONG_RUNDIR_JOB_IDS", ())
    with pytest.raises(ValueError) as plain:
        check_remote_path_length(long_path, limit=240, remote_rundir="/x", context="t")
    assert "alomancy cancel" not in str(plain.value)

    monkeypatch.setattr(alomancy, "_LONG_RUNDIR_JOB_IDS", ("run_sp_vasp_x_1",))
    with pytest.raises(ValueError, match="alomancy cancel") as pinned:
        check_remote_path_length(long_path, limit=240, remote_rundir="/x", context="t")
    assert "run_sp_vasp_x_1" in str(pinned.value)
