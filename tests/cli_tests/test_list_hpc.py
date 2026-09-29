"""Tests for `alomancy list-hpc` (cli/list_hpc.py) and its shared row
builder (configs/hpc_profiles.py)."""

from unittest import mock

import pytest
import yaml

_PROFILES = {
    "raven": {
        "hpc_name": "raven",
        "gpu": False,
        "partitions": ["general"],
        "node_info": {
            "ranks_per_system": 72,
            "ranks_per_node": 72,
            "threads_per_rank": 1,
            "max_mem_per_node": "240GB",
        },
        "pwx_path": "/opt/qe/pw.x",
        "max_num_of_concurrent_jobs": 30,
    },
    "raven_gpu": {
        "hpc_name": "raven_gpu",
        "gpu": True,
        "partitions": ["gpu"],
        "node_info": {"ranks_per_node": 4},
        "max_concurrent_jobs": 8,
    },
}


@pytest.fixture
def hpc_config(tmp_path, monkeypatch):
    path = tmp_path / "hpc_config.yaml"
    monkeypatch.setattr("alomancy.configs.global_config.ALOMANCY_HPC_CONFIG", path)
    monkeypatch.setattr("alomancy.cli.list_hpc.ALOMANCY_HPC_CONFIG", path)
    # Never read the real ~/.alomancy expyre config for ssh hosts.
    monkeypatch.setattr(
        "alomancy.configs.hpc_profiles.resolve_hpc_host", lambda name: f"{name}-host"
    )
    return path


@pytest.mark.unit
def test_lists_every_profile_with_details(hpc_config):
    from alomancy.cli.list_hpc import list_hpc

    hpc_config.write_text(yaml.safe_dump(_PROFILES))
    out = list_hpc()

    assert "2 HPC profile(s)" in out
    for expected in (
        "raven",
        "raven_gpu",
        "raven-host",
        "/opt/qe/pw.x",
        "240GB",
        "30",
        "8 (old key max_concurrent_jobs)",
    ):
        assert expected in out


@pytest.mark.unit
def test_no_ssh_without_check_remote(hpc_config):
    from alomancy.cli.list_hpc import list_hpc

    hpc_config.write_text(yaml.safe_dump(_PROFILES))
    with mock.patch(
        "alomancy.configs.hpc_profiles.get_alomancy_version_for_profile"
    ) as version:
        out = list_hpc()
    version.assert_not_called()
    assert "alomancy_version" not in out


@pytest.mark.unit
def test_check_remote_adds_version_column(hpc_config):
    from alomancy.cli.list_hpc import list_hpc

    hpc_config.write_text(yaml.safe_dump(_PROFILES))
    with mock.patch(
        "alomancy.configs.hpc_profiles.get_alomancy_version_for_profile",
        return_value="0.9.0",
    ) as version:
        out = list_hpc(check_remote=True)
    assert version.call_count == 2
    assert "alomancy_version" in out
    assert "0.9.0" in out


@pytest.mark.unit
def test_empty_config_points_at_add_hpc(hpc_config):
    from alomancy.cli.list_hpc import list_hpc

    assert "alomancy add-hpc" in list_hpc()


@pytest.mark.unit
def test_cli_dispatches_list_hpc(hpc_config, monkeypatch, capsys):
    from alomancy.cli.main import main

    hpc_config.write_text(yaml.safe_dump(_PROFILES))
    monkeypatch.setattr("sys.argv", ["alomancy", "list-hpc"])
    main()
    assert "raven_gpu" in capsys.readouterr().out
