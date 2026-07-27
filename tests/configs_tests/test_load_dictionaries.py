"""Unit tests for load_dictionaries HPC string resolution."""

import pytest
from yaml import safe_dump


def _write_run_config(path, sections):
    """Write a minimal jobs YAML."""
    with open(path, "w") as f:
        safe_dump(sections, f)


def _write_global_hpc(path, profiles):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        safe_dump(profiles, f)


@pytest.mark.unit
def test_hpc_string_resolved(tmp_path, monkeypatch):
    from alomancy.configs import global_config

    hpc_cfg = tmp_path / "hpc_config.yaml"
    _write_global_hpc(
        hpc_cfg, {"raven": {"hpc_name": "raven", "partitions": ["general"]}}
    )
    monkeypatch.setattr(global_config, "ALOMANCY_HPC_CONFIG", hpc_cfg)

    run_cfg = tmp_path / "config.yaml"
    _write_run_config(
        run_cfg, {"mlip_committee": {"name": "mlip_committee", "hpc": "raven"}}
    )

    from alomancy.configs.config_dictionaries import load_dictionaries

    result = load_dictionaries(run_cfg)
    assert isinstance(result["mlip_committee"]["hpc"], dict)
    assert result["mlip_committee"]["hpc"]["hpc_name"] == "raven"


@pytest.mark.unit
def test_hpc_dict_passthrough(tmp_path, monkeypatch):
    from alomancy.configs import global_config

    monkeypatch.setattr(global_config, "ALOMANCY_HPC_CONFIG", tmp_path / "empty.yaml")

    run_cfg = tmp_path / "config.yaml"
    hpc_dict = {"hpc_name": "raven", "partitions": ["general"]}
    _write_run_config(
        run_cfg, {"mlip_committee": {"name": "mlip_committee", "hpc": hpc_dict}}
    )

    from alomancy.configs.config_dictionaries import load_dictionaries

    result = load_dictionaries(run_cfg)
    assert result["mlip_committee"]["hpc"] == hpc_dict


@pytest.mark.unit
def test_missing_hpc_string_raises(tmp_path, monkeypatch):
    from alomancy.configs import global_config

    hpc_cfg = tmp_path / "hpc_config.yaml"
    _write_global_hpc(hpc_cfg, {})  # empty — no profiles
    monkeypatch.setattr(global_config, "ALOMANCY_HPC_CONFIG", hpc_cfg)

    run_cfg = tmp_path / "config.yaml"
    _write_run_config(
        run_cfg, {"mlip_committee": {"name": "mlip_committee", "hpc": "unknown_hpc"}}
    )

    from alomancy.configs.config_dictionaries import load_dictionaries

    with pytest.raises(ValueError, match="unknown_hpc"):
        load_dictionaries(run_cfg)


@pytest.mark.unit
def test_no_global_config_dict_passthrough(tmp_path, monkeypatch):
    """When ~/.alomancy/hpc_config.yaml is absent, dict hpc values pass through."""
    from alomancy.configs import global_config

    monkeypatch.setattr(
        global_config, "ALOMANCY_HPC_CONFIG", tmp_path / "nonexistent.yaml"
    )

    run_cfg = tmp_path / "config.yaml"
    hpc_dict = {"hpc_name": "raven", "partitions": ["general"]}
    _write_run_config(
        run_cfg, {"mlip_committee": {"name": "mlip_committee", "hpc": hpc_dict}}
    )

    from alomancy.configs.config_dictionaries import load_dictionaries

    result = load_dictionaries(run_cfg)
    assert result["mlip_committee"]["hpc"] == hpc_dict


@pytest.mark.unit
def test_partial_sections_only(tmp_path, monkeypatch):
    """Sections absent from the YAML are silently skipped."""
    from alomancy.configs import global_config

    hpc_cfg = tmp_path / "hpc_config.yaml"
    _write_global_hpc(hpc_cfg, {"raven": {"hpc_name": "raven"}})
    monkeypatch.setattr(global_config, "ALOMANCY_HPC_CONFIG", hpc_cfg)

    run_cfg = tmp_path / "config.yaml"
    _write_run_config(
        run_cfg, {"mlip_committee": {"name": "mlip_committee", "hpc": "raven"}}
    )

    from alomancy.configs.config_dictionaries import load_dictionaries

    result = load_dictionaries(run_cfg)
    assert isinstance(result["mlip_committee"]["hpc"], dict)
    assert "structure_generation" not in result
