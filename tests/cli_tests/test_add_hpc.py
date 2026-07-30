"""Unit tests for alomancy.cli.add_hpc."""

import json

import pytest
from yaml import safe_load


class TestBuildExpyreEntry:
    @pytest.mark.unit
    def test_cpu_header(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=False,
            partitions={
                "general": {"num_cores": 72, "max_time": "24:00:00", "max_mem": "240GB"}
            },
            commands=["module load python"],
            rundir="/scratch",
        )
        assert "#SBATCH --ntasks-per-node={num_cores}" in entry["header"]
        assert not any("cpus-per-task" in h for h in entry["header"])
        assert not any("gres" in h for h in entry["header"])

    @pytest.mark.unit
    def test_gpu_header_with_constraint_and_gres(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=True,
            partitions={
                "gpu": {"num_cores": 18, "max_time": "24:00:00", "max_mem": "12GB"}
            },
            commands=["module load cuda"],
            rundir="/scratch",
            gpu_constraint="gpu",
            gpu_gres="gpu:a100:1",
        )
        assert "#SBATCH --cpus-per-task={num_cores}" in entry["header"]
        assert any("constraint" in h for h in entry["header"])
        assert any("gres" in h for h in entry["header"])

    @pytest.mark.unit
    def test_gpu_header_no_optional_lines(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=True,
            partitions={
                "gpu": {"num_cores": 18, "max_time": "24:00:00", "max_mem": "12GB"}
            },
            commands=[],
            rundir="/scratch",
        )
        assert "#SBATCH --cpus-per-task={num_cores}" in entry["header"]
        assert not any("constraint" in h for h in entry["header"])
        assert not any("gres" in h for h in entry["header"])

    @pytest.mark.unit
    def test_mem_line_goes_on_partition_not_system_header(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=False,
            partitions={
                "general": {"num_cores": 72, "max_time": "24:00:00", "max_mem": "240GB"}
            },
            commands=["module load python"],
            rundir="/scratch",
        )
        assert not any("--mem" in h for h in entry["header"])
        assert "#SBATCH --mem=245760M" in entry["partitions"]["general"]["header"]

    @pytest.mark.unit
    def test_gpu_header_mem_line_on_partition(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=True,
            partitions={
                "gpu": {"num_cores": 18, "max_time": "24:00:00", "max_mem": "12GB"}
            },
            commands=[],
            rundir="/scratch",
        )
        assert not any("--mem" in h for h in entry["header"])
        assert "#SBATCH --mem=12288M" in entry["partitions"]["gpu"]["header"]

    @pytest.mark.unit
    def test_multi_partition_each_gets_own_mem(self):
        """Regression test: a single blanket --mem derived from one
        partition must not be applied to a different, differently-sized
        partition in the same system."""
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=False,
            partitions={
                "gpu": {"num_cores": 18, "max_time": "24:00:00", "max_mem": "12GB"},
                "general": {
                    "num_cores": 72,
                    "max_time": "24:00:00",
                    "max_mem": "240GB",
                },
            },
            commands=[],
            rundir="/scratch",
        )
        assert "#SBATCH --mem=12288M" in entry["partitions"]["gpu"]["header"]
        assert "#SBATCH --mem=245760M" in entry["partitions"]["general"]["header"]

    @pytest.mark.unit
    def test_unparseable_partition_mem_gets_no_mem_line(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="myhost",
            gpu=False,
            partitions={
                "general": {
                    "num_cores": 72,
                    "max_time": "24:00:00",
                    "max_mem": "lots",
                }
            },
            commands=[],
            rundir="/scratch",
        )
        assert not any(
            "--mem" in h for h in entry["partitions"]["general"].get("header", [])
        )

    @pytest.mark.unit
    def test_basic_fields_present(self):
        from alomancy.cli.add_hpc import build_expyre_entry

        entry = build_expyre_entry(
            host="raven",
            gpu=False,
            partitions={},
            commands=["module purge"],
            rundir="/ptmp/user",
        )
        assert entry["host"] == "raven"
        assert entry["remsh_cmd"] == "ssh"
        assert entry["scheduler"] == "slurm"
        assert entry["rundir"] == "/ptmp/user"
        assert entry["commands"] == ["module purge"]


class TestWithPartitionMemHeaders:
    @pytest.mark.unit
    def test_preserves_existing_header_entries(self):
        from alomancy.cli.add_hpc import _with_partition_mem_headers

        result = _with_partition_mem_headers(
            {
                "gpu": {
                    "num_cores": 18,
                    "max_time": "24:00:00",
                    "max_mem": "12GB",
                    "header": ["#SBATCH --gres=gpu:a100:1"],
                }
            }
        )
        header = result["gpu"]["header"]
        assert "#SBATCH --gres=gpu:a100:1" in header
        assert "#SBATCH --mem=12288M" in header

    @pytest.mark.unit
    def test_does_not_duplicate_existing_mem_line(self):
        from alomancy.cli.add_hpc import _with_partition_mem_headers

        result = _with_partition_mem_headers(
            {
                "general": {
                    "num_cores": 72,
                    "max_time": "24:00:00",
                    "max_mem": "240GB",
                    "header": ["#SBATCH --mem=100000M"],
                }
            }
        )
        mem_lines = [h for h in result["general"]["header"] if "--mem" in h]
        assert mem_lines == ["#SBATCH --mem=100000M"]

    @pytest.mark.unit
    def test_does_not_mutate_input(self):
        from alomancy.cli.add_hpc import _with_partition_mem_headers

        original = {
            "general": {"num_cores": 72, "max_time": "24:00:00", "max_mem": "240GB"}
        }
        _with_partition_mem_headers(original)
        assert "header" not in original["general"]


class TestMemStrToMb:
    @pytest.mark.unit
    def test_gb_converted(self):
        from alomancy.cli.add_hpc import mem_str_to_mb

        assert mem_str_to_mb("60GB") == 60 * 1024

    @pytest.mark.unit
    def test_tb_converted(self):
        from alomancy.cli.add_hpc import mem_str_to_mb

        assert mem_str_to_mb("1TB") == 1024 * 1024

    @pytest.mark.unit
    def test_empty_returns_none(self):
        from alomancy.cli.add_hpc import mem_str_to_mb

        assert mem_str_to_mb("") is None
        assert mem_str_to_mb(None) is None

    @pytest.mark.unit
    def test_unparseable_returns_none(self):
        from alomancy.cli.add_hpc import mem_str_to_mb

        assert mem_str_to_mb("lots of memory") is None

    @pytest.mark.unit
    def test_none_sentinel_string_returns_none(self):
        # expyre.units.mem_to_kB treats the literal string '_none_' as a
        # sentinel and returns None rather than raising -- mem_str_to_mb
        # must handle that None, not blow up on `None // 1024`.
        from alomancy.cli.add_hpc import mem_str_to_mb

        assert mem_str_to_mb("_none_") is None


class TestBuildAlomancyProfile:
    @pytest.mark.unit
    def test_cpu_profile_no_triton(self):
        from alomancy.cli.add_hpc import build_alomancy_profile

        profile = build_alomancy_profile(
            expyre_sys_name="raven",
            gpu=False,
            partitions=["general"],
            venv_cmd="source /u/user/.venvs/alomancy/bin/activate",
            node_info={
                "ranks_per_system": 72,
                "ranks_per_node": 72,
                "threads_per_rank": 1,
                "max_mem_per_node": "60GB",
            },
        )
        assert profile["gpu"] is False
        assert profile["hpc_name"] == "raven"
        assert not any("TRITON" in c for c in profile["pre_cmds"])

    @pytest.mark.unit
    def test_gpu_profile_with_triton(self):
        from alomancy.cli.add_hpc import build_alomancy_profile

        profile = build_alomancy_profile(
            expyre_sys_name="raccoon",
            gpu=True,
            partitions=["gpubig"],
            venv_cmd="source /home/user/.venvs/alomancy/bin/activate",
            node_info={},
            triton_cache="/home/user/.triton_cache",
        )
        assert profile["gpu"] is True
        assert any(
            "TRITON_CACHE_DIR=/home/user/.triton_cache" in c
            for c in profile["pre_cmds"]
        )

    @pytest.mark.unit
    def test_gpu_no_triton_not_added(self):
        from alomancy.cli.add_hpc import build_alomancy_profile

        profile = build_alomancy_profile(
            expyre_sys_name="raccoon",
            gpu=True,
            partitions=["gpubig"],
            venv_cmd="source /home/user/.venvs/alomancy/bin/activate",
            node_info={},
            triton_cache=None,
        )
        assert not any("TRITON" in c for c in profile["pre_cmds"])

    @pytest.mark.unit
    def test_qe_paths_written(self):
        from alomancy.cli.add_hpc import build_alomancy_profile

        profile = build_alomancy_profile(
            expyre_sys_name="raven",
            gpu=False,
            partitions=["general"],
            venv_cmd="source /u/user/.venvs/alomancy/bin/activate",
            node_info={},
            dft_code="qe",
            dft_paths={"pwx_path": "/path/to/pw.x", "pp_path": "/path/to/pps"},
        )
        assert profile["pwx_path"] == "/path/to/pw.x"
        assert profile["pp_path"] == "/path/to/pps"

    @pytest.mark.unit
    def test_vasp_paths_written(self):
        from alomancy.cli.add_hpc import build_alomancy_profile

        profile = build_alomancy_profile(
            expyre_sys_name="raven",
            gpu=False,
            partitions=["general"],
            venv_cmd="source /u/user/.venvs/alomancy/bin/activate",
            node_info={},
            dft_code="vasp",
            dft_paths={"vasp_path": "/path/to/vasp", "pp_path": "/path/to/potcars"},
        )
        assert profile["vasp_path"] == "/path/to/vasp"
        assert profile["pp_path"] == "/path/to/potcars"
        assert "pwx_path" not in profile


class TestWriteExpyreConfig:
    @pytest.mark.unit
    def test_creates_new_file(self, tmp_path):
        from alomancy.cli.add_hpc import build_expyre_entry, write_expyre_config

        entry = build_expyre_entry(
            host="h", gpu=False, partitions={}, commands=[], rundir="/s"
        )
        cfg = tmp_path / "config.json"
        write_expyre_config("myhost", entry, path=cfg)
        with open(cfg) as f:
            data = json.load(f)
        assert "systems" in data
        assert "myhost" in data["systems"]
        assert data["systems"]["myhost"]["host"] == "h"

    @pytest.mark.unit
    def test_merges_without_overwriting_existing(self, tmp_path):
        from alomancy.cli.add_hpc import build_expyre_entry, write_expyre_config

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"systems": {"existing": {"host": "old"}}}))
        entry = build_expyre_entry(
            host="new", gpu=False, partitions={}, commands=[], rundir="/s"
        )
        write_expyre_config("new_system", entry, path=cfg)
        with open(cfg) as f:
            data = json.load(f)
        assert "existing" in data["systems"]
        assert "new_system" in data["systems"]

    @pytest.mark.unit
    def test_malformed_json_raises_value_error(self, tmp_path):
        from alomancy.cli.add_hpc import build_expyre_entry, write_expyre_config

        cfg = tmp_path / "config.json"
        cfg.write_text("{not valid json")
        entry = build_expyre_entry(
            host="h", gpu=False, partitions={}, commands=[], rundir="/s"
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            write_expyre_config("myhost", entry, path=cfg)


class TestWriteAlomancyHpcConfig:
    @pytest.mark.unit
    def test_creates_dir_and_file(self, tmp_path):
        from alomancy.cli.add_hpc import (
            build_alomancy_profile,
            write_alomancy_hpc_config,
        )

        profile = build_alomancy_profile(
            "raven", False, ["general"], "source /venv/activate", {}
        )
        out = tmp_path / ".alomancy" / "hpc_config.yaml"
        write_alomancy_hpc_config("raven", profile, path=out)
        assert out.exists()
        with open(out) as f:
            data = safe_load(f)
        assert "raven" in data

    @pytest.mark.unit
    def test_merges_without_overwriting_existing(self, tmp_path):
        from alomancy.cli.add_hpc import (
            build_alomancy_profile,
            write_alomancy_hpc_config,
        )

        out = tmp_path / "hpc_config.yaml"
        out.write_text("existing_profile:\n  hpc_name: old\n")
        profile = build_alomancy_profile(
            "raven", False, ["general"], "source /venv/activate", {}
        )
        write_alomancy_hpc_config("new_profile", profile, path=out)
        with open(out) as f:
            data = safe_load(f)
        assert "existing_profile" in data
        assert "new_profile" in data
