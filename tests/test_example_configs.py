"""Every example config under examples/ must build its workflow and resolve
each module's settings with the current code -- so a renamed or removed key
breaks a test here instead of the next person who copies an example."""

import logging
from pathlib import Path

import pytest
import yaml

from alomancy.core.active_learning_workflow import build_workflow

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_CONFIGS = sorted(
    p
    for p in _EXAMPLES.rglob("*.yaml")
    if p.name != "hpc_config.yaml" and "results" not in p.parts
)
_PHASES = (
    "initialization",
    "training",
    "structure_generation",
    "high_accuracy_evaluation",
)
# Stands in for the named profiles in ~/.alomancy/hpc_config.yaml.
_HPC = {"hpc_name": "example", "pre_cmds": [], "partitions": ["p"]}


def _with_inline_hpc(config: dict) -> dict:
    for phase in _PHASES:
        section = config.get(phase) or {}
        if not isinstance(section.get("hpc"), dict):
            section["hpc"] = dict(_HPC)
        config[phase] = section
    return config


@pytest.mark.unit
def test_examples_are_found():
    assert len(_CONFIGS) >= 7


@pytest.mark.unit
@pytest.mark.parametrize(
    "path", _CONFIGS, ids=[str(p.relative_to(_EXAMPLES)) for p in _CONFIGS]
)
def test_example_config_builds_and_resolves(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _with_inline_hpc(yaml.safe_load(path.read_text()))

    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append  # type: ignore[method-assign]
    # The construction-time warnings (unknown keys) come from this module's
    # logger: setup_logging, called during construction, replaces the
    # handlers on "alomancy" itself.
    loggers = [
        logging.getLogger("alomancy.core.active_learning_workflow"),
        logging.getLogger("alomancy"),
    ]
    loggers[0].addHandler(handler)
    try:
        workflow = build_workflow(config)
        loggers[1].addHandler(handler)
        workflow.display_workflow_summary()
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
    # Config warnings (unknown keys and the like); the triton notice is about
    # the local PyTorch install, not the config.
    config_warnings = [
        r.getMessage() for r in records if "triton" not in r.getMessage().lower()
    ]
    assert config_warnings == []
