"""Every example config -- the files under examples/ and the YAML in
README.md -- must build its workflow and resolve each module's settings
with the current code, without warnings. A renamed or removed key then
breaks a test here instead of the next person who copies an example."""

import copy
import logging
import re
from pathlib import Path

import pytest
import yaml

from alomancy.core.active_learning_workflow import build_workflow

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples"
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


def _config_warnings(config: dict) -> list[str]:
    """Build the workflow and its settings summary; return the warnings."""
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
        workflow = build_workflow(_with_inline_hpc(config))
        loggers[1].addHandler(handler)
        workflow.display_workflow_summary()
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
    # The triton notice is about the local PyTorch install, not the config.
    return [r.getMessage() for r in records if "triton" not in r.getMessage().lower()]


# -- README.md ---------------------------------------------------------------

_README_YAML = [
    yaml.safe_load(block)
    for block in re.findall(
        r"```yaml\n(.*?)```", (_ROOT / "README.md").read_text(), re.S
    )
]
_README_FULL = [b for b in _README_YAML if {"general", *_PHASES} <= set(b)]
_README_SNIPPETS = [b for b in _README_YAML if b not in _README_FULL]


def _deep_merge(base: dict, snippet: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in snippet.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _merged(base: dict, snippet: dict) -> dict:
    """*snippet* applied over *base*, as the README's "one line" edits are."""
    out = _deep_merge(base, snippet)
    if out["general"].get("al_workflow") != "committee_uncertainty":
        # The README says to drop it: only the committee workflow reads it.
        out["general"].pop("committee_uncertainty_kwargs", None)
    return out


@pytest.mark.unit
def test_examples_are_found():
    assert len(_CONFIGS) >= 7
    assert len(_README_FULL) == 1
    assert len(_README_SNIPPETS) >= 6


@pytest.mark.unit
@pytest.mark.parametrize(
    "path", _CONFIGS, ids=[str(p.relative_to(_EXAMPLES)) for p in _CONFIGS]
)
def test_example_config_builds_and_resolves(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _config_warnings(yaml.safe_load(path.read_text())) == []


@pytest.mark.unit
def test_readme_config_builds_and_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _config_warnings(copy.deepcopy(_README_FULL[0])) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "snippet", _README_SNIPPETS, ids=[str(s)[:60] for s in _README_SNIPPETS]
)
def test_readme_snippet_applied_to_its_config_builds(snippet, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _config_warnings(_merged(_README_FULL[0], snippet)) == []


@pytest.mark.unit
def test_readme_config_matches_tested_example():
    """The README's config is examples/configs/committee_mace_cold_start.yaml,
    so the two can't drift apart."""
    example = yaml.safe_load(
        (_EXAMPLES / "configs" / "committee_mace_cold_start.yaml").read_text()
    )
    assert _README_FULL[0] == example
