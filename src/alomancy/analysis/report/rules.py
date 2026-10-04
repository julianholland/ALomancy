"""Turn triggers (triggers.py) and suggestions (suggestions.yaml) into the
report's findings.

Never raises over the suggestions file: an entry for an unknown trigger, a
trigger with no entry, a malformed entry or a template naming a value the
trigger doesn't provide are all reported (logged, and listed in the report)
instead of breaking the loop.
"""

import logging
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from alomancy.analysis.report.triggers import TRIGGERS

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTIONS = Path(__file__).with_name("suggestions.yaml")
SEVERITIES = ("error", "warning", "info")


@dataclass
class Finding:
    trigger: str
    severity: str
    message: str
    value: float


def load_suggestions(override: str | Path | None = None) -> dict[str, dict]:
    """The packaged suggestions, with *override*'s entries replacing them
    one key at a time (so an override can change just one threshold)."""
    entries: dict[str, dict] = yaml.safe_load(DEFAULT_SUGGESTIONS.read_text()) or {}
    if override:
        path = Path(override)
        try:
            extra = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Ignoring report suggestions file %s: %s", path, exc)
            extra = {}
        for name, entry in extra.items():
            if isinstance(entry, dict):
                entries[name] = {**entries.get(name, {}), **entry}
            else:
                logger.warning(
                    "Ignoring malformed suggestion entry %r in %s.", name, path
                )
    return entries


def _fill(template: str, values: dict[str, Any], trigger: str) -> str:
    try:
        return string.Formatter().vformat(template, (), values)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        logger.warning(
            "Suggestion text for %r could not be filled (%r); showing it unfilled.",
            trigger,
            exc,
        )
        return template


def evaluate(
    stats: dict, suggestions: dict[str, dict]
) -> tuple[list[Finding], list[str]]:
    """Findings for this loop, most severe first, plus notes about the
    suggestions file itself (unknown triggers, missing entries)."""
    notes = []
    for name in sorted(set(suggestions) - set(TRIGGERS)):
        notes.append(f"suggestions entry `{name}` has no trigger (typo?)")
        logger.warning("Report suggestions: entry %r has no matching trigger.", name)
    findings = []
    for name, fn in TRIGGERS.items():
        entry = suggestions.get(name)
        if entry is None:
            notes.append(f"trigger `{name}` has no suggestion configured")
            continue
        try:
            result = fn(stats)
        except Exception as exc:  # a broken trigger must not break the report
            logger.warning("Report trigger %r failed: %s", name, exc, exc_info=True)
            continue
        if result is None:
            continue
        try:
            threshold = float(entry.get("threshold", 0.0))
        except (TypeError, ValueError):
            notes.append(f"suggestions entry `{name}` has a non-numeric threshold")
            continue
        if float(result["value"]) < threshold:
            continue
        severity = entry.get("severity", "warning")
        if severity not in SEVERITIES:
            severity = "warning"
        message = _fill(str(entry.get("suggestion", "")).strip(), result, name)
        findings.append(Finding(name, severity, message, float(result["value"])))
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    return findings, notes
