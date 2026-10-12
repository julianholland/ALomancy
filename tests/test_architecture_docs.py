"""CLAUDE.md and docs/architecture/*.md must only reference code that exists.

These notes are what an assistant reads before changing a subsystem, so a
stale reference does more harm than none. Checked, inside backticks:
``path.py`` and ``path.py:name`` / ``path.py::test_name`` (paths relative
to the repo root, ``src/alomancy/`` or ``tests/``, or a bare file name);
``pkg/module.name`` and ``module.name`` (only when exactly one source file
has that module name); ``docs/...md`` and ``src/``/``tests/`` directory
paths. Other backticked text (config keys, shell commands, results paths)
is not checked.
"""

import re
from functools import cache
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "alomancy"
_DOCS = [_ROOT / "CLAUDE.md", *sorted((_ROOT / "docs" / "architecture").glob("*.md"))]

_PY_REF = re.compile(r"^(?P<path>[\w./-]+\.py)(?:::?(?P<name>[\w:]+))?$")
_DOTTED_REF = re.compile(r"^(?P<module>\w+(?:/\w+)*)\.(?P<name>\w+)$")
_DOC_OR_DIR_REF = re.compile(r"^(?:docs/[\w./-]+\.md|(?:src|tests)/[\w./-]+/)$")
# Results and data files (`results/run_config.yaml`), not modules.
_DATA_SUFFIXES = {"csv", "done", "json", "jsonl", "log", "md", "model", "png", "pth"}
_DATA_SUFFIXES |= {"txt", "xyz", "yaml", "yml"}


@cache
def _py_files() -> tuple[Path, ...]:
    return tuple(
        p
        for base in (_ROOT / "src", _ROOT / "tests")
        for p in base.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _resolve(path: str) -> list[Path]:
    """Every source file *path* can refer to: a file whose path ends with it
    (``mlip/base.py``, ``dft/run_qe.py``, ``alomancy/__init__.py``), or any
    file of that name for a bare file name."""
    return [p for p in _py_files() if p.as_posix().endswith(f"/{path}")]


def _defines(files: list[Path], name: str) -> bool:
    pattern = re.compile(
        rf"^\s*(?:async\s+)?(?:def|class)\s+{name}\b"
        rf"|^\s*(?:self\.)?{name}\s*(?::[^=\n]*)?="
        rf"|^\s*(?:from\s+\S+\s+)?import\s+.*\b{name}\b",
        re.M,
    )
    return any(pattern.search(p.read_text(encoding="utf-8")) for p in files)


def _references() -> list[tuple[str, str]]:
    refs = []
    for doc in _DOCS:
        for token in re.findall(r"`([^`\n]+)`", doc.read_text(encoding="utf-8")):
            refs.append((doc.relative_to(_ROOT).as_posix(), token.removesuffix("()")))
    return refs


def _problem(token: str) -> str | None:
    """Why *token* is a stale reference, or None if it's fine or unchecked."""
    if _DOC_OR_DIR_REF.match(token):
        return None if (_ROOT / token).exists() else "path does not exist"
    if m := _PY_REF.match(token):
        files = _resolve(m["path"])
        if not files:
            return "file does not exist"
        name = m["name"].split(":")[-1] if m["name"] else None
        if name and not _defines(files, name):
            return f"{name} is not defined in {m['path']}"
        return None
    if m := _DOTTED_REF.match(token):
        module, name = m["module"], m["name"]
        if name in _DATA_SUFFIXES:
            return None
        if "/" in module:
            files = _resolve(f"{module}.py") or _resolve(f"{module}/__init__.py")
            if not files:
                return "module does not exist"
        else:
            files = [p for p in _py_files() if p.stem == module and _SRC in p.parents]
            if len(files) != 1:
                return None  # not one of ours (ase.units, job.status, ...)
        if not _defines(files, name):
            return f"{name} is not defined in {module}"
    return None


_REFS = _references()


@pytest.mark.unit
def test_references_are_found():
    assert len(_DOCS) >= 10
    assert sum(_PY_REF.match(token) is not None for _, token in _REFS) > 100


@pytest.mark.unit
def test_architecture_docs_reference_existing_code():
    stale = [
        f"{doc}: `{token}`: {problem}"
        for doc, token in _REFS
        if (problem := _problem(token)) is not None
    ]
    assert stale == [], "\n" + "\n".join(stale)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("mlip/base.py", None),
        ("mlip/base.py:run_training", None),
        ("mlip/base.run_training", None),
        ("md_wfl._run_md_with_spec", None),
        ("ase.units.GPa", None),
        ("docs/architecture/cli.md", None),
        ("mlip/mace/mace_wfl.py", "file does not exist"),
        ("mlip/base.no_such_function", "no_such_function is not defined in mlip/base"),
        ("docs/remote_submission_architecture.md", "path does not exist"),
    ],
)
def test_stale_references_are_caught(token, expected):
    assert _problem(token) == expected
