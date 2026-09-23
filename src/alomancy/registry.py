"""Generic name -> module dispatch for ALomancy's pluggable module categories.

Replaces per-category ad-hoc registries (the precedent being
``high_accuracy_evaluation.dft._CALCULATOR_REGISTRY``) with one shared
mechanism used identically for every category: MLIP trainer, structure
generator, DFT evaluator, initialiser, and any future category (e.g. a
domain-classification module) that wants the same name -> entry-points
dispatch for free.

Registration is added incrementally as each module lands (see
docs/refactor plan) -- ``register()`` never imports anything, so recording
a name before its target module exists is harmless as long as nothing
calls ``resolve()`` for it yet.
"""

import importlib
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, dict[str, dict]] = {}


def register(category: str, name: str, module: str, **entry_points: str) -> None:
    """Record a name -> (module path, entry-point attribute names) mapping.

    Registration is purely string bookkeeping -- it never imports `module`,
    so registering every known backend across all categories costs nothing
    at package-import time (heavy deps like torch/mace/lammps stay lazy,
    only ever imported inside ``resolve()``).

    Calling this twice for the same (category, name) overwrites the earlier
    entry -- deliberate, so a later registration (e.g. from an optional
    plugin package) can override a default without needing a separate API.
    """
    _REGISTRY.setdefault(category, {})[name] = {
        "module": module,
        "entry_points": dict(entry_points),
    }


def resolve(category: str, name: str) -> SimpleNamespace:
    """Lazily import the registered module; return its named entry points
    as attributes on a SimpleNamespace.

    Raises ValueError (never AssertionError -- see CLAUDE.md's guidance on
    guarding config errors, since a bad `category`/`name` combination is
    always ultimately traceable back to user config) naming what *is*
    available, for both an unknown category and an unknown name within a
    known category.
    """
    known_names = _REGISTRY.get(category, {})
    if name not in known_names:
        if category in _REGISTRY:
            raise ValueError(
                f"Unknown {category} {name!r}. Available names: {sorted(known_names)}"
            )
        raise ValueError(
            f"Unknown {category} {name!r}. Available categories: {sorted(_REGISTRY)}"
        )
    entry = known_names[name]
    mod = importlib.import_module(entry["module"])
    return SimpleNamespace(
        **{
            attr: getattr(mod, fn_name)
            for attr, fn_name in entry["entry_points"].items()
        }
    )


def registered(category: str | None = None) -> dict:
    """Introspection helper: what's registered, optionally scoped to one
    category. Returns a shallow copy -- never a live reference to
    ``_REGISTRY`` -- so callers can't mutate registration state through it.
    """
    if category is None:
        return {cat: dict(names) for cat, names in _REGISTRY.items()}
    return dict(_REGISTRY.get(category, {}))
