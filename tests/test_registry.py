"""Tests for alomancy/registry.py -- the generic name -> module dispatch
mechanism replacing per-category ad-hoc registries (the precedent being
high_accuracy_evaluation.dft._CALCULATOR_REGISTRY).

Uses synthetic categories/modules throughout (not the real production
backends) so these tests exercise register()/resolve()'s own mechanics
independently of whichever real modules have landed for any given slice of
the refactor.
"""

import sys
import types

import pytest

from alomancy import registry


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Every test gets its own empty _REGISTRY so registrations in one test
    can never leak into another."""
    monkeypatch.setattr(registry, "_REGISTRY", {})


def _install_fake_module(monkeypatch, name: str, **attrs):
    """Install a throwaway module into sys.modules so resolve() can really
    import it -- exercising the genuine importlib.import_module path rather
    than mocking it away."""
    mod = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.mark.unit
class TestRegister:
    def test_register_does_not_import_the_module(self, monkeypatch):
        """Registration is pure string bookkeeping -- registering a module
        path that doesn't exist (and could never be imported) must not
        raise, so registering every known backend costs nothing even when
        most of their heavy deps (torch/mace/lammps) aren't installed."""
        registry.register(
            "mlip_trainer",
            "definitely_not_a_real_module",
            "no.such.module.anywhere",
            train="train",
        )
        assert "mlip_trainer" in registry.registered()

    def test_re_registering_overwrites(self, monkeypatch):
        _install_fake_module(monkeypatch, "fake_mod_a", value="a")
        _install_fake_module(monkeypatch, "fake_mod_b", value="b")
        registry.register("cat", "x", "fake_mod_a", val="value")
        registry.register("cat", "x", "fake_mod_b", val="value")
        assert registry.resolve("cat", "x").val == "b"


@pytest.mark.unit
class TestResolve:
    def test_resolve_returns_named_entry_points_as_attributes(self, monkeypatch):
        def train(x):
            return x * 2

        def get_calculator(x):
            return x

        _install_fake_module(
            monkeypatch, "fake_trainer_mod", train=train, get_calculator=get_calculator
        )
        registry.register(
            "mlip_trainer",
            "fake",
            "fake_trainer_mod",
            train="train",
            get_calculator="get_calculator",
        )
        resolved = registry.resolve("mlip_trainer", "fake")
        assert resolved.train is train
        assert resolved.get_calculator is get_calculator

    def test_resolve_is_lazy_until_called(self, monkeypatch):
        """register() must never import; only resolve() does. Verified by
        registering a module path that would raise on import and confirming
        register() alone doesn't trigger it."""
        registry.register("cat", "broken", "this.module.does.not.exist")
        with pytest.raises(ModuleNotFoundError):
            registry.resolve("cat", "broken")

    def test_unknown_name_raises_value_error_listing_available_names(self, monkeypatch):
        _install_fake_module(monkeypatch, "fake_mod", fn=lambda: None)
        registry.register("dft_evaluator", "qe", "fake_mod", sp="fn")
        registry.register("dft_evaluator", "vasp", "fake_mod", sp="fn")

        with pytest.raises(ValueError, match=r"Unknown dft_evaluator 'castep'"):
            registry.resolve("dft_evaluator", "castep")
        with pytest.raises(
            ValueError, match=r"Available names.*qe.*vasp|Available names.*vasp.*qe"
        ):
            registry.resolve("dft_evaluator", "castep")

    def test_unknown_category_raises_value_error_listing_available_categories(
        self, monkeypatch
    ):
        registry.register("mlip_trainer", "mace", "fake_mod")
        registry.register("structure_generator", "md", "fake_mod")

        with pytest.raises(ValueError, match=r"Unknown made_up_category 'x'"):
            registry.resolve("made_up_category", "x")
        with pytest.raises(
            ValueError,
            match=r"Available categories.*mlip_trainer.*structure_generator"
            r"|Available categories.*structure_generator.*mlip_trainer",
        ):
            registry.resolve("made_up_category", "x")


@pytest.mark.unit
class TestRegistered:
    def test_registered_scoped_to_one_category(self, monkeypatch):
        registry.register("mlip_trainer", "mace", "fake_mod_1")
        registry.register("structure_generator", "md", "fake_mod_2")
        assert set(registry.registered("mlip_trainer")) == {"mace"}
        assert set(registry.registered("structure_generator")) == {"md"}

    def test_registered_unknown_category_returns_empty_dict(self):
        assert registry.registered("nothing_registered_here") == {}

    def test_registered_returns_a_copy_not_a_live_reference(self, monkeypatch):
        registry.register("mlip_trainer", "mace", "fake_mod")
        snapshot = registry.registered("mlip_trainer")
        snapshot["injected"] = {"module": "should.not.leak", "entry_points": {}}
        assert "injected" not in registry.registered("mlip_trainer")

    def test_registered_with_no_category_lists_everything(self, monkeypatch):
        registry.register("mlip_trainer", "mace", "fake_mod_1")
        registry.register("structure_generator", "md", "fake_mod_2")
        everything = registry.registered()
        assert set(everything) == {"mlip_trainer", "structure_generator"}
        assert set(everything["mlip_trainer"]) == {"mace"}
