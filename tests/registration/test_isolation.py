"""Tests for registry isolation: ``conscribe.testing.isolated_registries()``.

Covers:
- af acceptance: fakes reach build_composed_config and exit restores the
  exact entry state; two nested levels restore level by level;
  ``unregister(..., missing_ok=True)`` twice
- exit rule: rollback of test-body classes, removals, registrars created in
  the block (shadowed names restored), shadowed registries, exceptions
  (BaseException included)
- keep rule: modules first imported inside a block keep their registrations
  (later blocks still see them); last import-time write wins; entry state
  wins; failed imports and modules dropped from sys.modules are rolled back;
  nested blocks hand survivors outward; warn_kept
- register() attribute writes on classes that predate the block
- capture / activate: net-diff replay (adds, removals, replacements,
  registrars, attribute writes), failed-enter unwinding, re-entering a capture
- LIFO enforcement, the discover() failure-record reset
- the opt-in pytest plugin (fixture, ini option, interleaved selections)
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from conscribe import build_composed_config, create_registrar
from conscribe.exceptions import (
    ConscribeWarning,
    DuplicateKeyError,
    IsolationWarning,
    KeyNotFoundError,
    WiringResolutionError,
)
from conscribe.registration.registry import (
    _ISOLATION_STACK,
    _JOURNAL,
    _REGISTRY_INDEX,
    _deregister,
    get_registry,
)
from conscribe.testing import RegistryIsolation, isolated_registries
from conscribe.wiring import WiredField

from . import _iso_shapes as shapes
from ._iso_shapes import (
    Act,
    ActionBase,
    Fast,
    Owner,
    OwnerBase,
    Plain,
    Strat,
    StrategyBase,
    StrategyProto,
)


def _state() -> dict[str, dict[str, type]]:
    """Exact state of every indexed registry: name -> {key: class}."""
    return {name: dict(reg._store) for name, reg in sorted(_REGISTRY_INDEX.items())}


def _index() -> dict:
    """The global index, name -> registry object (compared by identity)."""
    return dict(_REGISTRY_INDEX)


class _Abort(BaseException):
    """A non-Exception error, like KeyboardInterrupt, without pytest side effects."""


@pytest.fixture
def tmp_pkg(conscribe_isolated, tmp_path, monkeypatch):
    """An importable throwaway package whose modules a test imports lazily.

    Its modules leave ``sys.modules`` on teardown, *before* the enclosing
    ``conscribe_isolated`` block exits (fixture dependency order), so
    whatever they registered is rolled back for the tests that follow.
    """
    name = f"isopkg_{uuid.uuid4().hex[:10]}"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(module: str, source: str) -> str:
        code = textwrap.dedent(source).replace("SHAPES", shapes.__name__)
        (root / f"{module}.py").write_text(code)
        return f"{name}.{module}"

    yield SimpleNamespace(name=name, write=write)
    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[module]


# ===================================================================
# af acceptance
# ===================================================================


class TestAfAcceptance:
    def test_fake_reaches_composed_config_and_exit_restores_exact_state(self) -> None:
        entry, entry_index = _state(), _index()

        with isolated_registries() as iso:
            class FakeDone(ActionBase):
                __registry_key__ = "fake.done"

            class FakeOwner(OwnerBase):
                __wiring__ = {"action": ("iso_act", ["done"])}

                def __init__(self, action: str = "fake.done") -> None:
                    self.action = action

            result = build_composed_config({"iso_owner": Owner, "iso_act": Act})
            assert set(result.layer_results["iso_act"].per_key_models) == {
                "browser.click", "fake.done",
            }
            assert set(result.layer_results["iso_owner"].per_key_models) == {
                "good", "fake_owner",
            }
            config = result.top_level_type.model_validate(
                {"iso_owner": [{"name": "fake_owner", "action": {"name": "fake.done"}}]}
            )
            assert config.iso_owner[0].action.name == "fake.done"

        assert _state() == entry
        assert _index() == entry_index
        assert {(r, k) for r, k, _ in iso.rolled_back} == {
            ("iso_act", "fake.done"), ("iso_owner", "fake_owner"),
        }
        assert iso.kept == []

    def test_two_nested_levels_restore_level_by_level(self) -> None:
        entry = _state()
        with isolated_registries():
            class LevelOne(StrategyBase):
                pass

            level_one = _state()
            with isolated_registries():
                class LevelTwo(StrategyBase):
                    pass

                Strat.unregister("fast")
                assert set(Strat.keys()) == {"level_one", "level_two"}
            assert _state() == level_one
            assert Strat.get("fast") is Fast
        assert _state() == entry

    def test_unregister_missing_ok_twice_does_not_raise(self) -> None:
        with isolated_registries():
            class Once(StrategyBase):
                pass

            assert Strat.unregister("once", missing_ok=True) is Once
            assert Strat.unregister("once", missing_ok=True) is None
            assert Strat.unregister("once", missing_ok=True) is None
            with pytest.raises(KeyNotFoundError):
                Strat.unregister("once")


# ===================================================================
# Exit rule: what is rolled back
# ===================================================================


class TestRollback:
    def test_test_body_and_helper_function_classes_rolled_back(self) -> None:
        entry = _state()
        with isolated_registries() as iso:
            class BodyFake(StrategyBase):
                pass

            made = shapes.make_strategy("helper_made")
        assert _state() == entry
        assert ("iso_strategy", "body_fake", BodyFake) in iso.rolled_back
        assert ("iso_strategy", "helper_made", made) in iso.rolled_back

    def test_removal_and_replacement_of_real_key_restored(self) -> None:
        with isolated_registries():
            assert Strat.unregister("fast") is Fast

            class FakeFast(StrategyBase):
                __registry_key__ = "fast"

            assert Strat.get("fast") is FakeFast
        assert Strat.get("fast") is Fast

    def test_duplicate_key_still_raises_inside_block(self) -> None:
        with isolated_registries():
            with pytest.raises(DuplicateKeyError):
                class Fast(StrategyBase):  # noqa: F811 - the point of the test
                    pass

    def test_registrar_created_in_block_dropped_and_shadowed_name_restored(self) -> None:
        real = get_registry("iso_strategy")
        with isolated_registries() as iso:
            temp = create_registrar("iso_block_layer", StrategyProto)
            shadow = create_registrar("iso_strategy", StrategyProto)
            assert get_registry("iso_strategy") is shadow._registry

            class ShadowBase(metaclass=shadow.Meta):
                __abstract__ = True

                def plan(self) -> str:
                    return ""

            class InShadow(ShadowBase):
                pass

            assert get_registry("iso_block_layer") is temp._registry

        assert get_registry("iso_block_layer") is None
        assert get_registry("iso_strategy") is real
        assert shadow.keys() == []  # a registry born in the block ends empty
        assert ("iso_strategy", "in_shadow", InShadow) in iso.rolled_back

    def test_registry_shadowed_out_of_index_is_still_restored(self) -> None:
        with isolated_registries():
            hidden = create_registrar("iso_hidden", StrategyProto)

            class HiddenBase(metaclass=hidden.Meta):
                __abstract__ = True

                def plan(self) -> str:
                    return ""

            create_registrar("iso_hidden", StrategyProto)  # hidden leaves the index
            assert get_registry("iso_hidden") is not hidden._registry

            with isolated_registries():
                class Inside(HiddenBase):
                    pass

                assert hidden.keys() == ["inside"]
            assert hidden.keys() == []

    def test_deregister_undone(self) -> None:
        real = get_registry("iso_act")
        with isolated_registries():
            _deregister("iso_act")
            assert get_registry("iso_act") is None
        assert get_registry("iso_act") is real

    def test_exception_restores_and_propagates(self) -> None:
        entry = _state()
        with pytest.raises(ValueError, match="inside"):
            with isolated_registries():
                class Doomed(StrategyBase):
                    pass

                raise ValueError("raised inside")
        assert _state() == entry

    def test_base_exception_restores_and_propagates(self) -> None:
        entry = _state()
        with pytest.raises(_Abort):
            with isolated_registries():
                class Doomed(StrategyBase):
                    pass

                Strat.unregister("fast")
                raise _Abort()
        assert _state() == entry

    def test_journal_is_empty_after_outermost_block(self) -> None:
        depth = len(_ISOLATION_STACK)
        with isolated_registries():
            class Journaled(StrategyBase):
                pass

            assert _JOURNAL
        assert len(_ISOLATION_STACK) == depth
        if depth == 0:
            assert _JOURNAL == []

    def test_restore_keeps_dict_identity(self) -> None:
        store = Strat._registry._store
        index = _REGISTRY_INDEX
        with isolated_registries():
            class Identity(StrategyBase):
                pass

            create_registrar("iso_identity_layer", StrategyProto)
        assert Strat._registry._store is store
        assert _REGISTRY_INDEX is index

    def test_negative_protocol_cache_invalidated_by_rollback(self) -> None:
        counter = Strat._registry._invalidation_counter
        with isolated_registries():
            class Counted(StrategyBase):
                pass
        assert Strat._registry._invalidation_counter > counter

    def test_entry_key_order_restored(self) -> None:
        with isolated_registries():
            for key in ("first", "second"):
                shapes.make_strategy(key)
            order = Strat.keys()
            with isolated_registries():
                Strat.unregister("fast")

                class Fast(StrategyBase):  # noqa: F811 - re-added at the end
                    pass

                assert Strat.keys()[-1] == "fast"
            assert Strat.keys() == order

    def test_registration_from_another_thread_is_rolled_back(self) -> None:
        import threading

        made: list[type] = []
        with isolated_registries() as iso:
            worker = threading.Thread(target=lambda: made.append(shapes.make_strategy("threaded")))
            worker.start()
            worker.join()
            assert Strat.get("threaded") is made[0]
        assert "threaded" not in Strat.keys()
        assert iso.kept == []

    def test_exec_with_unimported_module_name_is_rolled_back(self) -> None:
        source = "from {0} import StrategyBase\nclass Execd(StrategyBase):\n    pass\n"
        with isolated_registries() as iso:
            exec(source.format(shapes.__name__), {"__name__": "never_imported_mod"})
            assert "execd" in Strat.keys()
        assert "execd" not in Strat.keys()
        assert iso.kept == []


# ===================================================================
# Keep rule: modules first imported inside the block
# ===================================================================


class TestKeepRule:
    def test_module_first_imported_in_block_keeps_registrations(self, tmp_pkg) -> None:
        lazy = tmp_pkg.write("lazy_layer", """
            from SHAPES import StrategyProto
            from conscribe import create_registrar

            Lazy = create_registrar("iso_lazy_layer", StrategyProto)

            class LazyBase(metaclass=Lazy.Meta):
                __abstract__ = True

                def plan(self) -> str:
                    return ""

            class Slow(LazyBase):
                pass
        """)
        extends = tmp_pkg.write("extends", """
            from SHAPES import ActionBase, StrategyBase

            class Turbo(StrategyBase):
                pass

            class Kick(ActionBase):
                __registry_key__ = "browser.kick"
        """)

        with isolated_registries() as iso:
            importlib.import_module(lazy)
            importlib.import_module(extends)

            class BodyFake(StrategyBase):
                pass

        lazy_mod, ext_mod = sys.modules[lazy], sys.modules[extends]
        assert get_registry("iso_lazy_layer") is lazy_mod.Lazy._registry
        assert lazy_mod.Lazy.get("slow") is lazy_mod.Slow
        assert Strat.get("turbo") is ext_mod.Turbo
        assert Act.get("browser.kick") is ext_mod.Kick
        assert "body_fake" not in Strat.keys()
        assert sorted(iso.kept, key=str) == sorted([
            ("iso_lazy_layer", "slow", lazy_mod.Slow),
            ("iso_strategy", "turbo", ext_mod.Turbo),
            ("iso_act", "browser.kick", ext_mod.Kick),
        ], key=str)
        assert iso.rolled_back == [("iso_strategy", "body_fake", BodyFake)]

    def test_later_blocks_still_see_kept_registrations(self, tmp_pkg) -> None:
        """The hazard: a naive restore orphans the lazily imported layer."""
        lazy = tmp_pkg.write("lazy_owner", """
            from SHAPES import OwnerBase

            class Kicker(OwnerBase):
                __wiring__ = {"action": ("iso_act", ["click"])}

                def __init__(self, action: str = "browser.click") -> None:
                    self.action = action
        """)
        with isolated_registries():  # "test 1"
            importlib.import_module(lazy)

        with isolated_registries():  # "test 2"
            assert Owner.get("kicker") is sys.modules[lazy].Kicker
            shapes.make_strategy("again")  # a helper fake works in every test
            result = build_composed_config({"iso_owner": Owner, "iso_act": Act})
            assert "kicker" in result.layer_results["iso_owner"].per_key_models
        with isolated_registries():  # "test 3"
            shapes.make_strategy("again")

    def test_registration_through_factory_in_older_module_is_kept(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("uses_factory", """
            from SHAPES import make_strategy

            Made = make_strategy("factory_made")
        """)
        with isolated_registries() as iso:
            importlib.import_module(mod)
        assert Strat.get("factory_made") is sys.modules[mod].Made
        assert iso.kept == [("iso_strategy", "factory_made", sys.modules[mod].Made)]

    def test_code_a_new_module_execs_counts_as_its_import(self, tmp_pkg) -> None:
        """Registrations are attributed to the module whose top-level code
        runs: a plugin loader exec()-ing generated classes with a synthetic
        module name (never imported) is part of the loader's import."""
        mod = tmp_pkg.write("loader", """
            from SHAPES import StrategyBase

            namespace = {"StrategyBase": StrategyBase, "__name__": "never_imported_plugins"}
            exec("class Execd(StrategyBase):\\n    __registry_key__ = 'execd_by_loader'\\n", namespace)
            Generated = namespace["Execd"]
        """)
        with isolated_registries() as iso:
            importlib.import_module(mod)
        generated = sys.modules[mod].Generated
        assert Strat.get("execd_by_loader") is generated
        assert iso.kept == [("iso_strategy", "execd_by_loader", generated)]

    def test_failed_import_rolls_back_partial_registrations(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("broken", """
            from SHAPES import StrategyBase

            class Partial(StrategyBase):
                pass

            raise RuntimeError("boom at import")
        """)
        with isolated_registries() as iso:
            with pytest.raises(RuntimeError, match="boom"):
                importlib.import_module(mod)
        assert "partial" not in Strat.keys()
        assert iso.kept == []
        # A retry reports the original error, not a DuplicateKeyError.
        with isolated_registries():
            with pytest.raises(RuntimeError, match="boom"):
                importlib.import_module(mod)

    def test_last_import_time_write_wins(self, tmp_pkg) -> None:
        """dataclass(slots=True) registers, then rebinds to the final class."""
        mod = tmp_pkg.write("slotted", """
            from dataclasses import dataclass

            from SHAPES import StrategyBase

            @dataclass(slots=True)
            class Slotted(StrategyBase):
                x: int = 0
        """)
        with isolated_registries() as iso:
            importlib.import_module(mod)
        final = sys.modules[mod].Slotted
        assert "__slots__" in final.__dict__
        assert Strat.get("slotted") is final
        assert iso.kept == [("iso_strategy", "slotted", final)]

    def test_import_time_add_and_removal_cancel_out(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("adds_and_removes", """
            from SHAPES import Strat, StrategyBase

            class Temporary(StrategyBase):
                pass

            Strat.unregister("temporary")
        """)
        with isolated_registries() as iso:
            importlib.import_module(mod)
        assert "temporary" not in Strat.keys()
        assert iso.kept == []

    def test_entry_key_wins_over_import_time_write(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("override", """
            from SHAPES import StrategyBase

            class Fast(StrategyBase):
                pass
        """)
        with isolated_registries() as iso:
            Strat.unregister("fast")
            importlib.import_module(mod)
            assert Strat.get("fast") is sys.modules[mod].Fast
        assert Strat.get("fast") is Fast
        assert iso.kept == []
        assert ("iso_strategy", "fast", sys.modules[mod].Fast) in iso.rolled_back

    def test_block_time_removal_of_kept_key_undone(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("removed_later", """
            from SHAPES import StrategyBase

            class Durable(StrategyBase):
                pass
        """)
        with isolated_registries():
            importlib.import_module(mod)
            Strat.unregister("durable")
        assert Strat.get("durable") is sys.modules[mod].Durable

    def test_module_dropped_from_sys_modules_is_not_kept(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("dropped", """
            from SHAPES import StrategyBase

            class Dropped(StrategyBase):
                pass
        """)
        with isolated_registries() as iso:
            importlib.import_module(mod)
            del sys.modules[mod]
        assert "dropped" not in Strat.keys()
        assert iso.kept == []

    def test_module_imported_before_block_is_not_import_time(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("early", """
            from SHAPES import make_strategy

            def define_later():
                return make_strategy("defined_later")
        """)
        early = importlib.import_module(mod)
        with isolated_registries() as iso:
            early.define_later()
        assert "defined_later" not in Strat.keys()
        assert iso.kept == []

    def test_import_in_inner_block_survives_outer_exit(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("inner_import", """
            from SHAPES import StrategyBase

            class Nested(StrategyBase):
                pass
        """)
        with isolated_registries() as outer:
            with isolated_registries() as inner:
                importlib.import_module(mod)
            assert inner.kept == [("iso_strategy", "nested", sys.modules[mod].Nested)]

            class OuterFake(StrategyBase):
                pass

        assert Strat.get("nested") is sys.modules[mod].Nested
        assert outer.kept == [("iso_strategy", "nested", sys.modules[mod].Nested)]
        assert "outer_fake" not in Strat.keys()

    def test_registry_created_by_imported_module_takes_its_name(self, tmp_pkg) -> None:
        leaked = create_registrar("iso_dup_name", StrategyProto)
        mod = tmp_pkg.write("dup_name", """
            from SHAPES import StrategyProto
            from conscribe import create_registrar

            Real = create_registrar("iso_dup_name", StrategyProto)
        """)
        with isolated_registries():
            importlib.import_module(mod)
        assert get_registry("iso_dup_name") is sys.modules[mod].Real._registry
        assert get_registry("iso_dup_name") is not leaked._registry

    def test_import_time_index_removal_is_kept(self, tmp_pkg) -> None:
        create_registrar("iso_victim", StrategyProto)
        mod = tmp_pkg.write("deregisters", """
            from conscribe.registration.registry import _deregister

            _deregister("iso_victim")
        """)
        with isolated_registries():
            importlib.import_module(mod)
        assert get_registry("iso_victim") is None

    def test_helper_module_capturing_its_fakes_leaves_nothing_behind(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("helper_fakes", """
            from SHAPES import StrategyBase
            from conscribe.testing import isolated_registries

            FAKES = isolated_registries()
            with FAKES:
                class HelperFake(StrategyBase):
                    pass
        """)
        with isolated_registries() as iso:
            helpers = importlib.import_module(mod)
            assert "helper_fake" not in Strat.keys()
        assert "helper_fake" not in Strat.keys()
        assert iso.kept == []
        with helpers.FAKES.activate():
            assert Strat.get("helper_fake") is helpers.HelperFake
        assert "helper_fake" not in Strat.keys()

    def test_warn_kept_names_the_modules(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("warned", """
            from SHAPES import StrategyBase

            class Warned(StrategyBase):
                pass
        """)
        with pytest.warns(IsolationWarning, match=r"kept 1 registration") as record:
            with isolated_registries(warn_kept=True):
                importlib.import_module(mod)
        assert issubclass(IsolationWarning, ConscribeWarning)
        assert record[0].category is IsolationWarning
        message = str(record[0].message)
        assert mod in message
        assert "iso_strategy:warned" in message
        assert "FAKES.activate()" in message

    def test_no_warning_by_default(self, tmp_pkg, recwarn) -> None:
        mod = tmp_pkg.write("silent", """
            from SHAPES import StrategyBase

            class Silent(StrategyBase):
                pass
        """)
        with isolated_registries():
            importlib.import_module(mod)
        assert not [w for w in recwarn if issubclass(w.category, ConscribeWarning)]


# ===================================================================
# register() attribute writes
# ===================================================================


class TestRegisterAttributeWrites:
    def test_register_on_preexisting_class_is_undone(self) -> None:
        with isolated_registries() as iso:
            Strat.register("plain")(Plain)
            assert Strat.get("plain") is Plain
            assert Plain.__dict__["__registry_key__"] == "plain"
            assert isinstance(Plain.__dict__["helper"], WiredField)
        assert "plain" not in Strat.keys()
        for attr in ("__registry_key__", "__registry_origin__", "helper"):
            assert attr not in Plain.__dict__
        assert iso.rolled_back == [("iso_strategy", "plain", Plain)]

    def test_register_twice_restores_original_attributes(self) -> None:
        class Twice:
            def plan(self) -> str:
                return ""

        with isolated_registries():
            Strat.register("twice_a")(Twice)
            original = dict(Twice.__dict__)
            with isolated_registries():
                Strat.register("twice_b")(Twice)
                Act_like = create_registrar("iso_twice_other", StrategyProto)
                Act_like.register("elsewhere")(Twice)
                assert Twice.__dict__["__registry_key__"] == "elsewhere"
            assert dict(Twice.__dict__) == original
            assert Strat.get("twice_a") is Twice
            assert "twice_b" not in Strat.keys()
        assert "__registry_key__" not in Twice.__dict__

    def test_register_propagate_hook_removed_after_block(self) -> None:
        class Root:
            def plan(self) -> str:
                return ""

        with isolated_registries():
            Strat.register("iso_root", propagate=True)(Root)

            class Child(Root):
                pass

            assert Strat.get("child") is Child
        assert "__init_subclass__" not in Root.__dict__

        class Later(Root):
            pass

        assert "later" not in Strat.keys()

    def test_register_at_import_time_keeps_attribute_writes(self, tmp_pkg) -> None:
        mod = tmp_pkg.write("registers", """
            from SHAPES import Strat

            @Strat.register("decorated")
            class Decorated:
                def plan(self) -> str:
                    return ""
        """)
        with isolated_registries():
            importlib.import_module(mod)
        decorated = sys.modules[mod].Decorated
        assert Strat.get("decorated") is decorated
        assert decorated.__dict__["__registry_key__"] == "decorated"


# ===================================================================
# Capture / activate
# ===================================================================


class TestCaptureActivate:
    def test_capture_rolls_back_and_activate_reapplies(self) -> None:
        entry = _state()
        fakes = isolated_registries()
        with fakes:
            class Captured(StrategyBase):
                pass

        assert _state() == entry
        for _ in range(2):  # repeatable
            with fakes.activate() as activation:
                assert Strat.get("captured") is Captured
            assert _state() == entry
            assert activation.rolled_back == [("iso_strategy", "captured", Captured)]

    def test_activate_reapplies_removals_and_replacements(self) -> None:
        fakes = isolated_registries()
        with fakes:
            Strat.unregister("fast")

            class FakeFast(StrategyBase):
                __registry_key__ = "fast"

            Act.unregister("browser.click")

        with fakes.activate():
            assert Strat.get("fast") is FakeFast
            assert "browser.click" not in Act.keys()
        assert Strat.get("fast") is Fast
        assert Act.get("browser.click") is shapes.BrowserClick

    def test_activate_reapplies_registrars_created_in_capture(self) -> None:
        fakes = isolated_registries()
        with fakes:
            fake_layer = create_registrar("iso_fake_layer", StrategyProto)

            class FakeLayerBase(metaclass=fake_layer.Meta):
                __abstract__ = True

                def plan(self) -> str:
                    return ""

            class FakeImpl(FakeLayerBase):
                pass

        assert get_registry("iso_fake_layer") is None
        with fakes.activate():
            assert get_registry("iso_fake_layer") is fake_layer._registry
            assert fake_layer.get("fake_impl") is FakeImpl
        assert get_registry("iso_fake_layer") is None
        assert fake_layer.keys() == []

    def test_activate_reapplies_register_attribute_writes(self) -> None:
        fakes = isolated_registries()
        with fakes:
            Strat.register("plain")(Plain)
        assert "__registry_key__" not in Plain.__dict__
        with fakes.activate():
            assert Strat.get("plain") is Plain
            assert Plain.__dict__["__registry_key__"] == "plain"
            assert isinstance(Plain.__dict__["helper"], WiredField)
        assert "__registry_key__" not in Plain.__dict__
        assert "helper" not in Plain.__dict__

    def test_activation_undoes_its_writes_when_binding_existed_at_entry(self) -> None:
        """Regression: a binding present at entry must not keep the writes
        the activation re-applied for it."""
        class Twice:
            def plan(self) -> str:
                return ""

        fakes = isolated_registries()
        with fakes:
            Strat.register("dup_bind")(Twice)

        with isolated_registries():
            Strat.register("dup_bind")(Twice)  # the same binding, made here
            attrs_before = dict(Twice.__dict__)
            with fakes.activate():
                assert Strat.get("dup_bind") is Twice
            assert all(Twice.__dict__[name] is value for name, value in attrs_before.items())
        assert "__registry_key__" not in Twice.__dict__
        assert "__registry_origin__" not in Twice.__dict__

    def test_activate_conflict_raises_and_unwinds(self) -> None:
        fakes = isolated_registries()
        with fakes:
            class FakeA(StrategyBase):
                pass

            Strat.unregister("fast")

            class FakeFast(StrategyBase):
                __registry_key__ = "fast"

        with isolated_registries():
            Strat.unregister("fast")

            class Usurper(StrategyBase):
                __registry_key__ = "fast"

            state, depth = _state(), len(_ISOLATION_STACK)
            with pytest.raises(DuplicateKeyError, match="activate") as info:
                with fakes.activate():
                    pass  # pragma: no cover - never reached
            assert info.value.existing_cls is Usurper
            assert info.value.new_cls is FakeFast
            assert _state() == state  # fake_a, applied before the conflict, is undone
            assert len(_ISOLATION_STACK) == depth

    def test_activate_reapplies_index_removal(self) -> None:
        real = get_registry("iso_act")
        fakes = isolated_registries()
        with fakes:
            _deregister("iso_act")

        with fakes.activate():
            assert get_registry("iso_act") is None
        assert get_registry("iso_act") is real

        with isolated_registries():
            replacement = create_registrar("iso_act", shapes.ActionProto)
            with fakes.activate():  # the name now belongs to another registry
                assert get_registry("iso_act") is replacement._registry
        assert get_registry("iso_act") is real

    def test_activate_skips_removal_already_applied(self) -> None:
        fakes = isolated_registries()
        with fakes:
            Strat.unregister("fast")

        with isolated_registries():
            Strat.unregister("fast")
            with fakes.activate() as activation:
                assert "fast" not in Strat.keys()
            assert activation.rolled_back == []
        assert Strat.get("fast") is Fast

    def test_activate_removal_conflict_names_the_captured_removal(self) -> None:
        fakes = isolated_registries()
        with fakes:
            Strat.unregister("fast")

        with isolated_registries():
            Strat.unregister("fast")

            class Usurper(StrategyBase):
                __registry_key__ = "fast"

            with pytest.raises(DuplicateKeyError, match="removed 'fast'"):
                with fakes.activate():
                    pass  # pragma: no cover - never reached
            assert Strat.get("fast") is Usurper

    def test_atomic_purge_inside_block_is_rolled_back_cleanly(self) -> None:
        entry = _state()
        with isolated_registries() as iso:
            with pytest.raises(DuplicateKeyError):
                class Partial(StrategyBase):
                    __registry_key__ = ["partial_fresh", "fast"]

            assert "partial_fresh" not in Strat.keys()
        assert _state() == entry
        assert iso.rolled_back == []

    def test_failed_register_inside_block_is_rolled_back_cleanly(self, monkeypatch) -> None:
        import conscribe.wiring

        def failing_inject(cls: type) -> None:
            raise RuntimeError("inject failed")

        entry = _state()
        with isolated_registries() as iso:
            monkeypatch.setattr(conscribe.wiring, "inject_wired_descriptors", failing_inject)
            with pytest.raises(RuntimeError, match="inject failed"):
                Strat.register("plain")(Plain)
            monkeypatch.undo()
            assert "plain" not in Strat.keys()
        assert _state() == entry
        assert iso.rolled_back == []
        assert "__registry_key__" not in Plain.__dict__

    def test_reentering_capture_keeps_and_extends_it(self) -> None:
        fakes = isolated_registries()
        with fakes:
            class FakeOne(StrategyBase):
                pass

        with fakes:
            assert Strat.get("fake_one") is FakeOne

            class FakeTwo(StrategyBase):
                pass

        assert {"fake_one", "fake_two"}.isdisjoint(Strat.keys())
        with fakes.activate():
            assert Strat.get("fake_one") is FakeOne
            assert Strat.get("fake_two") is FakeTwo

    def test_failed_reentry_keeps_previous_capture(self) -> None:
        fakes = isolated_registries()
        with fakes:
            class Keeper(StrategyBase):
                pass

        with isolated_registries():
            Strat.register("keeper")(Plain)
            with pytest.raises(DuplicateKeyError):
                with fakes:
                    pass  # pragma: no cover - never reached
        with fakes.activate():
            assert Strat.get("keeper") is Keeper

    def test_activate_guards(self) -> None:
        with pytest.raises(RuntimeError, match="nothing captured"):
            with isolated_registries().activate():
                pass  # pragma: no cover

        fakes = isolated_registries()
        with fakes:
            with pytest.raises(RuntimeError, match="still open"):
                with fakes.activate():
                    pass  # pragma: no cover

        with fakes.activate() as activation:
            with pytest.raises(RuntimeError, match="activate"):
                with fakes:
                    pass  # pragma: no cover
            with pytest.raises(RuntimeError, match="activation"):
                activation.activate()

    def test_interleaved_module_level_fakes(self, pytester) -> None:
        """a::t1, b::t, a::t2 — where guarded teardown_module loses the fake."""
        _write_inner_layer(pytester)
        pytester.makepyfile(
            test_a="""
                import pytest
                from conscribe.testing import isolated_registries
                from inner_layer import Loop, LoopBase

                FAKES = isolated_registries()
                with FAKES:
                    class ModFake(LoopBase):
                        __registry_key__ = "mod_fake"

                @pytest.fixture(autouse=True)
                def _fakes():
                    with FAKES.activate():
                        yield

                def test_t1():
                    assert Loop.get("mod_fake") is ModFake

                def test_t2():
                    assert Loop.get("mod_fake") is ModFake
            """,
            test_b="""
                from inner_layer import Loop

                def test_b():
                    assert Loop.get_or_none("mod_fake") is None
            """,
        )
        with isolated_registries():
            result = pytester.runpytest(
                "test_a.py::test_t1", "test_b.py::test_b", "test_a.py::test_t2",
                "-p", "no:cacheprovider",
            )
        result.assert_outcomes(passed=3)


# ===================================================================
# Nesting discipline
# ===================================================================


class TestNesting:
    def test_block_is_not_reentrant(self) -> None:
        iso = isolated_registries()
        with iso:
            with pytest.raises(RuntimeError, match="already active"):
                iso.__enter__()
        assert not iso.active

    def test_out_of_order_exit_unwinds_restores_and_raises(self) -> None:
        entry, depth = _state(), len(_ISOLATION_STACK)
        outer, inner = isolated_registries(), isolated_registries()
        outer.__enter__()

        class OuterFake(StrategyBase):
            pass

        inner.__enter__()

        class InnerFake(StrategyBase):
            pass

        with pytest.raises(RuntimeError, match="LIFO"):
            outer.__exit__(None, None, None)
        assert _state() == entry
        assert len(_ISOLATION_STACK) == depth
        with pytest.raises(RuntimeError, match="already closed"):
            inner.__exit__(None, None, None)
        with pytest.raises(RuntimeError, match="not active"):
            inner.__exit__(None, None, None)

    def test_exit_of_inactive_block_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not active"):
            isolated_registries().__exit__(None, None, None)

    def test_repr_and_types(self) -> None:
        iso = isolated_registries()
        assert isinstance(iso, RegistryIsolation)
        assert "inactive" in repr(iso)
        with iso:
            assert "active" in repr(iso)
        with pytest.raises(TypeError, match="warn_kept"):
            isolated_registries(warn_kept="yes")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            isolated_registries(Strat)  # type: ignore[call-arg]


# ===================================================================
# discover() failure record
# ===================================================================


class TestDiscoverReset:
    @pytest.fixture
    def failures(self):
        from conscribe.discover import _record_failure, clear_import_failures, import_failures

        clear_import_failures()
        yield _record_failure, import_failures
        clear_import_failures()

    def test_exit_restores_the_failure_record_of_entry(self, failures) -> None:
        """Review probe 18: failures recorded before the block (a session-level
        discover() in conftest) survive it; those recorded inside are dropped."""
        record, import_failures = failures
        record("iso_before_block.envs", "ModuleNotFoundError: No module named 'dep'")

        with isolated_registries():
            record("iso_inside_block.mod", "SyntaxError: bad")
            with isolated_registries():
                record("iso_inner_block.mod", "RuntimeError: boom")
            assert set(import_failures()) == {"iso_before_block.envs", "iso_inside_block.mod"}
        assert import_failures() == {
            "iso_before_block.envs": "ModuleNotFoundError: No module named 'dep'"
        }

    def test_wiring_errors_after_a_block_still_cite_earlier_failures(self, failures) -> None:
        from conscribe.wiring import resolve_wiring

        record, _ = failures

        class Owner:
            __wiring__ = {"env": "iso_failure_env"}

        with isolated_registries():  # keeps the registrar out of other tests
            create_registrar("iso_failure_env", StrategyProto)
            record("iso_failure_pkg.envs", "ModuleNotFoundError: No module named 'dep'")
            with isolated_registries():  # e.g. one test using conscribe_isolated
                pass
            with pytest.raises(
                WiringResolutionError, match=r"iso_failure_pkg\.envs \(ModuleNotFoundError"
            ):
                resolve_wiring(Owner)


# ===================================================================
# pytest plugin
# ===================================================================


def _write_inner_layer(pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["conscribe.testing.pytest_plugin"]\n')
    pytester.makepyfile(
        inner_layer="""
            from typing import Protocol, runtime_checkable

            from conscribe import create_registrar

            @runtime_checkable
            class LoopProto(Protocol):
                def run(self) -> str: ...

            Loop = create_registrar("iso_inner_loop", LoopProto)

            class LoopBase(metaclass=Loop.Meta):
                __abstract__ = True

                def run(self) -> str:
                    return ""
        """,
    )


class TestPytestPlugin:
    def test_fixture_rolls_back_between_tests(self, conscribe_isolated) -> None:
        assert isinstance(conscribe_isolated, RegistryIsolation)
        assert conscribe_isolated.active

        class FixtureFake(StrategyBase):
            pass

        assert Strat.get("fixture_fake") is FixtureFake

    def test_previous_fixture_fake_is_gone(self) -> None:
        assert "fixture_fake" not in Strat.keys()

    def test_autouse_wrapper_with_lazy_first_import(self, pytester) -> None:
        _write_inner_layer(pytester)
        pytester.makepyfile(
            lazy_real="""
                from inner_layer import LoopBase

                class LazyImpl(LoopBase):
                    pass
            """,
            test_lazy="""
                import importlib
                import pytest
                from inner_layer import Loop, LoopBase

                @pytest.fixture(autouse=True)
                def _iso(conscribe_isolated):
                    yield conscribe_isolated

                def test_1():
                    importlib.import_module("lazy_real")

                    class BodyFake(LoopBase):
                        pass

                def test_2():
                    assert Loop.get("lazy_impl").__module__ == "lazy_real"
                    assert Loop.get_or_none("body_fake") is None
            """,
        )
        with isolated_registries():
            result = pytester.runpytest("-p", "no:cacheprovider")
        result.assert_outcomes(passed=2)

    def test_warn_kept_ini_option(self, pytester) -> None:
        _write_inner_layer(pytester)
        pytester.makeini("[pytest]\nconscribe_warn_kept = true\n")
        pytester.makepyfile(
            lazy_warned="""
                from inner_layer import LoopBase

                class Warned(LoopBase):
                    pass
            """,
            test_warn="""
                import importlib

                def test_import(conscribe_isolated):
                    importlib.import_module("lazy_warned")
            """,
        )
        with isolated_registries():
            # -W default: an outer ``-W error`` must not turn the warning under
            # test into an error of the inner run.
            result = pytester.runpytest("-p", "no:cacheprovider", "-W", "default")
        result.assert_outcomes(passed=1, warnings=1)
        result.stdout.fnmatch_lines(["*isolated_registries() kept 1 registration*lazy_warned*"])

    def test_plugin_loads_from_root_conftest_in_a_fresh_process(self, pytester) -> None:
        _write_inner_layer(pytester)
        pytester.makepyfile(
            test_fresh="""
                from conscribe.testing import RegistryIsolation
                from inner_layer import Loop, LoopBase

                def test_one(conscribe_isolated):
                    assert isinstance(conscribe_isolated, RegistryIsolation)

                    class Fresh(LoopBase):
                        pass

                    assert Loop.keys() == ["fresh"]

                def test_two():
                    assert Loop.keys() == []
            """,
        )
        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
        result.assert_outcomes(passed=2)

    def test_testing_package_does_not_import_pytest(self) -> None:
        code = "import sys, conscribe.testing; print('pytest' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        )
        assert out.stdout.strip() == "False"


# ===================================================================
# dataclass(slots=True) re-creation inside a block
# ===================================================================


class TestRecreationInBlock:
    def test_slots_class_defined_in_block_is_rolled_back(self) -> None:
        entry = _state()
        with isolated_registries() as iso:
            @dataclass(slots=True)
            class SlotFake(StrategyBase):
                x: int = 0

            assert Strat.get("slot_fake") is SlotFake
        assert _state() == entry
        assert iso.rolled_back == [("iso_strategy", "slot_fake", SlotFake)]

    def test_slots_class_captured_and_activated(self) -> None:
        fakes = isolated_registries()
        with fakes:
            @dataclass(slots=True)
            class SlotCaptured(StrategyBase):
                x: int = 0

        with fakes.activate():
            assert Strat.get("slot_captured") is SlotCaptured
        assert "slot_captured" not in Strat.keys()
