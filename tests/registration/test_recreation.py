"""Tests for class re-creation (``@dataclass(slots=True)``, ``attrs.define``,
pydantic ``dataclass(slots=True)``) of registered classes.

Those decorators replace a freshly defined class with a new class built from
a copy of its ``__dict__``, which runs the registration machinery a second
time.  Registered classes carry ``__registry_origin__ = id(cls)``; the copy
carries the original's id, so the copy rebinds every key of the original in
every registry instead of raising ``DuplicateKeyError``.  A class body that
runs twice carries no stamp and still raises, with a hint toward
``isolated_registries()``.

Covers: module-level, class-nested and function-local classes; multi-key and
hierarchical keys; cross-registry diamonds with different key transforms;
attrs and pydantic dataclasses; the ``register(propagate=True)`` hook
(subclass re-creation and the ``super()`` owner fix); a class auto-registered
in one registry and ``@register``-ed in another; fresh redefinitions;
copy-to-derive with a changed key.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.exceptions import DuplicateKeyError
from conscribe.testing import isolated_registries


@runtime_checkable
class WorkProto(Protocol):
    def work(self) -> str: ...


Work = create_registrar("rc_work", WorkProto)
Tree = create_registrar("rc_tree", WorkProto, key_separator=".")
LLM = create_registrar("rc_llm", WorkProto, strip_suffixes=["LLM"])
AgentR = create_registrar("rc_agent", WorkProto, strip_suffixes=["Agent"])
Manual = create_registrar("rc_manual", WorkProto)


class WorkBase(metaclass=Work.Meta):
    __abstract__ = True

    def work(self) -> str:
        return type(self).__name__


class TreeRoot(metaclass=Tree.Meta):
    __abstract__ = True
    __registry_key__ = "browser"

    def work(self) -> str:
        return type(self).__name__


class DualBase(metaclass=LLM.Meta | AgentR.Meta):
    __abstract__ = True

    def work(self) -> str:
        return type(self).__name__


# Module-level re-created classes (registered at import of this module).


@dataclass(slots=True)
class ModuleSlotted(WorkBase):
    x: int = 1


@dataclass(slots=True)
class MultiSlotted(WorkBase):
    __registry_key__ = ["multi_a", "multi_b"]
    y: int = 0


@dataclass(slots=True)
class Click(TreeRoot):
    pass


@dataclass(slots=True)
class ChatAgentLLM(DualBase):
    pass


class Outer:
    @dataclass(slots=True)
    class Inner(WorkBase):
        z: int = 0


@pytest.fixture(autouse=True)
def _isolate(conscribe_isolated):
    """Each test's registrations roll back (module-level ones stay)."""
    yield conscribe_isolated


class TestDataclassSlots:
    def test_module_level_class_registers_final_class(self) -> None:
        assert "__slots__" in ModuleSlotted.__dict__
        assert Work.get("module_slotted") is ModuleSlotted
        assert ModuleSlotted.__registry_origin__ == id(ModuleSlotted)
        assert ModuleSlotted(x=3).work() == "ModuleSlotted"

    def test_class_nested_in_class(self) -> None:
        # dataclasses re-creates nested classes with __qualname__ == __name__
        # during the metaclass pass; the stamp does not depend on names.
        assert Work.get("inner") is Outer.Inner
        assert Outer.Inner.__qualname__ == "Outer.Inner"

    def test_function_local_class(self) -> None:
        @dataclass(slots=True)
        class Local(WorkBase):
            v: int = 0

        assert Work.get("local") is Local
        assert "__slots__" in Local.__dict__

    def test_multi_key_rebinds_every_key(self) -> None:
        assert Work.get("multi_a") is MultiSlotted
        assert Work.get("multi_b") is MultiSlotted
        assert MultiSlotted.__registry_keys__ == ["multi_a", "multi_b"]

    def test_hierarchical_key(self) -> None:
        assert Tree.keys() == ["browser.click"]
        assert Tree.get("browser.click") is Click

    def test_diamond_with_different_key_transforms(self) -> None:
        # One key per registry, each bound to the final class; no stale or
        # spurious keys.
        assert LLM.get_all() == {"chat_agent": ChatAgentLLM}
        assert AgentR.get_all() == {"chat_agent_llm": ChatAgentLLM}

    def test_diamond_with_skip_registries(self) -> None:
        @dataclass(slots=True)
        class OnlyLLM(DualBase):
            __skip_registries__ = ["rc_agent"]

        assert LLM.get("only") is OnlyLLM
        assert AgentR.get_or_none("only_llm") is None

    def test_abstract_class_is_not_registered(self) -> None:
        @dataclass(slots=True)
        class AbstractSlotted(WorkBase):
            __abstract__ = True

        assert "abstract_slotted" not in Work.keys()
        assert "__registry_origin__" not in AbstractSlotted.__dict__

    def test_subclass_of_slots_class_registers_normally(self) -> None:
        class Child(ModuleSlotted):
            pass

        assert Work.get("child") is Child
        assert "__registry_origin__" in Child.__dict__

    def test_frozen_slots_dataclass(self) -> None:
        @dataclass(slots=True, frozen=True)
        class Frozen(WorkBase):
            f: int = 0

        assert Work.get("frozen") is Frozen
        assert dataclasses.is_dataclass(Work.get("frozen"))


class TestOtherDecorators:
    def test_attrs_define(self) -> None:
        attrs = pytest.importorskip("attrs")

        @attrs.define
        class AttrsWorker(WorkBase):
            a: int = 1

        assert "__slots__" in AttrsWorker.__dict__
        assert Work.get("attrs_worker") is AttrsWorker
        assert AttrsWorker(a=2).a == 2

    def test_attrs_frozen(self) -> None:
        attrs = pytest.importorskip("attrs")

        @attrs.frozen
        class AttrsFrozen(WorkBase):
            a: int = 1

        assert Work.get("attrs_frozen") is AttrsFrozen

    def test_pydantic_dataclass_slots(self) -> None:
        from pydantic.dataclasses import dataclass as pydantic_dataclass

        @pydantic_dataclass(slots=True)
        class PydanticSlotted(WorkBase):
            p: int = 0

        assert Work.get("pydantic_slotted") is PydanticSlotted
        assert PydanticSlotted(p="5").p == 5


class TestPropagateHook:
    def test_slots_subclass_through_propagate_hook(self) -> None:
        @Manual.register("rc_root", propagate=True)
        class Root:
            def work(self) -> str:
                return ""

        @dataclass(slots=True)
        class Leaf(Root):
            w: int = 0

        assert Manual.get("leaf") is Leaf
        assert Leaf.__registry_origin__ == id(Leaf)

    def test_subclassing_a_recreated_propagate_root(self) -> None:
        # register(propagate=True) BELOW dataclass(slots=True): the hook was
        # installed on the pre-slots class and copied to the final one.
        # Subclassing used to crash in super(); the root's own registration
        # stays on the pre-slots class (documented as unsupported).
        @dataclass(slots=True)
        @Manual.register("rc_root2", propagate=True)
        class Root2:
            def work(self) -> str:
                return ""

        class Leaf2(Root2):
            pass

        assert Manual.get("leaf2") is Leaf2
        assert Manual.get("rc_root2") is not Root2  # the stale pre-slots class

    def test_original_init_subclass_still_chained(self) -> None:
        seen: list[str] = []

        @dataclass(slots=True)
        @Manual.register("rc_root3", propagate=True)
        class Root3:
            def __init_subclass__(cls, **kwargs):
                seen.append(cls.__name__)

            def work(self) -> str:
                return ""

        class Leaf3(Root3):
            pass

        assert seen == ["Leaf3"]
        assert Manual.get("leaf3") is Leaf3

    @pytest.mark.parametrize("decorator", ["dataclass_slots", "attrs_define"])
    def test_two_propagating_registrars_on_one_root(self, decorator) -> None:
        """Review probe 16: the outer hook used to read the re-creation stamp
        after the chained inner hook had re-stamped the copy, and registered
        it again (DuplicateKeyError)."""
        if decorator == "attrs_define":
            decorate = pytest.importorskip("attrs").define
        else:
            decorate = dataclass(slots=True)
        Left = create_registrar("rc_dp_left", WorkProto)
        Right = create_registrar("rc_dp_right", WorkProto)

        @Right.register("root_right", propagate=True)
        @Left.register("root_left", propagate=True)
        class Root:
            def work(self) -> str:
                return ""

        @decorate
        class ChatAgent(Root):
            x: int = 1

        assert Left.get("chat_agent") is ChatAgent
        assert Right.get("chat_agent") is ChatAgent
        assert ChatAgent.__registry_origin__ == id(ChatAgent)

        @decorate
        class Grandchild(ChatAgent):
            y: int = 2

        assert Left.get("grandchild") is Grandchild
        assert Right.get("grandchild") is Grandchild

    def test_two_propagating_registrars_on_a_slots_root(self) -> None:
        Left = create_registrar("rc_dp2_left", WorkProto)
        Right = create_registrar("rc_dp2_right", WorkProto)

        @Right.register("root_right", propagate=True)
        @Left.register("root_left", propagate=True)
        @dataclass(slots=True)
        class Root:
            x: int = 1

            def work(self) -> str:
                return ""

        @dataclass(slots=True)
        class Sub(Root):
            y: int = 2

        class Sub2(Sub):
            pass

        assert sorted(Left.keys()) == ["root_left", "sub", "sub2"]
        assert sorted(Right.keys()) == ["root_right", "sub", "sub2"]
        assert Left.get("sub") is Right.get("sub") is Sub

    def test_hook_passes_class_kwargs_up_the_chain(self) -> None:
        seen: dict[str, object] = {}

        class Configurable:
            def __init_subclass__(cls, flavour: str = "plain", **kwargs):
                super().__init_subclass__(**kwargs)
                seen[cls.__name__] = flavour

        @Manual.register("rc_conf", propagate=True)
        class ConfRoot(Configurable):
            def work(self) -> str:
                return ""

        class Spicy(ConfRoot, flavour="spicy"):
            pass

        assert seen["Spicy"] == "spicy"
        assert Manual.get("spicy") is Spicy


class TestMixedPaths:
    def test_metaclass_registry_and_register_below_slots_both_rebound(self) -> None:
        # Auto-registered in rc_work, @register-ed in rc_manual on the
        # pre-slots class: the metaclass pass of the re-creation rebinds the
        # copy source in every registry.
        @dataclass(slots=True)
        @Manual.register("mixed_manual")
        class Mixed(WorkBase):
            m: int = 0

        assert Work.get("mixed") is Mixed
        assert Manual.get("mixed_manual") is Mixed


class TestStillRaises:
    def test_class_body_run_twice_raises_with_isolation_hint(self) -> None:
        def define():
            @dataclass(slots=True)
            class Twice(WorkBase):
                pass

            return Twice

        define()
        with pytest.raises(DuplicateKeyError, match="isolated_registries") as info:
            define()
        assert "ran a second time" in str(info.value)
        assert info.value.hint is not None

    def test_plain_redefinition_raises_with_isolation_hint(self) -> None:
        source = "class Again(WorkBase):\n    pass\n"
        namespace = {"WorkBase": WorkBase, "__name__": __name__}
        exec(source, namespace)
        with pytest.raises(DuplicateKeyError, match="isolated_registries"):
            exec(source, dict(namespace))

    def test_pydantic_over_stdlib_dataclass_stacking_is_rejected(self) -> None:
        # Documented as unsupported: pydantic wraps the stdlib dataclass in a
        # same-named subclass, which inherits nothing but collides on the key.
        from pydantic.dataclasses import dataclass as pydantic_dataclass

        with pytest.raises(DuplicateKeyError, match="one dataclass decorator"):
            @pydantic_dataclass
            @dataclass
            class Stacked(WorkBase):
                s: int = 0

    def test_different_class_same_key_keeps_default_advice(self) -> None:
        class First(WorkBase):
            __registry_key__ = "contested"

        with pytest.raises(DuplicateKeyError, match="To fix: add __registry_key__"):
            class Second(WorkBase):
                __registry_key__ = "contested"

    def test_copy_to_derive_with_changed_key_registers_new_key(self) -> None:
        namespace = dict(ModuleSlotted.__dict__)
        namespace.pop("__slots__")
        for member in ModuleSlotted.__slots__:
            namespace.pop(member, None)
        namespace["__registry_key__"] = "derived_copy"
        Derived = type(WorkBase)("ModuleSlotted", ModuleSlotted.__bases__, namespace)

        assert Work.get("derived_copy") is Derived
        assert Work.get("module_slotted") is ModuleSlotted
        assert Derived.__registry_origin__ == id(Derived)

    def test_copy_of_unregistered_class_registers_normally(self) -> None:
        class Gone(WorkBase):
            pass

        Work.unregister("gone")
        namespace = dict(Gone.__dict__)
        Copy = type(WorkBase)("Gone", Gone.__bases__, namespace)
        assert Work.get("gone") is Copy


class TestIsolationInteraction:
    def test_slots_class_in_block_rolled_back(self) -> None:
        before = Work.keys()
        with isolated_registries() as iso:
            @dataclass(slots=True)
            class Temp(WorkBase):
                pass

            assert Work.get("temp") is Temp
        assert Work.keys() == before
        assert iso.rolled_back == [("rc_work", "temp", Temp)]

    def test_rebind_journaled_for_class_registered_before_block(self) -> None:
        # Re-create (copy) a class registered before the block, inside it:
        # the rebind must be undone on exit.
        namespace = dict(ModuleSlotted.__dict__)
        namespace.pop("__slots__")
        for member in ModuleSlotted.__slots__:
            namespace.pop(member, None)
        with isolated_registries() as iso:
            Copy = type(WorkBase)("ModuleSlotted", ModuleSlotted.__bases__, namespace)
            assert Work.get("module_slotted") is Copy
        assert Work.get("module_slotted") is ModuleSlotted
        assert iso.rolled_back == [("rc_work", "module_slotted", Copy)]
