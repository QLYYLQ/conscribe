"""Integration tests: Path B (bridge) edge cases.

Path B: bridge(ExternalClass) creates an abstract bridge base class,
enabling "inherit to register" for external classes that don't
have the metaclass.

Tests cover:
- Basic bridge workflow
- Bridge with custom name
- Bridge + multiple variants
- Bridge with ABCMeta external class
- Bridge + __registry_key__ override in variant
- Bridge isolation (bridge itself not registered)
- Nested bridges
- Bridge + Path C combination
"""
from __future__ import annotations

from abc import ABCMeta
from typing import Protocol, runtime_checkable

import pytest

from layer_registry import create_registrar
from layer_registry.exceptions import DuplicateKeyError


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


@runtime_checkable
class SimpleProtocol(Protocol):
    def do_work(self) -> str: ...


class TestBridgeBasicWorkflow:
    """Basic bridge usage: external class -> bridge -> variants."""

    def test_bridge_one_variant(self) -> None:
        R = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class ExtFrameworkAgent:
            async def step(self, task: str) -> str:
                return "ext"
            def reset(self) -> None:
                pass

        Bridge = R.bridge(ExtFrameworkAgent)

        class MyVariant(Bridge):
            async def step(self, task: str) -> str:
                return "variant"
            def reset(self) -> None:
                pass

        assert R.get("my_variant") is MyVariant

    def test_bridge_multiple_variants(self) -> None:
        R = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class ExtLib:
            async def step(self, task: str) -> str:
                return "lib"
            def reset(self) -> None:
                pass

        Bridge = R.bridge(ExtLib)

        class V1Agent(Bridge):
            async def step(self, task: str) -> str:
                return "v1"
            def reset(self) -> None:
                pass

        class V2Agent(Bridge):
            async def step(self, task: str) -> str:
                return "v2"
            def reset(self) -> None:
                pass

        class V3Agent(Bridge):
            async def step(self, task: str) -> str:
                return "v3"
            def reset(self) -> None:
                pass

        assert set(R.keys()) & {"v1", "v2", "v3"} == {"v1", "v2", "v3"}


class TestBridgeIsolation:
    """Bridge itself should be abstract and NOT registered."""

    def test_bridge_not_in_registry(self) -> None:
        R = create_registrar("test", SimpleProtocol)

        class External:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(External)
        # Bridge is abstract -> should not appear in registry
        assert len(R.keys()) == 0

    def test_bridge_has_abstract_flag(self) -> None:
        R = create_registrar("test", SimpleProtocol)

        class External:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(External)
        # Bridge should have __abstract__ = True in its __dict__
        assert Bridge.__dict__.get("__abstract__", False) is True


class TestBridgeCustomName:
    """Bridge with custom name parameter."""

    def test_custom_bridge_name(self) -> None:
        R = create_registrar("test", SimpleProtocol)

        class ExtTool:
            def do_work(self) -> str:
                return "tool"

        Bridge = R.bridge(ExtTool, name="MyToolBridge")
        assert Bridge.__name__ == "MyToolBridge"

    def test_default_bridge_name(self) -> None:
        R = create_registrar("test", SimpleProtocol)

        class SomeThing:
            def do_work(self) -> str:
                return "thing"

        Bridge = R.bridge(SomeThing)
        # Default name should include the external class name
        assert "SomeThing" in Bridge.__name__


class TestBridgeMetaclassConflict:
    """Bridge must resolve metaclass conflicts between external class and registrar Meta."""

    def test_bridge_plain_external(self) -> None:
        """Strategy 1: external is plain class (metaclass=type) -> use registrar Meta."""
        R = create_registrar("test", SimpleProtocol)

        class Plain:
            def do_work(self) -> str:
                return "plain"

        Bridge = R.bridge(Plain)

        class Impl(Bridge):
            def do_work(self) -> str:
                return "impl"

        assert R.get("impl") is Impl

    def test_bridge_abcmeta_external(self) -> None:
        """Strategy for ABCMeta external: combined metaclass must be created."""
        R = create_registrar("test", SimpleProtocol)

        class AbcExternal(metaclass=ABCMeta):
            def do_work(self) -> str:
                return "abc"

        Bridge = R.bridge(AbcExternal)

        class AbcImpl(Bridge):
            def do_work(self) -> str:
                return "abc_impl"

        assert R.get("abc_impl") is AbcImpl

    def test_bridge_custom_metaclass_external(self) -> None:
        """External class with custom metaclass -> combined metaclass."""
        R = create_registrar("test", SimpleProtocol)

        class CustomMeta(type):
            pass

        class ExtWithMeta(metaclass=CustomMeta):
            def do_work(self) -> str:
                return "custom_meta"

        Bridge = R.bridge(ExtWithMeta)

        class MetaImpl(Bridge):
            def do_work(self) -> str:
                return "meta_impl"

        assert R.get("meta_impl") is MetaImpl


class TestBridgeRegistryKeyOverride:
    """Variants created via bridge can use __registry_key__ override."""

    def test_variant_with_custom_key(self) -> None:
        R = create_registrar("test", SimpleProtocol, strip_suffixes=["Worker"])

        class External:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(External)

        class MyWorker(Bridge):
            __registry_key__ = "custom_name"
            def do_work(self) -> str:
                return "custom"

        assert R.get("custom_name") is MyWorker

    def test_variant_with_inferred_key(self) -> None:
        R = create_registrar("test", SimpleProtocol, strip_suffixes=["Worker"])

        class External:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(External)

        class MyFancyWorker(Bridge):
            def do_work(self) -> str:
                return "fancy"

        assert R.get("my_fancy") is MyFancyWorker


class TestBridgeInheritance:
    """Tests for deeper inheritance chains via bridge."""

    def test_grandchild_of_bridge(self) -> None:
        """Grandchild of bridge class should also be registered."""
        R = create_registrar("test", SimpleProtocol)

        class External:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(External)

        class Child(Bridge):
            def do_work(self) -> str:
                return "child"

        class GrandChild(Child):
            def do_work(self) -> str:
                return "grandchild"

        assert R.get("child") is Child
        assert R.get("grand_child") is GrandChild

    def test_abstract_variant_of_bridge(self) -> None:
        """Abstract variant of bridge should not be registered."""
        R = create_registrar("test", SimpleProtocol)

        class External:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(External)

        class AbstractVariant(Bridge):
            __abstract__ = True

        class ConcreteChild(AbstractVariant):
            def do_work(self) -> str:
                return "concrete"

        assert R.get_or_none("abstract_variant") is None
        assert R.get("concrete_child") is ConcreteChild


class TestBridgeCombinedWithPathC:
    """Combining bridge (Path B) and @register (Path C) in the same registrar."""

    def test_bridge_and_register_coexist(self) -> None:
        R = create_registrar("test", SimpleProtocol)

        class BaseW(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        # Path A
        class InternalWorker(BaseW):
            def do_work(self) -> str:
                return "internal"

        # Path B
        class ExternalLib:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(ExternalLib)

        class BridgedWorker(Bridge):
            def do_work(self) -> str:
                return "bridged"

        # Path C
        @R.register("manual_worker")
        class ManualWorker:
            def do_work(self) -> str:
                return "manual"

        assert R.get("internal_worker") is InternalWorker
        assert R.get("bridged_worker") is BridgedWorker
        assert R.get("manual_worker") is ManualWorker
        assert len(R.keys()) == 3
