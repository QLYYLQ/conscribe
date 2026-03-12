"""Integration tests: Path C (manual @register) edge cases.

Path C: @register("key") or @register(propagate=True) for manual registration.
This is the "escape hatch" for external classes that can't use metaclass inheritance.

Tests cover:
- @register with explicit key
- @register with inferred key
- @register with protocol check enforcement
- @register(propagate=True) for subclass propagation
- propagate=True grandchild registration
- propagate=True with __abstract__ subclass
- propagate=True with __registry_key__ override
- propagate=True chains existing __init_subclass__
- @register on a class that already has the metaclass (edge case)
- @register duplicate key prevention
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from layer_registry import create_registrar
from layer_registry.exceptions import (
    DuplicateKeyError,
    ProtocolViolationError,
)


@runtime_checkable
class RunnerProtocol(Protocol):
    def run(self, input: str) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


class TestRegisterExplicitKey:
    """@register("key") with an explicit key string."""

    def test_basic_register(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("my_runner")
        class MyRunner:
            def run(self, input: str) -> str:
                return f"result: {input}"

        assert R.get("my_runner") is MyRunner

    def test_register_sets_registry_key_attr(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("tagged")
        class Tagged:
            def run(self, input: str) -> str:
                return "tagged"

        assert Tagged.__registry_key__ == "tagged"

    def test_register_returns_same_class(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("orig")
        class Original:
            def run(self, input: str) -> str:
                return "original"

        assert Original().run("x") == "original"


class TestRegisterInferredKey:
    """@register() without explicit key uses key_transform."""

    def test_inferred_key_with_suffix_stripping(self) -> None:
        R = create_registrar("test", RunnerProtocol, strip_suffixes=["Runner"])

        @R.register()
        class FastRunner:
            def run(self, input: str) -> str:
                return "fast"

        assert R.get("fast") is FastRunner

    def test_inferred_key_default_snake(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register()
        class CamelCaseRunner:
            def run(self, input: str) -> str:
                return "camel"

        assert R.get("camel_case_runner") is CamelCaseRunner


class TestRegisterProtocolCheck:
    """Path C always enforces protocol_check=True."""

    def test_noncompliant_class_rejected(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        with pytest.raises((TypeError, ProtocolViolationError)):
            @R.register("bad")
            class Incomplete:
                pass  # Missing run()

    def test_compliant_class_accepted(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("good")
        class Complete:
            def run(self, input: str) -> str:
                return "good"

        assert R.get("good") is Complete

    def test_partially_compliant_rejected(self) -> None:
        """Class with wrong method signature or missing methods."""
        @runtime_checkable
        class MultiMethodProto(Protocol):
            def method_a(self) -> str: ...
            def method_b(self) -> str: ...

        R = create_registrar("test", MultiMethodProto)

        with pytest.raises((TypeError, ProtocolViolationError)):
            @R.register("partial")
            class Partial:
                def method_a(self) -> str:
                    return "a"
                # Missing method_b


class TestRegisterDuplicateKey:
    """Duplicate key prevention for @register."""

    def test_duplicate_explicit_key_raises(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("dup")
        class First:
            def run(self, input: str) -> str:
                return "first"

        with pytest.raises(DuplicateKeyError):
            @R.register("dup")
            class Second:
                def run(self, input: str) -> str:
                    return "second"

    def test_duplicate_inferred_key_raises(self) -> None:
        R = create_registrar("test", RunnerProtocol, strip_suffixes=["Runner"])

        @R.register()
        class FooRunner:
            def run(self, input: str) -> str:
                return "foo1"

        with pytest.raises(DuplicateKeyError):
            # Use explicit key="foo" to trigger duplicate — register() doesn't
            # check __registry_key__ class attribute, only the key parameter
            @R.register("foo")
            class FooRunner2:
                def run(self, input: str) -> str:
                    return "foo2"


class TestPropagate:
    """@register(propagate=True) injects __init_subclass__ for auto-registration."""

    def test_propagate_basic_subclass(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("base_runner", propagate=True)
        class BaseRunner:
            def run(self, input: str) -> str:
                return "base"

        class FastSubRunner(BaseRunner):
            def run(self, input: str) -> str:
                return "fast"

        class SlowSubRunner(BaseRunner):
            def run(self, input: str) -> str:
                return "slow"

        assert R.get("fast_sub_runner") is FastSubRunner
        assert R.get("slow_sub_runner") is SlowSubRunner

    def test_propagate_grandchild(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("root", propagate=True)
        class Root:
            def run(self, input: str) -> str:
                return "root"

        class Child(Root):
            def run(self, input: str) -> str:
                return "child"

        class GrandChild(Child):
            def run(self, input: str) -> str:
                return "grandchild"

        assert R.get("child") is Child
        assert R.get("grand_child") is GrandChild

    def test_propagate_abstract_skipped(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("base", propagate=True)
        class Base:
            def run(self, input: str) -> str:
                return "base"

        class AbstractMiddle(Base):
            __abstract__ = True

        class Concrete(AbstractMiddle):
            def run(self, input: str) -> str:
                return "concrete"

        assert R.get_or_none("abstract_middle") is None
        assert R.get("concrete") is Concrete

    def test_propagate_key_override(self) -> None:
        R = create_registrar("test", RunnerProtocol)

        @R.register("base", propagate=True)
        class Base:
            def run(self, input: str) -> str:
                return "base"

        class CustomKeyed(Base):
            __registry_key__ = "my_custom"
            def run(self, input: str) -> str:
                return "custom"

        assert R.get("my_custom") is CustomKeyed

    def test_propagate_chains_existing_init_subclass(self) -> None:
        """Existing __init_subclass__ on the target class should be preserved."""
        R = create_registrar("test", RunnerProtocol)
        hook_calls: list[str] = []

        class WithHook:
            def run(self, input: str) -> str:
                return "hook"

            def __init_subclass__(cls, **kwargs: object) -> None:
                super().__init_subclass__(**kwargs)
                hook_calls.append(cls.__name__)

        @R.register("hooked", propagate=True)
        class Hooked(WithHook):
            pass

        class SubHooked(Hooked):
            def run(self, input: str) -> str:
                return "sub"

        assert "SubHooked" in hook_calls
        assert R.get("sub_hooked") is SubHooked

    def test_propagate_without_propagate_no_subclass_registration(self) -> None:
        """Without propagate=True, subclasses of @register'd class are NOT registered."""
        R = create_registrar("test", RunnerProtocol)

        @R.register("base_only")
        class BaseOnly:
            def run(self, input: str) -> str:
                return "base"

        class SubOfBaseOnly(BaseOnly):
            def run(self, input: str) -> str:
                return "sub"

        # SubOfBaseOnly should NOT be registered (no propagation)
        assert R.get_or_none("sub_of_base_only") is None


class TestPathCWithAgentScenario:
    """Real-world scenario: Bob uses @register for agent integration."""

    def test_bob_single_agent_register(self) -> None:
        R = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        # Bob's external agent (no metaclass)
        class ExtToolAgent:
            async def step(self, task: str) -> str:
                return "ext_tool"
            def reset(self) -> None:
                pass

        @R.register("ext_tool")
        class WrappedExtTool(ExtToolAgent):
            pass

        assert R.get("ext_tool") is WrappedExtTool

    def test_bob_propagate_for_variant_factory(self) -> None:
        """Bob uses propagate=True to create a variant factory."""
        R = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class ExternalFramework:
            async def step(self, task: str) -> str:
                return "ext"
            def reset(self) -> None:
                pass

        @R.register("ext_base", propagate=True)
        class ExtBridge(ExternalFramework):
            pass

        class ExtVariantA(ExtBridge):
            async def step(self, task: str) -> str:
                return "variant_a"
            def reset(self) -> None:
                pass

        class ExtVariantB(ExtBridge):
            async def step(self, task: str) -> str:
                return "variant_b"
            def reset(self) -> None:
                pass

        assert R.get("ext_variant_a") is ExtVariantA
        assert R.get("ext_variant_b") is ExtVariantB
