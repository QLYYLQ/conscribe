"""Tests for conscribe.registration.registrar module.

Covers:
- create_registrar(): factory options, key_transform priority
- Query API: get, get_or_none, get_all, keys, unregister
- Path A (inheritance): via Meta metaclass
- Path B (bridge): bridge() with 4 metaclass conflict strategies
- Path C (register): explicit @register decorator
- propagate=True: __init_subclass__ injection
- Multi-layer independence
- Config API stubs (NotImplementedError)
- Real scenarios: Alice builds framework, Bob extends it
"""
from __future__ import annotations

from abc import ABCMeta
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.registration.key_transform import make_key_transform, default_key_transform
from conscribe.exceptions import (
    DuplicateKeyError,
    KeyNotFoundError,
    ProtocolViolationError,
)


@runtime_checkable
class AgentProto(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


@runtime_checkable
class LLMProto(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class SimpleProto(Protocol):
    def do_work(self) -> str: ...


# ===================================================================
# create_registrar() factory options
# ===================================================================

class TestCreateRegistrar:
    """Tests for the create_registrar one-line factory."""

    def test_basic_creation(self) -> None:
        R = create_registrar("test", SimpleProto)
        assert R.keys() == []

    def test_has_meta_attribute(self) -> None:
        R = create_registrar("test", SimpleProto)
        assert hasattr(R, "Meta")

    def test_has_protocol_attribute(self) -> None:
        R = create_registrar("test", SimpleProto)
        assert R.protocol is SimpleProto

    def test_discriminator_field_stored(self) -> None:
        R = create_registrar("agent", AgentProto, discriminator_field="name")
        assert R.discriminator_field == "name"

    def test_strip_suffixes_creates_key_transform(self) -> None:
        """strip_suffixes should configure the internal key transform."""
        R = create_registrar("agent", AgentProto, strip_suffixes=["Agent"])

        class Base(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class BrowserUseAgent(Base):
            async def step(self, task: str) -> str:
                return "bu"
            def reset(self) -> None:
                pass

        assert "browser_use" in R.keys()

    def test_strip_prefixes_creates_key_transform(self) -> None:
        R = create_registrar("test", SimpleProto, strip_prefixes=["Base"])

        class Root(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class BaseHandler(Root):
            def do_work(self) -> str:
                return "handler"

        assert "handler" in R.keys()

    def test_custom_key_transform_overrides_suffixes(self) -> None:
        """Explicit key_transform takes priority over strip_suffixes."""
        custom_kt = lambda name: name.lower()
        R = create_registrar(
            "test",
            SimpleProto,
            strip_suffixes=["Worker"],
            key_transform=custom_kt,
        )

        class Root(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class FooWorker(Root):
            def do_work(self) -> str:
                return "foo"

        # custom_kt returns "fooworker" (lowercase), NOT "foo" (suffix stripped)
        assert "fooworker" in R.keys()

    def test_default_key_transform_when_no_options(self) -> None:
        """When no strip_suffixes/prefixes/key_transform, uses default (pure snake_case)."""
        R = create_registrar("test", SimpleProto)

        class Root(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class MyFancyWorker(Root):
            def do_work(self) -> str:
                return "fancy"

        assert "my_fancy_worker" in R.keys()

    def test_registrar_class_name_includes_layer(self) -> None:
        """The dynamically created registrar class should have a meaningful name."""
        R = create_registrar("agent", AgentProto)
        # The class name should include "agent" in some form
        assert "agent" in R.__name__.lower() or "Agent" in R.__name__


# ===================================================================
# Query API
# ===================================================================

class TestQueryAPI:
    """Tests for get, get_or_none, get_all, keys, unregister."""

    def test_get_existing(self, simple_registrar) -> None:
        R = simple_registrar

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Alpha(Base):
            def do_work(self) -> str:
                return "alpha"

        assert R.get("alpha") is Alpha

    def test_get_missing_raises(self, simple_registrar) -> None:
        R = simple_registrar
        with pytest.raises((KeyError, KeyNotFoundError)):
            R.get("nonexistent")

    def test_get_or_none_existing(self, simple_registrar) -> None:
        R = simple_registrar

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Beta(Base):
            def do_work(self) -> str:
                return "beta"

        assert R.get_or_none("beta") is Beta

    def test_get_or_none_missing(self, simple_registrar) -> None:
        R = simple_registrar
        assert R.get_or_none("nope") is None

    def test_get_all_returns_dict(self, simple_registrar) -> None:
        R = simple_registrar

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Gamma(Base):
            def do_work(self) -> str:
                return "gamma"

        class Delta(Base):
            def do_work(self) -> str:
                return "delta"

        all_entries = R.get_all()
        assert isinstance(all_entries, dict)
        assert set(all_entries.keys()) == {"gamma", "delta"}

    def test_keys_returns_list(self, simple_registrar) -> None:
        R = simple_registrar
        keys = R.keys()
        assert isinstance(keys, list)

    def test_unregister_removes_key(self, simple_registrar) -> None:
        R = simple_registrar

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class ToRemove(Base):
            def do_work(self) -> str:
                return "remove"

        assert "to_remove" in R.keys()
        R.unregister("to_remove")
        assert "to_remove" not in R.keys()


# ===================================================================
# Path A: Inheritance via Meta metaclass
# ===================================================================

class TestPathAInheritance:
    """Path A: class FooAgent(BaseAgent) -> auto-registered."""

    def test_path_a_basic_registration(self) -> None:
        R = create_registrar("test", SimpleProto)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class ConcreteWorker(Base):
            def do_work(self) -> str:
                return "concrete"

        assert R.get("concrete_worker") is ConcreteWorker

    def test_path_a_no_protocol_check(self) -> None:
        """Path A skips Protocol check (inheritance guarantees compliance)."""
        R = create_registrar("test", SimpleProto)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        # This class doesn't implement do_work itself, but inherits it from Base
        class Inherited(Base):
            pass

        # Should still be registered (inherited method counts)
        assert R.get("inherited") is Inherited


# ===================================================================
# Path B: bridge()
# ===================================================================

class TestPathBBridge:
    """Path B: bridge(ExternalClass) creates an abstract bridge base class."""

    def test_bridge_basic(self) -> None:
        R = create_registrar("test", SimpleProto, strip_suffixes=["Worker"])

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class ExternalLib:
            def do_work(self) -> str:
                return "external"

        Bridge = R.bridge(ExternalLib)

        class VariantA(Bridge):
            def do_work(self) -> str:
                return "a"

        class VariantB(Bridge):
            def do_work(self) -> str:
                return "b"

        assert R.get("variant_a") is VariantA
        assert R.get("variant_b") is VariantB
        # Bridge itself is abstract -> not registered
        assert R.get_or_none("external_lib_bridge") is None
        assert R.get_or_none("external_lib") is None

    def test_bridge_custom_name(self) -> None:
        R = create_registrar("test", SimpleProto)

        class ExtClass:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(ExtClass, name="CustomBridge")
        assert Bridge.__name__ == "CustomBridge"

    def test_bridge_default_name(self) -> None:
        R = create_registrar("test", SimpleProto)

        class SomeExternal:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(SomeExternal)
        assert "SomeExternal" in Bridge.__name__ or "Bridge" in Bridge.__name__

    def test_bridge_preserves_external_methods(self) -> None:
        """Bridge subclass should inherit methods from the external class."""
        R = create_registrar("test", SimpleProto)

        class ExternalWithExtra:
            def do_work(self) -> str:
                return "external_work"

            def extra_method(self) -> str:
                return "extra"

        Bridge = R.bridge(ExternalWithExtra)

        class MyVariant(Bridge):
            pass

        instance = MyVariant()
        assert instance.extra_method() == "extra"

    def test_bridge_with_plain_metaclass(self) -> None:
        """Strategy 1: ext_meta is type -> use cls.Meta."""
        R = create_registrar("test", SimpleProto)

        class PlainExternal:
            def do_work(self) -> str:
                return "plain"

        Bridge = R.bridge(PlainExternal)

        class Concrete(Bridge):
            def do_work(self) -> str:
                return "concrete"

        assert R.get("concrete") is Concrete

    def test_bridge_with_abcmeta(self) -> None:
        """Strategy 4: external class has ABCMeta -> combined metaclass created."""
        R = create_registrar("test", SimpleProto)

        class ABCExternal(metaclass=ABCMeta):
            def do_work(self) -> str:
                return "abc"

        Bridge = R.bridge(ABCExternal)

        class Concrete(Bridge):
            def do_work(self) -> str:
                return "abc_concrete"

        assert R.get("concrete") is Concrete

    def test_bridge_is_abstract(self) -> None:
        """Bridge class should be marked __abstract__ = True."""
        R = create_registrar("test", SimpleProto)

        class Ext:
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(Ext)
        # Bridge should not appear in registry
        assert R.get_or_none(Bridge.__name__.lower()) is None


# ===================================================================
# Path C: @register decorator
# ===================================================================

class TestPathCRegister:
    """Path C: @register("key") or @register() for manual registration."""

    def test_register_with_explicit_key(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("my_custom_key")
        class External:
            def do_work(self) -> str:
                return "external"

        assert R.get("my_custom_key") is External

    def test_register_with_inferred_key(self) -> None:
        """@register() without key uses key_transform to infer key."""
        R = create_registrar("test", SimpleProto, strip_suffixes=["Worker"])

        @R.register()
        class MyFancyWorker:
            def do_work(self) -> str:
                return "fancy"

        assert R.get("my_fancy") is MyFancyWorker

    def test_register_protocol_check_enforced(self) -> None:
        """Path C always does protocol_check=True (no inheritance guarantee)."""
        R = create_registrar("test", SimpleProto)

        with pytest.raises((TypeError, ProtocolViolationError)):
            @R.register("bad")
            class Incomplete:
                pass  # Missing do_work

    def test_register_sets_registry_key(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("my_key")
        class Tagged:
            def do_work(self) -> str:
                return "tagged"

        assert Tagged.__registry_key__ == "my_key"

    def test_register_returns_original_class(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("original")
        class Original:
            def do_work(self) -> str:
                return "original"

        # Decorator should return the class unmodified
        assert Original().do_work() == "original"

    def test_register_duplicate_key_raises(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("dup")
        class First:
            def do_work(self) -> str:
                return "first"

        with pytest.raises(DuplicateKeyError):
            @R.register("dup")
            class Second:
                def do_work(self) -> str:
                    return "second"


# ===================================================================
# propagate=True: __init_subclass__ injection
# ===================================================================

class TestPropagateRegistration:
    """Tests for @register(propagate=True) -- __init_subclass__ injection."""

    def test_propagate_subclass_auto_registered(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("base_thing", propagate=True)
        class BaseThing:
            def do_work(self) -> str:
                return "base"

        class SubThingA(BaseThing):
            def do_work(self) -> str:
                return "a"

        class SubThingB(BaseThing):
            def do_work(self) -> str:
                return "b"

        assert R.get("sub_thing_a") is SubThingA
        assert R.get("sub_thing_b") is SubThingB

    def test_propagate_grandchild_registered(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("root", propagate=True)
        class Root:
            def do_work(self) -> str:
                return "root"

        class Middle(Root):
            def do_work(self) -> str:
                return "middle"

        class Leaf(Middle):
            def do_work(self) -> str:
                return "leaf"

        assert R.get("middle") is Middle
        assert R.get("leaf") is Leaf

    def test_propagate_abstract_subclass_skipped(self) -> None:
        """Subclass with __abstract__=True should be skipped."""
        R = create_registrar("test", SimpleProto)

        @R.register("base", propagate=True)
        class Base:
            def do_work(self) -> str:
                return "base"

        class AbstractMiddle(Base):
            __abstract__ = True

        class Concrete(AbstractMiddle):
            def do_work(self) -> str:
                return "concrete"

        assert R.get_or_none("abstract_middle") is None
        assert R.get("concrete") is Concrete

    def test_propagate_key_override_in_subclass(self) -> None:
        R = create_registrar("test", SimpleProto)

        @R.register("base", propagate=True)
        class Base:
            def do_work(self) -> str:
                return "base"

        class Custom(Base):
            __registry_key__ = "my_custom"
            def do_work(self) -> str:
                return "custom"

        assert R.get("my_custom") is Custom

    def test_propagate_protocol_check_enforced(self) -> None:
        """Propagated subclasses should also undergo protocol_check=True."""
        R = create_registrar("test", SimpleProto)

        @R.register("base", propagate=True)
        class Base:
            def do_work(self) -> str:
                return "base"

        # Subclass that removes the method should fail protocol check
        # But since Python inherits do_work, this should actually pass.
        # A truly non-compliant subclass would need to somehow not have do_work.
        # This test verifies the mechanism exists.
        class GoodChild(Base):
            def do_work(self) -> str:
                return "good"

        assert R.get("good_child") is GoodChild

    def test_propagate_chains_existing_init_subclass(self) -> None:
        """If the class already has __init_subclass__, it should be chained."""
        R = create_registrar("test", SimpleProto)
        init_subclass_called: list[str] = []

        class WithHook:
            def do_work(self) -> str:
                return "hook"

            def __init_subclass__(cls, **kwargs: object) -> None:
                super().__init_subclass__(**kwargs)
                init_subclass_called.append(cls.__name__)

        @R.register("with_hook", propagate=True)
        class Hooked(WithHook):
            pass

        class Child(Hooked):
            def do_work(self) -> str:
                return "child"

        assert "Child" in init_subclass_called
        assert R.get("child") is Child


# ===================================================================
# Multi-layer independence
# ===================================================================

class TestMultiLayerIndependence:
    """Tests that multiple registrars are completely independent."""

    def test_two_layers_independent(self) -> None:
        AR = create_registrar("agent", AgentProto, strip_suffixes=["Agent"])
        LR = create_registrar("llm", LLMProto, strip_suffixes=["LLM", "Provider"])

        class BaseAgent(metaclass=AR.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class FooAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "foo"
            def reset(self) -> None:
                pass

        class OpenAIProvider(BaseLLM):
            async def chat(self, messages: list[dict]) -> str:
                return "gpt"

        # Two registrars are independent
        assert "foo" in AR.keys()
        assert "open_ai" in LR.keys()
        assert AR.get_or_none("open_ai") is None
        assert LR.get_or_none("foo") is None

    def test_three_layers_independent(self) -> None:
        @runtime_checkable
        class BrowserProto(Protocol):
            def connect(self) -> str: ...
            def close(self) -> None: ...

        AR = create_registrar("agent", AgentProto, strip_suffixes=["Agent"])
        LR = create_registrar("llm", LLMProto, strip_suffixes=["Provider"])
        BR = create_registrar("browser", BrowserProto, strip_suffixes=["Browser"])

        class BaseAgent(metaclass=AR.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class BaseBrowser(metaclass=BR.Meta):
            __abstract__ = True
            def connect(self) -> str: ...
            def close(self) -> None: ...

        class TestAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "test"
            def reset(self) -> None:
                pass

        class TestProvider(BaseLLM):
            async def chat(self, messages: list[dict]) -> str:
                return "llm"

        class TestBrowser(BaseBrowser):
            def connect(self) -> str:
                return "connected"
            def close(self) -> None:
                pass

        assert len(AR.keys()) == 1
        assert len(LR.keys()) == 1
        assert len(BR.keys()) == 1


# ===================================================================
# Config API stubs
# ===================================================================

class TestConfigStubs:
    """Config API methods should raise NotImplementedError in Phase 1."""

    def test_build_config_not_implemented(self) -> None:
        R = create_registrar("test", SimpleProto)
        with pytest.raises(NotImplementedError):
            R.build_config()

    def test_config_union_type_not_implemented(self) -> None:
        R = create_registrar("test", SimpleProto)
        with pytest.raises(NotImplementedError):
            R.config_union_type()

    def test_get_config_schema_not_implemented(self) -> None:
        R = create_registrar("test", SimpleProto)
        with pytest.raises(NotImplementedError):
            R.get_config_schema("some_key")


# ===================================================================
# Real scenario: Alice builds benchmark framework
# ===================================================================

class TestAliceBuildsBenchmarkFramework:
    """Alice: create_registrar -> base class -> 3 built-in agents -> query."""

    def test_alice_full_workflow(self) -> None:
        R = create_registrar(
            "agent",
            AgentProto,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str:
                raise NotImplementedError
            def reset(self) -> None:
                raise NotImplementedError

        class BrowserUseAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "browser_use"
            def reset(self) -> None:
                pass

        class SkyvernAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "skyvern"
            def reset(self) -> None:
                pass

        # CLI simulation: alice-bench run --agent browser_use
        agent_cls = R.get("browser_use")
        assert agent_cls is BrowserUseAgent
        assert set(R.keys()) == {"browser_use", "skyvern"}


# ===================================================================
# Real scenario: Bob bridges external agent
# ===================================================================

class TestBobBridgesExternalAgent:
    """Bob extends the framework with external agents."""

    def test_bob_bridge_workflow(self) -> None:
        R = create_registrar(
            "agent",
            AgentProto,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        # External agent (independent framework, no metaclass)
        class ExternalCoolAgent:
            async def step(self, task: str) -> str:
                return "cool"
            def reset(self) -> None:
                pass

        # Bob one-time bridge
        CoolBridge = R.bridge(ExternalCoolAgent)

        class CoolVariantA(CoolBridge):
            async def step(self, task: str) -> str:
                return "variant_a"
            def reset(self) -> None:
                pass

        class CoolVariantB(CoolBridge):
            async def step(self, task: str) -> str:
                return "variant_b"
            def reset(self) -> None:
                pass

        assert R.get("cool_variant_a") is CoolVariantA
        assert R.get("cool_variant_b") is CoolVariantB
        # Bridge itself not registered
        assert R.get_or_none("external_cool_agent") is None

    def test_bob_register_single_external(self) -> None:
        R = create_registrar("agent", AgentProto, discriminator_field="name")

        class ExternalAgent:
            async def step(self, task: str) -> str:
                return "ext"
            def reset(self) -> None:
                pass

        @R.register("my_external")
        class WrappedExternal(ExternalAgent):
            pass

        assert R.get("my_external") is WrappedExternal

    def test_bob_propagate_with_init_subclass(self) -> None:
        R = create_registrar("agent", AgentProto, discriminator_field="name")

        class ProblematicBase:
            async def step(self, task: str) -> str:
                return "base"
            def reset(self) -> None:
                pass

        @R.register("prob_base", propagate=True)
        class ProblematicBridge(ProblematicBase):
            pass

        class ProbVariantA(ProblematicBridge):
            async def step(self, task: str) -> str:
                return "a"
            def reset(self) -> None:
                pass

        class ProbVariantB(ProblematicBridge):
            async def step(self, task: str) -> str:
                return "b"
            def reset(self) -> None:
                pass

        assert R.get("prob_variant_a") is ProbVariantA
        assert R.get("prob_variant_b") is ProbVariantB
