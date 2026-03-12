"""Tests for conscribe.registration.auto.create_auto_registrar.

Covers:
- Root class (no bases) is skipped
- __abstract__ = True classes are skipped
- Concrete subclasses are automatically registered
- __abstract__ is NOT inherited (namespace.get vs getattr distinction)
- __registry_key__ is NOT inherited
- Explicit __registry_key__ overrides inferred key
- __config_schema__ validation (must be BaseModel subclass)
- ABCMeta compatibility via base_metaclass
- Multiple inheritance
- kwargs passthrough in class definition
- Real scenario: Alice defines Agent layer with 3 built-in agents
"""
from __future__ import annotations

from abc import ABCMeta
from typing import Protocol, runtime_checkable

import pytest

from conscribe.registration.registry import LayerRegistry
from conscribe.registration.key_transform import make_key_transform, default_key_transform
from conscribe.registration.auto import create_auto_registrar


@runtime_checkable
class AgentLikeProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


@runtime_checkable
class SimpleProto(Protocol):
    def do_work(self) -> str: ...


# ===================================================================
# Root class (no bases) is skipped
# ===================================================================

class TestRootClassSkipping:
    """The base class (metaclass=Meta, no bases) should NOT be registered."""

    def test_root_class_not_registered(self) -> None:
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            def do_work(self) -> str:
                return "base"

        assert registry.keys() == []

    def test_root_class_gets_registry_name(self) -> None:
        """Root class should still get __registry_name__ set."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            def do_work(self) -> str:
                return "base"

        assert Base.__registry_name__ == "test"


# ===================================================================
# Abstract classes are skipped
# ===================================================================

class TestAbstractSkipping:
    """Classes with __abstract__ = True in their own namespace should be skipped."""

    def test_abstract_class_not_registered(self) -> None:
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            def do_work(self) -> str:
                return "base"

        class Middle(Base):
            __abstract__ = True

        assert registry.keys() == []

    def test_abstract_not_inherited_by_subclass(self) -> None:
        """CRITICAL: BaseAgent.__abstract__=True must NOT leak to FooAgent.

        This tests the namespace.get vs getattr distinction.
        """
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return "base"

        class Concrete(Base):
            def do_work(self) -> str:
                return "concrete"

        # Base should NOT be registered (abstract + root)
        # Concrete SHOULD be registered (no __abstract__ in its own namespace)
        assert "concrete" in registry.keys()
        assert len(registry.keys()) == 1

    def test_multi_level_abstract_chain(self) -> None:
        """Abstract at middle level, concrete at leaf -- only leaf registered."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Root(metaclass=Meta):
            def do_work(self) -> str:
                return "root"

        class Middle(Root):
            __abstract__ = True

        class Leaf(Middle):
            def do_work(self) -> str:
                return "leaf"

        assert registry.keys() == ["leaf"]


# ===================================================================
# Concrete subclass registration
# ===================================================================

class TestConcreteRegistration:
    """Concrete subclasses (non-abstract, non-root) should be auto-registered."""

    def test_single_concrete_registered(self) -> None:
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            def do_work(self) -> str:
                return "base"

        class FooWorker(Base):
            def do_work(self) -> str:
                return "foo"

        assert registry.get("foo_worker") is FooWorker

    def test_multiple_concrete_registered(self) -> None:
        registry = LayerRegistry("test", SimpleProto)
        kt = make_key_transform(suffixes=["Worker"])
        Meta = create_auto_registrar(registry, kt)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class AlphaWorker(Base):
            def do_work(self) -> str:
                return "alpha"

        class BetaWorker(Base):
            def do_work(self) -> str:
                return "beta"

        assert set(registry.keys()) == {"alpha", "beta"}

    def test_grandchild_also_registered(self) -> None:
        """Grandchild classes are also registered."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Root(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Parent(Root):
            def do_work(self) -> str:
                return "parent"

        class Child(Parent):
            def do_work(self) -> str:
                return "child"

        assert "parent" in registry.keys()
        assert "child" in registry.keys()


# ===================================================================
# __registry_key__ override
# ===================================================================

class TestRegistryKeyOverride:
    """Explicit __registry_key__ takes precedence over inferred key."""

    def test_explicit_key_used(self) -> None:
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class MySpecialWorker(Base):
            __registry_key__ = "custom_key"
            def do_work(self) -> str:
                return "special"

        assert registry.get("custom_key") is MySpecialWorker

    def test_registry_key_not_inherited(self) -> None:
        """CRITICAL: Parent's __registry_key__ must NOT leak to child.

        Child should use its own inferred key, not parent's explicit key.
        """
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Parent(Base):
            __registry_key__ = "parent_key"
            def do_work(self) -> str:
                return "parent"

        class Child(Parent):
            def do_work(self) -> str:
                return "child"

        # Child should NOT inherit parent's __registry_key__
        assert registry.get("parent_key") is Parent
        assert registry.get("child") is Child

    def test_class_gets_registry_key_attribute(self) -> None:
        """After registration, class should have __registry_key__ set."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class MyWorker(Base):
            def do_work(self) -> str:
                return "work"

        assert MyWorker.__registry_key__ == "my_worker"


# ===================================================================
# __config_schema__ validation
# ===================================================================

class TestConfigSchemaValidation:
    """__config_schema__ must be a pydantic BaseModel subclass if provided."""

    def test_valid_config_schema(self) -> None:
        """Valid BaseModel subclass should pass."""
        from pydantic import BaseModel

        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class MyConfig(BaseModel):
            param: str = "default"

        class MyWorker(Base):
            __config_schema__ = MyConfig
            def do_work(self) -> str:
                return "work"

        # Should register successfully
        assert registry.get("my_worker") is MyWorker

    def test_invalid_config_schema_raises(self) -> None:
        """Non-BaseModel __config_schema__ should raise TypeError."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        with pytest.raises(TypeError, match="__config_schema__"):
            class BadWorker(Base):
                __config_schema__ = "not_a_model"
                def do_work(self) -> str:
                    return "bad"

    def test_no_config_schema_is_fine(self) -> None:
        """Omitting __config_schema__ is valid (optional)."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class PlainWorker(Base):
            def do_work(self) -> str:
                return "plain"

        assert registry.get("plain_worker") is PlainWorker


# ===================================================================
# ABCMeta compatibility
# ===================================================================

class TestABCMetaCompatibility:
    """Tests that base_metaclass parameter resolves ABCMeta conflicts."""

    def test_abc_meta_base(self) -> None:
        """Using base_metaclass=ABCMeta allows combining with ABC."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform, base_metaclass=ABCMeta)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class ConcreteWorker(Base):
            def do_work(self) -> str:
                return "concrete"

        assert registry.get("concrete_worker") is ConcreteWorker
        # The class should also be an instance of ABCMeta
        assert isinstance(ConcreteWorker, ABCMeta)


# ===================================================================
# Multiple inheritance
# ===================================================================

class TestMultipleInheritance:
    """Tests for classes with multiple base classes."""

    def test_diamond_inheritance(self) -> None:
        """Diamond inheritance: multiple paths to metaclass -> still registered once."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class MixinA(Base):
            __abstract__ = True

        class MixinB(Base):
            __abstract__ = True

        class Diamond(MixinA, MixinB):
            def do_work(self) -> str:
                return "diamond"

        assert "diamond" in registry.keys()


# ===================================================================
# kwargs passthrough
# ===================================================================

class TestKwargsPassthrough:
    """Test that **kwargs are passed through to super().__new__."""

    def test_class_with_init_subclass_kwargs(self) -> None:
        """Class definitions with extra kwargs should work."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def __init_subclass__(cls, tag: str = "", **kwargs: object) -> None:
                super().__init_subclass__(**kwargs)
                cls.tag = tag  # type: ignore[attr-defined]
            def do_work(self) -> str:
                return ""

        class Tagged(Base, tag="hello"):
            def do_work(self) -> str:
                return "tagged"

        assert Tagged.tag == "hello"  # type: ignore[attr-defined]
        assert registry.get("tagged") is Tagged


# ===================================================================
# AutoRegistrar qualname
# ===================================================================

class TestAutoRegistrarQualname:
    """Tests for the qualname of the generated metaclass."""

    def test_qualname_includes_registry_name(self) -> None:
        registry = LayerRegistry("agent", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)
        assert "agent" in Meta.__qualname__.lower() or "agent" in Meta.__name__.lower()


# ===================================================================
# Real scenario: Alice defines Agent layer
# ===================================================================

class TestAliceDefinesAgentLayer:
    """Real scenario: Alice creates Agent registrar + base class + 3 built-in agents."""

    def test_alice_defines_agent_layer(self) -> None:
        registry = LayerRegistry("agent", AgentLikeProtocol)
        kt = make_key_transform(suffixes=["Agent"])
        Meta = create_auto_registrar(registry, kt)

        class BaseAgent(metaclass=Meta):
            __abstract__ = True
            async def step(self, task: str) -> str:
                raise NotImplementedError
            def reset(self) -> None:
                raise NotImplementedError

        # Alice writes 3 built-in agents
        class BrowserUseAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "browser_use result"
            def reset(self) -> None:
                pass

        class SkyvernAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "skyvern result"
            def reset(self) -> None:
                pass

        class AgentTarsAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "agent_tars result"
            def reset(self) -> None:
                pass

        # Verify: all 3 registered, keys correct
        assert registry.get("browser_use") is BrowserUseAgent
        assert registry.get("skyvern") is SkyvernAgent
        assert registry.get("agent_tars") is AgentTarsAgent
        # Base class NOT registered
        registered_classes = [v for _, v in registry.items()]
        assert BaseAgent not in registered_classes


# ===================================================================
# Additional edge cases
# ===================================================================

class TestAutoRegistrarEdgeCases:
    """Additional edge cases for create_auto_registrar."""

    def test_config_schema_as_dict_raises(self) -> None:
        """__config_schema__ = {} (dict instance) should raise TypeError."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        with pytest.raises(TypeError, match="__config_schema__"):
            class BadSchema(Base):
                __config_schema__ = {"key": "value"}
                def do_work(self) -> str:
                    return "bad"

    def test_config_schema_as_int_raises(self) -> None:
        """__config_schema__ = 42 should raise TypeError."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        with pytest.raises(TypeError, match="__config_schema__"):
            class IntSchema(Base):
                __config_schema__ = 42
                def do_work(self) -> str:
                    return "int"

    def test_two_registries_same_class_name_different_registries(self) -> None:
        """Two different registries can register classes with the same inferred key."""
        reg1 = LayerRegistry("layer_a", SimpleProto)
        reg2 = LayerRegistry("layer_b", SimpleProto)
        Meta1 = create_auto_registrar(reg1, default_key_transform)
        Meta2 = create_auto_registrar(reg2, default_key_transform)

        class Base1(metaclass=Meta1):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Base2(metaclass=Meta2):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Worker1(Base1):
            def do_work(self) -> str:
                return "1"

        class Worker2(Base2):
            def do_work(self) -> str:
                return "2"

        # Both have "worker1" / "worker2" but in different registries
        assert reg1.get("worker1") is Worker1
        assert reg2.get("worker2") is Worker2
        assert reg1.get_or_none("worker2") is None
        assert reg2.get_or_none("worker1") is None

    def test_key_transform_with_suffix_applied(self) -> None:
        """Verify that the key_transform is actually applied during registration."""
        registry = LayerRegistry("test", SimpleProto)
        kt = make_key_transform(suffixes=["Impl"])
        Meta = create_auto_registrar(registry, kt)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class FastImpl(Base):
            def do_work(self) -> str:
                return "fast"

        assert registry.get("fast") is FastImpl
        assert FastImpl.__registry_key__ == "fast"
