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

import enum
from abc import ABCMeta
from typing import Protocol, runtime_checkable

import pytest

from conscribe.exceptions import DuplicateKeyError
from conscribe.registration.registry import LayerRegistry
from conscribe.registration.key_transform import make_key_transform, default_key_transform
from conscribe.registration.auto import create_auto_registrar
from conscribe.testing import isolated_registries


class Kind(str, enum.Enum):
    REACT = "react"


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


# ===================================================================
# Pydantic Generic skipping
# ===================================================================

class TestPydanticGenericSkipping:
    """Classes with '[' in their name (Pydantic Generic intermediates) should be skipped."""

    def test_bracket_name_skipped_by_default(self) -> None:
        """Class with '[' in name is NOT registered when skip_pydantic_generic=True (default)."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        # Simulate Pydantic Generic intermediate by creating class with bracket name
        BracketClass = Meta("Foo[bar]", (Base,), {"do_work": lambda self: "bracket"})

        assert registry.keys() == []
        # Class should still be created (not None)
        assert BracketClass is not None

    def test_bracket_name_registered_when_disabled(self) -> None:
        """Class with '[' in name IS registered when skip_pydantic_generic=False."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(
            registry, default_key_transform, skip_pydantic_generic=False
        )

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        BracketClass = Meta("Foo[bar]", (Base,), {"do_work": lambda self: "bracket"})

        assert len(registry.keys()) == 1

    def test_custom_skip_filter(self) -> None:
        """skip_filter callable can reject specific classes."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(
            registry,
            default_key_transform,
            skip_filter=lambda cls: cls.__name__ == "Rejected",
        )

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Accepted(Base):
            def do_work(self) -> str:
                return "accepted"

        class Rejected(Base):
            def do_work(self) -> str:
                return "rejected"

        assert "accepted" in registry.keys()
        assert len(registry.keys()) == 1

    def test_hierarchical_key_no_cascade(self) -> None:
        """Deep inheritance should NOT cascade parent keys into grandchild keys.

        Without __dict__.get fix, a 3-level chain A→B→C with separator="."
        would produce "a.b.a.b.c" instead of "a.b.c".
        """
        registry = LayerRegistry("test", SimpleProto, separator=".")
        Meta = create_auto_registrar(
            registry, default_key_transform, key_separator=".",
        )

        class Root(metaclass=Meta):
            __abstract__ = True
            __registry_key__ = "a"
            def do_work(self) -> str:
                return ""

        class Middle(Root):
            __registry_key__ = "a.b"
            def do_work(self) -> str:
                return "mid"

        class Leaf(Middle):
            def do_work(self) -> str:
                return "leaf"

        # Leaf should be "a.b.leaf", NOT "a.b.a.b.leaf"
        assert "a.b.leaf" in registry.keys()
        assert not any(k.count("a.b") > 1 for k in registry.keys())

    def test_normal_classes_unaffected_by_skip(self) -> None:
        """Normal classes (no bracket in name) are registered as usual."""
        registry = LayerRegistry("test", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class NormalWorker(Base):
            def do_work(self) -> str:
                return "normal"

        assert "normal_worker" in registry.keys()


# ===================================================================
# Atomic registration (Path A)
# ===================================================================

class TestAtomicRegistration:
    """A class whose definition fails leaves no binding in any registry."""

    @staticmethod
    def _base(registry: LayerRegistry) -> type:
        Meta = create_auto_registrar(registry, default_key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True

            def do_work(self) -> str:
                return ""

        return Base

    def test_multi_key_collision_leaves_no_partial_key(self) -> None:
        registry = LayerRegistry("test_atomic_multi", SimpleProto)
        Base = self._base(registry)

        class Holder(Base):
            __registry_key__ = "taken"

        with pytest.raises(DuplicateKeyError, match="taken"):
            class Partial(Base):
                __registry_key__ = ["fresh", "taken"]

        assert registry.keys() == ["taken"]
        assert registry.get("taken") is Holder

    def test_key_listed_twice_leaves_nothing(self) -> None:
        registry = LayerRegistry("test_atomic_twice", SimpleProto)
        Base = self._base(registry)

        with pytest.raises(DuplicateKeyError, match="listed twice"):
            class Twice(Base):
                __registry_key__ = ["dup", "dup"]

        assert registry.keys() == []

    def test_invalid_config_schema_registers_nothing(self) -> None:
        registry = LayerRegistry("test_atomic_schema", SimpleProto)
        Base = self._base(registry)

        with pytest.raises(TypeError, match="__config_schema__"):
            class BadSchema(Base):
                __config_schema__ = {"not": "a model"}

        assert registry.keys() == []

    def test_failure_after_add_purges(self) -> None:
        class Picky(type):
            """Refuses the re-creation stamp, written after the registry adds."""

            def __setattr__(cls, name: str, value: object) -> None:
                if name == "__registry_origin__":
                    raise RuntimeError("late failure")
                super().__setattr__(name, value)

        registry = LayerRegistry("test_atomic_late", SimpleProto)
        Meta = create_auto_registrar(registry, default_key_transform, base_metaclass=Picky)

        class Base(metaclass=Meta):
            __abstract__ = True

            def do_work(self) -> str:
                return ""

        with pytest.raises(RuntimeError, match="late failure"):
            class Late(Base):
                __registry_key__ = ["late_a", "late_b"]

        assert registry.keys() == []

    def test_failure_in_injection_registers_nothing(self, monkeypatch) -> None:
        import conscribe.wiring

        registry = LayerRegistry("test_atomic_inject", SimpleProto)
        Base = self._base(registry)

        def failing_inject(cls: type) -> None:
            raise RuntimeError("inject failed")

        monkeypatch.setattr(conscribe.wiring, "inject_wired_descriptors", failing_inject)
        with pytest.raises(RuntimeError, match="inject failed"):
            class Broken(Base):
                pass

        assert registry.keys() == []

    def test_registered_class_is_stamped(self) -> None:
        registry = LayerRegistry("test_stamp_a", SimpleProto)
        Base = self._base(registry)

        class Stamped(Base):
            pass

        assert Stamped.__registry_origin__ == id(Stamped)
        assert "__registry_origin__" not in Base.__dict__  # abstract: not registered


# ===================================================================
# Registry keys are validated up front (spec 4.9)
# ===================================================================

class TestRegistryKeyValidation:
    """Review probes 4 and 26: odd keys registered under keys nobody looks up."""

    @staticmethod
    def _base(registry: LayerRegistry, key_transform=default_key_transform) -> type:
        Meta = create_auto_registrar(registry, key_transform)

        class Base(metaclass=Meta):
            __abstract__ = True

            def do_work(self) -> str:
                return ""

        return Base

    @pytest.mark.parametrize(
        ("key", "fragment"),
        [
            (Kind.REACT, "not the Enum member <Kind.REACT: 'react'> (str() of it is 'Kind.REACT'"),
            (("a", "b"), "a tuple is not a key; list several keys in a list"),
            (3, "a registry key must be a non-empty str, got int 3"),
            ("", "a registry key must be a non-empty str"),
            ([], "an empty list registers the class under no key"),
            (["a", 1], "every key in the list must be a non-empty str, got int 1"),
            (["a", ""], "every key in the list must be a non-empty str"),
            (["a", Kind.REACT], "every key in the list must be a plain str, not the Enum member"),
        ],
    )
    def test_invalid_keys_raise_before_registering(self, key, fragment) -> None:
        from conscribe import InvalidRegistryKeyError, RegistryError

        with isolated_registries():
            registry = LayerRegistry("test_key_validation", SimpleProto)
            Base = self._base(registry)
            with pytest.raises(InvalidRegistryKeyError) as info:
                class Odd(Base):
                    __registry_key__ = key

            assert registry.keys() == []
            assert isinstance(info.value, RegistryError) and isinstance(info.value, TypeError)
            assert info.value.source == "__registry_key__"
            assert str(info.value).startswith(
                "TestRegistryKeyValidation.test_invalid_keys_raise_before_registering.<locals>"
                ".Odd: invalid registry key "
            )
            assert fragment in str(info.value)

    def test_enum_member_fix_names_the_value(self) -> None:
        from conscribe import InvalidRegistryKeyError

        with isolated_registries():
            registry = LayerRegistry("test_key_enum", SimpleProto)
            Base = self._base(registry)
            with pytest.raises(InvalidRegistryKeyError, match="write the plain string 'react'"):
                class ReactLoop(Base):
                    __registry_key__ = Kind.REACT

    def test_valid_keys_still_register(self) -> None:
        with isolated_registries():
            registry = LayerRegistry("test_key_valid", SimpleProto)
            Base = self._base(registry)

            class One(Base):
                __registry_key__ = "one"

            class Many(Base):
                __registry_key__ = ["a", "b"]

            assert registry.keys() == ["one", "a", "b"]
            assert Many.__registry_keys__ == ["a", "b"]

    def test_key_transform_result_is_validated(self) -> None:
        from conscribe import InvalidRegistryKeyError

        with isolated_registries():
            registry = LayerRegistry("test_key_transform", SimpleProto)
            Base = self._base(registry, key_transform=lambda name: None)
            with pytest.raises(InvalidRegistryKeyError, match="from key_transform"):
                class Inferred(Base):
                    pass
            assert registry.keys() == []
