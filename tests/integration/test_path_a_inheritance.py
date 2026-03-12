"""Integration tests: Path A (inheritance) edge cases.

Path A is the primary registration path: class FooAgent(BaseAgent) -> auto-registered.
This file tests edge cases and corner cases specific to metaclass-based inheritance.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.exceptions import DuplicateKeyError


@runtime_checkable
class WorkProtocol(Protocol):
    def do_work(self) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


class TestAbstractInheritanceEdgeCases:
    """Edge cases around __abstract__ flag and inheritance."""

    def test_abstract_does_not_leak_to_subclass(self) -> None:
        """CRITICAL: Base.__abstract__=True must not leak to Child via getattr."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Child(Base):
            def do_work(self) -> str:
                return "child"

        assert "child" in R.keys()

    def test_abstract_at_middle_level(self) -> None:
        """Abstract at middle level: only concrete leaves registered."""
        R = create_registrar("test", WorkProtocol)

        class Root(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Middle(Root):
            __abstract__ = True

        class LeafA(Middle):
            def do_work(self) -> str:
                return "a"

        class LeafB(Middle):
            def do_work(self) -> str:
                return "b"

        registered_keys = set(R.keys())
        assert "leaf_a" in registered_keys
        assert "leaf_b" in registered_keys
        assert "middle" not in registered_keys

    def test_abstract_false_at_child_still_registers(self) -> None:
        """Child with __abstract__ = False should be registered."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class ExplicitConcrete(Base):
            __abstract__ = False
            def do_work(self) -> str:
                return "explicit"

        # __abstract__ = False in namespace -> not truthy -> registered
        assert "explicit_concrete" in R.keys()

    def test_re_abstract_in_grandchild(self) -> None:
        """Grandchild re-declares __abstract__=True -> should be skipped."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Concrete(Base):
            def do_work(self) -> str:
                return "concrete"

        class ReAbstract(Concrete):
            __abstract__ = True

        class FinalLeaf(ReAbstract):
            def do_work(self) -> str:
                return "final"

        keys = set(R.keys())
        assert "concrete" in keys
        assert "re_abstract" not in keys
        assert "final_leaf" in keys


class TestRegistryKeyInheritanceEdgeCases:
    """Edge cases around __registry_key__ override."""

    def test_explicit_key_does_not_leak_to_child(self) -> None:
        """Parent's __registry_key__ should not propagate to child."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Parent(Base):
            __registry_key__ = "custom_parent"
            def do_work(self) -> str:
                return "parent"

        class Child(Parent):
            def do_work(self) -> str:
                return "child"

        assert R.get("custom_parent") is Parent
        assert R.get("child") is Child
        assert R.get_or_none("custom_parent") is Parent

    def test_child_can_also_override_key(self) -> None:
        """Both parent and child can have explicit __registry_key__."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Parent(Base):
            __registry_key__ = "p"
            def do_work(self) -> str:
                return "parent"

        class Child(Parent):
            __registry_key__ = "c"
            def do_work(self) -> str:
                return "child"

        assert R.get("p") is Parent
        assert R.get("c") is Child


class TestMultipleInheritanceEdgeCases:
    """Edge cases with multiple inheritance (diamond, mixins)."""

    def test_diamond_inheritance_single_registration(self) -> None:
        """Diamond pattern: class is registered once (not duplicated)."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Left(Base):
            __abstract__ = True

        class Right(Base):
            __abstract__ = True

        class Diamond(Left, Right):
            def do_work(self) -> str:
                return "diamond"

        assert R.keys().count("diamond") == 1

    def test_mixin_without_metaclass(self) -> None:
        """Mixin class (no metaclass) combined with registered base."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class LogMixin:
            def log(self) -> str:
                return "logged"

        class WithMixin(Base, LogMixin):
            def do_work(self) -> str:
                return "mixed"

        assert R.get("with_mixin") is WithMixin
        assert WithMixin().log() == "logged"

    def test_deep_inheritance_chain(self) -> None:
        """Deep chain: Root -> L1 -> L2 -> L3 -> Leaf, all registered except Root."""
        R = create_registrar("test", WorkProtocol)

        class Root(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Level1(Root):
            def do_work(self) -> str:
                return "l1"

        class Level2(Level1):
            def do_work(self) -> str:
                return "l2"

        class Level3(Level2):
            def do_work(self) -> str:
                return "l3"

        class Leaf(Level3):
            def do_work(self) -> str:
                return "leaf"

        keys = set(R.keys())
        assert keys == {"level1", "level2", "level3", "leaf"}


class TestABCMetaIntegration:
    """Path A with ABCMeta (base_metaclass parameter)."""

    def test_abc_abstract_method_and_registration(self) -> None:
        """Combine ABCMeta abstract methods with auto-registration."""
        R = create_registrar("test", WorkProtocol, base_metaclass=ABCMeta)

        class Base(metaclass=R.Meta):
            __abstract__ = True

            @abstractmethod
            def do_work(self) -> str: ...

        class Concrete(Base):
            def do_work(self) -> str:
                return "concrete"

        assert R.get("concrete") is Concrete


class TestKeyTransformEdgeCases:
    """Edge cases for key inference during Path A registration."""

    def test_class_name_equals_suffix(self) -> None:
        """When class name equals the suffix, don't strip (would produce empty)."""
        R = create_registrar("test", WorkProtocol, strip_suffixes=["Agent"])

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class Agent(Base):
            def do_work(self) -> str:
                return "agent"

        # "Agent" -> strip "Agent" -> empty -> don't strip -> "agent"
        assert "agent" in R.keys()

    def test_single_char_class_name(self) -> None:
        """Single character class name."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class X(Base):
            def do_work(self) -> str:
                return "x"

        assert "x" in R.keys()

    def test_all_uppercase_class_name(self) -> None:
        """All-uppercase class name (acronym)."""
        R = create_registrar("test", WorkProtocol)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class DOM(Base):
            def do_work(self) -> str:
                return "dom"

        assert "dom" in R.keys()
