"""Tests for predicate-based registration filter system."""
from __future__ import annotations

import pytest
from typing import Protocol, runtime_checkable

from conscribe import create_registrar
from conscribe.registration.filters import (
    AbstractFilter,
    ChildSkipFilter,
    CustomCallableFilter,
    ParentRegistrationFilter,
    PropagationFilter,
    PydanticGenericFilter,
    RegistrationContext,
    RootFilter,
    build_filter_chain,
    should_skip_registration,
)


def _make_ctx(
    *,
    name: str = "MyClass",
    bases: tuple = (object,),
    namespace: dict | None = None,
    registry_name: str = "test",
    cls: type | None = None,
) -> RegistrationContext:
    """Helper to build a RegistrationContext for testing."""
    if namespace is None:
        namespace = {}
    if cls is None:
        cls = type(name, bases, namespace)
    return RegistrationContext(
        cls=cls,
        name=name,
        bases=bases,
        namespace=namespace,
        registry_name=registry_name,
    )


class TestRootFilter:
    def test_skips_when_no_bases(self):
        ctx = _make_ctx(bases=())
        assert RootFilter().should_skip(ctx) is True

    def test_passes_when_has_bases(self):
        ctx = _make_ctx(bases=(object,))
        assert RootFilter().should_skip(ctx) is False


class TestPydanticGenericFilter:
    def test_skips_bracket_name(self):
        ctx = _make_ctx(name="BaseEvent[str]")
        assert PydanticGenericFilter().should_skip(ctx) is True

    def test_passes_normal_name(self):
        ctx = _make_ctx(name="NormalClass")
        assert PydanticGenericFilter().should_skip(ctx) is False


class TestAbstractFilter:
    def test_skips_abstract_in_namespace(self):
        ctx = _make_ctx(namespace={"__abstract__": True})
        assert AbstractFilter().should_skip(ctx) is True

    def test_passes_no_abstract(self):
        ctx = _make_ctx(namespace={})
        assert AbstractFilter().should_skip(ctx) is False

    def test_passes_abstract_false(self):
        ctx = _make_ctx(namespace={"__abstract__": False})
        assert AbstractFilter().should_skip(ctx) is False


class TestCustomCallableFilter:
    def test_skips_when_fn_returns_true(self):
        ctx = _make_ctx(name="SkipMe")
        f = CustomCallableFilter(lambda c: True)
        assert f.should_skip(ctx) is True

    def test_passes_when_fn_returns_false(self):
        ctx = _make_ctx(name="KeepMe")
        f = CustomCallableFilter(lambda c: False)
        assert f.should_skip(ctx) is False


class TestChildSkipFilter:
    def test_skips_when_registry_in_skip_list(self):
        ctx = _make_ctx(
            namespace={"__skip_registries__": ["test", "other"]},
            registry_name="test",
        )
        assert ChildSkipFilter().should_skip(ctx) is True

    def test_passes_when_registry_not_in_skip_list(self):
        ctx = _make_ctx(
            namespace={"__skip_registries__": ["other"]},
            registry_name="test",
        )
        assert ChildSkipFilter().should_skip(ctx) is False

    def test_passes_when_no_skip_list(self):
        ctx = _make_ctx(namespace={}, registry_name="test")
        assert ChildSkipFilter().should_skip(ctx) is False


class TestParentRegistrationFilter:
    def test_skips_when_parent_filter_returns_false(self):
        class Parent:
            @staticmethod
            def __registration_filter__(child_cls):
                return False

        class Child(Parent):
            pass

        ctx = _make_ctx(
            cls=Child,
            bases=(Parent,),
            name="Child",
        )
        assert ParentRegistrationFilter().should_skip(ctx) is True

    def test_passes_when_parent_filter_returns_true(self):
        class Parent:
            @staticmethod
            def __registration_filter__(child_cls):
                return True

        class Child(Parent):
            pass

        ctx = _make_ctx(
            cls=Child,
            bases=(Parent,),
            name="Child",
        )
        assert ParentRegistrationFilter().should_skip(ctx) is False

    def test_passes_when_no_parent_filter(self):
        class Parent:
            pass

        class Child(Parent):
            pass

        ctx = _make_ctx(
            cls=Child,
            bases=(Parent,),
            name="Child",
        )
        assert ParentRegistrationFilter().should_skip(ctx) is False


class TestPropagationFilter:
    def test_skips_when_parent_propagate_false(self):
        class Parent:
            __propagate__ = False

        class Child(Parent):
            pass

        ctx = _make_ctx(cls=Child, bases=(Parent,), name="Child")
        assert PropagationFilter().should_skip(ctx) is True

    def test_passes_when_parent_propagate_true(self):
        class Parent:
            __propagate__ = True

        class Child(Parent):
            pass

        ctx = _make_ctx(cls=Child, bases=(Parent,), name="Child")
        assert PropagationFilter().should_skip(ctx) is False

    def test_skips_when_depth_exceeded(self):
        class Parent:
            __propagate_depth__ = 1

        class Child(Parent):
            pass

        class GrandChild(Child):
            pass

        # GrandChild has bases=(Child,). Child inherits __propagate_depth__=1
        # from Parent. Depth from GrandChild to Child (which has the attr) is 1,
        # but depth from GrandChild to Parent (originator) is 2.
        # The filter measures depth from cls to the base that has the attribute.
        # Since Child inherits it from Parent, getattr(Child, '__propagate_depth__')
        # returns 1. Depth from GrandChild to Parent in MRO is 2 > 1.
        ctx = _make_ctx(cls=GrandChild, bases=(Child,), name="GrandChild")
        assert PropagationFilter().should_skip(ctx) is True

    def test_passes_when_depth_within_limit(self):
        class Parent:
            __propagate_depth__ = 1

        class Child(Parent):
            pass

        ctx = _make_ctx(cls=Child, bases=(Parent,), name="Child")
        assert PropagationFilter().should_skip(ctx) is False


class TestBuildFilterChain:
    def test_default_chain_has_expected_filters(self):
        chain = build_filter_chain(registry_name="test")
        filter_types = [type(f).__name__ for f in chain]
        assert "RootFilter" in filter_types
        assert "AbstractFilter" in filter_types
        assert "PydanticGenericFilter" in filter_types
        assert "ChildSkipFilter" in filter_types
        assert "ParentRegistrationFilter" in filter_types
        assert "PropagationFilter" in filter_types

    def test_no_pydantic_generic_filter_when_disabled(self):
        chain = build_filter_chain(
            skip_pydantic_generic=False, registry_name="test"
        )
        filter_types = [type(f).__name__ for f in chain]
        assert "PydanticGenericFilter" not in filter_types

    def test_custom_filter_included(self):
        chain = build_filter_chain(
            skip_filter=lambda c: False, registry_name="test"
        )
        filter_types = [type(f).__name__ for f in chain]
        assert "CustomCallableFilter" in filter_types


class TestShouldSkipRegistration:
    def test_returns_true_on_first_match(self):
        chain = [RootFilter()]
        ctx = _make_ctx(bases=())
        assert should_skip_registration(chain, ctx) is True

    def test_returns_false_when_no_match(self):
        chain = [RootFilter(), AbstractFilter()]
        ctx = _make_ctx(bases=(object,), namespace={})
        assert should_skip_registration(chain, ctx) is False


class TestChildSkipFilterIntegration:
    """Test __skip_registries__ with actual registrars."""

    def test_skip_specific_registry(self):
        @runtime_checkable
        class P1(Protocol):
            def work(self) -> str: ...

        @runtime_checkable
        class P2(Protocol):
            def work(self) -> str: ...

        R1 = create_registrar("r1", P1)
        R2 = create_registrar("r2", P2)

        CombinedMeta = R1.Meta | R2.Meta

        class Base(metaclass=CombinedMeta):
            __abstract__ = True
            def work(self) -> str:
                return "base"

        class RegisterBoth(Base):
            pass

        class SkipR2(Base):
            __skip_registries__ = ["r2"]

        assert R1.get_or_none("register_both") is RegisterBoth
        assert R2.get_or_none("register_both") is RegisterBoth
        assert R1.get_or_none("skip_r2") is SkipR2
        assert R2.get_or_none("skip_r2") is None

        # Cleanup
        for key in list(R1.keys()):
            R1.unregister(key)
        for key in list(R2.keys()):
            R2.unregister(key)


class TestParentRegistrationFilterIntegration:
    """Test __registration_filter__ with actual registrars."""

    def test_parent_blocks_child_by_name(self):
        @runtime_checkable
        class P(Protocol):
            def work(self) -> str: ...

        R = create_registrar("pf_test", P)

        class Base(metaclass=R.Meta):
            __abstract__ = True

            @staticmethod
            def __registration_filter__(child_cls):
                return "Test" not in child_cls.__name__

            def work(self) -> str:
                return "base"

        class ValidImpl(Base):
            pass

        class TestImpl(Base):
            pass

        assert R.get_or_none("valid_impl") is ValidImpl
        assert R.get_or_none("test_impl") is None

        for key in list(R.keys()):
            R.unregister(key)


class TestPropagationFilterIntegration:
    """Test __propagate__ and __propagate_depth__ with actual registrars."""

    def test_propagate_false_blocks_children(self):
        @runtime_checkable
        class P(Protocol):
            def work(self) -> str: ...

        R = create_registrar("prop_test", P)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            __propagate__ = False

            def work(self) -> str:
                return "base"

        class Child(Base):
            pass

        assert R.get_or_none("child") is None

        for key in list(R.keys()):
            R.unregister(key)

    def test_propagate_depth_limits_registration(self):
        @runtime_checkable
        class P(Protocol):
            def work(self) -> str: ...

        R = create_registrar("depth_test", P)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            __propagate_depth__ = 1

            def work(self) -> str:
                return "base"

        class Child(Base):
            """Depth 1 from Base — should register."""
            pass

        class GrandChild(Child):
            """Depth 2 from Base — should NOT register."""
            pass

        assert R.get_or_none("child") is Child
        assert R.get_or_none("grand_child") is None

        for key in list(R.keys()):
            R.unregister(key)
