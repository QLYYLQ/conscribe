"""Cross-framework Generic compatibility tests for conscribe.

Tests that Pydantic Generic specializations (which create real class objects
with ``[`` in their name) are correctly filtered from registration, while
stdlib/third-party Generic aliases (which are NOT real classes) never
trigger metaclass hooks.

Also includes a full pipeline test: register + Generic[T] via bridge
→ clean registry → config extraction → discriminated union → codegen.
"""
from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import pytest
from pydantic import BaseModel, Field

from conscribe import create_registrar
from conscribe.config.builder import build_layer_config
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.extractor import extract_config_schema


T = TypeVar("T")


@runtime_checkable
class EventProto(Protocol):
    def handle(self) -> str: ...


# ===================================================================
# Group A: Pydantic-family (create real classes with [ in name)
# ===================================================================


class TestPydanticBaseModelGeneric:
    """Pydantic BaseModel + Generic[T] — bracket intermediates must be filtered."""

    def test_bridge_generic_basemodel_no_bracket_keys(self) -> None:
        """bridge(BaseModel) + Generic[T] specialization → no bracket keys."""
        R = create_registrar("event", EventProto, discriminator_field="type")

        class BaseEvent(BaseModel, Generic[T]):
            payload: T  # type: ignore[valid-type]

            def handle(self) -> str:
                return f"handled {self.payload}"

        Bridge = R.bridge(BaseEvent)

        class StringEvent(Bridge):
            payload: str = "hello"

            def handle(self) -> str:
                return f"string: {self.payload}"

        # Verify no bracket keys
        bracket_keys = [k for k in R.keys() if "[" in k]
        assert bracket_keys == [], f"Unexpected bracket keys: {bracket_keys}"

        # StringEvent should be registered
        assert R.get("string_event") is StringEvent

    def test_metaclass_generic_specialization_filtered(self) -> None:
        """Classes with metaclass + Generic[T] specialization are filtered."""
        R = create_registrar("event", EventProto, discriminator_field="type")

        class Base(metaclass=R.Meta):
            __abstract__ = True
            data: Any = None

            def handle(self) -> str:
                return "base"

        class GenericBase(Base, Generic[T]):
            __abstract__ = True

        class ConcreteEvent(GenericBase):
            def __init__(self, *, message: str = "hello"):
                self.message = message

            def handle(self) -> str:
                return self.message

        # ConcreteEvent should be registered, no bracket keys
        assert "concrete_event" in R.keys()
        bracket_keys = [k for k in R.keys() if "[" in k]
        assert bracket_keys == []

    def test_config_extraction_works_for_basemodel_subclass(self) -> None:
        """Config extraction works for BaseModel subclasses via fast path."""

        class MyEvent(BaseModel):
            name: str
            priority: int = Field(default=0, ge=0)

            def handle(self) -> str:
                return self.name

        result = extract_config_schema(MyEvent)
        assert result is not None
        assert "name" in result.model_fields
        assert "priority" in result.model_fields


# ===================================================================
# Group B: stdlib/typing (return _GenericAlias, NOT real classes)
# ===================================================================


class TestStdlibGeneric:
    """stdlib Generic[T] specializations return _GenericAlias, not real classes."""

    def test_stdlib_generic_is_not_type(self) -> None:
        """Foo[int] where Foo(Base, Generic[T]) returns _GenericAlias, not a type."""
        R = create_registrar("test", EventProto)

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def handle(self) -> str:
                return ""

        class Foo(Base, Generic[T]):
            __abstract__ = True
            def handle(self) -> str:
                return "foo"

        # Foo[int] should be a _GenericAlias, NOT a real class
        specialized = Foo[int]
        # It should NOT be a real type instance usable with isinstance
        assert not isinstance(specialized, type) or "[" not in getattr(specialized, "__name__", "")

    def test_list_int_is_not_real_class(self) -> None:
        """PEP 585 builtins like list[int] are types.GenericAlias, not real classes."""
        import types

        result = list[int]
        assert isinstance(result, types.GenericAlias)
        # It's NOT a real class, so it would never enter metaclass __new__

    def test_dict_str_int_is_not_real_class(self) -> None:
        """dict[str, int] is types.GenericAlias."""
        import types

        result = dict[str, int]
        assert isinstance(result, types.GenericAlias)


# ===================================================================
# Group C: Third-party frameworks
# ===================================================================


class TestAttrsGeneric:
    """attrs + Generic[T] produces _GenericAlias, not real class."""

    def test_attrs_generic_is_alias(self) -> None:
        """attrs @define class with Generic[T] returns _GenericAlias on specialization."""
        try:
            import attrs
        except ImportError:
            pytest.skip("attrs not installed")

        @attrs.define
        class AttrsFoo(Generic[T]):
            value: T  # type: ignore[valid-type]

        specialized = AttrsFoo[int]
        # attrs Generic specialization returns _GenericAlias
        assert not isinstance(specialized, type)


# ===================================================================
# Full pipeline test
# ===================================================================


class TestFullPipelinePydanticGeneric:
    """End-to-end: register + Generic[T] → clean registry → config → codegen."""

    def test_full_pipeline(self) -> None:
        """Full pipeline: bridge BaseModel Generic → register → config → codegen."""
        R = create_registrar(
            "event",
            EventProto,
            discriminator_field="type",
            strip_suffixes=["Event"],
        )

        class BaseEvent(BaseModel, Generic[T]):
            payload: T  # type: ignore[valid-type]

            def handle(self) -> str:
                return "base"

        Bridge = R.bridge(BaseEvent)

        class StringEvent(Bridge):
            payload: str = "hello"
            priority: int = Field(default=0, ge=0, description="Event priority")

            def handle(self) -> str:
                return f"string: {self.payload}"

        class IntEvent(Bridge):
            payload: int = 0
            label: str = Field(default="default", description="Event label")

            def handle(self) -> str:
                return f"int: {self.payload}"

        # 1. Clean registry — no bracket keys
        bracket_keys = [k for k in R.keys() if "[" in k]
        assert bracket_keys == []
        assert set(R.keys()) == {"string", "int"}

        # 2. Config extraction works for each key
        string_schema = R.get_config_schema("string")
        assert string_schema is not None
        assert "payload" in string_schema.model_fields
        assert "priority" in string_schema.model_fields

        int_schema = R.get_config_schema("int")
        assert int_schema is not None
        assert "payload" in int_schema.model_fields
        assert "label" in int_schema.model_fields

        # 3. Build discriminated union
        result = build_layer_config(R)
        assert result is not None
        assert "string" in result.per_key_models
        assert "int" in result.per_key_models

        # 4. Codegen produces valid source
        source = generate_layer_config_source(result)
        assert isinstance(source, str)
        assert "StringEventConfig" in source or "StringConfig" in source
        assert "IntEventConfig" in source or "IntConfig" in source

    def test_disable_skip_allows_bracket_registration(self) -> None:
        """With skip_pydantic_generic=False, bracket names register normally."""
        R = create_registrar(
            "event",
            EventProto,
            skip_pydantic_generic=False,
        )

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def handle(self) -> str:
                return ""

        # Manually create bracket-named class
        BracketClass = R.Meta(
            "Base[str]", (Base,), {"handle": lambda self: "bracket"}
        )

        # Should be registered since skip is disabled
        assert len(R.keys()) >= 1
        bracket_keys = [k for k in R.keys() if "[" in k or "base" in k]
        assert len(bracket_keys) >= 1
