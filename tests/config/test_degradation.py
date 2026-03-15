"""Tests for conscribe.config.degradation — type degradation core logic.

Tests ``check_type_compatibility()``, ``degrade_field_definitions()``,
``format_type_repr()``, and ``DegradedField`` data class.

Uses a synthetic incompatible type (plain class with no Pydantic support)
to test degradation without depending on specific third-party packages.
"""
from __future__ import annotations

from typing import Any, ForwardRef, Optional, Union

import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from conscribe.config.degradation import (
    DegradedField,
    check_type_compatibility,
    degrade_field_definitions,
    format_type_repr,
)


# ===================================================================
# Synthetic incompatible type
# ===================================================================

class _Incompatible:
    """A class Pydantic cannot serialize (no validators, no __get_pydantic_core_schema__)."""

    def __init__(self, x: int):
        self.x = x


# ===================================================================
# check_type_compatibility
# ===================================================================

class TestCheckTypeCompatibility:
    """Tests for check_type_compatibility()."""

    def test_basic_types_compatible(self) -> None:
        """str, int, float, bool are compatible."""
        for tp in (str, int, float, bool):
            assert check_type_compatibility(tp) is True

    def test_list_str_compatible(self) -> None:
        """list[str] is compatible."""
        assert check_type_compatibility(list[str]) is True

    def test_optional_str_compatible(self) -> None:
        """Optional[str] is compatible."""
        assert check_type_compatibility(Optional[str]) is True

    def test_any_compatible(self) -> None:
        """Any is compatible."""
        assert check_type_compatibility(Any) is True

    def test_basemodel_compatible(self) -> None:
        """BaseModel subclass is compatible."""

        class MyModel(BaseModel):
            x: int

        assert check_type_compatibility(MyModel) is True

    def test_arbitrary_class_incompatible(self) -> None:
        """A plain class without Pydantic support is incompatible."""
        assert check_type_compatibility(_Incompatible) is False

    def test_dict_str_any_compatible(self) -> None:
        """dict[str, Any] is compatible."""
        assert check_type_compatibility(dict[str, Any]) is True


# ===================================================================
# degrade_field_definitions
# ===================================================================

class TestDegradeFieldDefinitions:
    """Tests for degrade_field_definitions()."""

    def test_degrade_replaces_type_with_any(self) -> None:
        """Incompatible type is replaced with Any."""
        field_defs = {"auth": (_Incompatible, None)}
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert len(degraded) == 1
        assert degraded[0].field_name == "auth"
        assert cleaned["auth"][0] is Any

    def test_degrade_preserves_default(self) -> None:
        """Default value is preserved when type is degraded."""
        sentinel = object()
        field_defs = {"auth": (_Incompatible, sentinel)}
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert cleaned["auth"][1] is sentinel
        assert len(degraded) == 1

    def test_degrade_preserves_field_info(self) -> None:
        """FieldInfo instance is preserved when type is degraded."""
        info = Field(description="auth field")
        field_defs = {"auth": (_Incompatible, info)}
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert isinstance(cleaned["auth"][1], FieldInfo)
        assert cleaned["auth"][1].description == "auth field"
        assert len(degraded) == 1

    def test_degrade_returns_empty_on_all_compatible(self) -> None:
        """No degradation when all fields are compatible."""
        field_defs = {
            "name": (str, "default"),
            "count": (int, 0),
        }
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert degraded == []
        assert cleaned == field_defs

    def test_degrade_mixed_compatible_and_incompatible(self) -> None:
        """Only incompatible fields are degraded; compatible ones are kept."""
        field_defs = {
            "name": (str, "default"),
            "auth": (_Incompatible, None),
            "count": (int, 0),
        }
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert len(degraded) == 1
        assert degraded[0].field_name == "auth"
        assert cleaned["name"][0] is str
        assert cleaned["auth"][0] is Any
        assert cleaned["count"][0] is int

    def test_degrade_records_source_class(self) -> None:
        """DegradedField records the source_class name."""
        field_defs = {"auth": (_Incompatible, None)}
        _, degraded = degrade_field_definitions(field_defs, "httpx.Client")

        assert degraded[0].source_class == "httpx.Client"

    def test_degrade_records_original_type_repr(self) -> None:
        """DegradedField records a human-readable original type."""
        field_defs = {"auth": (_Incompatible, None)}
        _, degraded = degrade_field_definitions(field_defs, "TestClass")

        # Should contain the class name
        assert "_Incompatible" in degraded[0].original_type_repr

    def test_degrade_reason_default(self) -> None:
        """DegradedField reason defaults to 'pydantic_incompatible'."""
        field_defs = {"auth": (_Incompatible, None)}
        _, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert degraded[0].reason == "pydantic_incompatible"


# ===================================================================
# format_type_repr
# ===================================================================

class TestFormatTypeRepr:
    """Tests for format_type_repr()."""

    def test_basic_types(self) -> None:
        """Basic types render as their names."""
        assert format_type_repr(str) == "str"
        assert format_type_repr(int) == "int"
        assert format_type_repr(float) == "float"
        assert format_type_repr(bool) == "bool"

    def test_none_type(self) -> None:
        """None and NoneType render as 'None'."""
        assert format_type_repr(None) == "None"
        assert format_type_repr(type(None)) == "None"

    def test_optional(self) -> None:
        """Optional[str] renders as 'str | None'."""
        result = format_type_repr(Optional[str])
        assert "str" in result
        assert "None" in result
        assert "|" in result

    def test_union(self) -> None:
        """Union[str, int] renders with '|'."""
        result = format_type_repr(Union[str, int])
        assert "str" in result
        assert "int" in result
        assert "|" in result

    def test_list_type(self) -> None:
        """list[str] renders as 'list[str]'."""
        result = format_type_repr(list[str])
        assert result == "list[str]"

    def test_dict_type(self) -> None:
        """dict[str, int] renders correctly."""
        result = format_type_repr(dict[str, int])
        assert "dict" in result
        assert "str" in result
        assert "int" in result

    def test_class_with_module(self) -> None:
        """A class renders as module.ClassName."""
        result = format_type_repr(_Incompatible)
        assert "_Incompatible" in result

    def test_any(self) -> None:
        """Any renders as 'Any'."""
        assert format_type_repr(Any) == "Any"


# ===================================================================
# DegradedField dataclass
# ===================================================================

class TestDegradedField:
    """Tests for the DegradedField frozen dataclass."""

    def test_frozen(self) -> None:
        """DegradedField is frozen (immutable)."""
        df = DegradedField(
            field_name="auth",
            original_type_repr="httpx.Auth",
            source_class="httpx.Client",
        )
        with pytest.raises(AttributeError):
            df.field_name = "changed"  # type: ignore[misc]

    def test_default_reason(self) -> None:
        """Default reason is 'pydantic_incompatible'."""
        df = DegradedField(
            field_name="auth",
            original_type_repr="httpx.Auth",
            source_class="httpx.Client",
        )
        assert df.reason == "pydantic_incompatible"


# ===================================================================
# format_type_repr — edge cases
# ===================================================================

class TestFormatTypeReprEdgeCases:
    """Edge case tests for format_type_repr covering uncommon type forms."""

    def test_forward_ref(self) -> None:
        """ForwardRef renders as the referenced string."""
        ref = ForwardRef("MyClass")
        result = format_type_repr(ref)
        assert result == "MyClass"

    def test_bare_union_no_args(self) -> None:
        """typing.Union without args (bare) renders via repr fallback."""
        # Union without subscript — accessing __args__ gives None
        result = format_type_repr(Union)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generic_origin_without_args(self) -> None:
        """A type with __origin__ but no __args__ uses origin name."""
        # Create a synthetic object with __origin__ but no __args__
        class _FakeGeneric:
            __origin__ = list
            __args__ = None

        result = format_type_repr(_FakeGeneric())
        assert result == "list"

    def test_non_type_non_none_fallback(self) -> None:
        """A value that is not a type, not None, not Any falls back to repr."""
        result = format_type_repr(42)
        assert result == "42"

        result2 = format_type_repr("hello")
        assert result2 == "'hello'"

    def test_builtin_type_no_module_prefix(self) -> None:
        """Builtin types (int, str) render without module prefix."""
        assert format_type_repr(int) == "int"
        assert format_type_repr(str) == "str"


# ===================================================================
# degrade_field_definitions — edge cases
# ===================================================================

class TestDegradeFieldDefinitionsEdgeCases:
    """Edge case tests for degrade_field_definitions."""

    def test_non_tuple_value_passed_through(self) -> None:
        """Non-tuple field_definitions values are passed through unchanged."""
        field_defs = {
            "normal": (str, "default"),
            "weird": "not_a_tuple",
        }
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert degraded == []
        assert cleaned["weird"] == "not_a_tuple"
        assert cleaned["normal"] == (str, "default")

    def test_three_element_tuple_passed_through(self) -> None:
        """A 3-element tuple is treated as non-standard and passed through."""
        field_defs = {
            "triple": (str, "default", "extra"),
        }
        cleaned, degraded = degrade_field_definitions(field_defs, "TestClass")

        assert degraded == []
        assert cleaned["triple"] == (str, "default", "extra")
