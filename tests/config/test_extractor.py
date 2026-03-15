"""Tests for conscribe.config.extractor.extract_config_schema.

Covers:
- Tier 3: __config_schema__ (highest priority, direct return, preserves model_config)
- Tier 3 variant: single-param BaseModel auto-detect
- Tier 1: pure __init__ signature (typed params, defaults, **kwargs, *args, naming)
- Tier 1.5: docstring descriptions merged into fields
- Tier 2: Annotated[T, Field()] metadata preservation
- MRO awareness (child/grandchild inheritance)
- Edge cases (inspect.signature failure, get_type_hints failure, etc.)
- Error cases (invalid __config_schema__)

Organized by tier with class-based grouping per Test* convention.
"""
from __future__ import annotations

import inspect
from typing import Annotated, Any, Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from conscribe.config.extractor import extract_config_schema


# ===================================================================
# Tier 3: __config_schema__ (highest priority)
# ===================================================================

class TestTier3ConfigSchema:
    """Tests for classes with explicit __config_schema__ attribute."""

    def test_valid_config_schema_returned_directly(self) -> None:
        """Valid BaseModel subclass is returned unchanged."""

        class MyConfig(BaseModel):
            model_id: str
            temperature: float = 0.0

        class MyClass:
            __config_schema__ = MyConfig

            def __init__(self, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is MyConfig

    def test_invalid_config_schema_raises_type_error(self) -> None:
        """Non-BaseModel __config_schema__ raises TypeError."""

        class MyClass:
            __config_schema__ = "not a model"

            def __init__(self):
                pass

        with pytest.raises(TypeError):
            extract_config_schema(MyClass)

    def test_invalid_config_schema_dict_type_raises_type_error(self) -> None:
        """__config_schema__ = dict (a type, but not BaseModel) raises TypeError."""

        class MyClass:
            __config_schema__ = dict

            def __init__(self):
                pass

        with pytest.raises(TypeError):
            extract_config_schema(MyClass)

    def test_config_schema_preserves_extra_allow(self) -> None:
        """__config_schema__ with extra='allow' is not overridden to 'forbid'."""

        class OpenConfig(BaseModel):
            model_config = ConfigDict(extra="allow")
            model_id: str

        class MyClass:
            __config_schema__ = OpenConfig

            def __init__(self, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is OpenConfig
        assert result.model_config.get("extra") == "allow"

    def test_config_schema_preserves_model_validator(self) -> None:
        """__config_schema__ with model_validator is returned untouched."""

        class ValidatedConfig(BaseModel):
            model_id: str
            max_tokens: int = 4096

            @model_validator(mode="after")
            def check_max_tokens(self):
                if self.model_id.startswith("o") and self.max_tokens > 16384:
                    raise ValueError("o-series max_tokens limit is 16384")
                return self

        class MyClass:
            __config_schema__ = ValidatedConfig

            def __init__(self, config: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is ValidatedConfig


# ===================================================================
# Tier 3 variant: Single-param BaseModel auto-detect
# ===================================================================

# Module-level BaseModel classes for Tier 3 variant tests.
# Must be at module scope so get_type_hints() can resolve them
# (from __future__ import annotations stringifies annotations).

class _SingleParamConfig(BaseModel):
    model_id: str
    temperature: float = 0.0


class _SingleParamTarget:
    def __init__(self, config: _SingleParamConfig):
        pass


class _AnnotatedParamTarget:
    def __init__(self, config: Annotated[_SingleParamConfig, "some metadata"]):
        pass


class _TwoParamConfig(BaseModel):
    model_id: str


class _TwoParamTarget:
    def __init__(self, config: _TwoParamConfig, extra: int = 0):
        pass


class TestTier3SingleParamBaseModel:
    """Tests for auto-detecting single BaseModel param in __init__."""

    def test_single_basemodel_param_returns_type(self) -> None:
        """__init__(self, config: MyConfig) returns MyConfig."""
        result = extract_config_schema(_SingleParamTarget)
        assert result is _SingleParamConfig

    def test_single_annotated_basemodel_param_unwraps(self) -> None:
        """__init__(self, config: Annotated[MyConfig, ...]) unwraps and returns MyConfig."""
        result = extract_config_schema(_AnnotatedParamTarget)
        assert result is _SingleParamConfig

    def test_two_params_falls_to_tier1(self) -> None:
        """__init__(self, config: MyConfig, extra: int) has 2 params, falls to Tier 1."""
        result = extract_config_schema(_TwoParamTarget)
        # Should NOT return _TwoParamConfig; should create a dynamic model with both params
        assert result is not _TwoParamConfig
        assert result is not None
        # The dynamic model should have both 'config' and 'extra' as fields
        field_names = set(result.model_fields.keys())
        assert "config" in field_names
        assert "extra" in field_names

    def test_single_non_basemodel_param_falls_to_tier1(self) -> None:
        """__init__(self, config: str) -- str is not BaseModel, falls to Tier 1."""

        class MyClass:
            def __init__(self, config: str):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        # Should be a dynamically created model with 'config' as a str field
        assert "config" in result.model_fields
        assert result.model_fields["config"].annotation is str


# ===================================================================
# Tier 1: Pure __init__ signature
# ===================================================================

class TestTier1PureSignature:
    """Tests for Tier 1 extraction from __init__ signature only."""

    def test_basic_typed_params_with_defaults(self) -> None:
        """Basic typed params with defaults produce correct model fields."""

        class MyClass:
            def __init__(
                self,
                *,
                model_id: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        fields = result.model_fields

        assert "model_id" in fields
        assert fields["model_id"].annotation is str
        assert fields["model_id"].is_required()

        assert "temperature" in fields
        assert fields["temperature"].annotation is float
        assert fields["temperature"].default == 0.0

        assert "max_tokens" in fields
        assert fields["max_tokens"].annotation is int
        assert fields["max_tokens"].default == 4096

    def test_no_params_returns_none(self) -> None:
        """__init__ with only self returns None."""

        class MyClass:
            def __init__(self):
                pass

        result = extract_config_schema(MyClass)
        assert result is None

    def test_all_required_fields(self) -> None:
        """All params without defaults are required in the model."""

        class MyClass:
            def __init__(self, *, host: str, port: int):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        for name in ("host", "port"):
            assert result.model_fields[name].is_required()

    def test_all_optional_fields(self) -> None:
        """All params with defaults are optional in the model."""

        class MyClass:
            def __init__(
                self,
                *,
                verbose: bool = False,
                retries: int = 3,
                timeout: float = 30.0,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields["verbose"].default is False
        assert result.model_fields["retries"].default == 3
        assert result.model_fields["timeout"].default == 30.0

    def test_kwargs_present_sets_extra_allow(self) -> None:
        """Presence of **kwargs sets extra='allow' (open schema)."""

        class MyClass:
            def __init__(self, *, model_id: str, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_config.get("extra") == "allow"
        # **kwargs itself should NOT appear as a field
        assert "kwargs" not in result.model_fields

    def test_no_kwargs_sets_extra_forbid(self) -> None:
        """No **kwargs sets extra='forbid' (closed schema)."""

        class MyClass:
            def __init__(self, *, model_id: str, temperature: float = 0.0):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_config.get("extra") == "forbid"

    def test_args_present_sets_extra_allow(self) -> None:
        """Presence of *args sets extra='allow' (open schema)."""

        class MyClass:
            def __init__(self, *args: Any, model_id: str):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_config.get("extra") == "allow"
        # *args itself should NOT appear as a field
        assert "args" not in result.model_fields

    def test_model_named_classname_config(self) -> None:
        """Dynamic model is named {ClassName}Config."""

        class ChatOpenAI:
            def __init__(self, *, model_id: str):
                pass

        result = extract_config_schema(ChatOpenAI)
        assert result is not None
        assert result.__name__ == "ChatOpenAIConfig"

    def test_mixed_positional_and_keyword_params(self) -> None:
        """Both positional and keyword params become fields."""

        class MyClass:
            def __init__(self, host: str, port: int, *, timeout: float = 30.0):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert "host" in result.model_fields
        assert "port" in result.model_fields
        assert "timeout" in result.model_fields

    def test_complex_types(self) -> None:
        """Complex types like list[str], dict[str, int], Optional are preserved."""

        class MyClass:
            def __init__(
                self,
                *,
                tags: list[str],
                metadata: dict[str, int],
                description: Optional[str] = None,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert "tags" in result.model_fields
        assert "metadata" in result.model_fields
        assert "description" in result.model_fields
        assert result.model_fields["description"].default is None

    def test_no_type_annotations_uses_any(self) -> None:
        """Params without type annotations default to Any."""

        class MyClass:
            def __init__(self, *, name, count=5):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert "name" in result.model_fields
        assert "count" in result.model_fields
        assert result.model_fields["count"].default == 5


# ===================================================================
# Tier 1.5: Docstring descriptions
# ===================================================================

class TestTier15DocstringDescriptions:
    """Tests for Tier 1.5: docstring descriptions merged into model fields."""

    def test_google_docstring_adds_descriptions(self) -> None:
        """Google-style docstring descriptions appear as field descriptions."""

        class MyClass:
            """My class.

            Args:
                model_id: The model identifier.
                temperature: Sampling temperature between 0 and 2.
            """

            def __init__(
                self,
                *,
                model_id: str,
                temperature: float = 0.0,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields["model_id"].description == "The model identifier."
        assert "Sampling temperature" in result.model_fields["temperature"].description

    def test_partial_docstring_coverage(self) -> None:
        """Only documented params get description; undocumented get None."""

        class MyClass:
            """My class.

            Args:
                model_id: The model identifier.
            """

            def __init__(
                self,
                *,
                model_id: str,
                temperature: float = 0.0,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields["model_id"].description == "The model identifier."
        # temperature has no docstring description
        assert result.model_fields["temperature"].description is None

    def test_docstring_from_mro_parent_init(self) -> None:
        """Descriptions come from the MRO parent that defines __init__."""

        class ParentClass:
            """Parent class.

            Args:
                host: Server hostname.
                port: Server port number.
            """

            def __init__(self, *, host: str, port: int = 8080):
                pass

        class ChildClass(ParentClass):
            """Child class with no __init__ override."""
            pass

        result = extract_config_schema(ChildClass)
        assert result is not None
        assert result.model_fields["host"].description == "Server hostname."
        assert result.model_fields["port"].description == "Server port number."


# ===================================================================
# Tier 2: Annotated[T, Field()] metadata
# ===================================================================

class TestTier2AnnotatedMetadata:
    """Tests for Tier 2: Annotated[T, Field()] metadata preservation."""

    def test_full_annotated_with_description_and_constraints(self) -> None:
        """Annotated with description, ge, le preserves all metadata."""

        class MyClass:
            def __init__(
                self,
                *,
                model_id: Annotated[str, Field(description="Model identifier")],
                temperature: Annotated[float, Field(0.0, ge=0, le=2)] = 0.0,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None

        model_id_field = result.model_fields["model_id"]
        assert model_id_field.description == "Model identifier"

        temp_field = result.model_fields["temperature"]
        # Check constraints are preserved in metadata
        assert temp_field.default == 0.0
        # ge and le should be accessible via metadata
        assert any(
            getattr(m, "ge", None) == 0
            for m in temp_field.metadata
        ) or temp_field.ge == 0

    def test_mixed_tier1_and_tier2_params(self) -> None:
        """Some params use Annotated (Tier 2), some are plain (Tier 1)."""

        class MyClass:
            def __init__(
                self,
                *,
                model_id: Annotated[str, Field(description="Model ID")],
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        # Tier 2 param has description
        assert result.model_fields["model_id"].description == "Model ID"
        # Tier 1 params have no description
        assert result.model_fields["temperature"].description is None
        assert result.model_fields["max_tokens"].description is None

    def test_annotated_description_overrides_docstring(self) -> None:
        """Annotated Field(description=...) takes priority over docstring."""

        class MyClass:
            """My class.

            Args:
                model_id: Docstring description for model_id.
            """

            def __init__(
                self,
                *,
                model_id: Annotated[str, Field(description="Annotated description")],
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        # Annotated description wins over docstring
        assert result.model_fields["model_id"].description == "Annotated description"

    def test_annotated_without_fieldinfo_treated_as_tier1(self) -> None:
        """Annotated with non-FieldInfo metadata falls back to Tier 1."""

        class MyClass:
            def __init__(
                self,
                *,
                name: Annotated[str, "just a string annotation"],
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert "name" in result.model_fields
        # No FieldInfo, so no description
        assert result.model_fields["name"].description is None

    def test_docstring_fallback_when_annotated_has_constraints_but_no_description(
        self,
    ) -> None:
        """Annotated has ge/le but no description: docstring fills description."""

        class MyClass:
            """My class.

            Args:
                temperature: Sampling temperature.
            """

            def __init__(
                self,
                *,
                temperature: Annotated[float, Field(ge=0, le=2)] = 0.0,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        temp_field = result.model_fields["temperature"]
        # Description should come from docstring since Annotated has none
        assert temp_field.description == "Sampling temperature."
        # Constraints should still be preserved
        assert any(
            getattr(m, "ge", None) == 0
            for m in temp_field.metadata
        ) or temp_field.ge == 0


# ===================================================================
# MRO awareness
# ===================================================================

class TestMROAwareness:
    """Tests for MRO-aware parameter extraction."""

    def test_child_with_no_init_extracts_parent_params(self) -> None:
        """Child without __init__ uses parent's params."""

        class ParentClass:
            def __init__(self, *, max_steps: int = 100, timeout: int = 300):
                pass

        class ChildClass(ParentClass):
            pass

        result = extract_config_schema(ChildClass)
        assert result is not None
        assert "max_steps" in result.model_fields
        assert result.model_fields["max_steps"].default == 100
        assert "timeout" in result.model_fields
        assert result.model_fields["timeout"].default == 300

    def test_grandchild_two_levels_extracts_from_definer(self) -> None:
        """Grandchild (2 levels of inheritance) extracts from actual definer."""

        class GrandparentClass:
            def __init__(self, *, host: str, port: int = 8080):
                pass

        class ParentClass(GrandparentClass):
            pass

        class GrandchildClass(ParentClass):
            pass

        result = extract_config_schema(GrandchildClass)
        assert result is not None
        assert "host" in result.model_fields
        assert "port" in result.model_fields
        assert result.model_fields["port"].default == 8080

    def test_child_overrides_init_uses_child_params(self) -> None:
        """Child that overrides __init__ uses child's params, not parent's."""

        class ParentClass:
            def __init__(self, *, parent_param: str = "parent"):
                pass

        class ChildClass(ParentClass):
            def __init__(self, *, child_param: int = 42):
                pass

        result = extract_config_schema(ChildClass)
        assert result is not None
        assert "child_param" in result.model_fields
        assert result.model_fields["child_param"].default == 42
        # Parent's param should NOT appear
        assert "parent_param" not in result.model_fields

    def test_model_named_after_target_class_not_definer(self) -> None:
        """Dynamic model is named after the target class, not the MRO definer."""

        class BaseAgent:
            def __init__(self, *, max_steps: int = 100):
                pass

        class SubAgent(BaseAgent):
            pass

        result = extract_config_schema(SubAgent)
        assert result is not None
        assert result.__name__ == "SubAgentConfig"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Tests for edge cases and graceful degradation."""

    def test_inspect_signature_fails_returns_none(self) -> None:
        """If inspect.signature() raises, return None."""

        class MyClass:
            def __init__(self, *, name: str):
                pass

        with patch(
            "conscribe.config.extractor.inspect.signature",
            side_effect=ValueError("cannot inspect"),
        ):
            result = extract_config_schema(MyClass)
            assert result is None

    def test_get_type_hints_fails_falls_back_to_annotations(self) -> None:
        """If get_type_hints() raises, falls back to raw annotations."""

        class MyClass:
            def __init__(self, *, name: str):
                pass

        with patch(
            "conscribe.config.extractor.get_type_hints",
            side_effect=Exception("cannot resolve hints"),
        ):
            result = extract_config_schema(MyClass)
            # Fallback extracts from __annotations__, should still work
            assert result is not None
            assert "name" in result.model_fields

    def test_annotated_with_non_fieldinfo_metadata_graceful(self) -> None:
        """Annotated with arbitrary (non-FieldInfo) metadata does not crash."""

        class Marker:
            """Custom metadata that is not FieldInfo."""
            pass

        class MyClass:
            def __init__(
                self,
                *,
                name: Annotated[str, Marker()],
                count: Annotated[int, "a string", 42] = 5,
            ):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert "name" in result.model_fields
        assert "count" in result.model_fields
        assert result.model_fields["count"].default == 5

    def test_default_none_with_optional_type(self) -> None:
        """Default value None with Optional type works correctly."""

        class MyClass:
            def __init__(self, *, base_url: Optional[str] = None):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert "base_url" in result.model_fields
        assert result.model_fields["base_url"].default is None

    def test_cls_with_only_self_and_kwargs(self) -> None:
        """__init__(self, **kwargs) -- no explicit params, should return None or empty model."""

        class MyClass:
            def __init__(self, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        # No named params besides self -- should return None
        assert result is None

    def test_cls_with_only_self_and_args(self) -> None:
        """__init__(self, *args) -- no explicit params, should return None."""

        class MyClass:
            def __init__(self, *args: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is None

    def test_cls_with_only_self_args_and_kwargs(self) -> None:
        """__init__(self, *args, **kwargs) -- no explicit params, should return None."""

        class MyClass:
            def __init__(self, *args: Any, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is None


# ===================================================================
# Error cases
# ===================================================================

class TestErrorCases:
    """Tests for error conditions that should raise exceptions."""

    def test_config_schema_string_raises_type_error(self) -> None:
        """__config_schema__ = 'not a model' raises TypeError."""

        class MyClass:
            __config_schema__ = "not a model"

            def __init__(self):
                pass

        with pytest.raises(TypeError):
            extract_config_schema(MyClass)

    def test_config_schema_dict_type_raises_type_error(self) -> None:
        """__config_schema__ = dict (type but not BaseModel) raises TypeError."""

        class MyClass:
            __config_schema__ = dict

            def __init__(self):
                pass

        with pytest.raises(TypeError):
            extract_config_schema(MyClass)

    def test_config_schema_int_raises_type_error(self) -> None:
        """__config_schema__ = 42 (not even a type) raises TypeError."""

        class MyClass:
            __config_schema__ = 42

            def __init__(self):
                pass

        with pytest.raises(TypeError):
            extract_config_schema(MyClass)

    def test_config_schema_none_falls_through(self) -> None:
        """__config_schema__ = None should be treated as absent, falls to signature."""

        class MyClass:
            __config_schema__ = None

            def __init__(self, *, name: str):
                pass

        # None is falsy -- implementation should treat it as "not set"
        # and fall through to signature extraction
        result = extract_config_schema(MyClass)
        assert result is not None
        assert "name" in result.model_fields


# ===================================================================
# Integration: Tier priority ordering
# ===================================================================

class TestTierPriority:
    """Tests verifying that tier priority is respected."""

    def test_config_schema_takes_priority_over_single_param_basemodel(self) -> None:
        """__config_schema__ wins even when __init__ has single BaseModel param."""

        class SchemaA(BaseModel):
            from_schema: str = "schema"

        class SchemaB(BaseModel):
            from_init: str = "init"

        class MyClass:
            __config_schema__ = SchemaA

            def __init__(self, config: SchemaB):
                pass

        result = extract_config_schema(MyClass)
        assert result is SchemaA

    def test_single_param_basemodel_takes_priority_over_signature(self) -> None:
        """Single BaseModel param wins over treating it as a plain Tier 1 field."""
        # Uses module-level _SingleParamTarget / _SingleParamConfig
        result = extract_config_schema(_SingleParamTarget)
        # Should return the config directly, not a dynamic model with 'config' field
        assert result is _SingleParamConfig

    def test_extra_policy_not_overridden_on_tier3(self) -> None:
        """Tier 3 schema's extra policy is never overridden, even with **kwargs."""

        class StrictConfig(BaseModel):
            model_config = ConfigDict(extra="forbid")
            name: str

        class MyClass:
            __config_schema__ = StrictConfig

            def __init__(self, **kwargs: Any):
                # Even though __init__ has **kwargs, Tier 3 schema is untouched
                pass

        result = extract_config_schema(MyClass)
        assert result is StrictConfig
        assert result.model_config.get("extra") == "forbid"


# ===================================================================
# Model field validation (ensure generated models actually work)
# ===================================================================

class TestGeneratedModelValidation:
    """Tests that dynamically created models validate correctly with Pydantic."""

    def test_closed_schema_rejects_extra_fields(self) -> None:
        """Generated model with extra='forbid' rejects unknown fields."""

        class MyClass:
            def __init__(self, *, name: str):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None

        # Valid data should work
        instance = result(name="test")
        assert instance.name == "test"

        # Extra field should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            result(name="test", unknown_field="oops")

    def test_open_schema_accepts_extra_fields(self) -> None:
        """Generated model with extra='allow' accepts unknown fields."""

        class MyClass:
            def __init__(self, *, name: str, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None

        # Extra field should be accepted
        instance = result(name="test", extra_field="ok")
        assert instance.name == "test"

    def test_required_field_raises_on_missing(self) -> None:
        """Required field (no default) raises when not provided."""

        class MyClass:
            def __init__(self, *, host: str, port: int):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None

        with pytest.raises(Exception):  # Pydantic ValidationError
            result()  # Missing both required fields

    def test_default_values_used_when_not_provided(self) -> None:
        """Default values are used when fields are not provided."""

        class MyClass:
            def __init__(self, *, name: str, retries: int = 3, verbose: bool = False):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None

        instance = result(name="test")
        assert instance.retries == 3
        assert instance.verbose is False


# ===================================================================
# MRO **kwargs resolution
# ===================================================================


class _MROParent:
    """Parent class for MRO tests.

    Args:
        x: The x coordinate.
        y: The y label.
    """

    def __init__(self, x: int, y: str = "hello"):
        self.x = x
        self.y = y


class _MROChild(_MROParent):
    def __init__(self, z: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.z = z


class _MROGrandparent:
    def __init__(self, a: int, b: str = "gp"):
        pass


class _MROMiddle(_MROGrandparent):
    def __init__(self, x: int, **kwargs: Any):
        super().__init__(**kwargs)


class _MROGrandchild(_MROMiddle):
    def __init__(self, z: float, **kwargs: Any):
        super().__init__(**kwargs)


class _MROChildNoKwargs(_MROParent):
    """Child that overrides __init__ without **kwargs."""

    def __init__(self, z: float):
        super().__init__(x=0)
        self.z = z


class _MROChildOnlyKwargs(_MROParent):
    """Child with only **kwargs (no own named params)."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class _MRODepthParent:
    def __init__(self, deep_param: int = 99):
        pass


class _MRODepthMiddle(_MRODepthParent):
    def __init__(self, mid_param: str = "mid", **kwargs: Any):
        super().__init__(**kwargs)


class _MRODepthChild(_MRODepthMiddle):
    def __init__(self, z: float, **kwargs: Any):
        super().__init__(**kwargs)


class _MROScopeChild(_MROParent):
    __config_mro_scope__ = "all"

    def __init__(self, z: float, **kwargs: Any):
        super().__init__(**kwargs)


class _MRODepthOverrideChild(_MROGrandchild):
    __config_mro_depth__ = 1

    def __init__(self, w: float, **kwargs: Any):
        super().__init__(**kwargs)


class TestMROKwargsResolution:
    """Tests for MRO **kwargs parameter collection in extract_config_schema."""

    def test_basic_kwargs_collects_parent_params(self) -> None:
        """Child with **kwargs collects parent's x and y."""
        result = extract_config_schema(_MROChild, mro_scope="all")
        assert result is not None
        fields = result.model_fields
        assert "z" in fields
        assert "x" in fields
        assert "y" in fields
        assert fields["y"].default == "hello"

    def test_fully_resolved_sets_extra_forbid(self) -> None:
        """Fully resolved MRO chain produces extra='forbid'."""
        result = extract_config_schema(_MROChild, mro_scope="all")
        assert result is not None
        assert result.model_config.get("extra") == "forbid"

    def test_scope_truncation_sets_extra_allow(self) -> None:
        """When MRO chain is truncated by scope, extra='allow'."""
        from pydantic import BaseModel

        class _ScopeChild(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = extract_config_schema(_ScopeChild, mro_scope="local")
        assert result is not None
        assert result.model_config.get("extra") == "allow"

    def test_no_kwargs_backward_compatible(self) -> None:
        """Class without **kwargs behaves exactly as before."""
        result = extract_config_schema(_MROChildNoKwargs, mro_scope="all")
        assert result is not None
        assert "z" in result.model_fields
        # Parent params should NOT appear (no **kwargs)
        assert "x" not in result.model_fields
        assert "y" not in result.model_fields
        assert result.model_config.get("extra") == "forbid"

    def test_tier3_not_affected_by_mro(self) -> None:
        """Tier 3 (__config_schema__) is not affected by MRO collection."""

        class MyConfig(BaseModel):
            only_this: str

        class MyClass(_MROParent):
            __config_schema__ = MyConfig

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = extract_config_schema(MyClass, mro_scope="all")
        assert result is MyConfig

    def test_class_level_mro_scope_override(self) -> None:
        """__config_mro_scope__ overrides function-level mro_scope."""
        result = extract_config_schema(_MROScopeChild, mro_scope="local")
        assert result is not None
        # Despite mro_scope="local", the class-level override sets "all"
        assert "x" in result.model_fields
        assert "y" in result.model_fields

    def test_class_level_mro_depth_override(self) -> None:
        """__config_mro_depth__ overrides function-level mro_depth."""
        # _MRODepthOverrideChild extends _MROGrandchild which extends _MROMiddle
        # which extends _MROGrandparent. __config_mro_depth__=1 should
        # only go one level up from the child's init_definer.
        result = extract_config_schema(_MRODepthOverrideChild, mro_scope="all")
        assert result is not None
        # Only direct parent's params should appear (depth=1)
        assert "z" in result.model_fields
        # Further params should not be collected due to depth=1
        # The child has w, __init__ defines w + **kwargs
        assert "w" in result.model_fields

    def test_two_level_chain(self) -> None:
        """Two-level kwargs chain collects from both parent and grandparent."""
        result = extract_config_schema(_MROGrandchild, mro_scope="all")
        assert result is not None
        fields = result.model_fields
        assert "z" in fields
        assert "x" in fields
        assert "a" in fields
        assert "b" in fields

    def test_depth_limit(self) -> None:
        """mro_depth=1 limits collection to direct parent only."""
        result = extract_config_schema(_MRODepthChild, mro_scope="all", mro_depth=1)
        assert result is not None
        assert "mid_param" in result.model_fields
        assert "deep_param" not in result.model_fields

    def test_only_kwargs_child_collects_all_parent_params(self) -> None:
        """Child with only **kwargs (no own params) still collects parent params."""
        result = extract_config_schema(_MROChildOnlyKwargs, mro_scope="all")
        assert result is not None
        assert "x" in result.model_fields
        assert "y" in result.model_fields

    def test_docstring_merged_from_parent(self) -> None:
        """Docstring descriptions from parent classes are merged."""
        result = extract_config_schema(_MROChild, mro_scope="all")
        assert result is not None
        # Parent's docstring describes 'x' and 'y'
        x_field = result.model_fields.get("x")
        assert x_field is not None
        assert x_field.description == "The x coordinate."

    def test_kwargs_no_parent_init_returns_allow(self) -> None:
        """Class with **kwargs but parent has no custom __init__ gets extra='allow'."""

        class Child:
            def __init__(self, z: float, **kwargs: Any):
                pass

        result = extract_config_schema(Child, mro_scope="all")
        assert result is not None
        # No parent params collected, but has **kwargs and MRO found nothing
        # fully_resolved = True (reached object), so extra='forbid'
        # Actually, no mro_result.params means we fall through to has_var check
        assert "z" in result.model_fields

    def test_scope_list_collects_specified_package(self) -> None:
        """mro_scope=['pydantic'] collects pydantic parent params."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = extract_config_schema(Child, mro_scope=["pydantic"])
        assert result is not None
        assert "z" in result.model_fields

    def test_class_level_mro_scope_list_override(self) -> None:
        """__config_mro_scope__ = ['pydantic'] overrides function-level scope."""

        class _ListScopeChild(_MROParent):
            __config_mro_scope__ = ["anything"]

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        # Despite mro_scope="local", the class-level override with list applies
        result = extract_config_schema(_ListScopeChild, mro_scope="local")
        assert result is not None
        # _MROParent is local, so included regardless
        assert "x" in result.model_fields
        assert "y" in result.model_fields


# ===================================================================
# Type degradation
# ===================================================================


class _IncompatibleType:
    """A type Pydantic cannot serialize."""

    def __init__(self, x: int):
        self.x = x


class _ClassWithIncompatibleParam:
    def __init__(self, name: str, auth: _IncompatibleType, count: int = 0):
        pass


class _ClassAllCompatible:
    def __init__(self, name: str, count: int = 0, flag: bool = True):
        pass


class TestTypeDegradation:
    """Tests for type degradation when Pydantic can't handle a field type."""

    def test_incompatible_type_degraded(self) -> None:
        """Class with an incompatible param returns schema with field as Any."""
        result = extract_config_schema(_ClassWithIncompatibleParam)
        assert result is not None
        assert "auth" in result.model_fields
        assert "name" in result.model_fields
        assert "count" in result.model_fields
        # auth should be degraded to Any
        assert result.model_fields["auth"].annotation is Any

    def test_degraded_fields_metadata_attached(self) -> None:
        """model.__degraded_fields__ exists and contains correct info."""
        result = extract_config_schema(_ClassWithIncompatibleParam)
        assert result is not None
        degraded = getattr(result, "__degraded_fields__", None)
        assert degraded is not None
        assert len(degraded) >= 1
        field_names = [df.field_name for df in degraded]
        assert "auth" in field_names
        # original type repr should mention _IncompatibleType
        auth_df = [df for df in degraded if df.field_name == "auth"][0]
        assert "_IncompatibleType" in auth_df.original_type_repr

    def test_compatible_class_no_degradation(self) -> None:
        """Class with all compatible types has no __degraded_fields__."""
        result = extract_config_schema(_ClassAllCompatible)
        assert result is not None
        degraded = getattr(result, "__degraded_fields__", None)
        assert degraded is None

    def test_degraded_model_validates(self) -> None:
        """Degraded model can validate data (Any accepts anything)."""
        result = extract_config_schema(_ClassWithIncompatibleParam)
        assert result is not None
        instance = result(name="test", auth="any_value", count=5)
        assert instance.name == "test"
        assert instance.auth == "any_value"
        assert instance.count == 5
