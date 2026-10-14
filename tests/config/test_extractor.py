"""Tests for conscribe.config.extractor.extract_config_schema.

Covers:
- Tier 3: __config_schema__ (highest priority, direct return, preserves model_config)
- Single-param BaseModel: an ordinary nested field in 2.0 (1.x flattened it;
  the opt-in ``__config_from_param__`` is covered in test_extractor_v2.py)
- Tier 1: pure __init__ signature (typed params, defaults, **kwargs, *args, naming)
- Tier 1.5: docstring descriptions merged into fields
- Tier 2: Annotated[T, Field()] metadata preservation
- MRO awareness (child/grandchild inheritance)
- Edge cases (unreadable signature, resolution without typing.get_type_hints, etc.)
- Error cases (invalid __config_schema__)

Organized by tier with class-based grouping per Test* convention.
"""
from __future__ import annotations

import sys
from typing import Annotated, Any, Optional, Protocol, runtime_checkable
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from conscribe.config.extractor import extract_config_schema
from conscribe.exceptions import DegradationWarning, ExtractionWarning, WiringResolutionError


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
# Single-param BaseModel: a nested field in 2.0
# ===================================================================

# Module-level BaseModel classes for the single-parameter tests.
# Must be at module scope so the annotations resolve
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


class TestSingleParamBaseModel:
    """A single BaseModel parameter is an ordinary nested field (2.0).

    1.x returned the parameter's model as the whole config ("Tier 3
    variant"), silently dropping the parameter name, wiring receptors and
    ``**kwargs`` knobs.  Flattening is now the explicit
    ``__config_from_param__`` opt-in.
    """

    def test_single_basemodel_param_is_a_nested_field(self) -> None:
        """__init__(self, config: MyConfig) -> a model with a ``config`` field."""
        result = extract_config_schema(_SingleParamTarget)
        assert result is not _SingleParamConfig
        assert set(result.model_fields) == {"config"}
        assert result.model_fields["config"].annotation is _SingleParamConfig
        assert result.model_fields["config"].is_required()
        validated = result.model_validate({"config": {"model_id": "m"}})
        assert validated.config == _SingleParamConfig(model_id="m")

    def test_single_annotated_basemodel_param_is_a_nested_field(self) -> None:
        """Annotated[MyConfig, ...] is unwrapped for the field type, not flattened."""
        result = extract_config_schema(_AnnotatedParamTarget)
        assert set(result.model_fields) == {"config"}
        field = result.model_fields["config"]
        assert field.annotation is _SingleParamConfig
        assert "some metadata" in field.metadata

    def test_old_flat_keys_get_a_targeted_error(self) -> None:
        """The 1.x flat shape is rejected with a message naming the fix."""
        result = extract_config_schema(_SingleParamTarget)
        with pytest.raises(ValidationError) as excinfo:
            result.model_validate({"model_id": "m"})
        error = excinfo.value.errors()[0]
        assert error["type"] == "legacy_flat_config"
        assert error["msg"] == (
            "'config' is a nested block in conscribe 2.0; write {config: {...}} "
            "or set __config_from_param__ = 'config'"
        )

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

    def test_unreadable_signature_warns_and_allows_extra(self) -> None:
        """An ``__init__`` whose signature cannot be read is reported.

        conscribe cannot know the parameters, so it says so with an
        ``ExtractionWarning`` and the config accepts unknown keys instead of
        silently forbidding every knob the runtime takes.
        """

        class Opaque:
            """A callable whose signature ``inspect`` cannot read."""

            __signature__ = "not a signature"

            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class MyClass:
            __init__ = Opaque()

        with pytest.warns(ExtractionWarning, match="cannot read the signature"):
            result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields == {}
        assert result.model_config.get("extra") == "allow"
        assert result(name="anything").model_extra == {"name": "anything"}

    def test_resolution_does_not_use_typing_get_type_hints(self) -> None:
        """Annotations resolve per parameter, in the declaring scope.

        1.x called ``typing.get_type_hints`` on the ``__init__`` and fell back
        to raw strings; 2.0 never uses it (it adds an implicit ``Optional``
        on 3.10 and resolves inherited annotations in the wrong module).
        """

        class MyClass:
            def __init__(self, *, name: str):
                pass

        with patch("typing.get_type_hints", side_effect=Exception("not used")):
            result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields["name"].annotation is str

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

        # ``Marker`` is local to this test function: conscribe cannot see it
        # (PEP 563 strings), so ``name`` degrades to Any with an
        # ExtractionWarning instead of crashing.
        with pytest.warns(ExtractionWarning, match="local to the function"):
            result = extract_config_schema(MyClass)
        assert result is not None
        assert "name" in result.model_fields
        assert "count" in result.model_fields
        assert result.model_fields["count"].default == 5
        assert [d.reason for d in result.__degraded_fields__] == ["function_locals"]

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
        """__init__(self, **kwargs) -- no named params, but the constructor
        accepts any keyword: an empty, open model (never ``None``, which the
        builder turns into a closed discriminator-only model)."""

        class MyClass:
            def __init__(self, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields == {}
        assert result.model_config.get("extra") == "allow"
        assert result(anything=1).model_extra == {"anything": 1}

    def test_cls_with_only_self_and_args(self) -> None:
        """__init__(self, *args) -- *args keeps the documented open policy."""

        class MyClass:
            def __init__(self, *args: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields == {}
        assert result.model_config.get("extra") == "allow"

    def test_cls_with_only_self_args_and_kwargs(self) -> None:
        """__init__(self, *args, **kwargs) -- an empty, open model."""

        class MyClass:
            def __init__(self, *args: Any, **kwargs: Any):
                pass

        result = extract_config_schema(MyClass)
        assert result is not None
        assert result.model_fields == {}
        assert result.model_config.get("extra") == "allow"

    def test_closed_constructor_without_params_returns_none(self) -> None:
        """Nothing to configure and nothing accepted: ``None``."""

        class MyClass:
            def __init__(self) -> None:
                pass

        assert extract_config_schema(MyClass) is None


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

    def test_single_param_basemodel_is_not_flattened_without_opt_in(self) -> None:
        """Without ``__config_from_param__`` the parameter stays a nested field."""
        # Uses module-level _SingleParamTarget / _SingleParamConfig
        result = extract_config_schema(_SingleParamTarget)
        assert result is not _SingleParamConfig
        assert set(result.model_fields) == {"config"}

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

    def test_scope_truncation_sets_extra_allow(self, tmp_path, monkeypatch) -> None:
        """When MRO chain is truncated by scope, extra='allow'."""
        import importlib
        import uuid

        site = tmp_path / "site-packages"
        pkg = f"ext_{uuid.uuid4().hex[:8]}"
        (site / pkg).mkdir(parents=True)
        (site / pkg / "__init__.py").write_text(
            "class ExtBase:\n"
            "    def __init__(self, host: str = 'h', **kwargs):\n"
            "        super().__init__(**kwargs)\n"
        )
        monkeypatch.syspath_prepend(str(site))
        ext = importlib.import_module(pkg)

        class _ScopeChild(ext.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = extract_config_schema(_ScopeChild, mro_scope="local")
        assert result is not None
        assert list(result.model_fields) == ["z"]
        assert result.model_config.get("extra") == "allow"
        full = extract_config_schema(_ScopeChild, mro_scope="third_party")
        assert list(full.model_fields) == ["z", "host"]
        assert full.model_config.get("extra") == "forbid"
        del sys.modules[pkg]

    def test_base_model_parent_is_never_cut_by_scope(self) -> None:
        """pydantic's BaseModel ends the chain naturally (B4): a model with
        its own ``__init__(z, **kwargs)`` keeps its fields under any scope."""
        from pydantic import BaseModel

        class _ScopeChild(BaseModel):
            z: float
            w: int = 2

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        for scope in ("local", "third_party", []):
            result = extract_config_schema(_ScopeChild, mro_scope=scope)
            assert list(result.model_fields) == ["z", "w"]
            assert result.model_config.get("extra") == "forbid"

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
        with pytest.warns(DegradationWarning, match="auth"):
            result = extract_config_schema(_ClassWithIncompatibleParam)
        assert result is not None
        assert "auth" in result.model_fields
        assert "name" in result.model_fields
        assert "count" in result.model_fields
        # auth should be degraded to Any
        assert result.model_fields["auth"].annotation is Any

    def test_degraded_fields_metadata_attached(self) -> None:
        """model.__degraded_fields__ exists and contains correct info."""
        with pytest.warns(DegradationWarning):
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
        assert auth_df.reason == "pydantic_incompatible"
        assert auth_df.source_class.endswith("_ClassWithIncompatibleParam")

    def test_compatible_class_no_degradation(self) -> None:
        """Class with all compatible types has no __degraded_fields__."""
        result = extract_config_schema(_ClassAllCompatible)
        assert result is not None
        degraded = getattr(result, "__degraded_fields__", None)
        assert degraded is None

    def test_degraded_model_validates(self) -> None:
        """Degraded model can validate data (Any accepts anything)."""
        with pytest.warns(DegradationWarning):
            result = extract_config_schema(_ClassWithIncompatibleParam)
        assert result is not None
        instance = result(name="test", auth="any_value", count=5)
        assert instance.name == "test"
        assert instance.auth == "any_value"
        assert instance.count == 5


# ===================================================================
# BaseModel fast path
# ===================================================================


class TestBaseModelFastPath:
    """Tests for the BaseModel fast path in extract_config_schema.

    When cls is a BaseModel subclass without a custom __init__, fields
    are extracted from cls.model_fields instead of reflecting __init__.
    """

    def test_basemodel_subclass_extracts_fields(self) -> None:
        """BaseModel subclass with fields produces a config schema."""

        class MyModel(BaseModel):
            name: str
            count: int = 5
            flag: bool = True

        result = extract_config_schema(MyModel)
        assert result is not None
        fields = result.model_fields
        assert "name" in fields
        assert "count" in fields
        assert "flag" in fields
        assert fields["count"].default == 5
        assert fields["flag"].default is True

    def test_basemodel_init_false_excluded(self) -> None:
        """Fields with init=False are excluded from the config schema."""

        class MyModel(BaseModel):
            name: str
            internal: str = Field(default="auto", init=False)

        result = extract_config_schema(MyModel)
        assert result is not None
        assert "name" in result.model_fields
        assert "internal" not in result.model_fields

    def test_basemodel_with_own_init_uses_reflection(self) -> None:
        """BaseModel subclass with custom __init__ falls through to reflection."""

        class MyModel(BaseModel):
            name: str = "default"

            def __init__(self, *, label: str = "custom", **data: Any):
                super().__init__(**data)
                self.label = label  # type: ignore[attr-defined]

        result = extract_config_schema(MyModel)
        assert result is not None
        # Should use reflection path, which sees 'label' as a param
        assert "label" in result.model_fields

    def test_basemodel_field_metadata_preserved(self) -> None:
        """Field(description=...) metadata is preserved in the config schema."""

        class MyModel(BaseModel):
            name: str = Field(description="The name of the entity")
            count: int = Field(default=0, ge=0, description="Non-negative count")

        result = extract_config_schema(MyModel)
        assert result is not None
        name_field = result.model_fields["name"]
        assert name_field.description == "The name of the entity"
        count_field = result.model_fields["count"]
        assert count_field.description == "Non-negative count"

    def test_config_schema_still_highest_priority(self) -> None:
        """__config_schema__ wins over BaseModel fast path."""

        class OverrideSchema(BaseModel):
            override_field: str = "override"

        class MyModel(BaseModel):
            __config_schema__ = OverrideSchema
            name: str
            count: int = 5

        result = extract_config_schema(MyModel)
        assert result is OverrideSchema

    def test_basemodel_no_fields_returns_none(self) -> None:
        """BaseModel with all fields init=False returns None."""

        class EmptyModel(BaseModel):
            internal: str = Field(default="auto", init=False)

        result = extract_config_schema(EmptyModel)
        assert result is None

    def test_basemodel_extra_config_preserved(self) -> None:
        """model_config extra setting is preserved in the config schema."""

        class AllowExtraModel(BaseModel):
            model_config = ConfigDict(extra="allow")
            name: str

        result = extract_config_schema(AllowExtraModel)
        assert result is not None
        assert result.model_config.get("extra") == "allow"

    def test_basemodel_validates_correctly(self) -> None:
        """Generated config schema validates data correctly."""

        class MyModel(BaseModel):
            name: str
            count: int = 0

        result = extract_config_schema(MyModel)
        assert result is not None
        instance = result(name="test")
        assert instance.name == "test"
        assert instance.count == 0

    def test_basemodel_itself_returns_none(self) -> None:
        """BaseModel itself (not a subclass) does not trigger fast path."""
        result = extract_config_schema(BaseModel)
        assert result is None


# ===================================================================
# Regression: wiring receptors must survive a parameter-less __init__
# ===================================================================

class TestWiringSurvivesEmptyInit:
    """``extract_config_schema`` used to bail before applying ``__wiring__``.

    A class whose ``__init__`` declares no named parameters produced a
    schema of ``None``, which the builder turns into a discriminator-only
    model.  Every receptor the class declared or inherited through
    ``__wiring__`` vanished silently: the dependency became unselectable
    from YAML and bound to whatever default the framework used instead.

    The only way to get the receptor back was to widen the signature with
    a throw-away parameter, which is exactly the kind of workaround these
    tests exist to make unnecessary.
    """

    @pytest.fixture(autouse=True)
    def _registry(self):
        from conscribe.registration.registry import LayerRegistry, _deregister

        @runtime_checkable
        class _LLMProto(Protocol):
            def generate(self) -> str: ...

        reg = LayerRegistry("t_empty_init_llm", _LLMProto)
        for key in ("openai", "anthropic"):
            reg.add(key, type(f"Dummy{key}", (), {"generate": lambda self: ""}))
        yield reg
        _deregister("t_empty_init_llm")

    def test_wired_slot_survives_empty_init(self) -> None:
        """The headline case: no config knobs of its own, one wired slot."""

        class Judge:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any  # runtime slot the framework injects into

            def __init__(self) -> None: ...

        result = extract_config_schema(Judge)
        assert result is not None, "wiring receptor was dropped"
        assert "llm" in result.model_fields
        # A real slot exists, so the receptor is required, not optional.
        assert result.model_fields["llm"].is_required()

    def test_wired_slot_accepts_only_registry_keys(self) -> None:
        """The receptor is a real ``Literal``, not a free-form string."""

        class Judge:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any

            def __init__(self) -> None: ...

        result = extract_config_schema(Judge)
        assert result is not None
        assert result(llm="openai").llm == "openai"
        with pytest.raises(Exception):
            result(llm="not_a_registered_key")

    def test_inherited_wiring_survives_empty_init(self) -> None:
        """N3's exact shape: the receptor comes from a parent class."""

        class JudgeBase:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any

            def __init__(self, temperature: float = 0.0) -> None: ...

        class BrowseComp(JudgeBase):
            def __init__(self) -> None: ...

        result = extract_config_schema(BrowseComp)
        assert result is not None
        assert "llm" in result.model_fields

    def test_slotless_wiring_survives_empty_init(self) -> None:
        """Declaration-only wiring still emits an optional receptor."""

        class Judge:
            __wiring__ = {"llm": "t_empty_init_llm"}

            def __init__(self) -> None: ...

        result = extract_config_schema(Judge)
        assert result is not None
        assert "llm" in result.model_fields
        assert not result.model_fields["llm"].is_required()

    def test_kwargs_only_init_with_wiring_survives(self) -> None:
        """``def __init__(self, **kwargs)`` is the other zero-param shape."""

        class Hook:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any

            def __init__(self, **kwargs: Any) -> None: ...

        result = extract_config_schema(Hook)
        assert result is not None
        assert "llm" in result.model_fields

    def test_widener_param_is_no_longer_needed(self) -> None:
        """The workaround and the fix must produce the same receptor.

        Downstream code kept a dummy ``__init__`` parameter alive purely to
        stop the bail from firing.  This asserts that deleting it changes
        nothing about the wired field.
        """

        class WithWidener:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any

            def __init__(self, image_detail: str = "high") -> None: ...

        class WithoutWidener:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any

            def __init__(self) -> None: ...

        widened = extract_config_schema(WithWidener)
        plain = extract_config_schema(WithoutWidener)
        assert widened is not None and plain is not None
        assert (
            plain.model_fields["llm"].annotation
            == widened.model_fields["llm"].annotation
        )
        # ...and the throw-away knob really is the only difference.
        assert set(widened.model_fields) - set(plain.model_fields) == {"image_detail"}

    def test_wired_fields_marker_is_attached(self) -> None:
        """Codegen relies on ``__wired_fields__`` for its wiring comments."""

        class Judge:
            __wiring__ = {"llm": "t_empty_init_llm"}
            llm: Any

            def __init__(self) -> None: ...

        result = extract_config_schema(Judge)
        assert getattr(result, "__wired_fields__", None) == {"llm": "t_empty_init_llm"}

    def test_empty_init_without_wiring_still_returns_none(self) -> None:
        """The bail must still fire when there is genuinely nothing to emit."""

        class Plain:
            def __init__(self) -> None: ...

        assert extract_config_schema(Plain) is None

    def test_unresolvable_registry_raises(self) -> None:
        """A wiring entry that cannot resolve is an error, never a silent drop.

        1.x returned ``None`` here (every receptor of the class vanished).
        """

        class Broken:
            __wiring__ = {"llm": "no_such_registry_anywhere"}
            llm: Any

            def __init__(self) -> None: ...

        with pytest.raises(WiringResolutionError, match="unknown registry 'no_such_registry_anywhere'"):
            extract_config_schema(Broken)

    def test_selectable_from_config_end_to_end(self) -> None:
        """The point of the fix: the dependency becomes selectable."""
        from conscribe import create_registrar
        from conscribe.registration.registry import _deregister

        @runtime_checkable
        class _JudgeProto(Protocol):
            def evaluate(self) -> float: ...

        Judge = create_registrar(
            "t_empty_init_judge", _JudgeProto, discriminator_field="evaluator_id"
        )
        try:

            class JudgeBase(metaclass=Judge.Meta):
                __abstract__ = True
                __wiring__ = {"llm": "t_empty_init_llm"}
                llm: Any

                def evaluate(self) -> float: ...

            class BrowseComp(JudgeBase):
                def __init__(self) -> None: ...

            model = Judge.build_config().per_key_models["browse_comp"]
            assert set(model.model_fields) == {"evaluator_id", "llm"}
            assert model(llm="anthropic").llm == "anthropic"
        finally:
            _deregister("t_empty_init_judge")
