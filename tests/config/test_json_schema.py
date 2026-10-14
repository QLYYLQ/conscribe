"""Tests for conscribe.config.json_schema — TDD RED phase.

Tests ``generate_layer_json_schema()``:
- Basic structure: returns dict, has standard JSON Schema fields, has x-discriminator
- Schema content: additionalProperties, descriptions, constraints, required, defaults
- Model schema: single model, multi-model discriminated union, discriminator const/enum
- Serialization: json.dumps succeeds
- Type representation: basic types (str, int, float, bool), complex types (list, dict, Optional)

All registrars and implementation classes are defined at MODULE LEVEL
so that ``get_type_hints()`` can resolve forward references under
``from __future__ import annotations``.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional, Protocol, runtime_checkable

import pytest
from pydantic import BaseModel, Field

from conscribe import create_registrar
from conscribe.config.builder import LayerConfigResult, build_layer_config
from conscribe.config.degradation import DegradedField
from conscribe.config.json_schema import generate_layer_json_schema


# ===================================================================
# Protocols for test registrars
# ===================================================================

@runtime_checkable
class _LLMProto(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class _AgentProto(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


# ===================================================================
# Registrars (module-level for type hint resolution)
# ===================================================================

_single_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_multi_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["Multi"],
)

_closed_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_description_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_constraint_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_required_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_default_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_basic_types_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_complex_types_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_open_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_disc_const_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["DiscConst"],
)


# ===================================================================
# Implementation classes: single model (_single_reg)
# ===================================================================

class _SingleBase(metaclass=_single_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SingleOpenAI(_SingleBase):
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes: multiple models (_multi_reg)
# ===================================================================

class _MultiBase(metaclass=_multi_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OpenAIMulti(_MultiBase):
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class AnthropicMulti(_MultiBase):
    __registry_key__ = "anthropic"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes: closed schema (_closed_reg) for additionalProperties
# ===================================================================

class _ClosedBase(metaclass=_closed_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ClosedProvider(_ClosedBase):
    __registry_key__ = "closed"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes: open schema (_open_reg) — no additionalProperties restriction
# ===================================================================

class _OpenBase(metaclass=_open_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OpenProvider(_OpenBase):
    __registry_key__ = "flexible"

    def __init__(self, *, model_id: str, **kwargs: Any):
        self.model_id = model_id


# ===================================================================
# Implementation classes: description fields (_description_reg)
# ===================================================================

class _DescBase(metaclass=_description_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DescProvider(_DescBase):
    __registry_key__ = "desc"

    def __init__(
        self,
        *,
        model_id: Annotated[str, Field(description="The model identifier")],
        temperature: Annotated[float, Field(description="Sampling temperature")] = 0.0,
    ):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes: constraint fields (_constraint_reg)
# ===================================================================

class _ConstraintBase(metaclass=_constraint_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ConstraintProvider(_ConstraintBase):
    __registry_key__ = "constrained"

    def __init__(
        self,
        *,
        model_id: str,
        temperature: Annotated[float, Field(ge=0, le=2)] = 0.0,
        max_tokens: Annotated[int, Field(gt=0)] = 4096,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes: required fields (_required_reg)
# ===================================================================

class _RequiredBase(metaclass=_required_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class RequiredFieldsProvider(_RequiredBase):
    __registry_key__ = "required"

    def __init__(self, *, model_id: str, api_key: str, temperature: float = 0.0):
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature


# ===================================================================
# Implementation classes: default values (_default_reg)
# ===================================================================

class _DefaultBase(metaclass=_default_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DefaultsProvider(_DefaultBase):
    __registry_key__ = "defaults"

    def __init__(
        self,
        *,
        model_id: str,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose


# ===================================================================
# Implementation classes: basic types (_basic_types_reg)
# ===================================================================

class _BasicTypesBase(metaclass=_basic_types_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class BasicTypesProvider(_BasicTypesBase):
    __registry_key__ = "basic"

    def __init__(
        self,
        *,
        name: str,
        count: int,
        score: float = 0.0,
        enabled: bool = True,
    ):
        self.name = name
        self.count = count
        self.score = score
        self.enabled = enabled


# ===================================================================
# Implementation classes: complex types (_complex_types_reg)
# ===================================================================

class _ComplexTypesBase(metaclass=_complex_types_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ComplexTypesProvider(_ComplexTypesBase):
    __registry_key__ = "complex"

    def __init__(
        self,
        *,
        tags: list[str],
        metadata: dict[str, int],
        description: Optional[str] = None,
    ):
        self.tags = tags
        self.metadata = metadata
        self.description = description


# ===================================================================
# Implementation classes: discriminator const check (_disc_const_reg)
# ===================================================================

class _DiscConstBase(metaclass=_disc_const_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class AlphaDiscConst(_DiscConstBase):
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


class BetaDiscConst(_DiscConstBase):
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Helpers
# ===================================================================

def _find_subschemas(schema: dict) -> list[dict]:
    """Extract all variant subschemas from a JSON Schema dict.

    Looks in ``anyOf``, ``oneOf``, or resolves ``$ref`` from ``$defs``.
    Returns a list of resolved subschema dicts.
    """
    defs = schema.get("$defs", {})
    variants: list[dict] = []

    for key in ("anyOf", "oneOf"):
        if key in schema:
            for item in schema[key]:
                if "$ref" in item:
                    # Resolve $ref like "#/$defs/ModelName"
                    ref_name = item["$ref"].rsplit("/", 1)[-1]
                    if ref_name in defs:
                        variants.append(defs[ref_name])
                else:
                    variants.append(item)
            return variants

    # Single model — no anyOf/oneOf, schema itself is the variant
    return [schema]


def _find_property_in_schema(schema: dict, prop_name: str) -> dict | None:
    """Find a property dict in a schema (handles top-level or $defs)."""
    # Check top-level properties
    if "properties" in schema and prop_name in schema["properties"]:
        return schema["properties"][prop_name]

    # Check $defs
    for _def_name, def_schema in schema.get("$defs", {}).items():
        if "properties" in def_schema and prop_name in def_schema["properties"]:
            return def_schema["properties"][prop_name]

    return None


# ===================================================================
# Basic structure tests
# ===================================================================

class TestBasicStructure:
    """Tests for the basic structure of the generated JSON Schema."""

    def test_returns_dict(self) -> None:
        """generate_layer_json_schema returns a dict, not a string."""
        result = build_layer_config(_single_reg)
        schema = generate_layer_json_schema(result)

        assert isinstance(schema, dict)

    def test_has_standard_json_schema_fields(self) -> None:
        """Schema has standard JSON Schema fields like $defs or properties."""
        result = build_layer_config(_multi_reg)
        schema = generate_layer_json_schema(result)

        # Multi-model schema should have $defs for the variant models
        # or at least have anyOf/oneOf at the top level
        has_defs = "$defs" in schema
        has_anyof = "anyOf" in schema
        has_oneof = "oneOf" in schema
        has_properties = "properties" in schema

        assert has_defs or has_anyof or has_oneof or has_properties, (
            f"Schema lacks standard JSON Schema structure fields. "
            f"Keys found: {list(schema.keys())}"
        )

    def test_has_x_discriminator_extension_field(self) -> None:
        """Schema has 'x-discriminator' extension field with the discriminator field name."""
        result = build_layer_config(_single_reg)
        schema = generate_layer_json_schema(result)

        assert "x-discriminator" in schema
        assert schema["x-discriminator"] == "provider"

    def test_x_discriminator_matches_registrar_discriminator_field(self) -> None:
        """x-discriminator value matches the registrar's discriminator_field."""
        result = build_layer_config(_multi_reg)
        schema = generate_layer_json_schema(result)

        assert schema["x-discriminator"] == result.discriminator_field


# ===================================================================
# Schema content tests
# ===================================================================

class TestSchemaContent:
    """Tests for schema content: additionalProperties, descriptions, constraints, etc."""

    def test_additional_properties_false_for_closed_schema(self) -> None:
        """Closed schema (extra='forbid') has additionalProperties: false."""
        result = build_layer_config(_closed_reg)
        schema = generate_layer_json_schema(result)

        subschemas = _find_subschemas(schema)
        assert len(subschemas) >= 1

        # At least one subschema should have additionalProperties: false
        found = any(
            sub.get("additionalProperties") is False
            for sub in subschemas
        )
        assert found, (
            "Expected at least one subschema with 'additionalProperties: false' "
            f"for closed schema. Subschemas: {subschemas}"
        )

    def test_open_schema_no_additional_properties_false(self) -> None:
        """Open schema (extra='allow') should NOT have additionalProperties: false."""
        result = build_layer_config(_open_reg)
        schema = generate_layer_json_schema(result)

        subschemas = _find_subschemas(schema)
        assert len(subschemas) >= 1

        # None of the subschemas should restrict additionalProperties
        for sub in subschemas:
            assert sub.get("additionalProperties") is not False, (
                f"Open schema should not have additionalProperties: false. "
                f"Subschema: {sub}"
            )

    def test_field_descriptions_mapped(self) -> None:
        """Field descriptions from Annotated[T, Field(description=...)] appear in schema."""
        result = build_layer_config(_description_reg)
        schema = generate_layer_json_schema(result)

        model_id_prop = _find_property_in_schema(schema, "model_id")
        assert model_id_prop is not None, "model_id property not found in schema"
        assert model_id_prop.get("description") == "The model identifier"

        temp_prop = _find_property_in_schema(schema, "temperature")
        assert temp_prop is not None, "temperature property not found in schema"
        assert temp_prop.get("description") == "Sampling temperature"

    def test_field_constraints_mapped(self) -> None:
        """Field constraints (ge, le, gt) are mapped into the schema."""
        result = build_layer_config(_constraint_reg)
        schema = generate_layer_json_schema(result)

        temp_prop = _find_property_in_schema(schema, "temperature")
        assert temp_prop is not None, "temperature property not found in schema"

        # Pydantic maps ge -> minimum (or exclusiveMinimum for gt)
        # and le -> maximum (or exclusiveMaximum for lt)
        assert "minimum" in temp_prop or "exclusiveMinimum" in temp_prop, (
            f"Expected minimum or exclusiveMinimum for ge=0. Got: {temp_prop}"
        )
        assert "maximum" in temp_prop or "exclusiveMaximum" in temp_prop, (
            f"Expected maximum or exclusiveMaximum for le=2. Got: {temp_prop}"
        )

        max_tokens_prop = _find_property_in_schema(schema, "max_tokens")
        assert max_tokens_prop is not None, "max_tokens property not found in schema"
        assert "exclusiveMinimum" in max_tokens_prop or "minimum" in max_tokens_prop, (
            f"Expected exclusiveMinimum for gt=0. Got: {max_tokens_prop}"
        )

    def test_required_fields_in_required_array(self) -> None:
        """Required fields (no default) are listed in the 'required' array."""
        result = build_layer_config(_required_reg)
        schema = generate_layer_json_schema(result)

        subschemas = _find_subschemas(schema)
        assert len(subschemas) >= 1

        # Find the subschema that has model_id, api_key, temperature
        target_sub = None
        for sub in subschemas:
            if "properties" in sub and "model_id" in sub["properties"]:
                target_sub = sub
                break

        assert target_sub is not None, "Could not find subschema with model_id"
        required = target_sub.get("required", [])

        # model_id and api_key are required (no default), temperature has default
        assert "model_id" in required, f"model_id should be required. Got: {required}"
        assert "api_key" in required, f"api_key should be required. Got: {required}"

    def test_default_values_in_schema(self) -> None:
        """Default values appear in the schema."""
        result = build_layer_config(_default_reg)
        schema = generate_layer_json_schema(result)

        temp_prop = _find_property_in_schema(schema, "temperature")
        assert temp_prop is not None, "temperature property not found in schema"
        assert "default" in temp_prop, (
            f"Expected 'default' key for temperature. Got: {temp_prop}"
        )
        assert temp_prop["default"] == 0.5

        max_tokens_prop = _find_property_in_schema(schema, "max_tokens")
        assert max_tokens_prop is not None, "max_tokens property not found in schema"
        assert max_tokens_prop.get("default") == 4096

        verbose_prop = _find_property_in_schema(schema, "verbose")
        assert verbose_prop is not None, "verbose property not found in schema"
        assert verbose_prop.get("default") is False


# ===================================================================
# Model schema tests
# ===================================================================

class TestModelSchema:
    """Tests for single model, multi-model union, and discriminator values."""

    def test_single_model_valid_schema(self) -> None:
        """Single model produces a valid JSON Schema dict."""
        result = build_layer_config(_single_reg)
        schema = generate_layer_json_schema(result)

        assert isinstance(schema, dict)
        # Single model should have properties at top level or in $defs
        has_properties = "properties" in schema
        has_defs = "$defs" in schema
        assert has_properties or has_defs, (
            f"Single model schema should have properties or $defs. Keys: {list(schema.keys())}"
        )

    def test_multiple_models_discriminated_union(self) -> None:
        """Multiple models produce a schema with anyOf or oneOf (discriminated union)."""
        result = build_layer_config(_multi_reg)
        schema = generate_layer_json_schema(result)

        has_anyof = "anyOf" in schema
        has_oneof = "oneOf" in schema
        assert has_anyof or has_oneof, (
            f"Multi-model schema should have anyOf or oneOf. Keys: {list(schema.keys())}"
        )

    def test_discriminator_field_has_const_or_enum_per_variant(self) -> None:
        """Each variant's discriminator field has const or enum with the key value."""
        result = build_layer_config(_disc_const_reg)
        schema = generate_layer_json_schema(result)

        subschemas = _find_subschemas(schema)
        assert len(subschemas) == 2, (
            f"Expected 2 variant subschemas, got {len(subschemas)}"
        )

        expected_keys = {"alpha", "beta"}
        found_keys: set[str] = set()

        for sub in subschemas:
            props = sub.get("properties", {})
            disc_prop = props.get("provider", {})

            # Pydantic may use "const" or "enum" for Literal types
            if "const" in disc_prop:
                found_keys.add(disc_prop["const"])
            elif "enum" in disc_prop:
                for val in disc_prop["enum"]:
                    found_keys.add(val)

        assert expected_keys.issubset(found_keys), (
            f"Expected discriminator values {expected_keys} in schema. "
            f"Found: {found_keys}"
        )


# ===================================================================
# Serialization test
# ===================================================================

class TestSerialization:
    """Tests that the schema dict is serializable."""

    def test_serializable_to_json(self) -> None:
        """Schema dict is serializable to JSON string via json.dumps."""
        result = build_layer_config(_multi_reg)
        schema = generate_layer_json_schema(result)

        json_str = json.dumps(schema)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_roundtrip_json_serialization(self) -> None:
        """Schema survives json.dumps -> json.loads roundtrip."""
        result = build_layer_config(_multi_reg)
        schema = generate_layer_json_schema(result)

        json_str = json.dumps(schema)
        deserialized = json.loads(json_str)

        assert deserialized == schema


# ===================================================================
# Type representation tests
# ===================================================================

class TestTypeRepresentation:
    """Tests for correct JSON Schema representation of Python types."""

    def test_str_represented_as_string(self) -> None:
        """Python str maps to JSON Schema type 'string'."""
        result = build_layer_config(_basic_types_reg)
        schema = generate_layer_json_schema(result)

        name_prop = _find_property_in_schema(schema, "name")
        assert name_prop is not None, "name property not found in schema"
        assert name_prop.get("type") == "string"

    def test_int_represented_as_integer(self) -> None:
        """Python int maps to JSON Schema type 'integer'."""
        result = build_layer_config(_basic_types_reg)
        schema = generate_layer_json_schema(result)

        count_prop = _find_property_in_schema(schema, "count")
        assert count_prop is not None, "count property not found in schema"
        assert count_prop.get("type") == "integer"

    def test_float_represented_as_number(self) -> None:
        """Python float maps to JSON Schema type 'number'."""
        result = build_layer_config(_basic_types_reg)
        schema = generate_layer_json_schema(result)

        score_prop = _find_property_in_schema(schema, "score")
        assert score_prop is not None, "score property not found in schema"
        assert score_prop.get("type") == "number"

    def test_bool_represented_as_boolean(self) -> None:
        """Python bool maps to JSON Schema type 'boolean'."""
        result = build_layer_config(_basic_types_reg)
        schema = generate_layer_json_schema(result)

        enabled_prop = _find_property_in_schema(schema, "enabled")
        assert enabled_prop is not None, "enabled property not found in schema"
        assert enabled_prop.get("type") == "boolean"

    def test_list_represented_as_array(self) -> None:
        """Python list[str] maps to JSON Schema type 'array' with items."""
        result = build_layer_config(_complex_types_reg)
        schema = generate_layer_json_schema(result)

        tags_prop = _find_property_in_schema(schema, "tags")
        assert tags_prop is not None, "tags property not found in schema"
        assert tags_prop.get("type") == "array"
        # Should have items specifying element type
        assert "items" in tags_prop, f"Expected 'items' in array property. Got: {tags_prop}"

    def test_dict_represented_as_object(self) -> None:
        """Python dict[str, int] maps to JSON Schema type 'object'."""
        result = build_layer_config(_complex_types_reg)
        schema = generate_layer_json_schema(result)

        metadata_prop = _find_property_in_schema(schema, "metadata")
        assert metadata_prop is not None, "metadata property not found in schema"
        assert metadata_prop.get("type") == "object"

    def test_optional_represented_with_anyof_or_nullable(self) -> None:
        """Python Optional[str] is represented with anyOf (type + null) or nullable."""
        result = build_layer_config(_complex_types_reg)
        schema = generate_layer_json_schema(result)

        desc_prop = _find_property_in_schema(schema, "description")
        assert desc_prop is not None, "description property not found in schema"

        # Pydantic v2 represents Optional as anyOf: [{type: string}, {type: null}]
        # or may use "type": ["string", "null"]
        has_anyof = "anyOf" in desc_prop
        has_type_list = isinstance(desc_prop.get("type"), list)
        has_null = any(item.get("type") == "null" for item in desc_prop.get("anyOf", [])) or (
            has_type_list and "null" in desc_prop["type"]
        )

        assert has_anyof or has_type_list, (
            f"Expected Optional[str] to use anyOf or type list for nullable. "
            f"Got: {desc_prop}"
        )
        assert has_null, f"Expected a null branch for Optional[str]. Got: {desc_prop}"


# ===================================================================
# Degraded fields in JSON Schema
# ===================================================================

def _build_degraded(registrar: Any) -> LayerConfigResult:
    """Build a layer whose fields degrade; the DegradationWarning is part of the contract."""
    from conscribe.exceptions import DegradationWarning

    with pytest.warns(DegradationWarning, match="degrades to Any"):
        return build_layer_config(registrar)


class _IncompatibleType:
    """A type Pydantic cannot serialize."""

    def __init__(self, x: int):
        self.x = x


_degraded_json_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _DegradedJsonBase(metaclass=_degraded_json_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DegradedJsonProvider(_DegradedJsonBase):
    __registry_key__ = "degraded"

    def __init__(self, *, model_id: str, auth: _IncompatibleType, count: int = 0):
        self.model_id = model_id
        self.auth = auth
        self.count = count


class TestDegradedFieldsJsonSchema:
    """Tests for degraded fields metadata in JSON Schema."""

    def test_x_degraded_fields_present(self) -> None:
        """JSON schema contains x-degraded-fields when degradation occurred."""
        result = _build_degraded(_degraded_json_reg)
        schema = generate_layer_json_schema(result)

        assert "x-degraded-fields" in schema
        x_degraded = schema["x-degraded-fields"]
        assert "degraded" in x_degraded
        field_entries = x_degraded["degraded"]
        field_names = [e["field"] for e in field_entries]
        assert "auth" in field_names

    def test_no_degraded_no_extension(self) -> None:
        """JSON schema does NOT contain x-degraded-fields when no degradation."""
        result = build_layer_config(_single_reg)
        schema = generate_layer_json_schema(result)

        assert "x-degraded-fields" not in schema

    def test_degraded_property_has_description(self) -> None:
        """Degraded field property has a description with [degraded] prefix."""
        result = _build_degraded(_degraded_json_reg)
        schema = generate_layer_json_schema(result)

        auth_prop = _find_property_in_schema(schema, "auth")
        assert auth_prop is not None
        desc = auth_prop.get("description", "")
        assert "[degraded]" in desc

    def test_degraded_property_has_x_degraded_from(self) -> None:
        """Degraded field property has x-degraded-from extension."""
        result = _build_degraded(_degraded_json_reg)
        schema = generate_layer_json_schema(result)

        auth_prop = _find_property_in_schema(schema, "auth")
        assert auth_prop is not None
        assert "x-degraded-from" in auth_prop


# ===================================================================
# Multi-model degraded JSON Schema ($defs path)
# ===================================================================

_multi_degraded_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["Multi"],
)


class _MultiDegradedBase(metaclass=_multi_degraded_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DegradedMulti(_MultiDegradedBase):
    """Provider with an incompatible param for multi-model $defs test."""
    __registry_key__ = "degraded"

    def __init__(self, *, model_id: str, auth: _IncompatibleType, count: int = 0):
        self.model_id = model_id
        self.auth = auth
        self.count = count


class CleanMulti(_MultiDegradedBase):
    """Provider with all-compatible params."""
    __registry_key__ = "clean"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class TestDegradedFieldsMultiModelJsonSchema:
    """Tests for degraded fields in multi-model JSON Schema ($defs path)."""

    def test_multi_model_x_degraded_fields_present(self) -> None:
        """Multi-model schema has x-degraded-fields with the degraded key."""
        result = _build_degraded(_multi_degraded_reg)
        schema = generate_layer_json_schema(result)

        assert "x-degraded-fields" in schema
        assert "degraded" in schema["x-degraded-fields"]
        # Clean provider should NOT be in x-degraded-fields
        assert "clean" not in schema["x-degraded-fields"]

    def test_multi_model_defs_annotated(self) -> None:
        """In multi-model schema, the degraded model inside $defs has
        x-degraded-from and description on the degraded property."""
        result = _build_degraded(_multi_degraded_reg)
        schema = generate_layer_json_schema(result)

        # Multi-model schema should have $defs
        assert "$defs" in schema

        auth_prop = _find_property_in_schema(schema, "auth")
        assert auth_prop is not None
        assert "x-degraded-from" in auth_prop
        assert "[degraded]" in auth_prop.get("description", "")

    def test_multi_model_clean_model_not_annotated(self) -> None:
        """The clean (non-degraded) model in $defs has no x-degraded-from."""
        result = _build_degraded(_multi_degraded_reg)
        schema = generate_layer_json_schema(result)

        temp_prop = _find_property_in_schema(schema, "temperature")
        assert temp_prop is not None
        assert "x-degraded-from" not in temp_prop


# ===================================================================
# Degraded field with existing description (append path)
# ===================================================================

_desc_degraded_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _DescDegradedBase(metaclass=_desc_degraded_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DescDegradedProvider(_DescDegradedBase):
    """Provider with a degraded field that has an existing description."""
    __registry_key__ = "desc_degraded"

    def __init__(
        self,
        *,
        model_id: str,
        auth: Annotated[_IncompatibleType, Field(description="Authentication handler")],
    ):
        self.model_id = model_id
        self.auth = auth


class TestDegradedFieldDescriptionAppend:
    """Test that degradation info is appended when field already has a description."""

    def test_existing_description_preserved_and_appended(self) -> None:
        """When a degraded field already has a description, the degradation
        info is appended rather than replacing it."""
        result = _build_degraded(_desc_degraded_reg)
        schema = generate_layer_json_schema(result)

        auth_prop = _find_property_in_schema(schema, "auth")
        assert auth_prop is not None
        desc = auth_prop.get("description", "")
        # Original description should be preserved
        assert "Authentication handler" in desc
        # Degradation info should be appended
        assert "[degraded]" in desc


# ===================================================================
# conscribe 2.0: deterministic schemas, published == in-process
# ===================================================================

import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402

from pydantic import create_model  # noqa: E402

from conscribe.config.json_schema import _DeterministicJsonSchema  # noqa: E402

from tests.config.artifact_parity import normalize_schema, schema_of, same_value  # noqa: E402


_SEED_SCRIPT = textwrap.dedent(
    """
    import json
    from typing import Protocol, runtime_checkable
    from conscribe import create_registrar
    from conscribe.config.builder import build_layer_config
    from conscribe.config.codegen import generate_layer_config_source
    from conscribe.config.json_schema import generate_layer_json_schema

    @runtime_checkable
    class P(Protocol):
        def run(self) -> None: ...

    reg = create_registrar("seeded", P, discriminator_field="kind")

    class Base(metaclass=reg.Meta):
        __abstract__ = True
        def run(self) -> None: ...

    class Tagged(Base):
        __registry_key__ = "tagged"
        def __init__(
            self,
            tags: frozenset[str] = frozenset({"alpha", "beta", "gamma", "delta", "eps"}),
            names: set[str] = {"zeta", "eta", "theta", "iota"},
        ):
            pass

    result = build_layer_config(reg)
    print(json.dumps(generate_layer_json_schema(result), sort_keys=True))
    print(generate_layer_config_source(result))
    """
)


def _run_with_seed(seed: str) -> str:
    env = {**__import__("os").environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", _SEED_SCRIPT], capture_output=True, text=True, env=env, check=True,
    )
    return proc.stdout


class TestDeterminism:
    """Published artifacts do not depend on ``PYTHONHASHSEED``."""

    def test_set_defaults_are_sorted(self) -> None:
        model = create_model(
            "SetDefaults",
            tags=(frozenset[str], frozenset({"b", "c", "a"})),
            nested=(dict[str, set[int]], {"k": {3, 1, 2}}),
        )
        schema = model.model_json_schema(schema_generator=_DeterministicJsonSchema)
        assert schema["properties"]["tags"]["default"] == ["a", "b", "c"]
        assert schema["properties"]["nested"]["default"] == {"k": [1, 2, 3]}

    def test_schema_and_source_identical_across_hash_seeds(self) -> None:
        outputs = {_run_with_seed(seed) for seed in ("1", "2", "3")}
        assert len(outputs) == 1


class TestPublishedSchemaParity:
    """The published schema equals the in-process schema (``x-`` keys aside)."""

    @pytest.mark.parametrize("registrar", [
        _single_reg, _multi_reg, _description_reg, _constraint_reg, _required_reg,
        _default_reg, _basic_types_reg, _complex_types_reg, _open_reg, _disc_const_reg,
    ])
    def test_layer_schema(self, registrar: Any) -> None:
        result = build_layer_config(registrar)
        published = generate_layer_json_schema(result)
        stripped = {k: v for k, v in published.items() if not k.startswith("x-")}
        assert same_value(normalize_schema(stripped), normalize_schema(schema_of(result.union_type)))

    def test_factory_fields_publish_no_default(self) -> None:
        def make() -> list[str]:
            return ["*"]

        model = create_model("FactorySchema", globs=(list[str], Field(default_factory=make)))
        prop = schema_of(model)["properties"]["globs"]
        assert "default" not in prop


# ===================================================================
# The composed schema is the top-level model's own schema (F3)
# ===================================================================

from conscribe.config.composed import ComposedConfigResult, build_composed_config  # noqa: E402
from conscribe.config.json_schema import generate_composed_json_schema  # noqa: E402


@runtime_checkable
class _SinkProto(Protocol):
    def run(self) -> None: ...


def _sink_and_hook(*, inline_only: bool) -> dict[str, Any]:
    sink = create_registrar("js_sink", _SinkProto, discriminator_field="sink_id", inline_only=inline_only)
    hook = create_registrar("js_hook", _SinkProto, discriminator_field="hook_id")

    class _SinkBase(metaclass=sink.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class Disk(_SinkBase):
        __registry_key__ = "disk"

        def __init__(self, tags: frozenset[str] = frozenset({"b", "a"}), rotate: bool = False): ...

    class Net(_SinkBase):
        __registry_key__ = "net"

    class _HookBase(metaclass=hook.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class Writer(_HookBase):
        __registry_key__ = "writer"
        __wiring__ = {"archive": {"registry": "js_sink", "keys": ["disk", "net"]}}

        # (``label`` keeps the 1.x extractor off its single-parameter check,
        # which raises on a lone ``Optional[list[X]]`` under Python 3.10.)
        def __init__(self, archive: Optional[list[_SinkProto]] = None, label: str = "w"): ...

    return {"js_sink": sink, "js_hook": hook}


class TestComposedSchemaIsTheModelSchema:
    @pytest.mark.parametrize("mode", ["forbid", "ignore", "allow"])
    @pytest.mark.parametrize("inline", [True, False])
    def test_published_equals_model_json_schema(self, conscribe_isolated, inline: bool, mode: str) -> None:
        result = build_composed_config(
            _sink_and_hook(inline_only=inline), inline_wiring=inline, top_level_extra=mode,
        )
        top = result.top_level_type
        published = generate_composed_json_schema(result)
        assert published == top.model_json_schema(schema_generator=_DeterministicJsonSchema)
        assert published["x-composed-layers"] == result.dependency_order
        assert published["x-inline-wiring"] is inline
        assert ("x-inline-only-layers" in published) is inline
        assert ("not" in published) is (inline and mode != "forbid")
        assert published.get("additionalProperties") == {"forbid": False, "ignore": None, "allow": True}[mode]
        # Set-valued defaults are sorted, independently of PYTHONHASHSEED.
        disk = published["$defs"][result.layer_results["js_sink"].per_key_models["disk"].__name__]
        assert disk["properties"]["tags"]["default"] == ["a", "b"]

    def test_nothing_is_added_to_the_model_schema(self) -> None:
        """A result whose top-level model carries no extensions publishes none."""
        top = create_model("ComposedConfig", hook=(list[int], []))
        result = ComposedConfigResult(
            layer_results={}, top_level_type=top, dependency_order=["hook"], inline_wiring=True,
        )
        published = generate_composed_json_schema(result)
        assert published == top.model_json_schema()
        assert not any(key.startswith("x-") for key in published)


class TestDegradedMarkerIdempotent:
    def test_marker_carried_by_the_field_is_kept(self) -> None:
        """When the field's own schema already carries ``x-degraded-from``
        (its FieldInfo), the layer schema adds no second hint."""
        model = create_model(
            "MarkedConfig",
            kind=(Any, "marked"),
            auth=(Any, Field(default=None, description="auth", json_schema_extra={"x-degraded-from": "httpx.Auth"})),
        )
        result = LayerConfigResult(
            union_type=model,
            per_key_models={"marked": model},
            layer_name="probe",
            discriminator_field="kind",
            degraded_fields={"marked": [DegradedField("auth", "httpx.Auth", "pkg.Client")]},
        )
        schema = generate_layer_json_schema(result)
        prop = schema["properties"]["auth"]
        assert prop["x-degraded-from"] == "httpx.Auth"
        assert prop["description"] == "auth"
        assert schema["x-degraded-fields"] == {"marked": [{"field": "auth", "original_type": "httpx.Auth"}]}


# ===================================================================
# Render review fixes (2.0.0)
# ===================================================================

import dataclasses  # noqa: E402
import enum  # noqa: E402

from pydantic import ConfigDict  # noqa: E402

from conscribe.config.json_schema import _presort  # noqa: E402


class _Mode(enum.Enum):
    A = "a"


class _FrozenTag(BaseModel):
    model_config = ConfigDict(frozen=True)
    rank: int = 1
    labels: frozenset[str] = frozenset({"x", "y", "z", "w"})


class _Tagged(BaseModel):
    tags: frozenset[_FrozenTag] = frozenset({_FrozenTag(rank=2), _FrozenTag(rank=1)})
    by_alias: set[str] = Field(default={"q", "p"}, alias="byAlias")
    order: tuple[str, ...] = ("z", "y")


@dataclasses.dataclass
class _Box:
    items: set[str] = dataclasses.field(default_factory=lambda: {"b", "a", "c"})
    count: int = 0


class TestSetMembersInsideInstances:
    """R03: set members are sorted at any depth, also inside model and
    dataclass instances (pydantic cannot dump a set of models in python
    mode: the instance is walked next to its JSON dump)."""

    def test_models_and_dataclasses(self) -> None:
        assert _presort(_Tagged()) == {
            "tags": [
                {"rank": 1, "labels": ["w", "x", "y", "z"]},
                {"rank": 2, "labels": ["w", "x", "y", "z"]},
            ],
            "byAlias": ["p", "q"],
            "order": ["z", "y"],  # sequences keep their order
        }
        assert _presort(_Box()) == {"items": ["a", "b", "c"], "count": 0}
        assert _presort([_Box()]) == [{"items": ["a", "b", "c"], "count": 0}]

    def test_instances_without_sets_are_left_to_pydantic(self) -> None:
        tag = _FrozenTag.model_construct(rank=3, labels=("b", "a"))
        assert _presort(tag) is tag

    def test_published_default_of_a_model_instance(self) -> None:
        model = create_model("HoldsInstance", tagged=(_Tagged, _Tagged()))
        schema = model.model_json_schema(schema_generator=_DeterministicJsonSchema)
        default = schema["properties"]["tagged"]["default"]
        assert default["tags"][0] == {"rank": 1, "labels": ["w", "x", "y", "z"]}
        assert default["byAlias"] == ["p", "q"]

    def test_set_members_are_ordered_by_their_json_form(self) -> None:
        """Members sort by type name, then canonical JSON text (the order of
        composed default instances): enum members by value, types grouped."""
        assert _presort({_Mode.A, "b", 1}) == [1, _Mode.A, "b"]
        assert _presort(frozenset({"é", "z", "a"})) == ["a", "z", "é"]


class TestDegradedHintOnOneKeyLayers:
    def test_root_properties_are_annotated_when_defs_exist(self, conscribe_isolated) -> None:
        """R04: a one-key layer's schema is the per-key model; nested models or
        enums in ``$defs`` do not move its properties there."""
        reg = create_registrar("r04_solo", _LLMProto, discriminator_field="kind")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class Solo(_Base):
            __registry_key__ = "solo"

            def __init__(self, handle: _IncompatibleType = None, mode: _Mode = _Mode.A):  # type: ignore[assignment]
                pass

        result = _build_degraded(reg)
        schema = generate_layer_json_schema(result)
        assert list(schema["$defs"]) == ["_Mode"]
        handle = schema["properties"]["handle"]
        assert handle["x-degraded-from"] == result.degraded_fields["solo"][0].original_type_repr
        assert handle["description"].startswith("[degraded] Type was: ")
        assert "x-degraded-from" not in schema["properties"]["mode"]
