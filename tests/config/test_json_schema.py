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
        has_null_ref = any(
            item.get("type") == "null"
            for item in desc_prop.get("anyOf", [])
        ) if has_anyof else False

        assert has_anyof or has_type_list, (
            f"Expected Optional[str] to use anyOf or type list for nullable. "
            f"Got: {desc_prop}"
        )
