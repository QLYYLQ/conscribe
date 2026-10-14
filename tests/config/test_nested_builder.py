"""Tests for deeply nested compound discriminated config builder."""
from __future__ import annotations

import dataclasses
import inspect
import itertools
import json

import pydantic
import pytest
from typing import TYPE_CHECKING, Annotated, Any, Optional, Protocol, get_args, runtime_checkable

from pydantic import BaseModel, Discriminator, Field, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined

from conscribe import create_registrar
from conscribe.config.builder import GENERATED_MARKER, LayerConfigResult, _layer_union, build_layer_config
from conscribe.config.codegen import _union_alias_name, generate_layer_config_source
from conscribe.config.json_schema import generate_layer_json_schema
from conscribe.exceptions import (
    ConfigAnnotationError,
    ConfigDeclarationError,
    DegradationWarning,
    ExtractionWarning,
)

from tests.config.artifact_parity import assert_artifacts_agree, layer_expectations

if TYPE_CHECKING:
    # Only for the type checker: a nested-mode field annotated with it degrades.
    from decimal import Context as TypeCheckingOnly

_request_ids = itertools.count(1)


def _next_request_id() -> int:
    """An impure default factory: every config must get its own value."""
    return next(_request_ids)


@runtime_checkable
class LLMProtocol(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@pytest.fixture
def nested_registrar():
    """Create a registrar with nested discriminator fields."""
    registrar = create_registrar(
        "llm",
        LLMProtocol,
        discriminator_fields=["model_type", "provider"],
        key_separator=".",
    )
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)


class TestBasicNestedConfig:
    """Test basic nested config building."""

    def test_two_level_nested_config(self, nested_registrar):
        R = nested_registrar

        class OpenAIBase(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7, max_tokens: int = 1000):
                self.temperature = temperature
                self.max_tokens = max_tokens

        class AzureOpenAI(OpenAIBase):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, api_version: str = "2024-02", **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment
                self.api_version = api_version

        class OfficialOpenAI(OpenAIBase):
            __registry_key__ = "openai.official"
            def __init__(self, endpoint: str, **kwargs):
                super().__init__(**kwargs)
                self.endpoint = endpoint

        result = R.build_config()

        # Check result structure
        assert isinstance(result, LayerConfigResult)
        assert result.discriminator_fields == ["model_type", "provider"]
        assert result.key_separator == "."
        assert set(result.per_key_models.keys()) == {"openai.azure", "openai.official"}

        # Check per_segment_models
        assert result.per_segment_models is not None
        assert "provider" in result.per_segment_models

    def test_nested_config_validation(self, nested_registrar):
        """Test that the generated union validates correctly."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        class OfficialImpl(Base):
            __registry_key__ = "openai.official"
            def __init__(self, endpoint: str = "https://api.openai.com", **kwargs):
                super().__init__(**kwargs)
                self.endpoint = endpoint

        result = R.build_config()
        adapter = TypeAdapter(result.union_type)

        # Validate Azure config
        azure_config = adapter.validate_python({
            "model_type": "openai",
            "temperature": 0.9,
            "provider": {
                "name": "azure",
                "deployment": "my-deploy",
            },
        })
        assert azure_config.model_type == "openai"
        assert azure_config.temperature == 0.9
        assert azure_config.provider.name == "azure"
        assert azure_config.provider.deployment == "my-deploy"

        # Validate Official config
        official_config = adapter.validate_python({
            "model_type": "openai",
            "provider": {
                "name": "official",
                "endpoint": "https://custom.api.com",
            },
        })
        assert official_config.model_type == "openai"
        assert official_config.provider.name == "official"
        assert official_config.provider.endpoint == "https://custom.api.com"

    def test_invalid_combination_rejected(self, nested_registrar):
        """Unregistered key combinations should fail validation."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        result = R.build_config()
        adapter = TypeAdapter(result.union_type)

        # No "openai.bedrock" registered → should fail
        with pytest.raises(Exception):
            adapter.validate_python({
                "model_type": "openai",
                "provider": {
                    "name": "bedrock",
                    "region": "us-east-1",
                },
            })


class TestMultipleLevel0Groups:
    """Test with multiple level-0 groups (e.g., openai.*, anthropic.*)."""

    def test_multiple_model_types(self, nested_registrar):
        R = nested_registrar

        class OpenAIBase(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureOpenAI(OpenAIBase):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        class AnthropicBase(metaclass=R.Meta):
            __registry_key__ = "anthropic"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, max_tokens: int = 4096):
                self.max_tokens = max_tokens

        class AnthropicOfficial(AnthropicBase):
            __registry_key__ = "anthropic.official"
            def __init__(self, api_key: str = "", **kwargs):
                super().__init__(**kwargs)
                self.api_key = api_key

        result = R.build_config()
        adapter = TypeAdapter(result.union_type)

        # Validate OpenAI Azure
        azure = adapter.validate_python({
            "model_type": "openai",
            "temperature": 0.5,
            "provider": {"name": "azure", "deployment": "gpt4"},
        })
        assert azure.model_type == "openai"

        # Validate Anthropic Official
        anthropic = adapter.validate_python({
            "model_type": "anthropic",
            "max_tokens": 8192,
            "provider": {"name": "official", "api_key": "sk-xxx"},
        })
        assert anthropic.model_type == "anthropic"


class TestParamSplitting:
    """Test that params are correctly split by MRO level."""

    def test_parent_params_at_level_0(self, nested_registrar):
        """Protocol params should be at level 0 (flat)."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7, max_tokens: int = 1000):
                self.temperature = temperature
                self.max_tokens = max_tokens

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        result = R.build_config()
        model = result.per_key_models["openai.azure"]

        # Level 0 params should be flat on the combined model
        assert "model_type" in model.model_fields
        assert "temperature" in model.model_fields
        assert "max_tokens" in model.model_fields

        # Level 1 param should be in the nested model
        assert "provider" in model.model_fields
        provider_type = model.model_fields["provider"].annotation
        assert hasattr(provider_type, "model_fields")
        assert "deployment" in provider_type.model_fields

    def test_leaf_only_class_params_at_level_1(self, nested_registrar):
        """Leaf-only params should be at level 1 (nested)."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, api_version: str = "2024-02", **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment
                self.api_version = api_version

        result = R.build_config()
        model = result.per_key_models["openai.azure"]

        # deployment and api_version should be nested
        provider_type = model.model_fields["provider"].annotation
        assert "deployment" in provider_type.model_fields
        assert "api_version" in provider_type.model_fields


class TestNestedCodegen:
    """Test code generation for nested mode."""

    def test_codegen_has_discriminator_and_tag(self, nested_registrar):
        """Generated source should use Discriminator and Tag."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        class OfficialImpl(Base):
            __registry_key__ = "openai.official"
            def __init__(self, endpoint: str = "https://api.openai.com", **kwargs):
                super().__init__(**kwargs)
                self.endpoint = endpoint

        result = R.build_config()
        source = generate_layer_config_source(result)

        # Check imports
        assert "Discriminator" in source
        assert "Tag" in source

        # Check header
        assert "Discriminator fields:" in source
        assert "Key separator:" in source

        # Check discriminator function
        assert f"_discriminate_llm" in source

        # Check Tag usage in union
        assert 'Tag("openai.azure")' in source
        assert 'Tag("openai.official")' in source

    def test_codegen_has_nested_models(self, nested_registrar):
        """Generated source should have separate segment and combined models."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        result = R.build_config()
        source = generate_layer_config_source(result)

        # Should have nested submodel section
        assert "Nested submodels" in source
        # Should have combined model section
        assert "Combined models" in source


class TestNestedJsonSchema:
    """Test JSON schema generation for nested mode."""

    def test_json_schema_has_extensions(self, nested_registrar):
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        result = R.build_config()
        schema = generate_layer_json_schema(result)

        assert "x-discriminator-fields" in schema
        assert schema["x-discriminator-fields"] == ["model_type", "provider"]
        assert "x-key-separator" in schema
        assert schema["x-key-separator"] == "."


class TestBackwardCompat:
    """Ensure flat mode is completely unchanged."""

    def test_flat_mode_unchanged(self):
        """Flat mode should produce identical results to before."""
        R = create_registrar(
            "agent_bc",
            LLMProtocol,
            discriminator_field="provider",
        )

        class Base(metaclass=R.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class Impl(Base):
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        result = R.build_config()

        # Should be flat mode
        assert result.discriminator_fields is None
        assert result.key_separator == ""
        assert result.per_segment_models is None

        # Should validate normally
        adapter = TypeAdapter(result.union_type)
        config = adapter.validate_python({
            "provider": "impl",
            "temperature": 0.5,
        })
        assert config.temperature == 0.5

        for key in list(R.keys()):
            R.unregister(key)

    def test_flat_mode_codegen(self):
        """Flat mode codegen should use Field(discriminator=...) not Discriminator()."""
        R = create_registrar(
            "agent_bc2",
            LLMProtocol,
            discriminator_field="provider",
        )

        class Base(metaclass=R.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class ImplA(Base):
            __registry_key__ = "a"

        class ImplB(Base):
            __registry_key__ = "b"

        result = R.build_config()
        source = generate_layer_config_source(result)

        # Should use Field(discriminator=...) not Discriminator()
        assert 'Field(discriminator=' in source
        assert 'Discriminator(' not in source

        for key in list(R.keys()):
            R.unregister(key)


class TestLeafKeyFiltering:
    """Test that only leaf keys appear in the config union."""

    def test_non_leaf_keys_excluded(self, nested_registrar):
        """Keys with fewer segments than discriminator_fields are excluded."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        # Even if somehow registered (shouldn't happen with __abstract__),
        # non-leaf keys should be excluded from the config union
        class Leaf(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        result = R.build_config()
        # Only leaf key should be in the result
        assert "openai.azure" in result.per_key_models
        assert "openai" not in result.per_key_models


# ===================================================================
# Phase 4: Tests for uncommitted builder.py code
# ===================================================================


class TestLeafParamPromotion:
    """Test that leaf-only params are promoted to level 0 (4a)."""

    def test_single_provider_params_flat(self, nested_registrar):
        """Class with key 'google.default' but no abstract parent defining
        level-0 params should have its params appear flat at level 0."""
        R = nested_registrar

        class GoogleBase(metaclass=R.Meta):
            __registry_key__ = "google"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class GoogleDefault(GoogleBase):
            __registry_key__ = "google.default"
            def __init__(self, api_key: str, region: str = "us"):
                self.api_key = api_key
                self.region = region

        result = R.build_config()
        model = result.per_key_models["google.default"]

        # Params should be flat on the combined model (level 0), not nested
        assert "model_type" in model.model_fields
        # At least one of the params should be directly on the model
        flat_fields = set(model.model_fields.keys()) - {"model_type", "provider"}
        assert len(flat_fields) > 0, (
            "Leaf-only params should be promoted to level 0 (flat)"
        )


class TestNestedAutoDefault:
    """Test nested sub-model auto-default behavior (4b)."""

    def test_all_defaults_provider_optional(self, nested_registrar):
        """When ALL nested sub-model fields have defaults, the provider
        field itself becomes optional (has a default)."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class OfficialImpl(Base):
            __registry_key__ = "openai.official"
            def __init__(self, endpoint: str = "https://api.openai.com", **kwargs):
                super().__init__(**kwargs)
                self.endpoint = endpoint

        result = R.build_config()
        model = result.per_key_models["openai.official"]

        # The provider field should have a default (not PydanticUndefined)
        provider_field = model.model_fields.get("provider")
        assert provider_field is not None
        assert provider_field.default is not PydanticUndefined, (
            "Provider field should be optional when all sub-fields have defaults"
        )

    def test_required_field_provider_required(self, nested_registrar):
        """When a nested sub-model has at least one required field,
        the provider field stays required (...)."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(self, deployment: str, **kwargs):
                super().__init__(**kwargs)
                self.deployment = deployment

        result = R.build_config()
        model = result.per_key_models["openai.azure"]

        # The provider field should be required
        provider_field = model.model_fields.get("provider")
        assert provider_field is not None
        assert provider_field.default is PydanticUndefined, (
            "Provider field should be required when sub-model has required fields"
        )


class TestAnnotatedOnlyNested:
    """Test __config_annotated_only__ in nested mode (4c)."""

    def test_annotated_only_filters_bare_params(self, nested_registrar):
        """Parent with __config_annotated_only__ = True: only params with
        Annotated[..., Field()] should appear in the config."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            __config_annotated_only__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(
                self,
                temperature: Annotated[float, Field(description="Sampling temp")] = 0.7,
                internal_state: str = "init",
            ):
                self.temperature = temperature
                self.internal_state = internal_state

        class AzureImpl(Base):
            __registry_key__ = "openai.azure"
            def __init__(
                self,
                deployment: Annotated[str, Field(description="Azure deployment")],
                _debug: bool = False,
                **kwargs,
            ):
                super().__init__(**kwargs)
                self.deployment = deployment
                self._debug = _debug

        result = R.build_config()
        model = result.per_key_models["openai.azure"]

        # Collect all field names from combined model and nested models
        all_fields = set(model.model_fields.keys())
        for fname, finfo in model.model_fields.items():
            if hasattr(finfo.annotation, "model_fields"):
                all_fields.update(finfo.annotation.model_fields.keys())

        # Annotated params should be present
        assert "temperature" in all_fields or "deployment" in all_fields

        # Bare params should NOT be present
        assert "internal_state" not in all_fields, (
            "Bare-typed params should be excluded in annotated-only mode"
        )
        assert "_debug" not in all_fields, (
            "Bare-typed params should be excluded in annotated-only mode"
        )


class TestDiscriminatorDefaultFallback:
    """Test discriminator 'default' fallback (4d)."""

    def test_missing_name_field_returns_default(self, nested_registrar):
        """When the provider sub-model dict is missing the 'name' field,
        the discriminator should return 'default' (not empty string)."""
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class DefaultImpl(Base):
            __registry_key__ = "openai.default"
            def __init__(self, endpoint: str = "https://api.openai.com", **kwargs):
                super().__init__(**kwargs)
                self.endpoint = endpoint

        result = R.build_config()

        # The discriminator function should handle missing 'name' gracefully
        # Extract the discriminator function from the union_type
        from typing import get_args
        union_args = get_args(result.union_type)
        discriminator = None
        for arg in union_args:
            if hasattr(arg, "func") or hasattr(arg, "__metadata__"):
                # Get the Discriminator from metadata
                from typing import get_args as ga
                meta_args = ga(result.union_type)
                for ma in meta_args:
                    if hasattr(ma, "discriminator"):
                        discriminator = ma.discriminator
                        break
                break

        # Test with dict missing 'name' in provider
        test_dict = {
            "model_type": "openai",
            "provider": {},
        }

        # Validate via TypeAdapter — the "openai.default" key should match
        adapter = TypeAdapter(result.union_type)
        config = adapter.validate_python({
            "model_type": "openai",
            "provider": {"name": "default", "endpoint": "https://custom.api.com"},
        })
        assert config.model_type == "openai"


# ===================================================================
# 2.0: FieldInfo carry (descriptions, constraints, defaults, factories)
# ===================================================================


def _provider_model(model: type[BaseModel]) -> type[BaseModel]:
    return model.model_fields["provider"].annotation


class TestNestedFieldInfoCarry:
    """af's nested ``llm`` layer lost every description and ``ge``/``le``."""

    @staticmethod
    def _register(R):
        class OpenAIBase(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(
                self,
                temperature: Annotated[
                    float, Field(ge=0, le=2, description="Sampling temperature")
                ] = 0.7,
            ):
                self.temperature = temperature

        class AzureOpenAI(OpenAIBase):
            __registry_key__ = "openai.azure"

            def __init__(
                self,
                deployment: Annotated[str, Field(min_length=1, description="Azure deployment")],
                **kwargs,
            ):
                super().__init__(**kwargs)
                self.deployment = deployment

        return R.build_config()

    def test_descriptions_and_constraints_on_models(self, nested_registrar):
        result = self._register(nested_registrar)
        model = result.per_key_models["openai.azure"]
        temperature = model.model_fields["temperature"]
        assert temperature.description == "Sampling temperature"
        assert temperature.default == 0.7
        deployment = _provider_model(model).model_fields["deployment"]
        assert deployment.description == "Azure deployment"
        assert deployment.is_required()

    def test_constraints_are_enforced(self, nested_registrar):
        adapter = TypeAdapter(self._register(nested_registrar).union_type)
        with pytest.raises(ValidationError) as exc:
            adapter.validate_python({
                "model_type": "openai", "temperature": -1,
                "provider": {"name": "azure", "deployment": "d"},
            })
        assert [e["type"] for e in exc.value.errors()] == ["greater_than_equal"]
        with pytest.raises(ValidationError) as exc:
            adapter.validate_python({
                "model_type": "openai",
                "provider": {"name": "azure", "deployment": ""},
            })
        assert [e["type"] for e in exc.value.errors()] == ["string_too_short"]

    def test_json_schema_carries_descriptions_and_constraints(self, nested_registrar):
        result = self._register(nested_registrar)
        model = result.per_key_models["openai.azure"]
        schema = model.model_json_schema()
        temperature = schema["properties"]["temperature"]
        assert temperature["description"] == "Sampling temperature"
        assert (temperature["minimum"], temperature["maximum"]) == (0, 2)
        provider_def = next(d for name, d in schema["$defs"].items() if "Provider" in name)
        deployment = provider_def["properties"]["deployment"]
        assert deployment["description"] == "Azure deployment"
        assert deployment["minLength"] == 1
        layer_schema = json.dumps(generate_layer_json_schema(result))
        assert "Sampling temperature" in layer_schema and "Azure deployment" in layer_schema

    def test_generated_source_carries_descriptions(self, nested_registrar):
        source = generate_layer_config_source(self._register(nested_registrar))
        assert "Sampling temperature" in source
        assert "Azure deployment" in source

    def test_generated_source_keeps_distinct_segment_constraints(self, nested_registrar):
        R = nested_registrar

        class OpenAIBase(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, model: str = "gpt"):
                self.model = model

        class OpenAIDirect(OpenAIBase):
            __registry_key__ = "openai.direct"

            def __init__(self, timeout: Annotated[float, Field(ge=0)] = 1.0, **kwargs):
                super().__init__(**kwargs)

        class AnthropicBase(metaclass=R.Meta):
            __registry_key__ = "anthropic"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, model: str = "claude"):
                self.model = model

        class AnthropicDirect(AnthropicBase):
            __registry_key__ = "anthropic.direct"

            def __init__(self, timeout: Annotated[float, Field(ge=1)] = 1.0, **kwargs):
                super().__init__(**kwargs)

        result = R.build_config()
        payload = {"model_type": "openai", "provider": {"name": "direct", "timeout": 0.5}}
        TypeAdapter(result.union_type).validate_python(payload)  # in-process: ge=0

        ns: dict = {}
        exec(compile(generate_layer_config_source(result), "<nested-llm>", "exec"), ns)  # noqa: S102
        TypeAdapter(ns["LLMConfig"]).validate_python(payload)  # generated: must agree

    def test_docstring_descriptions(self, nested_registrar):
        R = nested_registrar

        class OpenAIBase(metaclass=R.Meta):
            """OpenAI models.

            Args:
                max_tokens: Token budget for one reply.
            """

            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, max_tokens: int = 1000):
                self.max_tokens = max_tokens

        class Official(OpenAIBase):
            """Official endpoint.

            Args:
                endpoint: Base URL of the API.
            """

            __registry_key__ = "openai.official"

            def __init__(self, endpoint: str = "https://api.openai.com", **kwargs):
                super().__init__(**kwargs)
                self.endpoint = endpoint

        model = R.build_config().per_key_models["openai.official"]
        assert model.model_fields["max_tokens"].description == "Token budget for one reply."
        endpoint = _provider_model(model).model_fields["endpoint"]
        assert endpoint.description == "Base URL of the API."

    def test_field_default_without_signature_default(self, nested_registrar):
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, retries: Annotated[int, Field(default=3, ge=0)]):
                self.retries = retries

        class Impl(Base):
            __registry_key__ = "openai.impl"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        model = R.build_config().per_key_models["openai.impl"]
        retries = model.model_fields["retries"]
        assert not retries.is_required()
        assert retries.default == 3

    def test_none_default_and_optional_annotated(self, nested_registrar):
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(
                self,
                seed: int = None,
                top_p: Optional[Annotated[float, Field(ge=0, le=1, description="nucleus")]] = None,
            ):
                self.seed = seed
                self.top_p = top_p

        class Impl(Base):
            __registry_key__ = "openai.impl"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        model = R.build_config().per_key_models["openai.impl"]
        assert model.model_fields["seed"].annotation == Optional[int]
        top_p = model.model_fields["top_p"]
        assert top_p.description == "nucleus"
        assert model(seed=None, top_p=None).top_p is None
        with pytest.raises(ValidationError):
            model(top_p=1.5)

    def test_factory_defaults_run_once_per_config(self, nested_registrar):
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class Official(Base):
            __registry_key__ = "openai.official"

            def __init__(
                self,
                request_id: Annotated[
                    int, Field(default_factory=_next_request_id, description="per-config id")
                ],
                tags: Annotated[list[str], Field(default_factory=list)],
            ):
                self.request_id = request_id
                self.tags = tags

        class Azure(Base):
            __registry_key__ = "openai.azure"

            def __init__(self, request_id: Annotated[int, Field(default_factory=_next_request_id)]):
                self.request_id = request_id

        result = R.build_config()
        model = result.per_key_models["openai.official"]
        # leaf-only params are promoted to level 0: plain fields, factories kept
        request_id = model.model_fields["request_id"]
        assert not request_id.is_required()
        assert request_id.default_factory is _next_request_id
        first, second = model(), model()
        assert first.request_id != second.request_id
        assert first.tags == [] and first.tags is not second.tags


class TestSegmentAutoDefault:
    """The segment block is optional iff no sub-model field is required."""

    def test_factory_only_segment_is_optional_and_not_frozen(self, nested_registrar):
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class Official(Base):
            __registry_key__ = "openai.official"

            def __init__(
                self,
                request_id: Annotated[int, Field(default_factory=_next_request_id)],
                **kwargs,
            ):
                super().__init__(**kwargs)
                self.request_id = request_id

        model = R.build_config().per_key_models["openai.official"]
        provider = model.model_fields["provider"]
        assert not provider.is_required()
        assert provider.default_factory is _provider_model(model)
        first, second = model(), model()
        assert first.provider.request_id != second.provider.request_id

    def test_three_levels_default_through_every_level(self):
        R = create_registrar(
            "llm3",
            LLMProtocol,
            discriminator_fields=["vendor", "region", "tier"],
            key_separator=".",
        )

        class Vendor(metaclass=R.Meta):
            __registry_key__ = "acme"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, x: int = 1):
                self.x = x

        class Region(Vendor):
            __registry_key__ = "acme.eu"
            __abstract__ = True

            def __init__(self, y: int = 2, **kwargs):
                super().__init__(**kwargs)

        class Tier(Region):
            __registry_key__ = "acme.eu.gold"

            def __init__(self, z: int = 3, **kwargs):
                super().__init__(**kwargs)

        try:
            model = R.build_config().per_key_models["acme.eu.gold"]
            region = model.model_fields["region"]
            assert not region.is_required()
            tier = region.annotation.model_fields["tier"]
            assert not tier.is_required()
            built = model()
            assert (built.x, built.region.y, built.region.tier.z) == (1, 2, 3)
        finally:
            for key in list(R.keys()):
                R.unregister(key)


# ===================================================================
# 2.0: __wiring__ / __config_schema__ are refused on nested-mode layers
# ===================================================================


class _NestedSchema(BaseModel):
    temperature: float = 0.7


class TestNestedDeclarationsRejected:
    @staticmethod
    def _base(R):
        class OpenAIBase(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        return OpenAIBase

    def test_wiring_on_leaf_raises(self, nested_registrar):
        Base = self._base(nested_registrar)

        class Azure(Base):
            __registry_key__ = "openai.azure"
            __wiring__ = {"tools": ["search", "browse"]}

        with pytest.raises(ConfigDeclarationError) as exc:
            nested_registrar.build_config()
        msg = str(exc.value)
        assert "not supported for nested-mode layers yet" in msg
        assert "Azure" in msg and "['tools']" in msg and "'openai.azure'" in msg

    def test_inherited_wiring_raises(self, nested_registrar):
        class OpenAIBase(metaclass=nested_registrar.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            __wiring__ = {"tools": ["search"]}

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class Azure(OpenAIBase):
            __registry_key__ = "openai.azure"

        with pytest.raises(ConfigDeclarationError, match=r"\['tools'\]"):
            nested_registrar.build_config()

    def test_wiring_excluded_with_none_is_fine(self, nested_registrar):
        class OpenAIBase(metaclass=nested_registrar.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            __wiring__ = {"tools": ["search"]}

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class Azure(OpenAIBase):
            __registry_key__ = "openai.azure"
            __wiring__ = {"tools": None}

        result = nested_registrar.build_config()
        assert "openai.azure" in result.per_key_models

    def test_config_schema_raises(self, nested_registrar):
        Base = self._base(nested_registrar)

        class Azure(Base):
            __registry_key__ = "openai.azure"
            __config_schema__ = _NestedSchema

        with pytest.raises(ConfigDeclarationError) as exc:
            nested_registrar.build_config()
        msg = str(exc.value)
        assert "__config_schema__" in msg and "not supported for nested-mode layers yet" in msg
        assert "Azure" in msg


# ===================================================================
# 2.0: generated-model marker on nested models
# ===================================================================


class TestNestedGeneratedMarker:
    def test_combined_and_segment_models_are_marked(self, nested_registrar):
        R = nested_registrar

        class Base(metaclass=R.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class Azure(Base):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs):
                super().__init__(**kwargs)

        result = R.build_config()
        model = result.per_key_models["openai.azure"]
        assert model.__dict__.get(GENERATED_MARKER) is True
        for segment_models in result.per_segment_models.values():
            for segment_model in segment_models.values():
                assert segment_model.__dict__.get(GENERATED_MARKER) is True
        assert GENERATED_MARKER not in json.dumps(generate_layer_json_schema(result))
        assert GENERATED_MARKER not in model().model_dump()


# ===================================================================
# 2.0 review fixes: the leaf's real constructor, discriminator, names
# ===================================================================


class _Opaque:
    """A type pydantic cannot build a schema for (degrades when the model is built)."""


def _union_discriminator(union_type: Any) -> Any:
    """The callable of a nested union's ``Discriminator``."""
    return next(m for m in get_args(union_type) if isinstance(m, Discriminator)).discriminator


def _union_tags(union_type: Any) -> list[str]:
    members = get_args(get_args(union_type)[0])
    return [get_args(member)[1].tag for member in members]


def _class_line(cls: type) -> tuple[str, int]:
    return inspect.getsourcefile(cls), inspect.getsourcelines(cls)[1]


def _location(record: Any, category: type[Warning]) -> tuple[str, int]:
    """Where the first warning of *category* was reported (pydantic < 2.10
    adds its own ``model_`` namespace warnings to the record)."""
    warning = next(w for w in record if issubclass(w.category, category))
    return warning.filename, warning.lineno


@pytest.fixture
def isolated(conscribe_isolated):
    yield conscribe_isolated


def _nested(name: str, levels: tuple[str, ...] = ("model_type", "provider")):
    """A nested-mode registrar and its abstract root class."""
    reg = create_registrar(name, LLMProtocol, discriminator_fields=list(levels), key_separator=".")

    class _Root(metaclass=reg.Meta):
        __abstract__ = True

        async def chat(self, messages: list[dict]) -> str:
            return "ok"

    return reg, _Root


class TestLeafConstructor:
    """Nested mode reads the leaf's runtime constructor chain (decision 2)."""

    def test_dataclass_default_factories(self, isolated):
        reg, root = _nested("nl_dc")

        @dataclasses.dataclass
        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True
            temperature: float = 0.7
            stops: list[str] = dataclasses.field(default_factory=list)

        @dataclasses.dataclass
        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"
            deployment: str = "d"
            headers: dict = dataclasses.field(default_factory=lambda: {"x": 1})
            request_id: int = dataclasses.field(default_factory=_next_request_id)

        result = reg.build_config()
        model = result.per_key_models["openai.azure"]
        assert list(model.model_fields) == ["model_type", "temperature", "stops", "provider"]
        assert model.model_fields["stops"].default == []  # pure factory -> plain default
        provider = model.model_fields["provider"].annotation
        assert provider.model_fields["headers"].default == {"x": 1}
        assert provider.model_fields["request_id"].default_factory is _next_request_id
        first, second = model(), model()
        assert first.provider.request_id != second.provider.request_id
        assert "<factory>" not in generate_layer_config_source(result)

    def test_pydantic_dataclass_fields(self, isolated):
        reg, _ = _nested("nl_pdc")

        @pydantic.dataclasses.dataclass
        class _OpenAI:
            __registry_key__ = "openai"
            temperature: Annotated[float, Field(ge=0, le=2)] = 0.7
            stops: list[str] = Field(default_factory=list)

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        @reg.register("openai.azure")
        @pydantic.dataclasses.dataclass
        class Azure(_OpenAI):
            deployment: str = "d"

        model = reg.build_config().per_key_models["openai.azure"]
        assert list(model.model_fields) == ["model_type", "temperature", "stops", "provider"]
        assert model.model_fields["stops"].default == []
        assert list(model.model_fields["provider"].annotation.model_fields) == ["name", "deployment"]
        with pytest.raises(ValidationError):
            model(temperature=3)

    def test_basemodel_leaves_keep_their_fields(self, isolated):
        reg, _ = _nested("nl_bm")

        class _OpenAI(BaseModel):
            __registry_key__ = "openai"
            temperature: float = Field(0.7, ge=0, description="sampling")

            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        @reg.register("openai.azure")
        class Azure(_OpenAI):
            deployment: str = "d"

        model = reg.build_config().per_key_models["openai.azure"]
        assert model.model_fields["temperature"].description == "sampling"
        assert list(model.model_fields["provider"].annotation.model_fields) == ["name", "deployment"]
        with pytest.raises(ValidationError):
            model(temperature=-1)

    def test_the_leaf_definition_wins_and_the_root_most_class_sets_the_level(self, isolated):
        reg, root = _nested("nl_override")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = 0.7, **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, temperature: float = 0.2, deployment: str = "d", **kwargs):
                super().__init__(temperature=temperature, **kwargs)

        model = reg.build_config().per_key_models["openai.azure"]
        assert model.model_fields["temperature"].default == 0.2  # the leaf's constructor
        assert "temperature" not in model.model_fields["provider"].annotation.model_fields

    def test_parameters_the_leaf_does_not_accept_are_not_config(self, isolated):
        reg, root = _nested("nl_noforward")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = 0.7): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d"):  # does not forward
                super().__init__()

        model = reg.build_config().per_key_models["openai.azure"]
        assert "temperature" not in model.model_fields
        with pytest.raises(ValidationError) as exc:
            model.model_validate({"model_type": "openai", "temperature": 0.1})
        assert exc.value.errors()[0]["type"] == "extra_forbidden"

    def test_non_keyed_mixin_joins_the_nearest_leafward_level(self, isolated):
        reg, root = _nested("nl_mixin")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = 0.7, **kwargs): ...

        class RetryMixin:
            def __init__(self, max_retries: int = 3, **kwargs):
                super().__init__(**kwargs)

        class Azure(RetryMixin, _OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs):
                super().__init__(**kwargs)

        model = reg.build_config().per_key_models["openai.azure"]
        assert list(model.model_fields) == ["model_type", "temperature", "provider"]
        provider = model.model_fields["provider"].annotation
        assert list(provider.model_fields) == ["name", "max_retries", "deployment"]

    def test_field_used_as_default_keeps_its_constraints(self, isolated):
        reg, root = _nested("nl_fielddefault")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = Field(0.7, ge=0, le=2), **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        model = reg.build_config().per_key_models["openai.azure"]
        assert model().temperature == 0.7
        with pytest.raises(ValidationError) as exc:
            model(temperature=5)
        assert exc.value.errors()[0]["type"] == "less_than_equal"

    def test_type_checking_only_annotation_degrades(self, isolated):
        reg, root = _nested("nl_tc")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = 0.7, **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, ctx: TypeCheckingOnly = None, **kwargs): ...

        with pytest.warns(DegradationWarning, match="TYPE_CHECKING") as record:
            result = reg.build_config()
        assert _location(record, DegradationWarning) == _class_line(Azure)
        provider = result.per_key_models["openai.azure"].model_fields["provider"].annotation
        assert provider.model_fields["ctx"].annotation is Any
        [degraded] = result.degraded_fields["openai.azure"]
        assert (degraded.field_name, degraded.reason) == ("ctx", "type_checking_only")
        assert provider.__dict__["__degraded_fields__"] == [degraded]

    def test_a_typo_is_an_annotation_error(self, isolated):
        reg, root = _nested("nl_typo")

        class Typo(root):
            __registry_key__ = "openai.typo"

            def __init__(self, depth: Integr = 2): ...  # noqa: F821

        with pytest.raises(ConfigAnnotationError, match="Integr"):
            reg.build_config()

    def test_warnings_before_an_error_are_still_reported(self, isolated):
        reg, root = _nested("nl_warn_then_fail")

        class Bad(root):
            __registry_key__ = "openai.bad"

            def __init__(self, ctx: TypeCheckingOnly = None, depth: Integr = 2): ...  # noqa: F821

        with pytest.warns(DegradationWarning) as record:
            with pytest.raises(ConfigAnnotationError, match="Integr"):
                reg.build_config()
        assert _location(record, DegradationWarning) == _class_line(Bad)

    def test_pydantic_incompatible_type_degrades_when_the_model_is_built(self, isolated):
        reg, root = _nested("nl_opaque")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, opaque: _Opaque = None, **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        with pytest.warns(DegradationWarning, match="pydantic cannot build a schema") as record:
            result = reg.build_config()
        # Reported at the class that declares the parameter.
        assert _location(record, DegradationWarning) == _class_line(_OpenAI)
        model = result.per_key_models["openai.azure"]
        assert model.model_fields["opaque"].annotation is Any
        [degraded] = result.degraded_fields["openai.azure"]
        assert (degraded.field_name, degraded.reason) == ("opaque", "pydantic_incompatible")
        assert model.__dict__["__degraded_fields__"] == [degraded]
        assert "degraded from" in generate_layer_config_source(result)

    def test_unreadable_signature_warns_and_opens_its_levels(self, isolated):
        reg, root = _nested("nl_unreadable")

        class Prov(root):
            __registry_key__ = "prov"
            __abstract__ = True
            __init__ = min  # no readable signature

        class M(Prov):
            __registry_key__ = "prov.m"

        with pytest.warns(ExtractionWarning, match="cannot read the signature") as record:
            model = reg.build_config().per_key_models["prov.m"]
        assert _location(record, ExtractionWarning) == _class_line(Prov)
        assert model.model_config["extra"] == "allow"  # level 0: Prov's level
        assert model.model_fields["provider"].annotation.model_config["extra"] == "forbid"
        assert model.model_validate({"model_type": "prov", "anything": 1}).model_extra == {"anything": 1}

    def test_config_from_param_is_refused(self, isolated):
        reg, root = _nested("nl_fromparam")

        class Settings(BaseModel):
            deployment: str = "d"

        class Azure(root):
            __registry_key__ = "openai.azure"
            __config_from_param__ = "settings"

            def __init__(self, settings: Settings): ...

        with pytest.raises(ConfigDeclarationError, match="__config_from_param__ = 'settings' is not supported"):
            reg.build_config()

    def test_unsupported_declarations_are_reported_before_wiring(self, isolated):
        reg, root = _nested("nl_wiring_first")

        class Azure(root):
            __registry_key__ = "openai.azure"
            __wiring__ = {"tool": "no_such_registry"}

        with pytest.raises(ConfigDeclarationError, match="__wiring__ is not supported"):
            build_layer_config(reg)


class TestNestedKeyShape:
    def test_keys_with_the_wrong_segment_count_are_refused(self, isolated):
        reg, root = _nested("nl_shape")

        class Leaf(root):
            __registry_key__ = "openai.azure"

        class Long(root):
            __registry_key__ = "openai.azure.gpt4"

        class Short(root):
            __registry_key__ = "local"

        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(reg)
        msg = str(exc.value)
        assert "every registered key must have exactly 2 segments ('<model_type>.<provider>')" in msg
        assert "'local' (" in msg and "Short, 1 segment)" in msg
        assert "'openai.azure.gpt4' (" in msg and "Long, 3 segments)" in msg
        assert "'openai.azure' (" not in msg


class TestReservedNestedFields:
    """Parameters named like a level's discriminator fields are refused (M5)."""

    def test_segment_parameter_named_name(self, isolated):
        reg, root = _nested("nl_res_name")

        class _Family(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, x: int = 1, **kwargs): ...

        class Azure(_Family):
            __registry_key__ = "openai.azure"

            def __init__(self, name: str = "my-deployment", **kwargs): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(reg)
        assert exc.value.field == "name"
        assert "collides with the segment name field 'name' of the 'provider' block" in str(exc.value)

    def test_family_parameter_named_like_the_block(self, isolated):
        reg, root = _nested("nl_res_block")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, provider: str = "p", **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(reg)
        assert exc.value.owner.endswith("_OpenAI") and exc.value.field == "provider"
        assert "level 0" in str(exc.value) and "the 'provider' segment block" in str(exc.value)

    def test_family_parameter_named_like_the_level0_discriminator(self, isolated):
        reg, root = _nested("nl_res_head")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, model_type: str = "chat", **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        with pytest.raises(ConfigDeclarationError, match="the level-0 discriminator field 'model_type'"):
            build_layer_config(reg)

    def test_alias_named_like_the_segment_name(self, isolated):
        reg, root = _nested("nl_res_alias")

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = 0.7, **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, label: Annotated[str, Field(alias="name")] = "x", **kwargs): ...

        with pytest.raises(ConfigDeclarationError, match=r"parameter \(alias 'name'\)"):
            build_layer_config(reg)


class TestCompoundDiscriminator:
    @staticmethod
    def _two_families(name: str):
        reg, root = _nested(name)

        class _OpenAI(root):
            __registry_key__ = "openai"
            __abstract__ = True

            def __init__(self, temperature: float = 0.7, **kwargs): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        class Official(_OpenAI):
            __registry_key__ = "openai.official"

            def __init__(self, retries: int = 1, **kwargs): ...

        class _Anthropic(root):
            __registry_key__ = "anthropic"
            __abstract__ = True

            def __init__(self, max_tokens: int = 1024, **kwargs): ...

        class Direct(_Anthropic):
            __registry_key__ = "anthropic.direct"

            def __init__(self, org: str = "o", **kwargs): ...

        return reg

    @pytest.mark.parametrize(
        "payload",
        [
            {"model_type": None},
            {"model_type": 1},
            {"model_type": ["openai"]},
            {"model_type": "openai", "provider": {"name": 3}},
            {"model_type": "openai", "provider": {"name": None}},
        ],
    )
    def test_non_string_parts_report_union_tag_not_found(self, isolated, payload):
        result = build_layer_config(self._two_families("nl_m4"))
        with pytest.raises(ValidationError) as exc:
            TypeAdapter(result.union_type).validate_python(payload)
        assert [e["type"] for e in exc.value.errors()] == ["union_tag_not_found"]

    def test_one_function_per_layer(self, isolated):
        reg = self._two_families("nl_p5")
        result = build_layer_config(reg)
        function = _union_discriminator(result.union_type)
        subset = {k: result.per_key_models[k] for k in ("openai.azure", "anthropic.direct")}
        assert _union_discriminator(_layer_union(result, subset)) is function
        assert _union_discriminator(_layer_union(result, dict(result.per_key_models))) is function
        assert _union_discriminator(build_layer_config(reg).union_type) is function
        assert function.__name__ == "_discriminate_nl_p5"

    def test_an_omitted_block_selects_the_only_leaf_with_a_default(self, isolated):
        result = build_layer_config(self._two_families("nl_m2"))
        adapter = TypeAdapter(result.union_type)
        direct = result.per_key_models["anthropic.direct"]
        for payload in ({"model_type": "anthropic"}, {"model_type": "anthropic", "provider": {}}):
            value = adapter.validate_python(payload)
            assert type(value) is direct and value.provider.org == "o"
        # Two openai providers have default blocks: an omitted block is ambiguous
        # (the published oneOf matches both), so the union refuses it too.
        with pytest.raises(ValidationError) as exc:
            adapter.validate_python({"model_type": "openai"})
        assert exc.value.errors()[0]["type"] == "union_tag_invalid"
        assert _union_tags(result.union_type) == [
            "anthropic.direct", "openai.azure", "openai.official", "anthropic.default",
        ]
        # The published schema lists each model once and marks the block optional.
        schema = generate_layer_json_schema(result)
        assert len(schema["oneOf"]) == 3
        assert "provider" not in schema["$defs"][direct.__name__].get("required", [])

    def test_a_required_block_stays_required(self, isolated):
        reg, root = _nested("nl_m2_required")

        class _Anthropic(root):
            __registry_key__ = "anthropic"
            __abstract__ = True

            def __init__(self, max_tokens: int = 1024, **kwargs): ...

        class Direct(_Anthropic):
            __registry_key__ = "anthropic.direct"

            def __init__(self, org: str, **kwargs): ...

        class Azure(root):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d"): ...

        result = build_layer_config(reg)
        assert result.per_key_models["anthropic.direct"].model_fields["provider"].is_required()
        assert _union_tags(result.union_type) == ["anthropic.direct", "openai.azure", "openai.default"]
        with pytest.raises(ValidationError) as exc:
            TypeAdapter(result.union_type).validate_python({"model_type": "anthropic"})
        assert exc.value.errors()[0]["type"] == "union_tag_invalid"

    def test_a_registered_default_segment_keeps_its_tag(self, isolated):
        reg, root = _nested("nl_m2_default")

        class GoogleDefault(root):
            __registry_key__ = "google.default"

            def __init__(self, api_key: str = "k"): ...

        class Vertex(root):
            __registry_key__ = "google.vertex"

            def __init__(self, x: int = 1): ...

        result = build_layer_config(reg)
        assert _union_tags(result.union_type) == ["google.default", "google.vertex"]
        value = TypeAdapter(result.union_type).validate_python({"model_type": "google"})
        assert type(value) is result.per_key_models["google.default"]

    def test_subset_unions_compute_their_own_omitted_block_tags(self, isolated):
        result = build_layer_config(self._two_families("nl_m2_subset"))
        subset = {k: result.per_key_models[k] for k in ("openai.azure", "anthropic.direct")}
        union = _layer_union(result, subset)
        assert _union_tags(union) == ["anthropic.direct", "openai.azure", "anthropic.default", "openai.default"]
        value = TypeAdapter(union).validate_python({"model_type": "openai"})
        assert type(value) is result.per_key_models["openai.azure"]

    def test_all_three_artifacts_agree(self, isolated):
        result = build_layer_config(self._two_families("nl_m2_parity"))
        payloads = [
            {"model_type": "anthropic"},
            {"model_type": "anthropic", "provider": {}},
            {"model_type": "anthropic", "provider": {"name": "direct", "org": "x"}},
            {"model_type": "anthropic", "provider": {"name": "default"}},
            {"model_type": "openai"},
            {"model_type": "openai", "provider": {"name": "azure"}},
            {"model_type": "nope"},
            {},
        ]
        expected = layer_expectations(result)
        alias = _union_alias_name(result.layer_name)
        assert expected[alias] is result.union_type
        assert_artifacts_agree(
            generate_layer_config_source(result),
            expected,
            samples={alias: payloads},
            published=(generate_layer_json_schema(result), result.union_type),
            min_accepted=3,
        )


class TestNestedModelNames:
    def test_same_segment_under_two_families_gets_unique_names(self, isolated):
        reg, root = _nested("nl_names")

        class _Family(root):
            __abstract__ = True

            def __init__(self, temperature: float = 0.7, **kwargs): ...

        class _OpenAI(_Family):
            __registry_key__ = "openai"
            __abstract__ = True

        class _Mistral(_Family):
            __registry_key__ = "mistral"
            __abstract__ = True

        class OpenAIAzure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        class MistralAzure(_Mistral):
            __registry_key__ = "mistral.azure"

            def __init__(self, endpoint: str, **kwargs): ...

        result = build_layer_config(reg)
        names = [
            result.per_key_models[key].model_fields["provider"].annotation.__name__
            for key in ("mistral.azure", "openai.azure")
        ]
        assert names == ["AzureProviderConfig", "AzureProviderConfig2"]
        assert not any("__" in name for name in generate_layer_json_schema(result)["$defs"])

    def test_identical_segment_models_are_shared(self, isolated):
        reg, root = _nested("nl_shared")

        class Bedrock(root):
            __registry_key__ = "bedrock.default"

            def __init__(self, region: str = "us"): ...

        class Google(root):
            __registry_key__ = "google.default"

            def __init__(self, api_key: str = "k"): ...

        result = build_layer_config(reg)
        bedrock, google = (
            result.per_key_models[key].model_fields["provider"].annotation
            for key in ("bedrock.default", "google.default")
        )
        assert bedrock is google and bedrock.__name__ == "DefaultProviderConfig"

    def test_segment_names_are_identifiers(self, isolated):
        reg, root = _nested("nl_idents")

        class Gpt(root):
            __registry_key__ = "openai.gpt-4o"

        class Four(root):
            __registry_key__ = "openai.4o"

        result = build_layer_config(reg)
        names = {
            key: model.model_fields["provider"].annotation.__name__
            for key, model in result.per_key_models.items()
        }
        assert names == {"openai.gpt-4o": "Gpt_4OProviderConfig", "openai.4o": "_4OProviderConfig"}
        assert result.per_key_models["openai.gpt-4o"].__name__ == "OpenaiGpt4ONl_IdentsConfig"
        namespace: dict = {}
        exec(compile(generate_layer_config_source(result), "<nested>", "exec"), namespace)  # noqa: S102


class TestNestedRefusesMroSettings:
    """Nested levels follow the leaf constructor; scope/depth settings would be ignored."""

    @pytest.mark.parametrize("attr, value", [("__config_mro_scope__", "all"), ("__config_mro_depth__", 1)])
    def test_mro_setting_is_refused(self, attr: str, value: object) -> None:
        from conscribe.exceptions import ConfigDeclarationError
        from conscribe.testing import isolated_registries

        with isolated_registries():
            reg = create_registrar(
                "nested_mro_setting",
                LLMProtocol,
                discriminator_fields=["model_type", "provider"],
                key_separator=".",
            )

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                async def chat(self, messages: list[dict]) -> str:
                    return ""

            class Family(Base):
                __registry_key__ = "openai"
                __abstract__ = True

            leaf_ns = {"__registry_key__": "openai.direct", attr: value}
            type("Direct", (Family,), leaf_ns)
            with pytest.raises(ConfigDeclarationError, match=attr):
                build_layer_config(reg)
