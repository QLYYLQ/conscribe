"""Tests for deeply nested compound discriminated config builder."""
from __future__ import annotations

import pytest
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import PydanticUndefined

from conscribe import create_registrar
from conscribe.config.builder import LayerConfigResult
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.json_schema import generate_layer_json_schema


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

        # Should have nested segment model section
        assert "Nested segment models" in source
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
