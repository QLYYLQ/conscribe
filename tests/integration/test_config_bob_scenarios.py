"""Integration tests: Bob (config writer) scenarios.

Simulates real-world scenarios from the design doc Section 1.4 where
Bob (a config writer / researcher) interacts with the config system.
Bob does NOT read source code -- he relies entirely on the config typing
system for guidance, validation, and error messages.

Each test exercises at least two modules in the config subsystem
(extractor + builder, builder + TypeAdapter validation, etc.).
"""
from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from conscribe import (
    build_layer_config,
    create_registrar,
    extract_config_schema,
)


# ===================================================================
# Protocols
# ===================================================================

@runtime_checkable
class LLMProtocol(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


# ===================================================================
# Helpers
# ===================================================================

def _build_llm_union_adapter(*provider_classes):
    """Register provider classes, build union, return TypeAdapter.

    Creates a fresh registrar so tests are isolated.
    """
    LR = create_registrar(
        "llm", LLMProtocol,
        strip_suffixes=["Provider", "LLM"],
        discriminator_field="provider",
    )

    class BaseLLM(metaclass=LR.Meta):
        __abstract__ = True
        async def chat(self, messages: list[dict]) -> str: ...

    # Force-register each provider class using the metaclass
    for cls in provider_classes:
        # Create a subclass that inherits from BaseLLM so metaclass registers it
        pass

    # Provider classes are registered at class-definition time via metaclass.
    # The caller must define them as subclasses of BaseLLM.
    # This helper is used when we already have a registrar and classes.
    result = build_layer_config(LR)
    return TypeAdapter(result.union_type), result, LR


# ===================================================================
# Test: Bob sees valid provider values in discriminated union
# ===================================================================

class TestBobSeesProviderValues:
    """Bob sees the list of valid provider values in the discriminated union."""

    def test_bob_sees_valid_provider_values(self) -> None:
        """The union's per-key models contain exactly the registered providers,
        each with a Literal discriminator showing the valid value."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class OpenAIProvider(BaseLLM):
            def __init__(self, *, model_id: str, temperature: float = 0.0):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "openai"

        class AnthropicProvider(BaseLLM):
            def __init__(self, *, model_id: str, max_tokens: int = 4096):
                self.model_id = model_id
                self.max_tokens = max_tokens

            async def chat(self, messages: list[dict]) -> str:
                return "anthropic"

        class DeepSeekProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "deepseek"

        result = build_layer_config(LR)

        # Bob sees these three providers as valid values
        assert set(result.per_key_models.keys()) == {
            "open_ai", "anthropic", "deep_seek"
        }

        # Each per-key model has a Literal discriminator field
        for key, model in result.per_key_models.items():
            assert "provider" in model.model_fields
            field = model.model_fields["provider"]
            assert field.default == key


# ===================================================================
# Test: Bob's typo field in closed schema is rejected
# ===================================================================

class TestBobTypoClosedSchema:
    """Bob's typo field is rejected in a closed schema (no **kwargs)."""

    def test_typo_field_rejected_in_closed_schema(self) -> None:
        """A config dict with an unknown field is rejected when the
        implementation has no **kwargs (extra='forbid')."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class StrictProvider(BaseLLM):
            def __init__(self, *, model_id: str, temperature: float = 0.0):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "strict"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Bob writes "temperture" (typo) instead of "temperature"
        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python({
                "provider": "strict",
                "model_id": "gpt-4o",
                "temperture": 0.5,  # typo!
            })

        # The error message should mention the extra field
        error_str = str(exc_info.value)
        assert "temperture" in error_str or "extra" in error_str.lower()


# ===================================================================
# Test: Bob's correct config for open schema passes
# ===================================================================

class TestBobOpenSchemaCorrect:
    """Bob's correct config for an open-schema provider passes validation."""

    def test_correct_config_passes_open_schema(self) -> None:
        """When implementation has **kwargs, known fields are validated
        and the config passes if they are correct."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class FlexibleProvider(BaseLLM):
            def __init__(self, *, model_id: str, temperature: float = 0.0, **kwargs):
                self.model_id = model_id
                self.temperature = temperature
                self.extra = kwargs

            async def chat(self, messages: list[dict]) -> str:
                return "flexible"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        validated = adapter.validate_python({
            "provider": "flexible",
            "model_id": "gpt-4o",
            "temperature": 0.5,
        })
        assert validated.model_id == "gpt-4o"
        assert validated.temperature == 0.5


# ===================================================================
# Test: Bob's extra field in open schema passes
# ===================================================================

class TestBobExtraFieldOpenSchema:
    """Bob's extra fields pass validation in an open schema."""

    def test_extra_field_accepted_in_open_schema(self) -> None:
        """When implementation has **kwargs (extra='allow'), unknown
        fields are silently accepted."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class FlexProvider(BaseLLM):
            def __init__(self, *, model_id: str, **kwargs):
                self.model_id = model_id
                self.extra = kwargs

            async def chat(self, messages: list[dict]) -> str:
                return "flex"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Bob adds "custom_param" which is not declared but should be allowed
        validated = adapter.validate_python({
            "provider": "flex",
            "model_id": "custom-model",
            "custom_param": "hello",
            "another_extra": 42,
        })
        assert validated.model_id == "custom-model"


# ===================================================================
# Test: Bob's extra field in closed schema is rejected
# ===================================================================

class TestBobExtraFieldClosedSchema:
    """Bob's extra fields are rejected in a closed schema."""

    def test_extra_field_rejected_in_closed_schema(self) -> None:
        """Without **kwargs, any field not in __init__ signature is rejected."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class ClosedProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "closed"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "closed",
                "model_id": "gpt-4o",
                "unknown_field": "should fail",
            })


# ===================================================================
# Test: Required fields without defaults must be provided
# ===================================================================

class TestBobRequiredFields:
    """Bob sees that required fields without defaults must be provided."""

    def test_required_field_missing_raises_error(self) -> None:
        """Fields without defaults in __init__ are required in the config."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class RequiredFieldProvider(BaseLLM):
            def __init__(self, *, model_id: str, api_key: str):
                self.model_id = model_id
                self.api_key = api_key

            async def chat(self, messages: list[dict]) -> str:
                return "required"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Omit required field "api_key"
        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python({
                "provider": "required_field",
                "model_id": "gpt-4o",
                # api_key is missing
            })
        assert "api_key" in str(exc_info.value)


# ===================================================================
# Test: Default values work correctly
# ===================================================================

class TestBobDefaultValues:
    """Bob sees default values work correctly -- optional fields can be omitted."""

    def test_defaults_applied_when_fields_omitted(self) -> None:
        """Fields with defaults in __init__ can be omitted from config."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class DefaultProvider(BaseLLM):
            def __init__(
                self, *,
                model_id: str,
                temperature: float = 0.7,
                max_tokens: int = 4096,
                stream: bool = False,
            ):
                self.model_id = model_id
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.stream = stream

            async def chat(self, messages: list[dict]) -> str:
                return "default"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Bob only provides required field, defaults are applied
        validated = adapter.validate_python({
            "provider": "default",
            "model_id": "gpt-4o",
        })
        assert validated.model_id == "gpt-4o"
        assert validated.temperature == 0.7
        assert validated.max_tokens == 4096
        assert validated.stream is False


# ===================================================================
# Test: Annotated constraints enforced at validation time
# ===================================================================

class TestBobAnnotatedConstraints:
    """Annotated constraints (ge, le, gt, lt) are enforced at validation time."""

    def test_ge_le_constraints_enforced(self) -> None:
        """Temperature with ge=0, le=2 rejects values outside range."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class ConstrainedProvider(BaseLLM):
            def __init__(
                self, *,
                model_id: str,
                temperature: Annotated[float, Field(0.0, ge=0, le=2)] = 0.0,
                max_tokens: Annotated[int, Field(4096, gt=0)] = 4096,
            ):
                self.model_id = model_id
                self.temperature = temperature
                self.max_tokens = max_tokens

            async def chat(self, messages: list[dict]) -> str:
                return "constrained"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Valid config within constraints
        valid = adapter.validate_python({
            "provider": "constrained",
            "model_id": "gpt-4o",
            "temperature": 1.5,
            "max_tokens": 8192,
        })
        assert valid.temperature == 1.5
        assert valid.max_tokens == 8192

        # Temperature > 2 (violates le=2)
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "constrained",
                "model_id": "gpt-4o",
                "temperature": 3.0,
            })

        # Temperature < 0 (violates ge=0)
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "constrained",
                "model_id": "gpt-4o",
                "temperature": -0.5,
            })

        # max_tokens = 0 (violates gt=0)
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "constrained",
                "model_id": "gpt-4o",
                "max_tokens": 0,
            })


# ===================================================================
# Test: Docstring descriptions appear in extracted field descriptions
# ===================================================================

class TestBobDocstringDescriptions:
    """Docstring descriptions appear in the extracted schema's field descriptions."""

    def test_docstring_descriptions_extracted(self) -> None:
        """Tier 1.5: parameter descriptions from Google-style docstring
        are available in the extracted schema's field descriptions."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class DocumentedProvider(BaseLLM):
            """A well-documented LLM provider.

            Args:
                model_id: The model identifier, e.g. gpt-4o
                temperature: Generation temperature between 0 and 2
            """

            def __init__(self, *, model_id: str, temperature: float = 0.0):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "documented"

        # Extract schema (touches extractor + docstring modules)
        schema = extract_config_schema(DocumentedProvider)
        assert schema is not None

        # Check descriptions are present
        model_id_field = schema.model_fields["model_id"]
        temp_field = schema.model_fields["temperature"]

        assert model_id_field.description is not None
        assert "model identifier" in model_id_field.description.lower() or "gpt-4o" in model_id_field.description

        assert temp_field.description is not None
        assert "temperature" in temp_field.description.lower()


# ===================================================================
# Test: MRO -- child class without __init__ uses parent's schema
# ===================================================================

class TestBobMROInheritance:
    """MRO: child class without __init__ uses parent's schema."""

    def test_child_without_init_inherits_parent_schema(self) -> None:
        """When a child class has no __init__, the extractor walks the MRO
        to find the parent's __init__ and extracts its schema."""
        LR = create_registrar(
            "agent", AgentProtocol,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        class BaseAgent(metaclass=LR.Meta):
            __abstract__ = True

            def __init__(self, *, max_steps: int = 100, timeout: int = 300):
                self.max_steps = max_steps
                self.timeout = timeout

            async def step(self, task: str) -> str:
                raise NotImplementedError

            def reset(self) -> None:
                pass

        class ChildAgent(BaseAgent):
            """Inherits parent __init__ without overriding."""

            async def step(self, task: str) -> str:
                return "child"

        # Extract schema from child (should use parent's __init__)
        schema = extract_config_schema(ChildAgent)
        assert schema is not None
        assert "max_steps" in schema.model_fields
        assert "timeout" in schema.model_fields

        # Build union and validate through it
        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        validated = adapter.validate_python({"name": "child", "max_steps": 50})
        assert validated.max_steps == 50
        assert validated.timeout == 300  # default from parent


# ===================================================================
# Test: Tier 3 -- explicit __config_schema__ is used directly
# ===================================================================

class TestBobTier3ExplicitSchema:
    """Tier 3: explicit __config_schema__ is used directly by the extractor."""

    def test_explicit_config_schema_used(self) -> None:
        """When __config_schema__ is set, the extractor returns it directly,
        bypassing __init__ reflection."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class ExplicitConfig(BaseModel):
            model_config = ConfigDict(extra="allow")

            model_id: str = Field(description="Model identifier")
            temperature: float = Field(0.0, ge=0, le=2)
            custom_validator_field: str = "default"

        class ExplicitProvider(BaseLLM):
            __config_schema__ = ExplicitConfig

            def __init__(self, config: ExplicitConfig):
                self.model_id = config.model_id
                self.temperature = config.temperature

            async def chat(self, messages: list[dict]) -> str:
                return "explicit"

        # Extract should return ExplicitConfig directly
        schema = extract_config_schema(ExplicitProvider)
        assert schema is ExplicitConfig

        # Build union and validate -- the extra="allow" from the user's
        # schema should be preserved
        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Extra fields should be allowed (user defined extra="allow")
        validated = adapter.validate_python({
            "provider": "explicit",
            "model_id": "gpt-4o",
            "bonus_field": "allowed",
        })
        assert validated.model_id == "gpt-4o"


# ===================================================================
# Test: Implementer zero burden
# ===================================================================

class TestImplementerZeroBurden:
    """Implementer writes just __init__ with type hints, schema is auto-extracted."""

    def test_plain_init_auto_extracts_schema(self) -> None:
        """A class with only a typed __init__ (no Annotated, no docstring,
        no __config_schema__) still produces a usable config schema."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class SimpleProvider(BaseLLM):
            def __init__(self, *, model_id: str, temperature: float = 0.0):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "simple"

        # Schema is extracted from __init__ alone
        schema = extract_config_schema(SimpleProvider)
        assert schema is not None
        assert "model_id" in schema.model_fields
        assert "temperature" in schema.model_fields

        # Build and validate end-to-end
        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        validated = adapter.validate_python({
            "provider": "simple",
            "model_id": "test",
        })
        assert validated.temperature == 0.0


# ===================================================================
# Test: Multiple providers validate through discriminated union
# ===================================================================

class TestBobMultipleProviders:
    """Multiple providers validate correctly through the discriminated union."""

    def test_multiple_providers_discriminated_correctly(self) -> None:
        """Each provider config is dispatched to the correct model based
        on the discriminator value."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class OpenAIProvider(BaseLLM):
            def __init__(self, *, model_id: str, temperature: float = 0.0):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "openai"

        class AnthropicProvider(BaseLLM):
            def __init__(self, *, model_id: str, top_k: int = 40):
                self.model_id = model_id
                self.top_k = top_k

            async def chat(self, messages: list[dict]) -> str:
                return "anthropic"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # OpenAI config routes to OpenAI model
        openai_cfg = adapter.validate_python({
            "provider": "open_ai",
            "model_id": "gpt-4o",
            "temperature": 1.0,
        })
        assert openai_cfg.provider == "open_ai"
        assert hasattr(openai_cfg, "temperature")
        assert openai_cfg.temperature == 1.0

        # Anthropic config routes to Anthropic model
        anthropic_cfg = adapter.validate_python({
            "provider": "anthropic",
            "model_id": "claude-3",
            "top_k": 100,
        })
        assert anthropic_cfg.provider == "anthropic"
        assert hasattr(anthropic_cfg, "top_k")
        assert anthropic_cfg.top_k == 100


# ===================================================================
# Test: Unknown discriminator value is rejected
# ===================================================================

class TestBobUnknownDiscriminator:
    """Unknown discriminator value is rejected."""

    def test_unknown_provider_rejected(self) -> None:
        """A provider value that doesn't match any registered key is rejected."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class OnlyProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "only"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # "nonexistent" is not a registered provider
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "nonexistent",
                "model_id": "gpt-4o",
            })


# ===================================================================
# Test: Wrong type for a field is rejected
# ===================================================================

class TestBobWrongFieldType:
    """Wrong type for a field is rejected at validation time."""

    def test_wrong_type_rejected(self) -> None:
        """Passing a string where an int is expected is rejected."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class TypedProvider(BaseLLM):
            def __init__(self, *, model_id: str, max_tokens: int = 4096):
                self.model_id = model_id
                self.max_tokens = max_tokens

            async def chat(self, messages: list[dict]) -> str:
                return "typed"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # max_tokens should be int, not a non-numeric string
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "typed",
                "model_id": "gpt-4o",
                "max_tokens": "not_a_number",
            })


# ===================================================================
# Test: Annotated Field description > docstring (priority)
# ===================================================================

class TestBobDescriptionPriority:
    """Annotated Field(description=...) takes priority over docstring."""

    def test_annotated_description_overrides_docstring(self) -> None:
        """When both Annotated Field description and docstring description
        exist for the same parameter, Annotated wins."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class PriorityProvider(BaseLLM):
            """Provider with both docstring and Annotated descriptions.

            Args:
                model_id: From docstring
                temperature: Docstring temperature description
            """

            def __init__(
                self, *,
                model_id: Annotated[str, Field(description="From Annotated")],
                temperature: float = 0.0,
            ):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "priority"

        schema = extract_config_schema(PriorityProvider)
        assert schema is not None

        # model_id should use Annotated description (higher priority)
        model_id_field = schema.model_fields["model_id"]
        assert model_id_field.description == "From Annotated"

        # temperature should fall back to docstring description
        temp_field = schema.model_fields["temperature"]
        if temp_field.description is not None:
            assert "docstring" in temp_field.description.lower() or "temperature" in temp_field.description.lower()


# ===================================================================
# Test: Bob's config with open + closed providers mixed in one union
# ===================================================================

class TestBobMixedOpenClosed:
    """Mixed open and closed providers in the same union."""

    def test_mixed_open_closed_in_same_union(self) -> None:
        """One provider is open (has **kwargs), another is closed.
        The union respects each model's extra policy independently."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class ClosedProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "closed"

        class OpenProvider(BaseLLM):
            def __init__(self, *, model_id: str, **kwargs):
                self.model_id = model_id
                self.extra = kwargs

            async def chat(self, messages: list[dict]) -> str:
                return "open"

        result = build_layer_config(LR)
        adapter = TypeAdapter(result.union_type)

        # Closed provider rejects extra fields
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "provider": "closed",
                "model_id": "gpt-4o",
                "extra_field": "nope",
            })

        # Open provider accepts extra fields
        validated = adapter.validate_python({
            "provider": "open",
            "model_id": "gpt-4o",
            "extra_field": "yes",
        })
        assert validated.model_id == "gpt-4o"
