"""Tests for conscribe.config.builder — TDD RED phase.

Tests ``build_layer_config()`` and ``LayerConfigResult``:
- Happy path: single/multi key, discriminator injection, extra policy, per_key_models
- Naming convention: key + layer name -> model class name
- Edge cases: None schema, existing discriminator field, frozen dataclass
- Error cases: missing discriminator_field
- Validation: union type validates/rejects configs via Pydantic TypeAdapter

All registrars and implementation classes are defined at MODULE LEVEL
so that ``get_type_hints()`` can resolve forward references under
``from __future__ import annotations``.
"""
from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from conscribe import create_registrar
from conscribe.config.builder import LayerConfigResult, build_layer_config


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


@runtime_checkable
class _DOMProto(Protocol):
    def filter(self, html: str) -> str: ...


@runtime_checkable
class _EvalProto(Protocol):
    def evaluate(self, result: str) -> float: ...


@runtime_checkable
class _MinimalProto(Protocol):
    def run(self) -> None: ...


# ===================================================================
# Registrars (module-level for type hint resolution)
# ===================================================================

_single_llm_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_multi_llm_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["LLM"],
)

_agent_reg = create_registrar(
    "agent", _AgentProto, discriminator_field="name",
    strip_suffixes=["Agent"],
)

_dom_reg = create_registrar(
    "dom", _DOMProto, discriminator_field="filter_type",
)

_eval_reg = create_registrar(
    "eval", _EvalProto, discriminator_field="evaluator",
)

_no_disc_reg = create_registrar(
    "broken", _MinimalProto,
    # discriminator_field defaults to "" (empty string)
)

_none_schema_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_existing_disc_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_mixed_extra_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["LLM"],
)

_validation_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["LLM"],
)


# ===================================================================
# Implementation classes for _single_llm_reg (single key)
# ===================================================================

class _SingleBase(metaclass=_single_llm_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OnlyOpenAI(_SingleBase):
    """Sole implementation for single-key tests."""
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes for _multi_llm_reg (multiple keys)
# ===================================================================

class _MultiBase(metaclass=_multi_llm_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OpenAIMultiLLM(_MultiBase):
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class AnthropicMultiLLM(_MultiBase):
    __registry_key__ = "anthropic"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _agent_reg (naming: key + layer >3 chars)
# ===================================================================

class _AgentBase(metaclass=_agent_reg.Meta):
    __abstract__ = True
    async def step(self, task: str) -> str:
        return ""
    def reset(self) -> None:
        pass


class BrowserUseAgent(_AgentBase):
    """Registered as 'browser_use'."""
    __registry_key__ = "browser_use"

    def __init__(self, *, max_steps: int = 100):
        self.max_steps = max_steps


# ===================================================================
# Implementation classes for _dom_reg (naming: layer == 3 chars -> ALL_CAPS)
# ===================================================================

class _DOMBase(metaclass=_dom_reg.Meta):
    __abstract__ = True
    def filter(self, html: str) -> str:
        return html


class SimpleDOMFilter(_DOMBase):
    __registry_key__ = "simple"

    def __init__(self, *, threshold: float = 0.5):
        self.threshold = threshold


# ===================================================================
# Implementation classes for _eval_reg (naming: layer == 4 chars -> Title)
# ===================================================================

class _EvalBase(metaclass=_eval_reg.Meta):
    __abstract__ = True
    def evaluate(self, result: str) -> float:
        return 0.0


class MyEvalEvaluator(_EvalBase):
    __registry_key__ = "my_eval"

    def __init__(self, *, strict: bool = True):
        self.strict = strict


# ===================================================================
# Implementation classes for _none_schema_reg (no __init__ params)
# ===================================================================

class _NoneSchemaBase(metaclass=_none_schema_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class NoParamLLM(_NoneSchemaBase):
    """Class with no __init__ params (schema is None)."""
    __registry_key__ = "no_param"

    # No __init__ at all -- inherits from _NoneSchemaBase which has none
    pass


# ===================================================================
# Implementation classes for _existing_disc_reg (existing discriminator field)
# ===================================================================

class _ExistingDiscBase(metaclass=_existing_disc_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ExistingDiscLLM(_ExistingDiscBase):
    """Class whose __init__ already declares the discriminator field."""
    __registry_key__ = "has_disc"

    def __init__(self, *, provider: str = "default", model_id: str):
        self.provider = provider
        self.model_id = model_id


# ===================================================================
# Implementation classes for _mixed_extra_reg (mixed open/closed)
# ===================================================================

class _MixedBase(metaclass=_mixed_extra_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ClosedMixedLLM(_MixedBase):
    """Closed schema (no **kwargs)."""
    __registry_key__ = "closed"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class OpenMixedLLM(_MixedBase):
    """Open schema (has **kwargs)."""
    __registry_key__ = "open"

    def __init__(self, *, model_id: str, **kwargs: Any):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _validation_reg (validation tests)
# ===================================================================

class _ValidBase(metaclass=_validation_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class AlphaValidLLM(_ValidBase):
    """Closed schema with required field for validation tests."""
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class BetaValidLLM(_ValidBase):
    """Open schema for validation tests."""
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str, **kwargs: Any):
        self.model_id = model_id


# ===================================================================
# Happy path tests
# ===================================================================

class TestBuildLayerConfigHappyPath:
    """Happy path tests for build_layer_config()."""

    def test_single_key_no_union_wrapper(self) -> None:
        """Single registered key -> union_type is the model itself, not wrapped in Union."""
        result = build_layer_config(_single_llm_reg)

        assert isinstance(result, LayerConfigResult)
        # Single key: union_type should be the per-key model directly
        assert result.union_type is result.per_key_models["openai"]

    def test_multiple_keys_annotated_union(self) -> None:
        """Multiple keys -> union_type is Annotated[Union[...], Field(discriminator=...)]."""
        result = build_layer_config(_multi_llm_reg)

        assert isinstance(result, LayerConfigResult)
        # With 2 keys, we should get a discriminated union (not a plain model)
        # The union_type should NOT be any single per-key model
        assert result.union_type is not result.per_key_models["openai"]
        assert result.union_type is not result.per_key_models["anthropic"]

    def test_discriminator_injected_as_literal_with_default(self) -> None:
        """Each per-key model has discriminator as Literal[key] with default=key."""
        result = build_layer_config(_multi_llm_reg)

        for key, model in result.per_key_models.items():
            field = model.model_fields[result.discriminator_field]
            # Field type should be Literal[key]
            assert field.annotation == Literal[key]  # type: ignore[comparison-overlap]
            # Field default should be the key itself
            assert field.default == key

    def test_extra_forbid_preserved(self) -> None:
        """Closed schema (no **kwargs) preserves extra='forbid' in per-key model."""
        result = build_layer_config(_single_llm_reg)

        model = result.per_key_models["openai"]
        assert model.model_config.get("extra") == "forbid"

    def test_mixed_open_closed_schemas_in_union(self) -> None:
        """Mixed open/closed schemas preserve their respective extra policies."""
        result = build_layer_config(_mixed_extra_reg)

        closed_model = result.per_key_models["closed"]
        open_model = result.per_key_models["open"]

        assert closed_model.model_config.get("extra") == "forbid"
        assert open_model.model_config.get("extra") == "allow"

    def test_per_key_models_complete(self) -> None:
        """per_key_models dict has exactly one entry per registered key."""
        result = build_layer_config(_multi_llm_reg)

        registered_keys = set(_multi_llm_reg.get_all().keys())
        assert set(result.per_key_models.keys()) == registered_keys

    def test_layer_name_matches_registrar(self) -> None:
        """layer_name in result matches the registrar's registry name."""
        result = build_layer_config(_multi_llm_reg)

        assert result.layer_name == "llm"
        assert result.layer_name == _multi_llm_reg._registry.name

    def test_discriminator_field_matches_registrar(self) -> None:
        """discriminator_field in result matches the registrar's discriminator_field."""
        result = build_layer_config(_multi_llm_reg)

        assert result.discriminator_field == "provider"
        assert result.discriminator_field == _multi_llm_reg.discriminator_field

    def test_per_key_model_has_original_fields(self) -> None:
        """Per-key model retains the original extracted fields plus discriminator."""
        result = build_layer_config(_multi_llm_reg)

        openai_model = result.per_key_models["openai"]
        assert "provider" in openai_model.model_fields  # injected
        assert "model_id" in openai_model.model_fields   # from __init__
        assert "temperature" in openai_model.model_fields  # from __init__

        anthropic_model = result.per_key_models["anthropic"]
        assert "provider" in anthropic_model.model_fields
        assert "model_id" in anthropic_model.model_fields
        assert "max_tokens" in anthropic_model.model_fields

    def test_discriminated_union_validates_correct_dict(self) -> None:
        """Union type validates a correct config dict successfully."""
        result = build_layer_config(_multi_llm_reg)

        adapter = TypeAdapter(result.union_type)
        validated = adapter.validate_python(
            {"provider": "openai", "model_id": "gpt-4o", "temperature": 0.5}
        )
        assert validated.provider == "openai"  # type: ignore[attr-defined]
        assert validated.model_id == "gpt-4o"  # type: ignore[attr-defined]


# ===================================================================
# Naming convention tests
# ===================================================================

class TestNamingConvention:
    """Tests for per-key model naming: {KeyPart}{LayerPart}Config."""

    def test_key_openai_layer_llm_produces_openai_llm_config(self) -> None:
        """key 'openai' + layer 'llm' (<=3 chars) -> 'OpenaiLLMConfig'."""
        result = build_layer_config(_single_llm_reg)

        model = result.per_key_models["openai"]
        assert model.__name__ == "OpenaiLLMConfig"

    def test_key_browser_use_layer_agent_produces_browser_use_agent_config(self) -> None:
        """key 'browser_use' + layer 'agent' (>3 chars) -> 'BrowserUseAgentConfig'."""
        result = build_layer_config(_agent_reg)

        model = result.per_key_models["browser_use"]
        assert model.__name__ == "BrowserUseAgentConfig"

    def test_key_simple_layer_dom_exactly_3_chars_all_caps(self) -> None:
        """key 'simple' + layer 'dom' (exactly 3 chars) -> 'SimpleDOMConfig'."""
        result = build_layer_config(_dom_reg)

        model = result.per_key_models["simple"]
        assert model.__name__ == "SimpleDOMConfig"

    def test_key_my_eval_layer_eval_4_chars_title_case(self) -> None:
        """key 'my_eval' + layer 'eval' (4 chars, >3) -> 'MyEvalEvalConfig'."""
        result = build_layer_config(_eval_reg)

        model = result.per_key_models["my_eval"]
        assert model.__name__ == "MyEvalEvalConfig"

    def test_snake_case_key_converts_to_title_case(self) -> None:
        """Snake-case key parts are each title-cased: 'browser_use' -> 'BrowserUse'."""
        result = build_layer_config(_agent_reg)

        model = result.per_key_models["browser_use"]
        # KeyPart should be "BrowserUse" (each segment titled)
        assert model.__name__.startswith("BrowserUse")

    def test_single_word_key_title_cased(self) -> None:
        """Single word key: 'openai' -> 'Openai'."""
        result = build_layer_config(_single_llm_reg)

        model = result.per_key_models["openai"]
        # Name should start with "Openai"
        assert model.__name__.startswith("Openai")


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases for build_layer_config()."""

    def test_none_schema_produces_model_with_only_discriminator(self) -> None:
        """Class with no __init__ params (schema=None) produces model with only discriminator."""
        result = build_layer_config(_none_schema_reg)

        model = result.per_key_models["no_param"]
        # Should have at least the discriminator field
        assert result.discriminator_field in model.model_fields
        # Should have only the discriminator field (no other params)
        non_disc_fields = [
            f for f in model.model_fields if f != result.discriminator_field
        ]
        assert len(non_disc_fields) == 0

    def test_existing_discriminator_field_overridden_with_literal(self) -> None:
        """If original schema already has the discriminator field, it is overridden with Literal[key]."""
        result = build_layer_config(_existing_disc_reg)

        model = result.per_key_models["has_disc"]
        disc_field = model.model_fields[result.discriminator_field]
        # Must be Literal["has_disc"], not the original str type
        assert disc_field.annotation == Literal["has_disc"]
        assert disc_field.default == "has_disc"

    def test_layer_config_result_is_frozen_dataclass(self) -> None:
        """LayerConfigResult should be a frozen dataclass (immutable)."""
        assert dataclasses.is_dataclass(LayerConfigResult)

        result = build_layer_config(_single_llm_reg)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.layer_name = "mutated"  # type: ignore[misc]

    def test_none_schema_model_has_extra_forbid(self) -> None:
        """Model created for None-schema class defaults to extra='forbid'."""
        result = build_layer_config(_none_schema_reg)

        model = result.per_key_models["no_param"]
        assert model.model_config.get("extra") == "forbid"


# ===================================================================
# Error cases
# ===================================================================

class TestErrorCases:
    """Error conditions that should raise exceptions."""

    def test_no_discriminator_field_raises_value_error(self) -> None:
        """Registrar with empty discriminator_field raises ValueError."""
        with pytest.raises(ValueError):
            build_layer_config(_no_disc_reg)


# ===================================================================
# Validation tests (union type + Pydantic TypeAdapter)
# ===================================================================

class TestValidation:
    """Tests that the built union type validates and rejects configs correctly."""

    def test_union_validates_correct_config(self) -> None:
        """Union type accepts valid config dict with correct discriminator."""
        result = build_layer_config(_validation_reg)
        adapter = TypeAdapter(result.union_type)

        validated = adapter.validate_python(
            {"provider": "alpha", "model_id": "gpt-4o", "temperature": 0.5}
        )
        assert validated.provider == "alpha"  # type: ignore[attr-defined]
        assert validated.model_id == "gpt-4o"  # type: ignore[attr-defined]

    def test_union_rejects_unknown_discriminator_value(self) -> None:
        """Union type rejects config with unknown discriminator value."""
        result = build_layer_config(_validation_reg)
        adapter = TypeAdapter(result.union_type)

        with pytest.raises(ValidationError):
            adapter.validate_python(
                {"provider": "nonexistent", "model_id": "gpt-4o"}
            )

    def test_closed_schema_rejects_extra_fields(self) -> None:
        """Closed schema (extra='forbid') rejects unknown fields."""
        result = build_layer_config(_validation_reg)
        adapter = TypeAdapter(result.union_type)

        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    "provider": "alpha",
                    "model_id": "gpt-4o",
                    "unknown_field": "should_fail",
                }
            )

    def test_open_schema_accepts_extra_fields(self) -> None:
        """Open schema (extra='allow') accepts unknown fields."""
        result = build_layer_config(_validation_reg)
        adapter = TypeAdapter(result.union_type)

        validated = adapter.validate_python(
            {
                "provider": "beta",
                "model_id": "gpt-4o",
                "unknown_field": "should_pass",
            }
        )
        assert validated.model_id == "gpt-4o"  # type: ignore[attr-defined]

    def test_wrong_type_raises_validation_error(self) -> None:
        """Wrong type for a field raises ValidationError."""
        result = build_layer_config(_validation_reg)
        adapter = TypeAdapter(result.union_type)

        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    "provider": "alpha",
                    "model_id": 12345,  # should be str
                    "temperature": "not_a_float",
                }
            )

    def test_missing_required_field_raises_validation_error(self) -> None:
        """Missing required field raises ValidationError."""
        result = build_layer_config(_validation_reg)
        adapter = TypeAdapter(result.union_type)

        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    "provider": "alpha",
                    # missing model_id which is required
                    "temperature": 0.5,
                }
            )


# ===================================================================
# Degraded fields propagation tests
# ===================================================================

class _IncompatibleType:
    """A type Pydantic cannot serialize."""

    def __init__(self, x: int):
        self.x = x


_degraded_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _DegradedBase(metaclass=_degraded_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DegradedProvider(_DegradedBase):
    """Implementation with an incompatible param type."""
    __registry_key__ = "degraded"

    def __init__(self, *, model_id: str, auth: _IncompatibleType, count: int = 0):
        self.model_id = model_id
        self.auth = auth
        self.count = count


class TestDegradedFieldsPropagation:
    """Tests that degraded fields propagate through build_layer_config."""

    def test_degraded_fields_in_result(self) -> None:
        """LayerConfigResult.degraded_fields contains degraded field info."""
        result = build_layer_config(_degraded_reg)
        assert "degraded" in result.degraded_fields
        field_names = [df.field_name for df in result.degraded_fields["degraded"]]
        assert "auth" in field_names

    def test_no_degraded_fields_default_empty(self) -> None:
        """No degradation produces empty degraded_fields dict."""
        result = build_layer_config(_single_llm_reg)
        assert result.degraded_fields == {}

    def test_backward_compat(self) -> None:
        """Old code constructing LayerConfigResult without degraded_fields works."""
        # Should not raise — degraded_fields has a default
        r = LayerConfigResult(
            union_type=str,
            per_key_models={},
            layer_name="test",
            discriminator_field="type",
        )
        assert r.degraded_fields == {}
