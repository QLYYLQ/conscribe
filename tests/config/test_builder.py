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
import importlib
import inspect
import logging
import sys
import textwrap
import warnings
from typing import Annotated, Any, Literal, Optional, Protocol, runtime_checkable

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from conscribe import create_registrar, discover
from conscribe.discover import clear_import_failures
from conscribe.config.builder import (
    GENERATED_MARKER,
    LayerConfigResult,
    _build_validated_layer,
    _ModelNames,
    build_layer_config,
)
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.composed import build_composed_config
from conscribe.config.json_schema import generate_layer_json_schema
from conscribe.exceptions import (
    ConfigDeclarationError,
    DegradationWarning,
    EmptyLayerError,
    RegistryError,
    WiringResolutionError,
)


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

    def test_existing_discriminator_field_is_refused(self) -> None:
        """A parameter named like the discriminator field used to be silently
        replaced by ``Literal[key]``; 2.0 refuses it (M5)."""
        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(_existing_disc_reg)
        assert exc.value.owner.endswith("ExistingDiscLLM")
        assert exc.value.field == "provider"
        msg = str(exc.value)
        assert "collides with the discriminator field 'provider' of layer 'llm'" in msg
        assert "Literal['has_disc']" in msg and "Rename the field" in msg

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
        with pytest.warns(DegradationWarning):
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


# ===================================================================
# K — per-key models derive from their source model
# ===================================================================


class RecipeConfig(BaseModel):
    """The Tier-3 example from docs/recipes.md and docs/guide-bob.md."""

    model_id: str
    temperature: float = Field(0.0, ge=0, le=2, description="Sampling temperature")

    @model_validator(mode="after")
    def validate_combo(self) -> "RecipeConfig":
        if self.model_id.startswith("o1") and self.temperature != 1.0:
            raise ValueError("o1 models only support temperature=1.0")
        return self

    @field_validator("model_id")
    @classmethod
    def _no_blank_model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id must not be blank")
        return v


class OpenTier3Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        title="Open Config",
        json_schema_extra={"x-tier": 3},
    )

    name: str


class IgnoreTier3Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: int = 0


_tier3_reg = create_registrar("k_tier3", _LLMProto, discriminator_field="provider")


class _Tier3Base(metaclass=_tier3_reg.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class RecipeLLM(_Tier3Base):
    __registry_key__ = "recipe"
    __config_schema__ = RecipeConfig


class OpenTier3LLM(_Tier3Base):
    __registry_key__ = "open"
    __config_schema__ = OpenTier3Config


class IgnoreTier3LLM(_Tier3Base):
    __registry_key__ = "ignore"
    __config_schema__ = IgnoreTier3Config


class KnobConfig(BaseModel):
    """A user BaseModel knob class passed as the single constructor parameter."""

    depth: int = 1
    width: int = 1

    @field_validator("depth")
    @classmethod
    def _positive_depth(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("depth must be positive")
        return v

    @model_validator(mode="after")
    def _wide_enough(self) -> "KnobConfig":
        if self.width < self.depth:
            raise ValueError("width must be >= depth")
        return self


_knob_reg = create_registrar("k_knob", _LLMProto, discriminator_field="provider")


class _KnobBase(metaclass=_knob_reg.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class KnobLLM(_KnobBase):
    __registry_key__ = "knob"
    # 2.0 opt-in for the single-BaseModel-parameter flattening (D6); 1.x did
    # it implicitly.  Either way the per-key model derives from KnobConfig.
    __config_from_param__ = "config"

    def __init__(self, config: KnobConfig) -> None:
        self.config = config


_literal_wired_reg = create_registrar("k_literal_wired", _AgentProto, discriminator_field="name")


class _LiteralWiredBase(metaclass=_literal_wired_reg.Meta):
    __abstract__ = True

    async def step(self, task: str) -> str:
        return ""

    def reset(self) -> None:
        pass


class ChromiumUser(_LiteralWiredBase):
    __registry_key__ = "chromium_user"
    __wiring__ = {"browser": ["chromium", "firefox"]}

    def __init__(self, *, browser: str = "chromium") -> None:
        self.browser = browser


class TestRebuildByInheritance:
    """Per-key models subclass their source model (K)."""

    def test_tier3_model_validator_fires(self) -> None:
        """The documented ``model_validator`` example now runs on the layer model."""
        adapter = TypeAdapter(build_layer_config(_tier3_reg).union_type)
        ok = adapter.validate_python(
            {"provider": "recipe", "model_id": "o1-mini", "temperature": 1.0}
        )
        assert ok.temperature == 1.0
        with pytest.raises(ValidationError, match="o1 models only support temperature=1.0"):
            adapter.validate_python(
                {"provider": "recipe", "model_id": "o1-mini", "temperature": 0.5}
            )

    def test_tier3_field_validator_fires(self) -> None:
        adapter = TypeAdapter(build_layer_config(_tier3_reg).union_type)
        with pytest.raises(ValidationError, match="model_id must not be blank"):
            adapter.validate_python({"provider": "recipe", "model_id": "  "})

    def test_tier3_field_info_survives(self) -> None:
        model = build_layer_config(_tier3_reg).per_key_models["recipe"]
        field = model.model_fields["temperature"]
        assert field.description == "Sampling temperature"
        with pytest.raises(ValidationError):
            model(model_id="gpt", temperature=3.0)

    def test_tier3_without_extra_stays_forbid(self) -> None:
        """A schema that does not set ``extra`` must not drift to pydantic's 'ignore'."""
        assert RecipeConfig.model_config.get("extra") is None
        model = build_layer_config(_tier3_reg).per_key_models["recipe"]
        assert model.model_config["extra"] == "forbid"
        with pytest.raises(ValidationError) as exc:
            model(model_id="gpt", typo_knob=1)
        assert exc.value.errors()[0]["type"] == "extra_forbidden"

    def test_tier3_explicit_extra_is_kept(self) -> None:
        result = build_layer_config(_tier3_reg)
        assert result.per_key_models["open"].model_config["extra"] == "allow"
        assert result.per_key_models["ignore"].model_config["extra"] == "ignore"

    def test_tier3_model_config_survives(self) -> None:
        model = build_layer_config(_tier3_reg).per_key_models["open"]
        assert model(name="  padded  ").name == "padded"  # str_strip_whitespace
        schema = model.model_json_schema()
        assert schema["title"] == "Open Config"
        assert schema["x-tier"] == 3

    def test_per_key_model_subclasses_schema_which_stays_untouched(self) -> None:
        model = build_layer_config(_tier3_reg).per_key_models["recipe"]
        assert issubclass(model, RecipeConfig)
        assert model is not RecipeConfig
        assert "provider" not in RecipeConfig.model_fields
        assert GENERATED_MARKER not in RecipeConfig.__dict__
        assert list(model.model_fields) == ["model_id", "temperature", "provider"]

    def test_basemodel_knob_class_validators_survive(self) -> None:
        model = build_layer_config(_knob_reg).per_key_models["knob"]
        assert issubclass(model, KnobConfig)
        assert model(depth=2, width=3).depth == 2
        with pytest.raises(ValidationError, match="depth must be positive"):
            model(depth=0, width=3)
        with pytest.raises(ValidationError, match="width must be >= depth"):
            model(depth=3, width=1)

    def test_discriminator_appended_after_extracted_fields(self) -> None:
        model = build_layer_config(_multi_llm_reg).per_key_models["openai"]
        assert list(model.model_fields) == ["model_id", "temperature", "provider"]

    def test_custom_attributes_reattached(self) -> None:
        with pytest.warns(DegradationWarning):
            model = build_layer_config(_degraded_reg).per_key_models["degraded"]
        assert [df.field_name for df in model.__dict__["__degraded_fields__"]] == ["auth"]

    def test_wired_fields_reattached(self) -> None:
        model = build_layer_config(_literal_wired_reg).per_key_models["chromium_user"]
        assert model.__dict__["__wired_fields__"] == {"browser": "literal"}


class TestNeverDeriveFromRegisteredClasses:
    """A config model must never subclass an implementation class."""

    def test_registered_basemodel_class_is_not_subclassed(self) -> None:
        reg = create_registrar("k_bm_impl", _LLMProto, discriminator_field="provider")
        Bridge = reg.bridge(BaseModel)

        class ToolLLM(Bridge):  # type: ignore[misc, valid-type]
            __registry_key__ = "tool"
            depth: int = Field(1, description="search depth")

            async def chat(self, messages: list[dict]) -> str:
                return ""

        before = sorted(reg.keys())
        model = build_layer_config(reg).per_key_models["tool"]
        assert sorted(reg.keys()) == before == ["tool"]
        assert not issubclass(model, ToolLLM)
        assert model.model_fields["depth"].description == "search depth"

    def test_registered_class_as_config_schema_is_refused(self) -> None:
        impl_reg = create_registrar("k_schema_impl", _LLMProto, discriminator_field="provider")
        Bridge = impl_reg.bridge(BaseModel)

        class Registered(Bridge):  # type: ignore[misc, valid-type]
            __registry_key__ = "registered"
            depth: int = 1

            async def chat(self, messages: list[dict]) -> str:
                return ""

        owner_reg = create_registrar("k_schema_owner", _LLMProto, discriminator_field="provider")

        class _OwnerBase(metaclass=owner_reg.Meta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class UsesRegistered(_OwnerBase):
            __registry_key__ = "uses"
            __config_schema__ = Registered

        with pytest.raises(ConfigDeclarationError, match="registered class") as exc:
            build_layer_config(owner_reg)
        assert "UsesRegistered" in str(exc.value)
        assert sorted(impl_reg.keys()) == ["registered"]

    def test_root_model_config_schema_is_refused(self) -> None:
        reg = create_registrar("k_root_schema", _LLMProto, discriminator_field="provider")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class Rooted(_Base):
            __registry_key__ = "rooted"
            __config_schema__ = RootModel[int]

        with pytest.raises(ConfigDeclarationError, match="RootModel") as exc:
            build_layer_config(reg)
        assert "Rooted" in str(exc.value)


# ===================================================================
# Generated-model marker
# ===================================================================


class TestGeneratedMarker:
    """Every model the builder creates carries ``__conscribe_generated__``."""

    @pytest.mark.parametrize(
        "registrar, key",
        [
            (_multi_llm_reg, "openai"),        # extracted schema
            (_none_schema_reg, "no_param"),    # discriminator-only
            (_tier3_reg, "recipe"),            # Tier-3 schema
            (_knob_reg, "knob"),               # single BaseModel parameter
        ],
    )
    def test_marker_in_own_dict(self, registrar: Any, key: str) -> None:
        model = build_layer_config(registrar).per_key_models[key]
        assert model.__dict__.get(GENERATED_MARKER) is True

    def test_marker_is_invisible_to_pydantic(self) -> None:
        result = build_layer_config(_multi_llm_reg)
        model = result.per_key_models["openai"]
        assert GENERATED_MARKER not in model.model_fields
        instance = model(model_id="gpt-4o")
        assert GENERATED_MARKER not in instance.model_dump()
        schema = TypeAdapter(result.union_type).json_schema()
        assert GENERATED_MARKER not in str(schema)

    def test_user_classes_are_not_marked(self) -> None:
        build_layer_config(_tier3_reg)
        build_layer_config(_knob_reg)
        for user_model in (RecipeConfig, OpenTier3Config, KnobConfig):
            assert GENERATED_MARKER not in user_model.__dict__


# ===================================================================
# Empty layers
# ===================================================================


class TestEmptyLayer:
    def test_empty_flat_layer_raises(self) -> None:
        reg = create_registrar("k_empty_flat", _LLMProto, discriminator_field="provider")
        with pytest.raises(EmptyLayerError) as exc:
            build_layer_config(reg)
        assert exc.value.layer_names == ["k_empty_flat"]
        assert "'k_empty_flat'" in str(exc.value)
        assert isinstance(exc.value, RegistryError)

    def test_empty_nested_layer_raises(self) -> None:
        reg = create_registrar(
            "k_empty_nested", _LLMProto,
            discriminator_fields=["model_type", "provider"], key_separator=".",
        )
        with pytest.raises(EmptyLayerError, match="'k_empty_nested'"):
            build_layer_config(reg)

    def test_registrar_build_config_raises_too(self) -> None:
        reg = create_registrar("k_empty_via_registrar", _LLMProto, discriminator_field="provider")
        with pytest.raises(EmptyLayerError):
            reg.build_config()

    def test_abstract_only_layer_is_empty(self) -> None:
        reg = create_registrar("k_abstract_only", _LLMProto, discriminator_field="provider")

        class _OnlyAbstract(metaclass=reg.Meta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        with pytest.raises(EmptyLayerError):
            build_layer_config(reg)


# ===================================================================
# Validate first (F1): every wiring problem of the layer, before building
# ===================================================================


class TestValidateFirst:
    @pytest.fixture(autouse=True)
    def _isolated(self, conscribe_isolated: Any) -> Any:
        yield conscribe_isolated

    @staticmethod
    def _layer(name: str) -> tuple[Any, type]:
        reg = create_registrar(name, _MinimalProto, discriminator_field="kind")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        return reg, _Base

    def test_all_problems_of_all_classes_in_one_error(self) -> None:
        target, target_base = self._layer("k_vf_target")

        class Real(target_base):
            __registry_key__ = "real"

        owner, owner_base = self._layer("k_vf_owner")

        class TypoKey(owner_base):
            __registry_key__ = "typo_key"
            __wiring__ = {"dep": ("k_vf_target", ["reel"])}

            def __init__(self, dep: str): ...

        class TypoRegistry(owner_base):
            __registry_key__ = "typo_registry"
            __wiring__ = {"dep": "k_vf_targte", "other": {"registry": "k_vf_target", "default": "nope"}}
            other: object  # a default needs a receptor slot

        # Building this one fails (a RootModel schema): validation must come first.
        class Rooted(owner_base):
            __registry_key__ = "rooted"
            __config_schema__ = RootModel[int]

        with pytest.raises(WiringResolutionError) as exc:
            build_layer_config(owner)
        problems = [(p.owner_key, p.param_name, p.kind) for p in exc.value.problems]
        assert problems == [
            ("typo_key", "dep", "unknown_key"),
            ("typo_registry", "dep", "unknown_registry"),
            ("typo_registry", "other", "bad_default"),
        ]
        assert "did you mean 'k_vf_target'?" in str(exc.value)

    def test_problems_used_to_be_swallowed_by_extraction(self) -> None:
        """1.x dropped every wired field of a class with one bad entry."""
        target, target_base = self._layer("k_vf_target2")

        class Real(target_base):
            __registry_key__ = "real"

        owner, owner_base = self._layer("k_vf_owner2")

        class OneBadEntry(owner_base):
            __registry_key__ = "one_bad"
            __wiring__ = {"good": "k_vf_target2", "bad": ("k_vf_target2", ["missing"])}

            def __init__(self, good: str, bad: str): ...

        with pytest.raises(WiringResolutionError) as exc:
            build_layer_config(owner)
        assert [p.param_name for p in exc.value.problems] == ["bad"]

    def test_empty_layer_is_reported_before_wiring(self) -> None:
        empty, _ = self._layer("k_vf_empty")
        with pytest.raises(EmptyLayerError):
            build_layer_config(empty)


# ===================================================================
# 2.0 review fixes
# ===================================================================


@pytest.fixture
def isolated(conscribe_isolated: Any) -> Any:
    yield conscribe_isolated


def _layer(name: str, disc: str = "kind") -> tuple[Any, type]:
    """A flat layer and its abstract base."""
    reg = create_registrar(name, _MinimalProto, discriminator_field=disc)

    class _Base(metaclass=reg.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    return reg, _Base


def _with_plan(base: type) -> None:
    """A second, plain implementation (so the layer builds a discriminated union)."""

    class Plan(base):  # type: ignore[misc, valid-type]
        __registry_key__ = "plan"

        def __init__(self, depth: int = 1): ...


class CamelSchema(BaseModel):
    """A Tier-3 schema whose fields are written in camelCase."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    max_steps: int = 5


class TestDiscriminatorAlias:
    """M1: an alias_generator must not rename the discriminator field."""

    def test_tier3_schema(self, isolated: Any) -> None:
        reg, base = _layer("k_alias_t3", disc="loop_id")

        class React(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "react"
            __config_schema__ = CamelSchema

        _with_plan(base)
        result = build_layer_config(reg)
        model = result.per_key_models["react"]
        assert model.model_fields["loop_id"].alias == "loop_id"
        adapter = TypeAdapter(result.union_type)
        assert adapter.validate_python({"loop_id": "react", "maxSteps": 2}).max_steps == 2
        assert adapter.validate_python({"loop_id": "plan"}).depth == 1
        with pytest.raises(ValidationError):
            adapter.validate_python({"loopId": "react"})
        assert model().model_dump(by_alias=True) == {"maxSteps": 5, "loop_id": "react"}
        schema = generate_layer_json_schema(result)
        assert "loop_id" in schema["$defs"][model.__name__]["properties"]
        composed = build_composed_config({"k_alias_t3": reg})
        assert composed.top_level_type.model_validate(
            {"k_alias_t3": [{"loop_id": "react", "maxSteps": 1}]}
        )

    def test_config_from_param_model(self, isolated: Any) -> None:
        reg, base = _layer("k_alias_param", disc="loop_id")

        class React(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "react"
            __config_from_param__ = "settings"

            def __init__(self, settings: CamelSchema): ...

        _with_plan(base)
        result = build_layer_config(reg)
        adapter = TypeAdapter(result.union_type)
        assert adapter.validate_python({"loop_id": "react", "maxSteps": 2}).max_steps == 2

    def test_without_alias_generator_the_discriminator_has_no_alias(self) -> None:
        model = build_layer_config(_multi_llm_reg).per_key_models["openai"]
        assert model.model_fields["provider"].alias is None


class TestIncompleteSourceModels:
    """M2: a user model with forward references is completed before deriving."""

    _SOURCE = textwrap.dedent(
        """
        from __future__ import annotations

        from typing import Optional

        from pydantic import BaseModel


        class LoopSchema(BaseModel):
            retry: Optional[Retry] = None
            steps: int = 3


        class Retry(BaseModel):
            times: int = 1


        class Broken(BaseModel):
            retry: Optional[NowhereDefined] = None
        """
    )

    @pytest.fixture
    def schemas(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        name = f"k_fwd_schemas_{abs(hash(str(tmp_path)))}"
        (tmp_path / f"{name}.py").write_text(self._SOURCE)
        monkeypatch.syspath_prepend(str(tmp_path))
        module = importlib.import_module(name)
        yield module
        sys.modules.pop(name, None)

    def test_tier3_schema_naming_a_later_class(self, isolated: Any, schemas: Any) -> None:
        assert not schemas.LoopSchema.__pydantic_complete__
        reg, base = _layer("k_fwd_t3", disc="loop_id")

        class React(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "react"
            __config_schema__ = schemas.LoopSchema

        _with_plan(base)
        result = build_layer_config(reg)
        model = result.per_key_models["react"]
        assert model.__pydantic_complete__
        assert model.model_validate({"retry": {"times": 2}}).retry.times == 2
        composed = build_composed_config({"k_fwd_t3": reg}).top_level_type
        assert composed.model_validate({"k_fwd_t3": [{"loop_id": "react", "retry": {"times": 3}}]})

    def test_config_from_param_model_naming_a_later_class(self, isolated: Any, schemas: Any) -> None:
        assert not schemas.LoopSchema.__pydantic_complete__
        reg, base = _layer("k_fwd_param", disc="loop_id")

        def __init__(self: Any, settings: Any) -> None: ...

        __init__.__annotations__ = {"settings": schemas.LoopSchema, "return": None}
        type("React", (base,), {
            "__registry_key__": "react",
            "__config_from_param__": "settings",
            "__init__": __init__,
            "__module__": __name__,
        })
        _with_plan(base)
        model = build_layer_config(reg).per_key_models["react"]
        assert model.model_validate({"retry": {"times": 2}}).retry.times == 2

    def test_unresolvable_name_is_a_declaration_error(self, isolated: Any, schemas: Any) -> None:
        reg, base = _layer("k_fwd_broken", disc="loop_id")

        class React(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "react"
            __config_schema__ = schemas.Broken

        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(reg)
        msg = str(exc.value)
        assert exc.value.owner.endswith("React")
        assert "'NowhereDefined'" in msg and "cannot be completed" in msg
        assert "Broken" in msg


class Tier3Agent(BaseModel):
    """A Tier-3 schema with a wired field (and a validator that must see configs)."""

    name: str = "a"
    loop: Literal["react", "plan"] = "react"


class TestTier3Wiring:
    """M3: Tier-3 owners record their wired fields, so composed builds inline them."""

    @staticmethod
    def _layers() -> dict[str, Any]:
        loop, loop_base = _layer("k_t3w_loop", disc="loop_id")

        class React(loop_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "react"

            def __init__(self, steps: int = 3): ...

        _with_plan(loop_base)
        agent, agent_base = _layer("k_t3w_agent", disc="agent_id")

        class Agent(agent_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "a"
            __config_schema__ = Tier3Agent
            __wiring__ = {"loop": {"registry": "k_t3w_loop", "default": "react"}}
            loop: object

        return {"k_t3w_loop": loop, "k_t3w_agent": agent}

    def test_wired_schema_fields_are_recorded(self, isolated: Any) -> None:
        layers = self._layers()
        model = build_layer_config(layers["k_t3w_agent"]).per_key_models["a"]
        assert model.__dict__["__wired_fields__"] == {"loop": "k_t3w_loop"}
        assert set(model.__dict__["__wired_keys__"]["loop"]) == {"react", "plan"}
        assert "__wired_fields__" not in Tier3Agent.__dict__

    def test_composed_build_inlines_the_receptor(self, isolated: Any) -> None:
        result = build_composed_config(self._layers())
        model = result.layer_results["k_t3w_agent"].per_key_models["a"]
        assert model.__inline_wired__["loop"]["keys"] == ("react", "plan")
        assert model().loop.steps == 3  # default instance of the default key
        assert model.model_validate({"loop": {"loop_id": "plan", "depth": 2}}).loop.depth == 2
        with pytest.raises(ValidationError):
            model.model_validate({"loop": "plan"})

    def test_inline_only_target_reached_through_a_tier3_owner(self, isolated: Any) -> None:
        sink = create_registrar("k_t3w_sink", _MinimalProto, discriminator_field="sink_id", inline_only=True)

        class _SinkBase(metaclass=sink.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Disk(_SinkBase):
            __registry_key__ = "disk"

        class Schema(BaseModel):
            sink: Optional[str] = None

        owner, owner_base = _layer("k_t3w_owner")

        class Owner(owner_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "o"
            __config_schema__ = Schema
            __wiring__ = {"sink": "k_t3w_sink"}
            sink: object

        result = build_composed_config({"k_t3w_sink": sink, "k_t3w_owner": owner})
        assert [str(loc) for loc in result.inline_only_layers["k_t3w_sink"]] == ["k_t3w_owner[o].sink"]


class TestDiscriminatorClash:
    """M5: fields named (or aliased) like the discriminator are refused."""

    def test_tier3_schema_field(self, isolated: Any) -> None:
        reg, base = _layer("k_clash_t3")

        class S(BaseModel):
            kind: str = "whatever"

        class React(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "react"
            __config_schema__ = S

        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(reg)
        assert exc.value.field == "kind" and exc.value.owner.endswith("React")

    def test_alias_of_a_parameter(self, isolated: Any) -> None:
        reg, base = _layer("k_clash_alias")

        class T(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "t"

            def __init__(self, flavour: Annotated[str, Field(alias="kind")] = "x"): ...

        with pytest.raises(ConfigDeclarationError, match=r"\(alias 'kind'\) collides"):
            build_layer_config(reg)

    def test_wired_parameter(self, isolated: Any) -> None:
        target, target_base = _layer("k_clash_target")

        class K1(target_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "k1"

        reg, base = _layer("k_clash_wired")

        class W(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "w"
            __wiring__ = {"kind": "k_clash_target"}

            def __init__(self, kind: Optional[_MinimalProto] = None): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_layer_config(reg)
        assert exc.value.field == "kind"


class TestModelNames:
    """M6: model names are identifiers, unique per build, never logged."""

    @pytest.mark.parametrize(
        ("keys", "names"),
        [
            (["gpt-4o", "claude-3.5"], ["Gpt4OK_NamesConfig", "Claude35K_NamesConfig"]),
            (["4o", "o1"], ["_4OK_NamesConfig", "O1K_NamesConfig"]),
            (["a b", "-"], ["ABK_NamesConfig", "_K_NamesConfig"]),
        ],
    )
    def test_keys_become_identifiers(self, isolated: Any, keys: list[str], names: list[str]) -> None:
        reg, base = _layer("k_names")
        for key in keys:
            type(key, (base,), {"__registry_key__": key, "__module__": __name__})
        result = build_layer_config(reg)
        assert [result.per_key_models[key].__name__ for key in keys] == names
        namespace: dict[str, Any] = {}
        exec(compile(generate_layer_config_source(result), "<layer>", "exec"), namespace)  # noqa: S102
        assert all(name in namespace for name in names)

    def test_layer_name_becomes_identifier(self, isolated: Any) -> None:
        reg, base = _layer("llm-provider")

        class A(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "a"

        assert build_layer_config(reg).per_key_models["a"].__name__ == "ALlm_ProviderConfig"

    def test_colliding_names_get_suffixes(self, isolated: Any, caplog: pytest.LogCaptureFixture) -> None:
        reg = create_registrar("tools", _MinimalProto, discriminator_field="tool_id", key_separator=".")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class A(_Base):
            __registry_key__ = "web_search"

            def __init__(self, a: int = 1): ...

        class B(_Base):
            __registry_key__ = "web.search"

            def __init__(self, b: str = "x"): ...

        with caplog.at_level(logging.DEBUG, logger="conscribe"):
            result = build_layer_config(reg)
        assert not caplog.records
        assert [m.__name__ for m in result.per_key_models.values()] == [
            "WebSearchToolsConfig", "WebSearchToolsConfig2",
        ]
        assert sorted(generate_layer_json_schema(result)["$defs"]) == [
            "WebSearchToolsConfig", "WebSearchToolsConfig2",
        ]

    def test_one_allocator_makes_names_unique_across_layers(self, isolated: Any) -> None:
        """The allocator a composed build passes to every layer it builds."""
        first, first_base = _layer("dup")
        second, second_base = _layer("dup_")

        class X1(first_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "x"

        class X2(second_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "x"

        names = _ModelNames()
        one = _build_validated_layer(first, names=names).per_key_models["x"]
        two = _build_validated_layer(second, names=names).per_key_models["x"]
        assert (one.__name__, two.__name__) == ("XDUPConfig", "XDup_Config")
        three = _build_validated_layer(first, names=names).per_key_models["x"]
        assert three.__name__ == "XDUPConfig2"


class TestEmptyLayerCitesImportFailures:
    """m3: an empty layer names the modules discover() could not import."""

    def test_failed_module_is_cited(
        self, isolated: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        package = tmp_path / "k_broken_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "impl.py").write_text("raise ImportError('missing dependency: foo')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            with pytest.raises(ImportError):
                discover("k_broken_pkg")
            reg, _ = _layer("k_empty_failed")
            with pytest.raises(EmptyLayerError) as exc:
                build_layer_config(reg)
        finally:
            sys.modules.pop("k_broken_pkg", None)
        assert "k_broken_pkg.impl" in exc.value.import_failures
        assert "k_broken_pkg.impl (ImportError: missing dependency: foo)" in str(exc.value)

    def test_no_failures_no_citation(self, isolated: Any) -> None:
        clear_import_failures()  # (isolation clears the record again afterwards)
        reg, _ = _layer("k_empty_clean")
        with pytest.raises(EmptyLayerError) as exc:
            build_layer_config(reg)
        assert exc.value.import_failures == {}
        assert "discover() recorded" not in str(exc.value)


class TestBuildArguments:
    """m4: what build_layer_config refuses before looking at classes."""

    @pytest.mark.parametrize(
        ("value", "shown"),
        [("llm", "str 'llm'"), (None, "NoneType None"), ({"llm": 1}, "dict {'llm': 1}")],
    )
    def test_non_registrar_is_a_type_error(self, value: Any, shown: str) -> None:
        with pytest.raises(TypeError) as exc:
            build_layer_config(value)
        assert str(exc.value) == (
            "build_layer_config() expects a layer registrar created by "
            f"create_registrar(), got {shown}."
        )

    def test_a_class_that_is_not_a_registrar(self) -> None:
        with pytest.raises(TypeError, match=r"got the class .*\.OnlyOpenAI\."):
            build_layer_config(OnlyOpenAI)


def _noisy_marker() -> str:
    """Annotation metadata whose evaluation warns (user code run by extraction)."""
    warnings.warn("warned by user code", UserWarning, stacklevel=1)
    return "marker"


_NOISY_LINE = inspect.getsourcelines(_noisy_marker)[1] + 2


class TestWarningLocation:
    """m5: warnings point at the class being built, not at conscribe."""

    def test_extractor_degradation_warning(self) -> None:
        with pytest.warns(DegradationWarning) as record:
            build_layer_config(_degraded_reg)
        [warning] = [w for w in record if issubclass(w.category, DegradationWarning)]
        assert warning.filename == inspect.getsourcefile(DegradedProvider)
        assert warning.lineno == inspect.getsourcelines(DegradedProvider)[1]

    @staticmethod
    def _noisy_layer() -> Any:
        reg, base = _layer("k_noisy")

        class Noisy(base):  # type: ignore[misc, valid-type]
            __registry_key__ = "noisy"

            def __init__(self, x: Annotated[int, _noisy_marker()] = 1): ...

        return reg

    def test_other_warnings_keep_their_location(self, isolated: Any) -> None:
        reg = self._noisy_layer()
        with pytest.warns(UserWarning, match="warned by user code") as record:
            build_layer_config(reg)
        noisy = [w for w in record if str(w.message) == "warned by user code"]
        assert noisy and all(
            (w.category, w.filename, w.lineno) == (UserWarning, __file__, _NOISY_LINE) for w in noisy
        )

    def test_other_warnings_still_obey_user_filters(self, isolated: Any) -> None:
        reg = self._noisy_layer()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message="warned by user code")
            build_layer_config(reg)
        assert not [w for w in record if str(w.message) == "warned by user code"]


class TestTier3WiredKeysFollowTheSchemaLiteral:
    """A Tier-3 schema field typed ``Literal[...]`` narrows the inline union."""

    def _build(self) -> tuple[Any, Any]:
        loops, loop_base = _layer("t3n_loop", disc="loop_id")
        agents, agent_base = _layer("t3n_agent", disc="agent_id")

        class React(loop_base):
            __registry_key__ = "react"

        class Plan(loop_base):
            __registry_key__ = "plan"

        class Schema(BaseModel):
            loop: Literal["react"] = "react"

        class T3(agent_base):
            __registry_key__ = "t3"
            __config_schema__ = Schema
            __wiring__ = {"loop": "t3n_loop"}
            loop: loop_base

        return loops, agents

    def test_composed_accepts_only_what_the_schema_accepts(self, isolated: Any) -> None:
        loops, agents = self._build()
        top = build_composed_config({"t3n_loop": loops, "t3n_agent": agents}).top_level_type
        top.model_validate({"t3n_agent": [{"agent_id": "t3", "loop": {"loop_id": "react"}}]})
        with pytest.raises(ValidationError):
            top.model_validate({"t3n_agent": [{"agent_id": "t3", "loop": {"loop_id": "plan"}}]})

    def test_disjoint_schema_literal_is_refused(self, isolated: Any) -> None:
        loops, loop_base = _layer("t3d_loop", disc="loop_id")
        agents, agent_base = _layer("t3d_agent", disc="agent_id")

        class React(loop_base):
            __registry_key__ = "react"

        class Schema(BaseModel):
            loop: Literal["plan"] = "plan"

        class T3(agent_base):
            __registry_key__ = "t3"
            __config_schema__ = Schema
            __wiring__ = {"loop": "t3d_loop"}
            loop: loop_base

        with pytest.raises(ConfigDeclarationError, match="no key fits both"):
            build_composed_config({"t3d_loop": loops, "t3d_agent": agents})
