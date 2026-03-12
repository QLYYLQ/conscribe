"""Integration tests: Config typing end-to-end pipeline.

Tests the full config pipeline from registration through extraction,
union building, code generation, and validation. Each test exercises
at least two modules in the config subsystem.

Pipeline flow:
  register -> extract -> build -> codegen/json_schema -> validate
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from conscribe import (
    LayerConfigResult,
    build_layer_config,
    compute_registry_fingerprint,
    create_registrar,
    extract_config_schema,
    generate_layer_config_source,
    generate_layer_json_schema,
    load_cached_fingerprint,
    save_fingerprint,
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
# Test: Register -> Extract -> Build -> Codegen -> Validate
# ===================================================================

class TestFullPipelineCodegen:
    """End-to-end: register, extract, build union, codegen, validate output."""

    def test_register_extract_build_codegen_validate(self) -> None:
        """Full pipeline: register implementations, extract schemas, build
        union, generate source, compile and validate config dicts."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["LLM", "Provider"],
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

        # Extract
        openai_schema = extract_config_schema(OpenAIProvider)
        anthropic_schema = extract_config_schema(AnthropicProvider)
        assert openai_schema is not None
        assert anthropic_schema is not None

        # Build
        result = build_layer_config(LR)
        assert isinstance(result, LayerConfigResult)
        assert set(result.per_key_models.keys()) == {"open_ai", "anthropic"}
        assert result.discriminator_field == "provider"

        # Codegen
        source = generate_layer_config_source(result)
        assert "class OpenAiLLMConfig(BaseModel):" in source
        assert "class AnthropicLLMConfig(BaseModel):" in source
        assert "provider: Literal['open_ai']" in source
        assert "provider: Literal['anthropic']" in source

        # Compile and validate
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        adapter = TypeAdapter(result.union_type)

        valid_config = {"provider": "open_ai", "model_id": "gpt-4o", "temperature": 0.5}
        validated = adapter.validate_python(valid_config)
        assert validated.provider == "open_ai"
        assert validated.model_id == "gpt-4o"


# ===================================================================
# Test: Register -> Extract -> Build -> JSON Schema -> Validate
# ===================================================================

class TestPipelineJsonSchema:
    """End-to-end: register, extract, build, json_schema, validate structure."""

    def test_register_build_json_schema_structure(self) -> None:
        """JSON schema output has correct structure with discriminator info."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class AlphaProvider(BaseLLM):
            def __init__(self, *, model_id: str, temp: float = 0.5):
                self.model_id = model_id
                self.temp = temp

            async def chat(self, messages: list[dict]) -> str:
                return "alpha"

        class BetaProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "beta"

        result = build_layer_config(LR)
        schema = generate_layer_json_schema(result)

        # Structure checks
        assert "x-discriminator" in schema
        assert schema["x-discriminator"] == "provider"
        # JSON Schema should reference sub-schemas (via $defs or anyOf/oneOf)
        # With discriminated union, Pydantic typically uses $defs
        assert "$defs" in schema or "anyOf" in schema or "oneOf" in schema


# ===================================================================
# Test: Register -> build_config() via Registrar classmethod
# ===================================================================

class TestRegistrarBuildConfig:
    """Test registrar convenience methods for config building."""

    def test_registrar_build_config(self) -> None:
        """build_config() classmethod delegates to build_layer_config correctly."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class SimpleProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "simple"

        result = LR.build_config()
        assert isinstance(result, LayerConfigResult)
        assert "simple" in result.per_key_models
        assert result.layer_name == "llm"
        assert result.discriminator_field == "provider"

    def test_registrar_config_union_type_validates(self) -> None:
        """config_union_type() returns a type usable with TypeAdapter."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class FastProvider(BaseLLM):
            def __init__(self, *, model_id: str, speed: int = 10):
                self.model_id = model_id
                self.speed = speed

            async def chat(self, messages: list[dict]) -> str:
                return "fast"

        union_type = LR.config_union_type()
        adapter = TypeAdapter(union_type)

        valid = adapter.validate_python({"provider": "fast", "model_id": "turbo"})
        assert valid.model_id == "turbo"
        assert valid.speed == 10  # default

        with pytest.raises(ValidationError):
            adapter.validate_python({"provider": "fast"})  # missing required model_id

    def test_registrar_get_config_schema(self) -> None:
        """get_config_schema() returns a BaseModel for a given key."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class SmartProvider(BaseLLM):
            def __init__(self, *, model_id: str, reasoning: bool = True):
                self.model_id = model_id
                self.reasoning = reasoning

            async def chat(self, messages: list[dict]) -> str:
                return "smart"

        schema = LR.get_config_schema("smart")
        assert schema is not None
        assert issubclass(schema, BaseModel)
        assert "model_id" in schema.model_fields
        assert "reasoning" in schema.model_fields


# ===================================================================
# Test: Codegen output compiles and exec'd models validate
# ===================================================================

class TestCodegenCompileAndExec:
    """Codegen output is valid Python that creates usable Pydantic models."""

    def test_codegen_exec_models_validate_config_dicts(self) -> None:
        """Exec generated source, extract models, validate config dicts."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class GoodProvider(BaseLLM):
            def __init__(
                self, *,
                model_id: Annotated[str, Field(description="Model identifier")],
                temperature: Annotated[float, Field(0.0, ge=0, le=2)] = 0.0,
            ):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "good"

        result = build_layer_config(LR)
        source = generate_layer_config_source(result)

        # Compile and exec into a fresh namespace
        ns: dict = {}
        exec(compile(source, "<test-codegen>", "exec"), ns)  # noqa: S102

        # The generated module should have the model class
        model_cls = ns.get("GoodLLMConfig")
        assert model_cls is not None
        assert issubclass(model_cls, BaseModel)

        # Validate using the exec'd model
        instance = model_cls(provider="good", model_id="test-model")
        assert instance.model_id == "test-model"
        assert instance.temperature == 0.0

        # Constraint enforcement from the exec'd model
        with pytest.raises(ValidationError):
            model_cls(provider="good", model_id="test", temperature=5.0)


# ===================================================================
# Test: Full pipeline with TypeAdapter union validation
# ===================================================================

class TestFullPipelineTypeAdapterValidation:
    """Full pipeline ending with TypeAdapter-based discriminated union validation."""

    def test_register_extract_build_codegen_exec_type_adapter(self) -> None:
        """Full pipeline: register -> extract -> build -> codegen -> exec ->
        TypeAdapter validates correct discriminator dispatch."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["LLM"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class OpenAILLM(BaseLLM):
            def __init__(self, *, model_id: str, temperature: float = 0.0):
                self.model_id = model_id
                self.temperature = temperature

            async def chat(self, messages: list[dict]) -> str:
                return "openai"

        class DeepSeekLLM(BaseLLM):
            def __init__(self, *, model_id: str, top_k: int = 50):
                self.model_id = model_id
                self.top_k = top_k

            async def chat(self, messages: list[dict]) -> str:
                return "deepseek"

        result = build_layer_config(LR)
        source = generate_layer_config_source(result)

        # Compile and exec
        ns: dict = {}
        exec(compile(source, "<test-full>", "exec"), ns)  # noqa: S102

        # Use the runtime union_type with TypeAdapter
        adapter = TypeAdapter(result.union_type)

        # Validate OpenAI config
        openai_cfg = adapter.validate_python({
            "provider": "open_ai", "model_id": "gpt-4o"
        })
        assert openai_cfg.provider == "open_ai"
        assert openai_cfg.model_id == "gpt-4o"
        assert openai_cfg.temperature == 0.0

        # Validate DeepSeek config
        ds_cfg = adapter.validate_python({
            "provider": "deep_seek", "model_id": "deepseek-v2", "top_k": 100
        })
        assert ds_cfg.provider == "deep_seek"
        assert ds_cfg.top_k == 100


# ===================================================================
# Test: Fingerprint compute + save + load roundtrip
# ===================================================================

class TestFingerprintRoundtrip:
    """Fingerprint compute, save, load roundtrip using tmp_path."""

    def test_fingerprint_compute_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Compute fingerprint, save to file, load back, verify match."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class StableProvider(BaseLLM):
            def __init__(self, *, model_id: str, version: int = 1):
                self.model_id = model_id
                self.version = version

            async def chat(self, messages: list[dict]) -> str:
                return "stable"

        # Compute
        fp = compute_registry_fingerprint(LR)
        assert isinstance(fp, str)
        assert len(fp) == 16  # first 16 hex chars of SHA-256

        # Save
        fp_path = tmp_path / ".registry_fingerprint"
        save_fingerprint(fp_path, "llm", fp)
        assert fp_path.exists()

        # Load
        loaded = load_cached_fingerprint(fp_path, "llm")
        assert loaded == fp

        # Non-existent layer returns None
        assert load_cached_fingerprint(fp_path, "agent") is None

    def test_fingerprint_changes_on_new_registration(self) -> None:
        """Fingerprint changes when a new implementation is registered."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class FirstProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "first"

        fp1 = compute_registry_fingerprint(LR)

        # Register a second provider manually
        class SecondProvider(BaseLLM):
            def __init__(self, *, model_id: str, extra: bool = False):
                self.model_id = model_id
                self.extra = extra

            async def chat(self, messages: list[dict]) -> str:
                return "second"

        fp2 = compute_registry_fingerprint(LR)

        assert fp1 != fp2


# ===================================================================
# Test: Multiple independent layers build correctly
# ===================================================================

class TestMultiLayerIndependentBuild:
    """Multiple layers (llm + agent) each build independently correct unions."""

    def test_two_layers_build_independently(self) -> None:
        """LLM and Agent layers each produce their own union with correct
        discriminator fields and per-key models."""
        LR = create_registrar(
            "llm", LLMProtocol,
            strip_suffixes=["Provider"],
            discriminator_field="provider",
        )
        AR = create_registrar(
            "agent", AgentProtocol,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        # LLM layer
        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class AlphaProvider(BaseLLM):
            def __init__(self, *, model_id: str):
                self.model_id = model_id

            async def chat(self, messages: list[dict]) -> str:
                return "alpha"

        # Agent layer
        class BaseAgent(metaclass=AR.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class SmartAgent(BaseAgent):
            def __init__(self, *, max_steps: int = 100):
                self.max_steps = max_steps

            async def step(self, task: str) -> str:
                return "smart"

            def reset(self) -> None:
                pass

        # Build independently
        llm_result = build_layer_config(LR)
        agent_result = build_layer_config(AR)

        # LLM assertions
        assert llm_result.layer_name == "llm"
        assert llm_result.discriminator_field == "provider"
        assert "alpha" in llm_result.per_key_models
        llm_adapter = TypeAdapter(llm_result.union_type)
        llm_cfg = llm_adapter.validate_python({"provider": "alpha", "model_id": "a1"})
        assert llm_cfg.model_id == "a1"

        # Agent assertions
        assert agent_result.layer_name == "agent"
        assert agent_result.discriminator_field == "name"
        assert "smart" in agent_result.per_key_models
        agent_adapter = TypeAdapter(agent_result.union_type)
        agent_cfg = agent_adapter.validate_python({"name": "smart"})
        assert agent_cfg.max_steps == 100

        # Cross-layer isolation: LLM union should not accept agent configs
        with pytest.raises(ValidationError):
            llm_adapter.validate_python({"name": "smart", "max_steps": 50})
