"""Tests for composed config (multi-layer inline wiring).

Tests the ``build_composed_config()`` pipeline, including dependency graph
construction, topological sorting, inline wiring replacement, and
JSON Schema / Python source generation.
"""
from __future__ import annotations

from typing import Literal, get_args, get_origin
import pytest
from pydantic import BaseModel, TypeAdapter

from conscribe.config.builder import build_layer_config
from conscribe.config.codegen import generate_composed_config_source
from conscribe.config.composed import (
    ComposedConfigResult,
    _build_dependency_graph,
    _topological_sort,
    build_composed_config,
)
from conscribe.config.json_schema import generate_composed_json_schema
from conscribe.exceptions import CircularWiringError
from conscribe.registration import create_registrar
from conscribe.registration.registry import (
    _REGISTRY_INDEX,
    _deregister,
)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class LoopProtocol(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...


# ---------------------------------------------------------------------------
# Cleanup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_registries():
    yield
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("comp_"):
            _deregister(name)


# ---------------------------------------------------------------------------
# Registrar + implementation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_registrar():
    reg = create_registrar(
        "comp_llm", LLMProtocol,
        strip_suffixes=["LLM", "Provider"],
        discriminator_field="provider",
    )

    class _LLMBase(metaclass=reg.Meta):
        __abstract__ = True

        async def chat(self, messages: list[dict]) -> str:
            return ""

    class OpenAIProvider(_LLMBase):
        __registry_key__ = "openai"

        def __init__(self, *, model: str = "gpt-4", temperature: float = 0.7):
            self.model = model
            self.temperature = temperature

    class AnthropicProvider(_LLMBase):
        __registry_key__ = "anthropic"

        def __init__(self, *, model: str = "claude-3", max_tokens: int = 1024):
            self.model = model
            self.max_tokens = max_tokens

    return reg


@pytest.fixture
def loop_registrar():
    reg = create_registrar(
        "comp_loop", LoopProtocol,
        strip_suffixes=["Loop"],
        discriminator_field="name",
    )

    class _LoopBase(metaclass=reg.Meta):
        __abstract__ = True

        def run(self) -> None:
            pass

    class ReactLoop(_LoopBase):
        __registry_key__ = "react"

        def __init__(self, *, max_steps: int = 10):
            self.max_steps = max_steps

    class CodeactLoop(_LoopBase):
        __registry_key__ = "codeact"

        def __init__(self, *, sandbox: str = "docker"):
            self.sandbox = sandbox

    return reg


@pytest.fixture
def agent_registrar(llm_registrar, loop_registrar):
    """Agent registrar with wiring to LLM and Loop."""
    reg = create_registrar(
        "comp_agent", AgentProtocol,
        strip_suffixes=["Agent"],
        discriminator_field="name",
    )

    class _AgentBase(metaclass=reg.Meta):
        __abstract__ = True

        async def step(self, task: str) -> str:
            return ""

    class BrowserUseAgent(_AgentBase):
        __registry_key__ = "browser_use"
        __wiring__ = {
            "llm": "comp_llm",
            "loop": "comp_loop",
        }

        def __init__(self, *, use_vision: bool = True):
            self.use_vision = use_vision

    class ReactAgent(_AgentBase):
        __registry_key__ = "react"
        __wiring__ = {
            "llm": ("comp_llm", ["openai"]),  # Mode 2: subset
        }

        def __init__(self, *, max_retries: int = 3):
            self.max_retries = max_retries

    return reg


# ---------------------------------------------------------------------------
# TestDependencyGraph
# ---------------------------------------------------------------------------


class TestDependencyGraph:
    def test_no_wiring_no_edges(self, llm_registrar, loop_registrar):
        registrars = {
            "comp_llm": llm_registrar,
            "comp_loop": loop_registrar,
        }
        graph = _build_dependency_graph(registrars)
        assert graph == {"comp_llm": set(), "comp_loop": set()}

    def test_wiring_creates_edges(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        registrars = {
            "comp_llm": llm_registrar,
            "comp_loop": loop_registrar,
            "comp_agent": agent_registrar,
        }
        graph = _build_dependency_graph(registrars)
        assert graph["comp_agent"] == {"comp_llm", "comp_loop"}
        assert graph["comp_llm"] == set()
        assert graph["comp_loop"] == set()

    def test_edges_only_for_included_layers(
        self, llm_registrar, agent_registrar,
    ):
        """Agent wires to comp_loop, but loop is not in registrars."""
        registrars = {
            "comp_llm": llm_registrar,
            "comp_agent": agent_registrar,
        }
        graph = _build_dependency_graph(registrars)
        # comp_loop not included, so only comp_llm edge
        assert graph["comp_agent"] == {"comp_llm"}


# ---------------------------------------------------------------------------
# TestTopologicalSort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_leaves_first(self):
        graph = {"a": {"b", "c"}, "b": set(), "c": set()}
        order = _topological_sort(graph, {"a", "b", "c"})
        assert order.index("b") < order.index("a")
        assert order.index("c") < order.index("a")

    def test_deterministic_order(self):
        graph = {"x": set(), "y": set(), "z": set()}
        order = _topological_sort(graph, {"x", "y", "z"})
        assert order == ["x", "y", "z"]

    def test_cycle_raises(self):
        graph = {"a": {"b"}, "b": {"a"}}
        with pytest.raises(CircularWiringError):
            _topological_sort(graph, {"a", "b"})

    def test_three_node_cycle_raises(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        with pytest.raises(CircularWiringError):
            _topological_sort(graph, {"a", "b", "c"})

    def test_chain_ordering(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": set()}
        order = _topological_sort(graph, {"a", "b", "c"})
        assert order == ["c", "b", "a"]


# ---------------------------------------------------------------------------
# TestBuildComposedConfig
# ---------------------------------------------------------------------------


class TestBuildComposedConfig:
    def test_result_structure(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=False,
        )
        assert isinstance(result, ComposedConfigResult)
        assert set(result.layer_results.keys()) == {"comp_llm", "comp_loop", "comp_agent"}
        assert result.inline_wiring is False
        # Agent comes after LLM and Loop in dependency order
        assert result.dependency_order.index("comp_agent") > result.dependency_order.index("comp_llm")
        assert result.dependency_order.index("comp_agent") > result.dependency_order.index("comp_loop")

    def test_inline_false_preserves_literals(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=False,
        )
        agent_result = result.layer_results["comp_agent"]
        browser_model = agent_result.per_key_models["browser_use"]
        llm_field = browser_model.model_fields["llm"]
        # Should still be Literal[...]
        assert get_origin(llm_field.annotation) is Literal

    def test_inline_true_replaces_with_union(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        agent_result = result.layer_results["comp_agent"]
        browser_model = agent_result.per_key_models["browser_use"]
        llm_field = browser_model.model_fields["llm"]
        # Should NOT be Literal anymore
        assert get_origin(llm_field.annotation) is not Literal
        # Should be a complex type (Annotated union or single model)
        # Verify it can validate an inline dict
        adapter = TypeAdapter(browser_model)
        validated = adapter.validate_python({
            "name": "browser_use",
            "use_vision": True,
            "llm": {"provider": "openai", "model": "gpt-4", "temperature": 0.5},
            "loop": {"name": "react", "max_steps": 20},
        })
        assert validated.name == "browser_use"

    def test_top_level_model_has_list_fields(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
        )
        top = result.top_level_type
        assert "comp_llm" in top.model_fields
        assert "comp_loop" in top.model_fields
        assert "comp_agent" in top.model_fields
        # Each field should default to empty list
        instance = top()
        assert instance.comp_llm == []
        assert instance.comp_agent == []


# ---------------------------------------------------------------------------
# TestSubsetWiring
# ---------------------------------------------------------------------------


class TestSubsetWiring:
    def test_mode2_subset_inlines_only_subset(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        """ReactAgent wires llm to ("comp_llm", ["openai"]) — subset."""
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        agent_result = result.layer_results["comp_agent"]
        react_model = agent_result.per_key_models["react"]
        llm_field = react_model.model_fields["llm"]

        # Should accept openai config
        adapter = TypeAdapter(react_model)
        validated = adapter.validate_python({
            "name": "react",
            "max_retries": 5,
            "llm": {"provider": "openai", "model": "gpt-4o"},
        })
        assert validated.name == "react"

        # Should reject anthropic config (not in subset)
        with pytest.raises(Exception):
            adapter.validate_python({
                "name": "react",
                "max_retries": 5,
                "llm": {"provider": "anthropic", "model": "claude-3"},
            })


# ---------------------------------------------------------------------------
# TestMode3Passthrough
# ---------------------------------------------------------------------------


class TestMode3Passthrough:
    def test_literal_list_stays_literal(self, llm_registrar):
        """Mode 3 wiring (literal list) should not be inlined."""
        reg = create_registrar(
            "comp_m3_agent", AgentProtocol,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            async def step(self, task: str) -> str:
                return ""

        class SimpleAgent(_Base):
            __registry_key__ = "simple"
            __wiring__ = {"browser": ["chromium", "firefox"]}

            def __init__(self, *, timeout: int = 30):
                self.timeout = timeout

        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_m3_agent": reg},
            inline_wiring=True,
        )
        agent_result = result.layer_results["comp_m3_agent"]
        simple_model = agent_result.per_key_models["simple"]
        browser_field = simple_model.model_fields["browser"]
        # Mode 3: should still be Literal
        assert get_origin(browser_field.annotation) is Literal
        assert set(get_args(browser_field.annotation)) == {"chromium", "firefox"}


# ---------------------------------------------------------------------------
# TestComposedJsonSchema
# ---------------------------------------------------------------------------


class TestComposedJsonSchema:
    def test_has_layer_properties(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        schema = generate_composed_json_schema(result)

        assert "properties" in schema
        assert "comp_llm" in schema["properties"]
        assert "comp_loop" in schema["properties"]
        assert "comp_agent" in schema["properties"]

    def test_has_extension_fields(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
        )
        schema = generate_composed_json_schema(result)
        assert schema["x-composed-layers"] == result.dependency_order
        assert schema["x-inline-wiring"] is True

    def test_has_defs_for_nested_models(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        schema = generate_composed_json_schema(result)
        defs = schema.get("$defs", {})
        # Should have definitions for LLM models (referenced from agent inline)
        def_names = set(defs.keys())
        # At least the per-key models should be in $defs
        assert len(def_names) > 0

    def test_schema_validates_inline_config(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        adapter = TypeAdapter(result.top_level_type)
        validated = adapter.validate_python({
            "comp_llm": [
                {"provider": "openai", "model": "gpt-4"},
            ],
            "comp_agent": [
                {
                    "name": "browser_use",
                    "use_vision": True,
                    "llm": {"provider": "anthropic", "model": "claude-3"},
                    "loop": {"name": "codeact", "sandbox": "local"},
                },
            ],
        })
        assert len(validated.comp_llm) == 1
        assert len(validated.comp_agent) == 1


# ---------------------------------------------------------------------------
# TestComposedCodegen
# ---------------------------------------------------------------------------


class TestComposedCodegen:
    def test_source_compiles(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        source = generate_composed_config_source(result)
        # Should compile without errors
        compile(source, "<composed>", "exec")

    def test_source_has_header(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
        )
        source = generate_composed_config_source(result)
        assert "Auto-generated by conscribe" in source
        assert "Composed config" in source

    def test_source_has_composed_config_class(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
        )
        source = generate_composed_config_source(result)
        assert "class ComposedConfig(BaseModel):" in source
        assert "comp_llm: list[" in source
        assert "comp_agent: list[" in source

    def test_layers_defined_in_dependency_order(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
        )
        source = generate_composed_config_source(result)
        llm_pos = source.index("Layer: comp_llm")
        agent_pos = source.index("Layer: comp_agent")
        assert llm_pos < agent_pos


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_inline_config_round_trip(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        """Full round-trip: build → validate with nested inline config."""
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        adapter = TypeAdapter(result.top_level_type)

        data = {
            "comp_llm": [
                {"provider": "openai", "model": "gpt-4o", "temperature": 0.3},
                {"provider": "anthropic", "model": "claude-3", "max_tokens": 2048},
            ],
            "comp_loop": [
                {"name": "react", "max_steps": 5},
            ],
            "comp_agent": [
                {
                    "name": "browser_use",
                    "use_vision": False,
                    "llm": {"provider": "openai", "model": "gpt-4o"},
                    "loop": {"name": "react", "max_steps": 20},
                },
                {
                    "name": "react",
                    "max_retries": 5,
                    "llm": {"provider": "openai"},
                },
            ],
        }

        validated = adapter.validate_python(data)
        assert len(validated.comp_llm) == 2
        assert len(validated.comp_loop) == 1
        assert len(validated.comp_agent) == 2

    def test_invalid_inline_config_rejected(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        """Invalid nested config should be rejected."""
        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
            inline_wiring=True,
        )
        adapter = TypeAdapter(result.top_level_type)

        with pytest.raises(Exception):
            adapter.validate_python({
                "comp_agent": [
                    {
                        "name": "browser_use",
                        "llm": {"provider": "nonexistent_provider"},
                        "loop": {"name": "react"},
                    },
                ],
            })
