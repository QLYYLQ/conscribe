"""Tests for composed config (multi-layer inline wiring).

Tests the ``build_composed_config()`` pipeline, including dependency graph
construction, topological sorting, inline wiring replacement, and
JSON Schema / Python source generation.
"""
from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)


def _literal_members(annotation):
    """Return the ``Literal`` members of a wired field, or ``None``.

    Slot-less wired fields are emitted as ``Optional[Literal[...]]``, so the
    ``Optional`` wrapper has to come off before inspecting the constraint.
    """
    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            members = _literal_members(arg)
            if members is not None:
                return members
        return None
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    return None

from conscribe.config.builder import (
    GENERATED_MARKER,
    LayerConfigResult,
    _layer_union,
    build_layer_config,
)
from conscribe.config.codegen import generate_composed_config_source
from conscribe.config.composed import (
    ComposedConfigResult,
    _build_dependency_graph,
    _rebuild_model_with_replacements,
    _replace_wired_literals_with_unions,
    _topological_sort,
    build_composed_config,
)
from conscribe.config.json_schema import generate_composed_json_schema
from conscribe.exceptions import (
    CircularWiringError,
    ConfigDeclarationError,
    EmptyLayerError,
)
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


@runtime_checkable
class ArchiverProtocol(Protocol):
    """Receptor annotations below reference this module-level type, so the
    extractor can resolve them under ``from __future__ import annotations``."""

    def run(self) -> None: ...


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
        with pytest.raises(CircularWiringError) as exc:
            _topological_sort(graph, {"a", "b"})
        assert exc.value.cycle == ["a", "b"]
        assert str(exc.value).startswith("Circular wiring dependency detected: a -> b -> a.")

    def test_three_node_cycle_raises(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        with pytest.raises(CircularWiringError) as exc:
            _topological_sort(graph, {"a", "b", "c"})
        assert exc.value.cycle == ["a", "b", "c"]

    def test_reported_cycle_is_an_actual_cycle(self):
        """Layers that only depend on a cycle are not part of it (review m3:
        ``a <-> b`` plus ``c -> a`` used to be reported as ``a -> b -> c -> a``)."""
        graph = {"a": {"b"}, "b": {"a"}, "c": {"a"}, "d": set()}
        with pytest.raises(CircularWiringError) as exc:
            _topological_sort(graph, {"a", "b", "c", "d"})
        assert exc.value.cycle == ["a", "b"]
        assert "a -> b -> a." in str(exc.value)

    def test_reported_cycle_follows_dependencies(self):
        """The walk starts at the smallest stuck layer and follows its smallest
        stuck dependency; the cycle it closes is reported from its first node."""
        graph = {"a": {"x"}, "x": {"y"}, "y": {"z"}, "z": {"y"}}
        with pytest.raises(CircularWiringError) as exc:
            _topological_sort(graph, {"a", "x", "y", "z"})
        assert exc.value.cycle == ["y", "z"]
        for node, nxt in zip(exc.value.cycle, exc.value.cycle[1:] + exc.value.cycle[:1]):
            assert nxt in graph[node]

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
        # Should still be a Literal selector (wrapped in Optional because
        # ``llm`` is a slot-less wired declaration on the fixture agent).
        assert _literal_members(llm_field.annotation) is not None

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
        # Mode 3: should still be a Literal selector, never inlined
        assert set(_literal_members(browser_field.annotation)) == {"chromium", "firefox"}


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
# TestComposedNestedSource — nested/compound discriminator layers
# ---------------------------------------------------------------------------


class TestComposedNestedSource:
    """Composed Python source for a nested-discriminator layer must exec and
    build the compound union, including same-named provider submodels that
    differ in fields (regression for the four defects: undefined submodels,
    name collision, rebuild ordering, flat-discriminator on a nested layer).
    """

    @staticmethod
    def _nested_registrar():
        reg = create_registrar(
            "comp_nllm", LLMProtocol,
            discriminator_fields=["model_type", "provider"],
            key_separator=".",
        )

        class _AnthropicBase(metaclass=reg.Meta):
            __registry_key__ = "anthropic"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

            def __init__(self, model: str = "claude-3"):
                self.model = model

        class AnthropicDirect(_AnthropicBase):
            __registry_key__ = "anthropic.direct"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class AnthropicAws(_AnthropicBase):
            __registry_key__ = "anthropic.aws"

            def __init__(self, aws_region: str = "us-east-1", **kwargs):
                super().__init__(**kwargs)
                self.aws_region = aws_region

        class _OpenAIBase(metaclass=reg.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

            def __init__(self, model: str = "gpt-4"):
                self.model = model

        # Same provider segment "direct" as anthropic, but DIFFERENT fields —
        # both synthesize a class named ``DirectProviderConfig``.
        class OpenAIDirect(_OpenAIBase):
            __registry_key__ = "openai.direct"

            def __init__(self, organization: str = "", project: str = "", **kwargs):
                super().__init__(**kwargs)
                self.organization = organization
                self.project = project

        return reg

    def test_nested_composed_source_execs(self):
        reg = self._nested_registrar()
        result = build_composed_config({"comp_nllm": reg}, inline_wiring=True)
        source = generate_composed_config_source(result)

        # Bug A (undefined nested submodels) + Bug C (rebuild ordering under
        # ``from __future__ import annotations``): exec must not raise.
        ns: dict = {}
        exec(compile(source, "<composed-nested>", "exec"), ns)  # noqa: S102

        assert "ComposedConfig" in ns
        # Bug D: a flat ``Field(discriminator="model_type")`` alias would reject
        # the repeated ``openai``/``anthropic`` discriminator values; the
        # compound ``Discriminator`` callable must build cleanly.
        TypeAdapter(ns["ComposedConfig"])

    def test_colliding_provider_submodels_are_distinct(self):
        reg = self._nested_registrar()
        result = build_composed_config({"comp_nllm": reg}, inline_wiring=True)
        source = generate_composed_config_source(result)

        ns: dict = {}
        exec(compile(source, "<composed-nested>", "exec"), ns)  # noqa: S102

        # Bug B: the two same-named ``direct`` provider classes must both be
        # emitted, disambiguated, with their own (different) field sets.
        assert "DirectProviderConfig" in ns
        assert "DirectProviderConfig2" in ns
        fields_a = set(ns["DirectProviderConfig"].model_fields)
        fields_b = set(ns["DirectProviderConfig2"].model_fields)
        assert fields_a != fields_b
        # exactly one variant carries the openai-only fields
        assert ("organization" in fields_a) != ("organization" in fields_b)

    def test_source_has_compound_discriminator(self):
        reg = self._nested_registrar()
        result = build_composed_config({"comp_nllm": reg}, inline_wiring=True)
        source = generate_composed_config_source(result)
        # Nested layer emits a compound discriminator function + Tag union,
        # not a flat Field(discriminator=...).
        assert "def _discriminate_comp_nllm(" in source
        assert "Discriminator(_discriminate_comp_nllm)" in source
        assert 'Tag("anthropic.direct")' in source


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


# ---------------------------------------------------------------------------
# Inline receptors keep their metadata (D9)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _TraceSafe:
    """Plain marker (af style): pydantic ignores it, a framework reads it back."""


def _error_types(exc: ValidationError) -> list[str]:
    return [e["type"] for e in exc.errors()]


@pytest.fixture
def archiver_registrar():
    reg = create_registrar(
        "comp_archiver", ArchiverProtocol, discriminator_field="archiver_id",
    )

    class _ArchBase(metaclass=reg.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class Net(_ArchBase):
        __registry_key__ = "net"

        def __init__(self, *, host: str = "localhost"):
            self.host = host

    class Disk(_ArchBase):
        __registry_key__ = "disk"

        def __init__(self, *, path: str = "/tmp"):
            self.path = path

    return reg


@pytest.fixture
def hook_registrar():
    reg = create_registrar("comp_hook", ArchiverProtocol, discriminator_field="hook_id")

    class _HookBase(metaclass=reg.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    reg.test_base = _HookBase  # type: ignore[attr-defined]
    return reg


class TestInlineReceptorMetadata:
    def test_af_defect_9_description_survives(self, archiver_registrar, hook_registrar):
        """af's repro: ``trace_writer.archive`` keeps its description everywhere."""

        class _Receptors:
            archive: list[ArchiverProtocol]

        class TraceWriter(_Receptors, hook_registrar.test_base):
            __registry_key__ = "trace_writer"
            __wiring__ = {"archive": "comp_archiver"}

            def __init__(
                self,
                archive: Annotated[
                    list[ArchiverProtocol] | None,
                    Field(description="what this run records"),
                ] = None,
            ): ...

        per_layer = build_layer_config(hook_registrar).per_key_models["trace_writer"]
        assert per_layer.model_fields["archive"].description == "what this run records"

        result = build_composed_config(
            {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
        )
        model = result.layer_results["comp_hook"].per_key_models["trace_writer"]
        field = model.model_fields["archive"]
        assert field.description == "what this run records"
        assert not field.is_required() and field.default is None
        assert model.model_validate(
            {"archive": [{"archiver_id": "net"}, {"archiver_id": "disk", "path": "/x"}]}
        ).archive[1].path == "/x"

        schema = generate_composed_json_schema(result)
        prop = schema["$defs"][model.__name__]["properties"]["archive"]
        assert prop["description"] == "what this run records"
        assert "what this run records" in generate_composed_config_source(result)

    def test_list_receptor_min_length_parity(self, archiver_registrar, hook_registrar):
        class Batch(hook_registrar.test_base):
            __registry_key__ = "batch"
            __wiring__ = {"archive": "comp_archiver"}

            # (``label`` keeps the 1.x extractor off its single-parameter
            # shortcut, which crashes on ``list[X]`` under Python 3.10.)
            def __init__(
                self,
                archive: Annotated[
                    list[ArchiverProtocol],
                    Field(min_length=1, description="at least one archiver"),
                ],
                label: str = "batch",
            ): ...

        per_layer = build_layer_config(hook_registrar).per_key_models["batch"]
        with pytest.raises(ValidationError) as exc:
            per_layer(archive=[])
        assert _error_types(exc.value) == ["too_short"]

        result = build_composed_config(
            {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
        )
        model = result.layer_results["comp_hook"].per_key_models["batch"]
        assert model.model_fields["archive"].description == "at least one archiver"
        with pytest.raises(ValidationError) as exc:
            model(archive=[])
        assert _error_types(exc.value) == ["too_short"]
        assert len(model(archive=[{"archiver_id": "net"}]).archive) == 1

        prop = generate_composed_json_schema(result)["$defs"][model.__name__]["properties"]["archive"]
        assert '"minItems": 1' in __import__("json").dumps(prop)

    def test_scalar_receptor_pattern_narrows_keys(self, archiver_registrar, hook_registrar):
        class Sink(hook_registrar.test_base):
            __registry_key__ = "sink"
            __wiring__ = {"sink": "comp_archiver"}

            def __init__(
                self, sink: Annotated[str, Field(pattern="^n", description="where traces go")],
            ): ...

        registrars = {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar}

        # Literal mode: the pattern rejects the 'disk' key.
        selector = build_composed_config(registrars, inline_wiring=False)
        selector_model = selector.layer_results["comp_hook"].per_key_models["sink"]
        assert selector_model(sink="net").sink == "net"
        with pytest.raises(ValidationError) as exc:
            selector_model(sink="disk")
        assert _error_types(exc.value) == ["string_pattern_mismatch"]

        # Inline mode: the same key is refused, now as a config block.
        result = build_composed_config(registrars)
        model = result.layer_results["comp_hook"].per_key_models["sink"]
        assert model.__inline_wired__["sink"] == {
            "target_layer": "comp_archiver",
            "keys": ("net",),
            "container": False,
            "optional": False,
        }
        field = model.model_fields["sink"]
        assert field.description == "where traces go"
        assert field.metadata == []  # the key-string pattern was applied, then dropped
        assert model(sink={"archiver_id": "net", "host": "h"}).sink.host == "h"
        with pytest.raises(ValidationError):
            model(sink={"archiver_id": "disk"})

    def test_key_constraint_rejecting_every_key_raises(self, archiver_registrar, hook_registrar):
        class Nowhere(hook_registrar.test_base):
            __registry_key__ = "nowhere"
            __wiring__ = {"sink": "comp_archiver"}

            def __init__(self, sink: Annotated[str, Field(pattern="^x")]): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(
                {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
            )
        msg = str(exc.value)
        assert "Nowhere.sink" in msg
        assert "pattern='^x'" in msg
        assert "'net'" in msg and "'disk'" in msg

    def test_tagless_inline_input_is_rejected(self, archiver_registrar, hook_registrar):
        class Both(hook_registrar.test_base):
            __registry_key__ = "both"
            __wiring__ = {"one": "comp_archiver", "many": "comp_archiver"}

            def __init__(self, one: str, many: list[ArchiverProtocol]): ...

        result = build_composed_config(
            {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
        )
        model = result.layer_results["comp_hook"].per_key_models["both"]
        with pytest.raises(ValidationError) as exc:
            model(one={"host": "h"}, many=[{"archiver_id": "net"}])
        assert _error_types(exc.value) == ["union_tag_not_found"]
        with pytest.raises(ValidationError) as exc:
            model(one={"archiver_id": "net"}, many=[{"path": "/x"}])
        assert _error_types(exc.value) == ["union_tag_not_found"]

    def test_inline_wired_record(self, archiver_registrar, hook_registrar):
        class Owner(hook_registrar.test_base):
            __registry_key__ = "owner"
            __wiring__ = {
                "archive": "comp_archiver",
                "primary": ("comp_archiver", ["disk"]),
            }

            def __init__(
                self,
                primary: str,
                archive: list[ArchiverProtocol] | None = None,
            ): ...

        result = build_composed_config(
            {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
        )
        model = result.layer_results["comp_hook"].per_key_models["owner"]
        assert model.__inline_wired__ == {
            "primary": {
                "target_layer": "comp_archiver",
                "keys": ("disk",),
                "container": False,
                "optional": False,
            },
            "archive": {
                "target_layer": "comp_archiver",
                "keys": ("net", "disk"),
                "container": True,
                "optional": True,
            },
        }

    def test_optional_scalar_receptor(self, archiver_registrar, hook_registrar):
        class Maybe(hook_registrar.test_base):
            __registry_key__ = "maybe"
            __wiring__ = {"sink": "comp_archiver"}

            def __init__(self, sink: Optional[str] = None): ...

        result = build_composed_config(
            {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
        )
        model = result.layer_results["comp_hook"].per_key_models["maybe"]
        assert model.__inline_wired__["sink"]["optional"] is True
        assert model.__inline_wired__["sink"]["container"] is False
        assert model().sink is None
        assert model(sink={"archiver_id": "disk"}).sink.path == "/tmp"

    def test_literal_mode_records_nothing(self, archiver_registrar, hook_registrar):
        class Plain(hook_registrar.test_base):
            __registry_key__ = "plain"
            __wiring__ = {"sink": "comp_archiver"}

            def __init__(self, sink: str): ...

        result = build_composed_config(
            {"comp_hook": hook_registrar, "comp_archiver": archiver_registrar},
            inline_wiring=False,
        )
        model = result.layer_results["comp_hook"].per_key_models["plain"]
        assert not hasattr(model, "__inline_wired__")


# ---------------------------------------------------------------------------
# Rebuild by inheritance (K) — composed side
# ---------------------------------------------------------------------------


class _OwnerModel(BaseModel):
    """A per-key owner model with validators, config and wired fields."""

    model_config = ConfigDict(
        extra="forbid", title="Owner Title", json_schema_extra={"x-owner": True},
    )

    first: int = Field(1, description="first knob")
    archive: list[Literal["net", "disk"]] = Field(
        default_factory=list, description="what this run records",
    )
    sink: Annotated[Literal["net", "disk"], _TraceSafe()] = Field(
        "net", description="scalar receptor", examples=["net"], max_length=3,
    )
    last: str = "z"

    @model_validator(mode="after")
    def _rule(self) -> "_OwnerModel":
        if self.first > 10:
            raise ValueError("first must be <= 10")
        return self

    @field_validator("last")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


# What the extractor records on every model with wired fields: the registry
# of each wired field and its allowed keys (in ``Literal`` order).
_OwnerModel.__wired_fields__ = {"archive": "archiver", "sink": "archiver"}  # type: ignore[attr-defined]
_OwnerModel.__wired_keys__ = {"archive": ("net", "disk"), "sink": ("net", "disk")}  # type: ignore[attr-defined]
_OwnerModel.__degraded_fields__ = ["sentinel"]  # type: ignore[attr-defined]


class _NetCfg(BaseModel):
    archiver_id: Literal["net"] = "net"
    host: str = "localhost"


class _DiskCfg(BaseModel):
    archiver_id: Literal["disk"] = "disk"
    path: str = "/tmp"


def _archiver_result() -> LayerConfigResult:
    return LayerConfigResult(
        union_type=Annotated[Union[_NetCfg, _DiskCfg], Field(discriminator="archiver_id")],
        per_key_models={"net": _NetCfg, "disk": _DiskCfg},
        layer_name="archiver",
        discriminator_field="archiver_id",
    )


def _field_snapshot(model: type[BaseModel]) -> dict[str, Any]:
    return {
        name: (repr(fi), dict(getattr(fi, "_attributes_set", {})), list(fi.metadata))
        for name, fi in model.model_fields.items()
    }


class TestComposedRebuildByInheritance:
    def _rebuild(self) -> type[BaseModel]:
        owner = LayerConfigResult(
            union_type=_OwnerModel,
            per_key_models={"owner": _OwnerModel},
            layer_name="hook",
            discriminator_field="hook_id",
        )
        new = _replace_wired_literals_with_unions(owner, {"archiver": _archiver_result()})
        return new.per_key_models["owner"]

    def test_rebuilt_model_subclasses_the_per_key_model(self):
        model = self._rebuild()
        assert issubclass(model, _OwnerModel)
        assert model.__name__ == "_OwnerModel"
        assert list(model.model_fields) == ["first", "archive", "sink", "last"]

    def test_default_factory_and_description_survive(self):
        model = self._rebuild()
        archive = model.model_fields["archive"]
        assert not archive.is_required()
        assert archive.default_factory is list
        assert archive.description == "what this run records"
        assert model().archive == []
        first = model.model_fields["first"]
        assert first.description == "first knob" and first.default == 1

    def test_validators_and_model_config_survive(self):
        model = self._rebuild()
        with pytest.raises(ValidationError, match="first must be <= 10"):
            model(first=11)
        assert model(last="abc").last == "ABC"
        assert model.model_config["title"] == "Owner Title"
        assert model.model_config["extra"] == "forbid"
        assert model.model_json_schema()["x-owner"] is True

    def test_scalar_receptor_is_narrowed_and_filtered(self):
        model = self._rebuild()
        assert model.__inline_wired__["sink"]["keys"] == ("net",)  # max_length=3 rejects 'disk'
        sink = model.model_fields["sink"]
        assert sink.description == "scalar receptor"
        assert sink.examples is None
        assert sink.metadata == [_TraceSafe()]

    def test_custom_attributes_and_marker(self):
        model = self._rebuild()
        assert model.__dict__["__wired_fields__"] == {"archive": "archiver", "sink": "archiver"}
        assert model.__dict__["__degraded_fields__"] == ["sentinel"]
        assert model.__dict__[GENERATED_MARKER] is True
        assert GENERATED_MARKER not in model.model_fields
        assert "__inline_wired__" not in model.model_json_schema()["properties"]

    def test_source_model_is_not_mutated(self):
        before = _field_snapshot(_OwnerModel)
        self._rebuild()
        assert _field_snapshot(_OwnerModel) == before
        assert "__inline_wired__" not in _OwnerModel.__dict__

    def test_rebuild_helper_takes_field_definitions(self):
        definitions = {"first": (int, Field(5, description="redeclared"))}
        model = _rebuild_model_with_replacements(_OwnerModel, definitions)
        assert model.model_fields["first"].default == 5
        assert model.model_fields["first"].description == "redeclared"
        assert not hasattr(model, "__inline_wired__")
        assert model.__dict__[GENERATED_MARKER] is True


class _Tier3LLMConfig(BaseModel):
    model: str = "gpt-4"
    temperature: float = 0.7

    @model_validator(mode="after")
    def _rule(self) -> "_Tier3LLMConfig":
        if self.model.startswith("o1") and self.temperature != 1.0:
            raise ValueError("o1 models only support temperature=1.0")
        return self


class TestTier3InComposed:
    def test_tier3_validator_fires_at_top_level_and_inline(self):
        llm = create_registrar("comp_t3_llm", LLMProtocol, discriminator_field="provider")

        class _LLMBase(metaclass=llm.Meta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class O1(_LLMBase):
            __registry_key__ = "o1"
            __config_schema__ = _Tier3LLMConfig

        class Other(_LLMBase):
            __registry_key__ = "other"

            def __init__(self, *, model: str = "x"): ...

        agent = create_registrar("comp_t3_agent", AgentProtocol, discriminator_field="name")

        class _AgentBase(metaclass=agent.Meta):
            __abstract__ = True

            async def step(self, task: str) -> str:
                return ""

        class UsesLLM(_AgentBase):
            __registry_key__ = "uses"
            __wiring__ = {"llm": "comp_t3_llm"}

            def __init__(self, *, llm: str): ...

        result = build_composed_config({"comp_t3_llm": llm, "comp_t3_agent": agent})
        top = TypeAdapter(result.top_level_type)
        bad = {"provider": "o1", "model": "o1-mini", "temperature": 0.2}
        with pytest.raises(ValidationError, match="o1 models only support"):
            top.validate_python({"comp_t3_llm": [bad]})
        with pytest.raises(ValidationError, match="o1 models only support"):
            top.validate_python({"comp_t3_agent": [{"name": "uses", "llm": bad}]})
        ok = top.validate_python(
            {"comp_t3_agent": [{"name": "uses", "llm": {**bad, "temperature": 1.0}}]}
        )
        assert isinstance(ok.comp_t3_agent[0].llm, _Tier3LLMConfig)


# ---------------------------------------------------------------------------
# Nested (compound-discriminator) targets
# ---------------------------------------------------------------------------


class TestNestedTargets:
    @staticmethod
    def _nested_llm():
        reg = create_registrar(
            "comp_nt_llm", LLMProtocol,
            discriminator_fields=["model_type", "provider"], key_separator=".",
        )

        class _OpenAI(metaclass=reg.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs):
                super().__init__(**kwargs)

        class Official(_OpenAI):
            __registry_key__ = "openai.official"

            def __init__(self, endpoint: str = "e", **kwargs):
                super().__init__(**kwargs)

        class _Anthropic(metaclass=reg.Meta):
            __registry_key__ = "anthropic"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class Direct(_Anthropic):
            __registry_key__ = "anthropic.direct"

            def __init__(self, model: str = "claude"):
                self.model = model

        return reg

    @staticmethod
    def _agent(wiring: Any):
        reg = create_registrar("comp_nt_agent", AgentProtocol, discriminator_field="name")

        class _AgentBase(metaclass=reg.Meta):
            __abstract__ = True

            async def step(self, task: str) -> str:
                return ""

        class Pilot(_AgentBase):
            __registry_key__ = "pilot"
            __wiring__ = {"llm": wiring}

            def __init__(self, llm: Annotated[str, Field(description="the model")]): ...

        return reg

    def test_subset_sharing_level0_segment(self):
        """Used to crash: 'Value openai ... mapped to multiple choices'."""
        llm = self._nested_llm()
        agent = self._agent(("comp_nt_llm", ["openai.azure", "openai.official"]))
        result = build_composed_config({"comp_nt_llm": llm, "comp_nt_agent": agent})
        model = result.layer_results["comp_nt_agent"].per_key_models["pilot"]
        assert model.__inline_wired__["llm"] == {
            "target_layer": "comp_nt_llm",
            "keys": ("openai.azure", "openai.official"),
            "container": False,
            "optional": False,
        }
        assert model.model_fields["llm"].description == "the model"

        official = model(llm={"model_type": "openai", "provider": {"name": "official"}})
        assert official.llm.provider.endpoint == "e"
        azure = model(llm={"model_type": "openai", "provider": {"name": "azure"}})
        assert azure.llm.provider.deployment == "d"
        with pytest.raises(ValidationError) as exc:
            model(llm={"model_type": "anthropic", "provider": {"name": "direct"}})
        assert _error_types(exc.value) == ["union_tag_invalid"]

        ns: dict = {}
        exec(compile(generate_composed_config_source(result), "<nested-subset>", "exec"), ns)  # noqa: S102

    def test_full_nested_union_inline(self):
        llm = self._nested_llm()
        agent = self._agent("comp_nt_llm")
        result = build_composed_config({"comp_nt_llm": llm, "comp_nt_agent": agent})
        model = result.layer_results["comp_nt_agent"].per_key_models["pilot"]
        assert model.__inline_wired__["llm"]["keys"] == (
            "anthropic.direct", "openai.azure", "openai.official",
        )
        direct = model(llm={"model_type": "anthropic", "provider": {"name": "direct"}})
        assert direct.llm.model == "claude"

    def test_rebuilt_union_keeps_compound_discriminator(self):
        """The composed pass rebuilds a layer's union with ``_layer_union``."""
        result = build_layer_config(self._nested_llm())
        union = _layer_union(result, dict(result.per_key_models))
        adapter = TypeAdapter(union)
        validated = adapter.validate_python(
            {"model_type": "openai", "provider": {"name": "official"}}
        )
        assert validated.provider.name == "official"
        with pytest.raises(ValidationError):
            adapter.validate_python({"model_type": "openai", "provider": {"name": "bedrock"}})

    def test_keys_without_config_models_raise(self):
        """A non-leaf key of a nested layer has no config model: the layer
        itself refuses the registered 1-segment key (2.0), before any receptor
        could select it."""
        reg = create_registrar(
            "comp_nl_llm", LLMProtocol,
            discriminator_fields=["model_type", "provider"], key_separator=".",
        )

        class _Root(metaclass=reg.Meta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class OpenAIRoot(_Root):  # concrete and registered, but 1 segment: not a leaf
            __registry_key__ = "openai"

        class Azure(OpenAIRoot):
            __registry_key__ = "openai.azure"

        assert sorted(reg.keys()) == ["openai", "openai.azure"]

        agent = create_registrar("comp_nl_agent", AgentProtocol, discriminator_field="name")

        class _AgentBase(metaclass=agent.Meta):
            __abstract__ = True

            async def step(self, task: str) -> str:
                return ""

        class RootOnly(_AgentBase):
            __registry_key__ = "root_only"
            __wiring__ = {"llm": ("comp_nl_llm", ["openai"])}

            def __init__(self, llm: str): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"comp_nl_llm": reg, "comp_nl_agent": agent})
        msg = str(exc.value)
        assert "OpenAIRoot" in msg and "'openai' (" in msg
        assert "every registered key must have exactly 2 segments" in msg

    def test_nested_layer_with_wiring_is_refused(self, loop_registrar):
        reg = create_registrar(
            "comp_nw_llm", LLMProtocol,
            discriminator_fields=["model_type", "provider"], key_separator=".",
        )

        class _Base(metaclass=reg.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return ""

        class Azure(_Base):
            __registry_key__ = "openai.azure"
            __wiring__ = {"loop": "comp_loop"}

        with pytest.raises(ConfigDeclarationError, match="not supported for nested-mode layers yet"):
            build_composed_config({"comp_nw_llm": reg, "comp_loop": loop_registrar})


# ---------------------------------------------------------------------------
# Empty layers and the generated-model marker
# ---------------------------------------------------------------------------


class TestEmptyLayers:
    def test_every_empty_layer_is_named(self, llm_registrar):
        empty_a = create_registrar("comp_empty_a", LLMProtocol, discriminator_field="provider")
        empty_b = create_registrar("comp_empty_b", LLMProtocol, discriminator_field="provider")
        with pytest.raises(EmptyLayerError) as exc:
            build_composed_config(
                {"comp_empty_a": empty_a, "comp_llm": llm_registrar, "comp_empty_b": empty_b},
            )
        assert exc.value.layer_names == ["comp_empty_a", "comp_empty_b"]
        assert "'comp_empty_a'" in str(exc.value) and "'comp_empty_b'" in str(exc.value)


class TestGeneratedMarker:
    def test_rebuilt_and_top_level_models_are_marked(
        self, llm_registrar, loop_registrar, agent_registrar,
    ):
        import json

        result = build_composed_config(
            {"comp_llm": llm_registrar, "comp_loop": loop_registrar, "comp_agent": agent_registrar},
        )
        for layer in result.layer_results.values():
            for model in layer.per_key_models.values():
                assert model.__dict__.get(GENERATED_MARKER) is True
        top = result.top_level_type
        assert top.__dict__.get(GENERATED_MARKER) is True
        assert GENERATED_MARKER not in top.model_fields
        assert GENERATED_MARKER not in json.dumps(generate_composed_json_schema(result))
