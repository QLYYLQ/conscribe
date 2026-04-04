"""Tests for the wiring module: parse, MRO merge, and resolve."""
from __future__ import annotations

import pytest
from typing import Optional, Protocol, runtime_checkable

from conscribe.exceptions import WiringResolutionError
from conscribe.registration.registry import LayerRegistry, _REGISTRY_INDEX, _deregister
from conscribe.wiring import (
    ResolvedWiring,
    WiringSpec,
    collect_wiring_from_mro,
    parse_wiring,
    resolve_wiring,
)


# ---------------------------------------------------------------------------
# Fixtures: create test registries
# ---------------------------------------------------------------------------


@runtime_checkable
class LoopProtocol(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def generate(self) -> str: ...


@pytest.fixture(autouse=True)
def _cleanup_registries():
    """Clean up test registries after each test."""
    yield
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("test_"):
            _deregister(name)


def _make_registry(name: str, protocol: type, keys: list[str]) -> LayerRegistry:
    """Helper to create a populated test registry."""
    reg = LayerRegistry(name, protocol)
    for key in keys:
        # Create a dummy class for each key
        dummy = type(f"Dummy_{key}", (), {"run": lambda self: None, "generate": lambda self: ""})
        reg.add(key, dummy)
    return reg


# ---------------------------------------------------------------------------
# Tests: collect_wiring_from_mro
# ---------------------------------------------------------------------------


class TestCollectWiringFromMRO:
    def test_no_wiring(self):
        class Plain:
            pass

        assert collect_wiring_from_mro(Plain) == {}

    def test_single_class(self):
        class Agent:
            __wiring__ = {"loop": "agent_loop"}

        assert collect_wiring_from_mro(Agent) == {"loop": "agent_loop"}

    def test_inheritance_merge(self):
        class Base:
            __wiring__ = {"loop": "agent_loop", "llm": "llm"}

        class Child(Base):
            __wiring__ = {"loop": ("agent_loop", ["react"])}

        result = collect_wiring_from_mro(Child)
        assert result == {
            "loop": ("agent_loop", ["react"]),  # child overrides
            "llm": "llm",  # parent preserved
        }

    def test_none_exclusion(self):
        class Base:
            __wiring__ = {"loop": "agent_loop", "llm": "llm"}

        class Offline(Base):
            __wiring__ = {"llm": None}

        result = collect_wiring_from_mro(Offline)
        assert result == {"loop": "agent_loop"}
        assert "llm" not in result

    def test_deep_inheritance_chain(self):
        class A:
            __wiring__ = {"x": "reg_x", "y": "reg_y"}

        class B(A):
            __wiring__ = {"y": ("reg_y", ["b1"])}

        class C(B):
            __wiring__ = {"z": ["c1", "c2"]}

        result = collect_wiring_from_mro(C)
        assert result == {
            "x": "reg_x",
            "y": ("reg_y", ["b1"]),
            "z": ["c1", "c2"],
        }

    def test_no_wiring_inheritance(self):
        """Child without __wiring__ inherits nothing via __dict__ (MRO walk)."""
        class Base:
            __wiring__ = {"loop": "agent_loop"}

        class Child(Base):
            pass

        # collect_wiring_from_mro uses __dict__ so it DOES see parent's __wiring__
        result = collect_wiring_from_mro(Child)
        assert result == {"loop": "agent_loop"}


# ---------------------------------------------------------------------------
# Tests: parse_wiring
# ---------------------------------------------------------------------------


class TestParseWiring:
    def test_mode1_string(self):
        class Agent:
            __wiring__ = {"loop": "agent_loop"}

        specs = parse_wiring(Agent)
        assert len(specs) == 1
        assert specs[0] == WiringSpec(
            param_name="loop", registry_name="agent_loop", allowed_keys=None
        )

    def test_mode2_tuple(self):
        class Agent:
            __wiring__ = {"llm": ("llm_reg", ["openai", "anthropic"])}

        specs = parse_wiring(Agent)
        assert len(specs) == 1
        assert specs[0] == WiringSpec(
            param_name="llm",
            registry_name="llm_reg",
            allowed_keys=("openai", "anthropic"),
        )

    def test_mode3_list(self):
        class Agent:
            __wiring__ = {"browser": ["chromium", "firefox"]}

        specs = parse_wiring(Agent)
        assert len(specs) == 1
        assert specs[0] == WiringSpec(
            param_name="browser",
            registry_name="",
            allowed_keys=("chromium", "firefox"),
        )

    def test_mixed_modes(self):
        class Agent:
            __wiring__ = {
                "loop": "agent_loop",
                "llm": ("llm_reg", ["openai"]),
                "browser": ["chromium"],
            }

        specs = parse_wiring(Agent)
        assert len(specs) == 3
        param_names = {s.param_name for s in specs}
        assert param_names == {"loop", "llm", "browser"}

    def test_invalid_type_raises(self):
        class Bad:
            __wiring__ = {"x": 42}

        with pytest.raises(TypeError, match="Invalid __wiring__ entry"):
            parse_wiring(Bad)

    def test_invalid_tuple_raises(self):
        class Bad:
            __wiring__ = {"x": (42, "not_a_list")}

        with pytest.raises(TypeError, match="tuple mode expects"):
            parse_wiring(Bad)

    def test_mode2_tuple_with_optional(self):
        class Agent:
            __wiring__ = {"obs": ("obs_reg", ["terminal"], ["filesystem"])}

        specs = parse_wiring(Agent)
        assert len(specs) == 1
        assert specs[0] == WiringSpec(
            param_name="obs",
            registry_name="obs_reg",
            allowed_keys=("terminal",),
            optional_keys=("filesystem",),
        )

    def test_mode2_tuple_with_empty_optional(self):
        class Agent:
            __wiring__ = {"obs": ("obs_reg", ["terminal"], [])}

        specs = parse_wiring(Agent)
        assert len(specs) == 1
        assert specs[0].allowed_keys == ("terminal",)
        assert specs[0].optional_keys == ()

    def test_mode2_backward_compat_no_optional(self):
        """2-element tuple should have optional_keys=None."""
        class Agent:
            __wiring__ = {"llm": ("llm_reg", ["openai"])}

        specs = parse_wiring(Agent)
        assert specs[0].optional_keys is None

    def test_invalid_3tuple_third_element_raises(self):
        class Bad:
            __wiring__ = {"x": ("reg", ["a"], "not_a_list")}

        with pytest.raises(TypeError, match="third element.*must be list"):
            parse_wiring(Bad)

    def test_empty_wiring(self):
        class Agent:
            __wiring__ = {}

        assert parse_wiring(Agent) == []


# ---------------------------------------------------------------------------
# Tests: resolve_wiring
# ---------------------------------------------------------------------------


class TestResolveWiring:
    def test_mode1_auto_discovery(self):
        _make_registry("test_loop", LoopProtocol, ["react", "codeact", "plan_act"])

        class Agent:
            __wiring__ = {"loop": "test_loop"}

        resolved = resolve_wiring(Agent)
        assert "loop" in resolved
        assert sorted(resolved["loop"].allowed_keys) == ["codeact", "plan_act", "react"]
        assert resolved["loop"].registry_name == "test_loop"

    def test_mode2_explicit_subset(self):
        _make_registry("test_llm", LLMProtocol, ["openai", "anthropic", "deepseek"])

        class Agent:
            __wiring__ = {"llm": ("test_llm", ["openai", "anthropic"])}

        resolved = resolve_wiring(Agent)
        assert resolved["llm"].allowed_keys == ["openai", "anthropic"]
        assert resolved["llm"].registry_name == "test_llm"

    def test_mode3_literal_list(self):
        class Agent:
            __wiring__ = {"browser": ["chromium", "firefox"]}

        resolved = resolve_wiring(Agent)
        assert resolved["browser"].allowed_keys == ["chromium", "firefox"]
        assert resolved["browser"].registry_name is None

    def test_registry_not_found_raises(self):
        class Agent:
            __wiring__ = {"loop": "nonexistent_registry"}

        with pytest.raises(WiringResolutionError, match="not found"):
            resolve_wiring(Agent)

    def test_empty_registry_raises(self):
        LayerRegistry("test_empty", LoopProtocol)  # empty registry

        class Agent:
            __wiring__ = {"loop": "test_empty"}

        with pytest.raises(WiringResolutionError, match="empty"):
            resolve_wiring(Agent)

    def test_mode2_missing_key_raises(self):
        _make_registry("test_llm2", LLMProtocol, ["openai"])

        class Agent:
            __wiring__ = {"llm": ("test_llm2", ["openai", "nonexistent"])}

        with pytest.raises(WiringResolutionError, match="nonexistent"):
            resolve_wiring(Agent)

    def test_mode3_empty_list_raises(self):
        class Agent:
            __wiring__ = {"browser": []}

        with pytest.raises(WiringResolutionError, match="non-empty"):
            resolve_wiring(Agent)

    def test_no_wiring_returns_empty(self):
        class Plain:
            pass

        assert resolve_wiring(Plain) == {}

    def test_inheritance_with_resolution(self):
        _make_registry("test_loop2", LoopProtocol, ["react", "codeact"])
        _make_registry("test_llm3", LLMProtocol, ["openai", "anthropic"])

        class Base:
            __wiring__ = {"loop": "test_loop2", "llm": "test_llm3"}

        class Child(Base):
            __wiring__ = {"loop": ("test_loop2", ["react"])}

        resolved = resolve_wiring(Child)
        assert resolved["loop"].allowed_keys == ["react"]
        assert sorted(resolved["llm"].allowed_keys) == ["anthropic", "openai"]

    def test_none_exclusion_with_resolution(self):
        _make_registry("test_loop3", LoopProtocol, ["react"])
        _make_registry("test_llm4", LLMProtocol, ["openai"])

        class Base:
            __wiring__ = {"loop": "test_loop3", "llm": "test_llm4"}

        class Offline(Base):
            __wiring__ = {"llm": None}

        resolved = resolve_wiring(Offline)
        assert "loop" in resolved
        assert "llm" not in resolved

    def test_mode2_with_optional_keys(self):
        _make_registry("test_obs", LoopProtocol, ["terminal", "filesystem", "browser"])

        class Agent:
            __wiring__ = {"obs": ("test_obs", ["terminal"], ["filesystem"])}

        resolved = resolve_wiring(Agent)
        assert resolved["obs"].allowed_keys == ["terminal", "filesystem"]
        assert resolved["obs"].optional_keys == ["filesystem"]
        assert resolved["obs"].registry_name == "test_obs"

    def test_mode2_optional_key_missing_raises(self):
        _make_registry("test_obs2", LoopProtocol, ["terminal"])

        class Agent:
            __wiring__ = {"obs": ("test_obs2", ["terminal"], ["nonexistent"])}

        with pytest.raises(WiringResolutionError, match="Optional keys not found"):
            resolve_wiring(Agent)

    def test_mode2_backward_compat_resolved(self):
        """2-element tuple should produce optional_keys=None in ResolvedWiring."""
        _make_registry("test_llm_bc", LLMProtocol, ["openai", "anthropic"])

        class Agent:
            __wiring__ = {"llm": ("test_llm_bc", ["openai"])}

        resolved = resolve_wiring(Agent)
        assert resolved["llm"].optional_keys is None
        assert resolved["llm"].allowed_keys == ["openai"]
