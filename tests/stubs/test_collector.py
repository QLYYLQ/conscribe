"""Tests for conscribe.stubs.collector."""
from __future__ import annotations

import pytest
from typing import Protocol, runtime_checkable

from conscribe.registration.registry import LayerRegistry, _REGISTRY_INDEX, _deregister
from conscribe.stubs.collector import (
    ClassStubInfo,
    InjectedAttr,
    collect_class_stub_info,
    narrowest_common_base,
)


# ── Fixtures ─────────────────────────────────────────────────────


@runtime_checkable
class EnvProtocol(Protocol):
    def setup(self) -> None: ...


class Terminal(EnvProtocol):
    def setup(self) -> None: ...


class BashTerminal(Terminal):
    def setup(self) -> None: ...


class ZshTerminal(Terminal):
    def setup(self) -> None: ...


class FileSystem(EnvProtocol):
    def setup(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def generate(self) -> str: ...


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("test_"):
            _deregister(name)


def _make_env_registry() -> LayerRegistry:
    reg = LayerRegistry("test_env", EnvProtocol)
    reg.add("terminal.bash", BashTerminal)
    reg.add("terminal.zsh", ZshTerminal)
    reg.add("filesystem", FileSystem)
    return reg


def _make_llm_registry() -> LayerRegistry:
    reg = LayerRegistry("test_llm", LLMProtocol)
    dummy = type("OpenAI", (), {"generate": lambda self: ""})
    reg.add("openai", dummy)
    return reg


# ── narrowest_common_base ────────────────────────────────────────


class TestNarrowestCommonBase:
    def test_empty_list(self):
        assert narrowest_common_base([], str) is str

    def test_single_class(self):
        assert narrowest_common_base([BashTerminal], EnvProtocol) is BashTerminal

    def test_siblings_share_parent(self):
        result = narrowest_common_base([BashTerminal, ZshTerminal], EnvProtocol)
        assert result is Terminal

    def test_different_branches_fallback(self):
        result = narrowest_common_base([BashTerminal, FileSystem], object)
        # Both descend from EnvProtocol
        assert result is EnvProtocol

    def test_parent_and_child(self):
        result = narrowest_common_base([Terminal, BashTerminal], EnvProtocol)
        assert result is Terminal

    def test_identical_classes(self):
        result = narrowest_common_base([BashTerminal, BashTerminal], EnvProtocol)
        assert result is BashTerminal


# ── collect_class_stub_info ──────────────────────────────────────


class TestCollectClassStubInfo:
    def test_no_wiring_returns_none(self):
        class Plain:
            pass

        assert collect_class_stub_info(Plain) is None

    def test_wired_in_init_returns_none(self):
        """Wired field that's already in __init__ → not injected → None."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def __init__(self, env: str = "default"):
                self.env = env

        assert collect_class_stub_info(Agent) is None

    def test_injected_wiring_basic(self):
        """Wired field not in __init__ → injected → ClassStubInfo."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def __init__(self, name: str = "a"):
                self.name = name

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.class_name == "Agent"
        assert len(info.injected_attrs) == 1
        assert info.injected_attrs[0].name == "env"
        assert info.injected_attrs[0].registry_name == "test_env"

    def test_injected_type_uses_protocol(self):
        """Mode 1 (all keys) → type is registry protocol."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

        info = collect_class_stub_info(Agent)
        assert info is not None
        # All keys span Terminal + FileSystem → common base is EnvProtocol
        assert info.injected_attrs[0].resolved_type is EnvProtocol

    def test_injected_type_narrows_for_subset(self):
        """Mode 2 (explicit subset) → narrowed to common base."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": ("test_env", ["terminal.bash", "terminal.zsh"])}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is Terminal

    def test_injected_type_single_key(self):
        """Mode 2 single key → exact class."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": ("test_env", ["terminal.bash"])}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is BashTerminal

    def test_mode3_literal_uses_str(self):
        """Mode 3 (literal list) → str type."""

        class Agent:
            __wiring__ = {"browser": ["chromium", "firefox"]}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is str
        assert info.injected_attrs[0].registry_name is None

    def test_three_element_tuple_narrows(self):
        """Mode 2 with 3-element tuple → narrows over required+optional."""
        _make_env_registry()

        class Agent:
            __wiring__ = {
                "env": ("test_env", ["terminal.bash"], ["terminal.zsh"])
            }

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is Terminal

    def test_mixed_wired_and_non_wired(self):
        """Only injected fields appear, not those in __init__."""
        _make_env_registry()
        _make_llm_registry()

        class Agent:
            __wiring__ = {"env": "test_env", "llm": "test_llm"}

            def __init__(self, env: str = "default"):
                self.env = env

        info = collect_class_stub_info(Agent)
        assert info is not None
        # Only llm is injected (env is in __init__)
        assert len(info.injected_attrs) == 1
        assert info.injected_attrs[0].name == "llm"

    def test_no_own_init(self):
        """Class without own __init__ → init_signature is None."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.init_signature is None

    def test_own_init_captured(self):
        """Class with own __init__ → init_signature is a string."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def __init__(self, name: str = "default"):
                self.name = name

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.init_signature is not None
        assert "name" in info.init_signature
        assert "-> None" in info.init_signature

    def test_own_methods_collected(self):
        """Own methods appear in the stub info."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def step(self, task: str) -> str:
                return task

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert len(info.methods) == 1
        assert info.methods[0].name == "step"
        assert "task" in info.methods[0].signature
