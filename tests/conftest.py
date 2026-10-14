"""Shared fixtures for conscribe test suite.

Provides Protocol definitions, registrar factories, and cleanup fixtures
modeled on real browseruse-bench scenarios (Agent, LLM, Browser Provider layers).
"""
from __future__ import annotations

import pytest
from typing import Protocol, runtime_checkable

from conscribe import create_registrar
from conscribe.testing import isolated_registries

# The opt-in isolation fixture (``conscribe_isolated``) and pytester for the
# plugin's own tests.  Loaded here, never with ``-p``: ``-p`` imports the
# plugin before pytest-cov starts measuring.
pytest_plugins = ["conscribe.testing.pytest_plugin", "pytester"]


# ---------------------------------------------------------------------------
# Protocols (simulate benchmark framework interfaces)
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentProtocol(Protocol):
    """Agent layer interface -- models browseruse-bench AgentProtocol."""

    async def step(self, task: str) -> str: ...

    def reset(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    """LLM Provider layer interface."""

    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class BrowserProtocol(Protocol):
    """Browser Provider layer interface."""

    def connect(self) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class SimpleProtocol(Protocol):
    """Minimal Protocol for basic unit tests."""

    def do_work(self) -> str: ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Evaluator layer interface."""

    def evaluate(self, result: str) -> float: ...


# ---------------------------------------------------------------------------
# Non-runtime-checkable Protocol (for negative tests)
# ---------------------------------------------------------------------------

class NotRuntimeCheckable(Protocol):
    """Protocol WITHOUT @runtime_checkable -- used to test InvalidProtocolError."""

    def do_something(self) -> None: ...


# ---------------------------------------------------------------------------
# Registrar fixtures (isolated: the registrar, everything registered with it
# during the test, and any registry its name shadowed are restored afterwards)
# ---------------------------------------------------------------------------

def _isolated_registrar(name: str, protocol: type, **options):
    with isolated_registries():
        yield create_registrar(name, protocol, **options)


@pytest.fixture
def agent_registrar():
    """Create Agent layer registrar (simulates Alice's first step)."""
    yield from _isolated_registrar(
        "agent",
        AgentProtocol,
        strip_suffixes=["Agent"],
        discriminator_field="name",
    )


@pytest.fixture
def llm_registrar():
    """Create LLM Provider layer registrar."""
    yield from _isolated_registrar(
        "llm",
        LLMProtocol,
        strip_suffixes=["LLM", "Provider"],
        discriminator_field="provider",
    )


@pytest.fixture
def browser_registrar():
    """Create Browser Provider layer registrar."""
    yield from _isolated_registrar(
        "browser",
        BrowserProtocol,
        strip_suffixes=["Provider", "Browser"],
        discriminator_field="id",
    )


@pytest.fixture
def simple_registrar():
    """Minimal registrar for basic unit tests."""
    yield from _isolated_registrar("test", SimpleProtocol)


@pytest.fixture
def evaluator_registrar():
    """Create Evaluator layer registrar."""
    yield from _isolated_registrar(
        "evaluator",
        EvaluatorProtocol,
        strip_suffixes=["Evaluator"],
        discriminator_field="evaluator",
    )
