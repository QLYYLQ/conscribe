"""Shared fixtures for conscribe test suite.

Provides Protocol definitions, registrar factories, and cleanup fixtures
modeled on real browseruse-bench scenarios (Agent, LLM, Browser Provider layers).
"""
from __future__ import annotations

import pytest
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


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
# Registrar fixtures (with cleanup)
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_registrar():
    """Create Agent layer registrar (simulates Alice's first step)."""
    registrar = create_registrar(
        "agent",
        AgentProtocol,
        strip_suffixes=["Agent"],
        discriminator_field="name",
    )
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)


@pytest.fixture
def llm_registrar():
    """Create LLM Provider layer registrar."""
    registrar = create_registrar(
        "llm",
        LLMProtocol,
        strip_suffixes=["LLM", "Provider"],
        discriminator_field="provider",
    )
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)


@pytest.fixture
def browser_registrar():
    """Create Browser Provider layer registrar."""
    registrar = create_registrar(
        "browser",
        BrowserProtocol,
        strip_suffixes=["Provider", "Browser"],
        discriminator_field="id",
    )
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)


@pytest.fixture
def simple_registrar():
    """Minimal registrar for basic unit tests."""
    registrar = create_registrar("test", SimpleProtocol)
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)


@pytest.fixture
def evaluator_registrar():
    """Create Evaluator layer registrar."""
    registrar = create_registrar(
        "evaluator",
        EvaluatorProtocol,
        strip_suffixes=["Evaluator"],
        discriminator_field="evaluator",
    )
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)
