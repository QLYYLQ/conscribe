"""Shared fixtures for config typing tests.

Provides helper factories and registrar fixtures used across
test_docstring.py and test_extractor.py.
"""
from __future__ import annotations

import pytest
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


# ---------------------------------------------------------------------------
# Protocols (reusable across config test modules)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProtocol(Protocol):
    """LLM Provider layer interface for config tests."""

    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Agent layer interface for config tests."""

    async def step(self, task: str) -> str: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Registrar fixtures (with cleanup)
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_registrar():
    """Create LLM Provider layer registrar for config tests."""
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
def agent_registrar():
    """Create Agent layer registrar for config tests."""
    registrar = create_registrar(
        "agent",
        AgentProtocol,
        strip_suffixes=["Agent"],
        discriminator_field="name",
    )
    yield registrar
    for key in list(registrar.keys()):
        registrar.unregister(key)
