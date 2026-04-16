"""Registrar definitions for all layers.

This is where create_registrar() is called — the `scan` command
should find these definitions via AST analysis.
"""
from __future__ import annotations

from conscribe import create_registrar

from mock_framework.protocols import (
    AgentProtocol,
    EvaluatorProtocol,
    LLMProtocol,
    ToolProtocol,
)

AgentRegistrar = create_registrar(
    "mock_agent",
    AgentProtocol,
    strip_suffixes=["Agent"],
    discriminator_field="name",
)

LLMRegistrar = create_registrar(
    "mock_llm",
    LLMProtocol,
    strip_suffixes=["LLM", "Provider"],
    discriminator_field="provider",
)

EvaluatorRegistrar = create_registrar(
    "mock_evaluator",
    EvaluatorProtocol,
    strip_suffixes=["Evaluator"],
    discriminator_field="evaluator",
)

ToolRegistrar = create_registrar(
    "mock_tool",
    ToolProtocol,
    key_separator=".",
    discriminator_field="tool",
)
