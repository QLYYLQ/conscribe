"""Protocol definitions for all layers."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentProtocol(Protocol):
    """Agent layer interface."""

    async def step(self, task: str) -> str: ...

    def reset(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    """LLM provider layer interface."""

    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Evaluator layer interface."""

    def evaluate(self, result: str) -> float: ...


@runtime_checkable
class ToolProtocol(Protocol):
    """Tool layer interface — supports hierarchical keys."""

    def execute(self, **kwargs: object) -> str: ...
