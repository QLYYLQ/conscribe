"""Anthropic provider — Path A: metaclass inheritance."""
from __future__ import annotations

from mock_framework.llm.openai import BaseLLM


class AnthropicProvider(BaseLLM):
    """Anthropic LLM provider.

    Args:
        model_id: Model identifier (e.g. claude-sonnet-4-20250514).
        max_tokens: Maximum output tokens.
    """
    __registry_key__ = "anthropic"

    def __init__(self, *, model_id: str = "claude-sonnet-4-20250514", max_tokens: int = 8192):
        self.model_id = model_id
        self.max_tokens = max_tokens

    async def chat(self, messages: list[dict]) -> str:
        return f"anthropic({self.model_id}): response"
