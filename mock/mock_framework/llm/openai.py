"""OpenAI provider — Path A: metaclass inheritance."""
from __future__ import annotations

from mock_framework.layers import LLMRegistrar


class BaseLLM(metaclass=LLMRegistrar.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class OpenAIProvider(BaseLLM):
    """OpenAI LLM provider.

    Args:
        model_id: Model identifier (e.g. gpt-4o).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
    """
    __registry_key__ = "openai"

    def __init__(
        self,
        *,
        model_id: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: list[dict]) -> str:
        return f"openai({self.model_id}): response"
