"""Local LLM — Path B: bridge from external class."""
from __future__ import annotations

from mock_framework.layers import LLMRegistrar


# Simulate an external library class
class _ExternalOllamaClient:
    """Pretend this comes from an external library."""

    def __init__(self, *, model: str = "llama3", host: str = "localhost:11434"):
        self.model = model
        self.host = host

    async def chat(self, messages: list[dict]) -> str:
        return f"ollama({self.model}): response"


# Bridge the external class into our registry
OllamaBridge = LLMRegistrar.bridge(_ExternalOllamaClient)


class LocalLLM(OllamaBridge):
    """Local LLM via Ollama.

    Args:
        model: Ollama model name.
        host: Ollama server address.
        num_ctx: Context window size.
    """
    __registry_key__ = "local"

    def __init__(
        self,
        *,
        model: str = "llama3",
        host: str = "localhost:11434",
        num_ctx: int = 4096,
    ):
        super().__init__(model=model, host=host)
        self.num_ctx = num_ctx
