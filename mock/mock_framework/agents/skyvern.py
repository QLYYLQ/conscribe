"""Skyvern agent — Path A: metaclass inheritance."""
from __future__ import annotations

from mock_framework.agents.browser_use import BaseAgent


class SkyvernAgent(BaseAgent):
    """Skyvern cloud agent.

    Args:
        api_key: Skyvern API key.
        max_steps: Maximum steps per task.
    """

    def __init__(self, *, api_key: str, max_steps: int = 50):
        self.api_key = api_key
        self.max_steps = max_steps

    async def step(self, task: str) -> str:
        return f"skyvern: {task}"
