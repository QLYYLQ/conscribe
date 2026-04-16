"""BrowserUse agent — Path A: metaclass inheritance."""
from __future__ import annotations

from mock_framework.layers import AgentRegistrar


class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True

    async def step(self, task: str) -> str:
        return ""

    def reset(self) -> None:
        pass


class BrowserUseAgent(BaseAgent):
    """BrowserUse agent for web automation.

    Args:
        headless: Run browser in headless mode.
        timeout: Navigation timeout in seconds.
    """

    def __init__(self, *, headless: bool = True, timeout: int = 30):
        self.headless = headless
        self.timeout = timeout

    async def step(self, task: str) -> str:
        return f"browser_use: {task}"
