"""Custom agent — Path C: @register decorator."""
from __future__ import annotations

from mock_framework.layers import AgentRegistrar


@AgentRegistrar.register("custom_v2")
class CustomAgent:
    """A manually registered custom agent.

    Args:
        strategy: The solving strategy to use.
        verbose: Enable verbose logging.
    """

    def __init__(self, *, strategy: str = "default", verbose: bool = False):
        self.strategy = strategy
        self.verbose = verbose

    async def step(self, task: str) -> str:
        return f"custom({self.strategy}): {task}"

    def reset(self) -> None:
        pass
