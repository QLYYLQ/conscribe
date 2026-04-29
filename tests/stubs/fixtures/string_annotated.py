"""Fixture: class with PEP 563 string annotations and Annotated metadata."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field


class StringAnnotated:
    """Mimics spectragent agents — `from __future__ import annotations`
    causes every parameter's annotation to be captured as a *string* by
    ``inspect.signature``.
    """

    def __init__(
        self,
        max_steps: Annotated[int, Field(description="Max steps")] = 100,
        mode: Annotated[Literal["a", "b"], Field(description="Mode")] = "a",
    ) -> None:
        self.max_steps = max_steps
        self.mode = mode

    def echo(self, value: Annotated[str, Field(description="Input")]) -> str:
        return value
