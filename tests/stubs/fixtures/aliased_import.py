"""Fixture: dataclass with aliased pydantic Field import.

Mirrors the spectragent ``BrowserUseAgent`` pattern that exposed two
stub-generation regressions:

1. ``from pydantic import Field as PydanticField`` was being mangled
   into ``from pydantic.fields import PydanticField`` (invalid — the
   alias only exists locally).
2. Public dataclass fields were missing from the generated class body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field as PydanticField


@dataclass
class AliasedDataclass:
    """Dataclass that uses an aliased pydantic Field import."""

    __wiring__: ClassVar[dict[str, Any]] = {}

    max_steps: Annotated[int, PydanticField(description="Maximum steps")] = 100
    use_vision: Annotated[bool, PydanticField(description="Include screenshots")] = False
    vision_detail: Annotated[
        Literal["auto", "low", "high"],
        PydanticField(description="Screenshot detail"),
    ] = "low"
    _internal: str = field(default="", init=False, repr=False)

    def step(self, value: Annotated[int, PydanticField(description="Step")]) -> int:
        return value
