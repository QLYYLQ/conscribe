"""Fixture module reproducing the real downstream shapes that broke ``.pyi`` output.

Kept in its own file (not defined inside a test function) because the stub
generator reads the *source file* to resolve annotations and imports, and
because ``from __future__ import annotations`` must apply at module scope to
produce the PEP 563 string annotations the downstream package has.

Every construct here is copied from a shape that actually appears in the
``spectragent`` browser capability:

- a ``@dataclass`` with ``field(default_factory=...)``  → ``<factory>``
- a wired receptor whose descriptor becomes a dataclass default → ``WiredField(...)``
- a pydantic ``FieldInfo`` / model instance as a default
- module-private helper types referenced only in annotations
- ``async def`` methods, including ``@staticmethod`` / ``@classmethod`` and
  an async generator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class ProviderProtocol(Protocol):
    def connect(self) -> str: ...


@runtime_checkable
class CapabilityProtocol(Protocol):
    def describe(self) -> str: ...


class LocalProvider:
    def connect(self) -> str:
        return "local"


class RemoteProvider:
    def connect(self) -> str:
        return "remote"


class _ObserveCycleCache:
    """Module-private: referenced in annotations, never registered."""

    hits: int = 0


class _StateEventOutcome:
    """Module-private: appears only as a return annotation."""


class Settings(BaseModel):
    """Same-module pydantic model used as a default value."""

    retries: int = Field(default=3, description="retry count")
    tags: list[str] = Field(default_factory=list)


@dataclass
class DataclassCapability:
    """Dataclass receptor: ``default_factory`` and a ``WiredField`` default."""

    __wiring__ = {"backend": "test_stub_provider"}

    profile: str = "standard"
    action_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)
    limits: Any = Field(default=None, description="a raw FieldInfo default")

    async def setup(self, config: dict[str, Any] | None = None) -> None: ...

    async def observe(self, cache: _ObserveCycleCache) -> _StateEventOutcome: ...

    async def stream(self) -> AsyncIterator[str]:  # async generator
        yield "chunk"

    @staticmethod
    async def probe() -> bool:
        return True

    @classmethod
    async def build(cls) -> "DataclassCapability":
        return cls()

    def teardown(self) -> None:
        """Genuinely synchronous — must stay a plain ``def``."""

    def describe(self) -> str:
        return "dataclass"


class PlainCapability:
    """Non-dataclass receptor with a model instance default."""

    __wiring__ = {"backend": ("test_stub_provider", ["local"])}

    def __init__(self, opts: Settings = Settings(), retries: int = 3) -> None:
        self.opts = opts
        self.retries = retries

    async def resolve_fragment(self, key: str, **kwargs: Any) -> Any: ...

    def is_alive(self) -> bool:
        return True

    def describe(self) -> str:
        return "plain"
