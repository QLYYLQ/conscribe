"""Registrars and base classes shared by the isolation tests.

They live in an importable module (not in the test file) so the throwaway
modules the tests write to ``tmp_path`` can import them by name, the way a
real plugin module imports its framework's base classes.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class ActionProto(Protocol):
    def act(self) -> str: ...


@runtime_checkable
class OwnerProto(Protocol):
    def own(self) -> str: ...


@runtime_checkable
class StrategyProto(Protocol):
    def plan(self) -> str: ...


Act = create_registrar(
    "iso_act", ActionProto, key_separator=".", discriminator_field="name",
)
Owner = create_registrar("iso_owner", OwnerProto, discriminator_field="name")
Strat = create_registrar("iso_strategy", StrategyProto, discriminator_field="name")


class ActionBase(metaclass=Act.Meta):
    __abstract__ = True

    def act(self) -> str:
        return type(self).__name__


class BrowserClick(ActionBase):
    __registry_key__ = "browser.click"


class OwnerBase(metaclass=Owner.Meta):
    __abstract__ = True

    def own(self) -> str:
        return type(self).__name__


class GoodOwner(OwnerBase):
    __registry_key__ = "good"
    __wiring__ = {"action": ("iso_act", ["click"])}

    def __init__(self, action: str = "browser.click") -> None:
        self.action = action


class StrategyBase(metaclass=Strat.Meta):
    __abstract__ = True

    def plan(self) -> str:
        return type(self).__name__


class Fast(StrategyBase):
    pass


def make_strategy(key: str) -> type:
    """Define (and so register) a strategy class from inside a function."""
    return type(f"Made_{key}", (StrategyBase,), {"__registry_key__": key})


class Plain:
    """Registered by no one at import; tests apply ``Strat.register()`` to it.

    ``helper`` is wired but not an ``__init__`` parameter, so ``register()``
    installs a ``WiredField`` on the class.
    """

    __wiring__ = {"helper": "iso_act"}

    def plan(self) -> str:
        return "plain"
