# pyright: reportUndefinedVariable=false
"""Fixture: class with a string annotation that cannot be resolved.

The annotation references a name that exists in neither runtime
globals nor any TYPE_CHECKING block — the stub generator must skip
it silently rather than crashing.
"""
from __future__ import annotations


class HasUnresolved:
    def mystery(self, x: "NeverDefined") -> "NeverDefined":
        return x  # type: ignore[no-any-return]
