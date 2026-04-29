"""Fixture: class referencing types via ``if TYPE_CHECKING:`` imports."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import OrderedDict


class UsesTypeChecking:
    """References a name that exists only inside ``if TYPE_CHECKING:``.

    At runtime ``OrderedDict`` is *not* in this module's globals, so a
    naive ``get_type_hints`` would fail. The stub generator must walk
    the source AST to discover the import.
    """

    def build(self) -> OrderedDict[str, int]:
        from collections import OrderedDict as _OD

        return _OD()
