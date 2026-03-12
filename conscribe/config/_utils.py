"""Shared internal helpers for the config subsystem."""
from __future__ import annotations

from typing import Union


def find_init_definer(cls: type) -> Union[type, None]:
    """Walk MRO to find the class that actually defines ``__init__``.

    Skips ``object`` (its ``__init__`` is not useful for extraction).
    """
    for klass in cls.__mro__:
        if klass is object:
            continue
        if "__init__" in klass.__dict__:
            return klass
    return None
