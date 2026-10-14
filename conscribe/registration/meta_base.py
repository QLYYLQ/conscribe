"""Shared metaclass base for all conscribe AutoRegistrar metaclasses.

Provides the ``|`` operator for cross-registry diamond inheritance:
    ``metaclass=LLM.Meta | Agent.Meta``

Architecture:
    MetaRegistrarType (meta-metaclass) defines ``__or__``
    └── AutoRegistrarBase (metaclass base) — inherits from ``type`` via MetaRegistrarType
        └── AutoRegistrar[llm] (concrete metaclass per registry)

    ``LLM.Meta | Agent.Meta`` calls ``MetaRegistrarType.__or__`` because
    ``type(LLM.Meta)`` resolves through ``MetaRegistrarType``.
"""
from __future__ import annotations


class MetaRegistrarType(type):
    """Meta-metaclass that provides the ``|`` operator on AutoRegistrar classes.

    When ``LLM.Meta | Agent.Meta`` is written, Python dispatches to
    ``type(LLM.Meta).__or__``. By making ``AutoRegistrarBase`` use this
    as its metaclass, all AutoRegistrar subclasses get ``|`` support.
    """

    def __or__(cls, other: type) -> type:
        """Combine two AutoRegistrar metaclasses.

        If one is already a subclass of the other, returns the more
        specific one. Otherwise creates a new combined metaclass.
        """
        if not isinstance(other, MetaRegistrarType):
            return NotImplemented
        if issubclass(cls, other):
            return cls
        if issubclass(other, cls):
            return other
        name = f"Combined[{cls.__qualname__},{other.__qualname__}]"
        return MetaRegistrarType(name, (cls, other), {})

    def __ror__(cls, other: type) -> type:
        return cls.__or__(other)

    @property
    def inline_only(cls) -> bool:
        """Whether this AutoRegistrar's layer was declared inline-only.

        Set with ``create_auto_registrar(..., inline_only=True)`` (which
        ``create_registrar(..., inline_only=True)`` calls).  The property lives
        on the meta-metaclass, so it is visible on AutoRegistrar metaclasses
        only and never as an attribute of the classes they create (where it
        could be mistaken for a dataclass or pydantic field default).
        A combined metaclass reports the flag of its first AutoRegistrar base.
        """
        return bool(getattr(cls, "_conscribe_inline_only", False))


class AutoRegistrarBase(type, metaclass=MetaRegistrarType):
    """Base metaclass for all conscribe AutoRegistrar metaclasses.

    Uses ``MetaRegistrarType`` as its metaclass so that the ``|``
    operator works on AutoRegistrar instances (which are classes
    themselves).
    """
