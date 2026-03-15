"""Type degradation for Pydantic-incompatible field types.

When MRO traversal reaches third-party classes, their ``__init__``
parameters may use types that Pydantic cannot serialize (e.g.,
``httpx.Auth``, ``ssl.SSLContext``).  Rather than failing, this module
degrades incompatible types to ``Any`` while preserving field names,
defaults, and ``FieldInfo`` metadata.

The approach is *try-first*: ``_create_dynamic_model()`` is called
optimistically.  Only when it raises a Pydantic schema error do we
probe individual fields and replace incompatible types.

See ``config-typing-design.md`` for specification.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Union

from pydantic import TypeAdapter

# Union-like origins for format_type_repr (Python 3.10+ has types.UnionType)
_UNION_ORIGINS: set[Any] = {Union}
if hasattr(types, "UnionType"):
    _UNION_ORIGINS.add(types.UnionType)


@dataclass(frozen=True)
class DegradedField:
    """Record of a field whose type was degraded to ``Any``.

    Attributes:
        field_name: The parameter name (e.g., ``"auth"``).
        original_type_repr: Human-readable original type
            (e.g., ``"AuthTypes | None"``).
        source_class: Fully-qualified name of the class that defined
            this parameter (e.g., ``"httpx.Client"``).
        reason: Why the field was degraded (always
            ``"pydantic_incompatible"`` for now).
    """

    field_name: str
    original_type_repr: str
    source_class: str
    reason: str = "pydantic_incompatible"


def check_type_compatibility(tp: Any) -> bool:
    """Check whether Pydantic can handle *tp* as a field type.

    Uses ``TypeAdapter(tp)`` as the probe — if Pydantic can build a
    schema for it, the type is compatible.

    Returns:
        ``True`` if Pydantic accepts *tp*, ``False`` otherwise.
    """
    from pydantic import PydanticSchemaGenerationError, PydanticUserError

    try:
        TypeAdapter(tp)
        return True
    except (PydanticSchemaGenerationError, PydanticUserError, TypeError):
        return False


def degrade_field_definitions(
    field_definitions: dict[str, Any],
    source_class_name: str = "",
) -> tuple[dict[str, Any], list[DegradedField]]:
    """Replace Pydantic-incompatible types with ``Any``.

    Probes each field's type via :func:`check_type_compatibility`.
    Incompatible types are replaced with ``Any``; the original type
    representation and other metadata are recorded in a
    :class:`DegradedField`.

    Only the *type* part of each ``(type, default_or_FieldInfo)`` tuple
    is replaced — defaults and ``FieldInfo`` instances are preserved.

    Args:
        field_definitions: Dict ``{name: (type, default_or_FieldInfo)}``.
        source_class_name: Human-readable class name for provenance
            (stored in :attr:`DegradedField.source_class`).

    Returns:
        A 2-tuple of (cleaned field_definitions, list of DegradedField).
        If all fields are compatible the list is empty and the dict is
        returned unchanged.
    """
    degraded: list[DegradedField] = []
    cleaned: dict[str, Any] = {}

    for name, value in field_definitions.items():
        if not isinstance(value, tuple) or len(value) != 2:
            cleaned[name] = value
            continue

        field_type, default_or_info = value

        if check_type_compatibility(field_type):
            cleaned[name] = value
        else:
            degraded.append(
                DegradedField(
                    field_name=name,
                    original_type_repr=format_type_repr(field_type),
                    source_class=source_class_name,
                )
            )
            cleaned[name] = (Any, default_or_info)

    return cleaned, degraded


def format_type_repr(tp: Any) -> str:
    """Return a human-readable string for a type annotation.

    Handles common cases:
    - ``Union[X, None]`` → ``"X | None"``
    - ``list[X]`` → ``"list[X]"``
    - Ordinary classes → ``"module.ClassName"``
    - ``ForwardRef`` → the original string
    """
    # Handle None / NoneType
    if tp is None or tp is type(None):
        return "None"

    # Handle typing.ForwardRef
    if hasattr(tp, "__forward_arg__"):
        return tp.__forward_arg__

    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", None)

    # Union types (typing.Union or Python 3.10+ X | Y)
    if origin in _UNION_ORIGINS or isinstance(tp, type(Union[int, str])):
        if args:
            parts = [format_type_repr(a) for a in args]
            return " | ".join(parts)
        return repr(tp)

    # Generic types: list[X], dict[K, V], etc.
    if origin is not None and args:
        origin_name = getattr(origin, "__name__", str(origin))
        args_str = ", ".join(format_type_repr(a) for a in args)
        return f"{origin_name}[{args_str}]"

    if origin is not None:
        return getattr(origin, "__name__", str(origin))

    # Any (must check before isinstance(tp, type) since Any is not a type)
    if tp is Any:
        return "Any"

    # Plain class
    if isinstance(tp, type):
        module = getattr(tp, "__module__", "")
        name = getattr(tp, "__qualname__", getattr(tp, "__name__", repr(tp)))
        if module and module != "builtins":
            return f"{module}.{name}"
        return name

    return repr(tp)
