"""Type degradation: config fields whose type cannot be used are typed ``Any``.

A field degrades for one of three reasons (:attr:`DegradedField.reason`):

``"pydantic_incompatible"``
    pydantic cannot build a schema for the type (typically a third-party
    class reached through the ``**kwargs`` MRO walk: ``httpx.Auth``,
    ``ssl.SSLContext``...).  The approach is *try-first*: the extractor
    builds the model optimistically and only when pydantic raises a schema
    error are the fields probed one by one (:func:`degrade_field_definitions`).
``"type_checking_only"``
    the annotation needs a name the declaring module only imports under
    ``if TYPE_CHECKING:`` (conscribe never imports those blocks).
``"function_locals"``
    the declaring class is defined inside a function and its annotation
    needs that function's local names, which conscribe cannot see.

A degraded field keeps its name, default, description and every other
``Field(...)`` attribute; only its type stops being validated.  The extractor
emits a :class:`~conscribe.exceptions.DegradationWarning` (an
:class:`~conscribe.exceptions.ExtractionWarning` for ``"function_locals"``)
and records the field in the model's ``__degraded_fields__`` so the generated
artifacts carry their degraded markers (header list, ``# degraded from:``
comment, ``x-degraded-from``).
"""
from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import TypeAdapter

from conscribe.config._annotations import UNION_ORIGINS

#: Every value :attr:`DegradedField.reason` may take.
DEGRADATION_REASONS: frozenset[str] = frozenset(
    {"pydantic_incompatible", "type_checking_only", "function_locals"}
)


@dataclass(frozen=True)
class DegradedField:
    """Record of a field whose type was degraded to ``Any``.

    Attributes:
        field_name: The parameter name (e.g., ``"auth"``).
        original_type_repr: Human-readable original type
            (e.g., ``"AuthTypes | None"``).
        source_class: Fully-qualified name of the class that declares this
            parameter (e.g., ``"httpx.Client"``).
        reason: Why the field was degraded: ``"pydantic_incompatible"``
            (pydantic cannot build a schema for the type),
            ``"type_checking_only"`` (the annotation needs a name only
            imported under ``if TYPE_CHECKING:``) or ``"function_locals"``
            (the annotation needs names local to the function that defines
            the class).
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


def _degrade_type(tp: Any) -> Any:
    """The degraded form of an incompatible field type.

    ``Annotated[T, *metadata]`` keeps its metadata on ``Any``
    (``Annotated[Any, *metadata]``) when pydantic can apply it there, so
    constraints and markers survive; everything else becomes plain ``Any``.
    """
    if get_origin(tp) is Annotated:
        _, *metadata = get_args(tp)
        candidate = Annotated[(Any, *metadata)]  # type: ignore[valid-type]
        if check_type_compatibility(candidate):
            return candidate
    return Any


def degrade_field_definitions(
    field_definitions: dict[str, Any],
    source_class_name: str = "",
    *,
    source_classes: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[DegradedField]]:
    """Replace Pydantic-incompatible types with ``Any``.

    Probes each field's type via :func:`check_type_compatibility`.
    Incompatible types are replaced (see :func:`_degrade_type`: the
    ``Annotated`` metadata of the type is kept when pydantic can apply it to
    ``Any``); the original type representation is recorded in a
    :class:`DegradedField` with reason ``"pydantic_incompatible"``.

    Only the *type* part of each ``(type, default_or_FieldInfo)`` tuple
    is replaced — defaults and ``FieldInfo`` instances are preserved.

    Args:
        field_definitions: Dict ``{name: (type, default_or_FieldInfo)}``.
        source_class_name: Human-readable class name for provenance
            (stored in :attr:`DegradedField.source_class`).
        source_classes: Per-field provenance overriding *source_class_name*
            (``{field name: "module.QualName"}`` of the declaring class).

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
            continue
        base = get_args(field_type)[0] if get_origin(field_type) is Annotated else field_type
        degraded.append(
            DegradedField(
                field_name=name,
                original_type_repr=format_type_repr(base),
                source_class=(source_classes or {}).get(name, source_class_name),
            )
        )
        cleaned[name] = (_degrade_type(field_type), default_or_info)

    return cleaned, degraded


def format_type_repr(tp: Any) -> str:
    """Return a human-readable string for a type annotation.

    Handles common cases:
    - ``Union[X, None]`` → ``"X | None"``
    - ``list[X]`` → ``"list[X]"``
    - ``Annotated[X, ...]`` → the representation of ``X``
    - Ordinary classes → ``"module.ClassName"``
    - ``ForwardRef`` → the original string
    """
    # Handle None / NoneType
    if tp is None or tp is type(None):
        return "None"

    # Handle typing.ForwardRef
    forward = getattr(tp, "__forward_arg__", None)
    if isinstance(forward, str):
        return forward

    if get_origin(tp) is Annotated:
        return format_type_repr(get_args(tp)[0])

    # ``__origin__`` / ``__args__`` are read defensively: on a *class* such as
    # the bare ``typing.Union`` of Python 3.14 they are member descriptors,
    # not values.
    origin = getattr(tp, "__origin__", None)
    if isinstance(origin, (types.MemberDescriptorType, types.GetSetDescriptorType)):
        origin = None
    args = getattr(tp, "__args__", None)
    if not isinstance(args, tuple):
        args = None

    # Union types (typing.Union or Python 3.10+ X | Y)
    if get_origin(tp) in UNION_ORIGINS or (origin is Union and args):
        if args:
            return " | ".join(format_type_repr(a) for a in args)
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
