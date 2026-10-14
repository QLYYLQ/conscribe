"""FieldInfo carry rules for the config models conscribe builds.

A config field is never built by handing a user's ``FieldInfo`` (or one
taken from ``model_fields``) back to pydantic: depending on the pydantic
version that either mutates the source object (2.6-2.9 share
``_attributes_set``) or lets the copied, "all attributes set" FieldInfo
override metadata the new type carries (2.12 drops an inline union's
``discriminator``).  ``FieldInfo.merge_field_infos`` is deprecated for the
same reasons.  Every helper here therefore builds a *fresh*, public
``Field(...)`` from an explicit list of attributes and passes constraint /
marker metadata through ``Annotated``.

Two attribute sets, two consumers:

* :data:`CONFIG_FIELD_ATTRS` / :func:`config_field` -- a constructor
  parameter (``Annotated[T, Field(...), ...]`` plus its signature default)
  becomes a config field.  Everything the user wrote is carried, except
  ``exclude=`` (a :class:`~conscribe.exceptions.ConfigDeclarationError` on a
  constructor parameter: a config model carries constructor arguments,
  dropping one from ``model_dump()`` makes no sense; on the field of a
  registered pydantic model -- ``api_key: SecretStr = Field(exclude=True)``
  -- it is a serialization choice of the model and is simply not carried)
  and ``deprecated=`` (kept only as the JSON-schema marker
  ``{"deprecated": true}``, never as pydantic's runtime
  ``DeprecationWarning`` on attribute access).  :func:`config_field` is
  also the one implementation of the 2.0 ``None`` rule.  Used by the
  extractor and the nested-mode builder.
* :data:`INLINE_FIELD_ATTRS` / :func:`inline_field` -- the composed inline
  retype, where a wired *key selector* field becomes inline nested configs.
  Only what still describes the field is kept: title, description, the
  deprecated marker and the aliases (so the YAML key matches the per-layer
  model).  ``min_length`` / ``max_length`` survive only on list receptors
  (they count wired instances there); on scalar receptors the key-string
  constraints are applied to the allowed keys at build time by
  :func:`narrow_keys` and then dropped.  Examples and every piece of
  metadata pydantic interprets (validators, ``WithJsonSchema``,
  ``Strict``, ...) are dropped -- they were written for the key string.
  Plain markers (objects pydantic ignores, e.g. a frozen dataclass a
  framework reads back from ``FieldInfo.metadata``) are kept.
"""
from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, Optional, get_args, get_origin

import annotated_types
import typing_extensions
from pydantic import Field, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from conscribe.config._annotations import UNION_ORIGINS, admits_none
from conscribe.exceptions import ConfigDeclarationError

__all__ = [
    "CONFIG_FIELD_ATTRS",
    "INLINE_FIELD_ATTRS",
    "KEEP",
    "config_field",
    "field_infos_of",
    "inline_field",
    "inline_metadata",
    "is_deprecated",
    "is_pydantic_interpreted",
    "narrow_keys",
]

#: ``FieldInfo`` attributes carried from a user's ``Field(...)`` into the
#: in-process config model.  Constraints (``ge``, ``pattern``, ``strict``,
#: ...) are not attributes in pydantic 2 -- they live in
#: ``FieldInfo.metadata`` and are carried through it.  ``default`` /
#: ``default_factory`` are resolved separately (see :func:`config_field`);
#: ``exclude`` / ``exclude_if`` are refused and ``deprecated`` becomes a
#: schema marker.  ``init`` / ``init_var`` / ``kw_only`` only apply to
#: dataclasses and are not carried.
CONFIG_FIELD_ATTRS: tuple[str, ...] = (
    "alias",
    "alias_priority",
    "validation_alias",
    "serialization_alias",
    "title",
    "field_title_generator",  # pydantic >= 2.9
    "description",
    "examples",
    "discriminator",
    "json_schema_extra",
    "frozen",
    "validate_default",
    "repr",
)

#: ``FieldInfo`` attributes the composed inline retype keeps when a wired
#: key-selector field becomes inline nested configs.  Never carries
#: ``discriminator`` (the inline union brings its own), ``examples`` or the
#: rest of ``json_schema_extra`` (both describe the key string).  The
#: deprecated marker is carried separately.
INLINE_FIELD_ATTRS: tuple[str, ...] = (
    "title",
    "description",
    "alias",
    "alias_priority",
    "validation_alias",
    "serialization_alias",
)


class _Keep:
    """Sentinel type: "carry the source field's default unchanged"."""

    def __repr__(self) -> str:
        return "KEEP"


#: Default for the ``default`` / ``default_factory`` overrides of
#: :func:`inline_field`: keep whatever the source field declares.
KEEP: Any = _Keep()

_NONE_TYPE = type(None)

# Signature "defaults" that mean "no default" (the field is required).
_NO_DEFAULT_SENTINELS: tuple[Any, ...] = (
    inspect.Parameter.empty,
    dataclasses.MISSING,
    PydanticUndefined,
    Ellipsis,
)
# ``dataclasses`` puts this sentinel in a synthesized ``__init__`` signature
# for ``field(default_factory=...)``; the real factory is only reachable via
# ``__dataclass_fields__`` (the caller resolves it).
_DATACLASS_FACTORY_SENTINEL: Any = getattr(dataclasses, "_HAS_DEFAULT_FACTORY", object())


# ---------------------------------------------------------------------------
# Metadata classification
# ---------------------------------------------------------------------------


def is_pydantic_interpreted(meta: Any) -> bool:
    """Return ``True`` when pydantic gives *meta* a meaning of its own.

    That covers every ``annotated_types`` constraint (``Ge``, ``MinLen``,
    ``Predicate``, ``Len``, ...), pydantic's own metadata (``Strict``, the
    general ``pattern`` / ``union_mode`` holder, ``StringConstraints``) and
    any object that hooks into schema generation
    (``__get_pydantic_core_schema__`` / ``__get_pydantic_json_schema__``:
    validators, serializers, ``WithJsonSchema``, ``Discriminator``, ...).
    Everything else is a *plain marker* that pydantic stores in
    ``FieldInfo.metadata`` untouched.
    """
    if isinstance(meta, FieldInfo):
        return True
    if isinstance(meta, (annotated_types.BaseMetadata, annotated_types.GroupedMetadata)):
        return True
    return hasattr(meta, "__get_pydantic_core_schema__") or hasattr(
        meta, "__get_pydantic_json_schema__"
    )


def _is_count_constraint(meta: Any) -> bool:
    """``min_length`` / ``max_length`` style constraints (``MinLen``, ``MaxLen``, ``Len``)."""
    return isinstance(meta, (annotated_types.MinLen, annotated_types.MaxLen, annotated_types.Len))


def inline_metadata(metadata: Iterable[Any], *, container: bool) -> list[Any]:
    """Filter a wired field's metadata for the inline retype.

    Keeps ``MinLen`` / ``MaxLen`` / ``Len`` only when *container* is true
    (on a list receptor they bound the number of wired instances), drops
    everything else pydantic interprets (see :func:`is_pydantic_interpreted`)
    and keeps plain markers, in their original order.
    """
    kept: list[Any] = []
    for meta in metadata:
        if _is_count_constraint(meta):
            if container:
                kept.append(meta)
            continue
        if is_pydantic_interpreted(meta):
            continue
        kept.append(meta)
    return kept


def is_deprecated(field_info: FieldInfo) -> bool:
    """Whether *field_info* is deprecated, as a runtime flag or as the schema marker.

    Recognises pydantic's ``deprecated=`` attribute, the marker
    ``json_schema_extra={"deprecated": True}`` that :func:`config_field`
    writes in its place, and the callable wrapper it uses when the user's
    ``json_schema_extra`` is a function.
    """
    if getattr(field_info, "deprecated", None):
        return True
    extra = field_info.json_schema_extra
    if isinstance(extra, Mapping):
        return extra.get("deprecated") is True
    return isinstance(extra, _DeprecatedSchemaExtra)


@dataclasses.dataclass(frozen=True)
class _DeprecatedSchemaExtra:
    """Callable ``json_schema_extra`` that adds the deprecated marker, then runs the user's."""

    inner: Callable[..., Any]

    def __call__(self, schema: dict[str, Any], *args: Any) -> None:
        schema["deprecated"] = True
        self.inner(schema, *args)


def _with_deprecated_marker(extra: Any) -> Any:
    """Return *extra* (a ``json_schema_extra`` value) plus ``{"deprecated": True}``."""
    if extra is None:
        return {"deprecated": True}
    if isinstance(extra, Mapping):
        return {**extra, "deprecated": True}
    if isinstance(extra, _DeprecatedSchemaExtra):
        return extra
    return _DeprecatedSchemaExtra(extra)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _is_annotated(tp: Any) -> bool:
    return get_origin(tp) is Annotated


def _hoist_optional_annotated(annotation: Any) -> Any:
    """``Optional[Annotated[T, *meta]]`` -> ``Annotated[Optional[T], *meta]``.

    ``typing.get_type_hints`` adds the ``Optional`` *outside* the
    ``Annotated`` for a ``= None`` parameter on Python 3.10, and users write
    ``Optional[Annotated[T, Field(...)]]`` on every version; either way the
    ``FieldInfo`` would be invisible to the field.  Only ``FieldInfo``
    objects (documentation and declarative constraints, which pydantic
    applies to a nullable schema correctly) move out; other metadata --
    validators in particular, which must not see ``None`` -- stays on ``T``.
    Unions with more than one non-``None`` member are left alone.
    """
    if get_origin(annotation) not in UNION_ORIGINS:
        return annotation
    args = get_args(annotation)
    non_none = [a for a in args if a is not _NONE_TYPE]
    if len(non_none) != 1 or len(non_none) == len(args) or not _is_annotated(non_none[0]):
        return annotation
    inner, *meta = get_args(non_none[0])
    hoisted = [m for m in meta if isinstance(m, FieldInfo)]
    if not hoisted:
        return annotation
    kept = [m for m in meta if not isinstance(m, FieldInfo)]
    inner_type = Annotated[tuple([inner, *kept])] if kept else inner  # type: ignore[misc]
    return Annotated[tuple([Optional[inner_type], *hoisted])]  # type: ignore[misc]


def _runs_on_value(meta: Any) -> bool:
    """Metadata that would receive a ``None`` placed outside an ``Optional``.

    Validators, serializers and other objects that hook into pydantic's
    schema generation (``AfterValidator``, ``PlainSerializer``,
    ``WithJsonSchema``, custom types...).  Not: ``FieldInfo`` objects,
    ``annotated_types`` / pydantic constraints (pydantic applies them to the
    non-``None`` branch of a nullable type) and plain markers.
    """
    if isinstance(meta, FieldInfo):
        return False
    if isinstance(meta, (annotated_types.BaseMetadata, annotated_types.GroupedMetadata)):
        return False
    return hasattr(meta, "__get_pydantic_core_schema__") or hasattr(
        meta, "__get_pydantic_json_schema__"
    )


def field_infos_of(annotation: Any) -> tuple[FieldInfo, ...]:
    """The ``FieldInfo`` objects written in ``Annotated[...]`` metadata, in order.

    Looks through ``Optional[Annotated[T, Field(...)]]`` the same way
    :func:`config_field` does.  Empty for any other annotation.  Used for
    ``__config_annotated_only__`` ("does this parameter carry a Field?").
    """
    annotation = _hoist_optional_annotated(annotation)
    if not _is_annotated(annotation):
        return ()
    return tuple(m for m in get_args(annotation)[1:] if isinstance(m, FieldInfo))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _DefaultSource:
    label: str
    kind: str  # "value" | "factory"
    value: Any

    def same_as(self, other: _DefaultSource) -> bool:
        if self.kind != other.kind:
            return False
        if self.value is other.value:
            return True
        if self.kind == "factory" or type(self.value) is not type(other.value):
            return False
        try:
            return bool(self.value == other.value)
        except Exception:  # noqa: BLE001 - incomparable values are simply "different"
            return False

    def describe(self) -> str:
        if self.kind == "factory":
            name = getattr(self.value, "__qualname__", None) or repr(self.value)
            return f"{self.label} default_factory={name}"
        return f"{self.label} default {self.value!r}"


def _signature_default_source(default: Any, factory: Callable[[], Any] | None) -> _DefaultSource | None:
    if factory is not None:
        return _DefaultSource("the signature", "factory", factory)
    if default is _DATACLASS_FACTORY_SENTINEL:
        raise TypeError(
            "config_field(): the signature default is the dataclass <factory> "
            "sentinel; pass the field's real factory as signature_factory="
        )
    if any(default is sentinel for sentinel in _NO_DEFAULT_SENTINELS):
        return None
    from conscribe.wiring import WiredField

    if isinstance(default, WiredField):
        # A wiring descriptor leaked into the signature; it is not a value.
        return None
    return _DefaultSource("the signature", "value", default)


def _field_default_source(index: int, total: int, fi: FieldInfo) -> _DefaultSource | None:
    label = "Field(...)" if total == 1 else f"Field(...) #{index + 1}"
    if fi.default_factory is not None:
        return _DefaultSource(label, "factory", fi.default_factory)
    if fi.default is PydanticUndefined or fi.default is Ellipsis:
        return None
    return _DefaultSource(label, "value", fi.default)


def _resolve_default(
    sources: Sequence[_DefaultSource], *, owner: str, field: str,
) -> _DefaultSource | None:
    """Only one source may define a default; identical duplicates are fine."""
    if not sources:
        return None
    first = sources[0]
    for other in sources[1:]:
        if not first.same_as(other):
            raise ConfigDeclarationError(
                owner,
                f"two defaults are declared and they differ: {first.describe()} vs "
                f"{other.describe()}. Declare the default in exactly one place "
                f"(the signature or Field(...)).",
                field,
            )
    return first


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def config_field(
    annotation: Any,
    *,
    owner: str,
    field: str,
    signature_default: Any = inspect.Parameter.empty,
    signature_factory: Callable[[], Any] | None = None,
    description: str | None = None,
    model_field: bool = False,
) -> tuple[Any, FieldInfo]:
    """Build the ``create_model`` definition ``(type, FieldInfo)`` of one config field.

    This is the one implementation of the 2.0 ``None`` rule (identical on
    every Python version):

    * ``Optional[Annotated[T, *meta]]`` (also ``Union[T, None]`` and
      ``T | None``) carries its ``FieldInfo`` objects to the field, so a
      ``Field(...)`` is never hidden inside a union; the rest of *meta*
      stays on ``T``.
    * A ``None`` default makes the type admit ``None``
      (:func:`conscribe.config._annotations.admits_none` decides whether it
      already does): ``T`` becomes ``Optional[T]``.  Validators, serializers
      and other schema hooks written on ``T`` move inside with it --
      ``Annotated[T, AfterValidator(f)] = None`` becomes
      ``Optional[Annotated[T, AfterValidator(f)]]`` -- so they never see
      ``None``; ``FieldInfo`` attributes, constraints (applied to the
      non-``None`` branch by pydantic) and plain markers stay on the field.

    Args:
        annotation: The parameter's resolved annotation, typically
            ``Annotated[T, Field(...), <metadata>...]`` or a bare type.
        owner: ``module.QualName`` of the declaring class (error messages).
        field: The parameter name (error messages).
        signature_default: The signature default (``inspect.Parameter.empty``
            when there is none).  ``dataclasses.MISSING``,
            ``PydanticUndefined``, ``...`` and a leaked ``WiredField`` also
            mean "no default".  The dataclass ``<factory>`` sentinel must be
            resolved by the caller (pass ``signature_factory``).
        signature_factory: The real ``default_factory`` of a dataclass field.
        description: Fallback description (Tier 1.5 docstring) used when no
            ``Field(...)`` carries one.
        model_field: The ``FieldInfo`` objects come from a registered pydantic
            model / dataclass field, not from a constructor parameter:
            ``exclude=`` / ``exclude_if=`` (the model's serialization choice)
            are then dropped instead of refused.

    Returns:
        ``(type, field_info)``: *type* is the base type, wrapped as
        ``Annotated[T, *metadata]`` when constraints or markers exist;
        *field_info* is a fresh ``Field(...)`` holding the carried attributes
        and the resolved default.

    Raises:
        ConfigDeclarationError: ``exclude=`` / ``exclude_if=`` on a
            constructor parameter, or two different defaults (signature vs
            ``Field(default=...)`` / ``default_factory``, or between two
            ``Field(...)`` objects).
    """
    annotation = _hoist_optional_annotated(annotation)
    if _is_annotated(annotation):
        base, *extras = get_args(annotation)
        # ``Annotated[Optional[Annotated[T, Field(...)]], Field(...)]``: the
        # inner FieldInfo must not stay hidden in the union either.
        hoisted = _hoist_optional_annotated(base)
        if hoisted is not base and _is_annotated(hoisted):
            base, *inner = get_args(hoisted)
            extras = [*inner, *extras]
    else:
        base, extras = annotation, []

    field_infos: list[FieldInfo] = []
    metadata: list[Any] = []
    deprecated = False
    for item in extras:
        if isinstance(item, FieldInfo):
            field_infos.append(item)
            metadata.extend(item.metadata)
        elif isinstance(item, typing_extensions.deprecated):
            deprecated = True
        else:
            metadata.append(item)

    kwargs: dict[str, Any] = {}
    for fi in field_infos:
        if not model_field and (fi.exclude or getattr(fi, "exclude_if", None) is not None):
            raise ConfigDeclarationError(
                owner,
                "Field(exclude=...) is not allowed on a constructor parameter: a "
                "config model carries constructor arguments, and an excluded "
                "field would be missing from model_dump(). Remove exclude=.",
                field,
            )
        if fi.deprecated:
            deprecated = True
        for attr in CONFIG_FIELD_ATTRS:
            value = getattr(fi, attr, None)
            if attr == "repr":
                if value is False:
                    kwargs["repr"] = False
            elif value is not None:
                kwargs[attr] = value  # a later Field(...) wins, as in pydantic

    if "examples" in kwargs:
        kwargs["examples"] = list(kwargs["examples"])
    if isinstance(kwargs.get("json_schema_extra"), Mapping):
        kwargs["json_schema_extra"] = dict(kwargs["json_schema_extra"])
    if deprecated:
        kwargs["json_schema_extra"] = _with_deprecated_marker(kwargs.get("json_schema_extra"))
    if kwargs.get("description") is None and description:
        kwargs["description"] = description

    sources: list[_DefaultSource] = []
    sig_source = _signature_default_source(signature_default, signature_factory)
    if sig_source is not None:
        sources.append(sig_source)
    for index, fi in enumerate(field_infos):
        src = _field_default_source(index, len(field_infos), fi)
        if src is not None:
            sources.append(src)
    resolved = _resolve_default(sources, owner=owner, field=field)
    if resolved is not None:
        if resolved.kind == "factory":
            kwargs["default_factory"] = resolved.value
        else:
            kwargs["default"] = resolved.value
            if resolved.value is None and not admits_none(base):
                on_value = [m for m in metadata if _runs_on_value(m)]
                if on_value:
                    metadata = [m for m in metadata if not _runs_on_value(m)]
                    base = Optional[Annotated[tuple([base, *on_value])]]  # type: ignore[misc]
                else:
                    base = Optional[base]

    field_type = Annotated[tuple([base, *metadata])] if metadata else base  # type: ignore[misc]
    return field_type, Field(**kwargs)


def inline_field(
    src: FieldInfo,
    inline_type: Any,
    *,
    container: bool,
    default: Any = KEEP,
    default_factory: Any = KEEP,
    json_schema_extra: Mapping[str, Any] | None = None,
) -> tuple[Any, FieldInfo]:
    """Build the ``(type, FieldInfo)`` definition of an inline-retyped receptor.

    Args:
        src: The wired field's ``FieldInfo`` in the per-layer model (read
            only; never handed back to pydantic).
        inline_type: The new type: the target layer's union (or a subset
            union / single model), already wrapped in ``list[...]`` /
            ``Optional[...]`` to match the receptor's shape.
        container: Whether the receptor is a list receptor (keeps
            ``min_length`` / ``max_length``).
        default: Override the carried default (``KEEP`` = carry ``src``'s).
        default_factory: Override the carried factory (``KEEP`` = carry).
            Pass exactly one of ``default`` / ``default_factory`` to
            override; a default-instance factory (F4) goes here.
        json_schema_extra: Extra JSON-schema keys merged into the field's
            ``json_schema_extra`` (e.g. F4's ``{"default": <dump>}``).

    Returns:
        ``(type, field_info)`` ready for ``create_model``.
    """
    if default is not KEEP and default_factory is not KEEP:
        raise TypeError("inline_field(): pass default or default_factory, not both")
    kwargs: dict[str, Any] = {
        attr: getattr(src, attr)
        for attr in INLINE_FIELD_ATTRS
        if getattr(src, attr, None) is not None
    }
    schema_extra: dict[str, Any] = {}
    if is_deprecated(src):
        schema_extra["deprecated"] = True
    if json_schema_extra:
        schema_extra.update(json_schema_extra)
    if schema_extra:
        kwargs["json_schema_extra"] = schema_extra

    if default_factory is not KEEP:
        kwargs["default_factory"] = default_factory
    elif default is not KEEP:
        kwargs["default"] = default
    elif src.default_factory is not None:
        kwargs["default_factory"] = src.default_factory
    elif src.default is not PydanticUndefined:
        kwargs["default"] = src.default

    kept = inline_metadata(src.metadata, container=container)
    field_type = Annotated[tuple([inline_type, *kept])] if kept else inline_type  # type: ignore[misc]
    return field_type, Field(**kwargs)


def _key_constraint(meta: Any) -> dict[str, Any] | None:
    """The ``min_length`` / ``max_length`` / ``pattern`` part of *meta*, if any.

    Only pydantic / ``annotated_types`` constraint objects count; a plain
    marker that happens to have a ``pattern`` attribute is not a constraint.
    """
    if not isinstance(meta, (annotated_types.BaseMetadata, annotated_types.GroupedMetadata)):
        return None
    found: dict[str, Any] = {}
    for name in ("min_length", "max_length", "pattern"):
        value = getattr(meta, name, None)
        if value is not None:
            found[name] = value
    return found or None


def narrow_keys(
    keys: Sequence[str],
    metadata: Iterable[Any],
    *,
    owner: str,
    field: str,
) -> tuple[str, ...]:
    """Apply a scalar receptor's key-string constraints to its allowed keys.

    In the per-layer model a scalar receptor is a ``Literal[...]`` key
    selector, and ``pattern`` / ``min_length`` / ``max_length`` declared on
    the parameter reject some keys.  Under inline wiring the field holds a
    nested config, where those constraints cannot apply; instead the keys
    they reject are removed from the inline union here, at build time, so
    the composed model accepts exactly the keys the per-layer model does.

    Args:
        keys: The allowed keys, in order.
        metadata: The wired field's ``FieldInfo.metadata``.
        owner: ``module.QualName`` of the owning class (error messages).
        field: The receptor name (error messages).

    Returns:
        The keys every constraint accepts, in their original order (all of
        *keys* when there is no key-string constraint).

    Raises:
        ConfigDeclarationError: The constraints reject every allowed key.
    """
    constraints = [c for c in (_key_constraint(m) for m in metadata) if c]
    if not constraints:
        return tuple(keys)
    adapters = [TypeAdapter(Annotated[str, Field(**c)]) for c in constraints]

    def _accepted(key: str) -> bool:
        for adapter in adapters:
            try:
                adapter.validate_python(key)
            except ValidationError:
                return False
        return True

    kept = tuple(k for k in keys if _accepted(k))
    if not kept:
        described = ", ".join(
            ", ".join(f"{name}={value!r}" for name, value in c.items()) for c in constraints
        )
        raise ConfigDeclarationError(
            owner,
            f"the key constraints ({described}) reject every key the wiring "
            f"allows ({list(keys)}), so no config could select a "
            f"{field!r}. Fix the constraint or the __wiring__ keys.",
            field,
        )
    return kept
