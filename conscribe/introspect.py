"""Public introspection API: how conscribe reads a class.

One rule decides which names reach a constructor, which attributes can
receive a wired value and in what shape.  conscribe's own extractor, wiring
injection, stubs and fingerprint use it, and so can frameworks built on
conscribe (routing config values to constructor keywords or ``setattr``).

* :func:`init_param_names` -- names ``cls(...)`` accepts by keyword,
  following the runtime ``**kwargs`` forwarding chain up the MRO.
* :func:`receptor_slots` -- every runtime slot: constructor parameters
  (``via="init"``) and class-level annotations (``via="annotation"``), with
  their shape.
* :func:`wiring_slots` -- each merged ``__wiring__`` entry with its slot
  (``None`` = slot-less, a declaration-only constraint).
* :func:`annotation_shape`, :func:`is_classvar`, :func:`config_from_param`.

Contract:

* Pure reflection: no registry lookup, no imports.  Answers reflect the
  class at call time (inside a metaclass ``__new__`` a ``@dataclass``
  decorator has not run yet, so its fields are not constructor parameters
  yet); only per-function signatures are cached, never per class.
* Scope-free: ``__config_mro_scope__`` / ``__config_mro_depth__`` and the
  registrar's ``mro_scope`` / ``mro_depth`` only filter which parent knobs a
  generated config exposes; they never change these answers.
* PEP 649-safe (Python 3.14) and identical on Python 3.10 to 3.14.
* Annotation problems never raise here: an annotation that cannot be
  resolved is reported in its raw form (string / ``ForwardRef``) and its
  shape is read from the source text.  Malformed ``__wiring__`` declarations
  do raise (from :func:`conscribe.wiring.parse_wiring`).
* Result dataclasses are frozen; new fields may be added in minor releases
  (do not unpack them positionally).
"""
from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ForwardRef, Literal

from pydantic_core import PydanticUndefined

from conscribe.config import _annotations as _ann
from conscribe.config._fieldinfo import field_infos_of
from conscribe.config.mro import (
    HAS_DEFAULT_FACTORY,
    InitParam,
    is_no_default,
    keyword_param_names,
    walk_init_chain,
)

if TYPE_CHECKING:
    from conscribe.wiring import WiringSpec

#: Receptor shape: one value, a list of values, or a mapping of values.
Shape = Literal["scalar", "list", "dict"]
#: How a slot receives its value: a constructor keyword, or ``setattr``.
Via = Literal["init", "annotation"]

__all__ = [
    "ReceptorSlot",
    "Shape",
    "Via",
    "WiringSlot",
    "annotation_shape",
    "config_from_param",
    "init_param_names",
    "is_classvar",
    "receptor_slots",
    "wiring_slots",
]


@dataclass(frozen=True)
class ReceptorSlot:
    """A place on a class that can receive a value.

    Attributes:
        name: Attribute / parameter name.
        via: ``"init"`` when the constructor accepts it by keyword (pass it
            to ``cls(...)``); ``"annotation"`` when it is only a class-level
            annotation (``setattr`` it on the instance after construction).
        shape: ``"scalar"``, ``"list"`` or ``"dict"`` (see
            :func:`annotation_shape`): how many wired instances it holds.
        annotation: The annotation, resolved in the namespace of the class
            that declares it (``Annotated`` metadata kept; for pydantic
            fields the type pydantic resolved).  The raw string /
            ``ForwardRef`` when it cannot be resolved; ``None`` when the
            slot is unannotated.  For ``via="init"`` the parameter's own
            annotation wins; a same-named class annotation is used when the
            parameter has none (or it cannot be resolved).
        declared_on: The class that declares it: the class whose
            ``__init__`` takes the parameter, the dataclass / model that
            declares the field, or the most-derived class annotating it.
        parameter: The constructor ``inspect.Parameter`` (kind, declared
            annotation, default) for ``via="init"``; ``None`` otherwise.
            Its ``default`` is the real runtime default, never a stand-in:
            ``inspect.Parameter.empty`` when the constructor requires the
            argument, the default value, or -- when the default comes from a
            factory (``dataclasses.field(default_factory=...)``, attrs
            ``Factory``, pydantic ``default_factory``) -- the
            ``dataclasses`` ``<factory>`` marker, exactly as stdlib
            dataclasses show it in their generated ``__init__``.  A
            ``Field(...)`` written as the default value is replaced by that
            field's default.
        default: The slot's default as a generated config uses it:
            ``inspect.Parameter.empty`` when there is none (or it comes
            from :attr:`default_factory`), else the value.  For
            ``via="init"`` the parameter's own default, else a
            ``Field(default=...)`` in its ``Annotated`` metadata; for
            ``via="annotation"`` a plain class-level value (``hooks:
            list[P] = []``) found along the MRO (descriptors, methods and
            wiring placeholders are not defaults).  Never a sentinel.
        default_factory: The zero-argument factory producing the default,
            or ``None``.
    """

    name: str
    via: Via
    shape: Shape
    annotation: Any
    declared_on: type | None
    parameter: inspect.Parameter | None
    default: Any = field(default=inspect.Parameter.empty, repr=False)
    default_factory: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class WiringSlot:
    """One merged ``__wiring__`` entry of a class and where it lands.

    Attributes:
        name: The wired attribute / parameter name.
        spec: The parsed entry (:class:`conscribe.wiring.WiringSpec`).
        slot: The runtime slot receiving the wired instance, or ``None``
            for a slot-less entry (nothing can receive a value: a
            declaration-only constraint).
        declare_only: The entry is explicitly declaration-only
            (``"declare_only": True`` in the dict form); never injected.
        default_key: The entry's default registry key (dict form
            ``"default"``), or ``None``.
    """

    name: str
    spec: WiringSpec
    slot: ReceptorSlot | None
    declare_only: bool
    default_key: str | None


def init_param_names(cls: type) -> tuple[str, ...]:
    """Names ``cls(...)`` accepts by keyword, child first.

    Follows the runtime ``**kwargs`` forwarding chain up the MRO wherever it
    leads (installed packages included); classes without an ``__init__`` of
    their own (``Generic``, ``ABC``, Protocols, bridges, mixins) never cut
    it, and Protocol placeholder / copied ``__init__`` functions are not
    definers.  Covers hand-written ``__init__`` methods (the instance
    parameter is dropped whatever it is called), dataclass-generated
    ``__init__`` methods (including inherited fields and ``InitVar``
    pseudo-fields), pydantic models (``model_fields`` with ``init`` not
    ``False``: field names, not aliases) and pydantic dataclasses.
    Positional-only parameters are excluded (they cannot be passed by name);
    ``*args`` / ``**kwargs`` themselves are not names.

    When the chain cannot be followed to its end (an ``__init__`` whose
    signature cannot be read, or a placeholder that swallows its arguments)
    the names found up to that point are returned.

    Args:
        cls: Any class.

    Returns:
        A tuple of names, unique, in constructor order.
    """
    _require_class(cls)
    return keyword_param_names(cls)


def receptor_slots(cls: type, *, names: Iterable[str] | None = None) -> dict[str, ReceptorSlot]:
    """Every runtime slot of *cls*: constructor keywords and annotated attributes.

    Keys are :func:`init_param_names` (``via="init"``) plus every
    non-``ClassVar``, non-dunder class-level annotation along the MRO that is
    not a constructor keyword (``via="annotation"``).  The most-derived
    declaration wins; a ``ClassVar`` re-declaration removes an annotation
    slot.  The ``dataclasses.KW_ONLY`` sentinel and ``InitVar`` annotations
    are not attribute slots.

    Args:
        cls: Any class.
        names: Only compute these slots (names that are not slots are
            absent from the result).  Resolving a slot's annotation is the
            expensive part, so pass the names you need when you do not need
            them all.

    Returns:
        ``{name: ReceptorSlot}``: constructor parameters first (constructor
        order), then annotation-only slots (base-first declaration order).
    """
    _require_class(cls)
    wanted = None if names is None else frozenset(names)
    chain = walk_init_chain(cls)
    annotated = _annotation_slots(cls)
    slots: dict[str, ReceptorSlot] = {}
    for param in chain.params:
        if not param.keyword or param.name in slots:
            continue
        if wanted is None or param.name in wanted:
            slots[param.name] = _init_slot(param, annotated.get(param.name))
    init_names = {p.name for p in chain.params if p.keyword}
    for name, owner in annotated.items():
        if name in init_names or name in slots:
            continue
        if wanted is None or name in wanted:
            slots[name] = _annotation_slot(cls, name, owner)
    return slots


def wiring_slots(cls: type) -> dict[str, WiringSlot]:
    """Each merged ``__wiring__`` entry of *cls* with its receptor slot.

    Uses :func:`conscribe.wiring.parse_wiring` (MRO deep-merge, ``None``
    excludes an inherited entry); needs no registry, so it works while target
    registries are empty or missing.

    Args:
        cls: Any class.

    Returns:
        ``{entry name: WiringSlot}`` in ``__wiring__`` order; empty when the
        class declares no wiring.

    Raises:
        Whatever :func:`conscribe.wiring.parse_wiring` raises for a malformed
        ``__wiring__`` declaration.
    """
    from conscribe.wiring import parse_wiring

    _require_class(cls)
    specs = parse_wiring(cls)
    if not specs:
        return {}
    slots = receptor_slots(cls, names=[spec.param_name for spec in specs])
    return {
        spec.param_name: WiringSlot(
            name=spec.param_name,
            spec=spec,
            slot=slots.get(spec.param_name),
            declare_only=bool(getattr(spec, "declare_only", False)),
            default_key=getattr(spec, "default_key", None),
        )
        for spec in specs
    }


def annotation_shape(annotation: Any, owner: Any = None) -> Shape:
    """Receptor shape of an annotation: ``"scalar"``, ``"list"`` or ``"dict"``.

    ``"dict"`` for mappings (``dict``, ``Mapping``, ``OrderedDict``,
    ``defaultdict``, ...), ``"list"`` for other collections (``list``,
    ``tuple``, ``set``, ``frozenset``, ``deque``, ``Sequence``, ``Set``,
    ``Iterable``, ``Collection``), ``"scalar"`` for everything else --
    including ``str``/``bytes``, ``TypedDict`` / ``NamedTuple`` records and
    classes that merely implement ``__iter__``.  ``Annotated``, ``Optional``,
    ``Union`` and ``X | None`` are looked through (a union is ``"dict"`` if any
    member is, else ``"list"`` if any member is).

    Works on resolved annotations, ``ForwardRef`` objects and strings; names
    in strings are looked up in *owner* (a class or module) first and judged
    by spelling only when they cannot be resolved.
    """
    return _ann.annotation_shape(annotation, owner)


def is_classvar(annotation: Any, owner: Any = None) -> bool:
    """True for ``ClassVar`` / ``ClassVar[...]`` annotations (also under ``Annotated``).

    String annotations are resolved against *owner*'s namespace, so aliases
    (``import typing as t`` → ``t.ClassVar[int]``, ``ClassVar as CV``) are
    recognised and a user type named ``ClassVarRegistry`` is not.
    """
    return _ann.is_classvar(annotation, owner)


def config_from_param(cls: type) -> str | None:
    """The ``__config_from_param__`` declared along *cls*'s MRO.

    The most-derived class that sets the attribute wins; ``None`` (also the
    explicit opt-out ``__config_from_param__ = None``) means the class's
    config is not flattened from a single parameter.  The value is returned
    as declared -- validating it is the config extractor's job.
    """
    _require_class(cls)
    for klass in cls.__mro__:
        if "__config_from_param__" in klass.__dict__:
            return klass.__dict__["__config_from_param__"]
    return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_class(cls: Any) -> None:
    if not isinstance(cls, type):
        raise TypeError(
            f"conscribe.introspect expects a class, got {type(cls).__name__}: {cls!r}"
        )


def _annotation_slots(cls: type) -> dict[str, type]:
    """``{name: declaring class}`` of class-level annotation slots along the MRO."""
    slots: dict[str, type] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        own = _ann.own_annotations(klass)
        if not own:
            continue
        context: Any = None
        for name, raw in own.items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(raw, (str, ForwardRef)):
                if context is None:
                    context = _ann.class_context(klass)  # once per class, only for text
                owner: Any = context
            else:
                owner = klass
            if _ann.is_pseudo_field(raw, owner):
                slots.pop(name, None)
                continue
            slots[name] = klass
    return slots


def _init_slot(param: InitParam, annotated_on: type | None) -> ReceptorSlot:
    pydantic_type = _pydantic_field_type(param)
    if pydantic_type is not None:
        annotation, context = pydantic_type, _ann.class_context(param.declared_on)
    else:
        resolved = param.resolve()
        annotation, context = _slot_annotation(resolved)
        if resolved.status != "resolved" and annotated_on is not None:
            from_class = _ann.resolve_class_annotation(annotated_on, param.name)
            if from_class is not None and (
                from_class.status == "resolved" or resolved.status == "unannotated"
            ):
                annotation, context = _slot_annotation(from_class)
    default, factory, runtime_default = _init_default(param, annotation)
    parameter = param.parameter
    if runtime_default is not parameter.default:
        parameter = parameter.replace(default=runtime_default)
    return ReceptorSlot(
        name=param.name,
        via="init",
        shape=_ann.annotation_shape(annotation, context) if annotation is not None else "scalar",
        annotation=annotation,
        declared_on=param.declared_on,
        parameter=parameter,
        default=default,
        default_factory=factory,
    )


def _annotation_slot(cls: type, name: str, owner: type) -> ReceptorSlot:
    resolved = _ann.resolve_class_annotation(owner, name)
    if resolved is None:  # pragma: no cover - owner came from own_annotations
        annotation, context = None, _ann.class_context(owner)
    else:
        annotation, context = _slot_annotation(resolved)
    default, factory = _class_default(cls, name)
    return ReceptorSlot(
        name=name,
        via="annotation",
        shape=_ann.annotation_shape(annotation, context) if annotation is not None else "scalar",
        annotation=annotation,
        declared_on=owner,
        parameter=None,
        default=default,
        default_factory=factory,
    )


def _init_default(param: InitParam, annotation: Any) -> tuple[Any, Any, Any]:
    """``(default, default_factory, runtime default)`` of a constructor slot.

    The first two are the config view (the parameter's own default, else a
    ``Field`` default in the annotation's metadata); the third is the
    ``inspect.Parameter`` default without stand-ins (``<factory>`` marker for
    any factory, ``empty`` when required).
    """
    declared = param.declared_default()
    if declared.takes_self:
        return inspect.Parameter.empty, None, HAS_DEFAULT_FACTORY
    if declared.factory is not None:
        return inspect.Parameter.empty, declared.factory, HAS_DEFAULT_FACTORY
    if declared.value is not inspect.Parameter.empty:
        return declared.value, None, declared.value
    own: tuple[Any, ...] = (declared.field_info,) if declared.field_info is not None else ()
    runtime: Any = inspect.Parameter.empty
    for info in own:  # a Field(...) default value / the pydantic field itself
        default, factory = _field_info_default(info)
        if factory is not None:
            runtime = HAS_DEFAULT_FACTORY
        elif default is not inspect.Parameter.empty:
            runtime = default
    metadata = field_infos_of(annotation) if annotation is not None else ()
    for info in (*own, *metadata):
        default, factory = _field_info_default(info)
        if factory is not None or default is not inspect.Parameter.empty:
            return default, factory, runtime
    return inspect.Parameter.empty, None, runtime


def _field_info_default(info: Any) -> tuple[Any, Any]:
    factory = getattr(info, "default_factory", None)
    if factory is not None:
        return inspect.Parameter.empty, factory
    default = getattr(info, "default", PydanticUndefined)
    if is_no_default(default):
        return inspect.Parameter.empty, None
    return default, None


def _class_default(cls: type, name: str) -> tuple[Any, Any]:
    """``(default, default_factory)`` of a class-level annotation slot.

    The value bound to *name* along *cls*'s MRO (``inspect.getattr_static``)
    when it is a plain value: not a descriptor (methods, properties, wiring
    placeholders) and not a "no default" marker.  A pydantic ``Field(...)``
    value contributes its own default / factory.  ``(empty, None)``
    otherwise.
    """
    try:
        value = inspect.getattr_static(cls, name)
    except AttributeError:
        return inspect.Parameter.empty, None
    if is_no_default(value):
        return inspect.Parameter.empty, None
    from pydantic.fields import FieldInfo

    if isinstance(value, FieldInfo):
        return _field_info_default(value)
    if hasattr(type(value), "__get__") or isinstance(value, (staticmethod, classmethod)):
        return inspect.Parameter.empty, None
    return value, None


def _slot_annotation(resolved: _ann.ResolvedAnnotation) -> tuple[Any, _ann.AnnotationContext]:
    if resolved.status == "resolved":
        return resolved.value, resolved.context
    if resolved.status == "unannotated":
        return None, resolved.context
    return resolved.raw, resolved.context


def _pydantic_field_type(param: InitParam) -> Any:
    """The type pydantic resolved for a model / pydantic-dataclass field."""
    if param.source not in ("model", "pydantic_dataclass"):
        return None
    annotation = getattr(param.field, "annotation", None)
    if annotation is None or _ann.contains_forward_refs(annotation):
        return None
    return annotation
