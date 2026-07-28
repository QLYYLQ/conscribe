"""Collect stub information from registered classes with wired attributes."""
from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InjectedAttr:
    """A wired attribute injected at runtime (not in ``__init__``)."""

    name: str
    resolved_type: type
    registry_name: str | None  # None for Mode 3 (literal list)


@dataclass(frozen=True)
class MethodStub:
    """Method signature for .pyi rendering.

    ``is_async`` records whether the underlying function is a coroutine
    (or async generator) function, so the renderer can emit ``async def``.
    Without it every ``await`` on the method is a type error downstream.
    """

    name: str
    signature: str  # e.g. "(self, x: int) -> str"
    decorators: tuple[str, ...]  # e.g. ("@classmethod",)
    is_async: bool = False


@dataclass(frozen=True)
class ClassAttrStub:
    """Class-level annotated attribute for .pyi rendering.

    Captures dataclass-style fields and other class-body annotations so
    IDEs can resolve ``obj.attr`` against the stub even when the
    attribute is also an ``__init__`` parameter.
    """

    name: str
    annotation: str  # rendered annotation text (string under PEP 563)


@dataclass(frozen=True)
class ClassStubInfo:
    """Everything needed to render a class in a .pyi stub."""

    cls: type
    class_name: str
    module: str
    source_file: str
    bases: tuple[type, ...]
    init_signature: str | None  # None if no own __init__
    injected_attrs: tuple[InjectedAttr, ...]
    methods: tuple[MethodStub, ...]
    class_attrs: tuple[ClassAttrStub, ...] = ()


# ── Type narrowing ──────────────────────────────────────────────


def narrowest_common_base(classes: list[type], fallback: type) -> type:
    """Find the most specific ancestor shared by all *classes*.

    Returns *fallback* when the list is empty or no common ancestor
    (other than ``object``) can be found.
    """
    if not classes:
        return fallback
    if len(classes) == 1:
        return classes[0]

    common = set.intersection(*(set(c.__mro__) for c in classes))
    common.discard(object)

    if not common:
        return fallback

    # The most specific candidate is a subclass of every other member.
    for candidate in sorted(common, key=lambda c: len(c.__mro__), reverse=True):
        if all(issubclass(candidate, other) for other in common):
            return candidate

    return fallback


# ── Main collection entry point ─────────────────────────────────


def collect_class_stub_info(cls: type) -> ClassStubInfo | None:
    """Return stub info for *cls* if it has injected wired attributes.

    Returns ``None`` when the class has no wiring, no *injected* fields
    (all wired params already appear in ``__init__``), or when the
    source file cannot be determined.
    """
    from conscribe.wiring import resolve_wiring

    try:
        resolved = resolve_wiring(cls)
    except Exception:
        return None

    if not resolved:
        return None

    # Determine own __init__ params ───────────────────────────────
    has_own_init = "__init__" in cls.__dict__
    init_param_names: set[str] = set()
    init_signature: str | None = None

    if has_own_init:
        try:
            sig = inspect.signature(cls.__init__)
            init_param_names = set(sig.parameters.keys()) - {"self"}
            init_signature = render_signature(sig)
            if sig.return_annotation is inspect.Parameter.empty:
                init_signature += " -> None"
        except (ValueError, TypeError):
            pass

    # Identify injected wired fields ──────────────────────────────
    injected_attrs: list[InjectedAttr] = []
    for param_name, wiring in resolved.items():
        if param_name in init_param_names:
            continue  # already in __init__ — not injected
        attr_type = _resolve_wired_type(wiring)
        injected_attrs.append(
            InjectedAttr(
                name=param_name,
                resolved_type=attr_type,
                registry_name=wiring.registry_name,
            )
        )

    if not injected_attrs:
        return None

    # Source file ──────────────────────────────────────────────────
    try:
        source_file = inspect.getfile(cls)
    except (TypeError, OSError):
        return None

    injected_names = {attr.name for attr in injected_attrs}
    class_attrs = _collect_class_attrs(cls, skip_names=injected_names)

    return ClassStubInfo(
        cls=cls,
        class_name=cls.__name__,
        module=cls.__module__,
        source_file=source_file,
        bases=tuple(cls.__bases__),
        init_signature=init_signature,
        injected_attrs=tuple(injected_attrs),
        methods=_collect_own_methods(cls),
        class_attrs=class_attrs,
    )


# ── Internal helpers ─────────────────────────────────────────────


def _resolve_wired_type(wiring: Any) -> type:
    """Resolve the narrowest type for a wired attribute."""
    from conscribe.registration.registry import get_registry

    if wiring.registry_name is None:
        return str  # Mode 3: literal list

    registry = get_registry(wiring.registry_name)
    if registry is None:
        return str

    classes: list[type] = []
    for key in wiring.allowed_keys:
        try:
            classes.append(registry.get(key))
        except Exception:
            continue

    if not classes:
        return registry.protocol

    return narrowest_common_base(classes, registry.protocol)


_SKIP_NAMES: frozenset[str] = frozenset(
    {
        "__dict__",
        "__weakref__",
        "__doc__",
        "__module__",
        "__qualname__",
        "__abstractmethods__",
        "__init__",
        "__init_subclass__",
        # conscribe-specific attributes
        "__wiring__",
        "__abstract__",
        "__registry_key__",
        "__registry_keys__",
        "__skip_registries__",
        "__registration_filter__",
        "__propagate__",
        "__propagate_depth__",
        "__config_schema__",
        "__config_annotated_only__",
        "__config_mro_scope__",
        "__config_mro_depth__",
        "__wired_fields__",
        "__degraded_fields__",
    }
)


def _collect_own_methods(cls: type) -> tuple[MethodStub, ...]:
    """Collect methods defined directly on *cls* (not inherited)."""
    methods: list[MethodStub] = []

    for name, obj in cls.__dict__.items():
        if name in _SKIP_NAMES:
            continue

        decorators: list[str] = []
        func: Any = None

        if isinstance(obj, classmethod):
            decorators.append("@classmethod")
            func = obj.__func__
        elif isinstance(obj, staticmethod):
            decorators.append("@staticmethod")
            func = obj.__func__
        elif isinstance(obj, property):
            decorators.append("@property")
            func = obj.fget
        elif callable(obj):
            func = obj
        else:
            continue

        if func is None:
            continue

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue

        methods.append(
            MethodStub(
                name=name,
                signature=render_signature(sig),
                decorators=tuple(decorators),
                is_async=_is_async_function(func),
            )
        )

    return tuple(methods)


def _is_async_function(func: Any) -> bool:
    """Return ``True`` for coroutine *and* async-generator functions.

    ``inspect.iscoroutinefunction`` alone misses ``async def`` bodies that
    ``yield`` — those are async generators, and they still need ``async def``
    in the stub.  Unwraps ``functools.wraps``-style chains via ``__wrapped__``.
    """
    seen: set[int] = set()
    target = func
    while target is not None and id(target) not in seen:
        seen.add(id(target))
        try:
            if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
                return True
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return False
        target = getattr(target, "__wrapped__", None)
    return False


# ── Signature rendering ──────────────────────────────────────────


class _Source:
    """Carries pre-rendered source text through ``str(inspect.Signature)``.

    ``inspect`` formats both annotations and defaults with ``repr()``, so
    wrapping a value in this shim lets us decide the exact text that lands
    in the ``.pyi`` while still reusing ``Signature.__str__`` for all the
    positional-only ``/``, keyword-only ``*`` and ``**kwargs`` placement.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def render_signature(sig: inspect.Signature) -> str:
    """Render *sig* as ``.pyi``-safe source text.

    ``str(inspect.Signature)`` cannot be trusted for stub output: it
    ``repr()``s defaults straight into the signature, which emits things
    like ``<factory>`` (the dataclass ``default_factory`` sentinel),
    ``FieldInfo(...)`` or ``<object at 0x...>`` — none of which are valid
    Python.  Defaults that cannot be rendered faithfully collapse to the
    ``...`` placeholder that PEP 484 defines for exactly this purpose.

    String annotations (PEP 563 / explicit forward refs) are emitted
    unquoted, since the stub already carries ``from __future__ import
    annotations``.  This also avoids double-quoting an annotation whose
    source text was itself a string literal.
    """
    params = []
    for param in sig.parameters.values():
        annotation = param.annotation
        if annotation is not inspect.Parameter.empty:
            annotation = _Source(_annotation_text(annotation))
        default = param.default
        if default is not inspect.Parameter.empty:
            default = _Source(_default_to_source(default))
        params.append(param.replace(annotation=annotation, default=default))

    return_annotation = sig.return_annotation
    if return_annotation is not inspect.Signature.empty:
        return_annotation = _Source(_annotation_text(return_annotation))

    try:
        return str(sig.replace(parameters=params, return_annotation=return_annotation))
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return str(sig)


def _annotation_text(ann: Any) -> str:
    """Render a single annotation as stub source text."""
    if isinstance(ann, str):
        text = ann.strip()
        if _parses_as_expression(text):
            return text
        return repr(ann)

    forward_arg = getattr(ann, "__forward_arg__", None)
    if isinstance(forward_arg, str):
        return _annotation_text(forward_arg)

    try:
        return inspect.formatannotation(ann)
    except Exception:  # pragma: no cover - defensive
        return "Any"


def _parses_as_expression(text: str) -> bool:
    """Return ``True`` when *text* is a syntactically valid expression."""
    if not text:
        return False
    try:
        ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return False
    return True


# Types whose ``repr()`` round-trips to valid, self-contained source.
_LITERAL_SCALARS: tuple[type, ...] = (bool, int, float, complex, str, bytes)
_LITERAL_CONTAINERS: tuple[type, ...] = (list, tuple, set, frozenset, dict)


def _default_to_source(value: Any) -> str:
    """Render a parameter default as stub source, or ``...`` if impossible.

    Only values whose ``repr()`` is guaranteed to be valid, self-contained
    source survive; everything else (``default_factory`` sentinels, pydantic
    ``FieldInfo``, ``WiredField`` descriptors, arbitrary instances, enums)
    degrades to ``...`` — the PEP 484 stub placeholder, which still tells a
    type checker the parameter is optional.
    """
    if value is None:
        return "None"
    if value is Ellipsis:
        return "..."
    # ``type(value) in`` (not isinstance): subclasses such as IntEnum or a
    # str-subclass may have a repr that is not valid source.
    if type(value) in _LITERAL_SCALARS:
        return repr(value)
    if type(value) in _LITERAL_CONTAINERS:
        rendered = repr(value)
        try:
            if ast.literal_eval(rendered) == value:
                return rendered
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return "..."


def _collect_class_attrs(
    cls: type, *, skip_names: set[str]
) -> tuple[ClassAttrStub, ...]:
    """Collect own class-level annotated attributes for stub rendering.

    Reads ``cls.__dict__["__annotations__"]`` (own annotations only, not
    inherited) and produces a ``ClassAttrStub`` per non-dunder, non-injected
    name. Skips names already represented as injected wired fields (those
    are rendered separately with a ``# wired from:`` comment).

    Annotations are kept as strings under ``from __future__ import
    annotations``; resolved type objects are rendered via ``__name__`` or
    ``repr`` as a best-effort fallback.
    """
    annotations = cls.__dict__.get("__annotations__")
    if not annotations:
        return ()

    out: list[ClassAttrStub] = []
    for name, ann in annotations.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in skip_names:
            continue
        out.append(ClassAttrStub(name=name, annotation=_render_annotation(ann)))
    return tuple(out)


def _render_annotation(ann: Any) -> str:
    """Render an annotation as a stub-safe string."""
    if isinstance(ann, str):
        return ann
    name = getattr(ann, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(ann)
