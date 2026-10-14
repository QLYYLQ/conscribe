"""Collect what goes into a ``.pyi`` stub for one registered class.

Selection
---------
A ``__wiring__`` entry is written into the stub only when it describes an
attribute the framework injects at runtime:

* the entry has **no runtime slot** -- no constructor parameter (inherited
  and ``**kwargs``-forwarded parameters included) and no non-``ClassVar``
  class annotation anywhere in the MRO (see
  :func:`conscribe.introspect.wiring_slots`).  An entry *with* a slot is
  already declared by the source; a stub could only make its type worse;
* the entry is **not** ``declare_only`` (a declaration-only constraint is
  never an attribute).

The attribute is typed as the target registry's **protocol** (``str`` for a
Mode 3 literal list) -- never as the narrowest common base of the registered
implementations, which breaks protocol conformance of mutable attributes and
drifts as implementations are registered.

A class with no such entry has no stub.  Wiring problems are never
swallowed: :func:`collect_class_stub_info` resolves the class's wiring and
lets :class:`~conscribe.exceptions.WiringResolutionError` propagate.

Members
-------
Besides the injected attributes, a class stub repeats the class's own
surface so the stub (which replaces its module for type checkers) does not
hide it (:func:`collect_class_members`):

* its own constructor, methods (``async`` kept), properties with their
  setters and deleters, ``functools.cached_property`` members;
* methods behind wrappers: ``functools.cache`` / ``lru_cache`` keep the
  wrapped function's signature, ``contextlib.contextmanager`` /
  ``asynccontextmanager`` methods return a context manager, and any other
  wrapper whose code is not the method's own (``functools.wraps``
  decorators, ``partialmethod``, ``singledispatchmethod``) is declared as
  accepting any arguments -- conscribe cannot see what it does;
* class-level annotations, and unannotated class-level values
  (``MAX_STEPS = 30``), typed from the value;
* classes nested in the class body.

This module only *collects* them (raw annotations and values plus the class
that declares them); :mod:`conscribe.stubs.generator` resolves and renders
them.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import types
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from conscribe.config import _annotations as _ann
from conscribe.config._utils import own_init


@dataclass(frozen=True)
class InjectedAttr:
    """A wired attribute the framework injects at runtime.

    Attributes:
        name: The attribute name (the ``__wiring__`` entry name).
        resolved_type: The type written into the stub: the target registry's
            protocol, or ``str`` for a Mode 3 literal list.
        registry_name: Target registry name (``None`` for Mode 3).
    """

    name: str
    resolved_type: type
    registry_name: str | None  # None for Mode 3 (literal list)


#: ``MethodStub.kind`` values.
METHOD_KINDS: frozenset[str] = frozenset(
    {"function", "contextmanager", "asynccontextmanager", "opaque"}
)


@dataclass(frozen=True)
class MethodStub:
    """A method-like member rendered as a ``def`` in the stub.

    Attributes:
        name: Member name.
        function: The function whose signature is rendered (for a property:
            its getter, setter or deleter).  For a context-manager method,
            the generator function the decorator wrapped; for an
            ``"opaque"`` member, the wrapper itself.
        decorators: Symbolic decorators, outermost first: ``"classmethod"``,
            ``"staticmethod"``, ``"property"``, ``"setter"``, ``"deleter"``,
            ``"cached_property"``, ``"abstractmethod"``.
        is_async: Render ``async def`` (coroutine and async-generator
            functions).  Without it every ``await`` on the method is a type
            error downstream.
        kind: How the signature is rendered (one of :data:`METHOD_KINDS`):
            ``"function"`` -- the function's own signature;
            ``"contextmanager"`` / ``"asynccontextmanager"`` -- the
            parameters of *function* returning
            ``AbstractContextManager[T]`` / ``AbstractAsyncContextManager[T]``
            (``T`` from the generator's ``Iterator[T]`` annotation; always a
            plain ``def``); ``"opaque"`` -- a wrapper conscribe cannot see
            through, declared as accepting any arguments and returning
            ``Any``.
    """

    name: str
    function: Any
    decorators: tuple[str, ...] = ()
    is_async: bool = False
    kind: str = "function"


@dataclass(frozen=True)
class ClassAttrStub:
    """A class-body annotation rendered as an attribute declaration.

    Attributes:
        name: Attribute name.
        annotation: The annotation as declared (a string under PEP 563),
            read PEP 649-safely.
        owner: The class whose body declares it; the annotation is resolved
            in that class's namespace.
        has_default: The class provides a value for it.  Rendered as
            ``= ...`` in classes a ``dataclass_transform`` applies to (e.g.
            pydantic models), where it decides whether the field is required.
    """

    name: str
    annotation: Any
    owner: type
    has_default: bool = False


@dataclass(frozen=True)
class ClassValueStub:
    """An unannotated class-level value (``MAX_STEPS = 30``).

    Rendered as an attribute declaration typed from the value
    (``MAX_STEPS: int``); a typing construct (``Pair = tuple[int, int]``)
    is rendered as a type alias.

    Attributes:
        name: Attribute name.
        value: The value in the class ``__dict__``.
        owner: The class whose body binds it.
    """

    name: str
    value: Any
    owner: type


@dataclass(frozen=True)
class ClassMembers:
    """The own surface of a class (see :func:`collect_class_members`).

    Attributes:
        init: The class's own constructor, or ``None`` when it inherits one.
        methods: Own methods / properties / cached properties, in
            definition order.
        class_attrs: Own class-level annotations.
        values: Own unannotated class-level values.
        nested: ``(attribute name, class)`` of the classes defined in the
            class body.
    """

    init: MethodStub | None = None
    methods: tuple[MethodStub, ...] = ()
    class_attrs: tuple[ClassAttrStub, ...] = ()
    values: tuple[ClassValueStub, ...] = ()
    nested: tuple[tuple[str, type], ...] = ()


@dataclass(frozen=True)
class ClassStubInfo:
    """Everything needed to render one class in a ``.pyi`` stub.

    Attributes:
        cls: The class.
        class_name: Its ``__name__``.
        module: Its ``__module__`` -- the module whose stub holds it.
        source_file: Its source file (``None`` when the module has none).
        bases: ``cls.__bases__`` (the generator renders ``__orig_bases__``
            when the class has them, so generic bases keep their arguments).
        injected_attrs: The attributes the framework injects (see the
            module docstring), in ``__wiring__`` order.
        init: The class's own constructor, or ``None`` when it inherits one.
        methods: Own methods / properties / cached properties, in
            definition order.
        class_attrs: Own class-level annotations (dunders, injected names,
            ``InitVar`` and ``KW_ONLY`` pseudo-fields excluded).
        values: Own unannotated class-level values (dunders, injected names
            and names a base class already declares excluded).
        nested: ``(attribute name, class)`` of the classes defined in the
            class body; the stub declares them (nested) as well.
    """

    cls: type
    class_name: str
    module: str
    source_file: str | None
    bases: tuple[type, ...]
    injected_attrs: tuple[InjectedAttr, ...]
    init: MethodStub | None = None
    methods: tuple[MethodStub, ...] = ()
    class_attrs: tuple[ClassAttrStub, ...] = ()
    values: tuple[ClassValueStub, ...] = ()
    nested: tuple[tuple[str, type], ...] = ()


# ── Type narrowing (public helper, kept for 1.x callers) ─────────


def narrowest_common_base(classes: list[type], fallback: type) -> type:
    """Find the most specific ancestor shared by all *classes*.

    Returns *fallback* when the list is empty or no common ancestor
    (other than ``object``) can be found.

    Stubs no longer use this (an injected attribute is typed as its
    registry's protocol); it stays available as a utility.
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
    """Return stub info for *cls*, or ``None`` when it injects nothing.

    The class's wiring is resolved first (:func:`conscribe.wiring.resolve_wiring`),
    so every wiring problem raises -- nothing is skipped silently.  Then
    every merged ``__wiring__`` entry with a runtime slot, and every
    ``declare_only`` entry, is left out (see the module docstring).

    Args:
        cls: A class (usually a registered implementation).

    Returns:
        A :class:`ClassStubInfo`, or ``None`` when the class declares no
        wiring or every entry has a slot / is ``declare_only``.

    Raises:
        WiringResolutionError: The class's ``__wiring__`` has problems
            (unknown registry or key, empty registry, malformed entry, ...).
    """
    from conscribe.introspect import wiring_slots
    from conscribe.wiring import resolve_wiring

    resolved = resolve_wiring(cls)
    if not resolved:
        return None

    injected: list[InjectedAttr] = []
    for name, entry in wiring_slots(cls).items():
        if entry.slot is not None or entry.declare_only:
            continue
        wiring = resolved[name]
        injected.append(
            InjectedAttr(
                name=name,
                resolved_type=_injected_type(wiring),
                registry_name=wiring.registry_name,
            )
        )
    if not injected:
        return None

    members = collect_class_members(cls, skip=frozenset(attr.name for attr in injected))
    return ClassStubInfo(
        cls=cls,
        class_name=cls.__name__,
        module=cls.__module__,
        source_file=_source_file(cls),
        bases=tuple(cls.__bases__),
        injected_attrs=tuple(injected),
        init=members.init,
        methods=members.methods,
        class_attrs=members.class_attrs,
        values=members.values,
        nested=members.nested,
    )


# ── Internal helpers ─────────────────────────────────────────────


def _injected_type(wiring: Any) -> type:
    """The registry protocol for registry entries, ``str`` for Mode 3."""
    from conscribe.registration.registry import get_registry

    if wiring.registry_name is None:
        return str  # Mode 3: literal list
    registry = get_registry(wiring.registry_name)
    if registry is None:  # pragma: no cover - resolve_wiring() already raised
        raise LookupError(f"registry {wiring.registry_name!r} disappeared during stub collection")
    return registry.protocol


def _source_file(cls: type) -> str | None:
    try:
        return inspect.getfile(cls)
    except (TypeError, OSError):
        return None


#: Own-``__dict__`` names that are never rendered as members.
_SKIP_NAMES: frozenset[str] = frozenset(
    {
        "__init__",  # rendered separately (``init``)
        # class-creation hooks, not instance API
        "__init_subclass__",
        "__subclasshook__",
        "__class_getitem__",
        # conscribe-specific attributes
        "__wiring__",
        "__abstract__",
        "__registry_key__",
        "__registry_keys__",
        "__registry_name__",
        "__registry_origin__",
        "__skip_registries__",
        "__registration_filter__",
        "__propagate__",
        "__propagate_depth__",
        "__config_schema__",
        "__config_annotated_only__",
        "__config_mro_scope__",
        "__config_mro_depth__",
        "__config_from_param__",
        "__wired_fields__",
        "__degraded_fields__",
    }
)

#: Machinery that ``abc`` / ``typing`` store in a class ``__dict__`` under
#: non-dunder names: never a value of the class's own API.
_MACHINERY_NAMES: frozenset[str] = frozenset({"_abc_impl", "_is_protocol", "_is_runtime_protocol"})

def _gen_probe() -> Iterator[None]:  # pragma: no cover - never iterated
    yield None


async def _async_gen_probe() -> AsyncIterator[None]:  # pragma: no cover - never iterated
    yield None


#: The code objects shared by every function ``contextlib.contextmanager`` /
#: ``asynccontextmanager`` return (their inner ``helper``).
_CONTEXTMANAGER_CODE = contextlib.contextmanager(_gen_probe).__code__
_ASYNCCONTEXTMANAGER_CODE = contextlib.asynccontextmanager(_async_gen_probe).__code__

#: Wrappers that keep the wrapped function's call signature by contract.
_SIGNATURE_PRESERVING: tuple[type, ...] = tuple(
    t for t in (getattr(functools, "_lru_cache_wrapper", None),) if isinstance(t, type)
)


def collect_class_members(cls: type, *, skip: frozenset[str] = frozenset()) -> ClassMembers:
    """Collect the own surface of *cls*: constructor, methods, attributes, nested classes.

    Only what *cls* itself defines (its ``__dict__`` and its own class-body
    annotations, read PEP 649-safely); inherited members come from the
    rendered bases.  Functions generated by ``@dataclass`` / attrs (other
    than ``__init__``) are left out.

    Unannotated values are collected unless a base class already declares
    the name (re-declaring it could only conflict with the base's type),
    the value is conscribe's own (a ``WiredField`` placeholder) or it is
    ``abc`` / ``typing`` machinery.

    Args:
        cls: The class.
        skip: Names to leave out (the injected attributes).

    Returns:
        A :class:`ClassMembers`.
    """
    init_func = own_init(cls)
    init = MethodStub("__init__", init_func) if init_func is not None else None
    own_annotations = _ann.own_annotations(cls)

    methods: list[MethodStub] = []
    values: list[ClassValueStub] = []
    nested: list[tuple[str, type]] = []
    for name, obj in list(cls.__dict__.items()):
        if name in skip or name in _SKIP_NAMES or not name.isidentifier():
            continue
        stubs = _member_stubs(cls, name, obj)
        if stubs is not None:  # method-like (possibly nothing to render)
            methods.extend(stubs)
        elif _is_dunder(name) or name in own_annotations:
            continue  # dunders are machinery; annotated names are class_attrs
        elif _is_nested_class(cls, name, obj):
            nested.append((name, obj))
        elif _is_own_value(cls, name, obj):
            values.append(ClassValueStub(name=name, value=obj, owner=cls))

    class_attrs: list[ClassAttrStub] = []
    for name, raw in own_annotations.items():
        if _is_dunder(name):
            continue
        if name in skip or not name.isidentifier():
            continue
        if _ann.is_kw_only_sentinel(raw, cls) or _ann.is_initvar(raw, cls):
            continue  # constructor-only pseudo-fields, not attributes
        class_attrs.append(
            ClassAttrStub(
                name=name,
                annotation=raw,
                owner=cls,
                has_default=_has_default(cls, name),
            )
        )
    return ClassMembers(
        init=init,
        methods=tuple(methods),
        class_attrs=tuple(class_attrs),
        values=tuple(values),
        nested=tuple(nested),
    )


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _member_stubs(cls: type, name: str, obj: Any) -> list[MethodStub] | None:
    """The ``MethodStub`` objects for one own ``__dict__`` entry.

    ``None`` when *obj* is not method-like (a value, a nested class); an
    empty list for a method-like member that is not rendered.
    """
    if _is_dunder(name) and not _defined_in(cls, name, obj):
        # A special method copied in by the machinery (``Enum.__new__``,
        # a Protocol's replaced ``__init__``), not written in this class --
        # or a dunder value (``__slots__``, ``__match_args__``).
        return []
    if isinstance(obj, property):
        getter = _method_target(obj.fget)
        if getter is None:
            return []
        stubs = [_method(name, getter, ("property",))]
        for accessor, decorator in ((obj.fset, "setter"), (obj.fdel, "deleter")):
            target = _method_target(accessor)
            if target is not None:
                stubs.append(_method(name, target, (decorator,)))
        return stubs
    if isinstance(obj, functools.cached_property):
        target = _method_target(obj.func)
        return [_method(name, target, ("cached_property",))] if target is not None else []
    if isinstance(obj, staticmethod):
        target = _method_target(obj.__func__)
        if target is None:
            return []
        # ``__new__`` is an implicit static method: stubs write it bare.
        decorators = () if name == "__new__" else ("staticmethod",)
        return [_method(name, target, decorators)]
    if isinstance(obj, classmethod):
        target = _method_target(obj.__func__)
        return [_method(name, target, ("classmethod",))] if target is not None else []
    if isinstance(obj, (functools.partialmethod, functools.singledispatchmethod)):
        # Bound like a method, but its call signature is not one function's.
        inner = getattr(obj, "func", None)
        decorators: tuple[str, ...] = ()
        if isinstance(inner, classmethod):
            decorators = ("classmethod",)
        elif isinstance(inner, staticmethod):
            decorators = ("staticmethod",)
        return [MethodStub(name=name, function=obj, decorators=decorators, kind="opaque")]
    target = _method_target(obj)
    if target is None:
        return None
    kind, func = target
    if kind == "function" and _is_generated_function(cls, name, func):
        return []
    return [_method(name, target, ())]


def _method_target(obj: Any, depth: int = 0) -> tuple[str, Any] | None:
    """``(kind, function)`` of a method-like callable, ``None`` for anything else.

    * a plain function: ``("function", obj)``;
    * a ``contextlib.contextmanager`` / ``asynccontextmanager`` method:
      ``("contextmanager" | "asynccontextmanager", <the generator function>)``;
    * any other function whose ``__wrapped__`` chain leads to different code
      (a ``functools.wraps`` decorator): ``("opaque", obj)`` -- the wrapper
      decides how it is called, and conscribe cannot see that;
    * a ``functools.cache`` / ``lru_cache`` wrapper: the wrapped function
      (they keep its call signature);
    * any other callable with a ``__wrapped__``: ``("opaque", obj)``.
    """
    if inspect.isfunction(obj):
        code = obj.__code__
        if code is _CONTEXTMANAGER_CODE or code is _ASYNCCONTEXTMANAGER_CODE:
            wrapped = getattr(obj, "__wrapped__", None)
            if inspect.isfunction(wrapped):
                kind = "contextmanager" if code is _CONTEXTMANAGER_CODE else "asynccontextmanager"
                return kind, wrapped
            return "opaque", obj
        target = _ann.unwrap_function(obj)
        if target is not obj and getattr(target, "__code__", None) is not code:
            return "opaque", obj
        return "function", obj
    if _SIGNATURE_PRESERVING and isinstance(obj, _SIGNATURE_PRESERVING) and depth < 8:
        wrapped = getattr(obj, "__wrapped__", None)
        if inspect.isfunction(wrapped):
            return _method_target(wrapped, depth + 1)
        return "opaque", obj
    if callable(obj) and hasattr(obj, "__wrapped__") and not isinstance(obj, type):
        return "opaque", obj
    return None


def _is_nested_class(cls: type, name: str, obj: Any) -> bool:
    """True for a class defined in *cls*'s body (``class Outer: class Inner``)."""
    return (
        _is_class(obj)
        and getattr(obj, "__module__", None) == cls.__module__
        and getattr(obj, "__qualname__", None) == f"{cls.__qualname__}.{name}"
    )


def _is_class(obj: Any) -> bool:
    """``isinstance(obj, type)``, except ``list[int]`` (a class on Python 3.10)."""
    return isinstance(obj, type) and not isinstance(obj, types.GenericAlias)


def _is_own_value(cls: type, name: str, obj: Any) -> bool:
    """True for an unannotated value the stub should declare (see :func:`collect_class_members`)."""
    if name in _MACHINERY_NAMES or _is_conscribe_marker(obj):
        return False  # abc / typing machinery, WiredField placeholders, tombstones
    for base in cls.__mro__[1:]:
        if name in base.__dict__ or name in _ann.own_annotations(base):
            return False
    return True


def _defined_in(cls: type, name: str, obj: Any) -> bool:
    """True when the function behind *obj* was written in *cls*'s own body."""
    func = obj
    if isinstance(obj, (staticmethod, classmethod)):
        func = obj.__func__
    elif isinstance(obj, property):
        func = obj.fget
    elif isinstance(obj, functools.cached_property):
        func = obj.func
    qualname = getattr(_ann.unwrap_function(func), "__qualname__", None)
    return qualname == f"{cls.__qualname__}.{name}"


def _method(name: str, target: tuple[str, Any], decorators: tuple[str, ...]) -> MethodStub:
    kind, func = target
    if getattr(func, "__isabstractmethod__", False):
        decorators = decorators + ("abstractmethod",)
    if kind == "function":
        is_async = _is_async_function(func)
    elif kind == "opaque":
        # The wrapper's own nature decides what calling it returns.
        is_async = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
    else:  # a context-manager method is a plain function returning a manager
        is_async = False
    return MethodStub(name=name, function=func, decorators=decorators, is_async=is_async, kind=kind)


def _is_generated_function(cls: type, name: str, func: Any) -> bool:
    """True for methods written by ``@dataclass`` / attrs (``__repr__``, ``__eq__``...)."""
    target = _ann.unwrap_function(func)
    filename = getattr(getattr(target, "__code__", None), "co_filename", "")
    if filename.startswith("<attrs generated"):
        return True
    return (
        filename == "<string>"
        and "__dataclass_fields__" in cls.__dict__
        and getattr(target, "__qualname__", None) == f"{cls.__qualname__}.{name}"
    )


def _has_default(cls: type, name: str) -> bool:
    """True when *cls* gives attribute *name* a value (or a field default).

    conscribe's own descriptors (a ``WiredField`` placeholder, the tombstone
    over an excluded entry) are no value.
    """
    import dataclasses

    fields = _pydantic_fields(cls)
    if fields is not None and name in fields:
        is_required = getattr(fields[name], "is_required", None)
        return not is_required() if callable(is_required) else False
    private = getattr(cls, "__private_attributes__", None)
    if isinstance(private, dict) and name in private:
        return True
    dc_fields = getattr(cls, "__dataclass_fields__", None)
    if isinstance(dc_fields, dict) and name in dc_fields:
        field = dc_fields[name]
        return (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return not _is_conscribe_marker(klass.__dict__[name])
    return False


def _is_conscribe_marker(value: Any) -> bool:
    """True for conscribe's own class-level objects (``WiredField``, tombstones, ...)."""
    module = getattr(type(value), "__module__", "") or ""
    return module == "conscribe" or module.startswith("conscribe.")


def _pydantic_fields(cls: type) -> dict[str, Any] | None:
    """``{name: FieldInfo}`` of a pydantic model / pydantic dataclass, else ``None``."""
    fields = getattr(cls, "__pydantic_fields__", None)
    if isinstance(fields, dict):
        return fields
    from conscribe.config._utils import is_pydantic_model

    if is_pydantic_model(cls):
        fields = getattr(cls, "model_fields", None)  # pydantic < 2.10
        if isinstance(fields, dict):
            return fields
    return None


def _is_async_function(func: Any) -> bool:
    """Return ``True`` for coroutine *and* async-generator functions.

    ``inspect.iscoroutinefunction`` alone misses ``async def`` bodies that
    ``yield`` -- those are async generators, and they still need ``async def``
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
