"""Annotation reading and owner-aware resolution (internal engine).

Everything conscribe needs to know about a type annotation goes through this
module, so the extractor, the wiring injector, the stub writer, the
fingerprint and :mod:`conscribe.introspect` all agree:

Reading (PEP 649-safe)
    :func:`own_annotations` returns the annotations a class declares in its
    own body.  On Python 3.14 a class body no longer stores them in
    ``klass.__dict__["__annotations__"]`` (only ``__annotate__`` is there),
    so never read that key directly.  :func:`safe_signature` is
    :func:`inspect.signature` that cannot raise ``NameError`` on 3.14 (it
    asks for ``Format.FORWARDREF``) and returns ``None`` when a callable has
    no introspectable signature.

Resolution (D3)
    An annotation is evaluated in the namespace of the class that *declares*
    it: a hand-written ``__init__`` in its own ``__globals__``; a dataclass
    field in the module of the dataclass that declares the field; a class
    annotation in its class's module.  Lookup order matches
    :func:`typing.get_type_hints`: for class-body annotations module
    globals, then the class namespace, then builtins; for a function
    signature the function's globals, then builtins -- the class namespace is
    only a last resort there, so a method named ``set`` never shadows the
    builtin (see :class:`AnnotationContext`).  :func:`resolve_annotation`
    never imports anything (in
    particular it does **not** import ``if TYPE_CHECKING:`` blocks) and never
    emits a warning itself.  It returns a :class:`ResolvedAnnotation` that
    says what happened; the *caller* decides whether the field matters (a
    wired receptor takes its type from wiring, an annotated-only filter may
    drop it) and then calls :meth:`ResolvedAnnotation.raise_if_error` /
    :meth:`ResolvedAnnotation.warn`.  The outcomes:

    ``resolved``
        ``value`` is the evaluated annotation (``Annotated`` metadata kept).
    ``unannotated``
        No annotation was written; ``value`` is ``inspect.Parameter.empty``.
    ``degraded`` (a ``warn()`` is due)
        * ``reason="type_checking_only"``: every unresolvable name is bound
          only under ``if TYPE_CHECKING:`` in the declaring module (AST scan,
          no import) and none of them is needed by ``Annotated`` metadata.
          ``value`` is ``Any`` (or ``Annotated[Any, <recovered metadata>]``);
          the warning category is :class:`~conscribe.exceptions.DegradationWarning`.
        * ``reason="function_locals"``: the declaring class is defined inside
          a function and the annotation needs that function's locals, which
          conscribe cannot see.  ``value`` as above; the warning category is
          :class:`~conscribe.exceptions.ExtractionWarning`.
    ``error`` (a ``raise_if_error()`` is due)
        * ``reason="undefined_name"``: a name exists nowhere (module globals,
          class namespace, builtins, TYPE_CHECKING imports) -- a typo or a
          missing import.
        * ``reason="metadata"``: ``Annotated`` metadata (``Field(...)`` etc.)
          needs a name that is only imported under ``if TYPE_CHECKING:``, or
          the metadata of a degrading annotation raised when evaluated.
        * ``reason="evaluation_failed"``: evaluating the annotation raised
          something other than a missing name (``AttributeError`` on a module,
          a ``TypeError`` from ``Field(...)``, a ``SyntaxError``...).
        ``value`` is the raw annotation.

    Resolution itself never adds ``Optional`` (Python <= 3.10's
    ``typing.get_type_hints`` does so for functions; conscribe does not use
    it for that reason).  The 2.0 None rule lives in
    :func:`conscribe.config._fieldinfo.config_field`, backed by
    :func:`admits_none`.

Shape and kind
    :func:`annotation_shape` (``"scalar" | "list" | "dict"``),
    :func:`admits_none`, :func:`is_classvar`, :func:`is_kw_only_sentinel`,
    :func:`is_initvar` work on resolved objects, ``ForwardRef`` objects and
    unresolved strings alike (strings: names are looked up in the owner's
    namespace first, the spelling is only a fallback).

Warnings
    :func:`warn_at` emits conscribe's warnings at the user code they are
    about (the declaring class or function), not at conscribe's frames.
"""
from __future__ import annotations

import ast
import builtins
import collections.abc as _abc
import dataclasses
import functools
import importlib.util
import inspect
import operator
import os
import sys
import tokenize
import types
import typing
import warnings
import weakref
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    ForwardRef,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
)

from conscribe.exceptions import (
    ConfigAnnotationError,
    ConscribeWarning,
    DegradationWarning,
    ExtractionWarning,
)

_PY311 = sys.version_info >= (3, 11)
_PY312 = sys.version_info >= (3, 12)
_PY314 = sys.version_info >= (3, 14)

if _PY314:  # pragma: no cover - exercised by the 3.14 test run
    import annotationlib

#: ``inspect.Parameter.empty`` -- "no annotation" / "no default".
EMPTY: Any = inspect.Parameter.empty

NoneType = type(None)

#: ``get_origin()`` of ``Union[...]`` / ``Optional[...]`` and of PEP 604 ``X | Y``.
UNION_ORIGINS: tuple[Any, ...] = (Union, types.UnionType)

Shape = Literal["scalar", "list", "dict"]

__all__ = [
    "EMPTY",
    "AnnotationContext",
    "ResolvedAnnotation",
    "Shape",
    "TypeCheckingName",
    "UNION_ORIGINS",
    "admits_none",
    "annotation_owner",
    "annotation_shape",
    "annotation_text",
    "class_annotation_source",
    "class_context",
    "class_globals",
    "contains_forward_refs",
    "function_annotation_source",
    "function_context",
    "is_classvar",
    "is_initvar",
    "is_kw_only_sentinel",
    "is_none_node",
    "is_pseudo_field",
    "own_annotations",
    "resolve_annotation",
    "resolve_class_annotation",
    "safe_signature",
    "type_checking_imports",
    "type_checking_names",
    "unwrap_annotated",
    "unwrap_final",
    "unwrap_function",
    "unwrap_initvar",
    "warn_at",
]


# ===========================================================================
# Reading (PEP 649-safe)
# ===========================================================================


def own_annotations(klass: type) -> dict[str, Any]:
    """Return the annotations *klass* declares in its own body (not inherited).

    PEP 649-safe.  On Python 3.14 the values come from
    ``annotationlib.get_annotations(klass, format=Format.FORWARDREF)``: names
    that cannot be evaluated stay ``ForwardRef`` objects (possibly nested in
    typing constructs) instead of raising.  Modules using
    ``from __future__ import annotations`` yield plain strings on every
    version.  Before 3.14 the class's own ``__annotations__`` dict is copied.

    Use this for *what* a class declares (names, ``ClassVar`` checks, shape);
    use :func:`resolve_class_annotation` to evaluate one.

    Args:
        klass: Any class.

    Returns:
        A fresh ``{name: annotation}`` dict in declaration order (empty when
        the class declares none).
    """
    if _PY314:
        try:
            return dict(
                annotationlib.get_annotations(klass, format=annotationlib.Format.FORWARDREF)
            )
        except Exception:  # noqa: BLE001 - FORWARDREF failed; STRING never evaluates user code
            return dict(
                annotationlib.get_annotations(klass, format=annotationlib.Format.STRING)
            )
    ann = klass.__dict__.get("__annotations__")
    return dict(ann) if isinstance(ann, dict) else {}


def class_annotation_source(klass: type) -> tuple[dict[str, Any], dict[str, Any]]:
    """Own annotations of *klass* in their best *evaluable* form.

    Returns ``(annotations, closure)``.  Before 3.14 this is the class's own
    ``__annotations__`` and an empty closure.  On 3.14 each annotation is
    the native value (``Format.FORWARDREF``) when it evaluates; an
    annotation that still holds a forward reference (a name defined nowhere,
    or only under ``if TYPE_CHECKING:``) comes back as its source string
    (``Format.STRING``), and *closure* then holds the enclosing-function
    names the class body can see (so function-local classes resolve their
    local names), letting :func:`resolve_annotation` diagnose that
    annotation alone.  One failing annotation never turns its valid siblings
    into source text (which would lose, e.g., a lambda in their metadata).
    """
    if not _PY314:
        return own_annotations(klass), {}
    return _pep649_annotations(klass)  # pragma: no cover - 3.14 only


def function_annotation_source(func: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Annotations of *func* (unwrapped) in their best evaluable form.

    Same contract as :func:`class_annotation_source`, for functions.  The
    ``"return"`` entry is dropped.
    """
    target = unwrap_function(func)
    if not _PY314:
        ann = getattr(target, "__annotations__", None)
        result = dict(ann) if isinstance(ann, dict) else {}
        result.pop("return", None)
        return result, {}
    result, closure = _pep649_annotations(target)  # pragma: no cover - 3.14 only
    result.pop("return", None)  # pragma: no cover - 3.14 only
    return result, closure  # pragma: no cover - 3.14 only


def _pep649_annotations(obj: Any) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover - 3.14 only
    """Own annotations of a class or function on Python 3.14 (see
    :func:`class_annotation_source`): ``(annotations, closure)``."""
    try:
        values: dict[str, Any] | None = dict(
            annotationlib.get_annotations(obj, format=annotationlib.Format.FORWARDREF)
        )
    except (TypeError, ValueError):
        if not isinstance(obj, type):
            return {}, {}  # not a Python function: nothing to read
        values = None
    except Exception:  # noqa: BLE001 - evaluation failed even with stand-ins; use source text
        values = None
    if values is not None and not any(_holds_forward_ref(v) for v in values.values()):
        return values, {}
    try:
        strings = dict(annotationlib.get_annotations(obj, format=annotationlib.Format.STRING))
    except Exception:  # noqa: BLE001 - STRING never evaluates user code; nothing readable
        strings = {}
    closure = _closure_namespace(getattr(obj, "__annotate__", None))
    if values is None:
        return strings, closure
    merged = {
        name: strings.get(name, value) if _holds_forward_ref(value) else value
        for name, value in values.items()
    }
    return merged, closure


def _holds_forward_ref(value: Any) -> bool:  # pragma: no cover - 3.14 only
    """Does a ``Format.FORWARDREF`` annotation still hold a forward reference?

    Besides type positions (:func:`contains_forward_refs`) this looks inside
    ``Annotated`` metadata objects: ``Field(default=X)`` keeps a
    ``ForwardRef`` attribute when ``X`` cannot be evaluated.
    """
    if contains_forward_refs(value):
        return True
    return any(_object_holds_forward_ref(meta, 0) for meta in _metadata_objects(value, 0))


def _metadata_objects(tp: Any, depth: int) -> Iterator[Any]:  # pragma: no cover - 3.14 only
    """Every ``Annotated`` metadata object anywhere in a typing object."""
    if depth > 20:
        return
    origin = get_origin(tp)
    if origin is Literal:
        return
    args = get_args(tp)
    if origin is Annotated:
        yield from args[1:]
        args = args[:1]
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                yield from _metadata_objects(item, depth + 1)
        else:
            yield from _metadata_objects(arg, depth + 1)


_PLAIN_VALUES: tuple[type, ...] = (
    str, bytes, int, float, complex, bool, type(None), type, types.FunctionType,
    types.BuiltinFunctionType, types.MethodType, types.ModuleType, types.CodeType,
)


def _object_holds_forward_ref(obj: Any, depth: int) -> bool:  # pragma: no cover - 3.14 only
    if isinstance(obj, ForwardRef):
        return True
    if depth > 3 or isinstance(obj, _PLAIN_VALUES):
        return False
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_object_holds_forward_ref(item, depth + 1) for item in obj)
    if isinstance(obj, dict):
        return any(
            _object_holds_forward_ref(key, depth + 1) or _object_holds_forward_ref(value, depth + 1)
            for key, value in obj.items()
        )
    values: list[Any] = list(getattr(obj, "__dict__", {}).values())
    for klass in type(obj).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot not in ("__dict__", "__weakref__"):
                values.append(getattr(obj, slot, None))
    return any(_object_holds_forward_ref(value, depth + 1) for value in values)


#: :func:`safe_signature` results of plain functions:
#: ``func -> (validation key, signature)``.  The key holds the parts of a
#: function that :func:`inspect.signature` reads and that can be reassigned
#: (code, defaults, ``__wrapped__``, ``__signature__``); an entry whose key no
#: longer matches by identity is recomputed, and a replaced ``__init__`` is a
#: different function -- so nothing is ever cached across a class mutation.
_SIGNATURES: weakref.WeakKeyDictionary[Any, tuple[tuple[Any, ...], inspect.Signature | None]] = (
    weakref.WeakKeyDictionary()
)


def _signature_key(func: types.FunctionType) -> tuple[Any, ...]:
    own = func.__dict__
    return (
        func.__code__,
        func.__defaults__,
        func.__kwdefaults__,
        own.get("__wrapped__"),
        own.get("__signature__"),
        own.get("__text_signature__"),
    )


def safe_signature(obj: Any, *, follow_wrapped: bool = True) -> inspect.Signature | None:
    """:func:`inspect.signature` that never raises for annotation problems.

    On Python 3.14 annotations are requested in ``Format.FORWARDREF`` (names
    that cannot be evaluated become ``ForwardRef`` objects instead of a
    ``NameError``), falling back to ``Format.STRING``.  Earlier versions never
    evaluate annotations here (PEP 563 annotations stay strings).

    Signatures of plain functions are cached per function object (see
    ``_SIGNATURES``), except on 3.14 while a parameter annotation is still an
    unresolved ``ForwardRef`` (it may resolve later).

    Returns:
        The signature, or ``None`` when *obj* has no introspectable signature
        (C extensions, some class-based decorators): ``inspect.signature``
        raised ``ValueError`` or ``TypeError``.  Callers treat ``None`` as
        "unreadable" (and emit an ``ExtractionWarning`` where it matters).
    """
    if not (follow_wrapped and type(obj) is types.FunctionType):
        return _read_signature(obj, follow_wrapped=follow_wrapped)
    key = _signature_key(obj)
    cached = _SIGNATURES.get(obj)
    if cached is not None and all(a is b for a, b in zip(cached[0], key)):
        return cached[1]
    sig = _read_signature(obj, follow_wrapped=True)
    if sig is not None and not (
        _PY314
        and any(
            not isinstance(p.annotation, str) and contains_forward_refs(p.annotation)
            for p in sig.parameters.values()
        )
    ):
        _SIGNATURES[obj] = (key, sig)
    return sig


def _read_signature(obj: Any, *, follow_wrapped: bool) -> inspect.Signature | None:
    try:
        if _PY314:  # pragma: no cover - 3.14 only
            try:
                return inspect.signature(
                    obj,
                    follow_wrapped=follow_wrapped,
                    annotation_format=annotationlib.Format.FORWARDREF,
                )
            except (ValueError, TypeError):
                raise
            except Exception:  # noqa: BLE001 - FORWARDREF failed; STRING never evaluates user code
                return inspect.signature(
                    obj,
                    follow_wrapped=follow_wrapped,
                    annotation_format=annotationlib.Format.STRING,
                )
        return inspect.signature(obj, follow_wrapped=follow_wrapped)
    except (ValueError, TypeError):
        return None


def unwrap_function(func: Any) -> Any:
    """``inspect.unwrap`` that tolerates wrapper cycles."""
    try:
        return inspect.unwrap(func)
    except ValueError:  # a __wrapped__ cycle
        return func


def _closure_namespace(func: Any) -> dict[str, Any]:
    """Names an (annotate) function closes over, minus the class-body cell."""
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None)
    if code is None or not closure:
        return {}
    result: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, closure):
        if name == "__classdict__":
            continue
        try:
            result[name] = cell.cell_contents
        except ValueError:  # empty cell (name not bound yet)
            continue
    return result


# ===========================================================================
# Contexts: where an annotation is evaluated
# ===========================================================================


@dataclass(frozen=True)
class AnnotationContext:
    """The scope an annotation was written in.

    Name lookup order (:meth:`namespace`, :meth:`lookup`):

    * class-body annotations (``function is None``): closure, module
      globals, type parameters, the class namespace, builtins -- the
      :func:`typing.get_type_hints` order for classes;
    * annotations in a function signature (``function`` set): closure,
      module globals, type parameters, builtins and only then the class
      namespace, as a last resort -- what :func:`typing.get_type_hints` does
      for a function, plus class-level names nothing else defines.  A method
      called ``set`` or an attribute ``type = "x"`` never shadows a builtin
      in a constructor signature.

    Attributes:
        globalns: Globals of the declaring module (the live dict).
        owner: Declaring class (see the lookup order above).  ``None`` for a
            bare function.
        closure: Enclosing-function names visible to the annotation (only
            filled on Python 3.14, where annotations are closures).  They
            take precedence over the globals, like any enclosing scope.
        type_params: PEP 695 type parameters of the owner / function.
        function: The hand-written function whose signature holds the
            annotation (``None`` for class-body annotations).
    """

    globalns: Mapping[str, Any]
    owner: type | None = None
    closure: Mapping[str, Any] = field(default_factory=dict)
    type_params: tuple[Any, ...] = ()
    function: Any = None

    @property
    def module(self) -> str | None:
        """Name of the declaring module, if known."""
        name = self.globalns.get("__name__")
        if isinstance(name, str):
            return name
        if self.owner is not None:
            return getattr(self.owner, "__module__", None)
        return None

    @property
    def scope_qualname(self) -> str:
        """Qualified name of the declaring scope (function or class)."""
        if self.function is not None:
            return getattr(self.function, "__qualname__", "") or ""
        if self.owner is not None:
            return self.owner.__qualname__
        return ""

    @property
    def function_local(self) -> bool:
        """True when the declaring scope is defined inside a function."""
        return "<locals>" in self.scope_qualname

    def namespace(self) -> dict[str, Any]:
        """A fresh evaluation namespace in this context's lookup order.

        Builtins are found through the globals' ``__builtins__`` (``eval``
        inserts them when absent); in a function context, class-level names
        that shadow a builtin are left out so the builtin wins.
        """
        ns: dict[str, Any] = {}
        if self.owner is not None and self.function is None:
            ns.update(vars(self.owner))
        for param in self.type_params:
            name = getattr(param, "__name__", None)
            if isinstance(name, str):
                ns[name] = param
        ns.update(self.globalns)
        ns.update(self.closure)
        if self.owner is not None and self.function is not None:
            for name, value in vars(self.owner).items():
                if name not in ns and not hasattr(builtins, name):
                    ns[name] = value
        if "__builtins__" not in ns:
            ns["__builtins__"] = builtins
        return ns

    def eval_namespaces(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(globals, locals)`` for ``eval`` in this context's lookup order.

        The module globals are used as they are (no copy; lambdas created
        while evaluating see the live module).  *locals* only holds names
        that must win over them (the closure) or that the globals lack (type
        parameters, class-level names -- in a function context only those
        that are not builtins either), so ``eval``'s locals -> globals ->
        builtins order reproduces :meth:`namespace`.
        """
        globalns = self.globalns
        if not isinstance(globalns, dict):
            return self.namespace(), {}
        localns: dict[str, Any] = {}
        if self.owner is not None:
            function_scope = self.function is not None
            for name, value in vars(self.owner).items():
                if name not in globalns and not (function_scope and hasattr(builtins, name)):
                    localns[name] = value
        for param in self.type_params:
            name = getattr(param, "__name__", None)
            if isinstance(name, str) and name not in globalns:
                localns[name] = param
        localns.update(self.closure)
        return globalns, localns

    def lookup(self, name: str) -> tuple[bool, Any]:
        """Look *name* up the way evaluation would; ``(found, value)``."""
        if name in self.closure:
            return True, self.closure[name]
        if name in self.globalns:
            return True, self.globalns[name]
        for param in self.type_params:
            if getattr(param, "__name__", None) == name:
                return True, param
        in_class = self.owner is not None and name in vars(self.owner)
        if in_class and self.function is None:
            return True, vars(self.owner)[name]  # type: ignore[arg-type]
        if hasattr(builtins, name):
            return True, getattr(builtins, name)
        if in_class:
            return True, vars(self.owner)[name]  # type: ignore[arg-type]
        return False, None


_EMPTY_CONTEXT = AnnotationContext(globalns={})


#: :func:`class_globals` answers per class: ``klass -> (klass.__module__, globals)``.
#: The globals dict is the live module namespace, so later module-level names
#: stay visible; an entry is ignored when ``__module__`` was rewritten since.
_CLASS_GLOBALS: weakref.WeakKeyDictionary[type, tuple[Any, dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def class_globals(klass: type) -> dict[str, Any]:
    """Globals of the module whose code created *klass*.

    Prefers the ``__globals__`` of a function compiled as part of the class
    body (the 3.14 ``__annotate__`` function, a method): that is the
    namespace the class body really executed in, even when ``__module__`` was
    rewritten to a re-export module.  Decorated methods are unwrapped first
    (``functools.wraps`` copies the qualified name, so the wrapper of a
    ``@contextmanager`` method would otherwise point at ``contextlib``), and
    only functions whose *code* was compiled in the class body count;
    wrappers from other modules never do.  Methods generated by a class
    decorator (dataclasses, attrs) come next: they name the module the class
    was decorated in (attrs hands them a snapshot copy of its globals, so
    the live module dict is used).  Falls back to
    ``sys.modules[klass.__module__].__dict__`` (empty dict if unknown).
    """
    module_name = getattr(klass, "__module__", None)
    cached = _CLASS_GLOBALS.get(klass)
    if cached is not None and cached[0] == module_name:
        return cached[1]
    found = _body_globals(klass)
    if found is None:
        module = sys.modules.get(module_name or "") if isinstance(module_name, str) else None
        if module is None:
            return {}  # not cached: the module may be registered later
        found = module.__dict__
    try:
        _CLASS_GLOBALS[klass] = (module_name, found)
    except TypeError:  # pragma: no cover - a class that refuses weak references
        pass
    return found


def _body_globals(klass: type) -> dict[str, Any] | None:
    """Globals the class body of *klass* executed in, if a function tells."""
    prefix = f"{klass.__qualname__}."
    functions: list[Any] = []
    annotate = klass.__dict__.get("__annotate_func__") or klass.__dict__.get("__annotate__")
    if annotate is not None:
        functions.append(annotate)
    for value in vars(klass).values():
        func = value
        if isinstance(value, (staticmethod, classmethod)):
            func = value.__func__
        elif isinstance(value, property):
            func = value.fget
        if callable(func):
            functions.append(unwrap_function(func))
    generated: list[dict[str, Any]] = []
    for func in functions:
        if not isinstance(func, types.FunctionType):
            continue
        globalns = func.__globals__
        # Skip empty dicts: dataclasses generate methods with ``{}`` globals
        # when the class's module is not in ``sys.modules``.
        if not isinstance(globalns, dict) or not globalns:
            continue
        if _compiled_in_body(func, prefix):
            return globalns
        if func.__qualname__.startswith(prefix) and _decorator_generated(func.__code__):
            generated.append(globalns)
    for globalns in generated:
        name = globalns.get("__name__")
        module = sys.modules.get(name) if isinstance(name, str) else None
        return module.__dict__ if module is not None else globalns
    return None


def _decorator_generated(code: types.CodeType) -> bool:
    """Code a class decorator compiled from source text (dataclasses, attrs)."""
    return code.co_filename == "<string>" or code.co_filename.startswith("<attrs generated")


def _compiled_in_body(func: types.FunctionType, prefix: str) -> bool:
    """True when *func*'s code was compiled inside the class body named by *prefix*."""
    code = func.__code__
    co_qualname = getattr(code, "co_qualname", None)  # 3.11+: set by the compiler
    if isinstance(co_qualname, str):
        return co_qualname.startswith(prefix)
    # Python 3.10: the qualified name is writable (functools.wraps copies it),
    # so also require that the code object carries the same name, that the
    # function lives in the module it says it does, and that no class
    # decorator generated it (attrs: snapshot globals).
    if not func.__qualname__.startswith(prefix) or code.co_name != func.__name__:
        return False
    if code.co_filename.startswith("<attrs generated"):
        return False
    return func.__globals__.get("__name__") == func.__module__


def class_context(klass: type, *, closure: Mapping[str, Any] | None = None) -> AnnotationContext:
    """The context of annotations written in *klass*'s body."""
    return AnnotationContext(
        globalns=class_globals(klass),
        owner=klass,
        closure=dict(closure or {}),
        type_params=tuple(getattr(klass, "__type_params__", ()) or ()),
    )


def function_context(
    func: Any,
    owner: type | None = None,
    *,
    closure: Mapping[str, Any] | None = None,
) -> AnnotationContext:
    """The context of annotations in *func*'s signature (*owner*: its class).

    Names resolve in *func*'s closure, its ``__globals__`` and builtins; the
    class namespace of *owner* is consulted only for names none of those
    define (see :class:`AnnotationContext`).
    """
    target = unwrap_function(func)
    globalns = getattr(target, "__globals__", None)
    if not isinstance(globalns, dict):
        globalns = class_globals(owner) if owner is not None else {}
    type_params = tuple(getattr(owner, "__type_params__", ()) or ()) if owner is not None else ()
    type_params += tuple(getattr(target, "__type_params__", ()) or ())
    return AnnotationContext(
        globalns=globalns,
        owner=owner,
        closure=dict(closure or {}),
        type_params=type_params,
        function=target,
    )


def _as_context(owner: Any) -> AnnotationContext:
    """Accept a class, a module, a context or ``None``."""
    if isinstance(owner, AnnotationContext):
        return owner
    if isinstance(owner, type):
        return class_context(owner)
    if isinstance(owner, types.ModuleType):
        return AnnotationContext(globalns=owner.__dict__)
    return _EMPTY_CONTEXT


# ===========================================================================
# TYPE_CHECKING scan (AST only -- nothing is imported)
# ===========================================================================


@dataclass(frozen=True)
class TypeCheckingName:
    """A name a module binds only under ``if TYPE_CHECKING:``.

    Attributes:
        name: The name bound in the module.
        module: Absolute module it is imported from (relative imports are
            resolved against the module's package), or ``None`` for a
            non-import binding (assignment, ``class``/``def``, ``type`` alias)
            or an unresolvable relative import.
        attr: The imported attribute for ``from module import attr``;
            ``None`` for ``import module`` and non-import bindings.
        lineno: Line of the binding statement.
    """

    name: str
    module: str | None
    attr: str | None
    lineno: int


def type_checking_imports(scope: Any) -> dict[str, TypeCheckingName]:
    """Names bound only under ``if TYPE_CHECKING:`` in a module.

    Pure AST scan of the module source; nothing is imported or executed.
    Recognises ``if TYPE_CHECKING:``, ``if typing.TYPE_CHECKING:`` (any alias
    of ``typing`` / ``typing_extensions``) and ``if not TYPE_CHECKING: ...
    else:``, at module level (also inside top-level ``try`` / ``with``).

    A plain dotted import keeps the module it names: ``import a.b.c`` binds
    ``a`` and is recorded as ``TypeCheckingName("a", "a.b.c", None, ...)``,
    so a stub can repeat the statement exactly (``a.b.c.X`` needs ``a.b.c``
    imported, not just ``a``).  When several plain imports bind the same
    name (``import a.b`` and ``import a.c``), the first is recorded under
    the name and each further one under its dotted module (``"a.c"``); no
    identifier can collide with such a key.

    Args:
        scope: A module, a module globals dict, an :class:`AnnotationContext`
            or a class (its defining module is scanned).

    Returns:
        ``{name: TypeCheckingName}``; empty when the source is unavailable.
    """
    globalns = _scope_globals(scope)
    if not globalns:
        return {}
    source, path = _module_source(globalns)
    if source is None:
        return {}
    package = _module_package(globalns)
    if path is not None:
        try:
            stat = os.stat(path)
        except OSError:
            return _scan_type_checking(source, package)
        return dict(_scan_type_checking_cached(path, stat.st_mtime_ns, stat.st_size, package))
    return _scan_type_checking(source, package)


def type_checking_names(scope: Any) -> frozenset[str]:
    """Just the names of :func:`type_checking_imports`."""
    return frozenset(type_checking_imports(scope))


def _scope_globals(scope: Any) -> Mapping[str, Any]:
    if isinstance(scope, AnnotationContext):
        return scope.globalns
    if isinstance(scope, types.ModuleType):
        return scope.__dict__
    if isinstance(scope, type):
        return class_globals(scope)
    if isinstance(scope, Mapping):
        return scope
    return {}


def _module_source(globalns: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(source, path)`` of the module owning *globalns*, if readable."""
    path = globalns.get("__file__")
    if isinstance(path, str):
        if path.endswith((".pyc", ".pyo")):
            try:
                path = importlib.util.source_from_cache(path)
            except (ValueError, NotImplementedError):
                path = None
        if path and os.path.isfile(path):
            try:
                with tokenize.open(path) as handle:
                    return handle.read(), path
            except (OSError, SyntaxError, UnicodeDecodeError):
                pass
    loader = globalns.get("__loader__")
    name = globalns.get("__name__")
    get_source = getattr(loader, "get_source", None)
    if callable(get_source) and isinstance(name, str):
        try:
            source = get_source(name)
        except (ImportError, OSError):
            source = None
        if isinstance(source, str):
            return source, None
    return None, None


def _module_package(globalns: Mapping[str, Any]) -> str | None:
    package = globalns.get("__package__")
    if isinstance(package, str):
        return package
    spec = globalns.get("__spec__")
    parent = getattr(spec, "parent", None)
    if isinstance(parent, str):
        return parent
    name = globalns.get("__name__")
    if isinstance(name, str):
        return name.rpartition(".")[0]
    return None


@functools.lru_cache(maxsize=256)
def _scan_type_checking_cached(
    path: str, mtime_ns: int, size: int, package: str | None
) -> tuple[tuple[str, TypeCheckingName], ...]:
    try:
        with tokenize.open(path) as handle:
            source = handle.read()
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ()
    return tuple(_scan_type_checking(source, package).items())


def _scan_type_checking(source: str, package: str | None) -> dict[str, TypeCheckingName]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    flag_names = {"TYPE_CHECKING"}
    typing_modules = {"typing", "typing_extensions"}
    for node in _module_level_statements(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module in ("typing", "typing_extensions"):
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    flag_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("typing", "typing_extensions"):
                    typing_modules.add(alias.asname or alias.name)

    def is_flag(test: ast.expr) -> bool:
        if isinstance(test, ast.Name):
            return test.id in flag_names
        return (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and test.value.id in typing_modules
        )

    found: dict[str, TypeCheckingName] = {}

    def record(stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                # ``import a.b.c`` binds ``a``; ``import a.b.c as x`` binds
                # ``x``.  Either way the full dotted module is recorded.
                local = alias.asname or alias.name.split(".", 1)[0]
                entry = TypeCheckingName(local, alias.name, None, stmt.lineno)
                first = found.setdefault(local, entry)
                if (
                    first is not entry
                    and alias.asname is None
                    and first.attr is None
                    and first.module != alias.name
                ):
                    # A further plain import binding the same name.
                    found.setdefault(alias.name, entry)
        elif isinstance(stmt, ast.ImportFrom):
            module = _absolute_module(stmt.module, stmt.level, package)
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                found.setdefault(local, TypeCheckingName(local, module, alias.name, stmt.lineno))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        found.setdefault(node.id, TypeCheckingName(node.id, None, None, stmt.lineno))
        elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(stmt.name, TypeCheckingName(stmt.name, None, None, stmt.lineno))
        elif _PY312 and isinstance(stmt, ast.TypeAlias):  # pragma: no cover - 3.12+
            name = stmt.name.id if isinstance(stmt.name, ast.Name) else None
            if name:
                found.setdefault(name, TypeCheckingName(name, None, None, stmt.lineno))

    def visit(stmts: list[ast.stmt], guarded: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                test = stmt.test
                if is_flag(test):
                    visit(stmt.body, True)
                    visit(stmt.orelse, guarded)
                elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and is_flag(test.operand):
                    visit(stmt.body, guarded)
                    visit(stmt.orelse, True)
                else:
                    visit(stmt.body, guarded)
                    visit(stmt.orelse, guarded)
            elif isinstance(stmt, _TRY_NODES):
                block = typing.cast(ast.Try, stmt)  # ``ast.TryStar`` has the same fields
                visit(block.body, guarded)
                for handler in block.handlers:
                    visit(handler.body, guarded)
                visit(block.orelse, guarded)
                visit(block.finalbody, guarded)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                visit(stmt.body, guarded)
            elif guarded:
                record(stmt)

    visit(tree.body, False)
    return found


_TRY_NODES: tuple[type, ...] = (ast.Try,) + ((ast.TryStar,) if _PY311 else ())


def _module_level_statements(stmts: list[ast.stmt]) -> Iterator[ast.stmt]:
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, ast.If):
            yield from _module_level_statements(stmt.body)
            yield from _module_level_statements(stmt.orelse)
        elif isinstance(stmt, _TRY_NODES):
            yield from _module_level_statements(stmt.body)
            for handler in stmt.handlers:
                yield from _module_level_statements(handler.body)
            yield from _module_level_statements(stmt.orelse)
            yield from _module_level_statements(stmt.finalbody)


def _absolute_module(module: str | None, level: int, package: str | None) -> str | None:
    if not level:
        return module
    if not package:
        return None
    try:
        return importlib.util.resolve_name("." * level + (module or ""), package)
    except (ImportError, ValueError):
        return None


# ===========================================================================
# Resolution
# ===========================================================================

ResolutionStatus = Literal["resolved", "unannotated", "degraded", "error"]


@dataclass(frozen=True)
class ResolvedAnnotation:
    """Outcome of resolving one annotation in its declaring scope.

    The engine never warns or raises by itself: callers that keep the field
    call :meth:`raise_if_error` and :meth:`warn` (a wired receptor, whose
    config type comes from wiring, may skip both and only use :attr:`shape`).

    Attributes:
        name: Field / parameter name.
        raw: The annotation as declared (a string under PEP 563 or a 3.14
            ``Format.STRING`` fallback, a ``ForwardRef``, or a typing object);
            ``inspect.Parameter.empty`` when unannotated.
        value: The evaluated annotation (``Annotated`` kept, no
            ``Optional`` added) when ``status == "resolved"``; ``Any`` or
            ``Annotated[Any, <recovered metadata>]`` when ``"degraded"``;
            ``raw`` when ``"error"``; ``inspect.Parameter.empty`` when
            ``"unannotated"``.
        status: ``"resolved" | "unannotated" | "degraded" | "error"``.
        context: The scope the annotation was evaluated in.
        reason: Why it degraded (``"type_checking_only"``,
            ``"function_locals"``) or failed (``"undefined_name"``,
            ``"metadata"``, ``"evaluation_failed"``); ``None`` otherwise.
        missing: Names that could not be resolved.
        detail: Human-readable explanation including the fix.
    """

    name: str
    raw: Any
    value: Any
    status: ResolutionStatus
    context: AnnotationContext = field(default=_EMPTY_CONTEXT, repr=False, compare=False)
    reason: str | None = None
    missing: tuple[str, ...] = ()
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True for ``resolved`` and ``unannotated``."""
        return self.status in ("resolved", "unannotated")

    @property
    def degraded(self) -> bool:
        """True when the type degraded to ``Any``."""
        return self.status == "degraded"

    @property
    def owner(self) -> type | None:
        """The declaring class (``None`` for a bare function)."""
        return self.context.owner

    @property
    def annotation_text(self) -> str:
        """The annotation as source-like text, for messages."""
        return annotation_text(self.raw)

    @property
    def shape(self) -> Shape:
        """Receptor shape: from the resolved value, else from the raw text."""
        if self.status == "resolved":
            return annotation_shape(self.value)
        if self.status == "unannotated":
            return "scalar"
        return annotation_shape(self.raw, self.context)

    def error(self, leaf: type | str | None = None) -> ConfigAnnotationError | None:
        """The :class:`ConfigAnnotationError` to raise, or ``None``.

        Args:
            leaf: The class being processed (named in the message); defaults
                to the declaring class.
        """
        if self.status != "error":
            return None
        return ConfigAnnotationError(
            _leaf_name(leaf, self), self.name, self.annotation_text, self.detail
        )

    def raise_if_error(self, leaf: type | str | None = None) -> None:
        """Raise :meth:`error` if the annotation could not be resolved."""
        exc = self.error(leaf)
        if exc is not None:
            raise exc

    def warning(self, leaf: type | str | None = None) -> tuple[type[ConscribeWarning], str] | None:
        """``(category, message)`` of the warning due, or ``None``."""
        if self.status != "degraded":
            return None
        category: type[ConscribeWarning] = (
            ExtractionWarning if self.reason == "function_locals" else DegradationWarning
        )
        message = (
            f"{_leaf_name(leaf, self)}.{self.name}: annotation "
            f"{self.annotation_text!r} {self.detail}"
        )
        return category, message

    def warn(self, leaf: type | str | None = None, *, stacklevel: int = 2) -> None:
        """Emit :meth:`warning` (no-op unless degraded).

        The warning points at the class that declares the annotation (its
        ``class`` statement, see :func:`warn_at`; for a bare function, the
        function); *stacklevel* is only used when that location is unknown.
        """
        due = self.warning(leaf)
        if due is not None:
            category, message = due
            where = self.context.owner if self.context.owner is not None else self.context.function
            warn_at(message, category, where, stacklevel=stacklevel)


def warn_at(message: str, category: type[Warning], where: Any = None, *, stacklevel: int = 1) -> None:
    """Emit a warning attributed to user code.

    The warning's file and line are those of *where* -- a function (its
    first line) or a class (its ``class`` statement) -- so ``-W`` filters,
    ``pytest.warns`` and IDEs point at the code to fix, not at conscribe.
    Falls back to :func:`warnings.warn` with *stacklevel* when the location
    cannot be found.

    Args:
        message: The warning text.
        category: A ``Warning`` subclass (conscribe uses its
            :class:`~conscribe.exceptions.ConscribeWarning` subclasses).
        where: A class, a function, or ``None``.
        stacklevel: As for :func:`warnings.warn`: 1 is the code that calls
            :func:`warn_at`, 2 its caller, and so on.
    """
    location = _source_location(where)
    if location is None:
        warnings.warn(message, category=category, stacklevel=stacklevel + 1)
        return
    filename, lineno, module_globals = location
    registry: dict[Any, Any] | None = None
    module: str | None = None
    if isinstance(module_globals, dict):
        registry = module_globals.setdefault("__warningregistry__", {})
        name = module_globals.get("__name__")
        module = name if isinstance(name, str) else None
    warnings.warn_explicit(
        message,
        category,
        filename,
        lineno,
        module=module,
        registry=registry,
        module_globals=module_globals if isinstance(module_globals, dict) else None,
    )


def _source_location(where: Any) -> tuple[str, int, Any] | None:
    """``(filename, lineno, module globals)`` of a class or function, if known."""
    if where is None:
        return None
    if isinstance(where, type):
        try:
            filename = inspect.getsourcefile(where) or inspect.getfile(where)
        except (TypeError, OSError):
            return None
        lineno = getattr(where, "__firstlineno__", None)  # 3.13+
        if not isinstance(lineno, int):
            try:
                lineno = inspect.getsourcelines(where)[1]
            except (OSError, TypeError, IndexError, SyntaxError):
                lineno = 1
        return filename, lineno, class_globals(where)
    code = getattr(unwrap_function(where), "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    return code.co_filename, code.co_firstlineno, getattr(unwrap_function(where), "__globals__", None)


def _leaf_name(leaf: type | str | None, res: ResolvedAnnotation) -> str:
    if isinstance(leaf, str):
        return leaf
    if isinstance(leaf, type):
        return leaf.__qualname__
    if res.context.owner is not None:
        return res.context.owner.__qualname__
    return res.context.scope_qualname or "<annotation>"


def annotation_text(raw: Any) -> str:
    """Render an annotation for messages."""
    if raw is EMPTY:
        return "<no annotation>"
    if isinstance(raw, str):
        return raw
    if isinstance(raw, ForwardRef):
        return raw.__forward_arg__
    try:
        return inspect.formatannotation(raw)
    except Exception:  # noqa: BLE001 - message rendering must never fail
        return repr(raw)


def resolve_annotation(
    raw: Any,
    context: AnnotationContext | type | types.ModuleType | None,
    *,
    name: str,
) -> ResolvedAnnotation:
    """Resolve one annotation in its declaring scope.

    Never imports, never warns and never raises for annotation problems; see
    the module docstring for the outcomes.

    Args:
        raw: The annotation as declared (string, ``ForwardRef``, typing
            object or ``inspect.Parameter.empty``).
        context: Where it was written: an :class:`AnnotationContext` (see
            :func:`class_context` / :func:`function_context`), or a class /
            module as a shortcut.
        name: Field / parameter name (for messages).
    """
    ctx = _as_context(context)
    if raw is EMPTY:
        return ResolvedAnnotation(name, raw, EMPTY, "unannotated", ctx)
    if raw is None:
        return ResolvedAnnotation(name, raw, NoneType, "resolved", ctx)
    if not isinstance(raw, (str, ForwardRef)) and not contains_forward_refs(raw):
        return ResolvedAnnotation(name, raw, raw, "resolved", ctx)

    globalns, localns = ctx.eval_namespaces()
    failure: BaseException | None = None
    try:
        value = _evaluate(raw, globalns, localns)
    except Exception as exc:  # noqa: BLE001 - diagnosed below, never swallowed
        failure = exc
    else:
        if not contains_forward_refs(value):
            return ResolvedAnnotation(name, raw, value, "resolved", ctx)
    return _diagnose(raw, ctx, name, failure, ctx.namespace())


def resolve_class_annotation(
    klass: type, name: str, *, raw: Any = EMPTY
) -> ResolvedAnnotation | None:
    """Resolve the annotation of *name* declared in *klass*'s own body.

    Args:
        klass: The declaring class (see :func:`annotation_owner`).
        name: The annotated name.
        raw: Optional override of the raw annotation.

    Returns:
        The outcome, or ``None`` when *klass* does not annotate *name*.
    """
    annotations, closure = class_annotation_source(klass)
    if raw is EMPTY:
        if name not in annotations:
            return None
        raw = annotations[name]
    return resolve_annotation(raw, class_context(klass, closure=closure), name=name)


def annotation_owner(cls: type, name: str) -> type | None:
    """The most-derived class in *cls*'s MRO whose own body annotates *name*."""
    for klass in cls.__mro__:
        if klass is object:
            continue
        if name in own_annotations(klass):
            return klass
    return None


# -- evaluation --------------------------------------------------------------


def _parse_expr(text: str) -> ast.expr | None:
    """The parsed body of an annotation text (``None`` on a syntax error).

    Not cached on purpose: a parse is cheap, and cached AST nodes would stay
    alive as long-lived garbage-collector work.
    """
    try:
        return ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None


@functools.lru_cache(maxsize=1024)
def _compile(text: str) -> types.CodeType:
    return compile(text.strip(), "<annotation>", "eval")


def _evaluate(raw: Any, globalns: dict[str, Any], localns: dict[str, Any], depth: int = 0) -> Any:
    """Evaluate *raw* (see :meth:`AnnotationContext.eval_namespaces`); raises
    whatever evaluation raises."""
    if depth > 8:
        raise ValueError("annotation nests string literals too deeply")
    if isinstance(raw, ForwardRef):
        cell = getattr(raw, "__cell__", None)
        if isinstance(cell, types.CellType):  # pragma: no cover - 3.14 closures
            try:
                return cell.cell_contents
            except ValueError:
                pass
        raw = raw.__forward_arg__
    if isinstance(raw, str):
        value = eval(_compile(raw), globalns, localns)  # noqa: S307 - same contract as typing.get_type_hints
        if isinstance(value, (str, ForwardRef)):
            return _evaluate(value, globalns, localns, depth + 1)
        if value is None:
            return NoneType
        if contains_forward_refs(value):
            value = _eval_nested(value, globalns, localns)
        return value
    return _eval_nested(raw, globalns, localns)


#: ``typing._eval_type`` accepts ``type_params`` (3.12.4+ and 3.13+).  On
#: 3.12.0-3.12.3 its fourth positional parameter is ``recursive_guard``, so
#: ``type_params`` is only ever passed by keyword, and only when it exists.
_EVAL_TYPE_TAKES_TYPE_PARAMS: bool = (
    "type_params" in inspect.signature(typing._eval_type).parameters  # type: ignore[attr-defined]
)


def _eval_nested(value: Any, globalns: dict[str, Any], localns: dict[str, Any]) -> Any:
    """Evaluate forward references nested in a typing object."""
    if not _PY311:
        value = _str_args_to_forward_refs(value)
    eval_type = typing._eval_type  # type: ignore[attr-defined]
    if _EVAL_TYPE_TAKES_TYPE_PARAMS:
        return eval_type(value, globalns, localns, type_params=())
    return eval_type(value, globalns, localns)


def _str_args_to_forward_refs(tp: Any) -> Any:  # pragma: no cover - 3.10 only
    """3.10: ``typing._eval_type`` ignores ``str`` args of builtin generics."""
    if isinstance(tp, types.UnionType):
        # ``list["Leaf"] | None``: rebuild the PEP 604 union from its members.
        members = tuple(_str_args_to_forward_refs(arg) for arg in tp.__args__)
        if all(a is b for a, b in zip(members, tp.__args__)):
            return tp
        return functools.reduce(operator.or_, members)
    if isinstance(tp, types.GenericAlias):
        args = tuple(
            ForwardRef(arg) if isinstance(arg, str) else _str_args_to_forward_refs(arg)
            for arg in tp.__args__
        )
        if all(a is b for a, b in zip(args, tp.__args__)):
            return tp
        origin = tp.__origin__
        if origin is _abc.Callable:
            if args and args[0] is Ellipsis:
                return origin[..., args[-1]]
            return origin[list(args[:-1]), args[-1]]
        return types.GenericAlias(origin, args)
    origin = get_origin(tp)
    if origin is Literal or origin is None:
        return tp
    if origin is Annotated:
        base = tp.__origin__
        new_base = _str_args_to_forward_refs(base)
        return tp if new_base is base else tp.copy_with((new_base,))
    args = getattr(tp, "__args__", None)
    if not isinstance(args, tuple) or not hasattr(tp, "copy_with"):
        return tp
    new_args = tuple(_str_args_to_forward_refs(a) for a in args)
    if all(a is b for a, b in zip(new_args, args)):
        return tp
    return tp.copy_with(new_args)


def contains_forward_refs(tp: Any, _depth: int = 0) -> bool:
    """True when *tp* still holds a string / ``ForwardRef`` in a type position.

    ``Literal`` values and ``Annotated`` metadata are values, not types, so
    strings there do not count -- except a ``ForwardRef`` object in the
    metadata (3.14 ``FORWARDREF`` output for metadata that failed).
    """
    if _depth > 50:
        return False
    if isinstance(tp, (str, ForwardRef)):
        return True
    origin = get_origin(tp)
    if origin is Literal:
        return False
    if origin is Annotated:
        args = get_args(tp)
        return contains_forward_refs(args[0], _depth + 1) or any(
            isinstance(meta, ForwardRef) for meta in args[1:]
        )
    for arg in get_args(tp):
        if isinstance(arg, (list, tuple)):
            if any(contains_forward_refs(a, _depth + 1) for a in arg):
                return True
        elif contains_forward_refs(arg, _depth + 1):
            return True
    return False


# -- diagnosis ---------------------------------------------------------------


class _Placeholder:
    """Stands in for an undefined name while hunting for further missing names."""

    __slots__ = ("_conscribe_name",)

    def __init__(self, name: str) -> None:
        self._conscribe_name = name

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _Placeholder(f"{self._conscribe_name}.{item}")

    def __getitem__(self, item: Any) -> _Placeholder:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> _Placeholder:
        return self

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("placeholder is not iterable")

    def __or__(self, other: Any) -> _Placeholder:
        return self

    def __ror__(self, other: Any) -> _Placeholder:
        return self

    def __repr__(self) -> str:
        return self._conscribe_name


def _forward_texts(raw: Any) -> list[tuple[str, bool]]:
    """Source texts of the unresolved parts of *raw*: ``[(text, in_metadata)]``."""
    if isinstance(raw, str):
        return [(raw, False)]
    if isinstance(raw, ForwardRef):
        return [(raw.__forward_arg__, False)]
    found: list[tuple[str, bool]] = []

    def walk(tp: Any, depth: int) -> None:
        if depth > 50:
            return
        if isinstance(tp, ForwardRef):
            found.append((tp.__forward_arg__, False))
            return
        if isinstance(tp, str):
            found.append((tp, False))
            return
        origin = get_origin(tp)
        if origin is Literal:
            return
        if origin is Annotated:
            args = get_args(tp)
            walk(args[0], depth + 1)
            for meta in args[1:]:
                if isinstance(meta, ForwardRef):
                    found.append((meta.__forward_arg__, True))
            return
        for arg in get_args(tp):
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    walk(item, depth + 1)
            else:
                walk(arg, depth + 1)

    walk(raw, 0)
    return found


def _missing_names(
    text: str, ns: dict[str, Any], depth: int = 0
) -> tuple[list[str], BaseException | None]:
    """Names *text* needs that *ns* (plus builtins) does not define.

    Evaluates repeatedly, standing a placeholder in for each missing name, so
    every missing name is found -- not just the first -- including names
    behind forward references nested in the evaluated value
    (``"Optional['Heavy']"``).  Returns the names and the first
    non-``NameError`` failure seen while no name was missing.
    """
    if depth > 8:
        return [], None
    try:
        code = _compile(text)
    except SyntaxError as exc:
        return [], exc
    trial = dict(ns)
    missing: list[str] = []
    for _ in range(64):
        try:
            value = eval(code, trial)  # noqa: S307 - diagnosis of a failed evaluation
        except NameError as exc:
            name = getattr(exc, "name", None)
            if not isinstance(name, str) or name in trial:
                return missing, (exc if not missing else None)
            missing.append(name)
            trial[name] = _Placeholder(name)
            continue
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return missing, (exc if not missing else None)
        if isinstance(value, ForwardRef):
            nested = [value.__forward_arg__]
        elif isinstance(value, str):  # nested quotes: '"X"'
            nested = [value]
        elif contains_forward_refs(value):
            nested = [inner for inner, _ in _forward_texts(value)]
        else:
            nested = []
        failure: BaseException | None = None
        for inner in nested:
            more, error = _missing_names(inner, trial, depth + 1)
            missing.extend(m for m in more if m not in missing)
            if error is not None and failure is None:
                failure = error
        if not nested and not missing:
            # Everything evaluates on its own: the original failure came from
            # elsewhere (e.g. a typing construct rejecting an argument).
            return missing, None
        return missing, (failure if not missing else None)
    return missing, None


def _name_positions(text: str) -> dict[str, set[str]]:
    """Where each name occurs in *text*: ``"type"`` or ``"value"`` positions.

    Value positions are ``Annotated`` metadata, ``Literal`` arguments and call
    arguments: they are evaluated at runtime and can never degrade.
    """
    positions: dict[str, set[str]] = {}

    def record(name: str, where: str) -> None:
        positions.setdefault(name, set()).add(where)

    def visit(node: ast.AST, where: str, depth: int) -> None:
        if depth > 50:
            return
        if isinstance(node, ast.Name):
            record(node.id, where)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and where == "type":
                try:
                    inner = ast.parse(node.value.strip(), mode="eval").body
                except SyntaxError:
                    return
                visit(inner, "type", depth + 1)
        elif isinstance(node, ast.Subscript):
            visit(node.value, where, depth + 1)
            head = _dotted_tail(node.value)
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if head == "Annotated" and elements:
                visit(elements[0], where, depth + 1)
                for element in elements[1:]:
                    visit(element, "value", depth + 1)
            elif head == "Literal":
                for element in elements:
                    visit(element, "value", depth + 1)
            else:
                for element in elements:
                    visit(element, where, depth + 1)
        elif isinstance(node, ast.Call):
            visit(node.func, "value", depth + 1)
            for arg in node.args:
                visit(arg, "value", depth + 1)
            for keyword in node.keywords:
                visit(keyword.value, "value", depth + 1)
        elif isinstance(node, ast.Lambda):
            return  # evaluated lazily, never during annotation evaluation
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, where, depth + 1)

    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return positions
    visit(tree, "type", 0)
    return positions


def _dotted_tail(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _diagnose(
    raw: Any,
    ctx: AnnotationContext,
    name: str,
    failure: BaseException | None,
    ns: dict[str, Any],
) -> ResolvedAnnotation:
    """Classify a failed resolution (see the module docstring)."""
    missing: list[str] = []
    in_value: set[str] = set()
    other: BaseException | None = None
    for text, in_metadata in _forward_texts(raw):
        names, error = _missing_names(text, ns)
        if names:
            positions = _name_positions(text)
            for missing_name in names:
                root = missing_name.split(".", 1)[0]
                if missing_name not in missing:
                    missing.append(missing_name)
                if in_metadata or "value" in positions.get(root, ()):
                    in_value.add(missing_name)
        elif error is not None and other is None:
            other = error
    if not missing:
        exc = other or failure
        if exc is None:
            detail = (
                "it still holds an unresolved forward reference after evaluation. "
                f"Fix the annotation in {ctx.module or 'its module'}."
            )
        else:
            detail = (
                f"evaluating it raised {type(exc).__name__}: {exc}. "
                f"Fix the annotation in {ctx.module or 'its module'}."
            )
        return ResolvedAnnotation(
            name, raw, raw, "error", ctx, reason="evaluation_failed", detail=detail
        )

    tc_names = type_checking_names(ctx)
    local_names = _enclosing_locals(ctx) if ctx.function_local else None
    undefined: list[str] = []
    tc_value: list[str] = []
    locals_needed: list[str] = []
    tc_type: list[str] = []
    for missing_name in missing:
        if ctx.function_local and (local_names is None or missing_name in local_names):
            locals_needed.append(missing_name)
        elif missing_name in tc_names:
            (tc_value if missing_name in in_value else tc_type).append(missing_name)
        else:
            undefined.append(missing_name)

    module = ctx.module or "its module"
    declared = _declared_on(ctx)
    if undefined:
        listed = _names(undefined)
        detail = (
            f"{listed} {'is' if len(undefined) == 1 else 'are'} not defined{declared}: not a "
            f"global of module {module}, not a class attribute, not a builtin and not "
            f"imported under 'if TYPE_CHECKING:'. Fix the spelling, or import "
            f"{'it' if len(undefined) == 1 else 'them'} at module level in {module}."
        )
        return ResolvedAnnotation(
            name, raw, raw, "error", ctx, reason="undefined_name",
            missing=tuple(undefined), detail=detail,
        )
    if tc_value:
        listed = _names(tc_value)
        detail = (
            f"its Annotated metadata needs {listed}, which module {module} only imports "
            f"under 'if TYPE_CHECKING:'{declared}. Metadata such as Field(...) is evaluated "
            f"at runtime and conscribe does not import TYPE_CHECKING blocks: import "
            f"{'it' if len(tc_value) == 1 else 'them'} unconditionally in {module}."
        )
        return ResolvedAnnotation(
            name, raw, raw, "error", ctx, reason="metadata",
            missing=tuple(tc_value), detail=detail,
        )

    try:
        value = _degraded_value(raw, ns, tolerated=frozenset(locals_needed + tc_type))
    except _MetadataFailure as broken:
        detail = (
            f"its Annotated metadata {broken.text!r} cannot be evaluated: evaluating it "
            f"raised {type(broken.error).__name__}: {broken.error}{declared}. Fix the "
            f"metadata in {module}."
        )
        return ResolvedAnnotation(
            name, raw, raw, "error", ctx, reason="metadata", detail=detail,
        )
    if locals_needed:
        listed = _names(locals_needed + tc_type)
        detail = (
            f"needs {listed}, local to the function that defines "
            f"{ctx.scope_qualname or 'the class'}; conscribe cannot see function locals, "
            f"so the field's type degrades to Any (default, description and constraints "
            f"are kept). Define the class and the names it uses at module level to get "
            f"full validation."
        )
        return ResolvedAnnotation(
            name, raw, value, "degraded", ctx, reason="function_locals",
            missing=tuple(locals_needed + tc_type), detail=detail,
        )
    listed = _names(tc_type)
    detail = (
        f"needs {listed}, which module {module} only imports under 'if TYPE_CHECKING:'"
        f"{declared}; conscribe does not import TYPE_CHECKING blocks, so the field's type "
        f"degrades to Any (default, description and constraints are kept). Import "
        f"{'it' if len(tc_type) == 1 else 'them'} at runtime to get full validation."
    )
    return ResolvedAnnotation(
        name, raw, value, "degraded", ctx, reason="type_checking_only",
        missing=tuple(tc_type), detail=detail,
    )


def _names(names: list[str]) -> str:
    quoted = [f"'{n}'" for n in names]
    if len(quoted) == 1:
        return f"name {quoted[0]}"
    return "names " + ", ".join(quoted)


def _declared_on(ctx: AnnotationContext) -> str:
    scope = ctx.scope_qualname
    if not scope:
        return ""
    if ctx.function is not None and scope.endswith(".__init__"):
        scope = scope[: -len(".__init__")]
    module = ctx.module
    return f" (declared on {module}.{scope})" if module else f" (declared on {scope})"


class _MetadataFailure(Exception):
    """``Annotated`` metadata of a degrading annotation that cannot be evaluated."""

    def __init__(self, text: str, error: BaseException) -> None:
        super().__init__(text)
        self.text = text
        self.error = error


def _degraded_value(raw: Any, ns: dict[str, Any], *, tolerated: frozenset[str]) -> Any:
    """``Any``, keeping the ``Annotated`` metadata (see :func:`_recover_metadata`)."""
    metadata = _recover_metadata(raw, ns, tolerated=tolerated)
    if metadata:
        return Annotated[(Any, *metadata)]  # type: ignore[valid-type]
    return Any


def _recover_metadata(raw: Any, ns: dict[str, Any], *, tolerated: frozenset[str]) -> list[Any]:
    """Metadata of the outermost ``Annotated`` (possibly under ``Optional``).

    A metadata element that needs one of the *tolerated* names (the missing
    names the degradation already reports: function locals, TYPE_CHECKING
    imports) is dropped.  Any other failure raises :class:`_MetadataFailure`:
    metadata that cannot be evaluated is an error, never silently lost.
    """
    if isinstance(raw, ForwardRef):
        raw = raw.__forward_arg__
    if not isinstance(raw, str):
        annotated = _outer_annotated(raw)
        if annotated is None:
            return []
        return [m for m in get_args(annotated)[1:] if not isinstance(m, ForwardRef)]
    try:
        node = ast.parse(raw.strip(), mode="eval").body
    except SyntaxError:
        return []
    node = _outer_annotated_node(node, ns)
    if node is None:
        return []
    elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    recovered: list[Any] = []
    for element in elements[1:]:
        try:
            code = compile(ast.Expression(body=element), "<annotation metadata>", "eval")
            recovered.append(eval(code, dict(ns)))  # noqa: S307
        except NameError as exc:
            missing = getattr(exc, "name", None)
            if isinstance(missing, str) and missing.split(".", 1)[0] in tolerated:
                continue  # needs a name the degradation already reports
            raise _MetadataFailure(ast.unparse(element), exc) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised as a diagnosed failure
            raise _MetadataFailure(ast.unparse(element), exc) from exc
    return recovered


def _outer_annotated(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Annotated:
        return tp
    if origin in UNION_ORIGINS:
        members = [a for a in get_args(tp) if a is not NoneType]
        if len(members) == 1 and get_origin(members[0]) is Annotated:
            return members[0]
    return None


def _outer_annotated_node(node: ast.AST, ns: dict[str, Any]) -> ast.Subscript | None:
    def is_head(head: ast.AST, target: Any, spelling: str) -> bool:
        found, obj = _lookup_node(head, ns)
        if found:
            return obj is target
        return _dotted_tail(head) == spelling

    if isinstance(node, ast.Subscript):
        if is_head(node.value, Annotated, "Annotated"):
            return node
        if is_head(node.value, Optional, "Optional"):
            inner = node.slice
            return _outer_annotated_node(inner, ns)
        if is_head(node.value, Union, "Union"):
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            members = [e for e in elements if not is_none_node(e)]
            if len(members) == 1:
                return _outer_annotated_node(members[0], ns)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        members = [n for n in (node.left, node.right) if not is_none_node(n)]
        if len(members) == 1:
            return _outer_annotated_node(members[0], ns)
    return None


def is_none_node(node: ast.AST) -> bool:
    """True for an AST node spelling ``None`` (or ``NoneType``)."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    return isinstance(node, ast.Name) and node.id in ("None", "NoneType")


def _lookup_node(node: ast.AST, ns: Mapping[str, Any] | AnnotationContext) -> tuple[bool, Any]:
    """Resolve a ``Name`` / dotted ``Attribute`` chain without evaluating calls."""
    if isinstance(node, ast.Name):
        if isinstance(ns, AnnotationContext):
            return ns.lookup(node.id)
        if node.id in ns:
            return True, ns[node.id]
        if hasattr(builtins, node.id):
            return True, getattr(builtins, node.id)
        return False, None
    if isinstance(node, ast.Attribute):
        found, base = _lookup_node(node.value, ns)
        if not found:
            return False, None
        try:
            return True, getattr(base, node.attr)
        except AttributeError:
            return False, None
    return False, None


def _enclosing_locals(ctx: AnnotationContext) -> frozenset[str] | None:
    """Local names of the functions enclosing the declaring scope.

    Walks the qualified name (``outer.<locals>.Cls``) through the module and
    the nested code objects.  ``None`` when the walk fails (then every
    missing name is assumed to be a function local).
    """
    qualname = ctx.scope_qualname
    if "<generic parameters of" in qualname:
        return None
    parts = qualname.split(".")
    if "<locals>" not in parts:
        return None
    last_locals = len(parts) - 1 - parts[::-1].index("<locals>")
    names: set[str] = set()
    current: Any = None
    found, current = (True, ctx.globalns[parts[0]]) if parts[0] in ctx.globalns else (False, None)
    if not found:
        return None
    index = 1
    while index <= last_locals:
        part = parts[index]
        if part == "<locals>":
            code = current if isinstance(current, types.CodeType) else getattr(
                unwrap_function(current), "__code__", None
            )
            if not isinstance(code, types.CodeType):
                return None
            names.update(code.co_varnames, code.co_cellvars, code.co_freevars)
            if index == last_locals:
                break
            nested = parts[index + 1]
            current = next(
                (c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == nested),
                None,
            )
            if current is None:
                return None
            index += 2
            continue
        if isinstance(current, types.CodeType):
            current = next(
                (c for c in current.co_consts if isinstance(c, types.CodeType) and c.co_name == part),
                None,
            )
        else:
            current = getattr(current, part, None)
            if isinstance(current, (staticmethod, classmethod)):
                current = current.__func__
        if current is None:
            return None
        index += 1
    return frozenset(names)


# ===========================================================================
# Normalization
# ===========================================================================


def unwrap_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split ``Annotated[T, *meta]`` into ``(T, meta)``; ``(annotation, ())`` otherwise."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return args[0], tuple(args[1:])
    return annotation, ()


def unwrap_initvar(annotation: Any) -> Any:
    """``InitVar[T]`` → ``T`` (bare ``InitVar`` → ``Any``); anything else unchanged."""
    if isinstance(annotation, dataclasses.InitVar):
        return annotation.type
    if annotation is dataclasses.InitVar:
        return Any
    return annotation


def unwrap_final(annotation: Any) -> Any:
    """``Final[T]`` → ``T`` (bare ``Final`` → ``Any``), also under ``Annotated``.

    A dataclass / attrs field annotated ``Final[int]`` is an ordinary
    constructor parameter, but pydantic reads ``Final`` with a default as a
    class variable and would drop the config field.  Anything else is
    returned unchanged.
    """
    if annotation is Final:
        return Any
    origin = get_origin(annotation)
    if origin is Final:
        args = get_args(annotation)
        return args[0] if args else Any
    if origin is Annotated:
        base, metadata = unwrap_annotated(annotation)
        inner = unwrap_final(base)
        if inner is not base:
            return Annotated[(inner, *metadata)]  # type: ignore[valid-type]
    return annotation


def admits_none(annotation: Any, owner: Any = None) -> bool:
    """True when ``None`` is a valid value of *annotation*.

    ``None`` / ``NoneType``, ``Any``, ``object``, unions with ``None``,
    ``Literal[None, ...]``, an unconstrained ``TypeVar`` (it means ``Any``)
    and ``Annotated``/``NewType``/bound or constrained ``TypeVar`` over
    those.  Unresolved strings are analysed by AST (names looked up in
    *owner*).  This is the one ``None`` test of the 2.0 None rule (see
    :func:`conscribe.config._fieldinfo.config_field`).
    """
    if annotation is None or annotation is NoneType or annotation is Any or annotation is object:
        return True
    if isinstance(annotation, (str, ForwardRef)):
        ctx = _as_context(owner)
        text = annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation
        node = _parse_expr(text)
        if node is None:
            return False
        return _ast_admits_none(node, ctx, 0)
    if isinstance(annotation, typing.TypeVar):
        if annotation.__constraints__:
            return any(admits_none(c) for c in annotation.__constraints__)
        bound = annotation.__bound__
        return True if bound is None else admits_none(bound)
    origin = get_origin(annotation)
    if origin is Annotated:
        return admits_none(get_args(annotation)[0])
    if origin in UNION_ORIGINS:
        return any(admits_none(a) for a in get_args(annotation))
    if origin is Literal:
        return any(a is None for a in get_args(annotation))
    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        return admits_none(supertype)
    return False


def _ast_admits_none(node: ast.AST, ctx: AnnotationContext, depth: int) -> bool:
    if depth > 20:
        return False
    if is_none_node(node):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            inner = ast.parse(node.value.strip(), mode="eval").body
        except SyntaxError:
            return False
        return _ast_admits_none(inner, ctx, depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _ast_admits_none(node.left, ctx, depth + 1) or _ast_admits_none(
            node.right, ctx, depth + 1
        )
    if isinstance(node, (ast.Name, ast.Attribute)):
        found, obj = _lookup_node(node, ctx)
        if found:
            return obj is Any or obj is object or obj is NoneType
        return _dotted_tail(node) in ("Any", "object", "NoneType")
    if isinstance(node, ast.Subscript):
        found, head = _lookup_node(node.value, ctx)
        spelled = _dotted_tail(node.value)
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if (found and head is Optional) or (not found and spelled == "Optional"):
            return True
        if (found and head is Union) or (not found and spelled == "Union"):
            return any(_ast_admits_none(e, ctx, depth + 1) for e in elements)
        if (found and head is Annotated) or (not found and spelled == "Annotated"):
            return bool(elements) and _ast_admits_none(elements[0], ctx, depth + 1)
        if (found and head is Literal) or (not found and spelled == "Literal"):
            return any(is_none_node(e) for e in elements)
    return False


# ===========================================================================
# Shape, ClassVar, KW_ONLY, InitVar
# ===========================================================================

_DICT_NAMES: frozenset[str] = frozenset(
    {
        "dict", "Dict", "Mapping", "MutableMapping", "OrderedDict", "DefaultDict",
        "defaultdict", "ChainMap", "Counter",
    }
)
_LIST_NAMES: frozenset[str] = frozenset(
    {
        "list", "set", "frozenset", "tuple", "List", "Set", "FrozenSet", "Tuple",
        "Sequence", "MutableSequence", "AbstractSet", "MutableSet", "Collection",
        "Iterable", "deque", "Deque",
    }
)
_SCALAR_SEQUENCES: tuple[type, ...] = (str, bytes, bytearray, memoryview)


def annotation_shape(annotation: Any, owner: Any = None) -> Shape:
    """Receptor shape of an annotation: ``"scalar"``, ``"list"`` or ``"dict"``.

    * ``"dict"``: a mapping -- ``dict``, ``Mapping``/``MutableMapping``
      (and their ``typing`` aliases), ``OrderedDict``, ``defaultdict``,
      ``ChainMap``, ``Counter`` and other nominal ``Mapping`` subclasses.
    * ``"list"``: any other collection -- ``list``, ``tuple``, ``set``,
      ``frozenset``, ``deque``, nominal ``Sequence``/``Set`` subclasses, and
      exactly ``Iterable``/``Collection``.
    * ``"scalar"``: everything else, including ``str``/``bytes``,
      ``TypedDict`` and ``NamedTuple`` records, and classes that merely
      implement ``__iter__``/``__len__`` (no structural matching).

    ``Annotated``, ``Optional``, ``Union`` and ``X | None`` are looked
    through; a union is a ``"dict"`` if any member is, else a ``"list"`` if
    any member is.  Strings and ``ForwardRef`` objects are resolved in
    *owner* first; only names that cannot be resolved are judged by
    spelling (so a user class called ``Set`` is a scalar).

    Args:
        annotation: Resolved annotation, ``ForwardRef`` or string.
        owner: The declaring class (or module) used to look names up.
    """
    if annotation is EMPTY or annotation is None:
        return "scalar"
    if isinstance(annotation, (str, ForwardRef)):
        ctx = _as_context(owner)
        text = annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation
        node = _parse_expr(text)
        if node is None:
            return "scalar"
        return _ast_shape(node, ctx, 0)
    return _object_shape(annotation, 0)


def _merge_shapes(shapes: list[Shape]) -> Shape:
    if "dict" in shapes:
        return "dict"
    if "list" in shapes:
        return "list"
    return "scalar"


def _object_shape(tp: Any, depth: int) -> Shape:
    if depth > 50 or tp is None or tp is NoneType or tp is Any:
        return "scalar"
    if isinstance(tp, (str, ForwardRef)):
        return annotation_shape(tp)
    if isinstance(tp, dataclasses.InitVar):
        return _object_shape(tp.type, depth + 1)
    origin = get_origin(tp)
    if origin is Annotated or origin is ClassVar or origin is Final:
        args = get_args(tp)
        return _object_shape(args[0], depth + 1) if args else "scalar"
    if origin in UNION_ORIGINS:
        return _merge_shapes(
            [_object_shape(a, depth + 1) for a in get_args(tp) if a is not NoneType]
        )
    if origin is Literal:
        return "scalar"
    supertype = getattr(tp, "__supertype__", None)  # NewType
    if supertype is not None:
        return _object_shape(supertype, depth + 1)
    candidate = origin if origin is not None else tp
    return _class_shape(candidate)


def _class_shape(candidate: Any) -> Shape:
    if not isinstance(candidate, type):
        return "scalar"
    if _is_record_type(candidate):
        return "scalar"
    if candidate.__dict__.get("_is_protocol", False):
        # An interface: a collection only if it declares a collection ABC as
        # a base.  (``issubclass`` against the ABCs would also work, but it
        # fills their caches with an entry per protocol class.)
        mro = candidate.__mro__
        if _abc.Mapping in mro:
            return "dict"
        if _abc.Sequence in mro or _abc.Set in mro:
            return "list"
        return "scalar"
    try:
        if issubclass(candidate, _SCALAR_SEQUENCES):
            return "scalar"
        if issubclass(candidate, _abc.Mapping):
            return "dict"
        if candidate in (_abc.Iterable, _abc.Collection):
            return "list"
        if issubclass(candidate, (_abc.Sequence, _abc.Set)):
            return "list"
    except TypeError:  # exotic metaclass refusing issubclass
        return "scalar"
    return "scalar"


def _is_record_type(candidate: type) -> bool:
    """``TypedDict`` and ``NamedTuple`` classes are records, not collections."""
    if typing.is_typeddict(candidate):
        return True
    return issubclass(candidate, tuple) and hasattr(candidate, "_fields")


def _ast_shape(node: ast.AST, ctx: AnnotationContext, depth: int) -> Shape:
    if depth > 30:
        return "scalar"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _merge_shapes(
            [_ast_shape(node.left, ctx, depth + 1), _ast_shape(node.right, ctx, depth + 1)]
        )
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            try:
                inner = ast.parse(node.value.strip(), mode="eval").body
            except SyntaxError:
                return "scalar"
            return _ast_shape(inner, ctx, depth + 1)
        return "scalar"
    if isinstance(node, (ast.Name, ast.Attribute)):
        found, obj = _lookup_node(node, ctx)
        if found:
            return _object_shape(obj, depth + 1)
        return _spelled_shape(_dotted_tail(node))
    if isinstance(node, ast.Subscript):
        found, head = _lookup_node(node.value, ctx)
        spelled = _dotted_tail(node.value)
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]

        def head_is(target: Any, spelling: str) -> bool:
            return (found and head is target) or (not found and spelled == spelling)

        if head_is(Annotated, "Annotated") or head_is(ClassVar, "ClassVar") or head_is(
            Final, "Final"
        ):
            return _ast_shape(elements[0], ctx, depth + 1) if elements else "scalar"
        if head_is(Optional, "Optional") or head_is(Union, "Union"):
            return _merge_shapes(
                [_ast_shape(e, ctx, depth + 1) for e in elements if not is_none_node(e)]
            )
        if head_is(Literal, "Literal"):
            return "scalar"
        if found:
            return _object_shape(head, depth + 1)
        return _spelled_shape(spelled)
    return "scalar"


def _spelled_shape(name: str | None) -> Shape:
    if name in _DICT_NAMES:
        return "dict"
    if name in _LIST_NAMES:
        return "list"
    return "scalar"


def is_classvar(annotation: Any, owner: Any = None) -> bool:
    """True for ``ClassVar`` / ``ClassVar[...]`` (also under ``Annotated``).

    String annotations are parsed and their head resolved in *owner*'s
    namespace, so ``import typing as t`` / ``ClassVar as CV`` aliases are
    honoured and a user class named ``ClassVarRegistry`` is not a
    ``ClassVar``.  Unresolvable heads fall back to the spelling ``ClassVar``.
    """
    if annotation is ClassVar:
        return True
    if isinstance(annotation, (str, ForwardRef)):
        text = annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation
        return _ast_head_is(text, _as_context(owner), ClassVar, "ClassVar", unwrap_annotated=True)
    origin = get_origin(annotation)
    if origin is ClassVar:
        return True
    if origin is Annotated:
        return is_classvar(get_args(annotation)[0], owner)
    return False


def is_pseudo_field(annotation: Any, owner: Any = None) -> bool:
    """True when the annotation does not declare an instance attribute.

    ``ClassVar`` (also under ``Annotated``), the ``dataclasses.KW_ONLY``
    sentinel and ``InitVar`` -- :func:`is_classvar`,
    :func:`is_kw_only_sentinel` and :func:`is_initvar` in one pass (a string
    annotation is parsed once).
    """
    if not isinstance(annotation, (str, ForwardRef)):
        return (
            is_classvar(annotation)
            or annotation is dataclasses.KW_ONLY
            or is_initvar(annotation)
        )
    text = annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation
    node = _parse_expr(text)
    if node is None:
        return False
    ctx = _as_context(owner)
    top = True  # InitVar / KW_ONLY only count at the top; ClassVar also under Annotated
    for _ in range(10):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value.strip(), mode="eval").body
            except SyntaxError:
                return False
            continue
        head = node.value if isinstance(node, ast.Subscript) else node
        found, obj = _lookup_node(head, ctx)
        spelled = _dotted_tail(head)

        def head_is(target: Any, spelling: str) -> bool:
            return (found and obj is target) or (not found and spelled == spelling)

        if head_is(ClassVar, "ClassVar"):
            return True
        if top and (head_is(dataclasses.InitVar, "InitVar") or head_is(dataclasses.KW_ONLY, "KW_ONLY")):
            return True
        if isinstance(node, ast.Subscript) and head_is(Annotated, "Annotated"):
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if not elements:
                return False
            node, top = elements[0], False
            continue
        return False
    return False


def is_kw_only_sentinel(annotation: Any, owner: Any = None) -> bool:
    """True for the ``dataclasses.KW_ONLY`` pseudo-field annotation."""
    if annotation is dataclasses.KW_ONLY:
        return True
    if isinstance(annotation, (str, ForwardRef)):
        text = annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation
        return _ast_head_is(text, _as_context(owner), dataclasses.KW_ONLY, "KW_ONLY")
    return False


def is_initvar(annotation: Any, owner: Any = None) -> bool:
    """True for ``dataclasses.InitVar`` / ``InitVar[...]`` annotations."""
    if annotation is dataclasses.InitVar or isinstance(annotation, dataclasses.InitVar):
        return True
    if isinstance(annotation, (str, ForwardRef)):
        text = annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation
        return _ast_head_is(text, _as_context(owner), dataclasses.InitVar, "InitVar")
    return False


def _ast_head_is(
    text: str,
    ctx: AnnotationContext,
    target: Any,
    spelling: str,
    *,
    unwrap_annotated: bool = False,
    depth: int = 0,
) -> bool:
    node = _parse_expr(text)
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and depth < 5:
        return _ast_head_is(
            node.value, ctx, target, spelling, unwrap_annotated=unwrap_annotated, depth=depth + 1
        )
    head = node.value if isinstance(node, ast.Subscript) else node
    found, obj = _lookup_node(head, ctx)
    if unwrap_annotated and isinstance(node, ast.Subscript):
        is_annotated = (found and obj is Annotated) or (
            not found and _dotted_tail(head) == "Annotated"
        )
        if is_annotated:
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if elements:
                return _ast_head_is(
                    ast.unparse(elements[0]), ctx, target, spelling,
                    unwrap_annotated=True, depth=depth + 1,
                )
            return False
    if found:
        return obj is target
    return _dotted_tail(head) == spelling
