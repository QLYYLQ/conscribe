"""Shared internal helpers for the config subsystem: who defines ``__init__``.

Two generations of the "init definer" rule live here:

* :func:`find_init_definer` -- the 1.x rule (first class in the MRO whose
  ``__dict__`` holds an ``__init__``).  Only two 1.x compatibility views use
  it: :func:`conscribe.config.mro.collect_mro_params` (reports parameters
  relative to the 1.x definer) and the extractor's old-shape guard (which
  must recognise exactly the classes 1.x flattened).  It treats ``typing``'s
  Protocol placeholder ``__init__`` as a real definer and is
  instantiation-dependent (see below).
* :func:`constructor_definer` -- the 2.0 rule used by everything else:
  :mod:`conscribe.introspect`, :func:`conscribe.config.mro.walk_init_chain`
  and through them the extractor, wiring, stubs and fingerprint.

The 2.0 rule (:func:`own_init`) ignores an ``__init__`` found in a class's
``__dict__`` when it is not the class's own constructor:

* the Protocol placeholder ``typing._no_init_or_replace_init`` (it forwards
  to the next real ``__init__``);
* a copy of ``object.__init__`` -- ``_no_init_or_replace_init`` copies the
  next real ``__init__`` into a concrete Protocol subclass on its first
  instantiation, so without this rule the answer would change after the
  first ``Impl()``;
* a copy of an ``__init__`` defined further up the class's MRO (credited to
  the original definer, for the same reason).

``typing_extensions``' placeholder ``_no_init`` (Python 3.10/3.11) does
*not* forward: it swallows every argument.  It counts as a definer, and
:func:`is_swallowing_placeholder` lets callers report it.
"""
from __future__ import annotations

import inspect
import types
from typing import Any, Union

from pydantic import BaseModel
from pydantic.dataclasses import is_pydantic_dataclass as _is_pydantic_dataclass

from conscribe.config._annotations import safe_signature

#: Name of typing's forwarding Protocol placeholder ``__init__``.
_FORWARDING_PLACEHOLDERS = frozenset({"_no_init_or_replace_init"})
#: Names of Protocol placeholders that swallow their arguments.
_SWALLOWING_PLACEHOLDERS = frozenset({"_no_init"})
_PLACEHOLDER_MODULES = frozenset({"typing", "typing_extensions"})

__all__ = [
    "constructor_definer",
    "dataclass_field_owner",
    "find_init_definer",
    "is_bridge_class",
    "is_generated_init",
    "is_placeholder_init",
    "is_pydantic_dataclass",
    "is_pydantic_model",
    "is_swallowing_placeholder",
    "model_init_only_forwards",
    "named_params",
    "own_init",
    "qualified_name",
]


def find_init_definer(cls: type) -> Union[type, None]:
    """Walk MRO to find the class that actually defines ``__init__`` (1.x rule).

    Skips ``object`` (its ``__init__`` is not useful for extraction).

    Kept unchanged for 1.x callers.  New code uses
    :func:`constructor_definer`, which also skips Protocol placeholders and
    copied ``__init__`` functions.
    """
    for klass in cls.__mro__:
        if klass is object:
            continue
        if "__init__" in klass.__dict__:
            return klass
    return None


# ---------------------------------------------------------------------------
# Placeholder / copy detection
# ---------------------------------------------------------------------------


def is_placeholder_init(func: Any) -> bool:
    """True for ``typing`` / ``typing_extensions`` Protocol placeholder inits."""
    return (
        getattr(func, "__module__", None) in _PLACEHOLDER_MODULES
        and getattr(func, "__name__", None)
        in (_FORWARDING_PLACEHOLDERS | _SWALLOWING_PLACEHOLDERS)
    )


def is_swallowing_placeholder(func: Any) -> bool:
    """True for a Protocol placeholder ``__init__`` that ignores its arguments.

    ``typing_extensions.Protocol`` on Python 3.10/3.11 installs ``_no_init``,
    which never calls the next ``__init__``: a class whose MRO reaches it
    before a real ``__init__`` silently discards every constructor argument.
    """
    return is_placeholder_init(func) and getattr(func, "__name__", None) in _SWALLOWING_PLACEHOLDERS


def own_init(klass: type) -> Any:
    """The ``__init__`` *klass* itself defines, or ``None``.

    ``None`` when the class's ``__dict__`` has no ``__init__``, or holds the
    forwarding Protocol placeholder, ``object.__init__``, or a copy of an
    ``__init__`` defined by a class further up its own MRO.  A swallowing
    placeholder (``typing_extensions._no_init``) is returned: it really is
    the constructor the runtime calls.
    """
    if klass is object:
        return None
    init = klass.__dict__.get("__init__")
    if init is None or init is object.__init__:
        return None
    if is_placeholder_init(init):
        # The runtime calls the first one it finds, so a swallowing
        # placeholder is never a mere copy.
        return init if is_swallowing_placeholder(init) else None
    for base in klass.__mro__[1:]:
        if base.__dict__.get("__init__") is init:
            return None  # a copy: credited to the original definer
    return init


def constructor_definer(cls: type) -> Union[type, None]:
    """The class whose ``__init__`` ``cls(...)`` runs first (2.0 rule).

    First class in the MRO with :func:`own_init`; ``None`` when nothing but
    ``object`` (or placeholders / copies of ``object.__init__``) defines one,
    i.e. the constructor accepts no arguments.
    """
    for klass in cls.__mro__:
        if own_init(klass) is not None:
            return klass
    return None


# ---------------------------------------------------------------------------
# Class kinds
# ---------------------------------------------------------------------------


def is_pydantic_model(cls: Any) -> bool:
    """True for a strict subclass of ``pydantic.BaseModel``."""
    return isinstance(cls, type) and issubclass(cls, BaseModel) and cls is not BaseModel


def is_pydantic_dataclass(cls: Any) -> bool:
    """True for a class decorated with ``pydantic.dataclasses.dataclass``.

    Only the decorated class itself (its own ``__dict__`` carries the
    validator); a stdlib dataclass subclass of it does not count.
    """
    return isinstance(cls, type) and bool(_is_pydantic_dataclass(cls))


def model_init_only_forwards(cls: type) -> bool:
    """True when ``cls(...)`` reaches ``BaseModel.__init__`` with every keyword.

    The case for the BaseModel fast path: *cls* is a pydantic model and every
    ``__init__`` its constructor runs before ``BaseModel.__init__`` -- its
    own, a model parent's, a cooperative mixin's
    (``class Tool(SetupMixin, BaseModel)`` with
    ``SetupMixin.__init__(self, **kw): super().__init__(**kw)``) -- takes
    nothing but ``*args`` / ``**kwargs``.  Each of them forwards every keyword
    unchanged, so the model's fields are exactly what the constructor
    accepts.
    """
    if not is_pydantic_model(cls):
        return False
    for klass in cls.__mro__:
        if klass is BaseModel or klass is object:
            return True
        init = own_init(klass)
        if init is None:
            continue
        if is_swallowing_placeholder(init):
            return False
        sig = safe_signature(init)
        if sig is None or named_params(sig):
            return False
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return False
    return True  # pragma: no cover - a pydantic model's MRO always reaches BaseModel


def is_generated_init(definer: type, init: Any) -> bool:
    """True when *init* was generated by a class decorator, not written by hand.

    Covers stdlib dataclasses (code compiled from ``"<string>"`` on a class
    with ``__dataclass_fields__``) and attrs (``"<attrs generated ...>"``).
    Annotations of a generated ``__init__`` belong to the class body that
    declared each field, not to the generated function.
    """
    try:
        target = inspect.unwrap(init)
    except ValueError:  # a __wrapped__ cycle
        target = init
    code = getattr(target, "__code__", None)
    if not isinstance(code, types.CodeType):
        return False
    filename = code.co_filename
    if filename.startswith("<attrs generated"):
        return True
    if filename != "<string>":
        return False
    if "__dataclass_fields__" not in definer.__dict__:
        return False
    return getattr(target, "__qualname__", "") == f"{definer.__qualname__}.__init__"


def dataclass_field_owner(cls: type, name: str) -> Union[type, None]:
    """The dataclass in *cls*'s MRO that declares field *name*.

    Dataclass ``Field`` objects are shared by identity down the hierarchy and
    re-created when a subclass re-declares the field, so the declaring class
    is the most-base class whose own ``__dataclass_fields__`` holds the very
    ``Field`` object *cls* uses.  ``None`` if *cls* has no such field.
    """
    fields = getattr(cls, "__dataclass_fields__", None)
    if not isinstance(fields, dict) or name not in fields:
        return None
    target = fields[name]
    for klass in reversed(cls.__mro__):
        own = klass.__dict__.get("__dataclass_fields__")
        if isinstance(own, dict) and own.get(name) is target:
            return klass
    return None


def qualified_name(obj: Any) -> str:
    """``module.QualName`` of a class or function, for messages.

    Falls back to ``__name__`` / ``repr`` for objects without a qualified
    name, and omits the module when it is unknown.
    """
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
    return f"{module}.{qualname}" if module else qualname


def is_bridge_class(cls: type) -> bool:
    """True for a class created by ``LayerRegistrar.bridge()``.

    Bridges are abstract wrappers conscribe creates around an external class;
    their ``__module__`` is a conscribe module, so their install location says
    nothing about the user's code.  Honours an explicit
    ``__conscribe_bridge__`` marker if the registration layer sets one.
    """
    if cls.__dict__.get("__conscribe_bridge__") is not None:
        return True
    module = getattr(cls, "__module__", "") or ""
    return module.startswith("conscribe.registration.") and cls.__dict__.get("__abstract__") is True


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def named_params(sig: inspect.Signature) -> list[inspect.Parameter]:
    """Parameters of an unbound ``__init__`` signature a caller can name.

    Drops the instance parameter -- the first positional parameter, whatever
    it is called (``self``, ``this``, ``__dataclass_self__``) -- and
    ``*args`` / ``**kwargs``.  Positional-only parameters are kept (check
    ``kind`` if you need keyword-only routing).
    """
    params = list(sig.parameters.values())
    if params and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        params = params[1:]
    return [
        p
        for p in params
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
