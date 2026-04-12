"""Collect stub information from registered classes with wired attributes."""
from __future__ import annotations

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
    """Method signature for .pyi rendering."""

    name: str
    signature: str  # e.g. "(self, x: int) -> str"
    decorators: tuple[str, ...]  # e.g. ("@classmethod",)


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
            init_signature = str(sig)
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

    return ClassStubInfo(
        cls=cls,
        class_name=cls.__name__,
        module=cls.__module__,
        source_file=source_file,
        bases=tuple(cls.__bases__),
        init_signature=init_signature,
        injected_attrs=tuple(injected_attrs),
        methods=_collect_own_methods(cls),
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
                signature=str(sig),
                decorators=tuple(decorators),
            )
        )

    return tuple(methods)
