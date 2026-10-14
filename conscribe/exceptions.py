"""Custom exception classes for the conscribe package.

All registration-related exceptions inherit from RegistryError,
allowing consumers to catch all registry exceptions with a single handler.
Each exception carries structured context (layer name, key, class qualname, etc.)
to provide actionable error messages.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

__all__ = [
    # Errors
    "RegistryError",
    "DuplicateKeyError",
    "KeyNotFoundError",
    "InvalidRegistryKeyError",
    "ProtocolViolationError",
    "InvalidConfigSchemaError",
    "InvalidProtocolError",
    "CircularWiringError",
    "WiringResolutionError",
    "WiringProblem",
    "WIRING_PROBLEM_KINDS",
    "ConfigAnnotationError",
    "ConfigDeclarationError",
    "ConfigRenderError",
    "EmptyLayerError",
    # Warnings
    "ConscribeWarning",
    "ExtractionWarning",
    "DegradationWarning",
    "StaleStubWarning",
    "IsolationWarning",
]


class RegistryError(Exception):
    """Base class for all registry exceptions. Consumers can catch this uniformly."""


class DuplicateKeyError(RegistryError):
    """Raised when a duplicate key is registered in the same layer.

    Attributes:
        layer_name: Which layer (e.g. "agent").
        key: The conflicting key.
        existing_cls: The already-registered class.
        new_cls: The class that attempted to register.
        hint: Advice that replaced the default "To fix" sentence, or ``None``.
    """

    def __init__(
        self,
        layer_name: str,
        key: str,
        existing_cls: type,
        new_cls: type,
        *,
        hint: str | None = None,
    ) -> None:
        self.layer_name = layer_name
        self.key = key
        self.existing_cls = existing_cls
        self.new_cls = new_cls
        self.hint = hint
        if hint is None:
            hint = f"To fix: add __registry_key__ = '<unique_key>' to {new_cls.__name__}."
        message = (
            f"Duplicate key '{key}' in '{layer_name}' registry: "
            f"'{new_cls.__qualname__}' conflicts with existing "
            f"'{existing_cls.__qualname__}'. {hint}"
        )
        super().__init__(message)


class KeyNotFoundError(RegistryError, KeyError):
    """Raised when querying a key that does not exist in the registry.

    Inherits from KeyError for dict-like semantics.

    Attributes:
        layer_name: Which layer.
        key: The missing key.
        available_keys: All currently registered keys.
        hint: Advice that replaced the default discover() hint, or ``None``
            (removals use their own advice).
    """

    def __init__(
        self,
        layer_name: str,
        key: str,
        available_keys: list[str],
        *,
        hint: str | None = None,
    ) -> None:
        self.layer_name = layer_name
        self.key = key
        self.available_keys = available_keys
        self.hint = hint
        if hint is None:
            hint = (
                "Did you forget to call discover() or import the module "
                "containing this implementation?"
            )
        message = (
            f"Key '{key}' not found in '{layer_name}' registry. "
            f"Available keys: {', '.join(sorted(available_keys))}. {hint}"
        )
        # KeyError uses args[0] for repr, so pass the full message there
        super().__init__(message)


class InvalidRegistryKeyError(RegistryError, TypeError):
    """Raised when a class would be registered under an invalid key.

    A registry key is a non-empty ``str``; ``__registry_key__`` may also be a
    non-empty ``list`` of them (multi-key registration).  Anything else -- an
    Enum member, a tuple, an int, an empty string or list -- would register
    the class under a key nobody can look up (``str(Kind.REACT)`` is
    ``'Kind.REACT'``, not ``'react'``), so it is refused when the class is
    created, before it touches any registry.

    Inherits from TypeError for semantic consistency.

    Attributes:
        cls_name: Qualified name of the class being registered.
        key: The offending value.
        source: Where the value came from (``"__registry_key__"``,
            ``"register(key=...)"``, ``"key_transform"``).
        detail: What is wrong and how to fix it.
    """

    def __init__(self, cls_name: str, key: Any, source: str, detail: str) -> None:
        self.cls_name = cls_name
        self.key = key
        self.source = source
        self.detail = detail
        super().__init__(f"{cls_name}: invalid registry key {key!r} from {source}: {detail}")


class ProtocolViolationError(RegistryError, TypeError):
    """Raised when a class does not satisfy the layer's Protocol interface.

    Inherits from TypeError for semantic consistency.

    Attributes:
        layer_name: Which layer.
        cls: The offending class.
        missing_methods: Methods required by the protocol but not found on cls.
        protocol: The Protocol type that was violated.
    """

    def __init__(
        self,
        layer_name: str,
        cls: type,
        missing_methods: list[str],
        protocol: type,
    ) -> None:
        self.layer_name = layer_name
        self.cls = cls
        self.missing_methods = missing_methods
        self.protocol = protocol
        message = (
            f"'{cls.__qualname__}' cannot be registered in '{layer_name}' layer: "
            f"missing methods required by {protocol.__name__}: "
            f"{', '.join(sorted(missing_methods))}. "
            f"Implement these methods or inherit from the layer's base class."
        )
        super().__init__(message)


class InvalidConfigSchemaError(RegistryError, TypeError):
    """Raised when __config_schema__ is not a BaseModel subclass.

    Attributes:
        cls_name: Name of the class with the bad schema.
        actual: The actual value of __config_schema__.
    """

    def __init__(self, cls_name: str, actual: object) -> None:
        self.cls_name = cls_name
        self.actual = actual
        message = (
            f"{cls_name}.__config_schema__ must be a pydantic.BaseModel subclass, "
            f"got {type(actual).__name__}."
        )
        super().__init__(message)


class InvalidProtocolError(RegistryError, TypeError):
    """Raised when the provided protocol is not a @runtime_checkable Protocol.

    Attributes:
        protocol: The invalid protocol type.
    """

    def __init__(self, protocol: type) -> None:
        self.protocol = protocol
        message = (
            f"Protocol '{protocol.__name__}' must be decorated with "
            f"@runtime_checkable. "
            f"Add @runtime_checkable above your Protocol class definition."
        )
        super().__init__(message)


class CircularWiringError(RegistryError):
    """Raised when composed config detects circular wiring dependencies.

    Attributes:
        cycle: The list of layer names forming the cycle.
    """

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        cycle_str = " -> ".join(cycle + [cycle[0]]) if cycle else "(empty)"
        message = (
            f"Circular wiring dependency detected: {cycle_str}. "
            f"Composed config requires an acyclic dependency graph between layers."
        )
        super().__init__(message)


class WiringResolutionError(RegistryError):
    """Raised when one or more ``__wiring__`` entries are invalid.

    Every problem found in a build is collected and reported at once in
    ``problems`` (a list of :class:`WiringProblem`).  Causes include typos in
    keys or registry names, empty registries, and malformed entries.

    Attributes:
        problems: Every problem, in discovery order.
        cls_name, param_name, registry_name, detail: Mirror ``problems[0]``
            (kept for 1.x callers).
    """

    def __init__(
        self,
        cls_name: str | None = None,
        param_name: str | None = None,
        registry_name: str | None = None,
        detail: str | None = None,
        *,
        problems: "list[WiringProblem] | None" = None,
    ) -> None:
        if problems is None:
            problems = [
                WiringProblem(
                    owner=cls_name or "",
                    param_name=param_name or "",
                    registry_name=registry_name or "",
                    kind="unresolved",
                    detail=detail or "",
                )
            ]
        if not problems:
            raise ValueError("WiringResolutionError needs at least one problem")
        self.problems = list(problems)
        first = self.problems[0]
        self.cls_name = first.owner
        self.param_name = first.param_name
        self.registry_name = first.registry_name
        self.detail = first.detail
        if len(self.problems) == 1:
            message = str(first)
        else:
            lines = "\n".join(f"  - {p}" for p in self.problems)
            message = f"{len(self.problems)} wiring problems:\n{lines}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# 2.0 diagnostics taxonomy
#
# Rule: anything the user wrote wrong or contradictory raises one of the
# errors below at generation/build time, naming the class, the field, the bad
# value and the fix.  Warnings (``ConscribeWarning`` subclasses) are reserved
# for cases where conscribe itself cannot read a class, and for type
# degradation.  Consumers can escalate every warning with
# ``filterwarnings = error::conscribe.ConscribeWarning``.
# ---------------------------------------------------------------------------


class ConscribeWarning(UserWarning):
    """Base class for every warning conscribe emits."""


class ExtractionWarning(ConscribeWarning):
    """conscribe could not fully read a class and continued with less information.

    Example: an ``__init__`` whose signature cannot be introspected (a C
    extension), so the class contributes no knobs.
    """


class DegradationWarning(ConscribeWarning):
    """A config field's type was degraded to ``Any``.

    The field keeps its name, default and description; only its type is no
    longer validated.  The generated artifacts carry the usual degraded
    markers (header list, ``# degraded from:`` comment, ``x-degraded-from``).
    """


class WiringProblem:
    """One problem found in one ``__wiring__`` entry.

    Attributes:
        owner: Qualified name of the class that declares the wiring.
        owner_key: Registry key of the owner, if registered (else ``None``).
        param_name: The ``__wiring__`` entry name (``""`` when the whole
            ``__wiring__`` attribute is malformed).
        registry_name: Target registry name (``""`` when not applicable).
        kind: Stable machine-readable code, one of ``WIRING_PROBLEM_KINDS``.
        detail: Human-readable description, including how to fix it.
    """

    __slots__ = ("owner", "owner_key", "param_name", "registry_name", "kind", "detail")

    def __init__(
        self,
        owner: str,
        param_name: str,
        registry_name: str,
        kind: str,
        detail: str,
        owner_key: str | None = None,
    ) -> None:
        self.owner = owner
        self.owner_key = owner_key
        self.param_name = param_name
        self.registry_name = registry_name
        self.kind = kind
        self.detail = detail

    def __str__(self) -> str:
        who = self.owner if self.owner_key is None else f"{self.owner} (key '{self.owner_key}')"
        entry = f" wiring '{self.param_name}'" if self.param_name else " __wiring__"
        return f"{who}{entry}: {self.detail}"

    def __repr__(self) -> str:
        return (
            f"WiringProblem(owner={self.owner!r}, param_name={self.param_name!r}, "
            f"registry_name={self.registry_name!r}, kind={self.kind!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WiringProblem):
            return NotImplemented
        return all(getattr(self, s) == getattr(other, s) for s in self.__slots__)

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, s) for s in self.__slots__))


#: Every value ``WiringProblem.kind`` may take.
WIRING_PROBLEM_KINDS: frozenset[str] = frozenset(
    {
        "malformed",             # grammar: bad tuple shape, wrong types, unknown dict fields
        "not_a_dict",            # __wiring__ itself is not a dict
        "non_str_key",           # a key in a key list is not a str
        "empty_keys",            # an explicit key list (Mode 2 or Mode 3) is empty
        "unknown_registry",      # target registry does not exist
        "empty_registry",        # target registry exists but holds no keys
        "unknown_key",           # explicit required key not registered
        "unknown_optional_key",  # explicit optional key not registered
        "bad_default",           # F4 default key invalid or not allowed for this entry
        "contradiction",         # e.g. declare_only entry that has a runtime slot
        "unsupported",           # e.g. __wiring__ on a nested-mode layer class
        "unresolved",            # legacy single-problem construction
    }
)


class ConfigAnnotationError(RegistryError):
    """An annotation needed for a config field cannot be evaluated.

    Raised for names that exist nowhere (typos) and for ``Annotated``
    metadata (e.g. ``Field(...)``) that cannot be evaluated.  Names that are
    only imported under ``if TYPE_CHECKING:`` are *not* errors: those fields
    degrade to ``Any`` with a ``DegradationWarning``.
    """

    def __init__(self, owner: str, field: str, annotation: str, detail: str) -> None:
        self.owner = owner
        self.field = field
        self.annotation = annotation
        self.detail = detail
        super().__init__(
            f"{owner}.{field}: cannot evaluate annotation {annotation!r}: {detail}"
        )


class ConfigDeclarationError(RegistryError):
    """A class declaration that conscribe refuses because it is contradictory.

    Examples: a signature default and a ``Field(default_factory=...)`` on the
    same parameter, a ``__config_schema__`` that lacks a wired field, an
    inline-only layer built with ``inline_wiring=False``.
    """

    def __init__(self, owner: str, detail: str, field: str | None = None) -> None:
        self.owner = owner
        self.field = field
        self.detail = detail
        where = owner if field is None else f"{owner}.{field}"
        super().__init__(f"{where}: {detail}")


class ConfigRenderError(RegistryError):
    """A value or type cannot be written faithfully into generated source.

    conscribe refuses to write a generated file that would differ from the
    in-process model; the message names the model, the field and the value.
    """

    def __init__(self, model: str, field: str, value_repr: str, detail: str) -> None:
        self.model = model
        self.field = field
        self.value_repr = value_repr
        self.detail = detail
        super().__init__(f"{model}.{field}: cannot render {value_repr}: {detail}")


class EmptyLayerError(RegistryError):
    """A layer passed to a config builder has no registered implementations.

    The message cites the modules ``discover()`` recorded as failed to
    import (:func:`conscribe.discover.import_failures`): classes defined in
    them never registered, which is the usual reason a layer is empty.

    Attributes:
        layer_names: The empty layers.
        import_failures: ``{module: "ExcType: message"}`` -- the failures
            ``discover()`` had recorded when the error was raised (empty
            when it recorded none or was never imported).
    """

    def __init__(
        self,
        layer_names: list[str],
        import_failures: Mapping[str, str] | None = None,
    ) -> None:
        self.layer_names = list(layer_names)
        if import_failures is None:
            # Read lazily through sys.modules: a process that never imported
            # conscribe.discover has no failures to report (and importing it
            # here would be circular).
            discover = sys.modules.get("conscribe.discover")
            import_failures = discover.import_failures() if discover is not None else {}
        self.import_failures = dict(import_failures)
        names = ", ".join(repr(n) for n in self.layer_names)
        message = (
            f"Layer(s) {names} have no registered implementations. "
            f"Register at least one class, check that discover() imported the "
            f"package holding them, or leave the layer out of the build."
        )
        if self.import_failures:
            items = sorted(self.import_failures.items())
            shown = "; ".join(f"{module} ({detail})" for module, detail in items[:5])
            more = f"; +{len(items) - 5} more" if len(items) > 5 else ""
            message += (
                f" discover() recorded import failures, and classes defined in those "
                f"modules are not registered: {shown}{more}."
            )
        super().__init__(message)


class StaleStubWarning(ConscribeWarning):
    """conscribe-generated ``.pyi`` files that the current run did not rewrite.

    A stub left over from an earlier conscribe version still replaces its
    module for type checkers.  The warning lists the files; pass
    ``prune=True`` to ``write_layer_stubs`` (or ``--prune`` to the CLI) to
    delete exactly those files.
    """


class IsolationWarning(ConscribeWarning):
    """An isolation block kept registrations that outlive it.

    Emitted by ``conscribe.testing.isolated_registries(warn_kept=True)`` (and
    the pytest fixture with ``conscribe_warn_kept = true``) when modules
    imported for the first time inside the block registered classes: those
    modules stay imported and never run again, so their registrations stay
    for the rest of the process.  The message names the modules and keys.
    """
