"""LayerRegistry — thread-safe key-to-class mapping with Protocol check caching.

Each layer instantiates one LayerRegistry. It is a pure storage layer that
does not know about registration paths (metaclass/bridge/manual).
Protocol check caching borrows from CPython ABCMeta's positive/negative cache
with a global invalidation counter.

The module also owns the process-wide bookkeeping that
``conscribe.testing.isolated_registries()`` and ``conscribe.discover()`` rely
on: the set of live registries (including registries shadowed out of the
global index) and the mutation journal (see "Isolation journal" below).
"""
from __future__ import annotations

import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

if TYPE_CHECKING:
    from typing import IO
from weakref import WeakSet

from conscribe.exceptions import (
    DuplicateKeyError,
    InvalidProtocolError,
    KeyNotFoundError,
    ProtocolViolationError,
)

P = TypeVar("P")

# Global registry index: maps layer name -> LayerRegistry instance.
# Enables cross-registry lookups for wiring resolution.
_REGISTRY_INDEX: dict[str, "LayerRegistry[Any]"] = {}
_REGISTRY_INDEX_LOCK = threading.Lock()

# Every LayerRegistry that is still alive, including registries whose name
# was taken over by a later registry of the same name (shadowed out of
# ``_REGISTRY_INDEX``).  Isolation snapshots all of them; the atomic
# registration purge and the re-creation rebind search all of them.  A
# WeakSet must not be iterated while another thread adds to it ("Set changed
# size during iteration"), so additions and snapshots share one lock.
_LIVE_REGISTRIES: "WeakSet[LayerRegistry[Any]]" = WeakSet()
_LIVE_REGISTRIES_LOCK = threading.Lock()


# ── Isolation journal ──────────────────────────────────────────────────
#
# ``isolated_registries()`` restores every registry to its entry state on exit,
# except for registrations made by the top-level code of modules imported for
# the first time inside the block (those modules stay in ``sys.modules`` and
# never run again, so their registrations must survive).  Telling the two
# apart needs a record of *who* made each change, so while at least one block
# is open every mutation below appends an ``_Event``.  ``discover()`` uses the
# same events (through ``_recording()``) to roll back the registrations of a
# module whose import failed.  When neither is active the cost is two
# truthiness checks per mutation.
#
# INVARIANT: every change to a registry's ``_store`` and to ``_REGISTRY_INDEX``
# goes through ``LayerRegistry.__init__`` / ``add`` / ``remove`` /
# ``_replace_class`` / ``_discard_class`` / ``_discard_binding`` /
# ``_restore_binding`` or ``_deregister`` (and ``LayerRegistrar.register()``
# records its attribute writes).  The only other writers are isolation's own
# exit restore and ``activate()``, which journals what it applies.  A mutation
# that bypasses all of them is still undone by the snapshot restore, but it
# can never be *kept* by the import rule (nor rolled back by ``discover()``).


class _MissingType:
    """Sentinel type for "no value" (a key absent from a store, an unset attribute)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


_MISSING: Any = _MissingType()


class _Event(NamedTuple):
    """One recorded registry mutation.

    ``kind`` and the meaning of the other fields:

    - ``"set"``: ``target`` registry, ``key``; ``old`` / ``new`` are the classes
      bound before / after (``_MISSING`` for absent) — adds, removals, rebinds.
    - ``"create"``: ``target`` is the new registry, ``key`` its name, ``old`` the
      registry it shadowed in the index (or ``_MISSING``), ``new`` the registry.
    - ``"index"``: ``key`` is a registry name, ``old`` / ``new`` the registries
      the index bound it to before / after (``_deregister``, re-activation).
    - ``"attrs"``: ``target`` is a class, ``key`` the ``(registry, key)`` pairs
      the call registered it under, ``old`` / ``new`` map each attribute name
      the call changed to its value before / after (``_MISSING`` for absent).

    ``module`` is the ``__name__`` of the module whose top-level code was
    executing when the mutation happened (``None`` when there is none); see
    ``_importing_module()``.
    """

    kind: str
    target: Any
    key: Any
    old: Any
    new: Any
    module: str | None


# Open isolation blocks, innermost last (managed by registration.isolation).
_ISOLATION_STACK: list[Any] = []
# Mutations recorded since the outermost open block started.
_JOURNAL: list[_Event] = []
# Event lists of the module imports ``discover()`` is running (see
# ``_recording()``); every mutation is appended to each of them.
_RECORDERS: list[list[_Event]] = []


def _importing_module() -> str | None:
    """Return the name of the module whose top-level code is running.

    That is the innermost ``<module>`` frame on the stack that belongs to a
    module registered in ``sys.modules`` under its ``__name__``: its globals
    are that module's ``__dict__`` (or carry that module's ``__spec__``,
    for a module that replaced its own ``sys.modules`` entry).  A
    registration made while ``import pkg.mod`` executes ``pkg.mod``'s
    top-level code (directly, through a class statement, or through any
    helper it calls) is attributed to ``pkg.mod`` -- whatever ``__module__``
    the class claims, and also when ``pkg.mod`` runs code through ``exec()``
    with a namespace of its own (a plugin loader giving generated classes a
    synthetic module name): such a frame is part of the module that runs it.

    Returns ``None`` when no such frame exists or frames cannot be
    inspected; isolation then treats the mutation as made by the block itself
    (rolled back), and ``discover()`` never rolls it back.
    """
    getframe = getattr(sys, "_getframe", None)
    if getframe is None:  # pragma: no cover - CPython and PyPy provide it
        return None
    modules = sys.modules
    frame = getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "<module>":
            namespace = frame.f_globals
            name = namespace.get("__name__")
            if isinstance(name, str) and _runs_module(name, namespace, modules):
                return name
        frame = frame.f_back
    return None


def _runs_module(name: str, namespace: dict[str, Any], modules: Mapping[str, Any]) -> bool:
    """Whether a ``<module>`` frame with *namespace* executes module *name*."""
    module = modules.get(name)
    if module is None:
        return False
    if getattr(module, "__dict__", None) is namespace:
        return True
    spec = namespace.get("__spec__")
    return getattr(spec, "name", None) == name


def _record(kind: str, target: Any, key: Any, old: Any, new: Any) -> None:
    """Append one ``_Event`` to the isolation journal and every discover recorder.

    Callers check ``_ISOLATION_STACK or _RECORDERS`` first, so a mutation
    costs nothing extra while neither is active.
    """
    event = _Event(kind, target, key, old, new, _importing_module())
    if _ISOLATION_STACK:
        _JOURNAL.append(event)
    for recorder in tuple(_RECORDERS):
        recorder.append(event)


@contextmanager
def _recording() -> Iterator[list[_Event]]:
    """Collect every registry mutation made while the block runs.

    ``discover()`` wraps each first-time module import in one, to roll back
    exactly the registrations of modules whose import failed.  Recorders
    nest (a module may call ``discover()`` while being imported).
    """
    events: list[_Event] = []
    _RECORDERS.append(events)
    try:
        yield events
    finally:
        # By identity: another thread may have pushed a recorder meanwhile,
        # and two recorders with equal contents compare equal.
        for index in range(len(_RECORDERS) - 1, -1, -1):
            if _RECORDERS[index] is events:
                del _RECORDERS[index]
                break


def _apply_attrs(cls: type, values: Mapping[str, Any]) -> None:
    """Set each named attribute of ``cls``; a ``_MISSING`` value deletes it."""
    for name, value in values.items():
        if value is _MISSING:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, value)


def get_registry(name: str) -> "LayerRegistry[Any] | None":
    """Look up a LayerRegistry by layer name.

    Args:
        name: The layer name (e.g. ``"agent_loop"``).

    Returns:
        The LayerRegistry instance, or ``None`` if not found.
    """
    with _REGISTRY_INDEX_LOCK:
        return _REGISTRY_INDEX.get(name)


def _deregister(name: str) -> None:
    """Remove a registry from the global index. Intended for test cleanup.

    Prefer ``conscribe.testing.isolated_registries()``, which also undoes
    the registrations and restores any registry the name used to shadow.
    """
    with _REGISTRY_INDEX_LOCK:
        removed = _REGISTRY_INDEX.pop(name, _MISSING)
    if removed is not _MISSING and (_ISOLATION_STACK or _RECORDERS):
        _record("index", None, name, removed, _MISSING)


def _live_registries() -> list["LayerRegistry[Any]"]:
    """Snapshot of every live registry (indexed or shadowed)."""
    with _LIVE_REGISTRIES_LOCK:
        return list(_LIVE_REGISTRIES)


def _purge_class(cls: type) -> None:
    """Remove every binding of ``cls`` from every live registry.

    The atomic-registration rule: when registering a class that is being
    created fails part-way (a multi-key collision, the second registry of a
    cross-registry diamond, a failing hook), no registry may keep the class
    whose definition just failed.
    """
    for registry in _live_registries():
        registry._discard_class(cls)


def _rebind_recreated(origin: int, new_cls: type, namespace: Mapping[str, Any]) -> bool:
    """Rebind a registered class that was re-created from a copy of its namespace.

    ``@dataclass(slots=True)``, ``attrs.define`` and pydantic's
    ``dataclass(slots=True)`` replace a freshly defined class with a new class
    built from a copy of its ``__dict__``; the copy carries the
    ``__registry_origin__`` stamp of the original.  When ``origin`` is the id of
    a registered class that is the copy source of ``new_cls`` (same
    ``__module__`` and ``__name__``, same ``__registry_key__`` /
    ``__registry_keys__`` in the namespace), every key bound to the original
    in every live registry is rebound to ``new_cls`` (journaled, so isolation
    sees the change).

    Args:
        origin: ``__registry_origin__`` found in the new class's own namespace.
        new_cls: The re-created class.
        namespace: The namespace ``new_cls`` was created from (the metaclass
            namespace, or ``new_cls.__dict__`` in the ``__init_subclass__``
            path).

    Returns:
        ``True`` when ``new_cls`` now holds the original's registrations (the
        caller stamps it and skips normal registration); this includes the
        case where another metaclass of the same class creation (a
        cross-registry diamond) already rebound it.  ``False`` when the
        namespace is not a re-creation of a registered class.
    """
    registries = _live_registries()
    old: type | None = None
    for registry in registries:
        old = registry._find_by_id(origin)
        if old is not None:
            break
    if old is None:
        # A sibling metaclass of the same class statement (``A.Meta | B.Meta``)
        # may have rebound the copy source already.
        return any(registry._holds(new_cls) for registry in registries)
    if not _is_copy_source(old, new_cls, namespace):
        return False
    for registry in registries:
        registry._replace_class(old, new_cls)
    return True


def _is_copy_source(old: type, new_cls: type, namespace: Mapping[str, Any]) -> bool:
    """Whether ``new_cls`` was built from a copy of ``old``'s namespace."""
    return (
        old is not new_cls
        and old.__name__ == new_cls.__name__
        and old.__module__ == new_cls.__module__
        and old.__dict__.get("__registry_key__", _MISSING)
        == namespace.get("__registry_key__", _MISSING)
        and old.__dict__.get("__registry_keys__", _MISSING)
        == namespace.get("__registry_keys__", _MISSING)
    )


def _duplicate_hint(key: str, existing: type, new: type) -> str | None:
    """Replacement for DuplicateKeyError's default advice, or ``None``."""
    if existing is new:
        return (
            f"{new.__qualname__} is already registered under '{key}': the key "
            f"is listed twice in its __registry_key__, or the class was "
            f"registered twice."
        )
    module = getattr(new, "__module__", None)
    qualname = getattr(new, "__qualname__", None)
    if (
        module == getattr(existing, "__module__", None)
        and qualname == getattr(existing, "__qualname__", None)
    ):
        return (
            f"Both are '{module}.{qualname}': the class body ran a second time "
            f"(importlib.reload, a re-executed notebook cell, a test or helper "
            f"function that defines the class, or a module imported under two "
            f"names), and a registry keeps exactly one class per key. In tests, "
            f"define such classes inside "
            f"'with conscribe.testing.isolated_registries():' (or use the "
            f"'conscribe_isolated' pytest fixture) so every run starts from "
            f"the registry state it had before."
        )
    if (
        isinstance(new, type)
        and isinstance(existing, type)
        and new.__name__ == existing.__name__
        and issubclass(new, existing)
    ):
        return (
            f"'{new.__qualname__}' subclasses the registered class of the same "
            f"name, so it inherits its key. Give the subclass its own "
            f"__registry_key__. (Stacking @pydantic.dataclasses.dataclass on a "
            f"stdlib @dataclass does this implicitly: use one dataclass "
            f"decorator on registered classes.)"
        )
    return None


class LayerRegistry(Generic[P]):
    """Generic registry. One instance per layer.

    Internal state:
        name: layer name (for error messages)
        protocol: @runtime_checkable Protocol
        _store: dict[str, type] — strong references
        _lock: threading.Lock — thread safety (NOT RLock)
        _check_cache: WeakSet — positive Protocol check cache
        _check_negative_cache: WeakSet — negative Protocol check cache
        _invalidation_counter: int — instance variable, incremented whenever
            a binding is removed
        _negative_cache_version: int — instance variable, compared to own counter
    """

    def __init__(self, name: str, protocol: type, *, separator: str = "") -> None:
        """Initialize the registry.

        The new registry is added to the global index under ``name``; an
        existing registry of the same name is shadowed (it stays alive and
        tracked, but ``get_registry(name)`` returns the new one).

        Args:
            name: Layer name, used in error messages and debug output.
            protocol: A @runtime_checkable Protocol subclass.
            separator: Key separator for hierarchical keys (e.g. ``"."``).
                Empty string means flat keys (backward compatible).

        Raises:
            InvalidProtocolError: If protocol is not @runtime_checkable.
        """
        if not getattr(protocol, "_is_runtime_protocol", False):
            raise InvalidProtocolError(protocol)

        self.name = name
        self.protocol = protocol
        self.separator = separator
        self._store: dict[str, type] = {}
        self._lock = threading.Lock()
        self._check_cache: WeakSet = WeakSet()
        self._check_negative_cache: WeakSet = WeakSet()
        self._invalidation_counter: int = 0
        self._counter_lock = threading.Lock()
        self._negative_cache_version: int = self._invalidation_counter
        self._protocol_methods: frozenset[str] = frozenset(
            name
            for name in dir(protocol)
            if not name.startswith("_")
            and callable(getattr(protocol, name, None))
        )

        with _LIVE_REGISTRIES_LOCK:
            _LIVE_REGISTRIES.add(self)
        # Register in the global index for cross-registry wiring lookups
        with _REGISTRY_INDEX_LOCK:
            shadowed = _REGISTRY_INDEX.get(name, _MISSING)
            _REGISTRY_INDEX[name] = self
        if _ISOLATION_STACK or _RECORDERS:
            _record("create", self, name, shadowed, self)

    def runtime_check(self, cls: type) -> None:
        """Check whether cls satisfies this layer's Protocol.

        Args:
            cls: The class to validate.

        Raises:
            ProtocolViolationError: If cls is missing required protocol methods.
        """
        with self._lock:
            # 1. Positive cache hit -> return immediately
            if cls in self._check_cache:
                return

            # 2. Check if negative cache is stale
            if self._negative_cache_version != self._invalidation_counter:
                self._check_negative_cache = WeakSet()
                self._negative_cache_version = self._invalidation_counter

            # 3. Negative cache hit -> raise
            if cls in self._check_negative_cache:
                missing = self._compute_missing_methods(cls)
                raise ProtocolViolationError(
                    self.name, cls, list(missing), self.protocol
                )

        # 4. Compute missing methods (outside lock — no shared state mutation)
        missing = self._compute_missing_methods(cls)

        with self._lock:
            # 5. No missing methods -> add to positive cache
            if not missing:
                self._check_cache.add(cls)
                return

            # 6. Missing methods -> add to negative cache, raise
            self._check_negative_cache.add(cls)

        raise ProtocolViolationError(
            self.name, cls, list(missing), self.protocol
        )

    def _compute_missing_methods(self, cls: type) -> set[str]:
        """Compute the set of protocol methods missing from cls."""
        return {
            name
            for name in self._protocol_methods
            if not hasattr(cls, name) or not callable(getattr(cls, name))
        }

    def add(self, key: str, cls: type, *, protocol_check: bool = False) -> None:
        """Register a class under the given key.

        Args:
            key: The lookup key for this class.
            cls: The class to register.
            protocol_check: Whether to run Protocol compliance check first.
                True for path C (manual), False for path A/B (inheritance).

        Raises:
            DuplicateKeyError: If key already exists in this registry.  When
                the existing class has the same module and qualified name
                (a class body that ran twice), the message says so and
                points to ``conscribe.testing.isolated_registries()``.
            ProtocolViolationError: If protocol_check=True and cls fails check.
        """
        if protocol_check:
            self.runtime_check(cls)

        with self._lock:
            if key in self._store:
                existing = self._store[key]
                raise DuplicateKeyError(
                    self.name, key, existing, cls,
                    hint=_duplicate_hint(key, existing, cls),
                )
            self._store[key] = cls
        if _ISOLATION_STACK or _RECORDERS:
            _record("set", self, key, _MISSING, cls)

    def remove(self, key: str, *, missing_ok: bool = False) -> type | None:
        """Remove the binding of ``key`` and return the class it was bound to.

        Exactly one entry is removed.  A class registered under several keys
        (``__registry_key__ = [...]``) stays registered under its other keys,
        and the ``__registry_key__`` / ``__registry_keys__`` attributes of the
        class are left as they are.  Hierarchical keys are independent
        entries: removing ``"openai"`` does not remove ``"openai.azure"``.

        In tests prefer ``conscribe.testing.isolated_registries()``, which
        restores everything a test registered or removed.

        Args:
            key: The key to remove.
            missing_ok: If True, a key that is not registered is not an error
                and ``None`` is returned (idempotent teardown).

        Returns:
            The class that was bound to ``key``, or ``None`` when the key was
            absent and ``missing_ok`` is true.

        Raises:
            KeyNotFoundError: If key is not registered and ``missing_ok`` is
                false.  It is a ``KeyError`` subclass.
        """
        with self._lock:
            cls = self._store.pop(key, _MISSING)
            if cls is _MISSING:
                available = list(self._store)
        if cls is _MISSING:
            if missing_ok:
                return None
            raise KeyNotFoundError(
                self.name, key, available,
                hint=self._removal_hint(key, available),
            )
        self._bump_invalidation()
        if _ISOLATION_STACK or _RECORDERS:
            _record("set", self, key, cls, _MISSING)
        return cls

    def _removal_hint(self, key: str, available: list[str]) -> str:
        """Advice for removing a key that is not registered."""
        hint = (
            "Nothing to remove; pass missing_ok=True to make the removal a "
            "no-op when the key is absent (e.g. in teardown code)."
        )
        if self.separator:
            prefix = key + self.separator
            below = sorted(k for k in available if k.startswith(prefix))
            if below:
                hint = (
                    f"'{key}' is only a prefix here; the keys below it are "
                    f"separate entries ({', '.join(below)}) and must be "
                    f"removed one by one. " + hint
                )
        return hint

    def _bump_invalidation(self) -> None:
        """Invalidate the negative Protocol-check cache after a removal."""
        with self._counter_lock:
            self._invalidation_counter += 1

    # ── Internal mutations used by registration and isolation ──

    def _find_by_id(self, origin: int) -> type | None:
        """Return the registered class whose ``id()`` is ``origin``, if any."""
        with self._lock:
            for value in self._store.values():
                if id(value) == origin:
                    return value
        return None

    def _holds(self, cls: type) -> bool:
        """Whether ``cls`` is bound to at least one key."""
        with self._lock:
            return any(value is cls for value in self._store.values())

    def _replace_class(self, old: type, new: type) -> list[str]:
        """Rebind every key bound to ``old`` to ``new``; return those keys."""
        with self._lock:
            keys = [k for k, v in self._store.items() if v is old]
            for k in keys:
                self._store[k] = new
        if keys and (_ISOLATION_STACK or _RECORDERS):
            for k in keys:
                _record("set", self, k, old, new)
        return keys

    def _discard_class(self, cls: type) -> list[str]:
        """Remove every key bound to ``cls``; return those keys."""
        with self._lock:
            keys = [k for k, v in self._store.items() if v is cls]
            for k in keys:
                del self._store[k]
        if keys:
            self._bump_invalidation()
            if _ISOLATION_STACK or _RECORDERS:
                for k in keys:
                    _record("set", self, k, cls, _MISSING)
        return keys

    def _discard_binding(self, key: str, cls: type) -> bool:
        """Remove ``key`` only if it is still bound to ``cls``."""
        return self._restore_binding(key, cls, _MISSING)

    def _restore_binding(self, key: str, current: type, previous: Any) -> bool:
        """Bind ``key`` back to ``previous`` if it is still bound to ``current``.

        ``previous`` is a class, or ``_MISSING`` to remove the key.  Undoes
        one recorded ``"set"`` event (``discover()``'s rollback).

        Returns:
            ``True`` when the binding changed.
        """
        with self._lock:
            if self._store.get(key, _MISSING) is not current:
                return False
            if previous is _MISSING:
                del self._store[key]
            else:
                self._store[key] = previous
        self._bump_invalidation()
        if _ISOLATION_STACK or _RECORDERS:
            _record("set", self, key, current, previous)
        return True

    # ── Queries ──

    def get(self, key: str) -> type:
        """Look up a registered class.

        Args:
            key: The registration key.

        Returns:
            The registered class.

        Raises:
            KeyNotFoundError: If key does not exist.
        """
        with self._lock:
            if key not in self._store:
                raise KeyNotFoundError(
                    self.name, key, list(self._store.keys())
                )
            return self._store[key]

    def get_or_none(self, key: str) -> type | None:
        """Safe lookup — returns None if key is not found.

        Args:
            key: The registration key.

        Returns:
            The registered class, or None.
        """
        with self._lock:
            return self._store.get(key)

    def keys(self) -> list[str]:
        """Return a snapshot list of all registered keys."""
        with self._lock:
            return list(self._store.keys())

    def items(self) -> list[tuple[str, type]]:
        """Return a snapshot list of all (key, class) pairs."""
        with self._lock:
            return list(self._store.items())

    def children(self, prefix: str) -> dict[str, type]:
        """Return all entries whose key starts with ``prefix + separator``.

        Only meaningful when a ``separator`` is set.  Returns an empty
        dict when the separator is empty.

        Args:
            prefix: The parent key prefix (e.g. ``"openai"``).

        Returns:
            Dict of matching keys to classes.
        """
        if not self.separator:
            return {}
        full_prefix = prefix + self.separator
        with self._lock:
            return {
                k: v for k, v in self._store.items()
                if k.startswith(full_prefix)
            }

    def tree(self) -> dict:
        """Return a nested dict representing the key hierarchy.

        Only meaningful when a ``separator`` is set.  Returns a flat
        ``{key: class}`` dict when the separator is empty.

        For separator ``"."``, the key ``"openai.azure"`` produces:
        ``{"openai": {"azure": AzureOpenAI}}``

        Leaf values are classes; intermediate nodes are dicts.
        """
        with self._lock:
            items = list(self._store.items())

        if not self.separator:
            return dict(items)

        result: dict = {}
        for key, cls in items:
            parts = key.split(self.separator)
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                elif not isinstance(node[part], dict):
                    raise ValueError(
                        f"Key conflict in registry '{self.name}': "
                        f"'{part}' is registered as a leaf class "
                        f"({node[part].__name__}) but also used as a "
                        f"prefix for key '{key}'. Use distinct key "
                        f"prefixes to avoid collisions."
                    )
                node = node[part]
            node[parts[-1]] = cls
        return result

    def _dump_registry(self, file: IO[str] | None = None) -> None:
        """Debug introspection. Inspired by ABCMeta._dump_registry().

        Args:
            file: Output stream (defaults to sys.stderr).
        """
        if file is None:
            file = sys.stderr

        with self._lock:
            entries = list(self._store.items())
            check_cache = list(self._check_cache)
            neg_cache = list(self._check_negative_cache)
            neg_version = self._negative_cache_version

        print(f"Registry: {self.name}", file=file)
        print(
            f"Invalidation counter: {self._invalidation_counter}",
            file=file,
        )
        print(f"Entries ({len(entries)}):", file=file)
        for key, cls in entries:
            print(f"  {key} -> {cls.__qualname__}", file=file)
        print(
            f"Check cache: {{{', '.join(c.__qualname__ for c in check_cache)}}}",
            file=file,
        )
        print(
            f"Negative cache: {{{', '.join(c.__qualname__ for c in neg_cache)}}}",
            file=file,
        )
        print(
            f"Negative cache version: {neg_version}",
            file=file,
        )
