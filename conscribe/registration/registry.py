"""LayerRegistry — thread-safe key-to-class mapping with Protocol check caching.

Each layer instantiates one LayerRegistry. It is a pure storage layer that
does not know about registration paths (metaclass/bridge/manual).
Protocol check caching borrows from CPython ABCMeta's positive/negative cache
with a global invalidation counter.
"""
from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

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


class LayerRegistry(Generic[P]):
    """Generic registry. One instance per layer.

    Internal state:
        name: layer name (for error messages)
        protocol: @runtime_checkable Protocol
        _store: dict[str, type] — strong references
        _lock: threading.Lock — thread safety (NOT RLock)
        _check_cache: WeakSet — positive Protocol check cache
        _check_negative_cache: WeakSet — negative Protocol check cache
        _invalidation_counter: int — class variable (global), incremented on remove()
        _negative_cache_version: int — instance variable, compared to global counter
    """

    _invalidation_counter: int = 0
    _counter_lock: threading.Lock = threading.Lock()

    def __init__(self, name: str, protocol: type, *, separator: str = "") -> None:
        """Initialize the registry.

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
        self._negative_cache_version: int = type(self)._invalidation_counter
        self._protocol_methods: frozenset[str] = frozenset(
            name
            for name in dir(protocol)
            if not name.startswith("_")
            and callable(getattr(protocol, name, None))
        )

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
            if self._negative_cache_version != type(self)._invalidation_counter:
                self._check_negative_cache = WeakSet()
                self._negative_cache_version = type(self)._invalidation_counter

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
            DuplicateKeyError: If key already exists in this registry.
            ProtocolViolationError: If protocol_check=True and cls fails check.
        """
        if protocol_check:
            self.runtime_check(cls)

        with self._lock:
            if key in self._store:
                existing = self._store[key]
                raise DuplicateKeyError(self.name, key, existing, cls)
            self._store[key] = cls

    def remove(self, key: str) -> None:
        """Remove a registration. Intended for test cleanup.

        Args:
            key: The key to remove.

        Raises:
            KeyError: If key does not exist.
        """
        with self._lock:
            del self._store[key]  # KeyError if missing
        with type(self)._counter_lock:
            type(self)._invalidation_counter += 1

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

    def get_or_none(self, key: str) -> Optional[type]:
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
                node = node[part]
            node[parts[-1]] = cls
        return result

    def _dump_registry(self, file: Optional[IO[str]] = None) -> None:
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
            f"Invalidation counter: {type(self)._invalidation_counter}",
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
