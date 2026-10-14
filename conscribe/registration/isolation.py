"""Test isolation for registrations: ``isolated_registries()``.

Registration is a side effect of class creation and lands in process-wide
registries, so a class defined by one test is visible to every later test.
``isolated_registries()`` scopes those side effects::

    from conscribe.testing import isolated_registries

    with isolated_registries():
        class FakeAction(ActionBase):          # registered as usual: wiring,
            __registry_key__ = "fake.done"     # config generation etc. see it
        ...
    # every registry is back to its state at entry

The block covers every registry and the global registry index (there is no
per-registry form).  Blocks nest; each one restores the state of its own
entry.  The pytest fixture ``conscribe_isolated`` (opt-in plugin
``conscribe.testing.pytest_plugin``) wraps a test in one block.

Exit rule
---------
On exit each registry holds exactly what it held on entry **plus** the
registrations made by the top-level code of modules imported for the first
time inside the block that are still in ``sys.modules`` (the *keep rule*: an
import cannot be undone, and a module that stays imported never runs again to
re-register its classes).  What counts is the net effect of those modules'
writes: when they wrote the same key several times the last write wins (so a
``@dataclass(slots=True)`` class keeps its final class), and a key such a
module removed again stays removed.  A key that existed at entry always gets
its entry class back.  The global index likewise keeps registries those
modules created (they win over an entry registry of the same name).

Everything else done inside the block is undone: classes defined in test
bodies and helper functions, ``unregister()`` / ``remove()`` /
``_deregister()`` calls, registrars created in the block (a name they
shadowed gets its original registry back), the partial registrations of an
import that raised, and the attribute writes ``register()`` made on classes
whose registration is rolled back.  The failure record of ``discover()``
(``conscribe.discover.import_failures()``) is restored to its state at entry
as well: failures recorded inside the block are dropped, failures recorded
before it stay.

What was kept and what was rolled back is listed in ``kept`` and
``rolled_back`` after exit; ``isolated_registries(warn_kept=True)`` also emits
an :class:`~conscribe.exceptions.IsolationWarning` naming the modules whose
registrations were kept (a fake-defining helper module imported lazily inside
a block leaks its fakes this way).

Module-level fakes (capture / activate)
---------------------------------------
Classes defined at module level register when the module is imported (at
pytest collection time), outside any function-scoped fixture.  Capture them
instead::

    FAKES = isolated_registries()
    with FAKES:                          # registered, recorded, rolled back
        class FakeLoop(AgentLoopBase):
            __registry_key__ = "fake_loop"

    @pytest.fixture(autouse=True)
    def _fakes():
        with FAKES.activate():           # re-applied for each test
            yield

``FAKES`` records the net difference its block made (additions, removals and
replacements of existing keys, registrars created, attribute writes);
``activate()`` re-applies all of it on enter and undoes it on exit.  Entering
``FAKES`` again re-applies what it captured so far and adds the new block's
changes to the capture; tests should use ``activate()``, which never records.

Limits
------
- Blocks must exit in LIFO order.  An out-of-order exit closes the blocks
  opened after it (restoring their entry states), restores its own entry
  state and raises ``RuntimeError``; the blocks it closed raise
  ``RuntimeError`` from their own exit.  Blocks are process-global, so
  overlapping blocks in concurrent threads or asyncio tasks are an error.
- A registration counts as import-time only when the top-level code of the
  newly imported module is on the call stack of the thread that registers.
  It is attributed to the innermost module on the stack that is registered
  in ``sys.modules`` under its name, so code that module runs through
  ``exec()`` with a namespace of its own counts as part of it.
  Registrations from other threads, and code executed outside any imported
  module (``exec()`` in a test body), are rolled back like any other.
- Only attribute writes made by ``register()`` are undone; other changes
  user code makes to classes are not.
- Classes defined inside a block stay alive while something references them
  (for example ``Base.__subclasses__()`` until garbage collection).
"""
from __future__ import annotations

import sys
import threading
import warnings
from typing import Any, Literal

from conscribe.exceptions import DuplicateKeyError, IsolationWarning
from conscribe.registration.registry import (
    _ISOLATION_STACK,
    _JOURNAL,
    _MISSING,
    _REGISTRY_INDEX,
    _REGISTRY_INDEX_LOCK,
    LayerRegistry,
    _apply_attrs,
    _Event,
    _live_registries,
    _record,
)

__all__ = ["RegistryIsolation", "isolated_registries"]

# Serialises enter/exit of isolation blocks (they rewrite shared state).
_LOCK = threading.RLock()


class _Capture:
    """Net difference recorded by a capture block, re-applied by ``activate()``.

    Attributes:
        index: ``(name, before, after)`` changes of the global index.
        bindings: ``(registry, key, before, after)`` changes of registry keys.
        attrs: ``(cls, registrations, old, new)`` attribute writes of
            ``register()`` calls whose registration was rolled back.
    ``before`` / ``after`` are relative to the state the block's exit restored;
    ``_MISSING`` stands for absent.
    """

    __slots__ = ("index", "bindings", "attrs")

    def __init__(
        self,
        index: list[tuple[str, Any, Any]],
        bindings: list[tuple[LayerRegistry, str, Any, Any]],
        attrs: list[tuple[type, tuple, dict[str, Any], dict[str, Any]]],
    ) -> None:
        self.index = index
        self.bindings = bindings
        self.attrs = attrs


class RegistryIsolation:
    """A registry isolation block; create it with :func:`isolated_registries`.

    Use it as a context manager.  On exit every registry and the global
    registry index are restored to their state at entry, except for
    registrations made by modules imported for the first time inside the
    block (see the module documentation for the exact rule).  Exceptions
    raised inside the block (``BaseException`` included) propagate after the
    restore.

    The same object also records what its block changed, so module-level fakes
    can be re-applied per test with :meth:`activate`.

    Attributes:
        kept: After exit, ``(registry_name, key, cls)`` for every registration
            the keep rule preserved (made by a module first imported inside
            the block).
        rolled_back: After exit, ``(registry_name, key, cls)`` for every
            binding present when the block ended that the exit removed or
            replaced (keys the block had removed and the exit put back are
            not listed).
    """

    def __init__(
        self,
        *,
        warn_kept: bool = False,
        _source: RegistryIsolation | None = None,
    ) -> None:
        if not isinstance(warn_kept, bool):
            raise TypeError(
                f"warn_kept must be a bool, got {type(warn_kept).__name__} "
                f"({warn_kept!r})."
            )
        self._warn_kept = warn_kept
        self._source = _source  # set on objects returned by activate()
        self._capture: _Capture | None = None
        self._open_activations = 0
        self._active = False
        self._unwound = False
        self.kept: list[tuple[str, str, type]] = []
        self.rolled_back: list[tuple[str, str, type]] = []
        self._kept_modules: list[tuple[str, str, str, str]] = []
        # Frame state, valid while active.
        self._entry_stores: dict[LayerRegistry, dict[str, type]] = {}
        self._entry_index: dict[str, LayerRegistry] = {}
        self._entry_failures: dict[str, str] | None = None
        self._modules_before: frozenset[str] = frozenset()
        self._journal_start = 0

    # ── Public API ──

    @property
    def active(self) -> bool:
        """Whether the block is currently open."""
        return self._active

    def activate(self) -> RegistryIsolation:
        """Return an isolation block that re-applies this object's capture.

        On enter the block re-applies everything this object's ``with``
        block(s) recorded: keys it added, keys it removed, keys it rebound to
        another class, registrars it created and the attribute writes of its
        ``register()`` calls.  On exit (and when re-applying fails part-way)
        the state at enter is restored like in any isolation block.

        Raises (on enter):
            RuntimeError: If nothing was captured yet, or this object's own
                block is still open.
            DuplicateKeyError: If a key the capture touches is now bound to a
                class that is neither the captured class nor the class the
                capture replaced.
        """
        if self._source is not None:
            raise RuntimeError(
                "this object is an activation of a capture; call activate() "
                "on the isolated_registries() object that recorded the capture"
            )
        return RegistryIsolation(warn_kept=self._warn_kept, _source=self)

    def __enter__(self) -> RegistryIsolation:
        with _LOCK:
            if self._active:
                raise RuntimeError(
                    "this isolated_registries() block is already active; "
                    "blocks are not re-entrant (create another one to nest)"
                )
            source = self._source
            if source is not None:
                capture = source._capture_for_activation()
            else:
                if self._open_activations:
                    raise RuntimeError(
                        "cannot record into an isolated_registries() capture "
                        "while one of its activate() blocks is open"
                    )
                capture = self._capture  # re-entering keeps and extends it
            self._open()
            if source is not None:
                source._open_activations += 1  # _close() decrements it
            if capture is not None:
                try:
                    _apply_capture(capture)
                except BaseException:
                    self._close(record=False)
                    raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        with _LOCK:
            if not self._active:
                if self._unwound:
                    self._unwound = False
                    raise RuntimeError(
                        "this isolated_registries() block was already closed: "
                        "an enclosing block exited before it (blocks must exit "
                        "in LIFO order); every registry was restored then"
                    )
                raise RuntimeError("this isolated_registries() block is not active")
            position = next(
                i for i, frame in enumerate(_ISOLATION_STACK) if frame is self
            )
            above = _ISOLATION_STACK[position + 1:]
            for frame in reversed(above):
                frame._close(record=False)
                frame._unwound = True
            self._close(record=self._source is None)
            if above:
                raise RuntimeError(
                    f"isolated_registries() blocks must exit in LIFO order: this "
                    f"block exited while {len(above)} block(s) opened after it "
                    f"were still open. They were closed first and every "
                    f"registry is restored to its state at entry of this block."
                )
        if self._warn_kept and self._kept_modules:
            warnings.warn(_kept_message(self._kept_modules), IsolationWarning, stacklevel=2)
        return False

    def __repr__(self) -> str:
        role = "activation" if self._source is not None else "isolation"
        state = "active" if self._active else "inactive"
        captured = "" if self._capture is None else f", captured={len(self._capture.bindings)}"
        return (
            f"<RegistryIsolation {role} {state}, kept={len(self.kept)}, "
            f"rolled_back={len(self.rolled_back)}{captured}>"
        )

    # ── Frames ──

    def _capture_for_activation(self) -> _Capture:
        if self._active:
            raise RuntimeError(
                "cannot activate an isolated_registries() capture while its own "
                "block is still open"
            )
        if self._capture is None:
            raise RuntimeError(
                "nothing captured yet: run the fake class definitions inside "
                "'with <this object>:' first; activate() re-applies what that "
                "block recorded"
            )
        return self._capture

    def _open(self) -> None:
        """Snapshot every live registry and the index, then push this frame."""
        stores: dict[LayerRegistry, dict[str, type]] = {}
        for registry in _live_registries():
            with registry._lock:
                stores[registry] = dict(registry._store)
        with _REGISTRY_INDEX_LOCK:
            index = dict(_REGISTRY_INDEX)
        self._entry_stores = stores
        self._entry_index = index
        self._entry_failures = _discover_failures()
        self._modules_before = frozenset(sys.modules.copy())
        self._journal_start = len(_JOURNAL)
        self.kept = []
        self.rolled_back = []
        self._kept_modules = []
        self._unwound = False
        _ISOLATION_STACK.append(self)
        self._active = True

    def _close(self, *, record: bool) -> None:
        """Restore the entry state (keep rule applied) and pop this frame.

        Must only be called on the innermost open frame.
        """
        entries = _JOURNAL[self._journal_start:]
        modules_before = self._modules_before
        modules_now = sys.modules

        def import_time(module: str | None) -> bool:
            return (
                module is not None
                and module not in modules_before
                and module in modules_now
            )

        # 1. Net effect of the import-time mutations (journal order, so the
        #    last write per key or name wins).
        import_bindings: dict[tuple[LayerRegistry, str], tuple[Any, str]] = {}
        import_index: dict[str, Any] = {}
        created: dict[LayerRegistry, bool] = {}
        for event in entries:
            if event.kind == "set":
                if import_time(event.module):
                    import_bindings[(event.target, event.key)] = (event.new, event.module)
            elif event.kind == "create":
                keep = import_time(event.module)
                created[event.target] = keep
                if keep:
                    import_index[event.key] = event.new
            elif event.kind == "index" and import_time(event.module):
                import_index[event.key] = event.new

        # 2. Target state: entry snapshot (empty for registries born in the
        #    block) plus kept import-time bindings; the entry state wins.
        targets: dict[LayerRegistry, dict[str, type]] = {
            registry: dict(snapshot) for registry, snapshot in self._entry_stores.items()
        }
        for registry in created:
            targets.setdefault(registry, {})
        kept: list[tuple[LayerRegistry, str, type, str]] = []
        for (registry, key), (cls, module) in import_bindings.items():
            if cls is _MISSING or created.get(registry) is False:
                continue  # removed again, or a registry made by the block itself
            target = targets.get(registry)
            if target is None or key in target:
                continue
            target[key] = cls
            kept.append((registry, key, cls, module))

        index_target = dict(self._entry_index)
        for name, registry in import_index.items():
            if registry is _MISSING:
                index_target.pop(name, None)
            else:
                index_target[name] = registry

        # 3. What the block leaves behind: rolled_back and the capture diff.
        current_stores: dict[LayerRegistry, dict[str, type]] = {}
        for registry in targets:
            with registry._lock:
                current_stores[registry] = dict(registry._store)
        with _REGISTRY_INDEX_LOCK:
            current_index = dict(_REGISTRY_INDEX)

        rolled_back: list[tuple[str, str, type]] = []
        bindings: list[tuple[LayerRegistry, str, Any, Any]] = []
        for registry, target in targets.items():
            current = current_stores[registry]
            for key, cls in current.items():
                if target.get(key, _MISSING) is not cls:
                    rolled_back.append((registry.name, key, cls))
            if record:
                for key in _union_keys(current, target):
                    before = target.get(key, _MISSING)
                    after = current.get(key, _MISSING)
                    if before is not after:
                        bindings.append((registry, key, before, after))
        index_changes: list[tuple[str, Any, Any]] = []
        if record:
            for name in _union_keys(current_index, index_target):
                before = index_target.get(name, _MISSING)
                after = current_index.get(name, _MISSING)
                if before is not after:
                    index_changes.append((name, before, after))

        # 4. register()'s attribute writes stay only when the keep rule kept
        #    the registration they belong to; all others are undone (step 7)
        #    and captured.  (A binding that merely exists after exit, e.g.
        #    one present at entry, does not keep writes made in the block.)
        kept_bindings = {(registry, key, cls) for registry, key, cls, _ in kept}

        def survives(event: _Event) -> bool:
            return all(
                (registry, key, event.target) in kept_bindings
                for registry, key in event.key
            )

        undone = [e for e in entries if e.kind == "attrs" and not survives(e)]
        attrs = [(e.target, e.key, e.old, e.new) for e in undone]

        # 5. Apply the targets (in place: callers may hold the dicts).
        for registry, target in targets.items():
            current = current_stores[registry]
            if _same_bindings(current, target):
                continue
            removed = any(target.get(key, _MISSING) is not cls for key, cls in current.items())
            with registry._lock:
                registry._store.clear()
                registry._store.update(target)
            if removed:
                registry._bump_invalidation()
        if not _same_bindings(current_index, index_target):
            with _REGISTRY_INDEX_LOCK:
                _REGISTRY_INDEX.clear()
                _REGISTRY_INDEX.update(index_target)

        # 6. Pop the frame; hand only what survived to the enclosing block
        #    (so it can neither resurrect what this block rolled back nor
        #    lose what this block kept).
        _ISOLATION_STACK.pop()
        if _ISOLATION_STACK:
            survivors = [
                event
                for event in entries
                if (event.kind == "create" and created.get(event.target))
                or (event.kind == "index" and import_time(event.module))
            ]
            survivors.extend(
                _Event("set", registry, key, _MISSING, cls, module)
                for registry, key, cls, module in kept
            )
            survivors.extend(e for e in entries if e.kind == "attrs" and survives(e))
            _JOURNAL[self._journal_start:] = survivors
        else:
            _JOURNAL.clear()

        self._active = False
        self._entry_stores = {}
        self._entry_index = {}
        self.kept = [(registry.name, key, cls) for registry, key, cls, _ in kept]
        self._kept_modules = [
            (module, registry.name, key, cls.__qualname__)
            for registry, key, cls, module in kept
        ]
        self.rolled_back = rolled_back
        if record:
            self._capture = _Capture(index_changes, bindings, attrs)
        if self._source is not None:
            self._source._open_activations -= 1

        # 7. Undo the attribute writes (newest first) once the registries and
        #    the frame stack are consistent, so a class refusing setattr
        #    cannot leave a half-restored block behind.
        for event in reversed(undone):
            cls = event.target
            _apply_attrs(cls, {
                name: value
                for name, value in event.old.items()
                if cls.__dict__.get(name, _MISSING) is event.new[name]
            })
        failures, self._entry_failures = self._entry_failures, None
        _restore_discover_failures(failures)


def isolated_registries(*, warn_kept: bool = False) -> RegistryIsolation:
    """Isolate every registry and the global registry index for a block.

    ::

        with isolated_registries() as iso:
            class FakeAction(ActionBase):
                __registry_key__ = "fake.done"
            build_composed_config([...])      # sees FakeAction
        # registries are back to their state at entry
        iso.rolled_back                       # [('action', 'fake.done', FakeAction)]

    Registrations made by modules imported for the first time inside the
    block are kept (they cannot be re-imported); everything else is rolled
    back.  The returned object can also capture module-level fakes for
    re-application per test with :meth:`RegistryIsolation.activate`.

    Args:
        warn_kept: Emit an :class:`~conscribe.exceptions.IsolationWarning`
            on exit that names every module whose import-time registrations
            the block kept.

    Returns:
        A :class:`RegistryIsolation` context manager.
    """
    return RegistryIsolation(warn_kept=warn_kept)


# ── Helpers ──


def _apply_capture(capture: _Capture) -> None:
    """Re-apply a capture inside the frame that was just opened."""
    for name, before, after in capture.index:
        with _REGISTRY_INDEX_LOCK:
            current = _REGISTRY_INDEX.get(name, _MISSING)
            if current is after or (after is _MISSING and current is not before):
                continue
            if after is _MISSING:
                del _REGISTRY_INDEX[name]
            else:
                _REGISTRY_INDEX[name] = after
        _record("index", None, name, current, after)

    for registry, key, before, after in capture.bindings:
        with registry._lock:
            current = registry._store.get(key, _MISSING)
            if current is after or (after is _MISSING and current is _MISSING):
                continue
            conflict = current is not _MISSING and current is not before
            if not conflict:
                if after is _MISSING:
                    del registry._store[key]
                else:
                    registry._store[key] = after
        if conflict:
            raise DuplicateKeyError(
                registry.name, key, current, after if after is not _MISSING else before,
                hint=_activation_hint(key, current, before, after),
            )
        if current is not _MISSING:
            registry._bump_invalidation()
        _record("set", registry, key, current, after)

    for cls, registrations, _old, new in capture.attrs:
        previous = {name: cls.__dict__.get(name, _MISSING) for name in new}
        _apply_attrs(cls, new)
        _record("attrs", cls, registrations, previous, new)


def _activation_hint(key: str, current: type, before: Any, after: Any) -> str:
    """Advice for an activate() conflict."""
    if after is _MISSING:
        recorded = f"removed '{key}' (then bound to '{before.__qualname__}')"
    elif before is _MISSING:
        recorded = f"added '{key}'"
    else:
        recorded = f"rebound '{key}' from '{before.__qualname__}'"
    return (
        f"The captured block {recorded}, but the key is now bound to "
        f"'{current.__qualname__}', so activate() cannot re-apply the capture. "
        f"Unregister the key before activating, or capture under another key."
    )


def _kept_message(kept: list[tuple[str, str, str, str]]) -> str:
    """Text of the warn_kept warning."""
    by_module: dict[str, list[str]] = {}
    for module, registry_name, key, qualname in kept:
        by_module.setdefault(module, []).append(f"{registry_name}:{key} ({qualname})")
    details = "; ".join(f"{module}: {', '.join(items)}" for module, items in by_module.items())
    return (
        f"isolated_registries() kept {len(kept)} registration(s) made by "
        f"modules imported for the first time inside the block: {details}. "
        f"Those modules stay imported and never run again, so their "
        f"registrations stay for the rest of the process. If they are test "
        f"fakes, import the module before the block, or capture its fakes at "
        f"module level (FAKES = isolated_registries(); with FAKES: ...) and "
        f"use FAKES.activate() in tests."
    )


def _union_keys(first: dict, second: dict) -> list:
    """Keys of ``first`` then the keys only ``second`` has (stable order)."""
    return [*first, *(key for key in second if key not in first)]


def _same_bindings(first: dict, second: dict) -> bool:
    """Whether two mappings bind the same keys to identical objects."""
    if first.keys() != second.keys():
        return False
    return all(second[key] is value for key, value in first.items())


def _discover_failures() -> dict[str, str] | None:
    """Snapshot of ``discover()``'s record of failed imports (``None``: no record)."""
    module = sys.modules.get("conscribe.discover")
    snapshot = getattr(module, "_failures_snapshot", None)
    return snapshot() if callable(snapshot) else None


def _restore_discover_failures(failures: dict[str, str] | None) -> None:
    """Put ``discover()``'s record of failed imports back to *failures*."""
    module = sys.modules.get("conscribe.discover")
    restore = getattr(module, "_restore_failures", None)
    if callable(restore):
        restore(failures or {})
