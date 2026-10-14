"""Module discovery: recursively import packages to trigger registration.

Python metaclass registration depends on modules being imported.
``discover()`` ensures all implementation modules are imported,
triggering class definitions and filling registries.

Failure policy
--------------

By default (``strict=True``) any exception raised while importing a module
propagates — a broken module would otherwise surface much later as a
confusing "unknown key" or "empty registry" wiring error.  ``allow_missing``
names optional dependencies whose absence is planned (those modules are
skipped quietly), ``exclude`` names modules that are never imported, and
``strict=False`` restores the 1.x skip-and-continue behaviour for
``ImportError`` / ``SyntaxError`` with an
:class:`~conscribe.exceptions.ExtractionWarning` per skipped module.

Rollback
--------

While ``discover()`` imports a module for the first time it records every
registry change, attributed to the module whose top-level code made it (the
innermost module being executed, whatever ``__module__`` the class claims).
Afterwards the changes made by modules whose import was attempted and failed
-- the modules that are not in ``sys.modules`` any more -- are undone, also
when the failure was a ``BaseException`` (``KeyboardInterrupt``,
``SystemExit``).  Registrations of modules that imported successfully are
never touched, and a module that was already imported costs no recording or
rollback work at all.  Every failure is recorded (as a string) so that wiring
errors can name the module; see :func:`import_failures` and
:func:`clear_import_failures`.

Note that the package attribute ``conscribe.discover`` is the
:func:`discover` *function*; reach the helpers of this module with
``from conscribe.discover import import_failures`` (or through
``sys.modules["conscribe.discover"]``).
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
import threading
import warnings
from collections.abc import Iterable
from types import ModuleType
from typing import Any

from conscribe.exceptions import ExtractionWarning

__all__ = ["clear_import_failures", "discover", "import_failures"]

logger = logging.getLogger(__name__)

# Modules discover() could not import -> "ExcType: message".  Strings only:
# exception objects would keep their tracebacks (and every frame) alive.
_IMPORT_FAILURES: dict[str, str] = {}
_IMPORT_FAILURES_LOCK = threading.Lock()

#: Exception types ``strict=False`` skips (the 1.x behaviour); anything
#: else always propagates.
_LEGACY_SKIPPABLE = (ImportError, SyntaxError)


def import_failures() -> dict[str, str]:
    """Return the modules ``discover()`` could not import.

    The record maps each module name to ``"ExcType: message"``.  It covers
    modules that raised in strict mode, modules skipped by ``strict=False``
    or ``allow_missing`` (marked as such), and modules whose import failed
    inside another module's import while ``discover()`` was running.  A
    module that has been imported since (by ``discover()`` or otherwise) is
    no longer reported.  Wiring error messages cite this record when a
    registry or key is missing.

    Returns:
        A copy of the record (mutating it has no effect).
    """
    with _IMPORT_FAILURES_LOCK:
        for name in [n for n in _IMPORT_FAILURES if n in sys.modules]:
            del _IMPORT_FAILURES[name]
        return dict(_IMPORT_FAILURES)


def clear_import_failures() -> None:
    """Forget every import failure recorded by ``discover()``.

    Tests that call ``discover()`` on deliberately broken packages can call
    this so that their failures never show up in another test's wiring error
    messages.  (``conscribe.testing.isolated_registries()`` restores the
    record to its state at entry instead.)
    """
    with _IMPORT_FAILURES_LOCK:
        _IMPORT_FAILURES.clear()


def _failures_snapshot() -> dict[str, str]:
    """The raw failure record (for ``isolated_registries()`` to restore)."""
    with _IMPORT_FAILURES_LOCK:
        return dict(_IMPORT_FAILURES)


def _restore_failures(snapshot: dict[str, str]) -> None:
    """Replace the failure record with *snapshot* (taken by ``_failures_snapshot``)."""
    with _IMPORT_FAILURES_LOCK:
        _IMPORT_FAILURES.clear()
        _IMPORT_FAILURES.update(snapshot)


def _record_failure(name: str, description: str, *, overwrite: bool = True) -> None:
    with _IMPORT_FAILURES_LOCK:
        if overwrite or name not in _IMPORT_FAILURES:
            _IMPORT_FAILURES[name] = description


def _forget_failure(name: str) -> None:
    with _IMPORT_FAILURES_LOCK:
        _IMPORT_FAILURES.pop(name, None)


def discover(
    *package_paths: str,
    strict: bool = True,
    allow_missing: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[str]:
    """Recursively import all modules under the given packages.

    This triggers import-time registration for any classes using
    AutoRegistrar metaclasses or ``__init_subclass__`` hooks.  Each package
    is imported first, then its modules and subpackages depth-first, in
    name order.  ``__main__`` modules are never imported by the walk (that
    would run a command-line entry point); name one explicitly in
    *package_paths* to import it anyway.

    Args:
        *package_paths: Dotted package paths to discover
            (e.g. ``"my_app.agents"``, ``"my_app.llm_providers"``).  A plain
            module is imported and not walked.
        strict: If ``True`` (default), any exception raised while importing
            a module propagates as the original exception.  On Python 3.11+
            a PEP 678 note naming the module is attached; on Python 3.10,
            which has no notes, a failing *submodule* is re-raised as an
            ``ImportError`` naming it, chained from the original exception
            (``__cause__``).  If ``False``, a submodule raising
            ``ImportError`` or ``SyntaxError`` is skipped with an
            :class:`~conscribe.exceptions.ExtractionWarning` naming the
            exception (any other exception still propagates, as in 1.x).
            A ``BaseException`` that is not an ``Exception``
            (``KeyboardInterrupt``, ``SystemExit``) always propagates
            unchanged, after the rollback.
        allow_missing: Top-level module names of optional dependencies.  A
            module whose import fails with a ``ModuleNotFoundError`` for one
            of them (or for a submodule of one) is skipped, with only a
            ``logging.info`` record.  The ``ModuleNotFoundError`` may also be
            reached through ``__cause__`` / ``__context__`` links, but only
            from one ``ImportError`` to the next (``raise ImportError("install
            the extra") from exc``): an error of another type raised while
            handling the missing import (a bug in a fallback, say) is a real
            failure and is not skipped.  Applies in both modes and to the
            packages named in *package_paths*.
        exclude: Module or package names that are never imported; a name
            excludes itself and everything below it (``"pkg.generated"``
            excludes ``pkg.generated`` and ``pkg.generated.x``, not
            ``pkg.generated_old``).

    Returns:
        List of all successfully imported module names (including modules
        that were already imported before the call).

    Raises:
        ModuleNotFoundError: If a package in *package_paths* does not exist
            (unless it is listed in *allow_missing*).
        Exception: In strict mode, whatever a module raised while being
            imported (see *strict*).  The top-level packages always fail
            fast, whatever *strict* says.
        TypeError: If *allow_missing* or *exclude* contain anything but
            non-empty module-name strings.

    Notes:
        - Registrations made by the top-level code of a module whose import
          failed are removed from their registries again; registrations of
          modules imported successfully along the way are kept, whatever
          ``__module__`` their classes claim.  Modules that were already
          imported are not re-imported and cost no rollback bookkeeping.
        - Every failure is recorded by module name; see
          :func:`import_failures`.
    """
    allowed = _module_names(allow_missing, "allow_missing")
    excluded = _module_names(exclude, "exclude")
    imported: list[str] = []

    for pkg_path in package_paths:
        if _matches(pkg_path, excluded):
            logger.debug("discover(): %r is excluded", pkg_path)
            continue

        # The packages the caller named always fail fast (strict or not).
        package, _ = _import_module(pkg_path, strict=True, allowed=allowed, top_level=True)
        if package is None:
            continue  # optional dependency missing, allowed by allow_missing
        imported.append(pkg_path)

        _walk_package(
            package,
            strict=strict,
            allowed=allowed,
            excluded=excluded,
            imported=imported,
            seen_paths=set(),
            stacklevel=3,
        )

    return imported


# ── Walking ────────────────────────────────────────────────────


def _walk_package(
    package: ModuleType,
    *,
    strict: bool,
    allowed: tuple[str, ...],
    excluded: tuple[str, ...],
    imported: list[str],
    seen_paths: set[str],
    stacklevel: int,
) -> None:
    """Import every module below *package*, depth-first in name order.

    ``stacklevel`` makes skip warnings point at the caller of ``discover()``
    (it grows by one per recursion level).
    """
    search_path = [p for p in (getattr(package, "__path__", None) or ()) if p not in seen_paths]
    if not search_path:
        return  # a plain module, or path entries already walked
    seen_paths.update(search_path)

    for info in pkgutil.iter_modules(search_path, prefix=package.__name__ + "."):
        name = info.name
        if name.rpartition(".")[2] == "__main__":
            # Importing pkg.__main__ runs the package's command-line entry
            # point (argument parsing, sys.exit()); it is never a plugin.
            logger.debug("discover(): %r is a __main__ module, not imported", name)
            continue
        if _matches(name, excluded):
            logger.debug("discover(): %r is excluded", name)
            continue

        module, skipped = _import_module(name, strict=strict, allowed=allowed, top_level=False)
        if module is None:
            if skipped is not None:
                warnings.warn(
                    f"discover() skipped module {name!r} because importing it raised "
                    f"{skipped}. Classes defined in it are not registered; fix the "
                    f"module or exclude it with discover(..., exclude=[{name!r}]).",
                    ExtractionWarning,
                    stacklevel=stacklevel,
                )
            continue

        imported.append(name)
        if info.ispkg:
            _walk_package(
                module,
                strict=strict,
                allowed=allowed,
                excluded=excluded,
                imported=imported,
                seen_paths=seen_paths,
                stacklevel=stacklevel + 1,
            )


def _import_module(
    name: str,
    *,
    strict: bool,
    allowed: tuple[str, ...],
    top_level: bool,
) -> tuple[ModuleType | None, str | None]:
    """Import one module with rollback, failure recording and the skip policy.

    Returns:
        ``(module, None)`` on success, ``(None, None)`` when skipped through
        *allowed*, ``(None, description)`` when skipped by ``strict=False``.
        Every other failure raises.
    """
    if sys.modules.get(name) is not None:
        # Already imported: importing it again runs nothing, so there is
        # nothing to record or roll back.
        module = importlib.import_module(name)
        _forget_failure(name)
        return module, None

    from conscribe.registration.registry import _recording

    try:
        with _recording() as events:
            module = importlib.import_module(name)
    except BaseException as exc:
        # Undo what the modules that failed registered; modules imported
        # successfully along the way stay in sys.modules and never run
        # again, so their registrations are kept.
        _roll_back(events)
        description = _describe(exc)
        if not isinstance(exc, Exception):
            # KeyboardInterrupt, SystemExit, ...: rolled back and recorded,
            # never wrapped, skipped or annotated.
            _record_failure(name, description)
            raise

        missing = _allowed_missing_module(exc, allowed)
        if missing is not None:
            _record_failure(name, f"{description} (skipped: {missing!r} is in allow_missing)")
            logger.info(
                "discover(): skipped module %r: %s (allowed by allow_missing)",
                name,
                description,
            )
            return None, None

        _record_failure(name, description)
        if not strict and not top_level and isinstance(exc, _LEGACY_SKIPPABLE):
            return None, description

        if sys.version_info >= (3, 11):
            exc.add_note(f"(raised while conscribe.discover() was importing module {name!r})")
            raise
        if top_level:
            raise  # the caller named this module; nothing to add on 3.10
        raise ImportError(
            f"conscribe.discover() could not import module {name!r}: {description}",
            name=name,
        ) from exc

    # A module may import another one that fails and handle that error
    # itself; the failed module's registrations are leftovers too.
    for module_name in _roll_back(events):
        _record_failure(
            module_name,
            f"import failed inside the import of {name!r}, which handled the error; "
            f"its registrations were rolled back",
            overwrite=False,
        )
    _forget_failure(name)
    return module, None


# ── Rollback ───────────────────────────────────────────────────


def _roll_back(events: list[Any]) -> list[str]:
    """Undo the registry changes recorded for modules whose import failed.

    A module whose import was attempted and failed is no longer in
    ``sys.modules`` (the import system removes it); its changes are undone
    newest first.  Changes attributed to modules that are still imported,
    or to no module at all, are never touched.

    Returns:
        The names of the failed modules whose changes were undone, sorted.
    """
    modules = sys.modules
    failed = {
        event.module
        for event in events
        if event.kind == "set" and event.module is not None and event.module not in modules
    }
    if not failed:
        return []
    for event in reversed(events):
        if event.kind != "set" or event.module not in failed:
            continue
        if event.target._restore_binding(event.key, event.new, event.old):
            logger.debug(
                "discover(): rolled back %r in registry %r (module %r failed to import)",
                event.key,
                event.target.name,
                event.module,
            )
    return sorted(failed)


# ── Small helpers ──────────────────────────────────────────────


def _module_names(value: Iterable[str], param: str) -> tuple[str, ...]:
    """Normalise *allow_missing* / *exclude* to a tuple of module names."""
    names = (value,) if isinstance(value, str) else tuple(value)
    for item in names:
        if not isinstance(item, str) or not item:
            raise TypeError(
                f"discover({param}=...) expects module names (non-empty str), got {item!r}"
            )
    return names


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    """``True`` if *name* equals a prefix or lies below one (dotted)."""
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _allowed_missing_module(exc: BaseException, allowed: tuple[str, ...]) -> str | None:
    """The missing module's name if *exc* is an allowed ``ModuleNotFoundError``.

    The ``ModuleNotFoundError`` may be *exc* itself or be reached through
    ``__cause__`` / ``__context__`` links, but only from one ``ImportError``
    to the next: an exception of another type ends the search, so a bug
    raised while handling an optional import is never mistaken for the
    missing dependency.
    """
    if not allowed:
        return None
    seen: set[int] = set()
    queue: list[BaseException | None] = [exc]
    while queue:
        current = queue.pop(0)
        if not isinstance(current, ImportError) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError) and isinstance(current.name, str):
            if _matches(current.name, allowed):
                return current.name
        queue.append(current.__cause__)
        queue.append(current.__context__)
    return None


def _describe(exc: BaseException) -> str:
    """``"ExcType: message"`` on one line."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
