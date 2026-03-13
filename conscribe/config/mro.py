"""MRO parameter collection for ``**kwargs`` chains.

When a child class's ``__init__`` accepts ``**kwargs`` and passes them
to ``super().__init__(**kwargs)``, this module walks the MRO upwards to
collect parent parameters, producing a complete config schema.

Scope controls which classes in the MRO are included:
- ``"local"``: only classes NOT in site-packages (default)
- ``"third_party"``: include site-packages, exclude stdlib
- ``"all"``: all classes except ``object``
"""
from __future__ import annotations

import inspect
import sysconfig
from dataclasses import dataclass, field
from typing import Any, Literal, Union

from conscribe.config._utils import find_init_definer

MROScope = Literal["local", "third_party", "all"]


@dataclass(frozen=True)
class MROCollectionResult:
    """Result of collecting parameters from the MRO chain.

    Attributes:
        params: Parent-class parameters collected (child-first order, deduped).
        init_definers: Classes that contributed parameters.
        hints: Merged type hints from all contributing classes.
        fully_resolved: True if the chain terminated naturally (a parent
            had no ``**kwargs``), False if truncated by scope/depth.
    """

    params: list[inspect.Parameter] = field(default_factory=list)
    init_definers: list[type] = field(default_factory=list)
    hints: dict[str, Any] = field(default_factory=dict)
    fully_resolved: bool = True


def classify_class_scope(cls: type) -> Literal["local", "third_party", "stdlib"]:
    """Classify a class as local, third-party, or stdlib.

    Uses ``inspect.getfile()`` and ``sysconfig.get_paths()`` to
    determine origin.
    """
    try:
        source_file = inspect.getfile(cls)
    except (TypeError, OSError):
        # Built-in / C-extension classes have no file -> stdlib
        return "stdlib"

    # Check site-packages FIRST (more specific than stdlib paths,
    # which may be a prefix of site-packages dirs).
    if "site-packages" in source_file:
        return "third_party"

    # Check stdlib paths
    stdlib_paths = _get_stdlib_paths()
    for p in stdlib_paths:
        if source_file.startswith(p):
            return "stdlib"

    return "local"


def collect_mro_params(
    cls: type,
    scope: MROScope = "local",
    depth: Union[int, None] = None,
) -> MROCollectionResult:
    """Collect parameters from parent classes along the MRO ``**kwargs`` chain.

    Starting from the class that defines ``cls``'s ``__init__``, checks
    whether it accepts ``**kwargs``.  If so, walks upward through the MRO
    collecting named parameters from each parent that defines its own
    ``__init__``, stopping when:

    - A parent's ``__init__`` does NOT have ``**kwargs`` (natural termination).
    - A parent is excluded by *scope*.
    - The *depth* limit is reached.
    - ``object`` is reached.

    Parameters already seen (by name) are skipped — child wins.

    Args:
        cls: The class whose MRO to walk.
        scope: Which classes to include (``"local"``, ``"third_party"``,
            or ``"all"``).
        depth: Maximum number of MRO levels to traverse.  ``None`` means
            unlimited.  ``0`` disables MRO traversal entirely.

    Returns:
        An ``MROCollectionResult`` with collected parameters and metadata.
    """
    if depth is not None and depth == 0:
        return MROCollectionResult()

    init_definer = find_init_definer(cls)
    if init_definer is None:
        return MROCollectionResult()

    # Check if init_definer's __init__ has **kwargs
    if not _has_var_keyword(init_definer):
        return MROCollectionResult()

    # Collect the named params from init_definer itself to know which
    # names to skip (child's own params take precedence).
    own_param_names = _get_named_param_names(init_definer)

    # Find init_definer's position in the MRO and walk upward
    mro = cls.__mro__
    try:
        start_idx = mro.index(init_definer)
    except ValueError:
        return MROCollectionResult()

    collected_params: list[inspect.Parameter] = []
    collected_hints: dict[str, Any] = {}
    init_definers: list[type] = []
    seen_names: set[str] = set(own_param_names)
    fully_resolved = False
    levels_traversed = 0

    for klass in mro[start_idx + 1 :]:
        if klass is object:
            # Reached object — natural end
            fully_resolved = True
            break

        # Check scope
        if not _should_include_class(klass, scope):
            fully_resolved = False
            break

        # Check depth
        if depth is not None and levels_traversed >= depth:
            fully_resolved = False
            break

        # Only process classes that define their own __init__
        if "__init__" not in klass.__dict__:
            continue

        levels_traversed += 1

        # Collect named parameters from this class
        params, hints = _extract_class_params(klass)
        class_contributed = False

        for param in params:
            if param.name not in seen_names:
                collected_params.append(param)
                seen_names.add(param.name)
                class_contributed = True

        # Merge hints (child overrides)
        for name, hint in hints.items():
            if name not in collected_hints and name not in own_param_names:
                collected_hints[name] = hint

        if class_contributed:
            init_definers.append(klass)

        # Check if this parent also has **kwargs — if not, chain ends
        if not _has_var_keyword(klass):
            fully_resolved = True
            break

    else:
        # Exhausted MRO without hitting object (unusual but possible)
        fully_resolved = True

    return MROCollectionResult(
        params=collected_params,
        init_definers=init_definers,
        hints=collected_hints,
        fully_resolved=fully_resolved,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_var_keyword(cls: type) -> bool:
    """Check if a class's own ``__init__`` has a ``**kwargs`` parameter."""
    init = cls.__dict__.get("__init__")
    if init is None:
        return False
    try:
        sig = inspect.signature(init)
    except (ValueError, TypeError):
        return False
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


def _get_named_param_names(cls: type) -> set[str]:
    """Get the names of all named (non-VAR) parameters from a class's __init__."""
    init = cls.__dict__.get("__init__")
    if init is None:
        return set()
    try:
        sig = inspect.signature(init)
    except (ValueError, TypeError):
        return set()
    return {
        p.name
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _extract_class_params(
    cls: type,
) -> tuple[list[inspect.Parameter], dict[str, Any]]:
    """Extract named parameters and type hints from a class's own ``__init__``.

    Returns:
        A tuple of (named_params, type_hints).
    """
    init = cls.__dict__.get("__init__")
    if init is None:
        return [], {}

    try:
        sig = inspect.signature(init)
    except (ValueError, TypeError):
        return [], {}

    params = [
        p
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # Get type hints
    hints: dict[str, Any] = {}
    try:
        from typing import get_type_hints

        raw_hints = get_type_hints(init, include_extras=True)
        hints = {k: v for k, v in raw_hints.items() if k != "return"}
    except Exception:
        # Fall back to raw annotations
        raw = getattr(init, "__annotations__", {})
        hints = {k: v for k, v in raw.items() if k != "return"}

    return params, hints


def _should_include_class(cls: type, scope: MROScope) -> bool:
    """Check whether a class should be included given the scope."""
    if scope == "all":
        return True

    cls_scope = classify_class_scope(cls)

    if cls_scope == "stdlib":
        return False

    if scope == "local":
        return cls_scope == "local"

    # scope == "third_party": include local + third_party
    return cls_scope in ("local", "third_party")


_stdlib_paths_cache: Union[list[str], None] = None


def _get_stdlib_paths() -> list[str]:
    """Get stdlib directory paths, cached for performance."""
    global _stdlib_paths_cache  # noqa: PLW0603
    if _stdlib_paths_cache is None:
        paths = sysconfig.get_paths()
        _stdlib_paths_cache = [
            p
            for key in ("stdlib", "platstdlib", "purelib")
            if (p := paths.get(key))
            and "site-packages" not in p
        ]
    return _stdlib_paths_cache
