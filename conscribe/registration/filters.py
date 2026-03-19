"""Predicate-based registration filter system.

Replaces scattered if/else skip logic with composable filter objects.
Each filter implements ``should_skip(ctx)`` — returning True means
the class should NOT be registered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class RegistrationContext:
    """Context passed to registration filters.

    Attributes:
        cls: The class being registered.
        name: The class name.
        bases: The base classes tuple.
        namespace: The class ``__dict__`` (for metaclass path) or
            ``__dict__`` (for ``__init_subclass__`` path).
        registry_name: Name of the target registry.
    """

    cls: type
    name: str
    bases: tuple
    namespace: dict
    registry_name: str


class RegistrationFilter(Protocol):
    """Protocol for registration filters."""

    def should_skip(self, ctx: RegistrationContext) -> bool:
        """Return True if the class should NOT be registered."""
        ...


class RootFilter:
    """Skip root base classes (no bases = direct metaclass user)."""

    def should_skip(self, ctx: RegistrationContext) -> bool:
        return not ctx.bases


class PydanticGenericFilter:
    """Skip Pydantic Generic specializations (e.g. ``BaseEvent[str]``)."""

    def should_skip(self, ctx: RegistrationContext) -> bool:
        return "[" in ctx.name


class AbstractFilter:
    """Skip explicitly abstract classes.

    Uses ``namespace.get()`` (NOT ``getattr()``) to avoid inheriting
    parent's ``__abstract__ = True``.
    """

    def should_skip(self, ctx: RegistrationContext) -> bool:
        return bool(ctx.namespace.get("__abstract__", False))


class CustomCallableFilter:
    """Skip classes rejected by a user-provided callable."""

    def __init__(self, fn: Callable[[type], bool]) -> None:
        self._fn = fn

    def should_skip(self, ctx: RegistrationContext) -> bool:
        return self._fn(ctx.cls)


class ChildSkipFilter:
    """Skip registration if the class declares ``__skip_registries__``
    containing this registry's name."""

    def should_skip(self, ctx: RegistrationContext) -> bool:
        skip_list = ctx.namespace.get("__skip_registries__")
        if skip_list is None:
            return False
        return ctx.registry_name in skip_list


class ParentRegistrationFilter:
    """Check ``__registration_filter__`` on parent classes.

    If any parent defines ``__registration_filter__`` and it returns
    ``False`` for the child class, the child is skipped.
    """

    def should_skip(self, ctx: RegistrationContext) -> bool:
        for base in ctx.bases:
            reg_filter = getattr(base, "__registration_filter__", None)
            if reg_filter is not None:
                if not reg_filter(ctx.cls):
                    return True
        return False


class PropagationFilter:
    """Check ``__propagate__`` and ``__propagate_depth__`` on parents.

    If a parent sets ``__propagate__ = False``, subclasses through
    that parent don't auto-register.

    If a parent sets ``__propagate_depth__ = N``, only N levels of
    subclasses auto-register through that parent. Depth is measured
    from the class that **originally defined** ``__propagate_depth__``
    in its own ``__dict__``.
    """

    def should_skip(self, ctx: RegistrationContext) -> bool:
        for base in ctx.bases:
            # Check __propagate__ = False (use __dict__ to find originator)
            if getattr(base, "__propagate__", True) is False:
                return True

            # Check __propagate_depth__
            max_depth = getattr(base, "__propagate_depth__", None)
            if max_depth is not None:
                # Find the originator of __propagate_depth__ in the MRO
                originator = _find_attr_definer(ctx.cls, "__propagate_depth__")
                if originator is not None:
                    depth = _compute_depth(ctx.cls, originator)
                    if depth > max_depth:
                        return True

        return False


def _find_attr_definer(cls: type, attr: str) -> type | None:
    """Find the class in the MRO that defines ``attr`` in its own ``__dict__``."""
    for klass in cls.__mro__:
        if attr in klass.__dict__:
            return klass
    return None


def _compute_depth(cls: type, ancestor: type) -> int:
    """Compute the inheritance depth from cls to ancestor.

    Returns the number of generations between cls and ancestor.
    E.g., if cls is a direct subclass of ancestor, returns 1.
    """
    try:
        return cls.__mro__.index(ancestor)
    except ValueError:
        return 999


def build_filter_chain(
    *,
    skip_pydantic_generic: bool = True,
    skip_filter: Callable[[type], bool] | None = None,
    registry_name: str = "",
    include_propagation: bool = True,
) -> list[Any]:
    """Build the standard filter chain.

    Args:
        skip_pydantic_generic: Whether to include the Pydantic Generic filter.
        skip_filter: Optional custom callable filter.
        registry_name: Name of the target registry (for ChildSkipFilter).
        include_propagation: Whether to include propagation/parent filters.

    Returns:
        A list of filter objects in evaluation order.
    """
    filters: list[Any] = [RootFilter()]

    if skip_pydantic_generic:
        filters.append(PydanticGenericFilter())

    if skip_filter is not None:
        filters.append(CustomCallableFilter(skip_filter))

    filters.append(AbstractFilter())
    filters.append(ChildSkipFilter())

    if include_propagation:
        filters.append(ParentRegistrationFilter())
        filters.append(PropagationFilter())

    return filters


def should_skip_registration(
    filters: Sequence[Any],
    ctx: RegistrationContext,
) -> bool:
    """Run the filter chain. Returns True if the class should be skipped."""
    return any(f.should_skip(ctx) for f in filters)
