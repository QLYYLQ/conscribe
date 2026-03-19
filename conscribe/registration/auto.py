"""AutoRegistrar metaclass factory.

Creates metaclasses that automatically register concrete subclasses
into a LayerRegistry at class definition time.
This is the shared mechanism for Path A (inheritance) and Path B (bridge).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from conscribe.exceptions import InvalidConfigSchemaError
from conscribe.registration.filters import (
    RegistrationContext,
    build_filter_chain,
    should_skip_registration,
)
from conscribe.registration.key_transform import KeyTransform
from conscribe.registration.meta_base import AutoRegistrarBase, MetaRegistrarType

if TYPE_CHECKING:
    from conscribe.registration.registry import LayerRegistry


def create_auto_registrar(
    registry: LayerRegistry,
    key_transform: KeyTransform,
    *,
    base_metaclass: type = type,
    skip_pydantic_generic: bool = True,
    skip_filter: Optional[Callable[[type], bool]] = None,
    key_separator: str = "",
) -> type:
    """Create an AutoRegistrar metaclass via closure.

    Args:
        registry: Target registry instance. All subclasses register here.
        key_transform: Key inference function (class name -> registry key).
        base_metaclass: Parent metaclass for AutoRegistrar.
            Default is ``type``. Pass ``ABCMeta`` etc. to resolve conflicts.
        skip_pydantic_generic: If True (default), skip classes whose name
            contains ``[`` — these are Pydantic Generic specialization
            intermediates (e.g. ``BaseEvent[str]``) and should not be
            registered.
        skip_filter: Optional callable ``(cls) -> bool``. If it returns
            True for a class, that class is skipped (not registered).
        key_separator: Separator for hierarchical keys (e.g. ``"."``).
            Empty string means flat keys (backward compatible).

    Returns:
        An AutoRegistrar metaclass (subclass of base_metaclass).
    """
    # Ensure AutoRegistrarBase is in the metaclass chain
    if not issubclass(base_metaclass, AutoRegistrarBase):
        if base_metaclass is type:
            effective_base = AutoRegistrarBase
        else:
            # Combined metaclass: base_metaclass + AutoRegistrarBase
            # Use MetaRegistrarType so the | operator works
            effective_base = MetaRegistrarType(
                f"_{base_metaclass.__name__}AutoBase",
                (AutoRegistrarBase, base_metaclass),
                {},
            )
    else:
        effective_base = base_metaclass

    filters = build_filter_chain(
        skip_pydantic_generic=skip_pydantic_generic,
        skip_filter=skip_filter,
        registry_name=registry.name,
    )

    class AutoRegistrar(effective_base):  # type: ignore[misc]
        """Metaclass that auto-registers concrete subclasses."""

        def __new__(
            mcs,
            name: str,
            bases: tuple,
            namespace: dict,
            **kwargs: object,
        ) -> type:
            # Step 1: Create the class — pass **kwargs through!
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)

            # Step 2: Tag with layer name
            cls.__registry_name__ = registry.name  # type: ignore[attr-defined]

            # Step 3: Skip conditions via filter chain
            ctx = RegistrationContext(
                cls=cls,
                name=name,
                bases=bases,
                namespace=namespace,
                registry_name=registry.name,
            )
            if should_skip_registration(filters, ctx):
                return cls

            # Step 4: Key inference (supports hierarchical + multi-key)
            explicit_key = namespace.get("__registry_key__")
            keys = _resolve_keys(
                explicit_key, name, cls, registry, key_transform, key_separator,
            )

            # Step 5: Register
            for key in keys:
                registry.add(key, cls, protocol_check=False)

            # Step 6: Set key attributes
            if len(keys) == 1:
                cls.__registry_key__ = keys[0]  # type: ignore[attr-defined]
            else:
                cls.__registry_key__ = keys[0]  # type: ignore[attr-defined]
                cls.__registry_keys__ = keys  # type: ignore[attr-defined]

            # Step 7: Validate __config_schema__ (optional)
            config_schema = namespace.get("__config_schema__")
            if config_schema is not None:
                _validate_config_schema(name, config_schema)

            return cls

    # Debug-friendly qualname
    AutoRegistrar.__qualname__ = f"AutoRegistrar[{registry.name}]"
    AutoRegistrar.__name__ = f"AutoRegistrar[{registry.name}]"

    return AutoRegistrar


def _resolve_keys(
    explicit_key: object,
    name: str,
    cls: type,
    registry: LayerRegistry,
    key_transform: KeyTransform,
    key_separator: str,
) -> list[str]:
    """Resolve registration keys from explicit key, name, or hierarchy.

    Supports:
    - ``__registry_key__ = "foo"`` — single explicit key
    - ``__registry_key__ = ["foo", "bar"]`` — multi-key registration
    - Hierarchical key derivation when ``key_separator`` is set
    - Default: ``key_transform(name)``
    """
    if explicit_key is not None:
        if isinstance(explicit_key, list):
            return explicit_key
        return [str(explicit_key)]

    if key_separator:
        derived = _derive_hierarchical_key(
            cls, name, registry, key_transform, key_separator,
        )
        if derived is not None:
            return [derived]

    return [key_transform(name)]


def _derive_hierarchical_key(
    cls: type,
    name: str,
    registry: LayerRegistry,
    key_transform: KeyTransform,
    separator: str,
) -> str | None:
    """Derive a hierarchical key from the first parent with a key in MRO.

    Finds the nearest parent that has ``__registry_key__`` set (whether
    the parent is registered or abstract). This allows abstract parents
    with explicit keys to serve as hierarchy roots.
    """
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        parent_key = base.__dict__.get("__registry_key__", None)
        if parent_key and isinstance(parent_key, str):
            return f"{parent_key}{separator}{key_transform(name)}"
    return None


def _validate_config_schema(name: str, config_schema: object) -> None:
    """Validate that __config_schema__ is a BaseModel subclass."""
    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = None  # type: ignore[assignment,misc]

    if BaseModel is not None:
        if not (
            isinstance(config_schema, type)
            and issubclass(config_schema, BaseModel)
        ):
            raise InvalidConfigSchemaError(name, config_schema)
    else:
        if not isinstance(config_schema, type):
            raise InvalidConfigSchemaError(name, config_schema)
