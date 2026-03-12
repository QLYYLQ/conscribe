"""AutoRegistrar metaclass factory.

Creates metaclasses that automatically register concrete subclasses
into a LayerRegistry at class definition time.
This is the shared mechanism for Path A (inheritance) and Path B (bridge).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from layer_registry.exceptions import InvalidConfigSchemaError
from layer_registry.registration.key_transform import KeyTransform

if TYPE_CHECKING:
    from layer_registry.registration.registry import LayerRegistry


def create_auto_registrar(
    registry: LayerRegistry,
    key_transform: KeyTransform,
    *,
    base_metaclass: type = type,
) -> type:
    """Create an AutoRegistrar metaclass via closure.

    Args:
        registry: Target registry instance. All subclasses register here.
        key_transform: Key inference function (class name -> registry key).
        base_metaclass: Parent metaclass for AutoRegistrar.
            Default is ``type``. Pass ``ABCMeta`` etc. to resolve conflicts.

    Returns:
        An AutoRegistrar metaclass (subclass of base_metaclass).
    """

    class AutoRegistrar(base_metaclass):  # type: ignore[misc]
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

            # Step 3: Skip conditions
            # 3a: Root base class (no bases => direct metaclass user)
            if not bases:
                return cls

            # 3b: Explicitly abstract — MUST use namespace.get(), NOT getattr()
            #     to avoid inheriting parent's __abstract__=True
            if namespace.get("__abstract__", False):
                return cls

            # Step 4: Key inference
            # MUST use namespace.get() to avoid inheriting parent's __registry_key__
            explicit_key = namespace.get("__registry_key__")
            key = explicit_key if explicit_key is not None else key_transform(name)

            # Step 5: Register (no protocol check — inheritance guarantees compliance)
            registry.add(key, cls, protocol_check=False)
            cls.__registry_key__ = key  # type: ignore[attr-defined]

            # Step 6: Validate __config_schema__ (optional)
            config_schema = namespace.get("__config_schema__")
            if config_schema is not None:
                # Lazy import to avoid circular dependency and loading pydantic
                # when it's not needed
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
                    # Pydantic not installed — skip validation
                    if not isinstance(config_schema, type):
                        raise InvalidConfigSchemaError(name, config_schema)

            return cls

    # Debug-friendly qualname
    AutoRegistrar.__qualname__ = f"AutoRegistrar[{registry.name}]"
    AutoRegistrar.__name__ = f"AutoRegistrar[{registry.name}]"

    return AutoRegistrar
