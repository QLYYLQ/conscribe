"""layer-registry: Automatic class registration for layered Python architectures.

Inheritance is registration — when you write ``class FooAgent(BaseAgent)``,
the class is automatically registered in the appropriate layer registry.
"""
from __future__ import annotations

from layer_registry.exceptions import (
    DuplicateKeyError,
    InvalidConfigSchemaError,
    InvalidProtocolError,
    KeyNotFoundError,
    ProtocolViolationError,
    RegistryError,
)
from layer_registry.registration import (
    KeyTransform,
    LayerRegistrar,
    LayerRegistry,
    create_auto_registrar,
    create_registrar,
    default_key_transform,
    discover,
    make_key_transform,
)

__version__ = "0.1.0"

__all__ = [
    # Registration API
    "create_registrar",
    "create_auto_registrar",
    "LayerRegistrar",
    "LayerRegistry",
    "KeyTransform",
    "default_key_transform",
    "make_key_transform",
    "discover",
    # Exceptions
    "RegistryError",
    "DuplicateKeyError",
    "KeyNotFoundError",
    "ProtocolViolationError",
    "InvalidConfigSchemaError",
    "InvalidProtocolError",
    # Version
    "__version__",
]
