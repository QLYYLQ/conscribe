"""conscribe: Automatic class registration for layered Python architectures.

Inheritance is registration — when you write ``class FooAgent(BaseAgent)``,
the class is automatically registered in the appropriate layer registry.
"""
from __future__ import annotations

from conscribe.exceptions import (
    DuplicateKeyError,
    InvalidConfigSchemaError,
    InvalidProtocolError,
    KeyNotFoundError,
    ProtocolViolationError,
    RegistryError,
)
from conscribe.discover import discover
from conscribe.registration import (
    KeyTransform,
    LayerRegistrar,
    LayerRegistry,
    create_auto_registrar,
    create_registrar,
    default_key_transform,
    make_key_transform,
)
from conscribe.config import (
    LayerConfigResult,
    MROScope,
    build_layer_config,
    compute_registry_fingerprint,
    extract_config_schema,
    generate_layer_config_source,
    generate_layer_json_schema,
    load_cached_fingerprint,
    save_fingerprint,
)

__version__ = "0.2.0"

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
    # Config API
    "MROScope",
    "extract_config_schema",
    "build_layer_config",
    "LayerConfigResult",
    "generate_layer_config_source",
    "generate_layer_json_schema",
    "compute_registry_fingerprint",
    "load_cached_fingerprint",
    "save_fingerprint",
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
