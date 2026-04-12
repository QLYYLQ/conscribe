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
    WiringResolutionError,
)
from conscribe.discover import discover
from conscribe.registration import (
    AutoRegistrarBase,
    KeyTransform,
    LayerRegistrar,
    LayerRegistry,
    create_auto_registrar,
    create_registrar,
    default_key_transform,
    make_key_transform,
)
from conscribe.registration.registry import get_registry
from conscribe.config import (
    DegradedField,
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

__version__ = "0.7.0"

__all__ = [
    # Registration API
    "AutoRegistrarBase",
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
    "DegradedField",
    "generate_layer_config_source",
    "generate_layer_json_schema",
    "compute_registry_fingerprint",
    "load_cached_fingerprint",
    "save_fingerprint",
    # Stubs API (import from conscribe.stubs)
    # write_layer_stubs, collect_class_stub_info, generate_module_stub,
    # narrowest_common_base, ClassStubInfo, InjectedAttr
    # Wiring API
    "get_registry",
    "WiringResolutionError",
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
