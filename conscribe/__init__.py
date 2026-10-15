"""conscribe: Automatic class registration for layered Python architectures.

Inheritance is registration — when you write ``class FooAgent(BaseAgent)``,
the class is automatically registered in the appropriate layer registry.
"""
from __future__ import annotations

from conscribe.exceptions import (
    CircularWiringError,
    ConfigAnnotationError,
    ConfigDeclarationError,
    ConfigRenderError,
    ConscribeWarning,
    DegradationWarning,
    DuplicateKeyError,
    EmptyLayerError,
    ExtractionWarning,
    InvalidConfigSchemaError,
    InvalidProtocolError,
    InvalidRegistryKeyError,
    IsolationWarning,
    KeyNotFoundError,
    ProtocolViolationError,
    RegistryError,
    StaleStubWarning,
    WiringProblem,
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
from conscribe.wiring import (
    WiredField,
    WiringDeclaration,
    WiringProvenance,
    check_wiring_declaration,
    inject_wired_descriptors,
    validate_wiring,
    wired_field,
    wired_fields,
    wiring_provenance,
)
from conscribe.config import (
    ComposedConfigResult,
    ConfigFieldProvenance,
    ConfigProvenance,
    ConstructorProvenance,
    FieldDeclaration,
    ReceptorProvenance,
    DegradedField,
    LayerConfigResult,
    MROScope,
    build_composed_config,
    build_layer_config,
    compute_registry_fingerprint,
    extract_config_schema,
    extract_config_provenance,
    generate_composed_config_source,
    generate_composed_json_schema,
    generate_layer_config_source,
    generate_layer_json_schema,
    load_cached_fingerprint,
    save_fingerprint,
)
from conscribe.config.composed import (
    INLINE_ONLY_ERROR_TYPE,
    ReceptorLocation,
    TopLevelExtra,
)

__version__ = "2.1.0"

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
    "get_registry",
    "discover",
    # Wiring API
    "WiredField",
    "WiringDeclaration",
    "WiringProvenance",
    "wiring_provenance",
    "inject_wired_descriptors",
    "wired_field",
    "wired_fields",
    "check_wiring_declaration",
    "validate_wiring",
    # Config API
    "MROScope",
    "extract_config_schema",
    "extract_config_provenance",
    "ConfigProvenance",
    "ConfigFieldProvenance",
    "ConstructorProvenance",
    "FieldDeclaration",
    "ReceptorProvenance",
    "build_layer_config",
    "build_composed_config",
    "LayerConfigResult",
    "ComposedConfigResult",
    "DegradedField",
    "ReceptorLocation",
    "TopLevelExtra",
    "INLINE_ONLY_ERROR_TYPE",
    "generate_layer_config_source",
    "generate_composed_config_source",
    "generate_layer_json_schema",
    "generate_composed_json_schema",
    "compute_registry_fingerprint",
    "load_cached_fingerprint",
    "save_fingerprint",
    # Stubs API: import from conscribe.stubs (see conscribe.stubs.__all__).
    # Introspection API: import from conscribe.introspect.
    # Test isolation: import from conscribe.testing.
    # Errors (all are RegistryError subclasses)
    "RegistryError",
    "DuplicateKeyError",
    "KeyNotFoundError",
    "InvalidRegistryKeyError",
    "ProtocolViolationError",
    "InvalidConfigSchemaError",
    "InvalidProtocolError",
    "CircularWiringError",
    "WiringResolutionError",
    "WiringProblem",
    "EmptyLayerError",
    "ConfigDeclarationError",
    "ConfigAnnotationError",
    "ConfigRenderError",
    # Warnings (all are ConscribeWarning subclasses)
    "ConscribeWarning",
    "ExtractionWarning",
    "DegradationWarning",
    "StaleStubWarning",
    "IsolationWarning",
    # Version
    "__version__",
]
