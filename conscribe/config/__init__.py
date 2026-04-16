"""Config typing subpackage for conscribe.

Provides config schema extraction, discriminated union building,
code generation, JSON Schema generation, and fingerprinting.
"""
from __future__ import annotations

from conscribe.config.builder import LayerConfigResult, build_layer_config
from conscribe.config.codegen import generate_composed_config_source, generate_layer_config_source
from conscribe.config.composed import ComposedConfigResult, build_composed_config
from conscribe.config.degradation import DegradedField
from conscribe.config.extractor import extract_config_schema
from conscribe.config.fingerprint import (
    compute_registry_fingerprint,
    load_cached_fingerprint,
    save_fingerprint,
)
from conscribe.config.json_schema import (
    generate_composed_json_schema,
    generate_layer_json_schema,
)
from conscribe.config.mro import MROScope

__all__ = [
    "extract_config_schema",
    "build_layer_config",
    "build_composed_config",
    "LayerConfigResult",
    "ComposedConfigResult",
    "DegradedField",
    "generate_layer_config_source",
    "generate_composed_config_source",
    "generate_layer_json_schema",
    "generate_composed_json_schema",
    "compute_registry_fingerprint",
    "load_cached_fingerprint",
    "save_fingerprint",
    "MROScope",
]
