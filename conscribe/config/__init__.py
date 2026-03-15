"""Config typing subpackage for conscribe.

Provides config schema extraction, discriminated union building,
code generation, JSON Schema generation, and fingerprinting.
"""
from __future__ import annotations

from conscribe.config.builder import LayerConfigResult, build_layer_config
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.degradation import DegradedField
from conscribe.config.extractor import extract_config_schema
from conscribe.config.fingerprint import (
    compute_registry_fingerprint,
    load_cached_fingerprint,
    save_fingerprint,
)
from conscribe.config.json_schema import generate_layer_json_schema
from conscribe.config.mro import MROScope

__all__ = [
    "extract_config_schema",
    "build_layer_config",
    "LayerConfigResult",
    "DegradedField",
    "generate_layer_config_source",
    "generate_layer_json_schema",
    "compute_registry_fingerprint",
    "load_cached_fingerprint",
    "save_fingerprint",
    "MROScope",
]
