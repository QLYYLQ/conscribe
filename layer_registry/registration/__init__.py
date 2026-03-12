"""Registration subsystem for layer-registry.

Re-exports all public APIs from submodules.
"""
from __future__ import annotations

from layer_registry.registration.auto import create_auto_registrar
from layer_registry.registration.discover import discover
from layer_registry.registration.key_transform import (
    KeyTransform,
    default_key_transform,
    make_key_transform,
)
from layer_registry.registration.registrar import (
    LayerRegistrar,
    create_registrar,
)
from layer_registry.registration.registry import LayerRegistry

__all__ = [
    "KeyTransform",
    "default_key_transform",
    "make_key_transform",
    "LayerRegistry",
    "create_auto_registrar",
    "LayerRegistrar",
    "create_registrar",
    "discover",
]
