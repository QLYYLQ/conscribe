"""Registration subsystem for conscribe.

Re-exports all public APIs from submodules.
"""
from __future__ import annotations

from conscribe.registration.auto import create_auto_registrar
from conscribe.registration.discover import discover
from conscribe.registration.key_transform import (
    KeyTransform,
    default_key_transform,
    make_key_transform,
)
from conscribe.registration.registrar import (
    LayerRegistrar,
    create_registrar,
)
from conscribe.registration.registry import LayerRegistry

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
