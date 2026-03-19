"""Registration subsystem for conscribe.

Re-exports all public APIs from submodules.
"""
from __future__ import annotations

from conscribe.registration.auto import create_auto_registrar
from conscribe.registration.filters import (
    RegistrationContext,
    RegistrationFilter,
    build_filter_chain,
)
from conscribe.registration.key_transform import (
    KeyTransform,
    default_key_transform,
    make_key_transform,
)
from conscribe.registration.meta_base import AutoRegistrarBase
from conscribe.registration.registrar import (
    LayerRegistrar,
    create_registrar,
)
from conscribe.registration.registry import LayerRegistry

__all__ = [
    "AutoRegistrarBase",
    "KeyTransform",
    "default_key_transform",
    "make_key_transform",
    "LayerRegistry",
    "create_auto_registrar",
    "LayerRegistrar",
    "create_registrar",
    "RegistrationContext",
    "RegistrationFilter",
    "build_filter_chain",
]
