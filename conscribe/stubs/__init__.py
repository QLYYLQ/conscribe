"""Generate .pyi stub files for classes with wired attributes."""
from __future__ import annotations

from conscribe.stubs.collector import (
    ClassStubInfo,
    InjectedAttr,
    collect_class_stub_info,
    narrowest_common_base,
)
from conscribe.stubs.generator import generate_module_stub
from conscribe.stubs.writer import write_layer_stubs

__all__ = [
    "ClassStubInfo",
    "InjectedAttr",
    "collect_class_stub_info",
    "generate_module_stub",
    "narrowest_common_base",
    "write_layer_stubs",
]
