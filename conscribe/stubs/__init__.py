"""Generate ``.pyi`` stubs that declare the attributes a framework injects.

A ``__wiring__`` entry with no runtime slot (no constructor parameter, no
class annotation) and without ``"declare_only": True`` names an attribute the
framework sets on instances after construction.  :func:`write_layer_stubs`
writes a ``.pyi`` stub per module declaring those attributes, typed as the
target registry's protocol, next to the class's own surface.  A stub stands
for its whole module: it declares the stub-needing classes of every live
registry defined there, and stubs left over from earlier runs are reported
(:class:`~conscribe.exceptions.StaleStubWarning`) or pruned.
"""
from __future__ import annotations

from conscribe.stubs.collector import (
    ClassAttrStub,
    ClassMembers,
    ClassStubInfo,
    ClassValueStub,
    InjectedAttr,
    MethodStub,
    collect_class_members,
    collect_class_stub_info,
    narrowest_common_base,
)
from conscribe.stubs.generator import generate_module_stub
from conscribe.stubs.writer import write_layer_stubs

__all__ = [
    "ClassAttrStub",
    "ClassMembers",
    "ClassStubInfo",
    "ClassValueStub",
    "InjectedAttr",
    "MethodStub",
    "collect_class_members",
    "collect_class_stub_info",
    "generate_module_stub",
    "narrowest_common_base",
    "write_layer_stubs",
]
