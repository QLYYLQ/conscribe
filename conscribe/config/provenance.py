"""Immutable declaration evidence from the config extraction pipeline.

These records describe declarations, not user input or injected instances.
They intentionally omit mutable defaults, annotations and pydantic FieldInfo
objects. Class references identify owners; no source-file availability is
assumed. Mapping values and declaration sequences are read-only snapshots.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from conscribe.config.mro import InitChain
    from conscribe.introspect import ReceptorSlot
    from conscribe.wiring import WiringProvenance

FieldSource = Literal[
    "init", "dataclass", "generated", "model", "pydantic_dataclass",
    "annotation", "schema", "flattened", "wiring",
]
DeclarationStatus = Literal["included", "overridden", "excluded"]

__all__ = [
    "ConfigProvenance", "ConfigFieldProvenance", "ConstructorProvenance",
    "FieldDeclaration", "ReceptorProvenance", "FieldSource", "DeclarationStatus",
]


@dataclass(frozen=True)
class FieldDeclaration:
    """One observed declaration and the extraction decision for it.

    ``reason`` is ``annotated_only``, ``positional_only`` or ``private_name``
    for an excluded parameter, ``pseudo_field`` for an annotation such as
    ``ClassVar`` that removes a slot, ``overridden`` for a shadowed
    declaration, and ``None`` for an included declaration. Only declarations actually
    encountered by the extractor are reported.
    """

    name: str
    source: FieldSource
    declared_on: type
    init_owner: type | None = None
    status: DeclarationStatus = "included"
    reason: str | None = None


@dataclass(frozen=True)
class ReceptorProvenance:
    """Where a configured wired instance can be received, not whether it was injected."""

    via: Literal["init", "annotation"]
    shape: Literal["scalar", "list", "dict"]
    declared_on: type | None
    init_owner: type | None
    annotation_declared_on: type | None


@dataclass(frozen=True)
class ConstructorProvenance:
    """Snapshot of the existing constructor walker, including its honest boundary.

    ``parameters`` contains effective declarations; ``declarations`` also
    includes shadowed parameters actually read before the boundary. A
    non-forwarding constructor ends the walk naturally. A scope/depth or
    unreadable cut names ``stopped_at``; fields beyond it are unknown.
    """

    kind: str
    definer: type | None
    parameters: tuple[FieldDeclaration, ...]
    declarations: tuple[FieldDeclaration, ...]
    definers: tuple[type, ...]
    var_keyword: bool
    var_positional: bool
    fully_resolved: bool
    stop_reason: str | None
    stopped_at: type | None
    unreadable: tuple[type, ...]
    terminal: type | None


@dataclass(frozen=True)
class ConfigFieldProvenance:
    """The declaration behind one field of the extracted config schema.

    Wiring and receptor ownership are independent: a subclass can replace a
    wiring entry while an ancestor still declares its constructor slot.
    ``declarations`` is nearest-first, limited to evidence actually read.
    Schema/flattened fields name their model field owner; ``source_model``
    on the enclosing result names the selected model.
    """

    name: str
    source: FieldSource
    declared_on: type
    init_owner: type | None
    declarations: tuple[FieldDeclaration, ...]
    receptor: ReceptorProvenance | None = None
    wiring: WiringProvenance | None = None


@dataclass(frozen=True)
class ConfigProvenance:
    """Declaration evidence for one successful config extraction.

    ``fields`` includes exactly the extracted schema's fields (before a
    builder adds discriminators or composition). ``exclusions`` records
    parameters the extractor actually rejected as optional positional-only,
    private, or undeclared under ``__config_annotated_only__``. Excluded
    wiring entries live in ``wiring`` and may still have an ordinary knob.
    Tier-3 schemas are authoritative and have no constructor extraction
    chain. ``schema_declared_on`` identifies the owner selecting a schema
    or flattened parameter. New fields may be added; do not unpack records
    positionally.
    """

    cls: type
    path: Literal["reflection", "schema", "flattened"]
    fields: Mapping[str, ConfigFieldProvenance]
    exclusions: tuple[FieldDeclaration, ...]
    constructor_chain: ConstructorProvenance | None
    wiring: Mapping[str, WiringProvenance]
    source_model: type | None = None
    flattened_param: str | None = None
    schema_declared_on: type | None = None
    mro_scope: str | tuple[str, ...] | None = None
    mro_depth: int | None = None
    #: The class's complete Python MRO, distinct from the constructor walk.
    mro: tuple[type, ...] = ()

    def __post_init__(self) -> None:
        # Copy first: a caller retaining the input dict cannot mutate a snapshot.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "wiring", MappingProxyType(dict(self.wiring)))


def _constructor(chain: InitChain) -> ConstructorProvenance:
    seen: set[str] = set()
    declarations: list[FieldDeclaration] = []
    for param in chain.declarations:
        overridden = param.name in seen
        seen.add(param.name)
        declarations.append(FieldDeclaration(
            param.name, param.source, param.declared_on, param.init_owner,
            "overridden" if overridden else "included",
            "overridden" if overridden else None,
        ))
    return ConstructorProvenance(
        chain.kind, chain.definer,
        tuple(d for d in declarations if d.status == "included"),
        tuple(declarations), chain.definers, chain.var_keyword,
        chain.var_positional, chain.fully_resolved, chain.stop_reason,
        chain.stopped_at, chain.unreadable, chain.terminal,
    )


def _receptor(slot: ReceptorSlot | None) -> ReceptorProvenance | None:
    if slot is None:
        return None
    return ReceptorProvenance(
        slot.via, slot.shape, slot.declared_on, slot.init_owner,
        slot.annotation_declared_on,
    )
