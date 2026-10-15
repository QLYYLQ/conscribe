# Declaration provenance

Conscribe 2.1 exposes the evidence its extractor uses to decide which fields a
class contributes to a config. This answers “where was this field declared?”
without asking a framework to reconstruct constructor or wiring rules.

```python
from conscribe import extract_config_provenance, wiring_provenance


class Base:
    __wiring__ = {"strategy": ["basic"]}

    def __init__(self, *, strategy: str, budget: int = 3):
        pass


class Agent(Base):
    __wiring__ = {"strategy": ["fast", "careful"]}


result = extract_config_provenance(Agent)
assert result.fields["budget"].declared_on is Base
assert result.fields["strategy"].receptor.init_owner is Base
assert result.fields["strategy"].wiring.winner.declared_on is Agent
assert result.mro == (Agent, Base, object)
assert wiring_provenance(Agent)["strategy"].winner.declared_on is Agent
```

Three questions belong to three different records:

| Question | Owner |
| --- | --- |
| Which class declared a field, constructor slot or wiring entry? | Conscribe declaration provenance |
| Did the user supply the value, or did validation apply a default? | The consuming framework's input record |
| Which instance was constructed and injected, and where? | The consuming framework's execution record |

A runtime receptor in this API means a place capable of receiving a value. It
does not claim that any instance was built or injected.

## Entry points

`extract_config_provenance(cls, mro_scope="local", mro_depth=None)` is exported
from `conscribe` and `conscribe.config`. It follows the same extraction path as
`extract_config_schema`, including declaration validation, warnings, annotation
resolution and existing factory handling. It does not require a registrar,
though registry-backed wiring must resolve just as it does during extraction.

`Registrar.get_config_provenance(key)` supplies that registrar's scope/depth
settings. Class-level `__config_mro_scope__` and `__config_mro_depth__` overrides
win, just as with `get_config_schema`. Use this method when explaining a layer
instead of reproducing its private settings.

The result describes the per-class schema **before** a builder adds a
discriminator, assigns nested paths or inlines composed wiring. This also applies
to `get_config_provenance` on a nested-mode registrar: it matches that registrar's
`get_config_schema`, not its nested `build_config` layout. Provenance does not
add fields or extensions to generated Python or JSON Schema, mutate the source
schema, or change fingerprints apart from the normal release-version input.

`wiring_provenance(cls)`, exported from `conscribe` and `conscribe.wiring`, reads
wiring declarations without resolving registries. It works even when a target
registry is empty or absent.

## Result records

Result dataclasses are frozen; mappings are read-only snapshots and histories
are tuples. They contain class references, names, flags and reason strings,
without mutable defaults, annotations or `FieldInfo` objects. A consumer can
render an owner as `owner.__module__ + "." + owner.__qualname__`. Source files
and source line numbers are not required or fabricated. Later class or wiring
mutations are reflected by a new call, not by an existing snapshot. New record
fields may be added in minor releases; do not unpack records positionally.

`ConfigProvenance` contains:

| Field | Meaning |
| --- | --- |
| `cls`, `mro` | Inspected class and complete Python MRO, including `object` |
| `path` | `reflection`, `schema` (`__config_schema__`), or `flattened` (`__config_from_param__`) |
| `fields` | Field name to `ConfigFieldProvenance`; exactly the extracted schema's fields, in schema order |
| `exclusions` | Observed constructor declarations excluded by extraction, with reasons |
| `constructor_chain` | `ConstructorProvenance`, or `None` for an authoritative Tier-3 schema |
| `wiring` | Wiring histories, including entries excluded with `None` |
| `source_model` | Selected Tier-3 or flattened model; otherwise `None` |
| `flattened_param` | Parameter selected by `__config_from_param__`, or `None` |
| `schema_declared_on` | Class declaring the selected `__config_schema__` or `__config_from_param__` |
| `mro_scope`, `mro_depth` | Effective extraction settings (package lists become sorted tuples); `None` on the schema path |

`ConfigFieldProvenance` contains `name`, `source`, `declared_on`, `init_owner`,
`declarations`, `receptor` and `wiring`. `source` is one of `init`, `dataclass`,
`generated` (such as attrs), `model`, `pydantic_dataclass`, `annotation`, `schema`,
`flattened`, or `wiring`. `declared_on` identifies the declaration supplying the
field; `init_owner` identifies the constructor accepting it when applicable.
A generated dataclass constructor may accept a field declared by an ancestor.
A slot-less field's declaration belongs to the wiring owner, not automatically
to the inspected leaf class.

`FieldDeclaration` contains `name`, `source`, `declared_on`, `init_owner`,
`status` (`included`, `overridden`, `excluded`) and `reason`. Field histories
are nearest-first. Schema/flattened histories include model field annotations
in the model's MRO. Annotation-only receptor histories use the shared slot
resolver and retain pseudo-fields such as `ClassVar` as `excluded` with reason
`pseudo_field`. Constructor histories contain parameters actually visited
by the existing walker, including overridden names.

`ReceptorProvenance` contains `via` (`init` or `annotation`), `shape` (`scalar`,
`list` or `dict`), `declared_on`, `init_owner` and `annotation_declared_on`.
The last member identifies the annotation actually used, including a class
annotation fallback for an unannotated constructor parameter. The underlying
public `conscribe.introspect.ReceptorSlot` also exposes `init_owner` and
`annotation_declared_on`.

`ConstructorProvenance` contains `kind`, `definer`, `parameters` (effective
names), `declarations` (including observed overrides), `definers`,
`var_keyword`, `var_positional`, `fully_resolved`, `stop_reason`, `stopped_at`,
`unreadable` and `terminal`. It is an immutable projection of `walk_init_chain`;
that walk's `InitChain.declarations` now preserves all named parameters read
before deduplication.

## Wiring history

`wiring_provenance(cls)` returns a mapping from entry name to
`WiringProvenance(name, declarations, winner, excluded)`. Histories are in
leaf-first MRO order; mapping order is the existing base-first declaration order.

Each `WiringDeclaration` contains:

- `name`, `declared_on`: the entry and its class owner.
- `spec`: the existing frozen `WiringSpec`, with declared keys and
  `declare_only`; `None` for an exclusion or malformed overridden declaration.
- `disposition`: `winner`, `overridden`, or `excluded` (the winning `None`).
- `excluded`: whether this particular declaration was `None`, even when a
  subclass later overrides it.
- `problems`: immutable `(kind, registry_name, detail)` tuples for an
  overridden malformed declaration.

The first history item is `winner`, including when it excludes the name.
Excluding wiring does not necessarily exclude a config field: an ordinary
constructor parameter with that name can still be a knob. Its field record
retains the excluded wiring history.

Effective malformed declarations raise the same `WiringResolutionError` as
ordinary parsing. Overridden malformed declarations do not raise; they retain
the parser's existing diagnostic evidence. Unlike ordinary parsing, which drops
excluded entries, the provenance API also validates excluded entry names before
including them in its public mapping. `__wiring__ = None` declares no entries;
it is not an exclusion of every inherited name.

## Evidence boundaries and exclusions

`exclusions` records actual decisions made while extracting optional
positional-only parameters (`positional_only`), optional underscore-prefixed
parameters (`private_name`), or parameters without a `Field` declaration under
`__config_annotated_only__` (`annotated_only`). Required parameters that no
config can supply still raise; they do not produce a successful partial result.
Arbitrary class annotations are not all config candidates, and this API does
not label them as excluded fields.

Constructor history ends exactly where constructor extraction ends. An early
scope, depth, unreadable-signature or swallowing-placeholder stop is reported
with `stop_reason` and `stopped_at`; parameter names beyond it are unknown. A
constructor without `**kwargs` ends naturally and does not expose its ancestors'
constructor parameters. Model/generated signatures report their retained field
declarations, not source history that the generator discarded. The full `mro`
is available separately so a consumer can display the distinction.

Wired receptor discovery is scope-free. A wiring entry can expose a runtime
constructor slot beyond the configured ordinary-knob cutoff; its field record
then names that runtime owner while `constructor_chain` still reports the
configured cutoff. Tier-3 schemas bypass constructor extraction, so their
`constructor_chain` is `None`; their fields name schema declaration owners.
