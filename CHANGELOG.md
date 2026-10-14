# Changelog

All notable changes to conscribe are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

conscribe 1.x dropped many declaration problems silently: an unknown
wiring key weakened the field to a plain `str`, a module that failed to
import was skipped with a log line, `Field(...)` metadata and inherited
annotations went missing, a typo in a composed config's top level was
ignored. In 2.0, anything a class declares wrongly or contradictorily
raises at class creation or at build time, with a message that names the
class, the field and the bad value and says what to change. Warnings are
reserved for what conscribe cannot read, degraded types, stale stub files
and registrations a test isolation block kept.

On top of that, 2.0 adds registry isolation for tests, inline-only layers
and a strict top level for composed configs, default instances for wired
receptors, declaration-only wiring entries, a public introspection API, and
generated sources and JSON Schemas that preserve the models' field
declarations. JSON Schema cannot express Python validators; source
generation reports validators it cannot reproduce with an
`ExtractionWarning` and a header note.

Upgrading takes a few changes in most projects. The
[migration guide](conscribe/docs/migrating-to-2.0.md) walks through them
with before/after examples.

### Breaking changes

**Supported versions**

- Python 3.10 or newer (3.9 is no longer supported); tested on 3.10 to 3.14.
- pydantic 2.7 or newer, below 3 (was 2.0).

**Wiring**

- `build_layer_config()` and `build_composed_config()` resolve the
  `__wiring__` of every class registered in the layers they build, before
  building, and raise one `WiringResolutionError` listing every problem. In
  1.x an unknown key, unknown optional key, unknown registry or empty
  registry made the extractor skip the class's whole wiring silently:
  fields fell back to their plain annotation (usually `str`) and wired
  names without a constructor parameter vanished from the config.
- New build-time wiring errors: unknown optional keys, empty target
  registries, and defaults that are not allowed keys (including the
  receptor's own parameter default).
- Grammar errors raise `WiringResolutionError` when the class is created,
  and the class is not registered: a wrong tuple shape, keys not in a list,
  non-string keys, empty key lists, a key listed as both required and
  optional, unknown dict-form fields, entry names that are not identifiers
  (or are keywords or dunder names), and a `__wiring__` that is neither a dict nor `None`.
- `WiringResolutionError` messages have a new format
  (`Bad (key 'bad') wiring 'action': unknown key 'kick' in registry 'act'; registered: ['click']`,
  several problems on separate lines). Read `.problems` instead of parsing
  messages; `cls_name`, `param_name`, `registry_name` and `detail` describe
  the first problem.
- `parse_wiring()` raises `WiringResolutionError` for malformed
  declarations (was `TypeError`).
- `ResolvedWiring.injected` is removed (it was never set); `optional_keys`
  is now the fourth positional field.

**`discover()`**

- Strict by default: an exception raised while importing a module
  propagates as the original exception, with a note naming the module on
  Python 3.11+ (on 3.10 a failing submodule is re-raised as a chained
  `ImportError`). 1.x logged a warning and skipped `ImportError` and
  `SyntaxError`. `strict=False` keeps skipping them, now with an
  `ExtractionWarning` per module.
- Registrations made by a module whose import failed are rolled back.
- `__main__` submodules are no longer imported by the package walk.
- The `auto_update_stubs`, `stub_dir` and `fingerprint_path` parameters
  are removed. The feature never ran: the list of registrars it checked was
  always empty.

**Composed config**

- The top-level model forbids unknown keys by default
  (`top_level_extra="forbid"`, JSON Schema `additionalProperties: false`);
  1.x ignored them.
- Layer names (the keys of the `registrars` mapping) must be valid
  identifiers that are not private and do not shadow `BaseModel`
  attributes or use the `model_` prefix (`ConfigDeclarationError`). A value
  that is not a registrar raises `TypeError`.
- Wired fields of `__config_schema__` (Tier 3) owners are inlined like
  other receptors, so they take nested configs instead of key strings.
- A wired parameter whose default is a key string (`loop: str = "react"`)
  defaults to that key's config instance in inline builds (1.x left the
  string as the default of a field typed as configs) and raises
  `ConfigDeclarationError` when that config cannot be built from defaults.

**Config extraction**

- A class whose `__init__` takes a single pydantic-model parameter is no
  longer flattened: the parameter is a nested block. Opt back in with
  `__config_from_param__ = "<param>"`. For this major version, configs in
  the old flat shape fail with a `legacy_flat_config` error that names both
  fixes.
- Annotations resolve in the module of the class that declares them, and
  only there. 1.x fell back to a namespace pre-filled with `typing` names;
  a module must now import every name its annotations use, otherwise
  `ConfigAnnotationError`. Names imported only under `if TYPE_CHECKING:`
  degrade the field to `Any` with a `DegradationWarning`. `Annotated`
  metadata that cannot be evaluated raises `ConfigAnnotationError`.
- The whole `Field(...)` declaration is carried into the config: title,
  description, examples, `json_schema_extra`, discriminator, constraints,
  and the alias family. Aliases now change the accepted config keys.
- `Field(exclude=...)` on a constructor parameter raises
  `ConfigDeclarationError`; `deprecated=` becomes a JSON Schema
  `deprecated` marker only.
- A default declared both in the signature and in `Field(default=...)` /
  `default_factory`, with different values, raises
  `ConfigDeclarationError` (1.x kept the signature default).
- Validators and `model_config` of a `__config_schema__` model now apply to
  the config (1.x dropped them). Every wired entry that has a runtime slot
  must be a field of that schema.
- `x: T = None` makes the field admit `None` on every Python version.
- In wired annotations, a user-written `Literal[...]` member whose values
  are not all wiring keys raises `ConfigDeclarationError`; key-string
  constraints (`pattern`, `min_length`, `max_length`) narrow the allowed
  keys, and constraints that leave no key raise.
- More declarations raise `ConfigDeclarationError`: constructor parameters
  named `model_config` or another `model_*` member of `BaseModel`, required positional-only
  parameters, invalid `__config_mro_scope__` / `__config_mro_depth__`,
  config fields named like the layer's discriminator.
- An empty layer raises `EmptyLayerError` in every builder (1.x:
  `TypeError: Cannot take a Union of no types`).

**Nested mode (`discriminator_fields`)**

- `__wiring__`, `__config_schema__`, `__config_from_param__`,
  `__config_mro_scope__` and `__config_mro_depth__` on classes of a
  nested-mode layer raise `ConfigDeclarationError` (1.x ignored them), as
  do keys whose segment count does not match the levels (1.x dropped those
  classes) and fields named like a discriminator or segment field.
- Parameters are assigned to levels by the leaf class's real constructor
  (the level of the root-most class whose own constructor accepts them).

**`WiredField` and registration**

- Class access (`Agent.env`) raises `AttributeError` when the accessing
  class, or any class between it and the class holding the placeholder,
  annotates the name; `hasattr()` is then `False`. Use the new
  `wired_field()` / `wired_fields()` to look placeholders up.
- A class never gets a placeholder of its own for a wired name it
  annotates, so a dataclass or attrs field `env: Env` without a default is
  a required field (1.x made the placeholder its default).
- A subclass that excludes an inherited entry (`{"env": None}`) or
  redeclares it `declare_only` hides the inherited placeholder (class
  access raises; instance access raises the plain "no attribute" error).
- Registry keys must be non-empty strings (`__registry_key__` may be a
  non-empty list of them): Enum members, tuples, ints and empty values raise
  `InvalidRegistryKeyError`. 1.x registered an Enum member as
  `'Kind.REACT'`.
- Registration is atomic: a class statement that fails leaves nothing
  registered in any registry.

**Stubs**

- Only wiring entries without a runtime slot (no constructor parameter, no
  class annotation) and without `declare_only` are stubbed. Classes whose
  wired names all have a slot get no stub; leftover stubs are reported (see
  Added).
- Injected attributes are typed as the target registry's protocol; the
  narrowing to the common base of the allowed implementations is gone.
- Writing next to sources in site-packages or the standard library
  without an output directory raises `ConfigRenderError`; wiring problems
  raise before any file is written; an empty layer raises
  `EmptyLayerError`.
- Collector API: `ClassStubInfo.init_signature` is now `init`
  (`MethodStub | None`), `MethodStub.signature` is now `function` (plus
  `kind`), `ClassAttrStub` holds the annotation object with `owner` and
  `has_default`; `render_signature` is removed.

**Generated artifacts and fingerprints**

- Every generated config source, JSON Schema and stub changes once
  (header, field rendering, imports, model names); regenerate them.
- `compute_registry_fingerprint()` hashes the generated config source and
  JSON Schema (not stubs), so every stored fingerprint changes once.

### Added

- **Diagnostics.** Errors: `ConfigAnnotationError`,
  `ConfigDeclarationError`, `ConfigRenderError`, `EmptyLayerError`,
  `InvalidRegistryKeyError`, and `WiringProblem` (the items of
  `WiringResolutionError.problems`, with a stable `kind` from
  `WIRING_PROBLEM_KINDS`). Warnings, all subclasses of `ConscribeWarning`
  (those about a class are reported at its definition):
  `ExtractionWarning`, `DegradationWarning`, `StaleStubWarning`, and
  `IsolationWarning` (an isolation block with `warn_kept=True` kept
  registrations made by modules it imported). Make them errors with
  `filterwarnings = error::conscribe.ConscribeWarning`.
- `validate_wiring(registrars)` reports every wiring problem of every
  registered class at once; `check_wiring_declaration(cls)` runs the
  grammar checks only. Every registration path calls the latter before it
  registers a class.
- **Dict-form wiring entries**:
  `{"registry": ..., "keys": [...], "optional_keys": [...], "default": "<key>", "declare_only": bool}`.
  A subclass entry replaces the inherited one whole; `None` still excludes.
  `WiringSpec` and `ResolvedWiring` gain `default_key` and `declare_only`.
- **Default instances.** `"default"` gives a single-instance receptor (a
  constructor parameter or class annotation that receives one wired value)
  a default key. Without inline wiring the field defaults to the key
  string; with inline wiring it defaults to the key's full config, built
  from defaults (shown in full in the JSON Schema). `null` is accepted only
  when the receptor's annotation admits `None`.
- `declare_only` entries: constraints the framework only reads. They get
  no placeholder and no stub, and are never inlined; their config field is
  `Optional[Literal[...]] = None`.
- **`conscribe.testing`**: `isolated_registries()` / `RegistryIsolation`
  restore every registry and the registry index on exit, keep
  registrations made by modules first imported inside the block, list what
  was `kept` and `rolled_back`, and capture module-level fakes for
  `activate()` per test. Opt-in pytest plugin
  `conscribe.testing.pytest_plugin` with the `conscribe_isolated` fixture
  and the `conscribe_warn_kept` ini option.
- `LayerRegistrar.unregister(key, *, missing_ok=False)` and
  `LayerRegistry.remove(key, *, missing_ok=False)` return the removed
  class; a missing key raises `KeyNotFoundError` with a removal-specific
  message.
- **Inline-only layers**: `create_registrar(..., inline_only=True)`
  (`Registrar.inline_only`). Their instances may only be written inside
  other layers' receptors; writing the layer at top level fails with a
  pydantic error of type `inline_only_layer` (`INLINE_ONLY_ERROR_TYPE`)
  that lists where the instances belong
  (`layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive`).
  `ComposedConfigResult` gains `inline_only_layers` (`ReceptorLocation`
  values), `top_level_extra` and `top_level_layers`.
- `build_composed_config(..., top_level_extra=...)` (`"forbid"`,
  `"ignore"` or `"allow"`; type alias `TopLevelExtra`).
- `__config_from_param__` flattens a single pydantic-model parameter into
  the config, on request.
- **`conscribe.introspect`**: `init_param_names()`, `receptor_slots()`
  (`ReceptorSlot`), `wiring_slots()` (`WiringSlot`), `annotation_shape()`,
  `is_classvar()`, `config_from_param()`. These are the rules conscribe
  itself uses to decide which names reach a constructor and which
  attributes receive wired values.
- `wired_field(cls, name)` and `wired_fields(cls)` look up `WiredField`
  placeholders without triggering them.
- `discover(..., strict=True, allow_missing=(), exclude=())`, plus
  `import_failures()` and `clear_import_failures()` in
  `conscribe.discover`.
- **Stale stubs**: conscribe-generated `.pyi` files (1.x headers included)
  that a run did not rewrite are reported with a `StaleStubWarning`;
  `write_layer_stubs(..., prune=True)` / `generate-stubs --prune` delete
  exactly those files.
- Stubs repeat the class's own surface: constructor, methods (including
  cached, context-manager and wrapped ones), properties, class values and
  nested classes. New collector types `ClassMembers` and `ClassValueStub`,
  and `collect_class_members()`.
- CLI: `--registrar module:attribute` and `--discover package`
  (repeatable) for `generate-config`, `generate-composed-config`,
  `generate-stubs` and `inspect`; `--allow-missing` and `--exclude` (also
  for `list`); repeatable `--layers`; `generate-config --json-schema`;
  `generate-composed-config --top-level-extra`; `generate-stubs --prune`.
- Default factories (`dataclasses.field(default_factory=...)`, attrs
  `Factory`, pydantic `default_factory`) are read from the field. Pure
  factories (builtin containers, closure-free functions that use no
  globals) become plain defaults; others stay factories and the generated
  source references them.
- Generated files start with `# Generated by conscribe <version>`.
- A `LICENSE` file (MIT).

### Changed

- Config models are built by subclassing their source model (the
  extracted schema, a `__config_schema__` model, or the model of a
  `__config_from_param__` parameter), so the source's validators,
  `model_config` and field metadata survive. Validators inherited by
  inline-wired models see nested configs where the per-layer model passes
  key strings. Registered pydantic models are the exception: their config
  is rebuilt from their fields (see Known limitations).
- Every extraction path reaches the wiring pass: classes without any
  `__init__`, registered pydantic models and pydantic dataclasses,
  constructors whose signature cannot be read (`ExtractionWarning`,
  `extra="allow"`).
- `**kwargs` chains walk through `Generic`, `ABC`, Protocol and other
  mixins without an `__init__`; pydantic models and pydantic dataclasses
  end them.
- Inline receptors keep their description, title and deprecated marker;
  inline unions keep their discriminators in generated source.
- Generated sources render required fields as `name: T` (or
  `name: T = Field(...)` with other settings, never `= ...`), a default
  that comes with other settings as `Field(default=..., ...)`, every
  constraint (also inside type arguments),
  tuples as `tuple[str, ...]`, and import user types by identity (aliased
  on name clashes). Values go through a ladder of verified forms (literals,
  Enum members, `Path`, tz-aware `datetime`, `Decimal`, `re.compile`,
  constructor forms, importable references); anything that cannot be
  written faithfully raises `ConfigRenderError`. Validators the file cannot
  carry are reported with an `ExtractionWarning` and a header note.
  Composed sources reproduce the inline-only refusal and the old-shape
  guard and end with `ComposedConfig.model_rebuild()`. Output is
  deterministic and type-checks with pyright.
- The published composed JSON Schema equals
  `top_level_type.model_json_schema()`, including the
  `x-composed-layers`, `x-inline-wiring` and `x-inline-only-layers`
  extensions.
- Generated model class names are sanitized identifiers, unique per build.
- `compute_registry_fingerprint()` hashes the conscribe version, the layer
  settings, and the layer's generated config source and JSON Schema. It
  detects changes to those outputs and stays the same across Python
  versions and processes whenever those inputs match. Stubs are not
  hashed. For composed output, track every layer and the composition
  options. Fingerprinting never raises.
- Nested mode carries descriptions, constraints, defaults and factories;
  an omitted segment block selects the default segment; the discriminator
  never raises a raw `TypeError`.
- Warnings about a class carry the file and line of its definition, so
  `-W` filters, `pytest.warns` and IDEs point at the code to fix.
  conscribe no longer reports problems through `logging`; only skips you
  asked for (`discover(allow_missing=...)`) are logged, at info level.
- Config CLI commands render and stage outputs before replacing each file.
  Replacement is atomic per file, without rollback across several files;
  stub generation renders before writing but writes directly. Errors print
  `Error: ...` and exit with status 1; usage errors exit with status 2.
- The wheel ships `conscribe/docs/*.md`, so the links in `llms.txt` work in
  an installed package. The sdist contains only the package, `README.md`,
  `CHANGELOG.md`, `LICENSE` and `pyproject.toml`. Both use core metadata
  version 2.4 (`Metadata-Version: 2.4`).

### Fixed

- Classes with no `__init__` anywhere in their MRO lost their wiring
  receptors (the 1.3.0 known limitation).
- Fields inherited from a base class in another module that uses
  `from __future__ import annotations` were resolved against the
  subclass's imports and could be dropped silently.
- A `__wiring__` entry on a dataclass installed a `WiredField` that the
  dataclass took as the field's default.
- Default factories were rendered as `= ...  # default: <factory>`.
- `x: T = None` rejected an explicit `None`.
- Generated stubs could be invalid (unresolvable names, names bound to
  `Any`, missing class members); stubs for a second layer overwrote the
  first layer's classes in a shared module.
- The CLI `generate-*` and `inspect` commands could not find any registrar
  outside tests.
- `@dataclass(slots=True)`, `attrs.define` and pydantic slots dataclasses
  on a registered class raised `DuplicateKeyError`, because the decorator
  rebuilds the class. The rebuilt class now takes over every key of the
  original; running a class body twice still raises, with a hint toward
  `isolated_registries()`.
- Fingerprints depended on object addresses and hash seeds.
- Wheels since 0.5.3 did not contain the documentation `llms.txt` links
  to.

### Removed

- Python 3.9 support.
- `discover(auto_update_stubs=..., stub_dir=..., fingerprint_path=...)`.
- `ResolvedWiring.injected`.
- `conscribe.stubs.collector.render_signature`, `ClassStubInfo.init_signature`
  and `MethodStub.signature` (see Breaking changes).
- `conscribe.config.builder` no longer re-exports `extract_own_init_params`;
  import it from `conscribe.config.extractor`.

### Known limitations

- Validators of a registered pydantic model (an implementation class that
  is itself a `BaseModel`) are not part of its config; they run when the
  framework instantiates the class. Use `__config_schema__` to validate at
  config-load time.
- A `__config_schema__` model with a `@computed_field` dumps that field,
  and `extra="forbid"` refuses it when the dump is validated again.
- Nested-mode layers do not support `__wiring__`, `__config_schema__`,
  `__config_from_param__` or the MRO settings yet.
- `@register` applied below `@dataclass(slots=True)`, and a pydantic
  dataclass stacked on a stdlib dataclass, are not supported.
- `isolated_registries()` works on all registries at once; blocks must
  exit in LIFO order and must not overlap across threads or asyncio tasks.
- pydantic 2.11 and 2.12 are slow on deep chains of inline wiring; 2.13 is
  not.
- Planned for 2.1: wiring and `__config_schema__` in nested mode, and a
  typed `WiredField[T]` that can replace stubs.

## [1.3.0]

Two fixes to `__wiring__`, both of which remove a way for a declaration to be
silently lost or over-constrained. Neither changes the output for any wiring
declaration that was already legal, so upgrading regenerates byte-identical
config for existing consumers.

### Added

- **Capability-relative wiring keys.** When the target registry was created
  with a `key_separator` (e.g. `create_registrar(..., key_separator=".")`),
  the explicit key lists of the tuple modes may now name a bare **trailing
  segment** instead of a fully-qualified key:

  ```python
  # Registered: "browser.click", "desktop.click", "browser.scroll"
  __wiring__ = {"action": ("action", ["click"], ["scroll"])}
  # -> Literal["browser.click", "desktop.click", "browser.scroll"]
  ```

  Previously only `"browser.click"` was accepted and `"click"` was rejected as
  a missing key, which type-welded every declaring class to exactly one
  provider. A key that already contains the separator is passed through
  **unchanged**, so pinning a specific provider still works and remains the
  escape hatch. Registries without a `key_separator` are completely unaffected.

  This is **not** a new `__wiring__` grammar mode — the three existing modes
  are unchanged; only the matching of the key strings inside them is relaxed.

  If a short name matches several providers, *all* of them are included in the
  generated `Literal[...]`. Conscribe expands, it does not arbitrate: at config
  generation time it cannot know which providers a given assembled runtime will
  hold, so every candidate is legitimately possible. Narrowing to one belongs to
  the consuming framework, at the point where it knows the concrete provider set.

- `conscribe.wiring.expand_relative_keys(declared, registry_keys, separator)` —
  the expansion rule, exposed so consumers can project their own key spaces the
  same way.

### Fixed

- **`__wiring__` receptors are no longer dropped by a parameter-less
  `__init__`.** `extract_config_schema()` returned `None` as soon as
  `inspect.signature(cls.__init__)` yielded no named parameters — before
  `__wiring__` was applied. A class that declared no config knobs of its own
  therefore produced a discriminator-only model, and every receptor it declared
  or inherited through `__wiring__` disappeared silently: the dependency became
  unselectable from config and bound to a default instead. The only cure was to
  widen the signature with a throw-away parameter.

  Wiring is now applied first, and the "nothing to extract" bail fires only when
  the class contributes neither named parameters *nor* wiring receptors. Classes
  with no wiring are unaffected and still yield `None`.

### Changed

- The project tagline "`__init__` signature is config schema" was accurate but
  incomplete, and the incompleteness was load-bearing for the bug above. It now
  reads "`__init__` signature *plus* `__wiring__` is config schema"
  (`README.md`, `conscribe/llms.txt`).

### Known limitation

- The sibling bail at `config/extractor.py` — `find_init_definer(cls) is None`,
  i.e. *no* class in the MRO defines `__init__` at all — still returns `None`
  before wiring is applied, so such a class also loses its receptors. It was
  left alone deliberately: it is a different code path from the one fixed here
  and was not covered by the measurement that validated this release. Give the
  class an `__init__` (even an empty one) as a workaround.

## [1.2.0]

- Fixed the shape of wired fields in generated configs and made `.pyi` output
  valid. Container-annotated receptors become `list[Literal[...]]`; slot-less
  receptors become `Optional[Literal[...]] = None`.

## [1.1.2]

- Emit nested submodels in composed / nested config source.

## [1.1.1]

- Alias-faithful imports and class-level attributes in `.pyi` stubs.

## [1.1.0]

- Resolve string annotations and `TYPE_CHECKING` imports in `.pyi` stubs.

## [1.0.0]

- `WiredField` descriptor for wired attribute injection.

## [0.9.0]

- CLI `scan` and `list` commands for registry discovery.

## [0.8.0]

- Composed config with inline wiring for multi-layer YAML schema.

## [0.7.0]

- Generate `.pyi` stubs for wired class attributes.

## [0.6.1]

- 3-element `__wiring__` tuple for required / optional key subsets.

## [0.6.0]

- Cross-registry wiring via `__wiring__` for config generation.

## [0.5.0]

- Hierarchical keys, cross-registry diamond inheritance, predicate filters,
  nested config mode.
