# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Working Agreements

- Use the uv-managed environment for Python project commands. Run tests as `uv run pytest` and CLI commands as `uv run conscribe ...`; do not use bare `pytest` or `python -m pytest`.
- Never add `Co-Authored-By` lines to commit messages.
- Keep this file and `CLAUDE.md` synchronized when the current version, architecture map, commands, or non-obvious invariants change.
- For version bumps or publishing, follow `.agents/skills/conscribe-release/SKILL.md`. The user's review of the prepared release diff is the gate; after approval, complete push, build, and PyPI upload without inserting another confirmation stop.

## Project Overview

**Conscribe** (`conscribe` package, v2.0.0) — A Python library for automatic class registration and config typing stub generation for layered Python architectures. It targets framework developers building config-driven frameworks with pluggable layers (agents, LLM providers, etc.). The main downstream user is af. Python >= 3.10 (tested on 3.10–3.14), pydantic >= 2.7, < 3.

Core capabilities:
1. **Auto-registration**: Classes inheriting a base are automatically registered in their layer's registry (via metaclass). Also supports bridging external classes and explicit `@register` decorators. Keys are validated, registration is atomic, `isolated_registries()` scopes registrations for tests.
2. **Config typing**: Extracts `__init__` signatures (the whole `Field(...)` declaration, default factories, docstring descriptions) into Pydantic discriminated unions, generates Python source and JSON Schema for IDE autocomplete of YAML configs. Field declarations must agree across the in-process model, generated source and JSON Schema. Python validators have no JSON Schema equivalent; source rendering warns about validators it cannot reproduce.
3. **Cross-registry wiring**: Classes declare `__wiring__` to reference other registries, constraining config fields to `Literal[...]` with auto-discovery or explicit subsets; dict-form entries add a `default` key and `declare_only`. Every wiring problem raises.
4. **Wired field injection**: `WiredField` non-data descriptors are set on classes at definition time for wired attributes the framework injects (no constructor parameter, no own class-body value or annotation, not `declare_only`; inherited bindings are handled separately). They raise a clear `AttributeError` before injection; instance assignment shadows them.
5. **`.pyi` stub generation**: Generates PEP 484 `.pyi` files declaring injected wired attributes, typed as the target registry's protocol. Stale conscribe stubs are reported or pruned.
6. **Composed config**: Combines multiple layers into a single JSON Schema / Python source with inline wiring — wired fields become the target layer's full discriminated union. Strict top level (`top_level_extra="forbid"`), inline-only layers, default instances.
7. **Introspection** (`conscribe.introspect`): the public rules for which names reach a constructor and which attributes receive wired values.

## Commands

```bash
# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/config/test_extractor.py

# Run a single test
uv run pytest tests/config/test_extractor.py::TestExtractConfigSchema::test_basic

# Whole suite without coverage (faster)
uv run pytest -o addopts='' -q -p no:cacheprovider

# Version-sensitive changes (annotations, Union handling, inspect.signature): also run the oldest and newest
uv run --isolated --python 3.10 --with 'pydantic==2.7.*' --with pytest --with docstring-parser --with attrs pytest tests -o addopts='' -q -p no:cacheprovider
uv run --isolated --python 3.14 --with pydantic --with pytest --with docstring-parser --with attrs pytest tests -o addopts='' -q -p no:cacheprovider

# Install/synchronize development dependencies
uv sync --extra dev --extra docstring

# CLI (layers are loaded with --registrar MODULE:ATTR and/or --discover PACKAGE, both repeatable)
uv run conscribe generate-config --registrar <mod:attr> [--discover <pkg>] [--layer <name>] [--output <path>] [--json-schema <path>]
uv run conscribe generate-composed-config --registrar <mod:attr> ... [--layers <name> ...] [--format json-schema|python] [--no-inline-wiring] [--top-level-extra forbid|ignore|allow] [--output <path>]
uv run conscribe generate-stubs --registrar <mod:attr> [--discover <pkg>] [--output-dir <path>] [--partial-class-fallback] [--prune]
uv run conscribe inspect --registrar <mod:attr> [--discover <pkg>] [--layer <name>]
uv run conscribe scan [--path <dir>]
uv run conscribe list [--discover <pkg> ...] [--layer <name>] [--path <dir>]
# every loading command also takes --allow-missing MODULE and --exclude PREFIX (forwarded to discover())
```

pytest is pre-configured in `pyproject.toml` with `--cov=conscribe --cov-report=term-missing --tb=short -q` and `filterwarnings = ["error::conscribe.exceptions.ConscribeWarning"]`: a conscribe warning a test does not assert with `pytest.warns` fails the test. `tests/conftest.py` loads `conscribe.testing.pytest_plugin` through `pytest_plugins` (never `-p`, which imports conscribe before pytest-cov starts) and its registrar fixtures run inside `isolated_registries()`.

## Architecture

### Package map

- `__init__.py` — public API and `__version__`. Stubs API lives in `conscribe.stubs`, introspection in `conscribe.introspect`, test isolation in `conscribe.testing`.
- `exceptions.py` — the diagnostics taxonomy (see invariants): `RegistryError` subclasses, `WiringProblem` / `WIRING_PROBLEM_KINDS`, `ConscribeWarning` subclasses.
- `registration/` — registry storage, metaclasses, registrar facade, filters, key transform, `isolation.py`.
- `wiring.py` — `__wiring__` grammar, resolution, validation, `WiredField` placeholders.
- `introspect.py` — public introspection API (single source of truth for constructor parameters and receptor slots).
- `discover.py` — strict package import with rollback and a failure record.
- `config/` — extract → build → generate pipeline, composed config, fingerprint; private engines `_annotations.py`, `_fieldinfo.py`, `_render.py`, `_utils.py`.
- `stubs/` — `.pyi` collection, rendering and writing.
- `testing/` — `isolated_registries()` re-export and the opt-in pytest plugin.
- `scanner.py`, `cli.py` — static scan, runtime listing, command line.
- `docs/*.md`, `llms.txt` — shipped in the wheel; relative links must resolve inside the package (`tests/test_packaging.py`).

### Registration (`conscribe/registration/`)

- `registrar.py` — `LayerRegistrar` facade created via `create_registrar(name, protocol, ..., inline_only=False)`. One registrar class per layer. Supports `key_separator` for hierarchical keys, `discriminator_fields` for nested config mode, `inline_only` (exposed as `Registrar.inline_only`; read by composed builds only). `unregister(key, *, missing_ok=False)` returns the removed class. `register(key, propagate=...)` (Path C) undoes exactly its own work on failure; place it above `@dataclass(slots=True)`.
- `registry.py` — `LayerRegistry` key→class storage with thread-safe access and Protocol compliance caching (positive/negative WeakSets inspired by CPython's ABCMeta). `remove(key, *, missing_ok=False)`, `children(prefix)`, `tree()`. Global `_REGISTRY_INDEX` for cross-registry lookups, `_LIVE_REGISTRIES` (shadowed registries included), and the mutation journal used by isolation and `discover()` rollback.
- `auto.py` — `create_auto_registrar()` metaclass factory. Every registration path ends in the shared tail `_register_class`: grammar-check the merged `__wiring__` (`check_wiring_declaration`) → add every key → stamp `__registry_origin__` → install `WiredField` placeholders. Keys are validated first (`InvalidRegistryKeyError`). A failure after the first add purges the class from every live registry. A class re-created by `@dataclass(slots=True)` / `attrs.define` / pydantic slots dataclasses carries the origin stamp and rebinds every key of the original instead of registering again; a class body that runs twice still raises `DuplicateKeyError` (with a hint toward `isolated_registries()`).
- `meta_base.py` — `AutoRegistrarBase` (shared metaclass base) + `MetaRegistrarType` (meta-metaclass). Enables `|` operator for cross-registry diamond inheritance: `metaclass=LLM.Meta | Agent.Meta`. Holds the `inline_only` property of AutoRegistrar metaclasses.
- `filters.py` — Predicate-based filter system. Composable filters: `RootFilter` (a class with no bases is never registered), `PydanticGenericFilter`, `AbstractFilter`, `CustomCallableFilter`, `ChildSkipFilter` (`__skip_registries__`), `ParentRegistrationFilter` (`__registration_filter__`), `PropagationFilter` (`__propagate__`, `__propagate_depth__`).
- `key_transform.py` — CamelCase→snake_case key inference with optional prefix/suffix stripping.
- `isolation.py` — `isolated_registries()` / `RegistryIsolation`: global snapshot/restore of every live registry and the index; keep rule for registrations made by modules first imported inside the block (net effect, last write wins); `kept` / `rolled_back`; `warn_kept` (`IsolationWarning`); capture + `activate()` re-applies the full net diff; LIFO nesting (out-of-order exit unwinds and raises `RuntimeError`); BaseException-safe; restores `discover()`'s failure record.
- `discover.py` — backward-compatibility shim re-exporting `conscribe.discover.discover`.

### Discovery (`conscribe/discover.py`)

`discover(*packages, strict=True, allow_missing=(), exclude=())` imports packages depth-first in name order. Strict: any exception while importing a module propagates (PEP 678 note naming the module on 3.11+, chained `ImportError` for submodules on 3.10). `allow_missing` skips `ModuleNotFoundError` of listed top-level modules, following `__cause__`/`__context__` only through `ImportError` chains (`logging.info` only). `exclude` takes dotted prefixes. `strict=False` skips `ImportError` / `SyntaxError` with an `ExtractionWarning`. Registry changes made by a module whose import failed are rolled back (attributed to the module whose top-level code ran them); already-imported modules cost no bookkeeping; `__main__` submodules are skipped. Failures are recorded as strings: `import_failures()`, `clear_import_failures()` (reach them via `from conscribe.discover import ...`; the package attribute `conscribe.discover` is the function). The 1.x `stub_dir` / `auto_update_stubs` / `fingerprint_path` parameters are gone.

### Wiring (`conscribe/wiring.py`)

- Grammar: three modes — str (all keys of a registry), tuple `(registry, keys)` or `(registry, required_keys, optional_keys)` (explicit subset), list (literal values, no registry). The registry-backed modes may be written as a dict `{"registry", "keys", "optional_keys", "default", "declare_only"}`; unknown fields are `malformed`. Entry names must be identifiers, not keywords, not dunders; a key may not be both required and optional; `declare_only` + `default` is malformed. `None` excludes an inherited entry; a subclass entry replaces the inherited one whole; `__wiring__ = None` declares nothing.
- `WiringSpec` — frozen parse of one entry: `param_name`, `registry_name`, `allowed_keys`, `optional_keys`, `default_key`, `declare_only`. `ResolvedWiring` — concrete keys after lookup (`allowed_keys` = required + optional for `Literal[...]`), `optional_keys`, `default_key`, `declare_only` (`injected` was removed in 2.0).
- `parse_wiring(cls)` / `check_wiring_declaration(cls)` — grammar only, no registry lookups; the latter runs on every registration path before `registry.add`.
- `resolve_wiring(cls)` — resolves one class; `validate_wiring(registrars)` — every registered class of a registrar / mapping / iterable, every problem in one `WiringResolutionError` (deterministic order). Checks unknown registries (did-you-mean), empty registries (cites `import_failures()`), unknown required/optional keys, `bad_default` (default not registered / not allowed / on a slot-less or container receptor / contradicted by the receptor's own default, including parameter defaults that are not allowed keys), `contradiction` (`declare_only` entry with a runtime slot). `build_layer_config`, `build_composed_config` and `write_layer_stubs` call `validate_wiring` first.
- `expand_relative_keys(declared, registry_keys, separator)` — expands a bare trailing segment (`"click"`) into every registered fully-qualified key ending in it (`browser.click`, `desktop.click`). Keys already containing the separator pass through unchanged (escape hatch); unknown short names pass through so the caller's missing-key error still fires; a registry with no separator is a no-op. Multiple matches are ALL kept — conscribe expands, the consumer arbitrates at the point where it knows which providers are loaded. Do not "fix" this by raising on ambiguity. A `default` key is never expanded (it must be spelled exactly as registered).
- `WiredField` — placeholder for an injected attribute. Instance access before injection raises `AttributeError` naming the registry. Class access returns the descriptor, unless the accessing class or a class between it and the holder annotates the name (decided at access time) — then it raises, so dataclass/attrs/pydantic see "no default". `_WiringTombstone` covers an inherited placeholder on a class that excludes the entry (`None`) or redeclares it `declare_only`.
- `inject_wired_descriptors(cls)` — called from every registration path at class creation (before class decorators run). Installs a placeholder unless: the entry is `declare_only`; the class body binds the name; the class body annotates the name; an inherited binding can hold the value (plain value, writable descriptor, `cached_property`, same-registry placeholder); the name is a constructor parameter (`introspect.init_param_names`). Shadows inherited read-only properties, methods, other-registry placeholders and tombstones.
- `wired_field(cls, name)` / `wired_fields(cls)` — placeholder lookups through the MRO `__dict__`s, never invoking descriptors; only names injectable per the merged wiring and not constructor parameters at query time.
- `collect_wiring_from_mro(cls)` — merged raw `__wiring__` dict.

### Introspection (`conscribe/introspect.py`)

Public, pure reflection (no registry lookups, no imports), scope-free (MRO scope/depth only filter what a config exposes), PEP 649-safe. `init_param_names(cls)` follows the runtime `**kwargs` chain (dataclass/attrs generated `__init__`, pydantic fields, bridges); `receptor_slots(cls, names=None)` → `ReceptorSlot(name, via="init"|"annotation", shape="scalar"|"list"|"dict", annotation, declared_on, parameter, default, default_factory)`; `wiring_slots(cls)` → `WiringSlot(name, spec, slot, declare_only, default_key)`; `annotation_shape`, `is_classvar`, `config_from_param`. The extractor, wiring injection, stubs and composed builds all use it — do not add private duplicates.

### Config Typing (`conscribe/config/`)

- Pipeline: **extract** → **build** → **generate**.
- `_annotations.py` — internal annotation engine. `own_annotations(klass)` (PEP 649-safe), `safe_signature()` (no `NameError` on 3.14), owner-aware `resolve_annotation()` returning a `ResolvedAnnotation` (`resolved` / `unannotated` / `degraded` for `type_checking_only` or `function_locals` / `error` for `undefined_name`, `metadata`, `evaluation_failed`). Resolution happens in the namespace of the class that *declares* the annotation and never imports `TYPE_CHECKING` blocks. Shape / `admits_none` / ClassVar / KW_ONLY / InitVar predicates; `warn_at()` attributes warnings to the user's class or function.
- `_fieldinfo.py` — builds fresh `Field(...)` objects from explicit attribute lists (never re-passes a user's `FieldInfo`). `config_field()` carries a parameter's declaration into a config field (refuses `exclude=`, turns `deprecated=` into a JSON-schema marker, enforces one default source, the 2.0 None rule); `inline_field()` / `narrow_keys()` for composed inline receptors.
- `_utils.py` — init-definer rules: `constructor_definer()` / `own_init()` (2.0: Protocol placeholders and copied `__init__` functions are not definers), `find_init_definer()` (1.x rule, only for compat views), pydantic model / dataclass detection.
- `mro.py` — `walk_init_chain(cls, scope=..., depth=...)`: one walker, runtime view (scope-free, for introspection) and config view (cut by scope/depth). Init-less classes are skipped before scope/depth; pydantic models and pydantic dataclasses end the chain; unreadable signatures cut it (`extra="allow"`). `collect_mro_params()` is the 1.x wrapper. `classify_class_scope()`; `MROScope`.
- `extractor.py` — `extract_config_schema(cls, mro_scope, mro_depth)`. Paths: Tier 3 `__config_schema__` (returned by extraction; builds inherit it, default unset `extra` to `"forbid"`, and inline wired fields when composing; wired entries with a runtime slot must be schema fields); `__config_from_param__` (opt-in flattening of a single pydantic-model parameter; the model subclasses it); constructor reflection (Tier 1 / 1.5 / 2). Every path reaches the wiring pass. Defaults: one source; dataclass/attrs/pydantic factories read from the field; pure factories become literal defaults. Wired fields: every non-None, non-container union member maps to the wiring `Literal`, containers to `list[Literal]`; slot-less and `declare_only` entries are `Optional[Literal[...]] = None`; a dict-form default makes the key the default. Classes that 1.x flattened get the `legacy_flat_config` guard (`__legacy_flat_keys__`). Returns `None` only when there is no knob, no receptor and a closed constructor.
- `degradation.py` — Type degradation for Pydantic-incompatible field types (try-first, zero overhead on the happy path); `DegradedField.reason` is `pydantic_incompatible`, `type_checking_only` or `function_locals`.
- `docstring.py` — Tier 1.5 descriptions via the optional `docstring-parser` (empty when it is not installed).
- `builder.py` — `build_layer_config(registrar)`: check the layer (`EmptyLayerError`, nested-mode refusals), `validate_wiring`, then build. Per-key models are built by **subclassing** the source model (`_derive_model`), so validators, `model_config` and FieldInfo survive; conscribe's extra policy is set explicitly. Every model gets `__conscribe_generated__ = True` in its own `__dict__`. Model names are sanitized identifiers, unique per build (`_ModelNames`). Two modes:
  - **Flat mode** (`discriminator_field`): `Annotated[Union[...], Field(discriminator=...)]`.
  - **Nested mode** (`discriminator_fields`): each key segment maps to a nested sub-model; parameters follow the leaf's real constructor (`walk_init_chain`), level = the root-most class whose own constructor accepts the parameter; fields are built by the extractor's `_build_param_fields`. Compound `Discriminator(callable)` + `Tag`; an omitted block selects the `default` segment. `__wiring__`, `__config_schema__`, `__config_from_param__`, `__config_mro_scope__` / `__config_mro_depth__`, keys with the wrong segment count and fields colliding with discriminators raise `ConfigDeclarationError`.
- `composed.py` — `build_composed_config(registrars, *, inline_wiring=True, top_level_extra="forbid")`. Validates layer names and registrars, empty layers, per-layer checks, then `validate_wiring` over all layers, then builds leaves first (`CircularWiringError` names a real cycle; `declare_only` entries add no edge). Inline wiring retypes wired fields into the target layer's union (`__inline_wired__` records them), keeps title/description/deprecated/aliases, narrows keys by key-string constraints, and turns default keys into default instances (`Field(default_factory=<final per-key model>, json_schema_extra={"default": ...})`). `declare_only` entries are never inlined. Inline-only layers get a hidden phantom field + validator raising `inline_only_layer` (`INLINE_ONLY_ERROR_TYPE`) with the receptor paths (`ReceptorLocation`). `ComposedConfigResult` has `inline_only_layers`, `top_level_extra`, `top_level_layers`.
- `_render.py` — shared renderer for generated source: value ladder (literal → Enum member → constructor form verified by `eval` → JSON form + `validate_default` → importable reference by identity → `ConfigRenderError`), every constraint as `Field(...)` kwargs (also inside type args), imports by identity with deterministic aliasing, define-vs-import (`is_generated_model`). Metadata it cannot write (lambda validators) → `RenderNote` → `ExtractionWarning` + header note.
- `codegen.py` — `generate_layer_config_source()` / `generate_composed_config_source()`: layout (header `# Generated by conscribe <version>`, imports, nested submodels, per-key classes, `model_rebuild()`, discriminator functions, union aliases, the composed top level with the inline-only refusal and the legacy-flat guard).
- `json_schema.py` — `generate_layer_json_schema()` (union schema + `x-discriminator`, `x-discriminator-fields`, `x-key-separator`, `x-degraded-fields`) and `generate_composed_json_schema()` (exactly `top_level_type.model_json_schema()`; the top level carries `x-composed-layers`, `x-inline-wiring`, `x-inline-only-layers`). Set defaults are sorted (hash-seed independent).
- `fingerprint.py` — `compute_registry_fingerprint(registrar)`: sha256 (16 hex chars) of `conscribe.__version__`, layer settings that only show in composed output (name, discriminator field(s), separator, `inline_only`), the layer's generated source and JSON Schema. Warnings suppressed; never raises (an exception becomes `error:<Type>:<message without 0x addresses>`). `save_fingerprint` / `load_cached_fingerprint` keep a JSON cache. Stubs and composition options are not hashed; track every layer and `inline_wiring` / `top_level_extra` for a composed build.

**Stubs** (`conscribe/stubs/`):
- `collector.py` — `collect_class_stub_info(cls)`: a wiring entry is stubbed only when it has no runtime slot (`introspect.wiring_slots`: init params incl. inherited / `**kwargs`-forwarded, class annotations) and is not `declare_only`; typed as the target registry's protocol (`str` for a literal list). Wiring problems propagate. `collect_class_members()` gathers the class's own surface (constructor, methods incl. cached / context-manager / wrapped ones, properties, class annotations and values, nested classes). Types: `ClassStubInfo` (`init: MethodStub | None`), `MethodStub` (`function`, `kind`), `ClassAttrStub` (`annotation`, `owner`, `has_default`), `ClassValueStub`, `ClassMembers`, `InjectedAttr`. `narrowest_common_base()` remains as an unused utility.
- `generator.py` — `generate_module_stub(module_name, classes, *, partial_class_fallback=False)`: resolves annotations owner-aware and renders from objects (Annotated metadata dropped, `NoneType` → `None`, short names with verified imports, the declaring module's own import spelling kept, TYPE_CHECKING imports reproduced from source without importing them, same-module objects defined in the stub). The result is gated (parses, binds every name once, imports exist); otherwise `ConfigRenderError` / `ConfigAnnotationError`. Never returns an invalid stub.
- `writer.py` — `write_layer_stubs(registrar, output_dir=None, *, partial_class_fallback=False, prune=False)`: validates wiring and renders every module before writing; a module's stub covers the stub-needing classes defined there from registries currently in the global index plus the requested registrar (other shadowed registries are excluded); multi-key classes rendered once; refuses site-packages / stdlib without `output_dir`; stale conscribe stubs (2.x header or 1.x header) in the affected directories → `StaleStubWarning`, or deleted with `prune=True`. Never deletes anything else.

**Testing** (`conscribe/testing/`): `isolated_registries` / `RegistryIsolation` re-exported from `registration/isolation.py` (the package must not import pytest); `pytest_plugin.py` provides the `conscribe_isolated` fixture and the `conscribe_warn_kept` ini option. No pytest11 entry point.

**Scanner** (`conscribe/scanner.py`):
- `scan_registrar_definitions(root)` — Static AST analysis of `.py` files (excluding site-packages, .venv, etc.) for `create_registrar()` / `create_auto_registrar()` calls. No imports; unreadable files → `ExtractionWarning`.
- `find_packages(root)` — top-level packages under root.
- `list_registries(root, discover_packages, layer_filter, path_filter, allow_missing, exclude)` — imports via strict `discover()`, then `summarize_registries()` reports `RegistrySummary` / `RegistryEntry` per registry (entries filtered to files under root by default).

**CLI** (`conscribe/cli.py`): commands as listed above. Registrars come from `--registrar MODULE:ATTR` and from registrars created by `--discover` imports; `--layer` / `--layers` select by registry name. Config outputs are generated and staged before per-file replacement; replacement failures do not roll back files already replaced. Stub generation renders all modules before writing them directly (no atomic replacement). Errors print `Error: ...` to stderr and exit 1 (no traceback); usage errors exit 2.

### Key Class Attributes (non-obvious conventions)

- `__abstract__ = True` — Marks base classes to skip auto-registration. Checked via `namespace.get()`, not `getattr()`, to avoid inheritance.
- `__registry_key__` — Overrides inferred key. Accepts a non-empty `str` or a non-empty `list[str]` for multi-key registration (anything else: `InvalidRegistryKeyError`). Also set by metaclass after registration.
- `__registry_keys__` — Set by metaclass when multi-key registration is used. List of all keys (read-only).
- `__registry_origin__` — Stamped on every registered class (`id(cls)`); lets the metaclass recognise slots re-creation.
- `__skip_registries__` — List of registry names to skip. Used for selective opt-out in cross-registry diamond inheritance.
- `__registration_filter__` — Parent class `@staticmethod`. Called with child class; return `False` to block registration.
- `__propagate__ = False` — Subclasses don't auto-register through this parent.
- `__propagate_depth__ = N` — Only N levels of subclasses auto-register.
- `__config_schema__` — Tier 3: explicit Pydantic BaseModel; builds inherit its validators and `model_config`, with unset `extra` defaulting to `"forbid"`.
- `__config_from_param__` — Opt-in flattening of one pydantic-model constructor parameter (inherited; `None` opts out).
- `__config_annotated_only__` — Only include parameters declared with a `Field` (in `Annotated` metadata or as the default value); wiring receptors are fields regardless.
- `__config_mro_scope__` / `__config_mro_depth__` — Per-class override for MRO traversal scope (`"local"`, `"third_party"`, `"all"`, list of packages) / depth (int or None). Invalid values raise `ConfigDeclarationError`.
- `__wiring__` — Cross-registry field constraints (see Wiring). Deep-merged along the MRO.
- Model attributes set by conscribe: `__conscribe_generated__` (own `__dict__` marker), `__wired_fields__` (`{field: registry}`, `"literal"` for a literal list; used for `# wired from:` comments), `__wired_keys__` (`{field: keys}`), `__wired_defaults__` (`{field: key}`), `__inline_wired__` (composed retypes), `__degraded_fields__` (`DegradedField` list, only when degradation occurred), `__legacy_flat_keys__` (old-shape guard).

### Wired field *shape* in generated configs

How a `__wiring__` entry lands in the generated config depends on whether the class has a **runtime slot** for it — an `__init__` parameter (including inherited and `**kwargs`-forwarded ones), or a non-`ClassVar` class-level annotation anywhere in the MRO:

| Declaration | Generated config field |
|---|---|
| Slot + scalar annotation | `Literal[...]` (required, or the slot's own default; `Optional[Literal[...]]` when the annotation admits `None`) |
| Slot + container annotation (`dict[str, X]`, `list[X]`, bare `list`) | `list[Literal[...]]` — one selector per wired instance |
| Dict-form `default` (single-instance slot only) | `Literal[...] = '<key>'`; inline: the key's default config instance |
| **No slot at all**, or `declare_only` | `Optional[Literal[...]] = None` |

A slot-less entry names an attribute the framework injects; it is not a required config field. The `Literal` still applies whenever a value **is** supplied. Under composed config with `inline_wiring=True`, `Literal[...]` widens to the target layer union and `list[Literal[...]]` to `list[<target layer union>]`; `declare_only` entries are never inlined.

### Pydantic and dataclass classes

`create_registrar()` and `create_auto_registrar()` accept:
- `skip_pydantic_generic: bool = True` — Filters classes with `[` in name (Pydantic Generic intermediates like `BaseEvent[str]`).
- `skip_filter: Callable[[type], bool] | None = None` — Custom class filter. Return True to skip.

Both filters are part of the predicate filter chain, shared between `AutoRegistrar.__new__` (Path A) and the `register(propagate=True)` hook (Path C).

A registered pydantic model (or pydantic dataclass) whose own `__init__` only forwards `**data` contributes its fields; its config model is rebuilt from those fields on a fresh model, never derived from the registered class, so the class's validators are not part of its config (known limitation; `__config_schema__` is the way to validate at load time).

### 2.0 Invariants (do not break)

- **Fail fast, by taxonomy.** Anything a user declares wrong or contradictory raises, with a message naming the class, the field, the bad value and the fix. Pure grammar errors raise at class creation (`WiringResolutionError` from `check_wiring_declaration`, `InvalidRegistryKeyError`); everything else at build or generation time: `WiringResolutionError` (with `.problems`, aggregated per build), `ConfigAnnotationError`, `ConfigDeclarationError`, `ConfigRenderError`, `EmptyLayerError`. Warnings (`ConscribeWarning` subclasses, emitted at the user's class via `warn_at`) only for what conscribe cannot read (`ExtractionWarning`), type degradation (`DegradationWarning`), stale stubs (`StaleStubWarning`) and kept isolation registrations (`IsolationWarning`). No silent drops, no warn-and-continue for user errors, no bare `except Exception`, no `logging` for user-facing problems. Add new kinds/classes to `exceptions.py` only when truly needed. The fingerprint is the one place that never raises.
- **Never derive a config model from a registered class.** The registrar metaclass would register the config model as an implementation. Build from the extracted schema, a Tier-3 schema or a user's knob model; `builder._check_derivable` guards this.
- **Every registry mutation goes through the journaled methods** (`LayerRegistry.__init__` / `add` / `remove` / `_replace_class` / `_discard_class` / `_discard_binding` / `_restore_binding`, `_deregister`; `register()` journals its attribute writes). Isolation's keep rule and `discover()`'s rollback depend on the journal; a direct write to `_store` or `_REGISTRY_INDEX` breaks both.
- **No plain attributes on AutoRegistrar metaclasses.** Classes created by the metaclass see such an attribute as a class attribute (dataclass and pydantic would read it as a field default). Store state under a `_conscribe_`-prefixed name and expose it as a property on `MetaRegistrarType` (see `inline_only`).
- **The generated-model marker is checked in the class's own `__dict__`** (`vars(model).get("__conscribe_generated__")`), never with `getattr`: a user subclass of a generated model inherits the attribute but must be imported, not redefined.
- **Expand, don't arbitrate.** Capability-relative keys expand to every match; narrowing is the consumer's job.
- **Three-artifact parity.** Preserve field declarations across the in-process model, generated source and JSON Schema; `tests/config/test_parity_matrix.py` (harness in `tests/config/artifact_parity.py`) checks supported shapes. JSON Schema cannot run Python validators. Source rendering emits `ExtractionWarning` plus a header note for unsupported validators; unsupported defaults or types raise `ConfigRenderError`.
- **One introspection rule.** Use `conscribe.introspect` / `walk_init_chain` for "which names reach the constructor" and "where does a wired value go"; never re-derive it locally.
- **PEP 649 (Python 3.14).** Never read `klass.__dict__["__annotations__"]` directly — use `_annotations.own_annotations()`; use `safe_signature()` instead of `inspect.signature()` where annotations may be unresolvable.
- **Determinism.** Generated files, schemas and fingerprints must not depend on hash seeds (`PYTHONHASHSEED`), object addresses or the process that produced them.

### Deferred scope

Per-class `__config_rules__` (F6), nested-mode wiring / Tier-3 support, typed `WiredField[T]` and shared annotation-AST refactors are deferred to 2.1. The sibling `af` repository is outside this release task.

### Known Integration Traps

The 1.x traps are fixed: inherited fields declared in another module under `from __future__ import annotations` resolve in their declaring module (`_annotations.py`), and a wired name annotated on a dataclass no longer gets a `WiredField` as its default (no own placeholder for annotated names; class access raises). What remains:

- An annotation must use names its own module imports at runtime. There is no `typing` fallback namespace any more (a missing `Optional` import is a `ConfigAnnotationError`); names imported only under `if TYPE_CHECKING:` degrade the field to `Any` with a `DegradationWarning`.
- Annotating a wired name makes it a runtime slot: the config uses the slot or wiring default, is optional for an annotation admitting `None` without a value, and is otherwise required. No placeholder is installed in that declaring class, but it may inherit one; class access then raises when no value is present. A later subclass without its own annotation may receive a placeholder. Config/stub slot detection counts annotations anywhere in the MRO, so such names are not stubbed.
- `@register` applied below `@dataclass(slots=True)` / `attrs.define`, and `@pydantic.dataclasses.dataclass` stacked on a stdlib `@dataclass`, are unsupported.
- A Tier-3 schema with `@computed_field` dumps that field, and `extra="forbid"` refuses it on re-validation.
- Validators inherited by inline-wired models see nested configs where the per-layer model passes key strings.
- pydantic 2.11–2.12 are slow on deep inline-wiring chains (2.13+ is fine).

### Bridge Metaclass Conflict Resolution

`bridge()` handles 4 strategies for combining metaclasses when bridging external classes:
1. External is plain `type` → use our Meta
2. Our Meta subclasses external's → use ours
3. External is more specific → use theirs
4. Incompatible → dynamically create combined metaclass

## Dependencies

- Python ≥ 3.10 (tested on 3.10–3.14)
- `pydantic >= 2.7, < 3.0` (core)
- `docstring-parser >= 0.15` (optional, for Tier 1.5 docstring extraction)
- Build: Hatchling (core metadata pinned to 2.4)
