# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Conscribe** (`conscribe` package, v0.9.0) — A Python library for automatic class registration and config typing stub generation for layered Python architectures. It targets framework developers building config-driven frameworks with pluggable layers (agents, LLM providers, etc.).

Five core capabilities:
1. **Auto-registration**: Classes inheriting a base are automatically registered in their layer's registry (via metaclass). Also supports bridging external classes and explicit `@register` decorators.
2. **Config typing**: Extracts `__init__` signatures into Pydantic discriminated unions, generates Python stubs for IDE autocomplete of YAML configs, with fingerprint-based staleness detection.
3. **Cross-registry wiring**: Classes declare `__wiring__` to reference other registries, constraining config fields to `Literal[...]` with auto-discovery or explicit subsets. Enables Spring IoC-style dependency declarations between layers.
4. **`.pyi` stub generation**: Generates PEP 484 `.pyi` files for classes with injected wired attributes. Enables IDE autocomplete (PyCharm, VS Code, mypy) for runtime-injected dependencies. Uses type narrowing: wired attributes are typed as the target registry's protocol/base class, narrowed to the most specific common ancestor when constrained to a subset.
5. **Composed config**: Combines multiple layers into a single JSON Schema / Python source with inline wiring — wired fields become the target layer's full discriminated union type instead of `Literal[...]`. Enables recursive IDE autocompletion across layers when editing YAML configs.

## Commands

```bash
# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/config/test_extractor.py

# Run a single test
uv run pytest tests/config/test_extractor.py::TestExtractConfigSchema::test_basic

# Install for development
pip install -e ".[dev,docstring]"

# CLI
conscribe generate-config --layer <name> --output <path>
conscribe generate-composed-config --layers <name1> <name2> [--format json-schema|python] [--no-inline-wiring] --output <path>
conscribe generate-stubs --layer <name> --output-dir <path>
conscribe inspect --layer <name>
conscribe scan [--path <dir>]
conscribe list [--discover <pkg> ...] [--layer <name>] [--path <dir>]
```

pytest is pre-configured in `pyproject.toml` with `--cov=conscribe --cov-report=term-missing --tb=short -q`.

## Architecture

### Two Subsystems

**Registration** (`conscribe/registration/`):
- `registrar.py` — `LayerRegistrar` facade created via `create_registrar(name, protocol)`. One instance per layer. Supports `key_separator` for hierarchical keys and `discriminator_fields` for nested config mode.
- `registry.py` — `LayerRegistry` key→class storage with thread-safe access and Protocol compliance caching (positive/negative WeakSets inspired by CPython's ABCMeta). Supports `separator` param, `children(prefix)` and `tree()` for hierarchical key queries. Global `_REGISTRY_INDEX` enables cross-registry lookups for wiring resolution.
- `auto.py` — `create_auto_registrar()` metaclass factory. Intercepts `__new__` to auto-register on class definition. Uses predicate filter chain for skip conditions. Supports hierarchical key derivation and multi-key registration (`__registry_key__ = [...]`).
- `meta_base.py` — `AutoRegistrarBase` (shared metaclass base) + `MetaRegistrarType` (meta-metaclass). Enables `|` operator for cross-registry diamond inheritance: `metaclass=LLM.Meta | Agent.Meta`.
- `filters.py` — Predicate-based filter system. Composable filters: `RootFilter`, `PydanticGenericFilter`, `AbstractFilter`, `CustomCallableFilter`, `ChildSkipFilter` (`__skip_registries__`), `ParentRegistrationFilter` (`__registration_filter__`), `PropagationFilter` (`__propagate__`, `__propagate_depth__`).
- `key_transform.py` — CamelCase→snake_case key inference with optional prefix/suffix stripping.

**Config Typing** (`conscribe/config/`):
- Pipeline: **extract** → **build** → **generate**
- `extractor.py` — `extract_config_schema()` reflects `__init__` signatures into Pydantic models. Three tiers: Tier 1 (types+defaults), Tier 1.5 (+docstring descriptions), Tier 2 (+Annotated metadata), Tier 3 (explicit `__config_schema__`). MRO-aware. Also provides `extract_own_init_params()` for per-class param extraction (used by nested builder).
- `mro.py` — `collect_mro_params()` walks the MRO upward when `__init__` has `**kwargs`, collecting parent parameters. `classify_class_scope()` determines whether a class is local/third_party/stdlib. `MROScope` type alias (`"local" | "third_party" | "all"`).
- `degradation.py` — Type degradation for Pydantic-incompatible field types. Try-first approach: zero overhead on happy path.
- `builder.py` — Two builder paths:
  - **Flat mode** (`discriminator_field`): `Annotated[Union[...], Field(discriminator=...)]`. Original behavior.
  - **Nested mode** (`discriminator_fields`): Deeply nested compound discrimination. Each key segment maps to a nested Pydantic sub-model. Level 0 params are flat, level 1+ are nested. Uses `Discriminator(callable)` + `Tag(key)`. `_extract_params_by_level()` splits params by MRO level.
- `codegen.py` — `generate_layer_config_source()` outputs self-contained `.py` stub files. Dispatches to flat or nested generation. Nested mode adds `Discriminator`, `Tag` imports and compound discriminator function. Also `generate_composed_config_source()` for composed config.
- `json_schema.py` — `generate_layer_json_schema()` outputs JSON Schema dicts. Adds `x-discriminator-fields` and `x-key-separator` extensions in nested mode. Also `generate_composed_json_schema()` for composed config.
- `composed.py` — `build_composed_config(registrars, inline_wiring)` orchestrates multi-layer config. Builds dependency graph via wiring declarations, topologically sorts layers, builds each layer, then post-processes wired `Literal[...]` fields into target layer union types. Produces `ComposedConfigResult` with `top_level_type` (Pydantic model with `list[union]` per layer). Raises `CircularWiringError` on cycles.
- `fingerprint.py` — Hashes registry state for auto-freshness detection.

**Wiring** (`conscribe/wiring.py`):
- `WiringSpec` — Frozen dataclass for parsed `__wiring__` entries. Three modes: str (auto-discover all keys from registry), tuple (explicit subset from registry), list (literal list without registry). Mode 2 tuple supports 3-element form `(registry, required_keys, optional_keys)` to distinguish required vs optional keys. Fields: `allowed_keys` (required keys or all keys), `optional_keys` (optional subset, `None` for non-3-element modes).
- `ResolvedWiring` — Concrete key list after registry lookup, with `injected` flag for fields not in `__init__`. `allowed_keys` contains required+optional combined for backward-compatible `Literal[...]` generation. `optional_keys` stores the optional subset (`None` when not using 3-element tuple mode).
- `collect_wiring_from_mro(cls)` — Walks MRO bottom-up to deep-merge `__wiring__` dicts. `None` value excludes inherited keys.
- `parse_wiring(cls)` — Normalizes `__wiring__` dict to `WiringSpec` list.
- `resolve_wiring(cls)` — Resolves specs to concrete key lists via `get_registry()`. Raises `WiringResolutionError` on failures.

**Stubs** (`conscribe/stubs/`):
- `collector.py` — `collect_class_stub_info(cls)` reflects `__init__` signatures and wiring to produce `ClassStubInfo`. Only returns info for classes with *injected* wired fields (not in `__init__`). `narrowest_common_base(classes, fallback)` computes the most specific common ancestor via MRO intersection for type narrowing.
- `generator.py` — `generate_module_stub(module_name, classes)` renders `.pyi` source. Uses `def __getattr__(name: str) -> Any: ...` partial-stub pattern. Instance attributes typed as registry protocol (Mode 1), narrowed common base (Mode 2), or `str` (Mode 3).
- `writer.py` — `write_layer_stubs(registrar, output_dir)` groups classes by source module, generates and writes `.pyi` files. Default: alongside source `.py` files.

**Discovery** (`conscribe/discover.py`): Recursively imports modules to trigger metaclass registration. Optionally auto-regenerates stubs if fingerprint is stale.

**Scanner** (`conscribe/scanner.py`):
- `scan_registrar_definitions(root)` — Static AST analysis. Walks `.py` files (excluding site-packages, .venv, etc.) to find `create_registrar()` / `create_auto_registrar()` calls. Returns `RegistrarDefinition` (name, protocol, file, line, variable). No imports, no side effects.
- `find_packages(root)` — Detects top-level Python packages (dirs with `__init__.py`) under root, excluding common non-project dirs.
- `list_registries(root, discover_packages, layer_filter, path_filter)` — Runtime inspection. Imports packages via `discover()`, queries `_REGISTRY_INDEX`, returns `RegistrySummary` per registry with `RegistryEntry` per class (key, class name, source file, line). Filters entries to files under root by default (excludes site-packages). Supports `layer_filter` (single registry) and `path_filter` (source path prefix).

### Key Class Attributes (non-obvious conventions)

- `__abstract__ = True` — Marks base classes to skip auto-registration. Checked via `namespace.get()`, not `getattr()`, to avoid inheritance.
- `__registry_key__` — Overrides inferred key. Accepts `str` or `list[str]` for multi-key registration. Also set by metaclass after registration.
- `__registry_keys__` — Set by metaclass when multi-key registration is used. List of all keys (read-only).
- `__skip_registries__` — List of registry names to skip. Used for selective opt-out in cross-registry diamond inheritance.
- `__registration_filter__` — Parent class `@staticmethod`. Called with child class; return `False` to block registration.
- `__propagate__ = False` — Subclasses don't auto-register through this parent.
- `__propagate_depth__ = N` — Only N levels of subclasses auto-register.
- `__config_schema__` — Tier 3 escape hatch: explicit Pydantic BaseModel for full control.
- `__config_annotated_only__` — Only include parameters with `Annotated[..., Field(...)]`.
- `__config_mro_scope__` — Per-class override for MRO traversal scope (`"local"`, `"third_party"`, `"all"`).
- `__config_mro_depth__` — Per-class override for MRO traversal depth (int or None).
- `__degraded_fields__` — Attached to dynamically created models by `extract_config_schema()` when field types were degraded to `Any`. List of `DegradedField` instances. Only present when degradation occurred (zero overhead on happy path).
- `__wiring__` — Cross-registry field constraints. Dict mapping param names to registry references. Three modes: `{"loop": "agent_loop"}` (all keys), `{"loop": ("agent_loop", ["react"])}` (subset), `{"browser": ["chromium"]}` (literal list). Mode 2 also supports 3-element tuple: `{"obs": ("observation", ["terminal"], ["filesystem"])}` (required + optional). `None` value excludes inherited key. Deep-merged along MRO.
- `__wired_fields__` — Attached to dynamically created models by `extract_config_schema()` when wiring was applied. Dict mapping field names to registry names. Used by codegen for `# wired from:` comments.

### Pydantic Generic Compatibility

`create_registrar()` and `create_auto_registrar()` accept:
- `skip_pydantic_generic: bool = True` — Filters classes with `[` in name (Pydantic Generic intermediates like `BaseEvent[str]`).
- `skip_filter: Callable[[type], bool] | None = None` — Custom class filter. Return True to skip.

Both filters are part of the predicate filter chain, shared between `AutoRegistrar.__new__` (Path A) and `_inject_auto_registration` (Path C propagate).

`extract_config_schema()` has a **BaseModel fast path**: when `cls` is a `BaseModel` subclass without custom `__init__`, fields are extracted from `cls.model_fields` instead of reflecting `__init__` (which would resolve to `BaseModel.__init__` with zero named params).

`compute_registry_fingerprint()` has a **BaseModel-aware hashing** branch: hashes `model_fields` instead of `__init__` signature for BaseModel subclasses.

### Bridge Metaclass Conflict Resolution

`bridge()` handles 4 strategies for combining metaclasses when bridging external classes:
1. External is plain `type` → use our Meta
2. Our Meta subclasses external's → use ours
3. External is more specific → use theirs
4. Incompatible → dynamically create combined metaclass

## Dependencies

- Python ≥ 3.9
- `pydantic >= 2.0, < 3.0` (core)
- `docstring-parser >= 0.15` (optional, for Tier 1.5 docstring extraction)
- Build: Hatchling
