# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Conscribe** (`conscribe` package, v0.5.2) — A Python library for automatic class registration and config typing stub generation for layered Python architectures. It targets framework developers building config-driven frameworks with pluggable layers (agents, LLM providers, etc.).

Two core capabilities:
1. **Auto-registration**: Classes inheriting a base are automatically registered in their layer's registry (via metaclass). Also supports bridging external classes and explicit `@register` decorators.
2. **Config typing**: Extracts `__init__` signatures into Pydantic discriminated unions, generates Python stubs for IDE autocomplete of YAML configs, with fingerprint-based staleness detection.

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
conscribe inspect --layer <name>
```

pytest is pre-configured in `pyproject.toml` with `--cov=conscribe --cov-report=term-missing --tb=short -q`.

## Architecture

### Two Subsystems

**Registration** (`conscribe/registration/`):
- `registrar.py` — `LayerRegistrar` facade created via `create_registrar(name, protocol)`. One instance per layer. Supports `key_separator` for hierarchical keys and `discriminator_fields` for nested config mode.
- `registry.py` — `LayerRegistry` key→class storage with thread-safe access and Protocol compliance caching (positive/negative WeakSets inspired by CPython's ABCMeta). Supports `separator` param, `children(prefix)` and `tree()` for hierarchical key queries.
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
- `codegen.py` — `generate_layer_config_source()` outputs self-contained `.py` stub files. Dispatches to flat or nested generation. Nested mode adds `Discriminator`, `Tag` imports and compound discriminator function.
- `json_schema.py` — `generate_layer_json_schema()` outputs JSON Schema dicts. Adds `x-discriminator-fields` and `x-key-separator` extensions in nested mode.
- `fingerprint.py` — Hashes registry state for auto-freshness detection.

**Discovery** (`conscribe/discover.py`): Recursively imports modules to trigger metaclass registration. Optionally auto-regenerates stubs if fingerprint is stale.

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
