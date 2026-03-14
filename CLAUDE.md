# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Conscribe** (`conscribe` package, v0.1.2) — A Python library for automatic class registration and config typing stub generation for layered Python architectures. It targets framework developers building config-driven frameworks with pluggable layers (agents, LLM providers, etc.).

Two core capabilities:
1. **Auto-registration**: Classes inheriting a base are automatically registered in their layer's registry (via metaclass). Also supports bridging external classes and explicit `@register` decorators.
2. **Config typing**: Extracts `__init__` signatures into Pydantic discriminated unions, generates Python stubs for IDE autocomplete of YAML configs, with fingerprint-based staleness detection.

## Commands

```bash
# Run all tests with coverage
pytest

# Run a single test file
pytest tests/config/test_extractor.py

# Run a single test
pytest tests/config/test_extractor.py::TestExtractConfigSchema::test_basic

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
- `registrar.py` — `LayerRegistrar` facade created via `create_registrar(name, protocol)`. One instance per layer.
- `registry.py` — `LayerRegistry` key→class storage with thread-safe access and Protocol compliance caching (positive/negative WeakSets inspired by CPython's ABCMeta).
- `auto.py` — `create_auto_registrar()` metaclass factory. Intercepts `__new__` to auto-register on class definition. Skips root base and classes with `__abstract__ = True`.
- `key_transform.py` — CamelCase→snake_case key inference with optional prefix/suffix stripping.

**Config Typing** (`conscribe/config/`):
- Pipeline: **extract** → **build** → **generate**
- `extractor.py` — `extract_config_schema()` reflects `__init__` signatures into Pydantic models. Three tiers: Tier 1 (types+defaults), Tier 1.5 (+docstring descriptions), Tier 2 (+Annotated metadata), Tier 3 (explicit `__config_schema__`). MRO-aware. Supports `mro_scope` and `mro_depth` params for `**kwargs` chain resolution.
- `mro.py` — `collect_mro_params()` walks the MRO upward when `__init__` has `**kwargs`, collecting parent parameters. `classify_class_scope()` determines whether a class is local/third_party/stdlib. `MROScope` type alias (`"local" | "third_party" | "all"`).
- `builder.py` — `build_layer_config()` creates discriminated unions by injecting `Literal[key]` discriminator fields. Passes registrar-level `mro_scope`/`mro_depth` to extraction.
- `codegen.py` — `generate_layer_config_source()` outputs self-contained `.py` stub files.
- `fingerprint.py` — Hashes registry state (keys, qualnames, signatures, docstrings) for auto-freshness detection. When `**kwargs` is present, also hashes parent class signatures along the MRO chain.

**Discovery** (`conscribe/discover.py`): Recursively imports modules to trigger metaclass registration. Optionally auto-regenerates stubs if fingerprint is stale.

### Key Class Attributes (non-obvious conventions)

- `__abstract__ = True` — Marks base classes to skip auto-registration. Checked via `namespace.get()`, not `getattr()`, to avoid inheritance.
- `__registry_key__` — Overrides inferred key. Also set by metaclass after registration.
- `__config_schema__` — Tier 3 escape hatch: explicit Pydantic BaseModel for full control.
- `__config_annotated_only__` — Only include parameters with `Annotated[..., Field(...)]`.
- `__config_mro_scope__` — Per-class override for MRO traversal scope (`"local"`, `"third_party"`, `"all"`).
- `__config_mro_depth__` — Per-class override for MRO traversal depth (int or None).

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
