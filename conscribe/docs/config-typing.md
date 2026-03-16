# Config Typing Deep-Dive

This document covers the internals of `conscribe/config/`. For usage, see [Guide: Framework Developer](guide-alice.md) and [Guide: Framework User](guide-bob.md).

## Pipeline Overview

```
extract_config_schema(cls)     # per-class: __init__ -> Pydantic model
    |
build_layer_config(registrar)  # per-layer: models -> discriminated union
    |
    +-> generate_layer_config_source(result)  # Python stub file
    +-> generate_layer_json_schema(result)    # JSON Schema dict
```

## Extraction (`extractor.py`)

`extract_config_schema(cls)` is the core function. It follows a strict priority:

### Priority 1: Explicit `__config_schema__` (Tier 3)

```python
config_schema = getattr(cls, "__config_schema__", None)
if config_schema is not None:
    # Validate it's a BaseModel subclass, return directly
```

### Priority 2: BaseModel Fast Path

When `cls` is a `BaseModel` subclass without custom `__init__`:

```python
if issubclass(cls, BaseModel) and "__init__" not in cls.__dict__:
    # Extract from cls.model_fields instead of __init__
```

This handles Pydantic models where fields are declared as class attributes, not `__init__` parameters.

### Priority 3: Single-Param BaseModel (Tier 3 variant)

If `__init__` has exactly one named parameter whose type is a `BaseModel` subclass, return that type directly.

### Priority 4: `__init__` Signature Reflection (Tier 1/1.5/2)

The main path:

1. **Find `__init__` definer** via MRO walk (skip classes that don't define their own `__init__`)
2. **Get signature** via `inspect.signature()`
3. **Filter params**: exclude `self`, `*args`, `**kwargs`
4. **Get type hints**: try `get_type_hints(include_extras=True)`, fall back to raw `__annotations__`
5. **MRO collection**: if `**kwargs` present, walk MRO upward for parent params (see [MRO and Degradation](mro-and-degradation.md))
6. **Get docstring descriptions**: parse Google/NumPy-style docstrings for Tier 1.5
7. **Build field definitions**: for each param, extract type, default, FieldInfo, and description
8. **Determine extra policy**: `"forbid"` if no `**kwargs` or fully resolved MRO; `"allow"` if truncated
9. **Create model**: `pydantic.create_model()` with try/degrade fallback

### Field Definition Logic

For each parameter:

- **Type**: from type hints (resolved via `get_type_hints`), falling back to raw annotation, then `Any`
- **Default**: from `param.default`, or `...` (required)
- **Description**: Tier 2 `FieldInfo.description` > Tier 1.5 docstring > none
- **Constraints**: copied from `FieldInfo` attributes (`ge`, `gt`, `le`, `lt`, etc.)

### Annotated-Only Mode

When `cls.__config_annotated_only__ = True`, only parameters with `Annotated[..., Field(...)]` are included.

### Degradation Fallback

If `pydantic.create_model()` raises `PydanticSchemaGenerationError`:

```python
try:
    model = _create_dynamic_model(name, fields, extra)
except (PydanticSchemaGenerationError, PydanticUserError):
    fields, degraded = degrade_field_definitions(fields, source_class_name)
    model = _create_dynamic_model(name, fields, extra)
    model.__degraded_fields__ = degraded
```

See [MRO and Degradation](mro-and-degradation.md) for details.

## Building (`builder.py`)

`build_layer_config(registrar)` creates the discriminated union:

1. For each registered key, call `extract_config_schema()` with registrar-level MRO settings
2. If schema is `None`, create a discriminator-only model
3. Inject `Literal[key]` discriminator field with `default=key`
4. Preserve original `extra` policy from extracted schema
5. Collect `__degraded_fields__` from each schema before rebuilding
6. Build union: single key -> model itself; multiple -> `Annotated[Union[...], Field(discriminator=...)]`

### Model Naming

`_build_model_name(key, layer_name)` produces class names:
- Key part: snake_case segments each title-cased (`browser_use` -> `BrowserUse`)
- Layer part: <=3 chars -> ALL_CAPS (`llm` -> `LLM`); >3 chars -> Title (`agent` -> `Agent`)
- Combined: `BrowserUseAgentConfig`, `OpenAiLLMConfig`

## Code Generation (`codegen.py`)

`generate_layer_config_source(result)` outputs a self-contained Python file:

1. **Header**: auto-generated comment with layer info, degradation warnings
2. **Future imports**: `from __future__ import annotations`
3. **Import block**: typing + pydantic imports (only what's needed)
4. **Per-key classes**: sorted by key, with `model_config`, fields, inline degradation comments
5. **`model_rebuild()` calls**: needed because `from __future__ import annotations` defers type evaluation
6. **Union alias**: `LayerConfig = Annotated[Union[...], Field(discriminator=...)]` (if multiple models)

### Field Rendering

Each field is rendered as one of:
- `name: type = value` (plain default)
- `name: type = ...` (required)
- `name: type = Field(default, description=..., ge=...)` (with metadata)
- `name: type = value  # degraded from: original_type` (degraded)

## JSON Schema (`json_schema.py`)

`generate_layer_json_schema(result)` uses Pydantic's `TypeAdapter.json_schema()` and adds:

- `x-discriminator`: the discriminator field name
- `x-degraded-fields`: top-level dict `{key: [{field, original_type}]}`
- Per-property `x-degraded-from` and `description` annotations inside `$defs`

## Fingerprinting (`fingerprint.py`)

`compute_registry_fingerprint(registrar)` produces a 16-char hex hash of:

- Sorted registry keys
- Each class's qualname
- Each class's `__init__` signature (or `model_fields` for BaseModel)
- Each class's docstring
- Parent class signatures when `**kwargs` is present (MRO chain)

### Caching

- `save_fingerprint(path, layer, fp)`: writes to JSON `{layer: fp}`
- `load_cached_fingerprint(path, layer)`: reads from JSON, returns `None` on miss/corrupt

Used by `discover()` for auto-freshness: compare current vs cached, regenerate if different.
