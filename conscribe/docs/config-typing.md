# Config Typing Deep-Dive

This document covers the internals of `conscribe/config/`. For usage, see [Guide: Framework Developer](guide-alice.md) and [Guide: Framework User](guide-bob.md).

## Pipeline Overview

```
extract_config_schema(cls)     # per-class: __init__ -> Pydantic model
    |
build_layer_config(registrar)  # per-layer: models -> discriminated union
    |                            (flat mode OR nested mode)
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

### Priority 3: Single-Param BaseModel (Tier 3 variant)

If `__init__` has exactly one named parameter whose type is a `BaseModel` subclass, return that type directly.

### Priority 4: `__init__` Signature Reflection (Tier 1/1.5/2)

The main path:

1. **Find `__init__` definer** via MRO walk
2. **Get signature** via `inspect.signature()`
3. **Filter params**: exclude `self`, `*args`, `**kwargs`
4. **Get type hints**: try `get_type_hints(include_extras=True)`, fall back to raw `__annotations__`
5. **MRO collection**: if `**kwargs` present, walk MRO upward for parent params (see [MRO and Degradation](mro-and-degradation.md))
6. **Get docstring descriptions**: parse Google/NumPy-style docstrings for Tier 1.5
7. **Build field definitions**: for each param, extract type, default, FieldInfo, and description
8. **Determine extra policy**: `"forbid"` if no `**kwargs` or fully resolved MRO; `"allow"` if truncated
9. **Create model**: `pydantic.create_model()` with try/degrade fallback

### `extract_own_init_params(cls)`

Helper function that extracts parameters defined in `cls`'s own `__init__` (NOT inherited). Used by the nested config builder to split params by MRO level.

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

`build_layer_config(registrar)` dispatches to one of two paths:

### Flat Mode (`discriminator_field`)

Original behavior. For each registered key:

1. Call `extract_config_schema()` with registrar-level MRO settings
2. If schema is `None`, create a discriminator-only model
3. Inject `Literal[key]` discriminator field with `default=key`
4. Preserve original `extra` policy
5. Collect `__degraded_fields__` before rebuilding
6. Build union: single key -> model itself; multiple -> `Annotated[Union[...], Field(discriminator=...)]`

### Nested Mode (`discriminator_fields`)

Activated when the registrar has `discriminator_fields` set. Generates **nested Pydantic models** where each hierarchy level is a separate sub-model.

#### Algorithm

1. Filter to leaf keys (segment count == `len(discriminator_fields)`)
2. For each leaf key, call `_extract_params_by_level()` to split params by MRO level
3. Build nested segment models (level 1+) with `name: Literal[segment]` discriminator
4. Build combined models with level 0 params flat + level 1 as nested sub-model
5. Build compound discriminator function + union with `Discriminator(callable)` + `Tag(key)`

#### Param Splitting by MRO Level

`_extract_params_by_level(cls, all_classes, separator)` walks the MRO from root to leaf. Each ancestor with `__registry_key__` maps to a key segment level. Params are assigned to the **first class that defines them** (root → leaf):

```
MRO:     object ← Base ← OpenAIProtocol ← RetryMixin ← AzureOpenAI
Keys:                     "openai"                      "openai.azure"
Level:                    0 (model_type)                 1 (provider)

Params:  temperature → level 0 (from OpenAIProtocol)
         max_retries → level 1 (from RetryMixin, unregistered → nearest leaf-ward)
         deployment  → level 1 (from AzureOpenAI)
```

#### Generated Model Structure (Hybrid Format)

Level 0 (first discriminator) and its params are **flat** at the top level. Level 1+ are **nested sub-models**:

```python
class OpenaiAzureLLMConfig(BaseModel):
    model_type: Literal["openai"] = "openai"  # flat (level 0)
    temperature: float = 0.7                    # flat (level 0 param)
    provider: AzureProviderConfig               # nested (level 1)

class AzureProviderConfig(BaseModel):
    name: Literal["azure"] = "azure"
    deployment: str
    api_version: str = "2024-02"
```

### Model Naming

`_build_model_name(key, layer_name)` produces class names:
- Key part: segments each title-cased (`browser_use` -> `BrowserUse`, `openai.azure` -> `OpenaiAzure`)
- Layer part: <=3 chars -> ALL_CAPS (`llm` -> `LLM`); >3 chars -> Title (`agent` -> `Agent`)
- Combined: `BrowserUseAgentConfig`, `OpenaiAzureLLMConfig`

## Code Generation (`codegen.py`)

`generate_layer_config_source(result)` dispatches to flat or nested generation.

### Flat Mode Output

Self-contained Python file:

1. **Header**: auto-generated comment with layer info, degradation warnings
2. **Future imports**: `from __future__ import annotations`
3. **Import block**: typing + pydantic imports (only what's needed)
4. **Per-key classes**: sorted by key, with `model_config`, fields, inline degradation comments
5. **`model_rebuild()` calls**: needed because `from __future__ import annotations` defers type evaluation
6. **Union alias**: `LayerConfig = Annotated[Union[...], Field(discriminator=...)]` (if multiple models)

### Nested Mode Output

1. **Header**: includes `Discriminator fields:` and `Key separator:` metadata
2. **Imports**: adds `Discriminator`, `Tag` from pydantic
3. **Nested segment models** (level 1+): e.g., `AzureProviderConfig`
4. **Combined models** (level 0 flat + level 1 nested): e.g., `OpenaiAzureLLMConfig`
5. **`model_rebuild()` calls**
6. **Compound discriminator function**: `_discriminate_{layer}(v)` that walks nested dicts/models
7. **Union alias**: uses `Discriminator(fn)` + `Tag(key)` instead of `Field(discriminator=...)`

## JSON Schema (`json_schema.py`)

`generate_layer_json_schema(result)` uses Pydantic's `TypeAdapter.json_schema()` and adds:

- `x-discriminator`: the discriminator field name
- `x-discriminator-fields`: list of discriminator field names (nested mode)
- `x-key-separator`: the key separator (nested mode)
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
