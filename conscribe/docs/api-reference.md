# API Reference

All public APIs are exported from `conscribe` (i.e., `from conscribe import ...`).

## Registration API

### `create_registrar(name, protocol, **kwargs) -> type`

The recommended entry point. Creates a `LayerRegistrar` subclass with all components wired together.

```python
from conscribe import create_registrar

LLMRegistrar = create_registrar(
    "llm",
    ChatModelProtocol,
    discriminator_field="provider",
    strip_prefixes=["Chat"],
    mro_scope="local",
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | required | Layer name (e.g. `"agent"`, `"llm"`) |
| `protocol` | `type` | required | `@runtime_checkable` Protocol class |
| `discriminator_field` | `str` | `""` | Config union discriminator field name |
| `strip_suffixes` | `list[str] \| None` | `None` | Suffixes to strip during key inference |
| `strip_prefixes` | `list[str] \| None` | `None` | Prefixes to strip during key inference |
| `key_transform` | `KeyTransform \| None` | `None` | Custom key inference function (overrides strip_*) |
| `base_metaclass` | `type` | `type` | Parent metaclass for AutoRegistrar |
| `mro_scope` | `MROScope` | `"local"` | Scope for MRO parameter collection |
| `mro_depth` | `int \| None` | `None` | Max MRO levels (None = unlimited) |
| `skip_pydantic_generic` | `bool` | `True` | Filter classes with `[` in name |
| `skip_filter` | `Callable[[type], bool] \| None` | `None` | Custom skip filter |

**Returns:** A `LayerRegistrar` subclass.

**Raises:** `InvalidProtocolError` if protocol is not `@runtime_checkable`.

---

### `LayerRegistrar` (class)

Created by `create_registrar()`. All methods are classmethods.

#### `LayerRegistrar.get(key: str) -> type`

Look up a registered class by key.

**Raises:** `KeyNotFoundError` if key is not registered.

#### `LayerRegistrar.get_or_none(key: str) -> type | None`

Safe lookup. Returns `None` if key is not found.

#### `LayerRegistrar.get_all() -> dict[str, type]`

Return a snapshot dict of all `{key: class}` mappings.

#### `LayerRegistrar.keys() -> list[str]`

Return all registered keys.

#### `LayerRegistrar.unregister(key: str) -> None`

Remove a registration. Intended for test isolation.

#### `LayerRegistrar.bridge(external_class, *, name=None) -> type`

Create a bridge base class for an external class. Automatically resolves metaclass conflicts.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `external_class` | `type` | required | External class to bridge |
| `name` | `str \| None` | `None` | Bridge class name (default: `"{ClassName}Bridge"`) |

**Returns:** A bridge class marked `__abstract__=True` whose subclasses auto-register.

#### `LayerRegistrar.register(key=None, *, propagate=False) -> Callable`

Manual registration decorator with Protocol compliance check.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `key` | `str \| None` | `None` | Registration key (inferred if None) |
| `propagate` | `bool` | `False` | If True, subclasses also auto-register |

**Raises:** `DuplicateKeyError`, `ProtocolViolationError`.

#### `LayerRegistrar.build_config() -> LayerConfigResult`

Build discriminated union config from all registered classes.

#### `LayerRegistrar.config_union_type() -> type`

Get the discriminated union type for config validation.

#### `LayerRegistrar.get_config_schema(key: str) -> type[BaseModel] | None`

Get the config schema for a specific registered key.

---

### `discover(*package_paths, **kwargs) -> list[str]`

Recursively import all modules under the given packages to trigger registration.

```python
from conscribe import discover

imported = discover("my_app.agents", "my_app.llm_providers")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `*package_paths` | `str` | required | Dotted package paths to discover |
| `auto_update_stubs` | `bool` | `True` | Check fingerprints and regenerate stubs if stale |
| `stub_dir` | `Path \| None` | `None` | Directory for generated stubs (None = skip auto-freshness) |
| `fingerprint_path` | `Path \| None` | `None` | Path to fingerprint cache JSON (default: `stub_dir/.registry_fingerprint`) |

**Returns:** List of successfully imported module names.

**Raises:** `ModuleNotFoundError` if a top-level package doesn't exist.

---

### Supporting Types

#### `KeyTransform` (Protocol)

```python
class KeyTransform(Protocol):
    def __call__(self, class_name: str) -> str: ...
```

#### `default_key_transform(class_name: str) -> str`

Pure CamelCase -> snake_case conversion. Used when no key inference params are specified.

#### `make_key_transform(*, suffixes=None, prefixes=None) -> KeyTransform`

Factory for key transforms with suffix/prefix stripping. Suffix stripping happens first, then prefix, then CamelCase -> snake_case.

---

## Config Typing API

### `extract_config_schema(cls, mro_scope="local", mro_depth=None) -> type[BaseModel] | None`

Extract a Pydantic model from a class's `__init__` signature.

**Extraction priority:**
1. `cls.__config_schema__` -> return directly (Tier 3)
2. BaseModel subclass without custom `__init__` -> extract from `model_fields`
3. Single-param BaseModel in `__init__` -> return that type (Tier 3 variant)
4. `__init__` signature reflection -> dynamic Pydantic model (Tier 1/1.5/2)
5. No extractable params -> `None`

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cls` | `type` | required | Class to extract from |
| `mro_scope` | `MROScope` | `"local"` | Scope for MRO traversal (overridden by `__config_mro_scope__`) |
| `mro_depth` | `int \| None` | `None` | Max MRO levels (overridden by `__config_mro_depth__`) |

**Returns:** A Pydantic `BaseModel` subclass, or `None`.

---

### `build_layer_config(registrar) -> LayerConfigResult`

Build a discriminated union from all registered classes.

For each key: extract schema, inject `Literal[key]` discriminator, preserve extra policy. Single key -> model itself. Multiple keys -> `Annotated[Union[...], Field(discriminator=...)]`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `registrar` | `type` | A `LayerRegistrar` subclass |

**Returns:** `LayerConfigResult`

**Raises:** `ValueError` if registrar has no `discriminator_field`.

---

### `LayerConfigResult` (dataclass)

```python
@dataclass(frozen=True)
class LayerConfigResult:
    union_type: Any                              # The discriminated union type
    per_key_models: dict[str, type[BaseModel]]   # Key -> per-key model
    layer_name: str                              # Layer name
    discriminator_field: str                     # Discriminator field name
    degraded_fields: dict[str, list[DegradedField]]  # Key -> degraded fields (empty if none)
```

---

### `generate_layer_config_source(result) -> str`

Generate a self-contained Python source file from a `LayerConfigResult`.

Output structure: header -> imports -> per-key classes -> model_rebuild() calls -> union alias.

---

### `generate_layer_json_schema(result) -> dict`

Generate a JSON Schema dict from a `LayerConfigResult`. Includes `x-discriminator` extension and `x-degraded-fields` when degradation occurred.

---

### `compute_registry_fingerprint(registrar) -> str`

Compute a SHA-256 based fingerprint of all registered classes (keys, qualnames, signatures, docstrings). Returns a 16-character hex string. BaseModel-aware: hashes `model_fields` for BaseModel subclasses.

---

### `load_cached_fingerprint(fingerprint_path, layer_name) -> str | None`

Load a cached fingerprint from a JSON file.

### `save_fingerprint(fingerprint_path, layer_name, fingerprint) -> None`

Save a fingerprint to a JSON cache file, preserving other layers.

---

### `MROScope` (type alias)

```python
MROScope = Literal["local", "third_party", "all"] | list[str]
```

---

### `DegradedField` (dataclass)

```python
@dataclass(frozen=True)
class DegradedField:
    field_name: str           # Parameter name (e.g., "auth")
    original_type_repr: str   # Human-readable original type
    source_class: str         # Fully-qualified class name
    reason: str = "pydantic_incompatible"
```

---

## Exceptions

All inherit from `RegistryError`.

| Exception | Inherits | When |
|-----------|----------|------|
| `RegistryError` | `Exception` | Base for all registry exceptions |
| `DuplicateKeyError` | `RegistryError` | Same key registered twice in a layer |
| `KeyNotFoundError` | `RegistryError`, `KeyError` | `get()` with unknown key |
| `ProtocolViolationError` | `RegistryError`, `TypeError` | Class missing required protocol methods |
| `InvalidConfigSchemaError` | `RegistryError`, `TypeError` | `__config_schema__` is not a BaseModel |
| `InvalidProtocolError` | `RegistryError`, `TypeError` | Protocol not `@runtime_checkable` |

### Class Attributes for Configuration

| Attribute | Type | Purpose |
|-----------|------|---------|
| `__abstract__` | `bool` | Skip registration (checked via `namespace.get()`, not inherited) |
| `__registry_key__` | `str` | Override inferred registry key |
| `__config_schema__` | `type[BaseModel]` | Tier 3: explicit Pydantic model |
| `__config_annotated_only__` | `bool` | Only include `Annotated[..., Field()]` params |
| `__config_mro_scope__` | `MROScope` | Per-class MRO scope override |
| `__config_mro_depth__` | `int \| None` | Per-class MRO depth override |
