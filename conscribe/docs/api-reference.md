# API Reference

All public APIs are exported from `conscribe` (i.e., `from conscribe import ...`).

## Registration API

### `create_registrar(name, protocol, **kwargs) -> type`

The recommended entry point. Creates a `LayerRegistrar` subclass with all components wired together.

```python
from conscribe import create_registrar

# Flat mode (single discriminator)
LLMRegistrar = create_registrar(
    "llm",
    ChatModelProtocol,
    discriminator_field="provider",
    strip_prefixes=["Chat"],
    mro_scope="local",
)

# Nested mode (compound discriminator)
LLM = create_registrar(
    "llm",
    ChatModelProtocol,
    discriminator_fields=["model_type", "provider"],
    key_separator=".",
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | required | Layer name (e.g. `"agent"`, `"llm"`) |
| `protocol` | `type` | required | `@runtime_checkable` Protocol class |
| `discriminator_field` | `str` | `""` | Config union discriminator field name (flat mode) |
| `discriminator_fields` | `list[str] \| None` | `None` | Discriminator field names (nested mode). Mutually exclusive with `discriminator_field`. Requires `key_separator`. |
| `key_separator` | `str` | `""` | Separator for hierarchical keys (e.g. `"."`) |
| `strip_suffixes` | `list[str] \| None` | `None` | Suffixes to strip during key inference |
| `strip_prefixes` | `list[str] \| None` | `None` | Prefixes to strip during key inference |
| `key_transform` | `KeyTransform \| None` | `None` | Custom key inference function (overrides strip_*) |
| `base_metaclass` | `type` | `type` | Parent metaclass for AutoRegistrar |
| `mro_scope` | `MROScope` | `"local"` | Scope for MRO parameter collection |
| `mro_depth` | `int \| None` | `None` | Max MRO levels (None = unlimited) |
| `skip_pydantic_generic` | `bool` | `True` | Filter classes with `[` in name |
| `skip_filter` | `Callable[[type], bool] \| None` | `None` | Custom skip filter |

**Returns:** A `LayerRegistrar` subclass.

**Raises:**
- `InvalidProtocolError` if protocol is not `@runtime_checkable`.
- `ValueError` if both `discriminator_field` and `discriminator_fields` are set.
- `ValueError` if `discriminator_fields` is set without `key_separator`.

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

#### `LayerRegistrar.children(prefix: str) -> dict[str, type]`

Return all entries whose key starts with `prefix + separator`. Only meaningful when `key_separator` is set.

```python
LLM.children("openai")
# → {"openai.azure": AzureOpenAI, "openai.official": OfficialOpenAI}
```

#### `LayerRegistrar.tree() -> dict`

Return a nested dict representing the key hierarchy. Only meaningful when `key_separator` is set.

```python
LLM.tree()
# → {"openai": {"azure": AzureOpenAI, "official": OfficialOpenAI}}
```

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

### `AutoRegistrarBase` (metaclass)

Shared base for all conscribe AutoRegistrar metaclasses. Uses `MetaRegistrarType` as its meta-metaclass to support the `|` operator.

```python
from conscribe import AutoRegistrarBase

# Cross-registry diamond:
CombinedMeta = LLM.Meta | Agent.Meta
class DualClass(metaclass=CombinedMeta):
    ...  # registered in both
```

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

Factory for key transforms with suffix/prefix stripping.

---

## Config Typing API

### `extract_config_schema(cls, mro_scope="local", mro_depth=None) -> type[BaseModel] | None`

Extract a Pydantic model from a class's `__init__` signature. See [Config Typing Deep-Dive](config-typing.md) for extraction priority.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cls` | `type` | required | Class to extract from |
| `mro_scope` | `MROScope` | `"local"` | Scope for MRO traversal (overridden by `__config_mro_scope__`) |
| `mro_depth` | `int \| None` | `None` | Max MRO levels (overridden by `__config_mro_depth__`) |

---

### `build_layer_config(registrar) -> LayerConfigResult`

Build a discriminated union from all registered classes. Dispatches to flat or nested mode based on `discriminator_fields`.

**Returns:** `LayerConfigResult`

**Raises:** `ValueError` if registrar has no discriminator configuration.

---

### `LayerConfigResult` (dataclass)

```python
@dataclass(frozen=True)
class LayerConfigResult:
    union_type: Any                                 # The discriminated union type
    per_key_models: dict[str, type[BaseModel]]      # Key -> per-key model
    layer_name: str                                 # Layer name
    discriminator_field: str                        # Discriminator field name (flat mode)
    degraded_fields: dict[str, list[DegradedField]] # Key -> degraded fields (empty if none)
    # Nested mode fields:
    discriminator_fields: list[str] | None = None   # Discriminator field names (nested mode)
    key_separator: str = ""                         # Key separator
    per_segment_models: dict[str, dict[str, type[BaseModel]]] | None = None
    # e.g., {"provider": {"azure": AzureProviderConfig, "official": OfficialProviderConfig}}
```

---

### `generate_layer_config_source(result) -> str`

Generate a self-contained Python source file from a `LayerConfigResult`. Dispatches to flat or nested generation mode.

---

### `generate_layer_json_schema(result) -> dict`

Generate a JSON Schema dict. Includes:
- `x-discriminator`: discriminator field name
- `x-discriminator-fields`: list of discriminator field names (nested mode)
- `x-key-separator`: the key separator (nested mode)
- `x-degraded-fields`: degradation info (when applicable)

---

### `compute_registry_fingerprint(registrar) -> str`

Compute a 16-character hex fingerprint of all registered classes.

### `load_cached_fingerprint(fingerprint_path, layer_name) -> str | None`

Load a cached fingerprint from a JSON file.

### `save_fingerprint(fingerprint_path, layer_name, fingerprint) -> None`

Save a fingerprint to a JSON cache file.

---

### `MROScope` (type alias)

```python
MROScope = Literal["local", "third_party", "all"] | list[str]
```

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

## Wiring API

### `get_registry(name: str) -> LayerRegistry | None`

Look up a `LayerRegistry` by layer name. Used internally by wiring resolution, but also available for manual cross-registry queries.

```python
from conscribe import get_registry

loop_reg = get_registry("agent_loop")
if loop_reg:
    print(loop_reg.keys())  # ["react", "codeact", ...]
```

---

### `__wiring__` (class attribute)

Declares cross-registry field constraints. Three grammar modes:

```python
class SWEAgent(BaseAgent):
    __wiring__ = {
        "loop": "agent_loop",                              # Mode 1: all keys from registry
        "llm_provider": ("llm", ["openai", "anthropic"]),  # Mode 2: explicit subset
        "browser": ["chromium", "firefox"],                # Mode 3: literal list
    }
```

**Behavior during config generation:**
- If the param exists in `__init__`: type is constrained from `str` to `Literal[...keys...]`
- If the param does NOT exist in `__init__`: injected as a new required field with `Literal[...keys...]` type
- Generated stubs annotate wired fields: `loop: Literal["react"] = ...  # wired from: agent_loop`

**Inheritance:** Deep-merged along MRO. Child keys override parent keys. `None` excludes an inherited key:

```python
class BaseAgent:
    __wiring__ = {"loop": "agent_loop", "llm": "llm"}

class OfflineAgent(BaseAgent):
    __wiring__ = {"llm": None}  # excludes llm, inherits loop
```

**Validation timing:** Resolved at `build_layer_config()` time (after `discover()`), not at class definition time. Raises `WiringResolutionError` if registry not found, is empty, or explicit keys are missing.

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
| `WiringResolutionError` | `RegistryError` | `__wiring__` references missing registry, empty registry, or invalid keys |

### Class Attributes for Configuration

| Attribute | Type | Purpose |
|-----------|------|---------|
| `__abstract__` | `bool` | Skip registration (checked via `namespace.get()`, not inherited) |
| `__registry_key__` | `str \| list[str]` | Override inferred registry key. List for multi-key registration |
| `__registry_keys__` | `list[str]` | Set by metaclass when multi-key registration is used (read-only) |
| `__skip_registries__` | `list[str]` | Skip registration in named registries |
| `__registration_filter__` | `staticmethod` | Parent-level filter: called with child class, return `False` to block |
| `__propagate__` | `bool` | If `False`, subclasses don't auto-register through this parent |
| `__propagate_depth__` | `int` | Only N levels of subclasses auto-register |
| `__config_schema__` | `type[BaseModel]` | Tier 3: explicit Pydantic model |
| `__config_annotated_only__` | `bool` | Only include `Annotated[..., Field()]` params |
| `__config_mro_scope__` | `MROScope` | Per-class MRO scope override |
| `__config_mro_depth__` | `int \| None` | Per-class MRO depth override |
| `__wiring__` | `dict[str, str \| tuple \| list \| None]` | Cross-registry field constraints (3 modes + None for exclusion) |
| `__wired_fields__` | `dict[str, str]` | Set by extractor: maps wired field names to registry names (read-only) |
