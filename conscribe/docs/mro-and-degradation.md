# MRO and Type Degradation Deep-Dive

This document covers MRO-aware parameter collection (`conscribe/config/mro.py`) and type degradation (`conscribe/config/degradation.py`).

## MRO Parameter Collection

When a class's `__init__` accepts `**kwargs` and forwards to `super().__init__(**kwargs)`, the child's config schema would be incomplete without parent parameters. `collect_mro_params()` walks the MRO upward to collect them.

### Algorithm

```
1. Find the class that actually defines cls.__init__ (via MRO walk)
2. Check if that __init__ has **kwargs; if not, return empty
3. Record own param names (child wins on duplicates)
4. Walk MRO from init_definer + 1 upward:
   a. Skip object
   b. Check scope filter -> break if excluded (truncated)
   c. Check depth limit -> break if exceeded (truncated)
   d. Skip classes without own __init__
   e. Increment levels_traversed
   f. Collect named params (skip already-seen names)
   g. Merge type hints (child overrides)
   h. If this parent has no **kwargs -> natural termination (fully_resolved=True)
```

### Result

```python
@dataclass(frozen=True)
class MROCollectionResult:
    params: list[inspect.Parameter]  # collected parent params
    init_definers: list[type]        # classes that contributed
    hints: dict[str, Any]            # merged type hints
    fully_resolved: bool             # True if chain terminated naturally
```

`fully_resolved` determines the `extra` policy:
- `True` -> `extra="forbid"` (all params known)
- `False` -> `extra="allow"` (some params may be unknown)

## Scope Classification

`classify_class_scope(cls)` determines whether a class is local, third-party, or stdlib:

```python
1. inspect.getfile(cls) -> source file path
   - No file (built-in/C extension) -> "stdlib"
2. "site-packages" in path -> "third_party"
3. Path matches sysconfig stdlib paths -> "stdlib"
4. Otherwise -> "local"
```

### Scope Filtering

`_should_include_class(cls, scope)`:

| Scope | Includes |
|-------|----------|
| `"local"` | Local only |
| `"third_party"` | Local + third-party |
| `"all"` | Everything except object |
| `["pkg1", "pkg2"]` | Local + listed third-party packages |

For `list[str]` scope, `_extract_package_name(cls)` finds the top-level package by looking for `site-packages` in the file path and taking the next component.

### Package-Specific Scope

The `list[str]` form is the most practical for real-world use:

```python
# Only traverse into httpx and my_sdk, ignore all other third-party
LLMRegistrar = create_registrar(
    "llm", proto,
    mro_scope=["httpx", "my_sdk"],
)
```

Local classes always pass. Stdlib classes are always excluded. Only third-party classes whose top-level package is in the list are included.

## Configuration Levels

MRO scope and depth can be set at three levels (highest priority wins):

1. **Function-level**: `extract_config_schema(cls, mro_scope=..., mro_depth=...)`
2. **Registrar-level**: `create_registrar(..., mro_scope=..., mro_depth=...)`
3. **Per-class** (highest): `__config_mro_scope__` and `__config_mro_depth__` class attributes

The registrar passes its settings to `extract_config_schema()`, which checks for per-class overrides via `getattr(cls, "__config_mro_scope__", mro_scope)`.

## Type Degradation

When MRO traversal collects parameters from third-party classes, some types may be incompatible with Pydantic (e.g., `httpx.Auth`, `ssl.SSLContext`).

### Try-First Approach

Degradation is lazy -- it only triggers when `pydantic.create_model()` actually fails:

```python
try:
    model = _create_dynamic_model(name, fields, extra)
except (PydanticSchemaGenerationError, PydanticUserError):
    fields, degraded = degrade_field_definitions(fields, source_class)
    model = _create_dynamic_model(name, fields, extra)
    model.__degraded_fields__ = degraded
```

Zero overhead on the happy path. Local classes with standard types never pay this cost.

### Per-Field Probing

`degrade_field_definitions()` probes each field individually:

```python
for name, (field_type, default_or_info) in field_definitions.items():
    if check_type_compatibility(field_type):
        # Keep as-is
    else:
        # Replace type with Any, record DegradedField
```

`check_type_compatibility(tp)` uses `TypeAdapter(tp)` as the probe -- if Pydantic can build a schema for it, the type is compatible.

### DegradedField Record

```python
@dataclass(frozen=True)
class DegradedField:
    field_name: str           # e.g., "auth"
    original_type_repr: str   # e.g., "httpx._types.AuthTypes | None"
    source_class: str         # e.g., "httpx._client.Client"
    reason: str = "pydantic_incompatible"
```

### Type Representation

`format_type_repr(tp)` produces human-readable type strings:
- `Union[X, None]` -> `"X | None"`
- `list[X]` -> `"list[X]"`
- Plain class -> `"module.ClassName"`
- `ForwardRef` -> the original string

### Surfacing Degradation

Degraded fields appear in three places:

**1. Python stub header:**
```python
# WARNING: The following fields had types incompatible with config
# serialization and were degraded to Any:
#   httpx: auth (was: httpx._types.AuthTypes | None)
```

**2. Inline stub comments:**
```python
auth: Any = None  # degraded from: httpx._types.AuthTypes | None
```

**3. JSON Schema extensions:**
```json
{
  "x-degraded-fields": {
    "httpx": [{"field": "auth", "original_type": "httpx._types.AuthTypes | None"}]
  }
}
```

Plus per-property `x-degraded-from` and `description` annotations inside `$defs`.

## Example: Full MRO Chain

```python
class ThirdPartyBase:
    def __init__(self, *, auth: httpx.Auth = None, timeout: int = 30):
        ...

class LocalMiddle(ThirdPartyBase):
    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(**kwargs)

class MyImpl(LocalMiddle):
    __config_mro_scope__ = ["httpx"]

    def __init__(self, *, model_id: str, **kwargs):
        super().__init__(**kwargs)
```

With `mro_scope=["httpx"]`:

1. `MyImpl.__init__` has `**kwargs` -> start MRO collection
2. `LocalMiddle.__init__` contributes `api_key: str` (local -> always included)
3. `LocalMiddle.__init__` has `**kwargs` -> continue
4. `ThirdPartyBase.__init__` contributes `auth` and `timeout` (httpx is in scope list)
5. `ThirdPartyBase.__init__` has no `**kwargs` -> natural termination (`fully_resolved=True`)
6. `auth: httpx.Auth` fails Pydantic check -> degraded to `Any`
7. Final schema: `model_id: str, api_key: str, auth: Any = None, timeout: int = 30`, `extra="forbid"`
