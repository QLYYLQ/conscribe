# Registration Deep-Dive

This document covers the internals of `conscribe/registration/`. For usage, see [Guide: Framework Developer](guide-alice.md).

## Architecture

```
create_registrar(name, protocol, ...)
    -> LayerRegistry(name, protocol)       # storage
    -> create_auto_registrar(registry, kt) # metaclass
    -> LayerRegistrar subclass             # facade
```

Three registration paths feed into the same `LayerRegistry`:

| Path | Mechanism | Protocol Check |
|------|-----------|----------------|
| **A** (inheritance) | `AutoRegistrar.__new__` intercepts class creation | No (inheritance guarantees compliance) |
| **B** (bridge) | `bridge()` creates abstract base, subclasses use Path A | No (same as A) |
| **C** (manual) | `@register()` decorator calls `registry.add()` | Yes (explicit check) |

## LayerRegistry

Thread-safe key-to-class mapping. One instance per layer.

### Storage

```python
self._store: dict[str, type]  # strong references
self._lock: threading.Lock     # thread safety (NOT RLock)
```

### Protocol Check Caching

Inspired by CPython's `ABCMeta._abc_data`. Avoids re-checking Protocol compliance on every registration:

- **Positive cache** (`WeakSet`): classes that passed the check
- **Negative cache** (`WeakSet`): classes that failed
- **Invalidation counter** (class variable): incremented on `remove()`, invalidates all negative caches globally

The negative cache uses a version number compared against the global counter. When a class is unregistered, the counter increments, and all registries discard stale negative caches on next access.

### Key Operations

| Method | Thread Safety | Notes |
|--------|--------------|-------|
| `add(key, cls)` | Lock held | Raises `DuplicateKeyError` if key exists |
| `get(key)` | Lock held | Raises `KeyNotFoundError` if missing |
| `get_or_none(key)` | Lock held | Returns `None` if missing |
| `remove(key)` | Lock held + counter lock | Increments invalidation counter |
| `keys()` | Lock held | Returns snapshot list |
| `items()` | Lock held | Returns snapshot list of tuples |

## AutoRegistrar Metaclass

Created by `create_auto_registrar()` as a closure-based metaclass. Captures the registry and key transform function.

### `__new__` Flow

```
1. Create class via super().__new__()
2. Tag with __registry_name__
3. Skip conditions:
   a. No bases (root base class)
   b. Name contains "[" (Pydantic Generic intermediate)
   c. Custom skip_filter returns True
   d. namespace.get("__abstract__") is True
4. Infer key: namespace.get("__registry_key__") or key_transform(name)
5. registry.add(key, cls, protocol_check=False)
6. Set cls.__registry_key__ = key
7. Validate __config_schema__ if present
```

Critical: uses `namespace.get()` instead of `getattr()` for `__abstract__` and `__registry_key__` to avoid inheriting parent values.

### Pydantic Generic Filtering

When `skip_pydantic_generic=True` (default), any class whose `__name__` contains `[` is skipped. This filters Pydantic's runtime Generic specialization intermediates (e.g., `BaseEvent[str]`).

## Key Transform

`conscribe/registration/key_transform.py` provides CamelCase-to-snake_case conversion.

### Algorithm

1. Strip suffix (first match, if stripping wouldn't empty the string)
2. Strip prefix (first match, if stripping wouldn't empty the string)
3. Convert CamelCase to snake_case via two regex passes:
   - Insert `_` between consecutive uppercase and uppercase+lowercase: `HTTPSHandler` -> `HTTPS_Handler`
   - Insert `_` between lowercase/digit and uppercase: `browserUse` -> `browser_Use`
   - Lowercase everything

### Examples

| Input | strip_prefixes | strip_suffixes | Output |
|-------|---------------|----------------|--------|
| `ChatOpenAI` | `["Chat"]` | - | `open_ai` |
| `BrowserUseAgent` | - | `["Agent"]` | `browser_use` |
| `HTTPSHandler` | - | - | `https_handler` |
| `MyV2Agent` | - | `["Agent"]` | `my_v2` |

## Bridge Metaclass Conflict Resolution

`LayerRegistrar.bridge()` handles four metaclass scenarios:

```python
ext_meta = type(external_class)

# Strategy 1: External is plain type -> use our Meta
if ext_meta is type:
    meta = cls.Meta

# Strategy 2: Our Meta subclasses external's -> use ours
elif issubclass(cls.Meta, ext_meta):
    meta = cls.Meta

# Strategy 3: External's meta is more specific -> use theirs
elif issubclass(ext_meta, cls.Meta):
    meta = ext_meta

# Strategy 4: Incompatible -> create combined metaclass
else:
    meta = type(f"{bridge_name}Meta", (cls.Meta, ext_meta), {})
```

The bridge class itself is created with `__abstract__ = True`, so it's not registered.

## Path C: Propagation

When `@register(propagate=True)` is used, `_inject_auto_registration()` replaces the target's `__init_subclass__` with a hook that auto-registers future subclasses. This enables inheritance-based registration without the metaclass.

The injected hook:
1. Calls the original `__init_subclass__` (if any)
2. Applies the same skip conditions as AutoRegistrar (Pydantic generic, custom filter, abstract)
3. Infers key and registers

Uses `target_cls.__dict__.get("__init_subclass__")` (not `getattr`) to avoid inheriting the hook from a parent.
