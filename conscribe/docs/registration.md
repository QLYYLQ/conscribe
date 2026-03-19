# Registration Deep-Dive

This document covers the internals of `conscribe/registration/`. For usage, see [Guide: Framework Developer](guide-alice.md).

## Architecture

```
create_registrar(name, protocol, ...)
    -> LayerRegistry(name, protocol, separator=...)  # storage
    -> create_auto_registrar(registry, kt, ...)      # metaclass
    -> build_filter_chain(...)                        # predicate filters
    -> LayerRegistrar subclass                        # facade
```

Three registration paths feed into the same `LayerRegistry`:

| Path | Mechanism | Protocol Check |
|------|-----------|----------------|
| **A** (inheritance) | `AutoRegistrar.__new__` intercepts class creation | No (inheritance guarantees compliance) |
| **B** (bridge) | `bridge()` creates abstract base, subclasses use Path A | No (same as A) |
| **C** (manual) | `@register()` decorator calls `registry.add()` | Yes (explicit check) |

## AutoRegistrarBase + MetaRegistrarType

All conscribe metaclasses inherit from `AutoRegistrarBase`, which uses `MetaRegistrarType` as its meta-metaclass. This enables the `|` operator for cross-registry diamond inheritance:

```python
CombinedMeta = LLM.Meta | Agent.Meta  # combines two AutoRegistrar metaclasses

class LLMAgent(metaclass=CombinedMeta):
    ...  # registered in BOTH llm and agent registries
```

When `|` is applied:
- If one is already a subclass of the other, returns the more specific one
- Otherwise creates a new combined metaclass inheriting from both
- The combined metaclass's MRO ensures both `__new__` methods fire via `super()` chain

## Predicate Filter System

`conscribe/registration/filters.py` replaces scattered `if/else` skip logic with composable filter objects. Each filter implements `should_skip(ctx) -> bool`.

### RegistrationContext

```python
@dataclass(frozen=True)
class RegistrationContext:
    cls: type           # the class being registered
    name: str           # class name
    bases: tuple        # base classes
    namespace: dict     # class __dict__
    registry_name: str  # target registry name
```

### Built-in Filters

| Filter | What it checks | Replaces |
|--------|---------------|----------|
| `RootFilter` | `not ctx.bases` | root base class check |
| `PydanticGenericFilter` | `"[" in ctx.name` | Pydantic Generic intermediate |
| `AbstractFilter` | `ctx.namespace.get("__abstract__")` | explicit abstract check |
| `CustomCallableFilter(fn)` | `fn(ctx.cls)` | custom skip_filter |
| `ChildSkipFilter` | `__skip_registries__` on class | **NEW** opt-out |
| `ParentRegistrationFilter` | `__registration_filter__` on parents | **NEW** parent control |
| `PropagationFilter` | `__propagate__` / `__propagate_depth__` on parents | **NEW** depth control |

### Class-Level and Parent Control Attributes

| Attribute | Location | Effect |
|-----------|----------|--------|
| `__skip_registries__ = ["agent"]` | Child class | Skip registration in named registries |
| `__registration_filter__` | Parent class (`@staticmethod`) | Called with child class; return `False` to block |
| `__propagate__ = False` | Parent class | Subclasses don't auto-register through this parent |
| `__propagate_depth__ = N` | Parent class | Only N levels of subclasses auto-register |

The same filter chain is used in both `AutoRegistrar.__new__` (Path A) and `_inject_auto_registration` (Path C), eliminating duplicated skip logic.

## LayerRegistry

Thread-safe key-to-class mapping. One instance per layer.

### Storage

```python
self._store: dict[str, type]  # strong references
self._lock: threading.Lock     # thread safety (NOT RLock)
self.separator: str            # key separator (e.g. "." or "")
```

### Protocol Check Caching

Inspired by CPython's `ABCMeta._abc_data`. Avoids re-checking Protocol compliance on every registration:

- **Positive cache** (`WeakSet`): classes that passed the check
- **Negative cache** (`WeakSet`): classes that failed
- **Invalidation counter** (class variable): incremented on `remove()`, invalidates all negative caches globally

### Key Operations

| Method | Thread Safety | Notes |
|--------|--------------|-------|
| `add(key, cls)` | Lock held | Raises `DuplicateKeyError` if key exists |
| `get(key)` | Lock held | Raises `KeyNotFoundError` if missing |
| `get_or_none(key)` | Lock held | Returns `None` if missing |
| `remove(key)` | Lock held + counter lock | Increments invalidation counter |
| `keys()` | Lock held | Returns snapshot list |
| `items()` | Lock held | Returns snapshot list of tuples |
| `children(prefix)` | Lock held | Keys starting with `prefix + separator` |
| `tree()` | Lock held | Nested dict from splitting keys by separator |

### Hierarchical Key Queries

When `separator` is set (e.g. `"."`):

```python
registry.children("openai")
# → {"openai.azure": AzureOpenAI, "openai.official": OfficialOpenAI}

registry.tree()
# → {"openai": {"azure": AzureOpenAI, "official": OfficialOpenAI}}
```

## AutoRegistrar Metaclass

Created by `create_auto_registrar()` as a closure-based metaclass. Captures the registry, key transform, and filter chain.

### `__new__` Flow

```
1. Create class via super().__new__()
2. Tag with __registry_name__
3. Build RegistrationContext, run filter chain
4. Resolve keys:
   a. __registry_key__ = "foo"       -> single explicit key
   b. __registry_key__ = ["a", "b"]  -> multi-key registration
   c. Hierarchical derivation         -> parent_key + separator + key_transform(name)
   d. key_transform(name)             -> default
5. registry.add(key, cls) for each key
6. Set cls.__registry_key__ and cls.__registry_keys__ (if multi-key)
7. Validate __config_schema__ if present
```

Critical: uses `namespace.get()` instead of `getattr()` for `__abstract__` and `__registry_key__` to avoid inheriting parent values.

### Hierarchical Key Derivation

When `key_separator` is set and no explicit `__registry_key__` is provided, the metaclass derives a hierarchical key by finding the nearest parent with `__registry_key__` set:

```python
# Parent: __registry_key__ = "openai" (abstract)
# Child: AzureOpenAI -> derived key: "openai.azure_open_ai"
```

### Multi-Key Registration

`__registry_key__` accepts `str | list[str]`:

```python
class Universal(Base):
    __registry_key__ = ["openai.universal", "anthropic.universal"]
# Registered under BOTH keys
# cls.__registry_key__ = "openai.universal"  (primary)
# cls.__registry_keys__ = ["openai.universal", "anthropic.universal"]  (all)
```

## Key Transform

`conscribe/registration/key_transform.py` provides CamelCase-to-snake_case conversion.

### Algorithm

1. Strip suffix (first match, if stripping wouldn't empty the string)
2. Strip prefix (first match, if stripping wouldn't empty the string)
3. Convert CamelCase to snake_case via two regex passes

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

The injected hook uses the same filter chain as AutoRegistrar, so `__skip_registries__`, `__registration_filter__`, `__propagate__`, and `__propagate_depth__` all work consistently across paths.
