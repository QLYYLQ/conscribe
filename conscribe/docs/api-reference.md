# API Reference

The public API of conscribe 2.0, module by module. Supported versions: Python 3.10 to 3.14 and pydantic >= 2.7, < 3.

| Import from | What it holds |
|-------------|---------------|
| `conscribe` | Registration, wiring essentials, config typing, errors and warnings (`conscribe.__all__`) |
| `conscribe.wiring` | The full wiring API: grammar, resolution, placeholders |
| `conscribe.introspect` | How conscribe reads a class: constructor names, receptor slots, wiring slots |
| `conscribe.stubs` | `.pyi` stub generation |
| `conscribe.testing` | Test isolation; the opt-in pytest plugin is `conscribe.testing.pytest_plugin` |
| `conscribe.exceptions` | Every error and warning, plus `WiringProblem` and `WIRING_PROBLEM_KINDS` |
| `conscribe.discover` (module) | `discover()`, `import_failures()`, `clear_import_failures()` |

The package attribute `conscribe.discover` is the `discover()` function, so reach the other helpers of that module with `from conscribe.discover import import_failures`.

conscribe's own exceptions derive from `RegistryError` and its warnings from `ConscribeWarning` (see [Errors and warnings](#errors-and-warnings)); a wrong argument to a function raises the built-in `TypeError` or `ValueError`. Upgrading from 1.x: see [Migrating to 2.0](migrating-to-2.0.md).

---

## Registration

### `create_registrar()`

```
create_registrar(
    name: str,
    protocol: type,
    *,
    discriminator_field: str = "",
    discriminator_fields: list[str] | None = None,
    key_separator: str = "",
    strip_suffixes: list[str] | None = None,
    strip_prefixes: list[str] | None = None,
    key_transform: KeyTransform | None = None,
    base_metaclass: type = type,
    mro_scope: str | list[str] = "local",
    mro_depth: int | None = None,
    skip_pydantic_generic: bool = True,
    skip_filter: Callable[[type], bool] | None = None,
    inline_only: bool = False,
) -> type[LayerRegistrar]
```

The recommended entry point. Creates the layer's registry, its registration metaclass and filter chain, and returns a `LayerRegistrar` subclass named `"<Name>Registrar"` that ties them together.

A class with no bases (`class Base(metaclass=Registrar.Meta)`) is a root and is skipped automatically. A base class with mixins needs its own `__abstract__ = True` to stay out of the registry. Invalid class keys, including Enum members even when they subclass `str`, raise `InvalidRegistryKeyError` during registration, before any key is added.

| Parameter | Description |
|-----------|-------------|
| `name` | Layer name (`"agent"`, `"llm"`). Also the registry name that `__wiring__` entries use. |
| `protocol` | A `@runtime_checkable` Protocol class |
| `discriminator_field` | Discriminator field of the config union (flat mode) |
| `discriminator_fields` | Discriminator fields, one per key segment (nested mode). Exclusive with `discriminator_field`; requires `key_separator`. |
| `key_separator` | Separator of hierarchical keys (`"."`). `""` means flat keys. Also enables capability-relative wiring keys against this registry. |
| `strip_suffixes` / `strip_prefixes` | Stripped from class names during key inference (first match wins) |
| `key_transform` | A callable `class name -> key`; replaces the default inference |
| `base_metaclass` | Parent metaclass of the registration metaclass (e.g. `ABCMeta`, pydantic's `ModelMetaclass`) |
| `mro_scope` | Which parents a `**kwargs` chain may collect knobs from: `"local"`, `"third_party"`, `"all"` or a list of package names. See [MRO and Type Degradation](mro-and-degradation.md). |
| `mro_depth` | Maximum number of parent `__init__` definers to visit (`None` = unlimited, `0` = do not follow `**kwargs`) |
| `skip_pydantic_generic` | Skip classes whose name contains `[` (pydantic generic intermediates) |
| `skip_filter` | `(cls) -> bool`; return `True` to skip a class |
| `inline_only` | Declare the layer inline-only: in a composed config its instances may only be written inside other layers' wiring receptors. Registration is unaffected. |

**Raises:** `InvalidProtocolError` if `protocol` is not `@runtime_checkable`; `ValueError` if both discriminator options are set, or `discriminator_fields` is set without `key_separator`; `TypeError` if `inline_only` is not a bool.

`mro_scope` / `mro_depth` are validated on the constructor-reflection path of extraction (`ConfigDeclarationError`); a class's own settings take precedence. Nested-mode builds ignore these registrar settings, including invalid values, and follow each class's real constructor. `get_config_schema(key)` calls the extractor even on a nested-mode registrar, so its constructor schemas still apply and validate these settings. Creating a registrar whose name is already taken makes the new registry the one `get_registry(name)` and wiring see.

```python
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, prompt: str) -> str: ...


LLM = create_registrar("llm", ChatModel, discriminator_field="provider", strip_prefixes=["Chat"])


class BaseChatModel(metaclass=LLM.Meta):
    __abstract__ = True

    def complete(self, prompt: str) -> str:
        return prompt


class ChatOpenAI(BaseChatModel):
    def __init__(self, model: str, temperature: float = 0.7) -> None:
        self.model = model
        self.temperature = temperature


print(LLM.__name__, LLM.keys(), LLM.get("open_ai").__name__)
schema = LLM.get_config_schema("open_ai")
print(schema.__name__, list(schema.model_fields))
result = LLM.build_config()
print(result.per_key_models["open_ai"].__name__, list(result.per_key_models["open_ai"].model_fields))
```

```
LlmRegistrar ['open_ai'] ChatOpenAI
ChatOpenAIConfig ['model', 'temperature']
OpenAiLLMConfig ['model', 'temperature', 'provider']
```

---

### `LayerRegistrar`

The class `create_registrar()` returns. Every method is a classmethod.

| Attribute | Description |
|-----------|-------------|
| `protocol` | The layer's Protocol |
| `Meta` | The registration metaclass; `Meta.inline_only` reports the inline-only flag |
| `discriminator_field` | Flat-mode discriminator (`""` in nested mode) |
| `discriminator_fields` | Nested-mode discriminators, or `None` |
| `inline_only` | Whether the layer is inline-only |

#### `get(key) -> type`

The class registered under `key`. **Raises** `KeyNotFoundError` (a `KeyError` subclass) listing the registered keys.

#### `get_or_none(key) -> type | None`

Like `get()`, but returns `None` for an unknown key.

#### `get_all() -> dict[str, type]`

A snapshot of all `{key: class}` bindings, in registration order.

#### `keys() -> list[str]`

All registered keys, in registration order.

#### `unregister(key, *, missing_ok=False) -> type | None`

Removes the binding of `key` and returns the class it pointed to. Only that key is removed: a multi-key class stays registered under its other keys, and keys below a hierarchical prefix are separate entries. **Raises** `KeyNotFoundError` when `key` is not registered, unless `missing_ok=True` (then it returns `None`). For tests, prefer [`isolated_registries()`](testing.md).

#### `children(prefix) -> dict[str, type]`

Entries whose key starts with `prefix + key_separator`; `{}` when the layer has no separator.

#### `tree() -> dict`

The keys as a nested dict split on the separator (`{"openai": {"azure": AzureOpenAI}}`); a flat `{key: class}` dict without a separator. **Raises** `ValueError` when a key is both a leaf and the prefix of another key.

#### `bridge(external_class, *, name=None) -> type`

Creates an abstract base class (`"<External>Bridge"` by default) that inherits from `external_class` and uses a metaclass compatible with both sides; its subclasses register automatically. **Raises** `TypeError` when the two metaclasses cannot be combined. See [Bridge metaclass conflict resolution](registration.md#bridge-metaclass-conflict-resolution).

#### `register(key=None, *, propagate=False) -> Callable[[type], type]`

A class decorator that registers the class under `key` (inferred from the class name when omitted) after a Protocol check. With `propagate=True` it also installs an `__init_subclass__` hook so subclasses register on definition.

- `key` must be one non-empty `str`; to register under several keys, apply `register()` once per key.
- The key and the class's merged `__wiring__` grammar are checked before anything is registered.
- If something fails after the class was added, the decorator undoes its own binding and attribute writes (bindings the class has in other registries are left alone).
- Place `@register(...)` *above* `@dataclass(slots=True)` / `attrs.define`.

**Raises:** `InvalidRegistryKeyError`, `WiringResolutionError` (malformed `__wiring__`), `DuplicateKeyError`, `ProtocolViolationError`.

#### `build_config() -> LayerConfigResult`

Same as `build_layer_config(cls)`.

#### `config_union_type() -> Any`

`build_config().union_type`.

#### `get_config_schema(key) -> type[BaseModel] | None`

`extract_config_schema()` of the class registered under `key`, with the registrar's `mro_scope` / `mro_depth`: the per-class model without the discriminator field. `None` when the class has nothing to configure.

---

### `create_auto_registrar()`

```
create_auto_registrar(
    registry: LayerRegistry,
    key_transform: KeyTransform,
    *,
    base_metaclass: type = type,
    skip_pydantic_generic: bool = True,
    skip_filter: Callable[[type], bool] | None = None,
    key_separator: str = "",
    inline_only: bool = False,
) -> type[AutoRegistrarBase]
```

The lower-level factory `create_registrar()` uses: returns only the registration metaclass for an existing registry. **Raises** `TypeError` if `inline_only` is not a bool.

### `LayerRegistry`

```
LayerRegistry(name: str, protocol: type, *, separator: str = "")
```

The thread-safe key-to-class store behind a registrar. Creating one adds it to the global registry index under `name`; an existing registry of that name is shadowed (it stays alive, but `get_registry(name)` returns the new one). **Raises** `InvalidProtocolError` if `protocol` is not `@runtime_checkable`.

| Member | Description |
|--------|-------------|
| `name`, `protocol`, `separator` | As given |
| `add(key, cls, *, protocol_check=False)` | Registers `cls`. Raises `DuplicateKeyError`, and `ProtocolViolationError` when `protocol_check` fails. |
| `remove(key, *, missing_ok=False) -> type \| None` | Same contract as `LayerRegistrar.unregister()` |
| `get(key)`, `get_or_none(key)` | Lookups; `get` raises `KeyNotFoundError` |
| `keys() -> list[str]`, `items() -> list[tuple[str, type]]` | Snapshots |
| `children(prefix)`, `tree()` | As on `LayerRegistrar` |
| `runtime_check(cls)` | The Protocol check (cached); raises `ProtocolViolationError` |

### `AutoRegistrarBase`

The base of every registration metaclass. Its own metaclass defines `|`: `LLM.Meta | Agent.Meta` returns a metaclass that registers in both layers (cross-registry diamond). See [AutoRegistrarBase and the `|` operator](registration.md#autoregistrarbase-and-the--operator).

### `get_registry(name) -> LayerRegistry | None`

The registry currently registered under `name` in the global index, or `None`.

### Key transforms

| Name | Description |
|------|-------------|
| `KeyTransform` | Protocol: `__call__(self, class_name: str) -> str` |
| `default_key_transform(class_name) -> str` | CamelCase to snake_case: `HTTPSHandler` gives `https_handler` |
| `make_key_transform(*, suffixes=None, prefixes=None) -> KeyTransform` | Strips the first matching suffix, then the first matching prefix (never down to an empty name), then converts to snake_case |

The key a transform returns is validated like any other key (`InvalidRegistryKeyError`).

---

### `discover()`

```
discover(
    *package_paths: str,
    strict: bool = True,
    allow_missing: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[str]
```

Imports every module under the given packages (depth-first, name order) so their classes register. Returns the imported module names, including modules that were already imported.

| Parameter | Description |
|-----------|-------------|
| `*package_paths` | Dotted package (or module) names |
| `strict` | `True`: any exception while importing a module propagates as the original exception, with a note naming the module (Python 3.11+; on 3.10 a failing submodule is re-raised as an `ImportError` naming it, chained from the original). `False`: a submodule raising `ImportError` or `SyntaxError` is skipped with an `ExtractionWarning`; other exceptions still propagate. The named packages fail fast in both modes unless `allow_missing` covers the failure. |
| `allow_missing` | Top-level names of optional dependencies. A module that fails with `ModuleNotFoundError` for one of them (also through a chain of `ImportError`s) is skipped quietly (`logging.info`). |
| `exclude` | Module or package names never imported; each excludes everything below it (`"pkg.gen"` excludes `pkg.gen.x`, not `pkg.gen_old`). |

**Raises:** `ModuleNotFoundError` for a named package that does not exist (unless allowed); in strict mode, the import failure (wrapped in `ImportError` for a submodule on Python 3.10, as above); `TypeError` if `allow_missing` / `exclude` hold anything but non-empty strings.

Registrations made by a module whose import failed are rolled back; `__main__` modules are never imported by the walk; already imported modules cost nothing. See [`discover()`](registration.md#discover) for the details and an example.

#### `import_failures() -> dict[str, str]` (module `conscribe.discover`)

A copy of the modules `discover()` could not import, as `{module: "ExcType: message"}`. Skipped modules are included (marked as such); modules imported since are left out. `EmptyLayerError` and wiring error messages cite this record.

#### `clear_import_failures() -> None` (module `conscribe.discover`)

Forgets every recorded failure. (`isolated_registries()` restores the record to its state at entry instead.)

---

## Wiring

### `__wiring__` entries

A class attribute mapping a parameter or attribute name to a wiring entry. There are four entry forms:

| Form | Example | Allowed keys |
|------|---------|--------------|
| Mode 1: registry name | `"loop": "loop"` | Every key of the registry |
| Mode 2: tuple | `"loop": ("loop", ["react", "plan"])` | The listed keys |
| Mode 2: 3-tuple | `"obs": ("observation", ["terminal"], ["filesystem"])` | Required keys plus optional keys (both end up in the `Literal`; the split is metadata for your framework) |
| Mode 3: list | `"browser": ["chromium", "firefox"]` | The listed strings; no registry involved |
| Dict form | `"loop": {"registry": "loop", "keys": [...], "optional_keys": [...], "default": "react", "declare_only": False}` | As Mode 1 (without `keys`) or Mode 2 (with `keys`) |

Dict-form fields: `registry` (required), `keys` (omitted means every key), `optional_keys` (requires `keys`), `default` (the key a single-instance receptor uses when the config omits it) and `declare_only` (a declaration-only constraint: the framework injects nothing for it). Any other field is an error, with a "did you mean" hint for typos. `declare_only` cannot be combined with `default`. A third tuple element of `None` means "no optional keys".

Rules:

- Entry names must be identifiers that are neither keywords nor dunder names.
- Every key is a `str`; explicit key lists must not be empty; a key cannot be both required and optional.
- `__wiring__` is merged along the MRO, base classes first. A subclass entry replaces the inherited entry whole; `{"name": None}` removes an inherited entry. `__wiring__ = None` on a class declares nothing (it keeps every inherited entry).
- Against a registry with a `key_separator`, an explicit key may be a bare trailing segment (a capability-relative key): `"click"` expands to every registered key ending in `.click` (`browser.click`, `desktop.click`). All matches are kept, and the consuming framework decides which one it uses. A key that contains the separator passes through unchanged. A `default` key is never expanded, so it must be spelled exactly as registered.
- The grammar is checked when a class registers, so a malformed entry fails the class statement with `WiringResolutionError`. Registry lookups (unknown registries or keys, empty registries, bad defaults, `declare_only` contradictions) happen when a config is built, and every problem of the build is reported in one error.

`__wiring__` is not supported on classes of nested-mode layers yet (`ConfigDeclarationError` at build time).

```python
from typing import Protocol, runtime_checkable

from conscribe import WiringResolutionError, create_registrar, validate_wiring
from conscribe.wiring import parse_wiring, resolve_wiring


@runtime_checkable
class Component(Protocol):
    def start(self) -> None: ...


Loop = create_registrar("loop", Component, discriminator_field="name")
Action = create_registrar("action", Component, discriminator_field="name", key_separator=".")
Agent = create_registrar("agent", Component, discriminator_field="name")


class LoopBase(metaclass=Loop.Meta):
    def start(self) -> None: ...


class React(LoopBase):
    __registry_key__ = "react"


class Plan(LoopBase):
    __registry_key__ = "plan"


class ActionBase(metaclass=Action.Meta):
    def start(self) -> None: ...


class BrowserClick(ActionBase):
    __registry_key__ = "browser.click"


class DesktopClick(ActionBase):
    __registry_key__ = "desktop.click"


class BrowserScroll(ActionBase):
    __registry_key__ = "browser.scroll"


class AgentBase(metaclass=Agent.Meta):
    def start(self) -> None: ...


class SweAgent(AgentBase):
    __wiring__ = {
        "loop": "loop",                                     # Mode 1: every key
        "planner": ("loop", ["plan"]),                       # Mode 2: a subset
        "actions": ("action", ["click"], ["scroll"]),        # Mode 2: required + optional
        "browser": ["chromium", "firefox"],                  # Mode 3: literal values
        "fallback": {"registry": "loop", "default": "react"},  # dict form
    }

    def __init__(self, fallback: Component, loop: Component) -> None: ...


for spec in parse_wiring(SweAgent):
    print(spec)
print(resolve_wiring(SweAgent)["actions"])


class BrokenAgent(AgentBase):
    __wiring__ = {"loop": ("loop", ["reakt"]), "memory": "memroy"}


try:
    validate_wiring({"agent": Agent, "loop": Loop})
except WiringResolutionError as exc:
    print(exc)
    print([problem.kind for problem in exc.problems])
```

```
WiringSpec(param_name='loop', registry_name='loop', allowed_keys=None, optional_keys=None, default_key=None, declare_only=False)
WiringSpec(param_name='planner', registry_name='loop', allowed_keys=('plan',), optional_keys=None, default_key=None, declare_only=False)
WiringSpec(param_name='actions', registry_name='action', allowed_keys=('click',), optional_keys=('scroll',), default_key=None, declare_only=False)
WiringSpec(param_name='browser', registry_name='', allowed_keys=('chromium', 'firefox'), optional_keys=None, default_key=None, declare_only=False)
WiringSpec(param_name='fallback', registry_name='loop', allowed_keys=None, optional_keys=None, default_key='react', declare_only=False)
ResolvedWiring(param_name='actions', allowed_keys=['browser.click', 'desktop.click', 'browser.scroll'], registry_name='action', optional_keys=['browser.scroll'], default_key=None, declare_only=False)
2 wiring problems:
  - BrokenAgent (key 'broken_agent') wiring 'loop': unknown key 'reakt' in registry 'loop' (did you mean 'react'?); registered: ['plan', 'react']
  - BrokenAgent (key 'broken_agent') wiring 'memory': unknown registry 'memroy'; known registries: ['action', 'agent', 'loop']
['unknown_key', 'unknown_registry']
```

### What a wired entry becomes in a config

During constructor reflection, a wired field holds registry keys (literal values for Mode 3); composed configs may replace them with nested configs. Its shape depends on whether the class has a *runtime slot* for the entry (an `__init__` parameter, including `**kwargs`-forwarded ones, or a non-`ClassVar` class annotation anywhere in the MRO) and on the slot's annotation. A Tier-3 schema instead supplies its own fields and may omit slot-less entries.

| Declaration | Per-layer config field | Composed config, `inline_wiring=True` |
|-------------|------------------------|---------------------------------------|
| Slot, single-instance annotation without a default (`loop: P`) | `Literal[...]`, required | The target layer's union, required |
| Constructor parameter admitting `None`, without a default (`loop: P \| None`) | `Optional[Literal[...]]`, required | `Optional[<union>]`, required |
| Slot with default `None` (`loop: P \| None = None`), or a class annotation admitting `None` with no value | `Optional[Literal[...]] = None` | `Optional[<union>] = None` |
| Slot, container annotation (`list[P]`, `dict[str, P]`, `set[P]`, ...) | `list[Literal[...]]` | `list[<union>]` |
| Slot with a default key (dict-form `default`, or a parameter default that is a key) | `Literal[...] = "<key>"`; stays `Optional` when the annotation admits `None` | The default key's config built from defaults (`Field(default_factory=...)`); `None` remains accepted when the annotation admits it |
| No slot at all | `Optional[Literal[...]] = None` | `Optional[<union>] = None` |
| `declare_only` entry | `Optional[Literal[...]] = None` | Unchanged: never inlined |
| Mode 3 list | Uses the same slot/default rules above with the listed literal values | Unchanged |

A slot-less entry is never a required field: nothing can receive a value for it at construction time. Mode 2 subsets inline only the subset's models, keeping the target layer's discriminator. A registry outside the composed build stays a key selector. [Config Typing Deep-Dive](config-typing.md#wired-fields) has the full rules (annotation mapping, key constraints, defaults, `null`).

---

### `check_wiring_declaration(cls) -> None`

Checks the grammar of the class's merged `__wiring__` without touching any registry: each declaration is a dict or `None`; entry names are valid; every entry has a supported shape and well-typed fields; no unknown dict-form fields; `optional_keys` comes with `keys`; no key is both required and optional; `declare_only` is not combined with `default`; keys are strings; explicit key lists are not empty. Every registration path calls it before the first registry change. **Raises** `WiringResolutionError` listing every grammar problem.

### `validate_wiring(registrars) -> None`

Resolves the wiring of every class registered in `registrars` (a registrar, a mapping whose values are registrars, or an iterable of registrars), exactly like `resolve_wiring()`, and reports every problem across all classes in one `WiringResolutionError`. Each class is checked once even when registered under several keys or layers; problems are ordered by `(module, qualname)` and then declaration order. `build_layer_config()`, `build_composed_config()` and `write_layer_stubs()` call it first. **Raises** `WiringResolutionError`, or `TypeError` for anything that is not a registrar.

### `conscribe.wiring.parse_wiring(cls) -> list[WiringSpec]`

The merged `__wiring__` as `WiringSpec` objects in declaration order (exclusions applied), grammar-checked, without registry lookups. **Raises** `WiringResolutionError` for grammar problems.

### `conscribe.wiring.resolve_wiring(cls) -> dict[str, ResolvedWiring]`

Resolves each entry to concrete keys: Mode 1 reads the registry's keys (sorted); Mode 2 and the dict form check every required and optional key (after capability-relative expansion); Mode 3 is returned as is. It also checks each entry against its runtime slot (read with `conscribe.introspect.receptor_slots`):

- a `default` must be registered, fully qualified and among the entry's allowed keys, and belong to a single-instance receptor with a slot;
- a `declare_only` entry must not have a slot;
- the slot's own default (parameter default, `Field(default=...)` or class-level value) must be an allowed key (a list of them for a container receptor), and the same key as the entry's `default` if it declares one. `None` is always fine.

Factory-produced defaults and defaults whose annotations contain `Literal` or transforming metadata are checked during config extraction. An inline composed build further refuses a non-empty container default or a scalar factory, since those do not name a default inline config.

Every problem of the class is collected first. **Raises** `WiringResolutionError`.

### `conscribe.wiring.collect_wiring_from_mro(cls) -> dict[str, Any]`

The raw merged `__wiring__` dict (base classes first, subclass entries replace inherited ones, `None` entries removed). Never raises: a `__wiring__` that is not a mapping contributes nothing.

### `conscribe.wiring.expand_relative_keys(declared, registry_keys, separator) -> list[str]`

Expands capability-relative keys: a key without the separator becomes every registered key whose last segment equals it (sorted); a key with the separator, a key matching nothing and a non-str value pass through unchanged. No-op when `separator` is empty. The result is de-duplicated, in declaration order.

### `WiringSpec`

Frozen dataclass from `parse_wiring()`:

| Field | Description |
|-------|-------------|
| `param_name` | Entry name |
| `registry_name` | Target registry (`""` for Mode 3) |
| `allowed_keys` | Explicit keys as declared (`None` = every key); the required keys when optional keys exist |
| `optional_keys` | Optional keys, or `None` |
| `default_key` | The dict-form `default`, or `None` |
| `declare_only` | `True` for a declaration-only entry |

### `ResolvedWiring`

Dataclass from `resolve_wiring()`. Positional order: `param_name`, `allowed_keys`, `registry_name`, `optional_keys`, `default_key`, `declare_only`.

| Field | Description |
|-------|-------------|
| `param_name` | Entry name |
| `allowed_keys` | Concrete keys (required plus optional), as the `Literal` lists them |
| `registry_name` | Target registry, `None` for Mode 3 |
| `optional_keys` | The expanded optional keys, or `None` |
| `default_key` | The validated default key, or `None` |
| `declare_only` | Whether the entry is declaration-only |

`optional_keys` is the fourth positional field: 2.0 removed `injected`, which was never set.

The literal-only mode uses different sentinels at each stage: `WiringSpec.registry_name == ""`, `ResolvedWiring.registry_name is None`, and `__wired_fields__[name] == "literal"` on a generated model.

---

### `WiredField`

```
WiredField(name: str, registry_name: str = "")
```

The placeholder installed for a wired attribute that the framework injects after construction. A non-data descriptor, so injection is a plain instance assignment.

- Instance access before injection raises `AttributeError` naming the attribute and its registry.
- Class access returns the descriptor itself, unless the accessing class or any class between it and the holder annotates the name: then it raises `AttributeError` (the name has no class-level value), so dataclass and pydantic subclasses that re-annotate the name get a required field.

Attributes: `name`, `registry_name`. Who gets a placeholder, tombstones and patching are described in [WiredField placeholders](registration.md#wiredfield-placeholders).

### `wired_field(cls, name) -> WiredField | None`

The placeholder `cls` exposes under `name`, looked up through each class's `__dict__` without invoking descriptors. Returned only when the first binding of `name` along the MRO is a `WiredField`, the merged `__wiring__` injects `name` (not excluded, not `declare_only`), and `name` is not a constructor parameter of `cls` at call time. **Raises** `TypeError` if `cls` is not a class or `name` not a str.

### `wired_fields(cls) -> dict[str, WiredField]`

Every placeholder `cls` exposes, by the same rule, in `__wiring__` declaration order (base classes first). **Raises** `TypeError` if `cls` is not a class.

### `inject_wired_descriptors(cls) -> None`

Installs the placeholders (and tombstones) for `cls`. Every registration path calls it at class creation; call it yourself only for classes conscribe does not create. It skips malformed entries silently; `check_wiring_declaration()` is what reports them.

---

## Config typing

### `extract_config_schema()`

```
extract_config_schema(
    cls: type,
    mro_scope: str | list[str] = "local",
    mro_depth: int | None = None,
) -> type[BaseModel] | None
```

The config model of one class, without the discriminator. In order:

1. `__config_schema__`: the schema itself (after its wiring checks).
2. `__config_from_param__ = "<param>"`: the flattened model of that parameter.
3. Constructor reflection: a model built from everything the constructor accepts (following `**kwargs` chains within `mro_scope` / `mro_depth`), with the wiring receptors.
4. `None` when the class has no configurable constructor parameter (a *knob*), no wiring entry and a closed constructor (one that accepts no arbitrary extra arguments).

`__config_mro_scope__` / `__config_mro_depth__` on the class override the arguments.

**Raises:** `InvalidConfigSchemaError` (`__config_schema__` is not a `BaseModel` subclass); `WiringResolutionError`; `ConfigAnnotationError` (a typo or broken `Annotated` metadata in an annotation a knob needs); `ConfigDeclarationError` (contradictory declarations: two defaults, `Field(exclude=...)`, a bad `__config_from_param__`, a Tier-3 schema missing a wired field, a `Literal` member that is not a wiring key, an invalid MRO setting, a required parameter no config can pass, a private wiring name, a name colliding with a `model_*` member of `BaseModel`). **Warns:** `DegradationWarning`, `ExtractionWarning`.

See [Config Typing Deep-Dive](config-typing.md#extraction-extractorpy).

### `build_layer_config(registrar) -> LayerConfigResult`

Builds the layer's discriminated union: checks the layer, runs `validate_wiring(registrar)`, then builds every per-key model (flat or nested mode).

**Raises:** `TypeError` (not a registrar created by `create_registrar()`); `ValueError` (no discriminator configured); `EmptyLayerError`; `WiringResolutionError` (every problem of the layer); `ConfigDeclarationError` (nested-mode keys with the wrong number of segments, declarations nested mode does not support, a field named like a discriminator, a `RootModel` or registered class as config schema, an incomplete schema); `ConfigAnnotationError`. **Warns:** `DegradationWarning`, `ExtractionWarning`, located at the class definition.

### `LayerConfigResult`

Frozen dataclass:

| Field | Description |
|-------|-------------|
| `union_type` | The discriminated union (the model itself for a one-key layer) |
| `per_key_models` | `{key: model}` |
| `layer_name` | The layer name |
| `discriminator_field` | Flat discriminator (nested mode: the level-0 one) |
| `degraded_fields` | `{key: [DegradedField, ...]}`, empty when nothing degraded |
| `discriminator_fields` | Nested-mode discriminators, or `None` |
| `key_separator` | The key separator |
| `per_segment_models` | Nested mode: `{discriminator field: {segment: model}}`; `None` in flat mode |

### `build_composed_config()`

```
build_composed_config(
    registrars: Mapping[str, LayerRegistrar],
    *,
    inline_wiring: bool = True,
    top_level_extra: TopLevelExtra = "forbid",
) -> ComposedConfigResult
```

Combines several layers into one top-level model with a `list[<layer union>]` field per layer. The keys of `registrars` name the top-level sections; wiring entries find layers by registry name.

| Parameter | Description |
|-----------|-------------|
| `registrars` | `{section name: registrar}`. Section names must be valid identifiers, not private, not `BaseModel` attributes, without the `model_` prefix. |
| `inline_wiring` | `True`: wired fields hold the target layer's configs (default instances for default keys; `declare_only` entries stay key selectors). `False`: wired fields stay `Literal[...]` keys. |
| `top_level_extra` | `"forbid"` (unknown top-level keys are refused; the JSON schema says `additionalProperties: false`), `"ignore"` or `"allow"`. Inline-only layer names are refused in every mode. |

**Raises:** `TypeError` (not a mapping of registrars); `ValueError` (bad `top_level_extra`, a layer without a discriminator); `EmptyLayerError` (every empty layer named); `WiringResolutionError` (every problem of the build); `CircularWiringError`; `ConfigDeclarationError` (bad section names, layer declaration problems, key constraints that reject every key, wiring keys that select no config model, a default key whose config cannot be built from defaults, a non-empty container default or scalar `default_factory` under inline wiring, an inline-only layer without a writable location, an inline-only layer with `inline_wiring=False`, two sections sharing the registry a wiring entry targets).

### `ComposedConfigResult`

Frozen dataclass:

| Field | Description |
|-------|-------------|
| `layer_results` | `{section: LayerConfigResult}` for every layer, inline-only ones included; models that own inlined receptors are rebuilt |
| `top_level_type` | The `ComposedConfig` model |
| `dependency_order` | Every layer, leaves first |
| `inline_wiring` | Whether wired fields were inlined |
| `inline_only_layers` | `{layer: (ReceptorLocation, ...)}`: where each inline-only layer's instances are written; empty without inline-only layers |
| `top_level_extra` | The top-level extra policy |
| `top_level_layers` (property) | `dependency_order` without the inline-only layers |

### `ReceptorLocation`

Frozen dataclass naming one receptor that holds an inline-only layer's configs: `owner_layer`, `owner_key`, `receptor`, `target_layer`, `keys` (the target keys it accepts). `str()` renders it as a config path: `hook[trace_writer].archive`.

### `TopLevelExtra`, `INLINE_ONLY_ERROR_TYPE`

`TopLevelExtra = Literal["forbid", "ignore", "allow"]`. `INLINE_ONLY_ERROR_TYPE = "inline_only_layer"`, the pydantic error type raised when a config writes an inline-only layer at top level (`loc == (layer,)`, message `layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive`). `conscribe.config.composed` also exports the message template `INLINE_ONLY_MESSAGE`, `TOP_LEVEL_EXTRA_MODES` and `INLINE_WIRED_ATTR` (`"__inline_wired__"`).

### `generate_layer_config_source(result) -> str`

A self-contained Python module defining the layer's models and union alias. Fields, defaults and supported validation are reproduced; metadata the renderer cannot carry is omitted with an `ExtractionWarning` and listed in the file header, so those checks remain enforced only in-process. **Raises** `ConfigRenderError` when a type, default or factory cannot be written faithfully (no text is returned then). Also **warns** `ExtractionWarning` for types imported from `__main__`.

### `generate_composed_config_source(result) -> str`

The same for a `ComposedConfigResult`: every layer's models in dependency order, union aliases, the top-level `ComposedConfig` (with the inline-only refusal) and a final `ComposedConfig.model_rebuild()`.

### `generate_layer_json_schema(result) -> dict`

The JSON Schema of `result.union_type`, plus `x-discriminator`, nested mode's `x-discriminator-fields` / `x-key-separator`, and for degraded fields `x-degraded-fields` and per-property `x-degraded-from`.

### `generate_composed_json_schema(result) -> dict`

`result.top_level_type.model_json_schema()` (with set-valued defaults in a stable order). The `x-composed-layers`, `x-inline-wiring` and `x-inline-only-layers` extensions come from the model itself.

### Fingerprints

| Function | Description |
|----------|-------------|
| `compute_registry_fingerprint(registrar) -> str` | A 16-character hex SHA-256 over the conscribe version, the layer's name, discriminator(s), separator and `inline_only`, and the layer's generated source and JSON schema. Changes exactly when regenerating would change the outputs. Never raises: a broken layer hashes a stable error marker. |
| `load_cached_fingerprint(fingerprint_path, layer_name) -> str \| None` | Reads `{layer: fingerprint}` from a JSON file; `None` when missing or corrupt |
| `save_fingerprint(fingerprint_path, layer_name, fingerprint) -> None` | Writes one layer's fingerprint, keeping the others (not atomic) |

See [Fingerprint](config-typing.md#fingerprint-fingerprintpy).

### `MROScope`

`MROScope = Literal["local", "third_party", "all"] | list[str]`

The runtime validator also accepts a tuple, set or frozenset of non-empty package names; the public type alias uses `list[str]`.

### `DegradedField`

Frozen dataclass recording a field whose type was degraded to `Any`: `field_name`, `original_type_repr`, `source_class` (the declaring class), `reason` (`"pydantic_incompatible"`, `"type_checking_only"` or `"function_locals"`). See [Type degradation](mro-and-degradation.md#type-degradation).

---

## Introspection (`conscribe.introspect`)

One rule for "which names reach a constructor, which attributes can receive a wired value, and in what shape". conscribe's extractor, placeholders, stubs and composed builder use it, and so can frameworks built on conscribe (for example to route config values to constructor keywords or `setattr`).

Contract: pure reflection (no registry lookups, no imports); answers reflect the class at call time (inside a metaclass `__new__` a `@dataclass` decorator has not run yet); scope-free (the MRO scope and depth settings only filter what a generated config exposes); identical on Python 3.10 to 3.14; annotation problems never raise here (unresolvable annotations come back raw); malformed `__wiring__` raises from `wiring_slots()`. The result dataclasses are frozen and may gain fields in minor releases, so do not unpack them positionally.

| Function | Returns |
|----------|---------|
| `init_param_names(cls)` | `tuple[str, ...]`: the names `cls(...)` accepts by keyword, child first, following `**kwargs` forwarding up the MRO. Mixins without their own constructor and Protocols with a forwarding placeholder are skipped; a swallowing Protocol placeholder cuts the chain (see [Walk rules](mro-and-degradation.md#walk-rules)). Covers hand-written `__init__`, dataclass and attrs generated ones, pydantic models (field names, not aliases) and pydantic dataclasses. Positional-only parameters are left out. |
| `receptor_slots(cls, *, names=None)` | `dict[str, ReceptorSlot]`: constructor parameters (`via="init"`) first, then non-`ClassVar`, non-dunder class annotations that are not constructor names (`via="annotation"`). Includes slots whether or not they have a wiring entry. Pass `names` to compute only some slots. |
| `wiring_slots(cls)` | `dict[str, WiringSlot]`: every merged `__wiring__` entry with its slot (`None` for a slot-less entry). Needs no registry. |
| `annotation_shape(annotation, owner=None)` | `"scalar"`, `"list"` or `"dict"`. Mappings are `"dict"`, other collections `"list"`; `str`, `TypedDict`, `NamedTuple` are `"scalar"`. Looks through `Annotated`, `Optional` and unions; works on strings (names resolved against `owner`). |
| `is_classvar(annotation, owner=None)` | Whether the annotation is `ClassVar` (also under `Annotated`, also via aliases in string annotations) |
| `config_from_param(cls)` | The `__config_from_param__` value along the MRO, or `None` |

`ReceptorSlot` fields: `name`; `via` (`"init"` or `"annotation"`); `shape`; `annotation` (resolved in the declaring class's namespace, raw when unresolvable, `None` when absent); `declared_on`; `parameter` (the `inspect.Parameter` for `via="init"`, with the real default or the dataclass `<factory>` marker); `default` (the default a config uses, `inspect.Parameter.empty` when none); `default_factory`.

`WiringSlot` fields: `name`, `spec` (the `WiringSpec`), `slot` (`ReceptorSlot | None`), `declare_only`, `default_key`.

`Shape = Literal["scalar", "list", "dict"]`; `Via = Literal["init", "annotation"]`.

```python
from typing import Protocol, runtime_checkable

from conscribe.introspect import init_param_names, receptor_slots, wiring_slots


@runtime_checkable
class Hook(Protocol):
    def fire(self) -> None: ...


class Base:
    def __init__(self, *, timeout: int = 30) -> None: ...


class Agent(Base):
    __wiring__ = {
        "loop": {"registry": "loop", "default": "react"},
        "hooks": "hook",
        "tracer": "tracer",
        "budget": {"registry": "budget", "declare_only": True},
    }
    hooks: list[Hook] = []

    def __init__(self, loop: str = "react", *, steps: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)


print(init_param_names(Agent))
for name, slot in receptor_slots(Agent).items():
    print(f"{name}: via={slot.via} shape={slot.shape} declared_on={slot.declared_on.__name__} default={slot.default!r}")
for name, entry in wiring_slots(Agent).items():
    where = entry.slot.via if entry.slot else None
    print(f"{name}: registry={entry.spec.registry_name!r} slot={where} declare_only={entry.declare_only} default_key={entry.default_key!r}")
```

```
('loop', 'steps', 'timeout')
loop: via=init shape=scalar declared_on=Agent default='react'
steps: via=init shape=scalar declared_on=Agent default=3
timeout: via=init shape=scalar declared_on=Base default=30
hooks: via=annotation shape=list declared_on=Agent default=[]
loop: registry='loop' slot=init declare_only=False default_key='react'
hooks: registry='hook' slot=annotation declare_only=False default_key=None
tracer: registry='tracer' slot=None declare_only=False default_key=None
budget: registry='budget' slot=None declare_only=True default_key=None
```

---

## Stubs (`conscribe.stubs`)

A `.pyi` stub declares the attributes the framework injects, so type checkers see them. Only slot-less, non-`declare_only` wiring entries are stubbed, typed as the target registry's protocol (`str` for a Mode 3 list). Entries with a constructor parameter or class annotation are already declared by the source.

### `write_layer_stubs()`

```
write_layer_stubs(
    registrar,
    output_dir: str | Path | None = None,
    *,
    partial_class_fallback: bool = False,
    prune: bool = False,
) -> list[Path]
```

Writes one stub per module of the layer that has something to declare. A stub stands for its whole module, so it includes stub-needing classes from all registries currently in the global index, plus the requested registrar even if its name has been shadowed. Other shadowed registries are not included. Modules with nothing to declare get no file. Everything is validated and rendered before the first file is written.

| Parameter | Description |
|-----------|-------------|
| `output_dir` | Mirror module paths under this directory (`pkg/mod.pyi`, `pkg/__init__.pyi`). `None`: write next to each source file, never into an installed package or the standard library. |
| `partial_class_fallback` | Add `def __getattr__(self, name: str) -> Incomplete: ...` to each class so type checkers tolerate dynamic attributes |
| `prune` | Delete stale stubs instead of warning about them |

**Returns** the written paths, sorted by module. **Raises** `EmptyLayerError`, `WiringResolutionError` (also for another layer's class sharing a module), `ConfigRenderError` (a stub cannot be written faithfully, or has nowhere to go), `ConfigAnnotationError`.

A conscribe-generated `.pyi` file (recognised by its `# Generated by conscribe` header, or the older `# Auto-generated by conscribe.` followed by `# Stub file for:`) is stale when it sits in a directory this layer's stubs go to, this run did not write it, and no class in the registries considered above needs it. Stale stubs are listed in a `StaleStubWarning`; `prune=True` deletes exactly those files and nothing else.

### Lower-level functions

| Function | Description |
|----------|-------------|
| `collect_class_stub_info(cls) -> ClassStubInfo \| None` | Resolves the class's wiring (`WiringResolutionError` propagates) and collects what its stub needs; `None` when it injects nothing |
| `collect_class_members(cls, *, skip=frozenset()) -> ClassMembers` | The class's own surface: constructor, methods, properties, cached properties, class-level annotations and values, nested classes |
| `generate_module_stub(module_name, classes, *, partial_class_fallback=False) -> str` | The `.pyi` source for the given classes of one module; `""` for no classes. Raises `ConfigRenderError` rather than return an invalid stub, and `ConfigAnnotationError` for annotations naming something that exists nowhere. |
| `narrowest_common_base(classes, fallback) -> type` | The most specific common ancestor of `classes`. Kept as a utility; stubs no longer use it. |

| Dataclass | Fields |
|-----------|--------|
| `ClassStubInfo` | `cls`, `class_name`, `module`, `source_file`, `bases`, `injected_attrs`, `init` (`MethodStub \| None`, replaces 1.x `init_signature`), `methods`, `class_attrs`, `values`, `nested` |
| `InjectedAttr` | `name`, `resolved_type` (the registry protocol or `str`), `registry_name` |
| `MethodStub` | `name`, `function`, `decorators`, `is_async`, `kind` (`"function"`, `"contextmanager"`, `"asynccontextmanager"`, `"opaque"`) |
| `ClassAttrStub` | `name`, `annotation` (as declared), `owner`, `has_default` |
| `ClassValueStub` | `name`, `value`, `owner`: an unannotated class-level value |
| `ClassMembers` | `init`, `methods`, `class_attrs`, `values`, `nested` |

`render_signature` was removed in 2.0.

```python
from conscribe.stubs import collect_class_stub_info, generate_module_stub
from testing_app import Clicker, TypedClicker

info = collect_class_stub_info(Clicker)
print(info.injected_attrs)
print(collect_class_stub_info(TypedClicker))  # its wired name has a slot: no stub
print(generate_module_stub("testing_app", [info]))
```

```
(InjectedAttr(name='action', resolved_type=<class 'testing_app.ActionProtocol'>, registry_name='action'),)
None
# Generated by conscribe 2.0.0
# DO NOT EDIT. Stub for module: testing_app
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

def __getattr__(name: str) -> Any: ...

class AgentBase(Any): ...

@runtime_checkable
class ActionProtocol(Protocol):
    def act(self) -> str: ...

class Clicker(AgentBase):
    action: ActionProtocol  # wired from: action
```

`testing_app` is the package listed in [Testing with conscribe](testing.md#the-pytest-plugin). Objects defined in the stubbed module itself are defined in the stub: nominal classes as `class X(Any): ...` placeholders, Protocols structurally.

---

## Testing (`conscribe.testing`)

| Name | Description |
|------|-------------|
| `isolated_registries(*, warn_kept=False) -> RegistryIsolation` | A context manager that restores every registry, the global index and `discover()`'s failure record on exit, except registrations made by modules first imported inside the block |
| `RegistryIsolation` | The block object. Attributes after exit: `kept`, `rolled_back` (lists of `(registry_name, key, cls)`); property `active`; method `activate()` returns a block that re-applies this object's capture. `warn_kept=True` emits an `IsolationWarning` naming the kept registrations. |
| `conscribe.testing.pytest_plugin` | The opt-in plugin: fixture `conscribe_isolated`, ini option `conscribe_warn_kept`. Enable it with `pytest_plugins = ["conscribe.testing.pytest_plugin"]` in the root `conftest.py`. |

`RegistryIsolation.activate()` raises `RuntimeError` when nothing was captured yet, or while the capture's own block is open, and `DuplicateKeyError` (on enter) when a captured key is now bound to an unrelated class. Out-of-order exits raise `RuntimeError`. See [Testing with conscribe](testing.md).

---

## Errors and warnings

All in `conscribe.exceptions`; everything except `WIRING_PROBLEM_KINDS` is also exported from `conscribe`.

| Error | Bases | Raised when | Attributes |
|-------|-------|-------------|------------|
| `RegistryError` | `Exception` | Base class of every conscribe error | |
| `DuplicateKeyError` | `RegistryError` | A key is taken in the layer (also by `activate()`) | `layer_name`, `key`, `existing_cls`, `new_cls`, `hint` |
| `KeyNotFoundError` | `RegistryError`, `KeyError` | `get()` / `remove()` / `unregister()` with an unknown key | `layer_name`, `key`, `available_keys`, `hint` |
| `InvalidRegistryKeyError` | `RegistryError`, `TypeError` | A key is not a non-empty str (or a non-empty list of them for `__registry_key__`) | `cls_name`, `key`, `source`, `detail` |
| `ProtocolViolationError` | `RegistryError`, `TypeError` | A class misses protocol methods (path C and `protocol_check=True`) | `layer_name`, `cls`, `missing_methods`, `protocol` |
| `InvalidConfigSchemaError` | `RegistryError`, `TypeError` | `__config_schema__` is not a `BaseModel` subclass | `cls_name`, `actual` |
| `InvalidProtocolError` | `RegistryError`, `TypeError` | The protocol is not `@runtime_checkable` | `protocol` |
| `CircularWiringError` | `RegistryError` | Wiring between the layers of a composed build forms a cycle | `cycle` (one actual cycle) |
| `WiringResolutionError` | `RegistryError` | Any `__wiring__` problem; all problems of the class or build at once | `problems`; `cls_name`, `param_name`, `registry_name`, `detail` mirror `problems[0]` |
| `ConfigAnnotationError` | `RegistryError` | An annotation a config field needs names something that exists nowhere, or its `Annotated` metadata cannot be evaluated | `owner`, `field`, `annotation`, `detail` |
| `ConfigDeclarationError` | `RegistryError` | A class declares something contradictory or unsupported | `owner`, `field`, `detail` |
| `ConfigRenderError` | `RegistryError` | A value, type or factory cannot be written faithfully into generated source or a stub | `model`, `field`, `value_repr`, `detail` |
| `EmptyLayerError` | `RegistryError` | A layer passed to a config builder or stub writer has no registered class | `layer_names`, `import_failures` |

| Warning | Emitted when |
|---------|--------------|
| `ConscribeWarning` (`UserWarning`) | Base class of every conscribe warning |
| `ExtractionWarning` | conscribe cannot read something and continues with less: an unreadable signature (in a config or a stub), names local to a function, an attrs `Factory(takes_self=True)` default, metadata generated source cannot carry, a type imported from `__main__`, a module `discover(strict=False)` skipped, a file `scan` cannot parse |
| `DegradationWarning` | A field's type was degraded to `Any` because it is pydantic-incompatible or imported only under `TYPE_CHECKING`; function-local names use `ExtractionWarning` instead |
| `StaleStubWarning` | `write_layer_stubs()` found a conscribe-generated `.pyi` in a directory this layer's stubs go to, this run did not rewrite it, and no class in the registries the writer considers needs it |
| `IsolationWarning` | `isolated_registries(warn_kept=True)` kept registrations |

`ExtractionWarning` and `DegradationWarning` raised while a class is turned into a config point at the class definition (file and line of your code), so filters and IDEs lead to the code to fix. To make conscribe warnings errors, use pytest's `filterwarnings = error::conscribe.ConscribeWarning` or `warnings.simplefilter("error", ConscribeWarning)`; Python's `-W` option cannot name third-party categories (see [Turning conscribe warnings into errors](testing.md#turning-conscribe-warnings-into-errors)).

### `WiringProblem`

```
WiringProblem(owner, param_name, registry_name, kind, detail, owner_key=None)
```

One problem of one `__wiring__` entry. `owner` is the class's qualified name, `owner_key` its registry key, `param_name` the entry (`""` when the whole `__wiring__` is malformed), `registry_name` the target (`""` when not applicable), `kind` one of `WIRING_PROBLEM_KINDS`, `detail` the explanation with the fix. `str()` renders `"{owner} (key '{owner_key}') wiring '{param}': {detail}"`, omitting the key clause when `owner_key is None`; when the entry was inherited, the detail names the class that declared it.

`WiringResolutionError(problems=[...])` is the normal form; the 1.x four-argument form `WiringResolutionError(cls_name, param_name, registry_name, detail)` still works and creates one problem of kind `"unresolved"`.

### `WIRING_PROBLEM_KINDS`

A `frozenset[str]` of the stable problem codes below.

| Kind | Meaning |
|------|---------|
| `malformed` | Grammar: bad tuple shape, wrong types, unknown dict fields, invalid entry name, a key both required and optional, `declare_only` with `default` |
| `not_a_dict` | `__wiring__` itself is neither a dict nor `None` |
| `non_str_key` | A key in a key list is not a str |
| `empty_keys` | An explicit key list (Mode 2, Mode 3 or dict form) is empty |
| `unknown_registry` | The target registry does not exist (with a "did you mean") |
| `empty_registry` | The target registry holds no keys (the message cites `discover()` failures) |
| `unknown_key` | A required key is not registered (capability-relative keys included) |
| `unknown_optional_key` | An optional key is not registered |
| `bad_default` | A `default` key is unregistered, not allowed, not fully qualified, on a slot-less or container receptor, or contradicted by the receptor's own default; or a receptor's own default is not an allowed key |
| `contradiction` | A `declare_only` entry whose class has a runtime slot for it |
| `unsupported` | Reserved; not raised in 2.0 (wiring on nested-mode classes is a `ConfigDeclarationError`) |
| `unresolved` | Only from the legacy four-argument constructor |

---

## Class attributes

Attributes you set:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `__abstract__` | `bool` | Skip registration of this class (read from the class namespace, not inherited) |
| `__registry_key__` | `str \| list[str]` | Explicit key, or several keys (read from the namespace; validated) |
| `__skip_registries__` | `list[str]` | Registry names this class does not register in (this class only; not inherited) |
| `__registration_filter__` | `staticmethod` | On a parent: called with each subclass (it is inherited, so it sees all descendants); return `False` to skip that subclass |
| `__propagate__` | `bool` | On a parent: `False` stops its subclasses from registering |
| `__propagate_depth__` | `int` | On a parent: only this many levels of subclasses below it register |
| `__wiring__` | `dict \| None` | Cross-registry wiring (see [`__wiring__` entries](#__wiring__-entries)); not supported in nested-mode layers |
| `__config_schema__` | `type[BaseModel]` | Tier 3: extraction returns this model itself; the per-key subclass adds the discriminator and sets unset `extra` to `"forbid"` (not supported in nested-mode layers) |
| `__config_from_param__` | `str \| None` | Flatten the config from one `BaseModel` constructor parameter; inherited; `None` opts out (not supported in nested-mode layers) |
| `__config_annotated_only__` | `bool` | Only parameters declared with a `Field` become config fields |
| `__config_mro_scope__` | `MROScope` | Per-class MRO scope (validated; refused in nested-mode layers) |
| `__config_mro_depth__` | `int \| None` | Per-class MRO depth (validated; refused in nested-mode layers) |

Attributes conscribe sets on registered classes: `__registry_name__` (every class the metaclass creates), `__registry_key__` (the first key), `__registry_keys__` (all keys, multi-key only), `__registry_origin__` (the class's `id()`, used to recognise re-created classes) and `WiredField` placeholders.

Attributes on the config models conscribe builds:

| Attribute | Contents |
|-----------|----------|
| `__conscribe_generated__` | `True` in the model's own `__dict__`: code generation defines these models and imports every other type |
| `__wired_fields__` | `{field: registry name}` (`"literal"` for a Mode 3 list) |
| `__wired_keys__` | `{field: (key, ...)}`: the allowed keys, in `Literal` order |
| `__wired_defaults__` | `{field: key}`: default keys of single-instance receptors |
| `__inline_wired__` | Composed builds: `{field: {"target_layer", "keys", "container", "optional"}}` |
| `__degraded_fields__` | `[DegradedField, ...]` |
| `__config_from_param__` | The flattened parameter (opt-in models only) |
| `__legacy_flat_keys__` | The 1.x flat keys the model rejects with error type `legacy_flat_config` |
