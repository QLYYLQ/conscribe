# Registration Deep-Dive

`conscribe/registration/` puts classes into layer registries as they are defined, `conscribe/wiring.py` installs placeholders for the attributes a framework injects, and `conscribe.discover` imports the modules that hold the classes. For usage, see [Guide: Framework Developer](guide-alice.md); for signatures, see the [API Reference](api-reference.md).

## Architecture

```
create_registrar(name, protocol, ...)
    -> LayerRegistry(name, protocol, separator=...)   # storage, added to the global index
    -> create_auto_registrar(registry, key_transform, ...)   # the registration metaclass
    -> build_filter_chain(...)                         # skip conditions
    -> "<Name>Registrar", a LayerRegistrar subclass    # the facade you use
```

`create_registrar("agent", ...)` returns a class named `AgentRegistrar`. All its methods are classmethods (`Agent.get("react")`, `Agent.keys()`), and it exposes the metaclass as `Agent.Meta`.

## Three paths, one tail

Three registration paths feed the same `LayerRegistry`:

| Path | Mechanism | Protocol check |
|------|-----------|----------------|
| A (inheritance) | The metaclass registers each class as it is created | No (inheritance guarantees compliance) |
| B (bridge) | `bridge(External)` creates an abstract base class; its subclasses take path A | No |
| C (manual) | `@Registrar.register(key)`; with `propagate=True` it also installs an `__init_subclass__` hook so subclasses register | Yes (once per class) |

All three end in the same steps, in this order:

1. Validate the keys. A bad key raises `InvalidRegistryKeyError` before any registry changes.
2. Check the `__wiring__` grammar with `check_wiring_declaration()`. A malformed declaration raises `WiringResolutionError`, so the class statement fails and nothing is registered. Only the grammar is checked here; registry lookups happen when a config is built.
3. Add the class under every key. Path C runs the Protocol check on the first add.
4. Stamp the class: `__registry_key__` (the first key), `__registry_keys__` (all keys, multi-key registration only) and `__registry_origin__ = id(cls)` (see [Class re-creation](#class-re-creation)).
5. Install the `WiredField` placeholders (see [WiredField placeholders](#wiredfield-placeholders)).

Before these steps the metaclass also tags every class it creates with `__registry_name__`, recognises re-created classes, runs the skip filters, and checks that a `__config_schema__` is a `BaseModel` subclass (`InvalidConfigSchemaError` otherwise). A class the filters skip still gets its placeholders, because its subclasses inherit them. Its `__wiring__` is checked on every subclass that does register, since those inherit the entries.

## Keys

The metaclass picks the keys in this order:

1. `__registry_key__` in the class's own namespace: one key (`"react"`) or a list (`["react", "re_act"]`).
2. Hierarchical derivation, when the registrar has a `key_separator`: the nearest base class whose own `__dict__` holds a string `__registry_key__` (registered or abstract) becomes the prefix, so `AzureOpenAI` under a base keyed `"openai"` becomes `"openai.azure_open_ai"`.
3. The key transform applied to the class name.

When choosing a class's registration key, `__registry_key__` is read from its own namespace, never inherited; the same rule applies to `__abstract__`. The resulting key is still an ordinary class attribute, so later reflection can read an inherited value.

```python
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, prompt: str) -> str: ...


LLM = create_registrar("llm", ChatModel, discriminator_field="provider", key_separator=".")


class BaseLLM(metaclass=LLM.Meta):
    __abstract__ = True

    def complete(self, prompt: str) -> str:
        return prompt


class OpenAIFamily(BaseLLM):
    __abstract__ = True
    __registry_key__ = "openai"  # an abstract hierarchy root


class AzureOpenAI(OpenAIFamily):  # derived key
    pass


class Universal(BaseLLM):
    __registry_key__ = ["openai.universal", "anthropic.universal"]


print(LLM.keys())
print(Universal.__registry_key__, Universal.__registry_keys__)
print(LLM.children("openai"))
print(LLM.tree())
```

```
['openai.azure_open_ai', 'openai.universal', 'anthropic.universal']
openai.universal ['openai.universal', 'anthropic.universal']
{'openai.azure_open_ai': <class '__main__.AzureOpenAI'>, 'openai.universal': <class '__main__.Universal'>}
{'openai': {'azure_open_ai': <class '__main__.AzureOpenAI'>, 'universal': <class '__main__.Universal'>}, 'anthropic': {'universal': <class '__main__.Universal'>}}
```

A class registered under several keys keeps the first one in `__registry_key__` and all of them in `__registry_keys__`.

### Key validation

A registry key is a non-empty `str`; `__registry_key__` may also be a non-empty list of them. Anything else would register the class under a key nobody can look up (`str(Kind.REACT)` is `'Kind.REACT'`, not `'react'`), so it is refused when the class is created, before it touches any registry. The value a key transform returns is validated the same way.

The examples in this section and the next use this setup:

```python
import enum
from typing import Protocol, runtime_checkable

from conscribe import DuplicateKeyError, InvalidRegistryKeyError, create_registrar


@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, task: str) -> str: ...


Agent = create_registrar("agent", AgentProtocol, discriminator_field="name")


class BaseAgent(metaclass=Agent.Meta):
    __abstract__ = True

    def run(self, task: str) -> str:
        return task
```

```python
class Kind(enum.Enum):
    REACT = "react"

try:
    class ReactAgent(BaseAgent):
        __registry_key__ = Kind.REACT
except InvalidRegistryKeyError as exc:
    print(exc)
```

```
ReactAgent: invalid registry key <Kind.REACT: 'react'> from __registry_key__: a registry key must be a plain str, not the Enum member <Kind.REACT: 'react'> (str() of it is 'Kind.REACT', so the class would be registered under a key nobody looks up); write the plain string 'react'
```

All Enum members are refused, including `StrEnum` / `str, Enum` members; use their string value. Tuples, ints, empty strings and empty lists are refused too, each with its own fix in the message. `register(key=...)` takes exactly one key: to register a class under several keys, apply `register()` once per key, or set `__registry_key__ = [...]` on a class the metaclass creates.

### Duplicate keys

A registry keeps exactly one class per key. A second class under a taken key raises `DuplicateKeyError`, and the message adapts to the cause. The most common one in 2.0 is a class body that runs a second time:

```python
def define():
    class PlanAgent(BaseAgent):
        pass

    return PlanAgent


define()
try:
    define()  # the same class body runs a second time
except DuplicateKeyError as exc:
    print(exc)
```

```
Duplicate key 'plan_agent' in 'agent' registry: 'define.<locals>.PlanAgent' conflicts with existing 'define.<locals>.PlanAgent'. Both are '__main__.define.<locals>.PlanAgent': the class body ran a second time (importlib.reload, a re-executed notebook cell, a test or helper function that defines the class, or a module imported under two names), and a registry keeps exactly one class per key. In tests, define such classes inside 'with conscribe.testing.isolated_registries():' (or use the 'conscribe_isolated' pytest fixture) so every run starts from the registry state it had before.
```

See [Testing with conscribe](testing.md) for `isolated_registries()`. Other causes get their own advice: the same class registered twice (its key listed twice, or `register()` applied twice), and a subclass with the same name as a registered class, which ends up with the same key (stacking `@pydantic.dataclasses.dataclass` on a stdlib `@dataclass` creates one).

## Atomic registration

A class statement that fails leaves nothing registered.

On paths A and B, if anything fails after the class object exists (a collision on the second key of a multi-key class, the second registry of a cross-registry diamond, an invalid `__config_schema__`, a failing `__init_subclass__` hook), the metaclass removes the class from every live registry before the error propagates. A live registry is any registry that still exists, including one whose name a later registrar took over.

```python
from typing import Protocol, runtime_checkable

from conscribe import DuplicateKeyError, create_registrar


@runtime_checkable
class Runnable(Protocol):
    def run(self) -> None: ...


LLM = create_registrar("llm", Runnable, discriminator_field="provider")
Agent = create_registrar("agent", Runnable, discriminator_field="name")


class LLMBase(metaclass=LLM.Meta):
    def run(self) -> None: ...


class Planner(LLMBase):
    __registry_key__ = "planner"


# Registers in both layers; the right-hand operand of | registers first.
class LLMAgent(metaclass=LLM.Meta | Agent.Meta):
    def run(self) -> None: ...


try:
    class SelfPlanning(LLMAgent):
        __registry_key__ = "planner"  # added to "agent" first, then refused by "llm"
except DuplicateKeyError as exc:
    print(exc)

print("llm:", LLM.keys())
print("agent:", Agent.keys())
```

```
Duplicate key 'planner' in 'llm' registry: 'SelfPlanning' conflicts with existing 'Planner'. To fix: add __registry_key__ = '<unique_key>' to SelfPlanning.
llm: ['planner']
agent: []
```

`SelfPlanning` was already in the `agent` registry when the `llm` registry refused it; the failed class statement removed it again.

On path C, `register()` undoes exactly its own work when it fails after the add: the binding it made and every attribute it wrote. Bindings the class has elsewhere are left alone, including those of another `@register` stacked below it on the same class statement. When an outer `@R2.register(...)` fails, the inner `@R1.register(...)` stays registered; unregister it, or let `isolated_registries()` roll it back.

The propagation hook (`register(propagate=True)`) removes a subclass from every live registry when registering it fails part-way. One case cannot be undone: the hook registers the subclass while `type.__new__` runs `__init_subclass__`, so if another `__init_subclass__` in the MRO raises *after* the hook returned (a validating mixin placed before the decorated class), the class statement fails but the subclass stays registered. Put such validation in the subclass's own hook before calling `super().__init_subclass__()`, or unregister the key.

## Class re-creation

`@dataclass(slots=True)`, `attrs.define` and pydantic's `dataclass(slots=True)` replace a freshly defined class with a new class built from a copy of its namespace. The metaclass registered the first class; the copy has to take its place.

Every registered class is stamped with `__registry_origin__ = id(cls)`. The copied namespace carries that stamp, so when the metaclass sees a new class whose namespace holds the origin of a registered class with the same `__module__`, `__name__` and keys, it rebinds every key of the old class, in every registry, to the new class. Then it stamps the new class and skips normal registration:

```python
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import attrs

from conscribe import create_registrar


@runtime_checkable
class Tool(Protocol):
    def call(self) -> str: ...


Tools = create_registrar("tool", Tool, discriminator_field="kind")


class ToolBase(metaclass=Tools.Meta):
    def call(self) -> str:
        return type(self).__name__


@dataclass(slots=True)
class Search(ToolBase):
    top_k: int = 5


@attrs.define
class Fetch(ToolBase):
    retries: int = 3


print(Tools.get("search") is Search, "__slots__" in Search.__dict__)
print(Tools.get("fetch") is Fetch, "__slots__" in Fetch.__dict__)
print(Search.__registry_origin__ == id(Search))
```

```
True True
True True
True
```

A class body that runs a second time carries no stamp, so it still raises `DuplicateKeyError` (see [Duplicate keys](#duplicate-keys)).

Not supported:

- A path C `@register` placed *below* `@dataclass(slots=True)` / `attrs.define`. The registration stays on the replaced class. Put `@register(...)` on top.
- `@pydantic.dataclasses.dataclass` stacked on a stdlib `@dataclass`. pydantic wraps the class in a same-named subclass, which collides with the original's key; the `DuplicateKeyError` message says so. Use one dataclass decorator on registered classes.

## AutoRegistrarBase and the `|` operator

All conscribe metaclasses inherit from `AutoRegistrarBase`, whose own metaclass (`MetaRegistrarType`) defines `|`. That makes cross-registry diamond inheritance a one-liner: `class LLMAgent(metaclass=LLM.Meta | Agent.Meta)` creates a base whose concrete subclasses register in both layers (see the example in [Atomic registration](#atomic-registration)).

When `|` is applied:

- if one metaclass already subclasses the other, the more specific one is returned;
- otherwise a combined metaclass inheriting from both is created, and both registration steps run through the `super()` chain. The right-hand operand's registry adds the class first: with `LLM.Meta | Agent.Meta`, `agent` before `llm`.

`MetaRegistrarType` also provides the read-only `Meta.inline_only` property (see [Inline-only layers](#inline-only-layers)). A combined metaclass reports the flag of its first base.

## Filters

`conscribe/registration/filters.py` decides which classes are skipped. Each filter implements `should_skip(ctx) -> bool`, and a class is skipped when any filter says so. The same chain runs in the metaclass (paths A and B) and in the propagation hook (path C). A filter receives a frozen `RegistrationContext` with five fields: `cls` (the class), `name` (its name), `bases` (its direct bases), `namespace` (its class namespace) and `registry_name` (the target registry).

| Filter | Skips when | Notes |
|--------|------------|-------|
| `RootFilter` | `not ctx.bases` | A class without any base (`class Base(metaclass=Agent.Meta)`) is never registered; `class Base(Mixin, metaclass=Agent.Meta)` has a base and needs `__abstract__ = True` to stay out |
| `PydanticGenericFilter` | `"[" in ctx.name` | Pydantic generic intermediates such as `BaseEvent[str]`; off with `skip_pydantic_generic=False` |
| `CustomCallableFilter(fn)` | `fn(ctx.cls)` is true | `create_registrar(skip_filter=fn)` |
| `AbstractFilter` | `__abstract__ = True` in the namespace | Not inherited |
| `ChildSkipFilter` | the registry name is in the class's own `__skip_registries__` | Opt out of one layer of a diamond |
| `ParentRegistrationFilter` | the `__registration_filter__(cls)` a direct base defines or inherits returns false | A `@staticmethod` on a parent; it sees every descendant |
| `PropagationFilter` | a direct base defines or inherits `__propagate__ = False`, or the class is deeper than `__propagate_depth__` below the class that set it | Depth 1 is a direct subclass |

## LayerRegistry

`LayerRegistry` is the thread-safe key-to-class store, one per layer. It knows nothing about the registration paths. It keeps strong references in a `dict[str, type]` guarded by a `threading.Lock`, and `separator` holds the key separator (`""` for flat keys).

Protocol checks (path C only) are cached the way `ABCMeta` caches subclass checks: classes that passed go into a positive `WeakSet`, classes that failed into a negative one. Each registry bumps an invalidation counter whenever a binding is removed, which clears its negative cache.

Creating a `LayerRegistry` (which `create_registrar()` does) adds it to a process-wide index under its name, and `get_registry(name)` and wiring resolution read that index. A second registry with the same name takes the name over. The old registry stays alive and tracked, but `get_registry()` no longer returns it. An `isolated_registries()` block gives the name back on exit unless the new registry was created by a module first imported inside the block and retained by the [keep rule](testing.md#the-keep-rule).

| Method | Notes |
|--------|-------|
| `add(key, cls, *, protocol_check=False)` | `DuplicateKeyError` if the key is taken; `ProtocolViolationError` if `protocol_check` fails |
| `remove(key, *, missing_ok=False)` | Returns the removed class; see below |
| `get(key)` | `KeyNotFoundError` if missing |
| `get_or_none(key)` | `None` if missing |
| `keys()` / `items()` | Snapshot lists |
| `children(prefix)` | Entries whose key starts with `prefix + separator`; `{}` without a separator |
| `tree()` | Nested dict split on the separator; `ValueError` if a key is both a leaf and a prefix |
| `runtime_check(cls)` | The Protocol check |

### Removing keys

`LayerRegistry.remove(key, *, missing_ok=False)` and `LayerRegistrar.unregister(key, *, missing_ok=False)` remove exactly one binding and return the class it pointed to:

- A class registered under several keys stays registered under its other keys, and its `__registry_key__` / `__registry_keys__` attributes are left as they are.
- Hierarchical keys are independent entries: removing `"openai"` does not remove `"openai.azure"`. If you try, the error lists the keys below the prefix.
- A missing key raises `KeyNotFoundError` (a `KeyError` subclass) with removal-specific advice; with `missing_ok=True` it returns `None` instead, so teardown code can run twice.

In tests, prefer `isolated_registries()`: it also restores what a test removed (see [Testing with conscribe](testing.md)).

## Key transform

`conscribe/registration/key_transform.py` infers keys from class names:

1. Strip the first matching suffix, unless that would leave nothing.
2. Strip the first matching prefix, unless that would leave nothing.
3. Convert CamelCase to snake_case with two regex passes.

| Input | `strip_prefixes` | `strip_suffixes` | Output |
|-------|------------------|------------------|--------|
| `ChatOpenAI` | `["Chat"]` | - | `open_ai` |
| `BrowserUseAgent` | - | `["Agent"]` | `browser_use` |
| `HTTPSHandler` | - | - | `https_handler` |
| `MyV2Agent` | - | `["Agent"]` | `my_v2` |
| `Agent` | - | `["Agent"]` | `agent` |

A `key_transform=` callable replaces the whole algorithm. Its result is validated like any other key.

## Bridge metaclass conflict resolution

`LayerRegistrar.bridge(External, *, name=None)` creates an abstract base class (`"<External>Bridge"` by default) whose subclasses register through path A. It picks the bridge's metaclass by checking, in order:

1. The external class's metaclass is plain `type`: use the registrar's `Meta`.
2. `Meta` already subclasses the external metaclass: use `Meta`.
3. The external metaclass subclasses `Meta` (it is more specific): use the external one.
4. Otherwise: create a combined metaclass `"<bridge name>Meta"` inheriting from both.

When the two metaclasses cannot be combined, `bridge()` raises `TypeError` and suggests `create_registrar(base_metaclass=...)`. The bridge class has `__abstract__ = True`, so it is not registered itself.

## Path C: propagation

`@register(propagate=True)` replaces the decorated class's `__init_subclass__` with a hook that registers future subclasses, so inheritance-based registration works without the metaclass. The hook:

- finds the class that holds it at call time, so it keeps working after `@dataclass(slots=True)` / `attrs.define` re-created the decorated class;
- recognises a subclass re-created from a registered subclass's namespace and rebinds its keys instead of registering it again;
- installs the `WiredField` placeholders of every subclass, registered or skipped;
- registers through the shared tail (key validation, wiring grammar check, add, stamp, placeholders), and accepts a list of keys in `__registry_key__`;
- runs the same filter chain as the metaclass, so `__skip_registries__`, `__registration_filter__`, `__propagate__` and `__propagate_depth__` behave the same on every path.

## Inline-only layers

`create_registrar(..., inline_only=True)` (or `create_auto_registrar(..., inline_only=True)`) declares that the layer's instances may only be written inside the wiring receptors of other layers. Registration itself does not change. The flag is readable as `Registrar.inline_only` and `Registrar.Meta.inline_only`, never as an attribute of the registered classes (where it could be mistaken for a dataclass or pydantic field default). A non-bool value raises `TypeError`. What the flag does in a composed config is described in [Config Typing Deep-Dive](config-typing.md#inline-only-layers).

## WiredField placeholders

A `__wiring__` entry names a value the framework supplies, unless it is `declare_only`. When the class has a constructor parameter for it, the value arrives through the constructor. Otherwise the framework assigns the attribute on the instance (`agent.env = env`). conscribe installs a `WiredField` where the rules below call for a placeholder. It is a non-data descriptor: before assignment, instance access raises an `AttributeError` naming the attribute and its registry.

A *runtime slot* is a constructor parameter or a non-`ClassVar` class annotation, including inherited ones. Configs use this definition to decide requiredness; `.pyi` generation adds injected fields only for entries without a runtime slot. Placeholder installation has a different job: it protects attribute reads before assignment and must also work before decorators create constructor parameters.

The examples in this section continue one script:

```python
import inspect
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from conscribe import create_registrar, wired_field, wired_fields


@runtime_checkable
class EnvProtocol(Protocol):
    def reset(self) -> None: ...


@runtime_checkable
class AgentProtocol(Protocol):
    def run(self) -> None: ...


Env = create_registrar("env", EnvProtocol, discriminator_field="name")
Agent = create_registrar("agent", AgentProtocol, discriminator_field="name")


class EnvBase(metaclass=Env.Meta):
    def reset(self) -> None: ...


class Browser(EnvBase):
    pass
```

```python
class BaseAgent(metaclass=Agent.Meta):
    __abstract__ = True
    __wiring__ = {"env": "env"}  # the framework sets agent.env after construction

    def run(self) -> None: ...


class ReactAgent(BaseAgent):
    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps


print(ReactAgent.env)                       # class access: the placeholder
agent = ReactAgent()
try:
    agent.env
except AttributeError as exc:
    print(exc)
agent.env = Browser()                       # injection is a plain assignment
print(type(agent.env).__name__)
```

```
WiredField('env', registry='env')
Wired attribute 'env' on ReactAgent has not been injected yet. Expected injection from registry 'env'.
Browser
```

### Who gets a placeholder

Placeholders are installed when the class is created, by every registration path, also for classes the filters skip. That is *before* class decorators such as `@dataclass` run. Each merged `__wiring__` entry gets one unless:

- the entry is `declare_only` (nothing is injected for it);
- the class body binds the name (a default, `dataclasses.field(...)`, a property, a method, a hand-written placeholder). A class's own bindings are never replaced;
- the class body annotates the name (`env: Environment`). The class declares the slot itself, and attrs would read a placeholder in the class `__dict__` as the field default;
- an inherited binding can hold or produce the value: a plain value (an inherited default), a writable descriptor (a `__slots__` member, a property with a setter), a `functools.cached_property`, or a placeholder for the same registry;
- the name is a constructor parameter: a hand-written `__init__` parameter, a parameter forwarded through `**kwargs` up the MRO, a field of an already processed dataclass, or a pydantic model field.

A placeholder *does* replace an inherited binding that cannot receive the value or would hand out a wrong one before injection: a read-only property, a method (typically a Protocol or mixin stub of the member), a placeholder for another registry (the subclass re-wired the name), or a tombstone.

Only an annotation in the new class's own body prevents a new placeholder. An annotated class can still inherit a placeholder, and its unannotated subclass can get one if no inherited binding suffices. Those annotations remain runtime slots for configs and stubs; finding a `WiredField` does not mean that `.pyi` generation will add a field.

Malformed entries are skipped here; `check_wiring_declaration()` reports them when the class registers.

### Class access

`MyAgent.env` returns the descriptor itself, unless the accessing class, or any class between it and the class holding the descriptor, annotates the name in its own body. Then it raises `AttributeError`, like any annotation-only attribute:

```python
class TypedAgent(BaseAgent):
    env: EnvProtocol                        # annotates the wired name


try:
    TypedAgent.env
except AttributeError as exc:
    print(exc)
print(wired_field(TypedAgent, "env"), inspect.getattr_static(TypedAgent, "env"))
```

```
TypedAgent.env has no class-level value: TypedAgent annotates the wired attribute 'env' without a default, so every instance gets its own value (wired from registry 'env'). Use conscribe.wired_field(TypedAgent, 'env') to look up the injection placeholder, or inspect.getattr_static(TypedAgent, 'env') to read the raw class attribute.
WiredField('env', registry='env') WiredField('env', registry='env')
```

Field collectors depend on this. `@dataclass` treats a descriptor whose class access raises as "no default", and pydantic looks defaults up with `getattr(cls, name, <sentinel>)`, so a dataclass or pydantic subclass that re-annotates an inherited wired name gets a required field and never the placeholder as its default. The check happens at access time, so it also holds for classes created after the placeholder and for the copies `@dataclass(slots=True)` / `attrs.define` make. For such names `hasattr(TypedAgent, "env")` is `False`.

Placeholders never overwrite class-body values, so dataclass defaults keep working:

```python
@dataclass
class DataAgent(BaseAgent):
    env: EnvProtocol                        # a dataclass field: required


print(inspect.signature(DataAgent), wired_fields(DataAgent))
```

```
(env: __main__.EnvProtocol) -> None {}
```

`env` is now a constructor parameter, so `wired_fields()` no longer reports the inherited placeholder.

### Tombstones

A subclass that stops wiring an inherited name, by excluding it (`{"env": None}`) or by redeclaring it `declare_only`, gets a tombstone when it would otherwise inherit a placeholder. A tombstone is a private descriptor that hides that inherited placeholder. Class access raises `AttributeError`, instance access raises Python's plain missing-attribute error naming the subclass, and assigning the attribute on an instance still works:

```python
class OfflineAgent(BaseAgent):
    __wiring__ = {"env": None}              # stops wiring env


try:
    OfflineAgent().env
except AttributeError as exc:
    print(exc)
print(wired_fields(ReactAgent), wired_fields(OfflineAgent))
```

```
'OfflineAgent' object has no attribute 'env'
{'env': WiredField('env', registry='env')} {}
```

### Looking placeholders up

Use `conscribe.wired_field(cls, name)` and `conscribe.wired_fields(cls)` instead of `cls.attr`. They walk each class's `__dict__` along the MRO without triggering descriptors and report a placeholder only when:

- the first binding of the name is a `WiredField`,
- the name is injected according to the merged `__wiring__` (not excluded, not `declare_only`),
- and the name is not a constructor parameter at query time.

The last rule covers dataclasses. A class that annotates a wired name gets no placeholder of its own, may inherit one, and becomes a dataclass field after the metaclass ran. Its value arrives through the constructor, so the inherited placeholder is not reported.

When no placeholder qualifies, `wired_field(cls, name)` returns `None` and `wired_fields(cls)` omits the name. To inspect a raw inherited descriptor even after a decorator has made the name a constructor parameter, use `inspect.getattr_static(cls, name)`.

For patching wired attributes in tests, see [Testing with conscribe](testing.md#patching-wired-attributes).

## `discover()`

Metaclass registration happens when class statements run. `discover(*package_paths, strict=True, allow_missing=(), exclude=())` imports every module of the given packages so that their classes register, and returns the names of the modules it imported (including ones that were already imported).

The example uses this package:

```
tools_pkg/
    __init__.py            # the "tool" registrar and ToolBase
    __main__.py            # a command-line entry point (never imported by the walk)
    search.py              # class Search(ToolBase), key "search"
    vision.py              # class Vision(ToolBase), key "vision"; then `import fancy_ocr`
    generated/__init__.py  # raise RuntimeError("generated code is not meant to be imported")
```

`fancy_ocr` stands for an optional dependency that is not installed. This example uses Python 3.11+ exception notes; on 3.10, catch `ImportError` and inspect its `__cause__` for the original submodule failure.

```python
import warnings

from conscribe import discover
from conscribe.discover import import_failures

try:
    discover("tools_pkg")  # strict: the first failing module stops the walk
except RuntimeError as exc:
    print(type(exc).__name__, exc)
    print(exc.__notes__)

imported = discover(
    "tools_pkg",
    exclude=["tools_pkg.generated"],
    allow_missing=["fancy_ocr"],
)
print(imported)

from tools_pkg import Tool

print(Tool.keys())  # Vision registered, then its module failed: rolled back
print(import_failures())
```

```
RuntimeError generated code is not meant to be imported
["(raised while conscribe.discover() was importing module 'tools_pkg.generated')"]
['tools_pkg', 'tools_pkg.search']
['search']
{'tools_pkg.generated': 'RuntimeError: generated code is not meant to be imported', 'tools_pkg.vision': "ModuleNotFoundError: No module named 'fancy_ocr' (skipped: 'fancy_ocr' is in allow_missing)"}
```

The same package with `strict=False`:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    discover("tools_pkg", strict=False, exclude=["tools_pkg.generated"])
print(caught[0].category.__name__, caught[0].message)
```

```
ExtractionWarning discover() skipped module 'tools_pkg.vision' because importing it raised ModuleNotFoundError: No module named 'fancy_ocr'. Classes defined in it are not registered; fix the module or exclude it with discover(..., exclude=['tools_pkg.vision']).
```

### The walk

Each package is imported first, then its modules and subpackages depth-first, in name order. A plain module is imported and not walked. `__main__` modules are never imported by the walk, since importing `pkg.__main__` runs a command-line entry point; name one explicitly to import it anyway. A module that is already in `sys.modules` is not imported again and costs no bookkeeping.

### Failure policy

| Setting | Behaviour |
|---------|-----------|
| `strict=True` (default) | Any exception raised while importing a module propagates as the original exception. On Python 3.11+ a note naming the module is attached (PEP 678). On 3.10, which has no notes, a failing *submodule* is re-raised as an `ImportError` naming it, chained from the original. |
| `allow_missing=[...]` | Top-level names of optional dependencies. A module whose import fails with a `ModuleNotFoundError` for one of them (or a submodule of one) is skipped, with only a `logging.info` record. The error may also be reached through `__cause__` / `__context__`, but only along a chain of `ImportError`s (`raise ImportError("install the extra") from exc`). A different exception raised while handling the missing import (a bug in a fallback) is a real failure. Applies in both modes, also to the packages you name. |
| `exclude=[...]` | Module or package names that are never imported. A name excludes itself and everything below it: `"pkg.generated"` excludes `pkg.generated` and `pkg.generated.x`, but not `pkg.generated_old`. |
| `strict=False` | A submodule raising `ImportError` or `SyntaxError` is skipped with an `ExtractionWarning` naming the exception. Any other exception still propagates, and the packages you name fail fast unless `allow_missing` covers the failure. |

`KeyboardInterrupt`, `SystemExit` and other non-`Exception` errors always propagate unchanged, after the rollback. Passing anything but non-empty module-name strings in `allow_missing` / `exclude` raises `TypeError`.

### Rollback

While `discover()` imports a module for the first time, it records every registry change and attributes it to the module whose top-level code made it (the innermost module being executed, whatever `__module__` the class claims). Afterwards, the changes made by modules whose import was attempted and failed are undone, newest first. That is why `Vision` is not registered in the example. Registrations of modules that imported successfully are never touched. The rollback also runs when the failure is a `KeyboardInterrupt` or `SystemExit`.

A module may import another module that fails and handle the error itself. The failed module's registrations are rolled back as well, and the failure is recorded.

### Failure record

Every failure is recorded by module name as a string (`"ExcType: message"`), never as an exception object, so tracebacks are not kept alive. `import_failures()` returns a copy of the record and leaves out modules that have been imported since. `clear_import_failures()` forgets everything. Both live in the `conscribe.discover` module; the package attribute `conscribe.discover` is the function, so import them with `from conscribe.discover import import_failures, clear_import_failures`.

Later errors cite the record. `EmptyLayerError` and the wiring problems for unknown or empty registries and unknown keys name the recorded modules, since classes defined in them never registered, which is the usual reason a layer or key is missing. `isolated_registries()` restores the record to its state at entry.
