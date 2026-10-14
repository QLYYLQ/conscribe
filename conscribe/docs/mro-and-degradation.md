# MRO and Type Degradation Deep-Dive

conscribe follows `**kwargs` chains up the MRO to find every constructor parameter (`conscribe/config/mro.py`), and types a config field as `Any` when it cannot use the declared type (`conscribe/config/degradation.py`).

## MRO parameter collection

A constructor that accepts `**kwargs` and passes them to `super().__init__(**kwargs)` accepts its parents' parameters too. Without them the class's config would be incomplete, so conscribe walks the MRO to collect them.

### Walk rules

`walk_init_chain(cls)` (in `conscribe.config.mro`) finds every parameter `cls(...)` accepts:

1. The walk starts at the class that defines the constructor that runs first. The placeholder `__init__` that `typing` puts on Protocols forwards to the next real `__init__` and does not count as a definer; neither does the copy of a parent's `__init__` that `typing` makes when a Protocol subclass is first instantiated. A pydantic model whose constructor only forwards `**data` contributes its `model_fields`; a pydantic dataclass contributes its fields.
2. If that `__init__` takes `**kwargs`, the walk continues up the MRO. A parameter already seen keeps its nearest declaration (child wins).
3. Classes without an `__init__` of their own (`Generic`, `ABC`, Protocols with `typing`'s forwarding placeholder, bridges, plain mixins) are skipped before the scope and depth checks, so they never cut the chain.
4. `pydantic.BaseModel` (which contributes the fields of the pydantic model being walked) and a pydantic dataclass (which contributes its fields) end the chain naturally, also before the scope and depth checks: wherever pydantic is installed, its constructor is where the fields are validated.
5. A class outside the MRO scope, or the depth limit, cuts the chain.
6. An `__init__` whose signature cannot be read also cuts it, and so does the Protocol placeholder of `typing_extensions` on Python < 3.12, which swallows its arguments instead of forwarding them. Both come with an `ExtractionWarning`: conscribe cannot know what lies beyond.
7. A parent whose `__init__` takes no `**kwargs` ends the chain naturally, and so does `object`.

Generated `__init__` methods (`@dataclass`, attrs) are read with the real defaults and factories of their fields.

There are two views of the same walk:

- The runtime view (no scope, no depth) follows the chain wherever it leads, installed packages included. `conscribe.introspect.init_param_names()`, the `WiredField` placeholders, stubs and wiring slots use it.
- The config view cuts the walk where the MRO scope or depth says parent parameters stop being config fields. Only config extraction (and `collect_mro_params()`, which exposes it) uses this view, so the scope and depth settings never change what conscribe considers a constructor parameter.

### Extra policy

Where the chain ends decides whether the config accepts unknown keys. The first row that matches applies:

| Chain | `extra` |
|-------|---------|
| The class is, or the chain ends in, a pydantic model (`BaseModel`) or pydantic dataclass | That model's own `extra`, `"forbid"` when unset |
| Cut by the scope, the depth, an unreadable signature or a swallowing placeholder | `"allow"` |
| Followed to its natural end, with parents contributing parameters | `"forbid"` |
| `**kwargs` through which no parent contributes a parameter (also when they reach a parent whose `__init__` takes none) | `"allow"` |
| `*args` | `"allow"` |
| No `*args` / `**kwargs` | `"forbid"` |

### Example: a chain through mixins

```python
import abc
from typing import Generic, TypeVar

from conscribe import extract_config_schema

T = TypeVar("T")


class Base:
    def __init__(self, *, timeout: int = 30) -> None: ...


class Mixin(Generic[T], abc.ABC):  # no __init__: never cuts the chain
    pass


class Middle(Mixin[int], Base):
    def __init__(self, *, api_key: str, **kwargs) -> None:
        super().__init__(**kwargs)


class Impl(Middle):
    def __init__(self, *, model: str, **kwargs) -> None:
        super().__init__(**kwargs)


for depth in (None, 1, 0):
    config = extract_config_schema(Impl, mro_depth=depth)
    print(f"mro_depth={depth}:", list(config.model_fields), config.model_config["extra"])
```

```
mro_depth=None: ['model', 'api_key', 'timeout'] forbid
mro_depth=1: ['model', 'api_key'] allow
mro_depth=0: ['model'] allow
```

`mro_depth` counts the parent `__init__` definers visited; `0` does not follow `**kwargs` at all. `Mixin`, `Generic` and `ABC` define no `__init__`, so they are skipped without counting.

### Example: a chain ended by a pydantic model

```python
from pydantic import BaseModel

from conscribe import extract_config_schema


class Settings(BaseModel):
    region: str = "us"


class Hybrid(Settings):
    def __init__(self, *, verbose: bool = False, **data) -> None:
        super().__init__(**data)


config = extract_config_schema(Hybrid)
print(list(config.model_fields), config.model_config["extra"])
```

```
['verbose', 'region'] forbid
```

`BaseModel.__init__` ends the chain and contributes the model's fields, whatever the MRO scope says.

## Scope classification

`classify_class_scope(cls)` (in `conscribe.config.mro`) puts a class in one of three groups, `"local"`, `"third_party"` or `"stdlib"`, by the first of these rules that applies:

1. A class created by `LayerRegistrar.bridge()` is `"local"`, wherever conscribe is installed.
2. A class without a source file is `"stdlib"` when its module is a builtin or standard-library module, `"local"` otherwise (for example a class of a module created at runtime).
3. A source path containing `site-packages` or `dist-packages` is `"third_party"`.
4. A source path under the interpreter's standard-library directories is `"stdlib"`.
5. Anything else is `"local"`.

### Scope filtering

| Scope | Includes |
|-------|----------|
| `"local"` (default) | Local classes only |
| `"third_party"` | Local and third-party classes |
| `"all"` | Every class except `object` |
| `["pkg1", "pkg2"]` | Local classes and the third-party classes whose top-level package is listed; never the standard library. The package is the import name right after `site-packages` (or `dist-packages`) in the source path, `six` for `six.py`. |

The list form suits most real code: the config collects a known library's parameters, and other third-party base classes stay out.

```python
from packaging.version import Version

from conscribe import extract_config_schema
from conscribe.config.mro import classify_class_scope


class Pinned(Version):
    def __init__(self, *, channel: str = "stable", **kwargs) -> None:
        super().__init__(**kwargs)


print(classify_class_scope(Pinned), classify_class_scope(Version))
for scope in ("local", ["packaging"]):
    model = extract_config_schema(Pinned, mro_scope=scope)
    print(scope, list(model.model_fields), model.model_config["extra"])
```

```
local third_party
local ['channel'] allow
['packaging'] ['channel', 'version'] forbid
```

### Configuration levels

`extract_config_schema(cls, mro_scope="local", mro_depth=None)` takes the scope and depth as arguments, and a layer build passes the registrar's `mro_scope` / `mro_depth` (from `create_registrar()`) as those arguments. `__config_mro_scope__` / `__config_mro_depth__` on the class, set there or inherited from a base, win over the arguments. An inherited value applies to every subclass, counted from the subclass's own constructor; a subclass can set its own value but cannot fall back to the registrar's (`None` means an unlimited depth).

Flat-mode extraction validates these values when it reads them. A scope must be `"local"`, `"third_party"`, `"all"` or a list (tuple, set or frozenset also accepted at runtime) of non-empty package names; a depth must be `None` or an int >= 0. Anything else is a `ConfigDeclarationError` that says where the value was set:

```python
from conscribe import ConfigDeclarationError, extract_config_schema


class Impl:
    __config_mro_scope__ = "locals"

    def __init__(self, *, model: str, **kwargs) -> None: ...


try:
    extract_config_schema(Impl)
except ConfigDeclarationError as exc:
    print(exc)
```

```
__main__.Impl: __config_mro_scope__ = 'locals': unknown MRO scope 'locals' (did you mean 'local'?); use 'local', 'third_party', 'all' or a list of package names.
```

### Nested-mode layers

Layers built with `discriminator_fields` always follow each class's real constructor: their levels are built from the runtime view of the chain. `__config_mro_scope__` and `__config_mro_depth__` on a nested-mode class are refused with a `ConfigDeclarationError`. Nested-mode builds ignore the registrar's `mro_scope` / `mro_depth`, including invalid values. The registrar's `get_config_schema(key)` still calls the flat extractor, which applies and validates those settings. See [Nested mode](config-typing.md#nested-mode-discriminator_fields).

### `collect_mro_params()`

`collect_mro_params(cls, scope="local", depth=None)` is the 1.x interface to the config view, kept for compatibility. It reports, relative to the class that defines the first `__init__` in the MRO, the parameters the `**kwargs` chain contributes beyond that `__init__`'s own, as a frozen `MROCollectionResult`:

| Field | Contents |
|-------|----------|
| `params` | The parent parameters (`inspect.Parameter`), child first, unique by name |
| `init_definers` | The classes that contributed them |
| `hints` | Their annotations, resolved in the declaring class's namespace (raw when unresolvable) |
| `fully_resolved` | Whether the chain ended naturally |
| `declared_on` | `{parameter name: declaring class}` |
| `unreadable` | Classes whose `__init__` signature could not be read |
| `stop_reason` | Why the walk stopped early: `"scope"`, `"depth"`, `"unreadable"`, `"swallowing_placeholder"`, or `None` |
| `init_params` | The same parameters with their full provenance |

New code should use `conscribe.introspect` for the runtime view.

## Type degradation

A field's type degrades to `Any` when conscribe cannot use it. The field keeps its name, default, description and every other `Field(...)` attribute; only its type stops being validated.

### Reasons

| `DegradedField.reason` | Cause | Warning |
|------------------------|-------|---------|
| `"pydantic_incompatible"` | pydantic cannot build a schema for the type (often a third-party class reached through a `**kwargs` chain: `httpx.Auth`, `ssl.SSLContext`, ...) | `DegradationWarning` |
| `"type_checking_only"` | The annotation needs a name the declaring module only imports under `if TYPE_CHECKING:`, which conscribe never imports | `DegradationWarning` |
| `"function_locals"` | The class is defined inside a function and the annotation needs that function's locals, which conscribe cannot see | `ExtractionWarning` |

A name that exists nowhere (a typo, a missing import) is not degraded: it raises `ConfigAnnotationError`. See [Annotation resolution](config-typing.md#annotation-resolution).

### Try first

pydantic-incompatible types are found lazily. The extractor builds the model optimistically and only when pydantic raises a schema error does it probe the fields one by one with `TypeAdapter(tp)`. Classes with standard types never pay for the probe. The degraded type keeps its `Annotated` metadata when pydantic can apply it to `Any` (`Annotated[Any, <metadata>]`), so constraints and markers survive; otherwise it is plain `Any`. A wired field can never degrade: metadata pydantic cannot apply to a registry key is a `ConfigDeclarationError`.

### Warnings

Every degradation emits a warning located at the `class` statement of the class that declares the field, so warning filters, `pytest.warns` and IDEs point at the code to fix. The message names the field, the original type and the declaring class, and the fix when there is one (for a `TYPE_CHECKING`-only name: import it at runtime). `DegradationWarning` and `ExtractionWarning` are both `ConscribeWarning` subclasses, so filter on `ConscribeWarning` to catch every reason. To make them errors, see [Turning conscribe warnings into errors](testing.md#turning-conscribe-warnings-into-errors).

### Markers in the artifacts

Degraded fields are recorded in the model's `__degraded_fields__` and in `LayerConfigResult.degraded_fields`. In the generated source, the header lists them and the field carries a `# degraded from:` comment. In the per-layer JSON Schema, a top-level `x-degraded-fields` entry lists them, and the property gets `x-degraded-from` plus a `[degraded]` note in its description. The header and the note use the same wording for every reason ("incompatible with config serialization"); `DegradedField.reason` and the warning say which reason applies.

```python
import json
import ssl
import warnings
from typing import Annotated, Protocol, runtime_checkable

from pydantic import Field

from conscribe import (
    ConscribeWarning,
    build_layer_config,
    create_registrar,
    generate_layer_config_source,
    generate_layer_json_schema,
)


@runtime_checkable
class Client(Protocol):
    def get(self, url: str) -> str: ...


Http = create_registrar("http", Client, discriminator_field="kind")


class Base(metaclass=Http.Meta):
    def get(self, url: str) -> str:
        return url


class Secure(Base):
    def __init__(
        self,
        *,
        context: Annotated[ssl.SSLContext, Field(description="TLS context")] = None,
        timeout: int = 30,
    ) -> None: ...


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConscribeWarning)
    result = build_layer_config(Http)
for warning in caught:
    if issubclass(warning.category, ConscribeWarning):
        print(warning.category.__name__, warning.message)

print(result.degraded_fields)
print(generate_layer_config_source(result))
schema = generate_layer_json_schema(result)
print(json.dumps(schema["x-degraded-fields"]))
print(json.dumps(schema["properties"]["context"], indent=2))
```

```
DegradationWarning __main__.Secure.context: pydantic cannot build a schema for 'ssl.SSLContext | None' (declared on __main__.Secure); the field's type degrades to Any (default, description and constraints are kept).
{'secure': [DegradedField(field_name='context', original_type_repr='ssl.SSLContext | None', source_class='__main__.Secure', reason='pydantic_incompatible')]}
# Generated by conscribe 2.0.0
# Auto-generated by conscribe. DO NOT EDIT.
# Layer: http
# Discriminator: kind
# Entries (1): secure
#
# WARNING: The following fields had types incompatible with config
# serialization and were degraded to Any:
#   secure: context (was: ssl.SSLContext | None)

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SecureHttpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: Any = Field(default=None, description='TLS context')  # degraded from: ssl.SSLContext | None
    timeout: int = 30
    kind: Literal['secure'] = 'secure'


# Rebuild models for deferred annotation resolution
SecureHttpConfig.model_rebuild()

{"secure": [{"field": "context", "original_type": "ssl.SSLContext | None"}]}
{
  "default": null,
  "description": "TLS context [degraded] Type was: ssl.SSLContext | None (not serializable in config)",
  "title": "Context",
  "x-degraded-from": "ssl.SSLContext | None"
}
```

The type is `ssl.SSLContext | None` because of the None rule: a `None` default makes the type admit `None` before pydantic sees it.

### `DegradedField`

A frozen dataclass, exported from `conscribe`:

| Field | Example |
|-------|---------|
| `field_name` | `"context"` |
| `original_type_repr` | `"ssl.SSLContext \| None"` |
| `source_class` | The declaring class: `"__main__.Secure"` |
| `reason` | `"pydantic_incompatible"` (the default), `"type_checking_only"` or `"function_locals"` |

`conscribe.config.degradation.DEGRADATION_REASONS` lists the three reasons.

### Type representation

`format_type_repr(tp)` (in `conscribe.config.degradation`) renders the original type for these records: unions as `X | Y`, generics as `list[X]`, `Annotated[X, ...]` as `X`, classes as `module.QualName` (builtins without a module), a `ForwardRef` as its text, `None` as `None`. For a name only imported under `TYPE_CHECKING`, the annotation is recorded as written (`Context | None`).
