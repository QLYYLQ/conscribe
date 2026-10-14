# Conscribe Overview

Conscribe is a Python library for **automatic class registration** and **config typing stub generation** in layered architectures. It targets framework developers building config-driven systems with pluggable layers (agents, LLM providers, browser backends, evaluators, etc.).

## The Problem

In a pluggable architecture with N layers and M implementations per layer, you typically maintain:

- A registry mapping string keys to classes (`"openai"` -> `ChatOpenAI`)
- A factory to instantiate from config
- Protocol/interface compliance checks
- Config schemas for IDE autocomplete and validation

That's N x (registry + factory + protocol + schema) = a lot of boilerplate that stays in sync manually.

## Two Core Ideas

**1. Inheritance is registration.** When you write `class ChatOpenAI(ChatBaseModel)`, the class is automatically registered in the LLM registry. No decorators. No `registry["openai"] = ChatOpenAI`. Just Python inheritance.

**2. `__init__` signature is config schema.** Conscribe reflects your constructor parameters into Pydantic models, builds discriminated unions, and generates source files and JSON Schemas for IDE autocomplete. Your `__init__` is the primary source of truth. `__wiring__` adds the fields that reference other layers on top of it, so a class with no constructor parameters of its own still gets its wired fields.

Both ideas depend on conscribe reading your classes exactly, so it refuses what it cannot honour: a declaration that is wrong or contradicts itself raises at class creation or at build time, with the class, the field and the fix in the message. When conscribe cannot read something at all (a C-extension signature, a name imported only for type checkers), it continues with less information and emits a warning that points at your class.

## Two Subsystems

### Registration (`conscribe/registration/`)

Handles class discovery and storage. Components:

- **`create_registrar(name, protocol)`** -- creates a layer-specific registrar (the main entry point)
- **`LayerRegistry`** -- thread-safe key-to-class storage with Protocol compliance caching
- **`AutoRegistrar` metaclass** -- intercepts class creation to auto-register subclasses
- **`AutoRegistrarBase`** -- shared metaclass base with `|` operator for cross-registry diamond inheritance
- **Key transform** -- infers registry keys from class names (CamelCase -> snake_case)
- **Predicate filters** -- composable skip conditions (abstract, Pydantic generic, custom, propagation control)
- **`discover()`** -- imports every module of a package so that its classes register; strict by default
- **`conscribe.testing.isolated_registries()`** -- scopes registrations to a block, for tests

Three registration paths:
- **Path A (inheritance):** `class Foo(Base)` -- auto-registered via metaclass
- **Path B (bridge):** `Registrar.bridge(ExternalClass)` -- one-time bridge for external classes
- **Path C (manual):** `@Registrar.register("key")` -- decorator with protocol check

Every path validates the key (a non-empty string) and the grammar of the class's `__wiring__` before the class touches a registry, and registration is atomic: a class statement that fails leaves nothing registered.

Advanced features:
- **Hierarchical keys:** dotted keys like `"openai.azure"` with configurable separator, tree queries
- **Cross-registry diamond:** `metaclass=LLM.Meta | Agent.Meta` registers in both registries
- **Multi-key registration:** `__registry_key__ = ["alias_a", "alias_b"]`
- **Opt-out controls:** `__skip_registries__`, `__registration_filter__`, `__propagate__`, `__propagate_depth__`
- **Cross-registry wiring:** `__wiring__` declares which keys of another registry a field accepts, producing `Literal[...]` types in generated config. A wired field is called a *receptor*. Its *runtime slot*, when present, is a constructor parameter or non-`ClassVar` class annotation. The annotation decides whether the config holds one key or a list of keys; an entry without a slot becomes an optional field. An entry may carry a default key or be declared `declare_only` (a constraint the framework reads, with nothing injected).
- **Injected attributes:** a wired name the framework sets after construction (no constructor parameter, no class annotation, not `declare_only`) gets a `WiredField` placeholder that raises a clear `AttributeError` until the framework injects the value

### Config Typing (`conscribe/config/`)

Extracts config schemas and generates output. Pipeline: **extract -> build -> generate**.

- **Extract:** `extract_config_schema(cls)` reflects `__init__` into a Pydantic model, carrying supported `Field(...)` metadata, default factories and docstring descriptions. On constructor parameters, `exclude=` is rejected and `deprecated=` becomes a JSON Schema marker.
- **Build:** `build_layer_config(registrar)` first checks the wiring of every registered class (all problems in one `WiringResolutionError`), then creates discriminated unions with `Literal[key]` discriminators
  - **Flat mode** (single discriminator): `Annotated[Union[...], Field(discriminator=...)]`
  - **Nested mode** (compound discriminator): deeply nested sub-models with `Discriminator(callable)` + `Tag`
- **Generate:** `generate_layer_config_source(result)` writes the model declarations as Python source and warns about any validator it cannot reproduce; `generate_layer_json_schema(result)` describes the same fields, defaults and constraints. Python validators cannot be expressed in JSON Schema.
- **Compose:** `build_composed_config(registrars)` combines multiple layers into a single schema. Wired fields become inline config objects with recursive IDE autocompletion; the top level refuses unknown sections; inline-only layers have no top-level section at all.
- **Stubs:** `conscribe.stubs.write_layer_stubs(registrar)` writes `.pyi` files that declare the attributes a framework injects, so type checkers see them
- **Fingerprint:** `compute_registry_fingerprint(registrar)` hashes the layer's config source, JSON Schema, version and settings; it does not cover `.pyi` stubs or composition options.

## Introspection (`conscribe.introspect`)

The rules conscribe uses to read a class are public: `init_param_names(cls)` (names the constructor accepts, following `**kwargs` up the MRO), `receptor_slots(cls)` (every place that can receive a value, and its shape) and `wiring_slots(cls)` (each wiring entry with its slot). Frameworks use them to route config values to constructor arguments or attributes.

## Config Tiers

| Tier | What You Write | What's Extracted |
|------|---------------|-----------------|
| **1** | Plain `__init__(self, *, x: int = 5)` | Field names + types + defaults |
| **1.5** | + Google/NumPy docstring with `Args:` | + parameter descriptions |
| **2** | + `Annotated[int, Field(ge=0)]` | + constraints, descriptions, aliases, titles, examples |
| **3** | `__config_schema__ = MyModel` | Your model's fields, validators and `model_config`; an unset `extra` policy defaults to `"forbid"` |

## Learning Path

| Goal | Read |
|------|------|
| Build a framework with conscribe | [Guide: Framework Developer](guide-alice.md) |
| Use a conscribe-based framework's config | [Guide: Framework User](guide-bob.md) |
| Upgrade from conscribe 1.x | [Migrating to 2.0](migrating-to-2.0.md) |
| Isolate registrations in tests | [Testing](testing.md) |
| Understand registration internals | [Registration Deep-Dive](registration.md) |
| Understand config pipeline internals | [Config Typing Deep-Dive](config-typing.md) |
| Handle MRO chains and type degradation | [MRO and Degradation](mro-and-degradation.md) |
| Look up exact API signatures | [API Reference](api-reference.md) |
| Compose multiple layers into one schema | [Config Typing Deep-Dive](config-typing.md) |
| Answer "how do I X?" | [Recipes](recipes.md) |
| Use the CLI | [CLI Reference](cli.md) |

## Quick Example

```python
from typing import Protocol, runtime_checkable
from conscribe import create_registrar, build_layer_config, generate_layer_config_source

# 1. Define a layer
@runtime_checkable
class ChatProto(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

LLMRegistrar = create_registrar("llm", ChatProto, discriminator_field="provider")

# 2. Create a base class
class ChatBase(metaclass=LLMRegistrar.Meta):
    __abstract__ = True

# 3. Implement (auto-registered as "chat_open_ai")
class ChatOpenAI(ChatBase):
    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature
    def chat(self, messages): ...

# 4. Use
cls = LLMRegistrar.get("chat_open_ai")  # -> ChatOpenAI
result = build_layer_config(LLMRegistrar)
source = generate_layer_config_source(result)
```

The generated source defines one model per key:

```python
class ChatOpenAiLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    temperature: float = 0.0
    provider: Literal['chat_open_ai'] = 'chat_open_ai'
```

### Nested Config Example (Hierarchical Keys)

```python
LLM = create_registrar(
    "llm", ChatProto,
    discriminator_fields=["model_type", "provider"],
    key_separator=".",
)

class OpenAIBase(metaclass=LLM.Meta):
    __registry_key__ = "openai"
    __abstract__ = True
    def __init__(self, temperature: float = 0.7): ...

class AzureOpenAI(OpenAIBase):
    __registry_key__ = "openai.azure"
    def __init__(self, deployment: str, **kwargs): ...

# YAML config (hybrid format):
# llm:
#   model_type: openai           # flat (level 0)
#   temperature: 0.7             # flat (level 0 param)
#   provider:                    # nested (level 1)
#     name: azure
#     deployment: my-deployment
```

## Dependencies

- Python >= 3.10
- `pydantic >= 2.7, < 3.0` (core)
- `docstring-parser >= 0.15` (optional, for Tier 1.5 docstring extraction: `pip install conscribe[docstring]`)
