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

**2. `__init__` signature is config schema.** Conscribe reflects your constructor parameters into Pydantic models, builds discriminated unions, and generates stub files for IDE autocomplete. Your `__init__` is the single source of truth.

## Two Subsystems

### Registration (`conscribe/registration/`)

Handles class discovery and storage. Components:

- **`create_registrar(name, protocol)`** -- creates a layer-specific registrar (the main entry point)
- **`LayerRegistry`** -- thread-safe key-to-class storage with Protocol compliance caching
- **`AutoRegistrar` metaclass** -- intercepts class creation to auto-register subclasses
- **`AutoRegistrarBase`** -- shared metaclass base with `|` operator for cross-registry diamond inheritance
- **Key transform** -- infers registry keys from class names (CamelCase -> snake_case)
- **Predicate filters** -- composable skip conditions (abstract, Pydantic generic, custom, propagation control)

Three registration paths:
- **Path A (inheritance):** `class Foo(Base)` -- auto-registered via metaclass
- **Path B (bridge):** `Registrar.bridge(ExternalClass)` -- one-time bridge for external classes
- **Path C (manual):** `@Registrar.register("key")` -- decorator with protocol check

Advanced features:
- **Hierarchical keys:** dotted keys like `"openai.azure"` with configurable separator, tree queries
- **Cross-registry diamond:** `metaclass=LLM.Meta | Agent.Meta` registers in both registries
- **Multi-key registration:** `__registry_key__ = ["alias_a", "alias_b"]`
- **Opt-out controls:** `__skip_registries__`, `__registration_filter__`, `__propagate__`, `__propagate_depth__`
- **Cross-registry wiring:** `__wiring__` declares field constraints referencing other registries, producing `Literal[...]` types in generated config stubs

### Config Typing (`conscribe/config/`)

Extracts config schemas and generates output. Pipeline: **extract -> build -> generate**.

- **Extract:** `extract_config_schema(cls)` reflects `__init__` into a Pydantic model
- **Build:** `build_layer_config(registrar)` creates discriminated unions with `Literal[key]` discriminators
  - **Flat mode** (single discriminator): `Annotated[Union[...], Field(discriminator=...)]`
  - **Nested mode** (compound discriminator): deeply nested sub-models with `Discriminator(callable)` + `Tag`
- **Generate:** `generate_layer_config_source(result)` outputs Python stubs; `generate_layer_json_schema(result)` outputs JSON Schema
- **Compose:** `build_composed_config(registrars, inline_wiring)` combines multiple layers into a single schema — wired fields become inline config objects with recursive IDE autocompletion
- **Fingerprint:** `compute_registry_fingerprint(registrar)` hashes registry state for staleness detection

## Config Tiers

| Tier | What You Write | What's Extracted |
|------|---------------|-----------------|
| **1** | Plain `__init__(self, *, x: int = 5)` | Field names + types + defaults |
| **1.5** | + Google/NumPy docstring with `Args:` | + parameter descriptions |
| **2** | + `Annotated[int, Field(ge=0)]` | + constraints, descriptions |
| **3** | `__config_schema__ = MyModel` | Full Pydantic model (validators, nesting) |

## Learning Path

| Goal | Read |
|------|------|
| Build a framework with conscribe | [Guide: Framework Developer](guide-alice.md) |
| Use a conscribe-based framework's config | [Guide: Framework User](guide-bob.md) |
| Understand registration internals | [Registration Deep-Dive](registration.md) |
| Understand config pipeline internals | [Config Typing Deep-Dive](config-typing.md) |
| Handle MRO chains and type degradation | [MRO and Degradation](mro-and-degradation.md) |
| Look up exact API signatures | [API Reference](api-reference.md) |
| Compose multiple layers into one schema | [Config Typing Deep-Dive — Composed Config](config-typing.md#composed-config-composedpy) |
| Answer "how do I X?" | [Recipes](recipes.md) |
| Use the CLI | [CLI Reference](cli.md) |

## Quick Example

```python
from typing import Protocol, runtime_checkable
from conscribe import create_registrar, discover, build_layer_config, generate_layer_config_source

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

- Python >= 3.9
- `pydantic >= 2.0, < 3.0` (core)
- `docstring-parser >= 0.15` (optional, for Tier 1.5 docstring extraction)
