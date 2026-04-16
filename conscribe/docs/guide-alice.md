# Guide: Framework Developer (Alice)

You're building a config-driven framework with pluggable layers. This guide walks through setting up conscribe from scratch.

## Step 1: Define a Layer

Each pluggable layer needs a registrar. A registrar binds together a Protocol, a registry, and a metaclass.

```python
# my_app/llm/_registrar.py
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class ChatModelProtocol(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

LLMRegistrar = create_registrar(
    "llm",                              # layer name
    ChatModelProtocol,                  # protocol for compliance checking
    discriminator_field="provider",     # field name in config union
    strip_prefixes=["Chat"],            # ChatOpenAI -> "open_ai"
)
```

Key parameters for `create_registrar()`:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `discriminator_field` | Config union discriminator name | `""` (required for config) |
| `strip_prefixes` | Prefixes to strip from class names for key inference | `None` |
| `strip_suffixes` | Suffixes to strip from class names for key inference | `None` |
| `key_transform` | Fully custom key inference function | `None` (uses CamelCase->snake_case) |
| `mro_scope` | Scope for MRO `**kwargs` chain resolution | `"local"` |
| `mro_depth` | Max MRO levels to traverse | `None` (unlimited) |
| `skip_pydantic_generic` | Filter Pydantic Generic intermediates | `True` |
| `skip_filter` | Custom class filter callable | `None` |

## Step 2: Create a Base Class

The base class uses the registrar's metaclass. Mark it `__abstract__ = True` so it's not registered itself.

```python
# my_app/llm/base.py
from my_app.llm._registrar import LLMRegistrar

class ChatBaseModel(metaclass=LLMRegistrar.Meta):
    __abstract__ = True   # not registered
```

## Step 3: Write Implementations

Each subclass is automatically registered. The registry key is inferred from the class name.

```python
# my_app/llm/providers/openai.py
class ChatOpenAI(ChatBaseModel):
    """OpenAI LLM provider.

    Args:
        model_id: Model identifier, e.g. gpt-4o
        temperature: Sampling temperature, 0-2
    """
    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature

    def chat(self, messages):
        ...
# Registered as "open_ai" (strip_prefixes=["Chat"] + CamelCase->snake_case)
```

## Step 4: Discover and Use

Import all modules to trigger metaclass registration, then query the registry:

```python
# my_app/main.py
from conscribe import discover
from my_app.llm._registrar import LLMRegistrar

discover("my_app.llm.providers")

llm_cls = LLMRegistrar.get("open_ai")   # -> ChatOpenAI
llm = llm_cls(model_id="gpt-4o")

print(LLMRegistrar.keys())  # ["open_ai", "anthropic", ...]
```

## Step 5: Generate Config Stubs

Build a discriminated union and generate a Python stub file:

```python
from conscribe import build_layer_config, generate_layer_config_source

result = build_layer_config(LLMRegistrar)
source = generate_layer_config_source(result)

with open("generated/llm_config.py", "w") as f:
    f.write(source)
```

The generated file contains per-key Pydantic models and a union type:

```python
# generated/llm_config.py (auto-generated, DO NOT EDIT)
class OpenAiLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["open_ai"] = "open_ai"
    model_id: str = Field(..., description="Model identifier, e.g. gpt-4o")
    temperature: float = Field(0.0, description="Sampling temperature, 0-2")

LLMConfig = Annotated[
    Union[OpenAiLLMConfig, AnthropicLLMConfig],
    Field(discriminator="provider"),
]
```

## Step 6: Runtime Validation

Validate user config at startup, before any business logic:

```python
from pydantic import TypeAdapter
import yaml

raw = yaml.safe_load(open("experiment.yaml"))
union_type = LLMRegistrar.config_union_type()
config = TypeAdapter(union_type).validate_python(raw["llm"])
# ValidationError at startup if config is invalid
```

## Integrating External Classes

### Path A: Already Inherits Base

If an external class already extends your base, its subclasses auto-register:

```python
from cool_agent import CoolAgent  # extends BaseAgent

class MyCool(CoolAgent):           # auto-registered as "my_cool"
    ...
```

### Path B: Bridge (recommended)

For classes that don't use your metaclass:

```python
from ext_framework import ExtAgent

ExtBridge = AgentRegistrar.bridge(ExtAgent)

class MyExtV1(ExtBridge):   # auto-registered as "my_ext_v1"
    ...
class MyExtV2(ExtBridge):   # auto-registered as "my_ext_v2"
    ...
```

`bridge()` automatically resolves metaclass conflicts using four strategies:
1. External is plain `type` -> use our Meta
2. Our Meta subclasses external's -> use ours
3. External is more specific -> use theirs
4. Incompatible -> dynamically create combined metaclass

### Path C: Decorator (one-off)

```python
@AgentRegistrar.register("custom_agent")
class CustomAgent:
    def step(self): ...
```

Use `propagate=True` to also auto-register future subclasses:

```python
@AgentRegistrar.register("base_ext", propagate=True)
class ExtAgent:
    def step(self): ...

class SubExt(ExtAgent):  # auto-registered as "sub_ext"
    def step(self): ...
```

## Advanced: Hierarchical Keys

For layers with natural hierarchies (e.g., model_type → provider), use `key_separator` and `discriminator_fields`:

```python
LLM = create_registrar(
    "llm", ChatModelProtocol,
    discriminator_fields=["model_type", "provider"],
    key_separator=".",
)

# Abstract parent defines level 0 params
class OpenAIBase(metaclass=LLM.Meta):
    __registry_key__ = "openai"
    __abstract__ = True
    def __init__(self, *, temperature: float = 0.7, max_tokens: int = 1000):
        ...

# Leaf children define level 1 params
class AzureOpenAI(OpenAIBase):
    __registry_key__ = "openai.azure"
    def __init__(self, *, deployment: str, api_version: str = "2024-02", **kwargs):
        super().__init__(**kwargs)

class OfficialOpenAI(OpenAIBase):
    __registry_key__ = "openai.official"
    def __init__(self, *, endpoint: str, **kwargs):
        super().__init__(**kwargs)
```

### Tree Queries

```python
LLM.children("openai")
# → {"openai.azure": AzureOpenAI, "openai.official": OfficialOpenAI}

LLM.tree()
# → {"openai": {"azure": AzureOpenAI, "official": OfficialOpenAI}}
```

### Generated Nested Config

Config stubs use compound discrimination with nested sub-models:

```yaml
# experiment.yaml (hybrid format)
llm:
  model_type: openai           # flat (level 0 discriminator)
  temperature: 0.7             # flat (level 0 param)
  max_tokens: 1000
  provider:                    # nested (level 1)
    name: azure
    deployment: my-deployment
    api_version: 2024-02
```

## Advanced: Composed Config (Multi-Layer Inline Wiring)

When your framework has multiple layers wired together, generate a single composed schema. Wired fields become inline config objects instead of key selectors:

```python
from conscribe import build_composed_config, generate_composed_json_schema
import json

result = build_composed_config(
    {"llm": LLMRegistrar, "agent": AgentRegistrar},
    inline_wiring=True,
)

schema = generate_composed_json_schema(result)
with open("generated/composed.schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

Users can then write YAML with full IDE autocompletion across layers:

```yaml
agent:
  - name: browser_use
    use_vision: true
    llm:                          # IDE autocompletes all LLM config fields
      provider: openai
      model_id: gpt-4o
      temperature: 0.7
```

Layers are topologically sorted by wiring dependencies (leaves first). Circular wiring raises `CircularWiringError`.

CLI equivalent:

```bash
conscribe generate-composed-config \
  --layers llm agent \
  --format json-schema \
  --output generated/composed.schema.json
```

## Advanced: Cross-Registry Diamond Inheritance

Register a class in multiple registries using the `|` operator:

```python
LLM = create_registrar("llm", LLMProtocol, discriminator_field="provider")
Agent = create_registrar("agent", AgentProtocol, discriminator_field="name")

CombinedMeta = LLM.Meta | Agent.Meta

class LLMAgentBase(metaclass=CombinedMeta):
    __abstract__ = True
    # Implement both protocols...

class MyLLMAgent(LLMAgentBase):
    ...
# MyLLMAgent is in BOTH LLM and Agent registries
```

### Selective Opt-Out

```python
class LLMOnly(LLMAgentBase):
    __skip_registries__ = ["agent"]
# Only in LLM registry
```

### Parent Registration Control

```python
class StrictBase(metaclass=LLM.Meta):
    __abstract__ = True
    __propagate_depth__ = 1  # only direct children register

    @staticmethod
    def __registration_filter__(child_cls):
        return "Test" not in child_cls.__name__  # block test classes
```

## Pydantic BaseModel + Generic[T]

When Pydantic specializes `BaseModel[T]`, it creates real class objects that would pollute the registry. Conscribe filters these automatically (`skip_pydantic_generic=True` by default).

```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")
class BaseEvent(BaseModel, Generic[T]):
    payload: T

EventBridge = EventRegistrar.bridge(BaseEvent)

class StringEvent(EventBridge):
    payload: str = "hello"
# BaseEvent[str] intermediates are filtered out automatically
```

Config extraction also works -- fields are read from `model_fields` instead of `__init__`.

## Auto-Freshness

When passing `stub_dir` to `discover()`, conscribe auto-detects registry changes:

```python
discover("my_app.agents", stub_dir=Path("generated"))
# 1. Imports all modules -> fills registries
# 2. Computes fingerprint (keys + signatures + docstrings)
# 3. Compares with cached fingerprint
# 4. Changed -> regenerates stubs
# 5. Same -> skips (zero overhead)
```

## CLI Usage

See [CLI Reference](cli.md) for full details.

```bash
# Generate stubs
conscribe generate-config \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers" \
  --output "generated/llm_config.py" \
  --json-schema "generated/llm_config.schema.json"

# Batch generate
conscribe generate-config --config conscribe.yaml

# Inspect registry
conscribe inspect \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers"
```

## Design Principles

- **Zero registration burden** -- Inherit a base class = registered
- **`__init__` is the single source of truth** -- No duplicate config definitions
- **Fail-fast** -- Duplicate keys raise immediately; invalid config rejects at startup
- **Domain-agnostic** -- Pure infrastructure, knows nothing about agents or LLMs
- **Stubs and runtime are separate** -- Stale stubs don't affect correctness
