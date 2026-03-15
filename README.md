# Conscribe

**Inheritance is registration. `__init__` signature is config schema.**

Conscribe is a Python library that provides **automatic class registration** and **config typing stub generation** for layered architectures. It eliminates two categories of boilerplate:

1. **Manual registration** — Write a class, inherit a base → it's registered. No `registry["foo"] = FooClass`.
2. **Config guesswork** — Your `__init__` parameters become the config schema. IDE autocomplete and fail-fast validation come for free.

```
pip install conscribe
```

Requires Python >= 3.9. Built on Pydantic v2.

---

## Who Is This For?

### Framework Developer (Alice)

You're building a config-driven framework with pluggable layers (agents, LLM providers, browser backends, evaluators, etc.). Each layer has N implementations, and you need:

- A registry to map `"openai"` → `ChatOpenAI`
- A factory to instantiate the right class from config
- Interface checking to ensure implementations satisfy a protocol
- IDE-friendly config types so your users don't fly blind

Without Conscribe, that's N layers × (registry + factory + protocol check + config schema) = a lot of repetitive code.

### Framework User (Bob)

You use Alice's framework. You write YAML configs to run experiments. Your pain:

```
1. Write config.yaml (blind — no autocomplete, no docs)
2. Launch the program (wait minutes for model/browser/env setup)
3. Run N steps
4. Framework instantiates a module, passes config to __init__
5. Typo in field name → crash
6. Fix config, go back to step 1
```

**Conscribe eliminates the wait.** Bob gets IDE autocomplete while writing config, and the program validates all config at startup — before any business logic runs.

---

## Quick Start

### 1. Define a Layer (one-time setup)

```python
# my_app/llm/_registrar.py
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class ChatModelProtocol(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

LLMRegistrar = create_registrar(
    "llm",
    ChatModelProtocol,
    discriminator_field="provider",   # config discriminator
    strip_prefixes=["Chat"],          # ChatOpenAI → "open_ai"
)
```

### 2. Create a Base Class

```python
# my_app/llm/base.py
from my_app.llm._registrar import LLMRegistrar

class ChatBaseModel(metaclass=LLMRegistrar.Meta):
    __abstract__ = True   # base classes are not registered
```

### 3. Write Implementations (auto-registered)

```python
# my_app/llm/providers/openai.py
from typing import Annotated
from pydantic import Field

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

# That's it. ChatOpenAI is now registered as "open_ai".
# No decorator. No registry call. Just inheritance.
```

### 4. Discover & Use

```python
# my_app/main.py
from conscribe import discover
from my_app.llm._registrar import LLMRegistrar

# Import all modules → trigger metaclass registration
discover("my_app.llm.providers")

# Query the registry
llm_cls = LLMRegistrar.get("open_ai")   # → ChatOpenAI
llm = llm_cls(model_id="gpt-4o")

# List all registered implementations
print(LLMRegistrar.keys())  # ["open_ai", "anthropic", ...]
```

---

## Config Typing: From `__init__` to IDE Autocomplete

The killer feature: **your `__init__` signature is the config schema.** Conscribe extracts it, builds a Pydantic discriminated union, and generates stub files for IDE autocomplete.

### Generate Config Stubs

```python
from conscribe import build_layer_config, generate_layer_config_source

result = build_layer_config(LLMRegistrar)
source = generate_layer_config_source(result)

with open("generated/llm_config.py", "w") as f:
    f.write(source)
```

Or use the CLI:

```bash
conscribe generate-config \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers" \
  --output "generated/llm_config.py" \
  --json-schema "generated/llm_config.schema.json"
```

### What Gets Generated

```python
# generated/llm_config.py (auto-generated, DO NOT EDIT)

class OpenAiLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["open_ai"] = "open_ai"
    model_id: str = Field(..., description="Model identifier, e.g. gpt-4o")
    temperature: float = Field(0.0, description="Sampling temperature, 0-2")

class AnthropicLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic"] = "anthropic"
    model_id: str
    max_tokens: int = 4096

LlmConfig = Annotated[
    Union[OpenAiLLMConfig, AnthropicLLMConfig],
    Field(discriminator="provider"),
]
```

Now Bob writes config with full IDE support:

```yaml
# experiment.yaml
llm:
  provider: openai       # ← autocomplete: openai | anthropic | ...
  model_id: gpt-4o       # ← autocomplete: str, "Model identifier"
  temperature: 0.5       # ← autocomplete: float, default 0.0
  typo_field: 123        # ← RED LINE: unknown field (extra="forbid")
```

### Runtime Validation (fail-fast)

```python
from conscribe import build_layer_config
import yaml

raw = yaml.safe_load(open("experiment.yaml"))
union_type = LLMRegistrar.config_union_type()

# Pydantic validates immediately — before any business logic
config = TypeAdapter(union_type).validate_python(raw["llm"])
```

If `typo_field` is present → `ValidationError` at startup. Not after 10 minutes of model loading.

---

## Three Config Declaration Tiers

Conscribe extracts config schema from your `__init__` with zero to minimal extra code:

| Tier | What You Write | What Bob Gets |
|------|---------------|---------------|
| **Tier 1** | Plain `__init__(self, *, x: int = 5)` | Field names + types + defaults |
| **Tier 1.5** | + Google/NumPy docstring with `Args:` | + parameter descriptions |
| **Tier 2** | + `Annotated[int, Field(ge=0, le=10)]` | + constraints, descriptions |
| **Tier 3** | `__config_schema__ = MyConfigModel` | Full Pydantic model (validators, nesting) |

**Tier 1** — zero extra code (default):

```python
class MyAgent(BaseAgent):
    def __init__(self, *, max_steps: int = 100, timeout: int = 300):
        ...
```

**Tier 1.5** — add docstring (recommended minimum):

```python
class MyAgent(BaseAgent):
    """My custom agent.

    Args:
        max_steps: Maximum number of steps before stopping
        timeout: Timeout in seconds
    """
    def __init__(self, *, max_steps: int = 100, timeout: int = 300):
        ...
```

**Tier 2** — add Annotated metadata (recommended for production):

```python
class MyAgent(BaseAgent):
    def __init__(
        self, *,
        max_steps: Annotated[int, Field(100, gt=0, description="Max steps")] = 100,
        timeout: Annotated[int, Field(300, gt=0)] = 300,
    ):
        ...
```

**Tier 3** — escape hatch for complex scenarios:

```python
class OpenAIConfig(BaseModel):
    model_id: str
    temperature: float = Field(0.0, ge=0, le=2)

    @model_validator(mode="after")
    def check_constraints(self): ...

class ChatOpenAI(ChatBaseModel):
    __config_schema__ = OpenAIConfig   # full control
```

---

## Integrating External Classes

Not every class inherits your framework's base class. Conscribe provides three paths:

### Path A: Already Inherits Base (zero effort)

```python
from cool_agent import CoolAgent  # CoolAgent extends BaseAgent

class MyCool(CoolAgent):           # auto-registered as "my_cool"
    ...
```

### Path B: Bridge (recommended for external classes)

```python
from ext_framework import ExtAgent
from my_app.agents import AgentRegistrar

# One-time bridge
ExtBridge = AgentRegistrar.bridge(ExtAgent)

# Now "inheritance is registration" works
class MyExtV1(ExtBridge):   # → auto-registered as "my_ext_v1"
    ...
class MyExtV2(ExtBridge):   # → auto-registered as "my_ext_v2"
    ...
```

### Path C: Manual Register (one-off)

```python
@AgentRegistrar.register("custom_agent")
class CustomAgent:
    def step(self): ...
```

---

## MRO-Aware Parameter Collection

When a subclass doesn't define its own `__init__`, Conscribe walks the MRO chain to find the actual definer and extracts its parameters. Inherited config is never lost:

```python
class AgentBase(metaclass=AgentRegistrar.Meta):
    __abstract__ = True

    def __init__(self, *, max_steps: int = 100, timeout: int = 300):
        ...

class SubAgent(AgentBase):
    """Inherits all of AgentBase's parameters, no __init__ override needed."""
    pass
```

Conscribe extracts `max_steps` and `timeout` from `AgentBase.__init__` for `SubAgent`'s config schema:

```python
# Generated config for SubAgent
class SubAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["sub_agent"] = "sub_agent"
    max_steps: int = 100    # ← from AgentBase.__init__
    timeout: int = 300      # ← from AgentBase.__init__
```

### `**kwargs` Chain Resolution

When a child's `__init__` accepts `**kwargs` and passes them to `super().__init__(**kwargs)`, Conscribe automatically walks the MRO upward to collect parent parameters:

```python
class Parent:
    def __init__(self, x: int, y: str = "hello"):
        ...

class Child(Parent):
    def __init__(self, z: float, **kwargs):
        super().__init__(**kwargs)
```

Without this feature, `Child`'s config would only contain `z`. With MRO collection, it contains `z`, `x`, and `y` — the complete picture:

```python
schema = extract_config_schema(Child, mro_scope="all")
print(schema.model_fields)  # z, x, y
print(schema.model_config)  # extra="forbid" (fully resolved chain)
```

The chain walks upward until:
- A parent has no `**kwargs` → **natural termination**, all params are known → `extra="forbid"`
- A parent is outside the configured scope → **truncated** → `extra="allow"`
- The depth limit is reached → **truncated** → `extra="allow"`

#### Scope Control

The `mro_scope` parameter controls which classes in the MRO are included:

| Scope | Includes | Use Case |
|-------|----------|----------|
| `"local"` (default) | Only your project code | Safe default — ignores third-party internals |
| `"third_party"` | + site-packages | Include params from libraries you depend on |
| `"all"` | Everything except `object` | Full traversal for maximum schema completeness |
| `["httpx", "pydantic"]` | Your code + only the listed packages | Precise control — traverse into specific libraries only |

The `list[str]` form is the most practical for real-world use: you know which third-party base classes your implementations extend, so you list exactly those packages. Local classes always pass the filter; stdlib classes are always excluded.

```python
# Only traverse into httpx and my_sdk, ignore all other third-party
LLMRegistrar = create_registrar(
    "llm", proto,
    mro_scope=["httpx", "my_sdk"],
)

# Per-class override with package list
class MyClient(HttpxBridge):
    __config_mro_scope__ = ["httpx"]

    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(**kwargs)
```

#### Configuration Levels

MRO scope and depth can be set at three levels (highest priority wins):

```python
# 1. Function-level default
extract_config_schema(cls, mro_scope="local", mro_depth=None)

# 2. Registrar-level
create_registrar("agent", proto, mro_scope=["httpx"], mro_depth=2)

# 3. Per-class override (highest priority)
class MyAgent(BaseAgent):
    __config_mro_scope__ = ["httpx", "my_sdk"]
    __config_mro_depth__ = 1

    def __init__(self, z: float, **kwargs):
        super().__init__(**kwargs)
```

---

## Type Degradation for Third-Party Types

When MRO traversal reaches third-party classes, some `__init__` parameter types may be incompatible with Pydantic (e.g., `httpx.Auth`, `ssl.SSLContext`). Instead of failing, Conscribe **degrades** these types to `Any` while preserving field names and defaults, and adds clear warnings:

```python
class MyHttpClient(HttpxBridge):
    def __init__(self, *, api_key: str, auth: httpx.Auth = None, **kwargs):
        super().__init__(**kwargs)
```

The generated stub clearly marks what was degraded:

```python
# Auto-generated by conscribe. DO NOT EDIT.
# Layer: transport
#
# WARNING: The following fields had types incompatible with config
# serialization and were degraded to Any:
#   httpx: auth (was: httpx._types.AuthTypes | None)

class HttpxTransportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal['httpx'] = 'httpx'
    api_key: str = ...
    auth: Any = None  # degraded from: httpx._types.AuthTypes | None
```

JSON Schema output also includes machine-readable degradation info:

```json
{
  "x-degraded-fields": {
    "httpx": [{"field": "auth", "original_type": "httpx._types.AuthTypes | None"}]
  }
}
```

**Zero overhead on the happy path** — degradation logic only triggers when `create_model()` actually fails. Local classes with standard types never pay this cost.

---

## Open vs Closed Config Schema

Conscribe uses `**kwargs` presence and MRO resolution to decide config strictness:

```python
# No **kwargs → extra="forbid" → strict validation
class StrictProvider(Base):
    def __init__(self, *, model_id: str, temperature: float = 0.0):
        ...
# → Unknown fields are rejected

# Has **kwargs, fully resolved MRO chain → extra="forbid"
class FullProvider(Base):
    def __init__(self, *, model_id: str, **kwargs):
        super().__init__(**kwargs)
# → Parent params are collected, all params known → strict

# Has **kwargs, MRO chain truncated (scope/depth) → extra="allow"
class PartialProvider(ThirdPartyBase):
    def __init__(self, *, model_id: str, **kwargs):
        super().__init__(**kwargs)
# → Some params may be unknown → lenient
```

---

## Auto-Freshness

When your registry changes (new implementations, modified `__init__` signatures), stubs can auto-update:

```python
discover("my_app.agents", "my_app.llm.providers")
# 1. Imports all modules → fills registry
# 2. Computes registry fingerprint (keys + signatures + docstrings)
# 3. Compares with cached fingerprint
# 4. Changed? → Regenerates stubs automatically
# 5. Same? → Skips (zero overhead)
```

Bob adds a new agent → runs the program → stubs update → next IDE session has autocomplete.

---

## CLI

```bash
# Generate stubs for one layer
conscribe generate-config \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers" \
  --output "generated/llm_config.py"

# Batch generate from config file
conscribe generate-config --config conscribe.yaml

# Force regenerate all stubs (ignore fingerprint cache)
conscribe update-stubs --config conscribe.yaml

# Inspect registry contents
conscribe inspect \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers"
```

### Batch Config File

```yaml
# conscribe.yaml
discover:
  - my_app.agents
  - my_app.llm.providers
  - my_app.evaluators

output_dir: generated

layers:
  - registrar: my_app.agents._registrar:AgentRegistrar
    output: generated/agent_config.py
    json_schema: generated/agent_config.schema.json

  - registrar: my_app.llm._registrar:LLMRegistrar
    output: generated/llm_config.py
    json_schema: generated/llm_config.schema.json
```

---

## API Reference

### Registration

| API | Purpose |
|-----|---------|
| `create_registrar(name, protocol, ..., mro_scope, mro_depth)` | Create a layer registrar (recommended entry point) |
| `Registrar.get(key)` | Look up a registered class |
| `Registrar.keys()` | List all registered keys |
| `Registrar.bridge(external_cls)` | Create bridge for external class |
| `Registrar.register(key)` | Manual registration decorator |
| `discover(*package_paths)` | Import modules to trigger registration |

### Config Typing

| API | Purpose |
|-----|---------|
| `extract_config_schema(cls, mro_scope, mro_depth)` | Extract Pydantic model from `__init__` (with MRO `**kwargs` resolution) |
| `build_layer_config(registrar)` | Build discriminated union for a layer |
| `generate_layer_config_source(result)` | Generate Python stub source code |
| `generate_layer_json_schema(result)` | Generate JSON Schema for YAML editors |
| `compute_registry_fingerprint(registrar)` | Compute registry fingerprint hash |
| `DegradedField` | Dataclass recording a field degraded to `Any` |
| `MROScope` | Type alias: `Literal["local", "third_party", "all"] \| list[str]` |

---

## Design Principles

- **Zero registration burden** — Inherit a base class = registered. No decorators, no manual calls.
- **`__init__` is the single source of truth** — Config schema is extracted from constructor signatures. No duplicate definitions.
- **Fail-fast** — Duplicate keys raise immediately. Invalid config rejects at startup, not after N steps.
- **Domain-agnostic** — The library knows nothing about agents, LLMs, or benchmarks. Pure infrastructure.
- **Stubs and runtime validation are separate** — Stubs serve IDE autocomplete. Runtime validation builds from the live registry. Even stale stubs don't affect correctness.

---

## License

MIT
