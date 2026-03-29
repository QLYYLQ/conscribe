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

**Framework developers** building config-driven systems with pluggable layers (agents, LLM providers, browser backends, etc.) who need registries, factories, protocol checks, and config schemas — without N layers × boilerplate.

**Framework users** who write YAML configs and want IDE autocomplete, parameter docs, and fail-fast validation at startup instead of crashing N steps in.

---

## Quick Start

### 1. Define a Layer

```python
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class ChatModelProtocol(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

LLMRegistrar = create_registrar(
    "llm",
    ChatModelProtocol,
    discriminator_field="provider",
    strip_prefixes=["Chat"],
)
```

### 2. Create a Base Class

```python
class ChatBaseModel(metaclass=LLMRegistrar.Meta):
    __abstract__ = True
```

### 3. Write Implementations (auto-registered)

```python
class ChatOpenAI(ChatBaseModel):
    """OpenAI LLM provider.

    Args:
        model_id: Model identifier, e.g. gpt-4o
        temperature: Sampling temperature, 0-2
    """
    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature

    def chat(self, messages): ...

# Registered as "open_ai". No decorator. No registry call.
```

### 4. Discover & Use

```python
from conscribe import discover

discover("my_app.llm.providers")

llm_cls = LLMRegistrar.get("open_ai")   # → ChatOpenAI
llm = llm_cls(model_id="gpt-4o")
print(LLMRegistrar.keys())              # ["open_ai", "anthropic", ...]
```

---

## Config Typing

Your `__init__` signature is the config schema. Conscribe extracts it, builds a Pydantic discriminated union, and generates stubs for IDE autocomplete:

```python
from conscribe import build_layer_config, generate_layer_config_source

result = build_layer_config(LLMRegistrar)
source = generate_layer_config_source(result)
```

See the [Config Typing Guide](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/guide-alice.md#step-5-generate-config-stubs) for full details.

### Config Tiers

| Tier | What You Write | What Users Get |
|------|---------------|---------------|
| **1** | Plain `__init__(self, *, x: int = 5)` | Names + types + defaults |
| **1.5** | + Google/NumPy docstring with `Args:` | + descriptions |
| **2** | + `Annotated[int, Field(ge=0)]` | + constraints |
| **3** | `__config_schema__ = MyModel` | Full Pydantic model |

### Nested Config (Hierarchical Keys)

For layers with natural hierarchies (e.g., model_type → provider):

```python
LLM = create_registrar(
    "llm", ChatModelProtocol,
    discriminator_fields=["model_type", "provider"],
    key_separator=".",
)
```

Produces hybrid YAML configs where level 0 is flat and level 1+ is nested:

```yaml
llm:
  model_type: openai        # flat (level 0)
  temperature: 0.7          # flat (level 0 param)
  provider:                 # nested (level 1)
    name: azure
    deployment: my-deploy
```

### Cross-Registry Wiring

Declare which values from other registries a class accepts. Conscribe constrains the config field to `Literal[...]`:

```python
# Given: LoopRegistrar with "react" and "codeact" registered

class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True
    __wiring__ = {"loop": "loop"}  # all keys from "loop" registry

class SWEAgent(BaseAgent):
    __wiring__ = {"loop": ("loop", ["react"])}  # narrows to subset

    def __init__(self, *, max_steps: int = 10): ...
```

Generated config:

```python
class SweAgentConfig(BaseModel):
    name: Literal["swe_agent"] = "swe_agent"
    max_steps: int = 10
    loop: Literal["react"] = ...  # wired from: loop
```

Three modes:
- `"loop": "agent_loop"` — auto-discover all keys from registry
- `"loop": ("agent_loop", ["react", "codeact"])` — explicit subset
- `"browser": ["chromium", "firefox"]` — literal list (no registry)

Inheritance: child `__wiring__` merges with parent (child keys override). Use `None` to exclude: `{"llm": None}`.

### Cross-Registry Diamond Inheritance

Register a class in multiple registries:

```python
CombinedMeta = LLM.Meta | Agent.Meta

class LLMAgent(metaclass=CombinedMeta):
    ...  # in both LLM and Agent registries
```

---

## API Reference

### Registration

| API | Purpose |
|-----|---------|
| `create_registrar(name, protocol, ...)` | Create a layer registrar (recommended entry point) |
| `Registrar.get(key)` | Look up a registered class |
| `Registrar.keys()` | List all registered keys |
| `Registrar.children(prefix)` | Query hierarchical key descendants |
| `Registrar.tree()` | Get nested dict of key hierarchy |
| `Registrar.bridge(external_cls)` | Create bridge for external class |
| `Registrar.register(key)` | Manual registration decorator |
| `discover(*package_paths)` | Import modules to trigger registration |

### Config Typing

| API | Purpose |
|-----|---------|
| `extract_config_schema(cls, mro_scope, mro_depth)` | Extract Pydantic model from `__init__` |
| `build_layer_config(registrar)` | Build discriminated union (flat or nested mode) |
| `generate_layer_config_source(result)` | Generate Python stub source code |
| `generate_layer_json_schema(result)` | Generate JSON Schema |
| `compute_registry_fingerprint(registrar)` | Compute registry fingerprint hash |
| `get_registry(name)` | Look up a registry by name (for wiring) |

---

## Design Principles

- **Zero registration burden** — Inherit a base class = registered
- **`__init__` is the single source of truth** — No duplicate config definitions
- **Fail-fast** — Duplicate keys raise immediately; invalid config rejects at startup
- **Domain-agnostic** — Pure infrastructure, knows nothing about agents or LLMs
- **Stubs and runtime are separate** — Stale stubs don't affect correctness

---

## Documentation

Full documentation is shipped inside the package (accessible at `site-packages/conscribe/`) and [browsable on GitHub](https://github.com/QLYYLQ/conscribe/tree/master/conscribe/docs):

| Document | Description |
|----------|-------------|
| [`llms.txt`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/llms.txt) | AI entry point — package summary and navigation |
| [`docs/overview.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/overview.md) | Core concepts and architecture |
| [`docs/guide-alice.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/guide-alice.md) | Tutorial: building a framework with conscribe |
| [`docs/guide-bob.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/guide-bob.md) | Tutorial: consuming a conscribe-based framework |
| [`docs/api-reference.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/api-reference.md) | Full API signatures and examples |
| [`docs/recipes.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/recipes.md) | Task-oriented "how do I X?" |
| [`docs/registration.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/registration.md) | Registration subsystem internals |
| [`docs/config-typing.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/config-typing.md) | Config typing pipeline internals |
| [`docs/mro-and-degradation.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/mro-and-degradation.md) | MRO chains and type degradation |
| [`docs/cli.md`](https://github.com/QLYYLQ/conscribe/blob/master/conscribe/docs/cli.md) | CLI reference |

---

## License

MIT
