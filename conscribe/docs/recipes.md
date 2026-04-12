# Recipes

Task-oriented answers to common questions. Each recipe is self-contained.

## How do I create a new layer (registry)?

```python
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class AgentProtocol(Protocol):
    def step(self) -> None: ...

AgentRegistrar = create_registrar(
    "agent",                          # layer name
    AgentProtocol,                    # protocol for compliance checking
    discriminator_field="agent_type", # config union discriminator
    strip_suffixes=["Agent"],         # MyCustomAgent -> "my_custom"
)
```

## How do I register a class automatically?

Create a base class with the registrar's metaclass, then subclass it:

```python
class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True  # base classes are not registered

class MyAgent(BaseAgent):
    def step(self): ...
# MyAgent is now registered as "my_agent"
```

## How do I register an external class?

Use `bridge()` for classes not inheriting your base:

```python
from external_lib import TheirAgent

TheirBridge = AgentRegistrar.bridge(TheirAgent)

class MyExtAgent(TheirBridge):
    def step(self): ...
# MyExtAgent is registered as "my_ext_agent"
```

For a one-off, use the decorator:

```python
@AgentRegistrar.register("special_agent")
class SpecialAgent:
    def step(self): ...
```

## How do I generate config stubs for IDE autocomplete?

```python
from conscribe import build_layer_config, generate_layer_config_source

result = build_layer_config(AgentRegistrar)
source = generate_layer_config_source(result)

with open("generated/agent_config.py", "w") as f:
    f.write(source)
```

Or via CLI:

```bash
conscribe generate-config \
  --registrar "my_app.agents._registrar:AgentRegistrar" \
  --discover "my_app.agents" \
  --output "generated/agent_config.py"
```

## How do I generate JSON Schema for YAML editors?

```python
from conscribe import build_layer_config, generate_layer_json_schema
import json

result = build_layer_config(AgentRegistrar)
schema = generate_layer_json_schema(result)

with open("generated/agent_config.schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

## How do I validate config at runtime?

```python
from pydantic import TypeAdapter
import yaml

union_type = AgentRegistrar.config_union_type()
raw = yaml.safe_load(open("config.yaml"))
config = TypeAdapter(union_type).validate_python(raw["agent"])
# ValidationError raised immediately if config is invalid
```

## How do I add parameter descriptions to generated stubs?

Use a Google/NumPy-style docstring (Tier 1.5, requires `pip install conscribe[docstring]`):

```python
class MyAgent(BaseAgent):
    """My custom agent.

    Args:
        max_steps: Maximum number of steps before stopping
        timeout: Timeout in seconds
    """
    def __init__(self, *, max_steps: int = 100, timeout: int = 300): ...
```

Or use `Annotated` with `Field` (Tier 2):

```python
from typing import Annotated
from pydantic import Field

class MyAgent(BaseAgent):
    def __init__(
        self, *,
        max_steps: Annotated[int, Field(100, gt=0, description="Max steps")] = 100,
    ): ...
```

## How do I use a full Pydantic model for complex config?

Use Tier 3's `__config_schema__` escape hatch:

```python
from pydantic import BaseModel, Field, model_validator

class MyAgentConfig(BaseModel):
    model_id: str
    temperature: float = Field(0.0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_combo(self): ...

class MyAgent(BaseAgent):
    __config_schema__ = MyAgentConfig
```

## How do I handle types Pydantic can't serialize?

Conscribe degrades incompatible types (e.g., `httpx.Auth`) to `Any` automatically. The generated stub marks degraded fields:

```python
auth: Any = None  # degraded from: httpx._types.AuthTypes | None
```

To include third-party parent parameters that may have incompatible types, set the MRO scope:

```python
AgentRegistrar = create_registrar(
    "agent", proto,
    mro_scope=["httpx"],  # traverse into httpx's __init__ params
)
```

## How do I control which parent classes are traversed for `**kwargs` chains?

Set `mro_scope` at the registrar level or per-class:

```python
# Registrar-level (applies to all registered classes)
AgentRegistrar = create_registrar("agent", proto, mro_scope="third_party")

# Per-class override
class MyAgent(BaseAgent):
    __config_mro_scope__ = ["httpx"]
    __config_mro_depth__ = 2

    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(**kwargs)
```

## How do I override the inferred registry key?

Set `__registry_key__` on the class:

```python
class ChatGPT4(ChatBase):
    __registry_key__ = "gpt4"
    def chat(self, messages): ...
# Registered as "gpt4" instead of "chat_gpt4"
```

## How do I register under multiple keys?

Set `__registry_key__` to a list:

```python
class UniversalProvider(Base):
    __registry_key__ = ["openai.universal", "anthropic.universal"]
    def __init__(self, endpoint: str): ...
# Registered under BOTH keys
# cls.__registry_key__ == "openai.universal"  (primary)
# cls.__registry_keys__ == ["openai.universal", "anthropic.universal"]
```

## How do I use hierarchical (dotted) keys?

Set `key_separator` on the registrar:

```python
LLM = create_registrar(
    "llm", proto,
    discriminator_field="provider",
    key_separator=".",
)

class OpenAIBase(metaclass=LLM.Meta):
    __registry_key__ = "openai"
    __abstract__ = True

class AzureOpenAI(OpenAIBase):
    __registry_key__ = "openai.azure"
    # Or omit __registry_key__ for auto-derivation: "openai.azure_open_ai"

# Query the hierarchy:
LLM.children("openai")  # {"openai.azure": AzureOpenAI, ...}
LLM.tree()               # {"openai": {"azure": AzureOpenAI, ...}}
```

## How do I register in multiple registries (diamond inheritance)?

Use the `|` operator on metaclasses:

```python
LLM = create_registrar("llm", LLMProtocol, discriminator_field="provider")
Agent = create_registrar("agent", AgentProtocol, discriminator_field="name")

CombinedMeta = LLM.Meta | Agent.Meta

class LLMAgentBase(metaclass=CombinedMeta):
    __abstract__ = True
    # Implement both protocols...

class MyLLMAgent(LLMAgentBase):
    ...
# Registered in BOTH llm and agent registries
```

To opt out of one registry, use `__skip_registries__`:

```python
class LLMOnly(LLMAgentBase):
    __skip_registries__ = ["agent"]
# Only registered in llm, not agent
```

## How do I control which subclasses get registered?

Use parent-level controls:

```python
class StrictBase(metaclass=R.Meta):
    __abstract__ = True

    # Block children whose name contains "Test"
    @staticmethod
    def __registration_filter__(child_cls):
        return "Test" not in child_cls.__name__

class ValidImpl(StrictBase): ...  # registered
class TestImpl(StrictBase): ...   # blocked

# Limit depth:
class ShallowBase(metaclass=R.Meta):
    __abstract__ = True
    __propagate_depth__ = 1  # only direct children register

# Stop propagation entirely:
class TerminalBase(metaclass=R.Meta):
    __abstract__ = True
    __propagate__ = False  # no children register
```

## How do I build nested config with compound discriminators?

Use `discriminator_fields` (list) instead of `discriminator_field` (string):

```python
LLM = create_registrar(
    "llm", LLMProtocol,
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
#   model_type: openai           # flat (level 0 discriminator)
#   temperature: 0.7             # flat (level 0 param)
#   provider:                    # nested (level 1)
#     name: azure
#     deployment: my-deploy

# Validation:
from pydantic import TypeAdapter
result = LLM.build_config()
adapter = TypeAdapter(result.union_type)
config = adapter.validate_python({
    "model_type": "openai",
    "temperature": 0.9,
    "provider": {"name": "azure", "deployment": "my-deploy"},
})
```

## How do I constrain a config field to values from another registry?

Use `__wiring__` to declare cross-registry constraints:

```python
# Setup: create a loop registry
LoopRegistrar = create_registrar("loop", LoopProtocol, discriminator_field="name")

class BaseLoop(metaclass=LoopRegistrar.Meta):
    __abstract__ = True

class ReactLoop(BaseLoop): ...
class CodeactLoop(BaseLoop): ...

# Agent registry with wiring
AgentRegistrar = create_registrar("agent", AgentProtocol, discriminator_field="name")

class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True
    __wiring__ = {"loop": "loop"}  # auto-discover: all keys from "loop" registry

class SWEAgent(BaseAgent):
    __wiring__ = {"loop": ("loop", ["react_loop"])}  # narrow to subset
    def __init__(self, *, max_steps: int = 10): ...

# Generated config constrains loop to Literal["react_loop"]
result = build_layer_config(AgentRegistrar)
source = generate_layer_config_source(result)
```

Four modes:
- `"field": "registry_name"` — all keys (auto-discovery)
- `"field": ("registry_name", ["key1", "key2"])` — explicit subset
- `"field": ("registry_name", ["required"], ["optional"])` — required + optional keys
- `"field": ["val1", "val2"]` — literal list (no registry)

The 3-element tuple distinguishes required and optional keys — both appear in `Literal[...]`, but the distinction is available as metadata for downstream negotiation logic.

Use `None` to exclude an inherited wiring key: `__wiring__ = {"llm": None}`.

## How do I get IDE autocomplete for wired attributes injected at runtime?

If a class declares `__wiring__` for fields not in `__init__`, those attributes are invisible to the IDE. Use `generate-stubs` to create `.pyi` files:

```bash
conscribe generate-stubs \
  --registrar "my_app.agents._registrar:AgentRegistrar" \
  --discover "my_app.agents"
```

This writes `.pyi` alongside each source `.py`. The wired attribute type is **narrowed** automatically:

```python
# If __wiring__ = {"env": "environment"} (all keys) → type is EnvironmentProtocol
# If __wiring__ = {"env": ("environment", ["terminal.bash", "terminal.zsh"])} → type is Terminal
# If __wiring__ = {"env": ("environment", ["terminal.bash"])} → type is BashTerminal
```

Or programmatically:

```python
from conscribe.stubs import write_layer_stubs

written = write_layer_stubs(AgentRegistrar, output_dir="generated/stubs")
```

See [CLI Reference — generate-stubs](cli.md#generate-stubs) for full options.

## How do I skip a class from registration?

Mark it abstract:

```python
class IntermediateBase(ChatBase):
    __abstract__ = True  # not registered
```

Or use a custom skip filter:

```python
AgentRegistrar = create_registrar(
    "agent", proto,
    skip_filter=lambda cls: cls.__name__.startswith("Internal"),
)
```

## How do I use conscribe with Pydantic BaseModel + Generic[T]?

It works automatically. `skip_pydantic_generic=True` (default) filters Generic intermediates:

```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class BaseEvent(BaseModel, Generic[T]):
    payload: T

EventBridge = EventRegistrar.bridge(BaseEvent)

class StringEvent(EventBridge):
    payload: str = "hello"
# BaseEvent[str] intermediates are filtered out
```

## How do I trigger module imports for registration?

Use `discover()` to recursively import all modules under a package:

```python
from conscribe import discover

discover("my_app.agents", "my_app.llm_providers")
# All classes in those packages are now registered
```

## How do I auto-regenerate stubs when the registry changes?

Pass `stub_dir` to `discover()`:

```python
from pathlib import Path

discover(
    "my_app.agents",
    stub_dir=Path("generated"),
)
# Fingerprints are compared; stubs regenerate only if stale
```

## How do I list all registered implementations?

```python
print(AgentRegistrar.keys())          # ["my_agent", "ext_agent", ...]
print(AgentRegistrar.get_all())       # {"my_agent": MyAgent, ...}
```

## How do I inspect a registry from the CLI?

```bash
conscribe inspect \
  --registrar "my_app.agents._registrar:AgentRegistrar" \
  --discover "my_app.agents"
```
