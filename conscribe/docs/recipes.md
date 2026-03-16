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
