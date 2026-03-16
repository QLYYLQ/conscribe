# Guide: Framework User (Bob)

You use a framework built with conscribe. You write YAML/JSON configs to run experiments. This guide covers everything you need to know as a config consumer.

## The Pain Without Conscribe

```
1. Write config.yaml (blind -- no autocomplete, no docs)
2. Launch program (wait minutes for model/browser/env setup)
3. Run N steps
4. Framework instantiates a module, passes config to __init__
5. Typo in field name -> crash
6. Fix config, go to step 1
```

Conscribe eliminates the wait: you get IDE autocomplete while writing config, and the program validates all config at startup.

## Using Generated Config Types

Your framework should provide generated stub files. Import them for IDE support:

```yaml
# experiment.yaml
llm:
  provider: open_ai        # <- autocomplete shows: open_ai | anthropic | ...
  model_id: gpt-4o         # <- autocomplete shows: str, "Model identifier"
  temperature: 0.5         # <- autocomplete shows: float, default 0.0
  typo_field: 123          # <- RED LINE: unknown field (extra="forbid")
```

The generated Python stubs look like:

```python
class OpenAiLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["open_ai"] = "open_ai"
    model_id: str = Field(..., description="Model identifier, e.g. gpt-4o")
    temperature: float = Field(0.0, description="Sampling temperature, 0-2")
```

## Config Declaration Tiers

Framework developers choose how much config metadata to expose. As a user, you benefit from richer tiers:

### Tier 1: Plain `__init__`

You get field names, types, and defaults:

```python
class MyAgent(BaseAgent):
    def __init__(self, *, max_steps: int = 100, timeout: int = 300):
        ...
```

Generated config: `max_steps: int = 100`, `timeout: int = 300`.

### Tier 1.5: Docstring Descriptions

You additionally get human-readable descriptions:

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

Generated config: `max_steps: int = Field(100, description="Maximum number of steps before stopping")`.

### Tier 2: Annotated Metadata

You additionally get constraints (validation rules):

```python
class MyAgent(BaseAgent):
    def __init__(
        self, *,
        max_steps: Annotated[int, Field(100, gt=0, description="Max steps")] = 100,
        timeout: Annotated[int, Field(300, gt=0)] = 300,
    ):
        ...
```

Generated config enforces `gt=0` -- negative values are rejected at validation time.

### Tier 3: Explicit Pydantic Model

Full Pydantic power, including cross-field validators:

```python
class OpenAIConfig(BaseModel):
    model_id: str
    temperature: float = Field(0.0, ge=0, le=2)

    @model_validator(mode="after")
    def check_constraints(self): ...

class ChatOpenAI(ChatBaseModel):
    __config_schema__ = OpenAIConfig
```

## MRO-Aware Parameter Collection

When a subclass uses `**kwargs` and forwards to `super().__init__()`, conscribe walks the MRO chain to collect all parent parameters:

```python
class Parent:
    def __init__(self, x: int, y: str = "hello"): ...

class Child(Parent):
    def __init__(self, z: float, **kwargs):
        super().__init__(**kwargs)
```

Without MRO collection, Child's config would only have `z`. With it, the config includes `z`, `x`, and `y`.

### How the Chain Terminates

| Condition | Result |
|-----------|--------|
| Parent has no `**kwargs` | Natural termination -- all params known -> `extra="forbid"` |
| Parent is outside configured scope | Truncated -> `extra="allow"` |
| Depth limit reached | Truncated -> `extra="allow"` |

### Scope Control

The `mro_scope` parameter controls which classes in the MRO are included:

| Scope | Includes | Use Case |
|-------|----------|----------|
| `"local"` (default) | Only your project code | Safe default |
| `"third_party"` | + site-packages | Include library params |
| `"all"` | Everything except `object` | Maximum completeness |
| `["httpx", "pydantic"]` | Your code + listed packages | Precise control |

### Configuration Levels

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
```

## Open vs Closed Config Schema

Conscribe uses `**kwargs` presence to decide strictness:

```python
# No **kwargs -> extra="forbid" -> unknown fields rejected
class StrictProvider(Base):
    def __init__(self, *, model_id: str, temperature: float = 0.0): ...

# Has **kwargs, fully resolved chain -> extra="forbid"
class FullProvider(Base):
    def __init__(self, *, model_id: str, **kwargs):
        super().__init__(**kwargs)

# Has **kwargs, chain truncated by scope/depth -> extra="allow"
class PartialProvider(ThirdPartyBase):
    def __init__(self, *, model_id: str, **kwargs):
        super().__init__(**kwargs)
```

When `extra="allow"`, the generated stub includes a docstring note:

```python
class PartialProviderConfig(BaseModel):
    """This implementation accepts extra parameters (**kwargs).
    Undeclared fields will not cause errors."""
    model_config = ConfigDict(extra="allow")
```

## Type Degradation

When MRO traversal reaches third-party types that Pydantic can't serialize (e.g., `httpx.Auth`, `ssl.SSLContext`), conscribe degrades them to `Any`:

```python
auth: Any = None  # degraded from: httpx._types.AuthTypes | None
```

The stub header also lists all degraded fields. JSON Schema output includes `x-degraded-fields` for tooling.

See [MRO and Degradation](mro-and-degradation.md) for full details.
