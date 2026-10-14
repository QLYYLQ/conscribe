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

Conscribe removes the wait: you get IDE autocomplete while writing the config, and a framework that validates the whole config when it loads it, before the expensive setup.

## Using Generated Config Types

Your framework should provide generated config files (a Python module and a JSON Schema). Point your editor at the JSON Schema for YAML autocompletion:

```yaml
# experiment.yaml
llm:
  provider: open_ai        # <- autocomplete shows: open_ai | anthropic | ...
  model_id: gpt-4o         # <- autocomplete shows: str, "Model identifier"
  temperature: 0.5         # <- autocomplete shows: float, default 0.0
  typo_field: 123          # <- RED LINE: unknown field (extra="forbid")
```

The generated Python models look like:

```python
class OpenAiLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description='Model identifier, e.g. gpt-4o')
    temperature: float = Field(default=0.0, description='Sampling temperature, 0-2')
    provider: Literal['open_ai'] = 'open_ai'
```

The generated Python file, JSON Schema and runtime models share the field declarations. JSON Schema cannot run Python validators, so editor checks do not replace the framework's validation. If a validator cannot be reproduced in generated Python, conscribe reports an `ExtractionWarning` and notes the omission in the file.

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

Generated config: `max_steps: int = Field(default=100, description='Maximum number of steps before stopping')`. (The framework needs the `conscribe[docstring]` extra installed for this tier.)

### Tier 2: Annotated Metadata

You additionally get constraints (validation rules), titles, examples and aliases from `Field(...)`:

```python
class MyAgent(BaseAgent):
    def __init__(
        self, *,
        max_steps: Annotated[int, Field(gt=0, description="Max steps")] = 100,
        timeout: Annotated[int, Field(gt=0)] = 300,
    ):
        ...
```

Generated config: `max_steps: int = Field(default=100, description='Max steps', gt=0)`. A negative value is rejected at validation time ("Input should be greater than 0"). If a field declares an alias (`Field(alias="apiKey")`), the alias is the key you write in the config.

On constructor parameters, `exclude=` is rejected and `deprecated=` becomes a JSON Schema marker.

### Tier 3: Explicit Pydantic Model

Full Pydantic power, including cross-field validators:

```python
class OpenAIConfig(BaseModel):
    model_id: str
    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = 1.0

    @model_validator(mode="after")
    def one_sampling_knob(self):
        if self.temperature != 0.0 and self.top_p != 1.0:
            raise ValueError("set temperature or top_p, not both")
        return self

class ChatOpenAI(ChatBaseModel):
    __config_schema__ = OpenAIConfig
    def __init__(self, **kwargs): ...
```

The validator runs on your config: `{"provider": "open_ai", "model_id": "gpt-4o", "temperature": 0.5, "top_p": 0.9}` fails with "Value error, set temperature or top_p, not both".

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
| A pydantic model or pydantic dataclass | Natural termination -- its fields are the parameters -> `extra="forbid"` |
| Parent is outside configured scope | Truncated -> `extra="allow"` |
| Depth limit reached | Truncated -> `extra="allow"` |
| A constructor whose signature cannot be read | Truncated -> `extra="allow"`, with an `ExtractionWarning` |

Classes without an `__init__` of their own (`Generic`, `ABC`, Protocols, plain mixins) never stop the walk.

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

When the config is built, an invalid value at any level raises `ConfigDeclarationError` naming where it was set. These settings only decide which parent parameters the config exposes; they never change what the class's constructor actually accepts.

## Open vs Closed Config Schema

Conscribe uses `**kwargs` presence to decide strictness:

```python
from vendor_sdk import ThirdPartyBase   # an installed package, outside the "local" scope

# No **kwargs -> extra="forbid" -> unknown fields rejected
class StrictProvider(ChatBaseModel):
    def __init__(self, *, model_id: str, temperature: float = 0.0): ...

# Has **kwargs, fully resolved chain -> extra="forbid"
class FullProvider(StrictProvider):
    def __init__(self, *, retries: int = 3, **kwargs):
        super().__init__(**kwargs)

# Has **kwargs, chain cut by the MRO scope -> extra="allow"
VendorBridge = LLM.bridge(ThirdPartyBase)

class PartialProvider(VendorBridge):
    def __init__(self, *, model_id: str, **kwargs):
        super().__init__(**kwargs)
```

When `extra="allow"`, the generated model says so:

```python
class PartialProviderLLMConfig(BaseModel):
    # This implementation accepts extra parameters (**kwargs).
    # Undeclared fields will not cause errors.
    model_config = ConfigDict(extra="allow")

    model_id: str
    provider: Literal['partial_provider'] = 'partial_provider'
```

## Nested Config (Hierarchical Keys)

Some frameworks use hierarchical keys (e.g., `"openai.azure"`) with compound discriminators. In this case, the config has a hybrid format — level 0 params are flat, level 1+ are nested:

```yaml
# experiment.yaml
llm:
  model_type: openai           # flat (level 0 discriminator)
  temperature: 0.7             # flat (level 0 param)
  max_tokens: 1000
  provider:                    # nested (level 1)
    name: azure                # level 1 discriminator
    deployment: my-deployment
    api_version: 2024-02
```

The generated models look like:

```python
class AzureProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal['azure'] = 'azure'
    deployment: str
    api_version: str = '2024-02'


class OpenaiAzureLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal['openai'] = 'openai'
    temperature: float = 0.7
    max_tokens: int = 1000
    provider: AzureProviderConfig
```

Invalid combinations (e.g., `model_type: openai` + `provider.name: bedrock` when `"openai.bedrock"` isn't registered) are rejected at validation time. A key segment literally named `default` (for example, `openai.default`) can be selected by leaving its block out when that block can be built from defaults.

## Composed Config (Multi-Layer Inline Wiring)

Some frameworks provide a composed config schema covering multiple layers. In this mode, wired fields (like `llm` in an agent config) accept a full inline config object instead of just a key name:

```yaml
# experiment.yaml (composed config)
agent:
  - name: browser_use
    use_vision: true
    llm:                          # inline LLM config -- IDE autocompletes all fields
      provider: open_ai
      model_id: gpt-4o
      temperature: 0.7
    loop:
      name: react
      max_steps: 20
  - name: swe
    llm:
      provider: anthropic
      model_id: claude-sonnet-4-5
      max_tokens: 4096
    # no loop: the framework's default loop config is used
```

The IDE provides full autocompletion for nested config objects — typing `llm:` shows all available LLM providers and their parameters. Some wired fields have a default: when you leave them out (like `loop` for `swe` above), the framework uses that implementation with its default settings.

The top level only accepts the framework's sections, so a misspelled section (`agents:` instead of `agent:`) is an error. Some layers have no top-level section at all because their instances belong inside another component; writing one at the top level fails with a message that says where it goes:

```
layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive
```

## Reading Validation Errors

Validation errors are pydantic errors. Their `type` and location tell you what to fix:

| Error type | Example location | Meaning |
|---|---|---|
| `extra_forbidden` | `agents` | Unknown key: a typo, or a section/field this framework does not have |
| `union_tag_invalid` | `agent.0.browser_use.llm` | The discriminator names no registered implementation (`provider: mistral`); the message lists the valid ones |
| `missing` | `agent.0.browser_use.llm.open_ai.model_id` | A required field is not set |
| `inline_only_layer` | `archiver` | The layer has no top-level section; the message lists where its instances go |
| `legacy_flat_config` | | A block written in the shape conscribe 1.x used; the message names the nested block to write instead |

The location reads like a path into your file: `agent.0.browser_use.llm.open_ai.model_id` is the `model_id` of the `open_ai` LLM config of the first agent (whose `name` is `browser_use`).

## Type Degradation

When conscribe cannot build a schema for a parameter's type (a third-party class such as `ssl.SSLContext`, or a name the framework only imports for type checkers), the field accepts any value:

```python
ssl_context: Any = None  # degraded from: ssl.SSLContext | None
```

The generated file's header lists all degraded fields, and the JSON Schema marks them with `x-degraded-fields` / `x-degraded-from`, so tooling can flag them. The field keeps its default and description; only the type is no longer checked.

See [MRO and Degradation](mro-and-degradation.md) for full details.
