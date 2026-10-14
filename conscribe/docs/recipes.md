# Recipes

Task-oriented answers to common questions. Registration examples use the registrar and base below. Later snippets assume your application supplies the named registrars, protocols and implementations; replace `my_app` imports with your own modules. Treat alternative class definitions as separate examples. YAML-loading examples require PyYAML (`pip install pyyaml`).

## Registration

### How do I create a new layer (registry)?

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

### How do I register a class automatically?

Create a base class with the registrar's metaclass, then subclass it:

```python
class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True  # base classes are not registered

    def step(self) -> None: ...

class MyAgent(BaseAgent):
    def step(self): ...
# MyAgent is now registered as "my"
```

### How do I register an external class?

Use `bridge()` for classes not inheriting your base:

```python
from ext_framework import ExtAgent as TheirAgent

TheirBridge = AgentRegistrar.bridge(TheirAgent)

class MyExtAgent(TheirBridge):
    def step(self): ...
# MyExtAgent is registered as "my_ext"
```

For a one-off, use the decorator:

```python
@AgentRegistrar.register("special_agent")
class SpecialAgent:
    def step(self): ...
```

### How do I override the inferred registry key?

Set `__registry_key__` on the class:

```python
class ChatGPT4(ChatBase):
    __registry_key__ = "gpt4"
    def chat(self, messages): ...
# Registered as "gpt4" instead of "chat_gpt4"
```

A key must be a non-empty string. An Enum member, a tuple or an int raises `InvalidRegistryKeyError` when the class is created (write `Kind.REACT.value`, not `Kind.REACT`).

### How do I register under multiple keys?

Set `__registry_key__` to a list:

```python
class UniversalProvider(ChatBase):
    __registry_key__ = ["openai_compatible", "local_server"]
    def __init__(self, endpoint: str): ...
# Registered under BOTH keys
# cls.__registry_key__ == "openai_compatible"  (primary)
# cls.__registry_keys__ == ["openai_compatible", "local_server"]
```

If one of the keys is already taken, `DuplicateKeyError` is raised and the class is registered under none of them.

### How do I use hierarchical (dotted) keys?

Set `key_separator` on the registrar:

```python
LLM = create_registrar(
    "llm", ChatProtocol,
    discriminator_field="provider",
    key_separator=".",
)

class OpenAIBase(metaclass=LLM.Meta):
    __registry_key__ = "openai"
    __abstract__ = True

class AzureOpenAI(OpenAIBase):
    __registry_key__ = "openai.azure"
    def chat(self, messages): ...

class OpenAIProxy(OpenAIBase):  # no __registry_key__: derived as "openai.open_ai_proxy"
    def chat(self, messages): ...

# Query the hierarchy:
LLM.children("openai")  # {"openai.azure": AzureOpenAI, "openai.open_ai_proxy": OpenAIProxy}
LLM.tree()               # {"openai": {"azure": AzureOpenAI, "open_ai_proxy": OpenAIProxy}}
```

### How do I register in multiple registries (diamond inheritance)?

Use the `|` operator on metaclasses:

```python
LLM = create_registrar("llm", LLMProtocol, discriminator_field="provider")
Agent = create_registrar("agent", AgentProtocol, discriminator_field="name")

CombinedMeta = LLM.Meta | Agent.Meta

class LLMAgentBase(metaclass=CombinedMeta):
    __abstract__ = True
    def chat(self, messages: list) -> str: ...
    def step(self) -> None: ...

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

### How do I control which subclasses get registered?

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

### How do I skip a class from registration?

Mark it abstract:

```python
class IntermediateBase(StrictBase):
    __abstract__ = True  # not registered
```

Or use a custom skip filter:

```python
AgentRegistrar = create_registrar(
    "agent", proto,
    discriminator_field="name",
    skip_filter=lambda cls: cls.__name__.startswith("Internal"),
)
```

### How do I use conscribe with Pydantic BaseModel + Generic[T]?

It works automatically. `skip_pydantic_generic=True` (default) filters Generic intermediates such as `BaseEvent[str]`; see [Guide: Framework Developer](guide-alice.md#pydantic-basemodel--generict) for an example. The config of a registered model is built from its fields. Its validators are not part of that config (they run when your framework instantiates the class); use `__config_schema__` if configs must pass them when they are loaded.

### How do I trigger module imports for registration?

Use `discover()` to recursively import all modules under a package:

```python
from conscribe import discover

discover("my_app.agents", "my_app.llm.providers")
# All classes in those packages are now registered
```

`discover()` stops at the first module that fails to import. On Python 3.11+ the original exception carries a note naming the module; on 3.10 a failing submodule is wrapped in a chained `ImportError`. Skip expected missing dependencies or exclude modules explicitly:

```python
discover(
    "my_app.llm.providers",
    allow_missing=["vllm"],               # skip providers whose optional dependency is missing
    exclude=["my_app.llm.providers.dev"], # never import this module or package
)
```

`allow_missing` names the missing dependency (`vllm` here), and only covers `ModuleNotFoundError`, including one wrapped in an `ImportError` chain. `strict=False` skips submodules that raise `ImportError` or `SyntaxError`, with an `ExtractionWarning` for each; explicitly named packages still raise unless covered by `allow_missing`. Read the failure record with `from conscribe.discover import import_failures`, then call `import_failures()`.

### How do I list all registered implementations?

```python
print(LLM.keys())          # ['my_llm_agent', 'llm_only']
print(LLM.get_all())       # {'my_llm_agent': <class '...MyLLMAgent'>, 'llm_only': <class '...LLMOnly'>}
```

### How do I remove a registration?

```python
removed = LLM.unregister("llm_only")      # returns the removed class
LLM.unregister("llm_only", missing_ok=True)  # already gone: returns None, no error
```

Without `missing_ok=True`, removing a key that is not registered raises `KeyNotFoundError`. In tests, prefer `isolated_registries()` (next section): it also undoes registrars a test created and restores what a test removed.

## Testing

### How do I keep test classes out of the registries?

Classes register when they are created, so a class defined in one test would stay registered for every later test. Load the pytest plugin in the root `conftest.py` and use the `conscribe_isolated` fixture, or make it autouse:

```python
# conftest.py (the root one)
import pytest

pytest_plugins = ["conscribe.testing.pytest_plugin"]

@pytest.fixture(autouse=True)          # optional: isolate every test
def _isolated(conscribe_isolated):
    yield conscribe_isolated
```

```python
from my_app.llm import ChatBaseModel, LLMRegistrar

def test_with_a_fake_provider():
    class ChatFake(ChatBaseModel):
        def chat(self, messages):
            return "canned answer"

    assert LLMRegistrar.get("fake") is ChatFake
    # unregistered again when the test ends
```

Without pytest, use the context manager directly: `with conscribe.testing.isolated_registries(): ...`. Registrations made by modules imported for the first time inside the block are kept (the module stays imported and would not register again). See [Testing](testing.md) for the exact rules.

### How do I reuse module-level fakes across tests?

Classes defined at module level register when pytest collects the module, outside any fixture. Capture them once and re-apply them per test:

```python
import pytest
from conscribe.testing import isolated_registries
from my_app.llm import ChatBaseModel, LLMRegistrar

FAKES = isolated_registries()
with FAKES:                              # registered, recorded, rolled back
    class ChatEcho(ChatBaseModel):
        def chat(self, messages):
            return messages[-1]["content"]

@pytest.fixture
def echo_llm():
    with FAKES.activate():               # re-applied for this test only
        yield LLMRegistrar.get("echo")

def test_echo(echo_llm):
    assert echo_llm().chat([{"content": "hi"}]) == "hi"

def test_without_fakes():
    assert "echo" not in LLMRegistrar.keys()
```

### How do I make conscribe's warnings fail my tests?

All of conscribe's warnings subclass `ConscribeWarning`. Turn them into errors:

```ini
[pytest]
filterwarnings =
    error::conscribe.ConscribeWarning
```

Outside pytest: `python -W error::conscribe.ConscribeWarning ...`.

### How do I check every wiring declaration at once?

`validate_wiring()` resolves the `__wiring__` of every registered class and raises one `WiringResolutionError` listing every problem:

```python
from conscribe import discover, validate_wiring
from my_app.agents import AgentRegistrar
from my_app.llm import LLMRegistrar

def test_wiring():
    discover("my_app.llm.providers", "my_app.agents")
    validate_wiring({"llm": LLMRegistrar, "agent": AgentRegistrar})
```

`build_layer_config()` and `build_composed_config()` run the same check before they build anything.

## Config Typing

### How do I generate config files for IDE autocomplete?

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
  --registrar my_app.agents:AgentRegistrar \
  --registrar my_app.llm:LLMRegistrar \
  --layer agent \
  --discover my_app.llm.providers \
  --output generated/agent_config.py
```

The generated source behaves exactly like the in-process models. If a default or type cannot be written faithfully, generation raises `ConfigRenderError` instead of writing a file that differs.

### How do I generate JSON Schema for YAML editors?

```python
from conscribe import build_layer_config, generate_layer_json_schema
import json

result = build_layer_config(AgentRegistrar)
schema = generate_layer_json_schema(result)

with open("generated/agent_config.schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

### How do I validate config at runtime?

```python
from pydantic import TypeAdapter
import yaml

union_type = AgentRegistrar.config_union_type()
raw = yaml.safe_load(open("config.yaml"))
config = TypeAdapter(union_type).validate_python(raw["agent"])
# ValidationError raised immediately if config is invalid
```

### How do I add parameter descriptions to generated config?

Use a Google/NumPy-style docstring (Tier 1.5, requires `pip install conscribe[docstring]`):

```python
class MyAgent(BaseAgent):
    """My custom agent.

    Args:
        max_steps: Maximum number of steps before stopping
        timeout: Timeout in seconds
    """
    def __init__(self, *, llm: str, max_steps: int = 100, timeout: int = 300): ...
```

Or use `Annotated` with `Field` (Tier 2):

```python
from typing import Annotated
from pydantic import Field

class MyAgent(BaseAgent):
    def __init__(
        self, *,
        llm: str,
        max_steps: Annotated[int, Field(gt=0, description="Max steps")] = 100,
    ): ...
```

Declare the default once: in the signature or in `Field(default=...)`. Two different defaults raise `ConfigDeclarationError`.

### How do I use a full Pydantic model for complex config?

Use Tier 3's `__config_schema__` escape hatch. The per-key config inherits its fields, validators and `model_config`, adds the discriminator, and defaults an unset `extra` policy to `"forbid"`. Composed builds also inline its wired fields:

```python
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

class MyAgentConfig(BaseModel):
    model_id: str
    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = 1.0

    @model_validator(mode="after")
    def one_sampling_knob(self):
        if self.temperature != 0.0 and self.top_p != 1.0:
            raise ValueError("set temperature or top_p, not both")
        return self

class MyAgent(BaseAgent):
    __config_schema__ = MyAgentConfig

    def __init__(self, **kwargs): ...
```

```python
adapter = TypeAdapter(AgentRegistrar.config_union_type())
try:
    adapter.validate_python({"agent_type": "my", "model_id": "m", "temperature": 0.5, "top_p": 0.9})
except ValidationError as exc:
    print(exc.errors()[0]["msg"])  # Value error, set temperature or top_p, not both
```

Every `__wiring__` entry that has a constructor parameter or class annotation must also be a field of the schema (`ConfigDeclarationError` otherwise).

### How do I keep the fields of a single model parameter at the top of the config?

A parameter typed with a pydantic model is a nested block in the config. To flatten it, name it in `__config_from_param__`:

```python
class SearchSettings(BaseModel):
    engine: str = "duckduckgo"
    max_results: int = 10

class SearchAgent(BaseAgent):
    __config_from_param__ = "settings"

    def __init__(self, settings: SearchSettings = SearchSettings()):
        self.settings = settings

# config: {agent_type: search, engine: bing}   instead of   {agent_type: search, settings: {engine: bing}}
```

The attribute is inherited (set it to `None` in a subclass to opt out). The other constructor parameters, if any, must be wiring receptors.

### How do I handle types Pydantic can't serialize?

Conscribe degrades incompatible types (e.g., `ssl.SSLContext`) to `Any` and emits a `DegradationWarning` that points at the class:

```python
class SecureAgent(BaseAgent):
    def __init__(self, *, ssl_context: ssl.SSLContext | None = None): ...
```

The generated file marks the field, and its header lists every degraded field:

```python
ssl_context: Any = None  # degraded from: ssl.SSLContext | None
```

The same happens for a name the module only imports under `if TYPE_CHECKING:` (conscribe never imports those blocks). A name that exists nowhere is a typo and raises `ConfigAnnotationError`.

### How do I control which parent classes are traversed for `**kwargs` chains?

Set `mro_scope` at the registrar level or per-class:

```python
# Registrar-level (applies to all registered classes)
AgentRegistrar = create_registrar("agent", proto, discriminator_field="name", mro_scope="third_party")

# Per-class override
class MyAgent(BaseAgent):
    __config_mro_scope__ = ["httpx"]
    __config_mro_depth__ = 2

    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(**kwargs)
```

### How do I build nested config with compound discriminators?

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

### How do I regenerate files only when something changed?

`compute_registry_fingerprint()` hashes a layer's config source, JSON Schema, conscribe version and layer settings. Use it to detect changes to those config outputs; `.pyi` stubs are not covered:

```python
from pathlib import Path
from conscribe import (
    build_layer_config,
    compute_registry_fingerprint,
    generate_layer_config_source,
    load_cached_fingerprint,
    save_fingerprint,
)

cache = Path("generated/.fingerprints.json")
current = compute_registry_fingerprint(LLMRegistrar)
if load_cached_fingerprint(cache, "llm") != current:
    source = generate_layer_config_source(build_layer_config(LLMRegistrar))
    Path("generated/llm_config.py").write_text(source, encoding="utf-8")
    save_fingerprint(cache, "llm", current)
```

The fingerprint never raises: a layer that cannot be generated gets a fingerprint of its error, so a fix counts as a change. For composed configs, track every participating layer and the composition options (`inline_wiring` and `top_level_extra`). Regenerate stubs separately.

## Wiring

### How do I constrain a config field to values from another registry?

Use `__wiring__` to declare cross-registry constraints:

```python
# Setup: create a loop registry
LoopRegistrar = create_registrar("loop", LoopProtocol, discriminator_field="name")

class BaseLoop(metaclass=LoopRegistrar.Meta):
    __abstract__ = True
    def run(self) -> None: ...

class ReactLoop(BaseLoop): ...
class CodeactLoop(BaseLoop): ...

# Agent registry with wiring
AgentRegistrar = create_registrar("agent", AgentProtocol, discriminator_field="name")

class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True
    __wiring__ = {"loop": "loop"}  # auto-discover: all keys from "loop" registry
    def step(self) -> None: ...

class SWEAgent(BaseAgent):
    __wiring__ = {"loop": ("loop", ["react_loop"])}  # narrow to subset
    def __init__(self, *, loop: str = "react_loop", max_steps: int = 10): ...

# Generated config constrains loop to Literal["react_loop"]
result = build_layer_config(AgentRegistrar)
source = generate_layer_config_source(result)
```

```python
class SweAgentAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loop: Literal['react_loop'] = 'react_loop'  # wired from: loop
    max_steps: int = 10
    name: Literal['swe_agent'] = 'swe_agent'
```

Three modes:
- `"field": "registry_name"` — all keys (auto-discovery)
- `"field": ("registry_name", ["key1", "key2"])` — explicit subset; a third element lists optional keys: `("registry_name", ["required"], ["optional"])`. Both appear in `Literal[...]`; the distinction is metadata for your framework's negotiation logic.
- `"field": ["val1", "val2"]` — literal list (no registry)

The registry-backed modes can also be written as a dict: `{"registry": "loop", "keys": [...], "optional_keys": [...]}`, which adds the `default` and `declare_only` options described below.

Use `None` to exclude an inherited wiring key: `__wiring__ = {"llm": None}`. A subclass entry replaces the inherited entry as a whole.

Every problem is an error: an unknown key or registry, an empty registry, a malformed entry, a parameter default that is not an allowed key. Malformed entries fail when the class is created; the rest fail when a config is built, all reported in one `WiringResolutionError`.

### How do I wire to a capability without naming it?

If the target registry uses a `key_separator`, name only the trailing segment. Conscribe expands it to every provider that offers it, so the same class works against any of them:

```python
# Registered: "browser.click", "desktop.click", "browser.scroll"

class PortableAgent(BaseAgent):
    __wiring__ = {"action": ("action", ["click"], ["scroll"])}
    # -> Literal["browser.click", "desktop.click", "browser.scroll"]

class PinnedAgent(BaseAgent):
    __wiring__ = {"action": ("action", ["browser.click"])}  # escape hatch
    # -> Literal["browser.click"]
```

Write the full key (`"browser.click"`) to pin one provider instead. All matches stay in the `Literal[...]` when a short name is ambiguous — deciding between them is your framework's job, at the point where it knows which providers are loaded.

### How do I give a wired field a default?

Use the dict form with `"default"`. It needs a single-instance receptor (a constructor parameter or class annotation that is not a container) and an exact, registered key:

```python
class CodingAgent(BaseAgent):
    __wiring__ = {"env": {"registry": "env", "default": "bash"}}

    def __init__(self, *, env: EnvProtocol, loop: str = "react_loop"): ...
```

With key selectors the field becomes `Literal['bash', 'browser'] = 'bash'`. In a composed config with inline wiring, an omitted `env` becomes the whole default config of `bash`, and the JSON Schema shows that object as the default:

```python
composed = build_composed_config({"agent": AgentRegistrar, "env": EnvRegistrar, "loop": LoopRegistrar})
config = composed.top_level_type.model_validate({"agent": [{"name": "coding_agent"}]})
print(config.agent[0].env)   # shell='/bin/bash' name='bash'
```

The default config must be buildable from defaults (`ConfigDeclarationError` otherwise). `env: null` is refused unless the receptor's annotation admits `None`; then `null` means "no environment":

```python
class OptionalEnvAgent(BaseAgent):
    __wiring__ = {"env": {"registry": "env", "default": "bash"}}

    def __init__(self, *, env: EnvProtocol | None = None, loop: str = "react_loop"): ...
```

For `OptionalEnvAgent`, omitting `env` uses the `bash` default config; explicitly writing `null` gives `None`. A conflicting non-`None` parameter default is an error. A wired parameter whose own default is a key string (`loop: str = "react_loop"`) works the same way in inline builds.

### How do I declare a constraint the framework only reads?

Mark the entry `declare_only`. conscribe installs no placeholder for it, never writes it into a stub and never inlines it; the config field stays an optional key selector:

```python
class ShellAgent(BaseAgent):
    __wiring__ = {
        "obs": {
            "registry": "observation",
            "keys": ["terminal"],             # required by this agent
            "optional_keys": ["filesystem"],  # used when available
            "declare_only": True,             # read by the framework, never injected
        },
    }
```

Read the declaration with `from conscribe.introspect import wiring_slots`, then `wiring_slots(ShellAgent)["obs"]`. Its `.spec.allowed_keys` holds the required keys and `.spec.optional_keys` the optional keys; `.declare_only` identifies this kind of entry. A `declare_only` entry cannot have a `default`, and the class must not have a parameter or annotation of that name. A supplied config selector is still validated against the keys; the framework decides how to use that selection without assigning an attribute.

### How do I accept several instances of another layer?

Annotate the receptor as a container. The config field becomes a list of keys, or a list of inline configs in a composed build:

```python
class ObservedAgent(BaseAgent):
    __wiring__ = {"hooks": "hook"}

    def __init__(self, *, hooks: list[HookProtocol] = []):
        self.hooks = hooks
```

Per layer: `hooks: list[Literal['logger', 'tracer']] = []`. Composed: `hooks: [{name: logger}, {name: tracer}]`.

### How do I find the injection placeholders of a class?

A wired name without a constructor parameter or a value or annotation in the class body can receive a `WiredField` placeholder. It raises `AttributeError` until your framework assigns the attribute. Look placeholders up with `wired_field()` / `wired_fields()`; these functions also find inherited placeholders without triggering them:

```python
from conscribe import wired_field, wired_fields

class Planner(BaseAgent):
    __wiring__ = {"env": "env"}   # no parameter, no annotation: injected after construction

class TypedPlanner(Planner):
    env: EnvProtocol              # a runtime slot: config requires env; no stub for env

print(Planner.env)                          # WiredField('env', registry='env')
print(wired_field(TypedPlanner, "env"))     # WiredField('env', registry='env')
print(wired_fields(TypedPlanner))           # {'env': WiredField('env', registry='env')}
print(hasattr(TypedPlanner, "env"))         # False: TypedPlanner.env raises AttributeError
```

An annotation-only subclass inherits the placeholder, but class access raises so that `@dataclass`, attrs and pydantic see no default. `wired_field()` returns `None` if no placeholder exists. A class-body value takes precedence and is read normally.

### How do I route config values to the constructor or to attributes?

`conscribe.introspect` answers the question the same way conscribe does: `wiring_slots(cls)` gives each wired entry and its runtime slot, `receptor_slots(cls)` every slot, `init_param_names(cls)` the names the constructor accepts:

```python
from conscribe.introspect import wiring_slots

def assemble(cls, built: dict):
    """Create cls with the already-built wired instances in *built*."""
    slots = wiring_slots(cls)
    built = {name: value for name, value in built.items() if not slots[name].declare_only}
    kwargs = {name: value for name, value in built.items()
              if slots[name].slot is not None and slots[name].slot.via == "init"}
    obj = cls(**kwargs)
    for name, value in built.items():
        if name not in kwargs:
            setattr(obj, name, value)   # annotation slots and injected attributes
    return obj
```

`slot.via` is `"init"` or `"annotation"`; `slot.shape` is `"scalar"`, `"list"` or `"dict"`. An entry without a slot has `.slot is None`. Declaration-only entries are skipped during injection.

### How do I get IDE autocomplete for wired attributes injected at runtime?

An attribute your framework sets after construction (a wiring entry with no constructor parameter and no class annotation) is invisible to type checkers. Generate `.pyi` stubs that declare it:

```bash
conscribe generate-stubs \
  --registrar my_app.agents:AgentRegistrar \
  --discover my_app.agents \
  --prune
```

This writes a `.pyi` next to each source module that needs one. The attribute is typed as the target registry's protocol:

```python
class ShellAgent(BaseAgent):
    env: EnvironmentProtocol  # wired from: environment

    def step(self) -> None: ...
```

`--prune` deletes a generated stub only if it is in a directory used by this layer's modules, this run did not rewrite it, and no registered class needs it. Both current and legacy generated-stub headers are recognised. Without `--prune`, those files are reported as warnings. Discover the whole package first, since undiscovered classes cannot protect their stubs. `--output-dir` applies the same rule to the corresponding output directories.

Entries with a parameter or annotation, and `declare_only` entries, are never stubbed. To give an attribute a narrower type than the protocol, annotate it in the class body. It becomes a runtime slot: its config field uses the slot or wiring default, is optional for an annotation admitting `None` without a value, and is otherwise required.

Or programmatically:

```python
from conscribe.stubs import write_layer_stubs
from my_app.agents import AgentRegistrar

written = write_layer_stubs(AgentRegistrar, output_dir="typings", prune=True)
```

See [CLI Reference](cli.md) for all options.

## Composed Config

### How do I compose multiple layers into a single config schema?

Use `build_composed_config()` to combine layers. Wired fields become inline config objects:

```python
from conscribe import build_composed_config
from conscribe import generate_composed_json_schema, generate_composed_config_source

result = build_composed_config(
    {"llm": LLMRegistrar, "loop": LoopRegistrar, "agent": AgentRegistrar},
    inline_wiring=True,   # wired fields become target layer union types
)

# JSON Schema for YAML editors
schema = generate_composed_json_schema(result)

# Or Python source
source = generate_composed_config_source(result)
```

The resulting YAML config supports inline wiring:

```yaml
agent:
  - name: browser_use
    use_vision: true
    llm:                        # inline LLM config -- full IDE autocompletion
      provider: open_ai
      model_id: gpt-4o
      temperature: 0.7
```

Use `inline_wiring=False` to keep wired fields as `Literal[...]` key selectors.

Or via CLI:

```bash
conscribe generate-composed-config \
  --registrar my_app.llm:LLMRegistrar \
  --registrar my_app.agents:AgentRegistrar \
  --discover my_app.llm.providers \
  --format json-schema \
  --output composed.schema.json
```

### How do I allow other sections at the top level of a composed config?

The top level refuses keys that are not layer names. Relax it with `top_level_extra` (CLI: `--top-level-extra`):

```python
result = build_composed_config(
    {"llm": LLMRegistrar, "agent": AgentRegistrar},
    top_level_extra="allow",   # keep unknown sections in model_extra ("ignore" drops them)
)
config = result.top_level_type.model_validate({"agent": [], "logging": {"level": "debug"}})
print(config.model_extra)      # {'logging': {'level': 'debug'}}
```

### How do I keep a layer out of the top level?

Declare it inline-only. Its instances can then only be written inside the receptors of other layers:

```python
from conscribe import build_composed_config, create_registrar
from pydantic import ValidationError

Hook = create_registrar("hook", HookProtocol, discriminator_field="name")
Archiver = create_registrar("archiver", ArchiverProtocol, discriminator_field="name", inline_only=True)


class ArchiverBase(metaclass=Archiver.Meta):
    __abstract__ = True


class DiskArchiver(ArchiverBase):
    __registry_key__ = "disk"

    def __init__(self, *, path: str = "traces/"): ...


class HookBase(metaclass=Hook.Meta):
    __abstract__ = True


class TraceWriter(HookBase):
    __registry_key__ = "trace_writer"
    __wiring__ = {"archive": {"registry": "archiver", "default": "disk"}}

    def __init__(self, *, archive: object): ...


ComposedConfig = build_composed_config({"hook": Hook, "archiver": Archiver}).top_level_type

config = ComposedConfig.model_validate({"hook": [{"name": "trace_writer"}]})
print(config.hook[0].archive)       # the default instance

try:
    ComposedConfig.model_validate({"archiver": [{"name": "disk"}]})
except ValidationError as exc:
    print(exc.errors()[0]["msg"])
```

```
path='traces/' name='disk'
layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive
```

`result.inline_only_layers` lists where each inline-only layer can be written, and `result.top_level_layers` the layers that do have a section. An inline-only layer that no receptor can hold, or one passed with `inline_wiring=False`, raises `ConfigDeclarationError`.

### How do I handle circular wiring in composed config?

`build_composed_config()` raises `CircularWiringError` if layers have circular wiring dependencies:

```python
Agent = create_registrar("agent", AgentProtocol, discriminator_field="name")
LLM = create_registrar("llm", LLMProtocol, discriminator_field="provider")

class AgentBase(metaclass=Agent.Meta):
    __abstract__ = True
    __wiring__ = {"llm": "llm"}           # agent -> llm

class LLMBase(metaclass=LLM.Meta):
    __abstract__ = True
    __wiring__ = {"judge": "agent"}       # llm -> agent: a cycle
```

```
Circular wiring dependency detected: agent -> llm -> agent. Composed config requires an acyclic dependency graph between layers.
```

Fix by removing one direction of the cycle, or exclude one of the layers from the composed config. `declare_only` entries do not count as dependencies.

## CLI

### How do I inspect a registry from the CLI?

```bash
conscribe inspect \
  --registrar my_app.agents:AgentRegistrar \
  --discover my_app.llm.providers
```

```
Layer: agent
Discriminator: name
Registered keys (2):

  browser_use (BrowserUse):
    llm: Literal['anthropic', 'open_ai']  # wired from: llm
    use_vision: bool = False

  react (React):
    llm: Literal['anthropic', 'open_ai']  # wired from: llm
    max_steps: int = 20
```
