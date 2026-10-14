# CLI Reference

The `conscribe` command generates config sources, JSON Schemas and `.pyi` stubs from your registries, and inspects what is registered.

| Command | Does |
|---------|------|
| `generate-config` | Writes one layer's Python config source (`--json-schema` adds its JSON Schema) |
| `generate-composed-config` | Writes one composed JSON Schema or Python source for several layers |
| `generate-stubs` | Writes `.pyi` stubs for the attributes the framework injects into one layer's classes |
| `inspect` | Prints one layer's registered keys and their config fields |
| `scan` | Finds `create_registrar()` calls by parsing source files (imports nothing) |
| `list` | Imports packages, then lists what the registries hold |

## Loading layers

`generate-config`, `generate-composed-config`, `generate-stubs` and `inspect` load layers with the same options:

| Option | Description |
|--------|-------------|
| `--registrar MODULE:ATTRIBUTE` | Import `MODULE` and use the registrar at `ATTRIBUTE` (a class returned by `create_registrar()`; a dotted attribute path works) |
| `--discover PACKAGE` | Run `discover(PACKAGE)` so implementations register |
| `--allow-missing MODULE` | Forwarded to `discover(allow_missing=...)`: modules that fail because this optional dependency is missing are skipped |
| `--exclude PREFIX` | Forwarded to `discover(exclude=...)`: a dotted module or package name and everything below it is never imported (`pkg.generated` excludes `pkg.generated.x`, but not `pkg.generated_old`) |
| `--layer NAME` | The layer (registry name) of a one-layer command. Optional when exactly one `--registrar` is given. |
| `--layers NAME ...` | The layers of `generate-composed-config`. Default: every `--registrar` given. |

All of them except `--layer` are repeatable, and each takes one or more values (`--registrar a:A b:B` equals `--registrar a:A --registrar b:B`).

- The registrar modules are imported first, then the `--discover` packages, in the order given.
- A layer named by `--layer` / `--layers` uses the first matching explicit `--registrar`. If none matches, it uses the current registry of that name from the imports; a later `create_registrar()` takes over that global name. An unknown name is an error that lists the loaded layers, with a "did you mean".
- `discover()` runs strict, as it does by default in Python: a module that fails to import stops the command with an error naming the module and suggesting `--exclude` (or `--allow-missing` when an optional dependency is missing).
- The current directory is importable while a command runs, as with `python -m`.

## Output, errors and exit codes

A command exits with status `0` on success and `1` on an error, which it prints as `Error: <message>` on stderr without a traceback. Usage errors (an unknown option, a missing command) exit with `2` and are reported by argparse. A command also exits with `1`, silently, when stdout is closed early (`conscribe inspect ... | head`).

Errors include a bad `MODULE:ATTRIBUTE`, import failures, unknown or empty layers, and every registry, wiring, annotation, declaration and render error. A `WiringResolutionError` lists all its problems.

All requested content is generated before writing starts. Config commands stage each output in a temporary file next to its target, then replace the targets (`generate-config --json-schema` replaces the schema before the source) and finally print stdout output. A failure during staging leaves every target untouched and removes temporary files. Replacement is atomic per file; if a later replacement fails, an earlier target may already have changed. Missing parent directories are created, and config outputs follow symlinks while preserving the link.

`generate-stubs` validates and renders every module first, then writes the `.pyi` files directly. An I/O failure can leave earlier stubs written.

`generate-stubs` prints `StaleStubWarning` reports as `Warning: ...` lines on stderr, and `scan` does the same for `ExtractionWarning` reports about files it cannot parse. These two handlers force their warnings to be shown, overriding warning filters; those reports alone do not fail the command, even with `PYTHONWARNINGS=error::UserWarning`.

Other conscribe warnings use Python's warning filters; config extraction warnings point at your class definition. Set `PYTHONWARNINGS=error::UserWarning` to make these warnings fail the command: they become `Error: ...` and generation stops before writing. This also escalates other libraries' `UserWarning`s. Python's startup `-W` / `PYTHONWARNINGS` settings cannot name conscribe's own categories; pytest filters and `warnings.simplefilter("error", ConscribeWarning)` can.

## Commands

### `generate-config`

```
conscribe generate-config [--registrar MODULE:ATTRIBUTE ...] [--discover PACKAGE ...]
                          [--allow-missing MODULE ...] [--exclude PREFIX ...] [--layer NAME]
                          [--output OUTPUT] [--json-schema PATH]
```

| Option | Description |
|--------|-------------|
| `--output OUTPUT` | Write the Python source here; stdout when omitted |
| `--json-schema PATH` | Also write the layer's JSON Schema here. Must differ from `--output` (the same path is an `Error:`). |

### `generate-composed-config`

```
conscribe generate-composed-config [--registrar MODULE:ATTRIBUTE ...] [--discover PACKAGE ...]
                                   [--allow-missing MODULE ...] [--exclude PREFIX ...]
                                   [--layers NAME ...] [--inline-wiring | --no-inline-wiring]
                                   [--top-level-extra {forbid,ignore,allow}]
                                   [--format {python,json-schema}] [--output OUTPUT]
```

| Option | Description |
|--------|-------------|
| `--layers NAME ...` | Layers to compose (registry names); default: every `--registrar` |
| `--inline-wiring` / `--no-inline-wiring` | Wired fields hold the target layer's configs (default), or stay `Literal[...]` keys |
| `--top-level-extra` | Unknown top-level keys: `forbid` (default), `ignore` or `allow`. Inline-only layer names are refused in every mode. |
| `--format` | `json-schema` (default) or `python` |
| `--output OUTPUT` | Output file; stdout when omitted |

Each layer's section is named after its registry, and [inline-only layers](config-typing.md#inline-only-layers) have no section. Without `--layers`, composing two different explicit `--registrar` values with the same name is an error. With `--layers`, each name selects the first matching explicit registrar, then falls back to the global registry lookup described above.

### `generate-stubs`

```
conscribe generate-stubs [--registrar MODULE:ATTRIBUTE ...] [--discover PACKAGE ...]
                         [--allow-missing MODULE ...] [--exclude PREFIX ...] [--layer NAME]
                         [--output-dir OUTPUT_DIR] [--partial-class-fallback] [--prune]
```

| Option | Description |
|--------|-------------|
| `--output-dir OUTPUT_DIR` | Mirror module paths under this directory; default: next to each source file (never into an installed package) |
| `--partial-class-fallback` | Give each stubbed class `def __getattr__(self, name: str) -> Incomplete: ...` to tolerate dynamic attributes (looser type checking) |
| `--prune` | Delete the stale conscribe-generated stubs instead of reporting them |

Only `__wiring__` entries without a runtime slot (no constructor parameter, no non-`ClassVar` class annotation anywhere in the MRO) and without [`declare_only`](api-reference.md#__wiring__-entries) are stubbed, typed as the target registry's protocol. The command prints the written paths and `Generated N stub file(s).` on stdout, or `No classes with injected wired attributes found.` on stderr.

A conscribe-generated `.pyi` is stale when it sits in a directory this layer's stubs go to, this run did not rewrite it, and no class in the registries the [writer considers](api-reference.md#write_layer_stubs) needs it. These files are reported as `Warning: ...` lines; `--prune` deletes exactly those files. conscribe recognises both `# Generated by conscribe` and the older `# Auto-generated by conscribe.` followed by `# Stub file for:`.

Type checkers pick up stubs next to the source files without configuration. With `--output-dir`, mark the directory as a source root in PyCharm, or add it to your type checker's stub search path (for example mypy's `mypy_path`).

### `inspect`

```
conscribe inspect [--registrar MODULE:ATTRIBUTE ...] [--discover PACKAGE ...]
                  [--allow-missing MODULE ...] [--exclude PREFIX ...] [--layer NAME]
```

Prints the layer, its discriminator ([nested-mode](config-typing.md#nested-mode-discriminator_fields) layers: all fields and the separator), `Inline-only: yes` for inline-only layers, and every key with its config fields: `name: type = default  # description; wired from: registry`. The discriminator itself is not listed. A key whose config cannot be built is reported in an `Error:` line after the listing, and the exit status is `1`.

### `scan`

```
conscribe scan [--path PATH]
```

Parses the `.py` files under `PATH` (default: the current directory) for `create_registrar()` and `create_auto_registrar()` calls, without importing anything. (`--path` is the directory to search here; for `list` it filters entries.) Skips `site-packages`, `.venv`, `venv`, `.git`, `__pycache__`, `.eggs`, `.tox`, `node_modules`, `build`, `dist`, `.mypy_cache`, `.pytest_cache` and `.ruff_cache`.

### `list`

```
conscribe list [--discover PACKAGE ...] [--allow-missing MODULE ...] [--exclude PREFIX ...]
               [--layer LAYER] [--path PATH]
```

Imports the packages (strict `discover()`), then lists every registry with its entries: key, class and source location. Without `--discover` it imports every top-level package in the current directory (a directory with an `__init__.py`, one level deep, so a `src/` layout needs `--discover`). Only entries whose source file is under the current directory are shown, so classes from installed packages stay out.

| Option | Description |
|--------|-------------|
| `--discover PACKAGE ...` | Packages to import; auto-detected when omitted |
| `--layer LAYER` | Show only this registry |
| `--path PATH` | Show only entries whose source path (relative to the current directory) starts with `PATH`; empty registries are hidden |

## Worked example

A project with two layers:

```
my_app/
    __init__.py
    layers.py           # protocols, registrars, base classes
    llm/__init__.py
    llm/providers.py    # OpenAIChat ("openai"), AnthropicChat ("anthropic")
    agents/__init__.py
    agents/react.py     # ReactAgent: llm through the constructor, critic injected
```

```python
# my_app/layers.py
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class Agent(Protocol):
    def run(self, task: str) -> str: ...


LLM = create_registrar("llm", ChatModel, discriminator_field="provider")
AgentRegistrar = create_registrar(
    "agent", Agent, discriminator_field="name", strip_suffixes=["Agent"]
)


class BaseChatModel(metaclass=LLM.Meta):
    __abstract__ = True

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True

    def run(self, task: str) -> str:
        raise NotImplementedError
```

```python
# my_app/llm/providers.py
from typing import Annotated

from pydantic import Field

from my_app.layers import BaseChatModel


class OpenAIChat(BaseChatModel):
    """OpenAI chat models.

    Args:
        model: Model identifier, e.g. gpt-4o.
        temperature: Sampling temperature.
    """

    __registry_key__ = "openai"

    def __init__(
        self,
        model: str,
        temperature: Annotated[float, Field(ge=0, le=2)] = 0.7,
    ) -> None:
        self.model = model
        self.temperature = temperature


class AnthropicChat(BaseChatModel):
    __registry_key__ = "anthropic"

    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        self.model = model
        self.max_tokens = max_tokens
```

```python
# my_app/agents/react.py
from my_app.layers import BaseAgent, ChatModel


class ReactAgent(BaseAgent):
    """A ReAct loop.

    Args:
        llm: The model that plans each step.
        max_steps: Step budget.
    """

    # llm arrives through the constructor; the framework sets critic itself.
    __wiring__ = {"llm": "llm", "critic": "llm"}

    llm: ChatModel
    max_steps: int

    def __init__(self, llm: ChatModel, max_steps: int = 10) -> None:
        self.llm = llm
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        return self.llm.complete(task)
```

Every command below runs from the project root.

### Find the registrars

`scan` finds them without importing anything:

```bash
conscribe scan
```

```
Found 2 registrar(s):

  llm             ChatModel           my_app/layers.py:17  (var: LLM)
  agent           Agent               my_app/layers.py:18  (var: AgentRegistrar)
```

### List what registers

`list` imports the packages it finds and shows each registry's entries:

```bash
conscribe list
```

```
Registry: agent (Agent) — 1 entries
  react               ReactAgent              my_app/agents/react.py:5

Registry: llm (ChatModel) — 2 entries
  anthropic           AnthropicChat           my_app/llm/providers.py:28
  openai              OpenAIChat              my_app/llm/providers.py:9

```

### Inspect a layer's config fields

The agent's wiring targets the `llm` layer, so the whole `my_app` package is discovered:

```bash
conscribe inspect --registrar my_app.layers:AgentRegistrar --discover my_app
```

```
Layer: agent
Discriminator: name
Registered keys (1):

  react (ReactAgent):
    llm: Literal['anthropic', 'openai']  # The model that plans each step.; wired from: llm
    max_steps: int = 10  # Step budget.
    critic: Optional[Literal['anthropic', 'openai']] = None  # wired from: llm
```

`critic` has no slot in `ReactAgent` (no constructor parameter, no class annotation): the framework sets the attribute after construction, so the config field is an optional key, and `generate-stubs` declares the attribute below.

### Generate one layer's config source and JSON Schema

```bash
conscribe generate-config --registrar my_app.layers:LLM --discover my_app.llm --output generated/llm_config.py --json-schema generated/llm_config.schema.json
```

`generated/llm_config.py`:

```python
# Generated by conscribe 2.0.0
# Auto-generated by conscribe. DO NOT EDIT.
# Layer: llm
# Discriminator: provider
# Entries (2): anthropic, openai

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class AnthropicLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = 4096
    provider: Literal['anthropic'] = 'anthropic'


class OpenaiLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(description='Model identifier, e.g. gpt-4o.')
    temperature: float = Field(default=0.7, description='Sampling temperature.', ge=0, le=2)
    provider: Literal['openai'] = 'openai'


# Rebuild models for deferred annotation resolution
AnthropicLLMConfig.model_rebuild()
OpenaiLLMConfig.model_rebuild()


LLMConfig = Annotated[
    Union[OpenaiLLMConfig, AnthropicLLMConfig],
    Field(discriminator="provider"),
]
```

### Compose both layers

The composed config holds each agent's LLM config inline:

```bash
conscribe generate-composed-config --registrar my_app.layers:LLM my_app.layers:AgentRegistrar --discover my_app --format python --output generated/composed.py
```

The agent part of `generated/composed.py`:

```python
class ReactAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig = Field(description='The model that plans each step.')  # wired from: llm
    max_steps: int = Field(default=10, description='Step budget.')
    critic: Optional[LLMConfig] = None  # wired from: llm
    name: Literal['react'] = 'react'


# Rebuild models for deferred annotation resolution
ReactAgentConfig.model_rebuild()


# ── Composed Config ──


class ComposedConfig(BaseModel):
    # Unknown top-level keys are refused (top_level_extra='forbid').
    model_config = ConfigDict(extra="forbid", json_schema_extra={'x-composed-layers': ['llm', 'agent'], 'x-inline-wiring': True})

    llm: list[LLMConfig] = []
    agent: list[ReactAgentConfig] = []


# Rebuild models for deferred annotation resolution
ComposedConfig.model_rebuild()
```

The `agent` layer has a single key, so it has no union alias and its section is a list of `ReactAgentConfig`. A YAML config validated by this model (or by the JSON Schema `--format json-schema` writes) looks like:

```yaml
agent:
  - name: react
    max_steps: 20
    llm:
      provider: openai
      model: gpt-4o
```

### Generate stubs

The stub declares the injected `critic` attribute:

```bash
conscribe generate-stubs --registrar my_app.layers:AgentRegistrar --discover my_app
```

```
  /path/to/project/my_app/agents/react.pyi
Generated 1 stub file(s).
```

`my_app/agents/react.pyi`:

```python
# Generated by conscribe 2.0.0
# DO NOT EDIT. Stub for module: my_app.agents.react
from __future__ import annotations

from my_app.layers import BaseAgent, ChatModel
from typing import Any

def __getattr__(name: str) -> Any: ...

class ReactAgent(BaseAgent):
    critic: ChatModel  # wired from: llm

    llm: ChatModel
    max_steps: int

    def __init__(self, llm: ChatModel, max_steps: int = 10) -> None: ...
    def run(self, task: str) -> str: ...
```

Only `critic` gets a `# wired from:` line; `llm` reaches the class through the constructor. The stub repeats the class body's own annotations (`llm: ChatModel`, `max_steps: int`). That matters because a stub replaces its module for type checkers: an attribute that is only assigned inside `__init__` (`self.llm = llm`) is invisible to them once the stub exists, so mypy would report `"ReactAgent" has no attribute "llm"`. Declare such attributes in the class body, or pass `--partial-class-fallback`.

### Errors

A wrong attribute name:

```bash
conscribe generate-config --registrar my_app.layers:Llm --discover my_app
```

```
Error: --registrar 'my_app.layers:Llm': 'my_app.layers' has no attribute 'Llm' (registrars defined there: AgentRegistrar, LLM)
```

Discovering only the agents leaves the `llm` registry empty, which every wiring entry reports:

```bash
conscribe generate-config --registrar my_app.layers:AgentRegistrar --discover my_app.agents
```

```
Error: 2 wiring problems:
  - ReactAgent (key 'react') wiring 'llm': registry 'llm' is empty (no class is registered in it): discover() missed the package that implements it, or one of its modules failed to import; import every implementation module before resolving wiring
  - ReactAgent (key 'react') wiring 'critic': registry 'llm' is empty (no class is registered in it): discover() missed the package that implements it, or one of its modules failed to import; import every implementation module before resolving wiring
```

A typo in a layer name:

```bash
conscribe generate-config --layer agnt --discover my_app
```

```
Error: no registrar for layer 'agnt' is loaded (did you mean 'agent'?); loaded layers: agent, llm. Pass --registrar MODULE:ATTRIBUTE, or --discover the package that creates it.
```

Each of these exits with status `1`.

### Modules that fail to import

With the `tools_pkg` package from [Registration Deep-Dive](registration.md#discover) (a `generated` subpackage that raises when imported, and a module that needs the missing optional dependency `fancy_ocr`), the strict discovery stops at the first failure and says how to skip it:

```bash
conscribe list --discover tools_pkg
```

```
Error: could not import module 'tools_pkg.generated' (--discover tools_pkg): RuntimeError: generated code is not meant to be imported (tools_pkg/generated/__init__.py, line 1)
Fix the module, or skip it with --exclude tools_pkg.generated.
```

```bash
conscribe list --discover tools_pkg --exclude tools_pkg.generated
```

```
Error: could not import module 'tools_pkg.vision' (--discover tools_pkg): ModuleNotFoundError: No module named 'fancy_ocr' (tools_pkg/vision.py, line 8)
If 'fancy_ocr' is an optional dependency, skip the modules that need it with --allow-missing fancy_ocr; or skip this module with --exclude tools_pkg.vision.
```

```bash
conscribe list --discover tools_pkg --exclude tools_pkg.generated --allow-missing fancy_ocr
```

```
Registry: tool (ToolProtocol) — 1 entries
  search              Search                  tools_pkg/search.py:4

```

## Programmatic equivalents

Every command maps to a few calls (run from the project root):

```python
from conscribe import (
    build_composed_config,
    build_layer_config,
    discover,
    generate_composed_config_source,
    generate_composed_json_schema,
    generate_layer_config_source,
    generate_layer_json_schema,
)
from conscribe.stubs import write_layer_stubs
from my_app.layers import LLM, AgentRegistrar          # --registrar my_app.layers:...

discover("my_app")                                     # --discover my_app

result = build_layer_config(LLM)                       # generate-config
source = generate_layer_config_source(result)
schema = generate_layer_json_schema(result)            # --json-schema

composed = build_composed_config(                      # generate-composed-config
    {"llm": LLM, "agent": AgentRegistrar},             # section name -> registrar
    inline_wiring=True,
    top_level_extra="forbid",
)
composed_schema = generate_composed_json_schema(composed)   # --format json-schema
composed_source = generate_composed_config_source(composed)  # --format python

written = write_layer_stubs(AgentRegistrar)            # generate-stubs (prune=True for --prune)

for key in AgentRegistrar.keys():                      # inspect
    schema = AgentRegistrar.get_config_schema(key)
    print(key, list(schema.model_fields) if schema is not None else [])
```

```
react ['llm', 'max_steps', 'critic']
```

`conscribe.scanner` provides `scan_registrar_definitions(root)` and `list_registries(root, ...)` for `scan` and `list`.

## Removed in 2.0

- `update-stubs` and the `--config` batch YAML file are gone. Call the generation commands from your build or CI script instead. Compare `compute_registry_fingerprint()` with a stored value to track config-source and JSON-Schema changes; it does not cover `.pyi` stubs, which need separate regeneration.
- `discover()` no longer regenerates stubs (`auto_update_stubs`, `stub_dir` and `fingerprint_path` were removed).

See [Migrating to 2.0](migrating-to-2.0.md).
