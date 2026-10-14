# Migrating to conscribe 2.0

conscribe 1.x let many declaration problems pass without an error. An unknown wiring key turned the field into a plain `str`, a module that failed to import was skipped with a log line, a typo in a composed config's top level was ignored. 2.0 raises an error for each of them, at class creation or at build time, and the message names the class, the field, the bad value and the fix.

Most projects need a handful of changes. This guide lists them in the order you are likely to hit them, with the error you will see and the fix. The complete list of changes is in the [CHANGELOG](https://github.com/QLYYLQ/conscribe/blob/master/CHANGELOG.md).

## Before you upgrade

- Python 3.10 or newer and pydantic 2.7 or newer (still below 3). Python 3.9 support is gone. Python 3.13 and 3.14 need a pydantic release with wheels for them; the test suite runs with 2.8.2 on 3.13 and 2.12 on 3.14.
- Pin the new major version: `conscribe>=2,<3`.
- Make conscribe's warnings fail your test suite, so that nothing it cannot read goes unnoticed:

  ```ini
  # pytest.ini
  [pytest]
  filterwarnings =
      error::conscribe.ConscribeWarning
  ```

  Every warning conscribe emits is a `ConscribeWarning` subclass: `ExtractionWarning` (conscribe could not read part of a class, or `discover(strict=False)` skipped a module), `DegradationWarning` (a field type fell back to `Any`), `StaleStubWarning` (old generated stubs, see section 9) and `IsolationWarning` (see the [testing guide](testing.md)). Warnings about a class carry the file and line of its definition.

## 1. Wiring problems are errors

`build_layer_config()` and `build_composed_config()` now resolve the `__wiring__` of every class registered in the layers being built before they build anything, and raise one `WiringResolutionError` that lists every problem they found. In 1.x the same declarations built "successfully": the field just lost its `Literal[...]` and accepted any string, or a wired name with no matching constructor parameter disappeared from the config.

```python
class SWEAgent(AgentBase):
    __wiring__ = {"loop": ("loop", ["react", "reflexion"])}  # "reflexion" is not registered

    def __init__(self, *, loop: str = "react", max_steps: int = 10): ...


try:
    build_layer_config(Agent)
except WiringResolutionError as exc:
    print(exc)
    for problem in exc.problems:
        print(problem.kind, problem.owner, problem.param_name, problem.registry_name)
```

```
SWEAgent (key 'swe') wiring 'loop': unknown key 'reflexion' in registry 'loop'; registered: ['codeact', 'react']
unknown_key SWEAgent loop loop
```

Build time reports what needs the registries: unknown required and optional keys, unknown registries (with a "did you mean"), empty registries, and defaults that are not allowed keys. A *receptor* is a wired field; a constructor parameter or non-`ClassVar` class annotation provides its runtime slot. Its own default (`loop: str = "react"`) must be one of the allowed keys too. When a registry is empty, the message also lists the modules `discover()` skipped or failed to import (through `allow_missing`, `strict=False`, or a failure your own code caught), since a missing import is the usual cause.

Grammar mistakes fail earlier, when the class statement runs, and nothing is registered: a value of the wrong shape (`("loop", "react")` instead of `("loop", ["react"])`), a key that is not a string, an empty key list, a key listed as both required and optional, a misspelled dict-form field, an entry name that is not a Python identifier (or is a keyword or a dunder name), and a `__wiring__` that is not a dict.

**Fix:** correct the declarations the error lists. To catch all of them in one test instead of one build at a time, call `validate_wiring()`:

```python
from conscribe import validate_wiring


def test_wiring_is_valid():
    validate_wiring({"agent": Agent, "loop": Loop})  # raises listing every problem
```

If you parse error messages, read `WiringResolutionError.problems` instead. Each `WiringProblem` has `owner`, `owner_key`, `param_name`, `registry_name`, `detail` and a stable `kind`: `malformed`, `not_a_dict`, `non_str_key` or `empty_keys` (grammar), `unknown_registry`, `empty_registry`, `unknown_key`, `unknown_optional_key`, `bad_default` or `contradiction` (a `declare_only` entry that has a receptor). The full set is `conscribe.exceptions.WIRING_PROBLEM_KINDS`. The old `cls_name` / `param_name` / `registry_name` / `detail` attributes still exist and describe the first problem.

### Test fakes

Builds now resolve the wiring of *every* registered class, so a fake that one test registered and never removed fails every later build whose registries do not contain the keys it wires to. Scope fakes to the test that needs them:

```python
from conscribe.testing import isolated_registries


def test_agent_config_with_a_fake_loop():
    with isolated_registries():
        class FakeLoop(LoopBase):
            __registry_key__ = "fake"

        class FakeAgent(AgentBase):
            __wiring__ = {"loop": ("loop", ["fake"])}

            def __init__(self, *, loop: str = "fake"): ...

        build_layer_config(Agent)  # sees both fakes
    # here both fakes are gone again: Loop and Agent hold what they held before
```

With pytest, load the plugin in the root `conftest.py` and request the `conscribe_isolated` fixture. Fakes defined at module level register when pytest collects the module and, left alone, stay registered for the whole session. Capture them once and re-apply them in a fixture with `activate()`:

```python
# conftest.py (the root one)
pytest_plugins = ["conscribe.testing.pytest_plugin"]
```

```python
import pytest
from conscribe import build_layer_config
from conscribe.testing import isolated_registries
from my_layers import Agent, AgentBase, LoopBase

# Module-level fakes: captured once, re-applied for each test that wants them.
FAKES = isolated_registries()
with FAKES:
    class FakeLoop(LoopBase):
        __registry_key__ = "fake"

@pytest.fixture
def fake_loop():
    with FAKES.activate():
        yield

def test_fake_agent(conscribe_isolated, fake_loop):
    class FakeAgent(AgentBase):  # registered for this test only
        __wiring__ = {"loop": ("loop", ["fake"])}
        def __init__(self, *, loop: str = "fake"): ...

    build_layer_config(Agent)  # sees FakeAgent and FakeLoop
```

Use `pytest_plugins`, not `-p conscribe.testing.pytest_plugin`: `-p` imports conscribe before pytest-cov starts measuring. The [testing guide](testing.md) covers the exit rules, nesting and the `kept` / `rolled_back` reports.

Running a class body twice (a helper function that defines a class, a re-executed notebook cell) still raises `DuplicateKeyError`; its message now points at `isolated_registries()`.

## 2. `discover()` stops at the first broken module

`discover()` is strict by default. Import failures stop the call. On Python 3.11+ the original exception carries a note naming the module; on 3.10 a failing submodule is re-raised as an `ImportError` that names it and chains the original. 1.x logged a warning, skipped the module, and you found out later from an "unknown key" or an empty registry.

```python
discover("shop.plugins", exclude=["shop.plugins.generated"])
```

```
ModuleNotFoundError: No module named 'torch'
(raised while conscribe.discover() was importing module 'shop.plugins.vision')
```

**Fix:** decide what each failure means.

```python
imported = discover(
    "shop.plugins",
    exclude=["shop.plugins.generated"],  # never import this package
    allow_missing=["torch"],             # skip modules that fail because torch is missing
)
print(imported)
```

```
['shop.plugins', 'shop.plugins.search']
```

- `allow_missing` takes top-level module names of optional dependencies (here `torch`, not the plugin module). A module whose import fails with a `ModuleNotFoundError` for one of them is skipped without a warning, since you asked for it; it is logged at info level and still listed by `import_failures()`. The `ModuleNotFoundError` may be wrapped in another `ImportError` (`raise ImportError("install the extra") from exc`), but an error of a different type raised while handling it still propagates.
- `exclude` takes dotted module or package names that are never imported (generated code, scripts, test helpers). `"pkg.generated"` excludes `pkg.generated` and everything below it, not `pkg.generated_old`.
- `strict=False` brings back the 1.x skip for `ImportError` and `SyntaxError`, with an `ExtractionWarning` per skipped module instead of a log line. Other exceptions still propagate, as they did in 1.x.

Registrations made by a module whose import failed are rolled back, so a half-imported module no longer leaves classes behind. Every failure is recorded: `conscribe.discover.import_failures()` returns them, and wiring and empty-layer errors cite them. `discover()` also no longer imports `__main__` submodules (`pkg/__main__.py`); name one explicitly if you want it imported.

The CLI forwards the same options, for example `conscribe inspect --registrar shop:Tool --discover shop.plugins --allow-missing torch --exclude shop.plugins.generated` (both options are repeatable).

**Removed:** `discover(auto_update_stubs=..., stub_dir=..., fingerprint_path=...)`. Passing them is now a `TypeError`. The feature never ran in 1.x, because the list of registrars it checked was always empty. If you want "regenerate when stale", write it explicitly; see [section 10](#10-regenerate-generated-files).

## 3. Composed configs refuse unknown top-level sections

The top-level model of `build_composed_config()` now forbids keys that are not layer names, and the JSON Schema says `additionalProperties: false`. 1.x silently dropped a misspelled section.

```python
result = build_composed_config({"agent": Agent, "loop": Loop})
ComposedConfig = result.top_level_type

try:
    ComposedConfig.model_validate({"agents": [{"name": "swe"}]})  # typo: "agents"
except ValidationError as exc:
    print(exc)
```

```
1 validation error for ComposedConfig
agents
  Extra inputs are not permitted [type=extra_forbidden, input_value=[{'name': 'swe'}], input_type=list]
    For further information visit https://errors.pydantic.dev/2.12/v/extra_forbidden
```

**Fix:** correct the config. If your config files carry other top-level sections on purpose (settings your application reads itself), pass `top_level_extra="ignore"` (drop them) or `"allow"` (keep them in `model_extra`); the CLI option is `--top-level-extra`.

```python
lenient = build_composed_config({"agent": Agent, "loop": Loop}, top_level_extra="ignore")
```

With `"ignore"` or `"allow"` the JSON Schema no longer says `additionalProperties: false`, so editors accept those sections too.

The keys of the `registrars` mapping become fields of the top-level model, so they must now be valid identifiers that are not private (`_x`) and do not shadow a `BaseModel` attribute (`schema`, `json`, `copy`, `model_config`, anything starting with `model_`). A bad name raises `ConfigDeclarationError` with a suggested replacement.

## 4. Annotations, `Field` metadata and defaults are read faithfully

2.0 reads a constructor the way Python and pydantic do. Several 1.x approximations are gone, and some of them change what a config accepts:

| What you wrote | 1.x | 2.0 |
|---|---|---|
| An annotation using a name the module never imports (`Optional[int]` without `from typing import Optional`, under `from __future__ import annotations`) | resolved from a built-in fallback namespace | `ConfigAnnotationError`: import the name |
| A name imported only under `if TYPE_CHECKING:` | an unresolved forward reference | the field type becomes `Any`, with a `DegradationWarning` |
| `Annotated[str, Field(alias="apiKey")]` | alias dropped, config key `api_key` | config key `apiKey`, as in pydantic |
| `Field(title=..., examples=..., json_schema_extra=..., discriminator=...)` | dropped | carried into the model and the generated files |
| `Annotated[int, Field(default=5)] = 3` | the signature default won, the other was ignored | `ConfigDeclarationError`: declare the default once |
| `Field(exclude=True)` on a constructor parameter | ignored | `ConfigDeclarationError`: remove it |
| `Field(deprecated=...)` | dropped | a `deprecated` marker in the JSON Schema only |
| `x: int = None` | `None` rejected on validation | the field admits `None` |
| `__config_schema__` validators and `model_config` | dropped | applied |
| `dataclasses.field(default_factory=...)` | rendered as `= ...  # default: <factory>` | the real default |

The annotation error says exactly what to do:

```
WebSearch.region: cannot evaluate annotation 'Optional[str]': name 'Optional' is not defined (declared on app.search.WebSearch): not a global of module app.search, not a class attribute, not a builtin and not imported under 'if TYPE_CHECKING:'. Fix the spelling, or import it at module level in app.search.
```

Annotations now resolve in the module of the class that declares them. A field inherited from a base class in another module (under `from __future__ import annotations`) resolves against the base module's imports; 1.x used the subclass module's imports and could drop the field.

The alias change is the one most likely to break existing YAML files:

```python
class Search(ToolBase):
    def __init__(self, *, api_key: Annotated[str, Field(alias="apiKey")]): ...


adapter = TypeAdapter(build_layer_config(Tool).union_type)
print(adapter.validate_python({"name": "search", "apiKey": "k"}))
try:
    adapter.validate_python({"name": "search", "api_key": "k"})  # accepted by 1.x
except ValidationError as exc:
    print(sorted(err["type"] for err in exc.errors()))
```

```
api_key='k' name='search'
['extra_forbidden', 'missing']
```

**Fix:** rename the keys in your configs, or drop the alias. If a `__config_schema__` model has validators, configs that violate them now fail validation, so check your existing configs against them.

Validators defined on a *registered* pydantic model (an implementation class that is itself a `BaseModel`) are still not part of its config model; they run when your framework instantiates the class. Use `__config_schema__` if you need them at config-load time.

A few other declarations that 1.x accepted now raise `ConfigDeclarationError`: a constructor parameter named `model_config` or another `model_*` member of `BaseModel`, a required positional-only parameter (a config cannot pass it), an invalid `__config_mro_scope__` / `__config_mro_depth__`, a field named like the layer's discriminator, and a `Literal[...]` in a wired annotation that contains values that are not wiring keys.

## 5. Single-parameter models are nested blocks

1.x flattened a class whose `__init__` took exactly one parameter typed with a pydantic model: the model's fields became the config fields. 2.0 treats that parameter like any other, as a nested block.

```python
class OpenAISettings(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.0


class OpenAI(LLMBase):
    __registry_key__ = "openai"

    def __init__(self, settings: OpenAISettings):
        self.settings = settings
```

Throughout 2.x, configs still written in the old flat shape fail with an error that tells you both fixes:

```python
adapter = TypeAdapter(build_layer_config(LLM).union_type)
try:
    adapter.validate_python({"provider": "openai", "model": "gpt-4o-mini"})
except ValidationError as exc:
    print(exc.errors()[0]["type"])
    print(exc.errors()[0]["msg"])
```

```
legacy_flat_config
'settings' is a nested block in conscribe 2.0; write {settings: {...}} or set __config_from_param__ = 'settings'
```

**Fix:** nest the keys in your configs (`settings: {model: gpt-4o-mini}`), or keep the flat shape by opting in on the class (the attribute is inherited; `None` opts out again):

```python
class FlatOpenAI(LLMBase):
    __registry_key__ = "openai_flat"
    __config_from_param__ = "settings"

    def __init__(self, settings: OpenAISettings):
        self.settings = settings
```

The flattened config subclasses `OpenAISettings`, so its validators and `model_config` apply. Every other constructor parameter of the class must be a wiring receptor.

## 6. `WiredField`: class access and dataclass receptors

Reading a wired name on a class used to return the `WiredField` placeholder. In 2.0 it raises `AttributeError` when the class (or a class between it and the one holding the placeholder) annotates the name, just like any annotation-only attribute. This lets `@dataclass`, attrs and pydantic treat the name as a required field instead of taking the placeholder as its default.

```python
class AgentBase(metaclass=Agent.Meta):
    __abstract__ = True
    __wiring__ = {"env": "env"}   # AgentBase gets the placeholder for env


class CodingAgent(AgentBase):
    env: EnvProtocol  # the subclass annotates the wired name


try:
    CodingAgent.env  # 1.x returned the WiredField here
except AttributeError as exc:
    print(exc)

print(wired_field(CodingAgent, "env"))
print(wired_fields(CodingAgent))
```

```
CodingAgent.env has no class-level value: CodingAgent annotates the wired attribute 'env' without a default, so every instance gets its own value (wired from registry 'env'). Use conscribe.wired_field(CodingAgent, 'env') to look up the injection placeholder, or inspect.getattr_static(CodingAgent, 'env') to read the raw class attribute.
WiredField('env', registry='env')
{'env': WiredField('env', registry='env')}
```

**Fix:** replace `Cls.attr` / `getattr(Cls, name)` / `hasattr(Cls, name)` introspection with `conscribe.wired_field(cls, name)` (the placeholder or `None`) and `conscribe.wired_fields(cls)`. They never trigger the descriptor and skip constructor parameters, excluded entries and `declare_only` entries. A class that annotates a wired name gets no placeholder of its own; `wired_field()` reports the one it inherits, as here.

Related changes:

- A dataclass (or attrs class) that annotates a wired name without a default now gets a required field. 1.x silently made the placeholder its default.

  ```python
  @dataclasses.dataclass
  class DataAgent(AgentBase):
      env: EnvProtocol
      max_steps: int = 10


  print(inspect.signature(DataAgent))
  ```

  ```
  (env: __main__.EnvProtocol, max_steps: int = 10) -> None
  ```

- In tests, inject fakes on the instance. To patch an annotated name on the class, tell the patcher that the attribute may not exist yet:

  ```python
  agent = CodingAgent()
  agent.env = FakeEnv()  # preferred: inject on the instance

  monkeypatch.setattr(CodingAgent, "env", FakeEnv(), raising=False)  # class-level patch of an annotated name
  ```

  With `unittest.mock`, use `mock.patch.object(CodingAgent, "env", fake, create=True)`.

- A subclass that drops an inherited entry (`{"env": None}`) or redeclares it `declare_only` now hides the inherited placeholder: class access raises `AttributeError`, and instance access raises Python's plain "object has no attribute" error.

## 7. Mark negotiation-only wiring entries `declare_only`

Some `__wiring__` entries are never injected into the instance: the framework only reads them while it negotiates how to assemble a run (which observation providers are required, which are optional). In 1.x such an entry still got a `WiredField` placeholder and a stub attribute. Declare it with the dict form and `"declare_only": True`:

```python
class ShellAgent(AgentBase):
    __wiring__ = {
        # the framework reads this entry when it assembles a run; the agent
        # never receives an observation object for it
        "obs": {
            "registry": "observation",
            "keys": ["terminal"],
            "optional_keys": ["filesystem"],
            "declare_only": True,
        },
    }


print(wired_fields(ShellAgent))                       # {} -- no placeholder
print(wiring_slots(ShellAgent)["obs"].declare_only)   # True
```

A `declare_only` entry gets no placeholder, never appears in a stub, and is never inlined in a composed config: its config field stays `Optional[Literal[...]] = None` (`Optional[Literal['terminal', 'filesystem']]` here). A `declare_only` entry on a class that does have a constructor parameter or class annotation of that name is a contradiction and raises.

Entries *without* the flag keep their 1.x meaning: a slot-less entry is an attribute the framework injects.

## 8. Declare inline-only layers on the registrar

If your framework refused some layers at the top level of a composed config (their instances belong inside another layer's receptors), replace that code with the new registrar flag:

```python
Hook = create_registrar("hook", P, discriminator_field="name")
Archiver = create_registrar("archiver", P, discriminator_field="name", inline_only=True)


class TraceWriter(HookBase):
    __registry_key__ = "trace_writer"
    __wiring__ = {"archive": "archiver"}

    def __init__(self, *, archive: object): ...


result = build_composed_config({"hook": Hook, "archiver": Archiver})
print(result.top_level_layers)                       # ['hook']
print([str(loc) for loc in result.inline_only_layers["archiver"]])

try:
    result.top_level_type.model_validate({"archiver": [{"name": "disk"}]})
except ValidationError as exc:
    print(exc.errors()[0]["type"])
    print(exc.errors()[0]["msg"])
```

```
['hook']
['hook[trace_writer].archive']
inline_only_layer
layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive
```

conscribe does not read class attributes such as `__inline_only__`; the flag lives on the registrar (`create_registrar(..., inline_only=True)`, readable as `Registrar.inline_only`). An inline-only layer that no receptor can hold raises `ConfigDeclarationError`, and so does any inline-only layer in a build with `inline_wiring=False` (key strings instead of inline configs), since its instances could then be written nowhere.

Code that listed sections by walking `top_level_type.model_fields` should use `result.top_level_layers`. An inline-only layer still has a field in `model_fields`: a hidden one, left out of the JSON Schema and of dumps, whose only job is to refuse values written there.

## 9. Stubs: only injected attributes, typed by protocol

`generate-stubs` / `write_layer_stubs()` now write a stub only for wired attributes that your source does not declare already: no `__init__` parameter (inherited and `**kwargs`-forwarded ones included), no class annotation, not `declare_only`. Those attributes are typed as the target registry's protocol. 1.x narrowed them to the most specific common base of the allowed implementations, which drifted as implementations were registered.

Two consequences:

- Modules whose wired names all have a parameter or an annotation get no stub any more. The 1.x stub stays on disk and keeps hiding the module from your type checker. conscribe reports such files with a `StaleStubWarning` (a `Warning:` line in the CLI) and deletes exactly those files when you pass `--prune` / `prune=True`:

  ```
  $ conscribe generate-stubs --registrar my_app:Agent --discover my_app.agents
    my_app/agents/coder.pyi
  Generated 1 stub file(s).
  Warning: 1 stale conscribe-generated stub file(s) next to the modules of layer 'agent' -- not written by this run and needed by no registered class, but they still replace their modules for type checkers: my_app/agents/planner.pyi. Delete them, or pass prune=True (CLI: generate-stubs --prune) to delete exactly these files.
  ```

  (Paths shortened.) conscribe looks for such files in the directories this run writes to (the `--output-dir` tree, or next to the source modules), and only files whose header says they were generated by conscribe, 1.x headers included, are ever deleted.

- If you relied on the narrowed type, annotate the attribute in the class body instead (`env: Terminal`). An annotation is a runtime slot: the config field uses the slot or wiring default, is optional for an annotation admitting `None` without a value, and is otherwise required. With a class-level `None` default (`env: Terminal | None = None`), reading the attribute before injection returns `None` instead of raising.

The writer also refuses to write next to sources inside site-packages or the standard library (pass an output directory), raises on wiring problems before it writes any file, and never writes an invalid stub. If you use the collector API directly: `ClassStubInfo.init_signature` became `init` (a `MethodStub` or `None`), `MethodStub.signature` became `function` (plus `kind`), `ClassAttrStub` now holds the annotation object with `owner` and `has_default`, and `render_signature` is gone.

## 10. Regenerate generated files

Every generated file changes once: config sources start with `# Generated by conscribe 2.0.0`, a required field renders as `name: T` or `name: T = Field(...)` when it has metadata (never `= ...`), a default that comes with other settings as `Field(default=..., ...)`, user types are imported instead of redefined, every constraint is written out, and composed sources keep their discriminators and the inline-only refusal. Regenerate and commit the results.

Stored fingerprints all change too. `compute_registry_fingerprint()` now hashes the conscribe version, layer settings and the layer's generated config source and JSON Schema. Stubs are not part of it: regenerate them separately. For composed configs, track every participating layer and the composition options (`inline_wiring` and `top_level_extra`). The fingerprint never raises: a broken layer gets a stable error fingerprint, which is still an opaque hash; generation itself reports the error.

`discover(stub_dir=...)` is gone; if you want regeneration on demand, a few lines do it:

```python
from pathlib import Path

from conscribe import (
    build_layer_config,
    compute_registry_fingerprint,
    generate_layer_config_source,
    load_cached_fingerprint,
    save_fingerprint,
)


def regenerate_if_stale(registrar, layer: str, out_dir: Path) -> bool:
    cache = out_dir / ".fingerprints.json"
    current = compute_registry_fingerprint(registrar)
    if load_cached_fingerprint(cache, layer) == current:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    source = generate_layer_config_source(build_layer_config(registrar))
    (out_dir / f"{layer}_config.py").write_text(source, encoding="utf-8")
    save_fingerprint(cache, layer, current)
    return True


print(regenerate_if_stale(Agent, "agent", Path("generated")))  # True: first run
print(regenerate_if_stale(Agent, "agent", Path("generated")))  # False: nothing changed
```

If you hash the composed top-level model yourself (for example over `top_level_type.model_fields`), expect that hash to change as well: inline-only layers add hidden fields and the top level now forbids extra keys.

## 11. Smaller changes

- Registry keys are validated. A key must be a non-empty `str` (`__registry_key__` may also be a non-empty list of them). Enum members, tuples, ints and empty values raise `InvalidRegistryKeyError` when the class is created:

  ```
  ReactLoop: invalid registry key <LoopKind.REACT: 'react'> from __registry_key__: a registry key must be a plain str, not the Enum member <LoopKind.REACT: 'react'> (str() of it is 'LoopKind.REACT', so the class would be registered under a key nobody looks up); write the plain string 'react'
  ```

- Registration is atomic. A class statement that fails (a second key that collides, the second registry of `A.Meta | B.Meta`, a malformed `__wiring__`) leaves nothing registered.
- Empty layers raise `EmptyLayerError` in every builder and in the stub writer, naming the modules `discover()` failed to import. 1.x failed with `TypeError: Cannot take a Union of no types`.
- Nested-mode layers (`discriminator_fields`) refuse what they cannot honour yet: `__wiring__`, `__config_schema__`, `__config_from_param__`, `__config_mro_scope__` and `__config_mro_depth__` on their classes, keys whose segment count does not match the levels, and fields named like a discriminator. 1.x ignored or dropped these. Parameters are now assigned to levels by the leaf class's real constructor.
- In a class with a `__config_schema__` (Tier 3), every wired entry that has a runtime slot must be a field of the schema. In composed builds, those wired fields are now inlined like any other receptor, so they take nested configs instead of key strings.
- A wired parameter whose default is a key string (`loop: str = "react"`) becomes that key's default config instance in inline composed builds. In 1.x the key string stayed the default of a field typed as config objects, so an omitted block came out as the string `'react'`. The target config must be buildable from defaults.
- `ResolvedWiring.injected` is removed (it was never set), which makes `optional_keys` the fourth positional field. Pass it by keyword.
- `parse_wiring()` raises `WiringResolutionError` for a malformed declaration (1.x raised `TypeError`).
- Generated model names are sanitized identifiers, unique per build (a numeric suffix tells equal names apart).
- The `generate-config`, `generate-composed-config`, `generate-stubs` and `inspect` CLI commands load layers with `--registrar module:attribute` and `--discover package`; `--layer` / `--layers` still select layers by registry name. In 1.x these commands could not find any registrar outside tests. The `--config` batch-file option that the 1.x docs described never existed and still does not.

## Checklist for frameworks built on conscribe

A framework that defines its layers with conscribe usually needs these steps:

1. Require `conscribe>=2,<3`, Python 3.10+ and pydantic 2.7+.
2. Add `error::conscribe.ConscribeWarning` to your test warning filters.
3. Load `conscribe.testing.pytest_plugin` from the root `conftest.py`. Move test fakes into `isolated_registries()` blocks or the `conscribe_isolated` fixture; capture module-level fakes and `activate()` them per test.
4. Add a test that calls `validate_wiring()` on all your registrars.
5. Give `discover()` an `exclude` list (generated packages, scripts) and an `allow_missing` list (optional dependencies). Do the same for `--exclude` / `--allow-missing` in CLI calls.
6. Declare inline-only layers with `create_registrar(..., inline_only=True)` and delete your own top-level refusal code.
7. Mark negotiation-only wiring entries `"declare_only": True`.
8. Add `__config_from_param__` to classes whose single model parameter must stay flat, or migrate their configs to the nested shape.
9. Replace `Cls.attr` placeholder introspection with `conscribe.wired_field()` / `wired_fields()`. To route config values, use `conscribe.introspect` (`init_param_names`, `receptor_slots`, `wiring_slots`) instead of private helpers.
10. Use `result.top_level_layers` instead of walking the top-level model's fields.
11. Regenerate config sources, JSON Schemas and stubs (`generate-stubs --prune`), refresh stored fingerprints, and commit the new files.
