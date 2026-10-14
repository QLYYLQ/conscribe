# Testing with conscribe

Registration is a side effect of defining a class, and it lands in registries that live for the whole process. A fake class defined in one test is still registered in the next one, and defining it again raises `DuplicateKeyError`. conscribe 2.0 handles this with `isolated_registries()` and an opt-in pytest plugin, both in `conscribe.testing` (which does not import pytest). The [API Reference](api-reference.md#testing-conscribetesting) lists the signatures.

## `isolated_registries()`

`isolated_registries()` returns a context manager. On exit, every registry and the global registry index go back to the state they had on entry, apart from what the [keep rule](#the-keep-rule) preserves:

```python
from conscribe.testing import isolated_registries
from testing_app import Action, ActionBase   # registers Click under "click"

with isolated_registries() as iso:
    class FakeClick(ActionBase):
        __registry_key__ = "fake_click"

    Action.unregister("click")
    print("inside:", Action.keys())

print("after: ", Action.keys())
print("rolled_back:", iso.rolled_back)
print("kept:", iso.kept)
```

```
inside: ['fake_click']
after:  ['click']
rolled_back: [('action', 'fake_click', <class '__main__.FakeClick'>)]
kept: []
```

Inside the block registration works normally, so wiring resolution, config builds and stub generation all see the fakes. The block covers every registry at once; there is no per-registry form.

The block undoes:

- classes registered by the block's own code: test bodies, helper functions, `exec()` in a test;
- `unregister()` / `LayerRegistry.remove()` calls (the removed classes come back);
- registrars the block's own code created (a registry name they took over gets its original registry back);
- partial registrations left by an import that raised;
- the attributes `register()` wrote on classes whose registration is rolled back;
- failures that `discover()` recorded inside the block (the [failure record](registration.md#failure-record) goes back to its state at entry).

### The keep rule

Registrations (and registrars) made by the top-level code of a module that was imported for the first time inside the block, and is still in `sys.modules`, survive the block. Such a module never runs again (a later `import` finds it in `sys.modules`), so if its registrations were rolled back they would be gone for the rest of the process. A module whose import was already under way when the block started does not count: a test module that runs `with FAKES:` at its top level is in `sys.modules` before that line runs, so its fakes are rolled back.

```python
import warnings

from conscribe import IsolationWarning
from conscribe.testing import isolated_registries
from testing_app import Action

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", IsolationWarning)
    with isolated_registries(warn_kept=True) as iso:
        import testing_app.scroll  # first import of this module happens here

print(Action.keys())
print(iso.kept)
print(caught[0].message)
```

```
['click', 'scroll']
[('action', 'scroll', <class 'testing_app.scroll.Scroll'>)]
isolated_registries() kept 1 registration(s) made by modules imported for the first time inside the block: testing_app.scroll: action:scroll (Scroll). Those modules stay imported and never run again, so their registrations stay for the rest of the process. If they are test fakes, import the module before the block, or capture its fakes at module level (FAKES = isolated_registries(); with FAKES: ...) and use FAKES.activate() in tests.
```

How the keep rule settles conflicts and repeated writes:

- A key that existed at entry always gets its entry class back.
- When a newly imported module writes the same key several times, its last write wins. A module does that without a `DuplicateKeyError` when a class is re-created (a `@dataclass(slots=True)` copy takes over the key) or when it removes and re-adds a key; the final class is kept.
- A key that such a module registered and then removed stays removed.
- Registries created by such modules are kept in the global index, even over a registry of the same name that existed when the block started.

A registration counts as import-time only when the new module's top-level code is on the call stack of the thread that registers. Registrations made from other threads are rolled back like everything else.

### `kept` and `rolled_back`

After the block exits, the `RegistryIsolation` object lists what happened (and its `active` property tells whether the block is still open). Both lists hold `(registry_name, key, cls)` tuples:

| Attribute | Contents |
|-----------|----------|
| `kept` | Registrations the keep rule preserved |
| `rolled_back` | Bindings present when the block ended that the exit removed or replaced. Keys the block removed and the exit put back are not listed. |

### `warn_kept` and `IsolationWarning`

`isolated_registries(warn_kept=True)` emits an `IsolationWarning` on exit naming every module whose registrations were kept (see the example above). Use it to catch a fake-defining helper module that a test imports lazily: its fakes leak into every later test, and which tests see them depends on which test imported the module first. Importing the module before any block (for example in `conftest.py`) makes its registrations part of every test's starting state; capturing its fakes (see below) registers them only where a test asks for them.

### Exceptions, nesting and threads

- An exception raised inside the block (`BaseException` included, e.g. `KeyboardInterrupt`) propagates after the state is restored.
- Blocks nest, and each block restores the state of its own entry:

```python
from conscribe.testing import isolated_registries
from testing_app import Action, ActionBase

with isolated_registries():
    class Outer(ActionBase):
        __registry_key__ = "outer"

    with isolated_registries():
        class Inner(ActionBase):
            __registry_key__ = "inner"
        print("inner block:", Action.keys())

    print("outer block:", Action.keys())

try:
    with isolated_registries():
        class Doomed(ActionBase):
            __registry_key__ = "doomed"
        raise KeyboardInterrupt
except KeyboardInterrupt:
    pass

print("afterwards: ", Action.keys())
```

```
inner block: ['click', 'outer', 'inner']
outer block: ['click', 'outer']
afterwards:  ['click']
```

- Blocks must exit in LIFO order, which nested `with` statements guarantee. Only manual `__enter__` / `__exit__` calls or interleaved tasks can break it: if an outer block exits while inner ones are still open, it closes them first, restores its own entry state and raises `RuntimeError`, and each inner block it closed raises `RuntimeError` from its own exit.
- A block is not re-entrant while it is open. Create a new one to nest.
- Blocks are process-global, so overlapping blocks in concurrent threads or asyncio tasks are not supported.

Only the attribute writes made by `register()` are undone. Other changes your code makes to classes are not. A class defined inside a block stays alive as long as something references it (for example `Base.__subclasses__()`, until garbage collection).

## Module-level fakes: capture and `activate()`

A fake defined at module level in a test file registers when pytest imports the file at collection time, outside any function-scoped fixture. Capture such fakes instead of leaving them registered:

```python
from conscribe.testing import isolated_registries
from testing_app import Action, ActionBase

# At module level: define the fakes once, record them, roll them back.
FAKES = isolated_registries()
with FAKES:
    class FakeClick(ActionBase):
        __registry_key__ = "fake_click"

    Action.unregister("click")  # the capture records removals too

print("after capture:", Action.keys())

with FAKES.activate():  # what a fixture does around each test
    print("activated:    ", Action.keys())

print("after test:   ", Action.keys())
```

```
after capture: ['click']
activated:     ['fake_click']
after test:    ['click']
```

The `with FAKES:` block records the net difference it made: keys it added, keys it removed, keys it rebound to another class, registrars it created, and the attribute writes of `register()`. Then it rolls everything back like any isolation block.

`FAKES.activate()` returns a new isolation block. On enter it re-applies the full captured difference; on exit (or when re-applying fails part-way) it restores the state from before the enter. Activation raises:

- `RuntimeError` if nothing was captured yet, or while the capture's own block is open;
- `DuplicateKeyError` if a key the capture touches is now bound to a class that is neither the captured class nor the class the capture replaced. Unregister that key first, or capture under another key.

Entering `FAKES` itself again re-applies what it captured so far and adds the new block's changes to the capture. Tests should use `activate()`, which never records.

## The pytest plugin

conscribe ships an opt-in pytest plugin with one fixture, `conscribe_isolated`. Enable it in the root `conftest.py` of your test suite:

```python
# conftest.py (the root one)
pytest_plugins = ["conscribe.testing.pytest_plugin"]
```

pytest only accepts `pytest_plugins` in the root conftest. Do not enable the plugin with `-p conscribe.testing.pytest_plugin`: `-p` imports the plugin, and conscribe with it, before pytest-cov starts measuring, so coverage of conscribe (and of code imported with it) comes out incomplete. There is deliberately no `pytest11` entry point, so the fixture only exists in suites that opt in.

A test that requests `conscribe_isolated` runs inside `isolated_registries()`. The fixture value is the `RegistryIsolation` object; its `kept` and `rolled_back` lists are filled after the test, so code after the `yield` of a fixture that wraps it can read them.

To isolate every test, wrap the fixture in an autouse fixture of your own, in the root `conftest.py`:

```python
import pytest

@pytest.fixture(autouse=True)
def _isolated(conscribe_isolated):
    yield conscribe_isolated
```

The ini option `conscribe_warn_kept` (default `false`) turns on `warn_kept` for the fixture:

```ini
# pytest.ini
[pytest]
conscribe_warn_kept = true
filterwarnings =
    error::conscribe.ConscribeWarning
```

With both settings, a test that keeps registrations (it imported a registering module for the first time) fails with an error at teardown that names the module.

A test module that uses the fixture, a captured fake and the patterns from the next sections:

```python
import pytest

from conscribe.testing import isolated_registries
from testing_app import Action, ActionBase, Clicker, TypedClicker

# Module-level fakes: defined once at import, recorded, rolled back at once.
FAKES = isolated_registries()
with FAKES:
    class FakeClick(ActionBase):
        __registry_key__ = "fake_click"


@pytest.fixture
def fakes():
    with FAKES.activate():
        yield


def test_fake_is_registered(fakes):
    assert Action.get("fake_click") is FakeClick


def test_fake_is_gone_without_the_fixture():
    assert "fake_click" not in Action.keys()


def test_define_in_a_test(conscribe_isolated):
    class Temp(ActionBase):
        __registry_key__ = "temp"

    assert Action.get("temp") is Temp


def test_same_key_in_the_next_test(conscribe_isolated):
    class Temp(ActionBase):  # no DuplicateKeyError: the last test rolled back
        __registry_key__ = "temp"

    assert conscribe_isolated.active


def test_inject_on_the_instance():
    agent = Clicker()
    agent.action = FakeClick()
    assert agent.run() == "FakeClick"


def test_patch_the_class(monkeypatch):
    monkeypatch.setattr(Clicker, "action", FakeClick())
    assert Clicker().run() == "FakeClick"


def test_patch_an_annotated_name(monkeypatch):
    monkeypatch.setattr(TypedClicker, "action", FakeClick(), raising=False)
    assert TypedClicker().run() == "FakeClick"
```

```
.......                                                                  [100%]
7 passed in 0.00s
```

The `testing_app` package used on this page:

```python
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class ActionProtocol(Protocol):
    def act(self) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    def run(self) -> str: ...


Action = create_registrar("action", ActionProtocol, discriminator_field="kind")
Agent = create_registrar("agent", AgentProtocol, discriminator_field="kind")


class ActionBase(metaclass=Action.Meta):
    __abstract__ = True

    def act(self) -> str:
        return type(self).__name__


class Click(ActionBase):
    __registry_key__ = "click"


class AgentBase(metaclass=Agent.Meta):
    __abstract__ = True
    __wiring__ = {"action": "action"}  # injected by the framework

    def run(self) -> str:
        return self.action.act()


class Clicker(AgentBase):
    __registry_key__ = "clicker"


class TypedClicker(AgentBase):
    __registry_key__ = "typed_clicker"
    action: ActionProtocol  # the class annotates the wired name
```

`testing_app/scroll.py` holds one more action, `Scroll` (key `"scroll"`).

## `unregister(missing_ok=True)`

Prefer an isolation block over manual cleanup: it also restores what a test removed and undoes registrars the test created. When you do clean up by hand, make teardown idempotent:

```python
from testing_app import Action

print(Action.unregister("click"))
print(Action.unregister("click", missing_ok=True))  # None: the key is absent
```

```
<class 'testing_app.Click'>
None
```

`unregister()` (and `LayerRegistry.remove()`) removes exactly one key and returns the class that was bound to it. A class registered under several keys stays registered under the others, and the keys below a hierarchical prefix are separate entries. Without `missing_ok=True`, a missing key raises `KeyNotFoundError` (a `KeyError` subclass) whose message says how to remove what is there.

## `discover()` failures in tests

`discover()` records every module it failed to import, and wiring errors and `EmptyLayerError` cite that record. A test that runs `discover()` on a deliberately broken package can therefore leak its failure into another test's error message. An isolation block restores the record to its state at entry. Outside a block, call `clear_import_failures()`:

```python
from conscribe import discover
from conscribe.discover import clear_import_failures, import_failures

try:
    discover("broken_pkg")      # broken_pkg/bad.py raises RuntimeError("boom")
except (RuntimeError, ImportError):  # submodule failures are wrapped on Python 3.10
    pass

print(import_failures())
clear_import_failures()
print(import_failures())
```

```
{'broken_pkg.bad': 'RuntimeError: boom'}
{}
```

The helpers live in the `conscribe.discover` module. The package attribute `conscribe.discover` is the `discover()` function itself, so import them with `from conscribe.discover import ...`.

## Turning conscribe warnings into errors

Every warning conscribe emits is a `ConscribeWarning` subclass (`ExtractionWarning`, `DegradationWarning`, `StaleStubWarning`, `IsolationWarning`). In a test suite, make them errors with pytest's `filterwarnings` setting (or its `-W error::conscribe.ConscribeWarning` option); pytest resolves the category after conscribe can be imported:

```ini
[pytest]
filterwarnings =
    error::conscribe.ConscribeWarning
```

In a plain script, install the filter in code:

```python
import warnings

from conscribe import ConscribeWarning

warnings.simplefilter("error", ConscribeWarning)
```

Python's startup options cannot name conscribe's own categories: `python -W error::conscribe.ConscribeWarning` and `PYTHONWARNINGS=error::conscribe.ConscribeWarning` resolve them before third-party packages are importable and print `Invalid -W option ignored: invalid module name: 'conscribe'`. Since `ConscribeWarning` subclasses `UserWarning`, both `-W error::UserWarning` and `PYTHONWARNINGS=error::UserWarning` work, escalating every `UserWarning`. The CLI overrides filters for its scan and stale-stub reports; see [Output, errors and exit codes](cli.md#output-errors-and-exit-codes).

## Patching wired attributes

The framework assigns a wired attribute after construction when it is not a constructor parameter and the entry is not `declare_only`. Where conscribe installed a `WiredField` placeholder, reading the attribute before assignment raises an `AttributeError` naming the attribute and its registry. An annotation or existing binding can make a placeholder unnecessary. Class access to a placeholder returns it, except where the class or an intervening base annotates the name: there class access raises, so dataclasses and pydantic do not take it for a default (see [Registration Deep-Dive](registration.md#wiredfield-placeholders)). The test module above shows three ways to supply a fake:

- Inject it on the instance: `agent.action = FakeClick()`. This needs no patching.
- When you cannot reach the instance, patch the class. `monkeypatch.setattr(Clicker, "action", fake)` works when class access returns the placeholder.
- When the class, or a class between it and the placeholder, annotates the wired name (`action: ActionProtocol`), the name has no class-level value and `hasattr(TypedClicker, "action")` is `False`. Pass `raising=False` to `monkeypatch.setattr`, or `create=True` to `unittest.mock.patch.object`.

To look placeholders up, use `conscribe.wired_field(cls, name)` / `conscribe.wired_fields(cls)`. Reading `cls.attr` raises for annotated names.
