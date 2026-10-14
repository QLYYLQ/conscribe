"""Tests for conscribe.stubs.collector: which wired entries a stub declares."""
from __future__ import annotations

import abc
import contextlib
import functools
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.exceptions import WiringResolutionError
from conscribe.registration.registry import LayerRegistry
from conscribe.stubs.collector import (
    ClassMembers,
    ClassStubInfo,
    ClassValueStub,
    InjectedAttr,
    MethodStub,
    collect_class_members,
    collect_class_stub_info,
    narrowest_common_base,
)


# ── Fixtures ─────────────────────────────────────────────────────


@runtime_checkable
class EnvProtocol(Protocol):
    def setup(self) -> None: ...


class Terminal(EnvProtocol):
    def setup(self) -> None: ...


class BashTerminal(Terminal):
    def setup(self) -> None: ...


class ZshTerminal(Terminal):
    def setup(self) -> None: ...


class FileSystem(EnvProtocol):
    def setup(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def generate(self) -> str: ...


@pytest.fixture(autouse=True)
def _isolated(conscribe_isolated):
    """Registries and registrations made by a test are rolled back after it."""
    yield conscribe_isolated


def _make_env_registry() -> LayerRegistry:
    reg = LayerRegistry("test_env", EnvProtocol)
    reg.add("terminal.bash", BashTerminal)
    reg.add("terminal.zsh", ZshTerminal)
    reg.add("filesystem", FileSystem)
    return reg


def _make_llm_registry() -> LayerRegistry:
    reg = LayerRegistry("test_llm", LLMProtocol)
    dummy = type("OpenAI", (), {"generate": lambda self: ""})
    reg.add("openai", dummy)
    return reg


# ── narrowest_common_base (public helper, no longer used by stubs) ──


class TestNarrowestCommonBase:
    def test_empty_list(self):
        assert narrowest_common_base([], str) is str

    def test_single_class(self):
        assert narrowest_common_base([BashTerminal], EnvProtocol) is BashTerminal

    def test_siblings_share_parent(self):
        result = narrowest_common_base([BashTerminal, ZshTerminal], EnvProtocol)
        assert result is Terminal

    def test_different_branches_fallback(self):
        result = narrowest_common_base([BashTerminal, FileSystem], object)
        # Both descend from EnvProtocol
        assert result is EnvProtocol

    def test_parent_and_child(self):
        result = narrowest_common_base([Terminal, BashTerminal], EnvProtocol)
        assert result is Terminal

    def test_identical_classes(self):
        result = narrowest_common_base([BashTerminal, BashTerminal], EnvProtocol)
        assert result is BashTerminal


# ── Selection: which entries are injected attributes ──────────────


class TestSelection:
    def test_no_wiring_returns_none(self):
        class Plain:
            pass

        assert collect_class_stub_info(Plain) is None

    def test_wired_in_init_returns_none(self):
        """Wired field that's already in __init__ → not injected → None."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def __init__(self, env: str = "filesystem"):
                self.env = env

        assert collect_class_stub_info(Agent) is None

    def test_injected_wiring_basic(self):
        """Wired field with no runtime slot → injected → ClassStubInfo."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def __init__(self, name: str = "a"):
                self.name = name

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.class_name == "Agent"
        assert info.injected_attrs == (InjectedAttr("env", EnvProtocol, "test_env"),)

    def test_inherited_init_param_is_a_slot(self):
        """1.3.0 only looked at the class's *own* __init__."""
        _make_env_registry()

        class Parent:
            def __init__(self, env: str = "terminal.bash", other: int = 0) -> None: ...

        class Child(Parent):
            __wiring__ = {"env": "test_env"}

        assert collect_class_stub_info(Child) is None

    def test_kwargs_forwarded_param_is_a_slot(self):
        _make_env_registry()

        class Parent:
            def __init__(self, env: str = "terminal.bash") -> None: ...

        class Child(Parent):
            __wiring__ = {"env": "test_env", "llm_slotless": ["x"]}

            def __init__(self, extra: int = 0, **kwargs) -> None:
                super().__init__(**kwargs)

        info = collect_class_stub_info(Child)
        assert info is not None
        assert [a.name for a in info.injected_attrs] == ["llm_slotless"]

    def test_class_annotation_on_a_mixin_is_a_slot(self):
        """af defect 13: ``_Receptors.strategy: Strategy`` declares the slot."""
        _make_env_registry()

        class _Receptors:
            env: EnvProtocol

        class Agent(_Receptors):
            __wiring__ = {"env": "test_env"}

        assert collect_class_stub_info(Agent) is None

    def test_annotated_attribute_is_a_slot(self):
        """1.3.0 rendered ``env`` again (narrowed); the source declares it."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}
            env: str = "filesystem"

        assert collect_class_stub_info(Agent) is None

    def test_classvar_annotation_is_not_a_slot(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}
            env: ClassVar[str] = ""

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert [a.name for a in info.injected_attrs] == ["env"]
        # The injected name is not repeated as a class attribute.
        assert "env" not in {ca.name for ca in info.class_attrs}

    def test_declare_only_entry_is_never_injected(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {
                "env": {"registry": "test_env", "declare_only": True},
                "shell": {"registry": "test_env", "keys": ["terminal.bash"]},
            }

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert [a.name for a in info.injected_attrs] == ["shell"]

    def test_only_declare_only_entries_returns_none(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": {"registry": "test_env", "declare_only": True}}

        assert collect_class_stub_info(Agent) is None

    def test_mixed_wired_and_non_wired(self):
        """Only injected fields appear, not those in __init__."""
        _make_env_registry()
        _make_llm_registry()

        class Agent:
            __wiring__ = {"env": "test_env", "llm": "test_llm"}

            def __init__(self, env: str = "filesystem"):
                self.env = env

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert [a.name for a in info.injected_attrs] == ["llm"]


# ── Types: always the registry protocol ───────────────────────────


class TestInjectedType:
    def test_mode1_uses_protocol(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is EnvProtocol

    @pytest.mark.parametrize(
        "entry",
        [
            ("test_env", ["terminal.bash", "terminal.zsh"]),
            ("test_env", ["terminal.bash"]),
            ("test_env", ["terminal.bash"], ["terminal.zsh"]),
            {"registry": "test_env", "keys": ["terminal.bash"]},
        ],
    )
    def test_subsets_use_protocol_not_narrowest_base(self, entry):
        """1.3.0 narrowed to ``Terminal`` / ``BashTerminal``: that breaks protocol
        conformance of a mutable attribute and drifts as keys get registered."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": entry}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is EnvProtocol

    def test_mode3_literal_uses_str(self):
        class Agent:
            __wiring__ = {"browser": ["chromium", "firefox"]}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.injected_attrs[0].resolved_type is str
        assert info.injected_attrs[0].registry_name is None


# ── Wiring problems raise (no blanket except) ─────────────────────


class TestWiringProblemsRaise:
    def test_unknown_key(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": ("test_env", ["terminal.fish"])}

        with pytest.raises(WiringResolutionError, match="unknown key 'terminal.fish'"):
            collect_class_stub_info(Agent)

    def test_unknown_registry(self):
        class Agent:
            __wiring__ = {"env": "test_no_such_registry"}

        with pytest.raises(WiringResolutionError, match="unknown registry 'test_no_such_registry'"):
            collect_class_stub_info(Agent)

    def test_empty_registry(self):
        LayerRegistry("test_empty", EnvProtocol)

        class Agent:
            __wiring__ = {"env": "test_empty"}

        with pytest.raises(WiringResolutionError, match="registry 'test_empty' is empty"):
            collect_class_stub_info(Agent)

    def test_declare_only_entry_with_a_slot_is_a_contradiction(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": {"registry": "test_env", "declare_only": True}}

            def __init__(self, env: str = "terminal.bash") -> None: ...

        with pytest.raises(WiringResolutionError) as excinfo:
            collect_class_stub_info(Agent)
        assert [p.kind for p in excinfo.value.problems] == ["contradiction"]

    def test_problem_on_a_slotted_entry_still_raises(self):
        """The stub skips slotted entries, but a typo in one is still a typo."""
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": ("test_env", ["terminal.fsh"]), "shell": "test_env"}

            def __init__(self, env: str = "terminal.bash") -> None: ...

        with pytest.raises(WiringResolutionError) as excinfo:
            collect_class_stub_info(Agent)
        assert [p.param_name for p in excinfo.value.problems] == ["env"]


# ── Members ───────────────────────────────────────────────────────


class TestMembers:
    def test_no_own_init(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.init is None

    def test_own_init_captured(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def __init__(self, name: str = "default"):
                self.name = name

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert info.init == MethodStub("__init__", Agent.__init__)

    def test_own_methods_collected(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

            def step(self, task: str) -> str:
                return task

        info = collect_class_stub_info(Agent)
        assert info is not None
        assert [m.name for m in info.methods] == ["step"]
        assert info.methods[0].function is Agent.__dict__["step"]

    def test_class_attrs_captured(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}
            max_steps: int = 10
            use_vision: bool = False

        info = collect_class_stub_info(Agent)
        assert info is not None
        attrs = {ca.name: ca for ca in info.class_attrs}
        assert set(attrs) == {"max_steps", "use_vision"}
        assert attrs["max_steps"].annotation == "int"  # PEP 563: as declared
        assert attrs["max_steps"].owner is Agent
        assert attrs["max_steps"].has_default is True

    def test_class_attrs_skip_dunders(self):
        _make_env_registry()

        class Agent:
            __wiring__: dict = {"env": "test_env"}
            visible: int = 0

        info = collect_class_stub_info(Agent)
        assert info is not None
        names = {ca.name for ca in info.class_attrs}
        assert "__wiring__" not in names
        assert "visible" in names

    def test_property_setter_deleter_and_cached_property(self):
        class Thing:
            @property
            def size(self) -> int:
                return 1

            @size.setter
            def size(self, value: int) -> None: ...

            @size.deleter
            def size(self) -> None: ...

            @functools.cached_property
            def total(self) -> int:
                return 2

        methods = collect_class_members(Thing).methods
        assert [(m.name, m.decorators) for m in methods] == [
            ("size", ("property",)),
            ("size", ("setter",)),
            ("size", ("deleter",)),
            ("total", ("cached_property",)),
        ]

    def test_static_class_abstract_and_async(self):
        class Thing(abc.ABC):
            @staticmethod
            def make() -> int:
                return 1

            @classmethod
            async def build(cls) -> None: ...

            @abc.abstractmethod
            def run(self) -> None: ...

        by_name = {m.name: m for m in collect_class_members(Thing).methods}
        assert by_name["make"].decorators == ("staticmethod",)
        assert by_name["build"].decorators == ("classmethod",)
        assert by_name["build"].is_async is True
        assert by_name["run"].decorators == ("abstractmethod",)

    def test_generated_and_copied_members_are_skipped(self):
        @dataclass
        class Point:
            x: int = 0

            def norm(self) -> int:
                return self.x

        members = collect_class_members(Point)
        assert isinstance(members, ClassMembers)
        assert members.init is not None and members.init.name == "__init__"
        # ``__repr__`` / ``__eq__`` written by @dataclass are not repeated;
        # the field default ``x = 0`` is the annotated field, not a value.
        assert [m.name for m in members.methods] == ["norm"]
        assert [a.name for a in members.class_attrs] == ["x"]
        assert members.values == ()

    def test_non_function_callables_are_values_and_nested_classes(self):
        class Thing:
            class Inner:
                pass

            handler = functools.partial(print, "x")

        members = collect_class_members(Thing)
        assert members.methods == ()
        assert members.nested == (("Inner", Thing.Inner),)
        assert members.values == (ClassValueStub("handler", Thing.__dict__["handler"], Thing),)

    def test_info_shape(self):
        _make_env_registry()

        class Agent:
            __wiring__ = {"env": "test_env"}

        info = collect_class_stub_info(Agent)
        assert isinstance(info, ClassStubInfo)
        assert info.module == __name__
        assert info.source_file is not None and info.source_file.endswith("test_collector.py")
        assert info.bases == (object,)


# ── Wrapped methods, values, nested classes (review ST2 / ST3) ───


class _ValuesBase:
    INHERITED = 1
    declared: int


def _logged(func: Any) -> Any:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def _alogged(func: Any) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    return wrapper


class TestWrappedMethods:
    def test_contextmanager_methods_keep_the_generator(self):
        class Thing:
            @contextlib.contextmanager
            def session(self) -> Iterator[int]:
                yield 1

            @contextlib.asynccontextmanager
            async def asession(self) -> AsyncIterator[int]:
                yield 1

            @classmethod
            @contextlib.contextmanager
            def scoped(cls) -> Iterator[str]:
                yield ""

        by_name = {m.name: m for m in collect_class_members(Thing).methods}
        session = by_name["session"]
        assert session.kind == "contextmanager"
        assert session.function is Thing.__dict__["session"].__wrapped__
        assert session.is_async is False
        # A plain function returning an async context manager: never ``async def``.
        assert (by_name["asession"].kind, by_name["asession"].is_async) == ("asynccontextmanager", False)
        assert (by_name["scoped"].kind, by_name["scoped"].decorators) == ("contextmanager", ("classmethod",))

    def test_cache_wrappers_keep_the_wrapped_function(self):
        class Thing:
            @functools.cache
            def compute(self) -> int:
                return 1

            @functools.lru_cache(maxsize=4)
            def lookup(self, key: str) -> str:
                return key

            @staticmethod
            @functools.cache
            def shared(x: int) -> int:
                return x

        by_name = {m.name: m for m in collect_class_members(Thing).methods}
        assert by_name["compute"].kind == "function"
        assert by_name["compute"].function is Thing.__dict__["compute"].__wrapped__
        assert by_name["lookup"].function is Thing.__dict__["lookup"].__wrapped__
        assert (by_name["shared"].kind, by_name["shared"].decorators) == ("function", ("staticmethod",))

    def test_other_wrappers_are_opaque(self):
        class Thing:
            def _base(self, a: int, b: str) -> str:
                return b

            partial = functools.partialmethod(_base, 1)

            @functools.singledispatchmethod
            def dispatch(self, arg: object) -> str:
                return ""

            @functools.singledispatchmethod
            @classmethod
            def cdispatch(cls, arg: object) -> str:
                return ""

            @_logged
            def fetch(self, url: str) -> bytes:
                return b""

            @_alogged
            async def afetch(self, url: str) -> bytes:
                return b""

        by_name = {m.name: m for m in collect_class_members(Thing).methods}
        assert by_name["_base"].kind == "function"
        for name in ("partial", "dispatch", "cdispatch", "fetch", "afetch"):
            assert by_name[name].kind == "opaque", name
        assert by_name["cdispatch"].decorators == ("classmethod",)
        # Calling the wrapper itself decides what comes back.
        assert by_name["afetch"].is_async is True
        assert by_name["fetch"].is_async is False


class TestClassValues:
    def test_unannotated_values_are_collected(self):
        class Thing(_ValuesBase):
            MAX_STEPS = 30
            name = "thing"
            _registry: ClassVar[dict] = {}  # annotated: a class_attr, not a value
            INHERITED = 2  # a base binds it: left out
            declared = 3  # a base annotates it: left out

            def run(self) -> None: ...

        members = collect_class_members(Thing)
        assert [(v.name, v.value) for v in members.values] == [("MAX_STEPS", 30), ("name", "thing")]
        assert all(v.owner is Thing for v in members.values)
        assert [a.name for a in members.class_attrs] == ["_registry"]

    def test_conscribe_markers_and_machinery_are_not_values(self):
        @runtime_checkable
        class Probe(Protocol):
            LIMIT = 3

            def setup(self) -> None: ...

        Envs = create_registrar("test_values_env", EnvProtocol, discriminator_field="kind")
        Owners = create_registrar("test_values_owner", EnvProtocol, discriminator_field="kind")

        class EnvBase(metaclass=Envs.Meta):
            __abstract__ = True

            def setup(self) -> None: ...

        class Bash(EnvBase):
            pass

        class Base(metaclass=Owners.Meta):
            __abstract__ = True
            __wiring__ = {"env": "test_values_env"}

            def setup(self) -> None: ...

        class Excluding(Base):
            __wiring__ = {"env": None}

        class Annotating(Excluding):
            env: str

        # A WiredField placeholder and a tombstone are conscribe's, not values.
        assert type(vars(Base)["env"]).__name__ == "WiredField"
        assert type(vars(Excluding)["env"]).__name__ == "_WiringTombstone"
        assert collect_class_members(Base).values == ()
        assert collect_class_members(Excluding).values == ()
        # Nor does the tombstone count as a default of an annotated name.
        (attr,) = collect_class_members(Annotating).class_attrs
        assert (attr.name, attr.has_default) == ("env", False)
        # ``_is_protocol``, ``_abc_impl``... are typing / abc machinery.
        assert [v.name for v in collect_class_members(Probe).values] == ["LIMIT"]

    def test_nested_classes_are_collected(self):
        class Outer:
            class Config:
                retries = 2

            Alias = int  # a class value, not a nested definition

        members = collect_class_members(Outer)
        assert members.nested == (("Config", Outer.Config),)
        assert [v.name for v in members.values] == ["Alias"]


def test_public_api_is_exported_and_documented():
    import conscribe.stubs as stubs

    assert set(stubs.__all__) >= {
        "ClassStubInfo", "InjectedAttr", "MethodStub", "ClassAttrStub", "ClassValueStub",
        "ClassMembers", "collect_class_members", "collect_class_stub_info",
        "generate_module_stub", "narrowest_common_base", "write_layer_stubs",
    }
    for name in stubs.__all__:
        assert getattr(stubs, name).__doc__, name
