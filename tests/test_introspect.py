"""Tests for the public ``conscribe.introspect`` API (F7).

``init_param_names`` / ``receptor_slots`` / ``wiring_slots`` are the single
source of truth for "which names reach the constructor, which attributes can
receive a wired value, and in what shape".  The af divergence that motivated
the API (a ``**kwargs`` child whose parent knobs were invisible) is pinned in
``TestInitParamNames.test_kwargs_child_includes_parent_knobs``.
"""
from __future__ import annotations

import abc
import dataclasses
import functools
import importlib
import inspect
import sys
import textwrap
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import pytest
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.dataclasses import dataclass as pydantic_dataclass

from conscribe import create_registrar
from conscribe.config import _annotations as _ann
from conscribe.introspect import (
    ReceptorSlot,
    WiringSlot,
    annotation_shape,
    config_from_param,
    init_param_names,
    is_classvar,
    receptor_slots,
    wiring_slots,
)
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister

T = TypeVar("T")


@runtime_checkable
class _Proto(Protocol):
    def run(self) -> None: ...


@pytest.fixture
def make_package(tmp_path, monkeypatch) -> Iterator[Callable[[dict[str, str]], str]]:
    created: list[str] = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(files: dict[str, str]) -> str:
        name = f"cs_intro_{uuid.uuid4().hex[:10]}"
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("")
        for module, source in files.items():
            (root / f"{module}.py").write_text(textwrap.dedent(source))
        created.append(name)
        importlib.invalidate_caches()
        return name

    yield make
    for name in created:
        for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[module]


@pytest.fixture
def registry_names() -> Iterator[Callable[[str], str]]:
    made: list[str] = []

    def make(prefix: str) -> str:
        name = f"cs_intro_{prefix}_{uuid.uuid4().hex[:6]}"
        made.append(name)
        return name

    yield make
    for name in made:
        if name in _REGISTRY_INDEX:
            _deregister(name)


# ---------------------------------------------------------------------------
# init_param_names
# ---------------------------------------------------------------------------


class LoopBase:
    def __init__(self, max_steps: int = 10, temperature: float = 0.0):
        self.max_steps = max_steps
        self.temperature = temperature


class React(LoopBase):
    def __init__(self, verbose: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.verbose = verbose


class TestInitParamNames:
    def test_kwargs_child_includes_parent_knobs(self) -> None:
        """The af divergence (F7 s1): a ``**kwargs`` child forwards the parent's
        knobs; 1.x's private helper and af's copy both answered ['verbose']."""
        assert init_param_names(React) == ("verbose", "max_steps", "temperature")
        instance = React(max_steps=3, verbose=True)  # the runtime agrees
        assert instance.max_steps == 3

    def test_chain_through_initless_mixins(self) -> None:
        class Root:
            def __init__(self, a: int = 1):
                pass

        class GenericMid(Generic[T], Root):
            def __init__(self, b: int = 2, **kwargs: Any):
                super().__init__(**kwargs)

        class Leaf(abc.ABC, GenericMid[int]):
            def __init__(self, c: int = 3, **kwargs: Any):
                super().__init__(**kwargs)

        assert init_param_names(Leaf) == ("c", "b", "a")

    def test_inherited_init(self) -> None:
        class Child(LoopBase):
            pass

        assert init_param_names(Child) == ("max_steps", "temperature")

    def test_no_init_anywhere(self) -> None:
        class Bare:
            x: int = 1

        assert init_param_names(Bare) == ()

    def test_instance_parameter_with_any_name(self) -> None:
        class Odd:
            def __init__(this, a: int = 1):  # noqa: N805
                pass

        assert init_param_names(Odd) == ("a",)

    def test_positional_only_and_var_params_are_not_names(self) -> None:
        class Mixed:
            def __init__(self, a: int, /, b: int, *args: Any, c: int = 1, **kwargs: Any):
                pass

        assert init_param_names(Mixed) == ("b", "c")

    def test_functools_wraps_decorated_init(self) -> None:
        def logged(fn):
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any):
                return fn(*args, **kwargs)

            return wrapper

        class Decorated:
            @logged
            def __init__(self, knob: int = 1):
                pass

        assert init_param_names(Decorated) == ("knob",)

    def test_dataclass_generated_init(self) -> None:
        @dataclasses.dataclass
        class Base:
            a: int = 1
            only_class: ClassVar[int] = 3

        @dataclasses.dataclass
        class Child(Base):
            iv: dataclasses.InitVar[int] = 0
            _: dataclasses.KW_ONLY
            k: str = "x"
            hidden: int = dataclasses.field(default=0, init=False)

            def __post_init__(self, iv: int) -> None:
                pass

        assert init_param_names(Child) == ("a", "iv", "k")

    def test_dataclass_with_field_named_self(self) -> None:
        @dataclasses.dataclass
        class Weird:
            self: int = 1  # the generated __init__ calls its instance __dataclass_self__

        assert init_param_names(Weird) == ("self",)

    def test_pydantic_model_fields(self) -> None:
        class Model(BaseModel):
            a: int = 1
            b: list[int] = Field(default_factory=list)
            _private: int = PrivateAttr(default=0)
            cv: ClassVar[int] = 3

        assert init_param_names(Model) == ("a", "b")

    def test_pydantic_model_init_only_forwarding_data(self) -> None:
        class Model(BaseModel):
            a: int = 1
            b: str = "x"

            def __init__(self, **data: Any):
                super().__init__(**data)

        assert init_param_names(Model) == ("a", "b")

    def test_pydantic_model_with_named_init_params(self) -> None:
        class Model(BaseModel):
            a: int = 1

            def __init__(self, extra: int = 0, **data: Any):
                super().__init__(**data)

        assert init_param_names(Model) == ("extra", "a")

    def test_pydantic_dataclass(self) -> None:
        @pydantic_dataclass
        class PDC:
            x: int = 1
            y: list[int] = Field(default_factory=list)
            z: str = Field("zz", init=False)

        assert init_param_names(PDC) == ("x", "y")

    def test_kwargs_child_of_pydantic_dataclass(self) -> None:
        @pydantic_dataclass
        class PDC:
            x: int = 1

        class Child(PDC):
            def __init__(self, c: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        assert init_param_names(Child) == ("c", "x")

    def test_bridged_class(self, registry_names) -> None:
        class External:
            def __init__(self, host: str = "localhost", port: int = 80):
                pass

        reg = create_registrar(registry_names("bridge"), _Proto)
        bridge = reg.bridge(External)

        class Impl(bridge):
            __registry_key__ = "impl"

            def __init__(self, retries: int = 3, **kwargs: Any):
                super().__init__(**kwargs)

            def run(self) -> None: ...

        assert init_param_names(Impl) == ("retries", "host", "port")

    def test_protocol_subclass_is_instantiation_independent(self) -> None:
        @runtime_checkable
        class Runner(Protocol):
            def run(self) -> None: ...

        class Base:
            def __init__(self, a: int = 1):
                self.a = a

        class Impl(Runner, Base):
            def run(self) -> None: ...

        class Plain(Runner):
            def run(self) -> None: ...

        before = (init_param_names(Impl), init_param_names(Plain))
        Impl(a=2)
        Plain()
        # typing copied Base.__init__ / object.__init__ into the classes.
        assert "__init__" in Impl.__dict__ and "__init__" in Plain.__dict__
        assert (init_param_names(Impl), init_param_names(Plain)) == before == (("a",), ())

    def test_rejects_non_classes(self) -> None:
        with pytest.raises(TypeError, match="expects a class"):
            init_param_names(React())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# receptor_slots
# ---------------------------------------------------------------------------


class TestReceptorSlots:
    def test_via_shape_annotation_and_parameter(self) -> None:
        class Env:
            capabilities: Dict[str, Any]
            single: Any
            cv: ClassVar[int] = 1

            def __init__(self, name: str = "env", tools: Optional[list[str]] = None):
                pass

        slots = receptor_slots(Env)
        assert list(slots) == ["name", "tools", "capabilities", "single"]
        tools = slots["tools"]
        assert (tools.via, tools.shape) == ("init", "list")
        assert tools.annotation == Optional[list[str]]
        assert tools.declared_on is Env
        assert tools.parameter is not None and tools.parameter.default is None
        caps = slots["capabilities"]
        assert (caps.via, caps.shape, caps.parameter) == ("annotation", "dict", None)
        assert caps.annotation == Dict[str, Any]
        assert slots["single"].shape == "scalar"

    def test_forwarded_parent_param_is_an_init_slot(self) -> None:
        """F7 s2: a wired parent param forwarded through **kwargs is not a
        slot-less / injected attribute."""

        class AgentBase:
            def __init__(self, llm: Any = None, max_steps: int = 5):
                pass

        class KwChild(AgentBase):
            def __init__(self, verbose: bool = False, **kwargs: Any):
                super().__init__(**kwargs)

        slots = receptor_slots(KwChild)
        assert slots["llm"].via == "init"
        assert slots["llm"].declared_on is AgentBase

    def test_classvar_redeclaration_removes_slot(self) -> None:
        class Base:
            receptor: Any

        class Child(Base):
            receptor: ClassVar[Any] = None

        assert "receptor" in receptor_slots(Base)
        assert "receptor" not in receptor_slots(Child)

    def test_child_annotation_wins(self) -> None:
        class Base:
            caps: Any

        class Child(Base):
            caps: list[Any]

        slot = receptor_slots(Child)["caps"]
        assert slot.declared_on is Child and slot.shape == "list"

    def test_unannotated_param_uses_class_annotation(self) -> None:
        class Owner:
            caps: dict[str, Any]

            def __init__(self, caps=None):
                self.caps = caps

        slot = receptor_slots(Owner)["caps"]
        assert slot.via == "init" and slot.shape == "dict"
        assert slot.annotation == dict[str, Any]

    def test_unannotated_without_class_annotation(self) -> None:
        class Owner:
            def __init__(self, thing=None):
                pass

        slot = receptor_slots(Owner)["thing"]
        assert slot.annotation is None and slot.shape == "scalar"

    def test_dataclass_pseudo_fields(self) -> None:
        @dataclasses.dataclass
        class Holder:
            a: int = 1
            iv: dataclasses.InitVar[list[int]] = None
            _: dataclasses.KW_ONLY
            k: str = "x"

            def __post_init__(self, iv: Any) -> None:
                pass

        slots = receptor_slots(Holder)
        assert list(slots) == ["a", "iv", "k"]
        assert slots["iv"].via == "init"
        assert "_" not in slots

    def test_unresolvable_annotation_keeps_raw_text_and_shape(self, make_package) -> None:
        pkg = make_package(
            {
                "tc": """
                from __future__ import annotations
                from typing import TYPE_CHECKING, Any, Optional

                if TYPE_CHECKING:
                    from cs_absent_pkg import Cap

                class Env:
                    caps: Optional[dict[str, Cap]]
                    typo: Lsit[Cap]
                    fine: list[Any]

                    def __init__(self, tools: Optional[list[Cap]] = None):
                        pass
                """
            }
        )
        env = importlib.import_module(f"{pkg}.tc").Env
        slots = receptor_slots(env)
        assert slots["caps"].annotation == "Optional[dict[str, Cap]]"
        assert slots["caps"].shape == "dict"
        assert slots["typo"].shape == "scalar"  # 'Lsit' is not a container name
        assert slots["fine"].annotation == list[Any]
        assert slots["tools"].annotation == "Optional[list[Cap]]"
        assert slots["tools"].shape == "list"
        assert "cs_absent_pkg" not in sys.modules

    @pytest.mark.parametrize(
        "annotation, shape",
        [
            ("list", "list"),
            ("Optional[List[Any]]", "list"),
            ("list[Any] | None", "list"),
            ("Sequence[Any]", "list"),
            ("Mapping[str, Any]", "dict"),
            ("tuple[Any, ...]", "list"),
            ("Any", "scalar"),
        ],
    )
    def test_af_shapes(self, make_package, annotation: str, shape: str) -> None:
        """F7 s4: the receptor forms af's ``get_origin in (dict, list)`` check
        got wrong, including one next to an unresolvable sibling."""
        pkg = make_package(
            {
                "shapes": f"""
                from __future__ import annotations
                from collections.abc import Mapping, Sequence
                from typing import Any, List, Optional

                class Env:
                    caps: {annotation}
                    ghost: Ghost
                """
            }
        )
        env = importlib.import_module(f"{pkg}.shapes").Env
        assert receptor_slots(env)["caps"].shape == shape

    def test_lazily_annotated_module(self, make_package) -> None:
        """No ``from __future__ import annotations`` (PEP 649 on 3.14)."""
        pkg = make_package(
            {
                "lazy": """
                from typing import Any, ClassVar, Optional

                class Browser:
                    capabilities: dict[str, Any]
                    llm: Any
                    cv: ClassVar[int] = 3

                    def __init__(self, name: str = "b"):
                        pass
                """
            }
        )
        browser = importlib.import_module(f"{pkg}.lazy").Browser
        slots = receptor_slots(browser)
        assert list(slots) == ["name", "capabilities", "llm"]
        assert slots["capabilities"].shape == "dict"
        assert slots["llm"].via == "annotation"

    def test_pydantic_model_fields_and_private_attrs(self) -> None:
        class Model(BaseModel):
            items: list[int] = Field(default_factory=list)
            _cache: dict = PrivateAttr(default_factory=dict)
            note: ClassVar[str] = "x"

        slots = receptor_slots(Model)
        assert slots["items"].via == "init" and slots["items"].shape == "list"
        assert slots["items"].annotation == list[int]
        assert slots["_cache"].via == "annotation"
        assert "note" not in slots and "model_config" not in slots

    def test_slots_are_frozen(self) -> None:
        slot = receptor_slots(React)["verbose"]
        assert isinstance(slot, ReceptorSlot)
        with pytest.raises(dataclasses.FrozenInstanceError):
            slot.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# wiring_slots
# ---------------------------------------------------------------------------


class TestWiringSlots:
    def test_slot_and_slot_less_entries(self) -> None:
        class Agent:
            llm: Any
            __wiring__ = {
                "llm": "cs_intro_missing_llm",
                "loop": ("cs_intro_missing_loop", ["react"]),
                "browser": ["chromium", "firefox"],
                "decl": "cs_intro_missing_decl",
            }

            def __init__(self, loop: str = "react", browser: Optional[str] = None):
                pass

        slots = wiring_slots(Agent)  # no registry is needed
        assert list(slots) == ["llm", "loop", "browser", "decl"]
        assert slots["llm"].slot.via == "annotation"
        assert slots["loop"].slot.via == "init"
        assert slots["loop"].spec.allowed_keys == ("react",)
        assert slots["browser"].slot.via == "init"
        assert slots["decl"].slot is None
        assert all(not s.declare_only and s.default_key is None for s in slots.values())
        assert isinstance(slots["decl"], WiringSlot)

    def test_inherited_wiring_and_exclusion(self) -> None:
        class Base:
            __wiring__ = {"llm": "cs_intro_x", "memory": "cs_intro_y"}

            def __init__(self, llm: Any = None, memory: Any = None):
                pass

        class Child(Base):
            __wiring__ = {"memory": None}

            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)

        slots = wiring_slots(Child)
        assert list(slots) == ["llm"]
        assert slots["llm"].slot.via == "init"

    def test_reads_declare_only_and_default_key(self, monkeypatch) -> None:
        """Lane A adds ``declare_only`` / ``default_key`` to ``WiringSpec``;
        they are read with ``getattr`` so both grammars work."""
        from conscribe import wiring

        @dataclasses.dataclass(frozen=True)
        class FutureSpec:
            param_name: str
            registry_name: str = ""
            allowed_keys: Any = None
            optional_keys: Any = None
            default_key: Any = None
            declare_only: bool = False

        monkeypatch.setattr(
            wiring,
            "parse_wiring",
            lambda cls: [
                FutureSpec("loop", "loops", default_key="react"),
                FutureSpec("negotiated", "caps", declare_only=True),
            ],
        )

        class Agent:
            def __init__(self, loop: Any = None):
                pass

        slots = wiring_slots(Agent)
        assert slots["loop"].default_key == "react" and not slots["loop"].declare_only
        assert slots["negotiated"].declare_only and slots["negotiated"].slot is None

    def test_dict_form_grammar(self) -> None:
        """With lane A's grammar merged: dict-form ``default`` / ``declare_only``."""
        from conscribe.wiring import WiringSpec

        if "declare_only" not in WiringSpec.__dataclass_fields__:
            pytest.skip("dict-form wiring entries not available in this tree")

        class Agent:
            __wiring__ = {
                "loop": {"registry": "cs_intro_loops", "default": "react"},
                "negotiated": {"registry": "cs_intro_caps", "declare_only": True},
            }

            def __init__(self, loop: Any = None):
                pass

        slots = wiring_slots(Agent)
        assert slots["loop"].default_key == "react"
        assert slots["loop"].slot.via == "init"
        assert slots["negotiated"].declare_only is True
        assert slots["negotiated"].slot is None

    def test_no_wiring(self) -> None:
        assert wiring_slots(React) == {}

    def test_malformed_wiring_raises(self) -> None:
        from conscribe.exceptions import WiringResolutionError

        class Bad:
            __wiring__ = {"loop": ("registry", "not-a-list")}

        with pytest.raises((TypeError, WiringResolutionError)):
            wiring_slots(Bad)


# ---------------------------------------------------------------------------
# config_from_param, annotation_shape, is_classvar
# ---------------------------------------------------------------------------


class TestConfigFromParam:
    def test_inheritance_and_opt_out(self) -> None:
        class Base:
            __config_from_param__ = "settings"

        class Child(Base):
            pass

        class OptOut(Base):
            __config_from_param__ = None

        class Override(Base):
            __config_from_param__ = "other"

        class Plain:
            pass

        assert config_from_param(Base) == "settings"
        assert config_from_param(Child) == "settings"
        assert config_from_param(OptOut) is None
        assert config_from_param(Override) == "other"
        assert config_from_param(Plain) is None

    def test_value_returned_as_declared(self) -> None:
        class Odd:
            __config_from_param__ = 42

        assert config_from_param(Odd) == 42


class TestReExports:
    def test_annotation_shape(self) -> None:
        assert annotation_shape(Optional[Dict[str, int]]) == "dict"
        assert annotation_shape("Sequence[X] | None") == "list"
        assert annotation_shape(str) == "scalar"
        assert annotation_shape("Mapping[str, int]", sys.modules[__name__]) == "dict"
        assert annotation_shape(Mapping[str, int]) == "dict"
        assert annotation_shape(Sequence) == "list"

    def test_is_classvar(self) -> None:
        assert is_classvar(ClassVar[int])
        assert is_classvar("ClassVar[int]")
        assert not is_classvar("ClassVarRegistry")
        assert not is_classvar(List[int])

    def test_engine_identity(self) -> None:
        """The public functions are the engine the extractor uses."""
        assert annotation_shape(list[int]) == _ann.annotation_shape(list[int])
        assert inspect.signature(annotation_shape).parameters.keys() == {"annotation", "owner"}


# ===========================================================================
# Review fixes (2.0.0 release review)
# ===========================================================================


class TestSlotDefaults:
    """M6: ``ReceptorSlot`` carries the real default of every slot so wiring
    can check parameter defaults up front (``default`` / ``default_factory``;
    ``parameter.default`` is the runtime default without stand-ins)."""

    def test_dataclass_value_and_factory(self) -> None:
        @dataclasses.dataclass
        class D:
            a: int = 1
            b: list[int] = dataclasses.field(default_factory=list)
            c: Optional[str] = None
            d: int = dataclasses.field(default=4)
            r: str = dataclasses.field(default="x")

        slots = receptor_slots(D)
        assert (slots["a"].default, slots["a"].default_factory) == (1, None)
        assert slots["a"].parameter.default == 1
        b = slots["b"]
        assert b.default is inspect.Parameter.empty and b.default_factory is list
        assert b.parameter.default is dataclasses._HAS_DEFAULT_FACTORY  # type: ignore[attr-defined]
        assert slots["c"].default is None and slots["c"].parameter.default is None
        assert slots["d"].default == 4

    def test_hand_written_init_and_field_metadata(self) -> None:
        class H:
            def __init__(
                self,
                required: int,
                plain: str = "p",
                via_field: int = Field(3),
                via_metadata: Annotated[int, Field(default=5)] = None,  # type: ignore[assignment]
                meta_only: Annotated[list[int], Field(default_factory=list)] = None,  # type: ignore[assignment]
            ) -> None: ...

        slots = receptor_slots(H)
        assert slots["required"].default is inspect.Parameter.empty
        assert slots["required"].parameter.default is inspect.Parameter.empty
        assert slots["plain"].default == "p"
        assert slots["via_field"].default == 3
        assert slots["via_field"].parameter.default == 3  # the Field(...) stand-in is gone
        # the signature default (None) is the parameter's own; it wins
        assert slots["via_metadata"].default is None

        class Meta:
            def __init__(self, n: Annotated[int, Field(default=5)], f: Annotated[list, Field(default_factory=list)]) -> None: ...

        meta = receptor_slots(Meta)
        assert meta["n"].default == 5 and meta["n"].parameter.default is inspect.Parameter.empty
        assert meta["f"].default_factory is list

    def test_model_fields(self) -> None:
        class M(BaseModel):
            a: int = 1
            b: list[int] = Field(default_factory=list)
            c: str

        slots = receptor_slots(M)
        assert slots["a"].default == 1 and slots["a"].parameter.default == 1
        assert slots["b"].default_factory is list
        assert slots["c"].default is inspect.Parameter.empty

    def test_attrs_defaults(self) -> None:
        attrs = pytest.importorskip("attrs")

        @attrs.define
        class A:
            req: str = attrs.field()
            n: int = 1
            tags: list = attrs.Factory(list)

        slots = receptor_slots(A)
        assert slots["n"].default == 1
        assert slots["tags"].default_factory is list
        assert slots["tags"].parameter.default is dataclasses._HAS_DEFAULT_FACTORY  # type: ignore[attr-defined]
        assert slots["req"].default is inspect.Parameter.empty
        assert slots["req"].parameter.default is inspect.Parameter.empty

    def test_annotation_slots_use_class_level_values(self) -> None:
        class Owner:
            hooks: list[_Proto] = []
            primary: _Proto = "recorder"  # type: ignore[assignment]
            maybe: Optional[_Proto] = None
            bare: _Proto
            computed: int

            @property
            def computed(self) -> int:  # type: ignore[no-redef]
                return 1

        class Child(Owner):
            hooks = ["x"]  # type: ignore[assignment]

        slots = receptor_slots(Owner)
        assert slots["hooks"].default == [] and slots["hooks"].parameter is None
        assert slots["primary"].default == "recorder"
        assert slots["maybe"].default is None
        assert slots["bare"].default is inspect.Parameter.empty
        assert slots["computed"].default is inspect.Parameter.empty  # a property is no default
        assert receptor_slots(Child)["hooks"].default == ["x"]

    def test_wired_placeholder_is_not_a_default(self) -> None:
        from conscribe.wiring import WiredField

        class Owner:
            target: _Proto
            target = WiredField("target", "reg")  # type: ignore[misc]

        assert receptor_slots(Owner)["target"].default is inspect.Parameter.empty

    def test_names_filter(self) -> None:
        class H:
            x: int

            def __init__(self, a: int = 1, b: str = "") -> None: ...

        assert set(receptor_slots(H, names=["b", "x", "missing"])) == {"b", "x"}
        assert receptor_slots(H, names=[]) == {}


class TestReviewSlotShapes:
    def test_set_method_does_not_shadow_builtin_set(self, make_package) -> None:
        """M1: the receptor shape of ``listeners: set[Hook]`` is "list" even
        when the class defines a method named ``set``."""
        pkg = make_package(
            {
                "cache": """
                from __future__ import annotations
                from typing import Any, Protocol

                class Hook(Protocol):
                    def fire(self) -> None: ...

                class Cache:
                    def __init__(self, listeners: set[Hook] = frozenset(), size: int = 10) -> None:
                        pass

                    def set(self, key: str, value: Any) -> None:
                        pass
                """
            }
        )
        cache = importlib.import_module(f"{pkg}.cache").Cache
        slot = receptor_slots(cache)["listeners"]
        assert slot.shape == "list"
        assert slot.default == frozenset()

    def test_protocols_are_scalars(self) -> None:
        """A Protocol is an interface, never a collection receptor (only
        exactly ``Iterable`` / ``Collection`` count as lists)."""
        from collections.abc import Collection, Iterable

        class Plain(Protocol):
            def run(self) -> None: ...

        class IterLike(Iterable[int], Protocol):
            pass

        class Sized(Collection[int], Protocol):
            pass

        for proto in (Plain, IterLike, Sized):
            assert annotation_shape(proto) == "scalar"
            assert annotation_shape(Optional[proto]) == "scalar"
            assert annotation_shape(list[proto]) == "list"


class TestReviewInitParamNames:
    def test_cooperative_mixin_before_base_model(self) -> None:
        """B4: the runtime view of a model with a forwarding mixin."""

        class SetupMixin:
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

        class Tool(SetupMixin, BaseModel):
            name: str = "t"
            retries: int = 3

        assert init_param_names(Tool) == ("name", "retries")
