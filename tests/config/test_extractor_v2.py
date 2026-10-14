"""conscribe 2.0 extractor behaviour (lane G).

Every section pins one decision of the 2.0 spec, including af's acceptance
repros (feedback defects 1, 3, 6, 8, 10 and feature F4):

* X1   -- every extraction path reaches the wiring pass.
* F1   -- wiring problems raise, aggregated per class; nothing is swallowed.
* D10  -- wired annotations map member-wise onto the wiring ``Literal``.
* D3   -- owner-aware annotation resolution, TYPE_CHECKING degrade, typos.
* None -- ``x: T = None`` admits None on every Python version.
* F4   -- ``declare_only`` shape and Literal-mode ``default`` keys.
* D6   -- ``__config_from_param__`` opt-in flattening and the old-shape guard.
* F5   -- real defaults (factories, purity gate) and full ``Field`` carry.
* D9   -- receptors injected from slots keep their description.
"""
from __future__ import annotations

import abc
import dataclasses
import importlib
import inspect
import sys
import textwrap
import uuid
import warnings
from collections.abc import Callable, Iterator
from typing import (
    Annotated,
    Any,
    Final,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

import pytest
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, RootModel, ValidationError, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from conscribe import build_composed_config, create_registrar
from conscribe.config.builder import build_layer_config
from conscribe.config.extractor import (
    _build_param_fields,
    _evaluate_factory,
    _is_pure_factory,
    extract_config_schema,
    extract_own_init_params,
)
from conscribe.config.mro import walk_init_chain
from conscribe.exceptions import (
    ConfigAnnotationError,
    ConfigDeclarationError,
    DegradationWarning,
    ExtractionWarning,
    InvalidConfigSchemaError,
    WiringResolutionError,
)
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister

T = TypeVar("T")


@runtime_checkable
class _Proto(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class Strategy(Protocol):
    def pick(self) -> int: ...


@runtime_checkable
class OtherProto(Protocol):
    def other(self) -> None: ...


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def registrar() -> Iterator[Callable[..., Any]]:
    """Factory of throw-away registrars (unique names, removed afterwards)."""
    made: list[str] = []

    def make(prefix: str, *, protocol: Any = _Proto, **kwargs: Any) -> Any:
        name = f"xv2_{prefix}_{uuid.uuid4().hex[:6]}"
        kwargs.setdefault("discriminator_field", f"{prefix}_id")
        made.append(name)
        return create_registrar(name, protocol, **kwargs)

    yield make
    for name in made:
        if name in _REGISTRY_INDEX:
            _deregister(name)


@pytest.fixture
def target(registrar) -> Callable[..., Any]:
    """A registry holding plain implementations under the given keys."""

    def make(keys: list[str], prefix: str = "strat", **kwargs: Any) -> Any:
        reg = registrar(prefix, **kwargs)
        for key in keys:
            impl = type(f"Impl_{key.replace('.', '_')}", (), {"run": lambda self: None})
            reg.register(key)(impl)
        return reg

    return make


@pytest.fixture
def make_package(tmp_path, monkeypatch) -> Iterator[Callable[[dict[str, str]], str]]:
    """Write a throw-away package of real modules and return its name."""
    created: list[str] = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(files: dict[str, str]) -> str:
        name = f"xv2_pkg_{uuid.uuid4().hex[:10]}"
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


def _name(reg: Any) -> str:
    return reg._registry.name


def _literal_args(annotation: Any) -> tuple[Any, ...] | None:
    """The values of the (first) Literal inside Optional / list / Annotated."""
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    for arg in get_args(annotation):
        found = _literal_args(arg)
        if found is not None:
            return found
    return None


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _field(model: Any, name: str) -> Any:
    return model.model_fields[name]


# ===========================================================================
# X1 -- every path reaches wiring
# ===========================================================================


class TestNoInitAnywhere:
    """A class with no ``__init__`` in its MRO still gets its receptors."""

    def test_receptors_survive_without_init(self, target) -> None:
        strat = target(["fast", "slow"])
        caps = target(["browser", "docker"], prefix="cap")

        class NoInit:
            __wiring__ = {
                "strategy": _name(strat),
                "multi": (_name(caps), ["browser", "docker"]),
                "decl": (_name(strat), ["fast"]),
            }
            strategy: Strategy
            multi: dict[str, Any]

        model = extract_config_schema(NoInit)
        assert model is not None
        strategy = _field(model, "strategy")
        assert strategy.is_required() and _literal_args(strategy.annotation) == ("fast", "slow")
        multi = _field(model, "multi")
        assert get_origin(multi.annotation) is list and multi.is_required()
        decl = _field(model, "decl")
        assert not decl.is_required() and _is_optional(decl.annotation)
        assert model.model_config.get("extra") == "forbid"
        assert model.__wired_fields__ == {
            "strategy": _name(strat), "multi": _name(caps), "decl": _name(strat),
        }

    def test_no_init_matches_empty_init(self, target) -> None:
        """No ``__init__`` and ``def __init__(self)`` produce the same fields."""
        strat = target(["fast"])

        class NoInit:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy

        class EmptyInit:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy

            def __init__(self) -> None: ...

        a, b = extract_config_schema(NoInit), extract_config_schema(EmptyInit)
        for name in ("strategy",):
            fa, fb = _field(a, name), _field(b, name)
            assert (fa.annotation, fa.is_required(), fa.default) == (
                fb.annotation, fb.is_required(), fb.default,
            )

    def test_inherited_wiring_from_initless_base(self, target) -> None:
        strat = target(["fast"])

        class Base:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy

        class Child(Base):
            pass

        model = extract_config_schema(Child)
        assert _literal_args(_field(model, "strategy").annotation) == ("fast",)

    def test_dataclass_without_init(self, target) -> None:
        strat = target(["fast"])

        @dataclasses.dataclass(init=False)
        class Holder:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy = dataclasses.field(default=None)

        model = extract_config_schema(Holder)
        assert "strategy" in model.model_fields

    def test_nothing_to_emit_is_none(self) -> None:
        class Plain:
            pass

        assert extract_config_schema(Plain) is None

    def test_protocol_subclass_without_own_init_is_closed(self) -> None:
        """The Protocol placeholder ``__init__`` is not a constructor."""

        class Impl(_Proto):
            def run(self) -> None: ...

        assert extract_config_schema(Impl) is None
        Impl()  # instantiation copies an __init__ into the class...
        assert extract_config_schema(Impl) is None  # ...the answer is stable

    def test_generic_and_abc_mixins_do_not_cut_the_kwargs_chain(self) -> None:
        class Parent:
            def __init__(self, a: int = 1, b: str = "x") -> None: ...

        class Mixin(Generic[T], abc.ABC):  # no __init__ of its own
            pass

        class Child(Mixin[int], Parent):
            def __init__(self, c: float = 0.0, **kwargs: Any) -> None:
                super().__init__(**kwargs)

        model = extract_config_schema(Child, mro_scope="all")
        assert set(model.model_fields) == {"c", "a", "b"}
        assert model.model_config.get("extra") == "forbid"


class TestBaseModelFastPath:
    def test_wired_model_field_becomes_literal(self, target) -> None:
        strat = target(["fast", "slow"])

        class Holder(BaseModel):
            __wiring__ = {"mode": (_name(strat), ["fast"])}
            mode: str = Field("fast", description="the mode")
            depth: int = 2

        model = extract_config_schema(Holder)
        mode = _field(model, "mode")
        assert _literal_args(mode.annotation) == ("fast",)
        assert mode.default == "fast" and mode.description == "the mode"
        assert model.__wired_fields__ == {"mode": _name(strat)}
        with pytest.raises(ValidationError):
            model(mode="fsat")
        # A fresh model, never derived from the registered class.
        assert Holder not in model.__mro__

    def test_model_with_only_a_receptor(self, target) -> None:
        strat = target(["fast"])

        class Holder(BaseModel):
            __wiring__ = {"strategy": _name(strat)}

        model = extract_config_schema(Holder)
        assert _is_optional(_field(model, "strategy").annotation)

    def test_init_that_only_forwards_data_takes_the_fast_path(self) -> None:
        class Holder(BaseModel):
            name: str = "n"
            count: int = Field(0, ge=0)

            def __init__(self, **data: Any) -> None:
                super().__init__(**data)

        model = extract_config_schema(Holder)
        assert set(model.model_fields) == {"name", "count"}
        with pytest.raises(ValidationError):
            model(count=-1)

    def test_field_info_of_the_registered_model_is_not_mutated(self, target) -> None:
        strat = target(["fast"])

        class Holder(BaseModel):
            __wiring__ = {"mode": _name(strat)}
            mode: str = Field("fast", description="d")

        before = Holder.model_fields["mode"].annotation
        extract_config_schema(Holder)
        assert Holder.model_fields["mode"].annotation is before is str

    def test_registered_model_validators_do_not_run_on_the_config(self, registrar) -> None:
        reg = registrar("bm")

        class Holder(BaseModel, metaclass=type(BaseModel)):
            lo: int = 0
            hi: int = 1

            @model_validator(mode="after")
            def ordered(self) -> "Holder":
                if self.lo > self.hi:
                    raise ValueError("lo > hi")
                return self

            def run(self) -> None: ...

        reg.register("holder")(Holder)
        model = extract_config_schema(Holder)
        assert model(lo=5, hi=1).lo == 5  # documented: validators are not copied


class TestPydanticDataclassFastPath:
    def test_fields_and_factories(self) -> None:
        @pydantic_dataclass
        class Settings:
            name: str
            tags: list[str] = dataclasses.field(default_factory=list)
            limit: Annotated[int, Field(ge=1, description="max")] = 3

        model = extract_config_schema(Settings)
        assert set(model.model_fields) == {"name", "tags", "limit"}
        assert _field(model, "tags").default == []
        assert _field(model, "limit").description == "max"
        assert model.model_config.get("extra") == "forbid"
        with pytest.raises(ValidationError):
            model(name="x", limit=0)


class TestUnreadableSignatures:
    def test_unreadable_parent_in_kwargs_chain(self) -> None:
        class _Opaque:
            __signature__ = "not a signature"

            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class Parent:
            __init__ = _Opaque()

        class Child(Parent):
            def __init__(self, c: int = 1, **kwargs: Any) -> None: ...

        with pytest.warns(ExtractionWarning, match="Parent.__init__"):
            model = extract_config_schema(Child, mro_scope="all")
        assert set(model.model_fields) == {"c"}
        assert model.model_config.get("extra") == "allow"

    def test_unreadable_definer_keeps_receptors(self, target) -> None:
        strat = target(["fast"])

        class _Opaque:
            __signature__ = "not a signature"

            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class Holder:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy
            __init__ = _Opaque()

        with pytest.warns(ExtractionWarning):
            model = extract_config_schema(Holder)
        assert "strategy" in model.model_fields
        assert model.model_config.get("extra") == "allow"

    def test_signature_without_annotations_uses_the_raw_signature(self) -> None:
        """A readable signature on a non-Python ``__init__`` (builtin /
        extension): parameters come from the signature, typed Any, with an
        ``ExtractionWarning`` (1.x dropped every parameter)."""
        import inspect

        class _Native:
            __signature__ = inspect.Signature(
                [
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("size", inspect.Parameter.KEYWORD_ONLY, default=4),
                ]
            )

            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class Holder:
            __init__ = _Native()

        with pytest.warns(ExtractionWarning, match="not a Python function"):
            model = extract_config_schema(Holder)
        size = _field(model, "size")
        assert size.annotation is Any and size.default == 4
        assert model.model_config.get("extra") == "forbid"


class TestTier3:
    def test_schema_used_verbatim(self, target) -> None:
        strat = target(["fast"])

        class Schema(BaseModel):
            strategy: str = "fast"
            n: int = 1

            @model_validator(mode="after")
            def check(self) -> "Schema":
                if self.n > 5:
                    raise ValueError("n too big")
                return self

        class Holder:
            __config_schema__ = Schema
            __wiring__ = {"strategy": _name(strat), "decl": {"registry": _name(strat), "declare_only": True}}
            strategy: Strategy

        assert extract_config_schema(Holder) is Schema
        assert not hasattr(Schema, "__wired_fields__")

    def test_missing_wired_field_is_an_error(self, target) -> None:
        strat = target(["fast"])

        class Schema(BaseModel):
            n: int = 1

        class Holder:
            __config_schema__ = Schema
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy

        with pytest.raises(ConfigDeclarationError, match=r"no field for the wired receptor\(s\) \['strategy'\]"):
            extract_config_schema(Holder)

    def test_slot_less_entries_need_not_be_fields(self, target) -> None:
        strat = target(["fast"])

        class Schema(BaseModel):
            n: int = 1

        class Holder:
            __config_schema__ = Schema
            __wiring__ = {"hint": _name(strat)}

        assert extract_config_schema(Holder) is Schema

    def test_wiring_problems_still_raise(self, target) -> None:
        strat = target(["fast"])

        class Schema(BaseModel):
            strategy: str = "fast"

        class Holder:
            __config_schema__ = Schema
            __wiring__ = {"strategy": (_name(strat), ["fsat"])}
            strategy: Strategy

        with pytest.raises(WiringResolutionError, match="unknown key 'fsat'"):
            extract_config_schema(Holder)

    def test_default_key_must_match_the_schema(self, target) -> None:
        strat = target(["fast", "slow"])

        class Schema(BaseModel):
            strategy: str = "slow"

        class Holder:
            __config_schema__ = Schema
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "fast"}}
            strategy: Strategy

        with pytest.raises(ConfigDeclarationError, match="disagrees"):
            extract_config_schema(Holder)

    def test_invalid_schema_is_a_type_error(self) -> None:
        class Holder:
            __config_schema__ = dict

        with pytest.raises(InvalidConfigSchemaError):
            extract_config_schema(Holder)


# ===========================================================================
# F1 -- wiring problems raise (af defect 1)
# ===========================================================================


class TestWiringProblemsRaise:
    def _layers(self, registrar) -> tuple[Any, Any, Any]:
        strat = registrar("strategy")
        act = registrar("act")
        own = registrar("owner")

        class StratBase(metaclass=strat.Meta):
            __abstract__ = True

            def run(self): ...

        class Fast(StratBase):
            __registry_key__ = "fast"

        class ActBase(metaclass=act.Meta):
            __abstract__ = True

            def run(self): ...

        class Click(ActBase):
            __registry_key__ = "click"

        return strat, act, own

    def test_af_defect_1(self, registrar) -> None:
        strat, act, own = self._layers(registrar)
        s, a = _name(strat), _name(act)

        class OwnerBase(metaclass=own.Meta):
            __abstract__ = True

            def run(self): ...

        class Good(OwnerBase):
            __registry_key__ = "good"
            __wiring__ = {"strategy": s, "mode": (s, ["fast"]), "action": (a, ["click"])}
            strategy: Any

            def __init__(self, mode: str = "fast"): ...

        class Bad(OwnerBase):
            __registry_key__ = "bad"
            __wiring__ = {"strategy": s, "mode": (s, ["fast"]), "action": (a, ["click", "kick"])}
            strategy: Any

            def __init__(self, mode: str = "fast"): ...

        good = extract_config_schema(Good)
        assert set(good.model_fields) == {"mode", "strategy", "action"}
        assert _literal_args(_field(good, "mode").annotation) == ("fast",)

        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(Bad)
        error = excinfo.value
        assert len(error.problems) == 1
        problem = error.problems[0]
        assert (problem.param_name, problem.kind) == ("action", "unknown_key")
        message = str(error)
        assert "Bad (key 'bad') wiring 'action'" in message
        assert "'kick'" in message and "registered: ['click']" in message
        assert "strategy" not in message and "mode" not in message

    def test_every_problem_of_a_class_is_reported_at_once(self, target) -> None:
        strat = target(["fast"])

        class Broken:
            __wiring__ = {
                "a": (_name(strat), ["nope"]),
                "b": "no_such_registry_xv2",
                "c": {"registry": _name(strat), "declare_only": True},
            }
            c: Any

        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(Broken)
        kinds = [(p.param_name, p.kind) for p in excinfo.value.problems]
        assert kinds == [("a", "unknown_key"), ("b", "unknown_registry"), ("c", "contradiction")]

    def test_malformed_entry_is_not_swallowed(self, target) -> None:
        strat = target(["fast"])

        class Broken:
            __wiring__ = {"mode": (_name(strat), "fast")}  # keys must be a list

            def __init__(self, mode: str = "fast"): ...

        with pytest.raises(WiringResolutionError, match="malformed|must be a list"):
            extract_config_schema(Broken)


# ===========================================================================
# D10 -- wired annotations (af defect 10)
# ===========================================================================


def _owner_with(annotation: Any, registry: str, *, default: Any = ..., keys: Any = None) -> type:
    """A class whose ``__init__`` takes one wired parameter ``strategy``."""
    namespace: dict[str, Any] = {"__wiring__": {"strategy": registry if keys is None else (registry, keys)}}
    params = "self, strategy: _ANN" + ("" if default is ... else " = _DEFAULT")
    code = f"def __init__({params}):\n    pass\n"
    scope: dict[str, Any] = {"_ANN": annotation, "_DEFAULT": default}
    exec(code, scope)  # noqa: S102 - builds a signature with a real annotation object
    init = scope["__init__"]
    init.__annotations__ = {"strategy": annotation}
    namespace["__init__"] = init
    return type("Owner", (), namespace)


class TestD10WiredAnnotations:
    @pytest.mark.parametrize(
        "annotation",
        [
            Optional[Strategy],
            Union[Strategy, None],
            Strategy | None,
            Annotated[Strategy | None, Field(description="how to pick")],
            Optional[Any],
            object | None,
            Optional[int],
            Union[Strategy, OtherProto, None],
        ],
        ids=lambda a: repr(a)[:40],
    )
    def test_nullable_members_map_to_optional_literal(self, target, annotation) -> None:
        strat = target(["fast", "slow"])
        model = extract_config_schema(_owner_with(annotation, _name(strat), default=None))
        field = _field(model, "strategy")
        assert _is_optional(field.annotation), field.annotation
        assert _literal_args(field.annotation) == ("fast", "slow")
        assert field.default is None
        assert not getattr(model, "__degraded_fields__", None)
        assert model(strategy=None).strategy is None
        assert model(strategy="slow").strategy == "slow"
        with pytest.raises(ValidationError):
            model(strategy={"anything": 1})

    def test_af_defect_10(self, registrar) -> None:
        strat = registrar("strat", protocol=Strategy)
        own = registrar("owner")

        class StratBase(metaclass=strat.Meta):
            __abstract__ = True

            def pick(self): return 0

        class Fast(StratBase):
            __registry_key__ = "fast"

        class OwnerBase(metaclass=own.Meta):
            __abstract__ = True

            def run(self): ...

        class Owner(OwnerBase):
            __registry_key__ = "owner"
            __wiring__ = {"strategy": _name(strat)}

            def __init__(
                self, strategy: Annotated[Strategy | None, Field(description="how to pick")] = None
            ): ...

        model = extract_config_schema(Owner)
        field = _field(model, "strategy")
        assert field.annotation == Optional[Literal["fast"]]
        assert field.default is None and field.description == "how to pick"
        assert not getattr(model, "__degraded_fields__", None)
        composed = build_composed_config({_name(own): own, _name(strat): strat})
        inline = composed.layer_results[_name(own)].per_key_models["owner"]
        assert inline.model_fields["strategy"].description == "how to pick"

    def test_union_with_str_keeps_key_default(self, target) -> None:
        strat = target(["fast", "slow"])
        model = extract_config_schema(_owner_with(Union[Strategy, str], _name(strat), default="fast"))
        field = _field(model, "strategy")
        assert field.annotation == Literal["fast", "slow"]
        assert field.default == "fast"
        assert model.__wired_defaults__ == {"strategy": "fast"}

    def test_container_members(self, target) -> None:
        strat = target(["fast", "slow"])
        for annotation in (Optional[list[Strategy]], dict[str, Strategy] | None):
            model = extract_config_schema(_owner_with(annotation, _name(strat), default=None))
            field = _field(model, "strategy")
            assert field.annotation == Optional[list[Literal["fast", "slow"]]], annotation

    def test_mixed_scalar_and_container_union(self, target) -> None:
        strat = target(["fast"])
        model = extract_config_schema(_owner_with(Union[list[Strategy], Strategy, None], _name(strat), default=None))
        assert _field(model, "strategy").annotation == Optional[
            Union[list[Literal["fast"]], Literal["fast"]]
        ]

    def test_required_nullable_without_default(self, target) -> None:
        strat = target(["fast"])
        model = extract_config_schema(_owner_with(Strategy | None, _name(strat)))
        field = _field(model, "strategy")
        assert field.is_required() and _is_optional(field.annotation)

    def test_user_literal_of_wiring_keys_is_accepted(self, target) -> None:
        strat = target(["fast", "slow"])
        model = extract_config_schema(_owner_with(Optional[Literal["fast"]], _name(strat), default=None))
        assert _literal_args(_field(model, "strategy").annotation) == ("fast", "slow")

    def test_user_literal_with_other_values_is_an_error(self, target) -> None:
        strat = target(["fast", "slow"])
        owner = _owner_with(Union[Strategy, Literal["auto"]], _name(strat), default="auto")
        with pytest.raises(ConfigDeclarationError, match=r"\['auto'\] are not keys"):
            extract_config_schema(owner)

    def test_wired_keys_recorded(self, target) -> None:
        strat = target(["b", "a"])
        model = extract_config_schema(_owner_with(Strategy, _name(strat), keys=["b", "a", "b"]))
        assert model.__wired_keys__ == {"strategy": ("b", "a")}
        assert _field(model, "strategy").annotation == Literal["b", "a"]

    def test_string_annotations_use_their_text_shape(self, target, make_package) -> None:
        strat = target(["fast"])
        pkg = make_package(
            {
                "mod": f"""
                from __future__ import annotations
                from typing import TYPE_CHECKING, Any, Optional

                if TYPE_CHECKING:
                    from somewhere import Strategy

                class Owner:
                    __wiring__ = {{
                        "single": "{_name(strat)}",
                        "maybe": "{_name(strat)}",
                        "many": "{_name(strat)}",
                        "mapping": "{_name(strat)}",
                    }}
                    mapping: dict[str, Strategy]

                    def __init__(
                        self,
                        single: Strategy,
                        maybe: Strategy | None = None,
                        many: Optional[list[Strategy]] = None,
                    ) -> None: ...
                """
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # receptors never warn about their annotation
            model = extract_config_schema(importlib.import_module(f"{pkg}.mod").Owner)
        assert _field(model, "single").annotation == Literal["fast"]
        assert _field(model, "single").is_required()
        assert _field(model, "maybe").annotation == Optional[Literal["fast"]]
        assert _field(model, "maybe").default is None
        assert _field(model, "many").annotation == Optional[list[Literal["fast"]]]
        assert _field(model, "mapping").annotation == list[Literal["fast"]]
        assert _field(model, "mapping").is_required()

    def test_unresolvable_receptor_annotation_never_raises(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"strategy": _name(strat)}

            def __init__(self, strategy: "Strateggy | None" = None): ...  # noqa: F821 - typo on purpose

        model = extract_config_schema(Owner)
        assert _field(model, "strategy").annotation == Optional[Literal["fast"]]


# ===========================================================================
# D3 -- owner-aware resolution (af defect 3), TYPE_CHECKING degrade, typos
# ===========================================================================

_REG = """
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class P(Protocol):
    def run(self) -> None: ...

Loop = create_registrar("{registry}", P, discriminator_field="loop_id")
"""

_REACT = """
from __future__ import annotations          # annotations stored as strings
from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from .reg import Loop

@dataclass
class React(metaclass=Loop.Meta):
    __registry_key__ = "react"
    __config_annotated_only__ = True
    max_steps: Annotated[int, Field(ge=1, description="step budget")] = 5
    def run(self): ...
"""

_BRANCH_PLAIN = """
# no PEP 563, and no import of Annotated / Field
from dataclasses import dataclass
from .react import React

@dataclass
class BranchPlain(React):
    __registry_key__ = "branch_plain"
"""


@pytest.fixture
def unique_registry() -> Iterator[str]:
    name = f"xv2_loop_{uuid.uuid4().hex[:8]}"
    yield name
    if name in _REGISTRY_INDEX:
        _deregister(name)


class TestD3OwnerAwareResolution:
    def test_af_defect_3_inherited_pep563_knob(self, make_package, unique_registry) -> None:
        pkg = make_package(
            {"reg": _REG.format(registry=unique_registry), "react": _REACT, "branch_plain": _BRANCH_PLAIN}
        )
        React = importlib.import_module(f"{pkg}.react").React
        BranchPlain = importlib.import_module(f"{pkg}.branch_plain").BranchPlain
        # The bug's precondition: the generated __init__ runs in the subclass module.
        assert BranchPlain.__init__.__globals__["__name__"] == f"{pkg}.branch_plain"

        for cls in (React, BranchPlain):
            model = extract_config_schema(cls)
            assert list(model.model_fields) == ["max_steps"], cls
            field = _field(model, "max_steps")
            assert field.description == "step budget" and field.default == 5
            with pytest.raises(ValidationError):
                model(max_steps=0)

    def test_type_checking_name_degrades_with_marker(self, make_package, registrar) -> None:
        reg = registrar("tc")
        pkg = make_package(
            {
                "mod": """
                from __future__ import annotations
                from typing import TYPE_CHECKING, Annotated, Optional
                from pydantic import Field

                if TYPE_CHECKING:
                    from heavy_sdk import Session

                class Client:
                    def __init__(
                        self,
                        session: Annotated[Optional[Session], Field(description="the session")] = None,
                        retries: int = 3,
                    ) -> None: ...

                    def run(self) -> None: ...
                """
            }
        )
        Client = importlib.import_module(f"{pkg}.mod").Client
        reg.register("client")(Client)
        with pytest.warns(DegradationWarning, match="TYPE_CHECKING"):
            model = extract_config_schema(Client)
        session = _field(model, "session")
        assert session.annotation is Any
        assert session.description == "the session" and session.default is None
        (record,) = model.__degraded_fields__
        assert (record.field_name, record.reason) == ("session", "type_checking_only")
        assert "Session" in record.original_type_repr
        assert record.source_class.endswith("mod.Client")
        assert sys.modules.get("heavy_sdk") is None  # never imported

        # The usual degraded markers reach the artifacts.
        from conscribe.config.codegen import generate_layer_config_source
        from conscribe.config.json_schema import generate_layer_json_schema

        with pytest.warns(DegradationWarning):
            result = build_layer_config(reg)
        assert [d.field_name for d in result.degraded_fields["client"]] == ["session"]
        assert "# degraded from:" in generate_layer_config_source(result)
        assert "x-degraded-fields" in generate_layer_json_schema(result)

    def test_typo_is_an_error(self) -> None:
        class Client:
            def __init__(self, session: "Sesion") -> None: ...  # noqa: F821 - typo on purpose

        with pytest.raises(ConfigAnnotationError) as excinfo:
            extract_config_schema(Client)
        message = str(excinfo.value)
        assert "Client.session" in message and "'Sesion'" in message
        assert "Fix the spelling" in message

    def test_metadata_typo_is_an_error(self) -> None:
        class Client:
            def __init__(self, limit: "Annotated[int, Feild(ge=1)]" = 2) -> None: ...  # noqa: F821

        with pytest.raises(ConfigAnnotationError, match="Feild"):
            extract_config_schema(Client)

    def test_annotated_only_skips_unannotated_typos_but_not_knobs(self) -> None:
        class Skipped:
            __config_annotated_only__ = True

            def __init__(self, helper: "Hlper" = None, n: Annotated[int, Field(ge=0)] = 1) -> None: ...  # noqa: F821

        model = extract_config_schema(Skipped)
        assert list(model.model_fields) == ["n"]

        class Knob:
            __config_annotated_only__ = True

            def __init__(self, n: "Annotated[Nt, Field(ge=0)]" = 1) -> None: ...  # noqa: F821

        with pytest.raises(ConfigAnnotationError, match="'Nt'"):
            extract_config_schema(Knob)

    def test_function_local_names_degrade_with_extraction_warning(self) -> None:
        class LocalType:
            pass

        class Client:
            def __init__(self, thing: Optional[LocalType] = None) -> None: ...

        with pytest.warns(ExtractionWarning, match="local to the function"):
            model = extract_config_schema(Client)
        assert _field(model, "thing").annotation is Any
        assert model.__degraded_fields__[0].reason == "function_locals"

    def test_list_annotation_does_not_crash_on_310(self) -> None:
        """1.x called ``issubclass(list[int], BaseModel)`` (TypeError on 3.10)."""

        class Holder:
            def __init__(self, xs: list[int]) -> None: ...

        model = extract_config_schema(Holder)
        assert _field(model, "xs").annotation == list[int]


# ===========================================================================
# None / Optional normalization
# ===========================================================================


class TestNoneNormalization:
    def test_none_default_admits_none(self) -> None:
        class Holder:
            def __init__(self, x: int = None, y: Annotated[int, Field(ge=1)] = None) -> None: ...  # type: ignore[assignment]

        model = extract_config_schema(Holder)
        assert _field(model, "x").annotation == Optional[int]
        assert model(x=None).x is None
        assert model(y=None).y is None
        with pytest.raises(ValidationError):
            model(y=0)

    def test_optional_annotated_keeps_its_field_info(self) -> None:
        class Holder:
            __config_annotated_only__ = True

            def __init__(
                self, x: Optional[Annotated[int, Field(ge=1, description="x knob")]] = None
            ) -> None: ...

        model = extract_config_schema(Holder)
        field = _field(model, "x")
        assert field.description == "x knob"
        assert model(x=None).x is None
        with pytest.raises(ValidationError):
            model(x=0)

    def test_validators_inside_optional_never_see_none(self) -> None:
        seen: list[Any] = []

        def record(value: int) -> int:
            seen.append(value)
            return value

        def __init__(self: Any, x: Any = None) -> None: ...

        __init__.__annotations__["x"] = Optional[Annotated[int, AfterValidator(record)]]
        Holder = type("Holder", (), {"__init__": __init__})
        model = extract_config_schema(Holder)
        model(x=None)
        model(x=3)
        assert seen == [3]


# ===========================================================================
# F4 -- declare_only and Literal-mode default keys
# ===========================================================================


class TestDeclareOnly:
    def test_declare_only_shape(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"hint": {"registry": _name(strat), "keys": ["fast"], "declare_only": True}}

            def __init__(self, n: int = 1) -> None: ...

        model = extract_config_schema(Owner)
        field = _field(model, "hint")
        assert field.annotation == Optional[Literal["fast"]] and field.default is None
        with pytest.raises(ValidationError):
            model(hint="slow")

    @pytest.mark.parametrize("slot", ["init", "annotation"])
    def test_declare_only_with_a_slot_is_a_contradiction(self, target, slot) -> None:
        strat = target(["fast"])
        namespace: dict[str, Any] = {
            "__wiring__": {"hint": {"registry": _name(strat), "declare_only": True}},
        }
        if slot == "annotation":
            namespace["__annotations__"] = {"hint": Strategy}
        else:
            def __init__(self: Any, hint: Any = None) -> None: ...

            namespace["__init__"] = __init__
        owner = type("Owner", (), namespace)
        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(owner)
        (problem,) = excinfo.value.problems
        assert problem.kind == "contradiction" and problem.param_name == "hint"
        assert "declare_only" in str(excinfo.value)


class TestSlotChecksInResolution:
    """The slot-dependent checks are part of wiring resolution itself, so
    ``resolve_wiring`` / ``validate_wiring`` report them up front, aggregated
    with every other problem (before any model is built)."""

    def test_resolve_wiring_reports_slot_problems_in_declaration_order(self, target) -> None:
        from conscribe.wiring import resolve_wiring

        strat = target(["fast"])

        class Owner:
            __registry_key__ = "owner"
            __wiring__ = {
                "a": (_name(strat), ["nope"]),
                "hint": {"registry": _name(strat), "declare_only": True},
                "many": {"registry": _name(strat), "default": "fast"},
                "ghost": {"registry": _name(strat), "default": "fast"},
            }
            hint: Strategy
            many: list[Strategy]

        with pytest.raises(WiringResolutionError) as excinfo:
            resolve_wiring(Owner)
        assert [(p.param_name, p.kind) for p in excinfo.value.problems] == [
            ("a", "unknown_key"),
            ("hint", "contradiction"),
            ("many", "bad_default"),
            ("ghost", "bad_default"),
        ]
        message = str(excinfo.value)
        assert "Owner (key 'owner') wiring 'hint': the entry is declare_only" in message
        assert "class annotation 'hint' on" in message
        assert "single-instance receptor" in message and "no runtime slot" in message

    def test_validate_wiring_aggregates_across_classes(self, registrar, target) -> None:
        from conscribe.wiring import validate_wiring

        strat = target(["fast"])
        own = registrar("owner")

        class OwnerBase(metaclass=own.Meta):
            __abstract__ = True

            def run(self): ...

        class First(OwnerBase):
            __registry_key__ = "first"
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "fast"}}  # slot-less

        class Second(OwnerBase):
            __registry_key__ = "second"
            __wiring__ = {"hint": {"registry": _name(strat), "declare_only": True}}

            def __init__(self, hint: Any = None) -> None: ...

        with pytest.raises(WiringResolutionError) as excinfo:
            validate_wiring(own)
        assert [(p.owner_key, p.kind) for p in excinfo.value.problems] == [
            ("first", "bad_default"),
            ("second", "contradiction"),
        ]
        with pytest.raises(WiringResolutionError):
            build_layer_config(own)  # validate-first: before any model is built

    def test_valid_entries_need_no_slot_lookup_problems(self, target) -> None:
        from conscribe.wiring import resolve_wiring

        strat = target(["fast"])

        class Owner:
            __wiring__ = {
                "strategy": {"registry": _name(strat), "default": "fast"},
                "hint": {"registry": _name(strat), "declare_only": True},
            }
            strategy: Strategy

        resolved = resolve_wiring(Owner)
        assert resolved["strategy"].default_key == "fast"
        assert resolved["hint"].declare_only is True


class TestOptionalAnnotationSlots:
    def test_class_body_none_default_is_optional(self, target) -> None:
        """AB2 keeps ``archive: Optional[X] = None`` as None at runtime; the
        config field is ``Optional[...] = None`` accordingly."""
        caps = target(["net", "disk"], prefix="arch")

        class Owner:
            __wiring__ = {"archive": _name(caps), "archives": _name(caps)}
            archive: Optional[Strategy] = None
            archives: Optional[list[Strategy]] = None

            def __init__(self, n: int = 1) -> None: ...

        model = extract_config_schema(Owner)
        archive, archives = _field(model, "archive"), _field(model, "archives")
        assert archive.annotation == Optional[Literal["net", "disk"]] and archive.default is None
        assert archives.annotation == Optional[list[Literal["net", "disk"]]]
        assert archives.default is None
        assert model().archive is None
        assert model(archive="disk").archive == "disk"


class TestDefaultKey:
    def test_annotation_slot_default(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "fast"}}
            strategy: Strategy

        model = extract_config_schema(Owner)
        field = _field(model, "strategy")
        assert field.annotation == Literal["fast", "slow"] and field.default == "fast"
        assert model.__wired_defaults__ == {"strategy": "fast"}
        assert model().strategy == "fast"
        with pytest.raises(ValidationError):
            model(strategy=None)  # the receptor does not admit None

    def test_nullable_receptor_is_tri_state(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "fast"}}
            strategy: Strategy | None

        model = extract_config_schema(Owner)
        assert _field(model, "strategy").annotation == Optional[Literal["fast", "slow"]]
        assert model().strategy == "fast"
        assert model(strategy=None).strategy is None

    def test_init_param_none_default_is_tri_state(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "slow"}}

            def __init__(self, strategy: Strategy | None = None) -> None: ...

        model = extract_config_schema(Owner)
        assert model().strategy == "slow"
        assert model(strategy=None).strategy is None

    def test_equal_param_default_is_fine(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "slow"}}

            def __init__(self, strategy: str = "slow") -> None: ...

        assert extract_config_schema(Owner)().strategy == "slow"

    def test_different_param_default_is_an_error(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "slow"}}

            def __init__(self, strategy: str = "fast") -> None: ...

        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(Owner)
        (problem,) = excinfo.value.problems
        assert problem.kind == "bad_default"
        assert "'fast' names a different key than the wiring default 'slow'" in problem.detail

    def test_default_on_container_receptor_is_an_error(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"strategies": {"registry": _name(strat), "default": "fast"}}
            strategies: list[Strategy]

        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(Owner)
        assert [p.kind for p in excinfo.value.problems] == ["bad_default"]
        assert "single-instance" in str(excinfo.value)

    def test_default_on_slot_less_entry_is_an_error(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"hint": {"registry": _name(strat), "default": "fast"}}

        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(Owner)
        assert [p.kind for p in excinfo.value.problems] == ["bad_default"]
        assert "no runtime slot" in str(excinfo.value)

    def test_unregistered_default_is_an_error(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"strategy": {"registry": _name(strat), "default": "fsat"}}
            strategy: Strategy

        with pytest.raises(WiringResolutionError, match="default 'fsat' is not registered"):
            extract_config_schema(Owner)

    def test_param_default_that_is_not_a_key_is_an_error(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"strategy": _name(strat)}

            def __init__(self, strategy: str = "auto") -> None: ...

        with pytest.raises(WiringResolutionError) as excinfo:
            extract_config_schema(Owner)
        (problem,) = excinfo.value.problems
        assert problem.kind == "bad_default" and "'auto'" in problem.detail

    def test_container_receptor_list_default(self, target) -> None:
        strat = target(["fast", "slow"])

        class Owner:
            __wiring__ = {"strategies": _name(strat)}

            def __init__(self, strategies: tuple[Strategy, ...] = ()) -> None: ...

        model = extract_config_schema(Owner)
        field = _field(model, "strategies")
        assert field.annotation == list[Literal["fast", "slow"]]
        assert field.default == []  # validated into the field's own shape
        assert "strategies" not in getattr(model, "__wired_defaults__", {})


# ===========================================================================
# D6 -- single-param BaseModel: nested by default, opt-in flattening
# ===========================================================================


class UserModel(BaseModel):
    x: int = 1
    y: str = Field("a", description="the y knob")


class AliasedModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    value: int = Field(0, alias="val")


class CheckedModel(BaseModel):
    lo: int = 0
    hi: int = 10

    @model_validator(mode="after")
    def ordered(self) -> "CheckedModel":
        if self.lo > self.hi:
            raise ValueError("lo must not exceed hi")
        return self


class TestD6Nested:
    def _layer(self, registrar) -> tuple[Any, type]:
        own = registrar("owner")

        class OwnerBase(metaclass=own.Meta):
            __abstract__ = True

            def run(self): ...

        return own, OwnerBase

    def test_af_defect_6_owner_single_is_nested(self, registrar) -> None:
        own, OwnerBase = self._layer(registrar)

        class OwnerSingle(OwnerBase):
            __registry_key__ = "owner_single"

            def __init__(
                self, matcher: Annotated[UserModel, Field(description="matcher")] = UserModel()
            ): ...

        model = extract_config_schema(OwnerSingle)
        assert model is not UserModel and list(model.model_fields) == ["matcher"]
        field = _field(model, "matcher")
        assert field.annotation is UserModel and field.description == "matcher"
        assert field.default == UserModel()
        assert model.model_validate({"matcher": {"x": 3}}).matcher.x == 3
        assert model.__legacy_flat_keys__ == {"param": "matcher", "keys": ("x", "y")}

        # Old flat keys get the targeted message -- also on the per-key model
        # (which subclasses the extracted model) and the composed rebuild.
        per_key = build_layer_config(own).per_key_models["owner_single"]
        composed = build_composed_config({_name(own): own}).top_level_type
        expected = (
            "'matcher' is a nested block in conscribe 2.0; write {matcher: {...}} "
            "or set __config_from_param__ = 'matcher'"
        )
        for validate in (
            lambda: model.model_validate({"x": 3}),
            lambda: per_key.model_validate({"owner_id": "owner_single", "x": 3}),
            lambda: composed.model_validate({_name(own): [{"owner_id": "owner_single", "y": "b"}]}),
        ):
            with pytest.raises(ValidationError) as excinfo:
                validate()
            (error,) = excinfo.value.errors()
            assert error["type"] == "legacy_flat_config"
            assert error["msg"] == expected

    def test_guard_also_fires_when_extra_is_allowed(self) -> None:
        class Owner:
            def __init__(self, config: UserModel, **kwargs: Any) -> None: ...

        model = extract_config_schema(Owner)
        assert model.model_config.get("extra") == "allow"
        assert model.model_validate({"config": {"x": 2}, "other": 1}).config.x == 2
        with pytest.raises(ValidationError, match="nested block"):
            model.model_validate({"x": 2})

    def test_annotated_only_keeps_the_formerly_flattened_param(self) -> None:
        """1.x flattened before the annotated-only filter; the block must stay
        writable in 2.0 (otherwise every knob of the class would vanish)."""

        class Owner:
            __config_annotated_only__ = True

            def __init__(self, config: UserModel) -> None: ...

        model = extract_config_schema(Owner)
        assert list(model.model_fields) == ["config"]
        assert model.model_validate({"config": {"x": 2}}).config.x == 2
        with pytest.raises(ValidationError, match="nested block"):
            model.model_validate({"x": 2})

    def test_guard_only_for_the_one_param_shape(self) -> None:
        class Two:
            def __init__(self, config: UserModel, n: int = 0) -> None: ...

        assert not hasattr(extract_config_schema(Two), "__legacy_flat_keys__")

    def test_guard_accepts_aliases(self) -> None:
        class Owner:
            def __init__(self, cfg: AliasedModel) -> None: ...

        model = extract_config_schema(Owner)
        assert set(model.__legacy_flat_keys__["keys"]) == {"value", "val"}
        with pytest.raises(ValidationError, match="nested block"):
            model.model_validate({"val": 1})


class TestD6OptIn:
    def test_opt_in_restores_the_flat_shape(self, registrar, target) -> None:
        strat = target(["fast"])
        own = registrar("owner")

        class OwnerBase(metaclass=own.Meta):
            __abstract__ = True

            def run(self): ...

        class OwnerFlat(OwnerBase):
            __registry_key__ = "owner_flat"
            __config_from_param__ = "matcher"
            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy

            def __init__(
                self, matcher: Annotated[UserModel, Field(description="the matcher")] = UserModel(x=5)
            ): ...

        model = extract_config_schema(OwnerFlat)
        assert issubclass(model, UserModel)
        assert model.__config_from_param__ == "matcher"
        assert list(model.model_fields) == ["x", "y", "strategy"]
        assert _field(model, "x").default == 5  # the parameter default wins
        assert _field(model, "y").description == "the y knob"
        assert model.__doc__ == "the matcher"
        assert model.__wired_fields__ == {"strategy": _name(strat)}
        per_key = build_layer_config(own).per_key_models["owner_flat"]
        assert per_key.__config_from_param__ == "matcher"
        validated = per_key.model_validate({"owner_id": "owner_flat", "strategy": "fast"})
        assert (validated.x, validated.y) == (5, "a")
        with pytest.raises(ValidationError):
            per_key.model_validate({"owner_id": "owner_flat", "strategy": "fast", "matcher": {}})

    def test_flattened_model_validators_run(self) -> None:
        class Owner:
            __config_from_param__ = "limits"

            def __init__(self, limits: CheckedModel) -> None: ...

        model = extract_config_schema(Owner)
        with pytest.raises(ValidationError, match="lo must not exceed hi"):
            model(lo=5, hi=1)

    def test_opt_in_is_inherited_and_none_opts_out(self) -> None:
        class Base:
            __config_from_param__ = "config"

        class Flat(Base):
            def __init__(self, config: UserModel) -> None: ...

        class Nested(Base):
            __config_from_param__ = None

            def __init__(self, config: UserModel) -> None: ...

        assert list(extract_config_schema(Flat).model_fields) == ["x", "y"]
        assert list(extract_config_schema(Nested).model_fields) == ["config"]

    @pytest.mark.parametrize(
        ("init_source", "message"),
        [
            ("def __init__(self, config: UserModel) -> None: ...", "has no parameter 'missing'"),
            ("def __init__(self, missing: int) -> None: ...", "must be annotated with a pydantic BaseModel"),
            ("def __init__(self, missing: Optional[UserModel]) -> None: ...", "not Optional"),
            ("def __init__(self, missing: UserModel = None) -> None: ...", "default is None"),
            ("def __init__(self, missing: UserModel, n: int = 0) -> None: ...", r"also takes \['n'\]"),
            ("def __init__(self, missing: UserModel = 3) -> None: ...", "is not a UserModel instance"),
            ("def __init__(self, missing: RootModel[int]) -> None: ...", "RootModel"),
        ],
    )
    def test_strict_validation(self, init_source: str, message: str) -> None:
        scope: dict[str, Any] = {
            "UserModel": UserModel, "Optional": Optional, "RootModel": RootModel,
        }
        exec(init_source, scope)  # noqa: S102 - builds the signature under test
        owner = type("Owner", (), {"__init__": scope["__init__"], "__config_from_param__": "missing"})
        with pytest.raises(ConfigDeclarationError, match=message):
            extract_config_schema(owner)

    def test_never_applies_to_a_receptor(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __config_from_param__ = "strategy"
            __wiring__ = {"strategy": _name(strat)}

            def __init__(self, strategy: UserModel) -> None: ...

        with pytest.raises(ConfigDeclarationError, match="is a __wiring__ receptor"):
            extract_config_schema(Owner)

    def test_registered_model_cannot_be_flattened(self, registrar) -> None:
        reg = registrar("bm")

        class Registered(BaseModel):
            a: int = 1

            def run(self) -> None: ...

        reg.register("registered")(Registered)

        def __init__(self: Any, cfg: Any) -> None: ...

        __init__.__annotations__["cfg"] = Registered  # a real object, not a PEP 563 string
        owner = type("Owner", (), {"__init__": __init__, "__config_from_param__": "cfg"})
        with pytest.raises(ConfigDeclarationError, match="registered class"):
            extract_config_schema(owner)


# ===========================================================================
# F5 -- real defaults and full metadata fidelity (af defect 8)
# ===========================================================================


SOURCES = ["*"]


def _module_factory() -> list[str]:
    return list(SOURCES)  # uses a module global: impure


class TestF5Defaults:
    def test_af_defect_8_dataclass_factory(self, registrar) -> None:
        reg = registrar("loader")

        class Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self): ...

        @dataclasses.dataclass
        class DcFactory(Base):
            __registry_key__ = "dc_factory"
            sources: list[str] = dataclasses.field(default_factory=lambda: ["*"])

        class InitTuple(Base):
            __registry_key__ = "init_tuple"

            def __init__(
                self, sources: Annotated[tuple[str, ...], Field(description="globs")] = ("*",)
            ): ...

        dc = extract_config_schema(DcFactory)
        field = _field(dc, "sources")
        assert field.default == ["*"] and field.default_factory is None
        assert dc.model_validate({}).sources == ["*"]
        assert dc.model_validate({}).sources is not dc.model_validate({}).sources
        tup = extract_config_schema(InitTuple)
        assert _field(tup, "sources").default == ("*",)
        assert "<factory>" not in repr(dc.model_fields) + repr(tup.model_fields)

    def test_signature_and_field_defaults_conflict(self) -> None:
        class InitFieldFactory:
            def __init__(
                self,
                sources: Annotated[list[str], Field(default_factory=lambda: ["*"], description="globs")] = None,  # type: ignore[assignment]
            ): ...

        with pytest.raises(ConfigDeclarationError, match="two defaults are declared"):
            extract_config_schema(InitFieldFactory)

    def test_dataclass_initvar_and_init_false(self) -> None:
        @dataclasses.dataclass
        class Holder:
            seed: dataclasses.InitVar[int] = 3
            derived: int = dataclasses.field(init=False, default=0)
            name: str = "n"

        model = extract_config_schema(Holder)
        assert list(model.model_fields) == ["seed", "name"]
        seed = _field(model, "seed")
        assert seed.annotation is int and seed.default == 3

    def test_builtin_container_factory(self) -> None:
        @dataclasses.dataclass
        class Holder:
            tags: list[str] = dataclasses.field(default_factory=list)
            meta: dict[str, int] = dataclasses.field(default_factory=dict)

        model = extract_config_schema(Holder)
        assert (_field(model, "tags").default, _field(model, "meta").default) == ([], {})

    def test_impure_factories_stay_factories(self) -> None:
        prefix = "p"

        @dataclasses.dataclass
        class Holder:
            closure: list[str] = dataclasses.field(default_factory=lambda: [prefix])
            module_global: list[str] = dataclasses.field(default_factory=_module_factory)

        model = extract_config_schema(Holder)
        assert _field(model, "closure").default_factory is not None
        assert _field(model, "module_global").default_factory is _module_factory
        assert model().module_global == ["*"]

    def test_pydantic_field_factory(self) -> None:
        class Holder(BaseModel):
            names: list[str] = Field(default_factory=lambda: ["a", "b"], description="names")

        model = extract_config_schema(Holder)
        field = _field(model, "names")
        assert field.default == ["a", "b"] and field.description == "names"

    def test_raising_pure_factory_is_an_error(self) -> None:
        @dataclasses.dataclass
        class Holder:
            n: int = dataclasses.field(default_factory=lambda: int("x"))

        with pytest.raises(ConfigDeclarationError, match="raised ValueError"):
            extract_config_schema(Holder)

    def test_purity_gate(self) -> None:
        counter = [0]

        def counting() -> int:
            counter[0] += 1
            return counter[0]

        assert _is_pure_factory(list) and _is_pure_factory(lambda: {"a": [1, 2]})
        assert _is_pure_factory(lambda: sorted({3, 1}))
        assert not _is_pure_factory(counting)  # closure
        assert not _is_pure_factory(_module_factory)  # module global
        assert not _is_pure_factory(lambda: print("side effect"))  # noqa: T201
        assert not _is_pure_factory(lambda data: data)  # takes data
        assert _evaluate_factory(counting, owner="o", field="f") == (False, None)

    def test_field_default_value_declares_the_field(self) -> None:
        class Holder:
            __config_annotated_only__ = True

            def __init__(self, n: int = Field(2, ge=1, description="n knob"), skipped: int = 0): ...

        model = extract_config_schema(Holder)
        assert list(model.model_fields) == ["n"]
        assert _field(model, "n").default == 2 and _field(model, "n").description == "n knob"
        with pytest.raises(ValidationError):
            model(n=0)

    def test_full_field_info_carry(self) -> None:
        class Holder:
            def __init__(
                self,
                knob: Annotated[
                    int,
                    Field(title="Knob", examples=[1, 2], json_schema_extra={"x-unit": "s"}, alias="the_knob"),
                ] = 1,
                old: Annotated[int, Field(deprecated=True)] = 0,
            ): ...

        model = extract_config_schema(Holder)
        knob = _field(model, "knob")
        assert (knob.title, knob.examples, knob.alias) == ("Knob", [1, 2], "the_knob")
        assert model(the_knob=3).knob == 3  # alias passthrough changes the YAML key
        schema = model.model_json_schema()
        assert schema["properties"]["the_knob"]["x-unit"] == "s"
        assert schema["properties"]["old"]["deprecated"] is True

    def test_exclude_is_an_error(self) -> None:
        class Holder:
            def __init__(self, secret: Annotated[str, Field(exclude=True)] = "") -> None: ...

        with pytest.raises(ConfigDeclarationError, match="exclude"):
            extract_config_schema(Holder)

    def test_attrs_defaults(self) -> None:
        attrs = pytest.importorskip("attrs")

        @attrs.define
        class Holder:
            name: str
            tags: list = attrs.Factory(list)
            size: int = 3

        model = extract_config_schema(Holder)
        assert _field(model, "name").is_required()
        assert _field(model, "tags").default == []
        assert _field(model, "size").default == 3
        assert "NOTHING" not in repr(model.model_fields)

    def test_discriminated_union_knob(self) -> None:
        class Circle(BaseModel):
            kind: Literal["circle"] = "circle"
            r: float = 1.0

        class Square(BaseModel):
            kind: Literal["square"] = "square"
            side: float = 1.0

        def __init__(self: Any, shape: Any = None) -> None: ...

        __init__.__annotations__["shape"] = Annotated[Union[Circle, Square], Field(discriminator="kind")]
        model = extract_config_schema(type("Holder", (), {"__init__": __init__}))
        with pytest.raises(ValidationError, match="union_tag_not_found|discriminator"):
            model(shape={"r": 2.0})


# ===========================================================================
# D9 -- receptors injected from slots keep their description
# ===========================================================================


class TestD9SlotDescriptions:
    def test_annotation_slot_field_description(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Annotated[Strategy, Field(description="how to pick", title="Strategy")]

        field = _field(extract_config_schema(Owner), "strategy")
        assert (field.description, field.title) == ("how to pick", "Strategy")
        assert field.annotation == Literal["fast"]

    def test_annotation_slot_docstring_description(self, target) -> None:
        pytest.importorskip("docstring_parser")
        strat = target(["fast"])

        class Owner:
            """An owner.

            Attributes:
                strategy: The strategy, injected by the framework.
            """

            __wiring__ = {"strategy": _name(strat)}
            strategy: Strategy

        field = _field(extract_config_schema(Owner), "strategy")
        assert field.description == "The strategy, injected by the framework."

    def test_filtered_init_param_keeps_shape_default_and_description(self, target) -> None:
        pytest.importorskip("docstring_parser")
        caps = target(["browser", "docker"], prefix="cap")

        class Environment:
            """An environment.

            Args:
                capabilities: Capabilities owned by the environment.
            """

            __config_annotated_only__ = True
            __wiring__ = {"capabilities": _name(caps)}

            def __init__(self, capabilities: Optional[dict[str, Any]] = None) -> None: ...

        field = _field(extract_config_schema(Environment), "capabilities")
        assert field.annotation == Optional[list[Literal["browser", "docker"]]]
        assert field.default is None
        assert field.description == "Capabilities owned by the environment."

    def test_parameter_field_beats_slot_annotation(self, target) -> None:
        strat = target(["fast"])

        class Owner:
            __wiring__ = {"strategy": _name(strat)}
            strategy: Annotated[Strategy, Field(description="from the class")]

            def __init__(
                self, strategy: Annotated[Strategy, Field(description="from the parameter")]
            ) -> None: ...

        assert _field(extract_config_schema(Owner), "strategy").description == "from the parameter"


# ===========================================================================
# Compatibility helpers kept for the nested-mode builder
# ===========================================================================


class TestCompatHelpers:
    def test_extract_own_init_params_is_owner_aware(self, make_package, unique_registry) -> None:
        pkg = make_package(
            {"reg": _REG.format(registry=unique_registry), "react": _REACT, "branch_plain": _BRANCH_PLAIN}
        )
        React = importlib.import_module(f"{pkg}.react").React
        params, hints = extract_own_init_params(React)
        assert [p.name for p in params] == ["max_steps"]
        assert get_origin(hints["max_steps"]) is Annotated

        class Child(React):
            def __init__(self, extra: int = 0, **kwargs: Any) -> None: ...

        params, hints = extract_own_init_params(Child)
        assert [p.name for p in params] == ["extra"] and hints == {"extra": int}

        class NoOwnInit(React):
            pass

        assert extract_own_init_params(NoOwnInit) == ([], None)

    def test_build_param_fields(self) -> None:
        @dataclasses.dataclass
        class Holder:
            """Holder.

            Args:
                tags: The tags.
            """

            tags: list[str] = dataclasses.field(default_factory=lambda: ["t"])
            n: Annotated[int, Field(ge=0)] = 1

        params = walk_init_chain(Holder).params
        fields, degraded = _build_param_fields(Holder, params)
        assert set(fields) == {"tags", "n"} and degraded == []
        assert fields["tags"][1].default == ["t"]
        only, _ = _build_param_fields(Holder, params, annotated_only=True)
        assert set(only) == {"n"}


# ===========================================================================
# Review fixes (2.0.0 release review)
# ===========================================================================


class TestPurityGateRejectsSideEffects:
    """B3: a factory that imports (and can read the machine's state), defines
    classes, writes globals or walks dunder attributes is impure: it is never
    evaluated at build time and nothing machine-specific is baked into the
    artifacts."""

    def test_imports_and_escapes_are_impure(self, make_package, tmp_path) -> None:
        marker = tmp_path / "side_effect"
        pkg = make_package(
            {
                "fac": f"""
                from __future__ import annotations
                import dataclasses
                from typing import Annotated
                from pydantic import Field

                def default_home() -> str:
                    import os
                    return os.environ.get("HOME", "?")

                def default_cwd() -> str:
                    from os import getcwd
                    return getcwd()

                def make_marker() -> str:
                    import pathlib
                    pathlib.Path({str(marker)!r}).write_text("touched")
                    return "x"

                def builds_class():
                    class Local:
                        pass
                    return []

                def escapes():
                    return ().__class__.__base__.__subclasses__()

                def pure():
                    return {{"a": [i for i in range(3)]}}

                @dataclasses.dataclass
                class Agent:
                    home: str = dataclasses.field(default_factory=default_home)
                    cwd: str = dataclasses.field(default_factory=default_cwd)
                    marker: str = dataclasses.field(default_factory=make_marker)
                    table: dict = dataclasses.field(default_factory=pure)

                class Agent2:
                    def __init__(self, home: Annotated[str, Field(default_factory=default_home)]) -> None:
                        pass
                """
            }
        )
        module = importlib.import_module(f"{pkg}.fac")
        for factory in (
            module.default_home, module.default_cwd, module.make_marker,
            module.builds_class, module.escapes,
        ):
            assert not _is_pure_factory(factory), factory.__name__
        assert _is_pure_factory(module.pure)
        assert _is_pure_factory(lambda: [1, 2])

        model = extract_config_schema(module.Agent)
        for name in ("home", "cwd", "marker"):
            info = _field(model, name)
            assert info.default_factory is getattr(module, {"home": "default_home", "cwd": "default_cwd", "marker": "make_marker"}[name])
        assert _field(model, "table").default == {"a": [0, 1, 2]}
        assert not marker.exists()  # nothing ran at build time
        props = model.model_json_schema()["properties"]
        assert "default" not in props["home"] and "default" not in props["cwd"]
        assert _field(extract_config_schema(module.Agent2), "home").default_factory is module.default_home


class TestCooperativeMixinModels:
    """B4: registered pydantic models with a forwarding mixin or a model
    ``__init__`` with named parameters keep every field."""

    def test_mixin_model_in_a_layer(self, registrar) -> None:
        tools = registrar("tool", protocol=_Proto)

        class SetupMixin:
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

        @tools.register("search")
        class Search(SetupMixin, BaseModel):
            name: str = Field("t", description="tool name")
            retries: int = 3

            def run(self) -> None: ...

        for scope in ("local", "all"):
            model = extract_config_schema(Search, mro_scope=scope)
            assert list(model.model_fields) == ["name", "retries"]
            assert model.model_config.get("extra") == "forbid"
            assert _field(model, "name").description == "tool name"
        per_key = build_layer_config(tools).per_key_models["search"]
        assert per_key.model_validate({"tool_id": "search", "name": "x", "retries": 5}).retries == 5
        with pytest.raises(ValidationError):
            per_key.model_validate({"tool_id": "search", "nope": 1})

    def test_model_init_with_named_parameter(self) -> None:
        class Base(BaseModel):
            a: int = 1

            def __init__(self, flag: bool = False, **data: Any) -> None:
                super().__init__(**data)

        class Leaf(Base):
            b: int = 2

        for scope in ("local", "all"):
            model = extract_config_schema(Leaf, mro_scope=scope)
            assert list(model.model_fields) == ["flag", "a", "b"]
            assert model.model_config.get("extra") == "forbid"

    def test_model_extra_setting_survives_the_named_parameter_path(self) -> None:
        class Base(BaseModel):
            model_config = ConfigDict(extra="allow")
            a: int = 1

            def __init__(self, flag: bool = False, **data: Any) -> None:
                super().__init__(**data)

        assert extract_config_schema(Base).model_config.get("extra") == "allow"


class TestCutChainsStayOpen:
    """m5: a chain cut by scope / depth (or a constructor that takes any
    keyword) with no knob of its own is an *open* empty model -- ``None``
    would become the builder's closed discriminator-only model."""

    def test_depth_cut_without_own_knobs(self) -> None:
        class Parent:
            def __init__(self, host: str = "h", **kwargs: Any) -> None: ...

        class Child(Parent):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

        cut = extract_config_schema(Child, mro_depth=0)
        assert cut is not None and cut.model_fields == {}
        assert cut.model_config.get("extra") == "allow"
        assert cut(host="x").model_extra == {"host": "x"}
        full = extract_config_schema(Child)
        assert list(full.model_fields) == ["host"] and full.model_config.get("extra") == "forbid"

    def test_layer_accepts_what_the_constructor_accepts(self, registrar) -> None:
        layer = registrar("open")

        class Parent:
            def __init__(self, host: str = "h", **kwargs: Any) -> None: ...

        @layer.register("child")
        class Child(Parent):
            __config_mro_depth__ = 0

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

            def run(self) -> None: ...

        per_key = build_layer_config(layer).per_key_models["child"]
        assert per_key.model_validate({"open_id": "child", "host": "x"}).model_extra == {"host": "x"}


class TestIncompleteModelMetadata:
    """M4: a registered model whose annotations still name a later class
    keeps the ``Annotated`` metadata (description, constraints, factory)."""

    def test_forward_ref_to_a_later_class(self, make_package) -> None:
        pkg = make_package(
            {
                "fwd": """
                from __future__ import annotations
                from typing import Annotated
                from pydantic import BaseModel, Field

                class Owner(BaseModel):
                    ref: Annotated[Later, Field(description="the later model")] = None
                    n: Annotated[int, Field(ge=1, description="count")] = 1
                    m: Annotated[Later, Field(description="must be set")]
                    sub: Annotated[Later, Field(default_factory=lambda: Later(), description="sub-config")]
                    assigned: Later = Field(None, description="assigned")

                class Later(BaseModel):
                    v: int = 0
                """
            }
        )
        module = importlib.import_module(f"{pkg}.fwd")
        assert not module.Owner.__pydantic_complete__
        model = extract_config_schema(module.Owner)
        fields = model.model_fields
        assert fields["ref"].description == "the later model" and fields["ref"].default is None
        assert fields["n"].description == "count"
        assert fields["m"].description == "must be set" and fields["m"].is_required()
        assert fields["sub"].description == "sub-config" and not fields["sub"].is_required()
        assert fields["assigned"].description == "assigned"
        assert model(m={"v": 1}).sub == module.Later()
        with pytest.raises(ValidationError):
            model(m={"v": 1}, n=0)


class TestRegisteredModelExclude:
    """M5: ``Field(exclude=True)`` on a registered model's field is the
    model's serialization choice, not a config error."""

    def test_secret_field(self) -> None:
        from pydantic import SecretStr

        class Provider(BaseModel):
            model: str = "gpt"
            api_key: Optional[SecretStr] = Field(default=None, exclude=True, description="secret")

        config = extract_config_schema(Provider)
        assert list(config.model_fields) == ["model", "api_key"]
        assert config.model_fields["api_key"].exclude is None
        assert config.model_fields["api_key"].description == "secret"

    def test_constructor_parameter_exclude_is_still_refused(self) -> None:
        class Agent:
            def __init__(self, key: Annotated[str, Field(exclude=True)] = "") -> None: ...

        with pytest.raises(ConfigDeclarationError, match="exclude"):
            extract_config_schema(Agent)


def _positive(value: int) -> int:
    if value <= 0:
        raise ValueError("must be positive")
    return value


class _NoSchema:
    """A type pydantic cannot build a schema for."""


class TestValidatorsNeverSeeNone:
    """m1 at the extractor level."""

    def test_none_default_with_after_validator(self) -> None:
        class Knobs:
            def __init__(self, n: Annotated[int, AfterValidator(_positive)] = None) -> None:  # type: ignore[assignment]
                pass

        model = extract_config_schema(Knobs)
        assert model(n=None).n is None and model().n is None
        assert model(n=5).n == 5
        with pytest.raises(ValidationError, match="must be positive"):
            model(n=-1)


class TestDictReceptorKeys:
    """m2: mapping keys of a container receptor are role names, not registry
    keys; only element / value types are checked against the wiring."""

    def test_role_keyed_dict(self, target) -> None:
        reg = target(["greedy", "random"])
        name = _name(reg)

        class RoleKeyed:
            __wiring__ = {"helpers": name}

            def __init__(self, helpers: dict[Literal["primary", "backup"], _Proto]) -> None: ...

        model = extract_config_schema(RoleKeyed)
        assert _literal_args(_field(model, "helpers").annotation) == ("greedy", "random")

    def test_literal_values_are_still_checked(self, target) -> None:
        reg = target(["greedy", "random"])
        name = _name(reg)

        class BadValues:
            __wiring__ = {"helpers": name}

            def __init__(self, helpers: dict[str, Literal["zzz"]]) -> None: ...

        class BadElements:
            __wiring__ = {"helpers": name}

            def __init__(self, helpers: list[Literal["zzz"]]) -> None: ...

        for cls in (BadValues, BadElements):
            with pytest.raises(ConfigDeclarationError, match="zzz"):
                extract_config_schema(cls)


class TestMroSettingsValidation:
    """m4: invalid ``__config_mro_scope__`` / ``__config_mro_depth__`` (or
    registrar options) raise ``ConfigDeclarationError`` naming the class."""

    def test_scope_typo(self) -> None:
        class Agent:
            __config_mro_scope__ = "locl"

            def __init__(self, a: int = 1) -> None: ...

        with pytest.raises(ConfigDeclarationError) as exc:
            extract_config_schema(Agent)
        message = str(exc.value)
        assert "Agent" in message and "'locl'" in message and "did you mean 'local'" in message

    def test_depth_string_and_inherited(self) -> None:
        class Base:
            __config_mro_depth__ = "2"

            def __init__(self, a: int = 1) -> None: ...

        class Leaf(Base):
            pass

        with pytest.raises(ConfigDeclarationError, match=r"inherited from .*Base.*non-negative int"):
            extract_config_schema(Leaf)

    def test_registrar_option(self) -> None:
        class Agent:
            def __init__(self, a: int = 1) -> None: ...

        with pytest.raises(ConfigDeclarationError, match="registrar's mro_scope='nope'"):
            extract_config_schema(Agent, mro_scope="nope")
        with pytest.raises(ConfigDeclarationError, match="registrar's mro_depth=-1"):
            extract_config_schema(Agent, mro_depth=-1)


class TestFinalDataclassField:
    """m6: ``Final[T]`` dataclass fields are knobs (pydantic would read
    ``Final`` with a default as a class variable and drop them)."""

    def test_final_field_is_kept(self) -> None:
        @dataclasses.dataclass
        class Settings:
            retries: Final[int] = 3
            name: Final = "x"

        model = extract_config_schema(Settings)
        assert list(model.model_fields) == ["retries", "name"]
        assert _field(model, "retries").annotation is int
        assert _field(model, "retries").default == 3
        assert _field(model, "name").default == "x"

    def test_annotated_final_field(self) -> None:
        try:
            Annotated[Final[str], "m"]
        except TypeError:  # Python 3.10.0-3.10.2 reject it (bpo-46491)
            pytest.skip("typing rejects Annotated[Final[...]] on this Python version")

        @dataclasses.dataclass
        class Settings:
            name: Annotated[Final[str], Field(description="n")] = "x"

        model = extract_config_schema(Settings)
        assert _field(model, "name").annotation is str
        assert _field(model, "name").description == "n"


class TestParametersNoConfigCanPass:
    """m7: ``model_config`` and other ``model_*`` names, positional-only and
    ``_private`` parameters."""

    def test_model_config_name(self) -> None:
        class Agent:
            def __init__(self, model_config: dict = None) -> None:  # type: ignore[assignment]
                pass

        with pytest.raises(ConfigDeclarationError) as exc:
            extract_config_schema(Agent)
        assert "model_config" in str(exc.value) and "BaseModel.model_config" in str(exc.value)

    def test_model_dump_name(self) -> None:
        class Agent:
            def __init__(self, model_dump: bool = False) -> None: ...

        with pytest.raises(ConfigDeclarationError, match="model_dump"):
            extract_config_schema(Agent)

    def test_model_prefixed_names_that_are_not_members_are_fine(self) -> None:
        class Agent:
            def __init__(self, model_name: str = "gpt") -> None: ...

        assert list(extract_config_schema(Agent).model_fields) == ["model_name"]

    def test_positional_only(self) -> None:
        class Optional_:
            def __init__(self, a: int = 1, /, b: int = 2) -> None: ...

        class Required:
            def __init__(self, a: int, /, b: int = 2) -> None: ...

        assert list(extract_config_schema(Optional_).model_fields) == ["b"]
        with pytest.raises(ConfigDeclarationError) as exc:
            extract_config_schema(Required)
        assert "'a'" in str(exc.value) and "positional-only" in str(exc.value)

    def test_private_parameters(self) -> None:
        @dataclasses.dataclass
        class WithCache:
            size: int = 1
            _cache: dict = dataclasses.field(default_factory=dict)

        class Required:
            def __init__(self, _secret: str) -> None: ...

        assert list(extract_config_schema(WithCache).model_fields) == ["size"]
        with pytest.raises(ConfigDeclarationError, match="underscore"):
            extract_config_schema(Required)

    def test_flattened_config_ignores_parameters_no_config_can_pass(self) -> None:
        class Flat:
            __config_from_param__ = "user"

            def __init__(self, user: UserModel, _cache: Optional[dict] = None) -> None: ...

        model = extract_config_schema(Flat)
        assert list(model.model_fields) == ["x", "y"]


class TestAnnotationSlotDefaults:
    """m8: a class-level value is the default of an annotation-slot receptor."""

    def test_class_level_values(self, target) -> None:
        reg = target(["recorder", "trace_writer"], prefix="hook")
        name = _name(reg)

        class Desk:
            __wiring__ = {"hooks": name, "primary": name, "maybe": name, "bare": name}
            hooks: list[_Proto] = []
            primary: _Proto = "recorder"  # type: ignore[assignment]
            maybe: Optional[_Proto] = None
            bare: _Proto

        model = extract_config_schema(Desk)
        hooks, primary = _field(model, "hooks"), _field(model, "primary")
        assert not hooks.is_required() and hooks.default == []
        assert not primary.is_required() and primary.default == "recorder"
        assert _field(model, "maybe").default is None
        assert _field(model, "bare").is_required()
        assert model.__wired_defaults__ == {"primary": "recorder"}  # type: ignore[attr-defined]

    def test_bad_class_level_value(self, target) -> None:
        reg = target(["recorder"], prefix="hook")
        name = _name(reg)

        class Desk:
            __wiring__ = {"primary": name}
            primary: _Proto = "zzz"  # type: ignore[assignment]

        with pytest.raises(WiringResolutionError, match="'zzz'"):
            extract_config_schema(Desk)


class TestZeroKeyConstraint:
    """m10: key-string constraints that reject every allowed key fail in the
    Literal path too (not only under inline wiring)."""

    def test_pattern_rejects_every_key(self, target) -> None:
        reg = target(["alpha", "beta"])
        name = _name(reg)

        class Own:
            __wiring__ = {"t": name}

            def __init__(self, t: Annotated[_Proto, Field(pattern="^z", description="d")] = None) -> None:  # type: ignore[assignment]
                pass

        with pytest.raises(ConfigDeclarationError) as exc:
            extract_config_schema(Own)
        assert "pattern='^z'" in str(exc.value) and "['alpha', 'beta']" in str(exc.value)

    def test_partial_narrowing_is_fine(self, target) -> None:
        reg = target(["alpha", "beta"])
        name = _name(reg)

        class Own:
            __wiring__ = {"t": name}

            def __init__(self, t: Annotated[_Proto, Field(pattern="^a")] = None) -> None:  # type: ignore[assignment]
                pass

        model = extract_config_schema(Own)
        assert model(t="alpha").t == "alpha"
        with pytest.raises(ValidationError):
            model(t="beta")


class TestWarningLocations:
    """m11: extraction warnings point at the user's class / function."""

    def test_degradation_warning_points_at_the_class(self) -> None:
        class Incompatible:
            def __init__(self, x: _NoSchema = None) -> None:  # type: ignore[assignment]
                pass

        with pytest.warns(DegradationWarning) as record:
            extract_config_schema(Incompatible)
        assert record[0].filename == __file__
        assert record[0].lineno == inspect.getsourcelines(Incompatible)[1]


class TestAttrsChildModule:
    """m3 through the extractor: aliased attrs fields of a base declared in
    another module resolve in that module."""

    def test_child_in_another_module(self, make_package) -> None:
        pytest.importorskip("attrs")
        pkg = make_package(
            {
                "attrs_base": """
                from __future__ import annotations
                import attrs
                from decimal import Decimal

                @attrs.define
                class BaseA:
                    _budget: Decimal = Decimal("1.5")
                    renamed: int = attrs.field(default=3, alias="count")
                """,
                "attrs_child": """
                from __future__ import annotations
                import attrs
                from .attrs_base import BaseA

                @attrs.define
                class ChildA(BaseA):
                    extra: int = 0
                """,
            }
        )
        from decimal import Decimal

        child = importlib.import_module(f"{pkg}.attrs_child").ChildA
        model = extract_config_schema(child)
        assert list(model.model_fields) == ["budget", "count", "extra"]
        assert _field(model, "budget").annotation is Decimal
        assert _field(model, "budget").default == Decimal("1.5")
        assert _field(model, "count").annotation is int
