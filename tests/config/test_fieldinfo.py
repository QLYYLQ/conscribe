"""Tests for conscribe.config._fieldinfo — FieldInfo carry rules.

Covers the two attribute sets:
- ``config_field``: a constructor parameter becomes a config field (F5 carry).
- ``inline_field`` / ``narrow_keys``: the composed inline retype (D9).
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import warnings
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

import annotated_types
import pytest
import typing_extensions
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    WithJsonSchema,
    create_model,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from conscribe.config._fieldinfo import (
    CONFIG_FIELD_ATTRS,
    INLINE_FIELD_ATTRS,
    config_field,
    field_infos_of,
    inline_field,
    inline_metadata,
    is_deprecated,
    is_pydantic_interpreted,
    narrow_keys,
)
from conscribe.exceptions import ConfigDeclarationError
from conscribe.wiring import WiredField


@dataclasses.dataclass(frozen=True)
class TraceSafe:
    """A plain marker a framework reads back from FieldInfo.metadata."""

    reason: str = "ok"


def _model(name: str = "M", **defs: Any) -> type[BaseModel]:
    return create_model(name, **defs)


def _snapshot(fi: FieldInfo) -> tuple:
    return (repr(fi), dict(getattr(fi, "_attributes_set", {})), list(fi.metadata))


# ===================================================================
# Metadata classification
# ===================================================================


class TestClassification:
    @pytest.mark.parametrize(
        "meta",
        [
            annotated_types.Ge(0),
            annotated_types.MinLen(1),
            annotated_types.Len(1, 3),
            annotated_types.Predicate(str.islower),
            StringConstraints(pattern="^n"),
            AfterValidator(lambda v: v),
            WithJsonSchema({"type": "string"}),
            Field(description="x"),
        ],
    )
    def test_pydantic_interpreted(self, meta: Any) -> None:
        assert is_pydantic_interpreted(meta)

    @pytest.mark.parametrize("meta", [TraceSafe(), "a doc string", 42, ("t",)])
    def test_plain_markers(self, meta: Any) -> None:
        assert not is_pydantic_interpreted(meta)

    def test_inline_metadata_keeps_counts_only_on_containers(self) -> None:
        meta = [annotated_types.MinLen(1), TraceSafe(), annotated_types.MaxLen(4), WithJsonSchema({})]
        assert inline_metadata(meta, container=True) == [
            annotated_types.MinLen(1), TraceSafe(), annotated_types.MaxLen(4),
        ]
        assert inline_metadata(meta, container=False) == [TraceSafe()]

    def test_attribute_sets(self) -> None:
        assert "discriminator" in CONFIG_FIELD_ATTRS
        assert "examples" in CONFIG_FIELD_ATTRS
        assert "exclude" not in CONFIG_FIELD_ATTRS
        assert "deprecated" not in CONFIG_FIELD_ATTRS
        assert {"title", "description"} <= set(INLINE_FIELD_ATTRS)
        for dropped in ("discriminator", "examples", "json_schema_extra", "exclude"):
            assert dropped not in INLINE_FIELD_ATTRS


# ===================================================================
# config_field — parameter -> config field
# ===================================================================


class TestConfigFieldCarry:
    def test_carries_documentation_aliases_and_constraints(self) -> None:
        ann = Annotated[
            int,
            Field(
                title="Steps",
                description="max steps",
                examples=[3],
                json_schema_extra={"x-ui": "slider"},
                alias="maxSteps",
                ge=1,
                le=10,
            ),
        ]
        tp, fi = config_field(ann, owner="m.C", field="max_steps", signature_default=5)
        M = _model(max_steps=(tp, fi))
        f = M.model_fields["max_steps"]
        assert f.title == "Steps"
        assert f.description == "max steps"
        assert f.examples == [3]
        assert f.alias == "maxSteps"
        assert f.default == 5
        assert M(maxSteps=7).max_steps == 7
        with pytest.raises(ValidationError):
            M(maxSteps=0)
        props = M.model_json_schema()["properties"]["maxSteps"]
        assert props["minimum"] == 1 and props["maximum"] == 10
        assert props["x-ui"] == "slider"

    def test_carries_markers_and_annotated_metadata_in_order(self) -> None:
        ann = Annotated[str, Field(min_length=2), TraceSafe(), annotated_types.MaxLen(5)]
        tp, fi = config_field(ann, owner="m.C", field="s")
        M = _model(s=(tp, fi))
        meta = M.model_fields["s"].metadata
        assert meta == [annotated_types.MinLen(2), TraceSafe(), annotated_types.MaxLen(5)]
        assert M.model_fields["s"].is_required()

    def test_carries_discriminator(self) -> None:
        class Circle(BaseModel):
            kind: Literal["circle"] = "circle"
            r: float = 1.0

        class Square(BaseModel):
            kind: Literal["square"] = "square"
            side: float = 1.0

        ann = Annotated[Union[Circle, Square], Field(discriminator="kind")]
        tp, fi = config_field(ann, owner="m.C", field="shape")
        M = _model(shape=(tp, fi))
        with pytest.raises(ValidationError) as exc:
            M(shape={"r": 2.0})
        assert exc.value.errors()[0]["type"] == "union_tag_not_found"

    def test_later_field_wins_and_metadata_concatenates(self) -> None:
        ann = Annotated[int, Field(description="first", ge=0), Field(description="second", le=9)]
        tp, fi = config_field(ann, owner="m.C", field="x", signature_default=1)
        M = _model(x=(tp, fi))
        assert M.model_fields["x"].description == "second"
        assert M.model_fields["x"].metadata == [annotated_types.Ge(0), annotated_types.Le(9)]

    def test_docstring_fallback_only_without_field_description(self) -> None:
        _, fi = config_field(int, owner="m.C", field="x", signature_default=1, description="doc")
        assert fi.description == "doc"
        _, fi = config_field(
            Annotated[int, Field(description="own")], owner="m.C", field="x",
            signature_default=1, description="doc",
        )
        assert fi.description == "own"

    def test_user_field_info_not_mutated(self) -> None:
        user_fi = Field(description="d", ge=0, examples=[1], json_schema_extra={"k": 1})
        before = _snapshot(user_fi)
        extra_before = copy.deepcopy(user_fi.json_schema_extra)
        tp, fi = config_field(
            Annotated[int, user_fi], owner="m.C", field="x", signature_default=3,
        )
        _model(x=(tp, fi))
        assert _snapshot(user_fi) == before
        assert user_fi.json_schema_extra == extra_before
        assert fi is not user_fi


class TestConfigFieldRefusals:
    def test_exclude_is_an_error(self) -> None:
        with pytest.raises(ConfigDeclarationError) as exc:
            config_field(Annotated[int, Field(exclude=True)], owner="pkg.Agent", field="x")
        msg = str(exc.value)
        assert "pkg.Agent.x" in msg
        assert "exclude" in msg

    def test_deprecated_becomes_schema_marker_only(self) -> None:
        tp, fi = config_field(
            Annotated[int, Field(deprecated="use y", description="old knob")],
            owner="m.C", field="x", signature_default=1,
        )
        M = _model(x=(tp, fi))
        assert M.model_fields["x"].deprecated is None
        assert is_deprecated(M.model_fields["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert M().x == 1  # no runtime DeprecationWarning on access
        props = M.model_json_schema()["properties"]["x"]
        assert props["deprecated"] is True
        assert props["description"] == "old knob"

    def test_deprecated_marker_merges_with_user_schema_extra(self) -> None:
        _, fi = config_field(
            Annotated[int, Field(deprecated=True, json_schema_extra={"x-a": 1})],
            owner="m.C", field="x", signature_default=1,
        )
        assert fi.json_schema_extra == {"x-a": 1, "deprecated": True}

    def test_deprecated_marker_wraps_callable_schema_extra(self) -> None:
        def extra(schema: dict) -> None:
            schema["x-b"] = 2

        tp, fi = config_field(
            Annotated[int, Field(deprecated=True, json_schema_extra=extra)],
            owner="m.C", field="x", signature_default=1,
        )
        M = _model(x=(tp, fi))
        props = M.model_json_schema()["properties"]["x"]
        assert props["deprecated"] is True and props["x-b"] == 2
        assert is_deprecated(M.model_fields["x"])

    def test_typing_extensions_deprecated_marker(self) -> None:
        _, fi = config_field(
            Annotated[int, typing_extensions.deprecated("old")],
            owner="m.C", field="x", signature_default=1,
        )
        assert fi.json_schema_extra == {"deprecated": True}


class TestConfigFieldDefaults:
    def test_signature_default(self) -> None:
        _, fi = config_field(int, owner="m.C", field="x", signature_default=4)
        assert fi.default == 4

    def test_field_default_without_signature_default(self) -> None:
        _, fi = config_field(Annotated[int, Field(default=7)], owner="m.C", field="x")
        assert fi.default == 7 and not fi.is_required()

    def test_field_default_factory(self) -> None:
        _, fi = config_field(Annotated[list[str], Field(default_factory=list)], owner="m.C", field="x")
        assert fi.default_factory is list and not fi.is_required()

    def test_identical_defaults_are_fine(self) -> None:
        _, fi = config_field(
            Annotated[Optional[int], Field(default=None, description="d")],
            owner="m.C", field="x", signature_default=None,
        )
        assert fi.default is None

    def test_different_defaults_are_an_error(self) -> None:
        with pytest.raises(ConfigDeclarationError) as exc:
            config_field(
                Annotated[int, Field(default=1)], owner="pkg.Agent", field="x", signature_default=2,
            )
        msg = str(exc.value)
        assert "pkg.Agent.x" in msg and "signature default 2" in msg and "default 1" in msg

    def test_signature_default_plus_factory_is_an_error(self) -> None:
        with pytest.raises(ConfigDeclarationError, match="default_factory=list"):
            config_field(
                Annotated[list, Field(default_factory=list)],
                owner="m.C", field="x", signature_default=None,
            )

    def test_dataclass_factory_is_resolved_by_caller(self) -> None:
        @dataclasses.dataclass
        class D:
            tags: list[str] = dataclasses.field(default_factory=list)

        sentinel = inspect.signature(D).parameters["tags"].default
        with pytest.raises(TypeError, match="signature_factory"):
            config_field(list[str], owner="m.D", field="tags", signature_default=sentinel)
        _, fi = config_field(
            list[str], owner="m.D", field="tags",
            signature_default=sentinel, signature_factory=list,
        )
        assert fi.default_factory is list

    @pytest.mark.parametrize(
        "sentinel",
        [inspect.Parameter.empty, dataclasses.MISSING, PydanticUndefined, ..., WiredField("x", "reg")],
    )
    def test_sentinels_mean_required(self, sentinel: Any) -> None:
        _, fi = config_field(int, owner="m.C", field="x", signature_default=sentinel)
        assert fi.is_required()

    def test_none_default_admits_none(self) -> None:
        tp, fi = config_field(int, owner="m.C", field="x", signature_default=None)
        assert tp == Optional[int]
        M = _model(x=(tp, fi))
        assert M(x=None).x is None

    def test_optional_annotated_is_hoisted(self) -> None:
        ann = Optional[Annotated[int, Field(ge=0, description="d"), AfterValidator(lambda v: v)]]
        assert len(field_infos_of(ann)) == 1
        tp, fi = config_field(ann, owner="m.C", field="x", signature_default=None)
        M = _model(x=(tp, fi))
        f = M.model_fields["x"]
        assert f.description == "d"
        assert M(x=None).x is None
        assert M(x=3).x == 3
        with pytest.raises(ValidationError):
            M(x=-1)


def _positive(value: int) -> int:
    if value <= 0:
        raise ValueError("must be positive")
    return value


class TestNoneRule:
    """``config_field`` is the one implementation of the 2.0 None rule."""

    def test_none_default_makes_optional(self) -> None:
        tp, fi = config_field(int, owner="m.C", field="x", signature_default=None)
        assert tp == Optional[int] and fi.default is None

    def test_field_info_stays_on_the_field(self) -> None:
        info = Field(description="d", ge=1)
        tp, fi = config_field(Annotated[int, info], owner="m.C", field="x", signature_default=None)
        assert get_origin(tp) is Annotated
        assert get_args(tp)[0] == Optional[int]
        assert fi.description == "d"
        M = _model(x=(tp, fi))
        assert M(x=None).x is None
        with pytest.raises(ValidationError):
            M(x=0)

    @pytest.mark.parametrize("shape", ["optional", "union", "pep604"])
    def test_optional_annotated_field_info_is_hoisted(self, shape: str) -> None:
        info = Field(description="d")
        ann = {
            "optional": Optional[Annotated[int, info]],
            "union": Union[Annotated[int, info], None],
            "pep604": Annotated[int, info] | None,
        }[shape]
        tp, fi = config_field(ann, owner="m.C", field="x")
        assert tp == Optional[int]
        assert fi.description == "d"

    def test_hoisted_under_an_outer_annotated(self) -> None:
        """``Annotated[Optional[Annotated[T, Field()]], Field()]`` (a Field
        default value wrapped around an Optional annotation)."""
        ann = Annotated[Optional[Annotated[int, Field(description="d")]], Field(title="T")]
        tp, fi = config_field(ann, owner="m.C", field="x", signature_default=None)
        assert tp == Optional[int]
        assert (fi.description, fi.title) == ("d", "T")

    def test_unchanged_cases(self) -> None:
        for ann in (Optional[int], Any, object, Union[int, str, None]):
            tp, _ = config_field(ann, owner="m.C", field="x", signature_default=None)
            assert tp == ann
        assert config_field(int, owner="m.C", field="x")[0] is int
        assert config_field(int, owner="m.C", field="x", signature_default=3)[0] is int
        multi = Union[Annotated[int, "a"], str, None]
        assert config_field(multi, owner="m.C", field="x", signature_default=None)[0] == multi

    def test_unconstrained_typevar_admits_none(self) -> None:
        import typing

        free = typing.TypeVar("free")
        bound = typing.TypeVar("bound", bound=int)
        assert config_field(free, owner="m.C", field="x", signature_default=None)[0] is free
        assert config_field(bound, owner="m.C", field="x", signature_default=None)[0] == Optional[bound]

    def test_validators_never_see_none(self) -> None:
        """m1: ``x: Annotated[int, AfterValidator(f)] = None`` -- an explicit
        null must not reach the validator (it used to raise a raw TypeError)."""
        validator = AfterValidator(_positive)
        tp, fi = config_field(
            Annotated[int, validator, Field(description="d", ge=-10)],
            owner="m.C", field="x", signature_default=None,
        )
        assert get_origin(tp) is Annotated
        inner, *outer = get_args(tp)
        assert inner == Optional[Annotated[int, validator]]
        assert validator not in outer  # constraints stay outside, validators inside
        M = _model(x=(tp, fi))
        assert M(x=None).x is None
        assert M().x is None
        assert M(x=5).x == 5
        with pytest.raises(ValidationError, match="must be positive"):
            M(x=-1)
        with pytest.raises(ValidationError):
            M(x=-20)  # the Field(ge=-10) constraint still applies
        assert M.model_fields["x"].description == "d"

    def test_markers_stay_on_the_field(self) -> None:
        marker = TraceSafe()
        tp, fi = config_field(
            Annotated[int, marker], owner="m.C", field="x", signature_default=None,
        )
        assert get_args(tp) == (Optional[int], marker)
        assert _model(x=(tp, fi)).model_fields["x"].metadata == [marker]

    def test_model_field_exclude_is_dropped_not_refused(self) -> None:
        """M5: ``api_key: SecretStr = Field(exclude=True)`` on a registered
        pydantic model is the model's serialization choice."""
        info = Field(default=None, exclude=True, description="secret")
        tp, fi = config_field(Annotated[Optional[str], info], owner="m.P", field="k", model_field=True)
        assert fi.exclude is None and fi.description == "secret"
        assert _model(k=(tp, fi))(k="s").model_dump() == {"k": "s"}
        with pytest.raises(ConfigDeclarationError, match="exclude"):
            config_field(Annotated[Optional[str], info], owner="m.P", field="k")


# ===================================================================
# inline_field — composed inline retype
# ===================================================================


class _A(BaseModel):
    kind: Literal["a"] = "a"
    x: int = 0


class _B(BaseModel):
    kind: Literal["b"] = "b"
    y: int = 0


_UNION = Annotated[Union[_A, _B], Field(discriminator="kind")]


class TestInlineField:
    def _src(self, ann: Any, default: Any) -> FieldInfo:
        return _model(src=(ann, default)).model_fields["src"]

    def test_keeps_docs_aliases_and_drops_examples(self) -> None:
        src = self._src(
            Literal["a", "b"],
            Field(
                "a", title="Sink", description="where to", examples=["a"], alias="snk",
                json_schema_extra={"enum": ["a", "b"], "x-ui": 1},
            ),
        )
        tp, fi = inline_field(src, _UNION, container=False)
        M = _model(sink=(tp, fi))
        f = M.model_fields["sink"]
        assert (f.title, f.description, f.alias) == ("Sink", "where to", "snk")
        assert f.examples is None
        assert f.json_schema_extra is None
        assert f.default == "a"

    def test_scalar_union_keeps_its_discriminator(self) -> None:
        src = self._src(Literal["a", "b"], Field(description="d"))
        tp, fi = inline_field(src, _UNION, container=False)
        M = _model(sink=(tp, fi))
        assert M.model_fields["sink"].discriminator == "kind"
        with pytest.raises(ValidationError) as exc:
            M(sink={"x": 1})
        assert exc.value.errors()[0]["type"] == "union_tag_not_found"

    def test_list_receptor_keeps_count_constraints(self) -> None:
        src = self._src(Optional[list[Literal["a", "b"]]], Field(None, min_length=1, description="d"))
        tp, fi = inline_field(src, Optional[list[_UNION]], container=True)
        M = _model(sink=(tp, fi))
        with pytest.raises(ValidationError) as exc:
            M(sink=[])
        assert exc.value.errors()[0]["type"] == "too_short"
        assert M(sink=None).sink is None
        assert M(sink=[{"kind": "b"}]).sink[0].y == 0

    def test_scalar_receptor_drops_key_string_constraints(self) -> None:
        src = self._src(Literal["a", "b"], Field(max_length=1, pattern="^a"))
        tp, fi = inline_field(src, _UNION, container=False)
        M = _model(sink=(tp, fi))
        # No raw TypeError from a string constraint applied to a model union.
        assert M(sink={"kind": "a"}).sink.kind == "a"

    def test_plain_markers_survive_interpreted_metadata_dropped(self) -> None:
        src = self._src(
            Annotated[Literal["a", "b"], TraceSafe(), WithJsonSchema({"type": "string"}), AfterValidator(str)],
            "a",
        )
        tp, fi = inline_field(src, _UNION, container=False)
        M = _model(sink=(tp, fi))
        assert M.model_fields["sink"].metadata == [TraceSafe()]
        # WithJsonSchema would have replaced the union's schema with a string.
        assert "oneOf" in M.model_json_schema()["properties"]["sink"]

    def test_default_factory_is_carried(self) -> None:
        src = self._src(list[Literal["a"]], Field(default_factory=list, description="d"))
        tp, fi = inline_field(src, list[_UNION], container=True)
        M = _model(sink=(tp, fi))
        f = M.model_fields["sink"]
        assert not f.is_required() and f.default_factory is list
        assert M().sink == []

    def test_deprecated_marker_is_carried(self) -> None:
        src = self._src(Literal["a"], Field("a", json_schema_extra={"deprecated": True, "examples": ["a"]}))
        _, fi = inline_field(src, _UNION, container=False)
        assert fi.json_schema_extra == {"deprecated": True}
        src = self._src(Literal["a"], Field("a", deprecated="gone"))
        _, fi = inline_field(src, _UNION, container=False)
        assert fi.json_schema_extra == {"deprecated": True} and fi.deprecated is None

    def test_overrides_for_default_instances(self) -> None:
        src = self._src(Literal["a", "b"], Field("a", description="d"))
        _, fi = inline_field(
            src, _UNION, container=False,
            default_factory=_A, json_schema_extra={"default": {"kind": "a", "x": 0}},
        )
        assert fi.default_factory is _A
        assert fi.json_schema_extra == {"default": {"kind": "a", "x": 0}}
        with pytest.raises(TypeError):
            inline_field(src, _UNION, container=False, default=None, default_factory=_A)

    def test_source_field_info_not_mutated(self) -> None:
        src = self._src(Literal["a", "b"], Field("a", description="d", examples=["a"], max_length=3))
        before = _snapshot(src)
        tp, fi = inline_field(src, _UNION, container=False)
        _model(sink=(tp, fi))
        assert _snapshot(src) == before


# ===================================================================
# narrow_keys — scalar receptor key constraints
# ===================================================================


class TestNarrowKeys:
    def test_no_constraints_keeps_all(self) -> None:
        assert narrow_keys(("net", "disk"), [TraceSafe()], owner="m.C", field="f") == ("net", "disk")

    @pytest.mark.parametrize(
        "meta, expected",
        [
            (Field(pattern="^n").metadata, ("net",)),
            (Field(max_length=3).metadata, ("net",)),
            (Field(min_length=4).metadata, ("disk",)),
            ([annotated_types.Len(4, 4)], ("disk",)),
            ([StringConstraints(pattern="k$", to_upper=True)], ("disk",)),
        ],
    )
    def test_constraints_narrow(self, meta: list, expected: tuple) -> None:
        assert narrow_keys(("net", "disk"), meta, owner="m.C", field="f") == expected

    def test_all_constraints_must_pass(self) -> None:
        meta = [annotated_types.MinLen(3), *Field(pattern="^d").metadata]
        assert narrow_keys(("net", "disk", "d"), meta, owner="m.C", field="f") == ("disk",)

    def test_marker_with_pattern_attribute_is_not_a_constraint(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class Marker:
            pattern: str = "^zzz"

        assert narrow_keys(("net",), [Marker()], owner="m.C", field="f") == ("net",)

    def test_rejecting_every_key_is_an_error(self) -> None:
        with pytest.raises(ConfigDeclarationError) as exc:
            narrow_keys(("net", "disk"), Field(pattern="^x").metadata, owner="pkg.Hook", field="sink")
        msg = str(exc.value)
        assert "pkg.Hook.sink" in msg
        assert "pattern='^x'" in msg
        assert "['net', 'disk']" in msg

    def test_matches_per_layer_semantics(self) -> None:
        meta = Field(pattern="^n").metadata
        per_layer = TypeAdapter(Annotated[(Literal["net", "disk"], *meta)])
        kept = narrow_keys(("net", "disk"), meta, owner="m.C", field="f")
        for key in ("net", "disk"):
            try:
                per_layer.validate_python(key)
                accepted = True
            except ValidationError:
                accepted = False
            assert accepted == (key in kept)
