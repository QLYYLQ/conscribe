"""Partial composed builds: ``build_composed_config(..., coverage="partial")``.

A framework that imports only the packages one configuration names builds
its schema from part of the implementations.  The fixtures mirror the two
downstream failures that motivated the mode: an agent whose
``declare_only`` action declaration names actions the imported capability
lacks, and a trace hook whose ``archive`` list receptor targets an
``inline_only`` archiver layer that no imported package implements.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Annotated, Any, Literal, Optional, Protocol, get_args, runtime_checkable

import pytest
from pydantic import BaseModel, Field, ValidationError

import conscribe
from conscribe import (
    EmptyLayerError,
    UnresolvedWiring,
    WiringResolutionError,
    build_composed_config,
    check_wiring_declaration,
    create_registrar,
    validate_wiring,
)
from conscribe.config.codegen import generate_composed_config_source
from conscribe.config.composed import NO_IMPLEMENTATION_ERROR_TYPE, ComposedConfigResult
from conscribe.config.json_schema import generate_composed_json_schema


@runtime_checkable
class RunP(Protocol):
    def run(self) -> None: ...


@pytest.fixture(autouse=True)
def _isolated(conscribe_isolated):
    yield conscribe_isolated


def _layer(name: str, discriminator: str, **options: Any) -> Any:
    """A registrar plus an abstract base (``reg.Base``) to subclass."""
    reg = create_registrar(name, RunP, discriminator_field=discriminator, **options)

    class _Base(metaclass=reg.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    reg.Base = _Base  # type: ignore[attr-defined]
    return reg


def _errors(exc: ValidationError) -> list[tuple[str, tuple[Any, ...]]]:
    return [(e["type"], tuple(e["loc"])) for e in exc.errors()]


def _layers() -> dict[str, Any]:
    """An android-only run: two actions, an agent declaring more, an empty archiver layer.

    The archiver registry exists (its registrar module is imported) but no
    package registered an implementation, so the framework leaves the layer
    out of the build.
    """
    action = _layer("action", "action_id", key_separator=".")
    _layer("archiver", "archiver_id", key_separator=".", inline_only=True)
    agent = _layer("agent", "agent_id")
    hook = _layer("hook", "hook_id")

    class Click(action.Base):
        __registry_key__ = "android.click"

    class Navigate(action.Base):
        __registry_key__ = "android.navigate"

    class Dom(agent.Base):
        __registry_key__ = "dom"
        __wiring__ = {
            "action": {
                "registry": "action",
                "keys": ["navigate", "click"],
                "optional_keys": ["scroll", "send_keys", "drag", "wait"],
                "declare_only": True,
            },
        }

        def __init__(self, max_steps: int = 10): ...

    class TraceWriter(hook.Base):
        __registry_key__ = "trace_writer"
        __wiring__ = {"archive": "archiver"}

        def __init__(
            self, trace_dir: str = "traces", archive: Optional[list[RunP]] = None,
        ): ...

    return {"action": action, "agent": agent, "hook": hook}


def _partial(registrars: dict[str, Any], **options: Any) -> ComposedConfigResult:
    return build_composed_config(registrars, coverage="partial", **options)


def _hook_block(**fields: Any) -> dict[str, Any]:
    return {"hook": [{"hook_id": "trace_writer", **fields}]}


ARCHIVE_MESSAGE = (
    "this build holds no implementation of layer 'archiver' that 'archive' accepts; "
    "leave 'archive' out"
)


# ===========================================================================
# The strict default is unchanged
# ===========================================================================


class TestCompleteBuildsStayStrict:
    def test_both_downstream_failures_still_raise(self):
        registrars = _layers()
        with pytest.raises(WiringResolutionError) as default:
            build_composed_config(registrars)
        with pytest.raises(WiringResolutionError) as explicit:
            build_composed_config(registrars, coverage="complete")
        assert [p.kind for p in default.value.problems] == [
            "unknown_optional_key", "empty_registry",
        ]
        assert str(default.value) == str(explicit.value)
        assert "unknown optional keys 'scroll', 'send_keys', 'drag', 'wait'" in str(default.value)
        assert (
            "wiring 'archive': registry 'archiver' is empty (no class is registered in it): "
            "discover() missed the package that implements it, or one of its modules failed "
            "to import; import every implementation module before resolving wiring"
        ) in str(default.value)

    def test_the_same_problems_as_validate_wiring(self):
        registrars = _layers()
        with pytest.raises(WiringResolutionError) as validated:
            validate_wiring(registrars)
        with pytest.raises(WiringResolutionError) as built:
            build_composed_config(registrars)
        assert str(validated.value) == str(built.value)

    def test_a_complete_result_carries_no_partial_markers(self):
        action = _layer("action", "action_id", key_separator=".")
        agent = _layer("agent", "agent_id")

        class Click(action.Base):
            __registry_key__ = "android.click"

        class Dom(agent.Base):
            __registry_key__ = "dom"
            __wiring__ = {
                "action": {"registry": "action", "keys": ["click"], "declare_only": True},
            }

        result = build_composed_config({"action": action, "agent": agent})
        assert result.coverage == "complete"
        assert result.unresolved_wiring == ()
        assert "x-coverage" not in generate_composed_json_schema(result)
        assert "Coverage" not in generate_composed_config_source(result)

    @pytest.mark.parametrize("bad", ["Partial", "", None, True])
    def test_unknown_coverage_is_a_value_error(self, bad):
        with pytest.raises(ValueError, match="coverage must be one of"):
            build_composed_config(_layers(), coverage=bad)

    def test_coverage_modes(self):
        from conscribe.config.composed import COVERAGE_MODES

        assert COVERAGE_MODES == get_args(conscribe.Coverage) == ("complete", "partial")


# ===========================================================================
# Unregistered keys of declare_only entries
# ===========================================================================


class TestDeclareOnlyKeys:
    def test_unregistered_keys_are_left_out_of_the_literal(self):
        result = _partial(_layers())
        model = result.layer_results["agent"].per_key_models["dom"]
        field = model.model_fields["action"]
        assert field.annotation == Optional[Literal["android.navigate", "android.click"]]
        assert field.default is None
        assert model(action="android.click").action == "android.click"
        with pytest.raises(ValidationError) as exc:
            model(action="android.scroll")
        assert [t for t, _ in _errors(exc.value)] == ["literal_error"]

    def test_the_entry_is_reported_with_what_is_missing(self):
        result = _partial(_layers())
        [action, _] = result.unresolved_wiring
        assert action == UnresolvedWiring(
            owner_layer="agent",
            owner_key="dom",
            name="action",
            registry_name="action",
            declare_only=True,
            keys=("android.navigate", "android.click"),
            missing_keys=(),
            missing_optional_keys=("scroll", "send_keys", "drag", "wait"),
            registry_empty=False,
        )
        assert str(action) == "agent[dom].action"

    def test_missing_required_keys_are_tolerated_and_reported(self):
        """The framework's own negotiation decides; the build only reports."""
        registrars = _layers()

        class Mobile(registrars["agent"].Base):
            __registry_key__ = "mobile"
            __wiring__ = {
                "action": {"registry": "action", "keys": ["tap", "click"], "declare_only": True},
            }

        [record] = [r for r in _partial(registrars).unresolved_wiring if r.owner_key == "mobile"]
        assert record.missing_keys == ("tap",)
        assert record.missing_optional_keys == ()
        assert record.keys == ("android.click",)

    def test_an_entry_none_of_whose_keys_is_registered_accepts_only_none(self):
        registrars = _layers()

        class Desktop(registrars["agent"].Base):
            __registry_key__ = "desktop"
            __wiring__ = {
                "action": {"registry": "action", "keys": ["drag"], "declare_only": True},
            }

        result = _partial(registrars)
        model = result.layer_results["agent"].per_key_models["desktop"]
        assert model().action is None
        assert model(action=None).action is None
        with pytest.raises(ValidationError) as exc:
            model(action="android.click")
        [error] = exc.value.errors()
        assert error["type"] == NO_IMPLEMENTATION_ERROR_TYPE
        assert error["loc"] == ("action",)
        assert error["ctx"] == {"field": "action", "layer": "action"}
        [record] = [r for r in result.unresolved_wiring if r.owner_key == "desktop"]
        assert record.keys == () and record.missing_keys == ("drag",)
        assert not record.registry_empty

    def test_an_empty_target_registry_is_tolerated(self):
        registrars = _layers()
        _layer("observation", "fragment_id", key_separator=".")

        class Vision(registrars["agent"].Base):
            __registry_key__ = "vision"
            __wiring__ = {
                "observation": {
                    "registry": "observation", "keys": ["screenshot"], "declare_only": True,
                },
                "everything": {"registry": "observation", "declare_only": True},
            }

        result = _partial(registrars)
        records = [r for r in result.unresolved_wiring if r.owner_key == "vision"]
        assert [(r.name, r.registry_empty, r.missing_keys, r.keys) for r in records] == [
            ("observation", True, ("screenshot",), ()),
            ("everything", True, (), ()),
        ]
        model = result.layer_results["agent"].per_key_models["vision"]
        assert model().model_dump() == {"observation": None, "everything": None, "agent_id": "vision"}

    def test_complete_keys_report_nothing(self):
        registrars = _layers()

        class Plain(registrars["agent"].Base):
            __registry_key__ = "plain"
            __wiring__ = {
                "action": {"registry": "action", "keys": ["click"], "declare_only": True},
            }

        assert all(r.owner_key != "plain" for r in _partial(registrars).unresolved_wiring)


# ===========================================================================
# Receptors wired to an empty registry
# ===========================================================================


class TestEmptyTargetWithDefault:
    def test_downstream_archive_receptor(self):
        result = _partial(_layers())
        top = result.top_level_type
        assert top.model_validate(_hook_block()).hook[0].archive is None
        # Values that select no implementation stay valid, as in a complete build.
        assert top.model_validate(_hook_block(archive=None)).hook[0].archive is None
        assert top.model_validate(_hook_block(archive=[])).hook[0].archive == []
        for value in ([{"archiver_id": "browser.network"}], ["browser.network"], "x", {}):
            with pytest.raises(ValidationError) as exc:
                top.model_validate(_hook_block(archive=value))
            [error] = exc.value.errors()
            assert (error["type"], tuple(error["loc"])) == (
                NO_IMPLEMENTATION_ERROR_TYPE, ("hook", 0, "archive"),
            )
            assert error["msg"] == ARCHIVE_MESSAGE
            assert error["ctx"] == {"field": "archive", "layer": "archiver"}

    def test_the_receptor_is_reported(self):
        [_, archive] = _partial(_layers()).unresolved_wiring
        assert archive == UnresolvedWiring(
            "hook", "trace_writer", "archive", "archiver", False, (), (), (), True,
        )
        assert str(archive) == "hook[trace_writer].archive"

    def test_json_schema_accepts_exactly_the_same(self):
        result = _partial(_layers())
        schema = generate_composed_json_schema(result)
        model = result.layer_results["hook"].per_key_models["trace_writer"]
        prop = schema["$defs"][model.__name__]["properties"]["archive"]
        assert prop["default"] is None
        assert sorted(prop["anyOf"], key=json.dumps) == sorted(
            [{"items": {}, "maxItems": 0, "type": "array"}, {"type": "null"}], key=json.dumps,
        )
        assert schema["x-coverage"] == "partial"

    def test_list_receptor_without_none(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, archivers: list[RunP] = []): ...  # noqa: B006

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().archivers == []
        assert model(archivers=()).archivers == []
        with pytest.raises(ValidationError) as exc:
            model(archivers=None)
        assert _errors(exc.value) == [(NO_IMPLEMENTATION_ERROR_TYPE, ("archivers",))]

    def test_tuple_default_becomes_an_empty_list(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, archivers: tuple[RunP, ...] = ()): ...

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().archivers == []

    def test_default_factories_on_list_receptors(self):
        registrars = _layers()

        @dataclasses.dataclass
        class Pure(registrars["hook"].Base):
            __registry_key__ = "pure"
            __wiring__ = {"archivers": "archiver"}
            archivers: list[RunP] = dataclasses.field(default_factory=list)

        @dataclasses.dataclass
        class Opaque(registrars["hook"].Base):
            __registry_key__ = "opaque"
            __wiring__ = {"archivers": "archiver"}
            archivers: list[RunP] = dataclasses.field(default_factory=_no_archivers)

        layer = _partial(registrars).layer_results["hook"]
        pure = layer.per_key_models["pure"].model_fields["archivers"]
        assert pure.default == [] and pure.default_factory is None
        opaque = layer.per_key_models["opaque"].model_fields["archivers"]
        assert opaque.default_factory is _no_archivers
        assert layer.per_key_models["opaque"]().archivers == []

    def test_single_instance_receptor_accepts_only_none(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": "archiver"}

            def __init__(self, sink: Optional[RunP] = None): ...

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().sink is None and model(sink=None).sink is None
        with pytest.raises(ValidationError) as exc:
            model(sink="browser.network")
        assert _errors(exc.value) == [(NO_IMPLEMENTATION_ERROR_TYPE, ("sink",))]
        with pytest.raises(ValidationError):
            model(sink=[])

    def test_class_annotation_admitting_none(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver"}
            archivers: Optional[list[RunP]]

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().archivers is None

    def test_class_annotation_with_a_default_in_its_metadata(self):
        """A ``Field`` default inside ``Annotated`` counts, as in a complete build."""
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver", "mirrors": "archiver"}
            archivers: Annotated[list[RunP], Field(default_factory=list)]
            mirrors: Annotated[list[RunP], Field(default=[])]

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().model_dump() == {"archivers": [], "mirrors": [], "hook_id": "recorder"}
        with pytest.raises(ValidationError) as exc:
            model(mirrors=["browser.network"])
        assert _errors(exc.value) == [(NO_IMPLEMENTATION_ERROR_TYPE, ("mirrors",))]

    def test_slot_less_entry(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": "archiver"}  # injected by the framework

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().sink is None
        with pytest.raises(ValidationError):
            model(sink="browser.network")

    def test_description_and_alias_are_kept(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver"}

            def __init__(
                self,
                archivers: Annotated[
                    Optional[list[RunP]],
                    Field(description="Selected archivers.", alias="archive_with"),
                ] = None,
            ): ...

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        field = model.model_fields["archivers"]
        assert field.description == "Selected archivers."
        assert field.alias == "archive_with"
        assert model(archive_with=[]).archivers == []

    def test_inline_wiring_false(self):
        result = _partial(_layers(), inline_wiring=False)
        model = result.layer_results["hook"].per_key_models["trace_writer"]
        assert model().archive is None
        with pytest.raises(ValidationError):
            model(archive=["browser.network"])

    def test_defaults_are_never_validated(self):
        """The receptor's own default stands, also under a model that validates
        defaults: a factory conscribe cannot evaluate is the class's business."""
        registrars = _layers()

        @dataclasses.dataclass
        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __config_from_param__ = "knobs"
            __wiring__ = {"archivers": "archiver"}
            knobs: _ValidatedKnobs
            archivers: list[str] = dataclasses.field(default_factory=_preset_archivers)

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model.model_config.get("validate_default") is True
        assert model().model_dump() == {
            "retries": 1, "archivers": ["preset"], "hook_id": "recorder",
        }
        with pytest.raises(ValidationError) as exc:
            model(archivers=["browser.network"])
        assert _errors(exc.value) == [(NO_IMPLEMENTATION_ERROR_TYPE, ("archivers",))]


class _ValidatedKnobs(BaseModel):
    model_config = {"validate_default": True}
    retries: int = 1


_NO_ARCHIVERS: tuple[RunP, ...] = ()
_PRESET: tuple[str, ...] = ("preset",)


def _no_archivers() -> list[RunP]:
    """A factory conscribe does not evaluate (it reads a module global)."""
    return list(_NO_ARCHIVERS)


def _preset_archivers() -> list[str]:
    """Likewise, with a value the field's type would refuse."""
    return list(_PRESET)


# ===========================================================================
# What a partial build still refuses
# ===========================================================================


def _problem(registrars: dict[str, Any]) -> Any:
    with pytest.raises(WiringResolutionError) as exc:
        _partial(registrars)
    [problem] = exc.value.problems
    return problem


class TestStillRefused:
    def test_required_receptor_wired_to_an_empty_registry(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": "archiver"}

            def __init__(self, sink: RunP): ...

        problem = _problem(registrars)
        assert (problem.kind, problem.param_name) == ("empty_registry", "sink")
        assert "a partial build keeps a field wired to an empty registry only when" in problem.detail
        assert "'sink' has no default" in problem.detail

    def test_default_key_on_an_empty_registry(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": {"registry": "archiver", "default": "browser.network"}}

            def __init__(self, sink: Optional[RunP] = None): ...

        problem = _problem(registrars)
        assert problem.kind == "empty_registry"
        assert "the entry's default 'browser.network' names one of its keys" in problem.detail

    @pytest.mark.parametrize("default, why", [
        ("browser.network", "'sink' defaults to 'browser.network'"),
    ])
    def test_key_string_parameter_default(self, default, why):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": "archiver"}

            def __init__(self, sink: str = default): ...

        problem = _problem(registrars)
        assert problem.kind == "empty_registry" and why in problem.detail

    def test_key_default_in_class_annotation_metadata(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": "archiver"}
            sink: Annotated[str, Field(default="browser.network")]

        assert "'sink' defaults to 'browser.network'" in _problem(registrars).detail

    def test_non_empty_list_default(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, archivers: list[str] = ["browser.network"]): ...  # noqa: B006

        assert "defaults to ['browser.network']" in _problem(registrars).detail

    def test_pure_factory_naming_a_key(self):
        """Only the build evaluates factories, so the build refuses this one."""
        registrars = _layers()

        @dataclasses.dataclass
        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": "archiver"}
            archivers: list[str] = dataclasses.field(default_factory=lambda: ["browser.network"])

        problem = _problem(registrars)
        assert problem.kind == "empty_registry"
        assert "defaults to ['browser.network']" in problem.detail

    def test_factory_on_a_single_instance_receptor(self):
        registrars = _layers()

        @dataclasses.dataclass
        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __wiring__ = {"sink": "archiver"}
            sink: Optional[RunP] = dataclasses.field(default_factory=lambda: None)

        problem = _problem(registrars)
        assert "takes its default from a default_factory instead of None" in problem.detail

    def test_unknown_keys_of_a_receptor(self):
        registrars = _layers()

        class Runner(registrars["hook"].Base):
            __registry_key__ = "runner"
            __wiring__ = {"actions": ("action", ["click", "scroll"])}

            def __init__(self, actions: Optional[list[RunP]] = None): ...

        problem = _problem(registrars)
        assert (problem.kind, problem.param_name) == ("unknown_key", "actions")

    def test_unknown_registry(self):
        registrars = _layers()

        class Runner(registrars["hook"].Base):
            __registry_key__ = "runner"
            __wiring__ = {"sinks": {"registry": "archiverz", "declare_only": True}}

        problem = _problem(registrars)
        assert problem.kind == "unknown_registry"

    def test_layer_without_implementations(self):
        registrars = _layers()
        empty = _layer("empty", "empty_id")
        with pytest.raises(EmptyLayerError):
            _partial({**registrars, "empty": empty})

    def test_declaration_errors(self):
        registrars = _layers()

        class Runner(registrars["hook"].Base):
            __registry_key__ = "runner"
            __wiring__ = {"sinks": {"registry": "archiver", "declare_only": True}}

            def __init__(self, sinks: Optional[list[RunP]] = None): ...

        assert _problem(registrars).kind == "contradiction"


# ===========================================================================
# Report shape and order
# ===========================================================================


class TestReport:
    def test_order_and_every_registration(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = ["recorder", "another_recorder"]
            __wiring__ = {
                "sink": "archiver",
                "mirror": "archiver",
            }

            def __init__(self, sink: Optional[RunP] = None, mirror: Optional[RunP] = None): ...

        result = _partial(registrars)
        assert result.dependency_order == ["action", "agent", "hook"]
        assert [str(r) for r in result.unresolved_wiring] == [
            "agent[dom].action",
            "hook[another_recorder].sink",
            "hook[another_recorder].mirror",
            "hook[recorder].sink",
            "hook[recorder].mirror",
            "hook[trace_writer].archive",
        ]
        assert result.coverage == "partial"

    def test_records_are_frozen(self):
        [record, _] = _partial(_layers()).unresolved_wiring
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.name = "x"  # type: ignore[misc]

    def test_complete_wiring_builds_the_same_in_both_modes(self):
        """Partial mode changes only what it tolerates."""
        action = _layer("action", "action_id", key_separator=".")
        agent = _layer("agent", "agent_id")

        class Click(action.Base):
            __registry_key__ = "android.click"

        class Dom(agent.Base):
            __registry_key__ = "dom"
            __wiring__ = {
                "action": {"registry": "action", "keys": ["click"], "declare_only": True},
                "tool": {"registry": "action", "default": "android.click"},
                "tools": "action",
            }

            def __init__(self, tool: Optional[RunP] = None, tools: list[RunP] = []): ...  # noqa: B006

        registrars = {"action": action, "agent": agent}
        complete = generate_composed_json_schema(build_composed_config(registrars))
        result = _partial(registrars)
        partial = generate_composed_json_schema(result)
        assert result.unresolved_wiring == ()
        assert partial.pop("x-coverage") == "partial"
        assert partial == complete


# ===========================================================================
# Per-class extraction of a partial catalog
# ===========================================================================


class TestPerClassExtraction:
    """A framework explaining a partial run reads each component's provenance."""

    def test_provenance_and_schema_take_the_same_mode(self):
        registrars = _layers()
        hook, agent = registrars["hook"], registrars["agent"]
        for strict in (
            lambda: hook.get_config_provenance("trace_writer"),
            lambda: agent.get_config_provenance("dom"),
            lambda: hook.get_config_schema("trace_writer"),
        ):
            with pytest.raises(WiringResolutionError):
                strict()
        provenance = hook.get_config_provenance("trace_writer", coverage="partial")
        assert provenance.fields["archive"].receptor is not None
        assert "action" in agent.get_config_provenance("dom", coverage="partial").fields
        schema = hook.get_config_schema("trace_writer", coverage="partial")
        assert schema(archive=[]).archive == []
        with pytest.raises(ValidationError) as exc:
            schema(archive=["browser.network"])
        assert _errors(exc.value) == [(NO_IMPLEMENTATION_ERROR_TYPE, ("archive",))]

    @pytest.mark.parametrize("function", [
        conscribe.extract_config_schema, conscribe.extract_config_provenance,
    ])
    def test_unknown_coverage_is_a_value_error(self, function):
        registrars = _layers()
        with pytest.raises(ValueError, match=rf"{function.__name__}\(\): coverage must be one of"):
            function(registrars["hook"].get("trace_writer"), coverage="most")


# ===========================================================================
# Registration never depends on what is registered
# ===========================================================================


def test_classes_register_before_their_targets_exist():
    """Only the grammar is checked at class creation, so a partial build can
    tolerate what resolution reports: no registry, no key, empty registries."""
    agent = _layer("agent", "agent_id")

    class Early(agent.Base):
        __registry_key__ = "early"
        __wiring__ = {
            "action": {"registry": "not_yet", "keys": ["click"], "declare_only": True},
            "llm": "nor_this",
        }

    check_wiring_declaration(Early)
    assert agent.get("early") is Early
    with pytest.raises(WiringResolutionError) as exc:
        _partial({"agent": agent})
    assert [p.kind for p in exc.value.problems] == ["unknown_registry", "unknown_registry"]
    # The registries appear (empty) later: the partial build tolerates both.
    _layer("not_yet", "x_id")
    _layer("nor_this", "y_id")
    result = _partial({"agent": agent})
    assert [(r.name, r.registry_empty) for r in result.unresolved_wiring] == [
        ("action", True), ("llm", True),
    ]


# ===========================================================================
# Tier-3 schemas
# ===========================================================================


class _RecorderSchema(BaseModel):
    model_config = {"extra": "forbid"}
    path: str = "out"
    archivers: Optional[list[str]] = Field(default=None, description="Archivers.")


class _RequiredSchema(BaseModel):
    archivers: list[str]


class _FactorySchema(BaseModel):
    archivers: list[str] = Field(default_factory=list)
    sink: Optional[str] = Field(default_factory=lambda: None)


class TestTier3:
    def test_wired_schema_field_is_redeclared(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __config_schema__ = _RecorderSchema
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, path: str = "out", archivers: Optional[list[RunP]] = None): ...

        result = _partial(registrars)
        model = result.layer_results["hook"].per_key_models["recorder"]
        assert issubclass(model, _RecorderSchema)
        assert model().archivers is None and model(archivers=[]).archivers == []
        assert model.model_fields["archivers"].description == "Archivers."
        with pytest.raises(ValidationError) as exc:
            model(archivers=["browser.network"])
        assert _errors(exc.value) == [(NO_IMPLEMENTATION_ERROR_TYPE, ("archivers",))]
        assert "archivers" not in (getattr(model, "__wired_fields__", None) or {})
        assert [str(r) for r in result.unresolved_wiring][-2:] == [
            "hook[recorder].archivers", "hook[trace_writer].archive",
        ]
        _parity(result, {"ComposedConfig": [
            {"hook": [{"hook_id": "recorder"}]},
            {"hook": [{"hook_id": "recorder", "archivers": []}]},
            {"hook": [{"hook_id": "recorder", "archivers": ["browser.network"]}]},
        ]})

    def test_the_schema_default_counts_not_the_constructor(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __config_schema__ = _RecorderSchema
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, path: str, archivers: list[RunP]): ...  # both required

        model = _partial(registrars).layer_results["hook"].per_key_models["recorder"]
        assert model().archivers is None

    def test_schema_field_without_a_usable_default(self):
        registrars = _layers()

        class Recorder(registrars["hook"].Base):
            __registry_key__ = "recorder"
            __config_schema__ = _RequiredSchema
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, archivers: list[RunP] = []): ...  # noqa: B006

        problem = _problem(registrars)
        assert problem.kind == "empty_registry"
        assert "the __config_schema__ field 'archivers' has no default" in problem.detail

    def test_schema_field_defaults_from_a_factory(self):
        registrars = _layers()

        class Lists(registrars["hook"].Base):
            __registry_key__ = "lists"
            __config_schema__ = _FactorySchema
            __wiring__ = {"archivers": "archiver"}

            def __init__(self, archivers: list[RunP] = [], sink: Optional[str] = None): ...  # noqa: B006

        model = _partial(registrars).layer_results["hook"].per_key_models["lists"]
        assert model().archivers == [] and model.model_fields["archivers"].default_factory is list

        class Single(registrars["hook"].Base):
            __registry_key__ = "single"
            __config_schema__ = _FactorySchema
            __wiring__ = {"sink": "archiver"}

            def __init__(self, archivers: list[str] = [], sink: Optional[RunP] = None): ...  # noqa: B006

        problem = _problem(registrars)
        assert (problem.param_name, problem.kind) == ("sink", "empty_registry")
        assert (
            "the __config_schema__ field 'sink' takes its default from a default_factory "
            "instead of None"
        ) in problem.detail


# ===========================================================================
# Three-artifact parity
# ===========================================================================


def _parity(result: ComposedConfigResult, samples: dict[str, list[Any]]) -> None:
    from tests.config.artifact_parity import assert_artifacts_agree, composed_expectations

    assert_artifacts_agree(
        generate_composed_config_source(result),
        composed_expectations(result),
        samples=samples,
        published=(generate_composed_json_schema(result), result.top_level_type),
        min_accepted=1,
    )


def test_partial_build_renders_with_parity():
    registrars = _layers()

    class Desktop(registrars["agent"].Base):
        __registry_key__ = "desktop"
        __wiring__ = {"action": {"registry": "action", "keys": ["drag"], "declare_only": True}}

    class Runner(registrars["hook"].Base):  # an inlined receptor beside one without implementation
        __registry_key__ = "runner"
        __wiring__ = {"tool": "action", "archive": "archiver"}

        def __init__(self, tool: Optional[RunP] = None, archive: Optional[list[RunP]] = None): ...

    result = _partial(registrars)
    runner = result.layer_results["hook"].per_key_models["runner"]
    assert list(runner.__inline_wired__) == ["tool"]
    source = generate_composed_config_source(result)
    assert "# Coverage: partial" in source
    _parity(result, {"ComposedConfig": [
        _hook_block(),
        _hook_block(archive=None),
        _hook_block(archive=[]),
        _hook_block(archive=[{"archiver_id": "browser.network"}]),
        _hook_block(archive=["browser.network"]),
        {"agent": [{"agent_id": "dom", "action": "android.click"}]},
        {"agent": [{"agent_id": "dom", "action": "android.scroll"}]},
        {"agent": [{"agent_id": "desktop"}]},
        {"agent": [{"agent_id": "desktop", "action": "android.click"}]},
        {"hook": [{"hook_id": "runner", "tool": {"action_id": "android.click"}, "archive": []}]},
        {"hook": [{"hook_id": "runner", "archive": ["browser.network"]}]},
    ]})


def test_the_report_validator_without_its_table_is_a_render_error():
    from pydantic import create_model, field_validator

    from conscribe import ConfigRenderError, LayerConfigResult
    from conscribe.config.codegen import generate_layer_config_source
    from conscribe.config.extractor import _report_no_implementation

    model = create_model(
        "UntabledConfig",
        kind=(Literal["k"], "k"),
        sink=(Optional[str], None),
        __validators__={
            "_conscribe_report_no_implementation": field_validator("sink", mode="wrap")(
                _report_no_implementation,
            ),
        },
    )
    model.__conscribe_generated__ = True  # type: ignore[attr-defined]
    result = LayerConfigResult(
        union_type=model, per_key_models={"k": model}, layer_name="probe",
        discriminator_field="kind",
    )
    with pytest.raises(ConfigRenderError, match="does not name the target layer"):
        generate_layer_config_source(result)


# ===========================================================================
# Public names
# ===========================================================================


def test_partial_build_names_are_exported():
    import conscribe.config as config
    from conscribe.config import composed

    for module in (conscribe, config):
        assert module.Coverage is composed.Coverage
        assert module.UnresolvedWiring is composed.UnresolvedWiring
        assert module.NO_IMPLEMENTATION_ERROR_TYPE == "no_implementation"
        assert {"Coverage", "UnresolvedWiring", "NO_IMPLEMENTATION_ERROR_TYPE"} <= set(
            module.__all__,
        )
