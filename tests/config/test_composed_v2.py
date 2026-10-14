"""Composed config 2.0: validate-first, strict top level, inline-only layers,
default instances (F1 / F3 / F4 of the 2.0 spec, af acceptance cases).

The fixtures mirror af's shapes: a trace hook whose ``archive`` list
receptor is the only home of the ``archiver`` layer, a browser capability
with annotation-only receptors (``network_source`` defaulting to ``live``)
and a slot-less negotiation entry, and a replay chain
(``network_source`` -> ``miss_policy``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Optional,
    Protocol,
    Union,
    get_args,
    runtime_checkable,
)

import pytest
from annotated_types import MinLen
from pydantic import (
    BaseModel,
    Field,
    PlainSerializer,
    RootModel,
    StringConstraints,
    ValidationError,
    create_model,
    field_serializer,
    model_validator,
)

from conscribe import (
    CircularWiringError,
    ConfigDeclarationError,
    LayerConfigResult,
    ReceptorLocation,
    WiringResolutionError,
    build_layer_config,
    create_registrar,
)
from conscribe.registration.registrar import LayerRegistrar

if TYPE_CHECKING:
    # Visible to type checkers only: a receptor typed with it has an
    # annotation conscribe cannot resolve at runtime.
    from collections.abc import Sized as _TypeCheckingOnlySink
from conscribe.config.codegen import generate_composed_config_source
from conscribe.config.composed import (
    INLINE_ONLY_ERROR_TYPE,
    ComposedConfigResult,
    _build_dependency_graph,
    _replace_wired_literals_with_unions,
    _schema_default,
    build_composed_config,
)
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


# ---------------------------------------------------------------------------
# af-shaped fixtures
# ---------------------------------------------------------------------------


def _archiver(*, inline_only: bool = True) -> Any:
    archiver = _layer("archiver", "archiver_id", key_separator=".", inline_only=inline_only)

    class Network(archiver.Base):
        __registry_key__ = "browser.network"

        def __init__(self, max_bytes: int = 1024): ...

    class Storage(archiver.Base):
        __registry_key__ = "browser.storage"

    return archiver


def _hook(*, inline_only: bool = False, recorder: bool = False) -> Any:
    hook = _layer("hook", "hook_id", inline_only=inline_only)

    class _TraceWriterReceptors:
        archive: list[RunP]

    class TraceWriter(_TraceWriterReceptors, hook.Base):
        __registry_key__ = "trace_writer"
        __wiring__ = {"archive": "archiver"}

        def __init__(self, trace_dir: str = "traces", archive: Optional[list[RunP]] = None): ...

    if recorder:

        class Recorder(hook.Base):
            __registry_key__ = "recorder"
            __wiring__ = {"archivers": ("archiver", ["browser.storage"])}

            def __init__(self, archivers: Optional[list[RunP]] = None): ...

    return hook


def _negotiating_capability() -> Any:
    capability = _layer("capability", "capability_id")

    class BrowserCapability(capability.Base):
        __registry_key__ = "browser"
        # Negotiation-only: no slot receives it, so it is never a place to
        # write archiver instances.
        __wiring__ = {"archiver": ("archiver", ["browser.network"])}

        def __init__(self, headless: bool = True): ...

    return capability


def _af_build(**options: Any) -> ComposedConfigResult:
    return build_composed_config(
        {"archiver": _archiver(), "hook": _hook(), "capability": _negotiating_capability()},
        **options,
    )


AF_WHERE = "layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive"


# ===========================================================================
# F3 -- strict top level (af defect 2)
# ===========================================================================


class TestStrictTopLevel:
    def test_af_defect_2_repro(self):
        """af's repro: a misspelled section (``hooks``) used to vanish silently."""
        hook = _layer("hook", "hook_id")

        class TraceWriter(hook.Base):
            __registry_key__ = "trace_writer"

            def __init__(self, trace_dir: str = "traces"): ...

        result = build_composed_config({"hook": hook})
        top = result.top_level_type
        assert result.top_level_extra == "forbid"
        assert top.model_config["extra"] == "forbid"
        with pytest.raises(ValidationError) as exc:
            top.model_validate({"hooks": [{"hook_id": "trace_writer"}]})
        assert _errors(exc.value) == [("extra_forbidden", ("hooks",))]
        with pytest.raises(ValidationError) as exc:
            top.model_validate({"hook": [{"hook_id": "trace_writer", "trace_dri": "x"}]})
        assert [e[0] for e in _errors(exc.value)] == ["extra_forbidden"]
        assert top.model_validate({"hook": [{"hook_id": "trace_writer"}]}).hook[0].trace_dir == "traces"

    def test_schema_says_additional_properties_false(self):
        result = _af_build()
        schema = generate_composed_json_schema(result)
        assert schema["additionalProperties"] is False

    @pytest.mark.parametrize("mode", ["ignore", "allow"])
    def test_relaxed_modes(self, mode):
        hook = _hook()
        result = build_composed_config({"hook": hook, "archiver": _archiver(inline_only=False)}, top_level_extra=mode)
        top = result.top_level_type
        assert result.top_level_extra == mode
        cfg = top.model_validate({"hooks": [{"hook_id": "trace_writer"}]})
        assert cfg.hook == []
        schema = generate_composed_json_schema(result)
        if mode == "ignore":
            assert cfg.model_extra is None
            assert "additionalProperties" not in schema
        else:
            assert cfg.model_extra == {"hooks": [{"hook_id": "trace_writer"}]}
            assert schema["additionalProperties"] is True

    @pytest.mark.parametrize("bad", ["strict", "FORBID", None, ["forbid"]])
    def test_bad_mode_is_a_value_error(self, bad):
        with pytest.raises(ValueError) as exc:
            build_composed_config({"hook": _hook()}, top_level_extra=bad)
        assert str(exc.value) == (
            "build_composed_config(): top_level_extra must be one of "
            f"['forbid', 'ignore', 'allow'], got {bad!r}"
        )

    @pytest.mark.parametrize("mode", ["forbid", "ignore", "allow"])
    def test_in_process_schema_equals_published_schema(self, mode):
        result = build_composed_config(
            {"archiver": _archiver(), "hook": _hook(), "capability": _negotiating_capability()},
            top_level_extra=mode,
        )
        assert result.top_level_type.model_json_schema() == generate_composed_json_schema(result)

    def test_no_inline_only_layers(self):
        result = build_composed_config({"hook": _hook(), "archiver": _archiver(inline_only=False)})
        assert result.inline_only_layers == {}
        assert result.top_level_layers == result.dependency_order == ["archiver", "hook"]
        assert set(result.top_level_type.model_fields) == {"archiver", "hook"}
        extra = result.top_level_type.model_config["json_schema_extra"]
        assert extra == {"x-composed-layers": ["archiver", "hook"], "x-inline-wiring": True}


# ===========================================================================
# F3 -- inline-only layers (af acceptance)
# ===========================================================================


class TestInlineOnlyLayers:
    def test_af_top_level_archiver_is_refused_naming_the_receptor(self):
        top = _af_build().top_level_type
        with pytest.raises(ValidationError) as exc:
            top.model_validate({"archiver": [{"archiver_id": "browser.network"}]})
        (error,) = exc.value.errors()
        assert error["type"] == INLINE_ONLY_ERROR_TYPE == "inline_only_layer"
        assert error["loc"] == ("archiver",)
        assert error["msg"] == AF_WHERE
        assert error["ctx"] == {"layer": "archiver", "where": "hook[trace_writer].archive"}

    @pytest.mark.parametrize("value", [[], None, {}, "browser.network"])
    def test_empty_and_null_sections_are_refused_too(self, value):
        top = _af_build().top_level_type
        with pytest.raises(ValidationError) as exc:
            top.model_validate({"archiver": value})
        assert _errors(exc.value) == [("inline_only_layer", ("archiver",))]

    def test_af_hooks_typo_is_refused(self):
        with pytest.raises(ValidationError) as exc:
            _af_build().top_level_type.model_validate({"hooks": [{"hook_id": "trace_writer"}]})
        assert _errors(exc.value) == [("extra_forbidden", ("hooks",))]

    def test_af_inline_archive_is_accepted(self):
        top = _af_build().top_level_type
        cfg = top.model_validate({
            "hook": [{
                "hook_id": "trace_writer",
                "archive": [
                    {"archiver_id": "browser.network", "max_bytes": 5},
                    {"archiver_id": "browser.storage"},
                ],
            }],
        })
        archive = cfg.hook[0].archive
        assert [a.archiver_id for a in archive] == ["browser.network", "browser.storage"]
        assert archive[0].max_bytes == 5
        assert cfg.archiver is None
        assert "archiver" not in cfg.model_dump()
        assert cfg.model_dump()["hook"][0]["archive"][1] == {"archiver_id": "browser.storage"}

    def test_af_schema_has_no_archiver_section(self):
        result = _af_build()
        schema = generate_composed_json_schema(result)
        assert set(schema["properties"]) == {"hook", "capability"}
        assert "archiver" not in schema.get("required", [])
        assert schema["additionalProperties"] is False
        assert schema["x-inline-only-layers"] == {"archiver": ["hook[trace_writer].archive"]}
        assert "not" not in schema  # forbid already refuses unknown names
        # The archiver configs are still there, for the receptor.
        assert "BrowserNetworkArchiverConfig" in schema["$defs"]
        assert schema["x-composed-layers"] == result.dependency_order
        assert "archiver" in result.dependency_order

    @pytest.mark.parametrize("mode", ["ignore", "allow"])
    def test_relaxed_modes_still_refuse_inline_only_names(self, mode):
        result = _af_build(top_level_extra=mode)
        top = result.top_level_type
        for value in ([{"archiver_id": "browser.storage"}], [], None):
            with pytest.raises(ValidationError) as exc:
                top.model_validate({"archiver": value})
            assert _errors(exc.value) == [("inline_only_layer", ("archiver",))]
        schema = generate_composed_json_schema(result)
        assert "archiver" not in schema["properties"]
        assert schema["not"] == {"anyOf": [{"required": ["archiver"]}]}

    def test_refusal_is_reported_with_every_other_error(self):
        top = _af_build().top_level_type
        with pytest.raises(ValidationError) as exc:
            top.model_validate({
                "archiver": [],
                "hooks": [],
                "hook": [{"hook_id": "trace_writer", "trace_dri": "x"}],
            })
        errors = _errors(exc.value)
        assert ("inline_only_layer", ("archiver",)) in errors
        assert ("extra_forbidden", ("hooks",)) in errors
        assert any(t == "extra_forbidden" and loc[-1] == "trace_dri" for t, loc in errors)
        assert len(errors) == 3

    def test_json_constructor_and_nested_paths_are_refused(self):
        top = _af_build().top_level_type
        with pytest.raises(ValidationError) as exc:
            top.model_validate_json('{"archiver": null}')
        assert _errors(exc.value) == [("inline_only_layer", ("archiver",))]
        with pytest.raises(ValidationError) as exc:
            top(archiver=[])
        assert _errors(exc.value) == [("inline_only_layer", ("archiver",))]

        Outer = create_model("Outer", cfg=(top, ...))
        with pytest.raises(ValidationError) as exc:
            Outer.model_validate({"cfg": {"archiver": []}})
        assert _errors(exc.value) == [("inline_only_layer", ("cfg", "archiver"))]

    def test_result_fields(self):
        result = _af_build()
        location = ReceptorLocation(
            "hook", "trace_writer", "archive", "archiver", ("browser.network", "browser.storage"),
        )
        assert result.inline_only_layers == {"archiver": (location,)}
        assert str(location) == "hook[trace_writer].archive"
        assert result.top_level_layers == [
            layer for layer in result.dependency_order if layer != "archiver"
        ]
        # The phantom field exists (hidden, excluded); enumerate sections
        # with top_level_layers.
        phantom = result.top_level_type.model_fields["archiver"]
        assert phantom.default is None and phantom.exclude is True
        assert result.top_level_type().model_dump() == {"capability": [], "hook": []}
        # The archiver layer is built in full.
        assert set(result.layer_results["archiver"].per_key_models) == {
            "browser.network", "browser.storage",
        }

    def test_negotiation_only_entries_are_never_listed(self):
        """The capability's slot-less ``archiver`` entry holds inline configs
        but has no slot, so it is not a place to write archiver instances."""
        result = _af_build()
        capability = result.layer_results["capability"].per_key_models["browser"]
        assert capability.__inline_wired__["archiver"]["target_layer"] == "archiver"
        assert [str(loc) for loc in result.inline_only_layers["archiver"]] == [
            "hook[trace_writer].archive",
        ]

    def test_every_location_is_listed_with_its_keys(self):
        result = build_composed_config({"archiver": _archiver(), "hook": _hook(recorder=True)})
        locations = result.inline_only_layers["archiver"]
        assert [(str(loc), loc.keys) for loc in locations] == [
            ("hook[recorder].archivers", ("browser.storage",)),
            ("hook[trace_writer].archive", ("browser.network", "browser.storage")),
        ]
        with pytest.raises(ValidationError) as exc:
            result.top_level_type.model_validate({"archiver": []})
        assert exc.value.errors()[0]["msg"] == (
            "layer 'archiver' is inline-only; write its instances under: "
            "hook[recorder].archivers or hook[trace_writer].archive"
        )

    def test_full_path_through_inline_only_owners(self):
        archiver = _archiver()
        hook = _hook(inline_only=True)
        env = _layer("env", "env_id")

        class BrowserEnv(env.Base):
            __registry_key__ = "browser"
            __wiring__ = {"hooks": "hook"}

            def __init__(self, hooks: Optional[list[RunP]] = None): ...

        result = build_composed_config({"archiver": archiver, "hook": hook, "env": env})
        assert result.top_level_layers == ["env"]
        assert [str(loc) for loc in result.inline_only_layers["archiver"]] == [
            "hook[trace_writer].archive",
        ]
        top = result.top_level_type
        with pytest.raises(ValidationError) as exc:
            top.model_validate({"archiver": [], "hook": []})
        assert [e["msg"] for e in exc.value.errors()] == [
            "layer 'archiver' is inline-only; write its instances under: "
            "env[browser].hooks[trace_writer].archive",
            "layer 'hook' is inline-only; write its instances under: env[browser].hooks",
        ]
        schema = generate_composed_json_schema(result)
        assert schema["x-inline-only-layers"] == {
            "archiver": ["env[browser].hooks[trace_writer].archive"],
            "hook": ["env[browser].hooks"],
        }
        cfg = top.model_validate({
            "env": [{
                "env_id": "browser",
                "hooks": [{"hook_id": "trace_writer", "archive": [{"archiver_id": "browser.storage"}]}],
            }],
        })
        assert cfg.env[0].hooks[0].archive[0].archiver_id == "browser.storage"

    def test_inline_wiring_false_is_refused(self):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"archiver": _archiver(), "hook": _hook()}, inline_wiring=False)
        assert str(exc.value) == (
            "build_composed_config(inline_wiring=False): layer 'archiver' is inline-only "
            "(create_registrar(..., inline_only=True)): its instances can only be written "
            "inside receptors of other layers, and receptors hold inline configs only with "
            "inline_wiring=True, so without it they could not be written anywhere. Build "
            "with inline_wiring=True, or leave 'archiver' out of the registrars."
        )

    def test_inline_wiring_false_names_every_inline_only_layer(self):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(
                {"archiver": _archiver(), "hook": _hook(inline_only=True)}, inline_wiring=False,
            )
        assert str(exc.value).startswith(
            "build_composed_config(inline_wiring=False): layers 'archiver', 'hook' are "
            "inline-only (create_registrar(..., inline_only=True)): their instances can only"
        )

    def test_layer_without_its_owner_is_unreachable(self):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"archiver": _archiver()})
        assert str(exc.value) == (
            "build_composed_config: layer 'archiver' is inline-only "
            "(create_registrar(..., inline_only=True)), but no receptor in this build holds "
            "its configs inline, so its instances could not be written anywhere. Give a "
            "class of another layer in this build a receptor for it (a __wiring__ entry "
            "targeting 'archiver' named like one of its __init__ parameters or class-level "
            "annotations), include the owning layer in the build, or create the registrar "
            "without inline_only=True."
        )

    def test_entries_that_do_not_count_are_named(self):
        archiver = _archiver()

        class Chained(archiver.Base):
            __registry_key__ = "browser.chained"
            __wiring__ = {"next": "archiver"}

            def __init__(self, next: Optional[str] = None): ...

        capability = _negotiating_capability()

        class DeclaringCapability(capability.Base):
            __registry_key__ = "desktop"
            __wiring__ = {"archiver": {"registry": "archiver", "declare_only": True}}

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"archiver": archiver, "capability": capability})
        msg = str(exc.value)
        assert "layer 'archiver' is inline-only" in msg
        assert (
            "entries that do not count: archiver[browser.chained].next (a receptor of the "
            "layer itself; a layer's configs never nest inside its own), "
            "capability[browser].archiver (slot-less: no __init__ parameter or class-level "
            "annotation receives it), capability[desktop].archiver (a declare_only entry)"
        ) in msg


# ===========================================================================
# F4 -- default instances (inline mode)
# ===========================================================================


def _replay_layers(*, archive_needs_path: bool = False) -> dict[str, Any]:
    miss = _layer("miss_policy", "policy_id")

    class Strict(miss.Base):
        __registry_key__ = "strict"

    class Lenient(miss.Base):
        __registry_key__ = "lenient"

        def __init__(self, threshold: float = 0.5): ...

    source = _layer("network_source", "source_id")

    class Live(source.Base):
        __registry_key__ = "live"

    class _ArchiveReceptors:
        miss_policy: RunP

    if archive_needs_path:

        class ArchiveSource(_ArchiveReceptors, source.Base):
            __registry_key__ = "archive"
            __wiring__ = {"miss_policy": {"registry": "miss_policy", "default": "strict"}}

            def __init__(self, path: str, max_bytes: int = 1024): ...

    else:

        class ArchiveSource(_ArchiveReceptors, source.Base):  # type: ignore[no-redef]
            __registry_key__ = "archive"
            __wiring__ = {"miss_policy": {"registry": "miss_policy", "default": "strict"}}

            def __init__(self, max_bytes: int = 1024): ...

    backend = _layer("browser_provider", "provider_id")

    class Local(backend.Base):
        __registry_key__ = "local"

        def __init__(self, port: int = 9222): ...

    return {"miss_policy": miss, "network_source": source, "browser_provider": backend}


def _capability(default: str = "live", **extra_entries: Any) -> Any:
    capability = _layer("capability", "capability_id")

    class _BrowserReceptors:
        network_source: RunP
        backend: RunP

    class BrowserCapability(_BrowserReceptors, capability.Base):
        __registry_key__ = "browser"
        __wiring__ = {
            "network_source": {
                "registry": "network_source",
                "keys": ["archive", "live"],
                "default": default,
            },
            "backend": "browser_provider",
            **extra_entries,
        }

        def __init__(self, headless: bool = True): ...

    return capability


class TestDefaultInstances:
    def _build(self, **options: Any) -> ComposedConfigResult:
        return build_composed_config({**_replay_layers(), "capability": _capability()}, **options)

    def test_af_omitted_receptor_validates_to_the_default_instance(self):
        result = self._build()
        live = result.layer_results["network_source"].per_key_models["live"]
        cfg = result.top_level_type.model_validate(
            {"capability": [{"capability_id": "browser", "backend": {"provider_id": "local"}}]},
        )
        source = cfg.capability[0].network_source
        assert type(source) is live  # the final per-key model: configs map back by type
        assert cfg.model_dump()["capability"][0]["network_source"] == {"source_id": "live"}

    def test_field_definition(self):
        result = self._build()
        capability = result.layer_results["capability"].per_key_models["browser"]
        live = result.layer_results["network_source"].per_key_models["live"]
        field = capability.model_fields["network_source"]
        assert field.default_factory is live
        assert not field.is_required()
        assert field.json_schema_extra == {"default": {"source_id": "live"}}
        assert capability.model_fields["backend"].is_required()  # no default declared
        assert capability.__inline_wired__["network_source"] == {
            "target_layer": "network_source",
            "keys": ("live", "archive"),  # the layer's own union, registration order
            "container": False,
            "optional": False,
        }

    def test_af_null_is_refused(self):
        capability = self._build().layer_results["capability"].per_key_models["browser"]
        with pytest.raises(ValidationError) as exc:
            capability.model_validate({"backend": {"provider_id": "local"}, "network_source": None})
        assert [loc for _, loc in _errors(exc.value)] == [("network_source",)]

    def test_another_key_and_the_nested_default_chain(self):
        result = self._build()
        capability = result.layer_results["capability"].per_key_models["browser"]
        strict = result.layer_results["miss_policy"].per_key_models["strict"]
        cfg = capability.model_validate(
            {"backend": {"provider_id": "local"}, "network_source": {"source_id": "archive"}},
        )
        assert type(cfg.network_source.miss_policy) is strict
        assert cfg.model_dump()["network_source"] == {
            "source_id": "archive", "max_bytes": 1024, "miss_policy": {"policy_id": "strict"},
        }

    def test_default_instances_are_not_shared(self):
        capability = self._build().layer_results["capability"].per_key_models["browser"]
        a = capability(backend={"provider_id": "local"})
        b = capability(backend={"provider_id": "local"})
        assert a.network_source == b.network_source
        assert a.network_source is not b.network_source

    def test_schema_shows_the_full_default_object(self):
        result = build_composed_config(
            {**_replay_layers(), "capability": _capability(default="archive")},
        )
        capability = result.layer_results["capability"].per_key_models["browser"]
        schema = generate_composed_json_schema(result)
        definition = schema["$defs"][capability.__name__]
        prop = definition["properties"]["network_source"]
        expected = {"source_id": "archive", "max_bytes": 1024, "miss_policy": {"policy_id": "strict"}}
        assert prop["default"] == expected
        omitted = capability(backend={"provider_id": "local"})
        assert omitted.model_dump(mode="json")["network_source"] == expected
        assert "network_source" not in definition["required"]
        assert "backend" in definition["required"]
        assert result.top_level_type.model_json_schema() == schema

    def test_literal_mode_never_instantiates(self):
        result = self._build(inline_wiring=False)
        capability = result.layer_results["capability"].per_key_models["browser"]
        field = capability.model_fields["network_source"]
        assert field.default_factory is None
        assert "Literal" in repr(field.annotation)

    def test_optional_receptor_is_tri_state(self):
        owner = _layer("owner", "kind")

        class Player(owner.Base):
            __registry_key__ = "player"
            __wiring__ = {"source": {"registry": "network_source", "default": "archive"}}

            def __init__(self, source: Optional[str] = None): ...

        result = build_composed_config({**_replay_layers(), "owner": owner})
        model = result.layer_results["owner"].per_key_models["player"]
        archive = result.layer_results["network_source"].per_key_models["archive"]
        assert model.__inline_wired__["source"]["optional"] is True
        assert type(model().source) is archive  # omitted -> default instance
        assert model(source=None).source is None  # null -> None
        assert model(source={"source_id": "live"}).source.source_id == "live"

    def test_af_non_constructible_default_names_required_fields(self):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(
                {**_replay_layers(archive_needs_path=True), "capability": _capability(default="archive")},
            )
        msg = str(exc.value)
        assert ".BrowserCapability.network_source:" in msg
        assert (
            "the __wiring__ default 'archive' makes the field default to the "
            "ArchiveNetwork_SourceConfig config (layer 'network_source') built from "
            "defaults, but its required field(s) ['path'] have no default. Give them "
            "defaults, choose a default key whose config can be built from defaults, "
            "or remove the default."
        ) in msg

    def test_default_on_a_nested_mode_target(self):
        llm = create_registrar(
            "nllm", RunP, discriminator_fields=["model_type", "provider"], key_separator=".",
        )

        class _OpenAI(metaclass=llm.Meta):
            __registry_key__ = "openai"
            __abstract__ = True

            def run(self) -> None: ...

            def __init__(self, temperature: float = 0.7): ...

        class Azure(_OpenAI):
            __registry_key__ = "openai.azure"

            def __init__(self, deployment: str = "d", **kwargs): ...

        class Official(_OpenAI):
            __registry_key__ = "openai.official"

            def __init__(self, endpoint: str = "e", **kwargs): ...

        agent = _layer("agent", "name")

        class Pilot(agent.Base):
            __registry_key__ = "pilot"
            __wiring__ = {"llm": {"registry": "nllm", "default": "openai.azure"}}

            def __init__(self, llm: str): ...

        result = build_composed_config({"nllm": llm, "agent": agent})
        model = result.layer_results["agent"].per_key_models["pilot"]
        azure = result.layer_results["nllm"].per_key_models["openai.azure"]
        value = model().llm
        assert type(value) is azure
        assert value.provider.deployment == "d"
        assert model.model_fields["llm"].json_schema_extra["default"]["model_type"] == "openai"


class TestParameterKeyDefault:
    """A wired ``__init__`` parameter whose own default is a key string."""

    @staticmethod
    def _loops(*, branch_needs_count: bool = False) -> Any:
        loop = _layer("loop", "name")

        class React(loop.Base):
            __registry_key__ = "react"

            def __init__(self, max_steps: int = 10): ...

        if branch_needs_count:

            class Branch(loop.Base):
                __registry_key__ = "branch"

                def __init__(self, branches: int): ...

        else:

            class Branch(loop.Base):  # type: ignore[no-redef]
                __registry_key__ = "branch"

                def __init__(self, branches: int = 2): ...

        return loop

    @staticmethod
    def _agent(default: Any) -> Any:
        agent = _layer("agent", "name")

        class Runner(agent.Base):
            __registry_key__ = "runner"
            __wiring__ = {"loop": "loop"}

            def __init__(self, loop: str = default): ...

        return agent

    def test_key_string_default_becomes_the_default_instance(self):
        result = build_composed_config({"loop": self._loops(), "agent": self._agent("react")})
        model = result.layer_results["agent"].per_key_models["runner"]
        react = result.layer_results["loop"].per_key_models["react"]
        field = model.model_fields["loop"]
        assert field.default_factory is react
        assert field.json_schema_extra == {"default": {"name": "react", "max_steps": 10}}
        assert type(model().loop) is react
        # Literal mode keeps the key string.
        literal = build_composed_config(
            {"loop": self._loops(), "agent": self._agent("react")}, inline_wiring=False,
        )
        assert literal.layer_results["agent"].per_key_models["runner"]().loop == "react"

    def test_non_constructible_key_default_is_refused(self):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(
                {"loop": self._loops(branch_needs_count=True), "agent": self._agent("branch")},
            )
        msg = str(exc.value)
        assert ".Runner.loop:" in msg
        assert (
            "the parameter default 'branch' makes the field default to the BranchLoopConfig "
            "config (layer 'loop') built from defaults, but its required field(s) "
            "['branches'] have no default. Give them defaults, choose a parameter default "
            "whose config can be built from defaults, remove the parameter default (the "
            "field becomes required), or build with inline_wiring=False."
        ) in msg

    def test_non_key_default_is_refused(self):
        sentinel = object()
        with pytest.raises((ConfigDeclarationError, WiringResolutionError)) as exc:
            build_composed_config({"loop": self._loops(), "agent": self._agent(sentinel)})
        assert "Runner.loop" in str(exc.value) or "'loop'" in str(exc.value)
        assert repr(sentinel) in str(exc.value)

    def test_default_key_narrowed_away_by_a_constraint(self):
        loop = self._loops()
        agent = _layer("agent", "name")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": {"registry": "loop", "default": "branch"}}

            def __init__(self, loop: Annotated[str, Field(pattern="^r")]): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"loop": loop, "agent": agent})
        assert (
            "the __wiring__ default 'branch' selects no config this field accepts: it "
            "holds inline 'loop' configs for the keys ['react']"
        ) in str(exc.value)


class TestDefaultMisuse:
    """Defaults a receptor holding configs cannot honour.  A ``__wiring__``
    default on a list receptor or a slot-less entry is a ``bad_default``
    wiring problem that ``validate_wiring`` reports before anything is built
    (the composed pass relies on that and does not re-check it)."""

    def test_default_on_a_list_receptor(self):
        owner = _layer("owner", "kind")

        class Many(owner.Base):
            __registry_key__ = "many"
            __wiring__ = {"sources": {"registry": "network_source", "default": "live"}}

            def __init__(self, sources: Optional[list[RunP]] = None): ...

        for inline_wiring in (True, False):
            with pytest.raises(WiringResolutionError) as exc:
                build_composed_config({**_replay_layers(), "owner": owner}, inline_wiring=inline_wiring)
            assert [(p.param_name, p.kind) for p in exc.value.problems] == [("sources", "bad_default")]
            assert "'live'" in str(exc.value)

    def test_default_on_a_slot_less_entry(self):
        owner = _layer("owner", "kind")

        class Declares(owner.Base):
            __registry_key__ = "declares"
            __wiring__ = {"source": {"registry": "network_source", "default": "live"}}

            def __init__(self, verbose: bool = False): ...

        for inline_wiring in (True, False):
            with pytest.raises(WiringResolutionError) as exc:
                build_composed_config({**_replay_layers(), "owner": owner}, inline_wiring=inline_wiring)
            assert [(p.param_name, p.kind) for p in exc.value.problems] == [("source", "bad_default")]
            assert "'live'" in str(exc.value)

    def test_non_empty_key_list_default_on_a_list_receptor(self):
        owner = _layer("owner", "kind")

        class Many(owner.Base):
            __registry_key__ = "many"
            __wiring__ = {"sources": "network_source"}

            # (``label`` keeps the 1.x extractor off its single-parameter
            # shortcut, which crashes on ``list[X]`` under Python 3.10.)
            def __init__(self, sources: list[str] = ["live"], label: str = "x"): ...  # noqa: B006

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({**_replay_layers(), "owner": owner})
        assert (
            "the default ['live'] cannot be the default of a list of inline "
            "'network_source' configs"
        ) in str(exc.value)

    def test_conflicting_defaults(self):
        owner = _layer("owner", "kind")

        class Torn(owner.Base):
            __registry_key__ = "torn"
            __wiring__ = {"source": {"registry": "network_source", "default": "live"}}

            def __init__(self, source: str = "archive"): ...

        with pytest.raises((ConfigDeclarationError, WiringResolutionError)) as exc:
            build_composed_config({**_replay_layers(), "owner": owner})
        msg = str(exc.value)
        assert "source" in msg and "'archive'" in msg and "'live'" in msg


# ===========================================================================
# F1 -- validate first
# ===========================================================================


class _BrokenSchema(RootModel[int]):
    """Building a per-key model from a RootModel is a ConfigDeclarationError."""


class TestValidateFirst:
    def test_problems_of_every_layer_are_reported_before_building(self):
        target = _layer("v2c_target", "kind")

        class Real(target.Base):
            __registry_key__ = "real"

        first = _layer("v2c_first", "kind")

        class BadKey(first.Base):
            __registry_key__ = "bad_key"
            __wiring__ = {"dep": ("v2c_target", ["reel"])}

            def __init__(self, dep: str): ...

        second = _layer("v2c_second", "kind")

        class BadRegistry(second.Base):
            __registry_key__ = "bad_registry"
            __wiring__ = {"dep": "v2c_targte", "other": {"registry": "v2c_target", "default": "nope"}}
            other: object  # a default needs a receptor slot

        class Unbuildable(second.Base):
            __registry_key__ = "unbuildable"
            __config_schema__ = _BrokenSchema

        with pytest.raises(WiringResolutionError) as exc:
            build_composed_config({"v2c_target": target, "v2c_first": first, "v2c_second": second})
        problems = [(p.owner_key, p.param_name, p.kind) for p in exc.value.problems]
        assert problems == [
            ("bad_key", "dep", "unknown_key"),
            ("bad_registry", "dep", "unknown_registry"),
            ("bad_registry", "other", "bad_default"),
        ]
        assert str(exc.value).startswith("3 wiring problems:")

    def test_validation_runs_after_the_empty_layer_check(self):
        empty = _layer("v2c_empty", "kind")
        owner = _layer("v2c_owner", "kind")

        class Bad(owner.Base):
            __registry_key__ = "bad"
            __wiring__ = {"dep": "v2c_nowhere"}

        from conscribe import EmptyLayerError

        with pytest.raises(EmptyLayerError):
            build_composed_config({"v2c_empty": empty, "v2c_owner": owner})


# ===========================================================================
# Layers addressed by registry name
# ===========================================================================


class TestLayerNames:
    @staticmethod
    def _layers() -> tuple[Any, Any]:
        llm = _layer("v2c_llm_registry", "provider")

        class OpenAI(llm.Base):
            __registry_key__ = "openai"

            def __init__(self, model: str = "gpt"): ...

        agent = _layer("v2c_agent", "name")

        class Writer(agent.Base):
            __registry_key__ = "writer"
            __wiring__ = {"llm": "v2c_llm_registry"}

            def __init__(self, llm: str): ...

        return llm, agent

    def test_section_name_differs_from_registry_name(self):
        llm, agent = self._layers()
        result = build_composed_config({"llms": llm, "agents": agent})
        assert result.dependency_order == ["llms", "agents"]
        model = result.layer_results["agents"].per_key_models["writer"]
        assert model.__inline_wired__["llm"]["target_layer"] == "llms"
        cfg = result.top_level_type.model_validate(
            {"agents": [{"name": "writer", "llm": {"provider": "openai"}}], "llms": []},
        )
        assert cfg.agents[0].llm.model == "gpt"

    def test_registry_passed_twice_and_wired_is_ambiguous(self):
        llm, agent = self._layers()
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"llm_a": llm, "llm_b": llm, "agent": agent})
        assert (
            "Writer.llm: the wiring entry targets registry 'v2c_llm_registry', which this "
            "build holds under several layer names ['llm_a', 'llm_b']"
        ) in str(exc.value)


# ===========================================================================
# Wired fields are never skipped: allowed keys come from the extractor's
# __wired_keys__ record (always written together with __wired_fields__)
# ===========================================================================


class _NetCfg(BaseModel):
    archiver_id: Literal["net"] = "net"
    host: str = "localhost"


class _DiskCfg(BaseModel):
    archiver_id: Literal["disk"] = "disk"


def _archiver_layer_result() -> LayerConfigResult:
    return LayerConfigResult(
        union_type=Annotated[Union[_NetCfg, _DiskCfg], Field(discriminator="archiver_id")],
        per_key_models={"net": _NetCfg, "disk": _DiskCfg},
        layer_name="archiver",
        discriminator_field="archiver_id",
    )


def _owner_result(model: type[BaseModel]) -> LayerConfigResult:
    return LayerConfigResult(
        union_type=model, per_key_models={"owner": model}, layer_name="hook",
        discriminator_field="hook_id",
    )


def _wired(model: type[BaseModel], **fields: tuple[str, ...]) -> type[BaseModel]:
    """Record wired fields on a hand-built model the way the extractor does
    (``__wired_fields__`` and ``__wired_keys__``, always together)."""
    model.__wired_fields__ = {name: "archiver" for name in fields}  # type: ignore[attr-defined]
    model.__wired_keys__ = dict(fields)  # type: ignore[attr-defined]
    return model


class _KeyedOwner(BaseModel):
    sink: Literal["net", "disk"]


_wired(_KeyedOwner, sink=("disk",))


class _GhostOwner(BaseModel):
    label: str = "x"


_wired(_GhostOwner, ghost=("net", "disk"))


class TestWiredFieldsAreNeverSkipped:
    def test_wired_keys_record_wins(self):
        new = _replace_wired_literals_with_unions(
            _owner_result(_KeyedOwner), {"archiver": _archiver_layer_result()},
        )
        model = new.per_key_models["owner"]
        assert model.__inline_wired__["sink"]["keys"] == ("disk",)
        assert type(model(sink={"archiver_id": "disk"}).sink) is _DiskCfg
        with pytest.raises(ValidationError):
            model(sink={"archiver_id": "net"})

    def test_receptor_typed_only_for_type_checkers_is_inlined(self):
        """A receptor annotation that cannot be resolved at runtime is never a
        reason to skip (or degrade) the field: its keys come from the wiring."""
        archiver = _archiver(inline_only=False)
        hook = _layer("hook", "hook_id")

        class Owner(hook.Base):
            __registry_key__ = "owner"
            __wiring__ = {"sink": "archiver", "sinks": "archiver"}

            def __init__(
                self,
                sink: _TypeCheckingOnlySink,
                sinks: Optional[list[_TypeCheckingOnlySink]] = None,
            ): ...

        result = build_composed_config({"archiver": archiver, "hook": hook})
        model = result.layer_results["hook"].per_key_models["owner"]
        assert not getattr(model, "__degraded_fields__", None)
        assert model.__inline_wired__ == {
            "sink": {
                "target_layer": "archiver",
                "keys": ("browser.network", "browser.storage"),
                "container": False,
                "optional": False,
            },
            "sinks": {
                "target_layer": "archiver",
                "keys": ("browser.network", "browser.storage"),
                "container": True,
                "optional": True,
            },
        }
        cfg = model(sink={"archiver_id": "browser.storage"}, sinks=[{"archiver_id": "browser.network"}])
        assert cfg.sinks[0].max_bytes == 1024

    def test_wired_field_missing_from_the_model_raises(self):
        with pytest.raises(ConfigDeclarationError, match="has no such field"):
            _replace_wired_literals_with_unions(
                _owner_result(_GhostOwner), {"archiver": _archiver_layer_result()},
            )

    def test_targets_outside_the_build_stay_key_selectors(self):
        result = _owner_result(_KeyedOwner)
        assert _replace_wired_literals_with_unions(result, {}) is result


class _FailingNetCfg(BaseModel):
    archiver_id: Literal["net"] = "net"

    @model_validator(mode="after")
    def _needs_host(self) -> "_FailingNetCfg":
        raise ValueError("a host is required")


class _OpaqueNetCfg(BaseModel):
    archiver_id: Literal["net"] = "net"
    token: Any = object()


def _single_target(model: type[BaseModel]) -> LayerConfigResult:
    return LayerConfigResult(
        union_type=model, per_key_models={"net": model}, layer_name="archiver",
        discriminator_field="archiver_id",
    )


class TestInlineDefaultsUnit:
    @staticmethod
    def _owner(**field: Any) -> type[BaseModel]:
        """An owner model whose fields are all wired to the ``net`` key."""
        return _wired(create_model("Owner", **field), **{name: ("net",) for name in field})

    def test_scalar_default_factory_is_refused(self):
        owner = self._owner(sink=(Literal["net"], Field(default_factory=lambda: "net")))
        with pytest.raises(ConfigDeclarationError) as exc:
            _replace_wired_literals_with_unions(_owner_result(owner), {"archiver": _single_target(_NetCfg)})
        assert "the default_factory" in str(exc.value)
        assert "cannot be honoured under inline wiring" in str(exc.value)

    def test_list_default_factory_is_carried(self):
        owner = self._owner(sinks=(list[Literal["net"]], Field(default_factory=list)))
        new = _replace_wired_literals_with_unions(_owner_result(owner), {"archiver": _single_target(_NetCfg)})
        field = new.per_key_models["owner"].model_fields["sinks"]
        assert field.default_factory is list

    def test_default_instance_failing_a_validator(self):
        owner = self._owner(sink=(Literal["net"], "net"))
        with pytest.raises(ConfigDeclarationError) as exc:
            _replace_wired_literals_with_unions(
                _owner_result(owner), {"archiver": _single_target(_FailingNetCfg)},
            )
        assert (
            "the parameter default 'net' makes the field default to the _FailingNetCfg config "
            "(layer 'archiver') built from defaults, but building it from defaults fails: "
            "Value error, a host is required. Fix its defaults"
        ) in str(exc.value)

    def test_default_instance_that_cannot_be_dumped_has_no_schema_default(self):
        owner = self._owner(sink=(Literal["net"], "net"))
        new = _replace_wired_literals_with_unions(
            _owner_result(owner), {"archiver": _single_target(_OpaqueNetCfg)},
        )
        field = new.per_key_models["owner"].model_fields["sink"]
        assert field.default_factory is _OpaqueNetCfg
        assert not field.json_schema_extra


# ===========================================================================
# The generated source reproduces the new constructs (three-artifact parity)
# ===========================================================================


def _parity(result: ComposedConfigResult, samples: dict[str, list[Any]]) -> None:
    from tests.config.artifact_parity import assert_artifacts_agree, composed_expectations

    assert_artifacts_agree(
        generate_composed_config_source(result),
        composed_expectations(result),
        samples=samples,
        published=(generate_composed_json_schema(result), result.top_level_type),
    )


def test_composed_source_compiles_with_new_constructs():
    result = build_composed_config(
        {**_replay_layers(), "capability": _capability(), "archiver": _archiver(), "hook": _hook()},
    )
    compile(generate_composed_config_source(result), "<composed-v2>", "exec")
    assert json.dumps(generate_composed_json_schema(result))
    assert isinstance(result.top_level_type, type) and issubclass(result.top_level_type, BaseModel)


def test_default_instances_render_with_parity():
    result = build_composed_config({**_replay_layers(), "capability": _capability(default="archive")})
    _parity(result, {
        "ComposedConfig": [
            {"capability": [{"capability_id": "browser", "backend": {"provider_id": "local"}}]},
            {"capability": [{"capability_id": "browser", "backend": {"provider_id": "local"},
                             "network_source": None}]},
            {"capability": [{"capability_id": "browser", "backend": {"provider_id": "local"},
                             "network_source": {"source_id": "live"}}]},
            {"hooks": []},
        ],
    })


def test_strict_top_level_renders_with_parity():
    result = build_composed_config(
        {"hook": _hook(), "archiver": _archiver(inline_only=False)}, top_level_extra="forbid",
    )
    _parity(result, {"ComposedConfig": [{"hooks": []}, {"archiver": [], "hook": []}]})


def test_inline_only_layers_render_with_parity():
    result = _af_build()
    _parity(result, {
        "ComposedConfig": [
            {"archiver": []},
            {"archiver": None},
            {"hook": [{"hook_id": "trace_writer", "archive": [{"archiver_id": "browser.network"}]}]},
        ],
    })


# ===========================================================================
# F3 -- paths through inline-only owners follow the receptors' keys
# ===========================================================================


def _top_mid_leaf(
    *, t1_mid_keys: Optional[list[str]] = None, t2: bool = True, m1_holds_leaf: bool = True,
) -> dict[str, Any]:
    """``top`` (top level) -> ``mid`` (inline-only) -> ``leaf`` (inline-only).

    ``mid[m1].leaf`` (unless *m1_holds_leaf* is false) and ``mid[m2].leaves``
    hold leaf configs; ``top[t1].mid`` accepts *t1_mid_keys* (every mid key
    by default) and ``top[t2].mids`` (when *t2*) only ``m1``.
    """
    top = _layer("top", "top_id")
    mid = _layer("mid", "mid_id", inline_only=True)
    leaf = _layer("leaf", "leaf_id", inline_only=True)

    class L1(leaf.Base):
        __registry_key__ = "l1"

    if m1_holds_leaf:

        class M1(mid.Base):
            __registry_key__ = "m1"
            __wiring__ = {"leaf": "leaf"}

            def __init__(self, leaf: Optional[RunP] = None): ...

    else:

        class M1(mid.Base):  # type: ignore[no-redef]
            __registry_key__ = "m1"

    class M2(mid.Base):
        __registry_key__ = "m2"
        __wiring__ = {"leaves": "leaf"}
        leaves: dict[str, RunP]

    class T1(top.Base):
        __registry_key__ = "t1"
        __wiring__ = {"mid": "mid" if t1_mid_keys is None else ("mid", t1_mid_keys)}
        mid: RunP

    if t2:

        class T2(top.Base):
            __registry_key__ = "t2"
            __wiring__ = {"mids": ("mid", ["m1"])}
            mids: list[RunP]

    return {"top": top, "mid": mid, "leaf": leaf}


class TestInlineOnlyPathsFollowKeys:
    """review-spec ``f3_nested.py`` / ``f3_unreachable.py``: a path through an
    inline-only owner only exists when the receptor before it accepts the
    owner's key."""

    LEAF_WHERE = ("top[t1].mid[m1].leaf", "top[t2].mids[m1].leaf", "top[t1].mid[m2].leaves")

    def test_paths_only_extend_through_accepted_keys(self):
        result = build_composed_config(_top_mid_leaf())
        assert [str(loc) for loc in result.inline_only_layers["leaf"]] == [
            "mid[m1].leaf", "mid[m2].leaves",
        ]
        where = " or ".join(self.LEAF_WHERE)
        top = result.top_level_type
        with pytest.raises(ValidationError) as exc:
            top.model_validate({"leaf": []})
        assert exc.value.errors()[0]["msg"] == f"layer 'leaf' is inline-only; write its instances under: {where}"
        schema = generate_composed_json_schema(result)
        assert schema["x-inline-only-layers"] == {
            "leaf": list(self.LEAF_WHERE),
            "mid": ["top[t1].mid", "top[t2].mids"],
        }
        source = generate_composed_config_source(result)
        assert repr(where) in source
        assert "top[t2].mids[m2].leaves" not in source

        # Every advertised path is writable ...
        writable = [
            {"top": [{"top_id": "t1", "mid": {"mid_id": "m1", "leaf": {"leaf_id": "l1"}}}]},
            {"top": [{"top_id": "t2", "mids": [{"mid_id": "m1", "leaf": {"leaf_id": "l1"}}]}]},
            {"top": [{"top_id": "t1", "mid": {"mid_id": "m2", "leaves": [{"leaf_id": "l1"}]}}]},
        ]
        for data in writable:
            top.model_validate(data)
        # ... and the path advertised before the fix is not: top[t2].mids takes m1 only.
        refused = {"top": [{"top_id": "t2", "mids": [{"mid_id": "m2", "leaves": [{"leaf_id": "l1"}]}]}]}
        with pytest.raises(ValidationError):
            top.model_validate(refused)
        _parity(result, {"ComposedConfig": [*writable, refused, {"leaf": []}, {"mid": None}]})

    def test_layer_reachable_only_through_refused_keys_is_refused(self):
        layers = _top_mid_leaf(t1_mid_keys=["m1"], t2=False, m1_holds_leaf=False)
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(layers)
        assert str(exc.value) == (
            "build_composed_config: layer 'leaf' is inline-only (create_registrar(..., "
            "inline_only=True)), but no path from the top level reaches a receptor that "
            "holds its configs inline, so its instances could not be written anywhere: "
            "mid[m2].leaves (no path writes a 'm2' block of layer 'mid': top[t1].mid "
            "accepts ['m1']). Allow the owning keys in the receptors that lead to them "
            "(their __wiring__ keys), give a class of a top-level layer a receptor for it, "
            "or create the registrar without inline_only=True."
        )

    def test_unreachable_receptors_are_not_listed(self):
        result = build_composed_config(_top_mid_leaf(t1_mid_keys=["m1"], t2=False))
        assert [str(loc) for loc in result.inline_only_layers["leaf"]] == ["mid[m1].leaf"]
        assert generate_composed_json_schema(result)["x-inline-only-layers"]["leaf"] == [
            "top[t1].mid[m1].leaf",
        ]

    def test_receptors_of_an_unwritable_owner_do_not_count(self):
        layers = _top_mid_leaf()
        del layers["top"]
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(layers)
        msg = str(exc.value)
        assert (
            "layer 'mid' is inline-only (create_registrar(..., inline_only=True)), but no "
            "receptor in this build holds its configs inline"
        ) in msg
        assert (
            "mid[m1].leaf (layer 'mid' itself cannot be written anywhere), mid[m2].leaves "
            "(layer 'mid' itself cannot be written anywhere)"
        ) in msg

    def test_every_layer_without_a_writable_location_is_named(self):
        layers = _top_mid_leaf(t1_mid_keys=["m1"], t2=False, m1_holds_leaf=False)
        layers["archiver"] = _archiver()
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config(layers)
        msg = str(exc.value)
        assert msg.count("is inline-only (create_registrar(..., inline_only=True))") == 2
        assert "layer 'archiver' is inline-only" in msg and "no receptor in this build" in msg
        assert "layer 'leaf' is inline-only" in msg and "no path from the top level" in msg


# ===========================================================================
# declare_only entries are never inlined (post-review decision 6)
# ===========================================================================


def _declaring_capability() -> Any:
    capability = _layer("capability", "capability_id")

    class BrowserCapability(capability.Base):
        __registry_key__ = "browser"
        __wiring__ = {
            "archiver": {"registry": "archiver", "keys": ["browser.network"], "declare_only": True},
            "recorder": {"registry": "archiver", "declare_only": True},
            # Slot-less without the flag: the framework injects it, so it
            # holds an inline config.
            "negotiated": ("archiver", ["browser.storage"]),
        }

        def __init__(self, headless: bool = True): ...

    return capability


class TestDeclareOnlyEntries:
    def _build(self) -> ComposedConfigResult:
        return build_composed_config(
            {"archiver": _archiver(), "hook": _hook(), "capability": _declaring_capability()},
        )

    def test_declare_only_entries_stay_key_constraints(self):
        result = self._build()
        model = result.layer_results["capability"].per_key_models["browser"]
        assert set(model.__inline_wired__) == {"negotiated"}
        archiver = model.model_fields["archiver"]
        assert archiver.annotation == Optional[Literal["browser.network"]]
        assert archiver.default is None
        assert model.model_fields["recorder"].annotation == Optional[
            Literal["browser.network", "browser.storage"]
        ]
        assert model(archiver="browser.network").archiver == "browser.network"
        assert model().archiver is None
        # No config block is accepted that nothing would receive.
        with pytest.raises(ValidationError) as exc:
            model(archiver={"archiver_id": "browser.network"})
        assert [t for t, _ in _errors(exc.value)] == ["literal_error"]
        with pytest.raises(ValidationError):
            model(archiver="browser.storage")  # outside the entry's keys
        assert model(negotiated={"archiver_id": "browser.storage"}).negotiated.archiver_id == (
            "browser.storage"
        )

    def test_schema_and_generated_source_agree(self):
        result = self._build()
        model = result.layer_results["capability"].per_key_models["browser"]
        prop = generate_composed_json_schema(result)["$defs"][model.__name__]["properties"]["archiver"]
        assert "$ref" not in json.dumps(prop) and prop["default"] is None
        _parity(result, {
            "ComposedConfig": [
                {"capability": [{"capability_id": "browser", "archiver": "browser.network"}]},
                {"capability": [{"capability_id": "browser", "archiver": {"archiver_id": "browser.network"}}]},
                {"capability": [{"capability_id": "browser", "recorder": "browser.storage"}]},
                {"capability": [{"capability_id": "browser", "recorder": {"archiver_id": "browser.storage"}}]},
                {"capability": [{"capability_id": "browser", "negotiated": {"archiver_id": "browser.storage"}}]},
            ],
        })

    def test_afsim_declare_only_sample_is_refused(self):
        """review-spec ``afsim_run.py`` sample 7: a nested block under a
        declare_only entry used to be accepted (and dropped by the framework)."""
        result = self._build()
        with pytest.raises(ValidationError) as exc:
            result.top_level_type.model_validate({
                "capability": [{"capability_id": "browser", "archiver": {"archiver_id": "browser.network"}}],
            })
        assert _errors(exc.value) == [("literal_error", ("capability", 0, "archiver"))]

    def test_declare_only_entries_add_no_dependency(self):
        """``holder`` holds ``child`` configs; ``child`` only constrains keys of
        the holder layer.  Not a cycle: nothing needs the holder built first."""
        holders = _layer("holders", "holder_id")
        children = _layer("children", "child_id")

        class Holder(holders.Base):
            __registry_key__ = "holder"
            __wiring__ = {"child": "children"}

            def __init__(self, child: Optional[RunP] = None): ...

        class Child(children.Base):
            __registry_key__ = "child"
            __wiring__ = {"parent": {"registry": "holders", "declare_only": True}}

        registrars = {"holders": holders, "children": children}
        assert _build_dependency_graph(registrars) == {"holders": {"children"}, "children": set()}
        result = build_composed_config(registrars)
        assert result.dependency_order == ["children", "holders"]
        child = result.layer_results["children"].per_key_models["child"]
        assert child.model_fields["parent"].annotation == Optional[Literal["holder"]]
        cfg = result.top_level_type.model_validate(
            {"holders": [{"holder_id": "holder", "child": {"child_id": "child", "parent": "holder"}}]},
        )
        assert cfg.holders[0].child.parent == "holder"


# ===========================================================================
# Layer names (post-review decision 9)
# ===========================================================================


def _plain_layer(registry: str) -> Any:
    reg = _layer(registry, "kind")

    class Plain(reg.Base):
        __registry_key__ = "plain"

        def __init__(self, x: int = 1): ...

    return reg


class TestLayerNameValidation:
    @pytest.mark.parametrize(
        ("name", "why", "suggestion"),
        [
            ("_private", "is private (pydantic does not make it a field)", "private"),
            ("model_config", "shadows the BaseModel attribute 'model_config'", "config"),
            ("model_fields", "shadows the BaseModel attribute 'model_fields'", "fields"),
            ("schema", "shadows the BaseModel attribute 'schema'", "schema_layer"),
            ("json", "shadows the BaseModel attribute 'json'", "json_layer"),
            ("copy", "shadows the BaseModel attribute 'copy'", "copy_layer"),
            ("dict", "shadows the BaseModel attribute 'dict'", "dict_layer"),
            ("validate", "shadows the BaseModel attribute 'validate'", "validate_layer"),
            ("construct", "shadows the BaseModel attribute 'construct'", "construct_layer"),
            ("model_x", "starts with pydantic's protected prefix 'model_'", "x"),
            ("my-layer", "is not a valid identifier", "my_layer"),
            ("2fa", "is not a valid identifier", "layer_2fa"),
            ("class", "is a Python keyword", "class_layer"),
            (1, "is of type int, not str", "layer_1"),
        ],
    )
    def test_refused_names(self, name, why, suggestion):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({name: _plain_layer("plain_layer")})
        msg = str(exc.value)
        assert msg.startswith(
            f"build_composed_config: invalid layer name(s): {name!r} {why} (pass it as {suggestion!r})."
        )
        assert msg.endswith(
            "Pass the registrar under another name: only the section name changes, wiring "
            "entries find layers by their registry name."
        )

    def test_names_are_checked_before_anything_else(self):
        empty = create_registrar("empty_layer", RunP, discriminator_field="kind")
        with pytest.raises(ConfigDeclarationError, match="'_empty' is private"):
            build_composed_config({"_empty": empty})

    def test_every_bad_name_is_named(self):
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({
                "schema": _plain_layer("one"), "ok": _plain_layer("two"), "_x": _plain_layer("three"),
            })
        msg = str(exc.value)
        assert "'schema' shadows the BaseModel attribute 'schema' (pass it as 'schema_layer');" in msg
        assert "'_x' is private" in msg
        assert "'ok'" not in msg

    def test_renaming_the_section_keeps_the_wiring(self):
        schema = _layer("schema", "kind")

        class Strict(schema.Base):
            __registry_key__ = "strict"

        owner = _layer("owner", "owner_id")

        class Validator(owner.Base):
            __registry_key__ = "validator"
            __wiring__ = {"rules": {"registry": "schema", "default": "strict"}}
            rules: RunP

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"schema": schema, "owner": owner})
        assert (
            "'schema' shadows the BaseModel attribute 'schema' (pass it as 'schema_layer'; "
            "wiring entries still reach it as registry 'schema')"
        ) in str(exc.value)

        result = build_composed_config({"schema_layer": schema, "owner": owner})
        model = result.layer_results["owner"].per_key_models["validator"]
        assert model.__inline_wired__["rules"]["target_layer"] == "schema_layer"
        assert type(model().rules) is result.layer_results["schema_layer"].per_key_models["strict"]

    def test_accepted_names_build_without_warnings(self):
        """Soft keywords and names that are not BaseModel attributes are fine
        (``fields`` is not one in any supported pydantic version)."""
        names = ["fields", "type", "match", "model", "schemas", "validators"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = build_composed_config({name: _plain_layer(f"reg_{name}") for name in names})
        assert result.top_level_layers == sorted(names)
        _parity(result, {
            "ComposedConfig": [{name: [{"kind": "plain", "x": 2}] for name in names}, {"model": [{}]}],
        })


# ===========================================================================
# Registrar arguments
# ===========================================================================


class TestRegistrarArguments:
    def test_registrars_must_be_a_mapping(self):
        reg = _plain_layer("plain_layer")
        with pytest.raises(TypeError) as exc:
            build_composed_config([reg])  # type: ignore[arg-type]
        assert str(exc.value).startswith(
            "build_composed_config() expects a dict mapping layer names to registrars "
            "created by create_registrar(), got list ["
        )

    def test_every_non_registrar_is_named(self):
        reg = _plain_layer("plain_layer")
        registered = reg.get("plain")
        with pytest.raises(TypeError) as exc:
            build_composed_config({
                "ok": reg, "impl": registered, "name": "plain_layer", "base": LayerRegistrar,
                "instance": 3,
            })
        assert str(exc.value) == (
            "build_composed_config() expects registrars created by create_registrar() (the "
            "class it returns, not a registered class or an instance): layer 'impl' is "
            f"class {registered.__module__}.{registered.__qualname__}; layer 'name' is str "
            "'plain_layer'; layer 'base' is class "
            "conscribe.registration.registrar.LayerRegistrar; layer 'instance' is int 3"
        )


# ===========================================================================
# Key-string constraints inside Optional / unions (review m1) and
# constraints that reject every key (review m2)
# ===========================================================================


def _three_loops() -> Any:
    loop = _layer("loop", "loop_id")

    class React(loop.Base):
        __registry_key__ = "react"

    class Plan(loop.Base):
        __registry_key__ = "plan"

    class Tree(loop.Base):
        __registry_key__ = "tree"

    return loop


LOOP_KEYS = ("react", "plan", "tree")


def _per_layer_keys(agent: Any) -> set[str]:
    """The keys the per-layer model's ``loop`` field accepts."""
    model = build_layer_config(agent).per_key_models["picky"]
    accepted = set()
    for key in LOOP_KEYS:
        try:
            model.model_validate({"agent_id": "picky", "loop": key})
        except ValidationError:
            continue
        accepted.add(key)
    return accepted


class TestKeyConstraintsInsideAnnotations:
    def _inline_keys(self, loop: Any, agent: Any) -> set[str]:
        result = build_composed_config({"loop": loop, "agent": agent})
        model = result.layer_results["agent"].per_key_models["picky"]
        keys = set(model.__inline_wired__["loop"]["keys"])
        for key in LOOP_KEYS:
            data = {"agent_id": "picky", "loop": {"loop_id": key}}
            if key in keys:
                model.model_validate(data)
            else:
                with pytest.raises(ValidationError):
                    model.model_validate(data)
        return keys

    def test_string_constraints_inside_optional(self):
        """review-build ``s08_default_errors.py``: the pattern was ignored."""
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(self, loop: Optional[Annotated[str, StringConstraints(pattern="^re")]] = None): ...

        assert self._inline_keys(loop, agent) == _per_layer_keys(agent) == {"react"}

    def test_min_len_inside_optional(self):
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(self, loop: Optional[Annotated[str, MinLen(5)]] = None): ...

        assert self._inline_keys(loop, agent) == _per_layer_keys(agent) == {"react"}

    def test_union_members_narrow_like_pydantic(self):
        """A key passes when the field's constraints and those of at least one
        member accept it."""
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(
                self,
                loop: Union[
                    Annotated[str, StringConstraints(pattern="^re")],
                    Annotated[str, StringConstraints(pattern="^pl")],
                    None,
                ] = None,
            ): ...

        assert self._inline_keys(loop, agent) == _per_layer_keys(agent) == {"react", "plan"}

    def test_unconstrained_member_admits_every_key(self):
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(self, loop: Union[Annotated[str, MinLen(5)], RunP, None] = None): ...

        assert self._inline_keys(loop, agent) == _per_layer_keys(agent) == set(LOOP_KEYS)


class TestConstraintsRejectingEveryKey:
    @pytest.mark.parametrize("inline_wiring", [True, False])
    def test_field_constraint(self, inline_wiring):
        """review-build ``s31_zero_keys.py``: inline_wiring=False built silently."""
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(self, loop: Annotated[str, Field(pattern="^zz")]): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"loop": loop, "agent": agent}, inline_wiring=inline_wiring)
        msg = str(exc.value)
        assert ".Picky.loop: the key constraints (pattern='^zz') reject every key the wiring allows" in msg
        assert "Fix the constraint or the __wiring__ keys." in msg

    @pytest.mark.parametrize("inline_wiring", [True, False])
    def test_constraints_inside_and_around_optional(self, inline_wiring):
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(
                self, loop: Annotated[Optional[Annotated[str, MinLen(5)]], Field(max_length=4)] = None,
            ): ...

        assert _per_layer_keys(agent) == set()
        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"loop": loop, "agent": agent}, inline_wiring=inline_wiring)
        assert "the key constraints (max_length=4, min_length=5) reject every key" in str(exc.value)

    @pytest.mark.parametrize("inline_wiring", [True, False])
    def test_every_union_member_rejects_every_key(self, inline_wiring):
        loop = _three_loops()
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"loop": "loop"}

            def __init__(
                self,
                loop: Union[
                    Annotated[str, StringConstraints(pattern="^x")],
                    Annotated[str, StringConstraints(pattern="^y")],
                    None,
                ] = None,
            ): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"loop": loop, "agent": agent}, inline_wiring=inline_wiring)
        msg = str(exc.value)
        assert ".Picky.loop: the key constraints of every member of its annotation" in msg
        assert "reject every key the wiring allows" in msg

    def test_key_selectors_that_are_not_inlined_are_checked_too(self):
        agent = _layer("agent", "agent_id")

        class Picky(agent.Base):
            __registry_key__ = "picky"
            __wiring__ = {"browser": ["chromium", "firefox"]}  # a literal list: never inlined

            def __init__(self, browser: Annotated[str, Field(min_length=10)]): ...

        with pytest.raises(ConfigDeclarationError) as exc:
            build_composed_config({"agent": agent})
        assert ".Picky.browser: the key constraints (min_length=10) reject every key" in str(exc.value)


# ===========================================================================
# Cycles (review m3)
# ===========================================================================


def test_the_reported_cycle_is_real():
    """review-build ``s34_cycle.py``: ``a <-> b`` plus ``c -> a`` was reported
    as ``a -> b -> c -> a``."""
    a, b, c = (_layer(name, "k") for name in ("a", "b", "c"))

    class A(a.Base):
        __registry_key__ = "x"
        __wiring__ = {"t": "b"}

        def __init__(self, t: Optional[RunP] = None): ...

    class B(b.Base):
        __registry_key__ = "x"
        __wiring__ = {"t": "a"}

        def __init__(self, t: Optional[RunP] = None): ...

    class C(c.Base):
        __registry_key__ = "x"
        __wiring__ = {"t": "a"}

        def __init__(self, t: Optional[RunP] = None): ...

    with pytest.raises(CircularWiringError) as exc:
        build_composed_config({"a": a, "b": b, "c": c})
    assert exc.value.cycle == ["a", "b"]
    assert str(exc.value).startswith("Circular wiring dependency detected: a -> b -> a.")


# ===========================================================================
# Default instances do not depend on hash order (review m4)
# ===========================================================================


def _set_default_layers() -> dict[str, Any]:
    """An owner whose receptor defaults to a config holding sets.

    Module level so the hash-seed test can build it in a subprocess.
    """
    tagger = _layer("tagger", "tagger_id")

    class Tagger(tagger.Base):
        __registry_key__ = "tags"

        def __init__(
            self,
            tags: set[str] = {"red", "green", "blue", "cyan", "teal"},  # noqa: B006
            frozen: frozenset[str] = frozenset({"k1", "k2", "k3", "k4"}),
            nested: dict[str, frozenset[str]] = {"x": frozenset({"p", "q", "r", "s"})},  # noqa: B006
            ordered: tuple[str, ...] = ("z", "y", "x"),
        ): ...

    owner = _layer("owner", "owner_id")

    class Owner(owner.Base):
        __registry_key__ = "owner"
        __wiring__ = {"tagger": {"registry": "tagger", "default": "tags"}}
        tagger: RunP

    return {"tagger": tagger, "owner": owner}


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HASH_SEED_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from conscribe.config.codegen import generate_composed_config_source
    from conscribe.config.composed import build_composed_config
    from conscribe.config.json_schema import generate_composed_json_schema
    from conscribe.testing import isolated_registries
    import tests.config.test_composed_v2 as v2

    with isolated_registries():
        result = build_composed_config(v2._set_default_layers())
        sys.stdout.write(generate_composed_config_source(result))
        sys.stdout.write(json.dumps(generate_composed_json_schema(result), indent=1))
    """
)


class TestDefaultInstanceDeterminism:
    EXPECTED = {
        "tags": ["blue", "cyan", "green", "red", "teal"],
        "frozen": ["k1", "k2", "k3", "k4"],
        "nested": {"x": ["p", "q", "r", "s"]},
        "ordered": ["z", "y", "x"],
        "tagger_id": "tags",
    }

    def test_set_members_are_sorted_and_sequences_keep_their_order(self):
        result = build_composed_config(_set_default_layers())
        model = result.layer_results["owner"].per_key_models["owner"]
        assert model.model_fields["tagger"].json_schema_extra == {"default": self.EXPECTED}
        schema = generate_composed_json_schema(result)
        assert schema["$defs"][model.__name__]["properties"]["tagger"]["default"] == self.EXPECTED
        in_process = result.top_level_type.model_json_schema()
        assert in_process["$defs"][model.__name__]["properties"]["tagger"]["default"] == self.EXPECTED
        _parity(result, {"ComposedConfig": [{"owner": [{"owner_id": "owner"}]}]})

    def test_artifacts_do_not_depend_on_the_hash_seed(self):
        outputs = []
        for seed in ("1", "2", "3"):
            proc = subprocess.run(
                [sys.executable, "-c", _HASH_SEED_SCRIPT],
                cwd=_ROOT,
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            outputs.append(proc.stdout)
        assert "'tags': ['blue', 'cyan', 'green', 'red', 'teal']" in outputs[0]
        assert outputs[0] == outputs[1] == outputs[2]

    def test_json_only_member_serializers_keep_their_json_form(self):
        """Members whose JSON form only a JSON-mode serializer knows keep
        pydantic's form; they are still sorted."""

        class Marks(BaseModel):
            marks: set[
                Annotated[int, PlainSerializer(lambda v: f"#{v}", return_type=str, when_used="json")]
            ] = {30, 1, 200}

        assert _schema_default(Marks()) == {"default": {"marks": ["#1", "#200", "#30"]}}

    def test_nested_sets_inside_set_members_are_sorted(self):
        class Groups(BaseModel):
            groups: frozenset[frozenset[str]] = frozenset(
                {frozenset({"n2", "n1", "n3"}), frozenset({"m3", "m1", "m2"})}
            )
            pairs: set[tuple[str, str]] = {("b", "a"), ("a", "b")}

        assert _schema_default(Groups()) == {
            "default": {
                "groups": [["m1", "m2", "m3"], ["n1", "n2", "n3"]],
                "pairs": [["a", "b"], ["b", "a"]],  # tuples keep their order
            },
        }

    def test_sets_of_models_are_sorted_with_the_sets_inside_them(self):
        """pydantic cannot dump a set of models in python mode (its members
        become unhashable dicts); the JSON form is walked with the instance."""

        class Tag(BaseModel):
            model_config = {"frozen": True}
            rank: int = 1
            labels: frozenset[str] = frozenset({"x", "y", "z", "w"})

        class Tagged(BaseModel):
            tags: frozenset[Tag] = frozenset({Tag(rank=2), Tag(rank=1)})
            by_alias: set[str] = Field(default={"q", "p"}, alias="byAlias")

        assert _schema_default(Tagged()) == {
            "default": {
                "tags": [
                    {"rank": 1, "labels": ["w", "x", "y", "z"]},
                    {"rank": 2, "labels": ["w", "x", "y", "z"]},
                ],
                "byAlias": ["p", "q"],
            },
        }

    def test_serializer_failing_in_python_mode_does_not_matter(self):
        class JsonOnly(BaseModel):
            values: set[str] = {"b", "a", "c"}

            @field_serializer("values")
            def _ser(self, value: Any, info: Any) -> Any:
                if info.mode == "python":
                    raise ValueError("python mode is not supported")
                return sorted(value, reverse=True)

        assert _schema_default(JsonOnly()) == {"default": {"values": ["a", "b", "c"]}}


# ===========================================================================
# Public names
# ===========================================================================


def test_composed_names_are_exported():
    import conscribe
    import conscribe.config as config
    from conscribe.config import composed

    for module in (conscribe, config):
        assert module.ReceptorLocation is composed.ReceptorLocation
        assert module.TopLevelExtra is composed.TopLevelExtra
        assert module.INLINE_ONLY_ERROR_TYPE == composed.INLINE_ONLY_ERROR_TYPE == "inline_only_layer"
        assert {"ReceptorLocation", "TopLevelExtra", "INLINE_ONLY_ERROR_TYPE"} <= set(module.__all__)
    assert get_args(config.TopLevelExtra) == ("forbid", "ignore", "allow")


# ===========================================================================
# One build, one set of model names; layer checks before wiring problems
# ===========================================================================


def test_model_names_are_unique_across_layers():
    """review-build ``s32_cross_layer_names.py``: a nested segment model and
    another layer's per-key model were both ``AzureProviderConfig`` (mangled
    ``$defs``, ``AzureProviderConfig2`` in the generated file only)."""
    llm = create_registrar(
        "llm", RunP, discriminator_fields=["family", "provider"], key_separator=".",
    )

    class _OpenAI(metaclass=llm.Meta):
        __abstract__ = True
        __registry_key__ = "openai"

        def run(self) -> None: ...

        def __init__(self, temperature: float = 0.7, **kwargs: Any): ...

    class Azure(_OpenAI):
        __registry_key__ = "openai.azure"

        def __init__(self, deployment: str = "d", **kwargs: Any): ...

    provider = _layer("provider", "provider_id")

    class AzureProvider(provider.Base):
        __registry_key__ = "azure"

        def __init__(self, region: str = "eu"): ...

    agent = _layer("agent", "agent_id")

    class A(agent.Base):
        __registry_key__ = "a"
        __wiring__ = {
            "llm": {"registry": "llm", "default": "openai.azure"},
            "prov": {"registry": "provider", "default": "azure"},
        }

        def __init__(self, llm: Optional[RunP] = None, prov: Optional[RunP] = None): ...

    result = build_composed_config({"llm": llm, "provider": provider, "agent": agent})
    azure = result.layer_results["llm"].per_key_models["openai.azure"]
    segment = azure.model_fields["provider"].annotation
    per_key = result.layer_results["provider"].per_key_models["azure"]
    assert (segment.__name__, per_key.__name__) == ("AzureProviderConfig", "AzureProviderConfig2")
    schema = generate_composed_json_schema(result)
    assert {"AzureProviderConfig", "AzureProviderConfig2", "ComposedConfig"} - {
        *schema["$defs"], schema["title"],
    } == set()
    assert not [name for name in schema["$defs"] if "__" in name]
    _parity(result, {
        "ComposedConfig": [
            {"agent": [{"agent_id": "a"}]},
            {"agent": [{"agent_id": "a", "prov": {"provider_id": "azure", "region": "us"}}]},
            {"provider": [{"provider_id": "azure"}], "llm": [{"family": "openai", "provider": {"name": "azure"}}]},
        ],
    })


def test_layer_declaration_errors_come_before_wiring_problems():
    """A nested-mode class declaring ``__wiring__`` is reported as unsupported,
    not hidden behind the problems of the wiring it should not declare."""
    llm = create_registrar(
        "nested_llm", RunP, discriminator_fields=["family", "provider"], key_separator=".",
    )

    class _OpenAI(metaclass=llm.Meta):
        __abstract__ = True
        __registry_key__ = "openai"

        def run(self) -> None: ...

    class Azure(_OpenAI):
        __registry_key__ = "openai.azure"
        __wiring__ = {"loop": "no_such_registry"}

    with pytest.raises(ConfigDeclarationError, match="not supported for nested-mode layers yet"):
        build_composed_config({"nested_llm": llm})
