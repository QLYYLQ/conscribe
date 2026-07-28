"""Regression tests for wired-field *shape* in generated configs.

Two independent bugs, both in ``_apply_wiring`` / ``_replace_str_with_literal``:

Bug 1 — slot-less wired entries were emitted as **required** config fields.
    A ``__wiring__`` key with no runtime slot (no ``__init__`` parameter and
    no class-level annotation) is a purely declarative constraint: its only
    job is to produce a ``Literal[...]`` so illegal combinations fail
    validation.  Nothing can receive a value for it, yet every config file
    was forced to carry one.  It must be optional — without weakening the
    ``Literal`` when a value *is* supplied.

Bug 2 — container-typed receptors collapsed to a **scalar** ``Literal``.
    ``capabilities: dict[str, Capability]`` could only ever declare one
    capability, making composite environments unreachable from config.
    Container-annotated receptors must emit as a *list* of selectors, which
    the composed pass widens into a list of nested configs.

The shapes mirror ``spectragent``'s ``EnvironmentBase`` / ``BrowserUseAgent``.
"""
from __future__ import annotations

import sys
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Protocol,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

import pytest
from pydantic import Field, ValidationError

from conscribe import create_registrar
from conscribe.config.composed import build_composed_config
from conscribe.config.extractor import extract_config_schema
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister


# ── Fixtures ─────────────────────────────────────────────────────


@runtime_checkable
class CapabilityProtocol(Protocol):
    def go(self) -> None: ...


@runtime_checkable
class FragmentProtocol(Protocol):
    def resolve(self) -> None: ...


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("slot_"):
            _deregister(name)


@pytest.fixture
def capability_registrar():
    reg = create_registrar(
        "slot_capability", CapabilityProtocol, discriminator_field="capability_id"
    )

    class BrowserCapability:
        def go(self) -> None: ...

    class DockerCapability:
        def go(self) -> None: ...

    reg.register("browser")(BrowserCapability)
    reg.register("docker")(DockerCapability)
    return reg


@pytest.fixture
def fragment_registrar():
    reg = create_registrar(
        "slot_observation", FragmentProtocol, discriminator_field="fragment_id"
    )

    class DomFragment:
        def resolve(self) -> None: ...

    class ScreenshotFragment:
        def resolve(self) -> None: ...

    reg.register("browser.dom")(DomFragment)
    reg.register("browser.screenshot")(ScreenshotFragment)
    return reg


def _literal_members(annotation):
    """Peel ``Optional`` / ``list`` wrappers off a wired annotation."""
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin is Union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            members = _literal_members(arg)
            if members is not None:
                return members
        return None
    if origin in (list, set, frozenset, tuple):
        for arg in get_args(annotation):
            members = _literal_members(arg)
            if members is not None:
                return members
    return None


def _is_optional(annotation) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _is_list(annotation) -> bool:
    if get_origin(annotation) is list:
        return True
    if get_origin(annotation) is Union:
        return any(
            arg is not type(None) and _is_list(arg) for arg in get_args(annotation)
        )
    return False


# ── Bug 1: slot-less wired entries are optional ──────────────────


class TestSlotLessWiringIsOptional:
    def test_slot_less_entry_is_not_required(self, fragment_registrar):
        class Agent:
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

        model = extract_config_schema(Agent)
        field = model.model_fields["observation"]
        assert field.is_required() is False
        assert field.default is None
        assert _is_optional(field.annotation)

    def test_literal_constraint_still_applies(self, fragment_registrar):
        """Optional is about *presence*, not about weakening validation."""

        class Agent:
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

        model = extract_config_schema(Agent)

        # Omitted → fine (this is the fix).
        assert model().observation is None
        # Supplied and legal → fine.
        assert model(observation="browser.dom").observation == "browser.dom"
        # Supplied and illegal → still rejected.
        with pytest.raises(ValidationError):
            model(observation="browser.screenshot")

    def test_all_wired_modes_go_optional_when_slot_less(self, fragment_registrar):
        class Agent:
            __wiring__ = {
                "auto": "slot_observation",  # Mode 1
                "subset": ("slot_observation", ["browser.dom"]),  # Mode 2
                "three": (  # Mode 2, 3-element
                    "slot_observation",
                    ["browser.dom"],
                    ["browser.screenshot"],
                ),
                "literal": ["chromium", "firefox"],  # Mode 3
            }

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

        model = extract_config_schema(Agent)
        for name in ("auto", "subset", "three", "literal"):
            assert model.model_fields[name].is_required() is False, name
            assert _literal_members(model.model_fields[name].annotation) is not None

    def test_classvar_annotation_is_not_a_slot(self, fragment_registrar):
        """A ClassVar is class config, never a per-instance receptor."""

        class Agent:
            observation: ClassVar[Any] = None
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

        model = extract_config_schema(Agent)
        assert model.model_fields["observation"].is_required() is False

    def test_generated_source_renders_optional(self, fragment_registrar):
        from conscribe.config.builder import build_layer_config
        from conscribe.config.codegen import generate_layer_config_source

        agent_reg = create_registrar(
            "slot_agent", CapabilityProtocol, discriminator_field="agent_id"
        )

        class ReactAgent:
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

            def go(self) -> None: ...

        agent_reg.register("react")(ReactAgent)
        source = generate_layer_config_source(build_layer_config(agent_reg))

        assert "observation: Optional[Literal['browser.dom']] = None" in source
        compile(source, "<config>", "exec")


class TestSlotFulWiringStaysRequired:
    """A wired entry that *does* have a runtime slot must not become optional."""

    def test_init_param_slot_stays_required(self, fragment_registrar):
        class Agent:
            __config_annotated_only__ = True  # filters ``observation`` out
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(
                self,
                observation: str = "browser.dom",
                max_steps: Annotated[int, Field(default=10, description="steps")] = 10,
            ):
                self.observation = observation

        model = extract_config_schema(Agent)
        assert model.model_fields["observation"].is_required() is True

    def test_own_class_annotation_slot_stays_required(self, fragment_registrar):
        class Agent:
            observation: Any
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

        model = extract_config_schema(Agent)
        assert model.model_fields["observation"].is_required() is True

    def test_inherited_annotation_slot_stays_required(self, fragment_registrar):
        """Mirrors spectragent's ``AgentReceptors`` annotation-only base."""

        class Receptors:
            observation: Any

        class Agent(Receptors):
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, max_steps: int = 10):
                self.max_steps = max_steps

        model = extract_config_schema(Agent)
        assert model.model_fields["observation"].is_required() is True

    def test_constrained_init_param_keeps_its_default(self, fragment_registrar):
        """The pre-existing ``str`` → ``Literal`` path is untouched."""

        class Agent:
            __wiring__ = {"observation": ("slot_observation", ["browser.dom"])}

            def __init__(self, observation: str = "browser.dom"):
                self.observation = observation

        model = extract_config_schema(Agent)
        field = model.model_fields["observation"]
        assert field.is_required() is False
        assert field.default == "browser.dom"
        assert get_origin(field.annotation) is Literal


# ── Bug 2: container receptors emit a list ───────────────────────


class TestContainerReceptor:
    def test_dict_init_param_becomes_list_of_selectors(self, capability_registrar):
        """The exact ``EnvironmentBase`` shape: ``dict[str, X] | None = None``."""

        class Environment:
            __config_annotated_only__ = True
            capabilities: dict[str, Any]
            __wiring__ = {"capabilities": ("slot_capability", ["browser", "docker"])}

            def __init__(
                self,
                capabilities: Annotated[
                    Optional[Dict[str, Any]],
                    Field(default=None, description="Capabilities owned"),
                ] = None,
            ):
                self.capabilities = capabilities

        model = extract_config_schema(Environment)
        annotation = model.model_fields["capabilities"].annotation
        assert _is_list(annotation), annotation
        assert sorted(_literal_members(annotation)) == ["browser", "docker"]

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="PEP 604 unions need 3.10+"
    )
    def test_pep604_union_container_is_matched(self, capability_registrar):
        """``dict[str, X] | None`` — ``get_origin`` returns ``types.UnionType``.

        The old code only checked ``origin is Union``, so this fell through
        to the catch-all and was replaced by a bare scalar ``Literal``.
        """
        namespace: dict = {}
        exec(
            "from __future__ import annotations\n"
            "from typing import Annotated, Any\n"
            "from pydantic import Field\n"
            "class Environment:\n"
            "    __config_annotated_only__ = True\n"
            "    capabilities: dict[str, Any]\n"
            "    __wiring__ = {'capabilities': ('slot_capability', ['browser', 'docker'])}\n"
            "    def __init__(self, capabilities: Annotated[dict[str, Any] | None,"
            " Field(default=None, description='owned')] = None):\n"
            "        self.capabilities = capabilities\n",
            namespace,
        )
        model = extract_config_schema(namespace["Environment"])
        annotation = model.model_fields["capabilities"].annotation
        assert _is_list(annotation), annotation
        assert _is_optional(annotation), annotation
        assert sorted(_literal_members(annotation)) == ["browser", "docker"]

    def test_list_annotation_receptor(self, capability_registrar):
        class Environment:
            __wiring__ = {"capabilities": ("slot_capability", ["browser", "docker"])}

            def __init__(self, capabilities: list = []):
                self.capabilities = capabilities

        model = extract_config_schema(Environment)
        assert _is_list(model.model_fields["capabilities"].annotation)

    def test_container_class_annotation_without_init_param(self, capability_registrar):
        """Injected (no ``__init__`` param) but annotated as a container."""

        class Environment:
            capabilities: Dict[str, Any]
            __wiring__ = {"capabilities": ("slot_capability", ["browser", "docker"])}

            def __init__(self, name: str = "env"):
                self.name = name

        model = extract_config_schema(Environment)
        annotation = model.model_fields["capabilities"].annotation
        assert _is_list(annotation), annotation
        assert model.model_fields["capabilities"].is_required() is True

    def test_string_annotation_container_is_detected(self, capability_registrar):
        """PEP 563 leaves the annotation as text; detection must still work."""

        class Environment:
            __annotations__ = {"capabilities": "dict[str, Any]"}
            __wiring__ = {"capabilities": ("slot_capability", ["browser", "docker"])}

            def __init__(self, name: str = "env"):
                self.name = name

        model = extract_config_schema(Environment)
        assert _is_list(model.model_fields["capabilities"].annotation)

    def test_scalar_receptor_is_unaffected(self, capability_registrar):
        class Environment:
            capability: Any
            __wiring__ = {"capability": ("slot_capability", ["browser"])}

            def __init__(self, name: str = "env"):
                self.name = name

        model = extract_config_schema(Environment)
        annotation = model.model_fields["capability"].annotation
        assert get_origin(annotation) is Literal
        assert not _is_list(annotation)


class TestContainerReceptorComposed:
    """The payoff: multi-capability environments become expressible."""

    def _build(self, capability_registrar):
        env_reg = create_registrar(
            "slot_environment", CapabilityProtocol, discriminator_field="env_type"
        )

        class CompositeEnvironment:
            __config_annotated_only__ = True
            capabilities: Dict[str, Any]
            __wiring__ = {"capabilities": ("slot_capability", ["browser", "docker"])}

            def __init__(
                self,
                capabilities: Annotated[
                    Optional[Dict[str, Any]],
                    Field(default=None, description="Capabilities owned"),
                ] = None,
            ):
                self.capabilities = capabilities

            def go(self) -> None: ...

        env_reg.register("composite")(CompositeEnvironment)
        return build_composed_config(
            {"slot_capability": capability_registrar, "slot_environment": env_reg},
            inline_wiring=True,
        )

    def test_composed_field_is_a_list_of_nested_configs(self, capability_registrar):
        result = self._build(capability_registrar)
        model = result.layer_results["slot_environment"].per_key_models["composite"]
        annotation = model.model_fields["capabilities"].annotation
        assert _is_list(annotation), annotation

    def test_two_capabilities_validate(self, capability_registrar):
        """This is what was unreachable before: declaring more than one."""
        result = self._build(capability_registrar)
        model = result.layer_results["slot_environment"].per_key_models["composite"]

        validated = model(
            capabilities=[
                {"capability_id": "browser"},
                {"capability_id": "docker"},
            ]
        )
        assert len(validated.capabilities) == 2
        ids = {c.capability_id for c in validated.capabilities}
        assert ids == {"browser", "docker"}

    def test_composed_source_compiles(self, capability_registrar):
        from conscribe.config.codegen import generate_composed_config_source

        source = generate_composed_config_source(self._build(capability_registrar))
        assert "capabilities: Optional[list[" in source
        compile(source, "<composed>", "exec")
