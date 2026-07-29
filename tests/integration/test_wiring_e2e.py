"""End-to-end tests for __wiring__: two registries, full pipeline."""
from __future__ import annotations

import pytest
from typing import (
    Literal,
    Optional,
    Protocol,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

from pydantic import TypeAdapter

from conscribe import create_registrar
from conscribe.config import (
    build_layer_config,
    generate_layer_config_source,
    generate_layer_json_schema,
)
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def literal_keys(annotation) -> tuple:
    """Return the ``Literal`` members of a wired field's annotation.

    Slot-less wired fields (no ``__init__`` param, no class annotation) are
    emitted as ``Optional[Literal[...]]``, so config files need not carry a
    value that nothing can receive.  This unwraps that wrapper so a test can
    assert on the constraint itself.
    """
    if get_origin(annotation) is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        assert len(non_none) == 1, annotation
        annotation = non_none[0]
    assert get_origin(annotation) is Literal, annotation
    return get_args(annotation)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LoopProtocol(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class AgentProtocol(Protocol):
    def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean up registries after each test."""
    yield
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("e2e_"):
            _deregister(name)


# ---------------------------------------------------------------------------
# Test: Full pipeline with wiring
# ---------------------------------------------------------------------------


class TestWiringE2E:
    def test_full_pipeline_mode1_auto_discovery(self):
        """Mode 1: agent wires loop field to all keys in loop registry."""
        # Create loop registry and register implementations
        Loop = create_registrar("e2e_loop", LoopProtocol, discriminator_field="name")

        class BaseLoop(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop(BaseLoop):
            def run(self) -> None: ...

        class CodeactLoop(BaseLoop):
            def run(self) -> None: ...

        # Create agent registry
        Agent = create_registrar("e2e_agent", AgentProtocol, discriminator_field="name")

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True
            __wiring__ = {"loop": "e2e_loop"}

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class SWEAgent(BaseAgent):
            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "swe"

            def reset(self) -> None: ...

        # Build config
        result = build_layer_config(Agent)
        assert "swe_agent" in result.per_key_models

        model = result.per_key_models["swe_agent"]

        # Verify wired field was injected.  ``loop`` has no runtime slot
        # (no __init__ param, no class annotation), so it is optional.
        assert "loop" in model.model_fields
        assert model.model_fields["loop"].is_required() is False
        loop_type = model.model_fields["loop"].annotation

        # Verify Literal values contain all loop keys
        literal_args = literal_keys(loop_type)
        assert sorted(literal_args) == ["codeact_loop", "react_loop"]

        # Verify __init__ params are still there
        assert "max_steps" in model.model_fields

        # Verify __wired_fields__ metadata
        assert hasattr(model, "__wired_fields__")
        assert model.__wired_fields__["loop"] == "e2e_loop"

    def test_full_pipeline_mode2_explicit_subset(self):
        """Mode 2: agent wires loop to explicit subset."""
        Loop = create_registrar("e2e_loop2", LoopProtocol, discriminator_field="name")

        class BaseLoop2(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop2(BaseLoop2):
            def run(self) -> None: ...

        class CodeactLoop2(BaseLoop2):
            def run(self) -> None: ...

        class PlanActLoop2(BaseLoop2):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent2", AgentProtocol, discriminator_field="name")

        class BaseAgent2(metaclass=Agent.Meta):
            __abstract__ = True
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class SWEAgent2(BaseAgent2):
            __wiring__ = {"loop": ("e2e_loop2", ["react_loop2", "codeact_loop2"])}

            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "swe"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        model = result.per_key_models["swe_agent2"]

        literal_args = literal_keys(model.model_fields["loop"].annotation)
        assert sorted(literal_args) == ["codeact_loop2", "react_loop2"]  # type: ignore[type-var]

    def test_full_pipeline_mode3_literal_list(self):
        """Mode 3: literal list without registry reference."""
        Agent = create_registrar("e2e_agent3", AgentProtocol, discriminator_field="name")

        class BaseAgent3(metaclass=Agent.Meta):
            __abstract__ = True
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class BrowserAgent(BaseAgent3):
            __wiring__ = {"browser": ["chromium", "firefox"]}

            def __init__(self, *, headless: bool = True):
                self.headless = headless

            def step(self, task: str) -> str:
                return "browser"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        model = result.per_key_models["browser_agent"]

        literal_args = literal_keys(model.model_fields["browser"].annotation)
        assert sorted(literal_args) == ["chromium", "firefox"]

    def test_constrain_existing_init_param(self):
        """Wiring constrains an existing __init__ param from str to Literal."""
        Loop = create_registrar("e2e_loop3", LoopProtocol, discriminator_field="name")

        class BaseLoop3(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop3(BaseLoop3):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent4", AgentProtocol, discriminator_field="name")

        class BaseAgent4(metaclass=Agent.Meta):
            __abstract__ = True
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class MyAgent(BaseAgent4):
            __wiring__ = {"loop": "e2e_loop3"}

            def __init__(self, *, loop: str, max_steps: int = 10):
                self.loop = loop
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "my_agent"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        model = result.per_key_models["my_agent"]

        # loop should be constrained but NOT injected (it was in __init__)
        from typing import get_args
        literal_args = get_args(model.model_fields["loop"].annotation)
        assert literal_args == ("react_loop3",)

    def test_optional_param_handling(self):
        """Wiring handles Optional[str] → Optional[Literal[...]]."""
        Loop = create_registrar("e2e_loop4", LoopProtocol, discriminator_field="name")

        class BaseLoop4(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop4(BaseLoop4):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent5", AgentProtocol, discriminator_field="name")

        class BaseAgent5(metaclass=Agent.Meta):
            __abstract__ = True
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class OptionalAgent(BaseAgent5):
            __wiring__ = {"loop": "e2e_loop4"}

            def __init__(self, *, loop: Optional[str] = None):
                self.loop = loop

            def step(self, task: str) -> str:
                return "optional"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        model = result.per_key_models["optional_agent"]

        # Field should exist with Optional[Literal[...]] type
        assert "loop" in model.model_fields
        field_type = model.model_fields["loop"].annotation
        # Should accept None or "react_loop4"
        from typing import get_origin, get_args, Union
        origin = get_origin(field_type)
        assert origin is Union  # Optional is Union[X, None]
        args = get_args(field_type)
        assert type(None) in args

    def test_codegen_wired_comment(self):
        """Generated source contains '# wired from:' comment."""
        Loop = create_registrar("e2e_loop5", LoopProtocol, discriminator_field="name")

        class BaseLoop5(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop5(BaseLoop5):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent6", AgentProtocol, discriminator_field="name")

        class BaseAgent6(metaclass=Agent.Meta):
            __abstract__ = True
            __wiring__ = {"loop": "e2e_loop5"}
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class CodegenAgent(BaseAgent6):
            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "codegen"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        source = generate_layer_config_source(result)

        assert "# wired from: e2e_loop5" in source
        assert "Literal" in source
        # Verify the wired field appears in the generated model
        assert "loop:" in source or "loop :" in source

    def test_json_schema_includes_literal(self):
        """JSON schema output includes enum constraint for wired fields."""
        Loop = create_registrar("e2e_loop6", LoopProtocol, discriminator_field="name")

        class BaseLoop6(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop6(BaseLoop6):
            def run(self) -> None: ...

        class CodeactLoop6(BaseLoop6):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent7", AgentProtocol, discriminator_field="name")

        class BaseAgent7(metaclass=Agent.Meta):
            __abstract__ = True
            __wiring__ = {"loop": "e2e_loop6"}
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class SchemaAgent(BaseAgent7):
            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "schema"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        schema = generate_layer_json_schema(result)

        # JSON schema should contain enum values for the wired field
        import json
        schema_str = json.dumps(schema)
        assert "react_loop6" in schema_str
        assert "codeact_loop6" in schema_str

    def test_validation_rejects_invalid_wired_value(self):
        """Pydantic validation rejects values not in the wired Literal."""
        Loop = create_registrar("e2e_loop7", LoopProtocol, discriminator_field="name")

        class BaseLoop7(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop7(BaseLoop7):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent8", AgentProtocol, discriminator_field="name")

        class BaseAgent8(metaclass=Agent.Meta):
            __abstract__ = True
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class ValidatedAgent(BaseAgent8):
            __wiring__ = {"loop": "e2e_loop7"}

            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "validated"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        adapter = TypeAdapter(result.union_type)

        # Valid: should pass
        config = adapter.validate_python({"name": "validated_agent", "loop": "react_loop7", "max_steps": 5})
        assert config.loop == "react_loop7"  # type: ignore[attr-defined]

        # Invalid loop value: should fail
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            adapter.validate_python({"name": "validated_agent", "loop": "nonexistent", "max_steps": 5})

    def test_inheritance_narrowing_e2e(self):
        """Child narrows parent's wiring, resulting in different Literal constraints."""
        Loop = create_registrar("e2e_loop8", LoopProtocol, discriminator_field="name")

        class BaseLoop8(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop8(BaseLoop8):
            def run(self) -> None: ...

        class CodeactLoop8(BaseLoop8):
            def run(self) -> None: ...

        class PlanActLoop8(BaseLoop8):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent9", AgentProtocol, discriminator_field="name")

        class BaseAgent9(metaclass=Agent.Meta):
            __abstract__ = True
            __wiring__ = {"loop": "e2e_loop8"}
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class WideAgent(BaseAgent9):
            """Gets all loop keys."""
            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "wide"

            def reset(self) -> None: ...

        class NarrowAgent(BaseAgent9):
            """Only supports react."""
            __wiring__ = {"loop": ("e2e_loop8", ["react_loop8"])}

            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "narrow"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)

        # Wide agent: all 3 loop keys
        wide_model = result.per_key_models["wide_agent"]
        wide_literal = literal_keys(wide_model.model_fields["loop"].annotation)
        assert sorted(wide_literal) == ["codeact_loop8", "plan_act_loop8", "react_loop8"]

        # Narrow agent: only react
        narrow_model = result.per_key_models["narrow_agent"]
        narrow_literal = literal_keys(narrow_model.model_fields["loop"].annotation)
        assert narrow_literal == ("react_loop8",)

    def test_full_pipeline_mode2_with_optional(self):
        """Mode 2 with 3-element tuple: required + optional keys, both in Literal."""
        Loop = create_registrar("e2e_loop9", LoopProtocol, discriminator_field="name")

        class BaseLoop9(metaclass=Loop.Meta):
            __abstract__ = True
            def run(self) -> None: ...

        class ReactLoop9(BaseLoop9):
            def run(self) -> None: ...

        class CodeactLoop9(BaseLoop9):
            def run(self) -> None: ...

        class PlanActLoop9(BaseLoop9):
            def run(self) -> None: ...

        Agent = create_registrar("e2e_agent10", AgentProtocol, discriminator_field="name")

        class BaseAgent10(metaclass=Agent.Meta):
            __abstract__ = True
            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class OptAgent(BaseAgent10):
            __wiring__ = {
                "loop": ("e2e_loop9", ["react_loop9"], ["codeact_loop9"]),
            }

            def __init__(self, *, max_steps: int = 10):
                self.max_steps = max_steps

            def step(self, task: str) -> str:
                return "opt"

            def reset(self) -> None: ...

        result = build_layer_config(Agent)
        model = result.per_key_models["opt_agent"]

        literal_args = literal_keys(model.model_fields["loop"].annotation)
        # Both required and optional keys appear in the Literal
        assert sorted(literal_args) == ["codeact_loop9", "react_loop9"]

        # __wired_fields__ metadata preserved
        assert model.__wired_fields__["loop"] == "e2e_loop9"


# ---------------------------------------------------------------------------
# Capability-relative (short) keys through the full pipeline
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionProtocol(Protocol):
    def act(self) -> None: ...


def _build_action_layer(name: str):
    """A hierarchical action registry with two capabilities and one collision."""
    Action = create_registrar(
        name, ActionProtocol, discriminator_field="action_id", key_separator="."
    )

    class ActionBase(metaclass=Action.Meta):
        __abstract__ = True

        def act(self) -> None: ...

    class BrowserClick(ActionBase):
        __registry_key__ = "browser.click"

        def __init__(self, index: int = 0) -> None:
            self.index = index

    class DesktopClick(ActionBase):
        __registry_key__ = "desktop.click"

        def __init__(self, x: int = 0) -> None:
            self.x = x

    class BrowserScroll(ActionBase):
        __registry_key__ = "browser.scroll"

        def __init__(self, amount: int = 100) -> None:
            self.amount = amount

    return Action, ActionBase


class TestCapabilityRelativeKeysE2E:
    """A class that names no capability must still generate valid config.

    Before this, ``__wiring__`` could only name fully-qualified keys, which
    type-welded every declaring class to exactly one capability.
    """

    def test_codegen_widens_literal_across_capabilities(self):
        """extractor.py call site."""
        _build_action_layer("e2e_relkey_action1")
        Agent = create_registrar(
            "e2e_relkey_agent1", AgentProtocol, discriminator_field="name"
        )

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PortableAgent(BaseAgent):
            # Note: no capability named anywhere.
            __wiring__ = {"action": ("e2e_relkey_action1", ["click"], ["scroll"])}

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PinnedAgent(BaseAgent):
            # Escape hatch: fully-qualified key pins one capability.
            __wiring__ = {"action": ("e2e_relkey_action1", ["browser.click"])}

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        result = build_layer_config(Agent)

        portable = result.per_key_models["portable_agent"]
        assert sorted(literal_keys(portable.model_fields["action"].annotation)) == [
            "browser.click",
            "browser.scroll",
            "desktop.click",
        ]

        pinned = result.per_key_models["pinned_agent"]
        assert literal_keys(pinned.model_fields["action"].annotation) == (
            "browser.click",
        )

        # The generated source is valid Python and carries both shapes.
        source = generate_layer_config_source(result)
        assert "'browser.click', 'desktop.click', 'browser.scroll'" in source
        compile(source, "<generated>", "exec")

    def test_json_schema_enumerates_every_candidate(self):
        Action, _ = _build_action_layer("e2e_relkey_action2")
        Agent = create_registrar(
            "e2e_relkey_agent2", AgentProtocol, discriminator_field="name"
        )

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PortableAgent(BaseAgent):
            __wiring__ = {"action": ("e2e_relkey_action2", ["click"])}

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        schema = generate_layer_json_schema(build_layer_config(Agent))
        text = str(schema)
        assert "browser.click" in text and "desktop.click" in text

    def test_validation_accepts_every_expansion(self):
        _build_action_layer("e2e_relkey_action3")
        Agent = create_registrar(
            "e2e_relkey_agent3", AgentProtocol, discriminator_field="name"
        )

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PortableAgent(BaseAgent):
            __wiring__ = {"action": ("e2e_relkey_action3", ["click"])}
            action: str

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        model = build_layer_config(Agent).per_key_models["portable_agent"]
        adapter = TypeAdapter(model)
        assert adapter.validate_python(
            {"name": "portable_agent", "action": "browser.click"}
        ).action == "browser.click"
        assert adapter.validate_python(
            {"name": "portable_agent", "action": "desktop.click"}
        ).action == "desktop.click"
        with pytest.raises(Exception):
            adapter.validate_python({"name": "portable_agent", "action": "click"})

    def test_fingerprint_invalidates_when_a_capability_appears(self):
        """fingerprint.py call site: a new match must widen the Literal.

        If the fingerprint did not move, a stale generated config would
        silently omit the new capability.
        """
        from conscribe import compute_registry_fingerprint

        _, ActionBase = _build_action_layer("e2e_relkey_action4")
        Agent = create_registrar(
            "e2e_relkey_agent4", AgentProtocol, discriminator_field="name"
        )

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PortableAgent(BaseAgent):
            __wiring__ = {"action": ("e2e_relkey_action4", ["click"])}

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        before = compute_registry_fingerprint(Agent)

        class MobileClick(ActionBase):
            __registry_key__ = "mobile.click"

            def __init__(self, finger: int = 1) -> None: ...

        assert compute_registry_fingerprint(Agent) != before

    def test_stub_type_narrows_across_capabilities(self):
        """collector.py call site: narrowing still resolves a common base."""
        from conscribe.stubs.collector import collect_class_stub_info

        _, ActionBase = _build_action_layer("e2e_relkey_action5")
        Agent = create_registrar(
            "e2e_relkey_agent5", AgentProtocol, discriminator_field="name"
        )

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PortableAgent(BaseAgent):
            __wiring__ = {"action": ("e2e_relkey_action5", ["click"])}

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        info = collect_class_stub_info(PortableAgent)
        assert info is not None
        attr = next(a for a in info.injected_attrs if a.name == "action")
        assert attr.resolved_type is ActionBase

    def test_composed_config_inlines_every_candidate(self):
        """Composed config widens the union, not just the Literal."""
        from conscribe import build_composed_config, generate_composed_config_source

        Action, _ = _build_action_layer("e2e_relkey_action6")
        Agent = create_registrar(
            "e2e_relkey_agent6", AgentProtocol, discriminator_field="name"
        )

        class BaseAgent(metaclass=Agent.Meta):
            __abstract__ = True

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class PortableAgent(BaseAgent):
            __wiring__ = {"action": ("e2e_relkey_action6", ["click"])}
            action: list

            def __init__(self) -> None: ...

            def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        composed = build_composed_config(
            {"e2e_relkey_agent6": Agent, "e2e_relkey_action6": Action},
            inline_wiring=True,
        )
        source = generate_composed_config_source(composed)
        assert "BrowserClickE2E_Relkey_Action6Config" in source
        assert "DesktopClickE2E_Relkey_Action6Config" in source
        compile(source, "<generated>", "exec")
