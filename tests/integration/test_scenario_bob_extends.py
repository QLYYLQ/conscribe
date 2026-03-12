"""Integration tests: Bob extends the benchmark framework.

Simulates Bob (framework user) extending Alice's framework with
external agents via three paths:
- Ideal path: external agent inherits BaseAgent -> zero extra steps
- Bridge path: external agent doesn't inherit BaseAgent -> bridge once
- Single integration: one-off @register for a single external class
- Running benchmarks: Bob creates variants and runs benchmarks
"""
from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.exceptions import DuplicateKeyError


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


def _make_agent_registrar_with_builtins():
    """Helper: simulate Alice's published framework with built-in agents."""
    R = create_registrar(
        "agent",
        AgentProtocol,
        strip_suffixes=["Agent"],
        discriminator_field="name",
    )

    class BaseAgent(metaclass=R.Meta):
        __abstract__ = True
        async def step(self, task: str) -> str:
            raise NotImplementedError
        def reset(self) -> None:
            raise NotImplementedError

    class BrowserUseAgent(BaseAgent):
        async def step(self, task: str) -> str:
            return "browser_use"
        def reset(self) -> None:
            pass

    return R, BaseAgent


class TestBobIdealPath:
    """Bob scenario A: external agent inherits BaseAgent -> zero extra steps."""

    def test_bob_story_ideal_path(self) -> None:
        """External agent chose to depend on Alice's framework and inherit BaseAgent."""
        R, BaseAgent = _make_agent_registrar_with_builtins()

        # Simulate: from cool_agent import CoolAgent (inherits BaseAgent)
        class CoolAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "cool"
            def reset(self) -> None:
                pass

        # Bob creates variants -- they auto-register because CoolAgent has the metaclass
        class MyCoolV1(CoolAgent):
            async def step(self, task: str) -> str:
                return "cool_v1"
            def reset(self) -> None:
                pass

        class MyCoolV2(CoolAgent):
            async def step(self, task: str) -> str:
                return "cool_v2"
            def reset(self) -> None:
                pass

        assert R.get("my_cool_v1") is MyCoolV1
        assert R.get("my_cool_v2") is MyCoolV2
        # CoolAgent itself is also registered
        assert R.get("cool") is CoolAgent


class TestBobBridgePath:
    """Bob scenario B: external agent doesn't inherit BaseAgent -> bridge once."""

    def test_bob_story_bridge_path(self) -> None:
        """External agent is independent framework -- bridge required."""
        R, BaseAgent = _make_agent_registrar_with_builtins()

        # External agent (independent, no metaclass)
        class ExtAgent:
            async def step(self, task: str) -> str:
                return "ext"
            def reset(self) -> None:
                pass

        # Bob bridges once
        ExtBridge = R.bridge(ExtAgent)

        # Bob creates variants
        class MyExtV1(ExtBridge):
            async def step(self, task: str) -> str:
                return "ext_v1"
            def reset(self) -> None:
                pass

        class MyExtV2(ExtBridge):
            async def step(self, task: str) -> str:
                return "ext_v2"
            def reset(self) -> None:
                pass

        assert R.get("my_ext_v1") is MyExtV1
        assert R.get("my_ext_v2") is MyExtV2
        # Bridge is abstract -> not registered
        assert R.get_or_none("ext_agent_bridge") is None

    def test_bob_bridge_preserves_external_behavior(self) -> None:
        """Bridged subclass inherits external class methods."""
        R, BaseAgent = _make_agent_registrar_with_builtins()

        class ExtAgent:
            async def step(self, task: str) -> str:
                return "ext_step"
            def reset(self) -> None:
                pass
            def custom_method(self) -> str:
                return "custom"

        Bridge = R.bridge(ExtAgent)

        class MyVariant(Bridge):
            pass  # Inherits all methods from ExtAgent

        instance = MyVariant()
        assert instance.custom_method() == "custom"


class TestBobSingleIntegration:
    """Bob scenario C: single external agent, one-off @register."""

    def test_bob_story_single_integration(self) -> None:
        """Bob only needs one external agent -> @register is simplest."""
        R, BaseAgent = _make_agent_registrar_with_builtins()

        class ExtAgent:
            async def step(self, task: str) -> str:
                return "ext"
            def reset(self) -> None:
                pass

        @R.register("ext_agent")
        class WrappedExt(ExtAgent):
            pass

        assert R.get("ext_agent") is WrappedExt

    def test_bob_register_without_explicit_key(self) -> None:
        """@register() without key uses key_transform to infer."""
        R, BaseAgent = _make_agent_registrar_with_builtins()

        class ExtRunner:
            async def step(self, task: str) -> str:
                return "ext"
            def reset(self) -> None:
                pass

        # strip_suffixes=["Agent"] won't match "ExtRunner"
        # so key will be "ext_runner" (default snake_case)
        @R.register()
        class ExtRunner2(ExtRunner):
            async def step(self, task: str) -> str:
                return "ext2"

        assert R.get("ext_runner2") is ExtRunner2


class TestBobRunBenchmark:
    """Bob runs benchmarks with multiple agent variants."""

    def test_bob_story_run_benchmark(self) -> None:
        """Bob creates 3 variants (via different paths) and runs benchmarks."""
        R = create_registrar(
            "agent",
            AgentProtocol,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str:
                raise NotImplementedError
            def reset(self) -> None:
                raise NotImplementedError

        # Variant 1: Path A (inheritance)
        class BuiltinAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "builtin_result"
            def reset(self) -> None:
                pass

        # Variant 2: Path B (bridge)
        class ExternalLib:
            async def step(self, task: str) -> str:
                return "external_result"
            def reset(self) -> None:
                pass

        Bridge = R.bridge(ExternalLib)

        class BridgedVariant(Bridge):
            async def step(self, task: str) -> str:
                return "bridged_result"
            def reset(self) -> None:
                pass

        # Variant 3: Path C (register)
        class AnotherExternal:
            async def step(self, task: str) -> str:
                return "another_result"
            def reset(self) -> None:
                pass

        @R.register("manual_agent")
        class ManualVariant(AnotherExternal):
            pass

        # Bob simulates CLI: alice-bench run --agent <name>
        results = {}
        for agent_name in ["builtin", "bridged_variant", "manual_agent"]:
            agent_cls = R.get(agent_name)
            agent = agent_cls()
            result = asyncio.get_event_loop().run_until_complete(
                agent.step("benchmark task")
            )
            results[agent_name] = result

        assert len(results) == 3
        assert all(isinstance(v, str) for v in results.values())

    def test_bob_duplicate_key_prevented(self) -> None:
        """Bob accidentally creates two agents with same key -> error."""
        R = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class FooAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "foo1"
            def reset(self) -> None:
                pass

        with pytest.raises(DuplicateKeyError):
            # Another class that infers the same key "foo"
            class FooAgent2(BaseAgent):  # noqa: F811
                __registry_key__ = "foo"
                async def step(self, task: str) -> str:
                    return "foo2"
                def reset(self) -> None:
                    pass

    def test_bob_can_inspect_available_agents(self) -> None:
        """Bob can list all available agents before running benchmarks."""
        R = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])

        class BaseAgent(metaclass=R.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class AlphaAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "alpha"
            def reset(self) -> None:
                pass

        class BetaAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "beta"
            def reset(self) -> None:
                pass

        available = R.keys()
        assert "alpha" in available
        assert "beta" in available
