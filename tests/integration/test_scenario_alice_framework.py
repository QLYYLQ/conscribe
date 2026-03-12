"""Integration tests: Alice builds a benchmark framework.

Simulates the full story of Alice (framework developer) creating
a benchmark framework with multiple layers (Agent, LLM, Browser Provider).

Scenarios:
- End-to-end: define protocol -> create_registrar -> base class -> implementations -> query
- Multi-layer: three independent layers coexisting
- Custom key: evaluator with __registry_key__ override
"""
from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar


# ===================================================================
# Protocols (Alice defines these for her framework)
# ===================================================================

@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class BrowserProtocol(Protocol):
    def connect(self) -> str: ...
    def close(self) -> None: ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    def evaluate(self, result: str) -> float: ...


# ===================================================================
# Test: Alice end-to-end
# ===================================================================

class TestAliceEndToEnd:
    """Alice builds the benchmark framework from scratch."""

    def test_alice_end_to_end(self) -> None:
        """Alice's complete workflow:
        1. Define AgentProtocol
        2. create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])
        3. Create BaseAgent(metaclass=Meta, __abstract__=True)
        4. Implement BrowserUseAgent, SkyvernAgent, AgentTarsAgent
        5. Verify: 3 agents registered, keys correct
        6. Use get() to query -> instantiate -> call step()
        """
        # Step 1-2: create registrar
        AgentRegistrar = create_registrar(
            "agent",
            AgentProtocol,
            strip_suffixes=["Agent"],
            discriminator_field="name",
        )

        # Step 3: base class
        class BaseAgent(metaclass=AgentRegistrar.Meta):
            __abstract__ = True

            async def step(self, task: str) -> str:
                raise NotImplementedError

            def reset(self) -> None:
                raise NotImplementedError

        # Step 4: implementations
        class BrowserUseAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return f"browser_use: {task}"

            def reset(self) -> None:
                pass

        class SkyvernAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return f"skyvern: {task}"

            def reset(self) -> None:
                pass

        class AgentTarsAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return f"agent_tars: {task}"

            def reset(self) -> None:
                pass

        # Step 5: verify registrations
        assert set(AgentRegistrar.keys()) == {"browser_use", "skyvern", "agent_tars"}
        assert AgentRegistrar.get("browser_use") is BrowserUseAgent
        assert AgentRegistrar.get("skyvern") is SkyvernAgent
        assert AgentRegistrar.get("agent_tars") is AgentTarsAgent

        # Step 6: simulate CLI usage
        for agent_name in ["browser_use", "skyvern", "agent_tars"]:
            agent_cls = AgentRegistrar.get(agent_name)
            agent = agent_cls()
            result = asyncio.get_event_loop().run_until_complete(
                agent.step("test task")
            )
            assert agent_name.replace("_", " ") in result.replace("_", " ") or agent_name in result

    def test_alice_get_all(self) -> None:
        """Alice can use get_all() to list all registered implementations."""
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

        all_agents = R.get_all()
        assert isinstance(all_agents, dict)
        assert "alpha" in all_agents
        assert "beta" in all_agents


class TestAliceMultiLayer:
    """Alice builds three independent layers simultaneously."""

    def test_alice_multi_layer(self) -> None:
        """Three layers: Agent + LLM + Browser Provider, all independent."""
        # 1. Create registrars
        AR = create_registrar("agent", AgentProtocol, strip_suffixes=["Agent"])
        LR = create_registrar("llm", LLMProtocol, strip_suffixes=["LLM", "Provider"])
        BR = create_registrar("browser", BrowserProtocol, strip_suffixes=["Provider"])

        # 2. Base classes
        class BaseAgent(metaclass=AR.Meta):
            __abstract__ = True
            async def step(self, task: str) -> str: ...
            def reset(self) -> None: ...

        class BaseLLM(metaclass=LR.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str: ...

        class BaseBrowser(metaclass=BR.Meta):
            __abstract__ = True
            def connect(self) -> str: ...
            def close(self) -> None: ...

        # 3. Implementations
        class BrowserUseAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "bu"
            def reset(self) -> None:
                pass

        class SkyvernAgent(BaseAgent):
            async def step(self, task: str) -> str:
                return "sk"
            def reset(self) -> None:
                pass

        class OpenAIProvider(BaseLLM):
            async def chat(self, messages: list[dict]) -> str:
                return "openai"

        class AnthropicLLM(BaseLLM):
            async def chat(self, messages: list[dict]) -> str:
                return "anthropic"

        class LexmountProvider(BaseBrowser):
            def connect(self) -> str:
                return "lex"
            def close(self) -> None:
                pass

        # 4. Verify independence
        assert set(AR.keys()) == {"browser_use", "skyvern"}
        assert set(LR.keys()) == {"open_ai", "anthropic"}
        assert set(BR.keys()) == {"lexmount"}

        # Cross-layer isolation
        assert AR.get_or_none("open_ai") is None
        assert LR.get_or_none("browser_use") is None
        assert BR.get_or_none("skyvern") is None


class TestAliceEvaluatorWithCustomKey:
    """Evaluator layer where keys need to match dataset names."""

    def test_evaluator_custom_key(self) -> None:
        """Evaluator keys align with dataset names using __registry_key__."""
        ER = create_registrar(
            "evaluator",
            EvaluatorProtocol,
            strip_suffixes=["Evaluator"],
            discriminator_field="evaluator",
        )

        class BaseEvaluator(metaclass=ER.Meta):
            __abstract__ = True
            def evaluate(self, result: str) -> float:
                raise NotImplementedError

        class LexbenchBrowserEvaluator(BaseEvaluator):
            __registry_key__ = "lexbench_browser"
            def evaluate(self, result: str) -> float:
                return 0.95

        class BrowsecompEvaluator(BaseEvaluator):
            __registry_key__ = "browsecomp"
            def evaluate(self, result: str) -> float:
                return 0.88

        assert ER.get("lexbench_browser") is LexbenchBrowserEvaluator
        assert ER.get("browsecomp") is BrowsecompEvaluator

    def test_evaluator_mixed_key_strategies(self) -> None:
        """Some evaluators use custom key, others use inferred key."""
        ER = create_registrar("evaluator", EvaluatorProtocol, strip_suffixes=["Evaluator"])

        class BaseEvaluator(metaclass=ER.Meta):
            __abstract__ = True
            def evaluate(self, result: str) -> float:
                raise NotImplementedError

        class SimpleEvaluator(BaseEvaluator):
            # Inferred key: "simple"
            def evaluate(self, result: str) -> float:
                return 1.0

        class ComplexBenchmarkEvaluator(BaseEvaluator):
            __registry_key__ = "complex_bench"
            def evaluate(self, result: str) -> float:
                return 0.5

        assert ER.get("simple") is SimpleEvaluator
        assert ER.get("complex_bench") is ComplexBenchmarkEvaluator
