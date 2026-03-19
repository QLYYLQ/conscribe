"""Tests for diamond inheritance and cross-registry support."""
from __future__ import annotations

import pytest
from typing import Protocol, runtime_checkable

from conscribe import create_registrar
from conscribe.registration.meta_base import AutoRegistrarBase


@runtime_checkable
class LLMProtocol(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


class TestAutoRegistrarBase:
    """Test the base metaclass and | operator."""

    def test_or_returns_notimplemented_for_non_auto_registrar(self):
        """When | is used with a non-AutoRegistrarBase type, MetaRegistrarType.__or__
        returns NotImplemented. Python then falls back to type.__or__ which
        creates a union type (Python 3.10+)."""
        from conscribe.registration.meta_base import MetaRegistrarType
        # Direct method call to verify the behavior
        result = MetaRegistrarType.__or__(AutoRegistrarBase, int)
        assert result is NotImplemented

    def test_or_returns_self_when_subclass(self):
        class SubMeta(AutoRegistrarBase):
            pass
        result = SubMeta | AutoRegistrarBase
        assert result is SubMeta

    def test_or_returns_other_when_other_is_subclass(self):
        class SubMeta(AutoRegistrarBase):
            pass
        result = AutoRegistrarBase | SubMeta
        assert result is SubMeta

    def test_or_creates_combined_when_incompatible(self):
        class MetaA(AutoRegistrarBase):
            pass
        class MetaB(AutoRegistrarBase):
            pass
        combined = MetaA | MetaB
        assert issubclass(combined, MetaA)
        assert issubclass(combined, MetaB)
        assert "Combined[" in combined.__name__


class TestCrossRegistryDiamond:
    """Test cross-registry diamond inheritance via | operator."""

    def test_basic_cross_registry(self):
        """class C(metaclass=LLM.Meta | Agent.Meta) registers in both."""
        LLM = create_registrar("llm_d1", LLMProtocol, discriminator_field="provider")
        Agent = create_registrar("agent_d1", AgentProtocol, discriminator_field="name")

        CombinedMeta = LLM.Meta | Agent.Meta

        class LLMAgentBase(metaclass=CombinedMeta):
            __abstract__ = True

            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            async def step(self, task: str) -> str:
                return "ok"
            def reset(self) -> None:
                pass

        class MyLLMAgent(LLMAgentBase):
            pass

        assert LLM.get("my_llm_agent") is MyLLMAgent
        assert Agent.get("my_llm_agent") is MyLLMAgent

        # Cleanup
        for key in list(LLM.keys()):
            LLM.unregister(key)
        for key in list(Agent.keys()):
            Agent.unregister(key)

    def test_explicit_keys_cross_registry(self):
        """Explicit __registry_key__ is used in both registries."""
        LLM = create_registrar("llm_d2", LLMProtocol, discriminator_field="provider")
        Agent = create_registrar("agent_d2", AgentProtocol, discriminator_field="name")

        CombinedMeta = LLM.Meta | Agent.Meta

        class DualBase(metaclass=CombinedMeta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            async def step(self, task: str) -> str:
                return "ok"
            def reset(self) -> None:
                pass

        class DualImpl(DualBase):
            __registry_key__ = "custom_key"

        assert LLM.get("custom_key") is DualImpl
        assert Agent.get("custom_key") is DualImpl

        for key in list(LLM.keys()):
            LLM.unregister(key)
        for key in list(Agent.keys()):
            Agent.unregister(key)

    def test_skip_registries_with_combined_meta(self):
        """__skip_registries__ allows selective opt-out from one registry."""
        LLM = create_registrar("llm_d3", LLMProtocol, discriminator_field="provider")
        Agent = create_registrar("agent_d3", AgentProtocol, discriminator_field="name")

        CombinedMeta = LLM.Meta | Agent.Meta

        class DualBase(metaclass=CombinedMeta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"
            async def step(self, task: str) -> str:
                return "ok"
            def reset(self) -> None:
                pass

        class LLMOnly(DualBase):
            __skip_registries__ = ["agent_d3"]

        assert LLM.get_or_none("llm_only") is LLMOnly
        assert Agent.get_or_none("llm_only") is None

        for key in list(LLM.keys()):
            LLM.unregister(key)
        for key in list(Agent.keys()):
            Agent.unregister(key)


class TestSameRegistryHierarchical:
    """Test hierarchical keys within a single registry."""

    def test_auto_derived_hierarchical_key(self):
        """When key_separator is set, child key derived from parent."""
        LLM = create_registrar(
            "llm_h1", LLMProtocol,
            discriminator_field="provider",
            key_separator=".",
        )

        class OpenAIBase(metaclass=LLM.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class AzureOpenAI(OpenAIBase):
            """Should auto-derive key as 'openai.azure_open_ai'."""
            pass

        assert LLM.get_or_none("openai") is None  # abstract
        key = AzureOpenAI.__registry_key__
        assert key.startswith("openai.")
        assert LLM.get(key) is AzureOpenAI

        for key in list(LLM.keys()):
            LLM.unregister(key)

    def test_explicit_hierarchical_key(self):
        """Explicit __registry_key__ takes precedence over derivation."""
        LLM = create_registrar(
            "llm_h2", LLMProtocol,
            discriminator_field="provider",
            key_separator=".",
        )

        class OpenAIBase(metaclass=LLM.Meta):
            __registry_key__ = "openai"
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class AzureOpenAI(OpenAIBase):
            __registry_key__ = "openai.azure"

        assert LLM.get("openai.azure") is AzureOpenAI

        for key in list(LLM.keys()):
            LLM.unregister(key)

    def test_children_query(self):
        """children(prefix) returns matching descendants."""
        LLM = create_registrar(
            "llm_h3", LLMProtocol,
            discriminator_field="provider",
            key_separator=".",
        )

        class Base(metaclass=LLM.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class OpenAIAzure(Base):
            __registry_key__ = "openai.azure"

        class OpenAIOfficial(Base):
            __registry_key__ = "openai.official"

        class AnthropicClaude(Base):
            __registry_key__ = "anthropic.claude"

        children = LLM.children("openai")
        assert set(children.keys()) == {"openai.azure", "openai.official"}
        assert children["openai.azure"] is OpenAIAzure

        for key in list(LLM.keys()):
            LLM.unregister(key)

    def test_tree_query(self):
        """tree() returns nested dict."""
        LLM = create_registrar(
            "llm_h4", LLMProtocol,
            discriminator_field="provider",
            key_separator=".",
        )

        class Base(metaclass=LLM.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class OpenAIAzure(Base):
            __registry_key__ = "openai.azure"

        class OpenAIOfficial(Base):
            __registry_key__ = "openai.official"

        t = LLM.tree()
        assert "openai" in t
        assert "azure" in t["openai"]
        assert "official" in t["openai"]
        assert t["openai"]["azure"] is OpenAIAzure
        assert t["openai"]["official"] is OpenAIOfficial

        for key in list(LLM.keys()):
            LLM.unregister(key)

    def test_children_empty_without_separator(self):
        """children() returns empty dict when no separator is set."""
        LLM = create_registrar(
            "llm_h5", LLMProtocol, discriminator_field="provider"
        )

        class Base(metaclass=LLM.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class Impl(Base):
            __registry_key__ = "impl"

        assert LLM.children("impl") == {}

        for key in list(LLM.keys()):
            LLM.unregister(key)


class TestMultiKey:
    """Test __registry_key__ as list for multi-key registration."""

    def test_multi_key_registration(self):
        LLM = create_registrar(
            "llm_mk", LLMProtocol,
            discriminator_field="provider",
            key_separator=".",
        )

        class Base(metaclass=LLM.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return "ok"

        class Universal(Base):
            __registry_key__ = ["openai.universal", "anthropic.universal"]

        assert LLM.get("openai.universal") is Universal
        assert LLM.get("anthropic.universal") is Universal
        assert Universal.__registry_key__ == "openai.universal"  # primary
        assert Universal.__registry_keys__ == ["openai.universal", "anthropic.universal"]

        for key in list(LLM.keys()):
            LLM.unregister(key)

    def test_multi_key_flat_mode(self):
        """Multi-key also works in flat mode."""
        @runtime_checkable
        class P(Protocol):
            def work(self) -> str: ...

        R = create_registrar("mk_flat", P, discriminator_field="type")

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def work(self) -> str:
                return "ok"

        class Multi(Base):
            __registry_key__ = ["alias_a", "alias_b"]

        assert R.get("alias_a") is Multi
        assert R.get("alias_b") is Multi

        for key in list(R.keys()):
            R.unregister(key)


class TestCreateRegistrarValidation:
    """Test validation of new create_registrar parameters."""

    def test_discriminator_fields_requires_separator(self):
        with pytest.raises(ValueError, match="key_separator"):
            create_registrar(
                "test_v1", LLMProtocol,
                discriminator_fields=["model_type", "provider"],
            )

    def test_mutual_exclusion_discriminator_params(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            create_registrar(
                "test_v2", LLMProtocol,
                discriminator_field="type",
                discriminator_fields=["model_type", "provider"],
                key_separator=".",
            )

    def test_discriminator_fields_with_separator_ok(self):
        R = create_registrar(
            "test_v3", LLMProtocol,
            discriminator_fields=["model_type", "provider"],
            key_separator=".",
        )
        assert R.discriminator_fields == ["model_type", "provider"]
        assert R._key_separator == "."
