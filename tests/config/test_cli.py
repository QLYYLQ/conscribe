"""Tests for conscribe.cli — TDD RED phase.

Tests the CLI entry point ``main()`` with two subcommands:

- ``generate-config --layer <name> [--output <path>]``
    Generates a Python stub file from a registrar's registry.
- ``inspect --layer <name>``
    Prints registered keys, class names, and config field info.

All implementation classes and registrars are defined at MODULE LEVEL
so that ``get_type_hints()`` can resolve forward references under
``from __future__ import annotations``.

The CLI discovers registrars via an internal ``_get_registrar(layer_name)``
function, which is patched in tests to return a pre-populated test registrar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest.mock import patch

import pytest

from conscribe import create_registrar


# ===================================================================
# Protocols for test registrars
# ===================================================================

@runtime_checkable
class _CLILLMProtocol(Protocol):
    """LLM Provider interface for CLI tests."""

    async def chat(self, messages: list[dict]) -> str: ...


# ===================================================================
# Module-level registrar and implementations
# ===================================================================

_cli_llm_registrar = create_registrar(
    "llm",
    _CLILLMProtocol,
    strip_suffixes=["LLM", "Provider"],
    discriminator_field="provider",
)


class _CLILLMBase(metaclass=_cli_llm_registrar.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class CLIOpenAIProvider(_CLILLMBase):
    """OpenAI provider for CLI tests.

    Args:
        model_id: The model identifier, e.g. gpt-4o.
        temperature: Sampling temperature.
    """
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class CLIAnthropicProvider(_CLILLMBase):
    """Anthropic provider for CLI tests.

    Args:
        model_id: The model identifier.
        max_tokens: Maximum output tokens.
    """
    __registry_key__ = "anthropic"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Patch target: conscribe.cli._get_registrar
# We use the module path to the function we will implement.
# ===================================================================

_PATCH_TARGET = "conscribe.cli._get_registrar"


# ===================================================================
# Test: generate-config writes file
# ===================================================================

class TestGenerateConfig:
    """Tests for the ``generate-config`` subcommand."""

    def test_generate_config_writes_file(self, tmp_path: Path) -> None:
        """``generate-config --layer llm --output <path>`` writes a file
        containing valid Python source (class definitions, imports)."""
        from conscribe.cli import main

        output_file = tmp_path / "llm_config.py"

        with patch(_PATCH_TARGET, return_value=_cli_llm_registrar):
            exit_code = main([
                "generate-config",
                "--layer", "llm",
                "--output", str(output_file),
            ])

        assert exit_code == 0
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        # Should contain Python source with class definitions
        assert "class" in content
        assert "BaseModel" in content
        # Should contain the provider implementations
        assert "openai" in content.lower() or "Openai" in content

    def test_generate_config_prints_to_stdout_if_no_output(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``generate-config --layer llm`` (no ``--output``) prints
        the generated Python source to stdout."""
        from conscribe.cli import main

        with patch(_PATCH_TARGET, return_value=_cli_llm_registrar):
            exit_code = main(["generate-config", "--layer", "llm"])

        assert exit_code == 0

        captured = capsys.readouterr()
        # Source should be printed to stdout
        assert "class" in captured.out
        assert "BaseModel" in captured.out

    def test_unknown_layer_returns_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``generate-config --layer nonexistent`` returns non-zero exit code
        when the layer cannot be resolved to a registrar."""
        from conscribe.cli import main

        with patch(_PATCH_TARGET, side_effect=KeyError("nonexistent")):
            exit_code = main([
                "generate-config",
                "--layer", "nonexistent",
            ])

        assert exit_code != 0


# ===================================================================
# Test: inspect subcommand
# ===================================================================

class TestInspect:
    """Tests for the ``inspect`` subcommand."""

    def test_inspect_prints_layer_info(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``inspect --layer llm`` prints registered keys and model names
        to stdout."""
        from conscribe.cli import main

        with patch(_PATCH_TARGET, return_value=_cli_llm_registrar):
            exit_code = main(["inspect", "--layer", "llm"])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out

        # Should list the registered keys
        assert "openai" in output
        assert "anthropic" in output

        # Should show field/param info (at minimum the key names)
        assert "model_id" in output


# ===================================================================
# Test: no args / help
# ===================================================================

class TestCLIHelp:
    """Tests for help text and no-argument invocation."""

    def test_no_args_prints_help(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``main([])`` with no arguments returns non-zero exit code
        and prints usage/help information to stderr."""
        from conscribe.cli import main

        exit_code = main([])

        # Should indicate an error (no subcommand provided)
        assert exit_code != 0

        captured = capsys.readouterr()
        # Help or usage text should appear in stderr (argparse default)
        combined = captured.err + captured.out
        assert "usage" in combined.lower() or "help" in combined.lower()

    def test_help_flag(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``main(["--help"])`` prints help text to stdout.
        argparse raises SystemExit(0) for --help, which main() should handle."""
        from conscribe.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        # --help exits with code 0
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "generate-config" in captured.out or "generate" in captured.out
        assert "inspect" in captured.out
