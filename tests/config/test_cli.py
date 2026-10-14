"""Tests for conscribe.cli.

Two styles:

- The first classes patch ``conscribe.cli._get_registrar`` (the lookup of
  ``--layer NAME`` among loaded registrars) to hand the CLI a registrar
  defined in this module.
- The end-to-end classes write a throwaway package under ``tmp_path`` (the
  working directory, which the CLI makes importable) and drive every command
  through ``main([...])`` with ``--registrar MODULE:ATTRIBUTE``,
  ``--discover``, ``--allow-missing`` and ``--exclude``, including the error
  exits (``Error: ...`` on stderr, status 1, no traceback).  Registrations
  made by those packages are rolled back after each test.
"""
from __future__ import annotations

import errno
import io
import json
import os
import stat
import subprocess
import sys
import textwrap
import uuid
import warnings
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from unittest.mock import patch

import pytest

from conscribe import create_registrar
from conscribe.exceptions import ConscribeWarning, ExtractionWarning, StaleStubWarning
from conscribe.testing import isolated_registries


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

    def test_no_command_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``main([])`` is a usage error: argparse prints the usage and the
        missing command to stderr and exits with status 2."""
        from conscribe.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("usage: conscribe")
        assert "the following arguments are required: <command>" in captured.err

    def test_generate_config_is_described_as_a_config_source(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "Python config source file" in out
        assert "config stub" not in out

    def test_generate_stubs_help_lists_prune(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        with pytest.raises(SystemExit):
            main(["generate-stubs", "--help"])
        out = capsys.readouterr().out
        assert "--prune" in out and "# Generated by conscribe" in out

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


# ===================================================================
# End-to-end: a throwaway project under tmp_path
# ===================================================================

_LAYERS = '''\
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class LLMProtocol(Protocol):
    def chat(self, prompt: str) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    def step(self, task: str) -> str: ...


LLMRegistrar = create_registrar(
    "cli_llm", LLMProtocol, discriminator_field="provider", strip_suffixes=["LLM"]
)
AgentRegistrar = create_registrar(
    "cli_agent", AgentProtocol, discriminator_field="name", strip_suffixes=["Agent"]
)


class BaseLLM(metaclass=LLMRegistrar.Meta):
    __abstract__ = True

    def chat(self, prompt: str) -> str:
        return ""


class BaseAgent(metaclass=AgentRegistrar.Meta):
    __abstract__ = True

    def step(self, task: str) -> str:
        return ""
'''

_OPENAI = '''\
from {pkg}.layers import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI chat models.

    Args:
        model: Model identifier.
        temperature: Sampling temperature.
    """

    __registry_key__ = "openai"

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
'''

_LOCAL = '''\
from {pkg}.layers import BaseLLM


class LocalLLM(BaseLLM):
    __registry_key__ = "local"

    def __init__(self, path: str):
        self.path = path
'''

_REACT = '''\
from {pkg}.layers import BaseAgent


class ReactAgent(BaseAgent):
    """ReAct loop.

    Args:
        max_steps: Step budget.
    """

    __registry_key__ = "react"
    __wiring__ = {"llm": "cli_llm"}

    def __init__(self, llm: str, max_steps: int = 10):
        self.llm = llm
        self.max_steps = max_steps
'''

# Slot-less wiring: ``helper`` is injected by the framework (stub material).
_TOOL = '''\
from {pkg}.layers import BaseAgent


class ToolAgent(BaseAgent):
    __registry_key__ = "tool"
    __wiring__ = {"helper": "cli_llm"}

    def __init__(self, retries: int = 1):
        self.retries = retries
'''


# An inline-only layer whose instances live under ArchivingAgent.archive.
_ARCHIVE = '''\
from typing import Protocol, runtime_checkable

from conscribe import create_registrar
from {pkg}.layers import BaseAgent


@runtime_checkable
class ArchiveProtocol(Protocol):
    def store(self, data: str) -> None: ...


ArchiveRegistrar = create_registrar(
    "cli_archive", ArchiveProtocol, discriminator_field="kind", inline_only=True
)


class BaseArchive(metaclass=ArchiveRegistrar.Meta):
    __abstract__ = True

    def store(self, data: str) -> None: ...


class DiskArchive(BaseArchive):
    __registry_key__ = "disk"

    def __init__(self, path: str = "/tmp/archive"):
        self.path = path


class ArchivingAgent(BaseAgent):
    __registry_key__ = "archiving"
    __wiring__ = {"archive": "cli_archive"}

    def __init__(self, archive: str):
        self.archive = archive
'''


class _Project:
    """A throwaway package ``<name>`` in the working directory of a test."""

    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name

    def write(self, relative: str, source: str = "") -> Path:
        """Write ``<name>/<relative>``; ``{pkg}`` in *source* is the package name."""
        path = self.root / self.name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).replace("{pkg}", self.name), encoding="utf-8")
        return path

    def ref(self, attribute: str, module: str = "layers") -> str:
        """``<name>.<module>:<attribute>`` for ``--registrar``."""
        return f"{self.name}.{module}:{attribute}"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A two-layer project (cli_llm: openai, local; cli_agent: react, tool).

    The working directory is ``tmp_path``.  The package name is unique per
    test; its modules are dropped from ``sys.modules`` and everything they
    registered is rolled back afterwards.
    """
    monkeypatch.chdir(tmp_path)
    name = f"cliproj_{uuid.uuid4().hex[:10]}"
    proj = _Project(tmp_path, name)
    proj.write("__init__.py")
    proj.write("layers.py", _LAYERS)
    proj.write("llms/__init__.py")
    proj.write("llms/openai.py", _OPENAI)
    proj.write("llms/local.py", _LOCAL)
    proj.write("agents/__init__.py")
    proj.write("agents/react.py", _REACT)
    proj.write("agents/tool.py", _TOOL)

    before = set(sys.modules)
    with isolated_registries():
        try:
            yield proj
        finally:
            for module in [m for m in sys.modules if m not in before]:
                if module == name or module.startswith(name + "."):
                    del sys.modules[module]


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    from conscribe.cli import main

    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _assert_error(code: int, err: str, *fragments: str) -> None:
    """Status 1, ``Error: ...`` on stderr, no traceback, every fragment present."""
    assert code == 1, err
    assert err.startswith("Error: "), err
    assert "Traceback" not in err
    for fragment in fragments:
        assert fragment in err, f"{fragment!r} not in {err!r}"


class TestGenerateConfigEndToEnd:
    """generate-config with --registrar / --discover / --layer."""

    def test_registrar_and_discover_print_the_source(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), "--discover", project.name,
        )
        assert code == 0, err
        assert err == ""
        assert "class " in out and "BaseModel" in out
        assert "'openai'" in out and "'local'" in out
        assert "Model identifier." in out  # docstring description
        compile(out, "<generated>", "exec")

    def test_output_and_json_schema_files(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), "--discover", project.name,
            "--output", "generated/llm_config.py",
            "--json-schema", "generated/schema/llm.json",
        )
        assert code == 0, err
        assert out == ""
        source = (project.root / "generated" / "llm_config.py").read_text(encoding="utf-8")
        compile(source, "llm_config.py", "exec")
        assert "'openai'" in source
        schema = json.loads(
            (project.root / "generated" / "schema" / "llm.json").read_text(encoding="utf-8")
        )
        assert "openai" in json.dumps(schema) and "local" in json.dumps(schema)

    def test_layer_picks_one_of_several_registrars(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), project.ref("AgentRegistrar"),
            "--discover", project.name, "--layer", "cli_agent",
        )
        assert code == 0, err
        assert "max_steps" in out
        assert "'react'" in out
        assert "Model identifier." not in out  # not the llm layer

    def test_layer_found_among_registrars_the_imports_loaded(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-config", "--discover", project.name, "--layer", "cli_llm",
        )
        assert code == 0, err
        assert "'openai'" in out

    def test_dotted_attribute(self, project, capsys) -> None:
        project.write("holder.py", """\
            from {pkg}.layers import LLMRegistrar


            class Layers:
                LLM = LLMRegistrar
        """)
        code, out, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("Layers.LLM", module="holder"), "--discover", project.name,
        )
        assert code == 0, err
        assert "'openai'" in out

    def test_empty_layer_is_an_error(self, project, capsys) -> None:
        """Without --discover nothing registers: EmptyLayerError, cleanly."""
        code, _, err = _run(capsys, "generate-config", "--registrar", project.ref("LLMRegistrar"))
        _assert_error(code, err, "'cli_llm'", "no registered implementations")

    def test_registrar_without_discriminator_is_an_error(self, project, capsys) -> None:
        project.write("plain.py", """\
            from typing import Protocol, runtime_checkable

            from conscribe import create_registrar


            @runtime_checkable
            class P(Protocol):
                def run(self) -> None: ...


            Plain = create_registrar("cli_plain", P)


            class Base(metaclass=Plain.Meta):
                __abstract__ = True

                def run(self) -> None: ...


            class Thing(Base):
                pass
        """)
        code, _, err = _run(
            capsys, "generate-config", "--registrar", project.ref("Plain", module="plain"),
        )
        _assert_error(code, err, "'cli_plain'", "discriminator_field")

    def test_unwritable_output_is_an_error(self, project, capsys) -> None:
        (project.root / "taken").mkdir()
        code, _, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), "--discover", project.name,
            "--output", "taken",
        )
        _assert_error(code, err, "taken")


def _leftovers(root: Path) -> list[Path]:
    """Temporary files the staged writes left behind (there must be none)."""
    return [path for path in root.rglob(".*.tmp")]


class TestOutputFiles:
    """Generated files are staged, then moved into place: no half output."""

    def _generate(self, project, capsys, *options: str) -> tuple[int, str, str]:
        return _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), "--discover", project.name, *options,
        )

    def test_failed_schema_write_writes_no_source(self, project, capsys) -> None:
        (project.root / "out" / "schema.json").mkdir(parents=True)
        code, _, err = self._generate(
            project, capsys, "--output", "out/config.py", "--json-schema", "out/schema.json",
        )
        _assert_error(code, err, "out/schema.json", "is a directory")
        assert not (project.root / "out" / "config.py").exists()
        assert _leftovers(project.root) == []

    def test_failed_schema_write_keeps_the_previous_source(self, project, capsys) -> None:
        previous = project.root / "config.py"
        previous.write_text("# previous run\n", encoding="utf-8")
        (project.root / "schema.json").mkdir()
        code, _, err = self._generate(
            project, capsys, "--output", "config.py", "--json-schema", "schema.json",
        )
        _assert_error(code, err, "is a directory")
        assert previous.read_text(encoding="utf-8") == "# previous run\n"
        assert _leftovers(project.root) == []

    def test_failed_schema_write_prints_no_source(self, project, capsys) -> None:
        (project.root / "schema.json").mkdir()
        code, out, err = self._generate(project, capsys, "--json-schema", "schema.json")
        _assert_error(code, err, "is a directory")
        assert out == ""

    def test_output_and_schema_on_the_same_file(self, project, capsys) -> None:
        code, _, err = self._generate(
            project, capsys, "--output", "same.json", "--json-schema", "./same.json",
        )
        _assert_error(code, err, "--output and --json-schema name the same file")
        assert not (project.root / "same.json").exists()

    def test_schema_with_source_on_stdout(self, project, capsys) -> None:
        code, out, err = self._generate(project, capsys, "--json-schema", "s/llm.json")
        assert code == 0, err
        assert "'openai'" in out
        assert "openai" in (project.root / "s" / "llm.json").read_text(encoding="utf-8")

    def test_files_get_the_usual_permissions(self, project, capsys) -> None:
        umask = os.umask(0)
        os.umask(umask)
        kept = project.root / "kept.py"
        kept.write_text("", encoding="utf-8")
        kept.chmod(0o640)
        code, _, err = self._generate(
            project, capsys, "--output", "kept.py", "--json-schema", "new.json",
        )
        assert code == 0, err
        assert stat.S_IMODE((project.root / "new.json").stat().st_mode) == 0o666 & ~umask
        assert stat.S_IMODE(kept.stat().st_mode) == 0o640  # an existing file keeps its mode
        assert "'openai'" in kept.read_text(encoding="utf-8")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_output_through_a_symlink_keeps_the_link(self, project, capsys) -> None:
        real = project.root / "real_config.py"
        real.write_text("", encoding="utf-8")
        link = project.root / "config.py"
        link.symlink_to(real)
        code, _, err = self._generate(project, capsys, "--output", "config.py")
        assert code == 0, err
        assert link.is_symlink()
        assert "'openai'" in real.read_text(encoding="utf-8")

    def test_composed_output_file(self, project, capsys) -> None:
        (project.root / "composed.json").write_text("stale", encoding="utf-8")
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_llm", "--output", "composed.json",
        )
        assert code == 0, err
        assert out == ""
        schema = json.loads((project.root / "composed.json").read_text(encoding="utf-8"))
        assert "cli_llm" in schema["properties"]
        assert _leftovers(project.root) == []


class TestRegistrarOption:
    """--registrar MODULE:ATTRIBUTE parsing, import and validation errors."""

    @pytest.mark.parametrize("spec", ["{pkg}.layers", ":LLMRegistrar", "{pkg}.layers:", "a:b:c"])
    def test_malformed_spec(self, project, capsys, spec: str) -> None:
        code, _, err = _run(
            capsys, "generate-config", "--registrar", spec.replace("{pkg}", project.name),
        )
        _assert_error(code, err, "--registrar expects MODULE:ATTRIBUTE")

    def test_missing_module(self, project, capsys) -> None:
        code, _, err = _run(
            capsys, "generate-config", "--registrar", f"{project.name}.nope:Registrar",
        )
        _assert_error(
            code, err, f"could not import module '{project.name}.nope'", "ModuleNotFoundError"
        )

    def test_module_raising_on_import(self, project, capsys) -> None:
        project.write("bad_layers.py", """\
            raise RuntimeError("boom while importing")
        """)
        code, _, err = _run(
            capsys, "generate-config", "--registrar", project.ref("R", module="bad_layers"),
        )
        _assert_error(
            code, err,
            f"could not import module '{project.name}.bad_layers'",
            "RuntimeError: boom while importing",
            "bad_layers.py, line 1",
        )

    def test_missing_attribute_lists_the_registrars_of_the_module(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-config", "--registrar", project.ref("Nope"))
        _assert_error(
            code, err,
            "has no attribute 'Nope'",
            "registrars defined there: AgentRegistrar, LLMRegistrar",
        )

    def test_missing_nested_attribute(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-config", "--registrar", project.ref("BaseLLM.nope"))
        _assert_error(code, err, f"'{project.name}.layers:BaseLLM' has no attribute 'nope'")

    def test_class_that_is_not_a_registrar(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-config", "--registrar", project.ref("BaseLLM"))
        _assert_error(
            code, err,
            "expected a registrar returned by create_registrar()",
            f"got class {project.name}.layers.BaseLLM",
        )

    def test_value_that_is_not_a_registrar(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-config", "--registrar", project.ref("__name__"))
        _assert_error(code, err, "expected a registrar", f"got str '{project.name}.layers'")


class TestLayerSelection:
    """Which layer a one-layer command works on."""

    def test_unknown_layer_suggests_a_loaded_one(self, project, capsys) -> None:
        code, _, err = _run(
            capsys, "generate-config", "--discover", project.name, "--layer", "cli_agnet",
        )
        _assert_error(
            code, err,
            "no registrar for layer 'cli_agnet' is loaded",
            "did you mean 'cli_agent'?",
            "--registrar MODULE:ATTRIBUTE",
        )

    def test_nothing_selected(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-config")
        _assert_error(code, err, "no layer selected")

    def test_several_registrars_need_layer(self, project, capsys) -> None:
        code, _, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"),
            "--registrar", project.ref("AgentRegistrar"),
        )
        _assert_error(code, err, "2 registrars given (cli_llm, cli_agent)", "--layer NAME")

    def test_same_registrar_twice_counts_once(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), project.ref("LLMRegistrar"),
            "--discover", project.name,
        )
        assert code == 0, err
        assert "'openai'" in out

    def test_cwd_is_importable_only_while_the_command_runs(self, project, capsys) -> None:
        cwd = str(project.root)
        assert cwd not in sys.path
        code, _, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("LLMRegistrar"), "--discover", project.name,
        )
        assert code == 0, err
        assert cwd not in sys.path


class TestDiscoverOptions:
    """--discover is strict; --exclude / --allow-missing reach discover()."""

    def test_broken_module_names_module_and_location(self, project, capsys) -> None:
        project.write("agents/broken.py", """\
            from {pkg}.layers import BaseAgent

            VALUE = undefined_name
        """)
        code, _, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
        )
        _assert_error(
            code, err,
            f"could not import module '{project.name}.agents.broken'",
            "NameError",
            "undefined_name",
            "broken.py, line 3",
            f"--exclude {project.name}.agents.broken",
        )

    def test_exclude_skips_the_broken_module(self, project, capsys) -> None:
        project.write("agents/broken.py", "VALUE = undefined_name\n")
        code, out, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
            "--exclude", f"{project.name}.agents.broken",
        )
        assert code == 0, err
        assert "'react'" in out

    def test_missing_optional_dependency(self, project, capsys) -> None:
        project.write("agents/fancy.py", """\
            import cli_missing_dependency_xyz  # noqa: F401
        """)
        args = [
            "generate-config",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
        ]
        code, _, err = _run(capsys, *args)
        _assert_error(
            code, err,
            f"could not import module '{project.name}.agents.fancy'",
            "ModuleNotFoundError",
            "--allow-missing cli_missing_dependency_xyz",
        )

        code, out, err = _run(capsys, *args, "--allow-missing", "cli_missing_dependency_xyz")
        assert code == 0, err
        assert "'react'" in out

    def test_missing_module_of_the_project_itself(self, project, capsys) -> None:
        """A typo in an import of the project is not an optional dependency."""
        project.write("agents/typo.py", "import {pkg}.agnets.react  # noqa: F401\n")
        code, _, err = _run(
            capsys, "generate-config",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
        )
        _assert_error(code, err, "ModuleNotFoundError", f"--exclude {project.name}.agents.typo")
        assert "--allow-missing" not in err

    def test_missing_package(self, project, capsys) -> None:
        code, _, err = _run(
            capsys, "generate-config", "--discover", f"{project.name}_nope", "--layer", "cli_llm",
        )
        _assert_error(
            code, err, f"could not import module '{project.name}_nope'", "ModuleNotFoundError"
        )
        assert "--exclude" not in err  # the hint is for submodules


class TestGenerateComposedConfigEndToEnd:
    """generate-composed-config with --registrar / --layers."""

    @staticmethod
    def _react_model(schema: dict) -> dict:
        for model in schema.get("$defs", {}).values():
            if "max_steps" in model.get("properties", {}):
                return model
        raise AssertionError("no agent model in the composed schema")

    def test_registrars_define_the_layers(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config",
            "--registrar", project.ref("AgentRegistrar"), project.ref("LLMRegistrar"),
            "--discover", project.name,
        )
        assert code == 0, err
        schema = json.loads(out)
        assert set(schema["properties"]) >= {"cli_agent", "cli_llm"}
        assert "$ref" in json.dumps(self._react_model(schema)["properties"]["llm"])

    def test_layers_with_discover(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name, "--layers", "cli_llm",
        )
        assert code == 0, err
        schema = json.loads(out)
        assert "cli_llm" in schema["properties"]
        assert "cli_agent" not in schema["properties"]

    def test_repeated_layers_option_extends(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "--layers", "cli_llm",
        )
        assert code == 0, err
        assert {"cli_agent", "cli_llm"} <= set(json.loads(out)["properties"])

    def test_no_inline_wiring_keeps_key_literals(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm", "--no-inline-wiring",
        )
        assert code == 0, err
        llm = self._react_model(json.loads(out))["properties"]["llm"]
        assert "$ref" not in json.dumps(llm)
        assert "openai" in json.dumps(llm) and "local" in json.dumps(llm)

    def test_python_format_to_file(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm", "--format", "python", "--output", "out/composed.py",
        )
        assert code == 0, err
        assert out == ""
        source = (project.root / "out" / "composed.py").read_text(encoding="utf-8")
        compile(source, "composed.py", "exec")
        assert "max_steps" in source

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            ([], False),  # forbid is the default
            (["--top-level-extra", "forbid"], False),
            (["--top-level-extra", "ignore"], None),
            (["--top-level-extra", "allow"], True),
        ],
    )
    def test_top_level_extra(self, project, capsys, argv: list[str], expected) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm", *argv,
        )
        assert code == 0, err
        assert json.loads(out).get("additionalProperties") == expected

    def test_top_level_extra_reaches_the_python_source(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm", "--format", "python", "--top-level-extra", "allow",
        )
        assert code == 0, err
        assert 'extra="allow"' in out or "extra='allow'" in out

    def test_top_level_extra_rejects_unknown_modes(self, project, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(
                capsys, "generate-composed-config", "--discover", project.name,
                "--layers", "cli_llm", "--top-level-extra", "sometimes",
            )
        assert exc_info.value.code == 2

    def test_nothing_selected(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-composed-config")
        _assert_error(code, err, "no layers selected")

    def test_unknown_layer(self, project, capsys) -> None:
        code, _, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_llm", "cli_nope",
        )
        _assert_error(code, err, "no registrar for layer 'cli_nope' is loaded")

    def test_two_registrars_of_one_layer(self, project, capsys) -> None:
        project.write("dup.py", """\
            from typing import Protocol, runtime_checkable

            from conscribe import create_registrar


            @runtime_checkable
            class P(Protocol):
                def run(self) -> None: ...


            A = create_registrar("cli_dup", P, discriminator_field="kind")
            B = create_registrar("cli_dup", P, discriminator_field="kind")
        """)
        code, _, err = _run(
            capsys, "generate-composed-config",
            "--registrar", project.ref("A", module="dup"), project.ref("B", module="dup"),
        )
        _assert_error(code, err, "two different registrars of layer 'cli_dup'")

    def test_circular_wiring(self, project, capsys) -> None:
        project.write("llms/looping.py", """\
            from {pkg}.layers import BaseLLM


            class LoopingLLM(BaseLLM):
                __registry_key__ = "looping"
                __wiring__ = {"agent": "cli_agent"}

                def __init__(self, agent: str = "react"):
                    self.agent = agent
        """)
        code, _, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm",
        )
        _assert_error(code, err, "Circular wiring dependency")

    def test_malformed_wiring_fails_the_import(self, project, capsys) -> None:
        """Grammar errors fail at class creation: reported as an import failure."""
        project.write("agents/bad_wiring.py", """\
            from {pkg}.layers import BaseAgent


            class BadAgent(BaseAgent):
                __registry_key__ = "bad"
                __wiring__ = {"llm": [1]}
        """)
        code, _, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm",
        )
        _assert_error(
            code, err,
            f"could not import module '{project.name}.agents.bad_wiring'",
            "WiringResolutionError",
            "BadAgent",
            "non-str",
            "bad_wiring.py, line 4",
        )

    def test_wiring_problems_found_at_build_time(self, project, capsys) -> None:
        """Resolution problems (unknown keys) are validated before building."""
        project.write("agents/typo.py", """\
            from {pkg}.layers import BaseAgent


            class TypoAgent(BaseAgent):
                __registry_key__ = "typo"
                __wiring__ = {"llm": ("cli_llm", ["opnai"])}

                def __init__(self, llm: str):
                    self.llm = llm
        """)
        commands = [
            ["generate-config", "--layer", "cli_agent"],
            ["generate-composed-config", "--layers", "cli_agent", "cli_llm"],
        ]
        for command in commands:
            code, _, err = _run(capsys, *command, "--discover", project.name)
            _assert_error(code, err, "TypoAgent", "unknown key 'opnai'", "did you mean 'openai'?")

    def test_inline_only_layer_has_no_top_level_section(self, project, capsys) -> None:
        project.write("archive.py", _ARCHIVE)
        code, out, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm", "cli_archive",
        )
        assert code == 0, err
        properties = json.loads(out)["properties"]
        assert {"cli_agent", "cli_llm"} <= set(properties)
        assert "cli_archive" not in properties

    def test_inline_only_layer_without_inline_wiring(self, project, capsys) -> None:
        project.write("archive.py", _ARCHIVE)
        code, _, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_agent", "cli_llm", "cli_archive", "--no-inline-wiring",
        )
        _assert_error(code, err, "'cli_archive'", "inline-only", "inline_wiring=True")

    def test_inline_only_layer_nobody_holds(self, project, capsys) -> None:
        project.write("archive.py", _ARCHIVE)
        code, _, err = _run(
            capsys, "generate-composed-config", "--discover", project.name,
            "--layers", "cli_llm", "cli_archive",
        )
        _assert_error(code, err, "cli_archive", "inline-only")


#: ``llms/broken.py`` sources that make generation fail, with message fragments.
_GENERATION_FAILURES: dict[str, tuple[str, list[str]]] = {
    "annotation": (
        '''\
        from {pkg}.layers import BaseLLM


        class TypoLLM(BaseLLM):
            __registry_key__ = "typo"

            def __init__(self, x: "Undefinedd" = 1):
                self.x = x
        ''',
        ["TypoLLM.x", "cannot evaluate annotation", "Undefinedd"],
    ),
    "declaration": (
        '''\
        from typing import Annotated

        from pydantic import Field

        from {pkg}.layers import BaseLLM


        class ExcludingLLM(BaseLLM):
            __registry_key__ = "excluding"

            def __init__(self, x: Annotated[int, Field(exclude=True)] = 1):
                self.x = x
        ''',
        ["ExcludingLLM.x", "exclude"],
    ),
    "render": (
        '''\
        from {pkg}.layers import BaseLLM


        class LambdaLLM(BaseLLM):
            __registry_key__ = "lambda"

            def __init__(self, f=lambda: 1):
                self.f = f
        ''',
        ["LambdaLLM", "cannot render"],
    ),
}


class TestConscribeErrorsAreReported:
    """Every conscribe error ends as ``Error: ...`` on stderr with status 1."""

    @pytest.mark.parametrize("case", list(_GENERATION_FAILURES))
    def test_generation_errors(self, project, capsys, case: str) -> None:
        source, fragments = _GENERATION_FAILURES[case]
        project.write("llms/broken.py", source)
        code, out, err = _run(
            capsys, "generate-config", "--discover", project.name, "--layer", "cli_llm",
        )
        _assert_error(code, err, *fragments)
        assert out == ""

    def test_invalid_registry_key(self, project, capsys) -> None:
        project.write("llms/badkey.py", """\
            from {pkg}.layers import BaseLLM


            class BadKeyLLM(BaseLLM):
                __registry_key__ = ""
        """)
        code, _, err = _run(
            capsys, "generate-config", "--discover", project.name, "--layer", "cli_llm",
        )
        _assert_error(
            code, err,
            f"could not import module '{project.name}.llms.badkey'",
            "InvalidRegistryKeyError",
            "invalid registry key ''",
        )

    def test_conscribe_warning_escalated_by_the_filters(self, project, capsys) -> None:
        project.write("llms/opaque.py", """\
            from {pkg}.layers import BaseLLM


            class Opaque:
                pass


            class OpaqueLLM(BaseLLM):
                __registry_key__ = "opaque"

                def __init__(self, handle: Opaque | None = None):
                    self.handle = handle
        """)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConscribeWarning)
            code, out, err = _run(
                capsys, "generate-config", "--discover", project.name, "--layer", "cli_llm",
            )
        _assert_error(code, err, "OpaqueLLM", "Any")
        assert out == ""

    def test_registry_error_raised_anywhere(
        self, project, capsys, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import conscribe.scanner as scanner
        from conscribe.exceptions import RegistryError

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise RegistryError("the registry index is unreadable")

        monkeypatch.setattr(scanner, "summarize_registries", explode)
        code, _, err = _run(capsys, "list", "--discover", project.name)
        _assert_error(code, err, "the registry index is unreadable")


class TestGenerateStubsEndToEnd:
    """generate-stubs with --registrar / --discover."""

    def test_writes_stubs_for_injected_attributes(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-stubs",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
            "--output-dir", "stubs",
        )
        assert code == 0, err
        stub = project.root / "stubs" / project.name / "agents" / "tool.pyi"
        assert stub.exists()
        text = stub.read_text(encoding="utf-8")
        assert "class ToolAgent" in text and "helper" in text
        assert "tool.pyi" in out
        assert "Generated 1 stub file(s)." in out

    def test_no_wired_classes(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "generate-stubs",
            "--registrar", project.ref("LLMRegistrar"), "--discover", project.name,
            "--output-dir", "stubs",
        )
        assert code == 0
        assert "No classes with injected wired attributes found." in err
        assert not (project.root / "stubs").exists()

    def test_unknown_layer(self, project, capsys) -> None:
        code, _, err = _run(capsys, "generate-stubs", "--layer", "cli_nope")
        _assert_error(code, err, "no registrar for layer 'cli_nope' is loaded")

    def test_empty_layer_is_an_error(self, project, capsys) -> None:
        """Without --discover nothing registers: an error, not "no classes found"."""
        code, _, err = _run(capsys, "generate-stubs", "--registrar", project.ref("LLMRegistrar"))
        _assert_error(code, err, "'cli_llm'", "no registered implementations")

    @staticmethod
    def _fake_writer(monkeypatch: pytest.MonkeyPatch, emit: Any = None) -> list[dict[str, Any]]:
        """Replace ``write_layer_stubs``; record its keyword arguments."""
        import conscribe.stubs.writer as writer

        calls: list[dict[str, Any]] = []

        def fake(registrar: Any, output_dir: Any = None, **options: Any) -> list[Path]:
            calls.append(options)
            if emit is not None:
                emit()
            return []

        monkeypatch.setattr(writer, "write_layer_stubs", fake)
        return calls

    def test_prune_is_forwarded(self, project, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._fake_writer(monkeypatch)
        args = ["generate-stubs", "--registrar", project.ref("AgentRegistrar"), "--discover", project.name]
        assert _run(capsys, *args, "--prune")[0] == 0
        assert _run(capsys, *args, "--partial-class-fallback")[0] == 0
        assert calls == [
            {"partial_class_fallback": False, "prune": True},
            {"partial_class_fallback": True},
        ]

    def test_stale_stub_warnings_are_printed_as_warning_lines(
        self, project, capsys, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even when conscribe's warnings are errors (``-W error::...ConscribeWarning``)."""

        def emit() -> None:
            warnings.warn("stale stub: stubs/pkg/old.pyi", StaleStubWarning, stacklevel=2)

        self._fake_writer(monkeypatch, emit)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConscribeWarning)
            code, _, err = _run(
                capsys, "generate-stubs",
                "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
            )
        assert code == 0, err
        assert "Warning: stale stub: stubs/pkg/old.pyi" in err.splitlines()
        assert "Traceback" not in err

    def test_other_warnings_are_left_to_the_warning_filters(
        self, project, capsys, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def emit() -> None:
            warnings.warn("cannot read Foo", ExtractionWarning, stacklevel=2)

        self._fake_writer(monkeypatch, emit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            code, _, err = _run(
                capsys, "generate-stubs",
                "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
            )
        assert code == 0, err
        assert [str(w.message) for w in caught if w.category is ExtractionWarning] == [
            "cannot read Foo"
        ]
        assert "Warning: cannot read Foo" not in err

    def test_stale_stubs_are_reported_then_pruned(self, project, capsys) -> None:
        stub_dir = project.root / "stubs" / project.name / "agents"
        stub_dir.mkdir(parents=True)
        stale = stub_dir / "retired.pyi"
        stale.write_text(
            "# Generated by conscribe 1.9.0\n"
            "# Auto-generated by conscribe. DO NOT EDIT.\n"
            "class Retired: ...\n",
            encoding="utf-8",
        )
        manual = stub_dir / "manual.pyi"
        manual.write_text("class Manual: ...\n", encoding="utf-8")
        args = [
            "generate-stubs", "--registrar", project.ref("AgentRegistrar"),
            "--discover", project.name, "--output-dir", "stubs",
        ]

        code, _, err = _run(capsys, *args)
        assert code == 0, err
        assert any(line.startswith("Warning: ") and "retired.pyi" in line for line in err.splitlines())
        assert stale.exists()

        code, _, err = _run(capsys, *args, "--prune")
        assert code == 0, err
        assert not stale.exists()
        assert manual.exists()  # not generated by conscribe: never touched
        assert (stub_dir / "tool.pyi").exists()


class TestClosedStdout:
    """Output piped into a reader that stops early (``| head``)."""

    _MAIN = "import sys; from conscribe.cli import main; sys.exit(main(sys.argv[1:]))"

    def test_inspect_into_a_closed_pipe_exits_quietly(self, project) -> None:
        process = subprocess.Popen(
            [
                sys.executable, "-c", self._MAIN, "inspect",
                "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
            ],
            cwd=project.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        process.stdout.close()  # the reader is gone before anything is written
        err = process.stderr.read().decode()
        process.stderr.close()
        assert process.wait(timeout=120) == 1
        assert err == ""  # no traceback, no "Exception ignored" at exit

    class _ClosedPipe(io.StringIO):
        def write(self, s: str) -> int:
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")

        def flush(self) -> None:
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    @pytest.mark.parametrize("command", ["inspect", "generate-config"])
    def test_in_process(
        self, project, capsys, monkeypatch: pytest.MonkeyPatch, command: str,
    ) -> None:
        from conscribe.cli import main

        monkeypatch.setattr(sys, "stdout", self._ClosedPipe())
        code = main([
            command, "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
        ])
        assert code == 1
        assert capsys.readouterr().err == ""


class TestInspectEndToEnd:
    """inspect with --registrar / --discover."""

    def test_prints_keys_fields_descriptions_and_wiring(self, project, capsys) -> None:
        code, out, err = _run(
            capsys, "inspect",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
        )
        assert code == 0, err
        assert "Layer: cli_agent" in out
        assert "Discriminator: name" in out
        assert "Registered keys (2):" in out
        assert "react (ReactAgent):" in out
        assert "max_steps: int = 10  # Step budget." in out
        assert "wired from: cli_llm" in out
        assert "tool (ToolAgent):" in out

    def test_a_failing_key_is_reported_and_the_others_listed(
        self, project, capsys, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import conscribe.config.extractor as extractor
        from conscribe.exceptions import ConfigAnnotationError

        real = extractor.extract_config_schema

        def failing(cls, *args, **kwargs):
            if cls.__name__ == "ReactAgent":
                raise ConfigAnnotationError("ReactAgent", "llm", "Lmm", "name 'Lmm' is not defined")
            return real(cls, *args, **kwargs)

        monkeypatch.setattr(extractor, "extract_config_schema", failing)
        code, out, err = _run(
            capsys, "inspect",
            "--registrar", project.ref("AgentRegistrar"), "--discover", project.name,
        )
        assert code == 1
        assert "tool (ToolAgent):" in out and "retries: int = 1" in out
        assert "Traceback" not in err
        assert err.startswith("Error: react (ReactAgent): ReactAgent.llm: cannot evaluate")

    def test_nested_and_inline_only_layers(self, project, capsys) -> None:
        project.write("special.py", """\
            from typing import Protocol, runtime_checkable

            from conscribe import create_registrar


            @runtime_checkable
            class P(Protocol):
                def run(self) -> None: ...


            Nested = create_registrar(
                "cli_nested", P, discriminator_fields=["family", "model"], key_separator="."
            )
            Archive = create_registrar("cli_archive", P, discriminator_field="kind", inline_only=True)


            class NestedBase(metaclass=Nested.Meta):
                __abstract__ = True

                def run(self) -> None: ...


            class Small(NestedBase):
                __registry_key__ = "gpt.small"


            class ArchiveBase(metaclass=Archive.Meta):
                __abstract__ = True

                def run(self) -> None: ...


            class Disk(ArchiveBase):
                __registry_key__ = "disk"
        """)
        code, out, err = _run(
            capsys, "inspect", "--registrar", project.ref("Nested", module="special"),
        )
        assert code == 0, err
        assert "Discriminator: family, model (nested, key separator '.')" in out
        assert "gpt.small (Small):" in out

        code, out, err = _run(
            capsys, "inspect", "--registrar", project.ref("Archive", module="special"),
        )
        assert code == 0, err
        assert "Inline-only: yes" in out
