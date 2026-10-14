"""Tests for conscribe.scanner — AST scanning, package detection, and runtime listing."""
from __future__ import annotations

import sys
import textwrap
import uuid
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from conscribe.exceptions import ExtractionWarning
from conscribe.scanner import (
    RegistrarDefinition,
    find_packages,
    scan_registrar_definitions,
)
from conscribe.testing import isolated_registries


# ===================================================================
# Helpers
# ===================================================================

def _write_py(path: Path, source: str) -> Path:
    """Write a Python source file with dedented content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


# ===================================================================
# scan_registrar_definitions
# ===================================================================

class TestScanRegistrarDefinitions:
    """Tests for the AST-based registrar scanner."""

    def test_basic_create_registrar(self, tmp_path: Path) -> None:
        """Detects a basic create_registrar() call."""
        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent", AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        d = results[0]
        assert d.name == "agent"
        assert d.protocol_name == "AgentProto"
        assert d.variable_name == "R"
        assert d.function_name == "create_registrar"
        assert d.line_number == 2

    def test_create_auto_registrar(self, tmp_path: Path) -> None:
        """Detects create_auto_registrar() calls."""
        _write_py(tmp_path / "auto.py", """\
            from conscribe import create_auto_registrar
            Meta = create_auto_registrar("event", EventProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].name == "event"
        assert results[0].function_name == "create_auto_registrar"

    def test_attribute_access_form(self, tmp_path: Path) -> None:
        """Detects conscribe.create_registrar(...) via attribute access."""
        _write_py(tmp_path / "layers.py", """\
            import conscribe
            R = conscribe.create_registrar("agent", AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].name == "agent"

    def test_dynamic_name_arg(self, tmp_path: Path) -> None:
        """Variable as first arg produces '<dynamic>'."""
        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            name = "agent"
            R = create_registrar(name, AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].name == "<dynamic>"

    def test_no_assignment(self, tmp_path: Path) -> None:
        """Bare create_registrar() call has variable_name=None."""
        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            create_registrar("agent", AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].variable_name is None

    def test_inside_function(self, tmp_path: Path) -> None:
        """Finds create_registrar() inside a function body."""
        _write_py(tmp_path / "setup.py", """\
            from conscribe import create_registrar
            def setup():
                R = create_registrar("agent", AgentProto)
                return R
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].name == "agent"
        assert results[0].variable_name == "R"

    def test_excludes_venv(self, tmp_path: Path) -> None:
        """Files in .venv/ are skipped."""
        _write_py(tmp_path / ".venv" / "lib" / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent", AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)
        assert len(results) == 0

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        """Files in __pycache__/ are skipped."""
        _write_py(tmp_path / "__pycache__" / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent", AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)
        assert len(results) == 0

    def test_syntax_error_skipped(self, tmp_path: Path) -> None:
        """Files with syntax errors are skipped (with a warning naming them)."""
        (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
        _write_py(tmp_path / "good.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent", AgentProto)
        """)

        with pytest.warns(ExtractionWarning, match=r"scan skipped .*bad\.py because it does not parse"):
            results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].name == "agent"

    def test_null_bytes_skipped_with_warning(self, tmp_path: Path) -> None:
        (tmp_path / "binary.py").write_bytes(b"x = 1\x00\n")

        with pytest.warns(ExtractionWarning, match=r"binary\.py"):
            assert scan_registrar_definitions(tmp_path) == []

    def test_coding_declaration_is_honoured(self, tmp_path: Path) -> None:
        """PEP 263: a latin-1 file is decoded like Python decodes it."""
        (tmp_path / "latin.py").write_bytes(
            b"# -*- coding: latin-1 -*-\n"
            b"from conscribe import create_registrar\n"
            b"NAME = '\xe9t\xe9'\n"
            b"R = create_registrar('summer', SummerProto)\n"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = scan_registrar_definitions(tmp_path)

        assert [r.name for r in results] == ["summer"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        results = scan_registrar_definitions(tmp_path)
        assert results == []

    def test_multiple_files(self, tmp_path: Path) -> None:
        """Finds definitions across multiple files."""
        _write_py(tmp_path / "a.py", """\
            from conscribe import create_registrar
            A = create_registrar("agent", AgentProto)
        """)
        _write_py(tmp_path / "b.py", """\
            from conscribe import create_registrar
            B = create_registrar("llm", LLMProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 2
        names = {d.name for d in results}
        assert names == {"agent", "llm"}

    def test_protocol_as_attribute(self, tmp_path: Path) -> None:
        """module.Protocol extracts the attribute name."""
        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            import protos
            R = create_registrar("agent", protos.AgentProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 1
        assert results[0].protocol_name == "AgentProto"

    def test_fewer_than_two_args_skipped(self, tmp_path: Path) -> None:
        """Calls with fewer than 2 positional args are ignored."""
        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent")
        """)

        results = scan_registrar_definitions(tmp_path)
        assert len(results) == 0

    def test_nested_statement_bodies(self, tmp_path: Path) -> None:
        """Calls inside every kind of compound statement are found."""
        _write_py(tmp_path / "nested.py", """\
            from conscribe import create_registrar

            if FLAG:
                A = create_registrar("if_body", P)
            else:
                B = create_registrar("if_else", P)
            for _ in range(1):
                C = create_registrar("for_body", P)
            else:
                D = create_registrar("for_else", P)
            while False:
                E = create_registrar("while_body", P)
            else:
                F = create_registrar("while_else", P)
            with CONTEXT:
                G = create_registrar("with_body", P)
            try:
                H = create_registrar("try_body", P)
            except ValueError:
                I = create_registrar("try_handler", P)
            else:
                J = create_registrar("try_else", P)
            finally:
                K = create_registrar("try_finally", P)
            match MODE:
                case "a":
                    L = create_registrar("match_case", P)
                case _:
                    M = create_registrar("match_default", P)


            class Holder:
                N = create_registrar("class_body", P)


            async def setup():
                async with CONTEXT:
                    O = create_registrar("async_with_body", P)
                async for _ in STREAM:
                    Q = create_registrar("async_for_body", P)
        """)

        names = [d.name for d in scan_registrar_definitions(tmp_path)]

        assert names == [
            "if_body", "if_else", "for_body", "for_else", "while_body", "while_else",
            "with_body", "try_body", "try_handler", "try_else", "try_finally",
            "match_case", "match_default", "class_body", "async_with_body", "async_for_body",
        ]

    @pytest.mark.skipif(sys.version_info < (3, 11), reason="except* needs Python 3.11")
    def test_try_star_bodies(self, tmp_path: Path) -> None:
        _write_py(tmp_path / "star.py", """\
            from conscribe import create_registrar

            try:
                A = create_registrar("star_body", P)
            except* ValueError:
                B = create_registrar("star_handler", P)
        """)

        assert [d.name for d in scan_registrar_definitions(tmp_path)] == [
            "star_body", "star_handler",
        ]

    def test_sorted_by_file_then_line(self, tmp_path: Path) -> None:
        """Results are sorted by file path, then line number."""
        _write_py(tmp_path / "z.py", """\
            from conscribe import create_registrar
            Z = create_registrar("zzz", ZProto)
        """)
        _write_py(tmp_path / "a.py", """\
            from conscribe import create_registrar
            A = create_registrar("aaa", AProto)
        """)

        results = scan_registrar_definitions(tmp_path)

        assert len(results) == 2
        # a.py comes before z.py
        assert results[0].name == "aaa"
        assert results[1].name == "zzz"


# ===================================================================
# find_packages
# ===================================================================

class TestFindPackages:
    """Tests for top-level package detection."""

    def test_detects_package_with_init(self, tmp_path: Path) -> None:
        pkg = tmp_path / "my_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")

        result = find_packages(tmp_path)
        assert result == ["my_pkg"]

    def test_ignores_dir_without_init(self, tmp_path: Path) -> None:
        pkg = tmp_path / "not_a_pkg"
        pkg.mkdir()

        result = find_packages(tmp_path)
        assert result == []

    def test_excludes_venv(self, tmp_path: Path) -> None:
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "__init__.py").write_text("", encoding="utf-8")

        result = find_packages(tmp_path)
        assert result == []

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "__init__.py").write_text("", encoding="utf-8")

        result = find_packages(tmp_path)
        assert result == []

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = find_packages(tmp_path)
        assert result == []

    def test_only_top_level(self, tmp_path: Path) -> None:
        """Nested packages are not returned (only top-level)."""
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        (parent / "__init__.py").write_text("", encoding="utf-8")
        (child / "__init__.py").write_text("", encoding="utf-8")

        result = find_packages(tmp_path)
        assert result == ["parent"]

    def test_sorted_output(self, tmp_path: Path) -> None:
        for name in ["zzz", "aaa", "mmm"]:
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")

        result = find_packages(tmp_path)
        assert result == ["aaa", "mmm", "zzz"]


# ===================================================================
# list_registries (integration-style)
# ===================================================================

class TestListRegistries:
    """Tests for runtime registry listing.

    These create temporary packages, import them, and check the output.
    Cleanup removes the packages from sys.path and sys.modules.
    """

    def test_discovers_and_lists(self, tmp_path: Path) -> None:
        """End-to-end: create package, discover, list entries."""
        from conscribe import create_registrar
        from conscribe.registration.registry import _deregister

        # Create a temp package
        pkg = tmp_path / "test_list_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "registry.py").write_text(textwrap.dedent("""\
            from typing import Protocol, runtime_checkable
            from conscribe import create_registrar

            @runtime_checkable
            class WorkerProto(Protocol):
                def work(self) -> str: ...

            WorkerReg = create_registrar("_test_list_worker", WorkerProto)

            class Base(metaclass=WorkerReg.Meta):
                __abstract__ = True
                def work(self) -> str:
                    return ""

            class AlphaWorker(Base):
                def work(self) -> str:
                    return "alpha"
        """), encoding="utf-8")

        sys.path.insert(0, str(tmp_path))
        try:
            from conscribe.scanner import list_registries

            summaries = list_registries(
                tmp_path,
                discover_packages=["test_list_pkg"],
                layer_filter="_test_list_worker",
            )

            assert len(summaries) == 1
            s = summaries[0]
            assert s.name == "_test_list_worker"
            assert s.entry_count == 1
            assert s.entries[0].key == "alpha_worker"
            assert s.entries[0].class_name == "AlphaWorker"
            assert s.entries[0].file_path is not None
            assert "registry.py" in s.entries[0].file_path
        finally:
            sys.path.remove(str(tmp_path))
            # Cleanup
            to_remove = [k for k in sys.modules if k.startswith("test_list_pkg")]
            for k in to_remove:
                del sys.modules[k]
            _deregister("_test_list_worker")

    def test_no_packages_returns_empty(self, tmp_path: Path) -> None:
        """Returns empty list when no packages found."""
        from conscribe.scanner import list_registries

        result = list_registries(tmp_path)
        assert result == []

    def test_path_filter(self, tmp_path: Path) -> None:
        """--path filter only includes matching entries."""
        from conscribe.registration.registry import _deregister

        pkg = tmp_path / "test_path_pkg"
        sub_a = pkg / "sub_a"
        sub_b = pkg / "sub_b"
        sub_a.mkdir(parents=True)
        sub_b.mkdir(parents=True)

        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (sub_a / "__init__.py").write_text("", encoding="utf-8")
        (sub_b / "__init__.py").write_text("", encoding="utf-8")

        (pkg / "registry.py").write_text(textwrap.dedent("""\
            from typing import Protocol, runtime_checkable
            from conscribe import create_registrar

            @runtime_checkable
            class ItemProto(Protocol):
                def run(self) -> str: ...

            ItemReg = create_registrar("_test_path_item", ItemProto)

            class Base(metaclass=ItemReg.Meta):
                __abstract__ = True
                def run(self) -> str:
                    return ""
        """), encoding="utf-8")

        (sub_a / "impl.py").write_text(textwrap.dedent("""\
            from test_path_pkg.registry import Base

            class FooItem(Base):
                def run(self) -> str:
                    return "foo"
        """), encoding="utf-8")

        (sub_b / "impl.py").write_text(textwrap.dedent("""\
            from test_path_pkg.registry import Base

            class BarItem(Base):
                def run(self) -> str:
                    return "bar"
        """), encoding="utf-8")

        sys.path.insert(0, str(tmp_path))
        try:
            from conscribe.scanner import list_registries

            summaries = list_registries(
                tmp_path,
                discover_packages=["test_path_pkg"],
                layer_filter="_test_path_item",
                path_filter="test_path_pkg/sub_a",
            )

            assert len(summaries) == 1
            s = summaries[0]
            # Only sub_a entry should appear
            assert s.entry_count == 1
            assert s.entries[0].key == "foo_item"
        finally:
            sys.path.remove(str(tmp_path))
            to_remove = [k for k in sys.modules if k.startswith("test_path_pkg")]
            for k in to_remove:
                del sys.modules[k]
            _deregister("_test_path_item")


# ===================================================================
# CLI integration
# ===================================================================

class TestCLIScan:
    """Tests for the scan CLI subcommand."""

    def test_scan_returns_zero(self, tmp_path: Path) -> None:
        from conscribe.cli import main

        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent", AgentProto)
        """)

        exit_code = main(["scan", "--path", str(tmp_path)])
        assert exit_code == 0

    def test_scan_empty_returns_zero(self, tmp_path: Path) -> None:
        from conscribe.cli import main

        exit_code = main(["scan", "--path", str(tmp_path)])
        assert exit_code == 0

    def test_scan_output_contains_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("my_layer", MyProto)
        """)

        main(["scan", "--path", str(tmp_path)])
        captured = capsys.readouterr()
        assert "my_layer" in captured.out
        assert "MyProto" in captured.out


class TestCLIList:
    """Tests for the list CLI subcommand."""

    def test_list_no_packages_returns_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from conscribe.cli import main

        monkeypatch.chdir(tmp_path)
        exit_code = main(["list"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "no Python packages found" in captured.err


# ===================================================================
# list / summarize_registries with strict discover()
# ===================================================================

_WORKERS = """\
    from typing import Protocol, runtime_checkable

    from conscribe import create_registrar


    @runtime_checkable
    class WorkerProto(Protocol):
        def work(self) -> str: ...


    WorkerReg = create_registrar("_scan_worker", WorkerProto)


    class Base(metaclass=WorkerReg.Meta):
        __abstract__ = True

        def work(self) -> str:
            return ""


    class AlphaWorker(Base):
        def work(self) -> str:
            return "alpha"
"""


@pytest.fixture
def worker_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``<root>/<pkg>/{__init__,registry}.py`` with one registered worker.

    The working directory is the project root.  Yields ``(root, pkg)``; the
    package's modules and registrations are removed afterwards.
    """
    root = tmp_path / "proj"
    name = f"scanproj_{uuid.uuid4().hex[:10]}"
    _write_py(root / name / "__init__.py", "")
    _write_py(root / name / "registry.py", _WORKERS)
    monkeypatch.chdir(root)
    before = set(sys.modules)
    with isolated_registries():
        try:
            yield root, name
        finally:
            for module in [m for m in sys.modules if m not in before]:
                if module == name or module.startswith(name + "."):
                    del sys.modules[module]


class TestListStrictDiscover:
    """list_registries() / summarize_registries() / ``conscribe list``."""

    def test_list_registries_forwards_exclude_and_allow_missing(self, worker_project) -> None:
        from conscribe.scanner import list_registries

        root, name = worker_project
        _write_py(root / name / "broken.py", "VALUE = undefined_name\n")
        _write_py(root / name / "optional.py", "import scan_missing_dependency_xyz\n")

        # Strict: the original NameError (3.10 wraps it in an ImportError naming the module).
        with pytest.raises((NameError, ImportError), match="undefined_name"):
            list_registries(root, discover_packages=[name], exclude=[f"{name}.optional"])

        summaries = list_registries(
            root,
            discover_packages=[name],
            layer_filter="_scan_worker",
            exclude=[f"{name}.broken"],
            allow_missing=["scan_missing_dependency_xyz"],
        )
        assert [e.key for e in summaries[0].entries] == ["alpha_worker"]

    def test_summarize_registries_imports_nothing(
        self, worker_project, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from conscribe.scanner import summarize_registries

        root, name = worker_project
        assert name not in sys.modules
        assert all(s.name != "_scan_worker" for s in summarize_registries(root))

        monkeypatch.syspath_prepend(str(root))
        __import__(f"{name}.registry")
        (summary,) = summarize_registries(root, layer_filter="_scan_worker")
        assert summary.entries[0].file_path == f"{name}/registry.py"

    def test_directory_sharing_a_prefix_with_root_is_not_under_it(
        self, worker_project, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Classes of ``/x/proj`` are not listed for root ``/x/pro``."""
        from conscribe.scanner import summarize_registries

        root, name = worker_project
        monkeypatch.syspath_prepend(str(root))
        __import__(f"{name}.registry")
        prefix_root = root.parent / root.name[:-1]
        prefix_root.mkdir()

        assert summarize_registries(root, layer_filter="_scan_worker")[0].entry_count == 1
        assert summarize_registries(prefix_root, layer_filter="_scan_worker")[0].entry_count == 0

    def test_cli_list_reports_a_broken_module_cleanly(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        root, name = worker_project
        _write_py(root / name / "broken.py", "VALUE = undefined_name\n")

        code = main(["list", "--discover", name])
        err = capsys.readouterr().err
        assert code == 1
        assert err.startswith(f"Error: could not import module '{name}.broken'")
        assert "NameError" in err and "broken.py, line 1" in err
        assert f"--exclude {name}.broken" in err
        assert "Traceback" not in err

        code = main(["list", "--discover", name, "--exclude", f"{name}.broken"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Registry: _scan_worker (WorkerProto)" in out
        assert "alpha_worker" in out

    def test_cli_list_auto_detects_packages_and_allows_missing(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        root, name = worker_project
        _write_py(root / name / "optional.py", "import scan_missing_dependency_xyz\n")

        code = main(["list", "--layer", "_scan_worker"])
        err = capsys.readouterr().err
        assert code == 1
        assert "--allow-missing scan_missing_dependency_xyz" in err

        code = main(["list", "--layer", "_scan_worker", "--allow-missing", "scan_missing_dependency_xyz"])
        out = capsys.readouterr().out
        assert code == 0
        assert "alpha_worker" in out

    def test_cli_list_unknown_layer(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        _, name = worker_project
        code = main(["list", "--discover", name, "--layer", "_scan_nope"])
        err = capsys.readouterr().err
        assert code == 1
        assert err.startswith("Error: registry '_scan_nope' not found. Available: ")
        assert "_scan_worker" in err

    def test_cli_list_missing_package(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        code = main(["list", "--discover", "scanproj_does_not_exist"])
        err = capsys.readouterr().err
        assert code == 1
        assert "could not import module 'scanproj_does_not_exist'" in err
        assert "ModuleNotFoundError" in err

    def test_cli_list_path_without_a_match_says_so(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        root, name = worker_project
        code = main(["list", "--discover", name, "--path", "elsewhere"])
        out = capsys.readouterr().out
        assert code == 0
        assert out.startswith("No registered classes are defined under 'elsewhere'")
        assert "relative to" in out

        code = main(["list", "--discover", name, "--layer", "_scan_worker", "--path", "elsewhere"])
        out = capsys.readouterr().out
        assert code == 0
        assert "No registered classes in registry '_scan_worker' are defined under 'elsewhere'" in out

    def test_cli_list_path_forms(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``pkg``, ``./pkg``, ``pkg/``, an absolute path and a file all select the entry."""
        from conscribe.cli import main

        root, name = worker_project
        forms = [name, f"./{name}", f"{name}/", str(root / name), f"{name}/registry.py", "."]
        for form in forms:
            code = main(["list", "--discover", name, "--layer", "_scan_worker", "--path", form])
            out = capsys.readouterr().out
            assert code == 0
            assert "alpha_worker" in out, form

    def test_cli_list_path_trailing_separator_is_a_directory(
        self, worker_project, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``<pkg>/reg/`` does not match ``<pkg>/registry.py`` (``<pkg>/reg`` does)."""
        from conscribe.cli import main

        root, name = worker_project
        main(["list", "--discover", name, "--path", f"{name}/reg"])
        assert "alpha_worker" in capsys.readouterr().out
        main(["list", "--discover", name, "--path", f"{name}/reg/"])
        assert "No registered classes" in capsys.readouterr().out

    def test_cli_list_path_outside_the_root(
        self, worker_project, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        root, name = worker_project
        code = main(["list", "--discover", name, "--path", str(tmp_path / "other")])
        out = capsys.readouterr().out
        assert code == 0
        assert "No registered classes" in out


class TestCLIScanWarnings:
    def test_unparseable_file_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from conscribe.cli import main

        (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
        _write_py(tmp_path / "layers.py", """\
            from conscribe import create_registrar
            R = create_registrar("agent", AgentProto)
        """)

        code = main(["scan", "--path", str(tmp_path)])
        captured = capsys.readouterr()
        assert code == 0
        assert "Warning: scan skipped" in captured.err and "bad.py" in captured.err
        assert "agent" in captured.out

    def test_not_a_directory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from conscribe.cli import main

        code = main(["scan", "--path", str(tmp_path / "missing")])
        err = capsys.readouterr().err
        assert code == 1
        assert err.startswith("Error: ") and "is not a directory" in err
