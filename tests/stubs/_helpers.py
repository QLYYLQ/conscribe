"""Shared helpers for the stub tests.

* :func:`temp_package` writes a small package to disk and makes it
  importable, so the stub machinery sees real source files (for
  ``if TYPE_CHECKING:`` scans and ``inspect.getfile``) and modules with or
  without ``from __future__ import annotations``.  Registries the package
  creates are removed again afterwards.
* :func:`assert_valid_stub` is an *independent* validity check (it does not
  reuse the generator's own gate): the stub parses, binds every name it
  loads, binds no top-level name twice, and every ``from M import N`` names
  something that exists.
* :func:`run_pyright` type-checks stubs and consumer modules when ``pyright``
  is on ``PATH`` (tests using it skip otherwise).
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import json
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from conscribe.testing import isolated_registries

#: ``from M import N`` lines that are allowed to name stub-only / type-checking-only modules.
_UNVERIFIED_MODULES = frozenset({"__future__", "_typeshed"})


@contextlib.contextmanager
def temp_package(root: Path, name: str, files: dict[str, str]) -> Iterator[Path]:
    """Write *files* (relative path -> source) as package *name* under *root*.

    The package is importable inside the block, which runs inside
    :func:`conscribe.testing.isolated_registries`.  The package's modules
    leave ``sys.modules`` before the isolation ends, so the registries and
    registrations they created are rolled back too.
    """
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for relative, source in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    with isolated_registries():
        try:
            yield package
        finally:
            sys.path.remove(str(root))
            for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
                del sys.modules[module]


def assert_valid_stub(source: str, *, unverified: frozenset[str] = frozenset()) -> ast.Module:
    """Assert that *source* is a self-consistent stub; return its AST.

    Args:
        source: The stub text.
        unverified: Extra modules whose ``from`` imports are not checked
            (names imported only under ``if TYPE_CHECKING:`` in the source).
    """
    tree = ast.parse(source)

    top: list[str] = []
    packages: set[str] = set()  # bound by unaliased ``import a.b`` (may repeat)
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            top.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    packages.add(alias.name.split(".")[0])
                else:
                    top.append(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            top.extend(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            top.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top.append(node.target.id)
    top.extend(sorted(packages))
    duplicates = sorted({n for n in top if top.count(n) > 1})
    assert not duplicates, f"top-level names bound twice: {duplicates}"

    bound = set(dir(builtins)) | set(top)
    for node in ast.walk(tree):  # class-body names (``@x.setter``, enum members)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert used <= bound, f"names used but never bound: {sorted(used - bound)}"

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module in _UNVERIFIED_MODULES or node.module in unverified:
                continue
            assert node.module is not None and node.level == 0, ast.unparse(node)
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{ast.unparse(node)}: {alias.name} does not exist"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in unverified:
                    importlib.import_module(alias.name)
    return tree


#: ``pyright`` on PATH (the optional type-checker runs skip without it).
PYRIGHT = shutil.which("pyright")


def run_pyright(root: Path, include: list[str]) -> list[dict[str, Any]]:
    """Type-check *include* (paths relative to *root*) with pyright; return the errors.

    Uses this interpreter's environment (``--pythonpath``) and standard mode.
    """
    assert PYRIGHT is not None
    config = {"include": include, "pythonVersion": f"{sys.version_info[0]}.{sys.version_info[1]}"}
    (root / "pyrightconfig.json").write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.run(
        [PYRIGHT, "--pythonpath", sys.executable, "--outputjson", "-p", str(root)],
        cwd=root, capture_output=True, text=True, timeout=600,
    )
    report = json.loads(proc.stdout)
    return [d for d in report["generalDiagnostics"] if d["severity"] == "error"]


def class_body(source: str, name: str) -> str:
    """The source text of top-level class *name* in *source*."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"class {name} not in stub:\n{source}")
