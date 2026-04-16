"""Scanner: static AST analysis and runtime registry introspection.

Provides two discovery mechanisms:

1. **Static scan** (``scan_registrar_definitions``): Parses Python files
   via AST to find ``create_registrar()`` / ``create_auto_registrar()``
   calls without importing anything.

2. **Runtime list** (``list_registries``): Imports packages to trigger
   metaclass registration, then queries the global ``_REGISTRY_INDEX``
   to report what is registered.
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "site-packages", ".venv", "venv", ".git", "__pycache__",
    ".eggs", ".tox", "node_modules", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
})

_REGISTRAR_FUNCTION_NAMES: frozenset[str] = frozenset({
    "create_registrar",
    "create_auto_registrar",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegistrarDefinition:
    """Result of an AST scan: a ``create_registrar()`` call found in source."""

    name: str
    """Registry name (first arg), or ``"<dynamic>"`` if not a string literal."""
    protocol_name: str
    """Protocol class name (second arg), or ``"<dynamic>"``."""
    file_path: str
    """Absolute path to the ``.py`` file."""
    line_number: int
    """Line number of the call."""
    function_name: str
    """``"create_registrar"`` or ``"create_auto_registrar"``."""
    variable_name: Optional[str]
    """Assignment target (e.g. ``"R"``), or ``None`` for bare calls."""


@dataclass(frozen=True)
class RegistryEntry:
    """A single registered class in a runtime registry."""

    registry_name: str
    key: str
    class_name: str
    file_path: Optional[str]
    """Relative to scan root; ``None`` for built-in / C-extension classes."""
    line_number: Optional[int]
    module_name: Optional[str]


@dataclass(frozen=True)
class RegistrySummary:
    """Summary of a single ``LayerRegistry`` at runtime."""

    name: str
    protocol_name: str
    entry_count: int
    entries: list[RegistryEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Static AST scan
# ---------------------------------------------------------------------------

def scan_registrar_definitions(
    root: Path,
    *,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> list[RegistrarDefinition]:
    """Scan Python files under *root* for ``create_registrar()`` calls.

    Uses AST parsing — no imports, no side effects.

    Args:
        root: Directory to scan recursively.
        exclude_dirs: Directory names to skip (pruned during walk).

    Returns:
        List of :class:`RegistrarDefinition` found, sorted by file path
        then line number.
    """
    root = root.resolve()
    results: list[RegistrarDefinition] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs
        ]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                source = Path(fpath).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(source, filename=fpath)
            except SyntaxError:
                logger.warning("Skipping %s (syntax error)", fpath)
                continue
            _visit_module(tree, fpath, results)

    results.sort(key=lambda d: (d.file_path, d.line_number))
    return results


def _visit_module(
    tree: ast.Module,
    file_path: str,
    results: list[RegistrarDefinition],
) -> None:
    """Walk module AST to find registrar-creation calls."""
    _visit_body(tree.body, file_path, results)


def _visit_body(
    body: list[ast.stmt],
    file_path: str,
    results: list[RegistrarDefinition],
) -> None:
    """Recursively visit statement bodies for registrar calls."""
    for stmt in body:
        # Assignment: R = create_registrar(...)
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func_name = _extract_func_name(call)
            if func_name in _REGISTRAR_FUNCTION_NAMES:
                var_name = _extract_assign_target(stmt)
                defn = _build_definition(call, func_name, var_name, file_path)
                if defn is not None:
                    results.append(defn)

        # Annotated assignment: R: type = create_registrar(...)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func_name = _extract_func_name(call)
            if func_name in _REGISTRAR_FUNCTION_NAMES:
                var_name = _extract_ann_assign_target(stmt)
                defn = _build_definition(call, func_name, var_name, file_path)
                if defn is not None:
                    results.append(defn)

        # Bare expression: create_registrar(...)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func_name = _extract_func_name(call)
            if func_name in _REGISTRAR_FUNCTION_NAMES:
                defn = _build_definition(call, func_name, None, file_path)
                if defn is not None:
                    results.append(defn)

        # Recurse into nested bodies
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _visit_body(stmt.body, file_path, results)
        elif isinstance(stmt, (ast.If, ast.With, ast.AsyncWith)):
            _visit_body(stmt.body, file_path, results)
            if hasattr(stmt, "orelse") and stmt.orelse:
                _visit_body(stmt.orelse, file_path, results)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            _visit_body(stmt.body, file_path, results)
            if stmt.orelse:
                _visit_body(stmt.orelse, file_path, results)
        elif isinstance(stmt, ast.Try):
            _visit_body(stmt.body, file_path, results)
            for handler in stmt.handlers:
                _visit_body(handler.body, file_path, results)
            if stmt.orelse:
                _visit_body(stmt.orelse, file_path, results)
            if stmt.finalbody:
                _visit_body(stmt.finalbody, file_path, results)


def _extract_func_name(call: ast.Call) -> str:
    """Extract the function name from a Call node.

    Handles both ``create_registrar(...)`` and
    ``conscribe.create_registrar(...)``.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _extract_assign_target(stmt: ast.Assign) -> Optional[str]:
    """Extract variable name from ``X = ...``."""
    if len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _extract_ann_assign_target(stmt: ast.AnnAssign) -> Optional[str]:
    """Extract variable name from ``X: T = ...``."""
    target = stmt.target
    if isinstance(target, ast.Name):
        return target.id
    return None


def _build_definition(
    call: ast.Call,
    func_name: str,
    var_name: Optional[str],
    file_path: str,
) -> Optional[RegistrarDefinition]:
    """Build a RegistrarDefinition from a Call node.

    Returns None if the call has fewer than 2 positional args
    (not a valid ``create_registrar(name, protocol, ...)`` call).
    """
    args = call.args
    if len(args) < 2:
        return None

    # First arg: registry name
    name_node = args[0]
    if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
        name = name_node.value
    else:
        name = "<dynamic>"

    # Second arg: protocol class
    proto_node = args[1]
    if isinstance(proto_node, ast.Name):
        protocol_name = proto_node.id
    elif isinstance(proto_node, ast.Attribute):
        protocol_name = proto_node.attr
    else:
        protocol_name = "<dynamic>"

    return RegistrarDefinition(
        name=name,
        protocol_name=protocol_name,
        file_path=file_path,
        line_number=call.lineno,
        function_name=func_name,
        variable_name=var_name,
    )


# ---------------------------------------------------------------------------
# 2. Package detection
# ---------------------------------------------------------------------------

def find_packages(
    root: Path,
    *,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> list[str]:
    """Find top-level Python packages under *root*.

    A directory is considered a package if it contains ``__init__.py``.
    Only scans one level deep.

    Args:
        root: Directory to scan.
        exclude_dirs: Directory names to skip.

    Returns:
        Sorted list of package names.
    """
    root = root.resolve()
    packages: list[str] = []

    try:
        entries = os.listdir(root)
    except OSError:
        return []

    for entry in entries:
        if entry in exclude_dirs:
            continue
        full = os.path.join(root, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")):
            packages.append(entry)

    packages.sort()
    return packages


# ---------------------------------------------------------------------------
# 3. Runtime registry listing
# ---------------------------------------------------------------------------

def list_registries(
    root: Path,
    *,
    discover_packages: Optional[list[str]] = None,
    layer_filter: Optional[str] = None,
    path_filter: Optional[str] = None,
) -> list[RegistrySummary]:
    """Import packages and list runtime registry content.

    Args:
        root: Project root directory (added to ``sys.path`` if needed).
        discover_packages: Packages to import via ``discover()``.
            Auto-detected from *root* if ``None``.
        layer_filter: Only return the registry with this name.
        path_filter: Only include entries whose source file path
            (relative to *root*) starts with this prefix.

    Returns:
        List of :class:`RegistrySummary`, sorted by name.
    """
    from conscribe.discover import discover
    from conscribe.registration.registry import _REGISTRY_INDEX

    root = root.resolve()
    root_str = str(root)

    if discover_packages is None:
        discover_packages = find_packages(root)

    if not discover_packages:
        return []

    # Ensure root is on sys.path for imports
    added_to_path = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added_to_path = True

    try:
        discover(*discover_packages)

        summaries: list[RegistrySummary] = []

        for reg_name, registry in sorted(_REGISTRY_INDEX.items()):
            if layer_filter is not None and reg_name != layer_filter:
                continue

            entries: list[RegistryEntry] = []
            for key, cls in sorted(registry.items(), key=lambda kv: kv[0]):
                file_path, line_number, module_name = _get_class_location(cls, root)

                # Default filter: exclude entries not under root
                if file_path is None and path_filter is None:
                    # Cannot determine source — skip by default
                    continue
                if file_path is not None and "site-packages" in file_path:
                    continue

                if path_filter is not None:
                    if file_path is None or not file_path.startswith(path_filter):
                        continue

                entries.append(RegistryEntry(
                    registry_name=reg_name,
                    key=key,
                    class_name=cls.__qualname__,
                    file_path=file_path,
                    line_number=line_number,
                    module_name=module_name,
                ))

            summaries.append(RegistrySummary(
                name=reg_name,
                protocol_name=registry.protocol.__qualname__,
                entry_count=len(entries),
                entries=entries,
            ))

        return summaries

    finally:
        if added_to_path:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass


def _get_class_location(
    cls: type,
    root: Path,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Get source file (relative), line number, and module name for a class.

    Returns (None, None, module) if the source file cannot be determined
    or is not under *root*.
    """
    module_name = getattr(cls, "__module__", None)

    try:
        abs_path = inspect.getfile(cls)
    except (TypeError, OSError):
        return None, None, module_name

    abs_path = os.path.realpath(abs_path)
    root_str = str(root)

    if not abs_path.startswith(root_str):
        return None, None, module_name

    rel_path = os.path.relpath(abs_path, root_str)

    try:
        _, line_number = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        line_number = None

    return rel_path, line_number, module_name
