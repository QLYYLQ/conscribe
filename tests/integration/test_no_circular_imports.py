"""Integration tests: No circular imports between registration/ and config/ subsystems.

Enforced rules:
1. registration/ modules NEVER import from config/ at top level
2. config/ modules NEVER import from registration/ at top level
3. Cross-subsystem imports only inside function bodies (lazy imports)

These tests use ast.parse to statically verify the import structure.
"""
from __future__ import annotations

import ast
import os

import pytest


def _get_package_root() -> str:
    """Return the absolute path to the conscribe package directory."""
    import conscribe
    return os.path.dirname(os.path.abspath(conscribe.__file__))


def _get_top_level_imports(filepath: str) -> list[str]:
    """Parse a Python file and return all top-level import module names.

    Returns module names from:
    - `import X` -> "X"
    - `from X import Y` -> "X"

    Only considers imports at the top level of the module (not inside
    function/class bodies).
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r") as f:
        source = f.read()

    if not source.strip():
        return []

    tree = ast.parse(source, filename=filepath)
    imports: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def _collect_python_files(directory: str) -> list[str]:
    """Collect all .py files in a directory (non-recursive for simplicity)."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".py")
    ]


class TestNoCircularImports:
    """Static analysis: verify no circular imports between subsystems."""

    def test_registration_does_not_import_config(self) -> None:
        """registration/ modules must not have top-level imports from config/."""
        pkg_root = _get_package_root()
        registration_dir = os.path.join(pkg_root, "registration")
        py_files = _collect_python_files(registration_dir)

        violations: list[str] = []
        for filepath in py_files:
            filename = os.path.basename(filepath)
            imports = _get_top_level_imports(filepath)
            for imp in imports:
                if "config" in imp and "conscribe" in imp:
                    violations.append(f"{filename}: imports {imp}")

        assert violations == [], (
            f"registration/ has top-level imports from config/:\n"
            + "\n".join(violations)
        )

    def test_config_does_not_import_registration(self) -> None:
        """config/ modules must not have top-level imports from registration/."""
        pkg_root = _get_package_root()
        config_dir = os.path.join(pkg_root, "config")
        py_files = _collect_python_files(config_dir)

        violations: list[str] = []
        for filepath in py_files:
            filename = os.path.basename(filepath)
            imports = _get_top_level_imports(filepath)
            for imp in imports:
                if "registration" in imp and "conscribe" in imp:
                    violations.append(f"{filename}: imports {imp}")

        assert violations == [], (
            f"config/ has top-level imports from registration/:\n"
            + "\n".join(violations)
        )

    def test_registration_internal_dag(self) -> None:
        """Within registration/, verify no reverse-direction imports.

        Expected DAG:
        key_transform.py  <-  auto.py  <-  registrar.py
                                ^
        registry.py  -----------+
        discover.py  (independent)

        Specifically:
        - key_transform.py should NOT import from auto, registrar, discover, registry
        - registry.py should NOT import from auto, registrar, discover
        - auto.py should NOT import from registrar, discover
        - discover.py should NOT import from auto, registrar, key_transform, registry
        """
        pkg_root = _get_package_root()
        registration_dir = os.path.join(pkg_root, "registration")

        forbidden = {
            "key_transform.py": {"auto", "registrar", "discover", "registry"},
            "registry.py": {"auto", "registrar", "discover"},
            "auto.py": {"registrar", "discover"},
            "discover.py": {"auto", "registrar", "key_transform", "registry"},
        }

        violations: list[str] = []
        for filename, forbidden_modules in forbidden.items():
            filepath = os.path.join(registration_dir, filename)
            if not os.path.exists(filepath):
                continue
            imports = _get_top_level_imports(filepath)
            for imp in imports:
                for forbidden_mod in forbidden_modules:
                    if forbidden_mod in imp:
                        violations.append(f"{filename}: imports {imp} (forbidden: {forbidden_mod})")

        assert violations == [], (
            f"registration/ internal DAG violations:\n" + "\n".join(violations)
        )

    def test_conscribe_importable(self) -> None:
        """Smoke test: the package can be imported without circular import errors."""
        import conscribe
        assert hasattr(conscribe, "create_registrar")

    def test_registration_subpackage_importable(self) -> None:
        """Smoke test: registration subpackage can be imported."""
        from conscribe.registration import (
            LayerRegistry,
            create_auto_registrar,
            LayerRegistrar,
            create_registrar,
            discover,
            make_key_transform,
            default_key_transform,
        )
        # Just verify they exist
        assert LayerRegistry is not None
        assert create_auto_registrar is not None
        assert LayerRegistrar is not None
        assert create_registrar is not None
        assert discover is not None
        assert make_key_transform is not None
        assert default_key_transform is not None

    def test_exceptions_importable(self) -> None:
        """Smoke test: exceptions can be imported."""
        from conscribe.exceptions import (
            RegistryError,
            DuplicateKeyError,
            KeyNotFoundError,
            ProtocolViolationError,
            InvalidConfigSchemaError,
            InvalidProtocolError,
        )
        # Verify inheritance
        assert issubclass(DuplicateKeyError, RegistryError)
        assert issubclass(KeyNotFoundError, RegistryError)
        assert issubclass(KeyNotFoundError, KeyError)
        assert issubclass(ProtocolViolationError, RegistryError)
        assert issubclass(ProtocolViolationError, TypeError)
        assert issubclass(InvalidConfigSchemaError, RegistryError)
        assert issubclass(InvalidConfigSchemaError, TypeError)
        assert issubclass(InvalidProtocolError, RegistryError)
        assert issubclass(InvalidProtocolError, TypeError)
