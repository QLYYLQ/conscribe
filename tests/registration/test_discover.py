"""Tests for conscribe.registration.discover module.

Covers:
- discover() triggers module imports -> metaclass registration fills registry
- Bad submodule doesn't abort discovery of remaining modules
- Nonexistent package raises ModuleNotFoundError
- Multiple packages in single call
- Return value lists imported modules
- sys.path / sys.modules cleanup for test isolation
"""
from __future__ import annotations

import sys
import textwrap
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.registration.discover import discover


@runtime_checkable
class DiscoverProto(Protocol):
    def run(self) -> str: ...


# ===================================================================
# Helpers
# ===================================================================

def _create_package(
    tmp_path,
    package_name: str,
    modules: dict[str, str],
) -> str:
    """Create a temporary Python package with given modules.

    Args:
        tmp_path: pytest tmp_path fixture
        package_name: top-level package name
        modules: mapping of module filename -> source code

    Returns:
        The str path to add to sys.path
    """
    pkg_dir = tmp_path / package_name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")

    for filename, source in modules.items():
        (pkg_dir / filename).write_text(textwrap.dedent(source))

    return str(tmp_path)


# ===================================================================
# Tests
# ===================================================================

class TestDiscover:
    """Tests for the discover() function."""

    def test_discover_triggers_registration(self, tmp_path) -> None:
        """discover() imports modules -> metaclass auto-registration fills registry."""
        R = create_registrar("test", DiscoverProto)

        # We need the registrar's Meta accessible from inside the temp module.
        # We'll store it in a well-known module.
        import types
        bridge_mod = types.ModuleType("_discover_test_bridge")
        bridge_mod.R = R  # type: ignore[attr-defined]
        sys.modules["_discover_test_bridge"] = bridge_mod

        source = """\
        from _discover_test_bridge import R

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class FooRunner(Base):
            def run(self) -> str:
                return "foo"
        """
        path = _create_package(tmp_path, "my_agents", {"foo.py": source})
        sys.path.insert(0, path)

        try:
            result = discover("my_agents")
            assert isinstance(result, list)
            assert R.get("foo_runner") is not None
        finally:
            sys.path.remove(path)
            # Cleanup sys.modules
            to_remove = [k for k in sys.modules if k.startswith("my_agents")]
            for k in to_remove:
                del sys.modules[k]
            del sys.modules["_discover_test_bridge"]

    def test_discover_bad_submodule_continues(self, tmp_path) -> None:
        """Submodule import failure doesn't abort: good modules still discovered."""
        R = create_registrar("test", DiscoverProto)

        import types
        bridge_mod = types.ModuleType("_discover_test_bridge2")
        bridge_mod.R = R  # type: ignore[attr-defined]
        sys.modules["_discover_test_bridge2"] = bridge_mod

        good_source = """\
        from _discover_test_bridge2 import R

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class GoodRunner(Base):
            def run(self) -> str:
                return "good"
        """

        bad_source = """\
        # This will raise SyntaxError at import time
        def broken(
        """

        path = _create_package(
            tmp_path,
            "mixed_agents",
            {"good.py": good_source, "bad.py": bad_source},
        )
        sys.path.insert(0, path)

        try:
            result = discover("mixed_agents")
            # Good module's class should be registered despite bad module
            assert R.get("good_runner") is not None
        finally:
            sys.path.remove(path)
            to_remove = [k for k in sys.modules if k.startswith("mixed_agents")]
            for k in to_remove:
                del sys.modules[k]
            del sys.modules["_discover_test_bridge2"]

    def test_discover_nonexistent_package(self) -> None:
        """discover() with nonexistent package raises ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            discover("totally_nonexistent_package_xyz")

    def test_discover_returns_module_names(self, tmp_path) -> None:
        """discover() returns list of successfully imported module names."""
        R = create_registrar("test", DiscoverProto)

        import types
        bridge_mod = types.ModuleType("_discover_test_bridge3")
        bridge_mod.R = R  # type: ignore[attr-defined]
        sys.modules["_discover_test_bridge3"] = bridge_mod

        source_a = """\
        from _discover_test_bridge3 import R

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class RunnerA(Base):
            def run(self) -> str:
                return "a"
        """

        source_b = """\
        from _discover_test_bridge3 import R

        class Base2(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class RunnerB(Base2):
            def run(self) -> str:
                return "b"
        """

        path = _create_package(
            tmp_path,
            "multi_mod",
            {"mod_a.py": source_a, "mod_b.py": source_b},
        )
        sys.path.insert(0, path)

        try:
            result = discover("multi_mod")
            # Should have discovered at least the two submodules
            module_names = [r for r in result if "mod_a" in r or "mod_b" in r]
            assert len(module_names) >= 2
        finally:
            sys.path.remove(path)
            to_remove = [k for k in sys.modules if k.startswith("multi_mod")]
            for k in to_remove:
                del sys.modules[k]
            del sys.modules["_discover_test_bridge3"]

    def test_discover_nested_subpackage(self, tmp_path) -> None:
        """discover() recursively walks subpackages."""
        R = create_registrar("test", DiscoverProto)

        import types
        bridge_mod = types.ModuleType("_discover_test_bridge4")
        bridge_mod.R = R  # type: ignore[attr-defined]
        sys.modules["_discover_test_bridge4"] = bridge_mod

        # Create nested package: outer/inner/impl.py
        outer_dir = tmp_path / "outer_pkg"
        outer_dir.mkdir()
        (outer_dir / "__init__.py").write_text("")

        inner_dir = outer_dir / "inner"
        inner_dir.mkdir()
        (inner_dir / "__init__.py").write_text("")

        impl_source = textwrap.dedent("""\
        from _discover_test_bridge4 import R

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class NestedRunner(Base):
            def run(self) -> str:
                return "nested"
        """)
        (inner_dir / "impl.py").write_text(impl_source)

        sys.path.insert(0, str(tmp_path))

        try:
            discover("outer_pkg")
            assert R.get("nested_runner") is not None
        finally:
            sys.path.remove(str(tmp_path))
            to_remove = [k for k in sys.modules if k.startswith("outer_pkg")]
            for k in to_remove:
                del sys.modules[k]
            del sys.modules["_discover_test_bridge4"]

    def test_discover_multiple_packages(self, tmp_path) -> None:
        """discover() accepts multiple package paths."""
        R = create_registrar("test", DiscoverProto)

        import types
        bridge_mod = types.ModuleType("_discover_test_bridge5")
        bridge_mod.R = R  # type: ignore[attr-defined]
        sys.modules["_discover_test_bridge5"] = bridge_mod

        source_x = """\
        from _discover_test_bridge5 import R

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class RunnerX(Base):
            def run(self) -> str:
                return "x"
        """

        source_y = """\
        from _discover_test_bridge5 import R

        class Base2(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class RunnerY(Base2):
            def run(self) -> str:
                return "y"
        """

        path_x = _create_package(tmp_path / "dir1", "pkg_x", {"x.py": source_x})
        path_y = _create_package(tmp_path / "dir2", "pkg_y", {"y.py": source_y})

        sys.path.insert(0, path_x)
        sys.path.insert(0, path_y)

        try:
            result = discover("pkg_x", "pkg_y")
            assert R.get("runner_x") is not None
            assert R.get("runner_y") is not None
        finally:
            sys.path.remove(path_x)
            sys.path.remove(path_y)
            to_remove = [
                k for k in sys.modules
                if k.startswith("pkg_x") or k.startswith("pkg_y")
            ]
            for k in to_remove:
                del sys.modules[k]
            del sys.modules["_discover_test_bridge5"]

    def test_discover_empty_package(self, tmp_path) -> None:
        """discover() on a package with only __init__.py returns the package itself."""
        pkg_dir = tmp_path / "empty_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path))

        try:
            result = discover("empty_pkg")
            assert isinstance(result, list)
        finally:
            sys.path.remove(str(tmp_path))
            to_remove = [k for k in sys.modules if k.startswith("empty_pkg")]
            for k in to_remove:
                del sys.modules[k]

    def test_discover_idempotent(self, tmp_path) -> None:
        """Calling discover() twice on the same package should not cause errors."""
        R = create_registrar("test", DiscoverProto)

        import types
        bridge_mod = types.ModuleType("_discover_test_bridge6")
        bridge_mod.R = R  # type: ignore[attr-defined]
        sys.modules["_discover_test_bridge6"] = bridge_mod

        source = """\
        from _discover_test_bridge6 import R

        class Base(metaclass=R.Meta):
            __abstract__ = True
            def run(self) -> str:
                return ""

        class IdempotentRunner(Base):
            def run(self) -> str:
                return "idem"
        """
        path = _create_package(tmp_path, "idem_pkg", {"impl.py": source})
        sys.path.insert(0, path)

        try:
            discover("idem_pkg")
            # Second call should not raise (module already imported)
            discover("idem_pkg")
            assert R.get("idempotent_runner") is not None
        finally:
            sys.path.remove(path)
            to_remove = [k for k in sys.modules if k.startswith("idem_pkg")]
            for k in to_remove:
                del sys.modules[k]
            del sys.modules["_discover_test_bridge6"]
