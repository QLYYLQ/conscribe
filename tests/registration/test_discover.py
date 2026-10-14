"""Tests for conscribe.discover (also reachable as conscribe.registration.discover).

Covers:
- discover() triggers module imports -> metaclass registration fills registry
- strict (default): an import failure propagates, naming the module
- strict=False: ImportError / SyntaxError skipped with an ExtractionWarning
- allow_missing / exclude
- registrations of failed modules are rolled back
- the failure record (import_failures / clear_import_failures)
- removed 1.x parameters (auto_update_stubs, stub_dir, fingerprint_path)
- Nonexistent package raises ModuleNotFoundError
- Multiple packages in single call
- Return value lists imported modules
- sys.path / sys.modules cleanup for test isolation
"""
from __future__ import annotations

import importlib
import logging
import sys
import textwrap
import types
import warnings
from typing import Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.discover import clear_import_failures, discover, import_failures
from conscribe.exceptions import ExtractionWarning, WiringResolutionError
from conscribe.registration.registry import _deregister

PY311 = sys.version_info >= (3, 11)


@runtime_checkable
class DiscoverProto(Protocol):
    def run(self) -> str: ...


@pytest.fixture(autouse=True)
def _clean_failure_record():
    """discover() records failures globally; keep them out of other tests."""
    clear_import_failures()
    yield
    clear_import_failures()


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
        """With strict=False a broken submodule is skipped (with an
        ExtractionWarning) and the good modules are still discovered."""
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
            with pytest.warns(ExtractionWarning, match=r"skipped module 'mixed_agents\.bad'.*SyntaxError"):
                result = discover("mixed_agents", strict=False)
            # Good module's class should be registered despite bad module
            assert R.get("good_runner") is not None
            assert "mixed_agents.good" in result
            assert "mixed_agents.bad" not in result
            assert import_failures()["mixed_agents.bad"].startswith("SyntaxError: ")
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


# ===================================================================
# 2.0: strict discover, allow_missing, exclude, rollback, failure record
# ===================================================================

#: Source of a module that registers one class under *key* in the bridge
#: registrar (the metaclass root itself is never registered, hence the base).
_REGISTERING = """\
from _v2d_bridge import R

class _Base(metaclass=R.Meta):
    __abstract__ = True

    def run(self) -> str:
        return ""

class {cls}(_Base):
    __registry_key__ = "{key}"
"""


def _registering(cls: str, key: str, tail: str = "") -> str:
    return _REGISTERING.format(cls=cls, key=key) + textwrap.dedent(tail)


@pytest.fixture
def bridge():
    """A fresh registrar reachable from generated modules as ``_v2d_bridge.R``."""
    R = create_registrar("v2d_items", DiscoverProto)
    module = types.ModuleType("_v2d_bridge")
    module.R = R  # type: ignore[attr-defined]
    sys.modules["_v2d_bridge"] = module
    yield R
    del sys.modules["_v2d_bridge"]
    _deregister("v2d_items")


@pytest.fixture
def make_pkg(tmp_path):
    """Write a package (``{relative path: source}``) and make it importable.

    Every directory becomes a package; ``sys.path`` and ``sys.modules`` are
    restored afterwards.
    """
    roots: list[str] = []
    names: list[str] = []

    def make(name: str, files: dict[str, str]) -> str:
        root = tmp_path / f"root{len(roots)}"
        package_dir = root / name
        package_dir.mkdir(parents=True)
        for relative, source in files.items():
            path = package_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(source))
        for directory in [package_dir, *[d for d in package_dir.rglob("*") if d.is_dir()]]:
            if not (directory / "__init__.py").exists():
                (directory / "__init__.py").write_text("")
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        roots.append(str(root))
        names.append(name)
        return name

    yield make
    for root in roots:
        sys.path.remove(root)
    for name in names:
        for module_name in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[module_name]


def _failure_type(original: type[BaseException]) -> type[BaseException]:
    """What strict discover raises for a failing submodule on this Python."""
    return original if PY311 else ImportError


def _assert_names_module(exc: BaseException, module: str, original: type[BaseException]) -> None:
    """The raised error names the module that failed to import."""
    if PY311:
        assert isinstance(exc, original)
        assert (
            f"(raised while conscribe.discover() was importing module '{module}')"
            in exc.__notes__
        )
    else:  # pragma: no cover - Python 3.10 only
        assert type(exc) is ImportError
        assert exc.name == module
        assert isinstance(exc.__cause__, original)
        assert f"conscribe.discover() could not import module '{module}': " in str(exc)


class TestStrictDiscover:
    def test_syntax_error_propagates_by_default(self, make_pkg, bridge):
        name = make_pkg("v2d_syntax", {"bad.py": "def broken(\n", "good.py": _registering("Good", "good")})

        with pytest.raises(_failure_type(SyntaxError)) as exc_info:
            discover(name)
        _assert_names_module(exc_info.value, f"{name}.bad", SyntaxError)
        assert import_failures()[f"{name}.bad"].startswith("SyntaxError: ")

    @pytest.mark.parametrize(
        ("source", "original"),
        [
            ("import v2d_inpkg.does_not_exist\n", ModuleNotFoundError),
            ("from v2d_inpkg.good import nothing_here\n", ImportError),
            ("import v2d_dependency_not_installed\n", ModuleNotFoundError),
            ("raise RuntimeError('boom at import')\n", RuntimeError),
            ("undefined_name_at_import_time\n", NameError),
        ],
    )
    def test_any_import_failure_propagates(self, make_pkg, bridge, source, original):
        name = make_pkg("v2d_inpkg", {"a_bad.py": source, "good.py": _registering("Good", "good")})

        with pytest.raises(_failure_type(original)) as exc_info:
            discover(name)
        _assert_names_module(exc_info.value, f"{name}.a_bad", original)

    def test_registrations_of_the_failed_module_are_rolled_back(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_zombie",
            {
                "a_alive.py": _registering("Alive", "alive"),  # positive control
                "bad.py": _registering("Zombie", "zombie", "raise RuntimeError('late')\n"),
            },
        )

        with pytest.raises(_failure_type(RuntimeError)):
            discover(name)
        assert bridge.get_or_none("alive") is not None
        assert bridge.get_or_none("zombie") is None
        assert bridge.keys() == ["alive"]

    def test_successful_dependencies_keep_their_registrations(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_dep",
            {
                "a_bad.py": "import v2d_dep.z_dep\nraise RuntimeError('after importing a dependency')\n",
                "z_dep.py": _registering("Dep", "dep"),
            },
        )

        with pytest.raises(_failure_type(RuntimeError)):
            discover(name)
        # z_dep imported fine and stays in sys.modules; it will never run
        # again, so rolling its class back would lose it for good.
        assert bridge.get_or_none("dep") is not None
        assert f"{name}.z_dep" in sys.modules

    def test_failure_handled_by_another_module_is_rolled_back(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_handled",
            {
                "a_optional.py": """\
                    try:
                        import v2d_handled.b_broken
                    except RuntimeError:
                        HAVE_BROKEN = False
                """,
                "b_broken.py": _registering("Half", "half", "raise RuntimeError('broken extra')\n"),
            },
        )

        # a_optional imports fine; b_broken's leftover registration is rolled
        # back, so the walk's own import of b_broken fails with its real
        # error instead of a DuplicateKeyError.
        with pytest.raises(_failure_type(RuntimeError)) as exc_info:
            discover(name)
        _assert_names_module(exc_info.value, f"{name}.b_broken", RuntimeError)
        assert bridge.get_or_none("half") is None

    def test_top_level_package_always_fails_fast(self, make_pkg, bridge):
        name = make_pkg("v2d_toplevel", {"__init__.py": "def broken(\n"})

        for strict in (True, False):
            with pytest.raises(SyntaxError) as exc_info:
                discover(name, strict=strict)
            if PY311:
                assert (
                    f"(raised while conscribe.discover() was importing module '{name}')"
                    in exc_info.value.__notes__
                )
        assert import_failures()[name].startswith("SyntaxError: ")

    def test_nonexistent_package_raises_in_both_modes(self):
        for strict in (True, False):
            with pytest.raises(ModuleNotFoundError):
                discover("v2d_totally_nonexistent_package", strict=strict)


class TestNonStrictDiscover:
    def test_import_and_syntax_errors_are_skipped_with_warnings(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_lenient",
            {
                "a_import.py": "import v2d_dependency_not_installed\n",
                "b_syntax.py": "def broken(\n",
                "c_good.py": _registering("Good", "good"),
                "sub/d_bad.py": "from v2d_lenient.c_good import nothing_here\n",
            },
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = discover(name, strict=False)

        skipped = [w for w in caught if issubclass(w.category, ExtractionWarning)]
        messages = [str(w.message) for w in skipped]
        assert len(messages) == 3
        assert messages[0].startswith(
            f"discover() skipped module '{name}.a_import' because importing it raised "
            "ModuleNotFoundError: No module named 'v2d_dependency_not_installed'. "
            "Classes defined in it are not registered;"
        )
        assert "SyntaxError: " in messages[1]
        assert f"'{name}.sub.d_bad'" in messages[2] and "ImportError: cannot import name" in messages[2]
        # Warnings point at the caller of discover(), also from a subpackage.
        assert {w.filename for w in skipped} == {__file__}
        assert bridge.get_or_none("good") is not None
        assert result == [name, f"{name}.c_good", f"{name}.sub"]
        assert sorted(import_failures()) == [f"{name}.a_import", f"{name}.b_syntax", f"{name}.sub.d_bad"]

    def test_other_exceptions_still_propagate(self, make_pkg, bridge):
        name = make_pkg("v2d_lenient_rt", {"bad.py": "raise RuntimeError('not an import problem')\n"})

        with pytest.raises(_failure_type(RuntimeError)) as exc_info:
            discover(name, strict=False)
        _assert_names_module(exc_info.value, f"{name}.bad", RuntimeError)

    def test_skipped_module_registrations_are_rolled_back(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_lenient_zombie",
            {"bad.py": _registering("Zombie", "zombie", "import v2d_dependency_not_installed\n")},
        )

        with pytest.warns(ExtractionWarning):
            discover(name, strict=False)
        assert bridge.get_or_none("zombie") is None


class TestAllowMissing:
    def test_missing_optional_dependency_is_skipped_quietly(self, make_pkg, bridge, caplog):
        name = make_pkg(
            "v2d_allow",
            {
                "gan.py": _registering("Gan", "gan", "import v2d_numpy_fake\n"),
                "good.py": _registering("Good", "good"),
            },
        )

        with caplog.at_level(logging.INFO, logger="conscribe.discover"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # no warning of any kind
                result = discover(name, allow_missing=["v2d_numpy_fake"])

        assert result == [name, f"{name}.good"]
        assert bridge.get_or_none("good") is not None
        assert bridge.get_or_none("gan") is None  # rolled back
        assert import_failures()[f"{name}.gan"] == (
            "ModuleNotFoundError: No module named 'v2d_numpy_fake' "
            "(skipped: 'v2d_numpy_fake' is in allow_missing)"
        )
        assert any(
            r.levelno == logging.INFO and f"{name}.gan" in r.getMessage() for r in caplog.records
        )

    @pytest.mark.parametrize(
        "source",
        [
            # wrapped optional-dependency error (__cause__)
            """\
            try:
                import v2d_np
            except ImportError as exc:
                raise ImportError("install the [np] extra") from exc
            """,
            # an ImportError raised while handling the missing import (__context__)
            """\
            try:
                import v2d_np
            except ImportError:
                raise ImportError("v2d_np is required for this module")
            """,
            # a missing submodule of an allowed root
            "raise ModuleNotFoundError(\"No module named 'v2d_np.linalg'\", name='v2d_np.linalg')\n",
        ],
    )
    def test_chained_and_dotted_missing_modules(self, make_pkg, bridge, source):
        name = make_pkg("v2d_chain", {"opt.py": source})

        assert discover(name, allow_missing="v2d_np") == [name]  # a single str is accepted
        assert f"{name}.opt" in import_failures()

    @pytest.mark.parametrize(
        ("source", "original"),
        [
            # a deliberate non-import error raised while handling the import
            (
                """\
                try:
                    import v2d_np
                except ImportError:
                    raise RuntimeError("v2d_np is required for this module")
                """,
                RuntimeError,
            ),
            # review probe 11: the fallback the module turns to is broken
            (
                """\
                try:
                    import v2d_np
                except ImportError:
                    BROKEN = 1 / 0  # a real bug in the fallback path
                """,
                ZeroDivisionError,
            ),
        ],
    )
    def test_chain_is_followed_only_through_import_errors(
        self, make_pkg, bridge, source, original
    ):
        """Spec 4.7: a missing allowed module in ``__context__`` of an error of
        another type is a real failure, never a planned absence."""
        name = make_pkg("v2d_chain_bug", {"opt.py": source})

        with pytest.raises(_failure_type(original)) as exc_info:
            discover(name, allow_missing=["v2d_np"])
        _assert_names_module(exc_info.value, f"{name}.opt", original)

    def test_bug_in_a_fallback_outside_the_package_propagates(self, make_pkg, bridge):
        """Review probe 11b: the broken fallback lives in another package."""
        shared = make_pkg("v2d_shared_fallback", {"fallback.py": "BROKEN = 1 / 0\n"})
        name = make_pkg(
            "v2d_uses_fallback",
            {
                "backend.py": f"""\
                    try:
                        import v2d_torch_not_installed
                    except ImportError:
                        from {shared} import fallback
                """,
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(_failure_type(ZeroDivisionError)):
                discover(name, allow_missing=["v2d_torch_not_installed"])

    def test_unlisted_missing_module_still_raises(self, make_pkg, bridge):
        name = make_pkg("v2d_unlisted", {"opt.py": "import v2d_np\n"})

        with pytest.raises(_failure_type(ModuleNotFoundError)):
            discover(name, allow_missing=["v2d_other"])

    def test_does_not_cover_other_errors(self, make_pkg, bridge):
        name = make_pkg("v2d_allow_syntax", {"bad.py": "def broken(\n"})

        with pytest.raises(_failure_type(SyntaxError)):
            discover(name, allow_missing=["v2d_np"])

    def test_applies_in_non_strict_mode_too(self, make_pkg, bridge):
        name = make_pkg("v2d_allow_lenient", {"opt.py": "import v2d_np\n"})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert discover(name, strict=False, allow_missing=["v2d_np"]) == [name]

    def test_missing_top_level_package_can_be_allowed(self):
        assert discover("v2d_plugin_not_installed", allow_missing=["v2d_plugin_not_installed"]) == []


class TestExclude:
    def test_excluded_modules_are_never_imported(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_excl",
            {
                "bad.py": "raise RuntimeError('must never run')\n",
                "gen/broken.py": "def broken(\n",
                "generated.py": _registering("Kept", "kept"),
            },
        )

        result = discover(name, exclude=[f"{name}.bad", f"{name}.gen"])

        assert result == [name, f"{name}.generated"]
        assert f"{name}.bad" not in sys.modules
        assert f"{name}.gen" not in sys.modules
        # "<pkg>.gen" excludes <pkg>.gen and below, not <pkg>.generated.
        assert bridge.get_or_none("kept") is not None

    def test_excluding_the_package_itself(self, make_pkg, bridge):
        name = make_pkg("v2d_excl_all", {"bad.py": "raise RuntimeError('must never run')\n"})

        assert discover(name, exclude=name) == []
        assert name not in sys.modules

    @pytest.mark.parametrize("bad", [[1], [""], [None]])
    def test_rejects_non_names(self, bad):
        with pytest.raises(TypeError, match=r"discover\(exclude=\.\.\.\) expects module names"):
            discover("v2d_whatever", exclude=bad)
        with pytest.raises(TypeError, match=r"discover\(allow_missing=\.\.\.\) expects module names"):
            discover("v2d_whatever", allow_missing=bad)


class TestFailureRecord:
    def test_record_holds_strings_and_returns_a_copy(self, make_pkg, bridge):
        name = make_pkg("v2d_rec", {"bad.py": "import v2d_dependency_not_installed\n"})

        with pytest.warns(ExtractionWarning):
            discover(name, strict=False)
        record = import_failures()
        assert record == {
            f"{name}.bad": "ModuleNotFoundError: No module named 'v2d_dependency_not_installed'"
        }
        record.clear()
        assert import_failures() != {}

        clear_import_failures()
        assert import_failures() == {}

    def test_modules_imported_since_are_not_reported(self, make_pkg, bridge, tmp_path):
        name = make_pkg("v2d_since", {"mod.py": "import v2d_dependency_not_installed\n"})

        with pytest.warns(ExtractionWarning):
            discover(name, strict=False)
        assert f"{name}.mod" in import_failures()

        # Imported by other means (here: the module is replaced in place).
        sys.modules[f"{name}.mod"] = types.ModuleType(f"{name}.mod")
        assert f"{name}.mod" not in import_failures()

    def test_successful_reimport_forgets_the_failure(self, make_pkg, bridge, tmp_path):
        name = make_pkg("v2d_fixed", {"mod.py": "def broken(\n"})

        with pytest.warns(ExtractionWarning):
            discover(name, strict=False)
        assert f"{name}.mod" in import_failures()

        (tmp_path / "root0" / name / "mod.py").write_text(_registering("Fixed", "fixed"))
        importlib.invalidate_caches()
        discover(name)

        assert f"{name}.mod" not in import_failures()
        assert bridge.get_or_none("fixed") is not None

    def test_wiring_errors_cite_the_failed_module(self, make_pkg, bridge):
        """The af scenario: a broken implementation module surfaces in the
        wiring error that its missing class causes."""
        from conscribe.wiring import resolve_wiring

        name = make_pkg("v2d_cite", {"impl.py": _registering("Impl", "impl", "def broken(\n")})

        with pytest.warns(ExtractionWarning):
            discover(name, strict=False)

        class Owner:
            __wiring__ = {"item": "v2d_items"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Owner)
        problem = exc_info.value.problems[0]
        assert problem.kind == "empty_registry"
        assert f"discover() recorded import failures, and classes defined in those modules are not registered: {name}.impl (SyntaxError: " in problem.detail


class TestDiscoverWalk:
    def test_depth_first_name_order(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_order",
            {"b.py": "", "a/z.py": "", "a/y/x.py": "", "c.py": ""},
        )

        assert discover(name) == [
            name, f"{name}.a", f"{name}.a.y", f"{name}.a.y.x", f"{name}.a.z", f"{name}.b", f"{name}.c",
        ]

    def test_plain_module_is_imported_not_walked(self, tmp_path):
        (tmp_path / "v2d_plain_module.py").write_text("X = 1\n")
        sys.path.insert(0, str(tmp_path))
        importlib.invalidate_caches()
        try:
            assert discover("v2d_plain_module") == ["v2d_plain_module"]
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("v2d_plain_module", None)


class TestRemovedParameters:
    @pytest.mark.parametrize("kwarg", ["auto_update_stubs", "stub_dir", "fingerprint_path"])
    def test_auto_freshness_parameters_are_gone(self, kwarg):
        with pytest.raises(TypeError, match=kwarg):
            discover("v2d_whatever", **{kwarg: None})

    def test_auto_freshness_helpers_are_gone(self):
        module = sys.modules["conscribe.discover"]
        assert not hasattr(module, "_auto_update_stubs")
        assert not hasattr(module, "_get_known_registrars")


# ===================================================================
# 2.0 post-review: rollback attribution, fast path, __main__, BaseException
# ===================================================================


class TestRollbackAttribution:
    """Spec 4.7: rollback only touches modules whose import was attempted and
    failed, attributed by the module whose top-level code ran."""

    def test_synthetic_module_names_of_a_successful_import_are_kept(self, make_pkg, bridge):
        """Review probe 10 / discover_false_rollback: classes whose ``__module__``
        names a module that is not (yet) imported."""
        name = make_pkg(
            "v2d_synthetic",
            {
                "factory.py": """\
                    from _v2d_bridge import R

                    class _Base(metaclass=R.Meta):
                        __abstract__ = True

                        def run(self) -> str:
                            return ""

                    for key in ("alpha", "beta"):
                        type(key.title(), (_Base,), {"__registry_key__": key, "__module__": "v2d_synthetic.virtual"})

                    class Fancy(_Base):
                        __registry_key__ = "fancy"
                        __module__ = "v2d_synthetic.api"   # its documented public location
                """,
                "plugins.py": """\
                    from v2d_synthetic.factory import _Base

                    namespace = {"_Base": _Base, "__name__": "v2d_synthetic.generated_plugins"}
                    exec("class Generated(_Base):\\n    __registry_key__ = 'generated'\\n", namespace)
                """,
            },
        )

        discover(name)
        assert sorted(bridge.keys()) == ["alpha", "beta", "fancy", "generated"]
        assert import_failures() == {}

    def test_failure_rolls_back_classes_with_synthetic_module_names(self, make_pkg, bridge):
        name = make_pkg(
            "v2d_synthetic_bad",
            {
                "bad.py": _registering(
                    "Zombie", "zombie",
                    """\
                    type("Ghost", (_Base,), {"__registry_key__": "ghost", "__module__": "elsewhere"})
                    raise RuntimeError("late")
                    """,
                ),
            },
        )

        with pytest.raises(_failure_type(RuntimeError)):
            discover(name)
        assert bridge.keys() == []

    def test_rollback_restores_a_binding_the_failed_module_replaced(self, make_pkg, bridge):
        class Original:
            def run(self) -> str:
                return ""

        bridge.register("shared")(Original)
        name = make_pkg(
            "v2d_replacing",
            {
                "bad.py": """\
                    from _v2d_bridge import R

                    R.unregister("shared")

                    @R.register("shared")
                    class Replacement:
                        def run(self) -> str:
                            return ""

                    raise RuntimeError("late")
                """,
            },
        )

        with pytest.raises(_failure_type(RuntimeError)):
            discover(name)
        assert bridge.get("shared") is Original


def _count_recordings(monkeypatch) -> list[int]:
    """Count discover()'s rollback recordings (one per first-time import)."""
    from conscribe.registration import registry as registry_module

    calls: list[int] = []
    real = registry_module._recording

    def counting():
        calls.append(1)
        return real()

    monkeypatch.setattr(registry_module, "_recording", counting)
    return calls


class TestAlreadyImported:
    def test_imported_modules_cost_no_recording(self, make_pkg, bridge, monkeypatch):
        """Spec 4.7 / review bench: a second discover() of the same package does
        no snapshot or rollback work (it used to copy every registry per
        module)."""
        name = make_pkg("v2d_fast", {f"m{i}.py": _registering(f"C{i}", f"c{i}") for i in range(3)})
        first = discover(name)
        assert sorted(bridge.keys()) == ["c0", "c1", "c2"]

        calls = _count_recordings(monkeypatch)
        assert discover(name) == first
        assert calls == []

    def test_new_modules_are_recorded(self, make_pkg, bridge, monkeypatch):
        calls = _count_recordings(monkeypatch)
        name = make_pkg("v2d_counted", {"a.py": "", "b.py": ""})
        discover(name)
        assert len(calls) == 3  # the package and its two modules


class TestMainAndBaseExceptions:
    def test_main_modules_are_not_imported_by_the_walk(self, make_pkg, bridge):
        """Review probe 14: importing pkg.__main__ runs a command-line entry point."""
        name = make_pkg(
            "v2d_cli",
            {
                "__main__.py": _registering("CliOnly", "cli_only", "raise SystemExit(2)\n"),
                "sub/__main__.py": "raise SystemExit(3)\n",
                "good.py": _registering("Good", "good"),
            },
        )

        assert discover(name) == [name, f"{name}.good", f"{name}.sub"]
        assert f"{name}.__main__" not in sys.modules
        assert bridge.keys() == ["good"]

    def test_named_main_module_is_imported(self, make_pkg, bridge):
        name = make_pkg("v2d_cli_named", {"__main__.py": _registering("Named", "named")})

        assert discover(f"{name}.__main__") == [f"{name}.__main__"]
        assert bridge.keys() == ["named"]

    @pytest.mark.parametrize("raised", ["SystemExit(2)", "KeyboardInterrupt()"])
    def test_base_exception_rolls_back_and_propagates_unchanged(self, make_pkg, bridge, raised):
        name = make_pkg(
            "v2d_base_exc",
            {
                "a_alive.py": _registering("Alive", "alive"),
                "b_exit.py": _registering("BeforeExit", "before_exit", f"raise {raised}\n"),
            },
        )

        expected = SystemExit if raised.startswith("SystemExit") else KeyboardInterrupt
        with pytest.raises(expected) as exc_info:
            discover(name)
        assert type(exc_info.value) is expected
        assert not getattr(exc_info.value, "__notes__", None)
        assert bridge.keys() == ["alive"]
        assert import_failures()[f"{name}.b_exit"].startswith(expected.__name__)
