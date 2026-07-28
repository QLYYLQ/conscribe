"""Regression tests: generated ``.pyi`` stubs must be valid Python.

Bug: the stub generator ``repr()``-ed defaults straight into ``__init__``
signatures, emitting ``<factory>`` (the dataclass ``default_factory``
sentinel) and ``FieldInfo(...)``; it referenced module-private names it
never imported; and it dropped ``async`` from every coroutine method, so
every ``await`` on those methods became a type error downstream.

The acceptance here is deliberately *mechanical*: the stub must
``compile()``, must bind every name it references, and must keep
``async def``.  A stub that merely "looks right" is not enough — the
broken one looked plausible and did not parse.
"""
from __future__ import annotations

import ast
import builtins
import sys

import pytest

from conscribe.registration.registry import (
    _REGISTRY_INDEX,
    LayerRegistry,
    _deregister,
)
from conscribe.stubs.collector import collect_class_stub_info, render_signature
from conscribe.stubs.generator import generate_module_stub

from . import _downstream_shapes as shapes


@pytest.fixture(autouse=True)
def _provider_registry():
    reg = LayerRegistry("test_stub_provider", shapes.ProviderProtocol)
    reg.add("local", shapes.LocalProvider)
    reg.add("remote", shapes.RemoteProvider)
    yield reg
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("test_"):
            _deregister(name)


@pytest.fixture
def stub_source() -> str:
    infos = [
        info
        for info in (
            collect_class_stub_info(shapes.DataclassCapability),
            collect_class_stub_info(shapes.PlainCapability),
        )
        if info is not None
    ]
    assert len(infos) == 2, "fixture classes should both have injected wiring"
    return generate_module_stub(shapes.__name__, infos)


# ── The mechanical acceptance ────────────────────────────────────


class TestStubIsValidPython:
    def test_stub_compiles(self, stub_source):
        """The whole point: the emitted stub must be real Python source."""
        compile(stub_source, "<stub>", "exec")

    def test_no_factory_sentinel(self, stub_source):
        """``<factory>`` is the dataclass sentinel repr — never valid source."""
        assert "<factory>" not in stub_source
        # No repr-of-an-object leaked into the source at all.
        assert "<" not in stub_source.replace("->", "")

    def test_default_factory_renders_as_ellipsis(self, stub_source):
        assert "action_overrides: dict[str, dict[str, Any]] = ..." in stub_source

    def test_pydantic_defaults_do_not_leak(self, stub_source):
        """FieldInfo / model instances degrade to ``...`` (the v1.1.2 defect)."""
        assert "FieldInfo(" not in stub_source
        assert "Settings(" not in stub_source
        assert "opts: Settings = ..." in stub_source

    def test_wired_field_descriptor_default_does_not_leak(self, stub_source):
        """A ``WiredField`` becomes a dataclass default; it must not be emitted."""
        assert "WiredField(" not in stub_source

    def test_every_referenced_name_is_bound(self, stub_source):
        """No undefined names: a .pyi *replaces* its module for type checkers."""
        from conscribe.stubs.generator import _unresolved_names

        assert _unresolved_names(stub_source) == []

    def test_module_private_names_fall_back_to_any(self, stub_source):
        """Same-module helpers are not importable from the stub → Any."""
        assert "_ObserveCycleCache = Any" in stub_source
        assert "_StateEventOutcome = Any" in stub_source


class TestAsyncSurvives:
    def test_async_def_present(self, stub_source):
        assert "async def" in stub_source

    @pytest.mark.parametrize(
        "name",
        ["setup", "observe", "stream", "probe", "build", "resolve_fragment"],
    )
    def test_coroutine_methods_stay_async(self, stub_source, name):
        tree = ast.parse(stub_source)
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        assert found, f"{name} missing from stub"
        assert all(isinstance(node, ast.AsyncFunctionDef) for node in found), (
            f"{name} lost its 'async'"
        )

    def test_sync_methods_stay_sync(self, stub_source):
        tree = ast.parse(stub_source)
        for name in ("teardown", "is_alive", "__init__"):
            nodes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ]
            assert nodes, f"{name} missing from stub"
            assert all(isinstance(n, ast.FunctionDef) for n in nodes), (
                f"{name} wrongly marked async"
            )

    def test_async_decorated_methods_keep_decorator_order(self, stub_source):
        assert "@staticmethod\n    async def probe" in stub_source
        assert "@classmethod\n    async def build" in stub_source


class TestWriterOutputCompiles:
    def test_written_pyi_file_compiles(self, tmp_path, _provider_registry):
        """End-to-end through the public writer entry point."""
        from conscribe import create_registrar
        from conscribe.stubs.writer import write_layer_stubs

        registrar = create_registrar("test_stub_cap", shapes.CapabilityProtocol)
        registrar.register("dataclass_cap")(shapes.DataclassCapability)
        registrar.register("plain_cap")(shapes.PlainCapability)

        written = write_layer_stubs(registrar, output_dir=tmp_path)
        assert written, "expected at least one .pyi"

        for path in written:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            assert "async def" in source


# ── render_signature unit coverage ───────────────────────────────


class TestRenderSignature:
    def _sig(self, func):
        import inspect

        return render_signature(inspect.signature(func))

    def test_simple_literals_are_preserved(self):
        def f(a: int = 1, b: str = "x", c: bool = False, d: None = None): ...

        rendered = self._sig(f)
        assert "a: int = 1" in rendered
        assert "b: str = 'x'" in rendered
        assert "c: bool = False" in rendered
        assert "d: None = None" in rendered

    def test_empty_containers_are_preserved(self):
        def f(a: list = [], b: dict = {}, c: tuple = ()): ...

        rendered = self._sig(f)
        assert "a: list = []" in rendered
        assert "b: dict = {}" in rendered
        assert "c: tuple = ()" in rendered

    def test_arbitrary_object_default_becomes_ellipsis(self):
        sentinel = object()

        def f(a: int = sentinel): ...  # type: ignore[assignment]

        assert "a: int = ..." in self._sig(f)

    def test_enum_default_becomes_ellipsis(self):
        import enum

        class Color(enum.IntEnum):
            RED = 1

        def f(a: int = Color.RED): ...

        rendered = self._sig(f)
        assert "a: int = ..." in rendered
        assert "Color" not in rendered

    def test_string_annotations_are_unquoted(self):
        """PEP 563 gives string annotations; the stub already defers them."""
        rendered = self._sig(shapes.DataclassCapability.observe)
        assert "cache: _ObserveCycleCache" in rendered
        assert "'_ObserveCycleCache'" not in rendered

    def test_explicit_forward_ref_is_not_double_quoted(self):
        rendered = self._sig(shapes.DataclassCapability.build)
        # Was rendered as -> "'DataclassCapability'" (a nested string).
        assert '"\'' not in rendered
        assert "DataclassCapability" in rendered

    def test_star_args_and_kwargs_survive(self):
        rendered = self._sig(shapes.PlainCapability.resolve_fragment)
        assert "**kwargs: Any" in rendered

    def test_keyword_only_marker_survives(self):
        def f(a: int, *, b: int = 2): ...

        assert "*, b: int = 2" in self._sig(f)

    @pytest.mark.skipif(
        sys.version_info < (3, 8), reason="positional-only params need 3.8+"
    )
    def test_positional_only_marker_survives(self):
        namespace: dict = {}
        exec("def f(a: int, /, b: int = 2): ...", namespace)
        assert "/" in self._sig(namespace["f"])

    def test_builtin_names_are_not_aliased(self):
        """``_unresolved_names`` must not flag real builtins."""
        from conscribe.stubs.generator import _unresolved_names

        source = (
            "from typing import Any\n"
            "def f(a: Exception, b: property) -> BaseException: ...\n"
        )
        assert _unresolved_names(source) == []
        assert "Exception" in dir(builtins)

    def test_literal_string_args_are_not_treated_as_names(self):
        """``Literal['a']`` must not produce a bogus ``a = Any`` alias."""
        from conscribe.stubs.generator import _unresolved_names

        source = (
            "from typing import Any, Literal\n"
            "def f(mode: Literal['fast', 'slow']) -> Any: ...\n"
        )
        assert _unresolved_names(source) == []
