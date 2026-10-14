"""Regression tests: generated ``.pyi`` stubs must be valid Python.

Bugs this guards against: the stub generator ``repr()``-ed defaults straight
into ``__init__`` signatures, emitting ``<factory>`` (the dataclass
``default_factory`` sentinel) and ``FieldInfo(...)``; it referenced
module-private names it never imported (then bound them to ``Any``); and it
dropped ``async`` from every coroutine method, so every ``await`` on those
methods became a type error downstream.

The acceptance here is deliberately *mechanical*: the stub must parse, must
bind every name it references, must import only names that exist, and must
keep ``async def``.  A stub that merely "looks right" is not enough -- the
broken one looked plausible and did not parse.
"""
from __future__ import annotations

import ast
import enum
import math
import re

import pytest

from conscribe.registration.registry import LayerRegistry
from conscribe.stubs.collector import (
    ClassStubInfo,
    InjectedAttr,
    collect_class_members,
    collect_class_stub_info,
)
from conscribe.stubs.generator import generate_module_stub

from . import _downstream_shapes as shapes
from ._helpers import assert_valid_stub, class_body


@pytest.fixture(autouse=True)
def _provider_registry(conscribe_isolated):
    """The provider registry, rolled back (with everything else) after each test."""
    reg = LayerRegistry("test_stub_provider", shapes.ProviderProtocol)
    reg.add("local", shapes.LocalProvider)
    reg.add("remote", shapes.RemoteProvider)
    yield reg


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

    def test_stub_is_self_consistent(self, stub_source):
        """Every name bound, nothing bound twice, every import exists."""
        assert_valid_stub(stub_source)

    def test_no_factory_sentinel(self, stub_source):
        """``<factory>`` is the dataclass sentinel repr -- never valid source."""
        assert "<factory>" not in stub_source
        # No repr-of-an-object leaked into the source at all.
        assert "<" not in stub_source.replace("->", "")

    def test_default_factory_renders_as_ellipsis(self, stub_source):
        assert "action_overrides: dict[str, dict[str, Any]] = ..." in stub_source

    def test_pydantic_defaults_do_not_leak(self, stub_source):
        """FieldInfo / model instances degrade to ``...`` (the v1.1.2 defect)."""
        assert "FieldInfo(" not in stub_source
        assert "= Settings(" not in stub_source
        assert "opts: Settings = ..." in stub_source

    def test_wired_field_descriptor_default_does_not_leak(self, stub_source):
        """A ``WiredField`` can become a dataclass default; it must not be emitted."""
        assert "WiredField(" not in stub_source

    def test_injected_attribute_is_typed_as_the_protocol(self, stub_source):
        """Mode 2 ``["local"]`` would have narrowed to ``LocalProvider`` in 1.3.0."""
        plain = class_body(stub_source, "PlainCapability")
        assert "backend: ProviderProtocol  # wired from: test_stub_provider" in plain
        assert "LocalProvider" not in stub_source

    def test_module_private_names_become_placeholders(self, stub_source):
        """Same-module helpers cannot be imported from their own stub: they are
        defined as nominal placeholders, never bound to ``Any``."""
        assert "class _ObserveCycleCache(Any): ..." in stub_source
        assert "class _StateEventOutcome(Any): ..." in stub_source
        assert "class Settings(Any): ..." in stub_source
        assert not re.search(r"^\w+ = Any$", stub_source, flags=re.M)

    def test_same_module_protocol_is_structural(self, stub_source):
        assert (
            "@runtime_checkable\nclass ProviderProtocol(Protocol):\n"
            "    def connect(self) -> str: ...\n"
        ) in stub_source


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

    def test_typing_alias_keeps_its_spelling(self, stub_source):
        """``AsyncIterator[str]`` from ``typing`` stays that (not collections.abc)."""
        assert "async def stream(self) -> AsyncIterator[str]: ..." in stub_source
        assert "AsyncIterator_2" not in stub_source


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
            assert_valid_stub(source)
            assert "async def" in source


# ── Signature rendering ──────────────────────────────────────────


def _signature(func) -> str:
    """The rendered ``def`` line of *func* as a method of a throwaway class."""
    # A static method: the rendered stub must type-check, and *func* has no ``self``.
    owner = type("Owner", (), {"f": staticmethod(func), "__module__": __name__})
    info = ClassStubInfo(
        cls=owner,
        class_name="Owner",
        module=__name__,
        source_file=None,
        bases=(object,),
        injected_attrs=(InjectedAttr("dummy", str, None),),
        methods=collect_class_members(owner).methods,
    )
    source = generate_module_stub(__name__, [info])
    line = next(line for line in source.splitlines() if "def f(" in line)
    return line.strip()


class TestSignatureRendering:
    def test_simple_literals_are_preserved(self):
        def f(a: int = 1, b: str = "x", c: bool = False, d: None = None): ...

        assert _signature(f) == "def f(a: int = 1, b: str = 'x', c: bool = False, d: None = None): ..."

    def test_empty_containers_are_preserved(self):
        def f(a: list = [], b: dict = {}, c: tuple = ()): ...

        assert _signature(f) == "def f(a: list = [], b: dict = {}, c: tuple = ()): ..."

    def test_arbitrary_object_default_becomes_ellipsis(self):
        sentinel = object()

        def f(a: int = sentinel): ...  # type: ignore[assignment]

        assert "a: int = ..." in _signature(f)

    def test_enum_default_becomes_ellipsis(self):
        class Color(enum.IntEnum):
            RED = 1

        def f(a: int = Color.RED): ...

        rendered = _signature(f)
        assert "a: int = ..." in rendered
        assert "Color" not in rendered

    def test_non_finite_float_default_becomes_ellipsis(self):
        """``repr(inf)`` is ``inf``: not a literal, would be an unbound name."""

        def f(a: float = math.inf, b: float = 0.5): ...

        assert _signature(f) == "def f(a: float = ..., b: float = 0.5): ..."

    def test_string_annotations_are_unquoted(self):
        source = generate_module_stub(
            shapes.__name__, [collect_class_stub_info(shapes.DataclassCapability)]
        )
        assert "cache: _ObserveCycleCache" in source
        assert "'_ObserveCycleCache'" not in source

    def test_explicit_forward_ref_is_not_double_quoted(self):
        source = generate_module_stub(
            shapes.__name__, [collect_class_stub_info(shapes.DataclassCapability)]
        )
        # Was rendered as -> "'DataclassCapability'" (a nested string).
        assert '"\'' not in source
        assert "async def build(cls) -> DataclassCapability: ..." in source

    def test_star_args_and_kwargs_survive(self):
        def f(*args: int, **kwargs: str): ...

        assert _signature(f) == "def f(*args: int, **kwargs: str): ..."

    def test_keyword_only_marker_survives(self):
        def f(a: int, *, b: int = 2): ...

        assert _signature(f) == "def f(a: int, *, b: int = 2): ..."

    def test_positional_only_marker_survives(self):
        namespace: dict = {}
        exec("def f(a: int, /, b: int = 2): ...", namespace)
        assert _signature(namespace["f"]) == "def f(a: int, /, b: int = 2): ..."

    def test_builtin_names_are_used_as_is(self):
        def f(a: Exception, b: property) -> BaseException: ...

        assert _signature(f) == "def f(a: Exception, b: property) -> BaseException: ..."
