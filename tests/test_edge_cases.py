"""Robustness tests for edge-case branches.

Each test validates that defensive code paths behave correctly under
unusual but plausible conditions — graceful degradation on corrupt input,
correct fallback values, no silent data loss, no crashes.

Organised by module.  Every test docstring starts with the real-world
scenario it guards against, not the line number it covers.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import (
    Any,
    Annotated,
    Protocol,
    Union,
    runtime_checkable,
)
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from conscribe import create_registrar


# ===================================================================
# Shared protocol
# ===================================================================


@runtime_checkable
class _SimpleProto(Protocol):
    def do_work(self) -> str: ...


# ===================================================================
# CLI — error handling and edge-case output
# ===================================================================


class TestCLIEdgeCases:
    """Verify the CLI gives clear feedback on bad input, not stack traces."""

    def test_inspect_unknown_layer_prints_error_and_exits_1(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """User typos a layer name → helpful error on stderr, exit 1.

        This exercises the *real* ``_get_registrar`` raising KeyError
        (not a mock) and the ``except KeyError`` handler in ``_cmd_inspect``.
        """
        from conscribe.cli import main

        exit_code = main(["inspect", "--layer", "typo_layer"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "typo_layer" in captured.err

    def test_inspect_parameterless_class_shows_no_config(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A registered class whose __init__ takes only ``self`` should
        appear in inspect output as having no configurable parameters,
        rather than crashing or showing an empty table.
        """
        from conscribe.cli import main

        reg = create_registrar(
            "noparam_cli", _SimpleProto, discriminator_field="kind",
        )

        class _Base(metaclass=reg.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class _Minimal(_Base):
            __registry_key__ = "minimal"
            # intentionally no __init__

        with patch("conscribe.cli._get_registrar", return_value=reg):
            exit_code = main(["inspect", "--layer", "noparam_cli"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "(no config parameters)" in captured.out


# ===================================================================
# Codegen — type rendering with unusual type objects
# ===================================================================


class TestCodegenTypeEdgeCases:
    """Generated Python source must be syntactically valid even for
    exotic type annotations that rarely appear in normal code."""

    def test_three_way_union_with_none_renders_all_members(self) -> None:
        """``Union[str, int, None]`` is NOT a simple Optional, so codegen
        must render each member including ``None`` for NoneType."""
        from conscribe.config.codegen import _type_to_source

        source = _type_to_source(Union[str, int, None])
        # NoneType must appear as "None", not "NoneType"
        assert "None" in source
        assert "str" in source
        assert "int" in source

    def test_annotated_type_unwraps_to_base_in_source(self) -> None:
        """``Annotated[int, Field(ge=0)]`` should render as just ``int``
        in generated source (the metadata lives in the Field, not the type)."""
        from conscribe.config.codegen import _type_to_source

        source = _type_to_source(Annotated[int, Field(ge=0)])
        assert source == "int"

    def test_annotated_union_import_collection(self) -> None:
        """When a field has ``Annotated[Union[str, int], ...]``, the import
        collector must recurse through Annotated to find the Union inside."""
        from conscribe.config.codegen import _collect_type_imports

        imports: set[str] = set()
        _collect_type_imports(Annotated[Union[str, int], Field()], imports)
        assert "Union" in imports

    def test_bare_generic_renders_as_plain_name(self) -> None:
        """``typing.List`` (unsubscripted) should render as ``list``,
        not ``list[]`` or crash."""
        import typing
        from conscribe.config.codegen import _type_to_source

        assert _type_to_source(typing.List) == "list"
        assert _type_to_source(typing.Dict) == "dict"

    def test_type_without_dunder_name_falls_back_to_repr(self) -> None:
        """A synthetic type object that lacks ``__name__`` must not crash;
        the repr fallback should produce *some* string."""
        from conscribe.config.codegen import _get_type_name

        # Real types always have __name__; simulate a synthetic type-like
        # object that lacks it (e.g. a runtime-generated generic alias).
        class _FakeType:
            """Has no __name__ attribute."""
            def __repr__(self) -> str:
                return "<FakeType>"

        fake = _FakeType()
        # _FakeType instances don't have __name__, and it's not Any
        result = _get_type_name(fake)
        assert result == "<FakeType>"

    def test_ellipsis_default_renders_as_three_dots(self) -> None:
        """A FieldInfo whose default is ``...`` (Ellipsis) must render
        as ``...`` in source, not ``Ellipsis``."""
        from conscribe.config.codegen import _value_to_source

        assert _value_to_source(...) == "..."

    def test_non_primitive_default_uses_repr(self) -> None:
        """Default values that are not str/int/float/bool/list/dict
        (e.g. tuple, frozenset, custom objects) must fall back to repr
        without crashing."""
        from conscribe.config.codegen import _value_to_source

        assert _value_to_source((1, 2, 3)) == repr((1, 2, 3))
        assert _value_to_source(frozenset({1})) == repr(frozenset({1}))

    def test_string_default_is_properly_escaped(self) -> None:
        """String defaults containing quotes or backslashes must be
        safely escaped via repr, not naively wrapped in double quotes
        (which would enable code injection in generated .py files)."""
        from conscribe.config.codegen import _value_to_source

        dangerous = 'x"; import os; os.system("id"); x="'
        source = _value_to_source(dangerous)
        # repr wraps in single quotes with proper escaping
        assert source == repr(dangerous)
        # The generated source must not contain an unescaped double quote
        # outside of the string literal
        assert source.count("'") >= 2  # repr wraps in quotes


# ===================================================================
# Docstring — graceful degradation when docstring_parser missing
# ===================================================================


class TestDocstringDegradation:
    """Users who don't install the optional ``docstring_parser`` dep
    should still get working (if description-less) schemas."""

    def test_returns_empty_dict_when_parser_unavailable(self) -> None:
        """Even if the class has perfect Google-style docstrings,
        ``parse_param_descriptions`` returns {} when ``docstring_parser``
        is not installed — no ImportError leaks to the caller."""
        from conscribe.config.docstring import parse_param_descriptions

        class _WellDocumented:
            """A class with docs.

            Args:
                x: A perfectly documented parameter.
            """
            def __init__(self, x: int):
                self.x = x

        with patch("conscribe.config.docstring._HAS_DOCSTRING_PARSER", False):
            result = parse_param_descriptions(_WellDocumented)

        assert result == {}


# ===================================================================
# Extractor — robustness under broken / unusual annotations
# ===================================================================


class TestExtractorEdgeCases:
    """The extractor must never crash on user-defined classes, even if
    their annotations are missing, broken, or use exotic typing forms."""

    def test_fieldinfo_constraint_attrs_are_preserved(self) -> None:
        """``Annotated[int, Field(ge=0, le=100)]`` constraints must
        survive extraction into the dynamic Pydantic model so that
        validation actually rejects out-of-range values.

        This is critical for Bob: if the schema says ``ge=0`` but the
        generated model silently drops the constraint, invalid config
        would be accepted.
        """
        from conscribe.config.extractor import extract_config_schema

        # Define via exec so `from __future__ import annotations` in this
        # file doesn't turn the annotation into a string that might resolve
        # differently in the test module's scope.
        ns = {"Annotated": Annotated, "Field": Field}
        exec(
            "class Bounded:\n"
            "    def __init__(self, x: Annotated[int, Field(ge=0, le=100)]):\n"
            "        self.x = x\n",
            ns,
        )

        schema = extract_config_schema(ns["Bounded"])
        assert schema is not None

        # Happy path
        obj = schema(x=50)
        assert obj.x == 50

        # Constraint violation must raise, not silently pass
        with pytest.raises(Exception):
            schema(x=-1)
        with pytest.raises(Exception):
            schema(x=101)

    def test_hints_totally_unavailable_returns_none_not_crash(self) -> None:
        """If ``get_type_hints`` fails *and* the fallback annotation
        resolution also fails (e.g. a C extension class), the extractor
        must return None gracefully — not crash with an AttributeError
        or attempt to build a model from no information."""
        from conscribe.config.extractor import extract_config_schema

        # Create a class with params via exec (no annotations)
        ns = {}
        exec(
            "class Opaque:\n"
            "    def __init__(self, x, y):\n"
            "        pass\n",
            ns,
        )

        with patch(
            "conscribe.config.extractor._safe_get_type_hints",
            return_value=None,
        ):
            result = extract_config_schema(ns["Opaque"])

        assert result is None

    def test_safe_get_type_hints_with_empty_annotations(self) -> None:
        """When ``get_type_hints`` raises (unusual) and the function has
        no ``__annotations__`` at all, the fallback should return None
        cleanly — not attempt to iterate an empty dict."""
        from conscribe.config.extractor import _safe_get_type_hints

        def bare_func():
            pass

        with patch(
            "conscribe.config.extractor.get_type_hints",
            side_effect=TypeError("forced failure"),
        ):
            result = _safe_get_type_hints(bare_func)

        assert result is None

    def test_safe_get_type_hints_keeps_preresolved_annotations(self) -> None:
        """When ``get_type_hints`` raises but ``__annotations__`` contains
        actual type objects (not strings), the fallback must keep them
        as-is rather than trying to ``eval()`` a type object."""
        from conscribe.config.extractor import _safe_get_type_hints

        def func(x):
            pass
        func.__annotations__ = {"x": int}

        with patch(
            "conscribe.config.extractor.get_type_hints",
            side_effect=TypeError("forced failure"),
        ):
            result = _safe_get_type_hints(func)

        assert result == {"x": int}

    def test_safe_get_type_hints_catastrophic_failure_returns_none(self) -> None:
        """If even the fallback annotation processing crashes (e.g.
        ``__annotations__`` is corrupted to a non-dict), the function
        must return None — never propagate an unhandled exception."""
        from conscribe.config.extractor import _safe_get_type_hints

        def func(x):
            pass

        # Simulate a function whose __annotations__ is corrupted in a way
        # that causes iteration to fail.  Python 3.12+ won't allow setting
        # __annotations__ to a non-dict directly, so we patch getattr to
        # return something that blows up on .items().
        class _BrokenDict(dict):
            def items(self):
                raise RuntimeError("corrupted annotations")

        with (
            patch(
                "conscribe.config.extractor.get_type_hints",
                side_effect=TypeError("forced failure"),
            ),
            patch.object(func, "__annotations__", _BrokenDict({"x": "int"})),
        ):
            result = _safe_get_type_hints(func)

        assert result is None


# ===================================================================
# Fingerprint — corrupt cache files and unusual classes
# ===================================================================


class TestFingerprintEdgeCases:
    """Fingerprint computation must be deterministic and never crash.
    Cache load/save must tolerate corrupt or unexpected file contents."""

    def test_class_without_custom_init_gets_stable_hash(self) -> None:
        """A registered class that inherits ``object.__init__`` (no custom
        ``__init__`` anywhere in its MRO) should still produce a valid,
        deterministic fingerprint — not crash with an AttributeError."""
        from conscribe.config.fingerprint import compute_registry_fingerprint

        reg = create_registrar(
            "noinit_fp", _SimpleProto, discriminator_field="kind",
        )

        class _Base(metaclass=reg.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return ""

        class _NoInit(_Base):
            __registry_key__ = "noinit"
            def do_work(self) -> str:
                return "noinit"

        fp1 = compute_registry_fingerprint(reg)
        fp2 = compute_registry_fingerprint(reg)
        assert isinstance(fp1, str) and len(fp1) == 16
        assert fp1 == fp2  # deterministic

    def test_signature_failure_produces_different_hash(self) -> None:
        """When ``inspect.signature()`` raises on a class's __init__,
        the fingerprint must still succeed (using a placeholder) and
        the result must differ from the normal fingerprint, so that a
        stale-signature scenario triggers regeneration."""
        from conscribe.config.fingerprint import compute_registry_fingerprint

        reg = create_registrar(
            "sigfail_fp", _SimpleProto, discriminator_field="kind",
        )

        class _Base(metaclass=reg.Meta):
            __abstract__ = True
            def __init__(self, x: int):
                self.x = x
            def do_work(self) -> str:
                return ""

        class _Impl(_Base):
            __registry_key__ = "impl"

        normal_fp = compute_registry_fingerprint(reg)

        with patch(
            "conscribe.config.fingerprint.inspect.signature",
            side_effect=ValueError("cannot inspect"),
        ):
            degraded_fp = compute_registry_fingerprint(reg)

        assert isinstance(degraded_fp, str) and len(degraded_fp) == 16
        assert normal_fp != degraded_fp

    def test_load_returns_none_when_json_is_a_list(self, tmp_path: Path) -> None:
        """A fingerprint cache file containing valid JSON that is a list
        (not a dict) must return None — not crash with AttributeError
        when trying ``.get()`` on a list."""
        from conscribe.config.fingerprint import load_cached_fingerprint

        cache = tmp_path / "fp.json"
        cache.write_text(json.dumps(["not", "a", "dict"]))

        assert load_cached_fingerprint(cache, "llm") is None

    def test_save_overwrites_corrupt_json(self, tmp_path: Path) -> None:
        """If the fingerprint file contains unparseable garbage, ``save``
        must overwrite it cleanly — not crash or append to the corruption."""
        from conscribe.config.fingerprint import (
            load_cached_fingerprint,
            save_fingerprint,
        )

        cache = tmp_path / "fp.json"
        cache.write_text("{{{{ not json !!!!")

        save_fingerprint(cache, "llm", "abc123")

        assert load_cached_fingerprint(cache, "llm") == "abc123"

    def test_save_overwrites_non_dict_json(self, tmp_path: Path) -> None:
        """If the fingerprint file contains valid JSON that is a list,
        ``save`` must replace it with a proper dict."""
        from conscribe.config.fingerprint import (
            load_cached_fingerprint,
            save_fingerprint,
        )

        cache = tmp_path / "fp.json"
        cache.write_text(json.dumps([1, 2, 3]))

        save_fingerprint(cache, "llm", "def456")

        assert load_cached_fingerprint(cache, "llm") == "def456"
        # Other data should be gone
        data = json.loads(cache.read_text())
        assert isinstance(data, dict)


# ===================================================================
# Discover — auto-freshness edge cases
# ===================================================================


class TestDiscoverEdgeCases:
    """discover() auto-freshness must be a no-op when there's nothing
    to do, and must derive sensible defaults for optional parameters."""

    def test_auto_freshness_skips_when_no_registrars(
        self, tmp_path: Path,
    ) -> None:
        """With ``auto_update_stubs=True`` and a ``stub_dir`` set but no
        known registrars, discover() must complete normally and not create
        any fingerprint/stub files."""
        from conscribe.discover import discover

        stub_dir = tmp_path / "stubs"
        stub_dir.mkdir()
        fp_path = tmp_path / "fp.json"

        mod_file = tmp_path / "trivial_mod.py"
        mod_file.write_text("X = 1\n")

        sys.path.insert(0, str(tmp_path))
        try:
            result = discover(
                "trivial_mod",
                auto_update_stubs=True,
                stub_dir=stub_dir,
                fingerprint_path=fp_path,
            )
            assert "trivial_mod" in result
            # No registrars → no fingerprint file created
            assert not fp_path.exists()
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("trivial_mod", None)

    def test_auto_freshness_derives_fingerprint_path_from_stub_dir(
        self, tmp_path: Path,
    ) -> None:
        """When ``fingerprint_path`` is None, ``_auto_update_stubs`` must
        derive it as ``stub_dir / '.registry_fingerprint'`` — not crash
        with a NoneType error."""
        from conscribe.discover import _auto_update_stubs

        stub_dir = tmp_path / "stubs"
        stub_dir.mkdir()

        # Should not raise — even though _get_known_registrars returns [],
        # the path derivation code on line 126 must execute safely.
        _auto_update_stubs(stub_dir, fingerprint_path=None)


# ===================================================================
# Registration — bridge metaclass strategies & propagate hooks
# ===================================================================


class TestBridgeStrategies:
    """The bridge() method resolves metaclass conflicts using four
    strategies.  Strategies 2 and 3 are rare (only triggered when the
    external class uses a non-trivial metaclass chain) but must work
    correctly to avoid silent metaclass conflicts at class creation time."""

    def test_strategy_2_our_meta_already_satisfies_external(self) -> None:
        """Strategy 2: ``issubclass(cls.Meta, ext_meta)`` — our Meta is
        *already* a subclass of the external metaclass, so we reuse our
        Meta directly.

        Real-world scenario: a framework provides a base metaclass, and
        the registrar was configured with ``base_metaclass=FrameworkMeta``.
        An external class uses the *same* FrameworkMeta.
        """
        class FrameworkMeta(type):
            pass

        R = create_registrar(
            "strat2", _SimpleProto, base_metaclass=FrameworkMeta,
        )

        class ExtClass(metaclass=FrameworkMeta):
            def do_work(self) -> str:
                return "ext"

        Bridge = R.bridge(ExtClass)

        class Impl(Bridge):
            __registry_key__ = "impl"
            def do_work(self) -> str:
                return "impl"

        assert R.get("impl") is Impl
        # Bridge should use our Meta (strategy 2 reuses it)
        assert type(Bridge) is R.Meta

    def test_strategy_3_external_meta_is_more_specific(self) -> None:
        """Strategy 3: ``issubclass(ext_meta, cls.Meta)`` — the external
        metaclass is a *child* of our Meta, so we use the external one
        (which is more specific and satisfies both sides).

        Real-world scenario: two registrars where R2.Meta extends R1.Meta.
        A class created with R2.Meta is bridged into R1.
        """
        R1 = create_registrar("strat3_parent", _SimpleProto)
        R2 = create_registrar(
            "strat3_child", _SimpleProto, base_metaclass=R1.Meta,
        )

        class ExtBase(metaclass=R2.Meta):
            __abstract__ = True
            def do_work(self) -> str:
                return "ext"

        class ExtImpl(ExtBase):
            __registry_key__ = "ext_impl"

        Bridge = R1.bridge(ExtImpl)

        class BridgedImpl(Bridge):
            __registry_key__ = "bridged"
            def do_work(self) -> str:
                return "bridged"

        assert R1.get("bridged") is BridgedImpl


class TestPropagateHook:
    """When ``propagate=True``, the registrar injects an
    ``__init_subclass__`` hook.  If the target class already has its own
    hook, the injected one must chain-call it — otherwise the user's
    custom logic silently disappears."""

    def test_original_init_subclass_still_fires(self) -> None:
        """A class with its own ``__init_subclass__`` hook, registered
        with ``propagate=True``, should have BOTH the auto-registration
        AND the original hook fire when a child class is defined."""
        R = create_registrar("prop_test", _SimpleProto)
        hook_log: list[str] = []

        class _Base:
            def do_work(self) -> str:
                return "base"

        @R.register("base_key", propagate=True)
        class _Registered(_Base):
            def __init_subclass__(cls, **kwargs: Any) -> None:
                super().__init_subclass__(**kwargs)
                hook_log.append(cls.__name__)

        class Child(_Registered):
            def do_work(self) -> str:
                return "child"

        # Auto-registration via propagate
        assert R.get("child") is Child
        # Original __init_subclass__ also fired
        assert "Child" in hook_log


class TestNegativeCache:
    """The protocol check uses a negative cache to avoid re-computing
    missing methods for classes that already failed.  The cached path
    must raise the same error with the same detail as the computed path."""

    def test_second_check_raises_same_error_from_cache(self) -> None:
        """Two consecutive runtime_check calls on a non-conforming class
        must both raise TypeError mentioning the missing method — the
        second one coming from the negative cache, not a silent pass."""
        from conscribe.registration.registry import LayerRegistry

        registry = LayerRegistry("negcache", _SimpleProto)

        class _Bad:
            pass  # Missing do_work

        # First call: compute and cache
        with pytest.raises(TypeError, match="do_work"):
            registry.runtime_check(_Bad)

        # Second call: hit negative cache — same error
        with pytest.raises(TypeError, match="do_work"):
            registry.runtime_check(_Bad)
