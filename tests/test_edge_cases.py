"""Robustness tests for edge-case branches.

Each test validates that defensive code paths behave correctly under
unusual but plausible conditions — graceful degradation on corrupt input,
correct fallback values, no silent data loss, no crashes.

Organised by module.  Every test docstring starts with the real-world
scenario it guards against, not the line number it covers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Annotated,
    Protocol,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)
from unittest.mock import patch

import pytest
from pydantic import Field

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
        from tests.config.artifact_parity import type_to_source as _type_to_source

        source = _type_to_source(Union[str, int, None])
        # NoneType must appear as "None", not "NoneType"
        assert "None" in source
        assert "str" in source
        assert "int" in source

    def test_annotated_type_keeps_its_constraints_in_source(self) -> None:
        """``Annotated[int, Field(ge=0)]`` inside a type expression keeps its
        constraint (conscribe 2.0: nested constraints are rendered, not dropped)."""
        from tests.config.artifact_parity import type_to_source as _type_to_source

        source = _type_to_source(Annotated[int, Field(ge=0)])
        assert source == "Annotated[int, Field(ge=0)]"

    def test_bare_generic_renders_as_plain_name(self) -> None:
        """``typing.List`` (unsubscripted) should render as ``list``,
        not ``list[]`` or crash."""
        import typing
        from tests.config.artifact_parity import type_to_source as _type_to_source

        assert _type_to_source(typing.List) == "list"
        assert _type_to_source(typing.Dict) == "dict"

    def test_type_without_dunder_name_is_refused(self) -> None:
        """A synthetic type object that lacks ``__name__`` cannot be written
        as source; conscribe 2.0 raises instead of emitting its repr."""
        from tests.config.artifact_parity import type_to_source as _type_to_source
        from conscribe.exceptions import ConfigRenderError

        # Real types always have __name__; simulate a synthetic type-like
        # object that lacks it (e.g. a runtime-generated generic alias).
        class _FakeType:
            """Has no __name__ attribute."""
            def __repr__(self) -> str:
                return "<FakeType>"

        with pytest.raises(ConfigRenderError):
            _type_to_source(_FakeType())

    def test_ellipsis_default_renders_as_three_dots(self) -> None:
        """A FieldInfo whose default is ``...`` (Ellipsis) must render
        as ``...`` in source, not ``Ellipsis``."""
        from tests.config.artifact_parity import value_to_source as _value_to_source

        assert _value_to_source(...) == "..."

    def test_non_primitive_default_uses_repr(self) -> None:
        """Default values that are not str/int/float/bool/list/dict
        (e.g. tuple, frozenset, custom objects) must fall back to repr
        without crashing."""
        from tests.config.artifact_parity import value_to_source as _value_to_source

        assert _value_to_source((1, 2, 3)) == repr((1, 2, 3))
        assert _value_to_source(frozenset({1})) == repr(frozenset({1}))

    def test_string_default_is_properly_escaped(self) -> None:
        """String defaults containing quotes or backslashes must be
        safely escaped via repr, not naively wrapped in double quotes
        (which would enable code injection in generated .py files)."""
        from tests.config.artifact_parity import value_to_source as _value_to_source

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

    def test_unreadable_init_signature_warns_not_crash(self) -> None:
        """A constructor conscribe cannot introspect (e.g. a C extension
        or an opaque callable) must not crash extraction and must not be
        silent: an ``ExtractionWarning`` names the class, and the config
        accepts unknown keys rather than rejecting every real knob."""
        from conscribe.config.extractor import extract_config_schema
        from conscribe.exceptions import ExtractionWarning

        class _Opaque:
            __signature__ = "not a signature"

            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class Opaque:
            __init__ = _Opaque()

        with pytest.warns(ExtractionWarning, match="Opaque"):
            result = extract_config_schema(Opaque)

        assert result is not None
        assert result.model_fields == {}
        assert result.model_config.get("extra") == "allow"


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

        from conscribe.config import _annotations

        with patch("inspect.signature", side_effect=ValueError("cannot inspect")):
            _annotations._SIGNATURES.clear()  # signatures are cached per function
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
