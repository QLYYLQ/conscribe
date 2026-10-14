"""Tests for conscribe.config.fingerprint.

The fingerprint is a hash of what conscribe *generates* for a layer:
``conscribe.__version__``, the registrar settings that only show in composed
output (layer name, discriminator field(s), key separator, ``inline_only``),
the per-layer generated source and the per-layer JSON schema.

- Definition: the value is the first 16 hex chars of SHA-256 over exactly
  those inputs (error markers in their place when generation fails).
- Follows the outputs: an edit that changes the generated source or schema
  changes the fingerprint (field types, ``dict[str, int]`` vs
  ``dict[str, float]`` included, defaults, descriptions, constraints, wiring
  keys, default keys, registered keys, enum members, PEP 695 aliases); the
  composed-only settings and the conscribe version change it as well.
- Ignores edits that do not reach the outputs: comments, a class docstring's
  summary, the body of a factory the generated file imports by reference, a
  renamed class.
- Never raises: broken declarations give a deterministic fingerprint built
  from ``error:<ExceptionType>:<message>`` markers, different from the
  repaired layer's; warnings are suppressed while computing.
- Stable across processes (``PYTHONHASHSEED``) and, opt-in through
  ``CONSCRIBE_FINGERPRINT_PYTHONS`` (needs ``uv``), across Python versions.
- Cache files: ``load_cached_fingerprint`` / ``save_fingerprint``.

Most tests compare two *snapshots* -- the fingerprint together with the
outputs it hashes -- so each case also checks that its edit really does (or
does not) reach the generated files.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

import pydantic
import pytest
from pydantic import BaseModel, Field

import conscribe
from conscribe import create_registrar
from conscribe.config.builder import build_layer_config
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.fingerprint import (
    _error_marker,
    compute_registry_fingerprint,
    load_cached_fingerprint,
    save_fingerprint,
)
from conscribe.config.json_schema import generate_layer_json_schema
from conscribe.exceptions import ConscribeWarning
from conscribe.testing import isolated_registries

_HEX16 = re.compile(r"[0-9a-f]{16}")


# ===================================================================
# Helpers
# ===================================================================


@runtime_checkable
class _Proto(Protocol):
    def run(self) -> None: ...


@contextlib.contextmanager
def _isolated() -> Iterator[None]:
    """``isolated_registries()``, typed as a context manager that never swallows."""
    with isolated_registries():
        yield


@dataclasses.dataclass(frozen=True)
class _Snapshot:
    """A fingerprint and the outputs it hashes (or the text of their error)."""

    fingerprint: str
    outputs: tuple[str, str] | str


def _outputs(registrar: Any) -> tuple[str, str] | str:
    """``(generated source, schema JSON)`` of a layer, or ``"Type: message"``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = build_layer_config(registrar)
            source = generate_layer_config_source(result)
            schema = json.dumps(generate_layer_json_schema(result), sort_keys=True)
        except Exception as exc:  # noqa: BLE001 - the error is the output here
            return f"{type(exc).__name__}: {exc}"
    return source, schema


def _take(registrar: Any) -> _Snapshot:
    return _Snapshot(compute_registry_fingerprint(registrar), _outputs(registrar))


def _snapshot(define: Callable[[type], Any], **options: Any) -> _Snapshot:
    """Snapshot of a fresh layer ``fp_layer`` whose classes ``define(Base)`` creates.

    A target layer ``fp_target`` (keys ``a`` and ``b``) exists for wiring.
    Everything is rolled back afterwards.
    """
    if "discriminator_fields" not in options:
        options.setdefault("discriminator_field", "kind")
    with _isolated():
        target = create_registrar("fp_target", _Proto, discriminator_field="kind")

        class TargetBase(metaclass=target.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        type("TargetA", (TargetBase,), {"__registry_key__": "a"})
        type("TargetB", (TargetBase,), {"__registry_key__": "b"})

        reg = create_registrar("fp_layer", _Proto, **options)

        class Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        define(Base)
        return _take(reg)


def _impl(**namespace: Any) -> Callable[[type], Any]:
    """``define`` callback creating one class ``Impl`` (key ``impl``) with *namespace*."""

    def define(Base: type) -> Any:
        return type("Impl", (Base,), {"__registry_key__": "impl", **namespace})

    return define


def _dataclass_impl(annotation: str, default: Any = dataclasses.MISSING) -> Callable[[type], Any]:
    """``define`` callback creating a dataclass ``Impl`` with one field ``value``."""

    def define(Base: type) -> Any:
        namespace: dict[str, Any] = {
            "__registry_key__": "impl",
            "__annotations__": {"value": annotation},
        }
        if default is not dataclasses.MISSING:
            namespace["value"] = default
        return dataclasses.dataclass(type("Impl", (Base,), namespace))

    return define


def _bridged_snapshot(annotation: str, default: Any = None) -> _Snapshot:
    """Snapshot of a layer holding one bridged BaseModel class ``Impl`` (field ``value``)."""
    with _isolated():
        reg = create_registrar("fp_layer", _Proto, discriminator_field="kind")
        bridge = reg.bridge(BaseModel)
        type(
            "Impl",
            (bridge,),
            {
                "__registry_key__": "impl",
                "__annotations__": {"value": annotation},
                "value": default,
                "run": lambda self: None,
            },
        )
        return _take(reg)


def _init_x(self, x: int = 1) -> None:
    self.x = x


def _init_x_two(self, x: int = 2) -> None:
    self.x = x


def _init_x_float(self, x: float = 1) -> None:
    self.x = x


def _init_x_y(self, x: int = 1, y: str = "") -> None:
    self.x = x


def _init_gt0(self, x: Annotated[int, Field(gt=0)] = 1) -> None:
    self.x = x


def _init_gt1(self, x: Annotated[int, Field(gt=1)] = 1) -> None:
    self.x = x


def _init_limits_int(self, limits: dict[str, int] | None = None) -> None:
    self.limits = limits


def _init_limits_float(self, limits: dict[str, float] | None = None) -> None:
    self.limits = limits


def _init_items_str(self, items: tuple[str, ...] = ()) -> None:
    self.items = items


def _init_items_int(self, items: tuple[int, ...] = ()) -> None:
    self.items = items


def _init_peer(self, peer: str) -> None:
    self.peer = peer


def _init_described_one(self, x: int = 1) -> None:
    """Init.

    Args:
        x: The first description.
    """


def _init_described_two(self, x: int = 1) -> None:
    """Init.

    Args:
        x: A different description.
    """


class _Opaque:
    """A type pydantic cannot build a schema for (the field degrades to Any)."""


def _init_opaque(self, handle: _Opaque | None = None) -> None:
    self.handle = handle


def _make_one() -> list[int]:
    return [1]


def _make_two() -> list[int]:
    return [2]


# ===================================================================
# Definition and format
# ===================================================================


def _expected(document: dict[str, Any]) -> str:
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


class TestDefinition:
    """The fingerprint is the hash of the generated outputs and nothing else."""

    def test_is_the_hash_of_version_settings_source_and_schema(self) -> None:
        with _isolated():
            target = create_registrar("fp_def_target", _Proto, discriminator_field="kind")

            class TargetBase(metaclass=target.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("One", (TargetBase,), {"__registry_key__": "one"})
            reg = create_registrar("fp_def", _Proto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type(
                "Impl",
                (Base,),
                {"__init__": _init_peer, "__wiring__": {"peer": "fp_def_target"}},
            )
            result = build_layer_config(reg)
            document = {
                "conscribe": conscribe.__version__,
                "layer": {
                    "name": "fp_def",
                    "discriminator_field": "kind",
                    "discriminator_fields": None,
                    "key_separator": "",
                    "inline_only": False,
                },
                "source": generate_layer_config_source(result),
                "schema": json.dumps(generate_layer_json_schema(result), sort_keys=True),
            }
            assert compute_registry_fingerprint(reg) == _expected(document)

    def test_a_layer_that_cannot_be_built_hashes_the_error_marker(self) -> None:
        with _isolated():
            reg = create_registrar(
                "fp_def_empty", _Proto, discriminator_fields=["family", "model"], key_separator="."
            )
            from conscribe.exceptions import EmptyLayerError

            marker = f"error:EmptyLayerError:{EmptyLayerError(['fp_def_empty'])}"
            document = {
                "conscribe": conscribe.__version__,
                "layer": {
                    "name": "fp_def_empty",
                    "discriminator_field": "",
                    "discriminator_fields": ["family", "model"],
                    "key_separator": ".",
                    "inline_only": False,
                },
                "source": marker,
                "schema": marker,
            }
            assert compute_registry_fingerprint(reg) == _expected(document)

    def test_format_is_16_lowercase_hex_and_deterministic(self) -> None:
        first = _snapshot(_impl(__init__=_init_x))
        assert _HEX16.fullmatch(first.fingerprint)
        assert _snapshot(_impl(__init__=_init_x)) == first

    def test_computing_twice_gives_the_same_value(self) -> None:
        with _isolated():
            reg = create_registrar("fp_twice", _Proto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("Impl", (Base,), {"__init__": _init_x})
            assert compute_registry_fingerprint(reg) == compute_registry_fingerprint(reg)


# ===================================================================
# The fingerprint follows the outputs
# ===================================================================


def _nested_define(Base: type) -> None:
    type("Small", (Base,), {"__registry_key__": "gpt.small", "__init__": _init_x})


def _keys(*keys: str) -> Callable[[type], Any]:
    def define(Base: type) -> None:
        for key in keys:
            type(f"Impl_{key}", (Base,), {"__registry_key__": key, "__init__": _init_x})

    return define


def _wired(entry: Any) -> Callable[[type], Any]:
    return _impl(__init__=_init_peer, __wiring__={"peer": entry})


#: ``(before, after)`` definitions whose generated outputs differ.
_OUTPUT_CHANGES: dict[str, tuple[Callable[[type], Any], Callable[[type], Any]]] = {
    "param type": (_impl(__init__=_init_x), _impl(__init__=_init_x_float)),
    "param default": (_impl(__init__=_init_x), _impl(__init__=_init_x_two)),
    "param added": (_impl(__init__=_init_x), _impl(__init__=_init_x_y)),
    "param description": (
        _impl(__init__=_init_described_one),
        _impl(__init__=_init_described_two),
    ),
    "Field constraint": (_impl(__init__=_init_gt0), _impl(__init__=_init_gt1)),
    "dict value type of an init param": (
        _impl(__init__=_init_limits_int),
        _impl(__init__=_init_limits_float),
    ),
    "tuple item type of an init param": (
        _impl(__init__=_init_items_str),
        _impl(__init__=_init_items_int),
    ),
    "dict value type of a dataclass field": (
        _dataclass_impl("dict[str, int] | None", None),
        _dataclass_impl("dict[str, float] | None", None),
    ),
    "list item type of a dataclass field": (
        _dataclass_impl("list[str] | None", None),
        _dataclass_impl("list[bytes] | None", None),
    ),
    "pure factory value": (
        _dataclass_impl("list[int]", dataclasses.field(default_factory=_make_one)),
        _dataclass_impl("list[int]", dataclasses.field(default_factory=_make_two)),
    ),
    "wiring keys": (_wired(("fp_target", ["a"])), _wired(("fp_target", ["a", "b"]))),
    "wiring registry vs literal list": (_wired("fp_target"), _wired(["a", "b"])),
    "wiring default key": (
        _wired({"registry": "fp_target", "default": "a"}),
        _wired({"registry": "fp_target", "default": "b"}),
    ),
    "wired field gains a default key": (
        _wired({"registry": "fp_target"}),
        _wired({"registry": "fp_target", "default": "a"}),
    ),
    "registered key added": (_keys("alpha"), _keys("alpha", "beta")),
    "registered key renamed": (_keys("alpha"), _keys("omega")),
}


class TestFollowsTheOutputs:
    """Every edit that reaches the generated files changes the fingerprint."""

    @pytest.mark.parametrize("case", list(_OUTPUT_CHANGES))
    def test_output_change_changes_the_fingerprint(self, case: str) -> None:
        before, after = (_snapshot(define) for define in _OUTPUT_CHANGES[case])
        assert isinstance(before.outputs, tuple), before.outputs
        assert isinstance(after.outputs, tuple), after.outputs
        assert before.outputs != after.outputs  # the edit reaches the generated files
        assert before.fingerprint != after.fingerprint

    def test_basemodel_field_type(self) -> None:
        before = _bridged_snapshot("dict[str, int] | None")
        after = _bridged_snapshot("dict[str, float] | None")
        assert isinstance(before.outputs, tuple), before.outputs
        assert before.outputs != after.outputs
        assert before.fingerprint != after.fingerprint

    def test_basemodel_field_added(self) -> None:
        before = _bridged_snapshot("str", "x")
        with _isolated():
            reg = create_registrar("fp_layer", _Proto, discriminator_field="kind")
            bridge = reg.bridge(BaseModel)
            type(
                "Impl",
                (bridge,),
                {
                    "__registry_key__": "impl",
                    "__annotations__": {"value": "str", "extra_field": "str"},
                    "value": "x",
                    "extra_field": "new",
                    "run": lambda self: None,
                },
            )
            after = _take(reg)
        assert before.outputs != after.outputs
        assert before.fingerprint != after.fingerprint

    def test_basemodel_field_description(self) -> None:
        before = _bridged_snapshot("str", Field("x", description="first"))
        after = _bridged_snapshot("str", Field("x", description="second"))
        assert before.outputs != after.outputs
        assert before.fingerprint != after.fingerprint

    def test_target_registry_gaining_a_key_widens_the_literal(self) -> None:
        with _isolated():
            target = create_registrar("fp_grow", _Proto, discriminator_field="kind")

            class TargetBase(metaclass=target.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("One", (TargetBase,), {"__registry_key__": "one"})
            reg = create_registrar("fp_grow_owner", _Proto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("Owner", (Base,), {"__init__": _init_peer, "__wiring__": {"peer": "fp_grow"}})
            before = _take(reg)
            type("Two", (TargetBase,), {"__registry_key__": "two"})
            after = _take(reg)
        assert before.outputs != after.outputs
        assert before.fingerprint != after.fingerprint

    def test_unregister_changes_the_fingerprint(self) -> None:
        with _isolated():
            reg = create_registrar("fp_unreg", _Proto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("A", (Base,), {"__registry_key__": "a", "__init__": _init_x})
            type("B", (Base,), {"__registry_key__": "b", "__init__": _init_x_float})
            before = compute_registry_fingerprint(reg)
            reg.unregister("b")
            assert compute_registry_fingerprint(reg) != before

    def test_parent_signature_reached_through_kwargs(self) -> None:
        """A ``**kwargs`` chain forwards to a parent: its signature is in the config."""

        def define(parent_init: Callable[..., None]) -> Callable[[type], Any]:
            def body(Base: type) -> None:
                parent = type("Parent", (), {"__init__": parent_init})

                def child_init(self, *, model_id: str, **kwargs: Any) -> None:
                    super(child, self).__init__(**kwargs)

                child = type(
                    "Child",
                    (Base, parent),
                    {"__registry_key__": "child", "__init__": child_init},
                )

            return body

        before = _snapshot(define(_parent_init_a))
        after = _snapshot(define(_parent_init_ab))
        assert before.outputs != after.outputs
        assert before.fingerprint != after.fingerprint

    def test_registration_order_matters_exactly_when_the_outputs_do(self) -> None:
        """The union lists its members in some order; the fingerprint follows it."""
        alpha_beta = _snapshot(_keys("alpha", "beta"))
        beta_alpha = _snapshot(_keys("beta", "alpha"))
        assert (alpha_beta.fingerprint == beta_alpha.fingerprint) == (
            alpha_beta.outputs == beta_alpha.outputs
        )


def _parent_init_a(self, *, param_a: int = 10) -> None:
    self.param_a = param_a


def _parent_init_ab(self, *, param_a: int = 10, param_b: str = "new") -> None:
    self.param_a = param_a


# ===================================================================
# Registrar settings and the conscribe version
# ===================================================================


class TestSettingsAndVersion:
    """Inputs hashed explicitly: composed-only settings and the version."""

    def test_inline_only_changes_only_the_fingerprint(self) -> None:
        define = _impl(__init__=_init_x)
        plain = _snapshot(define)
        inline_only = _snapshot(define, inline_only=True)
        assert plain.outputs == inline_only.outputs  # not visible per layer
        assert plain.fingerprint != inline_only.fingerprint

    @pytest.mark.parametrize(
        "options",
        [
            {"discriminator_field": "type"},
            {"key_separator": "."},
        ],
        ids=["discriminator_field", "key_separator"],
    )
    def test_flat_settings(self, options: dict[str, Any]) -> None:
        define = _impl(__init__=_init_x)
        assert _snapshot(define).fingerprint != _snapshot(define, **options).fingerprint

    def test_discriminator_fields(self) -> None:
        nested = {"key_separator": "."}
        first = _snapshot(_nested_define, discriminator_fields=["family", "model"], **nested)
        second = _snapshot(_nested_define, discriminator_fields=["vendor", "model"], **nested)
        assert isinstance(first.outputs, tuple), first.outputs
        assert first.fingerprint != second.fingerprint

    def test_conscribe_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        define = _impl(__init__=_init_x)
        before = _snapshot(define)
        monkeypatch.setattr(conscribe, "__version__", "99.0.0")
        after = _snapshot(define)
        assert before.fingerprint != after.fingerprint


# ===================================================================
# Edits that do not reach the outputs
# ===================================================================


@pytest.fixture
def load_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., contextlib.AbstractContextManager[Any]]:
    """Import a throwaway module written to ``tmp_path`` (fresh every time).

    The module is imported inside ``isolated_registries()`` and dropped from
    ``sys.modules`` before the block ends, so everything it registered is
    rolled back.  No bytecode is written: a rewritten file is always re-read.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    @contextlib.contextmanager
    def load(source: str, name: str = "fp_probe_module") -> Iterator[Any]:
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
        importlib.invalidate_caches()
        sys.modules.pop(name, None)
        with _isolated():
            try:
                yield importlib.import_module(name)
            finally:
                sys.modules.pop(name, None)

    return load


_EDITABLE = '''\
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class P(Protocol):
    def run(self) -> None: ...


R = create_registrar("fpm_layer", P, discriminator_field="kind")

_BASE = ["x"]


class Base(metaclass=R.Meta):
    __abstract__ = True

    def run(self) -> None: ...


def make_tags():
    {comment}
    return ["a", "b"]


def make_items():
    # Reads a module global: impure, so the generated file imports it by name.
    return list(_BASE) + [{factory_value!r}]


class Impl(Base):
    """{summary}

    Args:
        retries: How often to retry.
    """

    __registry_key__ = "impl"

    def __init__(self, retries: int = 3, tags: list = None):  {comment}
        pass


@dataclass
class Data(Base):
    """{summary}"""

    __registry_key__ = "data"
    tags: list = field(default_factory=make_tags)
    items: list = field(default_factory=make_items)
'''

_EDITABLE_DEFAULTS = {"comment": "# v1", "summary": "Summary.", "factory_value": 1}


class TestIgnoresEditsThatDoNotReachTheOutputs:
    @pytest.mark.parametrize(
        "edit",
        [
            {"comment": "# a reworded comment, longer than before"},
            {"summary": "A reworded summary; the Args section is unchanged."},
            {"factory_value": 2},
        ],
        ids=["comment", "docstring-summary", "body-of-a-factory-imported-by-name"],
    )
    def test_same_fingerprint(self, load_module, edit: dict[str, Any]) -> None:
        snapshots = []
        for overrides in ({}, edit):
            source = _EDITABLE.format(**{**_EDITABLE_DEFAULTS, **overrides})
            with load_module(source) as module:
                snapshots.append(_take(module.R))
        before, after = snapshots
        assert isinstance(before.outputs, tuple), before.outputs
        assert before.outputs == after.outputs
        assert before.fingerprint == after.fingerprint

    def test_the_editable_module_is_sensitive_to_a_real_change(self, load_module) -> None:
        """Control: the same module with a changed default does change."""
        snapshots = []
        for retries in ("3", "4"):
            source = _EDITABLE.format(**_EDITABLE_DEFAULTS).replace(
                "retries: int = 3", f"retries: int = {retries}"
            )
            with load_module(source) as module:
                snapshots.append(_take(module.R))
        assert snapshots[0].fingerprint != snapshots[1].fingerprint

    def test_renamed_class_with_the_same_key(self) -> None:
        def renamed(Base: type) -> None:
            type("Renamed", (Base,), {"__registry_key__": "impl", "__init__": _init_x})

        original = _snapshot(_impl(__init__=_init_x))
        assert _snapshot(renamed) == original


_ENUM_MODULE = '''\
import enum
from typing import Protocol, runtime_checkable

from conscribe import create_registrar


@runtime_checkable
class P(Protocol):
    def run(self) -> None: ...


R = create_registrar("fpm_enum", P, discriminator_field="kind")


class Color(enum.Enum):
{members}


class Base(metaclass=R.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class Impl(Base):
    def __init__(self, color: Color = Color.RED):
        pass
'''

_ALIAS_MODULE = '''\
from typing import Protocol, runtime_checkable

from conscribe import create_registrar

type Pair = tuple[{item}, {item}]


@runtime_checkable
class P(Protocol):
    def run(self) -> None: ...


R = create_registrar("fpm_alias", P, discriminator_field="kind")


class Base(metaclass=R.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class Impl(Base):
    def __init__(self, pair: Pair | None = None):
        pass
'''


class TestTypesDefinedElsewhere:
    """Types the generated files reference: their structure reaches the outputs."""

    def test_enum_member_added(self, load_module) -> None:
        snapshots = []
        for members in ('    RED = "red"', '    RED = "red"\n    BLUE = "blue"'):
            with load_module(_ENUM_MODULE.format(members=members)) as module:
                snapshots.append(_take(module.R))
        assert snapshots[0].outputs != snapshots[1].outputs
        assert snapshots[0].fingerprint != snapshots[1].fingerprint

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type statements need 3.12")
    def test_pep695_alias_value(self, load_module) -> None:
        snapshots = []
        for item in ("int", "str"):
            with load_module(_ALIAS_MODULE.format(item=item)) as module:
                snapshots.append(_take(module.R))
        if snapshots[0].outputs == snapshots[1].outputs:
            # pydantic 2.7 cannot build a schema for a PEP 695 alias: the field
            # degrades to Any in both versions, so the outputs -- and therefore
            # the fingerprints -- agree.  That is the contract, not a miss.
            assert snapshots[0].fingerprint == snapshots[1].fingerprint
            pytest.skip("this pydantic version degrades PEP 695 aliases to Any")
        assert snapshots[0].fingerprint != snapshots[1].fingerprint


# ===================================================================
# Never raises
# ===================================================================


class _BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


class _Exploding:
    """A class attribute that raises when read."""

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        raise RuntimeError("boom")


class _BadStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str")


_SELF_REFERENCING: list[Any] = []
_SELF_REFERENCING.append(_SELF_REFERENCING)


def _init_undefined(self, x=1) -> None:
    self.x = x


# A typo in an annotation (set here so that type checkers do not flag it).
_init_undefined.__annotations__ = {"x": "Undefinedd", "return": "None"}


def _init_bad_repr(self, x=_BadRepr()) -> None: ...


def _init_self_referencing(self, x=_SELF_REFERENCING) -> None: ...


def _init_unrenderable(self, x: object = object()) -> None: ...


def _broken(**attributes: Any) -> Callable[[type], Any]:
    """``define`` creating a valid ``Impl``, then breaking it with *attributes*.

    Set after class creation, so that class-creation checks cannot reject
    the declaration first: the fingerprint must cope on its own.
    """

    def define(Base: type) -> None:
        cls = type("Impl", (Base,), {"__registry_key__": "impl", "__init__": _init_x})
        for name, value in attributes.items():
            setattr(cls, name, value)

    return define


def _nothing(Base: type) -> None:
    """``define`` registering no class at all (an empty layer)."""


_BROKEN: dict[str, Callable[[type], Any]] = {
    "wiring-not-a-dict": _broken(__wiring__=["not", "a", "dict"]),
    "mode3-mixed-int-str": _broken(__wiring__={"peer": [1, "a"]}),
    "wiring-set": _broken(__wiring__={"peer": {"one", "two", "three"}}),
    "unknown-registry": _broken(__wiring__={"peer": ("fp_nowhere", ["x"])}),
    "unknown-keys": _broken(__wiring__={"peer": ("fp_target", ["zz"], ["yy"])}),
    "unknown-dict-field": _broken(__wiring__={"peer": {"registry": "fp_target", "bogus": 1}}),
    "bad-default": _broken(__wiring__={"peer": {"registry": "fp_target", "default": "zz"}}),
    "non-str-entry-name": _broken(__wiring__={1: "fp_target"}),
    "declare-only-with-a-slot": _broken(
        __wiring__={"x": {"registry": "fp_target", "declare_only": True}}
    ),
    "raising-knob": _broken(__config_schema__=_Exploding()),
    "undefined-annotation": _impl(__init__=_init_undefined),
    "raising-repr-default": _impl(__init__=_init_bad_repr),
    "self-referencing-default": _impl(__init__=_init_self_referencing),
    "unrenderable-default": _impl(__init__=_init_unrenderable),
    "empty-layer": _nothing,
}


class TestNeverRaises:
    """Broken declarations still produce a deterministic fingerprint."""

    @pytest.mark.parametrize("case", list(_BROKEN))
    def test_broken_declaration_fingerprints(self, case: str) -> None:
        first = _snapshot(_BROKEN[case])
        assert _HEX16.fullmatch(first.fingerprint)
        assert _snapshot(_BROKEN[case]).fingerprint == first.fingerprint
        assert first.fingerprint != _snapshot(_impl(__init__=_init_x)).fingerprint

    def test_broken_states_are_told_apart(self) -> None:
        """Each broken state has its own marker (the messages differ)."""
        fingerprints = {case: _snapshot(define).fingerprint for case, define in _BROKEN.items()}
        assert len(set(fingerprints.values())) == len(fingerprints), fingerprints

    def test_marker_does_not_depend_on_object_addresses(self) -> None:
        def define(key: object) -> Callable[[type], Any]:
            return _broken(__wiring__={"peer": ("fp_target", [key])})

        first, second = object(), object()
        assert repr(first) != repr(second)
        assert _snapshot(define(first)).fingerprint == _snapshot(define(second)).fingerprint

    def test_registrar_without_discriminator(self) -> None:
        with _isolated():
            plain = create_registrar("fp_plain", _Proto)

            class Base(metaclass=plain.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("Impl", (Base,), {"__init__": _init_x})
            assert _HEX16.fullmatch(compute_registry_fingerprint(plain))

    def test_settings_that_are_not_plain_values(self) -> None:
        """A setting is hashed by type and repr, object addresses removed."""

        def registrar_like(discriminator: object) -> type:
            return type("Fake", (), {"discriminator_field": discriminator})

        first = compute_registry_fingerprint(registrar_like(object()))
        assert first == compute_registry_fingerprint(registrar_like(object()))
        assert first != compute_registry_fingerprint(registrar_like("kind"))

    def test_object_that_is_not_a_registrar(self) -> None:
        class Unreadable:
            discriminator_field = "kind"

            @staticmethod
            def get_all() -> dict[str, type]:
                raise RuntimeError("registry unavailable")

        assert _HEX16.fullmatch(compute_registry_fingerprint(Unreadable))

    def test_missing_target_registry_then_present(self) -> None:
        """A wiring target that appears later changes the fingerprint."""
        with _isolated():
            reg = create_registrar("fp_early", _Proto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            owner = type("Owner", (Base,), {"__init__": _init_peer})
            setattr(owner, "__wiring__", {"peer": "fp_late"})
            before = compute_registry_fingerprint(reg)

            late = create_registrar("fp_late", _Proto, discriminator_field="kind")

            class LateBase(metaclass=late.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("Late", (LateBase,), {"__registry_key__": "late"})
            assert compute_registry_fingerprint(reg) != before


class TestErrorMarker:
    def test_format_and_address_stripping(self) -> None:
        exc = RuntimeError("cannot render <X object at 0x10f3a2b40> or 0xDEADbeef")
        assert _error_marker(exc) == "error:RuntimeError:cannot render <X object at 0x...> or 0x..."

    def test_unprintable_message(self) -> None:
        assert _error_marker(_BadStr()) == "error:_BadStr:<message unavailable>"


class TestWarnings:
    def test_warnings_are_suppressed_and_filters_restored(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filters_before = list(warnings.filters)
            snapshot = _snapshot(_impl(__init__=_init_opaque))
            assert list(warnings.filters) == filters_before
        assert isinstance(snapshot.outputs, tuple), snapshot.outputs
        assert "degraded" in snapshot.outputs[0].lower()  # the layer does degrade
        assert caught == []

    def test_conscribe_warnings_as_errors_do_not_change_the_value(self) -> None:
        normal = _snapshot(_impl(__init__=_init_opaque))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConscribeWarning)
            strict = _snapshot(_impl(__init__=_init_opaque))
        assert strict.fingerprint == normal.fingerprint


# ===================================================================
# Determinism across processes and Python versions
# ===================================================================

_PROBE_MODULE = '''\
import dataclasses
import enum
from typing import Annotated, Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from conscribe import create_registrar


@runtime_checkable
class P(Protocol):
    def run(self) -> None: ...


Target = create_registrar("probe_target", P, discriminator_field="kind", key_separator=".")
Reg = create_registrar("probe_layer", P, discriminator_field="kind")
Nested = create_registrar(
    "probe_nested", P, discriminator_fields=["family", "model"], key_separator="."
)
Archive = create_registrar("probe_archive", P, discriminator_field="kind", inline_only=True)
Models = create_registrar(
    "probe_models", P, discriminator_field="kind", base_metaclass=type(BaseModel)
)


class TargetBase(metaclass=Target.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class BrowserClick(TargetBase):
    __registry_key__ = "browser.click"


class DesktopClick(TargetBase):
    __registry_key__ = "desktop.click"


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Sub(BaseModel):
    tags: set[str] = {"x", "y", "z", "w"}
    depth: int = Field(3, ge=0)


SENTINEL = object()


def make_tags() -> set[str]:
    return {"alpha", "beta", "gamma", "delta"}


class Base(metaclass=Reg.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class Impl(Base):
    """Impl.

    Args:
        x: An x.
    """

    __registry_key__ = "impl"
    __wiring__ = {"action": ("probe_target", ["click"]), "lit": ["b", "a"]}
    env: Optional[Any]

    def __init__(
        self,
        action: str,
        x: Annotated[int, Field(gt=0)] = 1,
        tags: frozenset = frozenset({"q", "r", "s", "t"}),
        sentinel: object = SENTINEL,
        color: Color = Color.RED,
        sub: Sub = Sub(),
        mapping: dict = {"k": {"v1", "v2", "v3"}},
        limits: dict[str, int] | None = None,
    ):
        pass


@dataclasses.dataclass
class Data(Base):
    __registry_key__ = "data"
    items: set = dataclasses.field(default_factory=make_tags)
    plain: frozenset = frozenset({"n", "o", "p", "m"})


class NestedBase(metaclass=Nested.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class Small(NestedBase):
    __registry_key__ = "gpt.small"

    def __init__(self, temperature: float = 0.5, stop: frozenset = frozenset({"a", "b", "c"})):
        pass


class ArchiveBase(metaclass=Archive.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class Disk(ArchiveBase):
    __registry_key__ = "disk"

    def __init__(self, path: str = "/tmp/archive"):
        pass


class ModelBase(BaseModel, metaclass=Models.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class ModelOwner(ModelBase):
    retries: int = Field(default=2, ge=0, le=10, description="How many retries")
    labels: set[str] = {"one", "two", "three"}
'''

_PROBE_LAYERS = ("Target", "Reg", "Nested", "Archive", "Models")

_PROBE_SCRIPT = """\
import json, sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("error")  # suppression must hold even when warnings are errors
import fp_probe_module
from conscribe import compute_registry_fingerprint
print(json.dumps({
    name: compute_registry_fingerprint(getattr(fp_probe_module, name))
    for name in sys.argv[2].split(",")
}, sort_keys=True))
"""

_PYTHONS_ENV = "CONSCRIBE_FINGERPRINT_PYTHONS"


def _package_root() -> str:
    return str(Path(conscribe.__file__).resolve().parent.parent)


def _probe(command: list[str], probe_dir: Path, *, seed: str = "random") -> dict[str, str]:
    """Fingerprint the layers of ``probe_dir/fp_probe_module.py`` with *command* (an interpreter)."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [_package_root(), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    completed = subprocess.run(
        [*command, "-c", _PROBE_SCRIPT, str(probe_dir), ",".join(_PROBE_LAYERS)],
        env=env,
        cwd=probe_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


class TestAcrossProcesses:
    def test_identical_with_different_hash_seeds(self, tmp_path: Path, load_module) -> None:
        """Set-valued defaults (also inside a model-instance default) are written sorted."""
        (tmp_path / "fp_probe_module.py").write_text(_PROBE_MODULE, encoding="utf-8")
        results = [
            _probe([sys.executable], tmp_path, seed=seed) for seed in ("0", "1", "4242", "random")
        ]
        assert all(result == results[0] for result in results), results
        assert set(results[0]) == set(_PROBE_LAYERS)
        assert all(_HEX16.fullmatch(value) for value in results[0].values())
        assert len(set(results[0].values())) == len(_PROBE_LAYERS)

        # ... and identical to the value computed in this process.
        with load_module(_PROBE_MODULE) as module:
            here = {name: compute_registry_fingerprint(getattr(module, name)) for name in _PROBE_LAYERS}
        assert here == results[0]

    @pytest.mark.skipif(
        not os.environ.get(_PYTHONS_ENV),
        reason=f"set {_PYTHONS_ENV}='3.10 3.12 3.14' to compare Python versions (needs uv)",
    )
    def test_identical_across_python_versions(self, tmp_path: Path) -> None:
        """Opt-in: the same code gives the same fingerprints on other Pythons.

        Each interpreter runs through ``uv run --isolated`` with this
        environment's pydantic (and docstring-parser) versions, so only the
        Python version differs.
        """
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv is not on PATH")
        (tmp_path / "fp_probe_module.py").write_text(_PROBE_MODULE, encoding="utf-8")
        requirements = [f"pydantic=={pydantic.VERSION}"]
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            requirements.append(f"docstring-parser=={importlib.metadata.version('docstring-parser')}")
        expected = _probe([sys.executable], tmp_path)
        for version in os.environ[_PYTHONS_ENV].replace(",", " ").split():
            command = [uv, "run", "--quiet", "--isolated", "--no-project", "--python", version]
            for requirement in requirements:
                command += ["--with", requirement]
            assert _probe([*command, "python"], tmp_path) == expected, version


# ===================================================================
# Cache files
# ===================================================================


class TestLoadCachedFingerprint:
    """Tests for load_cached_fingerprint()."""

    def test_returns_stored_fingerprint_for_known_layer(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text(json.dumps({"llm": "abcdef0123456789", "agent": "9876543210fedcba"}))
        assert load_cached_fingerprint(fp_file, "llm") == "abcdef0123456789"

    def test_returns_none_if_file_does_not_exist(self, tmp_path: Path) -> None:
        assert load_cached_fingerprint(tmp_path / ".registry_fingerprint", "llm") is None

    def test_returns_none_if_layer_name_not_in_json(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text(json.dumps({"agent": "9876543210fedcba"}))
        assert load_cached_fingerprint(fp_file, "llm") is None

    @pytest.mark.parametrize(
        "content",
        [b"this is not valid json {{{", b"", b"\xff\xfe not utf-8", b'["not", "a", "dict"]'],
        ids=["corrupt", "empty", "not-utf8", "not-a-dict"],
    )
    def test_returns_none_for_unusable_files(self, tmp_path: Path, content: bytes) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_bytes(content)
        assert load_cached_fingerprint(fp_file, "llm") is None

    def test_returns_none_for_a_value_that_is_not_a_string(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text(json.dumps({"llm": 12345, "agent": None}))
        assert load_cached_fingerprint(fp_file, "llm") is None
        assert load_cached_fingerprint(fp_file, "agent") is None


class TestSaveFingerprint:
    """Tests for save_fingerprint()."""

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        save_fingerprint(fp_file, "llm", "abcdef0123456789")
        assert json.loads(fp_file.read_text()) == {"llm": "abcdef0123456789"}

    def test_preserves_other_layers_entries(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text(json.dumps({"agent": "9876543210fedcba"}))
        save_fingerprint(fp_file, "llm", "abcdef0123456789")
        assert json.loads(fp_file.read_text()) == {
            "agent": "9876543210fedcba",
            "llm": "abcdef0123456789",
        }

    def test_overwrites_existing_fingerprint_for_same_layer(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text(json.dumps({"llm": "0000000000000000", "agent": "1111111111111111"}))
        save_fingerprint(fp_file, "llm", "aaaaaaaaaaaaaaaa")
        assert json.loads(fp_file.read_text()) == {
            "llm": "aaaaaaaaaaaaaaaa",
            "agent": "1111111111111111",
        }

    def test_multiple_saves_each_update_correctly(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        save_fingerprint(fp_file, "llm", "aaaa000000000000")
        save_fingerprint(fp_file, "agent", "bbbb000000000000")
        save_fingerprint(fp_file, "provider", "cccc000000000000")
        assert json.loads(fp_file.read_text()) == {
            "llm": "aaaa000000000000",
            "agent": "bbbb000000000000",
            "provider": "cccc000000000000",
        }

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        fp_file = tmp_path / "nested" / "dir" / ".registry_fingerprint"
        save_fingerprint(fp_file, "llm", "abcdef0123456789")
        assert json.loads(fp_file.read_text())["llm"] == "abcdef0123456789"

    @pytest.mark.parametrize(
        "content",
        [b"{{{{ not json !!!!", b"\xff\xfe not utf-8", b"[1, 2, 3]"],
        ids=["corrupt", "not-utf8", "not-a-dict"],
    )
    def test_replaces_an_unusable_file(self, tmp_path: Path, content: bytes) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_bytes(content)
        save_fingerprint(fp_file, "llm", "def456")
        assert json.loads(fp_file.read_text()) == {"llm": "def456"}


class TestEndToEndRoundtrip:
    """Compute a real fingerprint, save it, load it back, detect staleness."""

    def test_compute_save_load_and_detect_a_change(self, tmp_path: Path) -> None:
        fp_file = tmp_path / ".registry_fingerprint"
        with _isolated():
            reg = create_registrar("fp_roundtrip", _Proto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            type("Alpha", (Base,), {"__init__": _init_x})
            fingerprint = compute_registry_fingerprint(reg)
            save_fingerprint(fp_file, "fp_roundtrip", fingerprint)
            assert load_cached_fingerprint(fp_file, "fp_roundtrip") == fingerprint

            type("Beta", (Base,), {"__init__": _init_x_float})
            assert load_cached_fingerprint(fp_file, "fp_roundtrip") != compute_registry_fingerprint(reg)
