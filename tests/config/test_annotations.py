"""Tests for conscribe.config._annotations -- the annotation engine.

Covers PEP 649-safe reading, owner-aware resolution (D3: af defect 3),
TYPE_CHECKING-only names (degrade, never imported), typos and metadata
failures (ConfigAnnotationError), function-local classes, the None/Optional
normalization (Python 3.10 implicit Optional), and the shape / ClassVar /
KW_ONLY / InitVar predicates.

Multi-module scenarios are written as real packages under ``tmp_path``:
classes defined inside test functions would share one namespace and hide
exactly the bugs under test.
"""
from __future__ import annotations

import collections
import dataclasses
import importlib
import inspect
import sys
import textwrap
import typing
import uuid
import warnings
from collections.abc import Callable, Iterator
from datetime import date
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Union,
    get_args,
    get_origin,
)

import pytest
from pydantic import Field
from pydantic.fields import FieldInfo

from conscribe.config import _annotations as A
from conscribe.config.mro import walk_init_chain
from conscribe.exceptions import (
    ConfigAnnotationError,
    DegradationWarning,
    ExtractionWarning,
)

PY314 = sys.version_info >= (3, 14)


# ---------------------------------------------------------------------------
# Package factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_package(tmp_path, monkeypatch) -> Iterator[Callable[[dict[str, str]], str]]:
    """Write ``{module: source}`` as a fresh package; return its name."""
    created: list[str] = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(files: dict[str, str]) -> str:
        name = f"cs_ann_{uuid.uuid4().hex[:10]}"
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("")
        for module, source in files.items():
            (root / f"{module}.py").write_text(textwrap.dedent(source))
        created.append(name)
        importlib.invalidate_caches()
        return name

    yield make
    for name in created:
        for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[module]


def _import(package: str, module: str) -> Any:
    return importlib.import_module(f"{package}.{module}")


def _resolve_own(cls: type) -> dict[str, A.ResolvedAnnotation]:
    """Resolve every annotation *cls* declares in its own body."""
    return {name: A.resolve_class_annotation(cls, name) for name in A.own_annotations(cls)}


def _field_info(annotation: Any) -> FieldInfo | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, FieldInfo):
                return meta
    return None


# ---------------------------------------------------------------------------
# PEP 649-safe reading
# ---------------------------------------------------------------------------


class TestOwnAnnotations:
    def test_returns_only_own_annotations(self) -> None:
        class Base:
            a: int

        class Child(Base):
            b: str

        assert A.own_annotations(Child) == {"b": "str"}
        assert A.own_annotations(Base) == {"a": "int"}

    def test_returns_a_copy(self) -> None:
        class Holder:
            a: int

        A.own_annotations(Holder)["b"] = str
        assert "b" not in A.own_annotations(Holder)

    def test_class_without_annotations(self) -> None:
        class Empty:
            pass

        assert A.own_annotations(Empty) == {}

    def test_explicit_annotations_dict_in_body(self) -> None:
        class Explicit:
            __annotations__ = {"caps": "dict[str, Any]"}

        assert A.own_annotations(Explicit) == {"caps": "dict[str, Any]"}

    def test_lazily_annotated_module_is_read(self, make_package) -> None:
        """No ``from __future__ import annotations``: on 3.14 the class
        ``__dict__`` has no ``__annotations__`` key at all."""
        pkg = make_package(
            {
                "lazy": """
                from typing import Any, ClassVar

                class Receptors:
                    caps: dict[str, Any]
                    single: Any
                    cv: ClassVar[int] = 3
                """
            }
        )
        cls = _import(pkg, "lazy").Receptors
        annotations = A.own_annotations(cls)
        assert list(annotations) == ["caps", "single", "cv"]
        assert annotations["single"] is Any

    @pytest.mark.skipif(not PY314, reason="unquoted TYPE_CHECKING names only import on 3.14")
    def test_unresolvable_lazy_annotation_does_not_raise(self, make_package) -> None:
        pkg = make_package(
            {
                "lazy314": """
                from typing import TYPE_CHECKING, Optional

                if TYPE_CHECKING:
                    from collections import OrderedDict as Heavy

                class Holder:
                    tc: Optional[Heavy]
                    good: int
                """
            }
        )
        cls = _import(pkg, "lazy314").Holder
        annotations = A.own_annotations(cls)
        assert annotations["good"] is int
        assert "Heavy" in repr(annotations["tc"])


class TestSafeSignature:
    def test_regular_function(self) -> None:
        def f(a: int, b: str = "x") -> None: ...

        sig = A.safe_signature(f)
        assert sig is not None and list(sig.parameters) == ["a", "b"]

    def test_unintrospectable_returns_none(self) -> None:
        assert A.safe_signature(42) is None

    def test_class_based_decorator_without_wrapped_returns_none(self) -> None:
        class Opaque:
            """A callable whose signature inspect cannot read."""

            __signature__ = "not a signature"

            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        assert A.safe_signature(Opaque()) is None

    def test_follows_functools_wraps(self) -> None:
        import functools

        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @deco
        def f(self, knob: int = 3) -> None: ...

        sig = A.safe_signature(f)
        assert sig is not None and list(sig.parameters) == ["self", "knob"]

    @pytest.mark.skipif(not PY314, reason="PEP 649 deferred annotations")
    def test_type_checking_name_does_not_raise_on_314(self, make_package) -> None:
        pkg = make_package(
            {
                "sig314": """
                from typing import TYPE_CHECKING, Optional

                if TYPE_CHECKING:
                    from collections import OrderedDict as Heavy

                class Holder:
                    def __init__(self, a: int, b: Optional[Heavy] = None):
                        pass
                """
            }
        )
        cls = _import(pkg, "sig314").Holder
        with pytest.raises(NameError):
            inspect.signature(cls.__init__)
        sig = A.safe_signature(cls.__init__)
        assert sig is not None and list(sig.parameters) == ["self", "a", "b"]


# ---------------------------------------------------------------------------
# D3: owner-aware resolution
# ---------------------------------------------------------------------------

_REG = """
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class P(Protocol):
    def run(self) -> None: ...

Loop = create_registrar("{registry}", P, discriminator_field="loop_id")
"""

_REACT = """
from __future__ import annotations          # annotations stored as strings
from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from .reg import Loop

@dataclass
class React(metaclass=Loop.Meta):
    __registry_key__ = "react"
    __config_annotated_only__ = True
    max_steps: Annotated[int, Field(ge=1, description="step budget")] = 5
    def run(self): ...
"""

_BRANCH_PLAIN = """
# no PEP 563, and no import of Annotated / Field
from dataclasses import dataclass
from .react import React

@dataclass
class BranchPlain(React):
    __registry_key__ = "branch_plain"
"""


@pytest.fixture
def registry_name() -> Iterator[str]:
    from conscribe.registration.registry import _REGISTRY_INDEX, _deregister

    name = f"cs_ann_loop_{uuid.uuid4().hex[:8]}"
    yield name
    if name in _REGISTRY_INDEX:
        _deregister(name)


class TestDefect3Repro:
    """af feedback defect 3: inherited PEP 563 annotations must resolve in the
    module of the class that declares them, not in the subclass module."""

    def _package(self, make_package, registry_name: str, branch: str = _BRANCH_PLAIN) -> str:
        return make_package(
            {
                "reg": _REG.format(registry=registry_name),
                "react": _REACT,
                "branch_plain": branch,
            }
        )

    def test_inherited_field_resolves_in_declaring_module(
        self, make_package, registry_name
    ) -> None:
        pkg = self._package(make_package, registry_name)
        React = _import(pkg, "react").React
        BranchPlain = _import(pkg, "branch_plain").BranchPlain
        # The bug's precondition: the generated __init__ runs in the subclass module.
        assert BranchPlain.__init__.__globals__["__name__"] == f"{pkg}.branch_plain"

        chain = walk_init_chain(BranchPlain)
        assert chain.kind == "dataclass"
        param = chain.get("max_steps")
        assert param is not None
        assert param.declared_on is React
        assert param.source == "dataclass"

        resolved = param.resolve()
        assert resolved.status == "resolved", resolved.detail
        assert get_args(resolved.value)[0] is int
        info = _field_info(resolved.value)
        assert info is not None and info.description == "step budget"
        assert any(getattr(m, "ge", None) == 1 for m in info.metadata)
        # Same answer as the reference implementation.
        expected = typing.get_type_hints(BranchPlain, include_extras=True)["max_steps"]
        assert get_args(expected)[0] is int

    def test_subclass_with_aliased_field_import(self, make_package, registry_name) -> None:
        """S2: the subclass imports ``Field as PydanticField``; still resolves."""
        branch = """
        from __future__ import annotations
        from dataclasses import dataclass
        from pydantic import Field as PydanticField
        from .react import React

        @dataclass
        class BranchAliased(React):
            __registry_key__ = "branch_aliased"
            extra: int = PydanticField(default=1)
        """
        pkg = self._package(make_package, registry_name, branch)
        cls = _import(pkg, "branch_plain").BranchAliased
        resolved = walk_init_chain(cls).get("max_steps").resolve()
        assert resolved.status == "resolved"
        assert _field_info(resolved.value).description == "step budget"

    def test_grandparent_dataclass_through_kwargs_child(self, make_package, registry_name) -> None:
        """S7: dataclass grandparent -> dataclass parent in a module lacking the
        names -> hand-written ``**kwargs`` child."""
        mid = """
        from dataclasses import dataclass
        from typing import Optional
        from .react import React

        @dataclass
        class Mid(React):
            __registry_key__ = "mid"
            tag: Optional[str] = None
        """
        child = """
        from .mid import Mid

        class Child(Mid):
            __registry_key__ = "child"
            def __init__(self, verbose: bool = False, **kwargs):
                super().__init__(**kwargs)
        """
        pkg = make_package(
            {
                "reg": _REG.format(registry=registry_name),
                "react": _REACT,
                "mid": mid,
                "child": child,
            }
        )
        cls = _import(pkg, "child").Child
        chain = walk_init_chain(cls)
        assert chain.names() == ("verbose", "max_steps", "tag")
        assert chain.fully_resolved
        resolved = {p.name: p.resolve() for p in chain.params}
        assert all(r.status == "resolved" for r in resolved.values())
        assert _field_info(resolved["max_steps"].value).description == "step budget"
        assert resolved["tag"].value == Optional[str]


class TestResolutionNamespaces:
    def test_hand_written_init_resolves_in_its_own_globals(self, make_package) -> None:
        pkg = make_package(
            {
                "base": """
                from __future__ import annotations
                from typing import Annotated
                from pydantic import Field as PydanticField

                class Base:
                    def __init__(self, depth: Annotated[int, PydanticField(ge=0)] = 1):
                        self.depth = depth
                """,
                "child": """
                from .base import Base

                class Child(Base):
                    def __init__(self, name: str = "x", **kwargs):
                        super().__init__(**kwargs)
                """,
            }
        )
        cls = _import(pkg, "child").Child
        param = walk_init_chain(cls).get("depth")
        assert param.source == "init" and param.level == 1
        resolved = param.resolve()
        assert resolved.status == "resolved"
        assert get_args(resolved.value)[0] is int

    def test_module_globals_win_over_class_namespace(self, make_package) -> None:
        """``date: Optional[date] = None`` must not see the class attribute."""
        pkg = make_package(
            {
                "precedence": """
                from __future__ import annotations
                from dataclasses import dataclass
                from datetime import date
                from typing import Optional

                @dataclass
                class Event:
                    date: Optional[date] = None
                """
            }
        )
        cls = _import(pkg, "precedence").Event
        resolved = walk_init_chain(cls).get("date").resolve()
        assert resolved.value == Optional[date]
        assert resolved.value == typing.get_type_hints(cls)["date"]

    def test_class_namespace_is_searched_after_globals(self, make_package) -> None:
        pkg = make_package(
            {
                "aliases": """
                from __future__ import annotations
                from dataclasses import dataclass
                from typing import Literal

                @dataclass
                class Picker:
                    Mode = Literal["fast", "slow"]
                    mode: Mode = "fast"
                """
            }
        )
        cls = _import(pkg, "aliases").Picker
        resolved = walk_init_chain(cls).get("mode").resolve()
        assert resolved.status == "resolved"
        assert get_args(resolved.value) == ("fast", "slow")

    def test_rewritten_module_uses_creation_globals(self, make_package) -> None:
        """A re-export idiom rewrites ``__module__``; the class body's real
        module is still used (via the generated ``__init__``'s globals)."""
        pkg = make_package(
            {
                "impl": """
                from __future__ import annotations
                from dataclasses import dataclass
                from typing import Annotated
                from pydantic import Field

                @dataclass
                class Knobs:
                    depth: Annotated[int, Field(ge=0)] = 1

                Knobs.__module__ = __name__.rpartition(".")[0]
                """
            }
        )
        cls = _import(pkg, "impl").Knobs
        assert cls.__module__ == pkg
        resolved = walk_init_chain(cls).get("depth").resolve()
        assert resolved.status == "resolved", resolved.detail

    def test_nested_string_quotes(self, make_package) -> None:
        pkg = make_package(
            {
                "quotes": """
                from __future__ import annotations
                from typing import Optional

                class Thing:
                    pass

                class Holder:
                    a: "Thing"
                    b: Optional["Thing"]
                    c: list["Thing"]
                """
            }
        )
        module = _import(pkg, "quotes")
        resolved = _resolve_own(module.Holder)
        assert resolved["a"].value is module.Thing
        assert resolved["b"].value == Optional[module.Thing]
        assert resolved["c"].value == list[module.Thing]

    def test_forward_refs_in_evaluated_objects(self, make_package) -> None:
        """No PEP 563: quoted forward references inside typing objects."""
        pkg = make_package(
            {
                "objects": """
                from typing import Optional

                class Holder:
                    a: Optional["Thing"]
                    b: list["Thing"]
                    c: dict[str, "Thing"]

                class Thing:
                    pass
                """
            }
        )
        module = _import(pkg, "objects")
        resolved = _resolve_own(module.Holder)
        assert resolved["a"].value == Optional[module.Thing]
        assert resolved["b"].value == list[module.Thing]
        assert resolved["c"].value == dict[str, module.Thing]

    def test_plain_objects_resolve_without_evaluation(self) -> None:
        resolved = A.resolve_annotation(Optional[int], None, name="x")
        assert resolved.status == "resolved" and resolved.value == Optional[int]
        assert A.resolve_annotation(None, None, name="x").value is type(None)
        empty = A.resolve_annotation(inspect.Parameter.empty, None, name="x")
        assert empty.status == "unannotated" and empty.ok

    def test_annotation_owner_is_most_derived(self) -> None:
        class Base:
            a: int

        class Child(Base):
            a: str
            b: int

        assert A.annotation_owner(Child, "a") is Child
        assert A.annotation_owner(Child, "b") is Child
        assert A.annotation_owner(Base, "a") is Base
        assert A.annotation_owner(Child, "zzz") is None


# ---------------------------------------------------------------------------
# D3: unresolvable names
# ---------------------------------------------------------------------------

_TC_MODULE = """
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Optional
from pydantic import Field

if TYPE_CHECKING:
    from cs_never_imported_pkg_xyz import Heavy
    from .sibling import Session
    Alias = dict[str, int]

@dataclass
class Owner:
    plain: int = 1
    heavy: Optional[Heavy] = None
    inner_quotes: Optional["Heavy"] = None
    double_quotes: "Heavy" = None
    inner_typo: Optional["Hevy"] = None
    described: Annotated[Optional[Heavy], Field(description="the heavy one")] = None
    both: Optional[Session] = None
    alias: Optional[Alias] = None
    meta_tc: Annotated[int, Field(default_factory=Heavy)] = 1
    typo: Optional[Hevy] = None
    two_typos: dict[Hevy, Sesion] = None
    meta_typo: Annotated[int, Feild(ge=1)] = 1
    attr_typo: typing.Lsit[int] = None
    bad_call: Annotated[int, Field(default=1, default_factory=list)] = 1
"""


@pytest.fixture
def tc_owner(make_package):
    pkg = make_package({"tc": _TC_MODULE, "sibling": "class Session: ..."})
    return _import(pkg, "tc").Owner


def _resolve_field(cls: type, name: str) -> A.ResolvedAnnotation:
    return walk_init_chain(cls).get(name).resolve()


class TestTypeCheckingOnly:
    def test_degrades_to_any_with_reason(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "heavy")
        assert resolved.status == "degraded"
        assert resolved.reason == "type_checking_only"
        assert resolved.value is Any
        assert resolved.missing == ("Heavy",)
        assert resolved.error() is None
        assert resolved.shape == "scalar"

    @pytest.mark.parametrize("name", ["inner_quotes", "double_quotes"])
    def test_names_behind_inner_quotes(self, tc_owner, name: str) -> None:
        """``Optional["Heavy"]`` under PEP 563: the outer text evaluates, the
        inner forward reference does not."""
        resolved = _resolve_field(tc_owner, name)
        assert resolved.status == "degraded", resolved.detail
        assert resolved.missing == ("Heavy",)

    def test_typo_behind_inner_quotes(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "inner_typo")
        assert resolved.status == "error" and resolved.missing == ("Hevy",)

    def test_never_imports_the_type_checking_module(self, tc_owner) -> None:
        _resolve_field(tc_owner, "heavy")
        assert "cs_never_imported_pkg_xyz" not in sys.modules

    def test_relative_type_checking_import(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "both")
        assert resolved.status == "degraded" and resolved.missing == ("Session",)

    def test_assignment_under_type_checking(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "alias")
        assert resolved.status == "degraded" and resolved.missing == ("Alias",)

    def test_keeps_field_info(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "described")
        assert resolved.status == "degraded"
        assert get_args(resolved.value)[0] is Any
        assert _field_info(resolved.value).description == "the heavy one"

    def test_warning_category_and_message(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "heavy")
        category, message = resolved.warning("Leaf")
        assert category is DegradationWarning
        assert message.startswith("Leaf.heavy: annotation 'Optional[Heavy]' needs name 'Heavy'")
        assert "if TYPE_CHECKING:" in message
        with pytest.warns(DegradationWarning, match="Leaf.heavy"):
            resolved.warn("Leaf")

    def test_metadata_needing_type_checking_name_is_an_error(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "meta_tc")
        assert resolved.status == "error"
        assert resolved.reason == "metadata"
        assert resolved.missing == ("Heavy",)
        with pytest.raises(ConfigAnnotationError) as info:
            resolved.raise_if_error(tc_owner)
        message = str(info.value)
        assert message.startswith("Owner.meta_tc: cannot evaluate annotation")
        assert "Annotated metadata needs name 'Heavy'" in message
        assert "import it unconditionally" in message


class TestUnresolvedInsideObjects:
    """No PEP 563: quoted forward references nested in evaluated typing objects."""

    @pytest.fixture
    def owner(self, make_package):
        pkg = make_package(
            {
                "objs": """
                from dataclasses import dataclass
                from typing import TYPE_CHECKING, Annotated, Optional
                from pydantic import Field

                if TYPE_CHECKING:
                    from cs_never_imported_objs import Heavy

                @dataclass
                class Owner:
                    heavy: Optional["Heavy"] = None
                    described: Annotated[Optional["Heavy"], Field(description="obj")] = None
                    typo: Optional["Hevy"] = None
                    many: dict[str, "Heavy"] = None
                """
            }
        )
        return _import(pkg, "objs").Owner

    def test_type_checking_name_degrades(self, owner) -> None:
        resolved = _resolve_field(owner, "heavy")
        assert resolved.status == "degraded" and resolved.missing == ("Heavy",)

    def test_metadata_of_evaluated_object_is_kept(self, owner) -> None:
        resolved = _resolve_field(owner, "described")
        assert resolved.status == "degraded"
        assert _field_info(resolved.value).description == "obj"

    def test_typo_is_an_error(self, owner) -> None:
        resolved = _resolve_field(owner, "typo")
        assert resolved.status == "error" and resolved.missing == ("Hevy",)

    def test_shape_of_unresolved_object(self, owner) -> None:
        assert _resolve_field(owner, "many").shape == "dict"


class TestMetadataRecovery:
    @pytest.mark.parametrize(
        "annotation",
        [
            "Optional[Annotated[Heavy, Field(description='kept')]]",
            "Union[Annotated[Heavy, Field(description='kept')], None]",
            "Annotated[Heavy, Field(description='kept')] | None",
            "Annotated[Optional[Heavy], Field(description='kept'), 'marker']",
        ],
    )
    def test_field_info_survives_degradation(self, make_package, annotation: str) -> None:
        pkg = make_package(
            {
                "rec": f"""
                from __future__ import annotations
                from typing import TYPE_CHECKING, Annotated, Optional, Union
                from pydantic import Field

                if TYPE_CHECKING:
                    from cs_never_imported_rec import Heavy

                class Owner:
                    knob: {annotation}
                """
            }
        )
        owner = _import(pkg, "rec").Owner
        resolved = A.resolve_class_annotation(owner, "knob")
        assert resolved.status == "degraded"
        assert get_args(resolved.value)[0] is Any
        assert _field_info(resolved.value).description == "kept"


class TestUndefinedNames:
    def test_typo_is_an_error(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "typo")
        assert resolved.status == "error"
        assert resolved.reason == "undefined_name"
        assert resolved.missing == ("Hevy",)
        assert resolved.warning() is None
        exc = resolved.error("Leaf")
        assert isinstance(exc, ConfigAnnotationError)
        assert exc.owner == "Leaf" and exc.field == "typo"
        assert exc.annotation == "Optional[Hevy]"
        message = str(exc)
        assert "name 'Hevy' is not defined" in message
        assert f"(declared on {tc_owner.__module__}.Owner)" in message
        assert "Fix the spelling, or import it at module level" in message

    def test_every_missing_name_is_reported(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "two_typos")
        assert resolved.status == "error"
        assert resolved.missing == ("Hevy", "Sesion")
        assert "names 'Hevy', 'Sesion' are not defined" in resolved.detail

    def test_metadata_typo_is_an_error(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "meta_typo")
        assert resolved.status == "error" and resolved.missing == ("Feild",)
        with pytest.raises(ConfigAnnotationError, match="Feild"):
            resolved.raise_if_error()

    def test_attribute_typo_is_an_evaluation_error(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "attr_typo")
        assert resolved.status == "error"
        # ``typing`` itself is not imported in that module: a missing name.
        assert resolved.missing == ("typing",)

    def test_failing_metadata_call_is_an_evaluation_error(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "bad_call")
        assert resolved.status == "error"
        assert resolved.reason == "evaluation_failed"
        assert "raised" in resolved.detail and "default" in resolved.detail

    def test_attribute_error_on_real_module(self) -> None:
        resolved = A.resolve_annotation("typing.Lsit[int]", sys.modules[__name__], name="x")
        assert resolved.status == "error" and resolved.reason == "evaluation_failed"
        assert "AttributeError" in resolved.detail

    def test_syntax_error(self) -> None:
        resolved = A.resolve_annotation("list[", None, name="x")
        assert resolved.status == "error" and resolved.reason == "evaluation_failed"
        assert "SyntaxError" in resolved.detail

    def test_resolved_outcome_never_raises(self, tc_owner) -> None:
        resolved = _resolve_field(tc_owner, "plain")
        assert resolved.status == "resolved" and resolved.value is int
        resolved.raise_if_error()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved.warn()


class TestFunctionLocals:
    def test_function_local_names_degrade_with_extraction_warning(self) -> None:
        class LocalThing:
            pass

        @dataclasses.dataclass
        class Holder:
            thing: Optional[LocalThing] = None
            described: Annotated[LocalThing, Field(description="kept")] = None

        resolved = _resolve_field(Holder, "thing")
        if resolved.status == "resolved":  # pragma: no cover - 3.14 closures, no PEP 563
            pytest.skip("annotations are closures here")
        assert resolved.status == "degraded"
        assert resolved.reason == "function_locals"
        assert resolved.missing == ("LocalThing",)
        category, message = resolved.warning()
        assert category is ExtractionWarning
        assert "local to the function that defines" in message
        described = _resolve_field(Holder, "described")
        assert _field_info(described.value).description == "kept"

    def test_typo_in_function_local_class_is_still_an_error(self) -> None:
        class LocalThing:
            pass

        @dataclasses.dataclass
        class Holder:
            thing: Optional[LocalThng] = None  # noqa: F821 - deliberate typo
            ok: Optional[LocalThing] = None

        resolved = _resolve_field(Holder, "thing")
        assert resolved.status == "error"
        assert resolved.reason == "undefined_name"
        assert resolved.missing == ("LocalThng",)

    def test_nested_function_locals(self, make_package) -> None:
        pkg = make_package(
            {
                "nested": """
                from __future__ import annotations
                from typing import Optional

                def outer():
                    class OuterThing:
                        pass

                    def inner():
                        class InnerThing:
                            pass

                        class Holder:
                            a: Optional[OuterThing]
                            b: Optional[InnerThing]
                            c: Optional[Mispeled]

                            def method(self):
                                return OuterThing

                        return Holder

                    return inner()
                """
            }
        )
        holder = _import(pkg, "nested").outer()
        resolved = _resolve_own(holder)
        assert resolved["a"].reason == "function_locals"
        assert resolved["b"].reason == "function_locals"
        assert resolved["c"].reason == "undefined_name"

    def test_local_metadata_degrades(self) -> None:
        """``test_annotated_with_non_fieldinfo_metadata_graceful`` shape."""

        class Marker:
            pass

        class MyClass:
            def __init__(self, *, name: Annotated[str, Marker()], count: int = 5):
                pass

        resolved = walk_init_chain(MyClass).get("name").resolve()
        assert resolved.status == "degraded" and resolved.reason == "function_locals"

    @pytest.mark.skipif(not PY314, reason="PEP 649 closures")
    def test_closure_names_resolve_natively_on_314(self, make_package) -> None:
        pkg = make_package(
            {
                "closures": """
                from typing import Optional

                def make():
                    class LocalThing:
                        pass

                    class Holder:
                        thing: Optional[LocalThing]

                        def __init__(self, p: LocalThing):
                            pass

                    return Holder, LocalThing
                """
            }
        )
        holder, local = _import(pkg, "closures").make()
        assert A.resolve_class_annotation(holder, "thing").value == Optional[local]
        assert walk_init_chain(holder).get("p").resolve().value is local


@pytest.mark.skipif(not PY314, reason="unquoted TYPE_CHECKING names only import on 3.14")
class TestLazyAnnotations314:
    def test_type_checking_names_degrade_without_name_error(self, make_package) -> None:
        pkg = make_package(
            {
                "lazy": """
                from dataclasses import dataclass
                from typing import TYPE_CHECKING, Annotated, Optional
                from pydantic import Field

                if TYPE_CHECKING:
                    from collections import OrderedDict as Heavy

                @dataclass
                class Owner:
                    knob: Annotated[int, Field(ge=1, description="k")] = 3
                    heavy: Optional[Heavy] = None
                    typo: Optional[Hevy] = None
                    meta: Annotated[int, Feild(ge=1)] = 1

                @dataclass
                class Child(Owner):
                    z: int = 1
                """
            }
        )
        cls = _import(pkg, "lazy").Child
        chain = walk_init_chain(cls)
        assert chain.names() == ("knob", "heavy", "typo", "meta", "z")
        resolved = {p.name: p.resolve() for p in chain.params}
        assert resolved["knob"].status == "resolved"
        assert _field_info(resolved["knob"].value).description == "k"
        assert resolved["heavy"].reason == "type_checking_only"
        assert resolved["typo"].reason == "undefined_name"
        assert resolved["meta"].status == "error" and resolved["meta"].missing == ("Feild",)
        assert resolved["z"].value is int


# ---------------------------------------------------------------------------
# TYPE_CHECKING scan
# ---------------------------------------------------------------------------


class TestTypeCheckingScan:
    def test_scan_variants(self, make_package) -> None:
        pkg = make_package(
            {
                "scan": """
                import typing as t
                from typing import TYPE_CHECKING as TC

                if t.TYPE_CHECKING:
                    from cs_absent_one import A1
                if TC:
                    import cs_absent_two as two
                    import cs_absent_three.sub
                    from .peer import Peer
                    Alias = int
                if not TC:
                    RuntimeOnly = 1
                else:
                    from cs_absent_four import Four
                try:
                    if TC:
                        from cs_absent_five import Five
                except ImportError:
                    pass

                def f():
                    if TC:
                        from cs_absent_six import Six
                """,
                "peer": "class Peer: ...",
            }
        )
        module = _import(pkg, "scan")
        found = A.type_checking_imports(module)
        assert set(found) == {"A1", "two", "cs_absent_three", "Peer", "Alias", "Four", "Five"}
        assert found["A1"].module == "cs_absent_one" and found["A1"].attr == "A1"
        assert found["two"].module == "cs_absent_two" and found["two"].attr is None
        assert found["Peer"].module == f"{pkg}.peer" and found["Peer"].attr == "Peer"
        assert found["Alias"].module is None
        assert A.type_checking_names(module) == frozenset(found)
        assert A.type_checking_names(module.__dict__) == frozenset(found)
        for name in ("cs_absent_one", "cs_absent_two", "cs_absent_four"):
            assert name not in sys.modules

    def test_no_source_means_no_names(self) -> None:
        assert A.type_checking_names({"__name__": "nowhere"}) == frozenset()
        assert A.type_checking_names(None) == frozenset()


# ---------------------------------------------------------------------------
# None / Optional normalization
# ---------------------------------------------------------------------------


class TestNoImplicitOptional:
    """The None rule lives in ``_fieldinfo.config_field`` (tests in
    test_fieldinfo.py::TestNoneRule); resolution itself never adds it."""

    def test_resolution_never_adds_implicit_optional(self, make_package) -> None:
        """On 3.10 ``typing.get_type_hints`` wraps ``= None`` parameters in
        ``Optional`` and hides the FieldInfo; the engine does not, on any
        version, and ``config_field`` then puts Optional *inside*."""
        from conscribe.config._fieldinfo import config_field

        pkg = make_package(
            {
                "implicit": """
                from typing import Annotated
                from pydantic import Field

                class Owner:
                    def __init__(self, timeout: Annotated[float, Field(description="s")] = None):
                        self.timeout = timeout
                """
            }
        )
        cls = _import(pkg, "implicit").Owner
        param = walk_init_chain(cls).get("timeout")
        resolved = param.resolve()
        assert get_origin(resolved.value) is Annotated
        assert get_args(resolved.value)[0] is float
        tp, info = config_field(
            resolved.value, owner="m.Owner", field="timeout", signature_default=param.default,
        )
        assert tp == Optional[float]
        assert info.description == "s"
        hints = typing.get_type_hints(cls.__init__, include_extras=True)
        tp, info = config_field(hints["timeout"], owner="m.Owner", field="timeout", signature_default=None)
        assert tp == Optional[float] and info.description == "s"


class TestAdmitsNone:
    @pytest.mark.parametrize(
        "annotation, expected",
        [
            (None, True),
            (type(None), True),
            (Any, True),
            (object, True),
            (Optional[int], True),
            (int | None, True),
            (Annotated[Optional[int], "m"], True),
            (typing.Literal[None, 1], True),
            (int, False),
            (list[int], False),
            (Annotated[int, "m"], False),
            ("Optional[X]", True),
            ("X | None", True),
            ("None", True),
            ("Union[X, None]", True),
            ("Annotated[X | None, Field()]", True),
            ("'X | None'", True),
            ("X", False),
            ("list[X]", False),
        ],
    )
    def test_table(self, annotation: Any, expected: bool) -> None:
        assert A.admits_none(annotation) is expected


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class Iterates:
    """Structurally iterable, but not a collection."""

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


class Point(NamedTuple):
    x: int


class Record(typing.TypedDict):
    x: int


_SHAPES = [
    ("list[int]", list[int], "list"),
    ("List[int]", List[int], "list"),
    ("tuple[int, ...]", tuple[int, ...], "list"),
    ("set[int]", set[int], "list"),
    ("frozenset[int]", frozenset[int], "list"),
    ("collections.deque", collections.deque, "list"),
    ("typing.Sequence[int]", typing.Sequence[int], "list"),
    ("typing.AbstractSet[int]", typing.AbstractSet[int], "list"),
    ("typing.Iterable[int]", typing.Iterable[int], "list"),
    ("typing.Collection[int]", typing.Collection[int], "list"),
    ("list", list, "list"),
    ("dict[str, int]", dict[str, int], "dict"),
    ("Dict[str, int]", Dict[str, int], "dict"),
    ("typing.Mapping[str, int]", typing.Mapping[str, int], "dict"),
    ("collections.OrderedDict", collections.OrderedDict, "dict"),
    ("collections.defaultdict", collections.defaultdict, "dict"),
    ("collections.Counter", collections.Counter, "dict"),
    ("dict", dict, "dict"),
    ("Optional[Dict[str, int]]", Optional[Dict[str, int]], "dict"),
    ("dict[str, int] | None", dict[str, int] | None, "dict"),
    ("Annotated[Optional[list[int]], Field()]", Annotated[Optional[list[int]], Field()], "list"),
    ("Union[list[int], int, None]", Union[list[int], int, None], "list"),
    ("Union[dict[str, int], list[int]]", Union[dict[str, int], list[int]], "dict"),
    ("str", str, "scalar"),
    ("bytes", bytes, "scalar"),
    ("int", int, "scalar"),
    ("Optional[str]", Optional[str], "scalar"),
    ("Any", Any, "scalar"),
    ("Iterates", Iterates, "scalar"),
    ("Point", Point, "scalar"),
    ("Record", Record, "scalar"),
    ("typing.Literal['a', 'b']", typing.Literal["a", "b"], "scalar"),
    ("ClassVar[list[int]]", ClassVar[list[int]], "list"),
]


class _ShapeOwner:
    """Owner whose module namespace (this test module) resolves the names."""


class TestAnnotationShape:
    @pytest.mark.parametrize("text, obj, expected", _SHAPES, ids=[s[0] for s in _SHAPES])
    def test_object_string_and_forward_ref_agree(self, text: str, obj: Any, expected: str) -> None:
        assert A.annotation_shape(obj) == expected
        assert A.annotation_shape(text, _ShapeOwner) == expected
        assert A.annotation_shape(typing.ForwardRef(text), _ShapeOwner) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Sequence[Heavy]", "list"),
            ("Mapping[str, Heavy]", "dict"),
            ("Optional[List[Heavy]]", "list"),
            ("'dict[str, Heavy]' | None", "dict"),
            ("Heavy", "scalar"),
            ("Annotated[Heavy, Field()]", "scalar"),
            ("typing.Dict[str, Heavy]", "dict"),
            ("t.Set[Heavy]", "list"),
        ],
    )
    def test_unresolvable_names_judged_by_spelling(self, text: str, expected: str) -> None:
        assert A.annotation_shape(text) == expected

    def test_user_class_named_like_a_container_is_scalar(self, make_package) -> None:
        pkg = make_package(
            {
                "names": """
                from __future__ import annotations

                class Set:
                    pass

                class Holder:
                    a: Set
                    b: Set | None
                """
            }
        )
        holder = _import(pkg, "names").Holder
        assert A.annotation_shape("Set", holder) == "scalar"
        assert A.annotation_shape("Set | None", holder) == "scalar"
        assert A.annotation_shape("Set") == "list"  # no owner: spelling

    def test_empty_and_none(self) -> None:
        assert A.annotation_shape(inspect.Parameter.empty) == "scalar"
        assert A.annotation_shape(None) == "scalar"
        assert A.annotation_shape("not valid (") == "scalar"

    def test_initvar_and_newtype(self) -> None:
        assert A.annotation_shape(dataclasses.InitVar[list[int]]) == "list"
        user_ids = typing.NewType("user_ids", list)
        assert A.annotation_shape(user_ids) == "list"

    def test_resolved_annotation_shape_property(self, tc_owner) -> None:
        assert _resolve_field(tc_owner, "two_typos").shape == "dict"
        assert _resolve_field(tc_owner, "plain").shape == "scalar"


# ---------------------------------------------------------------------------
# ClassVar / KW_ONLY / InitVar
# ---------------------------------------------------------------------------

_CLASSVAR_MODULE = """
from __future__ import annotations
import typing as t
from dataclasses import dataclass
from typing import Annotated, ClassVar
from typing import ClassVar as CV

class ClassVarRegistry:
    pass

@dataclass
class Fixture:
    plain: ClassVar[int] = 1
    dotted: t.ClassVar[int] = 2
    aliased: CV[int] = 3
    registry: ClassVarRegistry = None
    real: int = 5
"""


def _annotated_classvar_supported() -> bool:
    try:
        Annotated[ClassVar[int], "m"]
    except TypeError:  # Python 3.10.0-3.10.2: typing rejects it (bpo-46491)
        return False
    return True


class TestIsClassVar:
    def test_objects(self) -> None:
        assert A.is_classvar(ClassVar)
        assert A.is_classvar(ClassVar[int])
        assert not A.is_classvar(int)
        assert not A.is_classvar(Optional[int])

    @pytest.mark.skipif(
        not _annotated_classvar_supported(),
        reason="typing rejects Annotated[ClassVar[...]] on this Python version",
    )
    def test_annotated_classvar_object(self) -> None:
        assert A.is_classvar(Annotated[ClassVar[int], "m"])
        assert A.is_pseudo_field(Annotated[ClassVar[int], "m"])

    def test_strings_without_owner_use_spelling(self) -> None:
        assert A.is_classvar("ClassVar[int]")
        assert A.is_classvar("typing.ClassVar[int]")
        assert A.is_classvar("t.ClassVar[int]")
        assert A.is_classvar("'ClassVar[int]'")
        assert not A.is_classvar("ClassVarRegistry")
        assert not A.is_classvar("ClassVarRegistry[int]")
        assert not A.is_classvar("int")

    def test_parity_with_dataclasses(self, make_package) -> None:
        """Ground truth: what stdlib dataclasses treat as a ClassVar.

        (conscribe additionally unwraps ``Annotated[ClassVar[...], ...]``,
        which dataclasses do not; see ``test_objects``.)
        """
        pkg = make_package({"cv": _CLASSVAR_MODULE})
        fixture = _import(pkg, "cv").Fixture
        field_names = {f.name for f in dataclasses.fields(fixture)}
        assert field_names == {"registry", "real"}
        for name, raw in A.own_annotations(fixture).items():
            assert A.is_classvar(raw, fixture) is (name not in field_names), name

    def test_alias_resolved_through_owner(self, make_package) -> None:
        pkg = make_package({"cv": _CLASSVAR_MODULE})
        fixture = _import(pkg, "cv").Fixture
        assert A.is_classvar("CV[int]", fixture)
        assert not A.is_classvar("CV[int]")  # unresolvable spelling is not ClassVar


class TestDataclassPseudoFields:
    def test_kw_only_sentinel(self) -> None:
        assert A.is_kw_only_sentinel(dataclasses.KW_ONLY)
        assert A.is_kw_only_sentinel("KW_ONLY")
        assert A.is_kw_only_sentinel("dataclasses.KW_ONLY")
        assert not A.is_kw_only_sentinel(int)

    def test_initvar(self) -> None:
        assert A.is_initvar(dataclasses.InitVar[int])
        assert A.is_initvar(dataclasses.InitVar)
        assert A.is_initvar("InitVar[int]")
        assert A.is_initvar("dataclasses.InitVar[int]")
        assert not A.is_initvar(int)
        assert A.unwrap_initvar(dataclasses.InitVar[int]) is int
        assert A.unwrap_initvar(dataclasses.InitVar) is Any
        assert A.unwrap_initvar(str) is str


class TestHelpers:
    def test_unwrap_annotated(self) -> None:
        assert A.unwrap_annotated(Annotated[int, "a", "b"]) == (int, ("a", "b"))
        assert A.unwrap_annotated(int) == (int, ())

    def test_contexts(self, make_package) -> None:
        pkg = make_package(
            {
                "ctx": """
                class Owner:
                    LEVEL = 3

                    def __init__(self, x: int = 1):
                        pass
                """
            }
        )
        module = _import(pkg, "ctx")
        ctx = A.class_context(module.Owner)
        assert ctx.module == f"{pkg}.ctx"
        assert ctx.lookup("Owner") == (True, module.Owner)
        assert ctx.lookup("LEVEL") == (True, 3)
        assert ctx.lookup("len") == (True, len)
        assert ctx.lookup("nope") == (False, None)
        fctx = A.function_context(module.Owner.__init__, module.Owner)
        assert fctx.function is module.Owner.__init__
        assert fctx.scope_qualname == "Owner.__init__"
        assert not fctx.function_local
        assert A.class_globals(module.Owner) is module.__dict__

    def test_class_globals_prefers_real_namespace(self) -> None:
        """A dataclass whose module is not in ``sys.modules`` gets generated
        methods with empty globals; the class body's namespace still wins."""
        namespace: dict[str, Any] = {"__name__": "cs_ann_not_a_module"}
        source = (
            "import dataclasses\n"
            "from typing import Optional\n"
            "@dataclasses.dataclass\n"
            "class Exec:\n"
            "    x: Optional[int] = None\n"
            "    def helper(self):\n"
            "        return 1\n"
        )
        # dont_inherit: no PEP 563 from this module (dataclasses would need
        # the class's module in sys.modules to read string annotations).
        exec(compile(source, "<exec>", "exec", dont_inherit=True), namespace)
        cls = namespace["Exec"]
        assert A.class_globals(cls) is namespace
        resolved = walk_init_chain(cls).get("x").resolve()
        assert resolved.value == Optional[int]


# ===========================================================================
# Review fixes (2.0.0 release review)
# ===========================================================================


class TestQuotedForwardRefsInUnions:
    """B1 (3.12.0-3.12.3: ``typing._eval_type``'s 4th positional parameter is
    ``recursive_guard``) and M2 (3.10: PEP 604 unions holding ``list["X"]``)."""

    def test_eval_type_takes_type_params_only_by_keyword(self) -> None:
        params = inspect.signature(typing._eval_type).parameters  # type: ignore[attr-defined]
        assert A._EVAL_TYPE_TAKES_TYPE_PARAMS is ("type_params" in params)

    def test_quoted_names_nested_in_unions_resolve(self, make_package) -> None:
        pkg = make_package(
            {
                "tree": """
                # no PEP 563: quoted forward references inside evaluated objects
                from typing import Optional

                class Node:
                    def __init__(self, children: list["Leaf"] | None = None, label: str = "") -> None:
                        pass

                class Node2:
                    def __init__(self, children: Optional[list["Leaf"]] = None) -> None:
                        pass

                class Node3:
                    def __init__(self, mapping: dict[str, "Leaf"] | None = None) -> None:
                        pass

                class Leaf:
                    pass
                """
            }
        )
        module = _import(pkg, "tree")
        leaf = module.Leaf
        expected = {
            "Node": ("children", Optional[list[leaf]]),
            "Node2": ("children", Optional[list[leaf]]),
            "Node3": ("mapping", Optional[dict[str, leaf]]),
        }
        for cls_name, (param, value) in expected.items():
            resolved = walk_init_chain(getattr(module, cls_name)).get(param).resolve()
            assert resolved.status == "resolved", resolved.detail
            assert resolved.value == value

    @pytest.mark.skipif(sys.version_info >= (3, 11), reason="the 3.10-only conversion")
    def test_str_args_inside_pep604_union_become_forward_refs(self) -> None:
        converted = A._str_args_to_forward_refs(list["Leaf"] | None)  # noqa: F821
        inner = [a for a in get_args(converted) if a is not type(None)][0]
        assert get_args(inner) == (typing.ForwardRef("Leaf"),)


class TestClassGlobalsIgnoresForeignWrappers:
    """B2: ``functools.wraps`` copies ``__qualname__``, so a decorated method
    must not make a class's annotations resolve in the decorator's module."""

    def test_contextmanager_method(self, make_package) -> None:
        pkg = make_package(
            {
                "wrapped": """
                from __future__ import annotations

                import contextlib
                import enum
                from collections.abc import Iterator
                from dataclasses import dataclass

                class Mode(enum.Enum):
                    A = "a"

                @dataclass
                class WrapsDecorated:
                    mode: Mode = Mode.A

                    @contextlib.contextmanager
                    def session(self) -> Iterator[int]:
                        yield 1

                class OnlyWrapped:
                    @contextlib.contextmanager
                    def session(self) -> Iterator[int]:
                        yield 1
                """
            }
        )
        module = _import(pkg, "wrapped")
        for cls in (module.WrapsDecorated, module.OnlyWrapped):
            assert A.class_globals(cls) is module.__dict__
        resolved = walk_init_chain(module.WrapsDecorated).get("mode").resolve()
        assert resolved.status == "resolved" and resolved.value is module.Mode

    def test_manual_qualname_copy_without_wrapped(self, make_package) -> None:
        """A wrapper that copies the qualified name but sets no
        ``__wrapped__`` is not compiled in the class body either."""
        pkg = make_package(
            {
                "deco": """
                def keep_name(func):
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
                    wrapper.__name__ = func.__name__
                    wrapper.__qualname__ = func.__qualname__
                    return wrapper
                """,
                "user": """
                from .deco import keep_name

                class Owner:
                    @keep_name
                    def method(self):
                        return 1
                """,
            }
        )
        module = _import(pkg, "user")
        assert A.class_globals(module.Owner) is module.__dict__

    def test_attrs_snapshot_globals_are_not_used(self, make_package) -> None:
        """attrs hands generated methods a *copy* of the module globals;
        names bound after the class (forward references) are missing there."""
        pytest.importorskip("attrs")
        pkg = make_package(
            {
                "late": """
                from __future__ import annotations
                import attrs

                @attrs.define
                class Early:
                    later: Later | None = None

                class Later:
                    pass
                """
            }
        )
        module = _import(pkg, "late")
        assert A.class_globals(module.Early) is module.__dict__
        resolved = walk_init_chain(module.Early).get("later").resolve()
        assert resolved.status == "resolved"
        assert resolved.value == Optional[module.Later]

    def test_cache_follows_module_rewrite(self) -> None:
        class Plain:
            pass

        assert A.class_globals(Plain) is sys.modules[__name__].__dict__
        Plain.__module__ = "collections"
        assert A.class_globals(Plain) is sys.modules["collections"].__dict__


class TestFunctionScopeLookupOrder:
    """M1: in a constructor signature the class namespace never shadows a
    builtin; it is only a last resort."""

    def test_builtins_win_over_class_names(self, make_package) -> None:
        pkg = make_package(
            {
                "shadow": """
                from __future__ import annotations

                from typing import Any, Literal, Optional

                class Store:
                    Mode = Literal["fast", "safe"]

                    def __init__(
                        self,
                        tags: set[str] | None = None,
                        options: dict[str, Any] | None = None,
                        mode: Mode = "fast",
                    ) -> None:
                        pass

                    def set(self, key: str, value: Any) -> None:
                        pass

                    def dict(self) -> dict:
                        return {}

                class Kind:
                    type = "react"

                    def __init__(self, model_cls: Optional[type] = None) -> None:
                        pass
                """
            }
        )
        module = _import(pkg, "shadow")
        store = walk_init_chain(module.Store)
        assert store.get("tags").resolve().value == Optional[set[str]]
        assert store.get("options").resolve().value == Optional[dict[str, Any]]
        # class-level names nothing else defines still resolve (last resort)
        mode = store.get("mode").resolve()
        assert mode.status == "resolved" and get_args(mode.value) == ("fast", "safe")
        kind = walk_init_chain(module.Kind).get("model_cls").resolve()
        assert kind.value == Optional[type]

    def test_lookup_and_namespaces_agree(self) -> None:
        class Owner:
            type = "react"
            Alias = int

            def __init__(self, x: type) -> None: ...

        fctx = A.function_context(Owner.__init__, Owner)
        assert fctx.lookup("type") == (True, type)
        assert fctx.lookup("Alias") == (True, int)
        ns = fctx.namespace()
        assert "type" not in ns and ns["Alias"] is int
        _, localns = fctx.eval_namespaces()
        assert "type" not in localns and localns["Alias"] is int
        # class-body annotations keep typing.get_type_hints' order
        assert A.class_context(Owner).lookup("type") == (True, "react")

    def test_shape_of_set_receptor_with_set_method(self, make_package) -> None:
        pkg = make_package(
            {
                "shape": """
                from __future__ import annotations
                from typing import Any

                class Cache:
                    def __init__(self, listeners: set[Any] = frozenset(), size: int = 10) -> None:
                        pass

                    def set(self, key: str, value: Any) -> None:
                        pass
                """
            }
        )
        cls = _import(pkg, "shape").Cache
        raw, ctx = walk_init_chain(cls).get("listeners").annotation_source()
        assert A.annotation_shape(raw, ctx) == "list"


class TestDegradedMetadataFailure:
    """m9: metadata of a degrading annotation that raises is an error."""

    def test_raising_metadata_is_an_error(self, make_package) -> None:
        pkg = make_package(
            {
                "mod": """
                from __future__ import annotations
                from typing import TYPE_CHECKING, Annotated
                from pydantic import Field
                if TYPE_CHECKING:
                    from decimal import Decimal

                def boom(**kw):
                    raise RuntimeError("metadata factory crashed")

                class A:
                    def __init__(
                        self,
                        x: Annotated[Decimal, Field(ge=1, description="kept")] = 2,
                        w: Annotated[Decimal, boom(a=1)] = 5,
                    ) -> None: ...
                """
            }
        )
        chain = walk_init_chain(_import(pkg, "mod").A)
        kept = chain.get("x").resolve()
        assert kept.status == "degraded" and kept.reason == "type_checking_only"
        assert _field_info(kept.value).description == "kept"
        broken = chain.get("w").resolve()
        assert broken.status == "error" and broken.reason == "metadata"
        assert "boom(a=1)" in broken.detail and "RuntimeError" in broken.detail
        with pytest.raises(ConfigAnnotationError, match="metadata factory crashed"):
            broken.raise_if_error()


class TestWarningsPointAtUserCode:
    """m11: conscribe's warnings carry the user's file and line."""

    def test_degradation_warning_location(self, make_package) -> None:
        pkg = make_package(
            {
                "tc": """
                from __future__ import annotations
                from typing import TYPE_CHECKING
                if TYPE_CHECKING:
                    from decimal import Decimal

                class Owner:
                    def __init__(self, budget: Decimal | None = None) -> None:
                        pass
                """
            }
        )
        module = _import(pkg, "tc")
        res = walk_init_chain(module.Owner).get("budget").resolve()
        with pytest.warns(DegradationWarning) as record:
            res.warn(module.Owner)
        assert record[0].filename == module.__file__
        assert record[0].lineno == inspect.getsourcelines(module.Owner)[1]  # the class line

    def test_warn_at_class_and_fallback(self) -> None:
        class Here:
            pass

        with pytest.warns(ExtractionWarning) as record:
            A.warn_at("about Here", ExtractionWarning, Here)
        assert record[0].filename == __file__
        assert record[0].lineno == inspect.getsourcelines(Here)[1]
        with pytest.warns(ExtractionWarning) as record:
            A.warn_at("nowhere", ExtractionWarning, None)
        assert record[0].filename == __file__


class TestSignatureCache:
    def test_cached_per_function_and_invalidated_by_mutation(self) -> None:
        def init(self, a: int = 1, **kwargs: Any) -> None: ...

        first = A.safe_signature(init)
        assert A.safe_signature(init) is first
        init.__defaults__ = (2,)
        second = A.safe_signature(init)
        assert second is not first and second.parameters["a"].default == 2

    def test_new_function_is_never_served_a_stale_answer(self) -> None:
        class Owner:
            def __init__(self, a: int) -> None: ...

        assert walk_init_chain(Owner).names() == ("a",)

        def replacement(self, b: str) -> None: ...

        Owner.__init__ = replacement  # type: ignore[method-assign]
        assert walk_init_chain(Owner).names() == ("b",)


class TestPseudoFields:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("ClassVar[int]", True),
            ("t.ClassVar[int]", True),
            ("Annotated[ClassVar[int], 'm']", True),
            ("KW_ONLY", True),
            ("InitVar[int]", True),
            ("'InitVar[int]'", True),
            ("Annotated[InitVar[int], 'm']", False),  # dataclasses only see a top-level InitVar
            ("ClassVarRegistry", False),
            ("int", False),
        ],
    )
    def test_matches_the_three_predicates(self, text: str, expected: bool) -> None:
        assert A.is_pseudo_field(text) is expected
        individually = A.is_classvar(text) or A.is_kw_only_sentinel(text) or A.is_initvar(text)
        assert individually is expected


@pytest.mark.skipif(not PY314, reason="PEP 649 annotations")
class TestPep649PerEntryFallback:
    """M3: one annotation that cannot be evaluated no longer turns its valid
    siblings into source text (a lambda in their metadata survives)."""

    def test_lambda_metadata_survives_a_failing_sibling(self, make_package) -> None:
        pkg = make_package(
            {
                "lam": """
                from typing import TYPE_CHECKING, Annotated, Optional
                from pydantic import AfterValidator, Field

                if TYPE_CHECKING:
                    from decimal import Decimal

                class WithLambda:
                    def __init__(
                        self,
                        a: Annotated[int, AfterValidator(lambda v: v * 2)] = 1,
                        b: Optional[Decimal] = None,
                    ) -> None:
                        pass
                """
            }
        )
        cls = _import(pkg, "lam").WithLambda
        annotations, _ = A.function_annotation_source(cls.__init__)
        assert not isinstance(annotations["a"], str)
        assert isinstance(annotations["b"], str)
        chain = walk_init_chain(cls)
        a = chain.get("a").resolve()
        assert a.status == "resolved"
        validator = get_args(a.value)[1]
        assert validator.func(3) == 6
        assert chain.get("b").resolve().status == "degraded"
