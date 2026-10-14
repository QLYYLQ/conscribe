"""Tests for conscribe.config.mro — MRO parameter collection.

Tests ``classify_class_scope()``, ``collect_mro_params()``, and
``MROCollectionResult``:
- Scope classification (local, third_party, stdlib)
- Basic **kwargs chain resolution
- Multi-level chain traversal
- Chain termination (no **kwargs parent)
- Child parameter override (dedup)
- Depth limiting
- Scope filtering
- fully_resolved flag semantics
- Diamond inheritance

2.0 additions (``walk_init_chain`` and the X1 rules):
- init-less classes (Generic / ABC / Protocol mixins, bridges) are skipped
  before scope and depth are checked
- an unreadable signature is truncation, never the natural end
- the runtime view ignores scope; fast paths for pydantic models and
  pydantic dataclasses; Protocol placeholders and copied ``__init__``
- owner-aware hints (D3)
"""
from __future__ import annotations

import abc
import dataclasses
import importlib
import importlib.util
import inspect
import sys
import textwrap
import types
import uuid
from typing import Annotated, Any, Generic, Protocol, TypeVar, get_args, get_origin, runtime_checkable

import pytest
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.fields import FieldInfo

from conscribe.config._utils import (
    constructor_definer,
    dataclass_field_owner,
    find_init_definer,
    is_bridge_class,
    is_placeholder_init,
    is_swallowing_placeholder,
    model_init_only_forwards,
    named_params,
    own_init,
)
from conscribe.config.mro import (
    HAS_DEFAULT_FACTORY,
    DeclaredDefault,
    InitChain,
    MROCollectionResult,
    _extract_package_name,
    classify_class_scope,
    collect_mro_params,
    is_no_default,
    keyword_param_names,
    mro_depth_problem,
    mro_scope_problem,
    walk_init_chain,
)

_T = TypeVar("_T")


# ===================================================================
# Scope classification
# ===================================================================

class TestClassifyClassScope:
    """Tests for classify_class_scope()."""

    def test_classify_local_class(self) -> None:
        """A class defined in this test file is classified as 'local'."""

        class LocalClass:
            pass

        assert classify_class_scope(LocalClass) == "local"

    def test_classify_stdlib_class(self) -> None:
        """A stdlib class (e.g. ``int``, ``dict``) is classified as 'stdlib'."""
        assert classify_class_scope(int) == "stdlib"
        assert classify_class_scope(dict) == "stdlib"

    def test_classify_third_party_class(self) -> None:
        """A class from site-packages is classified as 'third_party'."""
        from pydantic import BaseModel

        assert classify_class_scope(BaseModel) == "third_party"


# ===================================================================
# collect_mro_params — basic cases
# ===================================================================

class TestCollectMROParamsBasic:
    """Basic tests for collect_mro_params()."""

    def test_no_kwargs_returns_empty(self) -> None:
        """Class without **kwargs returns empty result."""

        class Parent:
            def __init__(self, x: int, y: str = "hello"):
                pass

        class Child(Parent):
            def __init__(self, z: float):
                pass

        result = collect_mro_params(Child, scope="all")
        assert result.params == []
        assert result.init_definers == []
        assert result.fully_resolved is True

    def test_single_level_kwargs_collects_parent_params(self) -> None:
        """Child with **kwargs collects parent's named params."""

        class Parent:
            def __init__(self, x: int, y: str = "hello"):
                pass

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        assert "x" in param_names
        assert "y" in param_names
        assert "z" not in param_names  # child's own param, not collected
        assert Parent in result.init_definers
        assert result.fully_resolved is True

    def test_two_level_chain_collects_grandparent_params(self) -> None:
        """Two-level chain: Child -> Parent -> Grandparent."""

        class Grandparent:
            def __init__(self, a: int, b: str = "gp"):
                pass

        class Parent(Grandparent):
            def __init__(self, x: int, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        assert "x" in param_names
        assert "a" in param_names
        assert "b" in param_names
        assert "z" not in param_names
        assert result.fully_resolved is True

    def test_chain_stops_at_no_kwargs_parent(self) -> None:
        """Chain stops when a parent has no **kwargs."""

        class Base:
            def __init__(self, a: int, b: str = "base"):
                pass

        class Parent(Base):
            def __init__(self, x: int, y: str = "parent", **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        # Parent params
        assert "x" in param_names
        assert "y" in param_names
        # Base params (chain continued through Parent's **kwargs)
        assert "a" in param_names
        assert "b" in param_names
        assert result.fully_resolved is True


# ===================================================================
# collect_mro_params — deduplication / override
# ===================================================================

class TestCollectMROParamsOverride:
    """Tests for parameter deduplication (child wins)."""

    def test_child_param_overrides_same_name_parent_param(self) -> None:
        """When child and parent both have param 'x', child's is kept."""

        class Parent:
            def __init__(self, x: str = "parent_x", y: int = 10):
                pass

        class Child(Parent):
            def __init__(self, x: float = 3.14, z: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        # 'x' is child's own, so NOT collected from parent
        assert "x" not in param_names
        # 'y' is only in parent, so collected
        assert "y" in param_names

    def test_diamond_inheritance_no_duplicate_params(self) -> None:
        """Diamond inheritance doesn't produce duplicate parameters."""

        class Base:
            def __init__(self, a: int = 1):
                pass

        class Left(Base):
            def __init__(self, b: int = 2, **kwargs: Any):
                super().__init__(**kwargs)

        class Right(Base):
            def __init__(self, c: int = 3, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Left, Right):
            def __init__(self, d: int = 4, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        # Each param should appear at most once
        assert len(param_names) == len(set(param_names))
        # All parent params should be collected
        assert "b" in param_names
        assert "c" in param_names
        assert "a" in param_names


# ===================================================================
# collect_mro_params — depth limiting
# ===================================================================

class TestCollectMROParamsDepth:
    """Tests for depth limiting."""

    def test_depth_1_collects_only_direct_parent(self) -> None:
        """depth=1 only collects from the direct parent."""

        class Grandparent:
            def __init__(self, a: int):
                pass

        class Parent(Grandparent):
            def __init__(self, x: int, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all", depth=1)
        param_names = [p.name for p in result.params]
        assert "x" in param_names
        assert "a" not in param_names
        assert result.fully_resolved is False

    def test_depth_0_disables_traversal(self) -> None:
        """depth=0 returns empty result (traversal disabled)."""

        class Parent:
            def __init__(self, x: int):
                pass

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all", depth=0)
        assert result.params == []
        assert result.fully_resolved is True

    def test_depth_2_collects_two_levels(self) -> None:
        """depth=2 collects from two parent levels."""

        class GreatGrandparent:
            def __init__(self, w: int):
                pass

        class Grandparent(GreatGrandparent):
            def __init__(self, a: int, **kwargs: Any):
                super().__init__(**kwargs)

        class Parent(Grandparent):
            def __init__(self, x: int, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all", depth=2)
        param_names = [p.name for p in result.params]
        assert "x" in param_names
        assert "a" in param_names
        assert "w" not in param_names
        assert result.fully_resolved is False


# ===================================================================
# collect_mro_params — scope filtering
# ===================================================================

class TestCollectMROParamsScope:
    """Tests for scope filtering."""

    def test_scope_local_skips_third_party_parent(self, third_party) -> None:
        """scope='local' skips parents from site-packages."""

        class Child(third_party.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="local")
        # ExtBase is third-party, should be skipped
        assert result.fully_resolved is False
        assert result.stop_reason == "scope"
        assert [p.name for p in result.params] == []

    def test_scope_all_includes_third_party(self) -> None:
        """scope='all' includes parents from any location."""

        class Parent:
            def __init__(self, x: int):
                pass

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        assert "x" in param_names
        assert result.fully_resolved is True


# ===================================================================
# collect_mro_params — fully_resolved semantics
# ===================================================================

class TestFullyResolved:
    """Tests for the fully_resolved flag."""

    def test_natural_termination_fully_resolved(self) -> None:
        """Chain ending at a parent without **kwargs is fully resolved."""

        class Parent:
            def __init__(self, x: int):
                pass

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        assert result.fully_resolved is True

    def test_scope_truncation_not_fully_resolved(self, third_party) -> None:
        """Chain truncated by scope filtering is NOT fully resolved."""

        class Child(third_party.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="local")
        assert result.fully_resolved is False

    def test_depth_truncation_not_fully_resolved(self) -> None:
        """Chain truncated by depth limit is NOT fully resolved."""

        class Grandparent:
            def __init__(self, a: int):
                pass

        class Parent(Grandparent):
            def __init__(self, x: int, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all", depth=1)
        assert result.fully_resolved is False


# ===================================================================
# collect_mro_params — edge cases
# ===================================================================

class TestCollectMROParamsEdgeCases:
    """Edge cases for collect_mro_params."""

    def test_no_init_definer_returns_empty(self) -> None:
        """Class with no __init__ at all returns empty result."""

        class Empty:
            pass

        result = collect_mro_params(Empty, scope="all")
        assert result.params == []

    def test_parent_without_init_in_dict_is_skipped(self) -> None:
        """A parent class that doesn't define its own __init__ is skipped."""

        class Grandparent:
            def __init__(self, a: int = 1):
                pass

        class Parent(Grandparent):
            # No __init__ here — inherits from Grandparent
            pass

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        param_names = [p.name for p in result.params]
        # Should collect Grandparent's 'a' (skipping Parent who has no __init__)
        assert "a" in param_names
        assert result.fully_resolved is True

    def test_mro_collection_result_is_frozen(self) -> None:
        """MROCollectionResult is a frozen dataclass."""
        result = MROCollectionResult()
        with pytest.raises(AttributeError):
            result.fully_resolved = False  # type: ignore[misc]


# ===================================================================
# _extract_package_name
# ===================================================================

class TestExtractPackageName:
    """Tests for _extract_package_name()."""

    def test_third_party_class(self) -> None:
        """A third-party class returns its top-level package name."""
        from pydantic import BaseModel

        assert _extract_package_name(BaseModel) == "pydantic"

    def test_local_class(self) -> None:
        """A locally defined class returns None."""

        class LocalClass:
            pass

        assert _extract_package_name(LocalClass) is None

    def test_stdlib_class(self) -> None:
        """A stdlib/builtin class returns None (no file or no site-packages)."""
        assert _extract_package_name(int) is None
        assert _extract_package_name(dict) is None


# ===================================================================
# collect_mro_params — package list scope
# ===================================================================

class TestPackageListScope:
    """Tests for list[str] scope (package-specific filtering)."""

    def test_scope_list_includes_specified_package(self, third_party) -> None:
        """scope=[<package>] includes that package's parent params."""

        class Child(third_party.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        package = third_party.__name__.split(".", 1)[0]
        result = collect_mro_params(Child, scope=[package])
        assert [p.name for p in result.params] == ["host"]
        assert result.init_definers == [third_party.ExtBase]
        assert classify_class_scope(third_party.ExtBase) == "third_party"

    def test_scope_list_excludes_unspecified_package(self, third_party) -> None:
        """scope=['some_other'] excludes a parent from another package."""

        class Child(third_party.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope=["some_other"])
        # ExtBase's package is not in ["some_other"], should be excluded
        assert result.fully_resolved is False
        # No pydantic classes in init_definers
        for cls in result.init_definers:
            assert classify_class_scope(cls) != "third_party"

    def test_scope_list_always_includes_local(self) -> None:
        """Local classes always pass the filter with list scope."""

        class Parent:
            def __init__(self, x: int):
                pass

        class Child(Parent):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope=["anything"])
        param_names = [p.name for p in result.params]
        assert "x" in param_names
        assert result.fully_resolved is True

    def test_scope_list_excludes_stdlib(self) -> None:
        """Stdlib classes are always excluded with list scope."""

        class Child(Exception):
            def __init__(self, msg: str, **kwargs: Any):
                super().__init__(msg, **kwargs)

        result = collect_mro_params(Child, scope=["builtins"])
        # Exception is stdlib, should be excluded regardless of list contents
        assert result.fully_resolved is False

    def test_scope_list_empty_excludes_all_third_party(self, third_party) -> None:
        """scope=[] effectively means local-only (all third-party excluded)."""

        class Child(third_party.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope=[])
        # Empty list: no third-party packages included
        assert result.fully_resolved is False
        for cls in result.init_definers:
            assert classify_class_scope(cls) != "third_party"

    def test_scope_list_truncation_not_fully_resolved(self, third_party) -> None:
        """Truncation by package filter sets fully_resolved=False."""

        class Child(third_party.ExtBase):
            def __init__(self, z: float, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope=["nonexistent_pkg"])
        assert result.fully_resolved is False


# ===================================================================
# 2.0: shared fixtures for the walker tests
# ===================================================================


@pytest.fixture
def make_package(tmp_path, monkeypatch):
    """Write ``{module: source}`` as a fresh package under *root*."""
    created: list[str] = []

    def make(files: dict[str, str], *, root=None) -> str:
        base = root if root is not None else tmp_path
        base.mkdir(parents=True, exist_ok=True)
        monkeypatch.syspath_prepend(str(base))
        name = f"cs_mro_{uuid.uuid4().hex[:10]}"
        pkg_dir = base / name
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        for module, source in files.items():
            (pkg_dir / f"{module}.py").write_text(textwrap.dedent(source))
        created.append(name)
        importlib.invalidate_caches()
        return name

    yield make
    for name in created:
        for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[module]


@pytest.fixture
def third_party(make_package, tmp_path):
    """A package that classifies as third-party (lives under site-packages)."""
    pkg = make_package(
        {
            "ext": """
            class ExtBase:
                def __init__(self, host: str = "localhost", **kwargs):
                    super().__init__(**kwargs)
                    self.host = host

            class ExtRoot:
                def __init__(self, port: int = 80):
                    self.port = port

            class ExtMid(ExtBase, ExtRoot):
                pass
            """
        },
        root=tmp_path / "site-packages",
    )
    return importlib.import_module(f"{pkg}.ext")


# ===================================================================
# X1: init-less classes are skipped BEFORE scope/depth
# ===================================================================


class TestInitlessClassesNeverCutTheChain:
    """A Generic / ABC / Protocol mixin used to end the walk (scope was
    checked before ``__init__``), dropping parent knobs and flipping the
    config to ``extra='allow'``."""

    def _root(self) -> type:
        class Root:
            def __init__(self, a: int = 1):
                pass

        return Root

    def test_generic_mixin(self) -> None:
        Root = self._root()

        class Mid(Generic[_T], Root):
            def __init__(self, b: int = 2, **kwargs: Any):
                super().__init__(**kwargs)

        assert Mid.__mro__[1] is Generic  # the mixin precedes Root
        result = collect_mro_params(Mid, scope="local")
        assert [p.name for p in result.params] == ["a"]
        assert result.fully_resolved is True

    def test_abc_mixin(self) -> None:
        Root = self._root()

        class Mid(abc.ABC, Root):
            def __init__(self, b: int = 2, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Mid, scope="local")
        assert [p.name for p in result.params] == ["a"]
        assert result.fully_resolved is True

    def test_protocol_mixin(self) -> None:
        Root = self._root()

        class Runner(Protocol):
            def run(self) -> None: ...

        class Mid(Runner, Root):
            def __init__(self, b: int = 2, **kwargs: Any):
                super().__init__(**kwargs)

            def run(self) -> None: ...

        # C3 puts typing.Protocol and Generic (stdlib) before Root.
        assert Mid.__mro__.index(Protocol) < Mid.__mro__.index(Root)
        result = collect_mro_params(Mid, scope="local")
        assert [p.name for p in result.params] == ["a"]
        assert result.fully_resolved is True

    def test_initless_classes_do_not_count_toward_depth(self) -> None:
        class Grand:
            def __init__(self, g: int = 0):
                pass

        class Parent(abc.ABC, Generic[_T], Grand):
            def __init__(self, p: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Generic[_T], Parent[int]):
            def __init__(self, c: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="local", depth=1)
        assert [p.name for p in result.params] == ["p"]
        assert result.fully_resolved is False
        assert result.stop_reason == "depth"

    def test_bridge_does_not_cut_and_third_party_does(self, third_party) -> None:
        from conscribe import create_registrar
        from conscribe.registration.registry import _REGISTRY_INDEX, _deregister

        @runtime_checkable
        class P(Protocol):
            def run(self) -> None: ...

        name = f"cs_mro_bridge_{uuid.uuid4().hex[:6]}"
        reg = create_registrar(name, P)
        try:
            bridge = reg.bridge(third_party.ExtBase)
            assert is_bridge_class(bridge)
            assert classify_class_scope(bridge) == "local"

            class Impl(bridge):
                __registry_key__ = "impl"

                def __init__(self, retries: int = 3, **kwargs: Any):
                    super().__init__(**kwargs)

                def run(self) -> None: ...

            local = collect_mro_params(Impl, scope="local")
            assert local.params == []
            assert local.fully_resolved is False
            assert local.stop_reason == "scope"

            everything = collect_mro_params(Impl, scope="all")
            assert [p.name for p in everything.params] == ["host"]
            assert everything.fully_resolved is True
        finally:
            if name in _REGISTRY_INDEX:
                _deregister(name)


# ===================================================================
# Unreadable signatures are truncation, never the natural end
# ===================================================================


class _Opaque:
    """A descriptor whose call signature ``inspect`` cannot read."""

    def __init__(self, func: Any) -> None:
        self.func = func

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    @property
    def __signature__(self) -> Any:
        raise ValueError("no signature")


class TestUnreadableSignatures:
    def test_unreadable_parent(self) -> None:
        class Parent:
            @_Opaque
            def __init__(self, a, b):
                pass

        class Mid(Parent):
            def __init__(self, m: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        class Child(Mid):
            def __init__(self, c: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        result = collect_mro_params(Child, scope="all")
        assert [p.name for p in result.params] == ["m"]
        assert result.fully_resolved is False
        assert result.stop_reason == "unreadable"
        assert result.unreadable == (Parent,)

        chain = walk_init_chain(Child)
        assert chain.names() == ("c", "m")
        assert chain.stopped_at is Parent and chain.fully_resolved is False

    def test_unreadable_definer(self) -> None:
        class Opaque:
            @_Opaque
            def __init__(self, a):
                pass

        chain = walk_init_chain(Opaque)
        assert chain.definer is Opaque
        assert chain.params == ()
        assert chain.fully_resolved is False and chain.stop_reason == "unreadable"
        result = collect_mro_params(Opaque)
        assert result.fully_resolved is False and result.unreadable == (Opaque,)


# ===================================================================
# Scope classification (2.0 additions)
# ===================================================================


class TestClassifyScopeEdges:
    def test_dynamic_module_without_file_is_local(self) -> None:
        module = types.ModuleType("cs_mro_dynamic_module")
        exec("class Dyn:\n    def __init__(self, a=1):\n        pass\n", module.__dict__)
        sys.modules[module.__name__] = module
        try:
            assert classify_class_scope(module.Dyn) == "local"
        finally:
            del sys.modules[module.__name__]

    def test_site_packages_class(self, third_party) -> None:
        assert classify_class_scope(third_party.ExtBase) == "third_party"
        assert _extract_package_name(third_party.ExtBase) == third_party.__name__.split(".")[0]

    def test_builtin_is_stdlib(self) -> None:
        assert classify_class_scope(object) == "stdlib"
        assert classify_class_scope(Exception) == "stdlib"


# ===================================================================
# Runtime view (walk_init_chain, scope-free)
# ===================================================================


class TestWalkInitChain:
    def test_runtime_view_ignores_scope(self, third_party) -> None:
        class Mine(third_party.ExtMid):
            def __init__(self, mine: int = 0, **kwargs: Any):
                super().__init__(**kwargs)

        runtime = walk_init_chain(Mine)
        assert runtime.names() == ("mine", "host", "port")
        assert runtime.fully_resolved is True
        assert runtime.definers == (Mine, third_party.ExtBase, third_party.ExtRoot)
        assert Mine(host="h", port=1).port == 1  # the runtime agrees

        scoped = walk_init_chain(Mine, scope="local")
        assert scoped.names() == ("mine",)
        assert scoped.stop_reason == "scope" and scoped.stopped_at is third_party.ExtBase

    def test_kind_and_lookup(self) -> None:
        class Plain:
            def __init__(self, a: int, b: str = "x"):
                pass

        chain = walk_init_chain(Plain)
        assert isinstance(chain, InitChain)
        assert (chain.kind, chain.definer, chain.var_keyword) == ("init", Plain, False)
        assert chain.get("b").default == "x"
        assert chain.get("missing") is None
        assert chain.get("a").source == "init" and chain.get("a").level == 0

    def test_no_init(self) -> None:
        class Bare:
            pass

        chain = walk_init_chain(Bare)
        assert (chain.kind, chain.definer, chain.params, chain.fully_resolved) == (
            "none",
            None,
            (),
            True,
        )

    def test_dataclass_params_know_their_declaring_class(self) -> None:
        @dataclasses.dataclass
        class Base:
            a: int = 1
            items: list = dataclasses.field(default_factory=list)

        @dataclasses.dataclass
        class Child(Base):
            a: int = 2  # re-declared: Child owns it now
            b: str = "x"

        chain = walk_init_chain(Child)
        assert chain.kind == "dataclass"
        owners = {p.name: p.declared_on for p in chain.params}
        assert owners == {"a": Child, "items": Base, "b": Child}
        assert chain.get("items").default is HAS_DEFAULT_FACTORY
        assert chain.get("items").field.default_factory is list
        assert dataclass_field_owner(Child, "items") is Base
        assert dataclass_field_owner(Child, "nope") is None

    def test_base_model_terminal_contributes_fields(self) -> None:
        class Model(BaseModel):
            z: float = 0.0
            extra_knob: int = 1

            def __init__(self, z: float = 0.0, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        runtime = walk_init_chain(Model)
        assert runtime.names() == ("z", "extra_knob")
        assert runtime.fully_resolved is True
        assert runtime.get("extra_knob").source == "model"
        # pydantic's BaseModel ends the chain naturally wherever pydantic is
        # installed: scope and depth never cut the leaf model's own fields.
        for scope in ("local", []):
            scoped = collect_mro_params(Model, scope=scope)
            assert scoped.fully_resolved is True
            assert [p.name for p in scoped.params] == ["extra_knob"]
            assert walk_init_chain(Model, scope=scope).terminal is BaseModel
        shallow = walk_init_chain(Model, scope="local", depth=0)
        assert shallow.names() == ("z", "extra_knob")
        assert shallow.fully_resolved is True and shallow.terminal is BaseModel

    def test_model_fast_path(self) -> None:
        class Model(BaseModel):
            a: int = 1
            b: list[int] = Field(default_factory=list)

        assert model_init_only_forwards(Model)
        chain = walk_init_chain(Model)
        assert chain.kind == "model"
        assert chain.names() == ("a", "b")
        assert chain.get("b").default is HAS_DEFAULT_FACTORY
        assert chain.get("a").parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert isinstance(chain.get("a").field, FieldInfo)

    def test_pydantic_dataclass(self) -> None:
        @pydantic_dataclass
        class PDC:
            x: int = 1
            y: str = Field("y", init=False)

        chain = walk_init_chain(PDC)
        assert chain.kind == "pydantic_dataclass"
        assert chain.names() == ("x",)
        assert chain.get("x").default == 1

    def test_protocol_copy_rule_is_instantiation_independent(self) -> None:
        @runtime_checkable
        class Runner(Protocol):
            def run(self) -> None: ...

        class Base:
            def __init__(self, a: int = 1, **kwargs: Any):
                pass

        class Impl(Runner, Base):
            def run(self) -> None: ...

        assert is_placeholder_init(Runner.__dict__["__init__"])
        assert own_init(Runner) is None
        before = (constructor_definer(Impl), walk_init_chain(Impl).names())
        Impl()
        assert Impl.__dict__["__init__"] is Base.__dict__["__init__"]  # typing's copy
        assert own_init(Impl) is None
        assert (constructor_definer(Impl), walk_init_chain(Impl).names()) == before
        assert before == (Base, ("a",))
        # The 1.x helper keeps its 1.x answer.
        assert find_init_definer(Impl) is Impl

    @pytest.mark.skipif(
        sys.version_info >= (3, 12), reason="typing_extensions.Protocol is typing's on 3.12+"
    )
    def test_swallowing_placeholder(self) -> None:  # pragma: no cover - 3.10/3.11
        import typing_extensions

        class Runner(typing_extensions.Protocol):
            def run(self) -> None: ...

        class Base:
            def __init__(self, a: int = 1):
                self.a = a

        class Impl(Runner, Base):
            def run(self) -> None: ...

        assert is_swallowing_placeholder(Runner.__dict__["__init__"])
        chain = walk_init_chain(Impl)
        assert chain.params == ()
        assert chain.fully_resolved is False
        assert chain.stop_reason == "swallowing_placeholder"
        assert not hasattr(Impl(a=5), "a")  # the runtime really drops it

    def test_named_params_drops_instance_param_by_position(self) -> None:
        def init(this, a, /, b, *args, c=1, **kwargs):
            pass

        assert [p.name for p in named_params(inspect.signature(init))] == ["a", "b", "c"]


# ===================================================================
# D3: owner-aware hints in the legacy view
# ===================================================================


class TestOwnerAwareHints:
    def test_hints_resolve_per_declaring_module(self, make_package) -> None:
        pkg = make_package(
            {
                "grand": """
                from __future__ import annotations
                from dataclasses import dataclass
                from typing import TYPE_CHECKING, Annotated, Optional
                from pydantic import Field

                if TYPE_CHECKING:
                    from cs_absent_heavy import Heavy

                @dataclass
                class Grand:
                    knob: Annotated[int, Field(ge=1, description="kn")] = 3
                    heavy: Optional[Heavy] = None
                """,
                "mid": """
                from dataclasses import dataclass
                from .grand import Grand

                @dataclass
                class Mid(Grand):
                    tag: str = "t"
                """,
                "child": """
                from .mid import Mid

                class Child(Mid):
                    def __init__(self, verbose: bool = False, **kwargs):
                        super().__init__(**kwargs)
                """,
            }
        )
        grand = importlib.import_module(f"{pkg}.grand").Grand
        child = importlib.import_module(f"{pkg}.child").Child
        result = collect_mro_params(child, scope="all")
        assert [p.name for p in result.params] == ["knob", "heavy", "tag"]
        assert result.declared_on["knob"] is grand
        knob = result.hints["knob"]
        assert get_origin(knob) is Annotated and get_args(knob)[0] is int
        # One unresolvable hint no longer turns every hint into a raw string.
        assert result.hints["heavy"] == "Optional[Heavy]"
        assert result.hints["tag"] is str
        assert [p.name for p in result.init_params] == ["knob", "heavy", "tag"]

    def test_new_result_fields_have_defaults(self) -> None:
        result = MROCollectionResult()
        assert result.declared_on == {}
        assert result.unreadable == ()
        assert result.stop_reason is None
        assert result.init_params == ()


# ===================================================================
# Review fixes (2.0.0 release review)
# ===================================================================


class SetupMixin:
    """Cooperative mixin: forwards every keyword to the next constructor."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class TestPydanticEndsTheChain:
    """B4: a registered BaseModel keeps its fields with a cooperative mixin
    or a parent model ``__init__(self, flag=False, **data)``."""

    def test_forwarding_mixin_takes_the_model_fast_path(self) -> None:
        class Tool(SetupMixin, BaseModel):
            name: str = Field("t", description="tool name")
            retries: int = 3

        class Tool2(BaseModel):
            name: str = "t"

            def __init__(self, **data: Any) -> None:
                super().__init__(**data)

        class Tool3(SetupMixin, Tool2):
            retries: int = 3

        for cls, names in ((Tool, ("name", "retries")), (Tool3, ("name", "retries"))):
            assert model_init_only_forwards(cls)
            for scope in (None, "local", "all", []):
                chain = walk_init_chain(cls, scope=scope)
                assert chain.kind == "model"
                assert chain.names() == names
                assert chain.fully_resolved
        assert Tool(name="x", retries=5).retries == 5  # what the config promises

    def test_named_model_init_parameter_then_base_model(self) -> None:
        class Base(BaseModel):
            a: int = 1

            def __init__(self, flag: bool = False, **data: Any) -> None:
                super().__init__(**data)

        class Leaf(Base):
            b: int = 2

        assert not model_init_only_forwards(Leaf)
        for scope, depth in ((None, None), ("local", None), ("local", 0), ([], 0)):
            chain = walk_init_chain(Leaf, scope=scope, depth=depth)
            assert chain.names() == ("flag", "a", "b"), (scope, depth)
            assert chain.terminal is BaseModel and chain.fully_resolved
        assert Leaf(flag=True, a=3).a == 3

    def test_pydantic_dataclass_parent_is_not_cut_by_scope(self) -> None:
        @pydantic_dataclass
        class Settings:
            level: int = 1

        class Child(Settings):
            def __init__(self, z: int = 0, **kwargs: Any) -> None:
                super().__init__(**kwargs)

        chain = walk_init_chain(Child, scope=[], depth=0)
        assert chain.names() == ("z", "level")
        assert chain.terminal is Settings


@pytest.mark.skipif(importlib.util.find_spec("attrs") is None, reason="attrs not installed")
class TestAttrsAliases:
    """m3: an attrs field passed to ``__init__`` under an alias resolves in
    the module of the class that annotates the attribute."""

    def test_private_and_explicit_aliases(self, make_package) -> None:
        pkg = make_package(
            {
                "base": """
                from __future__ import annotations
                import attrs
                from decimal import Decimal

                @attrs.define
                class BaseA:
                    _budget: Decimal = Decimal("1.5")
                    public: Decimal = Decimal("2")
                    renamed: int = attrs.field(default=3, alias="count")
                    tags: list = attrs.Factory(list)
                """,
                "child": """
                from __future__ import annotations
                import attrs
                from .base import BaseA

                @attrs.define
                class ChildA(BaseA):
                    extra: int = 0
                """,
            }
        )
        base = importlib.import_module(f"{pkg}.base")
        child = importlib.import_module(f"{pkg}.child").ChildA
        chain = walk_init_chain(child)
        assert chain.names() == ("budget", "public", "count", "tags", "extra")
        budget = chain.get("budget")
        assert budget.attribute == "_budget" and budget.declared_on is base.BaseA
        assert budget.resolve().value is base.Decimal
        count = chain.get("count")
        assert count.attribute == "renamed" and count.resolve().value is int
        assert chain.get("public").attribute is None
        assert budget.declared_default().value == base.Decimal("1.5")
        assert count.declared_default().value == 3
        assert chain.get("tags").declared_default().factory is list


class TestDeclaredDefault:
    """The real default of a parameter, never a signature stand-in (M6)."""

    def test_dataclass_value_and_factory(self) -> None:
        @dataclasses.dataclass
        class D:
            a: int = 1
            b: list[int] = dataclasses.field(default_factory=list)
            c: str = dataclasses.field(default="x")
            d: int = 0

        chain = walk_init_chain(D)
        assert chain.get("a").declared_default() == DeclaredDefault(value=1)
        assert chain.get("b").default is HAS_DEFAULT_FACTORY  # the signature stand-in
        assert chain.get("b").declared_default() == DeclaredDefault(factory=list)
        assert chain.get("c").declared_default().value == "x"

    def test_hand_written_init(self) -> None:
        info = Field(3, ge=0)

        class H:
            def __init__(self, a: int, b: int = 2, c: int = info) -> None: ...

        chain = walk_init_chain(H)
        assert chain.get("a").declared_default() == DeclaredDefault()
        assert chain.get("b").declared_default().value == 2
        declared = chain.get("c").declared_default()
        assert declared.field_info is info and declared.value is inspect.Parameter.empty

    def test_model_field(self) -> None:
        class M(BaseModel):
            a: int = 1

        param = walk_init_chain(M).get("a")
        assert param.declared_default().field_info is M.model_fields["a"]

    @pytest.mark.parametrize(
        "value, expected",
        [
            (inspect.Parameter.empty, True),
            (dataclasses.MISSING, True),
            (..., True),
            (None, False),
            (0, False),
        ],
    )
    def test_is_no_default(self, value: Any, expected: bool) -> None:
        from pydantic_core import PydanticUndefined

        from conscribe.wiring import WiredField

        assert is_no_default(value) is expected
        assert is_no_default(PydanticUndefined)
        assert is_no_default(WiredField("x", "reg"))


class TestScopeAndDepthValidation:
    """m4: invalid MRO settings are refused, never silently reinterpreted."""

    @pytest.mark.parametrize("value", ["local", "third_party", "all", [], ["pydantic"], ("a", "b")])
    def test_valid_scopes(self, value: Any) -> None:
        assert mro_scope_problem(value) is None

    @pytest.mark.parametrize(
        "value, fragment",
        [
            ("locl", "did you mean 'local'"),
            ("everything", "unknown MRO scope"),
            ([""], "non-empty strings"),
            (["ok", 3], "non-empty strings"),
            (3, "got int"),
        ],
    )
    def test_invalid_scopes(self, value: Any, fragment: str) -> None:
        problem = mro_scope_problem(value)
        assert problem is not None and fragment in problem
        with pytest.raises(ValueError, match="MRO scope"):
            walk_init_chain(int, scope=value)

    @pytest.mark.parametrize("value", [None, 0, 3])
    def test_valid_depths(self, value: Any) -> None:
        assert mro_depth_problem(value) is None

    @pytest.mark.parametrize("value", ["2", -1, True, 1.5])
    def test_invalid_depths(self, value: Any) -> None:
        assert mro_depth_problem(value) is not None
        with pytest.raises(ValueError, match="MRO depth"):
            walk_init_chain(int, depth=value)


def _fast_name_cases() -> list[type]:
    """Constructor shapes for the names fast path (each compared with the walk)."""
    import functools
    import typing_extensions

    class Plain:
        def __init__(self, a: int, b: int = 1, *, c: str = "") -> None: ...

    class Child(Plain):
        def __init__(self, d: int = 0, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    class PosOnly:
        def __init__(self, a: int, /, b: int = 1, **kwargs: Any) -> None: ...

    class PosOnlyShadow(Plain):
        def __init__(self, a: int = 0, /, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    class Args:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    class NoInit:
        pass

    @dataclasses.dataclass
    class DC:
        x: int = 1
        _: dataclasses.KW_ONLY
        y: int = 2
        z: dataclasses.InitVar[int] = 3

    class DCChild(DC):
        def __init__(self, w: int = 0, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    class Wrapped:
        @deco
        def __init__(self, q: int = 1) -> None: ...

    class Model(SetupMixin, BaseModel):
        m: int = 1

    @pydantic_dataclass
    class PDC:
        p: int = 1

    class Unreadable:
        __init__ = _Opaque(lambda self: None)

    @typing_extensions.runtime_checkable
    class Proto(typing_extensions.Protocol):
        def run(self) -> None: ...

    class Impl(Proto, Plain):
        def run(self) -> None: ...

    return [Plain, Child, PosOnly, PosOnlyShadow, Args, NoInit, DC, DCChild, Wrapped, Model, PDC, Unreadable, Impl]


@pytest.mark.parametrize("cls", _fast_name_cases(), ids=lambda c: c.__name__)
def test_keyword_param_names_matches_the_walk(cls: type) -> None:
    """The class-creation fast path never answers differently from the walk."""
    assert keyword_param_names(cls) == walk_init_chain(cls).names()
