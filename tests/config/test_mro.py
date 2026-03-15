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
"""
from __future__ import annotations

import inspect
from typing import Any

import pytest

from conscribe.config.mro import (
    MROCollectionResult,
    _extract_package_name,
    classify_class_scope,
    collect_mro_params,
)


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

    def test_scope_local_skips_third_party_parent(self) -> None:
        """scope='local' skips parents from site-packages."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = collect_mro_params(Child, scope="local")
        # BaseModel is third-party, should be skipped
        assert result.fully_resolved is False

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

    def test_scope_truncation_not_fully_resolved(self) -> None:
        """Chain truncated by scope filtering is NOT fully resolved."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

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

    def test_scope_list_includes_specified_package(self) -> None:
        """scope=['pydantic'] includes pydantic parent params."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = collect_mro_params(Child, scope=["pydantic"])
        # BaseModel is pydantic, should be included
        # At minimum the chain should not be truncated by package filter
        # (may still be truncated by other reasons)
        # Check that we traversed into pydantic
        assert any(
            classify_class_scope(cls) == "third_party"
            for cls in result.init_definers
        ) or result.fully_resolved

    def test_scope_list_excludes_unspecified_package(self) -> None:
        """scope=['some_other'] excludes pydantic parent."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = collect_mro_params(Child, scope=["some_other"])
        # BaseModel is pydantic, not in ["some_other"], should be excluded
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

    def test_scope_list_empty_excludes_all_third_party(self) -> None:
        """scope=[] effectively means local-only (all third-party excluded)."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = collect_mro_params(Child, scope=[])
        # Empty list: no third-party packages included
        assert result.fully_resolved is False
        for cls in result.init_definers:
            assert classify_class_scope(cls) != "third_party"

    def test_scope_list_truncation_not_fully_resolved(self) -> None:
        """Truncation by package filter sets fully_resolved=False."""
        from pydantic import BaseModel

        class Child(BaseModel):
            z: float

            def __init__(self, z: float, **kwargs: Any):
                super().__init__(z=z, **kwargs)

        result = collect_mro_params(Child, scope=["nonexistent_pkg"])
        assert result.fully_resolved is False
