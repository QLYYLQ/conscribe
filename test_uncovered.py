"""Standalone tests for uncovered code paths (no pytest)."""
from __future__ import annotations

import sys
import tempfile
import textwrap
import os
from typing import Protocol, runtime_checkable

from layer_registry import create_registrar
from layer_registry.registration.discover import discover
from layer_registry.registration.registry import LayerRegistry


@runtime_checkable
class SimpleProto(Protocol):
    def do_work(self) -> str: ...


@runtime_checkable
class AgentProto(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


def test_discover_module_not_package():
    """discover.py:48 — discover() a single module (no __path__) → continue."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a plain .py file, not a package
        mod_file = os.path.join(tmp, "solo_module.py")
        with open(mod_file, "w") as f:
            f.write("X = 42\n")

        sys.path.insert(0, tmp)
        try:
            result = discover("solo_module")
            assert result == ["solo_module"], f"Expected ['solo_module'], got {result}"
            print("PASS: discover() on a module (not package) hits continue branch")
        finally:
            sys.path.remove(tmp)
            sys.modules.pop("solo_module", None)


def test_discover_onerror_callback():
    """discover.py:52 — onerror callback fires when walk_packages can't import."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "err_pkg")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "__init__.py"), "w") as f:
            f.write("")

        # Create a subpackage whose __init__.py raises ImportError
        # This triggers onerror during walk_packages (not the outer try/except)
        sub = os.path.join(pkg, "bad_sub")
        os.makedirs(sub)
        with open(os.path.join(sub, "__init__.py"), "w") as f:
            f.write("raise ImportError('deliberate')\n")

        sys.path.insert(0, tmp)
        try:
            result = discover("err_pkg")
            assert "err_pkg" in result, f"Top package should be in result: {result}"
            print("PASS: discover() onerror callback triggered by bad subpackage")
        finally:
            sys.path.remove(tmp)
            for k in list(sys.modules):
                if k.startswith("err_pkg"):
                    del sys.modules[k]


def test_bridge_strategy2():
    """registrar.py:109 — bridge strategy 2: issubclass(cls.Meta, ext_meta)."""
    R = create_registrar("test", SimpleProto)

    # Create an external class whose metaclass is a PARENT of R.Meta
    # R.Meta inherits from type, so if ext_meta is type → strategy 1.
    # For strategy 2: cls.Meta must be a subclass of ext_meta.
    # We need ext_meta != type AND issubclass(cls.Meta, ext_meta).
    # Solution: make ext_meta a parent of R.Meta by creating R with
    # base_metaclass=CustomMeta, then external uses CustomMeta.

    class CustomMeta(type):
        pass

    R2 = create_registrar("test2", SimpleProto, base_metaclass=CustomMeta)
    # R2.Meta is AutoRegistrar which subclasses CustomMeta
    # So issubclass(R2.Meta, CustomMeta) is True

    class ExtWithCustomMeta(metaclass=CustomMeta):
        def do_work(self) -> str:
            return "ext"

    # ext_meta = CustomMeta, cls.Meta subclasses CustomMeta → strategy 2
    Bridge = R2.bridge(ExtWithCustomMeta)

    class Impl(Bridge):
        def do_work(self) -> str:
            return "impl"

    assert R2.get("impl") is Impl
    print("PASS: bridge strategy 2 (issubclass(cls.Meta, ext_meta)) covered")


def test_bridge_strategy3():
    """registrar.py:112 — bridge strategy 3: issubclass(ext_meta, cls.Meta)."""
    # We need a DIFFERENT registrar's Meta as parent, so the external class
    # doesn't auto-register into R's registry.
    # Strategy 3 condition: ext_meta is NOT type, NOT parent of cls.Meta,
    # but IS a subclass of cls.Meta.
    #
    # Approach: create two registrars sharing a base_metaclass chain.
    # R1's Meta is the parent. R2 uses R1.Meta as base_metaclass.
    # Then external class uses R2.Meta (subclass of R1.Meta).
    # When R1.bridge(external) is called: ext_meta=R2.Meta, cls.Meta=R1.Meta
    # issubclass(R2.Meta, R1.Meta) → True → strategy 3.

    R1 = create_registrar("s3_layer1", SimpleProto)
    R2 = create_registrar("s3_layer2", SimpleProto, base_metaclass=R1.Meta)

    # External class uses R2.Meta — this registers into R2, not R1
    class ExtBase(metaclass=R2.Meta):
        __abstract__ = True
        def do_work(self) -> str:
            return "ext"

    class ExtConcrete(ExtBase):
        def do_work(self) -> str:
            return "ext_concrete"

    # Now bridge ExtConcrete into R1
    # type(ExtConcrete) = R2.Meta, which is subclass of R1.Meta → strategy 3
    Bridge = R1.bridge(ExtConcrete)

    class S3Impl(Bridge):
        def do_work(self) -> str:
            return "s3_impl"

    assert R1.get("s3_impl") is S3Impl
    print("PASS: bridge strategy 3 (issubclass(ext_meta, cls.Meta)) covered")


def test_propagate_chains_own_init_subclass():
    """registrar.py:179 — original.__func__ path when class has its OWN __init_subclass__."""
    R = create_registrar("test", SimpleProto)
    hook_log = []

    class Base:
        def do_work(self) -> str:
            return "base"

        # This __init_subclass__ is defined ON Base itself (in its __dict__)
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            hook_log.append(cls.__name__)

    # register with propagate=True — _inject_auto_registration sees
    # original = Base.__dict__.get("__init_subclass__") → not None
    @R.register("base", propagate=True)
    class Registered(Base):
        # Registered inherits Base's __init_subclass__
        # But _inject_auto_registration looks at Registered.__dict__
        # We need __init_subclass__ in Registered's OWN __dict__
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            hook_log.append(f"registered_hook:{cls.__name__}")

    class Child(Registered):
        def do_work(self) -> str:
            return "child"

    assert R.get("child") is Child
    assert any("registered_hook" in x for x in hook_log), f"hook_log: {hook_log}"
    print("PASS: propagate chains original.__func__ from own __dict__")


def test_negative_cache_hit():
    """registry.py:93-94 — negative cache hit (class fails twice without invalidation)."""
    registry = LayerRegistry("test", SimpleProto)

    class Bad:
        pass  # Missing do_work

    # First call: computes missing, adds to negative cache, raises
    try:
        registry.runtime_check(Bad)
        assert False, "Should have raised"
    except TypeError:
        pass

    # Second call: negative cache HIT → raises from cache path (line 93-94)
    try:
        registry.runtime_check(Bad)
        assert False, "Should have raised from negative cache"
    except TypeError:
        pass

    print("PASS: negative cache hit path (lines 93-94) covered")


if __name__ == "__main__":
    tests = [
        test_discover_module_not_package,
        test_discover_onerror_callback,
        test_bridge_strategy2,
        test_bridge_strategy3,
        test_propagate_chains_own_init_subclass,
        test_negative_cache_hit,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"FAIL: {t.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        for name, err in failed:
            print(f"  FAILED: {name} — {err}")
