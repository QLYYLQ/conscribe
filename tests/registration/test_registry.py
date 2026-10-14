"""Tests for conscribe.registration.registry.LayerRegistry.

Covers:
- CRUD operations: add, get, get_or_none, remove, keys, items
- Duplicate key detection (DuplicateKeyError)
- Missing key detection (KeyNotFoundError)
- runtime_check: pass/fail, cache behavior (positive/negative/invalidation)
- Snapshot isolation (keys/items return copies)
- Thread safety (5 dedicated tests)
- Real scenario: Agent registry with multiple implementations
"""
from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import pytest

from conscribe.registration.registry import LayerRegistry
from conscribe.exceptions import (
    DuplicateKeyError,
    KeyNotFoundError,
    InvalidProtocolError,
)


# ---------------------------------------------------------------------------
# Local Protocols and stubs (defined in class to avoid metaclass side effects)
# ---------------------------------------------------------------------------

@runtime_checkable
class WorkerProtocol(Protocol):
    def do_work(self) -> str: ...


@runtime_checkable
class AgentLikeProtocol(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


# ===================================================================
# Construction
# ===================================================================

class TestLayerRegistryConstruction:
    """Tests for LayerRegistry.__init__."""

    def test_create_with_valid_protocol(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        assert registry.name == "test"

    def test_create_with_non_runtime_checkable_raises(self) -> None:
        """Protocol without @runtime_checkable should raise InvalidProtocolError."""

        class BadProtocol(Protocol):
            def run(self) -> None: ...

        with pytest.raises((TypeError, InvalidProtocolError)):
            LayerRegistry("test", BadProtocol)

    def test_empty_registry_has_no_keys(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        assert registry.keys() == []

    def test_empty_registry_has_no_items(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        assert registry.items() == []


# ===================================================================
# CRUD operations
# ===================================================================

class TestLayerRegistryCRUD:
    """Tests for add, get, get_or_none, remove, keys, items."""

    def test_add_and_get(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class MyWorker:
            def do_work(self) -> str:
                return "done"

        registry.add("my_worker", MyWorker)
        assert registry.get("my_worker") is MyWorker

    def test_add_multiple_keys(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class WorkerA:
            def do_work(self) -> str:
                return "a"

        class WorkerB:
            def do_work(self) -> str:
                return "b"

        registry.add("a", WorkerA)
        registry.add("b", WorkerB)
        assert set(registry.keys()) == {"a", "b"}

    def test_get_or_none_existing(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class W:
            def do_work(self) -> str:
                return "w"

        registry.add("w", W)
        assert registry.get_or_none("w") is W

    def test_get_or_none_missing(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        assert registry.get_or_none("nonexistent") is None

    def test_remove_existing(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class W:
            def do_work(self) -> str:
                return "w"

        registry.add("w", W)
        registry.remove("w")
        assert registry.get_or_none("w") is None

    def test_remove_missing_raises_key_error(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        with pytest.raises((KeyError, KeyNotFoundError)):
            registry.remove("nonexistent")

    def test_items_returns_all_entries(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("a", A)
        registry.add("b", B)
        items = registry.items()
        assert set(items) == {("a", A), ("b", B)}

    def test_keys_returns_list(self) -> None:
        """keys() should return a list, not a dict_keys view."""
        registry = LayerRegistry("test", WorkerProtocol)

        class W:
            def do_work(self) -> str:
                return "w"

        registry.add("w", W)
        keys = registry.keys()
        assert isinstance(keys, list)

    def test_items_returns_list(self) -> None:
        """items() should return a list, not a dict_items view."""
        registry = LayerRegistry("test", WorkerProtocol)
        items = registry.items()
        assert isinstance(items, list)


# ===================================================================
# Duplicate key detection
# ===================================================================

class TestDuplicateKeyDetection:
    """Tests for DuplicateKeyError on conflicting keys."""

    def test_duplicate_key_raises(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("same_key", A)
        with pytest.raises(DuplicateKeyError, match="Duplicate key"):
            registry.add("same_key", B)

    def test_duplicate_error_mentions_key(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("foo_key", A)
        with pytest.raises(DuplicateKeyError, match="foo_key"):
            registry.add("foo_key", B)

    def test_duplicate_error_mentions_conflicting_classes(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class OriginalWorker:
            def do_work(self) -> str:
                return "orig"

        class ConflictingWorker:
            def do_work(self) -> str:
                return "conflict"

        registry.add("worker", OriginalWorker)
        with pytest.raises(DuplicateKeyError) as exc_info:
            registry.add("worker", ConflictingWorker)
        msg = str(exc_info.value)
        assert "OriginalWorker" in msg
        assert "ConflictingWorker" in msg

    def test_add_after_remove_same_key_succeeds(self) -> None:
        """After removing a key, adding it again with a new class should succeed."""
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("key", A)
        registry.remove("key")
        registry.add("key", B)
        assert registry.get("key") is B


# ===================================================================
# Missing key detection
# ===================================================================

class TestMissingKeyDetection:
    """Tests for KeyNotFoundError on missing keys."""

    def test_get_missing_key_raises(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        with pytest.raises((KeyError, KeyNotFoundError)):
            registry.get("nonexistent")

    def test_get_missing_key_error_message_lists_available(self) -> None:
        """Error message should list available keys to help the user."""
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        registry.add("alpha", A)
        with pytest.raises((KeyError, KeyNotFoundError), match="alpha"):
            registry.get("beta")

    def test_get_missing_key_error_mentions_registry_name(self) -> None:
        registry = LayerRegistry("my_layer", WorkerProtocol)
        with pytest.raises((KeyError, KeyNotFoundError), match="my_layer"):
            registry.get("missing")


# ===================================================================
# runtime_check
# ===================================================================

class TestRuntimeCheck:
    """Tests for Protocol compliance checking via runtime_check."""

    def test_compliant_class_passes(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class Good:
            def do_work(self) -> str:
                return "good"

        # Should not raise
        registry.runtime_check(Good)

    def test_noncompliant_class_raises(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class Bad:
            pass

        with pytest.raises(TypeError, match="do_work"):
            registry.runtime_check(Bad)

    def test_partially_compliant_class_raises(self) -> None:
        """Class that has some but not all methods of the protocol."""
        @runtime_checkable
        class MultiMethodProtocol(Protocol):
            def method_a(self) -> str: ...
            def method_b(self) -> str: ...

        registry = LayerRegistry("test", MultiMethodProtocol)

        class Partial:
            def method_a(self) -> str:
                return "a"

        with pytest.raises(TypeError, match="method_b"):
            registry.runtime_check(Partial)

    def test_add_with_protocol_check_true(self) -> None:
        """add() with protocol_check=True should validate the class."""
        registry = LayerRegistry("test", WorkerProtocol)

        class Bad:
            pass

        with pytest.raises(TypeError):
            registry.add("bad", Bad, protocol_check=True)

    def test_add_with_protocol_check_false_skips_validation(self) -> None:
        """add() with protocol_check=False (default) skips Protocol check."""
        registry = LayerRegistry("test", WorkerProtocol)

        class NotCompliant:
            pass

        # Should NOT raise -- protocol_check defaults to False
        registry.add("not_compliant", NotCompliant)
        assert registry.get("not_compliant") is NotCompliant

    def test_runtime_check_positive_cache(self) -> None:
        """Once a class passes runtime_check, subsequent checks should pass immediately."""
        registry = LayerRegistry("test", WorkerProtocol)

        class Good:
            def do_work(self) -> str:
                return "good"

        registry.runtime_check(Good)
        # Second call should also pass (from cache)
        registry.runtime_check(Good)

    def test_runtime_check_negative_cache_invalidation(self) -> None:
        """After remove() bumps the invalidation counter, negative cache is stale."""
        registry = LayerRegistry("test", WorkerProtocol)

        class Bad:
            pass

        # First check: fails, goes into negative cache
        with pytest.raises(TypeError):
            registry.runtime_check(Bad)

        # Remove something to bump the invalidation counter
        class Dummy:
            def do_work(self) -> str:
                return "dummy"

        registry.add("dummy", Dummy)
        registry.remove("dummy")

        # After invalidation, the negative cache should be stale
        # Bad still doesn't comply, so it should still raise
        with pytest.raises(TypeError):
            registry.runtime_check(Bad)


# ===================================================================
# Snapshot isolation
# ===================================================================

class TestSnapshotIsolation:
    """Tests that keys() and items() return snapshots, not live views."""

    def test_keys_snapshot_not_affected_by_later_add(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("a", A)
        keys_snapshot = registry.keys()
        registry.add("b", B)
        # Snapshot should still only contain "a"
        assert keys_snapshot == ["a"]

    def test_items_snapshot_not_affected_by_later_add(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("a", A)
        items_snapshot = registry.items()
        registry.add("b", B)
        assert len(items_snapshot) == 1


# ===================================================================
# Thread safety (5 tests)
# ===================================================================

class TestThreadSafety:
    """Thread safety tests using threading.Thread with barriers."""

    def test_concurrent_add_100_threads(self) -> None:
        """100 threads simultaneously add different keys -- all must succeed."""
        registry = LayerRegistry("test", WorkerProtocol)
        barrier = threading.Barrier(100)
        errors: list[Exception] = []

        def add_class(index: int) -> None:
            # Create class inside thread to avoid sharing
            cls = type(f"Worker{index}", (), {"do_work": lambda self: str(index)})
            barrier.wait()
            try:
                registry.add(f"worker_{index}", cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_class, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.keys()) == 100

    def test_concurrent_add_same_key_race(self) -> None:
        """Multiple threads add the same key: exactly 1 succeeds, rest get DuplicateKeyError."""
        registry = LayerRegistry("test", WorkerProtocol)
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        successes: list[int] = []
        failures: list[int] = []

        def add_same_key(index: int) -> None:
            cls = type(f"Worker{index}", (), {"do_work": lambda self: str(index)})
            barrier.wait()
            try:
                registry.add("contested_key", cls)
                successes.append(index)
            except DuplicateKeyError:
                failures.append(index)

        threads = [
            threading.Thread(target=add_same_key, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1
        assert len(failures) == num_threads - 1

    def test_concurrent_get_during_add(self) -> None:
        """Reader threads calling get() while writer threads call add() -- no inconsistent state."""
        registry = LayerRegistry("test", WorkerProtocol)

        class Seed:
            def do_work(self) -> str:
                return "seed"

        registry.add("seed", Seed)

        barrier = threading.Barrier(40)
        read_results: list[type | None] = []
        write_errors: list[Exception] = []

        def reader() -> None:
            barrier.wait()
            for _ in range(50):
                result = registry.get_or_none("seed")
                read_results.append(result)

        def writer(index: int) -> None:
            cls = type(f"W{index}", (), {"do_work": lambda self: str(index)})
            barrier.wait()
            try:
                registry.add(f"w_{index}", cls)
            except Exception as e:
                write_errors.append(e)

        readers = [threading.Thread(target=reader) for _ in range(20)]
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        all_threads = readers + writers

        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        assert len(write_errors) == 0
        # Every read of "seed" should return the Seed class (never None)
        assert all(r is Seed for r in read_results)

    def test_concurrent_remove_and_add(self) -> None:
        """Threads removing and adding concurrently should not corrupt state."""
        registry = LayerRegistry("test", WorkerProtocol)

        # Pre-populate
        for i in range(50):
            cls = type(f"Pre{i}", (), {"do_work": lambda self: "pre"})
            registry.add(f"pre_{i}", cls)

        barrier = threading.Barrier(50)
        errors: list[Exception] = []

        def remove_then_add(index: int) -> None:
            barrier.wait()
            try:
                registry.remove(f"pre_{index}")
                new_cls = type(f"New{index}", (), {"do_work": lambda self: "new"})
                registry.add(f"new_{index}", new_cls)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=remove_then_add, args=(i,))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All "pre_*" removed, all "new_*" added
        keys = registry.keys()
        assert all(k.startswith("new_") for k in keys)
        assert len(keys) == 50

    def test_concurrent_keys_is_snapshot(self) -> None:
        """keys() called concurrently with add() returns consistent snapshots."""
        registry = LayerRegistry("test", WorkerProtocol)
        barrier = threading.Barrier(20)
        snapshots: list[list[str]] = []

        def add_keys(start: int) -> None:
            barrier.wait()
            for i in range(10):
                cls = type(f"W{start}_{i}", (), {"do_work": lambda self: ""})
                try:
                    registry.add(f"w_{start}_{i}", cls)
                except DuplicateKeyError:
                    pass

        def take_snapshot() -> None:
            barrier.wait()
            for _ in range(10):
                snap = registry.keys()
                snapshots.append(snap)

        writers = [threading.Thread(target=add_keys, args=(i,)) for i in range(10)]
        readers = [threading.Thread(target=take_snapshot) for _ in range(10)]

        for t in writers + readers:
            t.start()
        for t in writers + readers:
            t.join()

        # Every snapshot should be a valid list (not corrupted)
        for snap in snapshots:
            assert isinstance(snap, list)


# ===================================================================
# Real scenario: Agent registry with multiple implementations
# ===================================================================

class TestAgentRegistryScenario:
    """Simulates Alice's Agent layer: register multiple agent implementations."""

    def test_agent_registry_multi_implementation(self) -> None:
        registry = LayerRegistry("agent", AgentLikeProtocol)

        class BrowserUseAgentStub:
            async def step(self, task: str) -> str:
                return "browser_use result"
            def reset(self) -> None:
                pass

        class SkyvernAgentStub:
            async def step(self, task: str) -> str:
                return "skyvern result"
            def reset(self) -> None:
                pass

        class AgentTarsAgentStub:
            async def step(self, task: str) -> str:
                return "agent_tars result"
            def reset(self) -> None:
                pass

        registry.add("browser_use", BrowserUseAgentStub)
        registry.add("skyvern", SkyvernAgentStub)
        registry.add("agent_tars", AgentTarsAgentStub)

        assert registry.get("browser_use") is BrowserUseAgentStub
        assert set(registry.keys()) == {"browser_use", "skyvern", "agent_tars"}

        # Duplicate key -> fail-fast
        class AnotherAgentStub:
            async def step(self, task: str) -> str:
                return "another"
            def reset(self) -> None:
                pass

        with pytest.raises(DuplicateKeyError, match="Duplicate key"):
            registry.add("browser_use", AnotherAgentStub)

    def test_tree_raises_on_prefix_leaf_collision(self) -> None:
        """tree() raises ValueError when a leaf class blocks nesting for a deeper key."""
        registry = LayerRegistry("test", WorkerProtocol, separator=".")

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("openai", A)
        registry.add("openai.azure", B)

        with pytest.raises(ValueError, match="Key conflict"):
            registry.tree()

    def test_dump_registry_does_not_crash(self) -> None:
        """_dump_registry is a debug method -- it should not raise."""
        registry = LayerRegistry("test", WorkerProtocol)

        class W:
            def do_work(self) -> str:
                return "w"

        registry.add("w", W)
        # Just verify it doesn't raise
        registry._dump_registry()


# ===================================================================
# remove(key, *, missing_ok)
# ===================================================================

class TestRemove:
    """remove() returns the class, is idempotent with missing_ok, and says
    why a missing key cannot be removed."""

    @staticmethod
    def _worker(name: str) -> type:
        return type(name, (), {"do_work": lambda self: name})

    def test_remove_returns_removed_class(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        worker = self._worker("W")
        registry.add("w", worker)
        assert registry.remove("w") is worker
        assert registry.keys() == []

    def test_missing_key_raises_removal_specific_key_not_found(self) -> None:
        registry = LayerRegistry("my_layer", WorkerProtocol)
        registry.add("alpha", self._worker("A"))
        with pytest.raises(KeyNotFoundError) as info:
            registry.remove("beta")
        message = str(info.value)
        assert isinstance(info.value, KeyError)
        assert info.value.key == "beta"
        assert info.value.available_keys == ["alpha"]
        assert "'beta' not found in 'my_layer'" in message
        assert "alpha" in message
        assert "missing_ok=True" in message
        assert "discover()" not in message  # that advice is for lookups

    def test_missing_ok_returns_none_twice_without_invalidating(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        counter = registry._invalidation_counter
        assert registry.remove("nope", missing_ok=True) is None
        assert registry.remove("nope", missing_ok=True) is None
        assert registry._invalidation_counter == counter

    def test_counter_bumped_only_when_something_was_removed(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        registry.add("w", self._worker("W"))
        counter = registry._invalidation_counter
        registry.remove("w")
        assert registry._invalidation_counter == counter + 1
        registry.remove("w", missing_ok=True)
        assert registry._invalidation_counter == counter + 1

    def test_removing_one_key_of_multi_key_class_keeps_the_others(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        worker = self._worker("Multi")
        registry.add("alias_a", worker)
        registry.add("alias_b", worker)
        assert registry.remove("alias_a") is worker
        assert registry.get("alias_b") is worker

    def test_hierarchical_prefix_is_not_a_key(self) -> None:
        registry = LayerRegistry("llm", WorkerProtocol, separator=".")
        registry.add("openai.azure", self._worker("Azure"))
        registry.add("openai.chat", self._worker("Chat"))
        registry.add("anthropic.claude", self._worker("Claude"))
        with pytest.raises(KeyNotFoundError, match="only a prefix") as info:
            registry.remove("openai")
        advice = str(info.value).split("only a prefix")[1]
        assert "openai.azure, openai.chat" in advice
        assert "anthropic.claude" not in advice
        assert len(registry.keys()) == 3

    def test_removing_parent_leaf_keeps_children(self) -> None:
        registry = LayerRegistry("llm", WorkerProtocol, separator=".")
        parent, child = self._worker("OpenAI"), self._worker("Azure")
        registry.add("openai", parent)
        registry.add("openai.azure", child)
        assert registry.remove("openai") is parent
        assert registry.keys() == ["openai.azure"]
        assert registry.tree() == {"openai": {"azure": child}}


# ===================================================================
# DuplicateKeyError advice
# ===================================================================

class TestDuplicateKeyAdvice:
    def test_different_classes_get_default_advice(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        class B:
            def do_work(self) -> str:
                return "b"

        registry.add("k", A)
        with pytest.raises(DuplicateKeyError, match="To fix: add __registry_key__") as info:
            registry.add("k", B)
        assert info.value.hint is None

    def test_same_class_twice_names_the_cause(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class A:
            def do_work(self) -> str:
                return "a"

        registry.add("k", A)
        with pytest.raises(DuplicateKeyError, match="listed twice"):
            registry.add("k", A)

    def test_class_body_run_twice_points_to_isolation(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        def define() -> type:
            class Again:
                def do_work(self) -> str:
                    return "again"

            return Again

        registry.add("k", define())
        with pytest.raises(DuplicateKeyError, match="isolated_registries") as info:
            registry.add("k", define())
        assert "ran a second time" in str(info.value)
        assert "To fix: add __registry_key__" not in str(info.value)

    def test_same_named_subclass_is_told_to_take_its_own_key(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)

        class Worker:
            def do_work(self) -> str:
                return "w"

        registry.add("worker", Worker)
        Sub = type("Worker", (Worker,), {"__qualname__": "Other.Worker"})
        with pytest.raises(DuplicateKeyError, match="its own __registry_key__"):
            registry.add("worker", Sub)


# ===================================================================
# Internal mutation helpers (used by registration and isolation)
# ===================================================================

class TestInternalMutations:
    @staticmethod
    def _worker(name: str) -> type:
        return type(name, (), {"do_work": lambda self: name})

    def test_replace_class_rebinds_every_key(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        old, new, other = self._worker("Old"), self._worker("New"), self._worker("Other")
        registry.add("a", old)
        registry.add("b", old)
        registry.add("c", other)
        assert registry._replace_class(old, new) == ["a", "b"]
        assert registry.items() == [("a", new), ("b", new), ("c", other)]

    def test_discard_class_removes_every_key_and_invalidates(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        gone, other = self._worker("Gone"), self._worker("Other")
        registry.add("a", gone)
        registry.add("b", other)
        registry.add("c", gone)
        counter = registry._invalidation_counter
        assert registry._discard_class(gone) == ["a", "c"]
        assert registry.keys() == ["b"]
        assert registry._invalidation_counter == counter + 1
        assert registry._discard_class(gone) == []
        assert registry._invalidation_counter == counter + 1

    def test_discard_binding_only_if_still_bound(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        first, second = self._worker("First"), self._worker("Second")
        registry.add("k", first)
        assert registry._discard_binding("k", second) is False
        assert registry._discard_binding("k", first) is True
        assert registry._discard_binding("k", first) is False
        assert registry.keys() == []

    def test_find_by_id_and_holds(self) -> None:
        registry = LayerRegistry("test", WorkerProtocol)
        worker = self._worker("W")
        registry.add("w", worker)
        assert registry._find_by_id(id(worker)) is worker
        assert registry._find_by_id(id(object())) is None
        assert registry._holds(worker)
        assert not registry._holds(self._worker("X"))


# ===================================================================
# Live registry tracking
# ===================================================================

class TestLiveRegistries:
    def test_shadowed_registry_stays_tracked(self) -> None:
        from conscribe.registration.registry import _live_registries, get_registry

        first = LayerRegistry("test_live_shadow", WorkerProtocol)
        second = LayerRegistry("test_live_shadow", WorkerProtocol)
        assert get_registry("test_live_shadow") is second
        live = _live_registries()
        assert first in live and second in live

    def test_missing_sentinel_repr(self) -> None:
        from conscribe.registration.registry import _MISSING

        assert repr(_MISSING) == "<missing>"

    def test_live_snapshot_while_another_thread_creates_registries(self) -> None:
        """Review probe 8: the live-registry WeakSet was iterated without a
        lock, so a purge / rebind / isolation snapshot failed with "Set changed
        size during iteration" while another thread created a registrar."""
        import sys

        from conscribe.registration.registry import _live_registries
        from conscribe.testing import isolated_registries

        errors: list[BaseException] = []
        stop = threading.Event()

        def snapshot() -> None:
            while not stop.is_set():
                try:
                    _live_registries()
                except RuntimeError as exc:  # pragma: no cover - the bug
                    errors.append(exc)
                    return

        old_interval = sys.getswitchinterval()
        with isolated_registries():
            keep = [LayerRegistry(f"test_race_pre_{i}", WorkerProtocol) for i in range(200)]

            def create() -> None:
                for i in range(300):
                    keep.append(LayerRegistry(f"test_race_new_{i}", WorkerProtocol))

            sys.setswitchinterval(1e-5)
            try:
                reader = threading.Thread(target=snapshot)
                writer = threading.Thread(target=create)
                reader.start()
                writer.start()
                writer.join()
                stop.set()
                reader.join()
            finally:
                sys.setswitchinterval(old_interval)
        assert errors == []
        assert len(keep) == 500
