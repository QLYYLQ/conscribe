"""conscribe 2.0 wiring core: strict aggregated resolution (F1) and the
dict-form grammar with ``default`` / ``declare_only`` (F4).

Covers ``resolve_wiring``, ``validate_wiring``, ``check_wiring_declaration``,
``parse_wiring`` and the diagnostics they produce.  Every negative path
asserts the message, because the message is the product here.
"""
from __future__ import annotations

import dataclasses
import types
from typing import Annotated, Protocol, runtime_checkable

import pytest
from pydantic import AfterValidator, BeforeValidator

import conscribe
from conscribe import check_wiring_declaration, create_registrar, validate_wiring
from conscribe.discover import clear_import_failures
from conscribe.exceptions import WIRING_PROBLEM_KINDS, WiringProblem, WiringResolutionError
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister
from conscribe.wiring import (
    ResolvedWiring,
    WiredField,
    WiringSpec,
    expand_relative_keys,
    inject_wired_descriptors,
    parse_wiring,
    resolve_wiring,
)


@runtime_checkable
class _P(Protocol):
    def run(self) -> None: ...


@pytest.fixture(autouse=True)
def _isolate():
    """Registries prefixed ``v2w_`` and discover()'s failure record are per test."""
    clear_import_failures()
    yield
    clear_import_failures()
    for name in list(_REGISTRY_INDEX):
        if name.startswith("v2w_"):
            _deregister(name)


def _registrar(name: str, keys: list[str] = (), *, separator: str = "") -> type:
    """A registrar named *name* holding one dummy class per key."""
    reg = create_registrar(name, _P, key_separator=separator)
    for key in keys:
        dummy = type(f"Impl_{key.replace('.', '_')}", (), {"run": lambda self: None})
        reg.register(key)(dummy)
    return reg


def _problems(exc_info: pytest.ExceptionInfo) -> list[tuple[str, str]]:
    return [(p.param_name, p.kind) for p in exc_info.value.problems]


# ===================================================================
# af defect 1 — the original repro, now a hard, aggregated error
# ===================================================================


class TestAfDefectOne:
    """``conscribe_feedback.md`` §1: one typo used to drop every wired field."""

    def _layers(self):
        Strat = _registrar("v2w_strategy")
        Act = _registrar("v2w_act")
        Own = _registrar("v2w_owner")

        class StratBase(metaclass=Strat.Meta):
            __abstract__ = True

            def run(self): ...

        class Fast(StratBase):
            __registry_key__ = "fast"

        class ActBase(metaclass=Act.Meta):
            __abstract__ = True

            def run(self): ...

        class Click(ActBase):
            __registry_key__ = "click"

        class OwnerBase(metaclass=Own.Meta):
            __abstract__ = True

            def run(self): ...

        return Strat, Act, Own, StratBase, OwnerBase

    def test_bad_raises_naming_kick_and_registered_keys(self):
        Strat, Act, Own, StratBase, OwnerBase = self._layers()

        class Good(OwnerBase):
            __registry_key__ = "good"
            __wiring__ = {
                "strategy": "v2w_strategy",
                "mode": ("v2w_strategy", ["fast"]),
                "action": ("v2w_act", ["click"]),
            }
            strategy: StratBase

            def __init__(self, mode: str = "fast"): ...

        class Bad(OwnerBase):
            __registry_key__ = "bad"
            # Identical to Good except for one typo in the action list.
            __wiring__ = {
                "strategy": "v2w_strategy",
                "mode": ("v2w_strategy", ["fast"]),
                "action": ("v2w_act", ["click", "kick"]),
            }
            strategy: StratBase

            def __init__(self, mode: str = "fast"): ...

        assert set(resolve_wiring(Good)) == {"strategy", "mode", "action"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Bad)

        err = exc_info.value
        assert len(err.problems) == 1
        problem = err.problems[0]
        assert (problem.owner, problem.owner_key, problem.param_name) == (
            Bad.__qualname__, "bad", "action",
        )
        assert (problem.registry_name, problem.kind) == ("v2w_act", "unknown_key")
        # The spec's message format, verbatim; strategy and mode are not blamed.
        assert str(err) == (
            f"{Bad.__qualname__} (key 'bad') wiring 'action': "
            "unknown key 'kick' in registry 'v2w_act'; registered: ['click']"
        )
        # 1.x compatibility attributes mirror the first problem.
        assert (err.cls_name, err.param_name, err.registry_name) == (
            Bad.__qualname__, "action", "v2w_act",
        )

    def test_two_class_build_reports_both_problems_in_one_error(self):
        Strat, Act, Own, StratBase, OwnerBase = self._layers()

        class BadAction(OwnerBase):
            __registry_key__ = "bad_action"
            __wiring__ = {"action": ("v2w_act", ["click", "kick"])}

        class BadMode(OwnerBase):
            __registry_key__ = "bad_mode"
            __wiring__ = {"strategy": "v2w_strategy", "mode": ("v2w_strategy", ["fsat"])}

            def __init__(self, mode: str = "fast"): ...

        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring({"owner": Own, "strategy": Strat, "act": Act})

        err = exc_info.value
        assert [(p.owner_key, p.param_name, p.kind) for p in err.problems] == [
            ("bad_action", "action", "unknown_key"),
            ("bad_mode", "mode", "unknown_key"),
        ]
        assert str(err) == (
            "2 wiring problems:\n"
            f"  - {BadAction.__qualname__} (key 'bad_action') wiring 'action': "
            "unknown key 'kick' in registry 'v2w_act'; registered: ['click']\n"
            f"  - {BadMode.__qualname__} (key 'bad_mode') wiring 'mode': "
            "unknown key 'fsat' in registry 'v2w_strategy'; registered: ['fast']"
        )


# ===================================================================
# F1 — every problem of a class, aggregated
# ===================================================================


class TestResolveAggregates:
    def test_all_problems_of_one_class_in_declaration_order(self):
        _registrar("v2w_llm", ["openai", "anthropic"])
        _registrar("v2w_empty")

        class Agent:
            __wiring__ = {
                "llm": ("v2w_llm", ["openai", "claude"]),  # unknown required key
                "memory": "v2w_memroy",  # unknown registry
                "loop": "v2w_empty",  # empty registry
                "tools": ("v2w_llm", ["openai"], ["gemini"]),  # unknown optional key
                "good": ("v2w_llm", ["anthropic"]),
                "browser": [],  # empty literal list
            }

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)

        assert _problems(exc_info) == [
            ("llm", "unknown_key"),
            ("memory", "unknown_registry"),
            ("loop", "empty_registry"),
            ("tools", "unknown_optional_key"),
            ("browser", "empty_keys"),
        ]
        message = str(exc_info.value)
        assert message.startswith("5 wiring problems:\n  - ")
        assert "wiring 'good'" not in message
        assert all(p.kind in WIRING_PROBLEM_KINDS for p in exc_info.value.problems)
        assert all(isinstance(p, WiringProblem) for p in exc_info.value.problems)

    def test_unregistered_owner_has_no_key_in_message(self):
        class Plain:
            __wiring__ = {"x": "v2w_nope"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Plain)
        problem = exc_info.value.problems[0]
        assert problem.owner_key is None
        assert str(problem).startswith(f"{Plain.__qualname__} wiring 'x': unknown registry")

    def test_inherited_entry_names_the_declaring_class(self):
        _registrar("v2w_act2", ["click"])

        class Base:
            __wiring__ = {"action": ("v2w_act2", ["kick"])}

        class Child(Base):
            __registry_key__ = "child"

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Child)
        assert str(exc_info.value) == (
            f"{Child.__qualname__} (key 'child') wiring 'action': unknown key 'kick' in "
            "registry 'v2w_act2'; registered: ['click'] "
            f"(entry declared on {Base.__qualname__})"
        )

    def test_success_still_returns_resolved_wiring(self):
        _registrar("v2w_ok", ["a", "b"])

        class Agent:
            __wiring__ = {"x": "v2w_ok", "y": ("v2w_ok", ["b"]), "z": ["lit"]}

        resolved = resolve_wiring(Agent)
        assert resolved["x"] == ResolvedWiring("x", ["a", "b"], "v2w_ok")
        assert resolved["y"] == ResolvedWiring("y", ["b"], "v2w_ok")
        assert resolved["z"] == ResolvedWiring("z", ["lit"], None)


class TestUnknownRegistry:
    def test_did_you_mean_close_registry_name(self):
        _registrar("v2w_agent_loop", ["react"])

        class Agent:
            __wiring__ = {"loop": "v2w_agent_lop"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        problem = exc_info.value.problems[0]
        assert problem.kind == "unknown_registry"
        assert problem.registry_name == "v2w_agent_lop"
        assert problem.detail == (
            "unknown registry 'v2w_agent_lop' (did you mean 'v2w_agent_loop'?)"
        )

    def test_no_close_match_lists_known_registries(self):
        _registrar("v2w_zzz_known", ["a"])

        class Agent:
            __wiring__ = {"loop": "qqqqqqqqqqqqqqqqqqqq"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        detail = exc_info.value.problems[0].detail
        assert detail.startswith("unknown registry 'qqqqqqqqqqqqqqqqqqqq'; known registries: [")
        assert "did you mean" not in detail

    def test_no_registry_at_all(self, monkeypatch):
        from conscribe.registration import registry as registry_module

        monkeypatch.setattr(registry_module, "_REGISTRY_INDEX", {})

        class Agent:
            __wiring__ = {"loop": "agent_loop"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert exc_info.value.problems[0].detail == (
            "unknown registry 'agent_loop'; no registry exists yet "
            "(create_registrar() has not run)"
        )


class TestEmptyRegistry:
    def test_hint_points_at_discover(self):
        _registrar("v2w_hollow")

        class Agent:
            __wiring__ = {"loop": "v2w_hollow"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        problem = exc_info.value.problems[0]
        assert problem.kind == "empty_registry"
        assert problem.detail == (
            "registry 'v2w_hollow' is empty (no class is registered in it): discover() "
            "missed the package that implements it, or one of its modules failed to "
            "import; import every implementation module before resolving wiring"
        )

    def test_hint_cites_modules_discover_recorded_as_failed(self):
        from conscribe.discover import _record_failure

        _registrar("v2w_hollow2")
        _record_failure("pkg.archive.network", "SyntaxError: '(' was never closed (network.py, line 4)")

        class Agent:
            __wiring__ = {"archiver": "v2w_hollow2"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert exc_info.value.problems[0].detail.endswith(
            "; discover() recorded import failures, and classes defined in those "
            "modules are not registered: pkg.archive.network (SyntaxError: '(' was "
            "never closed (network.py, line 4))"
        )

    def test_unknown_key_also_cites_recorded_failures(self):
        """af's real case: one archiver module failed, its sibling keys exist."""
        from conscribe.discover import _record_failure

        _registrar("v2w_archiver", ["browser.dom_raw"], separator=".")
        _record_failure("pkg.archive.network", "SyntaxError: bad")

        class Cap:
            __wiring__ = {"archiver": ("v2w_archiver", ["browser.dom_raw", "browser.network"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Cap)
        assert "pkg.archive.network (SyntaxError: bad)" in exc_info.value.problems[0].detail

    def test_failure_list_is_capped(self):
        from conscribe.discover import _record_failure

        _registrar("v2w_hollow3")
        for i in range(7):
            _record_failure(f"pkg.m{i}", "ImportError: x")

        class Agent:
            __wiring__ = {"loop": "v2w_hollow3"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        detail = exc_info.value.problems[0].detail
        assert "pkg.m4 (ImportError: x); +2 more" in detail
        assert "pkg.m5" not in detail


class TestUnknownKeys:
    def test_several_unknown_keys_one_problem(self):
        _registrar("v2w_act3", ["click", "scroll"])

        class Agent:
            __wiring__ = {"action": ("v2w_act3", ["clik", "scrol", "zzzzzz"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert len(exc_info.value.problems) == 1
        assert exc_info.value.problems[0].detail == (
            "unknown keys 'clik', 'scrol', 'zzzzzz' in registry 'v2w_act3' "
            "(did you mean 'click' for 'clik', 'scroll' for 'scrol'?); "
            "registered: ['click', 'scroll']"
        )

    def test_required_and_optional_unknowns_are_separate_problems(self):
        _registrar("v2w_obs", ["terminal", "filesystem"])

        class Agent:
            __wiring__ = {"obs": ("v2w_obs", ["termnal"], ["filesytem", "browser"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert _problems(exc_info) == [("obs", "unknown_key"), ("obs", "unknown_optional_key")]
        assert exc_info.value.problems[1].detail.startswith(
            "unknown optional keys 'filesytem', 'browser' in registry 'v2w_obs'"
        )

    def test_suggestion_skips_keys_already_declared_in_the_entry(self):
        _registrar("v2w_act4", ["click", "clack"])

        class Agent:
            __wiring__ = {"action": ("v2w_act4", ["click", "clickk"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        detail = exc_info.value.problems[0].detail
        assert "did you mean 'click'" not in detail

    def test_no_misleading_sibling_suggestion(self):
        """The critique's case: 'browser.network' must not suggest 'browser.storage'."""
        _registrar(
            "v2w_arch", ["browser.dom_raw", "browser.replay_log", "browser.storage"], separator="."
        )

        class Cap:
            __wiring__ = {"archiver": ("v2w_arch", ["browser.network"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Cap)
        assert "did you mean" not in exc_info.value.problems[0].detail

    def test_qualified_key_suggestion_compares_within_prefix(self):
        _registrar("v2w_arch2", ["browser.storage", "desktop.storage"], separator=".")

        class Cap:
            __wiring__ = {"archiver": ("v2w_arch2", ["desktop.storag"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Cap)
        assert "(did you mean 'desktop.storage'?)" in exc_info.value.problems[0].detail

    def test_registered_list_is_capped(self):
        _registrar("v2w_big", [f"k{i:02d}" for i in range(35)])

        class Agent:
            __wiring__ = {"x": ("v2w_big", ["nope"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        detail = exc_info.value.problems[0].detail
        assert "'k29'] (+5 more)" in detail
        assert "'k30'" not in detail


class TestCapabilityRelativeKeys:
    def _registry(self, name: str) -> type:
        return _registrar(
            name, ["browser.click", "desktop.click", "browser.scroll"], separator="."
        )

    def test_short_key_matching_nothing_is_a_problem(self):
        self._registry("v2w_rel1")

        class Agent:
            __wiring__ = {"action": ("v2w_rel1", ["click", "teleport"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert exc_info.value.problems[0].kind == "unknown_key"
        assert exc_info.value.problems[0].detail == (
            "unknown key 'teleport' in registry 'v2w_rel1' (capability-relative: no "
            "registered key ends in '.teleport'); registered: ['browser.click', "
            "'browser.scroll', 'desktop.click']"
        )

    def test_short_key_typo_suggests_segment(self):
        self._registry("v2w_rel2")

        class Agent:
            __wiring__ = {"action": ("v2w_rel2", ["scrol"])}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert "(capability-relative: no registered key ends in '.scrol'; did you mean 'scroll'?)" in (
            exc_info.value.problems[0].detail
        )

    def test_ambiguity_still_expands_not_arbitrates(self):
        self._registry("v2w_rel3")

        class Agent:
            __wiring__ = {"action": {"registry": "v2w_rel3", "keys": ["click"]}}

        assert resolve_wiring(Agent)["action"].allowed_keys == ["browser.click", "desktop.click"]


class TestExpandRelativeKeysRobustness:
    def test_non_str_declared_keys_pass_through(self):
        assert expand_relative_keys((1, "click", None), ["a.click"], ".") == [1, "a.click", None]

    def test_unhashable_declared_keys_do_not_crash(self):
        assert expand_relative_keys((["x"],), ["a.click"], ".") == [["x"]]

    def test_non_str_registry_keys_are_ignored_for_matching(self):
        assert expand_relative_keys(("click",), [3, "a.click"], ".") == ["a.click"]


# ===================================================================
# F1 — grammar problems (also reported by parse_wiring / check)
# ===================================================================


class TestGrammarProblems:
    @pytest.mark.parametrize(
        ("value", "kind", "fragment"),
        [
            (("reg", [1]), "non_str_key", "second tuple element (keys) contains non-str key(s) 1 (int)"),
            (("reg", ["a"], [None]), "non_str_key", "third tuple element (optional keys) contains non-str key(s) None (NoneType)"),
            (["a", 2], "non_str_key", "literal list contains non-str value(s) 2 (int)"),
            ({"registry": "reg", "keys": [b"x"]}, "non_str_key", "'keys' contains non-str key(s) b'x' (bytes)"),
            ({"registry": "reg", "keys": ["a"], "optional_keys": [1.5]}, "non_str_key", "'optional_keys' contains non-str key(s) 1.5 (float)"),
            (("reg", []), "empty_keys", "explicit key list is empty"),
            (("reg", [], []), "empty_keys", "explicit key list is empty"),
            ([], "empty_keys", "literal list is empty"),
            ({"registry": "reg", "keys": []}, "empty_keys", "'keys' is empty"),
            ({"registry": "reg", "keys": [], "optional_keys": []}, "empty_keys", "'keys' is empty"),
            (42, "malformed", "expected a registry name (str)"),
            ({"a", "b"}, "malformed", "got set"),
            ("", "malformed", "registry name must be a non-empty str"),
            (("reg",), "malformed", "got a 1-tuple"),
            (("reg", ["a"], ["b"], ["c"]), "malformed", "got a 4-tuple"),
            ((None, ["a"]), "malformed", "first tuple element (registry name) must be a non-empty str, got NoneType None"),
            (("reg", ("a", "b")), "malformed", "second tuple element (keys) must be a list of str, got tuple"),
            (("reg", "a"), "malformed", "second tuple element (keys) must be a list of str, got str 'a'"),
        ],
    )
    def test_problem_kinds_and_messages(self, value, kind, fragment):
        class Agent:
            __wiring__ = {"x": value}

        for check in (parse_wiring, check_wiring_declaration, resolve_wiring):
            with pytest.raises(WiringResolutionError) as exc_info:
                check(Agent)
            assert _problems(exc_info) == [("x", kind)]
            assert fragment in exc_info.value.problems[0].detail

    def test_empty_required_with_optional_keys_is_fine(self):
        _registrar("v2w_opt", ["x"])

        class Agent:
            __wiring__ = {"obs": ("v2w_opt", [], ["x"])}

        resolved = resolve_wiring(Agent)["obs"]
        assert (resolved.allowed_keys, resolved.optional_keys) == (["x"], ["x"])

    def test_third_tuple_element_none_means_no_optional_keys(self):
        """1.x accepted ``(registry, keys, None)``; keep it."""

        class Agent:
            __wiring__ = {"x": ("reg", ["a"], None)}

        assert parse_wiring(Agent) == [WiringSpec("x", "reg", ("a",), None)]

    def test_malformed_optional_does_not_also_claim_empty(self):
        class Agent:
            __wiring__ = {"x": ("reg", [], "oops")}

        with pytest.raises(WiringResolutionError) as exc_info:
            parse_wiring(Agent)
        assert _problems(exc_info) == [("x", "malformed")]

    def test_bad_parameter_names(self):
        class Agent:
            __wiring__ = {1: "reg", "not-an-identifier": "reg"}

        with pytest.raises(WiringResolutionError) as exc_info:
            check_wiring_declaration(Agent)
        assert _problems(exc_info) == [("1", "malformed"), ("not-an-identifier", "malformed")]
        assert exc_info.value.problems[0].detail == "parameter name must be a str, got int 1"
        assert exc_info.value.problems[1].detail == (
            "parameter name 'not-an-identifier' is not a valid Python identifier"
        )

    def test_several_grammar_problems_of_one_entry(self):
        class Agent:
            __wiring__ = {"x": ({"r": 1}, [1, None], [])}

        with pytest.raises(WiringResolutionError) as exc_info:
            parse_wiring(Agent)
        assert _problems(exc_info) == [("x", "malformed"), ("x", "non_str_key")]


class TestWiringNotADict:
    @pytest.mark.parametrize("value", [["llm"], ("llm",), "llm", 3])
    def test_non_mapping_wiring_is_reported(self, value):
        class Agent:
            __wiring__ = value

        with pytest.raises(WiringResolutionError) as exc_info:
            check_wiring_declaration(Agent)
        problem = exc_info.value.problems[0]
        assert (problem.kind, problem.param_name) == ("not_a_dict", "")
        assert str(problem) == (
            f"{Agent.__qualname__} __wiring__: must be a dict mapping parameter names "
            f"to wiring entries, got {type(value).__name__} {value!r}"
        )

    def test_inherited_non_mapping_is_reported_with_its_class(self):
        _registrar("v2w_nd", ["a"])

        class Base:
            __wiring__ = ["v2w_nd"]

        class Child(Base):
            __wiring__ = {"x": "v2w_nd"}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Child)
        assert _problems(exc_info) == [("", "not_a_dict")]
        assert exc_info.value.problems[0].detail.endswith(f"(declared on {Base.__qualname__})")

    def test_none_and_mappings_are_accepted(self):
        _registrar("v2w_map", ["a"])

        class Base:
            __wiring__ = types.MappingProxyType({"x": "v2w_map"})

        class Child(Base):
            __wiring__ = None

        assert list(resolve_wiring(Child)) == ["x"]

    def test_collect_wiring_from_mro_stays_lenient(self):
        from conscribe.wiring import collect_wiring_from_mro

        class Agent:
            __wiring__ = ["x"]

        assert collect_wiring_from_mro(Agent) == {}


# ===================================================================
# validate_wiring
# ===================================================================


class TestValidateWiring:
    def _owner_layer(self, name: str):
        Own = _registrar(name)

        class OwnerBase(metaclass=Own.Meta):
            __abstract__ = True

            def run(self): ...

        return Own, OwnerBase

    def test_success_returns_none(self):
        target = _registrar("v2w_tgt", ["a"])
        Own, OwnerBase = self._owner_layer("v2w_own_ok")

        class Fine(OwnerBase):
            __wiring__ = {"x": "v2w_tgt"}

        assert validate_wiring({"own": Own, "tgt": target}) is None
        assert validate_wiring([Own, target]) is None
        assert validate_wiring((Own,)) is None
        assert validate_wiring(Own) is None  # a single registrar

    def test_mapping_iterable_and_single_registrar_agree(self):
        Own, OwnerBase = self._owner_layer("v2w_own_forms")

        class Broken(OwnerBase):
            __wiring__ = {"x": "v2w_missing_registry"}

        for arg in ({"own": Own}, [Own], iter([Own]), Own):
            with pytest.raises(WiringResolutionError) as exc_info:
                validate_wiring(arg)
            assert _problems(exc_info) == [("x", "unknown_registry")]
            assert exc_info.value.problems[0].owner_key == "broken"

    def test_order_is_by_module_and_qualname_not_registration(self):
        Own, OwnerBase = self._owner_layer("v2w_own_order")

        class Zeta(OwnerBase):
            __wiring__ = {"b": "v2w_nope_b", "a": "v2w_nope_a"}

        class Alpha(OwnerBase):
            __wiring__ = {"c": "v2w_nope_c"}

        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring(Own)
        # Alpha sorts first; Zeta keeps its declaration order (b before a).
        assert [(p.owner_key, p.param_name) for p in exc_info.value.problems] == [
            ("alpha", "c"),
            ("zeta", "b"),
            ("zeta", "a"),
        ]

    def test_each_class_reported_once(self):
        Own, OwnerBase = self._owner_layer("v2w_own_multi")

        class Multi(OwnerBase):
            __registry_key__ = ["m1", "m2"]
            __wiring__ = {"x": "v2w_nope"}

        assert Own.keys() == ["m1", "m2"]
        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring([Own, Own])
        assert len(exc_info.value.problems) == 1
        assert exc_info.value.problems[0].owner_key == "m1"

    def test_class_without_own_key_uses_the_registered_key(self):
        Own = _registrar("v2w_own_raw")

        class Raw:
            __wiring__ = {"x": "v2w_nope"}

            def run(self): ...

        Own._registry.add("raw_key", Raw)
        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring(Own)
        assert exc_info.value.problems[0].owner_key == "raw_key"

    def test_grammar_problems_of_registered_classes_are_reported(self):
        Own, OwnerBase = self._owner_layer("v2w_own_grammar")

        class Typo(OwnerBase):
            pass

        # Class creation rejects a malformed declaration
        # (check_wiring_declaration); one written after registration still
        # reaches build time, where validate_wiring must report it.
        Typo.__wiring__ = {"x": {"registry": "v2w_own_grammar", "defualt": "typo"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring(Own)
        assert _problems(exc_info) == [("x", "malformed")]
        assert "'defualt' (did you mean 'default'?)" in exc_info.value.problems[0].detail

    @pytest.mark.parametrize("bad", ["agent", 42, [object()], {"x": 1}])
    def test_rejects_non_registrars(self, bad):
        with pytest.raises(TypeError, match=r"validate_wiring\(\) expects"):
            validate_wiring(bad)

    def test_empty_input_is_fine(self):
        assert validate_wiring({}) is None
        assert validate_wiring([]) is None


# ===================================================================
# check_wiring_declaration
# ===================================================================


class TestCheckWiringDeclaration:
    def test_no_registry_lookups(self):
        class Agent:
            __wiring__ = {
                "loop": "v2w_does_not_exist",
                "llm": ("v2w_also_missing", ["anything"]),
                "x": {"registry": "v2w_nope", "keys": ["a"], "default": "zzz"},
            }

        assert check_wiring_declaration(Agent) is None

    def test_no_wiring_is_a_no_op(self):
        class Plain:
            pass

        assert check_wiring_declaration(Plain) is None

    def test_side_effect_free(self):
        reg = _registrar("v2w_side", ["a"])
        before = dict(reg.get_all())

        class Agent:
            __wiring__ = {"env": "v2w_side", "bad": 3}

        snapshot = dict(Agent.__dict__)
        with pytest.raises(WiringResolutionError):
            check_wiring_declaration(Agent)
        assert dict(Agent.__dict__) == snapshot  # no WiredField installed
        assert reg.get_all() == before

    def test_exported_from_package(self):
        assert conscribe.check_wiring_declaration is check_wiring_declaration
        assert conscribe.validate_wiring is validate_wiring
        assert {"check_wiring_declaration", "validate_wiring"} <= set(conscribe.__all__)


# ===================================================================
# F4 — dict-form grammar
# ===================================================================


class TestDictFormParse:
    def test_registry_only_is_mode_one(self):
        class Agent:
            __wiring__ = {"loop": {"registry": "agent_loop"}}

        assert parse_wiring(Agent) == [WiringSpec("loop", "agent_loop")]

    def test_keys_and_optional_keys(self):
        class Agent:
            __wiring__ = {
                "obs": {"registry": "obs", "keys": ["terminal"], "optional_keys": ["fs"]},
            }

        assert parse_wiring(Agent) == [
            WiringSpec("obs", "obs", allowed_keys=("terminal",), optional_keys=("fs",))
        ]

    def test_default_and_declare_only(self):
        class Agent:
            __wiring__ = {
                "source": {"registry": "net", "keys": ["archive", "live"], "default": "live"},
                "archiver": {"registry": "archiver", "declare_only": True},
                "plain": {"registry": "net", "declare_only": False},
            }

        specs = {s.param_name: s for s in parse_wiring(Agent)}
        assert specs["source"] == WiringSpec(
            "source", "net", ("archive", "live"), None, default_key="live"
        )
        assert specs["archiver"].declare_only is True
        assert specs["archiver"].default_key is None
        assert specs["plain"].declare_only is False

    def test_legacy_forms_have_no_default_and_are_not_declare_only(self):
        class Agent:
            __wiring__ = {"a": "r", "b": ("r", ["k"]), "c": ("r", ["k"], ["o"]), "d": ["v"]}

        for spec in parse_wiring(Agent):
            assert spec.default_key is None
            assert spec.declare_only is False

    def test_spec_and_resolved_field_layout(self):
        assert [f.name for f in dataclasses.fields(WiringSpec)] == [
            "param_name", "registry_name", "allowed_keys", "optional_keys",
            "default_key", "declare_only",
        ]
        names = [f.name for f in dataclasses.fields(ResolvedWiring)]
        assert names == [
            "param_name", "allowed_keys", "registry_name", "optional_keys",
            "default_key", "declare_only",
        ]
        assert "injected" not in names

    @pytest.mark.parametrize(
        ("entry", "fragment"),
        [
            ({"registry": "r", "defualt": "a"}, "unknown field(s) 'defualt' (did you mean 'default'?)"),
            ({"registry": "r", "kyes": ["a"]}, "unknown field(s) 'kyes' (did you mean 'keys'?)"),
            ({"registry": "r", "inline_only": True}, "unknown field(s) 'inline_only'; allowed fields: 'registry', 'keys', 'optional_keys', 'default', 'declare_only'"),
            ({"keys": ["live"], "default": "live"}, "dict-form entry needs 'registry'"),
            ({"registry": "", "keys": ["a"]}, "'registry' must be a non-empty str, got str ''"),
            ({"registry": None}, "'registry' must be a non-empty str, got NoneType None"),
            ({"registry": "r", "keys": "a"}, "'keys' must be a list of str, got str 'a'"),
            ({"registry": "r", "keys": None}, "'keys' must be a list of str, got NoneType None"),
            ({"registry": "r", "optional_keys": ["a"]}, "'optional_keys' requires 'keys'"),
            ({"registry": "r", "keys": ["a"], "optional_keys": ("b",)}, "'optional_keys' must be a list of str, got tuple"),
            ({"registry": "r", "default": ["a"]}, "'default' must be a single key (non-empty str), got list ['a']"),
            ({"registry": "r", "default": ""}, "'default' must be a single key (non-empty str), got str ''"),
            ({"registry": "r", "default": None}, "'default' must be a single key (non-empty str), got NoneType None"),
            ({"registry": "r", "declare_only": "yes"}, "'declare_only' must be a bool, got str 'yes'"),
            ({"registry": "r", "declare_only": True, "default": "a"}, "'declare_only' cannot be combined with 'default'"),
        ],
    )
    def test_malformed_dict_entries(self, entry, fragment):
        class Agent:
            __wiring__ = {"x": entry}

        with pytest.raises(WiringResolutionError) as exc_info:
            check_wiring_declaration(Agent)
        assert _problems(exc_info) == [("x", "malformed")]
        assert fragment in exc_info.value.problems[0].detail

    def test_subclass_entry_replaces_inherited_entry_whole(self):
        class Base:
            __wiring__ = {"src": {"registry": "net", "keys": ["archive", "live"], "default": "live"}}

        class Child(Base):
            __wiring__ = {"src": {"registry": "net"}}

        assert parse_wiring(Child) == [WiringSpec("src", "net")]

    def test_none_excludes_a_dict_entry(self):
        class Base:
            __wiring__ = {"src": {"registry": "net", "default": "live"}, "keep": "r"}

        class Child(Base):
            __wiring__ = {"src": None}

        assert [s.param_name for s in parse_wiring(Child)] == ["keep"]

    def test_wired_field_keeps_registry_name_for_dict_entries(self):
        class Agent:
            __wiring__ = {"src": {"registry": "network_source", "default": "live"}}

        inject_wired_descriptors(Agent)
        assert isinstance(Agent.__dict__["src"], WiredField)
        assert Agent.__dict__["src"].registry_name == "network_source"

    def test_injection_skips_malformed_parameter_names(self):
        class Agent:
            __wiring__ = {1: "r", "ok": "r"}

        inject_wired_descriptors(Agent)  # must not raise
        assert isinstance(Agent.__dict__["ok"], WiredField)


# ===================================================================
# F4 — default keys and declare_only during resolution
# ===================================================================


class TestDefaultResolution:
    def test_valid_default_is_carried(self):
        """af acceptance mirror: network_source defaults to 'live'."""
        _registrar("v2w_network_source", ["archive", "live"])

        class Browser:
            network_source: object  # the receptor slot a default needs
            __wiring__ = {
                "network_source": {
                    "registry": "v2w_network_source",
                    "keys": ["archive", "live"],
                    "default": "live",
                },
            }

        resolved = resolve_wiring(Browser)["network_source"]
        assert resolved == ResolvedWiring(
            "network_source", ["archive", "live"], "v2w_network_source", None, "live", False
        )

    def test_unregistered_default_names_the_key(self):
        _registrar("v2w_net2", ["archive", "live"])

        class Browser:
            __registry_key__ = "browser"
            network_source: object
            __wiring__ = {
                "network_source": {"registry": "v2w_net2", "keys": ["archive", "live"], "default": "livee"}
            }

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Browser)
        assert str(exc_info.value) == (
            f"{Browser.__qualname__} (key 'browser') wiring 'network_source': default "
            "'livee' is not registered in registry 'v2w_net2' (did you mean 'live'?); "
            "registered: ['archive', 'live']"
        )
        assert exc_info.value.problems[0].kind == "bad_default"

    def test_default_outside_the_allowed_keys(self):
        _registrar("v2w_net3", ["archive", "live", "mirror"])

        class Browser:
            src: object
            __wiring__ = {"src": {"registry": "v2w_net3", "keys": ["archive", "live"], "default": "mirror"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Browser)
        assert exc_info.value.problems[0].detail == (
            "default 'mirror' is not among the entry's allowed keys ['archive', 'live']; "
            "add it to the entry's keys or choose an allowed key"
        )

    def test_default_among_optional_keys_is_allowed(self):
        _registrar("v2w_net4", ["archive", "live"])

        class Browser:
            src: object
            __wiring__ = {
                "src": {"registry": "v2w_net4", "keys": ["archive"], "optional_keys": ["live"], "default": "live"}
            }

        assert resolve_wiring(Browser)["src"].default_key == "live"

    def test_mode_one_dict_default_only_needs_registration(self):
        _registrar("v2w_net5", ["archive", "live"])

        class Browser:
            src: object
            __wiring__ = {"src": {"registry": "v2w_net5", "default": "archive"}}

        resolved = resolve_wiring(Browser)["src"]
        assert (resolved.allowed_keys, resolved.default_key) == (["archive", "live"], "archive")

    def test_relative_default_must_be_fully_qualified(self):
        """Even with a single match: a default is one concrete choice."""
        _registrar("v2w_act_sep", ["browser.click", "browser.scroll"], separator=".")

        class Agent:
            action: object
            __wiring__ = {"action": {"registry": "v2w_act_sep", "keys": ["click"], "default": "click"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert exc_info.value.problems[0].detail == (
            "default 'click' must be a fully-qualified key: registry 'v2w_act_sep' uses "
            "key_separator '.', so write one of ['browser.click'] (a default is one "
            "concrete choice; it is never expanded)"
        )

    def test_qualified_default_inside_relative_keys(self):
        _registrar("v2w_act_sep2", ["browser.click", "desktop.click"], separator=".")

        class Agent:
            action: object
            __wiring__ = {
                "action": {"registry": "v2w_act_sep2", "keys": ["click"], "default": "desktop.click"}
            }

        resolved = resolve_wiring(Agent)["action"]
        assert resolved.allowed_keys == ["browser.click", "desktop.click"]
        assert resolved.default_key == "desktop.click"

    def test_flat_key_registered_in_a_separator_registry_is_exact(self):
        _registrar("v2w_mixed", ["click", "browser.click"], separator=".")

        class Agent:
            action: object
            __wiring__ = {"action": {"registry": "v2w_mixed", "default": "click"}}

        assert resolve_wiring(Agent)["action"].default_key == "click"

    def test_unknown_short_default_in_separator_registry(self):
        _registrar("v2w_act_sep3", ["browser.click"], separator=".")

        class Agent:
            action: object
            __wiring__ = {"action": {"registry": "v2w_act_sep3", "default": "teleport"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert exc_info.value.problems[0].detail.startswith(
            "default 'teleport' is not registered in registry 'v2w_act_sep3'"
        )

    def test_unknown_keys_do_not_cascade_into_the_default(self):
        _registrar("v2w_net6", ["archive", "live"])

        class Browser:
            src: object
            __wiring__ = {"src": {"registry": "v2w_net6", "keys": ["archve", "live"], "default": "archive"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Browser)
        assert _problems(exc_info) == [("src", "unknown_key")]

    def test_unregistered_default_is_reported_next_to_unknown_keys(self):
        _registrar("v2w_net7", ["archive", "live"])

        class Browser:
            src: object
            __wiring__ = {"src": {"registry": "v2w_net7", "keys": ["archve"], "default": "archve"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Browser)
        assert _problems(exc_info) == [("src", "unknown_key"), ("src", "bad_default")]

    def test_registry_problem_suppresses_default_checks(self):
        _registrar("v2w_hollow_net")

        class Browser:
            a: object
            b: object
            __wiring__ = {
                "a": {"registry": "v2w_hollow_net", "default": "live"},
                "b": {"registry": "v2w_no_such_net", "default": "live"},
            }

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Browser)
        assert _problems(exc_info) == [("a", "empty_registry"), ("b", "unknown_registry")]

    def test_declare_only_is_carried(self):
        _registrar("v2w_archiver2", ["dom_raw"])

        class Cap:
            __wiring__ = {"archiver": {"registry": "v2w_archiver2", "keys": ["dom_raw"], "declare_only": True}}

        resolved = resolve_wiring(Cap)["archiver"]
        assert resolved.declare_only is True
        assert resolved.default_key is None

    def test_validate_wiring_reports_default_problems(self):
        _registrar("v2w_net8", ["archive", "live"])
        Own = _registrar("v2w_own_default")

        class OwnerBase(metaclass=Own.Meta):
            __abstract__ = True

            def run(self): ...

        class Browser(OwnerBase):
            src: object
            __wiring__ = {"src": {"registry": "v2w_net8", "default": "nope"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring(Own)
        assert _problems(exc_info) == [("src", "bad_default")]


# ===================================================================
# Interaction with the composed builder (parse_wiring callers)
# ===================================================================


class TestComposedBuildSeesGrammarErrors:
    def test_malformed_entry_raises_wiring_resolution_error(self):
        """composed._build_dependency_graph calls parse_wiring: malformed
        declarations now surface as WiringResolutionError (1.x: TypeError)."""
        from conscribe import build_composed_config

        # A buildable layer (with a discriminator): build_composed_config
        # checks each layer before it validates the wiring.
        Own = create_registrar("v2w_own_composed", _P, discriminator_field="kind")

        class OwnerBase(metaclass=Own.Meta):
            __abstract__ = True

            def run(self): ...

        class Broken(OwnerBase):
            pass

        # Written after registration: class creation would reject it.
        Broken.__wiring__ = {"x": ("v2w_own_composed", "not-a-list")}

        with pytest.raises(WiringResolutionError) as exc_info:
            build_composed_config({"own": Own})
        assert _problems(exc_info) == [("x", "malformed")]
        assert exc_info.value.problems[0].owner_key == "broken"


# ===================================================================
# Spec 4.8: parameter-default problems are wiring problems (validate-first)
# ===================================================================


def _owner_layer(name: str):
    """An owner layer with a discriminator (buildable) and its abstract base."""
    Own = create_registrar(name, _P, discriminator_field="kind")

    class OwnerBase(metaclass=Own.Meta):
        __abstract__ = True

        def run(self): ...

    return Own, OwnerBase


class TestParameterDefaults:
    def test_validate_wiring_reports_every_class_at_once(self):
        """Review s10 / s27 / probe 19: the extractor found these one class at
        a time; validate_wiring (and so every build) reports them together."""
        from typing import Optional

        _registrar("v2w_pd_loop", ["react", "plan"])
        _registrar("v2w_pd_hook", ["trace", "record"])
        Own, OwnerBase = _owner_layer("v2w_pd_owner")

        class First(OwnerBase):
            __wiring__ = {"loop": {"registry": "v2w_pd_loop", "default": "react"}}

            def __init__(self, loop: str = "plan"): ...

        class Second(OwnerBase):
            __wiring__ = {"loop": "v2w_pd_loop"}

            def __init__(self, loop: str = "reactt"): ...

        class Third(OwnerBase):
            __wiring__ = {"hooks": "v2w_pd_hook"}

            def __init__(self, hooks: Optional[list[str]] = ["zz"]): ...  # noqa: B006

        with pytest.raises(WiringResolutionError) as exc_info:
            validate_wiring(Own)
        problems = exc_info.value.problems
        assert [(p.owner, p.param_name, p.kind) for p in problems] == [
            (First.__qualname__, "loop", "bad_default"),
            (Second.__qualname__, "loop", "bad_default"),
            (Third.__qualname__, "hooks", "bad_default"),
        ]
        assert problems[0].detail == (
            "the parameter default 'plan' names a different key than the wiring default "
            "'react'; declare the default once (remove the parameter default or make the "
            "two equal)."
        )
        assert problems[1].detail == (
            "the parameter default 'reactt' is not a valid value of the wired field: it "
            "accepts keys from ['plan', 'react']. Use one of them (or None), or declare "
            "the default with the dict-form wiring 'default'."
        )
        assert "it accepts a list of keys from ['record', 'trace']" in problems[2].detail

        # A build reports the same problems, all at once, before extracting.
        from conscribe import build_layer_config

        with pytest.raises(WiringResolutionError) as build_info:
            build_layer_config(Own)
        assert build_info.value.problems == problems

    def test_resolve_wiring_reports_it_too(self):
        _registrar("v2w_pd_env", ["local", "docker"])

        class Agent:
            __wiring__ = {"env": "v2w_pd_env"}

            def __init__(self, env: str = "remote"): ...

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert _problems(exc_info) == [("env", "bad_default")]

    @pytest.mark.parametrize(
        "signature",
        [
            "env: str = 'local'",
            "env: str = None",
            "env: 'Optional[str]' = None",
            "env: str = ''.join(['lo', 'cal'])",
            "env: list[str] = ['local', 'docker']",
            "env: list[str] = ('local',)",
        ],
    )
    def test_valid_defaults_pass(self, signature):
        _registrar("v2w_pd_ok", ["local", "docker"])
        namespace: dict = {}
        exec(  # noqa: S102 - the signature under test
            f"from typing import Optional\nclass Agent:\n"
            f"    __wiring__ = {{'env': 'v2w_pd_ok'}}\n"
            f"    def __init__(self, {signature}): ...\n",
            namespace,
        )
        assert set(resolve_wiring(namespace["Agent"])) == {"env"}

    def test_class_level_value_of_an_annotation_slot(self):
        _registrar("v2w_pd_cls", ["local"])

        class Agent:
            __wiring__ = {"env": "v2w_pd_cls"}
            env: str = "remote"

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert _problems(exc_info) == [("env", "bad_default")]

    def test_dataclass_and_pydantic_field_defaults(self):
        from pydantic import BaseModel, Field

        _registrar("v2w_pd_fields", ["local"])

        @dataclasses.dataclass
        class DC:
            __wiring__ = {"env": "v2w_pd_fields"}
            env: str = "remote"

        class Model(BaseModel):
            __wiring__ = {"env": "v2w_pd_fields"}
            env: str = Field(default="remote")

        for cls in (DC, Model):
            with pytest.raises(WiringResolutionError) as exc_info:
                resolve_wiring(cls)
            assert _problems(exc_info) == [("env", "bad_default")], cls

    def test_factories_are_left_to_the_build(self):
        """A default_factory is only evaluated by the config build."""
        _registrar("v2w_pd_factory", ["local"])

        @dataclasses.dataclass
        class DC:
            __wiring__ = {"envs": "v2w_pd_factory"}
            envs: list = dataclasses.field(default_factory=lambda: ["remote"])

        assert set(resolve_wiring(DC)) == {"envs"}

    def test_user_literal_annotation_is_left_to_the_build(self):
        """The extractor reports a Literal with non-key values more precisely."""
        from typing import Literal, Union

        _registrar("v2w_pd_lit", ["fast", "slow"])

        class Agent:
            __wiring__ = {"strategy": "v2w_pd_lit"}

            def __init__(self, strategy: Union[str, Literal["auto"]] = "auto"): ...

        assert set(resolve_wiring(Agent)) == {"strategy"}

    def test_transforming_validators_are_left_to_the_build(self):
        # Annotated / the validators are module globals: string annotations
        # (``from __future__ import annotations``) must resolve.
        _registrar("v2w_pd_val", ["react"])

        class Lowered:
            __wiring__ = {"loop": "v2w_pd_val"}

            def __init__(self, loop: Annotated[str, BeforeValidator(str.lower)] = "REACT"): ...

        class Checked:
            __wiring__ = {"loop": "v2w_pd_val"}

            def __init__(self, loop: Annotated[str, AfterValidator(str.lower)] = "REACT"): ...

        assert set(resolve_wiring(Lowered)) == {"loop"}
        with pytest.raises(WiringResolutionError):
            resolve_wiring(Checked)  # an AfterValidator never sees an invalid key

    def test_validators_of_a_pydantic_field_are_left_to_the_build(self):
        """pydantic moves Annotated validators into the field's metadata."""
        from conscribe import build_layer_config
        from pydantic import BaseModel

        _registrar("v2w_pd_model_val", ["react"])
        Own = create_registrar("v2w_pd_model_owner", _P, discriminator_field="kind")
        Bridge = Own.bridge(BaseModel)

        class Lowering(Bridge):
            __wiring__ = {"loop": "v2w_pd_model_val"}
            loop: Annotated[str, BeforeValidator(str.lower)] = "REACT"

            def run(self) -> None: ...

        validate_wiring(Own)  # the build accepts it, so validation must too
        assert "loop" in build_layer_config(Own).per_key_models["lowering"].model_fields

    def test_problem_is_reported_once_with_registry_problems_first(self):
        """No cascade: an entry with an unknown key is not judged by its default."""
        _registrar("v2w_pd_cascade", ["local"])

        class Agent:
            __wiring__ = {"env": ("v2w_pd_cascade", ["locl"])}

            def __init__(self, env: str = "remote"): ...

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        assert _problems(exc_info) == [("env", "unknown_key")]


# ===================================================================
# Spec 4.9 grammar, m10 suggestions, __wiring__ = None
# ===================================================================


class TestStricterGrammar:
    @pytest.mark.parametrize(
        ("entry", "fragment"),
        [
            (("reg", ["a", "b"], ["b"]), "key 'b' listed both as required and as optional"),
            (
                {"registry": "reg", "keys": ["a", "b"], "optional_keys": ["a", "b"]},
                "keys 'a', 'b' listed both as required and as optional",
            ),
        ],
    )
    def test_key_both_required_and_optional_is_malformed(self, entry, fragment):
        class Agent:
            __wiring__ = {"x": entry}

        with pytest.raises(WiringResolutionError) as exc_info:
            check_wiring_declaration(Agent)
        assert _problems(exc_info) == [("x", "malformed")]
        assert fragment in exc_info.value.problems[0].detail

    def test_expanded_overlap_is_still_deduplicated(self):
        """1.3.0 behaviour kept: only a key *written* in both lists is an error."""
        _registrar("v2w_dedupe", ["browser.click", "desktop.click"], separator=".")

        class Agent:
            __wiring__ = {"action": ("v2w_dedupe", ["click"], ["browser.click"])}

        assert resolve_wiring(Agent)["action"].allowed_keys == ["browser.click", "desktop.click"]

    @pytest.mark.parametrize(
        ("name", "fragment"),
        [
            ("class", "parameter name 'class' is a Python keyword"),
            ("None", "parameter name 'None' is a Python keyword"),
            ("__call__", "parameter name '__call__' is a dunder name"),
            ("__init__", "parameter name '__init__' is a dunder name"),
        ],
    )
    def test_entry_names_must_be_plain_identifiers(self, name, fragment):
        """Review probe 27: a placeholder named __call__ / run replaced them."""

        class Agent:
            __wiring__ = {name: "reg"}

        with pytest.raises(WiringResolutionError) as exc_info:
            check_wiring_declaration(Agent)
        assert _problems(exc_info) == [(name, "malformed")]
        assert fragment in exc_info.value.problems[0].detail

    def test_private_and_soft_keyword_names_stay_allowed(self):
        class Agent:
            __wiring__ = {"_llm": "reg", "match": "reg", "type": "reg"}

        check_wiring_declaration(Agent)

    def test_dunder_name_fails_class_creation(self):
        Own, OwnerBase = _owner_layer("v2w_dunder_owner")
        with pytest.raises(WiringResolutionError, match="dunder name"):
            class X(OwnerBase):
                __wiring__ = {"__call__": "v2w_whatever"}
        assert Own.keys() == []

    def test_wiring_attribute_none_keeps_inherited_entries(self):
        class Base:
            __wiring__ = {"llm": "reg"}

        class Child(Base):
            __wiring__ = None

        assert [s.param_name for s in parse_wiring(Child)] == ["llm"]

    def test_mistyped_short_default_suggests_fully_qualified_keys(self):
        """Review m10: suggesting 'click' led to a second, different error."""
        _registrar("v2w_m10", ["browser.click", "desktop.click", "browser.scroll"], separator=".")

        class Agent:
            act: object
            __wiring__ = {"act": {"registry": "v2w_m10", "default": "clik"}}

        with pytest.raises(WiringResolutionError) as exc_info:
            resolve_wiring(Agent)
        detail = exc_info.value.problems[0].detail
        assert "(did you mean 'browser.click' or 'desktop.click'?)" in detail


# ===================================================================
# The per-class parse cache never goes stale
# ===================================================================


class TestParseCache:
    def test_in_place_changes_are_noticed(self):
        class Base:
            __wiring__ = {"llm": ("reg", ["a"])}

        class Child(Base):
            pass

        check_wiring_declaration(Child)  # parsed and cached
        Base.__wiring__["llm"][1].append(3)  # a nested in-place change
        with pytest.raises(WiringResolutionError) as exc_info:
            check_wiring_declaration(Child)
        assert _problems(exc_info) == [("llm", "non_str_key")]

        Base.__wiring__["llm"][1].pop()
        check_wiring_declaration(Child)
        Base.__wiring__["env"] = 42  # a top-level change
        with pytest.raises(WiringResolutionError):
            check_wiring_declaration(Child)

    def test_reassigned_wiring_is_reparsed(self):
        class Base:
            __wiring__ = {"llm": "reg"}

        assert [s.registry_name for s in parse_wiring(Base)] == ["reg"]
        Base.__wiring__ = {"llm": "other"}
        assert [s.registry_name for s in parse_wiring(Base)] == ["other"]

    def test_validate_wiring_reads_each_registry_once(self, monkeypatch):
        """Each target registry's keys and capability-relative index are built
        once per validate_wiring() call, not once per entry."""
        from conscribe.registration.registry import LayerRegistry

        _registrar("v2w_once", ["browser.click", "desktop.click"], separator=".")
        Own, OwnerBase = _owner_layer("v2w_once_owner")
        for i in range(5):
            type(OwnerBase)(f"Agent{i}", (OwnerBase,), {
                "__wiring__": {"a": ("v2w_once", ["click"]), "b": "v2w_once"},
            })

        calls: list[str] = []
        real = LayerRegistry.keys

        def counting(self):
            calls.append(self.name)
            return real(self)

        monkeypatch.setattr(LayerRegistry, "keys", counting)
        validate_wiring(Own)
        assert calls.count("v2w_once") == 1


# ===================================================================
# API hygiene
# ===================================================================


class TestModuleExports:
    @pytest.mark.parametrize("module", ["conscribe.wiring", "conscribe.exceptions", "conscribe.discover"])
    def test_star_import_exports_only_public_names(self, module):
        namespace: dict = {}
        exec(f"from {module} import *", namespace)  # noqa: S102
        names = {name for name in namespace if not name.startswith("__")}
        assert names == set(__import__(module, fromlist=["_"]).__all__)
        assert not names & {"sys", "Mapping", "Any", "difflib", "types", "inspect", "weakref"}

    def test_new_public_names_are_exported(self):
        for name in ("InvalidRegistryKeyError", "IsolationWarning", "StaleStubWarning"):
            assert name in conscribe.__all__
            assert getattr(conscribe, name) is getattr(conscribe.exceptions, name)

    def test_create_registrar_is_annotated_with_the_registrar_type(self):
        import typing

        from conscribe.registration.registrar import LayerRegistrar, create_registrar

        hints = typing.get_type_hints(create_registrar)
        assert hints["return"] == type[LayerRegistrar[typing.Any]]
