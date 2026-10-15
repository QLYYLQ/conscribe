"""Immutable declaration history shares the effective wiring parser's rules."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from conscribe.exceptions import WiringResolutionError
from conscribe.wiring import (
    WiringDeclaration,
    WiringProvenance,
    WiringSpec,
    check_wiring_declaration,
    parse_wiring,
    wiring_provenance,
)


def test_empty_history_and_public_exports():
    class Empty:
        pass

    assert wiring_provenance(Empty) == {}
    assert isinstance(wiring_provenance(Empty), MappingProxyType)
    namespace = {}
    exec("from conscribe.wiring import *", namespace)
    assert namespace["WiringDeclaration"] is WiringDeclaration
    assert namespace["WiringProvenance"] is WiringProvenance
    assert namespace["wiring_provenance"] is wiring_provenance


def test_all_grammar_modes_are_normalized_without_registry_lookup():
    class Owner:
        __wiring__ = {
            "all_keys": "missing_registry",
            "subset": ("missing_registry", ["required"]),
            "optional": ("missing_registry", ["required"], ["optional"]),
            "literal": ["one", "two"],
            "defaulted": {"registry": "missing_registry", "default": "chosen"},
            "declaration": {
                "registry": "missing_registry", "keys": ["click"], "declare_only": True,
            },
        }

    result = wiring_provenance(Owner)
    assert [item.winner.spec for item in result.values()] == parse_wiring(Owner)
    for name, item in result.items():
        assert item.name == name
        assert item.winner.name == name
        assert item.winner.declared_on is Owner
        assert item.winner is item.declarations[0]
        assert item.winner.disposition == "winner"
        assert item.winner.problems == ()
        assert not item.winner.excluded
        assert not item.excluded
    assert result["declaration"].winner.spec.declare_only
    assert result["defaulted"].winner.spec.default_key == "chosen"


def test_whole_entry_replacement_and_declaration_only_keep_distinct_owners():
    class Base:
        __wiring__ = {
            "strategy": {"registry": "strategy", "keys": ["base"], "default": "base"},
            "untouched": "other",
        }

    class Child(Base):
        __wiring__ = {"strategy": {"registry": "strategy", "declare_only": True}}

    class Leaf(Child):
        __wiring__ = None

    result = wiring_provenance(Leaf)
    assert list(result) == ["strategy", "untouched"]
    item = result["strategy"]
    assert [entry.declared_on for entry in item.declarations] == [Child, Base]
    assert [entry.disposition for entry in item.declarations] == ["winner", "overridden"]
    assert item.winner.spec == WiringSpec("strategy", "strategy", declare_only=True)
    assert item.declarations[1].spec.default_key == "base"
    assert result["untouched"].winner.declared_on is Base


def test_diamond_history_follows_c3_mro_without_duplicate_ancestors():
    class Root:
        __wiring__ = {"shared": "root", "root_only": "root"}

    class Left(Root):
        __wiring__ = {"shared": "left", "left_only": "left"}

    class Right(Root):
        __wiring__ = {"shared": "right", "right_only": "right"}

    class Leaf(Left, Right):
        __wiring__ = {"leaf_only": "leaf"}

    result = wiring_provenance(Leaf)
    assert list(result) == [spec.param_name for spec in parse_wiring(Leaf)]
    assert [entry.declared_on for entry in result["shared"].declarations] == [Left, Right, Root]
    assert [entry.spec.registry_name for entry in result["shared"].declarations] == [
        "left", "right", "root",
    ]
    assert result["shared"].winner.declared_on is Left


def test_exclusions_and_revivals_preserve_every_declaration():
    class Base:
        __wiring__ = {"strategy": "base"}

    class Excluded(Base):
        __wiring__ = {"strategy": None, "never_present": None}

    class Revived(Excluded):
        __wiring__ = {"strategy": "revived"}

    class ExcludedAgain(Revived):
        __wiring__ = {"strategy": None}

    revived = wiring_provenance(Revived)["strategy"]
    assert not revived.excluded
    assert revived.winner.spec.registry_name == "revived"
    assert revived.declarations[1].excluded
    assert revived.declarations[1].disposition == "overridden"
    assert revived.declarations[1].spec is None
    assert revived.declarations[1].problems == ()

    result = wiring_provenance(ExcludedAgain)
    item = result["strategy"]
    assert item.excluded
    assert item.winner.excluded
    assert item.winner.spec is None
    assert item.winner.disposition == "excluded"
    assert [entry.declared_on for entry in item.declarations] == [
        ExcludedAgain, Revived, Excluded, Base,
    ]
    assert result["never_present"].excluded
    assert result["never_present"].winner.declared_on is Excluded
    assert parse_wiring(ExcludedAgain) == []


def test_result_and_nested_records_are_immutable_snapshots():
    class Base:
        __wiring__ = {"strategy": ("registry", ["base"], ["maybe"])}

    class Child(Base):
        __wiring__ = {"strategy": {"registry": "registry", "keys": ["child"]}}

    original = wiring_provenance(Child)
    item = original["strategy"]
    with pytest.raises(TypeError):
        original["extra"] = item
    with pytest.raises(FrozenInstanceError):
        item.excluded = True
    with pytest.raises(FrozenInstanceError):
        item.winner.declared_on = Base
    with pytest.raises(FrozenInstanceError):
        item.winner.spec.registry_name = "mutated"
    with pytest.raises(TypeError):
        item.winner.spec.allowed_keys[0] = "mutated"

    Base.__wiring__["strategy"][1].append("base_added")
    Base.__wiring__["strategy"][2].append("optional_added")
    Child.__wiring__["strategy"]["keys"].append("child_added")
    updated = wiring_provenance(Child)["strategy"]
    assert item.winner.spec.allowed_keys == ("child",)
    assert item.declarations[1].spec.allowed_keys == ("base",)
    assert item.declarations[1].spec.optional_keys == ("maybe",)
    assert updated.winner.spec.allowed_keys == ("child", "child_added")
    assert updated.declarations[1].spec.allowed_keys == ("base", "base_added")
    assert updated.declarations[1].spec.optional_keys == ("maybe", "optional_added")


def test_replacement_deletion_and_in_place_exclusion_are_observed():
    class Base:
        __wiring__ = {"strategy": "base"}

    class Child(Base):
        __wiring__ = {"strategy": "child"}

    original = wiring_provenance(Child)
    Child.__wiring__ = {"strategy": "replacement"}
    assert wiring_provenance(Child)["strategy"].winner.spec.registry_name == "replacement"
    Child.__wiring__["strategy"] = None
    assert wiring_provenance(Child)["strategy"].excluded
    del Child.__wiring__
    assert wiring_provenance(Child)["strategy"].winner.declared_on is Base
    del Base.__wiring__["strategy"]
    assert wiring_provenance(Child) == {}
    assert original["strategy"].winner.spec.registry_name == "child"


def test_overridden_malformed_entries_remain_evidence_without_new_errors():
    class Malformed:
        __wiring__ = {"strategy": {"registry": "registry", "keys": [42]}}

    class Excluded(Malformed):
        __wiring__ = {"strategy": None}

    class Replaced(Excluded):
        __wiring__ = {"strategy": "registry"}

    result = wiring_provenance(Replaced)["strategy"]
    assert [entry.spec for entry in result.declarations] == [
        WiringSpec("strategy", "registry"), None, None,
    ]
    excluded, malformed = result.declarations[1:]
    assert excluded.excluded and not excluded.problems
    assert not malformed.excluded
    assert malformed.problems[0][0:2] == ("non_str_key", "registry")
    assert "42" in malformed.problems[0][2]
    assert malformed.disposition == "overridden"
    check_wiring_declaration(Replaced)
    assert parse_wiring(Replaced) == [result.winner.spec]

    Malformed.__wiring__["strategy"]["keys"][0] = "repaired"
    assert result.declarations[2].spec is None
    repaired = wiring_provenance(Replaced)["strategy"].declarations[2]
    assert repaired.spec.allowed_keys == ("repaired",)
    assert repaired.problems == ()


@pytest.mark.parametrize("wiring", [42, {"one": 42, "two": []}])
def test_effective_grammar_errors_match_parse_wiring(wiring):
    class Base:
        __wiring__ = wiring

    class Child(Base):
        pass

    with pytest.raises(WiringResolutionError) as parsed:
        parse_wiring(Child)
    with pytest.raises(WiringResolutionError) as provenance:
        wiring_provenance(Child)
    assert provenance.value.problems == parsed.value.problems
    assert str(provenance.value) == str(parsed.value)


@pytest.mark.parametrize("name", [42, "bad-name", "class", "__call__"])
def test_excluded_names_are_validated_only_by_provenance(name):
    class Base:
        __wiring__ = {name: None}

    class Child(Base):
        pass

    # Existing APIs discard exclusions before checking grammar.
    assert parse_wiring(Child) == []
    check_wiring_declaration(Child)
    with pytest.raises(WiringResolutionError) as exc_info:
        wiring_provenance(Child)
    problem, = exc_info.value.problems
    assert problem.kind == "malformed"
    assert "entry declared on" in problem.detail


def test_distinct_classes_with_the_same_name_remain_distinct_owners():
    first = type("Owner", (), {"__wiring__": {"strategy": "first"}})
    second = type("Owner", (first,), {"__wiring__": {"strategy": "second"}})
    result = wiring_provenance(second)["strategy"]
    assert result.declarations[0].declared_on is second
    assert result.declarations[1].declared_on is first
