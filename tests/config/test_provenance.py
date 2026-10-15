"""Declaration evidence must agree with the extraction that makes fields."""
from dataclasses import FrozenInstanceError, dataclass, field
from typing import Annotated, ClassVar

import attrs
import pytest
from pydantic import BaseModel, Field

from conscribe import extract_config_provenance, extract_config_schema
from conscribe.config.mro import walk_init_chain


def test_constructor_history_keeps_overrides_without_changing_fields():
    class Base:
        def __init__(self, *, limit: int = 1, base: str = "b"):
            pass

    class Child(Base):
        def __init__(self, *, limit: int = 2, child: bool = True, **kwargs):
            pass

    chain = walk_init_chain(Child)
    assert [(p.name, p.init_owner) for p in chain.declarations] == [
        ("limit", Child), ("child", Child), ("limit", Base), ("base", Base),
    ]
    assert chain.names() == ("limit", "child", "base")
    provenance = extract_config_provenance(Child)
    assert tuple(provenance.fields) == tuple(extract_config_schema(Child).model_fields)
    limit = provenance.fields["limit"]
    assert (limit.source, limit.declared_on, limit.init_owner) == ("init", Child, Child)
    assert [(d.declared_on, d.status) for d in limit.declarations] == [
        (Child, "included"), (Base, "overridden"),
    ]
    assert provenance.constructor_chain.terminal is Base
    assert provenance.constructor_chain.fully_resolved


def test_closed_constructor_does_not_invent_unvisited_history():
    class Base:
        def __init__(self, *, invisible: int = 1):
            pass

    class Child(Base):
        def __init__(self, *, own: int = 2):
            pass

    result = extract_config_provenance(Child)
    assert list(result.fields) == ["own"]
    assert [d.name for d in result.constructor_chain.declarations] == ["own"]
    assert result.constructor_chain.fully_resolved


def test_depth_and_class_override_use_extraction_settings():
    class Base:
        def __init__(self, *, base: int = 1):
            pass

    class Child(Base):
        def __init__(self, *, own: int = 1, **kwargs):
            pass

    result = extract_config_provenance(Child, mro_depth=0)
    assert list(result.fields) == ["own"]
    assert result.mro_depth == 0
    assert result.constructor_chain.stop_reason == "depth"
    assert result.constructor_chain.stopped_at is Base
    assert not result.constructor_chain.fully_resolved
    Child.__config_mro_depth__ = 1
    Child.__config_mro_scope__ = ["some_package"]
    result = extract_config_provenance(Child, mro_depth=0, mro_scope="all")
    assert tuple(result.fields) == ("own", "base")
    assert result.mro_depth == 1
    assert result.mro_scope == ("some_package",)


def test_scope_cut_has_evidence_without_reading_excluded_signature(monkeypatch):
    from conscribe.config import mro

    class Base:
        def __init__(self, *, base: int = 1):
            pass

    class Child(Base):
        def __init__(self, **kwargs):
            pass

    original = mro.classify_class_scope
    monkeypatch.setattr(mro, "classify_class_scope", lambda cls: "third_party" if cls is Base else original(cls))
    result = extract_config_provenance(Child)
    assert result.constructor_chain.stop_reason == "scope"
    assert result.constructor_chain.stopped_at is Base
    assert not result.fields
    assert not result.exclusions  # The excluded signature was never read.


def test_real_filter_decisions_are_reported():
    class Filtered:
        __config_annotated_only__ = True

        def __init__(self, positional: int = 1, /, *, _private: int = 2,
                     ordinary: int = 3, marked: Annotated[int, Field(ge=0)] = 4):
            pass

    result = extract_config_provenance(Filtered)
    assert list(result.fields) == ["marked"]
    assert [(d.name, d.reason) for d in result.exclusions] == [
        ("positional", "positional_only"), ("_private", "private_name"),
        ("ordinary", "annotated_only"),
    ]
    assert all(d.status == "excluded" and d.declared_on is Filtered for d in result.exclusions)


def test_wiring_declaration_and_runtime_slot_have_independent_owners():
    class Base:
        __wiring__ = {"strategy": ["old"]}

        def __init__(self, *, strategy: str):
            pass

    class Child(Base):
        __wiring__ = {"strategy": ["new"]}

    result = extract_config_provenance(Child)
    strategy = result.fields["strategy"]
    assert strategy.declared_on is Base
    assert strategy.receptor.declared_on is Base
    assert strategy.receptor.init_owner is Base
    assert strategy.wiring.winner.declared_on is Child
    assert [d.declared_on for d in strategy.wiring.declarations] == [Child, Base]


def test_slotless_and_declare_only_fields_attribute_to_wiring_owner():
    class Base:
        __wiring__ = {"strategy": ["one"]}

    class Child(Base):
        pass

    result = extract_config_provenance(Child)
    item = result.fields["strategy"]
    assert item.source == "wiring" and item.declared_on is Base
    assert item.receptor is None
    assert item.wiring.winner.declared_on is Base


def test_wiring_exclusion_does_not_exclude_an_ordinary_constructor_knob():
    class Base:
        __wiring__ = {"strategy": ["one"]}

        def __init__(self, strategy: str = "ordinary"):
            pass

    class Child(Base):
        __wiring__ = {"strategy": None}

    result = extract_config_provenance(Child)
    assert result.fields["strategy"].source == "init"
    assert result.wiring["strategy"].excluded
    assert result.fields["strategy"].wiring.excluded
    assert not result.exclusions


def test_wired_parameter_beyond_config_cut_still_names_runtime_receptor():
    class Base:
        def __init__(self, *, strategy: str):
            pass

    class Child(Base):
        __wiring__ = {"strategy": ["one"]}

        def __init__(self, **kwargs):
            pass

    result = extract_config_provenance(Child, mro_depth=0)
    assert result.constructor_chain.stopped_at is Base
    assert result.fields["strategy"].init_owner is Base
    assert result.fields["strategy"].receptor.init_owner is Base
    assert result.fields["strategy"].declarations[0].declared_on is Base


def test_annotation_fallback_owner_is_preserved():
    class Base:
        __wiring__ = {"strategy": ["one"]}

        def __init__(self, *, strategy):
            pass

    class Child(Base):
        strategy: list[str]

    result = extract_config_provenance(Child)
    slot = result.fields["strategy"].receptor
    assert slot.declared_on is Base
    assert slot.annotation_declared_on is Child
    assert slot.shape == "list"


def test_annotation_only_receptor_and_plain_annotations():
    class Base:
        strategy: str
        ordinary: int
        ignored: ClassVar[int] = 1
        __wiring__ = {"strategy": ["one"]}

    result = extract_config_provenance(Base)
    assert list(result.fields) == ["strategy"]
    assert result.fields["strategy"].source == "annotation"
    assert result.fields["strategy"].receptor.via == "annotation"
    assert result.fields["strategy"].receptor.annotation_declared_on is Base
    assert not result.exclusions  # Arbitrary class attrs are not config candidates.


def test_schema_path_preserves_schema_and_field_owners():
    class BaseSchema(BaseModel):
        budget: int = 1
        inherited: str = "yes"

    class Schema(BaseSchema):
        budget: int = 2

    class Base:
        __config_schema__ = Schema

    class Child(Base):
        pass

    result = extract_config_provenance(Child)
    assert result.path == "schema" and result.source_model is Schema
    assert result.schema_declared_on is Base
    assert result.constructor_chain is None
    assert result.fields["budget"].declared_on is Schema
    assert result.fields["inherited"].declared_on is BaseSchema
    assert [d.declared_on for d in result.fields["budget"].declarations] == [Schema, BaseSchema]
    assert extract_config_schema(Child) is Schema
    assert "__config_provenance__" not in vars(Schema)


def test_flattened_model_path_and_field_owner():
    class Knobs(BaseModel):
        budget: int = 3

    class Base:
        __config_from_param__ = "config"

        def __init__(self, config: Knobs):
            pass

    class Child(Base):
        pass

    result = extract_config_provenance(Child)
    assert result.path == "flattened"
    assert result.source_model is Knobs and result.flattened_param == "config"
    assert result.schema_declared_on is Base
    assert result.fields["budget"].source == "flattened"
    assert result.fields["budget"].declared_on is Knobs
    assert tuple(result.fields) == tuple(extract_config_schema(Child).model_fields)


def test_dataclass_generated_constructor_and_declaring_class_differ():
    @dataclass
    class Base:
        values: list[int] = field(default_factory=list)

    @dataclass
    class Child(Base):
        budget: int = 2

    result = extract_config_provenance(Child)
    assert result.fields["values"].source == "dataclass"
    assert result.fields["values"].declared_on is Base
    assert result.fields["values"].init_owner is Child
    assert result.fields["budget"].declared_on is Child


def test_attrs_alias_is_constructor_field_name():
    @attrs.define
    class Item:
        _value: int = attrs.field(default=3, alias="count")

    result = extract_config_provenance(Item)
    assert tuple(result.fields) == ("count",)
    assert result.fields["count"].source == "generated"
    assert result.fields["count"].declared_on is Item


def test_model_path_rebuilt_fields_keep_original_owner():
    class Base(BaseModel):
        budget: int = 1

    class Child(Base):
        value: str = "v"

    result = extract_config_provenance(Child)
    assert result.path == "reflection"
    assert result.fields["budget"].source == "model"
    assert result.fields["budget"].declared_on is Base
    assert result.fields["value"].declared_on is Child
    assert result.constructor_chain.kind == "model"


def test_empty_class_still_has_provenance():
    class Empty:
        pass

    result = extract_config_provenance(Empty)
    assert not result.fields and not result.wiring and not result.exclusions
    assert result.constructor_chain.kind == "none"
    assert extract_config_schema(Empty) is None


def test_result_containers_are_frozen_and_are_mutation_snapshots():
    class Item:
        __wiring__ = {"strategy": ["one"]}

    result = extract_config_provenance(Item)
    with pytest.raises(TypeError):
        result.fields["extra"] = result.fields["strategy"]
    with pytest.raises(FrozenInstanceError):
        result.fields["strategy"].name = "new"
    with pytest.raises(FrozenInstanceError):
        result.constructor_chain.stop_reason = "invented"
    Item.__wiring__["strategy"].append("two")
    assert result.wiring["strategy"].winner.spec.allowed_keys == ("one",)
    assert extract_config_provenance(Item).wiring["strategy"].winner.spec.allowed_keys == ("one", "two")


def test_registrar_entry_uses_its_settings_and_reports_full_mro():
    from typing import Protocol, runtime_checkable
    from conscribe import create_registrar

    @runtime_checkable
    class Contract(Protocol):
        pass

    registrar = create_registrar("provenance_scope", Contract, mro_depth=0)

    class Base:
        def __init__(self, *, base: int = 1):
            pass

    @registrar.register("item")
    class Item(Base):
        def __init__(self, *, own: int = 2, **kwargs):
            pass

    result = registrar.get_config_provenance("item")
    assert tuple(result.fields) == tuple(registrar.get_config_schema("item").model_fields)
    assert tuple(result.fields) == ("own",)
    assert result.constructor_chain.stop_reason == "depth"
    assert result.mro == (Item, Base, object)


def test_annotation_receptor_history_uses_shared_slot_rules():
    class Base:
        strategy: str
        __wiring__ = {"strategy": ["one"]}

    class Removed(Base):
        strategy: ClassVar[str]

    class Child(Removed):
        strategy: list[str]

    result = extract_config_provenance(Child)
    assert [(d.declared_on, d.status, d.reason) for d in result.fields["strategy"].declarations] == [
        (Child, "included", None), (Removed, "excluded", "pseudo_field"),
        (Base, "overridden", "overridden"),
    ]


def test_provenance_does_not_change_generated_artifacts():
    from typing import Protocol, runtime_checkable
    from conscribe import (
        create_registrar, build_layer_config, generate_layer_config_source,
        generate_layer_json_schema, compute_registry_fingerprint,
    )

    @runtime_checkable
    class Contract(Protocol):
        pass

    registrar = create_registrar("provenance_parity", Contract, discriminator_field="kind")

    @registrar.register("item")
    class Item:
        __wiring__ = {"mode": ["fast", "slow"]}

        def __init__(self, *, budget: Annotated[int, Field(ge=1)] = 3):
            pass

    before = build_layer_config(registrar)
    source = generate_layer_config_source(before)
    schema = generate_layer_json_schema(before)
    fingerprint = compute_registry_fingerprint(registrar)
    record = registrar.get_config_provenance("item")
    assert tuple(record.fields) == ("budget", "mode")
    after = build_layer_config(registrar)
    assert generate_layer_config_source(after) == source
    assert generate_layer_json_schema(after) == schema
    assert compute_registry_fingerprint(registrar) == fingerprint
