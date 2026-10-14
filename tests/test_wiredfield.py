"""WiredField semantics, the injection rule and the class-creation wiring check.

Covers conscribe 2.0's D4 and the class-time half of F1:

* **R\\*** -- class-level access to a :class:`~conscribe.wiring.WiredField`
  raises ``AttributeError`` iff the class holding the placeholder annotates the
  name in its own body; otherwise it returns the descriptor.  This is what lets
  ``@dataclass`` / pydantic see "no default" for a receptor annotated without
  one, although the registration metaclass runs before the decorator.
* ``wired_field()`` / ``wired_fields()`` lookups.
* The injection rule of ``inject_wired_descriptors``: constructor parameters,
  ``declare_only`` entries and own bindings never get a placeholder; inherited
  bindings are kept or shadowed by kind.
* ``check_wiring_declaration`` on every registration path: a malformed
  ``__wiring__`` fails the class statement and registers nothing.
* af's defect-4 acceptance cases (dataclass receptors, slots, pydantic, bridges).

No ``from __future__ import annotations`` here: dataclass annotations are real
objects, as in the af repro.
"""
import contextlib
import dataclasses
import functools
import inspect
import sys
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable
from unittest import mock

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.dataclasses import dataclass as pydantic_dataclass

import conscribe
import conscribe.wiring
from conscribe import (
    build_composed_config,
    create_registrar,
    wired_field,
    wired_fields,
)
from conscribe.config import generate_composed_config_source
from conscribe.exceptions import WiringResolutionError
from conscribe.wiring import WiredField, inject_wired_descriptors


@runtime_checkable
class Runnable(Protocol):
    def run(self) -> None: ...


Archiver = create_registrar("wf_archiver", Runnable, discriminator_field="archiver_id")
Hook = create_registrar("wf_hook", Runnable, discriminator_field="hook_id")
Other = create_registrar("wf_other", Runnable, discriminator_field="other_id")


class ArchBase(metaclass=Archiver.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class NetArchiver(ArchBase):
    __registry_key__ = "net"


class HookBase(metaclass=Hook.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class OtherBase(metaclass=Other.Meta):
    __abstract__ = True

    def run(self) -> None: ...


class Disk(OtherBase):
    __registry_key__ = "disk"


# af defect 4, verbatim shape: a dataclass receptor with a None default ...
@dataclass
class TraceWriter(HookBase):
    __registry_key__ = "trace_writer"
    __wiring__ = {"archive": "wf_archiver"}
    archive: list | None = None


# ... and the annotation-only variant (a required receptor).  ``knob`` keeps
# the 1.x extractor off its single-parameter path.
@dataclass
class RequiredArchive(HookBase):
    __registry_key__ = "required_archive"
    __wiring__ = {"archive": "wf_archiver"}
    archive: list[ArchBase]
    knob: int = 0


@pytest.fixture(autouse=True)
def _isolate(conscribe_isolated):
    """Classes defined in a test are rolled back (module-level ones stay)."""
    yield conscribe_isolated


def _default(cls: type, name: str):
    return {f.name: f for f in dataclasses.fields(cls)}[name].default


@contextlib.contextmanager
def _generated_module(source: str, name: str = "wf_generated_composed"):
    """Execute generated source as an importable module (as users load it)."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(source, f"<{name}>", "exec"), module.__dict__)  # noqa: S102
        yield module
    finally:
        sys.modules.pop(name, None)


# ===================================================================
# af defect 4: dataclass receptors
# ===================================================================


class TestDataclassReceptors:
    def test_none_default_survives(self) -> None:
        assert _default(TraceWriter, "archive") is None
        assert TraceWriter().archive is None
        assert TraceWriter.__dict__["archive"] is None  # the body's default, no placeholder
        assert wired_field(TraceWriter, "archive") is None

    def test_none_default_config_field_is_optional(self) -> None:
        model = Hook.get_config_schema("trace_writer")
        info = model.model_fields["archive"]
        assert not info.is_required()
        assert info.default is None

    def test_composed_source_has_no_placeholder_and_execs(self) -> None:
        result = build_composed_config({"wf_hook": Hook, "wf_archiver": Archiver})
        source = generate_composed_config_source(result)
        assert "WiredField(" not in source
        assert "archive: Optional[list[" in source  # the None default kept it optional

        with _generated_module(source) as module:
            top = module.ComposedConfig
            config = top.model_validate({"wf_hook": [{"hook_id": "trace_writer"}]})
            assert config.wf_hook[0].archive is None
            config = top.model_validate(
                {"wf_hook": [{"hook_id": "trace_writer", "archive": [{"archiver_id": "net"}]}]}
            )
            assert config.wf_hook[0].archive[0].archiver_id == "net"
            with pytest.raises(ValidationError, match="archive"):
                top.model_validate({"wf_hook": [{"hook_id": "required_archive"}]})

    def test_in_process_composed_model_agrees(self) -> None:
        result = build_composed_config({"wf_hook": Hook, "wf_archiver": Archiver})
        config = result.top_level_type.model_validate({"wf_hook": [{"hook_id": "trace_writer"}]})
        assert config.wf_hook[0].archive is None

    def test_annotation_only_receptor_is_required(self) -> None:
        assert _default(RequiredArchive, "archive") is dataclasses.MISSING
        with pytest.raises(TypeError, match="archive"):
            RequiredArchive()
        assert RequiredArchive(archive=[NetArchiver()]).archive[0].__class__ is NetArchiver
        # The class annotates the receptor, so it never got a placeholder of
        # its own (attrs would read one from __dict__ as the default).
        assert "archive" not in RequiredArchive.__dict__
        assert wired_field(RequiredArchive, "archive") is None
        with pytest.raises(AttributeError):
            RequiredArchive.archive

    def test_annotation_only_config_field_is_required(self) -> None:
        from typing import Literal

        info = Hook.get_config_schema("required_archive").model_fields["archive"]
        assert info.is_required()
        assert info.annotation == list[Literal["net"]]

    def test_default_factory_survives(self) -> None:
        @dataclass
        class Factory(HookBase):
            __wiring__ = {"archive": "wf_archiver"}
            archive: list = field(default_factory=list)

        assert Factory().archive == []
        assert Factory().archive is not Factory().archive
        assert dataclasses.fields(Factory)[0].default_factory is list

    def test_field_init_false_keeps_init_false(self) -> None:
        # The class body binds a dataclasses.Field, so no placeholder is
        # installed over it; @dataclass then removes the class attribute.
        @dataclass
        class Late(HookBase):
            __wiring__ = {"archive": "wf_archiver"}
            archive: list = field(init=False)

        (f,) = dataclasses.fields(Late)
        assert f.init is False
        late = Late()
        with pytest.raises(AttributeError):
            late.archive
        late.archive = ["injected"]
        assert late.archive == ["injected"]

    def test_required_receptor_after_defaulted_field_needs_kw_only(self) -> None:
        # A receptor annotated without a default is a required dataclass field,
        # so the usual dataclass ordering rule applies to it.
        with pytest.raises(TypeError, match="non-default argument"):
            @dataclass
            class Misordered(HookBase):
                __wiring__ = {"archive": "wf_archiver"}
                knob: int = 1
                archive: list

        @dataclass(kw_only=True)
        class KwOnly(HookBase):
            __wiring__ = {"archive": "wf_archiver"}
            knob: int = 1
            archive: list

        assert KwOnly(archive=[]).archive == []
        assert _default(KwOnly, "archive") is dataclasses.MISSING

    def test_reannotating_child_of_unannotated_holder_is_required(self) -> None:
        # The placeholder lives on MixinOwner, which does not annotate the
        # name (its mixin does), so MixinOwner.archive returns it.  The
        # dataclass child annotates the name: class access through the child
        # raises, so @dataclass sees no default; the child gets no placeholder
        # of its own.
        class Receptors:
            archive: list

        class MixinOwner(Receptors, HookBase):
            __wiring__ = {"archive": "wf_archiver"}

        assert MixinOwner.archive is MixinOwner.__dict__["archive"]

        @dataclass
        class Child(MixinOwner):
            archive: list

        assert _default(Child, "archive") is dataclasses.MISSING
        assert "archive" not in Child.__dict__
        assert Child(archive=[1]).archive == [1]
        assert MixinOwner.archive is MixinOwner.__dict__["archive"]  # unaffected

    def test_reannotating_child_of_annotated_holder_reuses_it(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"archive": "wf_archiver"}
            archive: list

        @dataclass
        class Child(Holder):
            archive: list

        # The inherited placeholder already answers class access with
        # AttributeError, so it is reused rather than shadowed.
        assert "archive" not in Child.__dict__
        assert _default(Child, "archive") is dataclasses.MISSING

    def test_inherited_plain_default_stays_a_default(self) -> None:
        # Plain Python semantics: the child re-annotates the name without a
        # default of its own, so the inherited class attribute is the default.
        class Parent(HookBase):
            __wiring__ = {"archive": "wf_archiver"}
            archive = None

        @dataclass
        class Child(Parent):
            archive: Optional[list]

        assert Parent().archive is None
        assert "archive" not in Child.__dict__  # no placeholder over the inherited value
        assert wired_fields(Child) == {}
        assert _default(Child, "archive") is None
        assert Child().archive is None

    def test_slots_dataclass_registers_once_and_works(self) -> None:
        @dataclass(slots=True)
        class Slotted(HookBase):
            __registry_key__ = "wf_slotted"
            __wiring__ = {
                "archive": "wf_archiver",
                "required": "wf_archiver",
                "env": "wf_other",
            }
            required: list
            archive: Optional[list] = None

        assert Hook.get("wf_slotted") is Slotted
        assert Hook.keys().count("wf_slotted") == 1
        assert "__slots__" in Slotted.__dict__
        assert _default(Slotted, "archive") is None
        assert _default(Slotted, "required") is dataclasses.MISSING

        obj = Slotted(required=[])
        assert obj.archive is None
        # A wired name that is not a field keeps its placeholder; HookBase
        # gives the instances a __dict__, so injecting works.
        with pytest.raises(AttributeError, match="not been injected"):
            obj.env
        obj.env = Disk()
        assert isinstance(obj.env, Disk)
        assert wired_fields(Slotted) == {"env": Slotted.__dict__["env"]}

    def test_frozen_slots_dataclass_receptor(self) -> None:
        @dataclass(slots=True, frozen=True)
        class Frozen(HookBase):
            __wiring__ = {"archive": "wf_archiver"}
            archive: Optional[list] = None

        assert Frozen().archive is None
        assert Frozen(archive=[1]).archive == [1]


class TestDataclassRegistrationPaths:
    def test_register_above_dataclass(self) -> None:
        Manual = create_registrar("wf_manual_above", Runnable)

        @Manual.register("above")
        @dataclass
        class Above:
            __wiring__ = {"archive": "wf_archiver", "env": "wf_other"}
            archive: Optional[list] = None

            def run(self) -> None: ...

        # archive is a dataclass field (a constructor parameter) by now.
        assert "archive" in [f.name for f in dataclasses.fields(Above)]
        assert Above().archive is None
        assert wired_field(Above, "archive") is None
        assert isinstance(wired_field(Above, "env"), WiredField)

    def test_register_below_dataclass(self) -> None:
        Manual = create_registrar("wf_manual_below", Runnable)

        @dataclass
        @Manual.register("below")
        class Below:
            __wiring__ = {"archive": "wf_archiver", "required": "wf_archiver"}
            required: list
            archive: Optional[list] = None

            def run(self) -> None: ...

        assert _default(Below, "archive") is None
        assert _default(Below, "required") is dataclasses.MISSING
        assert Manual.get("below") is Below

    def test_propagate_hook_with_dataclass_child(self) -> None:
        Manual = create_registrar("wf_manual_prop", Runnable)

        @Manual.register("root", propagate=True)
        class Root:
            def run(self) -> None: ...

        @dataclass
        class Leaf(Root):
            __wiring__ = {"archive": "wf_archiver", "required": "wf_archiver"}
            required: list
            archive: Optional[list] = None

        assert Manual.get("leaf") is Leaf
        assert _default(Leaf, "archive") is None
        assert _default(Leaf, "required") is dataclasses.MISSING

    def test_pydantic_dataclass_receptors(self) -> None:
        @pydantic_dataclass
        class PydDC(HookBase):
            __wiring__ = {"archive": "wf_archiver", "sink": "wf_archiver"}
            sink: list
            archive: Optional[list] = None

        fields_ = PydDC.__pydantic_fields__
        assert fields_["archive"].default is None
        assert fields_["sink"].is_required()
        assert PydDC(sink=[]).archive is None
        with pytest.raises(ValidationError, match="sink"):
            PydDC()

    def test_pydantic_dataclass_slots(self) -> None:
        @pydantic_dataclass(slots=True)
        class PydSlots(HookBase):
            __registry_key__ = "wf_pyd_slots"
            __wiring__ = {"archive": "wf_archiver"}
            archive: Optional[list] = None

        assert Hook.get("wf_pyd_slots") is PydSlots
        assert PydSlots().archive is None


# ===================================================================
# pydantic models and bridges
# ===================================================================


class TestPydanticModels:
    def test_model_field_receptor_gets_no_placeholder(self) -> None:
        Bridge = Hook.bridge(BaseModel)

        class Parent(Bridge):
            __wiring__ = {"archive": "wf_archiver"}
            archive: Optional[List] = None

            def run(self) -> None: ...

        assert "archive" not in Parent.__dict__
        assert Parent().archive is None
        assert Parent.model_fields["archive"].default is None

    def test_subclass_keeps_default_without_shadow_warning(self, recwarn) -> None:
        Bridge = Hook.bridge(BaseModel)

        class Parent(Bridge):
            __wiring__ = {"archive": "wf_archiver"}
            archive: Optional[List] = None

            def run(self) -> None: ...

        class Child(Parent):
            archive: Optional[List] = None

        class Plain(Parent):
            pass

        assert Child.model_fields["archive"].default is None
        assert Plain.model_fields["archive"].default is None
        assert not [w for w in recwarn if "shadows an attribute" in str(w.message)]

    def test_wired_name_that_is_not_a_field_gets_a_placeholder(self) -> None:
        Bridge = Hook.bridge(BaseModel)

        class Model(Bridge):
            __wiring__ = {"llm": "wf_other"}
            knob: int = 0

            def run(self) -> None: ...

        placeholder = Model.__dict__["llm"]
        assert isinstance(placeholder, WiredField)
        assert Model.llm is placeholder
        with pytest.raises(AttributeError, match="not been injected"):
            Model().llm
        assert "llm" not in Model.model_fields

    def test_child_reannotating_an_inherited_placeholder_gets_a_required_field(
        self, recwarn
    ) -> None:
        """Review probe 22: pydantic collects a child's fields (with getattr)
        before conscribe sees the class, so the inherited placeholder must
        answer class access through the annotating child with AttributeError."""
        Bridge = Hook.bridge(BaseModel)

        class Parent(Bridge):
            __abstract__ = True
            __wiring__ = {"llm": "wf_other"}  # slot-less here: a placeholder

            def run(self) -> None: ...

        class Child(Parent):
            llm: str  # a required pydantic field, wired

        assert isinstance(Parent.__dict__["llm"], WiredField)
        assert Parent.llm is Parent.__dict__["llm"]
        assert Child.model_fields["llm"].is_required()
        with pytest.raises(ValidationError, match="llm"):
            Child()
        assert Child(llm="disk").llm == "disk"
        assert "llm" not in Child.__dict__ or not isinstance(Child.__dict__["llm"], WiredField)
        model = Hook.get_config_schema("child")
        assert model.model_fields["llm"].is_required()


class TestBridge:
    def test_bridged_external_class_read_only_property_is_shadowed(self) -> None:
        class External:
            @property
            def env(self):
                return "stub"

            def run(self) -> None: ...

        Bridge = Other.bridge(External)

        class Wrapped(Bridge):
            __wiring__ = {"env": "wf_other"}

        obj = Wrapped()
        with pytest.raises(AttributeError, match="not been injected"):
            obj.env
        obj.env = Disk()
        assert isinstance(obj.env, Disk)
        assert Other.get("wrapped") is Wrapped

    def test_bridged_external_init_param_gets_no_placeholder(self) -> None:
        class External:
            def __init__(self, env: str = "disk") -> None:
                self.env = env

            def run(self) -> None: ...

        Bridge = Other.bridge(External)

        class Wrapped(Bridge):
            __wiring__ = {"env": "wf_other"}

        assert "env" not in Wrapped.__dict__
        assert Wrapped().env == "disk"


# ===================================================================
# R*: class-level access
# ===================================================================


class TestClassAccess:
    def test_unannotated_name_returns_descriptor(self) -> None:
        class Agent(HookBase):
            __wiring__ = {"env": "wf_other"}

        placeholder = Agent.__dict__["env"]
        assert isinstance(placeholder, WiredField)
        assert Agent.env is placeholder
        assert hasattr(Agent, "env")

    def test_annotating_class_gets_no_placeholder_of_its_own(self) -> None:
        """Spec 4.5: attrs reads defaults from ``__dict__``, so a class that
        annotates a wired name never holds a placeholder for it."""

        class Agent(HookBase):
            __wiring__ = {"env": "wf_other"}
            env: OtherBase

        assert "env" not in Agent.__dict__
        assert wired_field(Agent, "env") is None
        assert not hasattr(Agent, "env")
        with pytest.raises(AttributeError, match="'Agent' object has no attribute 'env'"):
            Agent().env
        agent = Agent()
        agent.env = Disk()
        assert isinstance(agent.env, Disk)

    def test_child_annotating_an_inherited_name_raises_on_class_access(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Child(Holder):
            env: OtherBase

        placeholder = Holder.__dict__["env"]
        assert Holder.env is placeholder  # the holder itself does not annotate
        assert "env" not in Child.__dict__
        with pytest.raises(AttributeError) as info:
            Child.env
        message = str(info.value)
        assert ".Child.env has no class-level value: " in message
        assert ".Child annotates the wired attribute 'env' without a default" in message
        assert "(wired from registry 'wf_other')" in message
        assert "conscribe.wired_field(Child, 'env')" in message
        assert "inspect.getattr_static(Child, 'env')" in message
        assert not hasattr(Child, "env")
        assert getattr(Child, "env", "fallback") == "fallback"
        assert inspect.getattr_static(Child, "env") is placeholder
        assert wired_field(Child, "env") is placeholder
        # Still injected: instance access keeps the not-injected message.
        with pytest.raises(AttributeError, match="has not been injected yet"):
            Child().env

    def test_annotation_between_accessor_and_holder_raises(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Middle(Holder):
            env: OtherBase

        class Leaf(Middle):
            pass

        with pytest.raises(
            AttributeError,
            match=r"\.Leaf\.env has no class-level value: .*\.Middle annotates the wired "
            r"attribute 'env'",
        ):
            Leaf.env
        assert Holder.env is Holder.__dict__["env"]

    def test_mixin_annotation_keeps_descriptor_access(self) -> None:
        """af's AgentReceptors style: the annotation lives on a plain mixin
        above the class holding the placeholder."""

        class Receptors:
            env: OtherBase

        class Agent(Receptors, HookBase):
            __wiring__ = {"env": "wf_other"}

        class Concrete(Agent):
            pass

        assert Agent.env is Agent.__dict__["env"]
        assert Concrete.env is Agent.__dict__["env"]

    def test_instance_access_is_unchanged_for_injected_names(self) -> None:
        class Agent(HookBase):
            __wiring__ = {"env": "wf_other", "llm": "wf_other"}

        agent = Agent()
        for name in ("env", "llm"):
            with pytest.raises(
                AttributeError,
                match=rf"Wired attribute '{name}' on Agent has not been injected yet\. "
                r"Expected injection from registry 'wf_other'\.",
            ):
                getattr(agent, name)
        agent.env = Disk()
        assert isinstance(agent.env, Disk)

    def test_manual_descriptor_in_class_body(self) -> None:
        class Annotated_:
            env: OtherBase = WiredField("env", "wf_other")

        class Bare:
            env = WiredField("env", "wf_other")

        with pytest.raises(AttributeError, match="no class-level value"):
            Annotated_.env
        assert isinstance(inspect.getattr_static(Annotated_, "env"), WiredField)
        assert Bare.env is Bare.__dict__["env"]

    def test_dataclass_with_manual_annotated_descriptor_has_no_default(self) -> None:
        @dataclass
        class Manual:
            env: OtherBase = WiredField("env", "wf_other")

        assert _default(Manual, "env") is dataclasses.MISSING

    def test_pydantic_model_with_manual_annotated_descriptor_has_no_default(self) -> None:
        model = create_model("Manual", env=(Any, WiredField("env", "wf_other")))
        assert model.model_fields["env"].is_required()
        # Wrapped in Field(), the placeholder is an ordinary (leaked) default:
        # pydantic calls __get__ outside attribute lookup (no class holds it).
        leaked = create_model("Leaked", env=(Any, Field(default=WiredField("env", "wf_other"))))
        assert isinstance(leaked.model_fields["env"].default, WiredField)

    def test_introspection_tools_still_see_it(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Agent(Holder):
            env: OtherBase

        members = dict(inspect.getmembers(Agent))
        assert members["env"] is Holder.__dict__["env"]
        assert "env" in dir(Agent)

    def test_direct_get_without_owner(self) -> None:
        bare = WiredField("env", "wf_other")
        assert bare.__get__(None) is bare
        assert bare.__get__(None, None) is bare

    def test_repr(self) -> None:
        assert repr(WiredField("env", "wf_other")) == "WiredField('env', registry='wf_other')"
        assert repr(WiredField("env")) == "WiredField('env')"
        with pytest.raises(TypeError):
            WiredField("env", "wf_other", annotated=True)  # type: ignore[call-arg]


class TestPatchingWiredAttributes:
    """Guidance for tests that patch a wired attribute on the class."""

    def test_monkeypatch_unannotated_name(self) -> None:
        class Agent(HookBase):
            __wiring__ = {"env": "wf_other"}

        placeholder = Agent.__dict__["env"]
        fake = Disk()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Agent, "env", fake)
            assert Agent().env is fake
        assert Agent.__dict__["env"] is placeholder

    def test_monkeypatch_annotated_name_needs_raising_false(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Agent(Holder):
            env: OtherBase  # class access raises: nothing to replace

        placeholder = Holder.__dict__["env"]
        fake = Disk()
        with pytest.MonkeyPatch.context() as mp:
            with pytest.raises(AttributeError, match="has no attribute 'env'"):
                mp.setattr(Agent, "env", fake)
            mp.setattr(Agent, "env", fake, raising=False)
            assert Agent().env is fake
        assert "env" not in Agent.__dict__
        assert Holder.__dict__["env"] is placeholder

    def test_mock_patch_object_annotated_name_needs_create(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Sub(Holder):
            env: OtherBase

        fake = Disk()
        with pytest.raises(AttributeError, match="does not have the attribute 'env'"):
            with mock.patch.object(Sub, "env", fake):
                pass  # pragma: no cover
        with mock.patch.object(Sub, "env", fake, create=True):
            assert Sub().env is fake
        assert "env" not in Sub.__dict__
        # The holder's own placeholder patches fine (found in its __dict__).
        with mock.patch.object(Holder, "env", fake):
            assert Holder().env is fake
        assert isinstance(Holder.__dict__["env"], WiredField)


# ===================================================================
# wired_field() / wired_fields()
# ===================================================================


class TestLookup:
    def test_finds_own_and_inherited_placeholders(self) -> None:
        class Base(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Child(Base):
            __wiring__ = {"llm": "wf_other"}

        assert wired_field(Base, "env") is Base.__dict__["env"]
        assert wired_field(Child, "env") is Base.__dict__["env"]
        assert wired_field(Child, "llm") is Child.__dict__["llm"]
        assert wired_fields(Child) == {
            "env": Base.__dict__["env"],
            "llm": Child.__dict__["llm"],
        }
        assert list(wired_fields(Child)) == ["env", "llm"]  # base first

    def test_none_for_unknown_shadowed_and_constructor_names(self) -> None:
        class Base(HookBase):
            __wiring__ = {"env": "wf_other", "llm": "wf_other"}

        class Shadowing(Base):
            env = "bound in the subclass body"

        class WithInit(Base):
            def __init__(self, llm: str = "x") -> None:
                self.llm = llm

        assert wired_field(Base, "nope") is None
        assert wired_field(Shadowing, "env") is None
        assert wired_field(WithInit, "llm") is None
        assert wired_fields(Shadowing) == {"llm": Base.__dict__["llm"]}
        assert wired_fields(WithInit) == {"env": Base.__dict__["env"]}
        assert wired_fields(HookBase) == {}

    def test_does_not_invoke_descriptors(self) -> None:
        class Holder(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Agent(Holder):
            env: OtherBase

        with mock.patch.object(WiredField, "__get__", side_effect=AssertionError):
            assert wired_field(Agent, "env") is Holder.__dict__["env"]
            assert wired_fields(Agent) == {"env": Holder.__dict__["env"]}

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(TypeError, match=r"wired_field\(\) expects a class"):
            wired_field(object(), "env")
        with pytest.raises(TypeError, match=r"wired_field\(\) expects an attribute name"):
            wired_field(HookBase, 3)
        with pytest.raises(TypeError, match=r"wired_fields\(\) expects a class"):
            wired_fields("HookBase")

    def test_exported(self) -> None:
        assert conscribe.wired_field is wired_field
        assert conscribe.wired_fields is wired_fields
        assert {"wired_field", "wired_fields", "WiredField"} <= set(conscribe.__all__)


# ===================================================================
# Injection rule
# ===================================================================


class TestInjectionRule:
    def test_constructor_parameters_get_no_placeholder(self) -> None:
        class Base(HookBase):
            def __init__(self, inherited: str = "", **kwargs) -> None:
                super().__init__(**kwargs)
                self.inherited = inherited

        class Agent(Base):
            __wiring__ = {
                "own": "wf_other",
                "inherited": "wf_other",
                "injected": "wf_other",
            }

            def __init__(self, own: str = "", **kwargs) -> None:
                super().__init__(**kwargs)
                self.own = own

        assert "own" not in Agent.__dict__
        assert "inherited" not in Agent.__dict__  # forwarded through **kwargs
        assert isinstance(Agent.__dict__["injected"], WiredField)

    def test_declare_only_entries_get_no_placeholder(self) -> None:
        class Agent(HookBase):
            __wiring__ = {
                "negotiated": {"registry": "wf_other", "declare_only": True},
                "injected": {"registry": "wf_other"},
            }

        assert "negotiated" not in Agent.__dict__
        assert wired_field(Agent, "negotiated") is None
        assert isinstance(Agent.__dict__["injected"], WiredField)
        with pytest.raises(AttributeError, match="has no attribute 'negotiated'"):
            Agent().negotiated

    def test_own_bindings_are_never_shadowed(self) -> None:
        class Agent(HookBase):
            __wiring__ = {k: "wf_other" for k in ("value", "prop", "meth")}
            value = None

            @property
            def prop(self):
                return "own property"

            def meth(self):
                return "own method"

        agent = Agent()
        assert agent.value is None
        assert agent.prop == "own property"
        assert agent.meth() == "own method"
        assert wired_fields(Agent) == {}

    def test_inherited_bindings_kept_or_shadowed_by_kind(self) -> None:
        class Parent(HookBase):
            __abstract__ = True
            plain = "inherited default"

            @property
            def read_only(self):
                return "stub"

            @property
            def writable(self):
                return self.__dict__.get("_writable", "unset")

            @writable.setter
            def writable(self, value):
                self.__dict__["_writable"] = value

            def method(self):
                return "stub"

            @staticmethod
            def static():
                return "stub"

            @classmethod
            def klass(cls):
                return "stub"

            @functools.cached_property
            def cached(self):
                return "computed"

        names = ("plain", "read_only", "writable", "method", "static", "klass", "cached")

        class Agent(Parent):
            __wiring__ = {name: "wf_other" for name in names}

        assert set(wired_fields(Agent)) == {"read_only", "method", "static", "klass"}
        before = Agent()
        assert before.plain == "inherited default"
        assert before.writable == "unset"
        assert before.cached == "computed"
        for name in ("read_only", "method", "static", "klass"):
            with pytest.raises(AttributeError, match="not been injected"):
                getattr(before, name)

        agent = Agent()
        for name in names:
            setattr(agent, name, f"injected {name}")
            assert getattr(agent, name) == f"injected {name}"
        assert agent.__dict__["_writable"] == "injected writable"  # via the setter

    def test_inherited_slot_member_is_kept(self) -> None:
        Slot = create_registrar("wf_slot_layer", Runnable)

        class Root(metaclass=Slot.Meta):
            __abstract__ = True
            __slots__ = ("archive",)

            def run(self) -> None: ...

        class Leaf(Root):
            __slots__ = ()
            __wiring__ = {"archive": "wf_archiver"}

        assert "archive" not in Leaf.__dict__
        leaf = Leaf()
        leaf.archive = [NetArchiver()]
        assert isinstance(leaf.archive[0], NetArchiver)

    def test_explicit_protocol_property_stub_is_shadowed(self) -> None:
        """A read-only property inherited from an explicit Protocol base."""

        @runtime_checkable
        class HasEnv(Protocol):
            @property
            def env(self) -> OtherBase: ...

            def run(self) -> None: ...

        Layer = create_registrar("wf_proto_layer", Runnable, base_metaclass=type(HasEnv))

        class LayerBase(metaclass=Layer.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Impl(HasEnv, LayerBase):
            __wiring__ = {"env": "wf_other"}

            def run(self) -> None: ...

        impl = Impl()
        with pytest.raises(AttributeError, match="not been injected"):
            impl.env
        impl.env = Disk()
        assert isinstance(impl.env, Disk)

    def test_same_registry_placeholder_is_inherited(self) -> None:
        class Base(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Child(Base):
            __wiring__ = {"env": ("wf_other", ["disk"])}

        assert "env" not in Child.__dict__
        assert wired_field(Child, "env") is Base.__dict__["env"]

    def test_rewired_to_another_registry_gets_its_own_placeholder(self) -> None:
        class Base(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Child(Base):
            __wiring__ = {"env": "wf_archiver"}

        own = Child.__dict__["env"]
        assert own.registry_name == "wf_archiver"
        with pytest.raises(AttributeError, match="registry 'wf_archiver'"):
            Child().env

    def test_reannotating_plain_child_raises_on_class_access(self) -> None:
        class Base(HookBase):
            __wiring__ = {"env": "wf_other"}

        class Child(Base):
            env: OtherBase

        assert "env" not in Child.__dict__  # annotated: no placeholder of its own
        assert Base.env is Base.__dict__["env"]
        with pytest.raises(AttributeError, match="no class-level value"):
            Child.env

    def test_mode3_and_dict_form_registry_names(self) -> None:
        class Agent(HookBase):
            __wiring__ = {
                "browser": ["chromium", "firefox"],
                "src": {"registry": "wf_other", "keys": ["disk"]},
            }

        assert Agent.__dict__["browser"].registry_name == ""
        assert Agent.__dict__["src"].registry_name == "wf_other"

    def test_direct_call_skips_malformed_entries(self) -> None:
        class Agent:
            __wiring__ = {"bad": ("wf_other",), "ok": "wf_other", 1: "wf_other"}

        inject_wired_descriptors(Agent)  # never raises for a bad declaration
        assert set(wired_fields(Agent)) == {"ok"}

    def test_abstract_classes_get_placeholders(self) -> None:
        class Abstract(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Concrete(Abstract):
            pass

        assert isinstance(Abstract.__dict__["env"], WiredField)
        assert "env" not in Concrete.__dict__
        assert wired_field(Concrete, "env") is Abstract.__dict__["env"]


# ===================================================================
# Class-creation wiring grammar check (F1) on every registration path
# ===================================================================

MALFORMED = [
    pytest.param({"x": ("wf_archiver",)}, "malformed", id="one-tuple"),
    pytest.param({"x": ("wf_archiver", "net")}, "malformed", id="keys-not-a-list"),
    pytest.param(
        {"x": {"registry": "wf_archiver", "defualt": "net"}}, "malformed", id="unknown-dict-field"
    ),
    pytest.param({"x": ("wf_archiver", ["net", 3])}, "non_str_key", id="non-str-key"),
    pytest.param({1: "wf_archiver"}, "malformed", id="non-str-name"),
    pytest.param({"x": []}, "empty_keys", id="empty-literal-list"),
    pytest.param({"x": ("wf_archiver", [])}, "empty_keys", id="empty-key-list"),
    pytest.param(["x"], "not_a_dict", id="not-a-dict"),
]


def _kinds(info: pytest.ExceptionInfo) -> list[str]:
    return [p.kind for p in info.value.problems]


class TestClassCreationCheck:
    @pytest.mark.parametrize(("wiring", "kind"), MALFORMED)
    def test_path_a_metaclass(self, wiring, kind) -> None:
        before = Hook.get_all()
        with pytest.raises(WiringResolutionError) as info:
            class Bad(HookBase):
                __registry_key__ = "bad"
                __wiring__ = wiring

        assert _kinds(info) == [kind]
        assert Hook.get_all() == before

    def test_message_names_class_key_and_entry(self) -> None:
        with pytest.raises(WiringResolutionError) as info:
            class Bad(HookBase):
                __registry_key__ = "bad"
                __wiring__ = {"archive": {"registry": "wf_archiver", "defualt": "net"}}

        message = str(info.value)
        assert message.startswith(
            "TestClassCreationCheck.test_message_names_class_key_and_entry.<locals>.Bad "
            "(key 'bad') wiring 'archive': unknown field(s) 'defualt' (did you mean 'default'?)"
        )

    def test_every_problem_is_reported_at_once(self) -> None:
        with pytest.raises(WiringResolutionError) as info:
            class Bad(HookBase):
                __wiring__ = {"a": ("wf_archiver",), "b": [], "c": ("wf_archiver", [1])}

        assert _kinds(info) == ["malformed", "empty_keys", "non_str_key"]

    def test_multi_key_class_registers_nothing(self) -> None:
        before = Hook.get_all()
        with pytest.raises(WiringResolutionError):
            class Bad(HookBase):
                __registry_key__ = ["bad_a", "bad_b"]
                __wiring__ = {"x": ("wf_archiver",)}

        assert Hook.get_all() == before

    def test_cross_registry_diamond_registers_nothing(self) -> None:
        Left = create_registrar("wf_left", Runnable)
        Right = create_registrar("wf_right", Runnable)

        class Dual(metaclass=Left.Meta | Right.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        with pytest.raises(WiringResolutionError):
            class Bad(Dual):
                __wiring__ = {"x": ("wf_archiver",)}

        assert Left.keys() == [] and Right.keys() == []

        with pytest.raises(WiringResolutionError):
            class OnlyLeft(Dual):  # skipped by one registry, checked by the other
                __skip_registries__ = ["wf_right"]
                __wiring__ = {"x": []}

        assert Left.keys() == [] and Right.keys() == []

    def test_inherited_malformed_entry_fails_registering_subclass(self) -> None:
        # A class the filters skip is not checked (it never registers) ...
        class AbstractBad(HookBase):
            __abstract__ = True
            __wiring__ = {"x": ("wf_archiver",), "env": "wf_other"}

        # ... but every registering subclass inherits and reports the entry.
        before = Hook.get_all()
        with pytest.raises(WiringResolutionError) as info:
            class Concrete(AbstractBad):
                pass

        assert "(entry declared on " in info.value.problems[0].detail
        assert "AbstractBad)" in info.value.problems[0].detail
        assert Hook.get_all() == before
        # The well-formed entry still got its placeholder on the abstract class.
        assert isinstance(AbstractBad.__dict__["env"], WiredField)
        assert "x" not in AbstractBad.__dict__

    @pytest.mark.parametrize(("wiring", "kind"), MALFORMED)
    def test_path_c_register(self, wiring, kind) -> None:
        Manual = create_registrar("wf_manual_check", Runnable)

        class Target:
            __wiring__ = wiring

            def run(self) -> None: ...

        before = dict(Target.__dict__)
        with pytest.raises(WiringResolutionError) as info:
            Manual.register("target")(Target)
        assert _kinds(info) == [kind]
        assert Manual.keys() == []
        assert dict(Target.__dict__) == before  # no key, no stamp, no placeholder

    @pytest.mark.parametrize(("wiring", "kind"), MALFORMED)
    def test_propagate_hook(self, wiring, kind) -> None:
        Manual = create_registrar("wf_manual_hook", Runnable)

        @Manual.register("root", propagate=True)
        class Root:
            def run(self) -> None: ...

        with pytest.raises(WiringResolutionError) as info:
            class Child(Root):
                __wiring__ = wiring

        assert _kinds(info) == [kind]
        assert Manual.keys() == ["root"]

    def test_propagate_hook_with_metaclass_registers_nothing(self) -> None:
        Manual = create_registrar("wf_manual_mixed", Runnable)

        @Manual.register("root", propagate=True)
        class Root(HookBase):
            __abstract__ = True

        before = Hook.get_all()
        with pytest.raises(WiringResolutionError):
            class Child(Root):
                __wiring__ = {"x": ("wf_archiver",)}

        assert Manual.keys() == ["root"]
        assert Hook.get_all() == before

    def test_well_formed_wiring_needs_no_registry(self) -> None:
        # Grammar only: unknown registries / keys are build-time problems.
        class Fine(HookBase):
            __wiring__ = {"x": ("wf_not_created_yet", ["anything"])}

        assert Hook.get("fine") is Fine

    def test_recreated_slots_class_is_checked_once(self, monkeypatch) -> None:
        calls: list[type] = []
        real = conscribe.wiring.check_wiring_declaration

        def spy(cls: type) -> None:
            calls.append(cls)
            real(cls)

        monkeypatch.setattr(conscribe.wiring, "check_wiring_declaration", spy)

        @dataclass(slots=True)
        class Slotted(HookBase):
            __registry_key__ = "wf_checked_once"
            __wiring__ = {"archive": "wf_archiver"}
            archive: Optional[list] = None

        assert len(calls) == 1
        assert calls[0] is not Slotted  # the pre-slots class; the copy rebinds
        assert Hook.get("wf_checked_once") is Slotted

    def test_skipped_classes_are_not_checked(self, monkeypatch) -> None:
        calls: list[type] = []
        monkeypatch.setattr(conscribe.wiring, "check_wiring_declaration", calls.append)

        class Abstract(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Concrete(Abstract):
            pass

        assert calls == [Concrete]


# ===================================================================
# Spec 4.5: attrs reads defaults from the class __dict__
# ===================================================================


class TestAttrsReceptors:
    """Review probe 7: an own placeholder became the default of an attrs field."""

    @pytest.mark.parametrize("slots", [True, False])
    def test_metaclass_path_annotated_receptor_is_required(self, slots) -> None:
        attrs = pytest.importorskip("attrs")

        @attrs.define(slots=slots)
        class Owner(OtherBase):
            __wiring__ = {"env": "wf_other"}
            env: OtherBase

        assert attrs.fields(Owner).env.default is attrs.NOTHING
        with pytest.raises(TypeError, match="env"):
            Owner()
        assert isinstance(Owner(env=Disk()).env, Disk)
        assert Other.get("owner") is Owner

    def test_child_of_a_placeholder_holder(self) -> None:
        """Probe 5 (b): the placeholder lives on the base; attrs on the child."""
        attrs = pytest.importorskip("attrs")

        class Base(OtherBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        @attrs.define
        class Child(Base):
            env: OtherBase

        assert attrs.fields(Child).env.default is attrs.NOTHING
        with pytest.raises(TypeError, match="env"):
            Child()

    def test_register_path(self) -> None:
        attrs = pytest.importorskip("attrs")
        Manual = create_registrar("wf_attrs_manual", Runnable)

        @Manual.register("owner")
        @attrs.define
        class Owner:
            __wiring__ = {"env": "wf_other", "llm": "wf_other"}
            env: OtherBase

            def run(self) -> None: ...

        assert attrs.fields(Owner).env.default is attrs.NOTHING
        assert wired_fields(Owner) == {"llm": Owner.__dict__["llm"]}


# ===================================================================
# Spec 4.5: tombstones for entries a subclass stops wiring
# ===================================================================


class TestTombstones:
    """Review probes 20 and 23."""

    def test_excluded_entry_covers_the_inherited_placeholder(self) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Plain(Base):
            __wiring__ = {"env": None}

        assert wired_fields(Plain) == {}
        assert wired_field(Plain, "env") is None
        with pytest.raises(AttributeError, match=r"^'Plain' object has no attribute 'env'$"):
            Plain().env
        with pytest.raises(AttributeError, match="excludes the inherited __wiring__ entry"):
            Plain.env
        assert not hasattr(Plain, "env")
        tombstone = inspect.getattr_static(Plain, "env")
        assert not isinstance(tombstone, WiredField)
        assert repr(tombstone) == (
            f"<conscribe: {Plain.__qualname__} no longer wires 'env': it excludes the "
            f"inherited __wiring__ entry with None>"
        )
        plain = Plain()
        plain.env = "assigned"  # an ordinary attribute again
        assert plain.env == "assigned"
        # The base keeps its placeholder.
        assert Base.env is Base.__dict__["env"]
        with pytest.raises(AttributeError, match="not been injected"):
            type("Sibling", (Base,), {})().env

    def test_declare_only_redeclaration_covers_the_inherited_placeholder(self) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Negotiating(Base):
            __wiring__ = {"env": {"registry": "wf_other", "declare_only": True}}

        assert wired_fields(Negotiating) == {}
        with pytest.raises(AttributeError, match=r"^'Negotiating' object has no attribute 'env'$"):
            Negotiating().env
        with pytest.raises(AttributeError, match="redeclares the __wiring__ entry declare_only"):
            Negotiating.env

    def test_excluding_dataclass_child_field_is_required(self) -> None:
        """Probe 20: the excluded name becomes an ordinary dataclass field."""

        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        @dataclass
        class DC(Base):
            __wiring__ = {"env": None}
            env: str

        assert _default(DC, "env") is dataclasses.MISSING
        with pytest.raises(TypeError, match="env"):
            DC()
        assert DC(env="x").env == "x"
        assert "env" not in DC.__dict__ or not isinstance(DC.__dict__["env"], WiredField)

    def test_annotating_excluder_gets_no_tombstone_but_plain_errors(self) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Plain(Base):
            __wiring__ = {"env": None}
            env: str

        assert "env" not in Plain.__dict__  # attrs-safe: nothing in __dict__
        with pytest.raises(AttributeError, match="no class-level value"):
            Plain.env
        with pytest.raises(AttributeError, match=r"^'Plain' object has no attribute 'env'$"):
            Plain().env

    def test_subclasses_inherit_the_exclusion(self) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Plain(Base):
            __wiring__ = {"env": None}

        class Leaf(Plain):
            pass

        assert "env" not in Leaf.__dict__
        assert wired_fields(Leaf) == {}
        with pytest.raises(AttributeError, match=r"^'Leaf' object has no attribute 'env'$"):
            Leaf().env

    def test_rewiring_below_an_exclusion_installs_a_fresh_placeholder(self) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Plain(Base):
            __abstract__ = True
            __wiring__ = {"env": None}

        class Rewired(Plain):
            __wiring__ = {"env": "wf_archiver"}

        own = Rewired.__dict__["env"]
        assert type(own) is WiredField and own.registry_name == "wf_archiver"
        assert wired_field(Rewired, "env") is own
        with pytest.raises(AttributeError, match="registry 'wf_archiver'"):
            Rewired().env

    def test_exclusion_through_multiple_inheritance(self) -> None:
        class Holder(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Excluder:
            __wiring__ = {"env": None}

        class Both(Excluder, Holder):
            pass

        assert wired_fields(Both) == {}
        with pytest.raises(AttributeError):
            Both.env
        with pytest.raises(AttributeError, match=r"^'Both' object has no attribute 'env'$"):
            Both().env

    def test_own_bindings_are_never_replaced_by_a_tombstone(self) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Plain(Base):
            __wiring__ = {"env": None}
            env = "own value"

        assert Plain.env == "own value"


# ===================================================================
# Placeholders on every registration path (review m2)
# ===================================================================


class TestPropagateHookPlaceholders:
    def test_skipped_subclass_gets_its_placeholders(self) -> None:
        """Probe 5 (c''): the hook returned before installing placeholders on
        subclasses the filters skip."""
        Manual = create_registrar("wf_hook_skip", Runnable)

        @Manual.register("base", propagate=True)
        class CBase:
            def run(self) -> None: ...

        class Skipped(CBase):
            __skip_registries__ = ["wf_hook_skip"]
            __wiring__ = {"llm": "wf_other"}

        class AbstractMid(CBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other"}

        class Leaf(AbstractMid):
            pass

        assert Manual.keys() == ["base", "leaf"]
        assert isinstance(Skipped.__dict__["llm"], WiredField)
        assert isinstance(AbstractMid.__dict__["env"], WiredField)
        assert wired_field(Leaf, "env") is AbstractMid.__dict__["env"]
        with pytest.raises(AttributeError, match="not been injected"):
            Skipped().llm

    def test_hook_registers_list_keys_like_the_metaclass(self) -> None:
        Manual = create_registrar("wf_hook_multi", Runnable)

        @Manual.register("base", propagate=True)
        class CBase:
            def run(self) -> None: ...

        class Multi(CBase):
            __registry_key__ = ["one", "two"]

        assert Manual.get("one") is Manual.get("two") is Multi
        assert Multi.__registry_key__ == "one"
        assert Multi.__registry_keys__ == ["one", "two"]


# ===================================================================
# Class creation cost (review m9)
# ===================================================================


class TestInjectionCost:
    def test_inherited_placeholders_need_no_constructor_walk(self, monkeypatch) -> None:
        """The constructor walk is the expensive question; subclasses that
        inherit their placeholders never ask it."""
        import conscribe.introspect

        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other", "llm": "wf_archiver"}

            def __init__(self, a: int = 1, **kwargs) -> None: ...

        calls: list[type] = []
        real = conscribe.introspect.init_param_names

        def counting(cls: type) -> tuple[str, ...]:
            calls.append(cls)
            return real(cls)

        monkeypatch.setattr(conscribe.introspect, "init_param_names", counting)

        class Leaf(Base):
            def __init__(self, b: int = 2, **kwargs) -> None:
                super().__init__(**kwargs)

        assert calls == []
        assert wired_fields(Leaf).keys() == {"env", "llm"}

    def test_each_base_is_parsed_once(self, monkeypatch) -> None:
        class Base(HookBase):
            __abstract__ = True
            __wiring__ = {"env": "wf_other", "llm": ("wf_archiver", ["net"])}

        calls: list[object] = []
        real = conscribe.wiring._check_entry

        def counting(name, value):
            calls.append(name)
            return real(name, value)

        monkeypatch.setattr(conscribe.wiring, "_check_entry", counting)
        for i in range(3):
            type(Base)(f"Leaf{i}", (Base,), {})
        assert calls == []  # Base was parsed when it was created
