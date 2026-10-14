"""af feedback defects 11-13: the shared three-file repro, as acceptance tests.

The package is af's ``pkg`` (``reg.py``, ``strat.py``, ``owner.py``, the
latter deliberately *without* ``from __future__ import annotations``),
written to a temporary directory under a unique package name.  The 1.3.0
stub for it contained ``FieldInfo(annotation=NoneType, ...)``, bound
``OwnerBase``, ``_Receptors`` and ``pkg`` to ``Any`` and declared
``strategy: StratBase`` although ``_Receptors`` declares
``strategy: Strategy``.
"""
from __future__ import annotations

import ast
import re

import pytest

from conscribe.introspect import wiring_slots
from conscribe.stubs import collect_class_stub_info, write_layer_stubs

from ._helpers import assert_valid_stub, class_body, temp_package

REG = '''
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class Strategy(Protocol):
    def pick(self) -> int: ...

@runtime_checkable
class OwnerP(Protocol):
    def run(self) -> None: ...

Strat = create_registrar("{pkg}_strat", Strategy, discriminator_field="strategy_id")
Own = create_registrar("{pkg}_owner", OwnerP, discriminator_field="owner_id")
'''

STRAT = '''
from {pkg}.reg import Strat

class StratBase(metaclass=Strat.Meta):
    __abstract__ = True
    def pick(self) -> int: return 0

class Fast(StratBase):
    __registry_key__ = "fast"

class Slow(StratBase):
    __registry_key__ = "slow"
'''

# owner.py -- no ``from __future__ import annotations`` (on purpose).
OWNER = '''
from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from {pkg}.reg import Own, Strategy
from {pkg}.strat import StratBase  # noqa: F401

@dataclass
class Sibling:                       # a helper type in the same file
    y: int = 0

class _Receptors:
    strategy: Strategy               # the receptor, declared as the Protocol

class OwnerBase(metaclass=Own.Meta):
    __abstract__ = True
    def run(self): ...

class Owner(_Receptors, OwnerBase):
    __registry_key__ = "owner"
    __wiring__ = {wiring}
    def __init__(self, *, depth: Annotated[int, Field(ge=1, description="depth")] = 1,
                 sib: Annotated[Sibling | None, Field(description="sibling")] = None) -> None: ...
'''


def _files(pkg: str, wiring: str = '{"strategy": "strat"}') -> dict[str, str]:
    wiring = wiring.replace('"strat"', f'"{pkg}_strat"')
    return {
        "reg.py": REG.format(pkg=pkg),
        "strat.py": STRAT.format(pkg=pkg),
        "owner.py": OWNER.replace("{pkg}", pkg).replace("{wiring}", wiring),
    }


def _load(pkg: str):
    import importlib

    importlib.import_module(f"{pkg}.strat")
    owner = importlib.import_module(f"{pkg}.owner")
    reg = importlib.import_module(f"{pkg}.reg")
    return reg, owner


class TestSharedRepro:
    def test_strategy_has_a_slot_so_nothing_is_stubbed(self, tmp_path):
        """``strategy`` is annotated on ``_Receptors``: the source declares it."""
        with temp_package(tmp_path / "src", "afrepro_a", _files("afrepro_a")):
            reg, owner = _load("afrepro_a")
            slot = wiring_slots(owner.Owner)["strategy"].slot
            assert slot is not None and slot.declared_on is owner._Receptors

            assert collect_class_stub_info(owner.Owner) is None
            out = tmp_path / "stubs"
            assert write_layer_stubs(reg.Own, output_dir=out) == []
            assert not out.exists() or not any(out.rglob("*.pyi"))

    def test_stub_for_a_slotless_entry_is_valid(self, tmp_path):
        """Add an injected entry so a stub is written; it must be clean."""
        wiring = '{"strategy": "strat", "helper": "strat"}'
        with temp_package(tmp_path / "src", "afrepro_b", _files("afrepro_b", wiring)):
            reg, owner = _load("afrepro_b")
            written = write_layer_stubs(reg.Own, output_dir=tmp_path / "stubs")
            assert [p.relative_to(tmp_path / "stubs").as_posix() for p in written] == [
                "afrepro_b/owner.pyi"
            ]
            source = written[0].read_text(encoding="utf-8")
            assert_valid_stub(source)  # parses, binds every name, imports exist

        # Defect 11: evaluated Annotated metadata and NoneType never leak.
        assert "FieldInfo(" not in source
        assert "NoneType" not in source
        assert "Field" not in source
        # Defect 12: no ``X = Any`` bindings; same-module classes become
        # nominal placeholders and other modules' names are imported.
        assert not re.search(r"^\w+ = Any$", source, flags=re.M), source
        assert "class OwnerBase(Any): ..." in source
        assert "class _Receptors(Any): ..." in source
        assert "class Sibling(Any): ..." in source
        assert "afrepro_b.owner.Sibling" not in source
        assert "from afrepro_b.reg import Strategy" in source
        # Defect 13: the slotted ``strategy`` is not re-declared; the
        # injected ``helper`` is typed as the registry protocol.
        owner_cls = class_body(source, "Owner")
        assert "strategy" not in owner_cls
        assert "helper: Strategy  # wired from: afrepro_b_strat" in owner_cls
        assert "StratBase" not in source
        assert "def __init__(self, *, depth: int = 1, sib: Sibling | None = None) -> None: ..." in owner_cls

    def test_stub_written_next_to_the_source(self, tmp_path):
        wiring = '{"strategy": "strat", "helper": "strat"}'
        with temp_package(tmp_path, "afrepro_c", _files("afrepro_c", wiring)) as package:
            reg, _owner = _load("afrepro_c")
            written = write_layer_stubs(reg.Own)
            assert written == [package / "owner.pyi"]
            tree = assert_valid_stub(written[0].read_text(encoding="utf-8"))
        assert any(isinstance(n, ast.ClassDef) and n.name == "Owner" for n in tree.body)

    @pytest.mark.parametrize("entry", ['{"registry": "strat", "declare_only": True}'])
    def test_declare_only_entry_is_never_stubbed(self, tmp_path, entry):
        wiring = '{"strategy": "strat", "plan": ' + entry + "}"
        with temp_package(tmp_path / "src", "afrepro_d", _files("afrepro_d", wiring)):
            reg, owner = _load("afrepro_d")
            assert wiring_slots(owner.Owner)["plan"].declare_only is True
            assert collect_class_stub_info(owner.Owner) is None
            assert write_layer_stubs(reg.Own, output_dir=tmp_path / "stubs") == []
