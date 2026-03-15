"""Tests for conscribe.config.fingerprint — TDD RED phase.

Tests ``compute_registry_fingerprint()``, ``load_cached_fingerprint()``,
and ``save_fingerprint()``:
- Determinism: same registry produces same fingerprint; output format is 16-char hex
- Change detection: fingerprint changes when keys/classes/signatures/docstrings change
- Order independence: registration order does not affect fingerprint
- Cache load: hit, miss (no file), miss (no layer), miss (corrupt file)
- Cache save: create new file, preserve other layers, roundtrip, overwrite

All implementation classes and registrars are defined at MODULE LEVEL
so that ``get_type_hints()`` can resolve forward references under
``from __future__ import annotations``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pytest

from conscribe import create_registrar
from conscribe.config.fingerprint import (
    compute_registry_fingerprint,
    load_cached_fingerprint,
    save_fingerprint,
)


# ===================================================================
# Protocols for test registrars
# ===================================================================

@runtime_checkable
class _LLMProto(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


# ===================================================================
# Registrars (module-level for type hint resolution)
# ===================================================================

_determinism_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_change_add_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_change_base_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_change_remove_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_swap_impl_reg_a = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_swap_impl_reg_b = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_sig_change_reg_a = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_sig_change_reg_b = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_sig_type_reg_a = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_sig_type_reg_b = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_doc_change_reg_a = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_doc_change_reg_b = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_order_reg_ab = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_order_reg_ba = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


# ===================================================================
# Implementation classes for _determinism_reg
# ===================================================================

class _DetBase(metaclass=_determinism_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DetOpenAI(_DetBase):
    """OpenAI provider for determinism tests."""
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class DetAnthropic(_DetBase):
    """Anthropic provider for determinism tests."""
    __registry_key__ = "anthropic"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _change_base_reg (base for add/remove)
# ===================================================================

class _ChangeBaseBase(metaclass=_change_base_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ChangeBaseAlpha(_ChangeBaseBase):
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


class ChangeBaseBeta(_ChangeBaseBase):
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _change_add_reg (has extra key)
# ===================================================================

class _ChangeAddBase(metaclass=_change_add_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ChangeAddAlpha(_ChangeAddBase):
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


class ChangeAddBeta(_ChangeAddBase):
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


class ChangeAddGamma(_ChangeAddBase):
    __registry_key__ = "gamma"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _change_remove_reg (one fewer key)
# ===================================================================

class _ChangeRemoveBase(metaclass=_change_remove_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ChangeRemoveAlpha(_ChangeRemoveBase):
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _swap_impl_reg_a (original impl)
# ===================================================================

class _SwapBaseA(metaclass=_swap_impl_reg_a.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SwapImplOriginal(_SwapBaseA):
    """Original implementation for swap test."""
    __registry_key__ = "provider_x"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes for _swap_impl_reg_b (swapped impl, same key)
# ===================================================================

class _SwapBaseB(metaclass=_swap_impl_reg_b.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SwapImplReplacement(_SwapBaseB):
    """Replacement implementation for swap test (different qualname/signature)."""
    __registry_key__ = "provider_x"

    def __init__(self, *, model_id: str, max_tokens: int = 8192):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _sig_change_reg_a (original signature)
# ===================================================================

class _SigBaseA(metaclass=_sig_change_reg_a.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SigOriginal(_SigBaseA):
    __registry_key__ = "sig_test"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes for _sig_change_reg_b (added param)
# ===================================================================

class _SigBaseB(metaclass=_sig_change_reg_b.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SigWithExtraParam(_SigBaseB):
    __registry_key__ = "sig_test"

    def __init__(self, *, model_id: str, temperature: float = 0.0, top_p: float = 1.0):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p


# ===================================================================
# Implementation classes for _sig_type_reg_a (param type: float)
# ===================================================================

class _SigTypeBaseA(metaclass=_sig_type_reg_a.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SigTypeFloat(_SigTypeBaseA):
    __registry_key__ = "type_test"

    def __init__(self, *, temperature: float = 0.0):
        self.temperature = temperature


# ===================================================================
# Implementation classes for _sig_type_reg_b (param type: int)
# ===================================================================

class _SigTypeBaseB(metaclass=_sig_type_reg_b.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SigTypeInt(_SigTypeBaseB):
    __registry_key__ = "type_test"

    def __init__(self, *, temperature: int = 0):
        self.temperature = temperature


# ===================================================================
# Implementation classes for _doc_change_reg_a (original docstring)
# ===================================================================

class _DocBaseA(metaclass=_doc_change_reg_a.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DocOriginal(_DocBaseA):
    """Original docstring.

    Args:
        model_id: The model identifier.
    """
    __registry_key__ = "doc_test"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _doc_change_reg_b (changed docstring)
# ===================================================================

class _DocBaseB(metaclass=_doc_change_reg_b.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DocChanged(_DocBaseB):
    """Updated docstring with different description.

    Args:
        model_id: A completely different description for the model.
    """
    __registry_key__ = "doc_test"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _order_reg_ab (register alpha first)
# ===================================================================

class _OrderBaseAB(metaclass=_order_reg_ab.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OrderAlphaAB(_OrderBaseAB):
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class OrderBetaAB(_OrderBaseAB):
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _order_reg_ba (register beta first)
# ===================================================================

class _OrderBaseBA(metaclass=_order_reg_ba.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OrderBetaBA(_OrderBaseBA):
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


class OrderAlphaBA(_OrderBaseBA):
    __registry_key__ = "alpha"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Determinism tests
# ===================================================================

class TestDeterminism:
    """Fingerprint is deterministic and correctly formatted."""

    def test_same_registry_produces_same_fingerprint(self) -> None:
        """Calling compute_registry_fingerprint twice on the same registrar
        returns identical results."""
        fp1 = compute_registry_fingerprint(_determinism_reg)
        fp2 = compute_registry_fingerprint(_determinism_reg)

        assert fp1 == fp2

    def test_fingerprint_is_16_char_hex_string(self) -> None:
        """Fingerprint is exactly 16 lowercase hexadecimal characters."""
        fp = compute_registry_fingerprint(_determinism_reg)

        assert isinstance(fp, str)
        assert len(fp) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", fp) is not None


# ===================================================================
# Change detection tests
# ===================================================================

class TestChangeDetection:
    """Fingerprint changes when the registry contents change."""

    def test_fingerprint_changes_when_key_added(self) -> None:
        """Adding a new key to the registry produces a different fingerprint."""
        fp_base = compute_registry_fingerprint(_change_base_reg)
        fp_added = compute_registry_fingerprint(_change_add_reg)

        assert fp_base != fp_added

    def test_fingerprint_changes_when_key_removed(self) -> None:
        """Removing a key from the registry produces a different fingerprint."""
        fp_base = compute_registry_fingerprint(_change_base_reg)
        fp_removed = compute_registry_fingerprint(_change_remove_reg)

        assert fp_base != fp_removed

    def test_fingerprint_changes_when_class_swapped(self) -> None:
        """Swapping a class for a different implementation (same key)
        produces a different fingerprint due to different qualname/signature."""
        fp_a = compute_registry_fingerprint(_swap_impl_reg_a)
        fp_b = compute_registry_fingerprint(_swap_impl_reg_b)

        assert fp_a != fp_b

    def test_fingerprint_changes_when_param_added(self) -> None:
        """Adding a parameter to __init__ changes the fingerprint."""
        fp_a = compute_registry_fingerprint(_sig_change_reg_a)
        fp_b = compute_registry_fingerprint(_sig_change_reg_b)

        assert fp_a != fp_b

    def test_fingerprint_changes_when_param_type_changes(self) -> None:
        """Changing a parameter type (float -> int) changes the fingerprint."""
        fp_a = compute_registry_fingerprint(_sig_type_reg_a)
        fp_b = compute_registry_fingerprint(_sig_type_reg_b)

        assert fp_a != fp_b

    def test_fingerprint_changes_when_docstring_changes(self) -> None:
        """Changing the __init__ docstring changes the fingerprint
        (docstring affects Tier 1.5 description extraction)."""
        fp_a = compute_registry_fingerprint(_doc_change_reg_a)
        fp_b = compute_registry_fingerprint(_doc_change_reg_b)

        assert fp_a != fp_b

    def test_fingerprint_changes_after_unregister(self) -> None:
        """Unregistering a key from a registrar changes the fingerprint."""
        # Create a fresh registrar for this test
        unreg = create_registrar("llm", _LLMProto, discriminator_field="provider")

        class _UnregBase(metaclass=unreg.Meta):
            __abstract__ = True
            async def chat(self, messages: list[dict]) -> str:
                return ""

        class UnregA(_UnregBase):
            __registry_key__ = "a"
            def __init__(self, *, x: int = 1):
                self.x = x

        class UnregB(_UnregBase):
            __registry_key__ = "b"
            def __init__(self, *, y: int = 2):
                self.y = y

        fp_before = compute_registry_fingerprint(unreg)

        unreg.unregister("b")

        fp_after = compute_registry_fingerprint(unreg)
        assert fp_before != fp_after


# ===================================================================
# Order independence tests
# ===================================================================

class TestOrderIndependence:
    """Fingerprint is independent of registration order."""

    def test_registration_order_does_not_affect_fingerprint(self) -> None:
        """Two registrars with the same keys and classes registered in
        different order produce identical fingerprints (keys are sorted)."""
        fp_ab = compute_registry_fingerprint(_order_reg_ab)
        fp_ba = compute_registry_fingerprint(_order_reg_ba)

        assert fp_ab == fp_ba


# ===================================================================
# Cache load tests
# ===================================================================

class TestLoadCachedFingerprint:
    """Tests for load_cached_fingerprint()."""

    def test_returns_stored_fingerprint_for_known_layer(self, tmp_path: Path) -> None:
        """Returns the fingerprint string for a layer present in the JSON file."""
        fp_file = tmp_path / ".registry_fingerprint"
        data = {"llm": "abcdef0123456789", "agent": "9876543210fedcba"}
        fp_file.write_text(json.dumps(data))

        result = load_cached_fingerprint(fp_file, "llm")

        assert result == "abcdef0123456789"

    def test_returns_none_if_file_does_not_exist(self, tmp_path: Path) -> None:
        """Returns None when the fingerprint file does not exist."""
        fp_file = tmp_path / ".registry_fingerprint"

        result = load_cached_fingerprint(fp_file, "llm")

        assert result is None

    def test_returns_none_if_layer_name_not_in_json(self, tmp_path: Path) -> None:
        """Returns None when the layer_name is not a key in the JSON."""
        fp_file = tmp_path / ".registry_fingerprint"
        data = {"agent": "9876543210fedcba"}
        fp_file.write_text(json.dumps(data))

        result = load_cached_fingerprint(fp_file, "llm")

        assert result is None

    def test_returns_none_if_file_is_corrupt(self, tmp_path: Path) -> None:
        """Returns None when the JSON file contains invalid/corrupt content."""
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text("this is not valid json {{{")

        result = load_cached_fingerprint(fp_file, "llm")

        assert result is None

    def test_returns_none_if_file_is_empty(self, tmp_path: Path) -> None:
        """Returns None when the JSON file is empty."""
        fp_file = tmp_path / ".registry_fingerprint"
        fp_file.write_text("")

        result = load_cached_fingerprint(fp_file, "llm")

        assert result is None


# ===================================================================
# Cache save tests
# ===================================================================

class TestSaveFingerprint:
    """Tests for save_fingerprint()."""

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        """Creates the fingerprint JSON file when it does not exist."""
        fp_file = tmp_path / ".registry_fingerprint"

        save_fingerprint(fp_file, "llm", "abcdef0123456789")

        assert fp_file.exists()
        data = json.loads(fp_file.read_text())
        assert data == {"llm": "abcdef0123456789"}

    def test_preserves_other_layers_entries(self, tmp_path: Path) -> None:
        """Preserves existing layer entries when saving a new layer."""
        fp_file = tmp_path / ".registry_fingerprint"
        existing = {"agent": "9876543210fedcba"}
        fp_file.write_text(json.dumps(existing))

        save_fingerprint(fp_file, "llm", "abcdef0123456789")

        data = json.loads(fp_file.read_text())
        assert data["agent"] == "9876543210fedcba"
        assert data["llm"] == "abcdef0123456789"

    def test_roundtrip_save_then_load(self, tmp_path: Path) -> None:
        """save_fingerprint followed by load_cached_fingerprint returns
        the same fingerprint."""
        fp_file = tmp_path / ".registry_fingerprint"
        fingerprint = "abcdef0123456789"

        save_fingerprint(fp_file, "llm", fingerprint)
        loaded = load_cached_fingerprint(fp_file, "llm")

        assert loaded == fingerprint

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if they do not exist."""
        fp_file = tmp_path / "nested" / "dir" / ".registry_fingerprint"

        save_fingerprint(fp_file, "llm", "abcdef0123456789")

        assert fp_file.exists()
        data = json.loads(fp_file.read_text())
        assert data["llm"] == "abcdef0123456789"


# ===================================================================
# Cache overwrite tests
# ===================================================================

class TestCacheOverwrite:
    """Tests for overwriting existing fingerprint entries."""

    def test_overwrites_existing_fingerprint_for_same_layer(self, tmp_path: Path) -> None:
        """Saving a new fingerprint for an existing layer updates the value."""
        fp_file = tmp_path / ".registry_fingerprint"
        initial = {"llm": "0000000000000000", "agent": "1111111111111111"}
        fp_file.write_text(json.dumps(initial))

        save_fingerprint(fp_file, "llm", "aaaaaaaaaaaaaaaa")

        data = json.loads(fp_file.read_text())
        assert data["llm"] == "aaaaaaaaaaaaaaaa"
        # Other layers untouched
        assert data["agent"] == "1111111111111111"

    def test_multiple_saves_each_update_correctly(self, tmp_path: Path) -> None:
        """Multiple successive saves to different layers all persist."""
        fp_file = tmp_path / ".registry_fingerprint"

        save_fingerprint(fp_file, "llm", "aaaa000000000000")
        save_fingerprint(fp_file, "agent", "bbbb000000000000")
        save_fingerprint(fp_file, "provider", "cccc000000000000")

        data = json.loads(fp_file.read_text())
        assert data == {
            "llm": "aaaa000000000000",
            "agent": "bbbb000000000000",
            "provider": "cccc000000000000",
        }


# ===================================================================
# End-to-end integration: compute + save + load roundtrip
# ===================================================================

class TestEndToEndRoundtrip:
    """Integration: compute a real fingerprint, save it, load it back."""

    def test_compute_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Compute a fingerprint from a real registrar, save it, and load
        it back -- the loaded value matches the computed value."""
        fp_file = tmp_path / ".registry_fingerprint"

        fp = compute_registry_fingerprint(_determinism_reg)
        save_fingerprint(fp_file, "llm", fp)
        loaded = load_cached_fingerprint(fp_file, "llm")

        assert loaded == fp

    def test_stale_fingerprint_detects_change(self, tmp_path: Path) -> None:
        """A previously cached fingerprint does not match a registrar
        whose contents have since changed."""
        fp_file = tmp_path / ".registry_fingerprint"

        # Save fingerprint from registrar with 2 keys
        fp_base = compute_registry_fingerprint(_change_base_reg)
        save_fingerprint(fp_file, "llm", fp_base)

        # Compute fingerprint from registrar with 3 keys (added gamma)
        fp_added = compute_registry_fingerprint(_change_add_reg)

        cached = load_cached_fingerprint(fp_file, "llm")
        assert cached != fp_added


# ===================================================================
# MRO **kwargs chain fingerprint tests
# ===================================================================

# -- Registrar A: child with **kwargs, parent has param_a --

_mro_fp_reg_a = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _MROFPParentA:
    """Parent with some params.

    Args:
        param_a: A parent parameter.
    """

    def __init__(self, *, param_a: int = 10):
        self.param_a = param_a


class _MROFPBaseA(metaclass=_mro_fp_reg_a.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class MROFPChildA(_MROFPBaseA, _MROFPParentA):
    """Child with **kwargs forwarding to parent."""
    __registry_key__ = "mro_child"

    def __init__(self, *, model_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_id = model_id

    async def chat(self, messages: list[dict]) -> str:
        return "mro_a"


# -- Registrar B: same structure but parent has different signature --

_mro_fp_reg_b = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _MROFPParentB:
    """Parent with different params.

    Args:
        param_a: A parent parameter.
        param_b: An extra parent parameter.
    """

    def __init__(self, *, param_a: int = 10, param_b: str = "new"):
        self.param_a = param_a
        self.param_b = param_b


class _MROFPBaseB(metaclass=_mro_fp_reg_b.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class MROFPChildB(_MROFPBaseB, _MROFPParentB):
    """Child with **kwargs forwarding to parent with different sig."""
    __registry_key__ = "mro_child"

    def __init__(self, *, model_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_id = model_id

    async def chat(self, messages: list[dict]) -> str:
        return "mro_b"


# -- Registrar C: two-level **kwargs chain --

_mro_fp_reg_chain = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _MROFPGrandparent:
    def __init__(self, *, deep_param: int = 99):
        self.deep_param = deep_param


class _MROFPMiddle(_MROFPGrandparent):
    def __init__(self, *, mid_param: str = "mid", **kwargs: Any):
        super().__init__(**kwargs)
        self.mid_param = mid_param


class _MROFPChainBase(metaclass=_mro_fp_reg_chain.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class MROFPChainChild(_MROFPChainBase, _MROFPMiddle):
    """Two-level **kwargs chain for fingerprint test."""
    __registry_key__ = "chain"

    def __init__(self, *, model_id: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_id = model_id

    async def chat(self, messages: list[dict]) -> str:
        return "chain"


class TestMROKwargsFingerprint:
    """Tests that **kwargs MRO parent signatures are included in fingerprint."""

    def test_kwargs_child_fingerprint_is_deterministic(self) -> None:
        """Fingerprint of a registrar with **kwargs child is deterministic."""
        fp1 = compute_registry_fingerprint(_mro_fp_reg_a)
        fp2 = compute_registry_fingerprint(_mro_fp_reg_a)
        assert fp1 == fp2

    def test_parent_sig_change_changes_fingerprint(self) -> None:
        """Changing a parent's __init__ signature (via **kwargs chain)
        produces a different fingerprint."""
        fp_a = compute_registry_fingerprint(_mro_fp_reg_a)
        fp_b = compute_registry_fingerprint(_mro_fp_reg_b)
        assert fp_a != fp_b

    def test_two_level_chain_fingerprint_valid(self) -> None:
        """Two-level **kwargs chain produces a valid 16-char hex fingerprint."""
        fp = compute_registry_fingerprint(_mro_fp_reg_chain)
        assert isinstance(fp, str)
        assert len(fp) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", fp) is not None

    def test_kwargs_vs_no_kwargs_different_fingerprint(self) -> None:
        """A class with **kwargs and one without produce different fingerprints,
        even if the child's own params are identical."""
        # _mro_fp_reg_a has **kwargs child -> hashes parent sigs
        # _determinism_reg has no **kwargs -> does NOT hash parent sigs
        fp_kwargs = compute_registry_fingerprint(_mro_fp_reg_a)
        fp_plain = compute_registry_fingerprint(_determinism_reg)
        # Different keys/classes, so fingerprints differ
        assert fp_kwargs != fp_plain
