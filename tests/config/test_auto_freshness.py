"""Tests for auto-freshness in discover() — TDD RED phase.

Tests the new ``auto_update_stubs`` parameter on ``discover()`` and its
integration with fingerprint comparison and stub regeneration.

After the import loop, if ``auto_update_stubs=True`` and ``stub_dir``
is provided, ``discover()`` should:

1. For each known registrar: compute fingerprint.
2. Compare against cached fingerprint from ``fingerprint_path``.
3. If different (or missing): regenerate stubs via codegen, save new fingerprint.

These tests mock the fingerprint/codegen functions rather than creating
real module files, since ``discover()`` imports modules and we want to
isolate the auto-freshness logic from the import mechanism.

All implementation classes and registrars are defined at MODULE LEVEL
so that ``get_type_hints()`` can resolve forward references under
``from __future__ import annotations``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, call, patch

import pytest

from conscribe import create_registrar


# ===================================================================
# Protocols for test registrars
# ===================================================================

@runtime_checkable
class _FreshnessLLMProto(Protocol):
    """LLM Protocol for auto-freshness tests."""

    async def chat(self, messages: list[dict]) -> str: ...


# ===================================================================
# Module-level registrar and implementations
# ===================================================================

_freshness_reg = create_registrar(
    "llm",
    _FreshnessLLMProto,
    strip_suffixes=["LLM", "Provider"],
    discriminator_field="provider",
)


class _FreshnessBase(metaclass=_freshness_reg.Meta):
    __abstract__ = True

    async def chat(self, messages: list[dict]) -> str:
        return ""


class FreshnessOpenAI(_FreshnessBase):
    """OpenAI provider for freshness tests."""
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Patch targets for fingerprint/codegen functions
# ===================================================================

_PATCH_COMPUTE = "conscribe.discover.compute_registry_fingerprint"
_PATCH_LOAD = "conscribe.discover.load_cached_fingerprint"
_PATCH_SAVE = "conscribe.discover.save_fingerprint"
_PATCH_GENERATE = "conscribe.discover.generate_layer_config_source"
_PATCH_BUILD = "conscribe.discover.build_layer_config"

# We also need to patch the registrar discovery inside discover().
# discover() needs a way to find known registrars for freshness checks.
# The exact mechanism depends on implementation, but the simplest approach
# is that discover() iterates registrars it can find. We mock at the
# function level to control what registrars are seen.
_PATCH_GET_REGISTRARS = "conscribe.discover._get_known_registrars"


# ===================================================================
# Helper: patch importlib to avoid needing real packages
# ===================================================================

def _patch_imports():
    """Return a patch that makes discover()'s import loop a no-op.

    We're testing the auto-freshness logic, not the import mechanism.
    We create a mock importlib module that only stubs ``import_module``,
    and patch it as a module-level attribute on ``conscribe.discover``
    so it does NOT replace the global ``importlib.import_module`` (which
    would break subsequent ``unittest.mock.patch()`` target resolution).
    """
    import sys

    mock_importlib = MagicMock()
    mock_importlib.import_module.return_value = MagicMock(__path__=None)
    return patch.object(
        sys.modules["conscribe.discover"],
        "importlib",
        mock_importlib,
    )


# ===================================================================
# Tests
# ===================================================================

class TestAutoFreshness:
    """Tests for the auto_update_stubs parameter on discover()."""

    def test_discover_with_auto_update_false_does_not_regenerate(
        self, tmp_path: Path,
    ) -> None:
        """When ``auto_update_stubs=False``, stubs are NOT regenerated
        regardless of fingerprint state."""
        from conscribe.discover import discover

        stub_dir = tmp_path / "generated"
        stub_dir.mkdir()
        fingerprint_path = tmp_path / ".registry_fingerprint"

        mock_generate = MagicMock()

        with (
            _patch_imports(),
            patch(_PATCH_COMPUTE, return_value="new_fingerprint"),
            patch(_PATCH_LOAD, return_value="old_fingerprint"),
            patch(_PATCH_SAVE) as mock_save,
            patch(_PATCH_GENERATE, mock_generate),
            patch(_PATCH_BUILD, return_value=MagicMock()),
            patch(_PATCH_GET_REGISTRARS, return_value=[_freshness_reg]),
        ):
            discover(
                "fake_module",
                auto_update_stubs=False,
                stub_dir=stub_dir,
                fingerprint_path=fingerprint_path,
            )

        # generate should NOT have been called
        mock_generate.assert_not_called()
        mock_save.assert_not_called()

    def test_discover_generates_stubs_when_fingerprint_missing(
        self, tmp_path: Path,
    ) -> None:
        """When no cached fingerprint exists (returns None), stubs
        are generated and the new fingerprint is saved."""
        from conscribe.discover import discover

        stub_dir = tmp_path / "generated"
        stub_dir.mkdir()
        fingerprint_path = tmp_path / ".registry_fingerprint"

        mock_build_result = MagicMock()
        mock_generate = MagicMock(return_value="# generated source")

        with (
            _patch_imports(),
            patch(_PATCH_COMPUTE, return_value="abc123") as mock_compute,
            patch(_PATCH_LOAD, return_value=None),  # No cached fingerprint
            patch(_PATCH_SAVE) as mock_save,
            patch(_PATCH_GENERATE, mock_generate),
            patch(_PATCH_BUILD, return_value=mock_build_result),
            patch(_PATCH_GET_REGISTRARS, return_value=[_freshness_reg]),
        ):
            discover(
                "fake_module",
                auto_update_stubs=True,
                stub_dir=stub_dir,
                fingerprint_path=fingerprint_path,
            )

        # Should have computed fingerprint
        mock_compute.assert_called()
        # Should have generated stubs (fingerprint was missing)
        mock_generate.assert_called_once()
        # Should have saved the new fingerprint
        mock_save.assert_called_once()

    def test_discover_skips_regeneration_when_fingerprint_matches(
        self, tmp_path: Path,
    ) -> None:
        """When the cached fingerprint matches the computed fingerprint,
        stubs are NOT regenerated."""
        from conscribe.discover import discover

        stub_dir = tmp_path / "generated"
        stub_dir.mkdir()
        fingerprint_path = tmp_path / ".registry_fingerprint"

        same_fingerprint = "abc123def456"
        mock_generate = MagicMock()

        with (
            _patch_imports(),
            patch(_PATCH_COMPUTE, return_value=same_fingerprint),
            patch(_PATCH_LOAD, return_value=same_fingerprint),  # Matches!
            patch(_PATCH_SAVE) as mock_save,
            patch(_PATCH_GENERATE, mock_generate),
            patch(_PATCH_BUILD, return_value=MagicMock()),
            patch(_PATCH_GET_REGISTRARS, return_value=[_freshness_reg]),
        ):
            discover(
                "fake_module",
                auto_update_stubs=True,
                stub_dir=stub_dir,
                fingerprint_path=fingerprint_path,
            )

        # generate should NOT have been called (fingerprints match)
        mock_generate.assert_not_called()
        # save should NOT have been called (nothing changed)
        mock_save.assert_not_called()

    def test_discover_regenerates_when_fingerprint_differs(
        self, tmp_path: Path,
    ) -> None:
        """When the cached fingerprint differs from the computed fingerprint,
        stubs are regenerated and the new fingerprint is saved."""
        from conscribe.discover import discover

        stub_dir = tmp_path / "generated"
        stub_dir.mkdir()
        fingerprint_path = tmp_path / ".registry_fingerprint"

        mock_build_result = MagicMock()
        mock_generate = MagicMock(return_value="# regenerated source")

        with (
            _patch_imports(),
            patch(_PATCH_COMPUTE, return_value="new_fingerprint_789"),
            patch(_PATCH_LOAD, return_value="old_fingerprint_123"),  # Different!
            patch(_PATCH_SAVE) as mock_save,
            patch(_PATCH_GENERATE, mock_generate),
            patch(_PATCH_BUILD, return_value=mock_build_result),
            patch(_PATCH_GET_REGISTRARS, return_value=[_freshness_reg]),
        ):
            discover(
                "fake_module",
                auto_update_stubs=True,
                stub_dir=stub_dir,
                fingerprint_path=fingerprint_path,
            )

        # Should have regenerated stubs
        mock_generate.assert_called_once()
        # Should have saved the new fingerprint
        mock_save.assert_called_once()

    def test_discover_backward_compatible_without_stub_dir(self) -> None:
        """When ``stub_dir=None`` (the default), no stub generation happens
        at all. This ensures backward compatibility with existing callers
        that do not use auto-freshness."""
        from conscribe.discover import discover

        mock_generate = MagicMock()

        with (
            _patch_imports(),
            patch(_PATCH_COMPUTE) as mock_compute,
            patch(_PATCH_LOAD) as mock_load,
            patch(_PATCH_SAVE) as mock_save,
            patch(_PATCH_GENERATE, mock_generate),
            patch(_PATCH_BUILD) as mock_build,
            patch(_PATCH_GET_REGISTRARS, return_value=[_freshness_reg]),
        ):
            # Call with defaults: auto_update_stubs=True, stub_dir=None
            discover("fake_module")

        # None of the fingerprint/codegen functions should be called
        mock_compute.assert_not_called()
        mock_load.assert_not_called()
        mock_save.assert_not_called()
        mock_generate.assert_not_called()
        mock_build.assert_not_called()
