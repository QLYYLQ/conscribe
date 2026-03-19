"""Module discovery: recursively import packages to trigger registration.

Python metaclass registration depends on modules being imported.
``discover()`` ensures all implementation modules are imported,
triggering class definitions and filling registries.

Optionally supports auto-freshness: after imports, compares registry
fingerprints against cached values and regenerates config stubs if stale.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import List, Union

from conscribe.config.builder import build_layer_config
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.fingerprint import (
    compute_registry_fingerprint,
    load_cached_fingerprint,
    save_fingerprint,
)

logger = logging.getLogger(__name__)


def _get_known_registrars() -> list[type]:
    """Return all known registrars for auto-freshness checks.

    Override or patch this in tests. In production, this would
    scan a config file or global registry.
    """
    return []


def discover(
    *package_paths: str,
    auto_update_stubs: bool = True,
    stub_dir: Union[Path, None] = None,
    fingerprint_path: Union[Path, None] = None,
) -> List[str]:
    """Recursively import all modules under the given packages.

    This triggers import-time registration for any classes using
    AutoRegistrar metaclasses or __init_subclass__ hooks.

    After imports, optionally checks registry fingerprints and
    regenerates config stubs if they are stale.

    Args:
        *package_paths: Dotted package paths to discover
            (e.g. ``"my_app.agents"``, ``"my_app.llm_providers"``).
        auto_update_stubs: If True and ``stub_dir`` is provided,
            check fingerprints and regenerate stale stubs.
        stub_dir: Directory for generated config stub files.
            If None, auto-freshness is skipped (backward compatible).
        fingerprint_path: Path to the fingerprint cache JSON file.
            Defaults to ``stub_dir / ".registry_fingerprint"`` if
            ``stub_dir`` is provided.

    Returns:
        List of all successfully imported module names.

    Raises:
        ModuleNotFoundError: If a top-level package does not exist (fail-fast).

    Notes:
        - Submodule import failures are logged as warnings and skipped.
        - The top-level package itself is always imported first.
    """
    imported: List[str] = []

    for pkg_path in package_paths:
        # Import the top-level package (fail-fast if not found)
        package = importlib.import_module(pkg_path)
        imported.append(pkg_path)

        # Verify it's actually a package (has __path__)
        pkg_file_path = getattr(package, "__path__", None)
        if pkg_file_path is None:
            # It's a module, not a package — nothing to walk
            continue

        # Deduplicate error logging between walk_packages and our import loop.
        _already_failed: set[str] = set()

        def _handle_error(name: str) -> None:
            if name in _already_failed:
                return
            logger.warning(
                "Failed to import module '%s' during discover(): skipping",
                name,
            )

        for module_info in pkgutil.walk_packages(
            path=pkg_file_path,
            prefix=package.__name__ + ".",
            onerror=_handle_error,
        ):
            try:
                importlib.import_module(module_info.name)
                imported.append(module_info.name)
            except (ImportError, SyntaxError):
                _already_failed.add(module_info.name)
                logger.warning(
                    "Failed to import module '%s' during discover(): skipping",
                    module_info.name,
                    exc_info=True,
                )

    # -- Auto-freshness: check fingerprints and regenerate stubs if stale --
    if auto_update_stubs and stub_dir is not None:
        _auto_update_stubs(stub_dir, fingerprint_path)

    return imported


def _auto_update_stubs(
    stub_dir: Path,
    fingerprint_path: Union[Path, None],
) -> None:
    """Check registry fingerprints and regenerate stubs if stale."""
    if fingerprint_path is None:
        fingerprint_path = stub_dir / ".registry_fingerprint"

    registrars = _get_known_registrars()

    for registrar in registrars:
        layer_name = registrar._registry.name
        current_fp = compute_registry_fingerprint(registrar)
        cached_fp = load_cached_fingerprint(fingerprint_path, layer_name)

        if current_fp == cached_fp:
            continue

        # Fingerprint is stale or missing — regenerate
        logger.info("Regenerating config stubs for layer '%s'", layer_name)
        result = build_layer_config(registrar)
        source = generate_layer_config_source(result)

        stub_dir.mkdir(parents=True, exist_ok=True)
        stub_file = stub_dir / f"{layer_name}_config.py"
        stub_file.write_text(source, encoding="utf-8")

        save_fingerprint(fingerprint_path, layer_name, current_fp)
