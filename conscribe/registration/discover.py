"""Module discovery: recursively import packages to trigger registration.

Python metaclass registration depends on modules being imported.
``discover()`` ensures all implementation modules are imported,
triggering class definitions and filling registries.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import List

logger = logging.getLogger(__name__)


def discover(*package_paths: str) -> List[str]:
    """Recursively import all modules under the given packages.

    This triggers import-time registration for any classes using
    AutoRegistrar metaclasses or __init_subclass__ hooks.

    Args:
        *package_paths: Dotted package paths to discover
            (e.g. ``"my_app.agents"``, ``"my_app.llm_providers"``).

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
        #
        # pkgutil.walk_packages() sequence for a bad sub-package:
        #   1. yield module_info          (our for-loop receives it)
        #   2. our import_module() fails  (we log with traceback)
        #   3. walk_packages tries the same import for recursion → also fails
        #   4. walk_packages calls onerror callback
        #
        # Without dedup, steps 2 and 4 both log the same failure.
        # Fix: track modules that failed in our try/except (step 2), and
        # suppress the onerror callback (step 4) for those.
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
            except Exception:
                _already_failed.add(module_info.name)
                logger.warning(
                    "Failed to import module '%s' during discover(): skipping",
                    module_info.name,
                    exc_info=True,
                )

    return imported
