"""Test helpers for code that registers classes with conscribe.

``isolated_registries()`` scopes registrations to a ``with`` block: every
registry and the global registry index return to their state at entry,
except for registrations made by modules imported for the first time inside
the block (an import cannot be undone).  The same object captures
module-level fakes for re-application per test (``activate()``).  See
:mod:`conscribe.registration.isolation` for the exact rules.

pytest users can opt in to the ``conscribe_isolated`` fixture by adding
``pytest_plugins = ["conscribe.testing.pytest_plugin"]`` to their root
``conftest.py``.  This package itself does not import pytest.
"""
from __future__ import annotations

from conscribe.registration.isolation import RegistryIsolation, isolated_registries

__all__ = ["RegistryIsolation", "isolated_registries"]
