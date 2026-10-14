"""Opt-in pytest plugin: the ``conscribe_isolated`` fixture.

Enable it in the **root** ``conftest.py`` of a test suite::

    pytest_plugins = ["conscribe.testing.pytest_plugin"]

(pytest only accepts ``pytest_plugins`` in the root conftest.)  Prefer this
over ``-p conscribe.testing.pytest_plugin``: ``-p`` imports the plugin, and
with it the conscribe package, before pytest-cov starts measuring, so
coverage of conscribe (or of code imported with it) comes out incomplete.
There is deliberately no ``pytest11`` entry point: the fixture is only
available where a suite opts in.

Each test that requests ``conscribe_isolated`` runs inside
``conscribe.testing.isolated_registries()``; the fixture value is the
:class:`~conscribe.testing.RegistryIsolation` object, whose ``kept`` and
``rolled_back`` lists are filled after the test.  Set the ini option
``conscribe_warn_kept = true`` to get an
:class:`~conscribe.exceptions.IsolationWarning` naming the modules whose
import-time registrations a test kept.  To isolate every test, wrap the
fixture in an autouse fixture of your own::

    @pytest.fixture(autouse=True)
    def _isolated(conscribe_isolated):
        yield conscribe_isolated
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conscribe.testing import RegistryIsolation

_WARN_KEPT_INI = "conscribe_warn_kept"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``conscribe_warn_kept`` ini option."""
    parser.addini(
        _WARN_KEPT_INI,
        type="bool",
        default=False,
        help=(
            "conscribe_isolated: warn when a test kept registrations made by "
            "modules it imported for the first time"
        ),
    )


@pytest.fixture
def conscribe_isolated(request: pytest.FixtureRequest) -> Iterator[RegistryIsolation]:
    """Run the test inside ``conscribe.testing.isolated_registries()``.

    Registrations the test makes are rolled back after it; registrations made
    by modules it imports for the first time are kept.  Yields the
    :class:`~conscribe.testing.RegistryIsolation` object.
    """
    from conscribe.testing import isolated_registries

    warn_kept = bool(request.config.getini(_WARN_KEPT_INI))
    with isolated_registries(warn_kept=warn_kept) as isolation:
        yield isolation
