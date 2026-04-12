"""Tests for conscribe.stubs.writer."""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import Protocol, runtime_checkable

from conscribe import create_registrar
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister
from conscribe.stubs.writer import write_layer_stubs


@runtime_checkable
class EnvProtocol(Protocol):
    def setup(self) -> None: ...


class Terminal:
    def setup(self) -> None: ...


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for name in list(_REGISTRY_INDEX.keys()):
        if name.startswith("test_"):
            _deregister(name)


@pytest.fixture
def env_registrar():
    return create_registrar(
        "test_env_writer",
        EnvProtocol,
        discriminator_field="name",
    )


class TestWriteLayerStubs:
    def test_no_wired_classes(self, env_registrar):
        """No classes with injected wiring → empty result."""

        class Base(metaclass=env_registrar.Meta):
            __abstract__ = True
            def setup(self) -> None: ...

        class PlainImpl(Base):
            def setup(self) -> None: ...

        result = write_layer_stubs(env_registrar)
        assert result == []

    def test_writes_pyi_to_output_dir(self, env_registrar, tmp_path):
        """Classes with injected wiring → .pyi written to output_dir."""

        class Base(metaclass=env_registrar.Meta):
            __abstract__ = True
            def setup(self) -> None: ...

        class WiredImpl(Base):
            __wiring__ = {"dep": ["some_value"]}

            def setup(self) -> None: ...

        written = write_layer_stubs(env_registrar, output_dir=tmp_path)
        assert len(written) == 1
        assert written[0].suffix == ".pyi"
        assert written[0].exists()

        content = written[0].read_text()
        assert "dep: str" in content
        assert "class WiredImpl" in content

    def test_output_dir_mirrors_module(self, env_registrar, tmp_path):
        """Output path mirrors module structure."""

        class Base(metaclass=env_registrar.Meta):
            __abstract__ = True
            def setup(self) -> None: ...

        class WiredImpl2(Base):
            __wiring__ = {"dep": ["val"]}

            def setup(self) -> None: ...

        written = write_layer_stubs(env_registrar, output_dir=tmp_path)
        assert len(written) == 1
        # Path should be under tmp_path, ending with .pyi
        assert str(written[0]).startswith(str(tmp_path))
