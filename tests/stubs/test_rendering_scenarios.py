"""End-to-end stub rendering scenarios on real (temporary) packages.

Each scenario writes a small package, registers its classes, writes the
stubs with :func:`conscribe.stubs.write_layer_stubs` and checks the text and
its validity (:func:`assert_valid_stub`).  When ``pyright`` is on ``PATH``
the generated stubs are also type-checked against a consumer module.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap

import pytest

from conscribe.stubs import write_layer_stubs

from ._helpers import assert_valid_stub, class_body, temp_package

REG = """
from typing import Protocol, runtime_checkable
from conscribe import create_registrar

@runtime_checkable
class Strategy(Protocol):
    def pick(self) -> int: ...

@runtime_checkable
class Tool(Protocol):
    def use(self) -> None: ...

Strat = create_registrar("{pkg}_strat", Strategy, discriminator_field="strategy_id")
Tools = create_registrar("{pkg}_tool", Tool, discriminator_field="tool_id")

class StratBase(metaclass=Strat.Meta):
    __abstract__ = True
    def pick(self) -> int: return 0

class Fast(StratBase):
    __registry_key__ = "fast"

class ToolBase(metaclass=Tools.Meta):
    __abstract__ = True
    def use(self) -> None: ...
"""


def _package(pkg: str, files: dict[str, str]) -> dict[str, str]:
    """``reg.py`` plus *files*, with ``{pkg}`` replaced by the package name."""
    return {name: src.replace("{pkg}", pkg) for name, src in {"reg.py": REG, **files}.items()}


class TestScenarios:
    def test_bridge_base_becomes_a_placeholder(self, tmp_path):
        """A bridge's ``__module__`` is a conscribe module it cannot be imported from."""
        files = {
            "external.py": """
                class ExternalTool:
                    def use(self) -> None: ...
            """,
            "tools.py": """
                from {pkg}.external import ExternalTool
                from {pkg}.reg import Tools

                ExternalBridge = Tools.bridge(ExternalTool)

                class Wrapped(ExternalBridge):
                    __registry_key__ = "wrapped"
                    __wiring__ = {"strategy": "{pkg}_strat"}
            """,
        }
        with temp_package(tmp_path, "scen_bridge", _package("scen_bridge", files)):
            import scen_bridge.tools  # noqa: F401
            from scen_bridge.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            assert_valid_stub(source)
        assert "conscribe.registration" not in source
        assert "class ExternalBridge(Any): ..." in source
        assert "class Wrapped(ExternalBridge):" in source

    def test_pydantic_model_fields_and_injected_attrs_get_defaults(self, tmp_path):
        """pydantic's dataclass_transform reads the stub's class body as fields:
        an injected attribute or a defaulted field without ``= ...`` would be
        a required constructor argument for type checkers."""
        # A pydantic model registered with ``register()`` (Path C).
        files = {
            "models.py": """
                from pydantic import BaseModel, Field
                from {pkg}.reg import Tools

                @Tools.register("search")
                class SearchTool(BaseModel):
                    __wiring__ = {"strategy": "{pkg}_strat"}

                    top_k: int = Field(default=5, ge=1)
                    query: str

                    def use(self) -> None: ...
            """,
        }
        pkg = "scen_model"
        with temp_package(tmp_path, pkg, _package(pkg, files)):
            import scen_model.models  # noqa: F401
            from scen_model.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            assert_valid_stub(source)
        body = class_body(source, "SearchTool")
        assert "strategy: Strategy = ...  # wired from: scen_model_strat" in body
        assert "top_k: int = ..." in body
        assert "    query: str\n" in body + "\n"
        assert "class SearchTool(BaseModel):" in body
        assert "from pydantic import BaseModel" in source

    def test_type_checking_only_module_is_never_imported(self, tmp_path):
        files = {
            "heavy/__init__.py": "",
            "heavy/api.py": "class Client: ...\n",
            "heavy/extra.py": "class Session: ...\n",
            "tools.py": """
                from __future__ import annotations

                from typing import TYPE_CHECKING
                from {pkg}.reg import ToolBase

                if TYPE_CHECKING:
                    from {pkg}.heavy.api import Client
                    import {pkg}.heavy.extra as extra

                class Remote(ToolBase):
                    __registry_key__ = "remote"
                    __wiring__ = {"strategy": "{pkg}_strat"}

                    def connect(self, client: Client) -> extra.Session: ...
            """,
        }
        pkg = "scen_tc"
        with temp_package(tmp_path, pkg, _package(pkg, files)):
            import scen_tc.tools  # noqa: F401
            from scen_tc.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            # Read from the source, never imported (the modules exist, unloaded).
            assert "scen_tc.heavy" not in sys.modules
            assert_valid_stub(
                source, unverified=frozenset({"scen_tc.heavy.api", "scen_tc.heavy.extra"})
            )
        assert "from scen_tc.heavy.api import Client" in source
        assert "import scen_tc.heavy.extra as extra" in source
        assert "def connect(self, client: Client) -> extra.Session: ..." in source

    def test_inherited_dataclass_field_resolves_in_its_own_module(self, tmp_path):
        """D3 on the stub side: 1.3.0 resolved ``knob: Optional[Knob]`` in the
        subclass module and bound ``Knob = Any``."""
        files = {
            "base.py": """
                from __future__ import annotations

                from dataclasses import dataclass
                from typing import Optional

                class Knob:
                    level: int = 0

                @dataclass
                class BaseConfig:
                    knob: Optional[Knob] = None
            """,
            "tools.py": """
                from __future__ import annotations

                from dataclasses import dataclass
                from {pkg}.base import BaseConfig
                from {pkg}.reg import ToolBase

                @dataclass
                class Configured(BaseConfig, ToolBase):
                    __registry_key__ = "configured"
                    __wiring__ = {"strategy": "{pkg}_strat"}

                    extra: int = 0
            """,
        }
        pkg = "scen_d3"
        with temp_package(tmp_path, pkg, _package(pkg, files)):
            import scen_d3.tools  # noqa: F401
            from scen_d3.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            assert_valid_stub(source)
        assert "from scen_d3.base import BaseConfig, Knob" in source
        assert "def __init__(self, knob: Knob | None = None, extra: int = 0) -> None: ..." in source
        assert "Knob = Any" not in source

    def test_import_name_clash_gets_a_suffix(self, tmp_path):
        """The module defines ``Config``; another module's ``Config`` is imported aliased."""
        files = {
            "other.py": """
                class Config:
                    pass
            """,
            "tools.py": """
                from __future__ import annotations

                from {pkg} import other
                from {pkg}.reg import ToolBase

                class Config:
                    pass

                class Clashing(ToolBase):
                    __registry_key__ = "clashing"
                    __wiring__ = {"strategy": "{pkg}_strat"}

                    def setup(self, mine: Config, theirs: other.Config) -> None: ...
            """,
        }
        pkg = "scen_clash"
        with temp_package(tmp_path, pkg, _package(pkg, files)):
            import scen_clash.tools  # noqa: F401
            from scen_clash.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            assert_valid_stub(source)
        assert "class Config(Any): ..." in source
        assert "from scen_clash.other import Config as Config_2" in source
        assert "def setup(self, mine: Config, theirs: Config_2) -> None: ..." in source

    def test_typeddict_with_mixed_requiredness(self, tmp_path):
        files = {
            "tools.py": """
                from typing import TypedDict
                from {pkg}.reg import ToolBase

                class Required_(TypedDict):
                    host: str

                class Options(Required_, total=False):
                    port: int

                class Served(ToolBase):
                    __registry_key__ = "served"
                    __wiring__ = {"strategy": "{pkg}_strat"}

                    def serve(self, options: Options) -> None: ...
            """,
        }
        pkg = "scen_td"
        with temp_package(tmp_path, pkg, _package(pkg, files)):
            import scen_td.tools  # noqa: F401
            from scen_td.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            assert_valid_stub(source)
        # Flattened structurally: same keys, same requiredness.
        assert (
            "class Options(TypedDict, total=False):\n"
            "    host: Required[str]\n"
            "    port: int\n"
        ) in source


    @pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 syntax")
    def test_pep695_type_parameters_and_aliases(self, tmp_path):
        """PEP 695 type parameters belong to no module: the stub defines them."""
        files = {
            "tools.py": """
                from {pkg}.reg import ToolBase

                class Box[T]:
                    def __init__(self, item: T) -> None:
                        self.item = item

                type Pair = tuple[int, int]
                type Wrapped[U] = list[U]

                class Generic695(ToolBase):
                    __registry_key__ = "generic695"
                    __wiring__ = {"strategy": "{pkg}_strat"}

                    def combine(self, box: Box[str], pair: Pair, wrapped: Wrapped[int]) -> None: ...

                    def first[V](self, items: list[V]) -> V: ...
            """,
        }
        pkg = "scen_pep695"
        with temp_package(tmp_path, pkg, _package(pkg, files)):
            import scen_pep695.tools  # noqa: F401
            from scen_pep695.reg import Tools

            (path,) = write_layer_stubs(Tools)
            source = path.read_text(encoding="utf-8")
            assert_valid_stub(source)
        assert "T = TypeVar('T', infer_variance=True)" in source
        assert "class Box(Any, Generic[T]): ..." in source
        assert "Pair: TypeAlias = tuple[int, int]" in source
        assert "Wrapped: TypeAlias = list[U]" in source
        assert "def combine(self, box: Box[str], pair: Pair, wrapped: Wrapped[int]) -> None: ..." in source
        assert "def first(self, items: list[V]) -> V: ..." in source


# ── pyright (optional) ───────────────────────────────────────────


PYRIGHT = shutil.which("pyright")

_CONSUMER = """
from {pkg}.owner import Owner, Sibling
from {pkg}.reg import Strategy
from {pkg}.strat import Fast


def ok(o: Owner) -> None:
    s: Strategy = o.helper
    n: int = o.helper.pick()
    o.helper = Fast()
    Owner(depth=2, sib=Sibling(y=1))
    print(s, n)


def bad(o: Owner) -> None:
    o.helper.bogus()
    Owner(depth="x")
"""


@pytest.mark.skipif(PYRIGHT is None, reason="pyright is not installed")
def test_pyright_accepts_the_stub(tmp_path):
    from .test_af_repro import _files

    pkg = "afrepro_pyright"
    files = _files(pkg, '{"strategy": "strat", "helper": "strat"}')
    with temp_package(tmp_path, pkg, files) as package:
        import afrepro_pyright.owner  # noqa: F401
        import afrepro_pyright.strat  # noqa: F401
        from afrepro_pyright.reg import Own

        write_layer_stubs(Own)
        (tmp_path / "consumer.py").write_text(textwrap.dedent(_CONSUMER.replace("{pkg}", pkg)))
        (tmp_path / "pyrightconfig.json").write_text(
            json.dumps({"include": ["consumer.py", f"{pkg}/owner.pyi"]})
        )
        proc = subprocess.run(
            [PYRIGHT, "--pythonpath", sys.executable, "--outputjson", "-p", str(tmp_path)],
            cwd=tmp_path, capture_output=True, text=True, timeout=300,
        )
    report = json.loads(proc.stdout)
    errors = [d for d in report["generalDiagnostics"] if d["severity"] == "error"]
    in_stub = [d for d in errors if d["file"].endswith(".pyi")]
    assert in_stub == [], in_stub
    lines = sorted(d["range"]["start"]["line"] + 1 for d in errors)
    # Only the two intended errors in ``bad``.
    assert len(lines) == 2, errors
    assert package.exists()
