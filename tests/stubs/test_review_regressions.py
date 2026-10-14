"""Regression tests for the 2.0 stub review findings (ST2-ST6, ST9, class-scope shadowing).

One package covers every shape the reviewers found broken; its stubs are
written next to the sources, checked by :func:`assert_valid_stub` and the
exact rendered text, and -- when ``pyright`` is on ``PATH`` -- type-checked
together with a consumer module that uses every rendered member: 0 errors.
"""
from __future__ import annotations

import textwrap

import pytest

from conscribe.exceptions import ConfigRenderError
from conscribe.stubs import write_layer_stubs

from ._helpers import PYRIGHT, assert_valid_stub, class_body, run_pyright, temp_package

PKG = "revfix"

REG = """
    from typing import Protocol, runtime_checkable
    from conscribe import create_registrar

    @runtime_checkable
    class Strategy(Protocol):
        def pick(self) -> int: ...

    @runtime_checkable
    class Owner(Protocol):
        def run(self) -> None: ...

    Strats = create_registrar("{pkg}_strat", Strategy, discriminator_field="kind")
    Owners = create_registrar("{pkg}_owner", Owner, discriminator_field="kind")

    class StratBase(metaclass=Strats.Meta):
        __abstract__ = True
        def pick(self) -> int: return 0

    class Fast(StratBase):
        __registry_key__ = "fast"

    class OwnerBase(metaclass=Owners.Meta):
        __abstract__ = True
        def run(self) -> None: ...
"""

# ST2: context-manager methods and other wrappers.
CM = """
    from __future__ import annotations

    import contextlib
    import functools
    from collections.abc import AsyncIterator, Iterator
    from typing import TYPE_CHECKING, Any

    from {pkg}.reg import OwnerBase

    if TYPE_CHECKING:
        from decimal import Decimal


    def logged(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper


    class Sessions(OwnerBase):
        __registry_key__ = "sessions"
        __wiring__ = {"strategy": "{pkg}_strat"}

        @contextlib.contextmanager
        def session(self, name: str = "x") -> Iterator[int]:
            yield 1

        @contextlib.asynccontextmanager
        async def asession(self) -> AsyncIterator[str]:
            yield ""

        @contextlib.contextmanager
        def money(self) -> Iterator[Decimal]:
            yield 0  # type: ignore[misc]

        @contextlib.contextmanager
        def untyped(self):
            yield 1

        @classmethod
        @contextlib.contextmanager
        def scoped(cls) -> Iterator[str]:
            yield ""

        @logged
        def fetch(self, url: str) -> bytes:
            return b""
"""

# ST3 / ST9: members the stub dropped; a builtin shadowed by the class body.
MEMBERS = """
    from __future__ import annotations

    import enum
    import functools
    import logging
    from typing import Final

    from {pkg}.reg import OwnerBase


    class Mode(enum.Enum):
        A = "a"
        B = "b"


    class Members(OwnerBase):
        __registry_key__ = "members"
        __wiring__ = {"strategy": "{pkg}_strat"}

        MAX_STEPS = 30
        name = "members"
        ratio = 0.5
        tags = ("a", "b")
        lookup = {"a": 1}
        DEFAULT_MODE = Mode.A
        log = logging.getLogger("revfix")
        Pair = tuple[int, int]
        LIMIT: Final = 3
        WHEN: Final = Mode.B
        type = "worker"

        class Config:
            retries = 2

        @functools.cache
        def compute(self) -> int:
            return 1

        def _base(self, a: int, b: str) -> str:
            return b

        partial = functools.partialmethod(_base, 1)

        @functools.singledispatchmethod
        def dispatch(self, arg: object) -> str:
            return ""

        def make(self, factory: type[int] = int) -> dict[str, type[int]]:
            return {}
"""

# ST4: dotted imports under TYPE_CHECKING.
DOTTED = """
    from __future__ import annotations

    from typing import TYPE_CHECKING

    from {pkg}.reg import OwnerBase

    if TYPE_CHECKING:
        import email.mime.text
        import email.mime.multipart
        import xml.dom.minidom as md


    class Dotted(OwnerBase):
        __registry_key__ = "dotted"
        __wiring__ = {"strategy": "{pkg}_strat"}

        def build(self) -> email.mime.text.MIMEText: ...

        def multi(self) -> email.mime.multipart.MIMEMultipart: ...

        def doc(self) -> md.Document: ...
"""

# ST5: a module-level alias of a parametrized builtin, on the source-text path.
ALIASES = """
    from __future__ import annotations

    from typing import TYPE_CHECKING

    from {pkg}.reg import OwnerBase

    if TYPE_CHECKING:
        from decimal import Decimal

    Pair = tuple[int, int]


    class Prices(OwnerBase):
        __registry_key__ = "prices"
        __wiring__ = {"strategy": "{pkg}_strat"}

        def prices(self, p: Pair) -> dict[Pair, Decimal]: ...
"""

# ST6: registered classes nested in a class body.
NESTED = """
    from {pkg}.reg import OwnerBase


    class Outer:
        class Inner(OwnerBase):
            __registry_key__ = "inner"
            __wiring__ = {"strategy": "{pkg}_strat"}

            class Deeper(OwnerBase):
                __registry_key__ = "deeper"
                __wiring__ = {"strategy": "{pkg}_strat"}

        def helper(self) -> int:
            return 1
"""

CONSUMER = """
    from __future__ import annotations

    from collections.abc import Mapping

    from {pkg}.aliases import Prices
    from {pkg}.cm import Sessions
    from {pkg}.dotted import Dotted
    from {pkg}.members import Members, Mode
    from {pkg}.nested import Outer
    from {pkg}.reg import Strategy


    def sessions(s: Sessions) -> None:
        with s.session() as n:
            n_: int = n
        with Sessions.scoped() as t:
            t_: str = t
        with s.untyped() as u:
            print(n_, t_, u)
        s.fetch("u", retries=3)
        strategy: Strategy = s.strategy
        print(strategy)


    async def asessions(s: Sessions) -> None:
        async with s.asession() as a:
            a_: str = a
            print(a_)


    def members(m: Members) -> None:
        a: int = Members.MAX_STEPS
        b: str = m.name
        c: float = m.ratio
        d: tuple[str, str] = m.tags
        e: dict[str, int] = m.lookup
        f: Mode = m.DEFAULT_MODE
        g: int = Members.LIMIT
        h: Mode = Members.WHEN
        i: str = m.type
        j: int = m.compute()
        k: dict[str, type[int]] = m.make(int)
        p: Members.Pair = (1, 2)
        m.partial("x")
        m.dispatch(3)
        m.log.info("x")
        print(a, b, c, d, e, f, g, h, i, j, k, p, Members.Config.retries)


    def dotted(d: Dotted) -> None:
        print(d.build().get_payload(), d.multi(), d.doc().documentElement)


    def prices(p: Prices) -> None:
        q: Mapping[tuple[int, int], object] = p.prices((1, 2))
        print(q)


    def nested(i: Outer.Inner, d: Outer.Inner.Deeper) -> None:
        s1: Strategy = i.strategy
        s2: Strategy = d.strategy
        print(s1, s2, Outer().helper())
"""

FILES = {
    "reg.py": REG,
    "cm.py": CM,
    "members.py": MEMBERS,
    "dotted.py": DOTTED,
    "aliases.py": ALIASES,
    "nested.py": NESTED,
}


@pytest.fixture(scope="module")
def stubs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("revfix")
    files = {name: source.replace("{pkg}", PKG) for name, source in FILES.items()}
    with temp_package(root, PKG, files) as package:
        import importlib

        for module in ("cm", "members", "dotted", "aliases", "nested"):
            importlib.import_module(f"{PKG}.{module}")
        reg = importlib.import_module(f"{PKG}.reg")
        written = write_layer_stubs(reg.Owners)
        assert sorted(p.name for p in written) == [
            "aliases.pyi", "cm.pyi", "dotted.pyi", "members.pyi", "nested.pyi",
        ]
        sources = {p.stem: p.read_text(encoding="utf-8") for p in written}
        for source in sources.values():
            assert_valid_stub(source)
        if PYRIGHT is not None:
            (root / "consumer.py").write_text(
                textwrap.dedent(CONSUMER.replace("{pkg}", PKG)), encoding="utf-8"
            )
            errors = run_pyright(root, ["consumer.py", *(f"{PKG}/{p.name}" for p in written)])
            sources["__pyright_errors__"] = repr(errors)
        assert package.exists()
    return sources


def _lines(source: str, cls: str) -> list[str]:
    return [line.strip() for line in class_body(source, cls).splitlines()]


class TestContextManagers:
    """ST2: ``@contextmanager`` methods returned the generator's ``Iterator[T]``."""

    def test_rendered_as_context_managers(self, stubs):
        lines = _lines(stubs["cm"], "Sessions")
        assert "def session(self, name: str = 'x') -> AbstractContextManager[int]: ..." in lines
        assert "def asession(self) -> AbstractAsyncContextManager[str]: ..." in lines
        assert "def money(self) -> AbstractContextManager[Decimal]: ..." in lines
        assert "def untyped(self) -> AbstractContextManager[Any]: ..." in lines
        assert "def scoped(cls) -> AbstractContextManager[str]: ..." in lines
        assert "async def asession" not in stubs["cm"]
        assert "from contextlib import AbstractAsyncContextManager, AbstractContextManager" in stubs["cm"]

    def test_other_wrappers_accept_anything(self, stubs):
        assert "def fetch(self, *args: Any, **kwargs: Any) -> Any: ..." in _lines(stubs["cm"], "Sessions")


class TestDroppedMembers:
    """ST3 / ST9: constants, cached / partial / dispatch methods, nested classes, ``Final``."""

    @pytest.mark.parametrize(
        "line",
        [
            "MAX_STEPS: int",
            "name: str",
            "ratio: float",
            "tags: tuple[str, str]",
            "lookup: dict[str, int]",
            "DEFAULT_MODE: Mode",
            "log: Logger",
            "Pair: TypeAlias = tuple[int, int]",
            "LIMIT: Final = 3",
            "WHEN: Final[Mode]",
            "type: str",
            "def compute(self) -> int: ...",
            "def _base(self, a: int, b: str) -> str: ...",
            "def partial(self, *args: Any, **kwargs: Any) -> Any: ...",
            "def dispatch(self, *args: Any, **kwargs: Any) -> Any: ...",
            "class Config(Any): ...",
        ],
    )
    def test_member_is_declared(self, stubs, line):
        assert line in _lines(stubs["members"], "Members")

    def test_builtin_shadowed_by_the_class_body_is_qualified(self, stubs):
        """``type = "worker"`` in the body: ``type[int]`` there must mean the builtin."""
        lines = _lines(stubs["members"], "Members")
        assert (
            "def make(self, factory: builtins.type[int] = ...) -> dict[str, builtins.type[int]]: ..."
            in lines
        )
        assert "import builtins" in stubs["members"]


class TestDottedTypeCheckingImports:
    """ST4: ``import email.mime.text`` rendered as ``import email``."""

    def test_statements_are_repeated(self, stubs):
        source = stubs["dotted"]
        assert "import email.mime.multipart\nimport email.mime.text\nimport xml.dom.minidom as md\n" in source
        lines = _lines(source, "Dotted")
        assert "def build(self) -> email.mime.text.MIMEText: ..." in lines
        assert "def multi(self) -> email.mime.multipart.MIMEMultipart: ..." in lines
        assert "def doc(self) -> md.Document: ..." in lines


class TestAliases:
    """ST5: on 3.10 ``tuple[int, int]`` is a ``type``: it became ``class Pair(Any)``."""

    def test_alias_is_inlined(self, stubs):
        source = stubs["aliases"]
        assert "def prices(self, p: tuple[int, int]) -> dict[tuple[int, int], Decimal]: ..." in source
        assert "class Pair" not in source


class TestNested:
    """ST6: nested registered classes were written as top-level classes."""

    def test_nested_under_the_enclosing_class(self, stubs):
        source = stubs["nested"]
        assert (
            "class Outer(Any):\n"
            "    class Inner(OwnerBase):\n"
            "        strategy: Strategy  # wired from: revfix_strat\n"
            "\n"
            "        class Deeper(OwnerBase):\n"
            "            strategy: Strategy  # wired from: revfix_strat\n"
        ) in source
        assert "\nclass Inner" not in source and "\nclass Deeper" not in source

    def test_function_local_registered_class_is_refused(self, tmp_path):
        files = {
            "reg.py": REG.replace("{pkg}", "revfix_local"),
            "local.py": """
                from revfix_local.reg import OwnerBase

                def make():
                    class Local(OwnerBase):
                        __registry_key__ = "local"
                        __wiring__ = {"strategy": "revfix_local_strat"}

                    return Local

                LocalCls = make()
            """,
        }
        with temp_package(tmp_path, "revfix_local", files) as package:
            import revfix_local.local  # noqa: F401
            from revfix_local.reg import Owners

            with pytest.raises(ConfigRenderError, match=r"make\.<locals>\.Local.*inside a function"):
                write_layer_stubs(Owners)
            assert not (package / "local.pyi").exists()


@pytest.mark.skipif(PYRIGHT is None, reason="pyright is not installed")
def test_pyright_accepts_stubs_and_consumer(stubs):
    assert stubs["__pyright_errors__"] == "[]"
