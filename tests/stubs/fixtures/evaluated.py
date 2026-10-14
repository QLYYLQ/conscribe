"""Fixture: evaluated annotations (deliberately NO ``from __future__ import annotations``).

Every annotation here is a live typing object, which the 1.3.0 stub writer
rendered with ``inspect.formatannotation``: ``FieldInfo(...)`` reprs,
``NoneType``, fully qualified names (``tests.stubs.fixtures.evaluated.X``),
``Literal[<Color.RED: 'r'>]`` (a SyntaxError) and bare ``Optional`` /
``List`` for class attributes.
"""
import collections.abc
import enum
import typing
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field

T = TypeVar("T")
UserId = NewType("UserId", int)
Pair = tuple[int, int]


class Color(enum.Enum):
    RED = "r"
    BLUE = "b"


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2

    def describe(self) -> str:
        return self.name.lower()


@runtime_checkable
class Picker(Protocol):
    name: str

    def pick(self, items: list[int]) -> int: ...


class Options(TypedDict, total=False):
    retries: int
    label: str


class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item


class Outer:
    class Inner:
        pass


class Settings(BaseModel):
    retries: int = Field(default=3, ge=0)
    name: str


class Evaluated:
    """Own class attributes and a constructor with evaluated annotations."""

    a: Optional[int] = None
    b: List[str]
    c: Dict[str, int]
    d: Annotated[int, Field(ge=1, description="d")] = 1
    e: Literal["x", "y"] = "x"
    f: ClassVar[int] = 3
    g: "Optional[Color]" = None

    def __init__(
        self,
        *,
        depth: Annotated[int, Field(ge=1, description="depth")] = 1,
        color: Literal[Color.RED, Color.BLUE] = Color.RED,
        level: Optional[Level] = None,
        picker: Union[Picker, None] = None,
        options: Optional[Options] = None,
        box: Optional[Box[int]] = None,
        user: Optional[UserId] = None,
        inner: Optional[Outer.Inner] = None,
        callback: Optional[Callable[[int, str], bool]] = None,
        anything: Callable[..., Any] = print,
        pair: Pair = (0, 0),
        rest: Tuple[int, ...] = (),
        empty: tuple[()] = (),
        typed: typing.List[int] = [],
        iterable: Optional[collections.abc.Iterable[str]] = None,
        settings: Optional[Settings] = None,
        kind: type[Picker] = Picker,
    ) -> None:
        self.depth = depth

    def first(self, items: list[T]) -> T:
        return items[0]
