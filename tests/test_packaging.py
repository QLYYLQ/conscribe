"""Packaging invariants that would otherwise surface only in a published release.

* ``pyproject.toml`` and ``conscribe.__version__`` carry the same version.
* Every relative link in the shipped docs (``conscribe/llms.txt`` and
  ``conscribe/docs/*.md``) resolves to a file inside the ``conscribe`` package.
  Only the package is installed, so a link that leaves it, or names a file
  that does not exist, is dead in site-packages even when it works on GitHub.

Standard library only (plus ``tomli`` on Python 3.10, which pytest itself requires there).
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path
from urllib.parse import unquote

import pytest

import conscribe

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest depends on tomli below Python 3.11
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(conscribe.__file__).resolve().parent
LLMS_TXT = PACKAGE_DIR / "llms.txt"
SHIPPED_DOCS = [LLMS_TXT, *sorted((PACKAGE_DIR / "docs").glob("*.md"))]


# ===================================================================
# Version
# ===================================================================

def test_pyproject_version_matches_dunder_version() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        version = tomllib.load(f)["project"]["version"]
    assert version == conscribe.__version__, (
        f"pyproject.toml has version {version!r} but conscribe.__version__ is "
        f"{conscribe.__version__!r}; change both together"
    )


# ===================================================================
# Doc links
# ===================================================================

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_CODE_SPAN = re.compile(r"(`+).+?\1")
_INLINE_DEST = re.compile(r"\]\(\s*(?:<([^>\n]*)>|([^\s)]+))")
_REFERENCE_DEST = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(?:<([^>\n]*)>|(\S+))")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _relative_links(path: Path) -> list[tuple[int, str]]:
    """Return ``(line number, destination)`` for every relative link in a Markdown file.

    Covers inline links and images (``[text](dest)``) and reference definitions
    (``[label]: dest``). Fenced code blocks and inline code spans are skipped, as are
    destinations with a URL scheme (``https:``, ``mailto:``), ``//host`` links and
    same-page ``#anchor`` links.
    """
    links: list[tuple[int, str]] = []
    fence: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = _FENCE.match(line)
        if fence is not None:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            continue
        text = _CODE_SPAN.sub("", line)
        for match in (*_INLINE_DEST.finditer(text), *_REFERENCE_DEST.finditer(text)):
            dest = match.group(1) if match.group(1) is not None else match.group(2)
            if dest and not dest.startswith(("#", "//")) and not _URL_SCHEME.match(dest):
                links.append((lineno, dest))
    return links


def _is_package_file(target: Path) -> bool:
    """True if *target* is a file inside the package, spelled with its exact letter case.

    macOS file systems ignore case; Linux site-packages and GitHub do not.
    """
    current = PACKAGE_DIR
    for part in target.relative_to(PACKAGE_DIR).parts:
        if not current.is_dir() or part not in os.listdir(current):
            return False
        current = current / part
    return current.is_file()


def _link_problem(source: Path, dest: str) -> str | None:
    """Say why *dest*, linked from *source*, will not resolve once installed (None if it will)."""
    path = unquote(dest.split("#", 1)[0].split("?", 1)[0])
    if path.startswith("/"):
        return "absolute path; link relative to the linking file instead"
    target = (source.parent / path).resolve()
    if not target.is_relative_to(PACKAGE_DIR):
        return "outside the conscribe package, so it is not installed"
    if not _is_package_file(target):
        return "no such file in the conscribe package (letter case counts)"
    return None


def test_link_scanner_finds_only_relative_links(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(textwrap.dedent("""\
        [inline](a.md), ![image](img/b.png "title") and [angle](<c d.md>)
        [web](https://example.com) [mail](mailto:x@example.com) [anchor](#top) [net](//host/x)
        Code `[not](a link)` is not a link.
        ```python
        registry[key](arguments)
        ```
        [ref]: ref/e.md
        """), encoding="utf-8")
    assert _relative_links(doc) == [
        (1, "a.md"), (1, "img/b.png"), (1, "c d.md"), (7, "ref/e.md"),
    ]


@pytest.mark.parametrize(("dest", "problem"), [
    ("__init__.py", None),
    ("docs/../__init__.py#anchor", None),
    ("__INIT__.py", "no such file"),
    ("missing.md", "no such file"),
    ("docs", "no such file"),
    ("../pyproject.toml", "outside the conscribe package"),
    ("/conscribe/__init__.py", "absolute path"),
])
def test_link_problem(dest: str, problem: str | None) -> None:
    found = _link_problem(LLMS_TXT, dest)
    if problem is None:
        assert found is None
    else:
        assert found is not None and found.startswith(problem)


@pytest.mark.parametrize("source", SHIPPED_DOCS, ids=lambda p: p.relative_to(PACKAGE_DIR).as_posix())
def test_relative_links_resolve_inside_package(source: Path) -> None:
    shown = source.relative_to(PACKAGE_DIR.parent).as_posix()
    broken = [
        f"{shown}:{lineno}: {dest} -> {problem}"
        for lineno, dest in _relative_links(source)
        if (problem := _link_problem(source, dest)) is not None
    ]
    assert not broken, "links that will not resolve once installed:\n" + "\n".join(broken)
