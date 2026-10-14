"""Cross-registry wiring resolution.

Reads ``__wiring__`` declarations from classes, merges them along the MRO,
and resolves registry references to concrete key lists for config generation.

Four entry forms are supported::

    __wiring__ = {
        "loop": "agent_loop",                                    # Mode 1: auto-discovery (all keys)
        "llm_provider": ("llm", ["openai", "anthropic"]),        # Mode 2: explicit subset
        "obs": ("observation", ["terminal"], ["filesystem"]),     # Mode 2: required + optional
        "browser": ["chromium", "firefox"],                      # Mode 3: literal list
        "source": {                                              # dict form
            "registry": "network_source",
            "keys": ["archive", "live"],
            "default": "live",
        },
    }

The **dict form** spells out the same information with named fields and is the
only form that can carry the extra per-entry options:

- ``registry`` (str, required): target registry name.
- ``keys`` (list[str], optional): explicit subset; omitted = every key (Mode 1).
- ``optional_keys`` (list[str], optional, requires ``keys``): optional subset,
  like the third element of a Mode 2 tuple.
- ``default`` (str, optional): the key a single-instance receptor uses when the
  config omits it.  It must be registered in the target registry (spelled
  exactly as registered, i.e. fully qualified when the registry has a
  ``key_separator``) and be among the entry's allowed keys.
- ``declare_only`` (bool, optional, default ``False``): the entry is a
  declaration-only constraint — the framework does not inject this attribute.
  Cannot be combined with ``default``.

Any other field is an error.  Entry names must be valid Python identifiers
that are neither keywords nor dunder names (``__x__``), since each one names
a constructor parameter or an instance attribute.

A subclass entry replaces the inherited entry whole, and a ``None`` value
excludes an inherited key::

    __wiring__ = {"llm": None}  # remove parent's llm wiring

``__wiring__ = None`` on a subclass (the attribute itself, not an entry)
declares nothing: the class keeps every inherited entry, exactly as if it
did not set ``__wiring__`` at all.  Use ``{"name": None}`` entries to drop
inherited entries.

Strictness
----------

Every problem is reported, never skipped: :func:`resolve_wiring` collects all
problems of one class and :func:`validate_wiring` all problems of every class
registered in a set of registrars, and each raises a single
:class:`~conscribe.exceptions.WiringResolutionError` whose ``problems`` lists
them (each a :class:`~conscribe.exceptions.WiringProblem` with a stable
``kind``).  Resolution also checks each entry against its runtime slot (a
``default`` needs a single-instance receptor with a slot; a ``declare_only``
entry must not have one) and against the slot's own parameter default (which
must be one of the entry's allowed keys, and the same key as the entry's
``default`` if it declares one).  :func:`check_wiring_declaration` runs only
the grammar checks, without touching any registry; every registration path
(the registration metaclass, ``register()`` and the ``register(propagate=True)``
hook) runs it before a class is added to a registry, so a malformed
declaration fails at class creation and leaves nothing registered.

Injection placeholders
----------------------

A wired name that is not a constructor parameter is injected by the framework
after construction, by plain attribute assignment.  At class creation
:func:`inject_wired_descriptors` gives such a name a :class:`WiredField`
placeholder (unless the class body or an inherited binding already provides
a place for the value, or the class annotates the name itself), so reading it
before injection raises a clear ``AttributeError``.  A class that stops
wiring an inherited name (``{"env": None}`` or a ``declare_only`` entry)
gets a tombstone over the inherited placeholder.  :func:`wired_field` and
:func:`wired_fields` look the placeholders up without triggering them.  See
:class:`WiredField` for the exact class- and instance-access semantics.

Capability-relative keys
------------------------

When the target registry was created with a ``key_separator`` (e.g.
``create_registrar(..., key_separator=".")``), the explicit key lists of
Mode 2 (and of the dict form) may name a **trailing segment** instead of a
fully-qualified key::

    __wiring__ = {"action": ("action", ["click"], ["scroll"])}

``resolve_wiring`` expands ``"click"`` into every registered key whose
trailing segment is ``click`` (``browser.click``, ``desktop.click``, ...).
This is *not* a fourth grammar mode — it is a relaxation of how the key
strings inside Mode 2 are matched.  A key that already contains the
separator is passed through unchanged, which is the escape hatch for
pinning one specific provider.  A ``default`` key is never expanded: it names
one concrete choice, so it must be spelled exactly as registered.
"""
from __future__ import annotations

import difflib
import functools
import inspect
import keyword
import sys
import types
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from conscribe.exceptions import WiringProblem, WiringResolutionError

__all__ = [
    "ResolvedWiring",
    "WiredField",
    "WiringSpec",
    "check_wiring_declaration",
    "collect_wiring_from_mro",
    "expand_relative_keys",
    "inject_wired_descriptors",
    "parse_wiring",
    "resolve_wiring",
    "validate_wiring",
    "wired_field",
    "wired_fields",
]

_PY314 = sys.version_info >= (3, 14)

#: Fields accepted by a dict-form ``__wiring__`` entry, in documentation order.
_DICT_FIELDS: tuple[str, ...] = (
    "registry",
    "keys",
    "optional_keys",
    "default",
    "declare_only",
)

#: Cap on how many keys / registries / failures / suggestions a message lists.
_LIST_CAP = 30
_FAILURES_CAP = 5
_SUGGESTION_CAP = 5

#: Sentinel for "field not present" in dict-form entries.
_MISSING: Any = object()


@dataclass(frozen=True)
class WiringSpec:
    """Normalized representation of a single ``__wiring__`` entry.

    Produced by :func:`parse_wiring` without consulting any registry, so it
    is available at any time (e.g. for runtime negotiation).

    Attributes:
        param_name: The config field / ``__init__`` parameter name.
        registry_name: Target registry name (empty string for Mode 3).
        allowed_keys: Explicit key subset (``None`` = auto-discover all keys).
            For 3-element tuple mode (and dict form with ``optional_keys``),
            these are the *required* keys.  Keys are exactly as declared, so
            they may be capability-relative.
        optional_keys: Optional key subset for 3-element tuple mode / dict
            form ``optional_keys`` (``None`` for all other modes).
        default_key: The dict-form ``default`` key, exactly as declared
            (``None`` when the entry declares no default).
        declare_only: ``True`` for a dict-form entry with
            ``"declare_only": True`` — a declaration-only constraint for
            which nothing is injected.
    """

    param_name: str
    registry_name: str = ""
    allowed_keys: tuple[str, ...] | None = field(default=None)
    optional_keys: tuple[str, ...] | None = field(default=None)
    default_key: str | None = None
    declare_only: bool = False


@dataclass
class ResolvedWiring:
    """Result of resolving a single wiring entry to concrete keys.

    Attributes:
        param_name: The config field name.
        allowed_keys: Concrete list of allowed key strings.  When
            ``optional_keys`` is set, this contains required + optional
            combined (for ``Literal[...]`` generation).
        registry_name: Source registry name (``None`` for Mode 3 literal lists).
        optional_keys: Optional key subset, expanded (``None`` when the entry
            declares no optional keys).
        default_key: The validated default key (registered, allowed and fully
            qualified), or ``None`` when the entry declares no default.
        declare_only: Whether the entry is a declaration-only constraint.
    """

    param_name: str
    allowed_keys: list[str]
    registry_name: str | None = None
    optional_keys: list[str] | None = None
    default_key: str | None = None
    declare_only: bool = False


# ── Merging along the MRO ──────────────────────────────────────


class _Entry(NamedTuple):
    """One ``__wiring__`` entry, parsed, and the class that declared it.

    ``spec`` is ``None`` when the entry has a grammar problem (listed in
    ``problems`` as ``(kind, registry_name, detail)``) and for exclusions
    (``value is None``, ``excluded``).
    """

    param_name: Any
    value: Any
    declared_on: type
    spec: WiringSpec | None
    problems: tuple[tuple[str, str, str], ...]
    excluded: bool


class _OwnWiring(NamedTuple):
    """Cached parse of one class's own ``__wiring__`` mapping."""

    source: Any
    snapshot: dict[Any, Any]
    entries: dict[Any, _Entry]


#: Parsed own ``__wiring__`` per class.  Class creation merges the entries of
#: every base (twice: the grammar check and the placeholders), so each base's
#: entries are parsed once and reused; they are re-parsed whenever the
#: mapping (or a list / tuple / dict inside it) changed since.
_OWN_WIRING: weakref.WeakKeyDictionary[type, _OwnWiring] = weakref.WeakKeyDictionary()


class _Uncachable(Exception):
    """A ``__wiring__`` value the cache does not copy (it is re-parsed every time)."""


def _snapshot(value: Any, depth: int = 0) -> Any:
    """A copy of a wiring value, deep through ``list`` / ``tuple`` / ``dict``.

    Compared with ``==`` against the live mapping, so a later in-place change
    (``Base.__wiring__["llm"][1].append("x")``) is noticed.  Only ``str``,
    ``None`` and exact ``list`` / ``tuple`` / ``dict`` containers are copied;
    anything else (numbers, ``str`` subclasses such as Enum members,
    arbitrary objects -- all malformed or unusual) makes the mapping
    uncachable, and it is re-parsed on every use.
    """
    kind = type(value)
    if kind is str or value is None:
        return value
    if depth > 8:
        raise _Uncachable
    if kind is list:
        return [_snapshot(item, depth + 1) for item in value]
    if kind is tuple:
        return tuple(_snapshot(item, depth + 1) for item in value)
    if kind is dict:
        return {_snapshot(k, depth + 1): _snapshot(v, depth + 1) for k, v in value.items()}
    raise _Uncachable


def _parse_own(klass: type, wiring: Mapping[Any, Any]) -> dict[Any, _Entry]:
    """Parse *klass*'s own ``__wiring__`` entries (cached per class)."""
    cached = _OWN_WIRING.get(klass)
    if cached is not None and cached.source is wiring:
        try:
            unchanged = bool(cached.snapshot == wiring)
        except Exception:  # noqa: BLE001 - a value that cannot compare: re-parse
            unchanged = False
        if unchanged:
            return cached.entries
    entries: dict[Any, _Entry] = {}
    for param_name, value in wiring.items():
        if value is None:
            entries[param_name] = _Entry(param_name, None, klass, None, (), True)
            continue
        try:
            spec, problems = _check_entry(param_name, value)
        except Exception as exc:  # noqa: BLE001 - e.g. a value whose repr raises
            # Reported as a grammar problem; collect_wiring_from_mro never raises.
            spec, problems = None, [(
                "malformed", "", f"the entry cannot be read ({type(exc).__name__}: {exc})"
            )]
        entries[param_name] = _Entry(param_name, value, klass, spec, tuple(problems), False)
    try:
        snapshot = _snapshot(dict(wiring.items()))
    except _Uncachable:
        _OWN_WIRING.pop(klass, None)
    else:
        _OWN_WIRING[klass] = _OwnWiring(wiring, snapshot, entries)
    return entries


def _merge(cls: type) -> tuple[dict[Any, _Entry], list[tuple[type, Any]]]:
    """Merge ``__wiring__`` along the MRO (base first, so subclasses win).

    Returns:
        ``(entries, not_mappings)``: every merged entry in first-declaration
        order, exclusions included (``entry.excluded``), and every class in
        the MRO whose own ``__wiring__`` is neither ``None`` nor a mapping,
        together with the offending value.
    """
    merged: dict[Any, _Entry] = {}
    not_mappings: list[tuple[type, Any]] = []
    for klass in reversed(cls.__mro__):
        # __dict__ gives only the class's own __wiring__ (not the inherited one).
        wiring = klass.__dict__.get("__wiring__")
        if wiring is None:
            continue
        if not isinstance(wiring, Mapping):
            not_mappings.append((klass, wiring))
            continue
        merged.update(_parse_own(klass, wiring))
    return merged, not_mappings


def _merged_entries(cls: type) -> tuple[dict[Any, _Entry], list[tuple[type, Any]]]:
    """Like :func:`_merge`, with the ``None`` exclusions applied."""
    merged, not_mappings = _merge(cls)
    return {k: e for k, e in merged.items() if not e.excluded}, not_mappings


def collect_wiring_from_mro(cls: type) -> dict[str, Any]:
    """Walk MRO bottom-up and merge ``__wiring__`` dicts.

    Child entries replace parent entries for the same key (whole-entry
    replacement).  A value of ``None`` excludes the key entirely (even if a
    parent defines it).  A class whose ``__wiring__`` attribute itself is
    ``None`` contributes nothing, so it keeps every inherited entry.

    This function is deliberately lenient: it never raises, and a class whose
    ``__wiring__`` is not a mapping contributes nothing.  Use
    :func:`check_wiring_declaration` / :func:`parse_wiring` to have such
    declarations reported.

    Args:
        cls: The class to collect wiring for.

    Returns:
        Merged wiring dict.  Keys with ``None`` values are removed.
    """
    entries, _ = _merged_entries(cls)
    return {name: entry.value for name, entry in entries.items()}


# ── Grammar ────────────────────────────────────────────────────


def _type_name(value: object) -> str:
    return type(value).__name__


def _non_str_items(values: list[Any]) -> list[Any]:
    return [v for v in values if not isinstance(v, str)]


def _format_non_str(values: list[Any]) -> str:
    return ", ".join(f"{v!r} ({_type_name(v)})" for v in values)


def _check_key_list(
    value: Any,
    what: str,
    malformed: list[str],
    non_str: list[str],
) -> list[str] | None:
    """Validate that *value* is a ``list`` of ``str``.

    Appends to *malformed* / *non_str* and returns the list when its shape
    is a list (even with non-str items, so emptiness can still be judged),
    else ``None``.
    """
    if not isinstance(value, list):
        malformed.append(f"{what} must be a list of str, got {_type_name(value)} {value!r}")
        return None
    bad = _non_str_items(value)
    if bad:
        non_str.append(
            f"{what} contains non-str key(s) {_format_non_str(bad)}; every key must be a str"
        )
    return value


def _overlap(required: list[Any] | None, optional: list[Any] | None) -> list[str]:
    """Keys listed both as required and as optional (str keys only, in order)."""
    if not required or not optional:
        return []
    req = {k for k in required if isinstance(k, str)}
    return list(dict.fromkeys(k for k in optional if isinstance(k, str) and k in req))


def _overlap_detail(keys: list[str]) -> str:
    noun = "key" if len(keys) == 1 else "keys"
    return (
        f"{noun} {', '.join(repr(k) for k in keys)} listed both as required and as "
        "optional; a key is one or the other, so list it in only one of the two lists"
    )


def _check_entry(
    param_name: Any, value: Any
) -> tuple[WiringSpec | None, list[tuple[str, str, str]]]:
    """Validate the grammar of one merged entry (no registry lookups).

    Returns:
        ``(spec, problems)`` where ``spec`` is ``None`` whenever there is at
        least one problem, and each problem is ``(kind, registry_name, detail)``.
    """
    malformed: list[str] = []
    non_str: list[str] = []
    empty: list[str] = []
    registry_name = ""

    if not isinstance(param_name, str):
        malformed.append(
            f"parameter name must be a str, got {_type_name(param_name)} {param_name!r}"
        )
    elif not param_name.isidentifier():
        malformed.append(f"parameter name {param_name!r} is not a valid Python identifier")
    elif keyword.iskeyword(param_name):
        malformed.append(
            f"parameter name {param_name!r} is a Python keyword, so it can be neither a "
            f"constructor parameter nor an attribute; rename the entry"
        )
    elif param_name.startswith("__") and param_name.endswith("__"):
        malformed.append(
            f"parameter name {param_name!r} is a dunder name, which Python reserves for "
            f"special attributes (a placeholder there would replace {param_name}); "
            f"use an ordinary attribute name"
        )

    spec_kwargs: dict[str, Any] = {}

    if isinstance(value, str):
        # Mode 1: every key of the registry.
        if not value:
            malformed.append("registry name must be a non-empty str")
        registry_name = value
        spec_kwargs = {"registry_name": value}

    elif isinstance(value, tuple):
        # Mode 2: (registry, keys) or (registry, required_keys, optional_keys)
        if len(value) not in (2, 3):
            malformed.append(
                "tuple entries must be (registry, keys) or "
                f"(registry, required_keys, optional_keys), got a {len(value)}-tuple {value!r}"
            )
        else:
            reg = value[0]
            if not isinstance(reg, str) or not reg:
                malformed.append(
                    f"first tuple element (registry name) must be a non-empty str, "
                    f"got {_type_name(reg)} {reg!r}"
                )
            else:
                registry_name = reg
            keys = _check_key_list(value[1], "second tuple element (keys)", malformed, non_str)
            optional: list[str] | None = None
            optional_ok = True
            # A third element of None is accepted as "no optional keys" (1.x).
            if len(value) == 3 and value[2] is not None:
                optional = _check_key_list(
                    value[2], "third tuple element (optional keys)", malformed, non_str
                )
                optional_ok = optional is not None
            both = _overlap(keys, optional)
            if both:
                malformed.append(_overlap_detail(both))
            if keys is not None and optional_ok and not keys and not optional:
                empty.append(
                    "explicit key list is empty, so no value could ever be valid; "
                    "list at least one key, or use the bare registry name to allow every key"
                )
            spec_kwargs = {
                "registry_name": registry_name,
                "allowed_keys": tuple(keys) if keys is not None else None,
                "optional_keys": tuple(optional) if optional is not None else None,
            }

    elif isinstance(value, list):
        # Mode 3: literal list, no registry.
        bad = _non_str_items(value)
        if bad:
            non_str.append(
                f"literal list contains non-str value(s) {_format_non_str(bad)}; "
                f"every value must be a str"
            )
        if not value:
            empty.append(
                "literal list is empty, so no value could ever be valid; "
                "list at least one value"
            )
        spec_kwargs = {"allowed_keys": tuple(value)}

    elif isinstance(value, Mapping):
        # Dict form.
        unknown = [f for f in value if f not in _DICT_FIELDS]
        if unknown:
            hints = []
            for f in unknown:
                # 0.7 keeps real typos ('defualt', 'kyes') and drops look-alikes
                # such as 'inline_only' -> 'declare_only'.
                close = (
                    difflib.get_close_matches(f, _DICT_FIELDS, n=1, cutoff=0.7)
                    if isinstance(f, str)
                    else []
                )
                hints.append(f"{f!r} (did you mean {close[0]!r}?)" if close else repr(f))
            malformed.append(
                f"unknown field(s) {', '.join(hints)}; "
                f"allowed fields: {', '.join(repr(f) for f in _DICT_FIELDS)}"
            )

        reg = value.get("registry", _MISSING)
        if reg is _MISSING:
            malformed.append(
                "dict-form entry needs 'registry' (the target registry name); "
                "use a plain list for a literal list without a registry"
            )
        elif not isinstance(reg, str) or not reg:
            malformed.append(f"'registry' must be a non-empty str, got {_type_name(reg)} {reg!r}")
        else:
            registry_name = reg

        keys_raw = value.get("keys", _MISSING)
        keys = None
        if keys_raw is not _MISSING:
            keys = _check_key_list(keys_raw, "'keys'", malformed, non_str)

        optional_raw = value.get("optional_keys", _MISSING)
        optional = None
        optional_ok = True
        if optional_raw is not _MISSING:
            if keys_raw is _MISSING:
                malformed.append(
                    "'optional_keys' requires 'keys' (without 'keys' every registered "
                    "key is already allowed)"
                )
            optional = _check_key_list(optional_raw, "'optional_keys'", malformed, non_str)
            optional_ok = optional is not None
        both = _overlap(keys, optional)
        if both:
            malformed.append(_overlap_detail(both))

        if keys is not None and optional_ok and not keys and not optional:
            empty.append(
                "'keys' is empty, so no value could ever be valid; list at least one key, "
                "or omit 'keys' to allow every registered key"
            )

        default = value.get("default", _MISSING)
        if default is not _MISSING and (not isinstance(default, str) or not default):
            malformed.append(
                "'default' must be a single key (non-empty str), "
                f"got {_type_name(default)} {default!r}"
            )

        declare_only = value.get("declare_only", _MISSING)
        if declare_only is not _MISSING and not isinstance(declare_only, bool):
            malformed.append(
                f"'declare_only' must be a bool, got {_type_name(declare_only)} {declare_only!r}"
            )
        if declare_only is True and default is not _MISSING:
            malformed.append(
                "'declare_only' cannot be combined with 'default': a declaration-only "
                "entry has no receptor that a default could fill"
            )

        spec_kwargs = {
            "registry_name": registry_name,
            "allowed_keys": tuple(keys) if keys is not None else None,
            "optional_keys": tuple(optional) if optional is not None else None,
            "default_key": default if isinstance(default, str) else None,
            "declare_only": declare_only is True,
        }

    else:
        malformed.append(
            "expected a registry name (str), a (registry, keys) or "
            "(registry, required_keys, optional_keys) tuple, a dict-form entry "
            "{'registry': ..., ...} or a list of literal values, "
            f"got {_type_name(value)} {value!r}"
        )

    problems: list[tuple[str, str, str]] = []
    if malformed:
        problems.append(("malformed", registry_name, "; ".join(malformed)))
    for detail in non_str:
        problems.append(("non_str_key", registry_name, detail))
    for detail in empty:
        problems.append(("empty_keys", registry_name, detail))
    if problems:
        return None, problems
    return WiringSpec(param_name=param_name, **spec_kwargs), []


def _owner_key(cls: type) -> str | None:
    """The class's own registry key (``None`` when it has none)."""
    key = cls.__dict__.get("__registry_key__")
    return key if isinstance(key, str) else None


def _make_problem(
    cls: type,
    owner_key: str | None,
    param_name: Any,
    registry_name: str,
    kind: str,
    detail: str,
    declared_on: type | None = None,
) -> WiringProblem:
    if declared_on is not None and declared_on is not cls:
        detail = f"{detail} (entry declared on {declared_on.__qualname__})"
    return WiringProblem(
        owner=cls.__qualname__,
        param_name=param_name if isinstance(param_name, str) else repr(param_name),
        registry_name=registry_name,
        kind=kind,
        detail=detail,
        owner_key=owner_key,
    )


class _Parsed(NamedTuple):
    """Grammar result for one merged entry: a spec, or its problems."""

    entry: _Entry
    spec: WiringSpec | None
    problems: list[WiringProblem]


def _parse_class(
    cls: type, owner_key: str | None
) -> tuple[list[WiringProblem], list[_Parsed]]:
    """Grammar pass over the merged ``__wiring__`` of *cls*.

    Returns:
        ``(not_a_dict_problems, parsed_entries)``, the entries in
        declaration order.
    """
    entries, not_mappings = _merged_entries(cls)
    return _parse_entries(cls, owner_key, entries, not_mappings)


def _parse_entries(
    cls: type,
    owner_key: str | None,
    entries: dict[Any, _Entry],
    not_mappings: list[tuple[type, Any]],
) -> tuple[list[WiringProblem], list[_Parsed]]:
    """:func:`_parse_class` on already merged entries (exclusions applied)."""
    head: list[WiringProblem] = []
    for klass, wiring in not_mappings:
        detail = (
            "must be a dict mapping parameter names to wiring entries, "
            f"got {_type_name(wiring)} {wiring!r}"
        )
        if klass is not cls:
            detail = f"{detail} (declared on {klass.__qualname__})"
        head.append(
            WiringProblem(
                owner=cls.__qualname__,
                param_name="",
                registry_name="",
                kind="not_a_dict",
                detail=detail,
                owner_key=owner_key,
            )
        )

    parsed: list[_Parsed] = []
    for entry in entries.values():
        problems = [
            _make_problem(
                cls, owner_key, entry.param_name, registry_name, kind, detail,
                entry.declared_on,
            )
            for kind, registry_name, detail in entry.problems
        ]
        parsed.append(_Parsed(entry, entry.spec, problems))
    return head, parsed


def _grammar_problems(cls: type) -> tuple[list[WiringProblem], list[_Parsed]]:
    """Every grammar problem of *cls* (in order) plus the parsed entries."""
    head, parsed = _parse_class(cls, _owner_key(cls))
    return head + [p for item in parsed for p in item.problems], parsed


def parse_wiring(cls: type) -> list[WiringSpec]:
    """Parse a class's merged ``__wiring__`` into normalized ``WiringSpec`` list.

    Only the grammar is checked; no registry is consulted, so this works
    before any implementation is registered.

    Args:
        cls: The class to parse wiring for.

    Returns:
        List of ``WiringSpec`` instances in declaration order.  Empty list if
        no wiring declared.

    Raises:
        WiringResolutionError: If any entry is malformed (``malformed``,
            ``not_a_dict``, ``non_str_key`` or ``empty_keys`` problems); all
            of them are reported at once.
    """
    problems, parsed = _grammar_problems(cls)
    if problems:
        raise WiringResolutionError(problems=problems)
    return [item.spec for item in parsed if item.spec is not None]


def check_wiring_declaration(cls: type) -> None:
    """Validate the grammar of a class's merged ``__wiring__``.

    Grammar only: no registry is consulted, nothing is mutated, and the cost
    is one MRO walk, so it is safe to call at class-creation time (before a
    class is added to its registry).

    Checks performed: ``__wiring__`` is a dict (``not_a_dict``); every entry
    name is an identifier that is neither a keyword nor a dunder name, every
    entry has one of the supported shapes and well-typed fields, the dict
    form has no unknown fields, ``optional_keys`` comes with ``keys``, no key
    is listed both as required and as optional, ``declare_only`` is not
    combined with ``default`` (``malformed``); every key is a str
    (``non_str_key``); explicit key lists are not empty (``empty_keys``).

    Each class's own entries are parsed once and reused while they stay
    unchanged, so checking a subclass costs little more than the MRO walk.

    Args:
        cls: The class whose (merged) ``__wiring__`` to check.

    Raises:
        WiringResolutionError: Listing every grammar problem found.
    """
    entries, not_mappings = _merged_entries(cls)
    if not not_mappings and not any(entry.problems for entry in entries.values()):
        return
    head, parsed = _parse_entries(cls, _owner_key(cls), entries, not_mappings)
    raise WiringResolutionError(problems=head + [p for item in parsed for p in item.problems])


# ── Key expansion ──────────────────────────────────────────────


def expand_relative_keys(
    declared: tuple[str, ...] | list[str],
    registry_keys: list[str],
    separator: str,
) -> list[str]:
    """Expand capability-relative (short) keys into fully-qualified keys.

    A registry created with ``key_separator="."`` holds fully-qualified keys
    such as ``browser.click`` / ``desktop.click``.  Requiring every
    ``__wiring__`` declaration to spell out the provider prefix welds the
    declaring class to one provider, even though "I need a *click*" is a
    provider-independent statement.  This function lifts that restriction:

    - A declared key that **contains** the separator is passed through
      **unchanged**.  This is the escape hatch — spell out
      ``"browser.click"`` and you get exactly ``browser.click``.
    - A declared key that does **not** contain the separator is expanded
      into every registered key whose trailing segment equals it.
    - A short key matching nothing is passed through unchanged, so the
      caller's normal "key not found" validation reports it as usual.
      Non-str declared keys are passed through unchanged as well (the
      caller's validation reports them); they never raise here.
    - When the registry has no separator (flat keys) nothing is expanded.

    Ambiguity is deliberately **not** an error here
    ------------------------------------------------
    If ``"click"`` expands to several fully-qualified keys, *all* of them
    are returned and all of them land in the generated ``Literal[...]``.
    That is intentional, and it is the subtle part: conscribe's job is
    **expansion**, not **arbitration**.

    Config generation happens once, ahead of time, with no knowledge of
    which providers a given assembled runtime will actually hold.  At that
    moment every candidate is legitimately possible, so narrowing the
    ``Literal`` to one of them would reject configs that are perfectly
    valid for some other assembly.  Deciding *which* ``click`` is meant
    requires knowing the concrete set of providers present, and only the
    consuming framework knows that — at wiring/negotiation time, where it
    can refuse a genuine collision with both candidates named, or pick the
    single provider that is actually installed.

    So: conscribe widens the type, the consumer narrows the instance.  Do
    not "fix" this by raising on multiple matches — that would move a
    runtime decision into a build-time one and break the very portability
    the expansion exists to provide.  (A ``default`` key is different: it
    is one concrete choice made at generation time, so resolution requires
    it to be fully qualified instead of expanding it.)

    Args:
        declared: Key strings exactly as written in ``__wiring__``.
        registry_keys: All keys currently registered in the target registry.
        separator: The target registry's ``key_separator`` (``""`` = flat).

    Returns:
        The expanded key list, order-preserving and de-duplicated.
        Expansions of a single short key are sorted for determinism
        (registration order must not leak into generated sources).
    """
    if not separator:
        return list(declared)
    return _expand(declared, _segment_index(registry_keys, separator), separator)


def _segment_index(registry_keys: Iterable[Any], separator: str) -> dict[str, list[str]]:
    """``{trailing segment: sorted registered keys ending in it}``."""
    index: dict[str, list[str]] = {}
    for key in registry_keys:
        if isinstance(key, str):
            index.setdefault(key.rsplit(separator, 1)[-1], []).append(key)
    for keys in index.values():
        keys.sort()
    return index


def _expand(
    declared: Iterable[Any], index: dict[str, list[str]], separator: str
) -> list[Any]:
    """:func:`expand_relative_keys` over a prebuilt :func:`_segment_index`."""
    expanded: list[Any] = []
    seen: set[str] = set()
    for key in declared:
        if not isinstance(key, str):
            # Not a key at all: leave it for the caller's validation.
            expanded.append(key)
            continue
        if separator in key:
            # Already fully qualified — pass through untouched.
            candidates = [key]
        else:
            # Unknown short name: keep it so the caller's missing-key check
            # produces the normal error message.
            candidates = index.get(key) or [key]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


class _RegistryView:
    """One registry's keys, read once per resolution run.

    :func:`validate_wiring` resolves every entry of every class against the
    same few registries; the key list, its set and the trailing-segment
    index used for capability-relative keys are computed once per registry
    instead of once per entry.
    """

    __slots__ = ("keys", "key_set", "separator", "_index")

    def __init__(self, registry: Any) -> None:
        self.keys: list[str] = registry.keys()
        self.key_set = frozenset(self.keys)
        self.separator: str = getattr(registry, "separator", "") or ""
        self._index: dict[str, list[str]] | None = None

    def expand(self, declared: Iterable[Any]) -> list[Any]:
        """The declared keys, capability-relative ones expanded."""
        if not self.separator:
            return list(declared)
        if self._index is None:
            self._index = _segment_index(self.keys, self.separator)
        return _expand(declared, self._index, self.separator)


#: ``{registry name: view}`` shared by one resolution run (``None``: unknown).
_Views = dict[str, "_RegistryView | None"]


def _view(views: _Views, registry_name: str) -> _RegistryView | None:
    if registry_name not in views:
        from conscribe.registration.registry import get_registry

        registry = get_registry(registry_name)
        views[registry_name] = None if registry is None else _RegistryView(registry)
    return views[registry_name]


# ── Message helpers ────────────────────────────────────────────


def _fmt_list(values: Iterable[Any]) -> str:
    """Sorted list repr, capped at ``_LIST_CAP`` items."""
    items = sorted(values, key=str)
    if len(items) <= _LIST_CAP:
        return repr(items)
    return f"{items[:_LIST_CAP]!r} (+{len(items) - _LIST_CAP} more)"


def _failures_hint() -> str:
    """Name the modules ``discover()`` recorded as failed, if any."""
    from conscribe.discover import import_failures

    failures = import_failures()
    if not failures:
        return ""
    items = sorted(failures.items())
    shown = "; ".join(f"{name} ({desc})" for name, desc in items[:_FAILURES_CAP])
    more = f"; +{len(items) - _FAILURES_CAP} more" if len(items) > _FAILURES_CAP else ""
    return (
        "; discover() recorded import failures, and classes defined in those "
        f"modules are not registered: {shown}{more}"
    )


def _suggest_key(
    key: str, registered: list[str], separator: str, declared: set[str]
) -> str | None:
    """A close registered key for a mistyped *key*, or ``None``.

    For registries with a separator, a short key is compared with the trailing
    segments and a qualified key with the keys under the same prefix, so an
    unrelated sibling with a similar prefix is never suggested.  Keys already
    declared in the same entry are never suggested.
    """
    names = [k for k in registered if isinstance(k, str) and k not in declared]
    if separator and separator not in key:
        segments = sorted({k.rsplit(separator, 1)[-1] for k in names} - declared)
        close = difflib.get_close_matches(key, segments, n=1, cutoff=0.75)
        return close[0] if close else None
    if separator:
        prefix, _, segment = key.rpartition(separator)
        siblings = {
            k.rpartition(separator)[2]: k for k in names if k.rpartition(separator)[0] == prefix
        }
        close = difflib.get_close_matches(segment, sorted(siblings), n=1, cutoff=0.75)
        if close:
            return siblings[close[0]]
    close = difflib.get_close_matches(key, sorted(names), n=1, cutoff=0.8)
    return close[0] if close else None


def _unknown_keys_detail(
    missing: list[str],
    *,
    optional: bool,
    declared: set[str],
    registry_name: str,
    registered: list[str],
    separator: str,
) -> str:
    noun = "key" if len(missing) == 1 else "keys"
    qualifier = "optional " if optional else ""
    detail = (
        f"unknown {qualifier}{noun} {', '.join(repr(k) for k in missing)} "
        f"in registry '{registry_name}'"
    )
    notes = []
    short = [k for k in missing if separator and separator not in k]
    if short:
        ends = " or ".join(repr(f"{separator}{k}") for k in short)
        notes.append(f"capability-relative: no registered key ends in {ends}")
    suggestions = []
    for key in missing:
        best = _suggest_key(key, registered, separator, declared)
        if best is not None:
            suggestions.append((key, best))
    if len(suggestions) == 1 and len(missing) == 1:
        notes.append(f"did you mean {suggestions[0][1]!r}?")
    elif suggestions:
        pairs = ", ".join(f"{best!r} for {key!r}" for key, best in suggestions)
        notes.append(f"did you mean {pairs}?")
    if notes:
        detail += f" ({'; '.join(notes)})"
    detail += f"; registered: {_fmt_list(registered)}"
    return detail + _failures_hint()


def _unknown_registry_detail(registry_name: str) -> str:
    from conscribe.registration.registry import _REGISTRY_INDEX, _REGISTRY_INDEX_LOCK

    with _REGISTRY_INDEX_LOCK:
        known = list(_REGISTRY_INDEX)
    detail = f"unknown registry '{registry_name}'"
    close = difflib.get_close_matches(registry_name, sorted(known), n=3, cutoff=0.6)
    if close:
        alternatives = " or ".join(repr(c) for c in close)
        detail += f" (did you mean {alternatives}?)"
    elif known:
        detail += f"; known registries: {_fmt_list(known)}"
    else:
        detail += "; no registry exists yet (create_registrar() has not run)"
    return detail + _failures_hint()


def _empty_registry_detail(registry_name: str) -> str:
    return (
        f"registry '{registry_name}' is empty (no class is registered in it): "
        "discover() missed the package that implements it, or one of its "
        "modules failed to import; import every implementation module before "
        "resolving wiring" + _failures_hint()
    )


def _default_problem(
    default: str,
    *,
    registry_name: str,
    view: _RegistryView,
    allowed: list[str] | None,
) -> str | None:
    """Detail text when *default* is not a valid default key, else ``None``.

    ``allowed=None`` skips the allowed-keys membership check.
    """
    registered, separator = view.keys, view.separator
    if default not in view.key_set:
        if separator and separator not in default:
            matches = sorted(
                k
                for k in registered
                if isinstance(k, str) and k.rsplit(separator, 1)[-1] == default
            )
            if matches:
                return (
                    f"default {default!r} must be a fully-qualified key: registry "
                    f"'{registry_name}' uses key_separator {separator!r}, so write one of "
                    f"{matches!r} (a default is one concrete choice; it is never expanded)"
                )
        detail = f"default {default!r} is not registered in registry '{registry_name}'"
        best = _suggest_key(default, registered, separator, set())
        if best is not None:
            candidates = [best]
            if separator and separator not in best:
                # The suggestion is a trailing segment, but a default is never
                # expanded: suggest the keys it would have to be written as.
                candidates = sorted(
                    k
                    for k in registered
                    if isinstance(k, str) and k.rsplit(separator, 1)[-1] == best
                )[:_SUGGESTION_CAP] or [best]
            detail += f" (did you mean {' or '.join(repr(c) for c in candidates)}?)"
        detail += f"; registered: {_fmt_list(registered)}"
        return detail + _failures_hint()
    if allowed is not None and default not in allowed:
        return (
            f"default {default!r} is not among the entry's allowed keys {allowed!r}; "
            "add it to the entry's keys or choose an allowed key"
        )
    return None


# ── Resolution ─────────────────────────────────────────────────


def _expand_and_split(
    declared: tuple[str, ...], view: _RegistryView
) -> tuple[list[str], list[str]]:
    """Expand *declared* keys; return ``(found, missing)`` in declaration order."""
    expanded = view.expand(declared)
    registered = view.key_set
    found = [k for k in expanded if k in registered]
    missing = [k for k in expanded if k not in registered]
    return found, missing


def _resolve_spec(
    spec: WiringSpec, views: _Views | None = None
) -> tuple[ResolvedWiring | None, list[tuple[str, str]]]:
    """Resolve one grammatically valid spec; return ``(resolved, [(kind, detail)])``.

    *views* caches each registry's keys across the specs of one resolution
    run (:func:`validate_wiring` passes one for every class).
    """
    if not spec.registry_name:
        # Mode 3: literal list, validated by the grammar pass.
        return (
            ResolvedWiring(
                param_name=spec.param_name,
                allowed_keys=list(spec.allowed_keys or ()),
                registry_name=None,
            ),
            [],
        )

    view = _view({} if views is None else views, spec.registry_name)
    if view is None:
        return None, [("unknown_registry", _unknown_registry_detail(spec.registry_name))]

    registered = view.keys
    if not registered:
        return None, [("empty_registry", _empty_registry_detail(spec.registry_name))]

    separator = view.separator
    problems: list[tuple[str, str]] = []
    optional_resolved: list[str] | None = None

    if spec.allowed_keys is None:
        # Mode 1 (or dict form without "keys"): every registered key.
        allowed = sorted(registered, key=str)
    else:
        declared = set(spec.allowed_keys) | set(spec.optional_keys or ())
        required, missing = _expand_and_split(spec.allowed_keys, view)
        if missing:
            problems.append((
                "unknown_key",
                _unknown_keys_detail(
                    missing,
                    optional=False,
                    declared=declared,
                    registry_name=spec.registry_name,
                    registered=registered,
                    separator=separator,
                ),
            ))
        allowed = list(required)
        if spec.optional_keys is not None:
            optional_found, missing_opt = _expand_and_split(spec.optional_keys, view)
            if missing_opt:
                problems.append((
                    "unknown_optional_key",
                    _unknown_keys_detail(
                        missing_opt,
                        optional=True,
                        declared=declared,
                        registry_name=spec.registry_name,
                        registered=registered,
                        separator=separator,
                    ),
                ))
            optional_resolved = optional_found
            # Combined: required + optional for Literal type generation
            allowed += [k for k in optional_found if k not in required]

    if spec.default_key is not None:
        detail = _default_problem(
            spec.default_key,
            registry_name=spec.registry_name,
            view=view,
            # With unknown keys the allowed set is incomplete; judging the
            # default against it would only echo the unknown-key problem.
            allowed=None if problems else allowed,
        )
        if detail is not None:
            problems.append(("bad_default", detail))

    if problems:
        return None, problems
    return (
        ResolvedWiring(
            param_name=spec.param_name,
            allowed_keys=allowed,
            registry_name=spec.registry_name,
            optional_keys=optional_resolved,
            default_key=spec.default_key,
            declare_only=spec.declare_only,
        ),
        [],
    )


def _describe_slot(slot: Any) -> str:
    """``constructor parameter 'x' of Owner`` / ``class annotation 'x' on Owner``."""
    owner = getattr(slot.declared_on, "__qualname__", "?")
    if slot.via == "init":
        return f"constructor parameter '{slot.name}' of {owner}"
    return f"class annotation '{slot.name}' on {owner}"


def _slot_problems(spec: WiringSpec, slot: Any) -> list[tuple[str, str]]:
    """Problems of an entry that depend on its runtime slot: ``[(kind, detail)]``.

    * ``contradiction``: a ``declare_only`` entry whose class has a runtime
      slot for it (a constructor parameter or a class annotation) -- a
      declaration-only constraint receives nothing.
    * ``bad_default``: a ``default`` on an entry without a runtime slot, or
      on a multi-instance (container) receptor.  (``declare_only`` together
      with ``default`` is a grammar problem, reported as ``malformed``.)

    *slot* is the entry's :class:`conscribe.introspect.ReceptorSlot`, or
    ``None`` for a slot-less entry.
    """
    name = spec.param_name
    found: list[tuple[str, str]] = []
    if spec.declare_only and slot is not None:
        found.append((
            "contradiction",
            f"the entry is declare_only, but the class has a runtime slot for it "
            f"({_describe_slot(slot)}); a declaration-only constraint receives nothing. "
            f"Remove 'declare_only' if the framework injects '{name}', or remove the slot",
        ))
    if spec.default_key is not None and not spec.declare_only:
        if slot is None:
            found.append((
                "bad_default",
                f"default {spec.default_key!r} needs a receptor, but '{name}' has no "
                f"runtime slot (no constructor parameter, no class annotation), so nothing "
                f"could receive it; add the slot or remove 'default'",
            ))
        elif slot.shape != "scalar":
            found.append((
                "bad_default",
                f"default {spec.default_key!r} is only allowed on a single-instance "
                f"receptor, but '{name}' holds several instances ({_describe_slot(slot)}, "
                f"shape {slot.shape!r}); remove 'default'",
            ))
    return found


def _same_value(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - an object that cannot compare is not the key
        return False


@functools.lru_cache(maxsize=256)
def _keys_adapter(keys: tuple[str, ...], scalar: bool) -> Any:
    """A pydantic adapter accepting what a wired field of *keys* can accept.

    ``Literal[keys]`` for a single-instance receptor; for a multi-instance
    one either a key or a list of keys -- a superset of every type the
    extractor maps a receptor annotation to, so a value this adapter rejects
    is rejected by the generated config field too.
    """
    from typing import Literal, Union

    from pydantic import TypeAdapter

    literal = Literal[keys]  # type: ignore[valid-type]
    return TypeAdapter(literal if scalar else Union[literal, list[literal]])  # type: ignore[valid-type]


def _defers_default_check(annotation: Any, depth: int = 0) -> bool:
    """Whether the config build, not resolution, must judge a receptor's default.

    True when *annotation* carries metadata that could turn any input into a
    key -- a ``BeforeValidator`` / ``WrapValidator`` / ``PlainValidator`` or
    other metadata with its own pydantic core schema (an ``AfterValidator``
    only runs on an already valid key) -- or a user-written ``Literal``,
    which the extractor checks against the wiring keys itself (and reports
    more precisely).  Unresolvable annotation text naming either counts too.
    """
    from typing import Annotated, ForwardRef, Literal, get_args, get_origin

    if depth > 30:
        return True
    if isinstance(annotation, (str, ForwardRef)):
        text = annotation if isinstance(annotation, str) else annotation.__forward_arg__
        return "Annotated" in text or "Literal" in text
    origin = get_origin(annotation)
    if origin is Literal:
        return True
    if origin is Annotated:
        base, *metadata = get_args(annotation)
        if any(_transforms(item) for item in metadata):
            return True
        return _defers_default_check(base, depth + 1)
    return any(_defers_default_check(arg, depth + 1) for arg in get_args(annotation))


def _transforms(metadata: Any) -> bool:
    from pydantic.functional_validators import AfterValidator

    if isinstance(metadata, AfterValidator):
        return False
    return hasattr(type(metadata), "__get_pydantic_core_schema__")


def _field_metadata_transforms(slot: Any) -> bool:
    """Whether the pydantic field behind *slot* carries transforming metadata.

    For pydantic model / dataclass fields the slot annotation is the type
    pydantic resolved: ``Annotated`` validators live in the field's metadata.
    """
    owner = getattr(slot, "declared_on", None)
    fields = getattr(owner, "__pydantic_fields__", None)
    if not isinstance(fields, dict) and isinstance(owner, type):
        from conscribe.config._utils import is_pydantic_model

        if is_pydantic_model(owner):  # pydantic < 2.10 keeps model fields here
            fields = getattr(owner, "model_fields", None)
    if not isinstance(fields, dict):
        return False
    info = fields.get(slot.name)
    return any(_transforms(item) for item in getattr(info, "metadata", None) or ())


def _parameter_default_problem(resolved: ResolvedWiring, slot: Any) -> str | None:
    """``bad_default`` detail for a receptor whose own default contradicts the wiring.

    Reads the config-effective default of the slot
    (:attr:`conscribe.introspect.ReceptorSlot.default`: the parameter's
    default, a ``Field(default=...)``, or a class-level value), exactly what
    the extractor turns into the field default:

    * with a wiring ``default``, a non-``None`` slot default must name the
      same key;
    * without one, it must be a valid value of the wired field (an allowed
      key, or a list of allowed keys for a multi-instance receptor).

    Defaults produced by a ``default_factory`` are left to the config build
    (only it evaluates factories), and so are receptors the config build
    refuses for other reasons first (a name starting with ``_``) and the
    value check of receptors whose annotation carries a ``Literal`` or
    validators that may transform the value (see
    :func:`_defers_default_check`).  The detail texts are the extractor's, so
    both report a problem the same way.
    """
    if slot is None or slot.name.startswith("_"):
        return None
    default = slot.default
    if (
        default is inspect.Parameter.empty
        or default is None
        or getattr(slot, "default_factory", None) is not None
    ):
        return None
    key = resolved.default_key
    if key is not None:
        if _same_value(default, key):
            return None
        return (
            f"the parameter default {default!r} names a different key than the "
            f"wiring default {key!r}; declare the default once (remove the "
            f"parameter default or make the two equal)."
        )
    keys = tuple(dict.fromkeys(resolved.allowed_keys))
    scalar = slot.shape == "scalar"
    if not keys or _defers_default_check(slot.annotation) or _field_metadata_transforms(slot):
        return None
    from pydantic import ValidationError

    try:
        _keys_adapter(keys, scalar).validate_python(default)
    except (ValidationError, TypeError, ValueError):
        return (
            f"the parameter default {default!r} is not a valid value of the wired "
            f"field: it accepts {'a list of ' if not scalar else ''}keys from "
            f"{list(keys)}. Use one of them (or None), or declare the default with "
            f"the dict-form wiring 'default'."
        )
    return None


def _resolve_class(
    cls: type, owner_key: str | None, views: _Views | None = None
) -> tuple[dict[str, ResolvedWiring], list[WiringProblem]]:
    """Grammar + registry + slot pass over *cls*; never raises for wiring problems.

    Problems come in declaration order: ``not_a_dict`` first, then each
    entry's grammar, resolution, slot problems (see :func:`_slot_problems`)
    and parameter-default problems (see :func:`_parameter_default_problem`).
    An entry with a slot or default problem is not in the returned mapping.
    """
    resolved, problems, _ = _resolve_class_and_slots(cls, owner_key, views)
    return resolved, problems


def _resolve_class_and_slots(
    cls: type, owner_key: str | None, views: _Views | None = None
) -> tuple[dict[str, ResolvedWiring], list[WiringProblem], dict[str, Any]]:
    """:func:`_resolve_class`, also returning the receptor slots it read.

    The slots are :func:`conscribe.introspect.receptor_slots` of the entries
    that needed one (every resolved entry, and every ``default`` /
    ``declare_only`` entry), so a caller that needs them again can reuse
    them instead of resolving the annotations a second time.
    """
    if views is None:
        views = {}
    head, parsed = _parse_class(cls, owner_key)
    outcomes: list[tuple[_Parsed, ResolvedWiring | None, list[tuple[str, str]]]] = []
    wanted: list[str] = []
    for item in parsed:
        spec = item.spec
        if spec is None:
            outcomes.append((item, None, []))
            continue
        result, spec_problems = _resolve_spec(spec, views)
        outcomes.append((item, result, list(spec_problems)))
        if spec.declare_only or spec.default_key is not None or result is not None:
            wanted.append(spec.param_name)

    slots: dict[str, Any] = {}
    if wanted:
        from conscribe.introspect import receptor_slots

        slots = receptor_slots(cls, names=wanted)

    problems = list(head)
    resolved: dict[str, ResolvedWiring] = {}
    for item, result, details in outcomes:
        spec = item.spec
        if spec is None:
            problems.extend(item.problems)
            continue
        slot = slots.get(spec.param_name)
        if spec.declare_only or spec.default_key is not None:
            slot_problems = _slot_problems(spec, slot)
            if slot_problems:
                details.extend(slot_problems)
                result = None
        if result is not None and not spec.declare_only:
            detail = _parameter_default_problem(result, slot)
            if detail is not None:
                details.append(("bad_default", detail))
                result = None
        for kind, detail in details:
            problems.append(
                _make_problem(
                    cls, owner_key, spec.param_name, spec.registry_name, kind, detail,
                    item.entry.declared_on,
                )
            )
        if result is not None:
            resolved[spec.param_name] = result
    return resolved, problems, slots


def resolve_wiring(cls: type) -> dict[str, ResolvedWiring]:
    """Resolve a class's ``__wiring__`` to concrete key lists.

    For Mode 1 (registry name string), looks up the registry and uses all its keys.
    For Mode 2 (tuple) and the dict form, validates that each required and
    optional key exists in the referenced registry.
    For Mode 3 (literal list), returns the list as-is.

    Mode 2 / dict-form key lists additionally support **capability-relative
    keys** when the target registry declares a ``key_separator``: a bare
    trailing segment (``"click"``) expands to every matching fully-qualified
    key (``browser.click``, ``desktop.click``).  See
    :func:`expand_relative_keys` for the expansion rules and for why an
    ambiguous short name widens the ``Literal`` instead of raising.

    A dict-form ``default`` is validated here: it must be registered in the
    target registry exactly as written (it is never expanded, so with a
    ``key_separator`` it must be fully qualified), be among the entry's
    allowed keys (after expansion) and belong to a single-instance receptor
    with a runtime slot.  A ``declare_only`` entry must not have a runtime
    slot.  The receptor's own default (a parameter default, a
    ``Field(default=...)`` or a class-level value) must be an allowed key --
    a list of them for a multi-instance receptor -- and name the same key as
    the entry's ``default`` when there is one (``None`` is always fine).
    Slots and their defaults are read with
    :func:`conscribe.introspect.receptor_slots`.

    Every problem of the class is collected before anything is raised.

    Args:
        cls: The class whose wiring to resolve.

    Returns:
        Dict mapping param names to ``ResolvedWiring``.  Empty dict if no wiring.

    Raises:
        WiringResolutionError: Listing every problem of the class: grammar
            problems (see :func:`check_wiring_declaration`), unknown or empty
            registries, unknown required / optional keys (including
            capability-relative keys matching nothing), invalid defaults
            (``bad_default``: unregistered, not allowed, on a slot-less or
            multi-instance receptor, or contradicted by the receptor's own
            default) and ``declare_only`` entries that have a runtime slot
            (``contradiction``).
    """
    resolved, problems = _resolve_class(cls, _owner_key(cls))
    if problems:
        raise WiringResolutionError(problems=problems)
    return resolved


def _registered_classes(registrars: Any) -> list[tuple[type, str]]:
    """Every distinct class registered in *registrars*, deterministically ordered."""
    if isinstance(registrars, Mapping):
        items = list(registrars.values())
    elif callable(getattr(registrars, "get_all", None)):
        items = [registrars]
    elif isinstance(registrars, (str, bytes)):
        raise TypeError(
            "validate_wiring() expects registrars created by create_registrar(), not a "
            f"name; got {_type_name(registrars)} {registrars!r}"
        )
    else:
        try:
            items = list(registrars)
        except TypeError:
            raise TypeError(
                "validate_wiring() expects a registrar, a mapping of registrars or an "
                f"iterable of registrars, got {_type_name(registrars)} {registrars!r}"
            ) from None

    first_seen: dict[int, tuple[type, str]] = {}
    for registrar in items:
        get_all = getattr(registrar, "get_all", None)
        if not callable(get_all):
            raise TypeError(
                "validate_wiring() expects registrars created by create_registrar(), "
                f"got {_type_name(registrar)} {registrar!r}"
            )
        for key, cls in get_all().items():
            first_seen.setdefault(id(cls), (cls, key))

    return sorted(
        first_seen.values(),
        key=lambda pair: (
            str(getattr(pair[0], "__module__", "")),
            pair[0].__qualname__,
            str(pair[1]),
        ),
    )


def validate_wiring(registrars: Any) -> None:
    """Resolve the wiring of every registered class and report all problems at once.

    Walks every class registered in the given registrars (each class once,
    even when it is registered under several keys or in several of the
    given registrars), resolves its ``__wiring__`` exactly like
    :func:`resolve_wiring` (grammar, registries and keys, defaults, the
    slot-dependent ``bad_default`` / ``contradiction`` checks and the
    receptors' own defaults) and collects every problem across all classes,
    so a config build reports every wiring problem up front instead of one
    class at a time.

    Problems are ordered deterministically: classes by
    ``(module, qualname)``, and each class's problems in declaration order.

    Args:
        registrars: A registrar (as returned by ``create_registrar()``), a
            mapping whose values are registrars (e.g.
            ``{"agent": Agent, "llm": LLM}``) or an iterable of registrars.

    Raises:
        WiringResolutionError: One error whose ``problems`` lists every
            problem found in every class.
        TypeError: If *registrars* is not a registrar, mapping or iterable
            of registrars.
    """
    problems: list[WiringProblem] = []
    views: _Views = {}  # each target registry is read once for the whole run
    for cls, key in _registered_classes(registrars):
        _, class_problems = _resolve_class(cls, _owner_key(cls) or str(key), views)
        problems.extend(class_problems)
    if problems:
        raise WiringResolutionError(problems=problems)


# ── WiredField descriptor ──────────────────────────────────────


class WiredField:
    """Placeholder for a wired attribute that the framework injects per instance.

    :func:`inject_wired_descriptors` installs one at class creation for each
    injected ``__wiring__`` entry that has no other way to receive its value:
    the name is not a constructor parameter, the class body neither binds nor
    annotates it, and no inherited binding can hold the injected value (see
    :func:`inject_wired_descriptors` for the exact rule).  It is a non-data
    descriptor, so injecting is a plain instance assignment::

        agent.env              # AttributeError: ... has not been injected yet
        agent.env = my_env     # the framework injects (an instance attribute)
        agent.env              # my_env

    **Instance access** before injection raises ``AttributeError`` naming the
    attribute and its registry.  On an instance of a subclass that stopped
    wiring the name (``{"env": None}`` or a ``declare_only`` entry) it raises
    Python's plain ``'Agent' object has no attribute 'env'`` instead.

    **Class access** (``MyAgent.env``) returns the descriptor itself, unless
    the accessing class, or any class between it and the class holding the
    descriptor, annotates the name in its own body (``env: Environment``
    without a default).  Then it raises ``AttributeError``, exactly like any
    annotation-only attribute: the name has no class-level value.  That is the
    answer field collectors read -- ``@dataclass`` takes a descriptor whose
    class access raises for "no default", and pydantic looks defaults up with
    ``getattr(cls, name, <sentinel>)`` -- so a dataclass or pydantic subclass
    that re-annotates an inherited wired name gets a required field, never
    this placeholder as its default.  The check is made at access time, so it
    also holds for classes created after the placeholder (and for the copies
    ``@dataclass(slots=True)`` / ``attrs.define`` make).  For such names
    ``hasattr(MyAgent, "env")`` is ``False`` and ``getattr(MyAgent, "env",
    d)`` returns ``d``.  (A class never gets a placeholder of its own for a
    name it annotates: attrs reads defaults straight from the class
    ``__dict__``.)

    Look placeholders up without triggering ``__get__`` with
    :func:`wired_field` / :func:`wired_fields`, or read the raw class
    attribute with :func:`inspect.getattr_static`.  In tests, inject fakes on
    the instance (``agent.env = fake``).  When a test patches the class
    instead, ``monkeypatch.setattr(MyAgent, "env", fake)`` works for names
    whose class access returns the descriptor; for annotated names pass
    ``raising=False`` (and ``unittest.mock.patch.object(..., create=True)``),
    since the class has no attribute to replace.

    Attributes:
        name: The wired attribute name.
        registry_name: The target registry (``""`` for a literal-list entry).
    """

    __slots__ = ("name", "registry_name")

    def __init__(self, name: str, registry_name: str = "") -> None:
        self.name = name
        self.registry_name = registry_name

    def __repr__(self) -> str:
        if self.registry_name:
            return f"WiredField({self.name!r}, registry={self.registry_name!r})"
        return f"WiredField({self.name!r})"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            if isinstance(objtype, type):
                annotating = _annotating_class(objtype, self.name, self)
                if annotating is not None:
                    raise AttributeError(self._class_access_message(objtype, annotating))
            return self
        owner = type(obj)
        excluded, registry_name = _instance_wiring(owner, self.name, self)
        if excluded:
            raise AttributeError(
                f"'{owner.__name__}' object has no attribute '{self.name}'",
                name=self.name,
                obj=obj,
            )
        registry_name = self.registry_name if registry_name is None else registry_name
        msg = f"Wired attribute '{self.name}' on {owner.__name__} has not been injected yet."
        if registry_name:
            msg += f" Expected injection from registry '{registry_name}'."
        raise AttributeError(msg)

    def _class_access_message(self, owner: type, annotating: type) -> str:
        """Message of the ``AttributeError`` raised on class-level access."""
        source = f" (wired from registry '{self.registry_name}')" if self.registry_name else ""
        return (
            f"{owner.__qualname__}.{self.name} has no class-level value: "
            f"{annotating.__qualname__} annotates the wired attribute '{self.name}' "
            f"without a default, so every instance gets its own value{source}. Use "
            f"conscribe.wired_field({owner.__name__}, {self.name!r}) to look up the "
            f"injection placeholder, or inspect.getattr_static({owner.__name__}, "
            f"{self.name!r}) to read the raw class attribute."
        )


class _WiringTombstone:
    """Covers an inherited :class:`WiredField` on a class that no longer wires the name.

    :func:`inject_wired_descriptors` installs one on a class whose merged
    ``__wiring__`` excludes an inherited entry (``{"env": None}``) or
    redeclares it ``declare_only``, when the class would otherwise inherit a
    placeholder for it (and neither binds nor annotates the name itself).
    Class access raises ``AttributeError`` (the name is no wired attribute of
    the class, and field collectors see no default); instance access raises
    the plain ``AttributeError`` Python raises for a missing attribute.  It is
    a non-data descriptor, so assigning the attribute on an instance works as
    usual.  :func:`wired_field` / :func:`wired_fields` never report it.
    """

    __slots__ = ("name", "owner", "reason")

    def __init__(self, name: str, owner: str, reason: str) -> None:
        self.name = name
        self.owner = owner
        self.reason = reason

    def __repr__(self) -> str:
        return f"<conscribe: {self.owner} no longer wires {self.name!r}: it {self.reason}>"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            owner = objtype.__name__ if isinstance(objtype, type) else self.owner
            raise AttributeError(
                f"type object '{owner}' has no attribute '{self.name}': {self.owner} "
                f"{self.reason}, so '{self.name}' is not a wired attribute there",
                name=self.name,
                obj=objtype,
            )
        raise AttributeError(
            f"'{type(obj).__name__}' object has no attribute '{self.name}'",
            name=self.name,
            obj=obj,
        )


#: Inherited bindings that define a method: a placeholder shadows them, since
#: before injection they would hand out a bound method (a Protocol or mixin
#: stub of the member) instead of raising "not injected".
_METHOD_TYPES: tuple[type, ...] = (
    types.FunctionType,
    staticmethod,
    classmethod,
    functools.partialmethod,
    functools.singledispatchmethod,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
)


def _own_annotation_names(klass: type) -> Any:
    """Names *klass* annotates in its own body (PEP 649-safe, nothing resolved).

    Returns a container supporting ``in``; never mutate it.
    """
    if _PY314:
        from conscribe.config._annotations import own_annotations

        return own_annotations(klass)
    annotations = klass.__dict__.get("__annotations__")
    return annotations if isinstance(annotations, dict) else ()


def _annotates(klass: type, name: str) -> bool:
    """Whether *klass* annotates *name* in its own body."""
    return name in _own_annotation_names(klass)


def _annotating_class(owner: type, name: str, descriptor: Any) -> type | None:
    """The first class from *owner* up to the holder of *descriptor* annotating *name*.

    ``None`` as well when no class along *owner*'s MRO holds *descriptor*:
    then ``__get__`` was called outside attribute lookup (pydantic calls it
    on a descriptor given as ``Field(default=...)``), with no class context.
    """
    annotating: type | None = None
    for klass in owner.__mro__:
        if annotating is None and _annotates(klass, name):
            annotating = klass
        if klass.__dict__.get(name) is descriptor:
            return annotating
    return None


def _entry_registry(entry: Any) -> str | None:
    """The registry name a raw ``__wiring__`` entry names (``""``: literal list)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, tuple):
        return entry[0] if entry and isinstance(entry[0], str) else None
    if isinstance(entry, Mapping):
        registry = entry.get("registry")
        return registry if isinstance(registry, str) else None
    if isinstance(entry, list):
        return ""
    return None


def _instance_wiring(owner: type, name: str, descriptor: Any) -> tuple[bool, str | None]:
    """How *owner* wires *name*, read from the nearest own ``__wiring__`` entry.

    Walks *owner*'s MRO up to the class holding *descriptor*.  Returns
    ``(excluded, registry_name)``: ``excluded`` when the nearest entry is
    ``None`` or ``declare_only``; ``registry_name`` from the nearest entry
    that wires the name (``None`` when no class on the way declares one).
    """
    for klass in owner.__mro__:
        wiring = klass.__dict__.get("__wiring__")
        if isinstance(wiring, Mapping) and name in wiring:
            entry = wiring[name]
            if entry is None or (isinstance(entry, Mapping) and entry.get("declare_only") is True):
                return True, None
            return False, _entry_registry(entry)
        if klass.__dict__.get(name) is descriptor:
            break
    return False, None


def _first_binding(cls: type, name: str) -> Any:
    """The value bound to *name* in the first ``__dict__`` along ``cls.__mro__``.

    Descriptors are not invoked.  ``_MISSING`` when no class binds *name*.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return _MISSING


def _require_class(func: str, cls: Any) -> None:
    if not isinstance(cls, type):
        raise TypeError(f"{func}() expects a class, got {_type_name(cls)} {cls!r}")


def _injectable_names(cls: type) -> list[str]:
    """Merged entry names of *cls* the framework injects, in declaration order.

    Excluded, ``declare_only`` and malformed entries are left out.
    """
    entries, _ = _merged_entries(cls)
    return [
        name
        for name, entry in entries.items()
        if isinstance(name, str) and entry.spec is not None and not entry.spec.declare_only
    ]


def wired_field(cls: type, name: str) -> WiredField | None:
    """The :class:`WiredField` placeholder *cls* exposes under *name*, or ``None``.

    Walks ``cls.__mro__`` through each class's ``__dict__`` without invoking
    any descriptor (so it also works for annotated names, whose class-level
    access raises ``AttributeError``) and looks at the first binding of
    *name*.  The placeholder is returned only when that binding is a
    :class:`WiredField`, *name* is injected according to the merged
    ``__wiring__`` of *cls* (an entry that is neither excluded nor
    ``declare_only``), **and** *name* is not a constructor parameter of *cls*
    at call time (:func:`conscribe.introspect.init_param_names`).

    The last condition matters for dataclasses: a class annotating a wired
    name gets no placeholder of its own, but may inherit one, and becomes a
    dataclass field after the registration metaclass ran.  Such a receptor
    receives its value through the constructor; the inherited placeholder is
    not reported.

    Args:
        cls: Any class.
        name: Attribute name.

    Returns:
        The descriptor, or ``None`` when *cls* has no pending placeholder for
        *name*: nothing binds it, another binding shadows the placeholder,
        the name is not (or no longer) injected, or it is a constructor
        parameter.

    Raises:
        TypeError: If *cls* is not a class or *name* is not a str.
    """
    _require_class("wired_field", cls)
    if not isinstance(name, str):
        raise TypeError(
            f"wired_field() expects an attribute name (str), got {_type_name(name)} {name!r}"
        )
    value = _first_binding(cls, name)
    if not isinstance(value, WiredField) or name not in _injectable_names(cls):
        return None
    from conscribe.introspect import init_param_names

    if name in init_param_names(cls):
        return None
    return value


def wired_fields(cls: type) -> dict[str, WiredField]:
    """Every :class:`WiredField` placeholder *cls* exposes, as ``{name: descriptor}``.

    The mapping form of :func:`wired_field` (same rule: the first binding
    along the MRO must be the placeholder, the name must be injected, and
    constructor parameters are left out), in ``__wiring__`` declaration order
    (base classes first).

    Args:
        cls: Any class.

    Returns:
        A new dict; empty when the class exposes no placeholder.

    Raises:
        TypeError: If *cls* is not a class.
    """
    _require_class("wired_fields", cls)
    found: dict[str, WiredField] = {}
    params: frozenset[str] | None = None
    for name in _injectable_names(cls):
        value = _first_binding(cls, name)
        if not isinstance(value, WiredField):
            continue
        if params is None:
            from conscribe.introspect import init_param_names

            params = frozenset(init_param_names(cls))
        if name not in params:
            found[name] = value
    return found


def _inherited_binding_suffices(cls: type, name: str, registry_name: str) -> bool:
    """Whether the binding *cls* inherits for *name* makes a placeholder unnecessary.

    *cls* itself neither binds nor annotates *name*.

    Kept (``True``): plain values (an inherited default), writable
    descriptors (``__slots__`` members, properties with a setter, other data
    descriptors), ``functools.cached_property`` and other non-method
    descriptors, and a placeholder for the same registry.

    Shadowed (``False``): no binding at all, a read-only property, a method
    definition, a placeholder for another registry, a tombstone (an ancestor
    stopped wiring the name, *cls* wires it again).
    """
    value = _first_binding(cls, name)
    if value is _MISSING or isinstance(value, _WiringTombstone):
        return False
    if isinstance(value, WiredField):
        return value.registry_name == registry_name
    if isinstance(value, property):
        return value.fset is not None
    return not isinstance(value, _METHOD_TYPES)


def inject_wired_descriptors(cls: type) -> None:
    """Install :class:`WiredField` placeholders for the injected wired attributes of *cls*.

    Every registration path calls this at class creation (the registration
    metaclass, ``register()`` and the ``register(propagate=True)`` hook,
    also for classes they do not register) -- which is *before* class
    decorators such as ``@dataclass`` run.  Each merged ``__wiring__`` entry
    gets a placeholder unless:

    - the entry is ``declare_only`` -- nothing is injected for it;
    - the class body binds the name (a default, ``dataclasses.field(...)``,
      a property, a method, a hand-written placeholder) -- the class's own
      bindings are never shadowed;
    - the class body annotates the name (``env: Environment``) -- the class
      declares the slot itself; a placeholder in its own ``__dict__`` would
      become the default of an attrs field (an inherited placeholder still
      serves instance access, and its class access raises for this class,
      see :class:`WiredField`);
    - an inherited binding can hold or produce the value: a plain value (an
      inherited default), a writable descriptor (a ``__slots__`` member, a
      property with a setter), a ``functools.cached_property``, or a
      placeholder for the same registry;
    - the name is a constructor parameter
      (:func:`conscribe.introspect.init_param_names`: hand-written
      ``__init__`` parameters, parameters forwarded through ``**kwargs`` up
      the MRO, fields of an already processed dataclass, pydantic model
      fields) -- the value arrives through the constructor.  (Checked last:
      it is the expensive question, and most subclasses inherit their
      placeholders.)

    A placeholder does shadow an inherited binding that cannot receive the
    injected value, or that would hand out a wrong one before injection: a
    read-only property, a method (function, ``staticmethod``,
    ``classmethod``) -- typically a Protocol or mixin stub of the member --
    a placeholder for another registry (the subclass re-wired the name) and
    a tombstone (an ancestor stopped wiring the name).

    An entry the class excludes (``None``) or redeclares ``declare_only``
    gets a tombstone instead (see :class:`WiredField`) when the class would
    otherwise inherit a placeholder for it, unless the class binds or
    annotates the name itself.

    No registry is consulted, and a malformed entry is skipped rather than
    raised (:func:`check_wiring_declaration` reports it when the class
    registers).

    Args:
        cls: The newly created class.
    """
    merged, _ = _merge(cls)
    if not merged:
        return
    own = cls.__dict__
    annotated: Any = None
    params: frozenset[str] | None = None
    for name, entry in merged.items():
        if not isinstance(name, str) or name in own:
            continue  # own bindings are never shadowed
        spec = entry.spec
        if entry.excluded or (spec is not None and spec.declare_only):
            # No longer wired here: cover a placeholder the class inherits.
            if isinstance(_first_binding(cls, name), WiredField):
                if annotated is None:
                    annotated = _own_annotation_names(cls)
                if name not in annotated:
                    reason = (
                        "excludes the inherited __wiring__ entry with None"
                        if entry.excluded
                        else "redeclares the __wiring__ entry declare_only"
                    )
                    setattr(cls, name, _WiringTombstone(name, cls.__qualname__, reason))
            continue
        if spec is None:
            continue  # malformed: check_wiring_declaration reports it
        if annotated is None:
            annotated = _own_annotation_names(cls)
        if name in annotated:
            continue
        if _inherited_binding_suffices(cls, name, spec.registry_name):
            continue
        if params is None:
            from conscribe.introspect import init_param_names

            params = frozenset(init_param_names(cls))
        if name in params:
            continue
        setattr(cls, name, WiredField(name, spec.registry_name))
