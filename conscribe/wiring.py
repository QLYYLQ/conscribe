"""Cross-registry wiring resolution.

Reads ``__wiring__`` declarations from classes, merges them along the MRO,
and resolves registry references to concrete key lists for config generation.

Three grammar modes are supported::

    __wiring__ = {
        "loop": "agent_loop",                                    # Mode 1: auto-discovery (all keys)
        "llm_provider": ("llm", ["openai", "anthropic"]),        # Mode 2: explicit subset
        "obs": ("observation", ["terminal"], ["filesystem"]),     # Mode 2: required + optional
        "browser": ["chromium", "firefox"],                      # Mode 3: literal list
    }

A ``None`` value excludes an inherited key::

    __wiring__ = {"llm": None}  # remove parent's llm wiring

Capability-relative keys
------------------------

When the target registry was created with a ``key_separator`` (e.g.
``create_registrar(..., key_separator=".")``), the explicit key lists of
Mode 2 may name a **trailing segment** instead of a fully-qualified key::

    __wiring__ = {"action": ("action", ["click"], ["scroll"])}

``resolve_wiring`` expands ``"click"`` into every registered key whose
trailing segment is ``click`` (``browser.click``, ``desktop.click``, ...).
This is *not* a fourth grammar mode — it is a relaxation of how the key
strings inside Mode 2 are matched.  A key that already contains the
separator is passed through unchanged, which is the escape hatch for
pinning one specific provider.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WiringSpec:
    """Normalized representation of a single ``__wiring__`` entry.

    Attributes:
        param_name: The config field / ``__init__`` parameter name.
        registry_name: Target registry name (empty string for Mode 3).
        allowed_keys: Explicit key subset (``None`` = auto-discover all keys).
            For 3-element tuple mode, these are the *required* keys.
        optional_keys: Optional key subset for 3-element tuple mode
            (``None`` for all other modes).
    """
    param_name: str
    registry_name: str = ""
    allowed_keys: tuple[str, ...] | None = field(default=None)
    optional_keys: tuple[str, ...] | None = field(default=None)


@dataclass
class ResolvedWiring:
    """Result of resolving a single wiring entry to concrete keys.

    Attributes:
        param_name: The config field name.
        allowed_keys: Concrete list of allowed key strings.  When
            ``optional_keys`` is set, this contains required + optional
            combined (for backward-compatible ``Literal[...]`` generation).
        registry_name: Source registry name (``None`` for Mode 3 literal lists).
        injected: ``True`` if the param was NOT in ``__init__`` (field injection).
        optional_keys: Optional key subset (``None`` when not using
            3-element tuple mode).
    """
    param_name: str
    allowed_keys: list[str]
    registry_name: str | None = None
    injected: bool = False
    optional_keys: list[str] | None = None


def collect_wiring_from_mro(cls: type) -> dict[str, Any]:
    """Walk MRO bottom-up and deep-merge ``__wiring__`` dicts.

    Child entries override parent entries for the same key.
    A value of ``None`` excludes the key entirely (even if a parent defines it).

    Args:
        cls: The class to collect wiring for.

    Returns:
        Merged wiring dict.  Keys with ``None`` values are removed.
    """
    merged: dict[str, Any] = {}

    # Walk MRO from most distant ancestor to cls (reverse order),
    # so child entries naturally override parent entries.
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        # Use __dict__ to get only the class's own __wiring__ (not inherited)
        wiring = klass.__dict__.get("__wiring__")
        if wiring is not None and isinstance(wiring, dict):
            merged.update(wiring)

    # Remove None-excluded keys
    return {k: v for k, v in merged.items() if v is not None}


def parse_wiring(cls: type) -> list[WiringSpec]:
    """Parse a class's merged ``__wiring__`` into normalized ``WiringSpec`` list.

    Args:
        cls: The class to parse wiring for.

    Returns:
        List of ``WiringSpec`` instances.  Empty list if no wiring declared.
    """
    raw = collect_wiring_from_mro(cls)
    if not raw:
        return []

    specs: list[WiringSpec] = []
    for param_name, value in raw.items():
        if isinstance(value, str):
            # Mode 1: all keys from registry
            specs.append(WiringSpec(param_name=param_name, registry_name=value))
        elif isinstance(value, tuple) and len(value) in (2, 3):
            # Mode 2: (registry_name, [key_subset]) or
            #          (registry_name, [required_keys], [optional_keys])
            registry_name = value[0]
            required_keys = value[1]
            optional_keys_raw = value[2] if len(value) == 3 else None

            if not isinstance(registry_name, str) or not isinstance(required_keys, list):
                type_strs = ", ".join(type(v).__name__ for v in value)
                raise TypeError(
                    f"Invalid __wiring__ entry for '{param_name}': "
                    f"tuple mode expects (str, list[str]) or (str, list[str], list[str]), "
                    f"got ({type_strs})"
                )
            if optional_keys_raw is not None and not isinstance(optional_keys_raw, list):
                raise TypeError(
                    f"Invalid __wiring__ entry for '{param_name}': "
                    f"third element of tuple must be list[str], "
                    f"got {type(optional_keys_raw).__name__}"
                )

            specs.append(WiringSpec(
                param_name=param_name,
                registry_name=registry_name,
                allowed_keys=tuple(required_keys),
                optional_keys=tuple(optional_keys_raw) if optional_keys_raw is not None else None,
            ))
        elif isinstance(value, list):
            # Mode 3: literal list (no registry reference)
            specs.append(WiringSpec(
                param_name=param_name,
                registry_name="",
                allowed_keys=tuple(value),
            ))
        else:
            raise TypeError(
                f"Invalid __wiring__ entry for '{param_name}': "
                f"expected str, (str, list), (str, list, list), or list, "
                f"got {type(value).__name__}"
            )

    return specs


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
    the expansion exists to provide.

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

    expanded: list[str] = []
    seen: set[str] = set()
    for key in declared:
        if separator in key:
            # Already fully qualified — pass through untouched.
            candidates = [key]
        else:
            candidates = sorted(
                k for k in registry_keys if k.rsplit(separator, 1)[-1] == key
            )
            if not candidates:
                # Unknown short name: keep it so the caller's missing-key
                # check produces the normal error message.
                candidates = [key]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def resolve_wiring(cls: type) -> dict[str, ResolvedWiring]:
    """Resolve a class's ``__wiring__`` to concrete key lists.

    For Mode 1 (registry name string), looks up the registry and uses all its keys.
    For Mode 2 (tuple), validates that each key exists in the referenced registry.
    For Mode 3 (literal list), returns the list as-is.

    Mode 2 key lists additionally support **capability-relative keys** when
    the target registry declares a ``key_separator``: a bare trailing
    segment (``"click"``) expands to every matching fully-qualified key
    (``browser.click``, ``desktop.click``).  See
    :func:`expand_relative_keys` for the expansion rules and for why an
    ambiguous short name widens the ``Literal`` instead of raising.

    Args:
        cls: The class whose wiring to resolve.

    Returns:
        Dict mapping param names to ``ResolvedWiring``.  Empty dict if no wiring.

    Raises:
        WiringResolutionError: If a referenced registry is not found, is empty,
            or an explicit key is not present in the registry.
    """
    from conscribe.exceptions import WiringResolutionError
    from conscribe.registration.registry import get_registry

    specs = parse_wiring(cls)
    if not specs:
        return {}

    cls_name = cls.__qualname__

    result: dict[str, ResolvedWiring] = {}
    for spec in specs:
        if spec.registry_name:
            # Mode 1 or Mode 2: resolve from registry
            registry = get_registry(spec.registry_name)
            if registry is None:
                raise WiringResolutionError(
                    cls_name=cls_name,
                    param_name=spec.param_name,
                    registry_name=spec.registry_name,
                    detail=(
                        f"Registry '{spec.registry_name}' not found. "
                        f"Available registries can be checked after all "
                        f"create_registrar() calls have executed."
                    ),
                )

            registry_keys = registry.keys()
            if not registry_keys:
                raise WiringResolutionError(
                    cls_name=cls_name,
                    param_name=spec.param_name,
                    registry_name=spec.registry_name,
                    detail=(
                        f"Registry '{spec.registry_name}' is empty. "
                        f"Did you forget to call discover() or import "
                        f"the modules containing implementations?"
                    ),
                )

            if spec.allowed_keys is not None:
                # Mode 2: expand capability-relative keys, then validate.
                separator = getattr(registry, "separator", "")
                required_keys = expand_relative_keys(
                    spec.allowed_keys, registry_keys, separator
                )

                missing = [k for k in required_keys if k not in registry_keys]
                if missing:
                    raise WiringResolutionError(
                        cls_name=cls_name,
                        param_name=spec.param_name,
                        registry_name=spec.registry_name,
                        detail=(
                            f"Required keys not found in '{spec.registry_name}' registry: "
                            f"{', '.join(sorted(missing))}. "
                            f"Available: {', '.join(sorted(registry_keys))}."
                        ),
                    )

                # Validate optional keys if present
                optional_resolved: list[str] | None = None
                if spec.optional_keys is not None:
                    optional_keys = expand_relative_keys(
                        spec.optional_keys, registry_keys, separator
                    )
                    missing_opt = [k for k in optional_keys if k not in registry_keys]
                    if missing_opt:
                        raise WiringResolutionError(
                            cls_name=cls_name,
                            param_name=spec.param_name,
                            registry_name=spec.registry_name,
                            detail=(
                                f"Optional keys not found in '{spec.registry_name}' registry: "
                                f"{', '.join(sorted(missing_opt))}. "
                                f"Available: {', '.join(sorted(registry_keys))}."
                            ),
                        )
                    optional_resolved = optional_keys
                    # Combined: required + optional for Literal type generation
                    allowed = required_keys + [
                        k for k in optional_keys if k not in required_keys
                    ]
                else:
                    allowed = list(required_keys)
            else:
                # Mode 1: all keys
                allowed = sorted(registry_keys)
                optional_resolved = None

            result[spec.param_name] = ResolvedWiring(
                param_name=spec.param_name,
                allowed_keys=allowed,
                registry_name=spec.registry_name,
                optional_keys=optional_resolved,
            )
        else:
            # Mode 3: literal list
            if spec.allowed_keys is None or not spec.allowed_keys:
                raise WiringResolutionError(
                    cls_name=cls_name,
                    param_name=spec.param_name,
                    registry_name="",
                    detail="Literal list mode requires a non-empty list of keys.",
                )
            result[spec.param_name] = ResolvedWiring(
                param_name=spec.param_name,
                allowed_keys=list(spec.allowed_keys),
                registry_name=None,
            )

    return result


# ── WiredField descriptor ──────────────────────────────────────


class WiredField:
    """Non-data descriptor placeholder for wired attributes not in ``__init__``.

    Set automatically by the metaclass / ``register()`` decorator when a
    class declares ``__wiring__`` fields that are not ``__init__`` parameters.

    Before injection (framework has not set the attribute)::

        agent.env          # raises AttributeError with clear message
        MyAgent.env        # returns the WiredField descriptor itself

    After injection (framework sets instance attribute)::

        agent.env = my_env
        agent.env          # returns my_env (instance attr shadows descriptor)
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
            return self
        msg = f"Wired attribute '{self.name}' on {type(obj).__name__} has not been injected yet."
        if self.registry_name:
            msg += f" Expected injection from registry '{self.registry_name}'."
        raise AttributeError(msg)


def inject_wired_descriptors(cls: type) -> None:
    """Set :class:`WiredField` descriptors for injected wired attributes.

    Called at class creation time (metaclass ``__new__``, ``register()``
    decorator, or ``__init_subclass__`` propagation hook).

    Only sets descriptors for wired fields that are **not** in the class's
    ``__init__`` signature.  Uses :func:`collect_wiring_from_mro` (no
    registry resolution needed, so no timing issues with unpopulated
    registries).

    Args:
        cls: The newly created class.
    """
    merged = collect_wiring_from_mro(cls)
    if not merged:
        return

    # Collect __init__ param names (including inherited)
    init_param_names: set[str] = set()
    try:
        sig = inspect.signature(cls.__init__)  # type: ignore[misc]
        init_param_names = {
            p.name
            for p in sig.parameters.values()
            if p.name != "self"
        }
    except (ValueError, TypeError):
        pass

    for param_name, raw_value in merged.items():
        if param_name in init_param_names:
            continue  # already in __init__, not injected

        # Skip if a WiredField already exists (own or inherited from parent)
        if isinstance(getattr(cls, param_name, None), WiredField):
            continue

        # Extract registry name from raw wiring value
        registry_name = _extract_registry_name(raw_value)
        setattr(cls, param_name, WiredField(param_name, registry_name))


def _extract_registry_name(raw_value: Any) -> str:
    """Extract registry name from a raw ``__wiring__`` value."""
    if isinstance(raw_value, str):
        return raw_value
    if isinstance(raw_value, tuple) and len(raw_value) >= 2 and isinstance(raw_value[0], str):
        return raw_value[0]
    return ""
