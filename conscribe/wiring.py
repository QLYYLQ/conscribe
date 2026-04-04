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
"""
from __future__ import annotations

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


def resolve_wiring(cls: type) -> dict[str, ResolvedWiring]:
    """Resolve a class's ``__wiring__`` to concrete key lists.

    For Mode 1 (registry name string), looks up the registry and uses all its keys.
    For Mode 2 (tuple), validates that each key exists in the referenced registry.
    For Mode 3 (literal list), returns the list as-is.

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
                # Mode 2: validate required subset
                missing = [k for k in spec.allowed_keys if k not in registry_keys]
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
                    missing_opt = [k for k in spec.optional_keys if k not in registry_keys]
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
                    optional_resolved = list(spec.optional_keys)
                    # Combined: required + optional for Literal type generation
                    allowed = list(spec.allowed_keys) + list(spec.optional_keys)
                else:
                    allowed = list(spec.allowed_keys)
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
