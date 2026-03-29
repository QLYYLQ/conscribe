"""Cross-registry wiring resolution.

Reads ``__wiring__`` declarations from classes, merges them along the MRO,
and resolves registry references to concrete key lists for config generation.

Three grammar modes are supported::

    __wiring__ = {
        "loop": "agent_loop",                              # Mode 1: auto-discovery (all keys)
        "llm_provider": ("llm", ["openai", "anthropic"]),  # Mode 2: explicit subset
        "browser": ["chromium", "firefox"],                # Mode 3: literal list
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
    """
    param_name: str
    registry_name: str = ""
    allowed_keys: tuple[str, ...] | None = field(default=None)


@dataclass
class ResolvedWiring:
    """Result of resolving a single wiring entry to concrete keys.

    Attributes:
        param_name: The config field name.
        allowed_keys: Concrete list of allowed key strings.
        registry_name: Source registry name (``None`` for Mode 3 literal lists).
        injected: ``True`` if the param was NOT in ``__init__`` (field injection).
    """
    param_name: str
    allowed_keys: list[str]
    registry_name: str | None = None
    injected: bool = False


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
        elif isinstance(value, tuple) and len(value) == 2:
            # Mode 2: (registry_name, [key_subset])
            registry_name, key_subset = value
            if not isinstance(registry_name, str) or not isinstance(key_subset, list):
                raise TypeError(
                    f"Invalid __wiring__ entry for '{param_name}': "
                    f"tuple mode expects (str, list[str]), got ({type(registry_name).__name__}, {type(key_subset).__name__})"
                )
            specs.append(WiringSpec(
                param_name=param_name,
                registry_name=registry_name,
                allowed_keys=tuple(key_subset),
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
                f"expected str, (str, list), or list, got {type(value).__name__}"
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
                # Mode 2: validate subset
                missing = [k for k in spec.allowed_keys if k not in registry_keys]
                if missing:
                    raise WiringResolutionError(
                        cls_name=cls_name,
                        param_name=spec.param_name,
                        registry_name=spec.registry_name,
                        detail=(
                            f"Keys not found in '{spec.registry_name}' registry: "
                            f"{', '.join(sorted(missing))}. "
                            f"Available: {', '.join(sorted(registry_keys))}."
                        ),
                    )
                allowed = list(spec.allowed_keys)
            else:
                # Mode 1: all keys
                allowed = sorted(registry_keys)

            result[spec.param_name] = ResolvedWiring(
                param_name=spec.param_name,
                allowed_keys=allowed,
                registry_name=spec.registry_name,
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
