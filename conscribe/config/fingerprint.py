"""Registry fingerprinting for staleness detection.

Computes a SHA-256 based fingerprint of a registrar's contents
(keys, class qualnames, signatures, docstrings) and provides
JSON-file caching for comparing against previously generated stubs.

See ``config-typing-design.md`` Section 7 for specification.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Union


def compute_registry_fingerprint(registrar: type) -> str:
    """Compute a fingerprint of all registered classes in a registrar.

    The fingerprint captures the "shape" of all registered implementations:
    - Sorted registry keys
    - Each class's qualified name
    - Each class's ``__init__`` signature (via ``inspect.signature``)
    - Each class's docstring (affects Tier 1.5 description extraction)

    Args:
        registrar: A ``LayerRegistrar`` subclass.

    Returns:
        A 16-character lowercase hex string (first 16 chars of SHA-256).
    """
    hasher = hashlib.sha256()
    all_classes = registrar.get_all()

    _BM = _get_base_model()

    for key in sorted(all_classes.keys()):
        cls = all_classes[key]
        hasher.update(key.encode("utf-8"))

        # BaseModel fast path: hash model_fields instead of __init__
        if (
            _BM is not None
            and isinstance(cls, type)
            and issubclass(cls, _BM)
            and cls is not _BM
            and "__init__" not in cls.__dict__
        ):
            for fname in sorted(cls.model_fields.keys()):
                fi = cls.model_fields[fname]
                hasher.update(
                    f"pydantic:{fname}:{fi.annotation}:{fi.default}".encode("utf-8")
                )
            doc = cls.__doc__ or ""
            hasher.update(doc.encode("utf-8"))
            continue

        # Signature
        init_definer = _find_init_definer(cls)
        if init_definer is not None:
            try:
                sig = inspect.signature(init_definer.__init__)
                hasher.update(str(sig).encode("utf-8"))
            except (ValueError, TypeError):
                hasher.update(b"<no-signature>")

            # Docstring (class-level first, then __init__)
            doc = init_definer.__doc__ or ""
            init_doc = getattr(init_definer.__init__, "__doc__", None) or ""
            hasher.update(doc.encode("utf-8"))
            hasher.update(init_doc.encode("utf-8"))

            # MRO parent signatures: when **kwargs is present,
            # parent signatures affect the generated schema.
            _hash_mro_parent_signatures(hasher, cls, init_definer)
        else:
            hasher.update(b"<no-init>")

        # Hash __wiring__ declarations so that changes in referenced
        # registries trigger fingerprint invalidation.
        _hash_wiring(hasher, cls)

    return hasher.hexdigest()[:16]


def load_cached_fingerprint(
    fingerprint_path: Path,
    layer_name: str,
) -> Union[str, None]:
    """Load a cached fingerprint from a JSON file.

    The JSON file stores ``{layer_name: fingerprint_hex}`` mappings.

    Args:
        fingerprint_path: Path to the JSON cache file.
        layer_name: The layer name to look up.

    Returns:
        The cached fingerprint string, or ``None`` if the file
        doesn't exist, the layer is not found, or the file is corrupt.
    """
    if not fingerprint_path.exists():
        return None
    try:
        data = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data.get(layer_name)
    except (json.JSONDecodeError, OSError):
        return None


def save_fingerprint(
    fingerprint_path: Path,
    layer_name: str,
    fingerprint: str,
) -> None:
    """Save a fingerprint to a JSON cache file, preserving other layers.

    Note: This is not atomic — concurrent writes from separate processes
    could cause a race condition (read-modify-write). Acceptable for a
    CLI build tool. For atomic writes, use tempfile + os.replace().

    Args:
        fingerprint_path: Path to the JSON cache file.
        layer_name: The layer name key.
        fingerprint: The fingerprint hex string to store.
    """
    # Load existing data if file exists
    data: dict[str, Any] = {}
    if fingerprint_path.exists():
        try:
            data = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, OSError):
            data = {}

    data[layer_name] = fingerprint

    # Ensure parent directories exist
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_base_model() -> Union[type, None]:
    """Lazily import pydantic.BaseModel, returning None if unavailable."""
    try:
        from pydantic import BaseModel
        return BaseModel
    except ImportError:
        return None


def _find_init_definer(cls: type) -> Union[type, None]:
    """Walk MRO to find the class that actually defines ``__init__``."""
    from conscribe.config._utils import find_init_definer
    return find_init_definer(cls)


def _hash_wiring(hasher: hashlib._Hash, cls: type) -> None:
    """Hash ``__wiring__`` resolved keys into the fingerprint.

    This ensures that when a referenced registry changes (e.g., a new
    class is registered in the "agent_loop" registry), the fingerprint
    of the referencing registry invalidates, triggering stub regeneration.
    """
    from conscribe.exceptions import WiringResolutionError

    try:
        from conscribe.wiring import resolve_wiring
        resolved = resolve_wiring(cls)
    except (WiringResolutionError, TypeError):
        # Wiring resolution can fail if referenced registries aren't
        # populated yet. In that case, hash the raw __wiring__ dict
        # to at least detect changes in the declaration itself.
        from conscribe.wiring import collect_wiring_from_mro
        raw = collect_wiring_from_mro(cls)
        if raw:
            hasher.update(f"wiring_raw:{sorted(raw.items())}".encode("utf-8"))
        return

    for param_name in sorted(resolved.keys()):
        wiring = resolved[param_name]
        hasher.update(
            f"wiring:{param_name}:{sorted(wiring.allowed_keys)}".encode("utf-8")
        )


def _hash_mro_parent_signatures(
    hasher: hashlib._Hash,
    cls: type,
    init_definer: type,
) -> None:
    """Hash parent class signatures when ``init_definer`` has ``**kwargs``.

    This ensures that changes to parent ``__init__`` signatures cause
    fingerprint invalidation when the child uses ``**kwargs`` forwarding.
    """
    init = init_definer.__dict__.get("__init__")
    if init is None:
        return

    try:
        sig = inspect.signature(init)
    except (ValueError, TypeError):
        return

    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_kw:
        return

    # Walk MRO from init_definer upward
    mro = cls.__mro__
    try:
        start_idx = mro.index(init_definer)
    except ValueError:
        return

    for klass in mro[start_idx + 1 :]:
        if klass is object:
            break
        if "__init__" not in klass.__dict__:
            continue

        try:
            parent_sig = inspect.signature(klass.__init__)
            hasher.update(f"mro:{klass.__qualname__}:{parent_sig}".encode("utf-8"))
        except (ValueError, TypeError):
            hasher.update(f"mro:{klass.__qualname__}:<no-sig>".encode("utf-8"))

        # Hash parent docstrings too
        parent_doc = klass.__doc__ or ""
        parent_init_doc = getattr(klass.__init__, "__doc__", None) or ""
        hasher.update(parent_doc.encode("utf-8"))
        hasher.update(parent_init_doc.encode("utf-8"))

        # Check if this parent also has **kwargs; if not, stop
        try:
            parent_sig = inspect.signature(klass.__init__)
            parent_has_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in parent_sig.parameters.values()
            )
            if not parent_has_kw:
                break
        except (ValueError, TypeError):
            break
