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

    for key in sorted(all_classes.keys()):
        cls = all_classes[key]
        hasher.update(key.encode("utf-8"))

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
        else:
            hasher.update(b"<no-init>")

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


def _find_init_definer(cls: type) -> Union[type, None]:
    """Walk MRO to find the class that actually defines ``__init__``."""
    from conscribe.config._utils import find_init_definer
    return find_init_definer(cls)
