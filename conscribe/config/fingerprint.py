"""Registry fingerprinting for staleness detection.

:func:`compute_registry_fingerprint` condenses what conscribe *generates* for
a layer into a 16-character hex digest.  Store it next to the generated files
(:func:`save_fingerprint` / :func:`load_cached_fingerprint`) and compare it
later: a different value means that regenerating would change the files.

What is hashed
--------------
SHA-256 over a canonical JSON document (sorted keys, ASCII) holding:

* ``conscribe.__version__``;
* the registrar settings that only show in *composed* output: the layer
  name, the discriminator field (flat mode) or fields (nested mode), the key
  separator and ``inline_only``;
* the layer's generated config source,
  ``generate_layer_config_source(build_layer_config(registrar))``;
* the layer's JSON Schema, ``json.dumps(generate_layer_json_schema(result),
  sort_keys=True)``.

The fingerprint therefore changes exactly when regeneration would change the
outputs -- a field's type, default, description or constraint, the keys a
wired field allows (a target registry gaining a key widens its ``Literal``),
a wiring default key, ``declare_only``, the registered keys and their order
in the union, a conscribe upgrade -- and never for edits that do not reach
them (comments, a class docstring's summary, the body of a factory that the
generated file imports by reference).  It is identical across processes and
Python versions whenever the outputs are.

A composed build with inline wiring embeds the target layers' unions in the
owner's models; fingerprint every layer of the build to cover it.

Never raises
------------
A fingerprint must be computable in broken states (that is when generated
files most need to be flagged as stale).  Each part is computed on its own,
and an exception becomes the marker ``error:<ExceptionType>:<message>``
(object addresses ``0x...`` removed from the message), so a broken layer --
malformed ``__wiring__``, a typo in an annotation, an empty layer -- still has
a fingerprint, one that differs from the repaired layer's.  Warnings emitted
while generating (``DegradationWarning``, ``ExtractionWarning``, ...) are
suppressed: the outputs carry their own markers.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conscribe.config.builder import build_layer_config
from conscribe.config.codegen import generate_layer_config_source
from conscribe.config.json_schema import generate_layer_json_schema

#: Object addresses as CPython prints them: ``<X object at 0x10f3a2b40>``.
_ADDRESS = re.compile(r"0x[0-9A-Fa-f]+")

#: Number of hex characters kept from the SHA-256 digest.
_DIGEST_LENGTH = 16


def compute_registry_fingerprint(registrar: type) -> str:
    """Compute a fingerprint of a layer's generated config outputs.

    Hashes ``conscribe.__version__``, the layer name, discriminator field(s),
    key separator and ``inline_only`` of *registrar*, and the layer's
    generated config source and JSON Schema (see the module docstring).  The
    value changes exactly when regenerating the layer's files would change
    them; it does not depend on object addresses, hash seeds or the Python
    version beyond what the outputs themselves contain.

    Never raises: a layer that cannot be generated (wiring problems, an
    unresolvable annotation, no registered classes, ...) gets a fingerprint
    built from a stable error marker, which differs from the fingerprint of
    the repaired layer.  Warnings emitted while generating are suppressed.

    Args:
        registrar: A ``LayerRegistrar`` subclass (from ``create_registrar``).

    Returns:
        A 16-character lowercase hex string (first 16 chars of SHA-256).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        document = _fingerprint_document(registrar)
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:_DIGEST_LENGTH]


def load_cached_fingerprint(
    fingerprint_path: Path,
    layer_name: str,
) -> str | None:
    """Load a cached fingerprint from a JSON file.

    The JSON file stores ``{layer_name: fingerprint_hex}`` mappings.

    Args:
        fingerprint_path: Path to the JSON cache file.
        layer_name: The layer name to look up.

    Returns:
        The cached fingerprint string, or ``None`` if the file doesn't
        exist, the layer is not found (or not stored as a string), or the
        file is corrupt.
    """
    if not fingerprint_path.exists():
        return None
    try:
        data = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(layer_name)
    return value if isinstance(value, str) else None


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
            loaded = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            data = loaded

    data[layer_name] = fingerprint

    # Ensure parent directories exist
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The hashed document
# ---------------------------------------------------------------------------


def _fingerprint_document(registrar: Any) -> dict[str, Any]:
    """The inputs of the fingerprint, as plain JSON values.

    The generated source and schema are both error markers when the layer
    cannot be built; when only one of them fails, the other is still hashed.
    """
    document: dict[str, Any] = {
        "conscribe": _guarded(_conscribe_version),
        "layer": {
            "name": _guarded(lambda: registrar._registry.name),
            "discriminator_field": _guarded(lambda: registrar.discriminator_field),
            "discriminator_fields": _guarded(lambda: registrar.discriminator_fields),
            "key_separator": _guarded(lambda: registrar._key_separator),
            "inline_only": _guarded(lambda: registrar.inline_only),
        },
    }
    try:
        result = build_layer_config(registrar)
    except Exception as exc:  # noqa: BLE001 - a fingerprint must stay computable in broken states
        document["source"] = document["schema"] = _error_marker(exc)
        return document
    document["source"] = _guarded(lambda: generate_layer_config_source(result))
    document["schema"] = _guarded(
        lambda: json.dumps(generate_layer_json_schema(result), sort_keys=True)
    )
    return document


def _guarded(compute: Callable[[], Any]) -> Any:
    """``compute()`` as a plain JSON value, or the error marker of what it raised."""
    try:
        return _plain(compute())
    except Exception as exc:  # noqa: BLE001 - a fingerprint must stay computable in broken states
        return _error_marker(exc)


def _plain(value: Any) -> Any:
    """*value* as a JSON value that does not depend on the process."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return f"{type(value).__qualname__}:{_ADDRESS.sub('0x...', repr(value))}"


def _error_marker(exc: BaseException) -> str:
    """``error:<ExceptionType>:<message>``, object addresses removed."""
    try:
        message = str(exc)
    except Exception:  # noqa: BLE001 - a user __str__ may raise; the type still marks the state
        message = "<message unavailable>"
    return f"error:{type(exc).__qualname__}:{_ADDRESS.sub('0x...', message)}"


def _conscribe_version() -> str:
    import conscribe

    return conscribe.__version__
