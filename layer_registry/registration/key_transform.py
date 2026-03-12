"""Key inference: transform class names into registry keys.

Provides CamelCase-to-snake_case conversion with optional
suffix/prefix stripping. All transforms conform to the KeyTransform protocol.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

# Pre-compiled regex at module level for performance.
# Step 1: Insert '_' between consecutive uppercase and uppercase+lowercase.
#   e.g. "HTTPSHandler" -> "HTTPS_Handler"
_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
# Step 2: Insert '_' between lowercase/digit and uppercase.
#   e.g. "browserUse" -> "browser_Use"
_RE2 = re.compile(r"([a-z\d])([A-Z])")


@runtime_checkable
class KeyTransform(Protocol):
    """Callable protocol: class name -> registry key."""

    def __call__(self, class_name: str) -> str: ...


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case without any stripping.

    Examples:
        "BrowserUseAgent" -> "browser_use_agent"
        "ChatOpenAI"      -> "chat_open_ai"
        "HTTPSHandler"    -> "https_handler"
        "DOM"             -> "dom"
        "MyV2Agent"       -> "my_v2_agent"
    """
    result = _RE1.sub(r"\1_\2", name)
    result = _RE2.sub(r"\1_\2", result)
    return result.lower()


def default_key_transform(class_name: str) -> str:
    """Default key inference: pure CamelCase -> snake_case, no stripping.

    Used by create_registrar() when no key inference parameters are specified.
    """
    return _camel_to_snake(class_name)


def make_key_transform(
    *,
    suffixes: Optional[list[str]] = None,
    prefixes: Optional[list[str]] = None,
) -> KeyTransform:
    """Factory: create a KeyTransform with suffix/prefix stripping.

    Args:
        suffixes: Suffixes to strip from the end of the class name.
            Tried in order; first match wins.
        prefixes: Prefixes to strip from the start of the class name.
            Tried in order; first match wins.

    Returns:
        A callable satisfying the KeyTransform protocol.

    Notes:
        - Suffix stripping happens first, then prefix stripping. Both apply.
        - If stripping would result in an empty string, it is NOT stripped.
    """
    _suffixes = suffixes or []
    _prefixes = prefixes or []

    def _transform(class_name: str) -> str:
        name = class_name

        # Step 1: strip suffix (first match wins)
        for suffix in _suffixes:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                break

        # Step 2: strip prefix (first match wins)
        for prefix in _prefixes:
            if name.startswith(prefix) and len(name) > len(prefix):
                name = name[len(prefix) :]
                break

        # Step 3: CamelCase -> snake_case
        return _camel_to_snake(name)

    return _transform  # type: ignore[return-value]
