"""Docstring parameter description parsing.

Extracts parameter descriptions from class or function docstrings
using the ``docstring_parser`` library (optional dependency).
Supports Google, NumPy, and reST docstring formats.

When ``docstring_parser`` is not installed, gracefully returns
an empty dict — no error, no side-effect.
"""
from __future__ import annotations

import inspect
from typing import Any, Union

try:
    import docstring_parser

    _HAS_DOCSTRING_PARSER = True
except ImportError:
    _HAS_DOCSTRING_PARSER = False


def parse_param_descriptions(
    cls_or_func: Union[type, Any],
) -> dict[str, str]:
    """Extract parameter descriptions from a class or function docstring.

    For classes, checks the class-level docstring first. If no parameter
    descriptions are found there, falls back to the ``__init__`` docstring.

    Args:
        cls_or_func: A class or function whose docstring to parse.

    Returns:
        Dict mapping parameter names to their description strings.
        Empty dict if no docstring, no Args section, or ``docstring_parser``
        is not installed.
    """
    if not _HAS_DOCSTRING_PARSER:
        return {}

    # For classes: try class docstring first, then __init__ docstring
    if inspect.isclass(cls_or_func):
        result = _parse_docstring(cls_or_func.__doc__)
        if result:
            return result
        # Fall back to __init__ docstring
        init_method = getattr(cls_or_func, "__init__", None)
        if init_method is not None:
            return _parse_docstring(init_method.__doc__)
        return {}

    # For functions / methods: parse their docstring directly
    return _parse_docstring(getattr(cls_or_func, "__doc__", None))


def _parse_docstring(docstring: Union[str, None]) -> dict[str, str]:
    """Parse a single docstring and return param descriptions.

    Args:
        docstring: Raw docstring text, or None.

    Returns:
        Dict of {param_name: description}. Empty if nothing found.
    """
    if not docstring:
        return {}

    parsed = docstring_parser.parse(docstring)
    result: dict[str, str] = {}
    for param in parsed.params:
        if param.description:
            result[param.arg_name] = param.description.strip()
    return result
