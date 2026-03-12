"""Backward-compatibility shim — discover() has moved to conscribe.discover."""
from __future__ import annotations

from conscribe.discover import discover  # noqa: F401

__all__ = ["discover"]
