"""File tools — hierarchical keys: file.read, file.write."""
from __future__ import annotations

from mock_framework.tools.browser_tool import BaseTool


class FileReadTool(BaseTool):
    """Read a file.

    Args:
        encoding: File encoding.
    """
    __registry_key__ = "file.read"

    def __init__(self, *, encoding: str = "utf-8"):
        self.encoding = encoding

    def execute(self, **kwargs: object) -> str:
        return "read file"


class FileWriteTool(BaseTool):
    """Write a file.

    Args:
        encoding: File encoding.
        create_dirs: Whether to create parent directories.
    """
    __registry_key__ = "file.write"

    def __init__(self, *, encoding: str = "utf-8", create_dirs: bool = True):
        self.encoding = encoding
        self.create_dirs = create_dirs

    def execute(self, **kwargs: object) -> str:
        return "wrote file"
