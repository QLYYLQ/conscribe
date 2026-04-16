"""Browser tools — hierarchical keys: browser.click, browser.navigate."""
from __future__ import annotations

from mock_framework.layers import ToolRegistrar


class BaseTool(metaclass=ToolRegistrar.Meta):
    __abstract__ = True

    def execute(self, **kwargs: object) -> str:
        return ""


class BrowserClickTool(BaseTool):
    """Click an element on the page.

    Args:
        selector: CSS selector to click.
    """
    __registry_key__ = "browser.click"

    def __init__(self, *, selector: str = ""):
        self.selector = selector

    def execute(self, **kwargs: object) -> str:
        return f"clicked {self.selector}"


class BrowserNavigateTool(BaseTool):
    """Navigate to a URL.

    Args:
        url: Target URL.
    """
    __registry_key__ = "browser.navigate"

    def __init__(self, *, url: str = ""):
        self.url = url

    def execute(self, **kwargs: object) -> str:
        return f"navigated to {self.url}"
