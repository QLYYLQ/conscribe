"""Tests for layer_registry.registration.key_transform module.

Covers:
- _camel_to_snake: parametrized CamelCase -> snake_case conversion
- default_key_transform: default strategy (delegates to _camel_to_snake)
- make_key_transform: factory with suffix/prefix stripping
- Real scenario tests modeling browseruse-bench layer naming
- Edge cases: empty string, single char, suffix==class_name, etc.
"""
from __future__ import annotations

import pytest

from layer_registry.registration.key_transform import (
    _camel_to_snake,
    default_key_transform,
    make_key_transform,
)


# ===================================================================
# _camel_to_snake -- parametrized tests
# ===================================================================

class TestCamelToSnake:
    """Parametrized tests for the core CamelCase -> snake_case converter."""

    @pytest.mark.parametrize(
        "input_name, expected",
        [
            ("BrowserUseAgent", "browser_use_agent"),
            ("ChatOpenAI", "chat_open_ai"),
            ("HTTPSHandler", "https_handler"),
            ("DOM", "dom"),
            ("MyV2Agent", "my_v2_agent"),
            ("XMLParser", "xml_parser"),
            ("S3Bucket", "s3_bucket"),
            ("getHTTPResponse", "get_http_response"),
            ("Agent", "agent"),
            ("A", "a"),
            ("ABCDef", "abc_def"),
        ],
        ids=[
            "multi_word_camel",
            "trailing_acronym",
            "leading_acronym",
            "all_caps_short",
            "digit_in_middle",
            "xml_prefix",
            "digit_prefix",
            "camelCase_lower_start",
            "single_word",
            "single_char",
            "acronym_then_word",
        ],
    )
    def test_camel_to_snake(self, input_name: str, expected: str) -> None:
        assert _camel_to_snake(input_name) == expected

    def test_already_snake_case(self) -> None:
        """Input already in snake_case should pass through unchanged."""
        assert _camel_to_snake("already_snake") == "already_snake"

    def test_lowercase_single_word(self) -> None:
        """A single lowercase word with no boundaries."""
        assert _camel_to_snake("agent") == "agent"

    def test_all_uppercase_long(self) -> None:
        """All-uppercase multi-char abbreviation."""
        assert _camel_to_snake("HTML") == "html"

    def test_trailing_digits(self) -> None:
        """Class name ending with digits."""
        assert _camel_to_snake("AgentV2") == "agent_v2"

    def test_consecutive_digits_and_upper(self) -> None:
        """Mixed digits and uppercase letters."""
        assert _camel_to_snake("My2ndAgent") == "my2nd_agent"


# ===================================================================
# default_key_transform
# ===================================================================

class TestDefaultKeyTransform:
    """Tests for the default key transform (pure snake_case, no stripping)."""

    def test_delegates_to_camel_to_snake(self) -> None:
        """default_key_transform should produce the same result as _camel_to_snake."""
        assert default_key_transform("BrowserUseAgent") == "browser_use_agent"

    def test_single_word(self) -> None:
        assert default_key_transform("Agent") == "agent"

    def test_acronym(self) -> None:
        assert default_key_transform("HTTPSHandler") == "https_handler"

    def test_is_callable(self) -> None:
        """default_key_transform satisfies KeyTransform protocol (callable)."""
        assert callable(default_key_transform)


# ===================================================================
# make_key_transform -- suffix stripping
# ===================================================================

class TestMakeKeyTransformSuffix:
    """Tests for make_key_transform with suffix stripping."""

    def test_strip_single_suffix(self) -> None:
        kt = make_key_transform(suffixes=["Agent"])
        assert kt("BrowserUseAgent") == "browser_use"

    def test_strip_first_matching_suffix(self) -> None:
        """When multiple suffixes provided, the first match wins."""
        kt = make_key_transform(suffixes=["Agent", "Handler"])
        assert kt("BrowserUseAgent") == "browser_use"

    def test_second_suffix_matches(self) -> None:
        """If the first suffix doesn't match, try the second."""
        kt = make_key_transform(suffixes=["Agent", "Handler"])
        assert kt("HTTPSHandler") == "https"

    def test_no_suffix_match_returns_full_snake(self) -> None:
        """If no suffix matches, return full snake_case of class name."""
        kt = make_key_transform(suffixes=["Agent"])
        assert kt("HTTPSHandler") == "https_handler"

    def test_suffix_equals_class_name_no_strip(self) -> None:
        """If suffix == class_name, stripping would produce empty -> don't strip."""
        kt = make_key_transform(suffixes=["Agent"])
        assert kt("Agent") == "agent"

    def test_suffix_case_sensitive(self) -> None:
        """Suffix matching is case-sensitive."""
        kt = make_key_transform(suffixes=["agent"])
        # "BrowserUseAgent" does NOT end with "agent" (lowercase)
        assert kt("BrowserUseAgent") == "browser_use_agent"


# ===================================================================
# make_key_transform -- prefix stripping
# ===================================================================

class TestMakeKeyTransformPrefix:
    """Tests for make_key_transform with prefix stripping."""

    def test_strip_single_prefix(self) -> None:
        kt = make_key_transform(prefixes=["Base"])
        assert kt("BaseHandler") == "handler"

    def test_prefix_equals_class_name_no_strip(self) -> None:
        """If prefix == class_name, stripping would produce empty -> don't strip."""
        kt = make_key_transform(prefixes=["Base"])
        assert kt("Base") == "base"

    def test_no_prefix_match(self) -> None:
        kt = make_key_transform(prefixes=["Base"])
        assert kt("ConcreteHandler") == "concrete_handler"


# ===================================================================
# make_key_transform -- both suffix AND prefix
# ===================================================================

class TestMakeKeyTransformCombined:
    """Tests for combined suffix + prefix stripping (both active)."""

    def test_strip_both(self) -> None:
        """Both suffix and prefix stripped: BaseBrowserUseAgent -> browser_use."""
        kt = make_key_transform(suffixes=["Agent"], prefixes=["Base"])
        assert kt("BaseBrowserUseAgent") == "browser_use"

    def test_suffix_then_prefix_order(self) -> None:
        """Stripping order: suffix first, then prefix."""
        kt = make_key_transform(suffixes=["Agent"], prefixes=["Base"])
        # "BaseAgent" -> strip "Agent" -> "Base" -> strip "Base" -> empty -> don't strip "Base"
        # So result is "base"
        assert kt("BaseAgent") == "base"

    def test_only_suffix_matches(self) -> None:
        """Prefix doesn't match, only suffix stripped."""
        kt = make_key_transform(suffixes=["Agent"], prefixes=["Abstract"])
        assert kt("BrowserUseAgent") == "browser_use"

    def test_only_prefix_matches(self) -> None:
        """Suffix doesn't match, only prefix stripped."""
        kt = make_key_transform(suffixes=["Handler"], prefixes=["Base"])
        assert kt("BaseAgent") == "agent"


# ===================================================================
# Real scenario tests -- browseruse-bench layer naming
# ===================================================================

class TestRealScenarioKeyTransforms:
    """Simulate key transforms Alice would configure for browseruse-bench layers."""

    def test_agent_layer_key_transform(self) -> None:
        """Alice configures strip_suffixes=["Agent"] for Agent layer."""
        kt = make_key_transform(suffixes=["Agent"])
        assert kt("BrowserUseAgent") == "browser_use"
        assert kt("SkyvernAgent") == "skyvern"
        assert kt("AgentTarsAgent") == "agent_tars"

    def test_llm_layer_key_transform(self) -> None:
        """Alice configures strip_suffixes=["LLM", "Provider"] for LLM layer."""
        kt = make_key_transform(suffixes=["LLM", "Provider"])
        assert kt("OpenAIProvider") == "open_ai"
        assert kt("AnthropicLLM") == "anthropic"
        assert kt("DeepSeekProvider") == "deep_seek"

    def test_browser_layer_key_transform(self) -> None:
        """Browser Provider layer with strip_suffixes=["Provider", "Browser"]."""
        kt = make_key_transform(suffixes=["Provider", "Browser"])
        assert kt("LexmountProvider") == "lexmount"
        assert kt("AgentbayBrowser") == "agentbay"

    def test_dom_filter_layer_key_transform(self) -> None:
        """DOM Filter layer with strip_suffixes=["Filter"]."""
        kt = make_key_transform(suffixes=["Filter"])
        assert kt("VisibilityFilter") == "visibility"
        assert kt("InteractivityFilter") == "interactivity"

    def test_make_key_transform_returns_callable(self) -> None:
        """The returned transform should be callable (satisfies KeyTransform protocol)."""
        kt = make_key_transform(suffixes=["Agent"])
        assert callable(kt)
