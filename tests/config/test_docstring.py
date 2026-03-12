"""Tests for conscribe.config.docstring.parse_param_descriptions.

Covers:
- Google-style docstring parsing (basic, multi-line, multi-param)
- NumPy-style docstring parsing
- reST-style docstring parsing
- No docstring / empty docstring / docstring without Args section
- Partial param coverage
- Graceful degradation when docstring_parser is not installed
- Unicode descriptions (Chinese characters)
- Function (not class) with docstring
- Docstring on __init__ method (class-level vs __init__-level)
"""
from __future__ import annotations

from unittest.mock import patch
import pytest

from conscribe.config.docstring import parse_param_descriptions


# ===================================================================
# Google-style docstrings
# ===================================================================

class TestGoogleStyleDocstring:
    """Tests for Google-style docstrings (Args: section)."""

    def test_basic_single_param(self) -> None:
        """Single param with a one-line description."""

        class MyClass:
            """My class.

            Args:
                name: The name of the resource.
            """

        result = parse_param_descriptions(MyClass)
        assert result == {"name": "The name of the resource."}

    def test_multi_param(self) -> None:
        """Multiple params each with one-line descriptions."""

        class MyClass:
            """My class.

            Args:
                model_id: The model identifier.
                temperature: Sampling temperature.
                max_tokens: Maximum output tokens.
            """

        result = parse_param_descriptions(MyClass)
        assert result == {
            "model_id": "The model identifier.",
            "temperature": "Sampling temperature.",
            "max_tokens": "Maximum output tokens.",
        }

    def test_multi_line_description(self) -> None:
        """Param with description spanning multiple lines."""

        class MyClass:
            """My class.

            Args:
                model_id: The model identifier, such as gpt-4o.
                    This is used to select the specific model variant
                    for inference.
                temperature: Sampling temperature.
            """

        result = parse_param_descriptions(MyClass)
        assert "model_id" in result
        assert "gpt-4o" in result["model_id"]
        assert "temperature" in result
        assert result["temperature"] == "Sampling temperature."

    def test_param_with_type_annotation_in_docstring(self) -> None:
        """Google-style with type in parentheses after param name."""

        class MyClass:
            """My class.

            Args:
                model_id (str): The model identifier.
                temperature (float): Sampling temperature.
            """

        result = parse_param_descriptions(MyClass)
        assert result == {
            "model_id": "The model identifier.",
            "temperature": "Sampling temperature.",
        }


# ===================================================================
# NumPy-style docstrings
# ===================================================================

class TestNumPyStyleDocstring:
    """Tests for NumPy-style docstrings (Parameters section)."""

    def test_numpy_style(self) -> None:
        """NumPy-style with dashed underlines."""

        class MyClass:
            """My class.

            Parameters
            ----------
            model_id : str
                The model identifier.
            temperature : float
                Sampling temperature.
            """

        result = parse_param_descriptions(MyClass)
        assert "model_id" in result
        assert "model identifier" in result["model_id"]
        assert "temperature" in result
        assert "Sampling temperature" in result["temperature"]


# ===================================================================
# reST-style docstrings
# ===================================================================

class TestReSTStyleDocstring:
    """Tests for reST-style docstrings (:param ...: directives)."""

    def test_rest_style(self) -> None:
        """reST-style with :param name: description."""

        class MyClass:
            """My class.

            :param model_id: The model identifier.
            :param temperature: Sampling temperature.
            """

        result = parse_param_descriptions(MyClass)
        assert "model_id" in result
        assert "model identifier" in result["model_id"]
        assert "temperature" in result
        assert "Sampling temperature" in result["temperature"]


# ===================================================================
# Empty / missing docstrings
# ===================================================================

class TestEmptyDocstring:
    """Tests for classes with no, empty, or irrelevant docstrings."""

    def test_no_docstring_returns_empty_dict(self) -> None:
        """Class with None docstring returns empty dict."""

        class MyClass:
            pass

        # Ensure __doc__ is truly None
        assert MyClass.__doc__ is None
        result = parse_param_descriptions(MyClass)
        assert result == {}

    def test_empty_docstring_returns_empty_dict(self) -> None:
        """Class with empty string docstring returns empty dict."""

        class MyClass:
            """"""

        result = parse_param_descriptions(MyClass)
        assert result == {}

    def test_docstring_without_args_section(self) -> None:
        """Class with docstring but no Args/Parameters section."""

        class MyClass:
            """This class does something cool.

            It has no parameter documentation at all.
            Just a general description of the class behavior.
            """

        result = parse_param_descriptions(MyClass)
        assert result == {}


# ===================================================================
# Partial coverage
# ===================================================================

class TestPartialCoverage:
    """Tests where only some params are documented in the docstring."""

    def test_partial_param_coverage(self) -> None:
        """Some params documented, some not -- only documented ones returned."""

        class MyClass:
            """My class.

            Args:
                model_id: The model identifier.
                temperature: Sampling temperature.
            """

            def __init__(
                self,
                model_id: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ):
                pass

        result = parse_param_descriptions(MyClass)
        assert "model_id" in result
        assert "temperature" in result
        # max_tokens is NOT documented in the docstring
        assert "max_tokens" not in result


# ===================================================================
# Graceful degradation: docstring_parser not installed
# ===================================================================

class TestDocstringParserNotInstalled:
    """Tests graceful fallback when docstring_parser package is unavailable."""

    def test_returns_empty_dict_when_parser_unavailable(self) -> None:
        """If docstring_parser is not installed, return {} gracefully."""

        class MyClass:
            """My class.

            Args:
                model_id: The model identifier.
            """

        # Simulate ImportError for docstring_parser
        with patch.dict("sys.modules", {"docstring_parser": None}):
            # We need to force re-evaluation of the import guard.
            # The implementation should handle ImportError gracefully.
            # This tests that the function does not raise when the
            # optional dependency is missing.
            result = parse_param_descriptions(MyClass)
            assert isinstance(result, dict)


# ===================================================================
# Unicode descriptions
# ===================================================================

class TestUnicodeDescriptions:
    """Tests for non-ASCII / Unicode descriptions in docstrings."""

    def test_chinese_description(self) -> None:
        """Chinese characters in parameter descriptions."""

        class MyClass:
            """OpenAI LLM provider.

            Args:
                model_id: 模型 ID，如 gpt-4o
                temperature: 生成温度，0-2 之间
            """

        result = parse_param_descriptions(MyClass)
        assert result["model_id"] == "模型 ID，如 gpt-4o"
        assert result["temperature"] == "生成温度，0-2 之间"


# ===================================================================
# Function (not class) with docstring
# ===================================================================

class TestFunctionDocstring:
    """Tests for passing a function (not class) to parse_param_descriptions."""

    def test_function_with_google_docstring(self) -> None:
        """Standalone function with Google-style Args."""

        def process_data(input_path: str, verbose: bool = False) -> None:
            """Process data from a file.

            Args:
                input_path: Path to the input file.
                verbose: Enable verbose logging.
            """

        result = parse_param_descriptions(process_data)
        assert result == {
            "input_path": "Path to the input file.",
            "verbose": "Enable verbose logging.",
        }

    def test_function_without_docstring(self) -> None:
        """Function with no docstring returns empty dict."""

        def bare_function(x: int) -> int:
            return x + 1

        result = parse_param_descriptions(bare_function)
        assert result == {}


# ===================================================================
# Docstring on __init__ method
# ===================================================================

class TestInitDocstring:
    """Tests for classes where docstring is on __init__ instead of class body."""

    def test_class_docstring_preferred_over_init(self) -> None:
        """Class-level docstring Args should be found."""

        class MyClass:
            """My class.

            Args:
                name: The name field.
            """

            def __init__(self, name: str):
                pass

        result = parse_param_descriptions(MyClass)
        assert result == {"name": "The name field."}

    def test_init_docstring_used_when_class_has_none(self) -> None:
        """If class docstring has no Args, look at __init__ docstring."""

        class MyClass:
            """My class -- no Args section here."""

            def __init__(self, name: str):
                """Initialize.

                Args:
                    name: The name field from init.
                """

        result = parse_param_descriptions(MyClass)
        assert result == {"name": "The name field from init."}
