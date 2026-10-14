"""Tests for conscribe.config.codegen — TDD RED phase.

Tests ``generate_layer_config_source()`` which serializes a
``LayerConfigResult`` into a valid Python source file string.

Test categories:
- Output structure: header, imports, class definitions, union alias
- Extra policy: closed (extra="forbid") vs open (extra="allow")
- Field rendering: Field() with description/constraints, defaults
- Type rendering: Literal, Optional, Union, basic types, list, dict, Any
- Validity: output compiles, executes, produces Pydantic BaseModel subclasses

All registrars and implementation classes are defined at MODULE LEVEL
so that ``get_type_hints()`` can resolve forward references under
``from __future__ import annotations``.
"""
from __future__ import annotations

import enum
import pathlib
import re
from typing import Annotated, Any, Optional, Protocol, Union, runtime_checkable

import pytest
from pydantic import BaseModel, Field, ValidationError

from conscribe import create_registrar
from conscribe.config.builder import LayerConfigResult, build_layer_config
from conscribe.config.codegen import generate_layer_config_source

# A rendered ``= ...`` default (D5: never emitted).
_ELLIPSIS_DEFAULT = re.compile(r"=\s*\.\.\.\s*(#|$|,|\))", re.MULTILINE)


# ===================================================================
# Protocols for test registrars
# ===================================================================

@runtime_checkable
class _LLMProto(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


@runtime_checkable
class _AgentProto(Protocol):
    async def step(self, task: str) -> str: ...
    def reset(self) -> None: ...


# ===================================================================
# Registrars (module-level for type hint resolution)
# ===================================================================

_single_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_multi_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["LLM"],
)

_closed_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_open_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_mixed_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["LLM"],
)

_desc_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_constraint_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_required_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_optional_none_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_str_default_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_num_default_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_bool_default_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_literal_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_optional_type_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_union_type_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_basic_types_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_list_type_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_dict_type_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_any_type_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)

_validity_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
    strip_suffixes=["LLM"],
)

_agent_union_reg = create_registrar(
    "agent", _AgentProto, discriminator_field="name",
    strip_suffixes=["Agent"],
)


# ===================================================================
# Implementation classes for _single_reg (single key, no union alias)
# ===================================================================

class _SingleBase(metaclass=_single_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class SingleOpenAI(_SingleBase):
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes for _multi_reg (multiple keys, union alias)
# ===================================================================

class _MultiBase(metaclass=_multi_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OpenaiMultiLLM(_MultiBase):
    __registry_key__ = "openai"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class AnthropicMultiLLM(_MultiBase):
    __registry_key__ = "anthropic"

    def __init__(self, *, model_id: str, max_tokens: int = 4096):
        self.model_id = model_id
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _closed_reg (closed schema, extra="forbid")
# ===================================================================

class _ClosedBase(metaclass=_closed_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ClosedProvider(_ClosedBase):
    __registry_key__ = "closed"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes for _open_reg (open schema, extra="allow")
# ===================================================================

class _OpenBase(metaclass=_open_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OpenProvider(_OpenBase):
    __registry_key__ = "flexible"

    def __init__(self, *, model_id: str, **kwargs: Any):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _mixed_reg (mixed open/closed)
# ===================================================================

class _MixedBase(metaclass=_mixed_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ClosedMixLLM(_MixedBase):
    __registry_key__ = "closed"

    def __init__(self, *, model_id: str, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature


class OpenMixLLM(_MixedBase):
    __registry_key__ = "open"

    def __init__(self, *, model_id: str, **kwargs: Any):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _desc_reg (field with description)
# ===================================================================

class _DescBase(metaclass=_desc_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DescProvider(_DescBase):
    __registry_key__ = "desc"

    def __init__(
        self,
        *,
        model_id: Annotated[str, Field(description="The model identifier")],
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.temperature = temperature


# ===================================================================
# Implementation classes for _constraint_reg (field with constraints)
# ===================================================================

class _ConstraintBase(metaclass=_constraint_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ConstraintProvider(_ConstraintBase):
    __registry_key__ = "constrained"

    def __init__(
        self,
        *,
        model_id: str,
        temperature: Annotated[float, Field(ge=0, le=2)] = 0.0,
        max_tokens: Annotated[int, Field(gt=0, lt=100000)] = 4096,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _required_reg (required field, no default)
# ===================================================================

class _RequiredBase(metaclass=_required_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class RequiredProvider(_RequiredBase):
    __registry_key__ = "required"

    def __init__(self, *, model_id: str, api_key: str):
        self.model_id = model_id
        self.api_key = api_key


# ===================================================================
# Implementation classes for _optional_none_reg (Optional with None default)
# ===================================================================

class _OptNoneBase(metaclass=_optional_none_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OptNoneProvider(_OptNoneBase):
    __registry_key__ = "opt_none"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
    ):
        self.model_id = model_id
        self.base_url = base_url


# ===================================================================
# Implementation classes for _str_default_reg (string default)
# ===================================================================

class _StrDefaultBase(metaclass=_str_default_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class StrDefaultProvider(_StrDefaultBase):
    __registry_key__ = "str_default"

    def __init__(self, *, model_id: str = "gpt-4o"):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _num_default_reg (numeric default)
# ===================================================================

class _NumDefaultBase(metaclass=_num_default_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class NumDefaultProvider(_NumDefaultBase):
    __registry_key__ = "num_default"

    def __init__(
        self,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens


# ===================================================================
# Implementation classes for _bool_default_reg (bool default)
# ===================================================================

class _BoolDefaultBase(metaclass=_bool_default_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class BoolDefaultProvider(_BoolDefaultBase):
    __registry_key__ = "bool_default"

    def __init__(self, *, verbose: bool = False, strict: bool = True):
        self.verbose = verbose
        self.strict = strict


# ===================================================================
# Implementation classes for _literal_reg (Literal type in field)
# ===================================================================

class _LiteralBase(metaclass=_literal_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


# Note: The discriminator field "provider" is Literal["lit_provider"]
# by injection. We test that it renders correctly in source.
class LiteralProvider(_LiteralBase):
    __registry_key__ = "lit_provider"

    def __init__(self, *, model_id: str):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _optional_type_reg (Optional type rendering)
# ===================================================================

class _OptTypeBase(metaclass=_optional_type_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class OptTypeProvider(_OptTypeBase):
    __registry_key__ = "opt_type"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
    ):
        self.model_id = model_id
        self.base_url = base_url


# ===================================================================
# Implementation classes for _union_type_reg (Union type rendering)
# ===================================================================

class _UnionTypeBase(metaclass=_union_type_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class UnionTypeProvider(_UnionTypeBase):
    __registry_key__ = "union_type"

    def __init__(
        self,
        *,
        value: Union[str, int, float],
    ):
        self.value = value


# ===================================================================
# Implementation classes for _basic_types_reg (basic type rendering)
# ===================================================================

class _BasicTypesBase(metaclass=_basic_types_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class BasicTypesProvider(_BasicTypesBase):
    __registry_key__ = "basic"

    def __init__(
        self,
        *,
        name: str = "default",
        count: int = 0,
        ratio: float = 0.5,
        enabled: bool = True,
    ):
        self.name = name
        self.count = count
        self.ratio = ratio
        self.enabled = enabled


# ===================================================================
# Implementation classes for _list_type_reg (list[T] rendering)
# ===================================================================

class _ListTypeBase(metaclass=_list_type_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class ListTypeProvider(_ListTypeBase):
    __registry_key__ = "list_type"

    def __init__(self, *, tags: list[str] = []):
        self.tags = tags


# ===================================================================
# Implementation classes for _dict_type_reg (dict[K, V] rendering)
# ===================================================================

class _DictTypeBase(metaclass=_dict_type_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DictTypeProvider(_DictTypeBase):
    __registry_key__ = "dict_type"

    def __init__(self, *, metadata: dict[str, Any] = {}):
        self.metadata = metadata


# ===================================================================
# Implementation classes for _any_type_reg (Any type rendering)
# ===================================================================

class _AnyTypeBase(metaclass=_any_type_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class AnyTypeProvider(_AnyTypeBase):
    __registry_key__ = "any_type"

    def __init__(self, *, data: Any = None):
        self.data = data


# ===================================================================
# Implementation classes for _validity_reg (validity/compilation tests)
# ===================================================================

class _ValidityBase(metaclass=_validity_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class AlphaValidLLM(_ValidityBase):
    """Closed schema with rich metadata for validity tests."""
    __registry_key__ = "alpha"

    def __init__(
        self,
        *,
        model_id: Annotated[str, Field(description="Model identifier")],
        temperature: Annotated[float, Field(ge=0, le=2)] = 0.0,
        max_tokens: Annotated[int, Field(gt=0)] = 4096,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens


class BetaValidLLM(_ValidityBase):
    """Open schema for validity tests."""
    __registry_key__ = "beta"

    def __init__(self, *, model_id: str, **kwargs: Any):
        self.model_id = model_id


# ===================================================================
# Implementation classes for _agent_union_reg (union alias naming)
# ===================================================================

class _AgentUnionBase(metaclass=_agent_union_reg.Meta):
    __abstract__ = True
    async def step(self, task: str) -> str:
        return ""
    def reset(self) -> None:
        pass


class BrowserUseAgent(_AgentUnionBase):
    __registry_key__ = "browser_use"

    def __init__(self, *, max_steps: int = 100):
        self.max_steps = max_steps


class SkyvernAgent(_AgentUnionBase):
    __registry_key__ = "skyvern"

    def __init__(self, *, api_url: str):
        self.api_url = api_url


# ===================================================================
# Output structure tests
# ===================================================================

class TestOutputStructure:
    """Tests that the generated source has the correct overall structure."""

    def test_output_contains_auto_generated_header(self) -> None:
        """Output contains a header comment warning not to edit manually."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        assert "Auto-generated" in source or "auto-generated" in source
        assert "DO NOT EDIT" in source or "do not edit" in source.lower()

    def test_output_contains_future_annotations(self) -> None:
        """Output contains 'from __future__ import annotations'."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        assert "from __future__ import annotations" in source

    def test_output_contains_typing_imports(self) -> None:
        """Output imports from typing (at minimum Literal)."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        assert "from typing import" in source or "from typing " in source
        assert "Literal" in source

    def test_output_contains_pydantic_imports(self) -> None:
        """Output imports from pydantic (BaseModel, ConfigDict, Field)."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        assert "from pydantic import" in source or "from pydantic " in source
        assert "BaseModel" in source
        assert "ConfigDict" in source

    def test_output_contains_class_definitions_for_each_key(self) -> None:
        """Output contains a class definition for each registered key."""
        result = build_layer_config(_multi_reg)
        source = generate_layer_config_source(result)

        for model_name in result.per_key_models.values():
            assert f"class {model_name.__name__}(BaseModel):" in source

    def test_output_contains_union_alias_for_multiple_models(self) -> None:
        """Output contains a union type alias when there are multiple models."""
        result = build_layer_config(_multi_reg)
        source = generate_layer_config_source(result)

        # Should contain Union and the discriminator field reference
        assert "Union[" in source or "Union[\n" in source
        assert "discriminator=" in source

    def test_single_model_no_union_alias(self) -> None:
        """Single model does NOT produce a Union alias."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        # Should NOT contain "Union[" for single key
        # The source should still have the class, but no union wrapping
        assert "Union[" not in source


# ===================================================================
# Extra policy tests
# ===================================================================

class TestExtraPolicy:
    """Tests that open/closed schema extra policy is reflected in source."""

    def test_closed_schema_has_extra_forbid_in_source(self) -> None:
        """Closed schema (extra='forbid') is reflected in model_config."""
        result = build_layer_config(_closed_reg)
        source = generate_layer_config_source(result)

        assert 'extra="forbid"' in source

    def test_open_schema_has_extra_allow_in_source(self) -> None:
        """Open schema (extra='allow') is reflected in model_config."""
        result = build_layer_config(_open_reg)
        source = generate_layer_config_source(result)

        assert 'extra="allow"' in source

    def test_open_schema_has_comment_about_extra_fields(self) -> None:
        """Open schema class carries a comment (not a docstring) about extra fields.

        A docstring would become the generated model's JSON-schema
        ``description`` — one the in-process model does not have.
        """
        result = build_layer_config(_open_reg)
        source = generate_layer_config_source(result)

        open_model_name = None
        for key, model in result.per_key_models.items():
            if model.model_config.get("extra") == "allow":
                open_model_name = model.__name__
                break

        assert open_model_name is not None, "No open model found"

        class_idx = source.index(f"class {open_model_name}(BaseModel):")
        class_block = source[class_idx:class_idx + 500]
        assert "# This implementation accepts extra parameters (**kwargs)." in class_block
        assert '"""' not in class_block and "'''" not in class_block


# ===================================================================
# Field rendering tests
# ===================================================================

class TestFieldRendering:
    """Tests for how individual fields are rendered in source output."""

    def test_field_with_description_renders_field_description(self) -> None:
        """Field with description renders Field(description=...)."""
        result = build_layer_config(_desc_reg)
        source = generate_layer_config_source(result)

        assert 'description=' in source
        assert "The model identifier" in source

    def test_field_with_constraints_renders_field_constraints(self) -> None:
        """Field with constraints (ge, le, gt, lt) renders Field with constraints."""
        result = build_layer_config(_constraint_reg)
        source = generate_layer_config_source(result)

        assert "ge=" in source
        assert "le=" in source
        assert "gt=" in source
        assert "lt=" in source

    def test_required_field_renders_without_default(self) -> None:
        """Required field (no default) renders as a bare annotation (D5).

        ``= ...`` is an ``EllipsisType`` assignment error for pyright and makes
        type checkers treat the field as optional.
        """
        result = build_layer_config(_required_reg)
        source = generate_layer_config_source(result)

        assert re.search(r"^    model_id: str$", source, re.MULTILINE)
        assert re.search(r"^    api_key: str$", source, re.MULTILINE)
        assert not _ELLIPSIS_DEFAULT.search(source)

        ns: dict[str, Any] = {}
        exec(compile(source, "<test>", "exec"), ns)  # noqa: S102
        model = ns[result.per_key_models["required"].__name__]
        assert model.model_fields["model_id"].is_required()
        assert model.model_fields["api_key"].is_required()

    def test_optional_field_with_none_default(self) -> None:
        """Optional field with None default renders = None or Field(None, ...)."""
        result = build_layer_config(_optional_none_reg)
        source = generate_layer_config_source(result)

        # base_url should have None as its default
        assert "None" in source

    def test_field_with_string_default(self) -> None:
        """Field with string default renders the string correctly."""
        result = build_layer_config(_str_default_reg)
        source = generate_layer_config_source(result)

        assert '"gpt-4o"' in source or "'gpt-4o'" in source

    def test_field_with_numeric_default(self) -> None:
        """Field with numeric default renders the number correctly."""
        result = build_layer_config(_num_default_reg)
        source = generate_layer_config_source(result)

        assert "0.7" in source
        assert "4096" in source

    def test_field_with_bool_default(self) -> None:
        """Field with bool default renders True/False correctly."""
        result = build_layer_config(_bool_default_reg)
        source = generate_layer_config_source(result)

        assert "False" in source
        assert "True" in source


# ===================================================================
# Type rendering tests
# ===================================================================

class TestTypeRendering:
    """Tests for how types are rendered as source code strings."""

    def test_literal_type_renders_correctly(self) -> None:
        """Literal type renders as Literal['value'] in the discriminator field."""
        result = build_layer_config(_literal_reg)
        source = generate_layer_config_source(result)

        assert "Literal['lit_provider']" in source

    def test_optional_type_renders_correctly(self) -> None:
        """Optional type renders as 'type | None' or 'Optional[type]'."""
        result = build_layer_config(_optional_type_reg)
        source = generate_layer_config_source(result)

        # The design doc says Union[str, None] -> str | None
        # Accept either form
        assert (
            "str | None" in source
            or "Optional[str]" in source
            or "Union[str, None]" in source
        )

    def test_union_type_renders_correctly(self) -> None:
        """Union of multiple types renders correctly."""
        result = build_layer_config(_union_type_reg)
        source = generate_layer_config_source(result)

        # Should render as X | Y | Z or Union[X, Y, Z]
        assert (
            "str | int | float" in source
            or "Union[str, int, float]" in source
            # Order might vary
            or ("str" in source and "int" in source and "float" in source)
        )

    def test_basic_types_render_correctly(self) -> None:
        """Basic types (str, int, float, bool) render as their names."""
        result = build_layer_config(_basic_types_reg)
        source = generate_layer_config_source(result)

        # Should have type annotations with basic type names
        assert ": str" in source or ": str =" in source
        assert ": int" in source or ": int =" in source
        assert ": float" in source or ": float =" in source
        assert ": bool" in source or ": bool =" in source

    def test_list_type_renders_correctly(self) -> None:
        """list[T] renders correctly in source."""
        result = build_layer_config(_list_type_reg)
        source = generate_layer_config_source(result)

        assert "list[str]" in source or "List[str]" in source

    def test_dict_type_renders_correctly(self) -> None:
        """dict[K, V] renders correctly in source."""
        result = build_layer_config(_dict_type_reg)
        source = generate_layer_config_source(result)

        assert (
            "dict[str, Any]" in source
            or "Dict[str, Any]" in source
        )

    def test_any_type_renders_correctly(self) -> None:
        """Any type renders correctly in source."""
        result = build_layer_config(_any_type_reg)
        source = generate_layer_config_source(result)

        assert "Any" in source


# ===================================================================
# Validity tests (compilation and execution)
# ===================================================================

class TestValidity:
    """Tests that the generated source is valid Python that can be compiled,
    executed, and produces working Pydantic models."""

    def test_output_is_valid_python(self) -> None:
        """Output compiles as valid Python via compile()."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        # Must not raise SyntaxError
        code = compile(source, "<test>", "exec")
        assert code is not None

    def test_output_can_be_executed(self) -> None:
        """Output can be exec()'d successfully."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        # The namespace should contain the model class
        model_name = next(iter(result.per_key_models.values())).__name__
        assert model_name in ns

    def test_executed_classes_are_basemodel_subclasses(self) -> None:
        """Executed classes are Pydantic BaseModel subclasses."""
        result = build_layer_config(_validity_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        for model in result.per_key_models.values():
            model_cls = ns[model.__name__]
            assert issubclass(model_cls, BaseModel), (
                f"{model.__name__} should be a BaseModel subclass"
            )

    def test_executed_classes_validate_correct_data(self) -> None:
        """Executed classes can validate correct data."""
        result = build_layer_config(_validity_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        # Alpha model (closed schema)
        alpha_cls = ns["AlphaLLMConfig"]
        instance = alpha_cls(
            provider="alpha",
            model_id="gpt-4o",
            temperature=0.5,
            max_tokens=1024,
        )
        assert instance.model_id == "gpt-4o"
        assert instance.temperature == 0.5
        assert instance.provider == "alpha"

    def test_executed_classes_reject_invalid_data_closed_schema(self) -> None:
        """Executed classes with closed schema reject extra fields."""
        result = build_layer_config(_validity_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        alpha_cls = ns["AlphaLLMConfig"]
        with pytest.raises(ValidationError):
            alpha_cls(
                provider="alpha",
                model_id="gpt-4o",
                unknown_field="should_fail",
            )

    def test_multi_model_output_compiles_and_validates(self) -> None:
        """Multi-model output with union alias compiles and validates."""
        result = build_layer_config(_validity_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        # Both models should be present
        assert "AlphaLLMConfig" in ns
        assert "BetaLLMConfig" in ns

        # Both should be BaseModel subclasses
        assert issubclass(ns["AlphaLLMConfig"], BaseModel)
        assert issubclass(ns["BetaLLMConfig"], BaseModel)

    def test_open_schema_executed_model_accepts_extra(self) -> None:
        """Open schema model (extra='allow') from executed source accepts extra fields."""
        result = build_layer_config(_validity_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        beta_cls = ns["BetaLLMConfig"]
        # Open schema should accept extra fields without error
        instance = beta_cls(
            provider="beta",
            model_id="gpt-4o",
            extra_param="should_pass",
        )
        assert instance.model_id == "gpt-4o"

    def test_agent_layer_output_compiles(self) -> None:
        """Agent layer (layer name > 3 chars -> Title case) output compiles."""
        result = build_layer_config(_agent_union_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        # Naming: "browser_use" + "agent" -> "BrowserUseAgentConfig"
        assert "BrowserUseAgentConfig" in ns
        assert "SkyvernAgentConfig" in ns

        bu_cls = ns["BrowserUseAgentConfig"]
        instance = bu_cls(name="browser_use", max_steps=50)
        assert instance.max_steps == 50

    def test_constrained_fields_enforce_constraints_after_exec(self) -> None:
        """Fields with constraints (ge, le, gt, lt) enforce them in exec'd models."""
        result = build_layer_config(_constraint_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        model_cls = ns[next(iter(result.per_key_models.values())).__name__]

        # Valid: temperature=1.0 (ge=0, le=2) should pass
        instance = model_cls(
            provider="constrained",
            model_id="test",
            temperature=1.0,
            max_tokens=100,
        )
        assert instance.temperature == 1.0

        # Invalid: temperature=3.0 (le=2 violated) should fail
        with pytest.raises(ValidationError):
            model_cls(
                provider="constrained",
                model_id="test",
                temperature=3.0,
                max_tokens=100,
            )

    def test_header_contains_layer_info(self) -> None:
        """Header comment contains layer name, discriminator, and entry count."""
        result = build_layer_config(_multi_reg)
        source = generate_layer_config_source(result)

        # Per design doc section 6.1: header includes layer info and entry count
        # Check for layer name
        assert "llm" in source.lower().split("class")[0]
        # Check for discriminator
        assert "provider" in source.split("class")[0]

    def test_union_alias_name_uses_layer_name(self) -> None:
        """The union type alias name is derived from the layer name."""
        result = build_layer_config(_multi_reg)
        source = generate_layer_config_source(result)

        # Per design doc: LlmConfig = Annotated[Union[...], ...]
        # The alias should contain the layer name in some form
        # "LlmConfig" or "LLMConfig" or similar
        assert "LlmConfig" in source or "LLMConfig" in source


# ===================================================================
# Degraded field rendering tests
# ===================================================================

class _IncompatibleType:
    """A type Pydantic cannot serialize."""

    def __init__(self, x: int):
        self.x = x


_degraded_codegen_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _DegradedCodegenBase(metaclass=_degraded_codegen_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class DegradedCodegenProvider(_DegradedCodegenBase):
    __registry_key__ = "degraded"

    def __init__(self, *, model_id: str, auth: _IncompatibleType, count: int = 0):
        self.model_id = model_id
        self.auth = auth
        self.count = count


def _build_degraded(registrar: Any) -> LayerConfigResult:
    """Build a layer whose fields degrade; the DegradationWarning is part of the contract."""
    from conscribe.exceptions import DegradationWarning

    with pytest.warns(DegradationWarning, match="degrades to Any"):
        return build_layer_config(registrar)


class TestDegradedFieldRendering:
    """Tests for degraded field rendering in generated source."""

    def test_inline_comment_on_degraded_field(self) -> None:
        """Generated source contains '# degraded from:' for degraded fields."""
        result = _build_degraded(_degraded_codegen_reg)
        source = generate_layer_config_source(result)

        assert "# degraded from:" in source

    def test_header_warning_section(self) -> None:
        """Generated source header contains WARNING section."""
        result = _build_degraded(_degraded_codegen_reg)
        source = generate_layer_config_source(result)

        assert "WARNING" in source
        assert "degraded to Any" in source

    def test_no_warning_on_clean_build(self) -> None:
        """No WARNING in header when there are no degraded fields."""
        result = build_layer_config(_single_reg)
        source = generate_layer_config_source(result)

        assert "WARNING" not in source

    def test_degraded_source_compiles(self) -> None:
        """Generated source with degraded fields compiles and executes."""
        result = _build_degraded(_degraded_codegen_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test-degraded>", "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)  # noqa: S102

        # The model should be present and be a BaseModel subclass
        model_cls = ns.get("DegradedLLMConfig")
        assert model_cls is not None
        assert issubclass(model_cls, BaseModel)

        # Should validate with any value for the degraded field
        instance = model_cls(provider="degraded", model_id="test", auth="any_value")
        assert instance.model_id == "test"


# ===================================================================
# Enum and non-serializable defaults (3a)
# ===================================================================

class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


_enum_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _EnumBase(metaclass=_enum_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class EnumDefaultProvider(_EnumBase):
    __registry_key__ = "enum_default"

    def __init__(self, *, color: Color = Color.RED, model_id: str = "test"):
        self.color = color
        self.model_id = model_id


class TestEnumAndCallableDefaults:
    """Tests for Enum and non-serializable default rendering (3a)."""

    def test_enum_default_renders_as_member_access(self) -> None:
        """Enum default should render as 'Color.RED', not '<Color.RED: ...>'."""
        result = build_layer_config(_enum_reg)
        source = generate_layer_config_source(result)

        # Should contain "Color.RED" not "<Color.RED"
        assert "Color.RED" in source
        assert "<Color.RED" not in source

    def test_enum_import_generated(self) -> None:
        """Generated source should import the enum type."""
        result = build_layer_config(_enum_reg)
        source = generate_layer_config_source(result)

        # Should have an import for the Color enum
        assert "Color" in source

    def test_enum_source_compiles(self) -> None:
        """Generated source with enum default compiles and executes."""
        result = build_layer_config(_enum_reg)
        source = generate_layer_config_source(result)

        code = compile(source, "<test-enum>", "exec")
        assert code is not None


# ===================================================================
# Non-builtin type imports in stubs (3b)
# ===================================================================

_path_reg = create_registrar(
    "llm", _LLMProto, discriminator_field="provider",
)


class _PathBase(metaclass=_path_reg.Meta):
    __abstract__ = True
    async def chat(self, messages: list[dict]) -> str:
        return ""


class PathProvider(_PathBase):
    __registry_key__ = "path_provider"

    def __init__(self, *, config_path: pathlib.Path, model_id: str = "test"):
        self.config_path = config_path
        self.model_id = model_id


class TestNonBuiltinTypeImports:
    """Tests that non-builtin types get import statements in generated source (3b)."""

    def test_pathlib_path_import_generated(self) -> None:
        """pathlib.Path field type should generate 'from pathlib import Path'."""
        result = build_layer_config(_path_reg)
        source = generate_layer_config_source(result)

        assert "from pathlib import" in source
        assert "Path" in source

    def test_pathlib_path_type_in_field(self) -> None:
        """Field type should show 'Path' not 'pathlib.Path'."""
        result = build_layer_config(_path_reg)
        source = generate_layer_config_source(result)

        # Should contain ": Path" in the field definition
        assert ": Path" in source or ": Path =" in source


# ###################################################################
# conscribe 2.0 renderer (D5, F5, D6, D9, header)
# ###################################################################

import ast  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import decimal  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
import typing  # noqa: E402
import uuid  # noqa: E402
from typing import Literal, Tuple, TypeVar  # noqa: E402

import annotated_types  # noqa: E402
from pydantic import (  # noqa: E402
    AfterValidator,
    AliasChoices,
    AliasPath,
    ConfigDict,
    Discriminator,
    StringConstraints,
    Tag,
    conint,
    create_model,
    model_validator,
)

import conscribe  # noqa: E402
from conscribe.config._render import _FIELD_PARAMS, RenderContext, render_model_class  # noqa: E402
from conscribe.config.codegen import (  # noqa: E402
    _composed_alias_name,
    _union_alias_name,
    generate_composed_config_source,
)
from conscribe.config.composed import build_composed_config  # noqa: E402
from conscribe.config.json_schema import (  # noqa: E402
    generate_composed_json_schema,
    generate_layer_json_schema,
)
from conscribe.exceptions import ConfigRenderError, ExtractionWarning  # noqa: E402
from conscribe.registration.registry import _REGISTRY_INDEX, _deregister  # noqa: E402

from tests.config.artifact_parity import (  # noqa: E402
    assert_artifacts_agree,
    composed_expectations,
    exec_generated_namespace,
    layer_expectations,
    load_generated_module,
    type_to_source,
    value_to_source,
)


# -------------------------------------------------------------------
# Importable fixture types (module level: generated sources import them)
# -------------------------------------------------------------------

class Shade(enum.Enum):
    LIGHT = "light"
    DARK = "dark"


class Perm(enum.Flag):
    READ = 1
    WRITE = 2


@dataclasses.dataclass
class KnobDC:
    a: int = 1
    b: str = "x"


class UserModel(BaseModel):
    """A user BaseModel knob (af defect 6)."""

    x: int = 1


class UserSchema(BaseModel):
    """A user schema a generated per-key model may inherit from (K)."""

    model_config = ConfigDict(title="UserSchemaTitle")
    x: int = Field(default=1, ge=0, description="xx")
    y: str = "s"

    @model_validator(mode="after")
    def _not_thirteen(self) -> "UserSchema":
        if self.x == 13:
            raise ValueError("13 is not allowed")
        return self


_UNSET = object()
_T = TypeVar("_T")


def default_globs() -> list[str]:
    """A module-level default factory."""
    return ["*"]


def keep_positive(value: int) -> int:
    """A module-level validator."""
    if value < 0:
        raise ValueError("negative")
    return value


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _generated(name: str, **fields: Any) -> type[BaseModel]:
    """A model conscribe treats as generated (``__conscribe_generated__``)."""
    model = create_model(name, **fields)
    model.__conscribe_generated__ = True  # type: ignore[attr-defined]
    return model


def _helper_base(name: str, base: type[BaseModel], *, extra: str) -> type[BaseModel]:
    """A config-only helper base, made the way the extractor makes its ``_<Name>Base``."""
    return type(name, (base,), {
        "model_config": ConfigDict(extra=extra),  # type: ignore[typeddict-item]
        "__module__": "conscribe.config.extractor",
    })


def _layer_of(model: type[BaseModel], *, key: str = "k", disc: str = "kind") -> LayerConfigResult:
    """A one-model ``LayerConfigResult`` around a directly built model."""
    return LayerConfigResult(
        union_type=model,
        per_key_models={key: model},
        layer_name="probe",
        discriminator_field=disc,
    )


def _render(model: type[BaseModel]) -> str:
    """Generated source for one directly built model (public entry point)."""
    return generate_layer_config_source(_layer_of(model))


def _line(source: str, name: str, *, in_class: str | None = None) -> str:
    """The (stripped) line declaring field ``name`` (first match, or in ``in_class``)."""
    if in_class is not None:
        start = source.index(f"class {in_class}(")
        end = source.find("\nclass ", start + 1)
        source = source[start:] if end == -1 else source[start:end]
    for line in source.splitlines():
        if re.match(rf"^    {re.escape(name)}: ", line):
            return line.strip()
    raise AssertionError(f"no line for field {name!r} in:\n{source}")


def _pep681_has_default(stmt: ast.AnnAssign) -> bool:
    """Whether a PEP 681 type checker sees a default on this field statement.

    Annotation-only means required.  A ``Field(...)`` call has a default only
    with a ``default`` / ``default_factory`` / ``factory`` keyword (PEP 681
    ignores positional defaults).  Any other assigned value is a default.
    """
    if stmt.value is None:
        return False
    value = stmt.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Field":
        return any(kw.arg in ("default", "default_factory", "factory") for kw in value.keywords)
    return True


def _assert_pep681_matches_runtime(source: str) -> None:
    """Static (PEP 681) required-ness equals runtime ``is_required()`` for every field."""
    module = load_generated_module(source)
    checked = 0
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        model = getattr(module, node.name)
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            name = stmt.target.id
            if name == "model_config":
                continue
            static_required = not _pep681_has_default(stmt)
            assert static_required == model.model_fields[name].is_required(), (
                f"{node.name}.{name}: PEP 681 sees required={static_required}"
            )
            checked += 1
    assert checked


_CG2 = "cg2_"


@pytest.fixture
def cg2_cleanup():
    """Deregister every ``cg2_*`` registry a test created."""
    yield
    for name in list(_REGISTRY_INDEX):
        if name.startswith(_CG2):
            _deregister(name)


@runtime_checkable
class _RunProto(Protocol):
    def run(self) -> None: ...


def _nested_llm(name: str, *, identical_directs: bool = False):
    """A nested-mode layer: two model types, three or four providers."""
    reg = create_registrar(
        name, _RunProto, discriminator_fields=["model_type", "provider"], key_separator=".",
    )

    class _OpenAIBase(metaclass=reg.Meta):
        __registry_key__ = "openai"
        __abstract__ = True

        def run(self) -> None: ...

        def __init__(self, model: str = "gpt"):
            self.model = model

    class OpenAIDirect(_OpenAIBase):
        __registry_key__ = "openai.direct"

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)

    class OpenAIAzure(_OpenAIBase):
        __registry_key__ = "openai.azure"

        def __init__(self, deployment: str = "dep", **kwargs: Any):
            super().__init__(**kwargs)

    class _AnthBase(metaclass=reg.Meta):
        __registry_key__ = "anthropic"
        __abstract__ = True

        def run(self) -> None: ...

        def __init__(self, model: str = "claude"):
            self.model = model

    if identical_directs:
        class AnthropicDirect(_AnthBase):
            __registry_key__ = "anthropic.direct"

            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)
    else:
        class AnthropicDirectOrg(_AnthBase):
            __registry_key__ = "anthropic.direct"

            def __init__(self, organization: str = "", **kwargs: Any):
                super().__init__(**kwargs)

    return reg


def _archiver_and_hook(prefix: str) -> dict[str, Any]:
    """A flat target layer, a nested LLM layer, and an owner wiring both."""
    arch = create_registrar(f"{prefix}archiver", _RunProto, discriminator_field="archiver_id")
    hook = create_registrar(f"{prefix}hook", _RunProto, discriminator_field="hook_id")
    llm = _nested_llm(f"{prefix}llm")

    class _ArchBase(metaclass=arch.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class Net(_ArchBase):
        __registry_key__ = "net"

        def __init__(self, host: str = "h"):
            self.host = host

    class Disk(_ArchBase):
        __registry_key__ = "disk"

        def __init__(self, path: str = "/tmp"):
            self.path = path

    class Stream(_ArchBase):
        __registry_key__ = "stream"

        def __init__(self, url: str = "u"):
            self.url = url

    class _HookBase(metaclass=hook.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class TraceWriter(_HookBase):
        __registry_key__ = "trace_writer"
        __wiring__ = {
            "sink": f"{prefix}archiver",
            "pair": (f"{prefix}archiver", ["net", "disk"]),
            "many": f"{prefix}archiver",
            "maybe": f"{prefix}archiver",
            "llm": f"{prefix}llm",
        }
        many: list

        def __init__(self, sink: str, pair: str, llm: str, maybe: Optional[str] = None):
            self.sink = sink

    return {f"{prefix}archiver": arch, f"{prefix}hook": hook, f"{prefix}llm": llm}


# === end of 2.0 renderer fixtures ===


_MOD = __name__  # module the generated sources import fixture types from


class TestGeneratorHeader:
    """Generated files start with the conscribe version."""

    def test_flat_header(self) -> None:
        lines = generate_layer_config_source(build_layer_config(_multi_reg)).splitlines()
        assert lines[0] == f"# Generated by conscribe {conscribe.__version__}"
        assert lines[1] == "# Auto-generated by conscribe. DO NOT EDIT."

    def test_nested_and_composed_headers(self, cg2_cleanup) -> None:
        reg = _nested_llm(f"{_CG2}hdr")
        nested = generate_layer_config_source(build_layer_config(reg))
        composed = generate_composed_config_source(build_composed_config({f"{_CG2}hdr": reg}))
        for source in (nested, composed):
            assert source.splitlines()[0] == f"# Generated by conscribe {conscribe.__version__}"
            assert "# Auto-generated by conscribe. DO NOT EDIT." in source


class TestRequiredAndDefaultForms:
    """D5: required fields carry no default token; defaults use keywords."""

    def test_required_with_kwargs_has_no_positional_ellipsis(self) -> None:
        source = generate_layer_config_source(build_layer_config(_desc_reg))
        assert _line(source, "model_id") == "model_id: str = Field(description='The model identifier')"

    def test_defaults_with_kwargs_use_the_keyword_form(self) -> None:
        source = generate_layer_config_source(build_layer_config(_constraint_reg))
        assert _line(source, "temperature") == "temperature: float = Field(default=0.0, ge=0, le=2)"
        assert _line(source, "max_tokens") == "max_tokens: int = Field(default=4096, gt=0, lt=100000)"

    def test_plain_default_without_kwargs(self) -> None:
        source = generate_layer_config_source(build_layer_config(_num_default_reg))
        assert _line(source, "temperature") == "temperature: float = 0.7"

    def test_no_ellipsis_default_in_any_output(self, cg2_cleanup) -> None:
        regs = _archiver_and_hook(f"{_CG2}ell_")
        sources = [
            generate_layer_config_source(build_layer_config(reg))
            for reg in (_validity_reg, _required_reg, _desc_reg, _constraint_reg)
        ]
        sources.append(generate_layer_config_source(build_layer_config(regs[f"{_CG2}ell_llm"])))
        for inline in (True, False):
            sources.append(generate_composed_config_source(
                build_composed_config(regs, inline_wiring=inline)
            ))
        for source in sources:
            assert not _ELLIPSIS_DEFAULT.search(source), source
            assert "default=..." not in source

    def test_pep681_view_matches_runtime_required(self, cg2_cleanup) -> None:
        """What a PEP 681 checker reads from the source equals ``is_required()``."""
        regs = _archiver_and_hook(f"{_CG2}pep_")
        for reg in (_validity_reg, _required_reg, _desc_reg, _constraint_reg, _basic_types_reg):
            _assert_pep681_matches_runtime(generate_layer_config_source(build_layer_config(reg)))
        _assert_pep681_matches_runtime(
            generate_layer_config_source(build_layer_config(regs[f"{_CG2}pep_llm"]))
        )
        for inline in (True, False):
            _assert_pep681_matches_runtime(generate_composed_config_source(
                build_composed_config(regs, inline_wiring=inline)
            ))

    @pytest.mark.filterwarnings("ignore:Field name .* shadows an attribute")
    def test_basemodel_attribute_names_get_an_explicit_field(self) -> None:
        """On pydantic < 2.12 a bare ``validate: bool`` would default to the method."""
        model = _generated(
            "ShadowConfig",
            kind=(Literal["k"], "k"),
            validate=(bool, ...),
            json=(int, 5),
            copy=(str, Field(description="c")),
        )
        source = _render(model)
        ignore = "  # pyright: ignore[reportIncompatibleMethodOverride]"
        assert _line(source, "validate") == "validate: bool = Field()" + ignore
        assert _line(source, "json") == "json: int = Field(default=5)" + ignore
        assert _line(source, "copy") == "copy: str = Field(description='c')" + ignore
        module = load_generated_module(source)
        fields = module.ShadowConfig.model_fields
        assert fields["validate"].is_required()
        assert fields["copy"].is_required()
        assert not fields["json"].is_required()
        _assert_pep681_matches_runtime(source)

    def test_field_named_like_its_type(self) -> None:
        """A field never shares a name with an import, assigned or not.

        ``date: date`` alone would run, but type checkers scope the field name
        to the class body: a later ``date(2020, 1, 1)`` there reads as the
        unbound field (pyright: ``"date" is unbound``).  The import is aliased.
        """
        required = _generated(
            "WhenConfig",
            kind=(Literal["k"], "k"),
            date=(datetime.date, ...),
            day=(datetime.date, datetime.date(2020, 1, 1)),
        )
        source = _render(required)
        assert "from datetime import date as date__datetime" in source
        assert _line(source, "date") == "date: date__datetime"
        assert _line(source, "day") == "day: date__datetime = date__datetime(2020, 1, 1)"
        assert_artifacts_agree(source, {"WhenConfig": required})

        assigned = _generated(
            "ShadowsTypeConfig",
            kind=(Literal["k"], "k"),
            date=(str, "x"),
            when=(datetime.date, datetime.date(2020, 1, 1)),
        )
        source = _render(assigned)
        assert "from datetime import date as date__datetime" in source
        assert _line(source, "when") == "when: date__datetime = date__datetime(2020, 1, 1)"
        assert_artifacts_agree(source, {"ShadowsTypeConfig": assigned}, samples={
            "ShadowsTypeConfig": [{}, {"when": "2021-02-03"}, {"date": 1}],
        })


class TestValueLadder:
    """F5: defaults render as verified source, never as raw ``repr``."""

    def test_literals_and_containers(self) -> None:
        model = _generated(
            "LitConfig",
            kind=(Literal["k"], "k"),
            text=(str, "x'y\"z"),
            raw=(bytes, b"\x00a"),
            items=(list[int], [1, 2]),
            single=(tuple[int, ...], (1,)),
            mapping=(dict[str, int], {"b": 2, "a": 1}),
            tags=(set[str], {"b", "a", "c"}),
            frozen=(frozenset[int], frozenset({3, 1})),
            empty=(set[int], set()),
            big=(float, math.inf),
            small=(float, -math.inf),
            nan=(float, math.nan),
        )
        source = _render(model)
        assert _line(source, "items") == "items: list[int] = [1, 2]"
        assert _line(source, "single") == "single: tuple[int, ...] = (1,)"
        assert _line(source, "mapping") == "mapping: dict[str, int] = {'b': 2, 'a': 1}"
        assert _line(source, "tags") == "tags: set[str] = {'a', 'b', 'c'}"
        assert _line(source, "frozen") == "frozen: frozenset[int] = frozenset({1, 3})"
        assert _line(source, "empty") == "empty: set[int] = set()"
        assert _line(source, "big") == "big: float = float('inf')"
        assert _line(source, "small") == "small: float = float('-inf')"
        assert _line(source, "nan") == "nan: float = float('nan')"
        assert_artifacts_agree(source, {"LitConfig": model}, samples={"LitConfig": [{}]})

    def test_scalar_literals_standalone(self) -> None:
        assert value_to_source(1 + 2j) == "(1+2j)"
        assert value_to_source(b"\x00a") == "b'\\x00a'"
        assert value_to_source([Shade.DARK, None]) == "[Shade.DARK, None]"

    def test_enum_flag_path_and_constructor_forms(self) -> None:
        model = _generated(
            "CtorConfig",
            kind=(Literal["k"], "k"),
            shade=(Shade, Shade.DARK),
            perm=(Perm, Perm.READ | Perm.WRITE),
            path=(pathlib.Path, pathlib.Path("/tmp/x")),
            day=(datetime.date, datetime.date(2020, 1, 2)),
            span=(datetime.timedelta, datetime.timedelta(days=1)),
            amount=(decimal.Decimal, decimal.Decimal("1.5")),
            ident=(uuid.UUID, uuid.UUID(int=5)),
            knob=(KnobDC, KnobDC(a=2)),
            user=(UserModel, UserModel(x=3)),
            choice=(Literal[Shade.LIGHT, Shade.DARK], Shade.LIGHT),
        )
        source = _render(model)
        assert _line(source, "shade") == "shade: Shade = Shade.DARK"
        assert _line(source, "perm") == "perm: Perm = Perm(3)"
        assert _line(source, "path") == "path: Path = Path('/tmp/x')"
        assert _line(source, "day") == "day: date = date(2020, 1, 2)"
        assert _line(source, "span") == "span: timedelta = timedelta(days=1)"
        assert _line(source, "amount") == "amount: Decimal = Decimal('1.5')"
        assert _line(source, "ident") == "ident: UUID = UUID('00000000-0000-0000-0000-000000000005')"
        assert _line(source, "knob") == "knob: KnobDC = KnobDC(a=2)"
        assert _line(source, "user") == "user: UserModel = UserModel(x=3)"
        assert _line(source, "choice") == "choice: Literal[Shade.LIGHT, Shade.DARK] = Shade.LIGHT"
        assert f"from {_MOD} import KnobDC, Perm, Shade, UserModel" in source
        assert "<" not in "".join(l for l in source.splitlines() if not l.startswith("#"))
        assert_artifacts_agree(source, {"CtorConfig": model}, samples={
            "CtorConfig": [{}, {"shade": "light"}, {"choice": "dark"}, {"choice": "nope"}],
        })

    def test_tz_aware_values_have_a_constructor_form(self) -> None:
        """R02: tz-aware ``datetime`` / ``time`` values are constructors whose
        ``tzinfo`` goes through the ladder -- as defaults, constraint values and
        examples alike (no JSON form, no ``validate_default``)."""
        utc = datetime.timezone.utc
        plus2 = datetime.timezone(datetime.timedelta(hours=2))
        est = datetime.timezone(datetime.timedelta(hours=-5), "EST")
        model = _generated(
            "TzConfig",
            kind=(Literal["k"], "k"),
            start=(datetime.datetime, datetime.datetime(2020, 1, 1, tzinfo=utc)),
            strict=(Annotated[datetime.datetime, Field(strict=True)], datetime.datetime(2020, 1, 1, tzinfo=plus2)),
            alarm=(datetime.time, datetime.time(7, 30, tzinfo=est)),
            after=(Annotated[datetime.datetime, Field(
                gt=datetime.datetime(2030, 1, 1, tzinfo=utc),
                examples=[datetime.datetime(2030, 6, 1, tzinfo=utc)],
            )], ...),
        )
        source = _render(model)
        assert _line(source, "start") == "start: datetime = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)"
        assert _line(source, "strict") == (
            "strict: datetime = Field(default=datetime(2020, 1, 1, 0, 0, "
            "tzinfo=timezone(timedelta(seconds=7200))), strict=True)"
        )
        assert _line(source, "alarm") == (
            "alarm: time = time(7, 30, tzinfo=timezone(timedelta(days=-1, seconds=68400), 'EST'))"
        )
        assert _line(source, "after") == (
            "after: datetime = Field(examples=[datetime(2030, 6, 1, 0, 0, tzinfo=timezone.utc)], "
            "gt=datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc))"
        )
        assert "validate_default" not in source
        module = assert_artifacts_agree(source, {"TzConfig": model}, samples={"TzConfig": [
            {}, {"after": "2020-01-01T00:00:00Z"}, {"after": "2031-01-01T00:00:00Z"},
            {"after": "2031-01-01T00:00:00Z", "strict": "2020-01-01T00:00:00Z"},
        ]})
        assert module.TzConfig.model_fields["alarm"].default.tzinfo.tzname(None) == "EST"

    def test_module_level_objects_are_imported_by_identity(self) -> None:
        model = _generated("SentinelConfig", kind=(Literal["k"], "k"), timeout=(Any, _UNSET))
        source = _render(model)
        assert _line(source, "timeout") == "timeout: Any = _UNSET"
        assert f"from {_MOD} import _UNSET" in source
        module = load_generated_module(source)
        assert module.SentinelConfig().timeout is _UNSET

    def test_unrenderable_default_raises(self) -> None:
        opaque = object()  # not bound at module level anywhere
        model = _generated("OpaqueConfig", kind=(Literal["k"], "k"), thing=(Any, opaque))
        with pytest.raises(ConfigRenderError) as exc:
            _render(model)
        assert exc.value.model == "OpaqueConfig"
        assert exc.value.field == "thing"
        assert "module-level object" in str(exc.value)
        assert "0x" not in str(exc.value)

    def test_failure_names_the_function_local_class(self) -> None:
        @dataclasses.dataclass
        class LocalItem:
            v: int = 0

        model = _generated(
            "NestedFailConfig", kind=(Literal["k"], "k"),
            item=(Any, LocalItem()), items=(list[Any], []),
        )
        with pytest.raises(ConfigRenderError, match=r"NestedFailConfig\.item: .*LocalItem.*inside a function"):
            _render(model)
        nested = _generated("NestedListFailConfig", kind=(Literal["k"], "k"), items=(list[Any], [LocalItem()]))
        with pytest.raises(ConfigRenderError, match=r"NestedListFailConfig\.items: .*LocalItem.*inside a function"):
            _render(nested)

    def test_conscribe_objects_are_refused(self) -> None:
        from conscribe.wiring import WiredField

        # Wrapped in Field(): a bare WiredField in an annotated class body is
        # "no default" (WiredField.__set_name__), so it would not leak.
        model = _generated(
            "LeakConfig",
            kind=(Literal["k"], "k"),
            loop=(Any, Field(default=WiredField("loop", "agent_loop"))),
        )
        with pytest.raises(ConfigRenderError, match=r"LeakConfig\.loop: .*conscribe"):
            _render(model)


class TestDefaultFactories:
    """F5: factories render as importable references or fail loudly."""

    def test_importable_and_builtin_factories(self) -> None:
        model = _generated(
            "FactoryConfig",
            kind=(Literal["k"], "k"),
            globs=(list[str], Field(default_factory=default_globs)),
            items=(list[int], Field(default_factory=list)),
            extra=(dict[str, int], Field(default_factory=dict, description="d")),
        )
        source = _render(model)
        assert _line(source, "globs") == "globs: list[str] = Field(default_factory=default_globs)"
        assert _line(source, "items") == "items: list[int] = Field(default_factory=list)"
        assert _line(source, "extra") == "extra: dict[str, int] = Field(default_factory=dict, description='d')"
        assert f"from {_MOD} import default_globs" in source
        module = assert_artifacts_agree(source, {"FactoryConfig": model}, samples={"FactoryConfig": [{}]})
        assert module.FactoryConfig().globs == ["*"]

    def test_lambda_factory_raises(self) -> None:
        model = _generated(
            "LambdaFactoryConfig", kind=(Literal["k"], "k"),
            globs=(list[str], Field(default_factory=lambda: ["*"])),
        )
        with pytest.raises(ConfigRenderError, match=r"LambdaFactoryConfig\.globs: .*lambda"):
            _render(model)

    def test_function_local_factory_raises(self) -> None:
        def make() -> list[str]:
            return ["x"]

        model = _generated(
            "LocalFactoryConfig", kind=(Literal["k"], "k"), globs=(list[str], Field(default_factory=make)),
        )
        with pytest.raises(ConfigRenderError, match="defined inside a function"):
            _render(model)

    def test_dataclass_factory_sentinel_is_never_rendered(self, cg2_cleanup) -> None:
        """af defect 8: ``<factory>`` must not reach the generated file."""
        reg = create_registrar(f"{_CG2}dcf", _RunProto, discriminator_field="loader_id")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        @dataclasses.dataclass
        class DcFactory(_Base):
            __registry_key__ = "dc_factory"
            sources: list[str] = dataclasses.field(default_factory=default_globs)
            depth: int = 1

        result = build_layer_config(reg)
        default = result.per_key_models["dc_factory"].model_fields["sources"].default
        if default is getattr(dataclasses, "_HAS_DEFAULT_FACTORY", object()):
            # The extractor has not resolved the dataclass factory yet: refuse.
            with pytest.raises(ConfigRenderError, match="sentinel"):
                generate_layer_config_source(result)
        else:
            source = generate_layer_config_source(result)
            assert "<factory>" not in source
            assert_artifacts_agree(source, layer_expectations(result))


class TestConstraintsAndMetadata:
    """F5: every constraint reaches the generated file; unrenderable metadata warns."""

    def test_af_defect_7_min_length_survives(self, cg2_cleanup) -> None:
        reg = create_registrar(f"{_CG2}owner7", _RunProto, discriminator_field="owner_id")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Owner(_Base):
            __registry_key__ = "owner"

            def __init__(
                self,
                *,
                tags: Annotated[list[str], Field(min_length=1, description="at least one tag")],
                depth: int = 1,
            ):
                self.tags = tags

        res = build_composed_config({f"{_CG2}owner7": reg})
        source = generate_composed_config_source(res)
        assert _line(source, "tags") == "tags: list[str] = Field(description='at least one tag', min_length=1)"
        layer = f"{_CG2}owner7"
        assert_artifacts_agree(
            source,
            composed_expectations(res),
            samples={"ComposedConfig": [
                {layer: [{"owner_id": "owner", "tags": []}]},
                {layer: [{"owner_id": "owner", "tags": ["a"]}]},
            ]},
            published=(generate_composed_json_schema(res), res.top_level_type),
        )
        module = load_generated_module(source)
        with pytest.raises(ValidationError, match="too_short|at least 1"):
            module.ComposedConfig.model_validate({layer: [{"owner_id": "owner", "tags": []}]})

    def test_every_field_constraint_keyword(self) -> None:
        fields: dict[str, Any] = dict(
            kind=(Literal["k"], "k"),
            a=(int, Field(default=1, gt=0, lt=10, multiple_of=1)),
            b=(float, Field(default=1.0, ge=0, le=5, allow_inf_nan=False)),
            c=(str, Field(default="a", min_length=1, max_length=3, pattern="^a")),
            d=(decimal.Decimal, Field(default=decimal.Decimal("1.0"), max_digits=4, decimal_places=1)),
            e=(int, Field(default=1, strict=True)),
            f=(Union[int, str], Field(default=1, union_mode="left_to_right")),
        )
        if "coerce_numbers_to_str" in _FIELD_PARAMS:
            fields["g"] = (str, Field(default="1", coerce_numbers_to_str=True))
        if "fail_fast" in _FIELD_PARAMS:
            fields["h"] = (list[int], Field(default=[1], fail_fast=True))
        model = _generated("ConstraintKwConfig", **fields)
        source = _render(model)
        assert _line(source, "a") == "a: int = Field(default=1, gt=0, lt=10, multiple_of=1)"
        assert _line(source, "b") == "b: float = Field(default=1.0, ge=0, le=5, allow_inf_nan=False)"
        assert _line(source, "c") == "c: str = Field(default='a', min_length=1, max_length=3, pattern='^a')"
        assert _line(source, "d") == "d: Decimal = Field(default=Decimal('1.0'), max_digits=4, decimal_places=1)"
        assert _line(source, "e") == "e: int = Field(default=1, strict=True)"
        assert _line(source, "f") == "f: Union[int, str] = Field(default=1, union_mode='left_to_right')"
        if "g" in fields:
            assert _line(source, "g") == "g: str = Field(default='1', coerce_numbers_to_str=True)"
        if "h" in fields:
            assert _line(source, "h") == "h: list[int] = Field(default=[1], fail_fast=True)"
        assert_artifacts_agree(source, {"ConstraintKwConfig": model}, samples={"ConstraintKwConfig": [
            {}, {"a": 0}, {"a": 10}, {"b": 6}, {"b": "inf"}, {"c": "b"}, {"c": "abcd"},
            {"d": "123.45"}, {"e": "1"}, {"f": "1"}, {"g": 5}, {"h": ["x", "y"]},
        ]})

    def test_grouped_nested_and_string_constraints(self) -> None:
        model = _generated(
            "GroupedConfig",
            kind=(Literal["k"], "k"),
            i=(Annotated[int, annotated_types.Interval(gt=1, lt=5)], 2),
            n=(Annotated[list[int], annotated_types.Len(1, 3)], [1]),
            items=(list[Annotated[int, Field(ge=0)]], []),
            s=(Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True)], "A"),
            ci=(conint(gt=1), 2),
        )
        source = _render(model)
        assert _line(source, "i") == "i: int = Field(default=2, gt=1, lt=5)"
        assert _line(source, "n") == "n: list[int] = Field(default=[1], min_length=1, max_length=3)"
        assert _line(source, "items") == "items: list[Annotated[int, Field(ge=0)]] = []"
        assert _line(source, "s") == (
            "s: Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True)] = 'A'"
        )
        assert _line(source, "ci") == "ci: int = Field(default=2, gt=1)"
        assert_artifacts_agree(source, {"GroupedConfig": model}, samples={"GroupedConfig": [
            {}, {"i": 5}, {"n": []}, {"n": [1, 2, 3, 4]}, {"items": [1, -1]}, {"s": " ab "}, {"ci": 1},
        ]})

    def test_importable_validator_is_rendered(self) -> None:
        model = _generated(
            "ValidatedConfig", kind=(Literal["k"], "k"),
            n=(Annotated[int, AfterValidator(keep_positive)], 1),
        )
        source = _render(model)
        assert _line(source, "n") == "n: Annotated[int, AfterValidator(func=keep_positive)] = 1"
        assert_artifacts_agree(source, {"ValidatedConfig": model}, samples={
            "ValidatedConfig": [{"n": -1}, {"n": 2}],
        })

    def test_unrenderable_metadata_warns_and_is_dropped(self) -> None:
        model = _generated(
            "LambdaValidatedConfig",
            kind=(Literal["k"], "k"),
            n=(Annotated[int, AfterValidator(lambda v: v)], 1),
            p=(Annotated[int, annotated_types.Predicate(lambda v: v > 0)], 1),
        )
        with pytest.warns(ExtractionWarning) as record:
            source = _render(model)
        messages = [str(w.message) for w in record]
        assert any("LambdaValidatedConfig.n" in m and "AfterValidator" in m for m in messages)
        assert any("LambdaValidatedConfig.p" in m and "Predicate" in m for m in messages)
        assert all("0x" not in m for m in messages)
        assert _line(source, "n").startswith("n: int = 1  # not rendered: AfterValidator(")
        assert "# WARNING: The following metadata is enforced only by the in-process" in source
        module = load_generated_module(source)
        # The file lacks the predicate; the in-process model keeps it.
        module.LambdaValidatedConfig(p=-1)
        with pytest.raises(ValidationError):
            model(p=-1)


class TestTypeForms:
    """Type rendering: tuples, union order, unrenderable types."""

    def test_tuples(self) -> None:
        model = _generated(
            "TupleConfig",
            kind=(Literal["k"], "k"),
            var=(tuple[str, ...], ("*",)),
            empty=(tuple[()], ()),
            pair=(Tuple[int, str], (1, "a")),
            bare=(tuple, ()),
            bare_typing=(typing.Tuple, ()),
        )
        source = _render(model)
        assert _line(source, "var") == "var: tuple[str, ...] = ('*',)"
        assert _line(source, "empty") == "empty: tuple[()] = ()"
        assert _line(source, "pair") == "pair: tuple[int, str] = (1, 'a')"
        assert _line(source, "bare") == "bare: tuple = ()"
        assert _line(source, "bare_typing") == "bare_typing: tuple = ()"
        assert "Ellipsis" not in source
        assert_artifacts_agree(source, {"TupleConfig": model}, samples={"TupleConfig": [
            {}, {"var": ["a", "b"]}, {"empty": [1]}, {"pair": [1]},
        ]})

    def test_union_member_order_is_preserved(self) -> None:
        model = _generated(
            "OrderConfig",
            kind=(Literal["k"], "k"),
            a=(Union[None, int], None),
            b=(Optional[int], None),
            c=(Union[int, str, None], None),
        )
        source = _render(model)
        assert _line(source, "a") == "a: Union[None, int] = None"
        assert _line(source, "b") == "b: Optional[int] = None"
        assert _line(source, "c") == "c: Union[int, str, None] = None"
        assert_artifacts_agree(source, {"OrderConfig": model})

    def test_unbound_type_variable_raises(self) -> None:
        T = TypeVar("T")
        model = _generated("TypeVarConfig", kind=(Literal["k"], "k"), items=(list[T], []))  # type: ignore[valid-type]
        with pytest.raises(ConfigRenderError, match=r"TypeVarConfig\.items: .*type variable"):
            _render(model)

    def test_standalone_helpers(self) -> None:
        assert type_to_source(list[Annotated[int, Field(ge=0)]]) == "list[Annotated[int, Field(ge=0)]]"
        assert type_to_source(tuple[str, ...]) == "tuple[str, ...]"
        assert value_to_source({3, 1, 2}) == "{1, 2, 3}"


class TestFieldAttributes:
    """F5: FieldInfo attributes are carried into the generated field."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_documentation_alias_and_behaviour_attributes(self) -> None:
        model = _generated(
            "AttrConfig",
            kind=(Literal["k"], "k"),
            a=(int, Field(default=1, title="Alpha", description="first", examples=[1, 2],
                          json_schema_extra={"x-unit": "s"})),
            b=(int, Field(default=2, alias="bee")),
            c=(int, Field(default=3, validation_alias=AliasChoices("cee", AliasPath("nested", 0)),
                          serialization_alias="see")),
            d=(int, Field(default=4, frozen=True, validate_default=True, repr=False, deprecated="use a")),
            e=(int, Field(default=5, exclude=True)),
        )
        source = _render(model)
        assert _line(source, "a") == (
            "a: int = Field(default=1, title='Alpha', description='first', examples=[1, 2], "
            "json_schema_extra={'x-unit': 's'})"
        )
        assert _line(source, "b") == "b: int = Field(default=2, alias='bee')"
        assert _line(source, "c") == (
            "c: int = Field(default=3, validation_alias=AliasChoices('cee', AliasPath('nested', 0)), "
            "serialization_alias='see')"
        )
        assert _line(source, "d") == (
            "d: int = Field(default=4, deprecated='use a', frozen=True, validate_default=True, repr=False)"
        )
        assert _line(source, "e") == "e: int = Field(default=5, exclude=True)"
        assert_artifacts_agree(source, {"AttrConfig": model}, samples={"AttrConfig": [
            {"bee": 7}, {"b": 7}, {"nested": [9]}, {"cee": 8}, {"c": 1},
        ]})


class Page(BaseModel, typing.Generic[_T]):
    """A user pydantic generic model."""

    items: list[_T] = []


UserId = typing.NewType("UserId", int)

# Parametrized at module level: pydantic also binds it in this module's
# globals under the non-identifier name "Page[str]".
PageOfStr = Page[str]


class Factories:
    @classmethod
    def make(cls) -> list[str]:
        return ["made"]


class TestReferenceForms:
    """Less common type and value forms render by identity."""

    def test_field_named_like_a_builtin_qualifies_the_builtin(self) -> None:
        model = _generated(
            "BuiltinShadowConfig", kind=(Literal["k"], "k"),
            list=(int, 1), items=(list[int], []),
        )
        source = _render(model)
        assert "import builtins" in source
        assert _line(source, "items") == "items: builtins.list[int] = []"
        assert_artifacts_agree(source, {"BuiltinShadowConfig": model}, samples={
            "BuiltinShadowConfig": [{"items": [1]}, {"list": "x"}],
        })

    def test_generic_model_newtype_and_callable(self) -> None:
        model = _generated(
            "RefFormsConfig", kind=(Literal["k"], "k"),
            page=(Page[int], Page[int]()),
            pages=(PageOfStr, PageOfStr(items=["a"])),
            user=(UserId, UserId(3)),
            hook=(typing.Callable[[int], str], str),
            any_hook=(typing.Callable[..., str], str),
        )
        source = _render(model)
        assert _line(source, "page") == "page: Page[int] = Page[int]()"
        assert _line(source, "pages") == "pages: Page[str] = Page[str](items=['a'])"
        assert "[" not in "".join(l for l in source.splitlines() if l.startswith("from "))
        assert _line(source, "user") == "user: UserId = 3"
        assert _line(source, "hook") == "hook: Callable[[int], str] = str"
        assert _line(source, "any_hook") == "any_hook: Callable[..., str] = str"
        assert "from collections.abc import Callable" in source
        assert f"from {_MOD} import Page, UserId" in source
        module = load_generated_module(source)
        assert module.RefFormsConfig().hook is str
        assert module.RefFormsConfig.model_fields["user"].annotation is UserId

    def test_method_references(self) -> None:
        model = _generated(
            "MethodRefConfig", kind=(Literal["k"], "k"),
            made=(list[str], Field(default_factory=Factories.make)),
            lower=(Annotated[str, annotated_types.Predicate(str.islower)], "a"),
        )
        source = _render(model)
        assert _line(source, "made") == "made: list[str] = Field(default_factory=Factories.make)"
        assert _line(source, "lower") == "lower: Annotated[str, Predicate(func=str.islower)] = 'a'"
        assert_artifacts_agree(source, {"MethodRefConfig": model}, samples={
            "MethodRefConfig": [{}, {"lower": "B"}],
        })

    def test_main_module_warning_through_composed_entry_point(self, monkeypatch, cg2_cleanup) -> None:
        class MainKnob2(BaseModel):
            q: int = 0

        MainKnob2.__module__ = "__main__"
        MainKnob2.__qualname__ = "MainKnob2"
        monkeypatch.setattr(sys.modules["__main__"], "MainKnob2", MainKnob2, raising=False)
        reg = create_registrar(f"{_CG2}main", _RunProto, discriminator_field="kind")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Owner(_Base):
            __registry_key__ = "owner"

            def __init__(self, knob=MainKnob2(), depth: int = 1):  # noqa: B008
                self.knob = knob

            # A real (non-string) annotation: the class is function-local.
            __init__.__annotations__["knob"] = MainKnob2

        with pytest.warns(ExtractionWarning, match="defined in __main__"):
            source = generate_composed_config_source(build_composed_config({f"{_CG2}main": reg}))
        assert "from __main__ import MainKnob2" in source


class TestRendererHooks:
    """Extension points the layout code (and lane J) builds on."""

    def test_extra_class_body_lines(self) -> None:
        model = _generated("HookedConfig", x=(int, 0))
        ctx = RenderContext()
        ctx.define_model(model, "HookedConfig")
        validator = ctx.ns.helper("pydantic", "field_validator")
        text = render_model_class(model, ctx, extra_lines=[
            f"@{validator}('x')",
            "@classmethod",
            "def _double(cls, v):",
            "    return v * 2",
        ])
        source = (
            "from __future__ import annotations\n"
            + "\n".join(ctx.ns.import_lines()) + "\n"
            + ctx.ns.materialize(text)
        )
        assert "    @field_validator('x')" in source
        module = load_generated_module(source)
        assert module.HookedConfig(x=2).x == 4


class Circle(BaseModel):
    kind: Literal["circle"] = "circle"
    r: float = 1.0


class Square(BaseModel):
    kind: Literal["square"] = "square"
    side: float = 1.0


class TestInlineDiscriminators:
    """D9: inline unions keep their discriminators in the generated file."""

    def test_layer_aliases_and_subset_unions(self, cg2_cleanup) -> None:
        p = f"{_CG2}d9a_"
        res = build_composed_config(_archiver_and_hook(p), inline_wiring=True)
        source = generate_composed_config_source(res)
        archiver = _composed_alias_name(f"{p}archiver")
        llm = _composed_alias_name(f"{p}llm")
        assert _line(source, "sink") == f"sink: {archiver}  # wired from: {p}archiver"
        assert _line(source, "maybe") == f"maybe: Optional[{archiver}] = None  # wired from: {p}archiver"
        assert _line(source, "many") == f"many: list[{archiver}]  # wired from: {p}archiver"
        assert _line(source, "llm") == f"llm: {llm}  # wired from: {p}llm"
        arch_models = res.layer_results[f"{p}archiver"].per_key_models
        net, disk = arch_models["net"].__name__, arch_models["disk"].__name__
        assert _line(source, "pair") == (
            f"pair: Union[{net}, {disk}] = Field(discriminator='archiver_id')  # wired from: {p}archiver"
        )
        assert f"Discriminator(_discriminate_{p}llm)" in source

    def test_generated_source_rejects_tagless_inline_blocks(self, cg2_cleanup) -> None:
        p = f"{_CG2}d9b_"
        res = build_composed_config(_archiver_and_hook(p), inline_wiring=True)
        source = generate_composed_config_source(res)
        owner = res.layer_results[f"{p}hook"].per_key_models["trace_writer"]
        ok = {
            "hook_id": "trace_writer",
            "sink": {"archiver_id": "net"},
            "pair": {"archiver_id": "disk"},
            "many": [{"archiver_id": "stream"}],
            "llm": {"model_type": "openai", "provider": {"name": "azure"}},
        }
        bad = [
            {**ok, "sink": {"host": "x"}},
            {**ok, "pair": {"path": "/x"}},
            {**ok, "pair": {"archiver_id": "stream"}},
            {**ok, "many": [{"url": "x"}]},
            {**ok, "llm": {"model_type": "nope"}},
            {**ok, "llm": {"provider": {"name": "azure"}}},
        ]
        module = assert_artifacts_agree(
            source,
            composed_expectations(res),
            samples={owner.__name__: [ok, *bad]},
            published=(generate_composed_json_schema(res), res.top_level_type),
        )
        generated = getattr(module, owner.__name__)
        generated.model_validate(ok)
        for payload in bad:
            with pytest.raises(ValidationError, match="union_tag|literal_error|Input tag"):
                generated.model_validate(payload)

    def test_literal_mode_composed_parity(self, cg2_cleanup) -> None:
        p = f"{_CG2}d9c_"
        res = build_composed_config(_archiver_and_hook(p), inline_wiring=False)
        source = generate_composed_config_source(res)
        owner = res.layer_results[f"{p}hook"].per_key_models["trace_writer"]
        assert_artifacts_agree(source, composed_expectations(res), samples={owner.__name__: [
            {"sink": "net", "pair": "disk", "many": ["stream"], "llm": "openai.azure"},
            {"sink": "net", "pair": "stream", "many": [], "llm": "openai.azure"},
        ]}, published=(generate_composed_json_schema(res), res.top_level_type))

    def test_field_level_discriminated_user_union(self) -> None:
        model = _generated(
            "ShapeConfig",
            kind=(Literal["k"], "k"),
            shape=(Union[Circle, Square], Field(discriminator="kind")),
            shapes=(list[Annotated[Union[Circle, Square], Field(discriminator="kind")]], []),
        )
        source = _render(model)
        assert _line(source, "shape") == "shape: Union[Circle, Square] = Field(discriminator='kind')"
        assert _line(source, "shapes") == (
            "shapes: list[Annotated[Union[Circle, Square], Field(discriminator='kind')]] = []"
        )
        assert_artifacts_agree(source, {"ShapeConfig": model}, samples={"ShapeConfig": [
            {"shape": {"kind": "circle"}}, {"shape": {"r": 2.0}}, {"shape": {"kind": "circle"}, "shapes": [{"side": 1}]},
        ]})

    def test_conscribe_discriminators_match_by_identity_only(self) -> None:
        """Every union of a nested layer carries the layer's one cached
        discriminator function, so emitted functions are matched by identity.
        A conscribe function the file does not emit is never imported: the
        discriminator is reported as metadata the file lacks."""

        def foreign(v: Any) -> str:
            return "a"

        foreign.__module__ = "conscribe.config.builder"
        foreign.__qualname__ = foreign.__name__ = "_discriminate_nl"
        member_a = _generated("AModel", tag=(Literal["a"], "a"))
        member_b = _generated("BModel", tag=(Literal["b"], "b"))
        union = Annotated[
            Union[Annotated[member_a, Tag("a")], Annotated[member_b, Tag("b")]],
            Discriminator(foreign),
        ]
        owner = _generated("InlineOwnerConfig", llm=(Optional[union], None))

        def emitted(v: Any) -> str:
            return "a"

        ctx = RenderContext()
        for model in (member_a, member_b, owner):
            ctx.define_model(model, model.__name__)
        ctx.define_function(emitted, "_discriminate_nl")  # same name, another object
        text = ctx.ns.materialize(render_model_class(owner, ctx))
        code = "\n".join(line.split("  #", 1)[0] for line in text.splitlines())
        assert "_discriminate_nl" not in code
        assert "# not rendered: Discriminator(" in text
        assert [(n.model, n.field) for n in ctx.notes] == [("InlineOwnerConfig", "llm")]
        assert "conscribe" not in "\n".join(ctx.ns.import_lines())

        ctx = RenderContext()
        for model in (member_a, member_b, owner):
            ctx.define_model(model, model.__name__)
        ctx.define_function(foreign, "_discriminate_nl")  # the very object: matched
        text = ctx.ns.materialize(render_model_class(owner, ctx))
        assert "Discriminator(_discriminate_nl)" in text
        assert not ctx.notes


class TestDefineVsImport:
    """D6: define generated models, import everything else by identity."""

    def test_af_defect_6_user_basemodel_knob_is_imported(self, cg2_cleanup) -> None:
        reg = create_registrar(f"{_CG2}owner6", _RunProto, discriminator_field="owner_id")

        class _Base(metaclass=reg.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Owner(_Base):
            __registry_key__ = "owner"

            def __init__(
                self,
                depth: int = 1,
                matcher: Annotated[UserModel, Field(description="matcher")] = UserModel(),
            ):
                self.matcher = matcher

        res = build_composed_config({f"{_CG2}owner6": reg})
        source = generate_composed_config_source(res)
        assert f"from {_MOD} import UserModel" in source
        assert _line(source, "matcher") == "matcher: UserModel = Field(default=UserModel(), description='matcher')"
        layer = f"{_CG2}owner6"
        assert_artifacts_agree(source, composed_expectations(res), samples={"ComposedConfig": [
            {layer: [{"owner_id": "owner", "matcher": {"x": 5}}]},
            {layer: [{"owner_id": "owner", "matcher": {"x": "nope"}}]},
        ]}, published=(generate_composed_json_schema(res), res.top_level_type))

    def test_user_types_inside_wrappers_are_imported(self) -> None:
        model = _generated(
            "WrapConfig",
            kind=(Literal["k"], "k"),
            a=(Optional[UserModel], None),
            b=(list[KnobDC], []),
            c=(dict[str, Shade], {}),
            d=(Union[UserModel, KnobDC, None], None),
        )
        source = _render(model)
        assert f"from {_MOD} import KnobDC, Shade, UserModel" in source
        assert "class UserModel" not in source
        assert_artifacts_agree(source, {"WrapConfig": model}, samples={"WrapConfig": [
            {"a": {"x": 2}, "b": [{"a": 3}], "c": {"q": "dark"}, "d": {"x": 1}},
        ]})

    def test_conscribe_types_are_never_imported(self) -> None:
        from conscribe.exceptions import WiringProblem

        model = create_model(
            "LeakTypeConfig",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            problem=(WiringProblem, ...),
        )
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        with pytest.raises(ConfigRenderError, match="conscribe's own module"):
            _render(model)

    def test_function_local_types_raise(self) -> None:
        class LocalKnob(BaseModel):
            q: int = 0

        model = _generated("LocalConfig", kind=(Literal["k"], "k"), knob=(LocalKnob, ...))
        with pytest.raises(ConfigRenderError) as exc:
            _render(model)
        assert exc.value.model == "LocalConfig"
        assert exc.value.field == "knob"
        assert "<locals>" in str(exc.value)

    def test_main_module_types_warn(self, monkeypatch) -> None:
        class MainKnob(BaseModel):
            q: int = 0

        MainKnob.__module__ = "__main__"
        MainKnob.__qualname__ = "MainKnob"
        monkeypatch.setattr(sys.modules["__main__"], "MainKnob", MainKnob, raising=False)
        model = _generated("MainConfig", kind=(Literal["k"], "k"), knob=(MainKnob, ...))
        with pytest.warns(ExtractionWarning, match="defined in __main__"):
            source = _render(model)
        assert "from __main__ import MainKnob" in source

    def test_colliding_import_names_are_aliased_deterministically(self, monkeypatch) -> None:
        mod_a = types.ModuleType("cs_cg2_mod_a")
        mod_b = types.ModuleType("cs_cg2_mod_b")
        exec(
            "from pydantic import BaseModel\n"
            "class Matcher(BaseModel):\n    a: int = 1\n"
            "class Field(BaseModel):\n    f: int = 0\n",
            mod_a.__dict__,
        )
        exec(
            "from pydantic import BaseModel\n"
            "class Matcher(BaseModel):\n    b: str = 'x'\n"
            "class CollideConfig(BaseModel):\n    o: int = 0\n",
            mod_b.__dict__,
        )
        monkeypatch.setitem(sys.modules, "cs_cg2_mod_a", mod_a)
        monkeypatch.setitem(sys.modules, "cs_cg2_mod_b", mod_b)

        def build(order: list[str]) -> type[BaseModel]:
            spec = {
                "m1": (mod_a.Matcher, mod_a.Matcher()),
                "m2": (Optional[mod_b.Matcher], None),
                "fld": (mod_a.Field, mod_a.Field()),
                "clash": (Optional[mod_b.CollideConfig], None),
                "desc": (int, Field(default=1, description="uses pydantic's Field")),
            }
            return _generated("CollideConfig", kind=(Literal["k"], "k"), **{k: spec[k] for k in order})

        model = build(["m1", "m2", "fld", "clash", "desc"])
        source = _render(model)
        assert "from cs_cg2_mod_a import Field as Field__cs_cg2_mod_a, Matcher" in source
        assert (
            "from cs_cg2_mod_b import CollideConfig as CollideConfig__cs_cg2_mod_b, "
            "Matcher as Matcher__cs_cg2_mod_b"
        ) in source
        module = assert_artifacts_agree(source, {"CollideConfig": model})
        fields = module.CollideConfig.model_fields
        assert fields["m1"].annotation is mod_a.Matcher
        assert typing.get_args(fields["m2"].annotation)[0] is mod_b.Matcher
        assert fields["fld"].annotation is mod_a.Field

        reordered = _render(build(["desc", "clash", "fld", "m2", "m1"]))
        imports = [l for l in source.splitlines() if l.startswith(("from ", "import "))]
        assert imports == [l for l in reordered.splitlines() if l.startswith(("from ", "import "))]

    def test_generated_submodel_defaults_are_factories(self, cg2_cleanup) -> None:
        res = build_layer_config(_nested_llm(f"{_CG2}nl6"))
        source = generate_layer_config_source(res)
        azure = res.per_key_models["openai.azure"].__name__
        assert _line(source, "provider", in_class=azure) == (
            "provider: AzureProviderConfig = Field(default_factory=AzureProviderConfig, "
            "json_schema_extra={'default': {'name': 'azure', 'deployment': 'dep'}})"
        )
        assert "AzureProviderConfig()" not in source
        assert_artifacts_agree(source, layer_expectations(res), samples={
            azure: [{}, {"provider": {"name": "azure", "deployment": "x"}}, {"provider": {"name": "x"}}],
        }, published=(generate_layer_json_schema(res), res.union_type))

    def test_marker_decides_define_or_import(self) -> None:
        inner = _generated("GenKnob", z=(int, 0))
        model = _generated(
            "MarkerConfig", kind=(Literal["k"], "k"), user=(UserModel, UserModel()), gen=(inner, ...),
        )
        source = _render(model)
        assert "class GenKnob(BaseModel):" in source
        assert "import GenKnob" not in source
        assert f"from {_MOD} import UserModel" in source
        assert_artifacts_agree(source, {"MarkerConfig": model, "GenKnob": inner}, samples={
            "MarkerConfig": [{"gen": {"z": 1}}, {"gen": {"zz": 1}}],
        })

    def test_generated_model_with_user_base_subclasses_it(self) -> None:
        base = _helper_base("_KBase", UserSchema, extra="forbid")
        model = create_model("KProbeConfig", __base__=base, kind=(Literal["k"], "k"))
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        source = _render(model)
        assert "class KProbeConfig(UserSchema):" in source
        assert 'model_config = ConfigDict(extra="forbid")' in source
        assert _line(source, "kind") == "kind: Literal['k'] = 'k'"
        assert "    x:" not in source and "    y:" not in source
        assert_artifacts_agree(source, {"KProbeConfig": model}, samples={"KProbeConfig": [
            {"x": 13}, {"x": 2}, {"x": -1}, {"zz": 1},
        ]})

    def test_identical_same_name_submodels_are_emitted_once(self, cg2_cleanup) -> None:
        res = build_layer_config(_nested_llm(f"{_CG2}nl7", identical_directs=True))
        source = generate_layer_config_source(res)
        assert source.count("class DirectProviderConfig(BaseModel):") == 1
        assert "DirectProviderConfig2" not in source
        assert_artifacts_agree(source, layer_expectations(res))

    def test_layers_mapping_to_one_alias_name_raise(self, cg2_cleanup) -> None:
        def two_key_layer(name: str):
            reg = create_registrar(name, _RunProto, discriminator_field="kind")

            class _Base(metaclass=reg.Meta):
                __abstract__ = True

                def run(self) -> None: ...

            class _One(_Base):
                __registry_key__ = "one"

            class _Two(_Base):
                __registry_key__ = "two"

            return reg

        regs = {f"{_CG2}x_y": two_key_layer(f"{_CG2}x_y"), f"{_CG2}x__y": two_key_layer(f"{_CG2}x__y")}
        with pytest.raises(ConfigRenderError, match="both map to the generated name 'Cg2XYConfig'"):
            generate_composed_config_source(build_composed_config(regs))

    def test_field_named_like_a_generated_class_raises(self) -> None:
        """An assigned field would shadow the generated class its default uses."""
        inner = _generated("InnerKnob", z=(int, 0))
        model = _generated(
            "ConflictConfig", kind=(Literal["k"], "k"),
            InnerKnob=(inner, Field(default_factory=inner)),
        )
        with pytest.raises(ConfigRenderError, match=r"ConflictConfig\.InnerKnob: .*rename the field"):
            _render(model)

    def test_merged_duplicates_followed_by_more_submodels(self, cg2_cleanup) -> None:
        """Three identical ``direct`` providers merge into one class, and a later
        submodel's signature is still computed (regression)."""
        reg = create_registrar(
            f"{_CG2}nl9", _RunProto, discriminator_fields=["model_type", "provider"], key_separator=".",
        )

        def family(name: str) -> type:
            class _Base(metaclass=reg.Meta):
                __registry_key__ = name
                __abstract__ = True

                def run(self) -> None: ...

                def __init__(self, model: str = name):
                    self.model = model

            class _Direct(_Base):
                __registry_key__ = f"{name}.direct"

                def __init__(self, **kwargs: Any):
                    super().__init__(**kwargs)

            return _Base

        family("anthropic")
        family("google")
        openai_base = family("openai")

        class _Zeta(openai_base):  # type: ignore[misc, valid-type]
            __registry_key__ = "openai.zeta"

            def __init__(self, z: int = 1, **kwargs: Any):
                super().__init__(**kwargs)

        res = build_layer_config(reg)
        source = generate_layer_config_source(res)
        assert source.count("class DirectProviderConfig(BaseModel):") == 1
        assert "class ZetaProviderConfig(BaseModel):" in source
        assert_artifacts_agree(source, layer_expectations(res))

    def test_distinct_same_name_submodels_keep_their_schema_title(self, cg2_cleanup) -> None:
        res = build_layer_config(_nested_llm(f"{_CG2}nl8"))
        source = generate_layer_config_source(res)
        assert "class DirectProviderConfig2(BaseModel):" in source
        # The builder already names the second, different ``direct`` provider
        # uniquely (model names are unique per build), so no title is needed.
        provider = res.per_key_models["openai.direct"].model_fields["provider"].annotation
        assert provider.__name__ == "DirectProviderConfig2"
        assert "title='DirectProviderConfig'" not in source
        assert_artifacts_agree(source, layer_expectations(res), published=(
            generate_layer_json_schema(res), res.union_type,
        ))


class TestArtifactParity:
    """The generated source agrees with the in-process models (and the schema)."""

    @pytest.mark.parametrize("reg_name", [
        "_validity_reg", "_constraint_reg", "_desc_reg", "_enum_reg", "_path_reg",
        "_required_reg", "_optional_none_reg",
        # Single ``list[...]`` / ``dict[...]`` parameter (1.x crashed on 3.10).
        "_list_type_reg", "_dict_type_reg",
        "_any_type_reg", "_degraded_codegen_reg", "_agent_union_reg",
    ])
    def test_flat_layers(self, reg_name: str) -> None:
        if reg_name == "_degraded_codegen_reg":
            result = _build_degraded(globals()[reg_name])
        else:
            result = build_layer_config(globals()[reg_name])
        source = generate_layer_config_source(result)
        # Degraded layers too: the harness ignores the documented
        # ``[degraded]`` description hints of the published layer schema.
        published = (generate_layer_json_schema(result), result.union_type)
        assert_artifacts_agree(
            source, layer_expectations(result), published=published, plain_namespace=True,
        )

    def test_flat_layer_samples(self) -> None:
        result = build_layer_config(_validity_reg)
        source = generate_layer_config_source(result)
        assert_artifacts_agree(source, layer_expectations(result), samples={"LLMConfig": [
            {"provider": "alpha", "model_id": "m"},
            {"provider": "alpha", "model_id": "m", "temperature": 3},
            {"provider": "beta", "model_id": "m", "anything": 1},
            {"model_id": "m"},
        ]})

    def test_nested_layer(self, cg2_cleanup) -> None:
        result = build_layer_config(_nested_llm(f"{_CG2}nlp"))
        source = generate_layer_config_source(result)
        alias = _union_alias_name(f"{_CG2}nlp")
        assert f"\n{alias} = Annotated[" in source
        assert_artifacts_agree(source, layer_expectations(result), samples={alias: [
            {"model_type": "openai", "provider": {"name": "azure"}},
            {"model_type": "openai"},
            {"model_type": "anthropic", "provider": {"name": "direct", "organization": "o"}},
            {"model_type": "openai", "provider": {"name": "nope"}},
        ]}, published=(generate_layer_json_schema(result), result.union_type))

    def test_composed(self, cg2_cleanup) -> None:
        p = f"{_CG2}par_"
        res = build_composed_config(_archiver_and_hook(p), inline_wiring=True)
        source = generate_composed_config_source(res)
        assert_artifacts_agree(source, composed_expectations(res), samples={"ComposedConfig": [
            {f"{p}archiver": [{"archiver_id": "net"}], f"{p}llm": [{"model_type": "openai"}]},
            {f"{p}archiver": [{"host": "x"}]},
        ]}, exact_published=(generate_composed_json_schema(res), res.top_level_type))


# -------------------------------------------------------------------
# Inline-only layers, strict top level, plain-namespace exec (F3)
# -------------------------------------------------------------------


def _inline_only_layers(*, hook_inline_only: bool = False, env: bool = False) -> dict[str, Any]:
    """An inline-only ``archiver`` layer written under ``hook[trace_writer].archive``.

    ``hook_inline_only`` makes the hook layer inline-only too, written under
    ``env[browser].hooks`` (``env`` is added then).
    """
    archiver = create_registrar("archiver", _RunProto, discriminator_field="archiver_id", inline_only=True)
    hook = create_registrar("hook", _RunProto, discriminator_field="hook_id", inline_only=hook_inline_only)

    class _ArchBase(metaclass=archiver.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class Net(_ArchBase):
        __registry_key__ = "net"

        def __init__(self, port: Annotated[int, Field(gt=0)] = 80): ...

    class Disk(_ArchBase):
        __registry_key__ = "disk"

    class _HookBase(metaclass=hook.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    class TraceWriter(_HookBase):
        __registry_key__ = "trace_writer"
        __wiring__ = {"archive": "archiver"}

        def __init__(self, trace_dir: str = "traces", archive: Optional[list[_RunProto]] = None): ...

    layers = {"archiver": archiver, "hook": hook}
    if env or hook_inline_only:
        environment = create_registrar("env", _RunProto, discriminator_field="env_id")

        class _EnvBase(metaclass=environment.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Browser(_EnvBase):
            __registry_key__ = "browser"
            __wiring__ = {"hooks": "hook"}

            def __init__(self, hooks: Optional[list[_RunProto]] = None): ...

        layers["env"] = environment
    return layers


def _assert_composed_parity(result: Any, samples: list[Any]) -> Any:
    """The generated ``_composed.py`` agrees with ``result`` (module and plain dict)."""
    return assert_artifacts_agree(
        generate_composed_config_source(result),
        composed_expectations(result),
        samples={result.top_level_type.__name__: samples},
        exact_published=(generate_composed_json_schema(result), result.top_level_type),
        plain_namespace=True,
        min_accepted=1,
    )


class TestInlineOnlyRendering:
    """``_composed.py`` refuses inline-only layer names exactly like the in-process model."""

    def test_refusal_is_rendered(self, conscribe_isolated) -> None:
        result = build_composed_config(_inline_only_layers())
        source = generate_composed_config_source(result)
        assert "# Top-level extra: forbid" in source
        assert "# Inline-only layers: archiver" in source
        assert (
            "_INLINE_ONLY_WHERE: dict[str, str] = {\n"
            "    'archiver': 'hook[trace_writer].archive',\n"
            "}\n"
        ) in source
        assert _line(source, "archiver", in_class="ComposedConfig") == (
            "archiver: SkipJsonSchema[Any] = Field(default=None, exclude=True)"
        )
        assert "    @field_validator('archiver', mode='after')\n    @classmethod\n" in source
        assert "from pydantic_core import PydanticCustomError" in source
        assert source.rstrip().endswith("ComposedConfig.model_rebuild()")

    @pytest.mark.parametrize("mode", ["forbid", "ignore", "allow"])
    def test_refusal_agrees_in_every_mode(self, conscribe_isolated, mode: str) -> None:
        result = build_composed_config(_inline_only_layers(), top_level_extra=mode)
        source = generate_composed_config_source(result)
        assert {
            "forbid": "# Unknown top-level keys are refused (top_level_extra='forbid').",
            "ignore": "# Unknown top-level keys are dropped (top_level_extra='ignore').",
            "allow": "# Unknown top-level keys are kept in model_extra (top_level_extra='allow').",
        }[mode] in source
        assert "accepts extra parameters" not in source
        module = _assert_composed_parity(result, [
            {},
            {"archiver": []},
            {"archiver": None},
            {"archiver": {}},
            {"archiver": [{"archiver_id": "net"}]},
            {"zz_typo": []},
            {"archiver": [], "zz_typo": 1, "hook": [{"hook_id": "trace_writer", "archive": [{"port": 0}]}]},
            {"hook": [{"hook_id": "trace_writer", "archive": [{"archiver_id": "net", "port": 8}]}]},
        ])
        with pytest.raises(ValidationError) as exc:
            module.ComposedConfig.model_validate_json('{"archiver": null}')
        (error,) = exc.value.errors()
        assert (error["type"], error["loc"], error["msg"], error["ctx"]) == (
            "inline_only_layer",
            ("archiver",),
            "layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive",
            {"layer": "archiver", "where": "hook[trace_writer].archive"},
        )

    def test_full_paths_through_inline_only_owners(self, conscribe_isolated) -> None:
        result = build_composed_config(_inline_only_layers(hook_inline_only=True))
        source = generate_composed_config_source(result)
        assert "    'archiver': 'env[browser].hooks[trace_writer].archive',\n" in source
        assert "    'hook': 'env[browser].hooks',\n" in source
        assert "@field_validator('archiver', 'hook', mode='after')" in source
        _assert_composed_parity(result, [
            {"archiver": [], "hook": []},
            {"hook": None},
            {"env": [{"env_id": "browser", "hooks": [{"hook_id": "trace_writer", "archive": [{"archiver_id": "disk"}]}]}]},
        ])

    def test_layers_named_like_builtins(self, conscribe_isolated) -> None:
        """Field names assigned in the class body never shadow what the
        validator uses (``classmethod``, ``str``)."""
        sink = create_registrar("str", _RunProto, discriminator_field="sink_id", inline_only=True)
        owner = create_registrar("classmethod", _RunProto, discriminator_field="owner_id")

        class _SinkBase(metaclass=sink.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Disk(_SinkBase):
            __registry_key__ = "disk"

        class _OwnerBase(metaclass=owner.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Holder(_OwnerBase):
            __registry_key__ = "holder"
            __wiring__ = {"sinks": "str"}

            def __init__(self, sinks: Optional[list[_RunProto]] = None): ...

        result = build_composed_config({"str": sink, "classmethod": owner})
        source = generate_composed_config_source(result)
        assert "    @builtins.classmethod\n" in source
        assert "        layer = builtins.str(info.field_name)\n" in source
        _assert_composed_parity(result, [
            {"str": []}, {"classmethod": [{"owner_id": "holder", "sinks": [{"sink_id": "disk"}]}]},
        ])

    def test_refusal_without_its_where_table_is_an_error(self, conscribe_isolated) -> None:
        """A top-level model refusing a layer it does not list in
        ``x-inline-only-layers`` cannot be reproduced faithfully: loud."""
        import dataclasses as dc

        from pydantic import field_validator
        from pydantic.json_schema import SkipJsonSchema

        result = build_composed_config(_inline_only_layers())

        def _refuse_inline_only_layers(cls: Any, value: Any, info: Any) -> Any:
            raise ValueError("refused")

        broken = create_model(
            "ComposedConfig",
            __config__=ConfigDict(extra="forbid", json_schema_extra={"x-composed-layers": ["archiver"]}),
            __validators__={
                "_refuse_inline_only_layers": field_validator("archiver", mode="after")(
                    _refuse_inline_only_layers
                ),
            },
            archiver=(SkipJsonSchema[Any], Field(default=None, exclude=True)),
        )
        broken.__conscribe_generated__ = True  # type: ignore[attr-defined]
        with pytest.raises(ConfigRenderError, match=r"x-inline-only-layers.*could not name it"):
            generate_composed_config_source(dc.replace(result, top_level_type=broken))

    def test_no_inline_only_layer_no_validator(self, conscribe_isolated) -> None:
        layers = _inline_only_layers()
        result = build_composed_config({"hook": layers["hook"]})
        source = generate_composed_config_source(result)
        assert "_INLINE_ONLY_WHERE" not in source
        assert "field_validator" not in source
        assert "# Inline-only layers" not in source


class TestPlainNamespace:
    """Generated sources work exec'd into a plain dict, not only as a module."""

    def test_flat_nested_and_composed_sources(self, cg2_cleanup, conscribe_isolated) -> None:
        flat = build_layer_config(_multi_reg)
        nested = build_layer_config(_nested_llm(f"{_CG2}plain_nl"))
        composed = build_composed_config(_inline_only_layers(env=True))
        for result, source, expected in (
            (flat, generate_layer_config_source(flat), layer_expectations(flat)),
            (nested, generate_layer_config_source(nested), layer_expectations(nested)),
            (composed, generate_composed_config_source(composed), composed_expectations(composed)),
        ):
            namespace = exec_generated_namespace(source)
            assert "__name__" not in namespace
            for name in expected:
                assert namespace[name] is not None
            assert_artifacts_agree(source, expected, plain_namespace=True)


class TestNestedDiscriminatorNames:
    """Emitted compound discriminators carry the in-process function's name,
    which pydantic writes into union errors."""

    @staticmethod
    def _owner(registry: str) -> Any:
        agent = create_registrar(f"{_CG2}agent_dn", _RunProto, discriminator_field="name")

        class _Base(metaclass=agent.Meta):
            __abstract__ = True

            def run(self) -> None: ...

        class Pilot(_Base):
            __registry_key__ = "pilot"
            __wiring__ = {
                "llm": registry,
                "backup": (registry, ["openai.direct", "openai.azure"]),
            }

            def __init__(self, llm: str, backup: Optional[str] = None): ...

        return agent

    def test_layer_key_differs_from_registry_name(self, cg2_cleanup) -> None:
        registry = f"{_CG2}nl_registry"
        llm = _nested_llm(registry)
        result = build_composed_config({"llms": llm, "agents": self._owner(registry)})
        source = generate_composed_config_source(result)
        function = result.layer_results["llms"].union_type.__metadata__[0].discriminator.__name__
        assert function == f"_discriminate_{registry}"
        assert source.count(f"def {function}(v):") == 1
        assert source.count(f"Discriminator({function})") == 2  # layer alias + subset union
        assert "_discriminate_llms" not in source
        module = _assert_composed_parity(result, [
            {"agents": [{"name": "pilot", "llm": {"model_type": "openai", "provider": {"name": "azure"}}}]},
            {"agents": [{"name": "pilot", "llm": {"model_type": "nope"}}]},
            {"agents": [{"name": "pilot", "llm": {"model_type": "openai"}, "backup": {"model_type": "anthropic"}}]},
            {"agents": [{"name": "pilot", "llm": {}, "backup": {}}]},
            {"llms": [{"model_type": "zz"}]},
        ])
        with pytest.raises(ValidationError, match=rf"found using {function}\(\) does not match"):
            module.ComposedConfig.model_validate({"llms": [{"model_type": "zz"}]})

    def test_registrar_passed_twice_emits_its_function_once(self, cg2_cleanup) -> None:
        llm = _nested_llm(f"{_CG2}nl_twice")
        result = build_composed_config({"llms": llm, "llms_again": llm})
        source = generate_composed_config_source(result)
        assert source.count("def _discriminate_") == 1
        _assert_composed_parity(result, [
            {"llms": [{"model_type": "openai"}], "llms_again": [{"model_type": "zz"}]},
            {"llms_again": [{"model_type": "anthropic", "provider": {"name": "direct"}}]},
        ])


class TestFlattenedValidators:
    """Validators declared on generated classes the emitted class flattens."""

    @staticmethod
    def _guarded() -> type[BaseModel]:
        def _guard(cls: Any, data: Any) -> Any:
            return data

        base = create_model(
            "GuardedSchema",
            x=(int, 0),
            __validators__={"_guard": model_validator(mode="before")(_guard)},
        )
        base.__conscribe_generated__ = True  # type: ignore[attr-defined]
        model = create_model("GuardedConfig", __base__=base, kind=(Literal["k"], "k"))
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        return model

    def test_unrendered_validator_is_reported(self) -> None:
        with pytest.warns(ExtractionWarning, match=r"model validator '_guard' \(mode='before'\)"):
            source = _render(self._guarded())
        assert "#   GuardedConfig.<model>: model validator '_guard' (mode='before')" in source

    def test_rendered_and_inherited_validators_are_not_reported(self, conscribe_isolated) -> None:
        import warnings

        base = _helper_base("_KBase", UserSchema, extra="forbid")
        model = create_model("KProbeConfig", __base__=base, kind=(Literal["k"], "k"))
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtractionWarning)
            _render(model)  # UserSchema's validator: inherited by subclassing
            generate_composed_config_source(build_composed_config(_inline_only_layers()))


class TestLegacyFlatGuard:
    """D6: the extractor's old-shape guard is reproduced in generated files."""

    @staticmethod
    def _layer() -> Any:
        reg = create_registrar("legacy", _RunProto, discriminator_field="kind")

        @reg.register("one")
        class One:
            def run(self) -> None: ...

            def __init__(self, config: UserModel = UserModel()): ...

        @reg.register("two")
        class Two:
            def run(self) -> None: ...

            def __init__(self, knob: Annotated[UserModel, Field(description="the knob")] = UserModel()): ...

        @reg.register("free")
        class Free:
            def run(self) -> None: ...

            def __init__(self, config: UserModel = UserModel(), depth: int = 0): ...

        return reg

    def test_guard_is_rendered_once_per_file(self, conscribe_isolated) -> None:
        result = build_layer_config(self._layer())
        source = generate_layer_config_source(result)
        assert source.count("def _reject_legacy_flat_keys(cls: Any, data: Any) -> Any:") == 1
        def body(name: str) -> str:
            start = source.index(f"class {name}(")
            end = source.find("\nclass ", start + 1)
            return source[start:] if end == -1 else source[start:end]

        one = result.per_key_models["one"].__name__
        assert _line(source, "config", in_class=one) == "config: UserModel = UserModel()"
        assert "    __legacy_flat_keys__ = {'param': 'config', 'keys': ('x',)}\n" in body(one)
        assert (
            "    @model_validator(mode='before')\n"
            "    @classmethod\n"
            "    def _conscribe_reject_legacy_flat_keys(cls, data: Any) -> Any:\n"
            "        return _reject_legacy_flat_keys(cls, data)\n"
        ) in body(one)
        assert "__legacy_flat_keys__" not in body(result.per_key_models["free"].__name__)
        alias = _union_alias_name("legacy")
        module = assert_artifacts_agree(
            source, layer_expectations(result),
            samples={alias: [
                {"kind": "one", "x": 2},
                {"kind": "one", "config": {"x": 2}},
                {"kind": "two", "x": 2, "zz": 1},
                {"kind": "two", "knob": {"x": "q"}},
                {"kind": "free", "x": 2},
                {"kind": "one"},
            ]},
            published=(generate_layer_json_schema(result), result.union_type),
            plain_namespace=True,
            min_accepted=2,
        )
        with pytest.raises(ValidationError) as exc:
            getattr(module, one).model_validate({"x": 2})
        (error,) = exc.value.errors()
        assert error["type"] == "legacy_flat_config"
        assert error["msg"] == (
            "'config' is a nested block in conscribe 2.0; write {config: {...}} "
            "or set __config_from_param__ = 'config'"
        )

    def test_guard_without_its_table_is_an_error(self) -> None:
        from conscribe.config.extractor import _reject_legacy_flat_keys

        model = create_model(
            "UntabledConfig",
            kind=(Literal["k"], "k"),
            __validators__={
                "_conscribe_reject_legacy_flat_keys": model_validator(mode="before")(
                    _reject_legacy_flat_keys
                ),
            },
        )
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        with pytest.raises(ConfigRenderError, match=r"carries no __legacy_flat_keys__ table"):
            _render(model)

    def test_guard_survives_composed_rebuilds(self, conscribe_isolated) -> None:
        """A guarded owner whose model inline wiring rebuilds keeps its guard."""
        layers = _inline_only_layers()
        owner = create_registrar("legacy_owner", _RunProto, discriminator_field="kind")

        @owner.register("guarded")
        class Guarded:
            __wiring__ = {"archive": "archiver"}

            def run(self) -> None: ...

            def __init__(self, config: UserModel = UserModel()): ...

        result = build_composed_config({**layers, "legacy_owner": owner})
        model = result.layer_results["legacy_owner"].per_key_models["guarded"]
        assert "_conscribe_reject_legacy_flat_keys" in model.__pydantic_decorators__.model_validators
        _assert_composed_parity(result, [
            {"legacy_owner": [{"kind": "guarded", "x": 1}]},
            {"legacy_owner": [{"kind": "guarded", "config": {"x": 1}}]},
        ])


class TestSkipJsonSchemaSpelling:
    def test_type_and_field_forms(self) -> None:
        from pydantic.json_schema import SkipJsonSchema

        assert type_to_source(SkipJsonSchema[int]) == "SkipJsonSchema[int]"
        model = _generated(
            "SkipConfig",
            kind=(Literal["k"], "k"),
            hidden=(SkipJsonSchema[Optional[int]], None),
            shown=(Optional[int], None),
        )
        source = _render(model)
        assert _line(source, "hidden") == "hidden: SkipJsonSchema[Optional[int]] = None"
        assert "from pydantic.json_schema import SkipJsonSchema" in source
        assert_artifacts_agree(source, {"SkipConfig": model}, samples={"SkipConfig": [
            {}, {"hidden": 3}, {"hidden": "x"}, {"shown": 1},
        ]})


# ###################################################################
# Render review fixes (2.0.0): one regression test per finding
# ###################################################################

import os  # noqa: E402
import subprocess  # noqa: E402
import textwrap  # noqa: E402
import warnings  # noqa: E402

from pydantic import BeforeValidator, ImportString, Json, field_validator  # noqa: E402
from pydantic_core import core_schema  # noqa: E402

from tests.config.artifact_parity import validation_outcome  # noqa: E402


class Celsius(float):
    """A float with a custom repr: no source form, but a JSON form (R01)."""

    def __repr__(self) -> str:
        return f"{float(self)} degC"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.no_info_after_validator_function(cls, core_schema.float_schema())


class InstanceOnlyCelsius(Celsius):
    """Python input must already be an instance; JSON input may be a number."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(cls, core_schema.float_schema()),
            python_schema=core_schema.is_instance_schema(cls),
        )


def warm_up(value: Any) -> Any:
    """Not idempotent: one degree warmer on every validation."""
    return type(value)(value + 1)


def parse_celsius_text(value: Any) -> Any:
    """Accepts only text such as ``"21.5C"`` (a custom input format)."""
    if not isinstance(value, str):
        raise ValueError("write the temperature as text, e.g. '21.5C'")
    return float(value.rstrip("C"))


class DocumentedBase(BaseModel):
    """A user base contributing only config and a JSON-schema hook (R07)."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_: Any, handler: Any) -> Any:
        schema = handler(core_schema_)
        schema["description"] = "documented by the base"
        return schema


class _HashColor(enum.Enum):
    RED = "red"


class _HashSize(enum.Enum):
    BIG = "big"


class _HashMode(enum.Enum):
    FAST = "fast"


class TestJsonFormRung:
    """R01: the JSON form + ``validate_default`` rung is verified with the whole field."""

    def test_json_form_validates_back_to_the_default(self) -> None:
        model = _generated(
            "ThermoConfig",
            kind=(Literal["k"], "k"),
            temp=(Celsius, Celsius(21.5)),
            preset=(Celsius, Field(default=Celsius(4.0), validate_default=True)),
        )
        source = _render(model)
        assert _line(source, "temp") == "temp: Celsius = Field(default=21.5, validate_default=True)"
        assert _line(source, "preset") == "preset: Celsius = Field(default=4.0, validate_default=True)"
        module = assert_artifacts_agree(source, {"ThermoConfig": model}, samples={"ThermoConfig": [
            {}, {"temp": 3}, {"temp": "x"},
        ]})
        default = module.ThermoConfig().temp
        assert type(default) is Celsius and default == 21.5

    @pytest.mark.parametrize(("annotation", "default"), [
        (Annotated[Celsius, AfterValidator(warm_up)], Celsius(21.5)),  # validating changes it
        (Annotated[Celsius, BeforeValidator(parse_celsius_text)], Celsius(21.5)),  # refused
        (InstanceOnlyCelsius, InstanceOnlyCelsius(21.5)),  # python mode refuses the literal
    ], ids=["after-validator", "before-validator", "instance-only"])
    def test_json_form_the_field_would_not_reproduce_is_refused(self, annotation: Any, default: Any) -> None:
        model = _generated("ThermoConfig", kind=(Literal["k"], "k"), temp=(annotation, default))
        with pytest.raises(ConfigRenderError, match=r"ThermoConfig\.temp: .*no literal, constructor or JSON form"):
            _render(model)

    def test_field_level_validate_default_wins_over_the_model_config(self) -> None:
        """In-process, ``Field(validate_default=False)`` keeps the default as
        written even under ``model_config["validate_default"] = True``; a
        literal that validates to something else is refused."""
        model = create_model(
            "ConfiguredThermoConfig",
            __config__=ConfigDict(validate_default=True),
            kind=(Literal["k"], "k"),
            temp=(Annotated[Celsius, AfterValidator(warm_up)], Field(default=Celsius(21.5), validate_default=False)),
        )
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        assert model().temp == 21.5
        with pytest.raises(ConfigRenderError, match=r"ConfiguredThermoConfig\.temp: .*JSON form"):
            _render(model)

    def test_model_field_validators_keep_the_json_form_out(self) -> None:
        """A ``@field_validator`` would run on the generated default only."""

        def checked(cls: Any, value: Any) -> Any:
            return value

        model = create_model(
            "CheckedThermoConfig",
            kind=(Literal["k"], "k"),
            temp=(Celsius, Celsius(21.5)),
            __validators__={"_checked": field_validator("temp")(checked)},
        )
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        with pytest.raises(ConfigRenderError, match=r"CheckedThermoConfig\.temp: .*JSON form"):
            _render(model)


class TestReviewValueForms:
    """R02, R08, R09: values with constructor forms; NaN-aware verification."""

    def test_timezones(self) -> None:
        assert value_to_source(datetime.timezone.utc) == "timezone.utc"
        assert value_to_source(datetime.timezone(datetime.timedelta(hours=1))) == (
            "timezone(timedelta(seconds=3600))"
        )
        assert value_to_source(datetime.timezone(datetime.timedelta(hours=1), "CET")) == (
            "timezone(timedelta(seconds=3600), 'CET')"
        )
        # Equal to timezone.utc (offsets compare), but named: written with its name.
        assert value_to_source(datetime.timezone(datetime.timedelta(0), "UTC")) == "timezone(timedelta(0), 'UTC')"
        aware = datetime.datetime(2024, 5, 6, 7, 8, 9, 10, tzinfo=datetime.timezone.utc)
        assert value_to_source(aware) == "datetime(2024, 5, 6, 7, 8, 9, 10, tzinfo=timezone.utc)"
        folded = datetime.datetime(2024, 11, 3, 1, 30, fold=1)
        assert value_to_source(folded) == "datetime(2024, 11, 3, 1, 30, fold=1)"

    def test_compiled_patterns(self) -> None:
        assert value_to_source(re.compile(r"^[a-z]+$")) == "re.compile('^[a-z]+$')"
        assert value_to_source(re.compile("a+", re.IGNORECASE | re.MULTILINE)) == (
            "re.compile('a+', re.IGNORECASE | re.MULTILINE)"
        )
        assert value_to_source(re.compile(b"x.", re.DOTALL)) == "re.compile(b'x.', re.DOTALL)"
        model = _generated(
            "PatConfig",
            kind=(Literal["k"], "k"),
            code=(Annotated[str, Field(pattern=re.compile(r"^[a-z]+$"))], "abc"),
        )
        source = _render(model)
        (pattern,) = [getattr(m, "pattern") for m in model.model_fields["code"].metadata]
        if isinstance(pattern, re.Pattern):  # pydantic >= 2.8 keeps the compiled pattern
            assert _line(source, "code") == "code: str = Field(default='abc', pattern=re.compile('^[a-z]+$'))"
            assert "\nimport re\n" in source
        assert_artifacts_agree(source, {"PatConfig": model}, samples={"PatConfig": [
            {}, {"code": "ABC"}, {"code": "xyz"},
        ]})

    def test_nan_defaults_render_and_agree(self) -> None:
        model = _generated(
            "NanConfig",
            kind=(Literal["k"], "k"),
            values=(list[float], [math.nan, 1.0]),
            mapping=(dict[str, float], {"x": math.nan}),
            amount=(decimal.Decimal, decimal.Decimal("NaN")),
            pair=(tuple[float, float], (math.nan, -math.inf)),
        )
        source = _render(model)
        assert _line(source, "values") == "values: list[float] = [float('nan'), 1.0]"
        assert _line(source, "mapping") == "mapping: dict[str, float] = {'x': float('nan')}"
        assert _line(source, "amount") == "amount: Decimal = Decimal('NaN')"
        assert _line(source, "pair") == "pair: tuple[float, float] = (float('nan'), float('-inf'))"
        assert_artifacts_agree(source, {"NanConfig": model}, samples={"NanConfig": [{}, {"pair": [1, 2]}]})

    def test_messages_never_carry_placeholders(self) -> None:
        ctx = RenderContext()
        text = f"{ctx.ns.builtin('float')}('nan')"
        assert "\ue000" in text
        assert ctx.ns.readable(text) == "float('nan')"
        error = ctx.error(1, f"the rendered default {text} does not evaluate back to the value")
        assert "\ue000" not in str(error) and "\ue001" not in str(error)
        assert "float('nan')" in str(error)

    def test_conscribe_objects_are_refused_by_type(self) -> None:
        from conscribe.exceptions import WiringProblem

        problem = WiringProblem("X", "p", "r", "malformed", "d")
        model = _generated("LeakObjectConfig", kind=(Literal["k"], "k"), thing=(Any, Field(default=problem)))
        with pytest.raises(ConfigRenderError) as exc:
            _render(model)
        message = str(exc.value)
        assert message.startswith("LeakObjectConfig.thing: ")
        assert "conscribe's own type conscribe.exceptions.WiringProblem" in message
        assert "WiredField" not in message


class TestReviewMarkersAndBases:
    """R05, R06, R07, R13, R14: markers, bases and imports."""

    def test_json_and_import_string_keep_their_subscript_form(self) -> None:
        model = _generated(
            "MarkerConfig",
            kind=(Literal["k"], "k"),
            factory=(ImportString[Any], "math.cos"),
            payload=(Json[list[int]], Field(default="[1, 2]", validate_default=True)),
            hidden=(Optional[Annotated[Json[list[int]], Field(description="d")]], None),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtractionWarning)
            source = _render(model)
        assert _line(source, "factory") == "factory: ImportString[Any] = 'math.cos'"
        assert _line(source, "payload") == (
            "payload: Json[list[int]] = Field(default='[1, 2]', validate_default=True)"
        )
        assert "from pydantic import BaseModel, Field, ImportString, Json" in source
        module = assert_artifacts_agree(source, {"MarkerConfig": model}, samples={"MarkerConfig": [
            {"factory": "math.sin"}, {"factory": "nope.nope"}, {"payload": "[3]"}, {"payload": "x"},
            {"hidden": "[4]"},
        ]}, min_accepted=2)
        assert module.MarkerConfig(factory="math.sin").factory is math.sin

    def test_user_subclass_of_a_generated_model_is_imported(self, monkeypatch, cg2_cleanup) -> None:
        """R06: the marker is read from the class's own ``__dict__``."""
        store = create_registrar(f"{_CG2}r06_store", _RunProto, discriminator_field="kind")

        @store.register("store")
        class Store:
            def __init__(self, size: int = 1): ...

            def run(self) -> None: ...

        user = types.ModuleType("cs_cg2_r06_user")
        monkeypatch.setitem(sys.modules, user.__name__, user)
        user.Generated = build_layer_config(store).per_key_models["store"]  # type: ignore[attr-defined]
        exec(  # noqa: S102 -- a user module subclassing a generated model
            "from pydantic import model_validator\n"
            "class SmallStore(Generated):\n"
            "    @model_validator(mode='after')\n"
            "    def _small(self):\n"
            "        if self.size > 10:\n"
            "            raise ValueError('size too big')\n"
            "        return self\n",
            user.__dict__,
        )
        owner = create_registrar(f"{_CG2}r06_owner", _RunProto, discriminator_field="owner_kind")

        @owner.register("owner")
        class Owner:
            __config_schema__ = user.SmallStore  # type: ignore[attr-defined]

            def run(self) -> None: ...

        result = build_layer_config(owner)
        model = result.per_key_models["owner"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtractionWarning)
            source = generate_layer_config_source(result)
        assert f"class {model.__name__}(SmallStore):" in source
        assert "from cs_cg2_r06_user import SmallStore" in source
        assert_artifacts_agree(source, layer_expectations(result), samples={model.__name__: [
            {"size": 50}, {"size": 5}, {},
        ]}, min_accepted=1)

    def test_user_base_with_only_config_and_hooks_is_imported(self) -> None:
        """R07: a user base is never flattened, whatever it declares."""
        model = create_model("HookConfig", __base__=DocumentedBase, kind=(Literal["k"], "k"), n=(int, 1))
        model.__conscribe_generated__ = True  # type: ignore[attr-defined]
        source = _render(model)
        assert "class HookConfig(DocumentedBase):" in source
        module = assert_artifacts_agree(source, {"HookConfig": model}, samples={"HookConfig": [{}, {"zz": 1}]})
        assert module.HookConfig.model_json_schema()["description"] == "documented by the base"

    def test_modules_named_like_conscribe_are_searched(self, monkeypatch) -> None:
        """R13: only ``conscribe`` and ``conscribe.*`` are skipped."""
        plugins = types.ModuleType("conscribe_plugins_cg2")
        monkeypatch.setitem(sys.modules, plugins.__name__, plugins)
        exec("class _Mode:\n    def __repr__(self):\n        return 'FAST'\nFAST = _Mode()\n", plugins.__dict__)  # noqa: S102
        model = _generated("PluginModeConfig", kind=(Literal["k"], "k"), mode=(Any, plugins.FAST))  # type: ignore[attr-defined]
        source = _render(model)
        assert "from conscribe_plugins_cg2 import FAST" in source
        assert load_generated_module(source).PluginModeConfig().mode is plugins.FAST  # type: ignore[attr-defined]

    _WINDOW_MODULE = textwrap.dedent(
        '''
        from __future__ import annotations

        import datetime

        from pydantic import BaseModel


        class WindowSchema(BaseModel):
            """Future annotations + ``import datetime``: a very common module style."""

            start: datetime.datetime = datetime.datetime(2020, 1, 1)
        '''
    )

    def test_names_a_user_base_annotates_with_are_not_rebound(self, monkeypatch, cg2_cleanup) -> None:
        """R14: pydantic < 2.9 evaluates a base's string annotations in the
        subclassing module, where ``from datetime import datetime`` would make
        ``datetime.datetime`` fail; the import is aliased instead."""
        window = types.ModuleType("cs_cg2_window")
        monkeypatch.setitem(sys.modules, window.__name__, window)
        exec(compile(self._WINDOW_MODULE, "<cs_cg2_window>", "exec"), window.__dict__)  # noqa: S102
        reg = create_registrar(f"{_CG2}win", _RunProto, discriminator_field="kind")

        @reg.register("windowed")
        class Windowed:
            __config_schema__ = window.WindowSchema  # type: ignore[attr-defined]

            def run(self) -> None: ...

        @reg.register("stamped")
        class Stamped:
            def __init__(self, at: datetime.datetime = datetime.datetime(2021, 1, 1)): ...

            def run(self) -> None: ...

        result = build_layer_config(reg)
        source = generate_layer_config_source(result)
        assert "from datetime import datetime as datetime__datetime" in source
        assert f"class {result.per_key_models['windowed'].__name__}(WindowSchema):" in source
        alias = _union_alias_name(f"{_CG2}win")
        assert_artifacts_agree(source, layer_expectations(result), samples={alias: [
            {"kind": "windowed"}, {"kind": "windowed", "start": "2022-01-01T00:00:00"},
            {"kind": "stamped", "at": "x"}, {"kind": "stamped"},
        ]}, plain_namespace=True, min_accepted=2)


class TestReviewNames:
    """R10, R11, R12, R15: names in the generated module."""

    def test_annotation_only_field_named_like_a_builtin(self) -> None:
        """R15: ``frozenset: bytes`` never shadows the builtin a later default uses."""
        model = _generated(
            "PluginConfig",
            kind=(Literal["k"], "k"),
            frozenset=(bytes, ...),
            tags=(frozenset[str], frozenset({"a"})),
        )
        source = _render(model)
        assert _line(source, "frozenset") == "frozenset: bytes"
        assert _line(source, "tags") == "tags: builtins.frozenset[str] = builtins.frozenset({'a'})"
        assert "\nimport builtins\n" in source
        assert_artifacts_agree(source, {"PluginConfig": model}, samples={"PluginConfig": [
            {"frozenset": b"x"}, {},
        ]}, plain_namespace=True)

    def test_annotation_only_field_named_like_a_generated_class(self) -> None:
        """Only an *assigned* field shadows a generated class at runtime (that is
        refused, see ``test_field_named_like_a_generated_class_raises``); an
        annotation-only one is written as is."""
        inner = _generated("InnerKnob", z=(int, 0))
        model = _generated("NamedLikeConfig", kind=(Literal["k"], "k"), knob=(inner, ...), InnerKnob=(int, ...))
        source = _render(model)
        assert _line(source, "InnerKnob") == "InnerKnob: int"
        assert_artifacts_agree(source, {"NamedLikeConfig": model, "InnerKnob": inner}, samples={
            "NamedLikeConfig": [{"knob": {"z": 1}, "InnerKnob": 2}, {"InnerKnob": 2}],
        })

    def test_user_base_attributes_get_the_matching_ignore(self) -> None:
        """R10: overriding a method/property vs a class variable of the base."""

        class _Rules(BaseModel):
            limit: typing.ClassVar[int] = 3

            @property
            def label(self) -> str:
                return "x"

        from conscribe.config._render import _override_rule

        assert _override_rule(_Rules, "label") == "reportIncompatibleMethodOverride"
        assert _override_rule(_Rules, "limit") == "reportIncompatibleVariableOverride"
        assert _override_rule(_Rules, "kind") is None
        assert _override_rule(BaseModel, "schema") == "reportIncompatibleMethodOverride"

    def test_messages_name_the_key_layer_and_fix(self) -> None:
        """R11: identifiers the file cannot use are named with their origin."""
        bad = create_model("Gpt-4oConfig", kind=(Literal["gpt-4o"], "gpt-4o"))
        bad.__conscribe_generated__ = True  # type: ignore[attr-defined]
        with pytest.raises(ConfigRenderError, match=(
            r"for key 'gpt-4o' of layer 'probe' is named 'Gpt-4oConfig', which is not a valid "
            r"Python identifier"
        )):
            generate_layer_config_source(_layer_of(bad, key="gpt-4o"))

        keyword_field = _generated("KeywordConfig", kind=(Literal["k"], "k"), **{"global": (int, 1)})
        with pytest.raises(ConfigRenderError, match=(
            r"KeywordConfig\.global: .*'global' is a Python keyword, .*Field\(alias='global'\)"
        )):
            _render(keyword_field)

        a = _generated("AConfig", kind=(Literal["a"], "a"))
        b = _generated("BConfig", kind=(Literal["b"], "b"))
        dashed = LayerConfigResult(
            union_type=Annotated[Union[a, b], Field(discriminator="kind")],
            per_key_models={"a": a, "b": b},
            layer_name="my-llm",
            discriminator_field="kind",
        )
        with pytest.raises(ConfigRenderError, match=(
            r"'My-LlmConfig' derived from layer name 'my-llm' is not a valid Python identifier; "
            r"rename the layer"
        )):
            generate_layer_config_source(dashed)

    def test_layer_named_composed_keeps_the_top_level_name(self, conscribe_isolated) -> None:
        """R12: the top-level model is always ``ComposedConfig``."""
        reg = create_registrar("composed", _RunProto, discriminator_field="kind")

        @reg.register("a")
        class A:
            def run(self) -> None: ...

        @reg.register("b")
        class B:
            def __init__(self, n: int = 1): ...

            def run(self) -> None: ...

        result = build_composed_config({"composed": reg})
        source = generate_composed_config_source(result)
        assert "class ComposedConfig(BaseModel):" in source
        assert "\nComposedConfig2 = Annotated[" in source
        assert "composed: list[ComposedConfig2]" in source
        assert_artifacts_agree(source, composed_expectations(result), samples={"ComposedConfig": [
            {"composed": [{"kind": "a"}, {"kind": "b", "n": 2}]}, {"composed": [{"kind": "zz"}]}, {},
        ]}, exact_published=(generate_composed_json_schema(result), result.top_level_type),
            plain_namespace=True, min_accepted=1)


class TestReviewNestedDiscriminator:
    """M4: the emitted compound discriminator mirrors the builder's."""

    def test_non_string_parts_and_omitted_defaulted_blocks(self, cg2_cleanup) -> None:
        result = build_layer_config(_nested_llm(f"{_CG2}nl_m4"))
        source = generate_layer_config_source(result)
        assert "    if not all(isinstance(part, str) for part in parts):\n        return None\n" in source
        alias = _union_alias_name(f"{_CG2}nl_m4")
        non_string = [
            {"model_type": 1}, {"model_type": None}, {"model_type": ["openai"]},
            {"model_type": "openai", "provider": {"name": 3}},
        ]
        omitted = [
            {"model_type": "anthropic"},  # the one defaulted ``anthropic`` block
            {"model_type": "anthropic", "provider": {}},
            {"model_type": "anthropic", "provider": None},
            {"model_type": "openai"},  # two defaulted ``openai`` blocks: no tag
            {"model_type": "openai", "provider": "azure"},
        ]
        module = assert_artifacts_agree(
            source, layer_expectations(result), samples={alias: [*non_string, *omitted]},
            plain_namespace=[alias], min_accepted=1,
        )
        for payload in non_string:
            outcome = validation_outcome(getattr(module, alias), payload)
            assert outcome[0] == "error" and [e[0] for e in outcome[1]] == ["union_tag_not_found"]
        accepted = validation_outcome(getattr(module, alias), {"model_type": "anthropic"})
        assert accepted[0] == "ok" and accepted[1]["provider"]["name"] == "direct"


class TestReviewLayout:
    """Layout code turns internal failures into ConfigRenderError."""

    def test_union_member_not_emitted_is_a_config_render_error(self) -> None:
        a = _generated("AConfig", kind=(Literal["a"], "a"))
        b = _generated("BConfig", kind=(Literal["b"], "b"))
        stray = _generated("StrayConfig", kind=(Literal["c"], "c"))
        result = LayerConfigResult(
            union_type=Annotated[Union[a, b, stray], Field(discriminator="kind")],
            per_key_models={"a": a, "b": b},
            layer_name="probe",
            discriminator_field="kind",
        )
        with pytest.raises(ConfigRenderError, match=r"StrayConfig\.\(union\): .*not emitted"):
            generate_layer_config_source(result)

    def test_same_named_models_of_a_hand_assembled_result(self) -> None:
        """The builders name models uniquely; a hand-assembled result with two
        models of one name still renders: the second is suffixed and keeps
        its in-process name as schema title."""
        first = _generated("SameConfig", kind=(Literal["a"], "a"), x=(int, 1))
        second = _generated("SameConfig", kind=(Literal["b"], "b"), y=(str, "s"))
        union = Annotated[Union[first, second], Field(discriminator="kind")]
        result = LayerConfigResult(
            union_type=union, per_key_models={"a": first, "b": second},
            layer_name="probe", discriminator_field="kind",
        )
        source = generate_layer_config_source(result)
        assert "class SameConfig(BaseModel):" in source
        assert "class SameConfig2(BaseModel):" in source
        assert "model_config = ConfigDict(title='SameConfig')" in source
        assert_artifacts_agree(
            source, {"SameConfig": first, "SameConfig2": second, _union_alias_name("probe"): union},
            samples={_union_alias_name("probe"): [{"kind": "a"}, {"kind": "b", "y": 1}, {"kind": "c"}]},
        )


def _hash_order_layers() -> dict[str, Any]:
    """Layers whose artifacts hold sets in every position R03 found (module
    level: the hash-seed test builds them in subprocesses)."""
    llm = create_registrar(
        f"{_CG2}r03_llm", _RunProto, discriminator_fields=["family", "provider"], key_separator=".",
    )

    class _OpenAI(metaclass=llm.Meta):
        __registry_key__ = "openai"
        __abstract__ = True

        def __init__(self, temperature: float = 0.7): ...

        def run(self) -> None: ...

    class Azure(_OpenAI):
        __registry_key__ = "openai.azure"

        def __init__(
            self, tools: frozenset[str] = frozenset({"search", "browse", "code", "files", "shell"}), **kwargs: Any,
        ):
            super().__init__(**kwargs)

    class Official(_OpenAI):
        __registry_key__ = "openai.official"

        def __init__(self, scopes: set[str] = {"read", "write", "admin", "audit"}, **kwargs: Any):  # noqa: B006
            super().__init__(**kwargs)

    owner = create_registrar(f"{_CG2}r03_owner", _RunProto, discriminator_field="oid")

    @owner.register("owner")
    class Owner:
        def __init__(
            self,
            marks: frozenset[Any] = frozenset({_HashColor.RED, _HashSize.BIG, _HashMode.FAST}),
            groups: frozenset[frozenset[str]] = frozenset({frozenset({"b2", "b1"}), frozenset({"a2", "a1"})}),
        ): ...

        def run(self) -> None: ...

    return {f"{_CG2}r03_llm": llm, f"{_CG2}r03_owner": owner}


_HASH_ORDER_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from conscribe.config.builder import build_layer_config
    from conscribe.config.codegen import generate_composed_config_source, generate_layer_config_source
    from conscribe.config.composed import build_composed_config
    from conscribe.config.json_schema import generate_composed_json_schema, generate_layer_json_schema
    from conscribe.testing import isolated_registries
    import tests.config.test_codegen as cg

    with isolated_registries():
        layers = cg._hash_order_layers()
        for registrar in layers.values():
            result = build_layer_config(registrar)
            sys.stdout.write(generate_layer_config_source(result))
            sys.stdout.write(json.dumps(generate_layer_json_schema(result), indent=1))
        composed = build_composed_config(layers)
        sys.stdout.write(generate_composed_config_source(composed))
        sys.stdout.write(json.dumps(generate_composed_json_schema(composed), indent=1))
    """
)


def test_set_valued_artifacts_do_not_depend_on_the_hash_seed() -> None:
    """R03: segment default instances (source ``json_schema_extra`` and the
    published schema) and set members of several enum classes are written in
    an order independent of ``PYTHONHASHSEED``."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outputs = []
    for seed in ("1", "2", "3"):
        proc = subprocess.run(
            [sys.executable, "-c", _HASH_ORDER_SCRIPT],
            cwd=root, env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert "'tools': ['browse', 'code', 'files', 'search', 'shell']" in outputs[0]
    assert "frozenset({_HashColor.RED, _HashMode.FAST, _HashSize.BIG})" in outputs[0]
    assert "frozenset({frozenset({'a1', 'a2'}), frozenset({'b1', 'b2'})})" in outputs[0]
    assert outputs[0] == outputs[1] == outputs[2]


def test_set_valued_defaults_agree_in_process(cg2_cleanup) -> None:
    """R03: the json_schema_extra default the file writes for a segment
    default instance is the published schema's (same canonical order)."""
    layers = _hash_order_layers()
    result = build_layer_config(layers[f"{_CG2}r03_llm"])
    source = generate_layer_config_source(result)
    assert "json_schema_extra={'default': {'name': 'azure', 'tools': ['browse', 'code', 'files', 'search', 'shell']}}" in source
    assert_artifacts_agree(
        source, layer_expectations(result),
        published=(generate_layer_json_schema(result), result.union_type),
        plain_namespace=True,
    )


def _pyright_review_model() -> type[BaseModel]:
    """Constructs of the render review that once broke type checking (stdlib
    and pydantic types only: the file must resolve outside this repo)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pydantic: fields shadow BaseModel methods
        return _generated(
            "ReviewConfig",
            kind=(Literal["k"], "k"),
            schema=(str, "public"),  # R10: BaseModel method names
            validate=(bool, True),
            copy=(bool, False),
            json=(int, 1),
            frozenset=(bytes, ...),  # R15: annotation-only names used later in the class
            tags=(frozenset[str], frozenset({"a"})),
            date=(datetime.date, ...),
            day=(datetime.date, datetime.date(2020, 1, 1)),
            payload=(Json[list[int]], ...),  # R05
            factory=(ImportString[Any], "math.cos"),
            code=(Annotated[str, Field(pattern=re.compile("^[a-z]+$"))], "abc"),  # R08
            at=(datetime.datetime, datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)),  # R02
            ratios=(list[float], [math.nan]),  # R09
        )


@pytest.mark.skipif(
    __import__("shutil").which("pyright") is None, reason="pyright is not on PATH",
)
def test_generated_sources_type_check_with_pyright(tmp_path, cg2_cleanup) -> None:
    """Optional: generated files are pyright-clean (no ``EllipsisType`` errors)."""
    import subprocess

    regs = _archiver_and_hook(f"{_CG2}pyr_")
    files = {
        "flat_cfg.py": generate_layer_config_source(build_layer_config(_validity_reg)),
        "nested_cfg.py": generate_layer_config_source(build_layer_config(regs[f"{_CG2}pyr_llm"])),
        "composed_cfg.py": generate_composed_config_source(build_composed_config(regs)),
        "review_cfg.py": _render(_pyright_review_model()),
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    proc = subprocess.run(
        ["pyright", "--pythonpath", sys.executable, *files],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


class TestCyclicDefaultIsARenderError:
    """A default that contains itself has no source form: ConfigRenderError, not RecursionError."""

    def test_self_referencing_list_default(self) -> None:
        from conscribe.exceptions import ConfigRenderError
        from conscribe.testing import isolated_registries

        loop: list = []
        loop.append(loop)
        with isolated_registries():
            reg = create_registrar("cyclic_default", _LLMProto, discriminator_field="kind")

            class Base(metaclass=reg.Meta):
                __abstract__ = True

                async def chat(self, messages: list[dict]) -> str:
                    return ""

            class Holder(Base):
                __registry_key__ = "holder"

                def __init__(self, items: list = loop) -> None: ...

            result = build_layer_config(reg)
            with pytest.raises(ConfigRenderError, match="contains itself"):
                generate_layer_config_source(result)
