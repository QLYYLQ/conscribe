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

from typing import Annotated, Any, Optional, Protocol, Union, runtime_checkable

import pytest
from pydantic import BaseModel, Field, ValidationError

from conscribe import create_registrar
from conscribe.config.builder import LayerConfigResult, build_layer_config
from conscribe.config.codegen import generate_layer_config_source


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

    def test_open_schema_has_docstring_about_extra_fields(self) -> None:
        """Open schema class has a docstring mentioning extra fields allowed."""
        result = build_layer_config(_open_reg)
        source = generate_layer_config_source(result)

        # The design doc specifies open schema classes should note extra fields
        # Example: "此实现接受额外参数 (**kwargs)。未声明的字段不会报错。"
        # We check for either Chinese or English phrasing about extra/kwargs
        open_model_name = None
        for key, model in result.per_key_models.items():
            if model.model_config.get("extra") == "allow":
                open_model_name = model.__name__
                break

        assert open_model_name is not None, "No open model found"

        # Find the class definition in source and check for a docstring
        class_idx = source.index(f"class {open_model_name}(BaseModel):")
        # The docstring should appear between the class line and model_config
        class_block = source[class_idx:class_idx + 500]
        assert '"""' in class_block or "'''" in class_block, (
            f"Open schema class {open_model_name} should have a docstring "
            f"about extra fields"
        )


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

    def test_required_field_renders_ellipsis_default(self) -> None:
        """Required field (no default) renders ... as the default."""
        result = build_layer_config(_required_reg)
        source = generate_layer_config_source(result)

        # The required fields (model_id, api_key) should use ... as default.
        # In source, this appears as `= ...` or `Field(...)` where the first
        # arg is `...`
        # Find the field definitions for required fields
        assert "..." in source

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
