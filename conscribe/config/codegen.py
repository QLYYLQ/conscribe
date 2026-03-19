"""Python source code generation for config schemas.

Serializes a ``LayerConfigResult`` into a valid, self-contained Python
source file string containing Pydantic ``BaseModel`` class definitions
and an optional discriminated union type alias.

See ``config-typing-design.md`` Section 6.1 for specification.
"""
from __future__ import annotations

import enum
import sys
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from conscribe.config.builder import LayerConfigResult

# types.UnionType is the runtime type for PEP 604 `X | Y` syntax (Python 3.10+)
_UNION_TYPES: tuple[type, ...] = (Union,)
if sys.version_info >= (3, 10):
    _UNION_TYPES = (Union, types.UnionType)

# Constraint types from annotated_types (used by Pydantic's Annotated metadata)
try:
    from annotated_types import Ge, Gt, Le, Lt
    _CONSTRAINT_MAP = {Ge: "ge", Gt: "gt", Le: "le", Lt: "lt"}
except ImportError:
    _CONSTRAINT_MAP = {}


def generate_layer_config_source(result: LayerConfigResult) -> str:
    """Serialize a ``LayerConfigResult`` to a valid Python source file string.

    Output structure:
    1. Auto-generated header comment (with degradation warnings if any)
    2. ``from __future__ import annotations``
    3. Required imports (typing, pydantic)
    4. Per-key class definitions (for nested mode: segment models first)
    5. Union type alias (if multiple models)

    Args:
        result: The ``LayerConfigResult`` from ``build_layer_config()``.

    Returns:
        A string containing valid Python source code.
    """
    if result.discriminator_fields:
        return _generate_nested_source(result)
    return _generate_flat_source(result)


def _generate_flat_source(result: LayerConfigResult) -> str:
    """Generate flat mode source code (original behavior)."""
    parts: list[str] = []

    degraded = result.degraded_fields

    # 1. Header
    parts.append(_generate_header(result, degraded_fields=degraded))

    # 2. Future annotations
    parts.append("from __future__ import annotations\n")

    # 3. Imports
    parts.append(_generate_imports(result, has_degraded=bool(degraded)))

    # 4. Per-key class definitions
    model_names = []
    for key in sorted(result.per_key_models.keys()):
        model = result.per_key_models[key]
        key_degraded = degraded.get(key, [])
        parts.append(
            _generate_class(model, result.discriminator_field, degraded_list=key_degraded)
        )
        model_names.append(model.__name__)

    # 5. model_rebuild() calls
    rebuild_lines = [f"\n# Rebuild models for deferred annotation resolution"]
    for name in model_names:
        rebuild_lines.append(f"{name}.model_rebuild()")
    rebuild_lines.append("")
    parts.append("\n".join(rebuild_lines))

    # 6. Union type alias (only for multiple models)
    if len(result.per_key_models) > 1:
        parts.append(_generate_union_alias(result))

    return "\n".join(parts)


def _generate_nested_source(result: LayerConfigResult) -> str:
    """Generate nested mode source code with compound discriminator."""
    parts: list[str] = []
    degraded = result.degraded_fields
    disc_fields = result.discriminator_fields
    separator = result.key_separator

    # 1. Header
    parts.append(_generate_nested_header(result, degraded_fields=degraded))

    # 2. Future annotations
    parts.append("from __future__ import annotations\n")

    # 3. Imports (nested mode needs Discriminator and Tag)
    parts.append(_generate_nested_imports(result, has_degraded=bool(degraded)))

    # 4. Nested segment models (level 1+)
    all_model_names: list[str] = []
    if result.per_segment_models:
        parts.append("# ── Nested segment models (level 1+) ──")
        for field_name in disc_fields[1:]:
            segment_models = result.per_segment_models.get(field_name, {})
            for seg in sorted(segment_models.keys()):
                model = segment_models[seg]
                parts.append(_generate_class(model, "name"))
                all_model_names.append(model.__name__)

    # 5. Combined models (level 0 flat + level 1 nested)
    parts.append("# ── Combined models (level 0 flat + level 1 nested) ──")
    for key in sorted(result.per_key_models.keys()):
        model = result.per_key_models[key]
        key_degraded = degraded.get(key, [])
        parts.append(_generate_class(model, disc_fields[0], degraded_list=key_degraded))
        all_model_names.append(model.__name__)

    # 6. model_rebuild() calls
    rebuild_lines = [f"\n# Rebuild models for deferred annotation resolution"]
    for name in all_model_names:
        rebuild_lines.append(f"{name}.model_rebuild()")
    rebuild_lines.append("")
    parts.append("\n".join(rebuild_lines))

    # 7. Discriminator function
    parts.append(_generate_discriminator_fn(result))

    # 8. Union
    if len(result.per_key_models) > 1:
        parts.append(_generate_nested_union_alias(result))

    return "\n".join(parts)


def _generate_nested_header(
    result: LayerConfigResult,
    degraded_fields: dict[str, list[Any]] | None = None,
) -> str:
    """Generate header for nested mode."""
    count = len(result.per_key_models)
    keys = ", ".join(sorted(result.per_key_models.keys()))
    disc_fields_str = ", ".join(result.discriminator_fields or [])
    lines = [
        f'# Auto-generated by conscribe. DO NOT EDIT.',
        f'# Layer: {result.layer_name}',
        f'# Discriminator fields: {disc_fields_str}',
        f'# Key separator: {result.key_separator}',
        f'# Entries ({count}): {keys}',
    ]

    if degraded_fields:
        lines.append('#')
        lines.append(
            '# WARNING: The following fields had types incompatible with config'
        )
        lines.append('# serialization and were degraded to Any:')
        for key in sorted(degraded_fields.keys()):
            field_strs = [
                f"{df.field_name} (was: {df.original_type_repr})"
                for df in degraded_fields[key]
            ]
            lines.append(f'#   {key}: {", ".join(field_strs)}')

    lines.append('')
    return '\n'.join(lines)


def _generate_nested_imports(
    result: LayerConfigResult,
    has_degraded: bool = False,
) -> str:
    """Generate imports for nested mode."""
    typing_imports = {"Annotated", "Literal", "Union"}
    pydantic_imports = {"BaseModel", "ConfigDict", "Discriminator", "Tag"}

    if has_degraded:
        typing_imports.add("Any")

    # Scan for Field usage and non-builtin types
    has_field = False
    extra_imports: dict[str, set[str]] = {}
    all_models = list(result.per_key_models.values())
    if result.per_segment_models:
        for seg_models in result.per_segment_models.values():
            all_models.extend(seg_models.values())

    for model in all_models:
        for field_info in model.model_fields.values():
            if _needs_field(field_info):
                has_field = True
            tp = field_info.annotation
            _collect_type_imports(tp, typing_imports)
            _collect_non_builtin_types(tp, extra_imports)
            if isinstance(field_info.default, enum.Enum):
                enum_type = type(field_info.default)
                mod = getattr(enum_type, "__module__", None)
                name = enum_type.__name__
                if mod and mod != "builtins":
                    extra_imports.setdefault(mod, set()).add(name)

    if has_field:
        pydantic_imports.add("Field")

    lines = []
    lines.append(f"from typing import {', '.join(sorted(typing_imports))}")
    lines.append("")
    lines.append(f"from pydantic import {', '.join(sorted(pydantic_imports))}")
    for mod in sorted(extra_imports.keys()):
        names = ", ".join(sorted(extra_imports[mod]))
        lines.append(f"from {mod} import {names}")
    lines.append("")
    return "\n".join(lines)


def _generate_discriminator_fn(result: LayerConfigResult) -> str:
    """Generate the compound discriminator function for nested mode."""
    disc_fields = result.discriminator_fields
    separator = result.key_separator
    layer_name = result.layer_name

    lines = ["\n# ── Discriminator ──\n"]
    lines.append(f"def _discriminate_{layer_name}(v):")

    # Level 0: flat field
    lines.append(f"    if isinstance(v, dict):")
    lines.append(f'        _l0 = v.get("{disc_fields[0]}", "")')
    lines.append(f"    else:")
    lines.append(f"        _l0 = v.{disc_fields[0]}")

    # Level 1+: nested fields — traverse into nested dicts/models
    for i in range(1, len(disc_fields)):
        # Build nested traversal for dict path
        dict_chain = "v"
        for j in range(1, i + 1):
            dict_chain = f'{dict_chain}.get("{disc_fields[j]}", {{}}) if isinstance({dict_chain}, dict) else {{}}'
        # Build nested traversal for model path
        model_chain = "v"
        for j in range(1, i + 1):
            model_chain = f"getattr({model_chain}, '{disc_fields[j]}', None)"

        lines.append(f"    if isinstance(v, dict):")
        lines.append(f"        _nested{i} = {dict_chain}")
        lines.append(f'        _l{i} = _nested{i}.get("name", "default") if isinstance(_nested{i}, dict) else "default"')
        lines.append(f"    else:")
        lines.append(f"        _obj{i} = {model_chain}")
        lines.append(f'        _l{i} = getattr(_obj{i}, "name", "default") if _obj{i} else "default"')

    # Return joined string
    parts_str = ", ".join(f"_l{i}" for i in range(len(disc_fields)))
    lines.append(f'    return f"{{_l0}}' + "".join(
        f'{separator}{{_l{i}}}' for i in range(1, len(disc_fields))
    ) + '"')
    lines.append("")

    return "\n".join(lines)


def _generate_nested_union_alias(result: LayerConfigResult) -> str:
    """Generate the union alias for nested mode using Discriminator(callable)."""
    layer_name = result.layer_name
    if len(layer_name) <= 3:
        alias_name = f"{layer_name.upper()}Config"
    else:
        alias_name = f"{layer_name.title()}Config"

    members = []
    for key in sorted(result.per_key_models.keys()):
        model_name = result.per_key_models[key].__name__
        members.append(f'        Annotated[{model_name}, Tag("{key}")]')

    members_str = ",\n".join(members)
    fn_name = f"_discriminate_{layer_name}"

    return (
        f'\n{alias_name} = Annotated[\n'
        f'    Union[\n{members_str},\n    ],\n'
        f'    Discriminator({fn_name}),\n'
        f']\n'
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_header(
    result: LayerConfigResult,
    degraded_fields: dict[str, list[Any]] | None = None,
) -> str:
    """Generate the auto-generated header comment."""
    count = len(result.per_key_models)
    keys = ", ".join(sorted(result.per_key_models.keys()))
    lines = [
        f'# Auto-generated by conscribe. DO NOT EDIT.',
        f'# Layer: {result.layer_name}',
        f'# Discriminator: {result.discriminator_field}',
        f'# Entries ({count}): {keys}',
    ]

    if degraded_fields:
        lines.append('#')
        lines.append(
            '# WARNING: The following fields had types incompatible with config'
        )
        lines.append('# serialization and were degraded to Any:')
        for key in sorted(degraded_fields.keys()):
            field_strs = [
                f"{df.field_name} (was: {df.original_type_repr})"
                for df in degraded_fields[key]
            ]
            lines.append(f'#   {key}: {", ".join(field_strs)}')

    lines.append('')
    return '\n'.join(lines)


def _generate_imports(result: LayerConfigResult, has_degraded: bool = False) -> str:
    """Generate the import block."""
    # Collect which typing imports are needed
    typing_imports = {"Literal"}
    pydantic_imports = {"BaseModel", "ConfigDict"}

    if has_degraded:
        typing_imports.add("Any")

    # Scan fields to determine needed imports
    has_field = False
    extra_imports: dict[str, set[str]] = {}
    for model in result.per_key_models.values():
        for field_name, field_info in model.model_fields.items():
            tp = field_info.annotation
            _collect_type_imports(tp, typing_imports)
            _collect_non_builtin_types(tp, extra_imports)
            if _needs_field(field_info):
                has_field = True
            # Check enum defaults
            if isinstance(field_info.default, enum.Enum):
                enum_type = type(field_info.default)
                mod = getattr(enum_type, "__module__", None)
                name = enum_type.__name__
                if mod and mod != "builtins":
                    extra_imports.setdefault(mod, set()).add(name)

    if has_field:
        pydantic_imports.add("Field")

    # Check if Union alias is needed
    if len(result.per_key_models) > 1:
        typing_imports.add("Annotated")
        typing_imports.add("Union")
        pydantic_imports.add("Field")

    lines = []
    lines.append(f"from typing import {', '.join(sorted(typing_imports))}")
    lines.append("")
    lines.append(f"from pydantic import {', '.join(sorted(pydantic_imports))}")
    # Extra imports for non-builtin types
    for mod in sorted(extra_imports.keys()):
        names = ", ".join(sorted(extra_imports[mod]))
        lines.append(f"from {mod} import {names}")
    lines.append("")
    return "\n".join(lines)


def _collect_type_imports(tp: Any, imports: set[str]) -> None:
    """Collect typing imports needed for a type annotation."""
    origin = get_origin(tp)
    if origin in _UNION_TYPES:
        args = get_args(tp)
        if len(args) == 2 and type(None) in args:
            imports.add("Optional")
        else:
            imports.add("Union")
        for arg in args:
            if arg is not type(None):
                _collect_type_imports(arg, imports)
    elif origin is Annotated:
        args = get_args(tp)
        if args:
            _collect_type_imports(args[0], imports)
    elif tp is Any:
        imports.add("Any")
    elif origin is not None:
        # Generic like list[T], dict[K,V] — recurse into args
        for arg in get_args(tp):
            _collect_type_imports(arg, imports)


def _collect_non_builtin_types(tp: Any, extra_imports: dict[str, set[str]]) -> None:
    """Collect non-builtin type imports needed for source generation.

    Populates extra_imports as {module: {name, ...}} for types whose
    ``__module__`` is not ``builtins`` or ``typing``.
    """
    origin = get_origin(tp)
    if origin in _UNION_TYPES or origin is Annotated:
        for arg in get_args(tp):
            if arg is not type(None):
                _collect_non_builtin_types(arg, extra_imports)
        return
    if origin is Literal:
        return
    if origin is not None:
        # Generic: check the origin + recurse into args
        _check_type_module(origin, extra_imports)
        for arg in get_args(tp):
            _collect_non_builtin_types(arg, extra_imports)
        return
    _check_type_module(tp, extra_imports)


def _check_type_module(tp: Any, extra_imports: dict[str, set[str]]) -> None:
    """Add an import entry if tp is from a non-builtin, non-typing module."""
    if tp is Any or tp is type(None):
        return
    module = getattr(tp, "__module__", None)
    name = getattr(tp, "__name__", None)
    if module and name and module not in ("builtins", "typing"):
        extra_imports.setdefault(module, set()).add(name)


def _needs_field(field_info: FieldInfo) -> bool:
    """Check if a field needs Field() in its source representation."""
    if field_info.description is not None:
        return True
    if _extract_constraints(field_info):
        return True
    return False


def _extract_constraints(field_info: FieldInfo) -> dict[str, Any]:
    """Extract constraint kwargs from a FieldInfo (checking metadata)."""
    constraints: dict[str, Any] = {}
    # Check metadata for annotated_types constraint objects
    for meta in field_info.metadata:
        for cls, attr_name in _CONSTRAINT_MAP.items():
            if isinstance(meta, cls):
                constraints[attr_name] = getattr(meta, attr_name)
    return constraints


def _generate_class(
    model: type[BaseModel],
    disc_field: str,
    degraded_list: list[Any] | None = None,
) -> str:
    """Generate source code for a single model class."""
    extra = model.model_config.get("extra", "forbid")
    is_open = extra == "allow"

    # Build a lookup: field_name -> original_type_repr for inline comments
    degraded_lookup: dict[str, str] = {}
    if degraded_list:
        for df in degraded_list:
            degraded_lookup[df.field_name] = df.original_type_repr

    lines = []
    lines.append(f"\nclass {model.__name__}(BaseModel):")

    # Docstring for open schema
    if is_open:
        lines.append('    """This implementation accepts extra parameters (**kwargs).')
        lines.append('    Undeclared fields will not cause errors."""')
        lines.append("")

    # model_config
    lines.append(f'    model_config = ConfigDict(extra="{extra}")')
    lines.append("")

    # Fields
    for field_name, field_info in model.model_fields.items():
        degraded_from = degraded_lookup.get(field_name)
        field_line = _render_field(field_name, field_info, degraded_from=degraded_from)
        lines.append(f"    {field_line}")

    lines.append("")
    return "\n".join(lines)


def _render_field(
    name: str,
    field_info: FieldInfo,
    degraded_from: str | None = None,
) -> str:
    """Render a single field definition line."""
    type_str = _type_to_source(field_info.annotation)
    default = field_info.default
    is_required = default is PydanticUndefined

    # Check if we need Field()
    field_kwargs: dict[str, Any] = {}
    if field_info.description is not None:
        field_kwargs["description"] = field_info.description
    field_kwargs.update(_extract_constraints(field_info))

    # Inline comment for degraded fields
    comment = f"  # degraded from: {degraded_from}" if degraded_from else ""

    if field_kwargs:
        # Need Field()
        default_repr = "..." if is_required else _value_to_source(default)
        kwargs_str = ", ".join(
            f"{k}={_value_to_source(v)}" for k, v in field_kwargs.items()
        )
        return f"{name}: {type_str} = Field({default_repr}, {kwargs_str}){comment}"
    else:
        # Plain default
        if is_required:
            return f"{name}: {type_str} = ...{comment}"
        else:
            return f"{name}: {type_str} = {_value_to_source(default)}{comment}"


def _type_to_source(tp: Any) -> str:
    """Convert a type annotation to its source code representation."""
    if tp is None or tp is type(None):
        return "None"

    # Handle Annotated
    origin = get_origin(tp)
    if origin is Annotated:
        args = get_args(tp)
        return _type_to_source(args[0]) if args else repr(tp)

    # Handle Literal
    if origin is Literal:
        args = get_args(tp)
        args_str = ", ".join(_value_to_source(a) for a in args)
        return f"Literal[{args_str}]"

    # Handle Union (including Optional and PEP 604 X | Y)
    if origin in _UNION_TYPES:
        args = get_args(tp)
        if len(args) == 2 and type(None) in args:
            # Optional[T]
            non_none = [a for a in args if a is not type(None)][0]
            return f"Optional[{_type_to_source(non_none)}]"
        else:
            args_str = ", ".join(_type_to_source(a) for a in args)
            return f"Union[{args_str}]"

    # Handle generic types (list, dict, etc.)
    if origin is not None:
        args = get_args(tp)
        origin_name = _get_type_name(origin)
        if args:
            args_str = ", ".join(_type_to_source(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    # Basic types
    return _get_type_name(tp)


def _get_type_name(tp: Any) -> str:
    """Get the simple name for a type."""
    if tp is Any:
        return "Any"
    if hasattr(tp, "__name__"):
        return tp.__name__
    return repr(tp)


def _value_to_source(value: Any) -> str:
    """Convert a Python value to its source code representation."""
    if value is None:
        return "None"
    if value is ...:
        return "..."
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, list):
        return repr(value)
    if isinstance(value, dict):
        return repr(value)
    r = repr(value)
    if r.startswith("<"):
        return f"...  # default: {r}"
    return r


def _generate_union_alias(result: LayerConfigResult) -> str:
    """Generate the union type alias for multiple models."""
    layer_name = result.layer_name
    if len(layer_name) <= 3:
        alias_name = f"{layer_name.upper()}Config"
    else:
        alias_name = f"{layer_name.title()}Config"

    model_names = [
        result.per_key_models[k].__name__
        for k in sorted(result.per_key_models.keys())
    ]
    union_members = ", ".join(model_names)

    return (
        f'\n{alias_name} = Annotated[\n'
        f'    Union[{union_members}],\n'
        f'    Field(discriminator="{result.discriminator_field}"),\n'
        f']\n'
    )
