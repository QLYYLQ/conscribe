"""Config schema extractor.

Extracts Pydantic ``BaseModel`` config schemas from registered
implementation classes via ``__init__`` signature reflection.

Supports three extraction tiers:
- Tier 3: explicit ``__config_schema__`` attribute (highest priority)
- Tier 3 variant: single-param BaseModel auto-detection
- Tier 1/1.5/2: ``__init__`` signature with optional ``Annotated``
  metadata and docstring descriptions

See ``config-typing-design.md`` Section 4 for full specification.
"""
from __future__ import annotations

import inspect
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from conscribe.config.docstring import parse_param_descriptions


def extract_config_schema(cls: type) -> Union[type[BaseModel], None]:
    """Extract config schema from a class.

    Extraction priority:
    1. ``cls.__config_schema__`` -> return directly (Tier 3)
    2. Single-param BaseModel in ``__init__`` -> return that type (Tier 3 variant)
    3. ``__init__`` signature reflection -> dynamic Pydantic model (Tier 1/1.5/2)
    4. No extractable params -> ``None``

    Args:
        cls: The class to extract config schema from.

    Returns:
        A Pydantic ``BaseModel`` subclass, or ``None`` if no
        extractable parameters.

    Raises:
        TypeError: If ``__config_schema__`` exists but is not a
            ``BaseModel`` subclass.
    """
    # -- Tier 3: explicit __config_schema__ --
    config_schema = getattr(cls, "__config_schema__", None)
    if config_schema is not None:
        if not (isinstance(config_schema, type) and issubclass(config_schema, BaseModel)):
            raise TypeError(
                f"{cls.__name__}.__config_schema__ must be a pydantic.BaseModel "
                f"subclass, got {type(config_schema).__name__}: {config_schema!r}"
            )
        return config_schema

    # -- Find the actual __init__ definer via MRO --
    init_definer = _find_init_definer(cls)
    if init_definer is None:
        return None

    init_method = init_definer.__init__  # type: ignore[misc]

    # -- Get signature --
    try:
        sig = inspect.signature(init_method)
    except (ValueError, TypeError):
        return None

    # -- Filter params (exclude self, *args, **kwargs) --
    named_params = _filter_named_params(sig)
    has_var = _has_var_keyword_or_positional(sig)

    # -- Get type hints (with Annotated extras) --
    # Try get_type_hints first (resolves forward refs), fall back to
    # signature annotations if that fails.
    hints = _safe_get_type_hints(init_method)

    # -- Tier 3 variant: single-param BaseModel --
    if len(named_params) == 1:
        param = named_params[0]
        param_type = _get_param_type(param, hints)
        base_type = _unwrap_annotated_type(param_type)

        if (
            base_type is not inspect.Parameter.empty
            and isinstance(base_type, type)
            and issubclass(base_type, BaseModel)
        ):
            return base_type

    # -- No named params -> None --
    if not named_params:
        return None

    if hints is None:
        # Both get_type_hints and signature annotations failed
        return None

    # -- Get docstring descriptions for fallback --
    doc_descriptions = parse_param_descriptions(init_definer)

    # -- Check annotated-only mode --
    annotated_only = getattr(cls, "__config_annotated_only__", False)

    # -- Build field definitions --
    field_definitions: dict[str, Any] = {}

    for param in named_params:
        param_type = _get_param_type(param, hints)
        field_info = _extract_field_info_from_annotated(param_type)

        # In annotated-only mode, skip params without Annotated[..., Field(...)]
        if annotated_only and field_info is None:
            continue
        base_type = _unwrap_annotated_type(param_type)

        if base_type is inspect.Parameter.empty:
            base_type = Any

        # Determine default
        if param.default is not inspect.Parameter.empty:
            default = param.default
        else:
            default = ...  # required

        # Build field kwargs
        field_kwargs: dict[str, Any] = {}

        if field_info is not None:
            # Tier 2: merge FieldInfo metadata
            if field_info.description is not None:
                field_kwargs["description"] = field_info.description
            elif param.name in doc_descriptions:
                # Tier 1.5 fallback: docstring description
                field_kwargs["description"] = doc_descriptions[param.name]

            # Copy constraint metadata from FieldInfo attributes
            for attr in ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length"):
                val = getattr(field_info, attr, None)
                if val is not None:
                    field_kwargs[attr] = val

            # Preserve constraint metadata stored in field_info.metadata
            if field_info.metadata:
                extra_metadata = [
                    m for m in field_info.metadata if not isinstance(m, FieldInfo)
                ]
                if extra_metadata:
                    if field_kwargs:
                        new_field = Field(default, **field_kwargs)
                        args = (base_type, new_field) + tuple(extra_metadata)
                        base_type = Annotated[args]  # type: ignore[misc]
                        field_definitions[param.name] = (base_type, default)
                        continue
                    else:
                        base_type = Annotated[tuple([base_type] + extra_metadata)]  # type: ignore[misc]

        else:
            # Tier 1.5 fallback: docstring description (no FieldInfo at all)
            if param.name in doc_descriptions:
                field_kwargs["description"] = doc_descriptions[param.name]

        if field_kwargs:
            field_definitions[param.name] = (base_type, Field(default, **field_kwargs))
        else:
            field_definitions[param.name] = (base_type, default)

    # -- Determine extra policy --
    extra = "allow" if has_var else "forbid"

    # -- Create dynamic model --
    model_name = f"{cls.__name__}Config"
    model = _create_dynamic_model(model_name, field_definitions, extra)
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_init_definer(cls: type) -> Union[type, None]:
    """Walk MRO to find the class that actually defines ``__init__``."""
    from conscribe.config._utils import find_init_definer
    return find_init_definer(cls)


def _filter_named_params(
    sig: inspect.Signature,
) -> list[inspect.Parameter]:
    """Filter signature params, excluding self, *args, **kwargs."""
    result = []
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        result.append(param)
    return result


def _has_var_keyword_or_positional(sig: inspect.Signature) -> bool:
    """Check if signature has *args or **kwargs."""
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
    return False


def _safe_get_type_hints(func: Any) -> Union[dict[str, Any], None]:
    """Get type hints safely, returning None on failure.

    Uses ``include_extras=True`` to preserve ``Annotated`` metadata.
    Falls back to evaluating raw ``__annotations__`` in the function's
    global/local scope if ``get_type_hints`` fails (e.g., for locally
    defined classes with ``from __future__ import annotations``).
    """
    try:
        return get_type_hints(func, include_extras=True)
    except Exception:
        # Fall back to raw __annotations__ with manual eval
        try:
            raw_annotations = getattr(func, "__annotations__", {})
            if not raw_annotations:
                return None
            globalns = getattr(func, "__globals__", {})
            localns = None
            import typing as _typing_mod
            safe_ns = {
                k: v for k, v in vars(_typing_mod).items()
                if not k.startswith("_")
            }
            safe_ns["__builtins__"] = {
                "str": str, "int": int, "float": float, "bool": bool,
                "bytes": bytes, "list": list, "dict": dict, "set": set,
                "tuple": tuple, "type": type, "None": None,
                "Ellipsis": Ellipsis,
            }
            if globalns:
                safe_ns.update({
                    k: v for k, v in globalns.items()
                    if isinstance(v, type) or hasattr(v, "__origin__")
                })
            result: dict[str, Any] = {}
            for name, annotation in raw_annotations.items():
                if isinstance(annotation, str):
                    try:
                        result[name] = eval(annotation, safe_ns, localns)  # noqa: S307
                    except Exception:
                        result[name] = annotation
                else:
                    result[name] = annotation
            return result if result else None
        except Exception:
            return None


def _get_param_type(
    param: inspect.Parameter,
    hints: Union[dict[str, Any], None],
) -> Any:
    """Get the type annotation for a parameter.

    Prefers resolved type hints, falls back to raw annotation.
    """
    if hints is not None and param.name in hints:
        return hints[param.name]
    if param.annotation is not inspect.Parameter.empty:
        return param.annotation
    return Any


def _unwrap_annotated_type(tp: Any) -> Any:
    """Unwrap ``Annotated[T, ...]`` to get the base type ``T``.

    If not an ``Annotated`` type, returns ``tp`` unchanged.
    """
    if get_origin(tp) is Annotated:
        args = get_args(tp)
        if args:
            return args[0]
    return tp


def _extract_field_info_from_annotated(tp: Any) -> Union[FieldInfo, None]:
    """Extract ``FieldInfo`` from ``Annotated[T, Field(...), ...]``.

    Returns the first ``FieldInfo`` found in the Annotated metadata,
    or ``None`` if not Annotated or no FieldInfo present.
    """
    if get_origin(tp) is not Annotated:
        return None

    args = get_args(tp)
    for arg in args[1:]:
        if isinstance(arg, FieldInfo):
            return arg
    return None


def _create_dynamic_model(
    name: str,
    field_definitions: dict[str, Any],
    extra: str,
) -> type[BaseModel]:
    """Create a dynamic Pydantic BaseModel with the given fields.

    Args:
        name: Model class name.
        field_definitions: Dict of {field_name: (type, default_or_FieldInfo)}.
        extra: Extra policy -- ``"forbid"`` or ``"allow"``.

    Returns:
        A dynamically created BaseModel subclass.
    """
    # Create a base class with the desired model_config
    base = type(
        f"_{name}Base",
        (BaseModel,),
        {"model_config": ConfigDict(extra=extra)},  # type: ignore[arg-type]
    )

    model = create_model(
        name,
        __base__=base,
        **field_definitions,
    )

    return model
