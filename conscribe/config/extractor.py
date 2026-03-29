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
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from conscribe.config.docstring import parse_param_descriptions


def extract_config_schema(
    cls: type,
    mro_scope: Union[str, list[str]] = "local",
    mro_depth: Union[int, None] = None,
) -> Union[type[BaseModel], None]:
    """Extract config schema from a class.

    Extraction priority:
    1. ``cls.__config_schema__`` -> return directly (Tier 3)
    2. Single-param BaseModel in ``__init__`` -> return that type (Tier 3 variant)
    3. ``__init__`` signature reflection -> dynamic Pydantic model (Tier 1/1.5/2)
    4. No extractable params -> ``None``

    When the ``__init__`` has ``**kwargs``, walks the MRO upward to
    collect parent parameters, producing a more complete schema.

    Args:
        cls: The class to extract config schema from.
        mro_scope: Scope for MRO traversal.  Accepts ``"local"``,
            ``"third_party"``, ``"all"``, or a ``list[str]`` of
            package names (local classes + only the listed packages).
            Can be overridden per-class via ``__config_mro_scope__``.
        mro_depth: Max MRO levels to traverse.  ``None`` = unlimited.
            Can be overridden per-class via ``__config_mro_depth__``.

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

    # -- BaseModel fast path --
    # When cls IS a BaseModel subclass and does NOT define its own __init__,
    # the standard reflection path would resolve to BaseModel.__init__
    # which has signature (self, /, **data: Any) — zero named params.
    # Instead, extract fields directly from cls.model_fields.
    if (
        isinstance(cls, type)
        and issubclass(cls, BaseModel)
        and cls is not BaseModel
        and "__init__" not in cls.__dict__
    ):
        return _extract_from_pydantic_model(cls)

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
    has_var_kw = _has_var_keyword(sig)
    has_var = has_var_kw or _has_var_positional(sig)

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

    # -- MRO collection for **kwargs chains --
    # Resolve class-level overrides for mro_scope / mro_depth
    effective_scope = getattr(cls, "__config_mro_scope__", mro_scope)
    effective_depth = getattr(cls, "__config_mro_depth__", mro_depth)

    mro_result = None
    if has_var_kw:
        from conscribe.config.mro import collect_mro_params

        mro_result = collect_mro_params(cls, scope=effective_scope, depth=effective_depth)
        if mro_result.params:
            named_params = named_params + mro_result.params
            # Merge hints: child hints take precedence
            if hints is not None and mro_result.hints:
                merged_hints = dict(mro_result.hints)
                merged_hints.update(hints)
                hints = merged_hints
            elif hints is None and mro_result.hints:
                hints = dict(mro_result.hints)

    # -- No named params -> None --
    if not named_params:
        return None

    if hints is None:
        # Both get_type_hints and signature annotations failed
        return None

    # -- Get docstring descriptions for fallback --
    doc_descriptions = parse_param_descriptions(init_definer)
    # Merge docstring descriptions from MRO parent classes
    if mro_result is not None and mro_result.init_definers:
        for parent_cls in mro_result.init_definers:
            parent_docs = parse_param_descriptions(parent_cls)
            for name, desc in parent_docs.items():
                if name not in doc_descriptions:
                    doc_descriptions[name] = desc

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

    # -- Apply __wiring__ constraints / injection --
    wired_fields = _apply_wiring(cls, field_definitions)

    # -- Determine extra policy --
    if has_var_kw and mro_result is not None and mro_result.params:
        # MRO collection happened — use fully_resolved to decide
        extra = "forbid" if mro_result.fully_resolved else "allow"
    elif has_var:
        extra = "allow"
    else:
        extra = "forbid"

    # -- Create dynamic model --
    model_name = f"{cls.__name__}Config"
    from pydantic import PydanticSchemaGenerationError, PydanticUserError

    try:
        model = _create_dynamic_model(model_name, field_definitions, extra)
    except (PydanticSchemaGenerationError, PydanticUserError):
        # Pydantic couldn't handle some field types — degrade them to Any
        from conscribe.config.degradation import degrade_field_definitions

        source_class_name = f"{cls.__module__}.{cls.__qualname__}"
        field_definitions, degraded = degrade_field_definitions(
            field_definitions, source_class_name=source_class_name,
        )
        model = _create_dynamic_model(model_name, field_definitions, extra)
        model.__degraded_fields__ = degraded  # type: ignore[attr-defined]

    # Tag wired fields for codegen annotation
    if wired_fields:
        model.__wired_fields__ = wired_fields  # type: ignore[attr-defined]

    return model


def _extract_from_pydantic_model(
    cls: type,
) -> Union[type[BaseModel], None]:
    """Extract config schema from a Pydantic BaseModel subclass.

    Uses ``cls.model_fields`` to build field definitions, preserving
    all FieldInfo metadata (description, constraints, etc.).

    Args:
        cls: A Pydantic BaseModel subclass.

    Returns:
        A dynamically created BaseModel subclass, or None if no
        init-eligible fields exist.
    """
    field_definitions: dict[str, Any] = {}

    for fname, field_info in cls.model_fields.items():
        # Skip fields where init=False
        if field_info.init is not None and not field_info.init:
            continue

        annotation = field_info.annotation
        if annotation is None:
            annotation = Any

        # Pass the FieldInfo directly to preserve all metadata
        field_definitions[fname] = (annotation, field_info)

    if not field_definitions:
        return None

    # Read extra policy from the model's config
    extra = cls.model_config.get("extra", "forbid")

    model_name = f"{cls.__name__}Config"
    return _create_dynamic_model(model_name, field_definitions, extra)


def extract_own_init_params(
    cls: type,
) -> tuple[list[inspect.Parameter], dict[str, Any] | None]:
    """Extract parameters defined in ``cls``'s own ``__init__`` (NOT inherited).

    Returns only the named parameters (excluding self, *args, **kwargs)
    that are introduced by this class (not present in the immediate
    parent's ``__init__``).

    This is used by the nested config builder to split params by MRO level.

    Args:
        cls: The class to extract own init params from.

    Returns:
        A tuple of (params, hints) where params is a list of
        ``inspect.Parameter`` objects and hints is a dict of type hints
        (or None if unavailable).
    """
    if "__init__" not in cls.__dict__:
        return [], None

    init_method = cls.__init__  # type: ignore[misc]
    try:
        sig = inspect.signature(init_method)
    except (ValueError, TypeError):
        return [], None

    own_params = _filter_named_params(sig)
    hints = _safe_get_type_hints(init_method)

    # Find parent params to exclude
    parent_param_names: set[str] = set()
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        if "__init__" in base.__dict__:
            try:
                parent_sig = inspect.signature(base.__init__)  # type: ignore[misc]
                parent_params = _filter_named_params(parent_sig)
                parent_param_names.update(p.name for p in parent_params)
            except (ValueError, TypeError):
                pass
            break  # Only look at the nearest parent with __init__

    # Filter to params unique to this class
    unique_params = [p for p in own_params if p.name not in parent_param_names]
    return unique_params, hints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_wiring(
    cls: type,
    field_definitions: dict[str, Any],
) -> dict[str, str]:
    """Apply ``__wiring__`` constraints and inject missing wired fields.

    For each resolved wiring entry:
    - If the param exists in ``field_definitions``: replace its type with
      ``Literal[...keys...]``, preserving the original default/FieldInfo.
      Handles ``Optional[str]`` → ``Optional[Literal[...]]``.
    - If the param does NOT exist in ``field_definitions``: inject it as
      a new required field with ``Literal[...keys...]`` type.

    Args:
        cls: The class being extracted.
        field_definitions: The current field definitions dict (mutated in place).

    Returns:
        Dict mapping param names to registry names for codegen annotation
        (e.g. ``{"loop": "agent_loop", "browser": ""}``).
        Empty dict if no wiring was applied.
    """
    from conscribe.exceptions import WiringResolutionError
    from conscribe.wiring import resolve_wiring

    try:
        resolved = resolve_wiring(cls)
    except (WiringResolutionError, TypeError):
        # WiringResolutionError: registry not yet populated or not found.
        # TypeError: malformed __wiring__ declaration.
        # Both can occur during partial extraction; skip gracefully.
        return {}

    if not resolved:
        return {}

    wired_fields: dict[str, str] = {}

    for param_name, wiring in resolved.items():
        literal_type = Literal[tuple(wiring.allowed_keys)]  # type: ignore[valid-type]

        if param_name in field_definitions:
            # Constrain existing field: replace type, preserve default
            existing = field_definitions[param_name]
            if isinstance(existing, tuple) and len(existing) == 2:
                old_type, default_or_field = existing
                # Handle Optional[str] → Optional[Literal[...]]
                new_type = _replace_str_with_literal(old_type, literal_type)
                field_definitions[param_name] = (new_type, default_or_field)
            wiring.injected = False
        else:
            # Inject new required field
            field_definitions[param_name] = (literal_type, ...)
            wiring.injected = True

        wired_fields[param_name] = wiring.registry_name or "literal"

    return wired_fields


def _replace_str_with_literal(original_type: Any, literal_type: Any) -> Any:
    """Replace ``str`` in a type annotation with a ``Literal`` type.

    Handles:
    - ``str`` → ``Literal[...]``
    - ``Optional[str]`` → ``Optional[Literal[...]]``
    - ``Union[str, X]`` → ``Union[Literal[...], X]``
    - ``Annotated[str, Field(...)]`` → ``Annotated[Literal[...], Field(...)]``
    - Other types → replaced entirely with ``literal_type``

    Args:
        original_type: The existing type annotation.
        literal_type: The Literal type to substitute.

    Returns:
        The updated type annotation.
    """
    origin = get_origin(original_type)

    # Annotated[str, Field(...), ...] → Annotated[Literal[...], Field(...), ...]
    if origin is Annotated:
        args = get_args(original_type)
        base = args[0]
        metadata = args[1:]
        replaced_base = _replace_str_with_literal(base, literal_type)
        return Annotated[tuple([replaced_base] + list(metadata))]  # type: ignore[return-value]

    # Optional[str] or Union[str, None]
    if origin is Union:
        args = get_args(original_type)
        new_args = []
        for arg in args:
            if arg is str:
                new_args.append(literal_type)
            elif arg is type(None):
                new_args.append(arg)
            else:
                new_args.append(arg)
        if len(new_args) == 2 and type(None) in new_args:
            non_none = [a for a in new_args if a is not type(None)][0]
            return Union[non_none, None]  # type: ignore[return-value]
        return Union[tuple(new_args)]  # type: ignore[return-value]

    # Plain str or Any → just use the Literal type
    if original_type is str or original_type is Any:
        return literal_type

    # For any other type, replace entirely
    return literal_type


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


def _has_var_keyword(sig: inspect.Signature) -> bool:
    """Check if signature has ``**kwargs``."""
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


def _has_var_positional(sig: inspect.Signature) -> bool:
    """Check if signature has ``*args``."""
    return any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in sig.parameters.values()
    )


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
        # Fall back to raw __annotations__ with per-annotation eval.
        # This handles the case where get_type_hints fails globally
        # (e.g. one annotation references a name missing from the
        # module namespace) but other annotations are resolvable.
        try:
            raw_annotations = getattr(func, "__annotations__", {})
            if not raw_annotations:
                return None
            globalns = getattr(func, "__globals__", {})
            localns = None
            import typing as _typing_mod
            # Start with typing module exports (Annotated, Union, etc.)
            eval_ns = {
                k: v for k, v in vars(_typing_mod).items()
                if not k.startswith("_")
            }
            # Include ALL names from the function's module globals.
            # This is critical: callables like pydantic.Field, custom
            # types, and other objects referenced in annotations must
            # be available.  get_type_hints() itself evaluates with
            # full globals, so the fallback needs the same namespace.
            if globalns:
                eval_ns.update(globalns)
            # Restrict __builtins__ to type-like names only.
            eval_ns["__builtins__"] = {
                "str": str, "int": int, "float": float, "bool": bool,
                "bytes": bytes, "list": list, "dict": dict, "set": set,
                "tuple": tuple, "type": type, "None": None,
                "Ellipsis": Ellipsis,
            }
            result: dict[str, Any] = {}
            for name, annotation in raw_annotations.items():
                if isinstance(annotation, str):
                    try:
                        result[name] = eval(annotation, eval_ns, localns)  # noqa: S307
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
