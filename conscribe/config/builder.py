"""Config union builder.

Builds discriminated Pydantic unions from all registered implementations
in a ``LayerRegistrar``. Each per-key model gets a ``Literal[key]``
discriminator field injected, and the resulting union type can be used
with ``TypeAdapter`` for config validation.

Supports two modes:
- **Flat mode** (``discriminator_field``): existing behavior, single-level
  discriminator with ``Field(discriminator=...)``
- **Nested mode** (``discriminator_fields``): deeply nested compound
  discrimination where each key segment maps to a nested Pydantic sub-model

See ``config-typing-design.md`` Section 5 for full specification.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, create_model

from conscribe.config.extractor import (
    extract_config_schema,
    extract_own_init_params,
    _safe_get_type_hints,
    _unwrap_annotated_type,
)


@dataclasses.dataclass(frozen=True)
class LayerConfigResult:
    """Result of building a layer config union.

    Attributes:
        union_type: The discriminated union type (or single model if only one key).
        per_key_models: Dict mapping registry key to its per-key Pydantic model.
        layer_name: The layer name from the registrar (e.g. ``"llm"``).
        discriminator_field: The discriminator field name (e.g. ``"provider"``).
        degraded_fields: Dict mapping registry key to list of
            :class:`~conscribe.config.degradation.DegradedField` for fields
            whose types were degraded to ``Any``.  Empty dict when no
            degradation occurred (default).
        discriminator_fields: Discriminator field names for nested mode.
            ``None`` in flat mode.
        key_separator: Key separator used for hierarchical keys.
        per_segment_models: For nested mode, maps segment level name to
            {segment_value: model} dict. ``None`` in flat mode.
    """

    union_type: Any
    per_key_models: dict[str, type[BaseModel]]
    layer_name: str
    discriminator_field: str
    degraded_fields: dict[str, list[Any]] = dataclasses.field(default_factory=dict)
    discriminator_fields: list[str] | None = None
    key_separator: str = ""
    per_segment_models: dict[str, dict[str, type[BaseModel]]] | None = None


def build_layer_config(registrar: type) -> LayerConfigResult:
    """Build a discriminated union from all registered classes in a registrar.

    Dispatches to flat or nested mode based on whether the registrar has
    ``discriminator_fields`` set.

    Args:
        registrar: A ``LayerRegistrar`` subclass (created via ``create_registrar``).

    Returns:
        A ``LayerConfigResult`` with the union type and per-key models.

    Raises:
        ValueError: If the registrar has no discriminator configuration set.
    """
    disc_fields = getattr(registrar, "discriminator_fields", None)
    if disc_fields:
        return _build_nested_config(registrar)
    return _build_flat_config(registrar)


def _build_flat_config(registrar: type) -> LayerConfigResult:
    """Build flat (single-level) discriminated union. Original behavior."""
    disc_field = registrar.discriminator_field
    if not disc_field:
        raise ValueError(
            f"Registrar for layer {registrar._registry.name!r} has no "
            f"discriminator_field set. Cannot build config union."
        )

    layer_name = registrar._registry.name
    all_classes = registrar.get_all()

    per_key_models: dict[str, type[BaseModel]] = {}
    all_degraded: dict[str, list[Any]] = {}

    # Read MRO config from the registrar
    reg_mro_scope = getattr(registrar, "_mro_scope", "local")
    reg_mro_depth = getattr(registrar, "_mro_depth", None)

    for key, cls in all_classes.items():
        schema = extract_config_schema(
            cls, mro_scope=reg_mro_scope, mro_depth=reg_mro_depth,
        )
        model_name = _build_model_name(key, layer_name)

        if schema is None:
            # No extractable params — create discriminator-only model
            model = _create_discriminator_only_model(model_name, disc_field, key)
            schema_degraded = None
        else:
            # Read degraded fields BEFORE _inject_discriminator (which
            # rebuilds the model and would lose the custom attribute).
            schema_degraded = getattr(schema, "__degraded_fields__", None)
            # Inject discriminator into existing schema
            model = _inject_discriminator(schema, model_name, disc_field, key)

        per_key_models[key] = model
        if schema_degraded:
            all_degraded[key] = schema_degraded

    # Build union type
    if len(per_key_models) == 1:
        union_type = next(iter(per_key_models.values()))
    else:
        union_type = Annotated[
            Union[tuple(per_key_models.values())],  # type: ignore[arg-type]
            Field(discriminator=disc_field),
        ]

    return LayerConfigResult(
        union_type=union_type,
        per_key_models=per_key_models,
        layer_name=layer_name,
        discriminator_field=disc_field,
        degraded_fields=all_degraded,
    )


# ---------------------------------------------------------------------------
# Nested mode (compound discriminator)
# ---------------------------------------------------------------------------


def _build_nested_config(registrar: type) -> LayerConfigResult:
    """Build nested (compound) discriminated union.

    Each key segment maps to a Pydantic sub-model level. Level 0 is flat
    at the top level; level 1+ are nested sub-models.

    Example for ``discriminator_fields=["model_type", "provider"]``:
    - ``"openai.azure"`` → flat ``model_type: Literal["openai"]`` + nested
      ``provider: AzureProviderConfig``
    """
    disc_fields = registrar.discriminator_fields
    separator = getattr(registrar, "_key_separator", ".")
    layer_name = registrar._registry.name
    all_classes = registrar.get_all()
    n_levels = len(disc_fields)

    # 1. Filter to leaf keys (segment count == n_levels)
    leaf_keys: dict[str, type] = {}
    for key, cls in all_classes.items():
        segments = key.split(separator)
        if len(segments) == n_levels:
            leaf_keys[key] = cls

    if not leaf_keys:
        raise ValueError(
            f"No leaf keys found with {n_levels} segments for layer "
            f"{layer_name!r}. Registered keys: {list(all_classes.keys())}"
        )

    # 2. For each leaf key, extract params split by level
    per_key_models: dict[str, type[BaseModel]] = {}
    per_segment_models: dict[str, dict[str, type[BaseModel]]] = {
        disc_fields[i]: {} for i in range(1, n_levels)
    }

    for key, cls in sorted(leaf_keys.items()):
        segments = key.split(separator)
        level_params = _extract_params_by_level(cls, all_classes, separator)

        # Build nested segment models (level 1+)
        nested_field_type = None
        for level in range(n_levels - 1, 0, -1):
            seg = segments[level]
            field_name = disc_fields[level]
            params = level_params.get(level, {})

            # Build field definitions for this segment
            field_defs: dict[str, Any] = {
                "name": (Literal[seg], seg),  # type: ignore[valid-type]
            }
            for pname, (ptype, pdefault) in params.items():
                field_defs[pname] = (ptype, pdefault)

            # If there's a deeper nested level, add it
            if nested_field_type is not None:
                deeper_field_name = disc_fields[level + 1]
                field_defs[deeper_field_name] = (nested_field_type, ...)

            # Create segment model
            seg_model_name = f"{seg.title()}{field_name.title()}Config"
            seg_base = type(
                f"_{seg_model_name}Base",
                (BaseModel,),
                {"model_config": ConfigDict(extra="forbid")},
            )
            seg_model = create_model(seg_model_name, __base__=seg_base, **field_defs)
            per_segment_models.setdefault(field_name, {})[seg] = seg_model
            nested_field_type = seg_model

        # Build combined model (level 0 flat + level 1 nested)
        combined_name = _build_model_name(key, layer_name)
        combined_defs: dict[str, Any] = {
            disc_fields[0]: (Literal[segments[0]], segments[0]),  # type: ignore[valid-type]
        }

        # Add level 0 params flat
        level0_params = level_params.get(0, {})
        for pname, (ptype, pdefault) in level0_params.items():
            combined_defs[pname] = (ptype, pdefault)

        # Add nested sub-model for level 1 (if exists)
        if n_levels > 1 and nested_field_type is not None:
            combined_defs[disc_fields[1]] = (nested_field_type, ...)

        combined_base = type(
            f"_{combined_name}Base",
            (BaseModel,),
            {"model_config": ConfigDict(extra="forbid")},
        )
        combined_model = create_model(combined_name, __base__=combined_base, **combined_defs)
        per_key_models[key] = combined_model

    # 3. Build compound discriminator function
    union_type = _build_compound_union(
        per_key_models, disc_fields, separator, layer_name,
    )

    return LayerConfigResult(
        union_type=union_type,
        per_key_models=per_key_models,
        layer_name=layer_name,
        discriminator_field=disc_fields[0],
        discriminator_fields=disc_fields,
        key_separator=separator,
        per_segment_models=per_segment_models,
    )


def _extract_params_by_level(
    cls: type,
    all_classes: dict[str, type],
    separator: str,
) -> dict[int, dict[str, tuple[Any, Any]]]:
    """Extract params split by MRO level for nested config.

    Walks the MRO from root to leaf. Each ancestor with ``__registry_key__``
    maps to a key segment level (regardless of whether it's registered or
    abstract). Params are assigned to the **first class that defines them**
    (root → leaf). Non-keyed intermediate classes' params go to the nearest
    leaf-ward keyed level.

    Returns:
        Dict mapping level index to {param_name: (type, default)}.
    """
    # Walk MRO from root → leaf, find ancestors with __registry_key__
    # This includes abstract parents that set __registry_key__ but aren't registered
    keyed_mro: list[tuple[int, type]] = []  # (level, class)
    for ancestor in reversed(cls.__mro__):
        if ancestor is object:
            continue
        reg_key = getattr(ancestor, "__registry_key__", None)
        if reg_key and isinstance(reg_key, str):
            segments = reg_key.split(separator)
            level = len(segments) - 1
            keyed_mro.append((level, ancestor))

    # If the leaf class itself isn't in keyed_mro (shouldn't happen), add it
    if keyed_mro and keyed_mro[-1][1] is not cls:
        reg_key = getattr(cls, "__registry_key__", None)
        if reg_key:
            segments = reg_key.split(separator)
            keyed_mro.append((len(segments) - 1, cls))

    # Extract params for each class in MRO, assign to appropriate level
    seen_params: set[str] = set()
    level_params: dict[int, dict[str, tuple[Any, Any]]] = {}

    for level, ancestor in keyed_mro:
        if "__init__" not in ancestor.__dict__:
            continue

        own_params, hints = extract_own_init_params(ancestor)

        # For each keyed ancestor, also grab ALL its own params (not just unique to it)
        # since extract_own_init_params only returns params NOT in parent
        if not own_params:
            try:
                sig = inspect.signature(ancestor.__init__)  # type: ignore[misc]
                from conscribe.config.extractor import _filter_named_params
                all_params = _filter_named_params(sig)
                own_params = [p for p in all_params if p.name not in seen_params]
                hints = _safe_get_type_hints(ancestor.__init__)  # type: ignore[misc]
            except (ValueError, TypeError):
                pass

        if not own_params:
            continue

        for param in own_params:
            if param.name in seen_params:
                continue
            seen_params.add(param.name)

            # Get type
            param_type = Any
            if hints is not None and param.name in hints:
                param_type = hints[param.name]
                base = _unwrap_annotated_type(param_type)
                if base is not inspect.Parameter.empty:
                    param_type = base

            # Get default
            if param.default is not inspect.Parameter.empty:
                default = param.default
            else:
                default = ...

            level_params.setdefault(level, {})[param.name] = (param_type, default)

    return level_params


def _build_compound_union(
    per_key_models: dict[str, type[BaseModel]],
    disc_fields: list[str],
    separator: str,
    layer_name: str,
) -> Any:
    """Build compound discriminated union with custom discriminator function."""
    if len(per_key_models) == 1:
        return next(iter(per_key_models.values()))

    # Build discriminator function
    def _discriminate(v: Any) -> str:
        parts = []
        for i, field_name in enumerate(disc_fields):
            if i == 0:
                # Level 0 is flat
                if isinstance(v, dict):
                    parts.append(v.get(field_name, ""))
                else:
                    parts.append(getattr(v, field_name, ""))
            else:
                # Level 1+ is nested — walk into nested dicts/models
                if isinstance(v, dict):
                    nested = v
                    for j in range(1, i + 1):
                        nested = nested.get(disc_fields[j], {}) if isinstance(nested, dict) else {}
                    parts.append(nested.get("name", "") if isinstance(nested, dict) else getattr(nested, "name", ""))
                else:
                    obj = v
                    for j in range(1, i + 1):
                        obj = getattr(obj, disc_fields[j], None)
                    parts.append(getattr(obj, "name", "") if obj else "")
        return separator.join(parts)

    # Set a descriptive name
    _discriminate.__qualname__ = f"_discriminate_{layer_name}"
    _discriminate.__name__ = f"_discriminate_{layer_name}"

    # Build annotated union members with Tag
    members = []
    for key in sorted(per_key_models.keys()):
        model = per_key_models[key]
        members.append(Annotated[model, Tag(key)])

    union_type = Annotated[
        Union[tuple(members)],  # type: ignore[arg-type]
        Discriminator(_discriminate),
    ]

    return union_type


# ---------------------------------------------------------------------------
# Flat mode helpers
# ---------------------------------------------------------------------------


def _build_model_name(key: str, layer_name: str) -> str:
    """Build per-key model class name: ``{KeyPart}{LayerPart}Config``.

    KeyPart: snake_case segments each title-cased (``browser_use`` → ``BrowserUse``).
    LayerPart: ≤3 chars → ALL_CAPS (``llm`` → ``LLM``); >3 chars → Title (``agent`` → ``Agent``).

    For dotted keys (e.g. ``openai.azure``), dots are treated as separators too.
    """
    # Replace dots with underscores for consistent handling
    normalized_key = key.replace(".", "_")
    # KeyPart: split on '_', title each segment
    key_part = "".join(segment.title() for segment in normalized_key.split("_"))

    # LayerPart: short names ALL_CAPS, longer names Title
    if len(layer_name) <= 3:
        layer_part = layer_name.upper()
    else:
        layer_part = layer_name.title()

    return f"{key_part}{layer_part}Config"


def _create_discriminator_only_model(
    name: str,
    disc_field: str,
    key: str,
) -> type[BaseModel]:
    """Create a model with only the discriminator field (extra='forbid')."""
    base = type(
        f"_{name}Base",
        (BaseModel,),
        {"model_config": ConfigDict(extra="forbid")},
    )

    field_definitions: dict[str, Any] = {
        disc_field: (Literal[key], key),  # type: ignore[valid-type]
    }

    return create_model(name, __base__=base, **field_definitions)


def _inject_discriminator(
    schema: type[BaseModel],
    model_name: str,
    disc_field: str,
    key: str,
) -> type[BaseModel]:
    """Inject a ``Literal[key]`` discriminator field into an extracted schema.

    Creates a new model that inherits the original schema's fields and extra
    policy, but with the discriminator field overridden/added as ``Literal[key]``.
    """
    # Get the original extra policy
    extra = schema.model_config.get("extra", "forbid")

    # Collect existing fields (excluding the discriminator if it exists)
    field_definitions: dict[str, Any] = {}
    for field_name, field_info in schema.model_fields.items():
        if field_name == disc_field:
            continue  # Will be overridden
        # Reconstruct field definition as (type, FieldInfo)
        field_definitions[field_name] = (field_info.annotation, field_info)

    # Add discriminator field as Literal[key] with default=key
    field_definitions[disc_field] = (Literal[key], key)  # type: ignore[valid-type]

    # Create base with desired extra policy
    base = type(
        f"_{model_name}Base",
        (BaseModel,),
        {"model_config": ConfigDict(extra=extra)},  # type: ignore[arg-type]
    )

    return create_model(model_name, __base__=base, **field_definitions)
