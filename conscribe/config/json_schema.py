"""JSON Schema generation for config unions.

Serializes a ``LayerConfigResult`` to a JSON Schema dict suitable
for YAML editor autocompletion and validation.

See ``config-typing-design.md`` Section 6.2 for specification.
"""
from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from conscribe.config.builder import LayerConfigResult


def generate_layer_json_schema(result: LayerConfigResult) -> dict[str, Any]:
    """Serialize a ``LayerConfigResult`` to a JSON Schema dict.

    Uses Pydantic's ``TypeAdapter.json_schema()`` for the heavy lifting,
    then adds the ``x-discriminator`` extension field.  When degraded
    fields are present, injects ``x-degraded-fields`` at the top level
    and per-property ``x-degraded-from`` / ``description`` annotations
    inside ``$defs``.

    Args:
        result: The ``LayerConfigResult`` from ``build_layer_config()``.

    Returns:
        A JSON Schema dict with ``x-discriminator`` extension.
    """
    adapter = TypeAdapter(result.union_type)
    schema = adapter.json_schema()
    schema["x-discriminator"] = result.discriminator_field

    # Nested mode extensions
    if result.discriminator_fields:
        schema["x-discriminator-fields"] = result.discriminator_fields
    if result.key_separator:
        schema["x-key-separator"] = result.key_separator

    degraded = result.degraded_fields
    if degraded:
        _inject_degraded_info(schema, result)

    return schema


def generate_composed_json_schema(result: Any) -> dict[str, Any]:
    """Serialize a ``ComposedConfigResult`` to a JSON Schema dict.

    Uses Pydantic's ``TypeAdapter.json_schema()`` on the top-level model.
    Pydantic automatically places nested models in ``$defs`` with ``$ref``.

    Adds:
    - ``x-composed-layers``: list of layer names in dependency order.
    - ``x-inline-wiring``: whether inline wiring was applied.

    Args:
        result: The ``ComposedConfigResult`` from ``build_composed_config()``.

    Returns:
        A JSON Schema dict.
    """
    adapter = TypeAdapter(result.top_level_type)
    schema = adapter.json_schema()
    schema["x-composed-layers"] = result.dependency_order
    schema["x-inline-wiring"] = result.inline_wiring

    return schema


def _inject_degraded_info(
    schema: dict[str, Any],
    result: LayerConfigResult,
) -> None:
    """Inject degradation metadata into a JSON Schema dict.

    Adds:
    - Top-level ``x-degraded-fields`` for programmatic consumption.
    - Per-property ``x-degraded-from`` and ``description`` annotations
      inside ``$defs`` for IDE hover support.
    """
    degraded = result.degraded_fields

    # Top-level extension
    x_degraded: dict[str, list[dict[str, str]]] = {}
    for key in sorted(degraded.keys()):
        x_degraded[key] = [
            {"field": df.field_name, "original_type": df.original_type_repr}
            for df in degraded[key]
        ]
    schema["x-degraded-fields"] = x_degraded

    # Per-property annotations in $defs
    defs = schema.get("$defs", {})
    if not defs:
        # Single-model schema — properties are at top level
        _annotate_properties(schema, degraded)
        return

    # Build a model-name -> key lookup from per_key_models
    model_to_key: dict[str, str] = {}
    for key, model in result.per_key_models.items():
        model_to_key[model.__name__] = key

    for def_name, def_schema in defs.items():
        key = model_to_key.get(def_name)
        if key and key in degraded:
            _annotate_properties_for_key(def_schema, degraded[key])


def _annotate_properties(
    schema: dict[str, Any],
    degraded: dict[str, list[Any]],
) -> None:
    """Annotate top-level properties (single-model case)."""
    props = schema.get("properties", {})
    for key_degraded_list in degraded.values():
        for df in key_degraded_list:
            if df.field_name in props:
                _annotate_single_property(props[df.field_name], df)


def _annotate_properties_for_key(
    def_schema: dict[str, Any],
    degraded_list: list[Any],
) -> None:
    """Annotate properties inside a ``$defs`` entry."""
    props = def_schema.get("properties", {})
    for df in degraded_list:
        if df.field_name in props:
            _annotate_single_property(props[df.field_name], df)


def _annotate_single_property(prop: dict[str, Any], df: Any) -> None:
    """Annotate a single JSON Schema property with degradation info."""
    prop["x-degraded-from"] = df.original_type_repr
    desc = f"[degraded] Type was: {df.original_type_repr} (not serializable in config)"
    if "description" in prop:
        prop["description"] = f"{prop['description']} {desc}"
    else:
        prop["description"] = desc
