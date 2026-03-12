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
    then adds the ``x-discriminator`` extension field.

    Args:
        result: The ``LayerConfigResult`` from ``build_layer_config()``.

    Returns:
        A JSON Schema dict with ``x-discriminator`` extension.
    """
    adapter = TypeAdapter(result.union_type)
    schema = adapter.json_schema()
    schema["x-discriminator"] = result.discriminator_field
    return schema
