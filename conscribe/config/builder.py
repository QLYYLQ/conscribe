"""Config union builder.

Builds discriminated Pydantic unions from all registered implementations
in a ``LayerRegistrar``. Each per-key model gets a ``Literal[key]``
discriminator field injected, and the resulting union type can be used
with ``TypeAdapter`` for config validation.

See ``config-typing-design.md`` Section 5 for full specification.
"""
from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from conscribe.config.extractor import extract_config_schema


@dataclasses.dataclass(frozen=True)
class LayerConfigResult:
    """Result of building a layer config union.

    Attributes:
        union_type: The discriminated union type (or single model if only one key).
        per_key_models: Dict mapping registry key to its per-key Pydantic model.
        layer_name: The layer name from the registrar (e.g. ``"llm"``).
        discriminator_field: The discriminator field name (e.g. ``"provider"``).
    """

    union_type: Any
    per_key_models: dict[str, type[BaseModel]]
    layer_name: str
    discriminator_field: str


def build_layer_config(registrar: type) -> LayerConfigResult:
    """Build a discriminated union from all registered classes in a registrar.

    For each registered key:
    1. Extract config schema via ``extract_config_schema()``
    2. If schema is ``None``, create a model with only the discriminator field
    3. Inject ``Literal[key]`` discriminator field with ``default=key``
    4. Preserve original ``extra`` policy from extracted schema

    Single key → ``union_type`` is the model itself (no ``Union`` wrapper).
    Multiple keys → ``union_type`` is ``Annotated[Union[...], Field(discriminator=...)]``.

    Args:
        registrar: A ``LayerRegistrar`` subclass (created via ``create_registrar``).

    Returns:
        A ``LayerConfigResult`` with the union type and per-key models.

    Raises:
        ValueError: If the registrar has no ``discriminator_field`` set.
    """
    disc_field = registrar.discriminator_field
    if not disc_field:
        raise ValueError(
            f"Registrar for layer {registrar._registry.name!r} has no "
            f"discriminator_field set. Cannot build config union."
        )

    layer_name = registrar._registry.name
    all_classes = registrar.get_all()

    per_key_models: dict[str, type[BaseModel]] = {}

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
        else:
            # Inject discriminator into existing schema
            model = _inject_discriminator(schema, model_name, disc_field, key)

        per_key_models[key] = model

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
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_model_name(key: str, layer_name: str) -> str:
    """Build per-key model class name: ``{KeyPart}{LayerPart}Config``.

    KeyPart: snake_case segments each title-cased (``browser_use`` → ``BrowserUse``).
    LayerPart: ≤3 chars → ALL_CAPS (``llm`` → ``LLM``); >3 chars → Title (``agent`` → ``Agent``).
    """
    # KeyPart: split on '_', title each segment
    key_part = "".join(segment.title() for segment in key.split("_"))

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
