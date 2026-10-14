"""JSON Schema generation for config unions.

Serializes a ``LayerConfigResult`` (or a ``ComposedConfigResult``) to a JSON
Schema dict suitable for YAML editor autocompletion and validation.

Every published schema is pydantic's own JSON schema of the in-process
object, generated with :class:`_DeterministicJsonSchema`, which writes the
members of set-valued defaults sorted -- at any depth, also inside model and
dataclass instances -- so field defaults do not depend on
``PYTHONHASHSEED``.  Everything else is pydantic's encoding (values a user
puts into ``examples`` or ``json_schema_extra`` are written as given; a set
there keeps its iteration order):

* **Composed** -- the top-level model's own schema,
  ``top_level_type.model_json_schema()`` generated as above (so it differs
  from pydantic's default generator only in the order of set members).  The
  top-level model carries its ``x-`` extensions itself (``x-composed-layers``,
  ``x-inline-wiring``, ``x-inline-only-layers`` and, under
  ``top_level_extra="ignore"`` / ``"allow"``, the ``not`` clause refusing
  inline-only names) through ``model_config["json_schema_extra"]``, so the
  in-process model, the generated ``_composed.py`` and the published file
  agree by construction; nothing is added afterwards.
* **Per layer** -- the schema of ``union_type`` plus the layer's ``x-``
  extensions (a union is not a model and cannot carry them itself):
  ``x-discriminator``, nested mode's ``x-discriminator-fields`` /
  ``x-key-separator``, and for degraded fields ``x-degraded-fields`` with the
  per-property ``x-degraded-from`` marker and ``[degraded]`` description
  hint (unless the field's own schema already carries the marker).
"""
from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, PydanticSchemaGenerationError, PydanticUserError, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticSerializationError, to_jsonable_python

from conscribe.config.builder import LayerConfigResult
from conscribe.config.composed import _set_member_key


class _DeterministicJsonSchema(GenerateJsonSchema):
    """pydantic's JSON-schema generator with stable set/frozenset defaults.

    pydantic encodes a set default as a list in iteration order, which for
    strings depends on ``PYTHONHASHSEED``; generated schema files would then
    change between runs.  This generator puts the members of every set in a
    default -- at any depth, also inside model and dataclass instances -- in
    canonical order (:func:`_presort`) before handing the value to pydantic.
    """

    def encode_default(self, dft: Any) -> Any:
        """Encode ``dft`` like pydantic, with set-like values sorted."""
        return super().encode_default(_presort(dft))


def _presort(value: Any, _depth: int = 0) -> Any:
    """``value`` with the members of every set/frozenset in it in canonical order.

    Builtin containers are walked and each set becomes the list of its
    members sorted by their JSON form: its type name, then its canonical JSON
    text (:func:`~conscribe.config.composed._set_member_key`, the order of
    composed default instances).  A pydantic model or dataclass instance that
    holds a set becomes its JSON dump (``by_alias``) with every set's members
    sorted (:func:`_sorted_sets`), so member forms stay pydantic's own.
    Anything else is kept: pydantic encodes it exactly as usual.
    """
    if _depth > 50:
        return value
    depth = _depth + 1
    if isinstance(value, (set, frozenset)):
        members = [_presort(v, depth) for v in value]
        return sorted(members, key=_member_key)
    if type(value) is list:
        return [_presort(v, depth) for v in value]
    if type(value) is tuple:
        return tuple(_presort(v, depth) for v in value)
    if type(value) is dict:
        return {k: _presort(v, depth) for k, v in value.items()}
    if _is_record(value) and _holds_set(value, 0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dumped = _json_dump(value)
        except (PydanticSchemaGenerationError, PydanticSerializationError, PydanticUserError):
            return value  # pydantic encodes it its own way (or leaves the default out)
        return _sorted_sets(value, dumped, 0)
    return value


def _member_key(member: Any) -> tuple[str, str]:
    """Sort key of a set member: :func:`_set_member_key` of its JSON form."""
    try:
        form = to_jsonable_python(member)
    except (PydanticSerializationError, TypeError, ValueError):
        return (type(member).__name__, repr(member))
    return _set_member_key(form)


def _is_record(value: Any) -> bool:
    """A pydantic model or dataclass instance (not a class)."""
    if isinstance(value, BaseModel):
        return True
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _json_dump(value: Any) -> Any:
    """pydantic's JSON form of a model / dataclass instance (``by_alias``)."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    return TypeAdapter(type(value)).dump_python(value, mode="json", by_alias=True)


def _record_values(value: Any) -> dict[str, Any]:
    """A record's values under the keys of its JSON dump (``by_alias``)."""
    out: dict[str, Any] = {}
    if isinstance(value, BaseModel):
        for name, info in type(value).model_fields.items():
            out[info.serialization_alias or info.alias or name] = getattr(value, name, None)
        for name, computed in type(value).model_computed_fields.items():
            out[computed.alias or name] = getattr(value, name, None)
        out.update(value.model_extra or {})
        return out
    infos = getattr(type(value), "__pydantic_fields__", None) or {}
    for field in dataclasses.fields(value):
        info = infos.get(field.name)
        key = (info.serialization_alias or info.alias or field.name) if info is not None else field.name
        out[key] = getattr(value, field.name, None)
    return out


def _holds_set(value: Any, depth: int) -> bool:
    """Whether a set/frozenset is reachable in ``value`` (records, mappings, sequences)."""
    if depth > 50:
        return False
    if isinstance(value, (set, frozenset)):
        return True
    if _is_record(value):
        return any(_holds_set(v, depth + 1) for v in _record_values(value).values())
    if isinstance(value, Mapping):
        return any(_holds_set(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_set(v, depth + 1) for v in value)
    return False


def _sorted_sets(value: Any, dumped: Any, depth: int) -> Any:
    """``dumped`` -- pydantic's JSON form of ``value`` -- with every set's members sorted.

    The walk of :func:`conscribe.config.composed._sorted_sets`, extended to
    dataclass instances: ``value`` and its JSON form side by side (records by
    dump key, mappings item by item, lists and tuples item by item, set
    members in iteration order -- pydantic serializes a set by iterating
    it); every JSON list that is a set's form is sorted by
    :func:`_set_member_key`.  Where the JSON form does not mirror the value
    (a custom serializer changed its shape) it is left as it is.
    """
    if depth > 50:
        return dumped
    depth += 1
    if _is_record(value) and isinstance(dumped, dict):
        values = _record_values(value)
        return {
            key: _sorted_sets(values[key], form, depth) if key in values else form
            for key, form in dumped.items()
        }
    if isinstance(value, (set, frozenset)) and isinstance(dumped, list):
        if len(value) == len(dumped):
            dumped = [_sorted_sets(member, form, depth) for member, form in zip(value, dumped)]
        return sorted(dumped, key=_set_member_key)
    if isinstance(value, Mapping) and isinstance(dumped, dict) and len(value) == len(dumped):
        return {
            key: _sorted_sets(item, form, depth)
            for item, (key, form) in zip(value.values(), dumped.items())
        }
    if isinstance(value, (list, tuple)) and isinstance(dumped, list) and len(value) == len(dumped):
        return [_sorted_sets(item, form, depth) for item, form in zip(value, dumped)]
    return dumped


def generate_layer_json_schema(result: LayerConfigResult) -> dict[str, Any]:
    """Serialize a ``LayerConfigResult`` to a JSON Schema dict.

    The schema of the in-process ``union_type`` (the same generator as the
    composed schema), plus the layer's ``x-`` extensions, which a union
    cannot carry itself: ``x-discriminator``, nested mode's
    ``x-discriminator-fields`` / ``x-key-separator``.  When degraded fields
    are present, injects ``x-degraded-fields`` at the top level and
    per-property ``x-degraded-from`` / ``description`` annotations (on the
    root properties of a one-key layer, and inside the ``$defs`` entries of
    per-key models) -- the only non-``x-`` difference from the in-process
    schema.

    Args:
        result: The ``LayerConfigResult`` from ``build_layer_config()``.

    Returns:
        A JSON Schema dict with ``x-discriminator`` extension.
    """
    adapter = TypeAdapter(result.union_type)
    schema = adapter.json_schema(schema_generator=_DeterministicJsonSchema)
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

    The published schema *is* the top-level model's own schema,
    ``result.top_level_type.model_json_schema()`` generated with
    :class:`_DeterministicJsonSchema` (set members in canonical order --
    pydantic's default generator writes them in hash order): nested models
    sit in ``$defs`` behind ``$ref``s, and the
    ``x-`` extensions (``x-composed-layers``, ``x-inline-wiring``,
    ``x-inline-only-layers``, ...) come from the model's
    ``json_schema_extra`` -- see :func:`conscribe.config.composed.build_composed_config`.
    Nothing is added here, so the in-process model, the generated source and
    this schema cannot drift apart.

    Args:
        result: The ``ComposedConfigResult`` from ``build_composed_config()``.

    Returns:
        A JSON Schema dict.
    """
    return result.top_level_type.model_json_schema(schema_generator=_DeterministicJsonSchema)


def _inject_degraded_info(
    schema: dict[str, Any],
    result: LayerConfigResult,
) -> None:
    """Inject degradation metadata into a JSON Schema dict.

    Adds:
    - Top-level ``x-degraded-fields`` for programmatic consumption.
    - Per-property ``x-degraded-from`` and ``description`` annotations for
      IDE hover support: on the root ``properties`` when the schema is a
      single per-key model (a one-key layer -- also when its ``$defs`` hold
      nested models or enums), and inside the ``$defs`` entries of the
      per-key models.
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

    # A single-model schema: the per-key model's properties are at the root.
    if "properties" in schema:
        _annotate_properties(schema, degraded)

    # Per-property annotations in $defs, by per-key model name
    model_to_key: dict[str, str] = {}
    for key, model in result.per_key_models.items():
        model_to_key[model.__name__] = key

    for def_name, def_schema in schema.get("$defs", {}).items():
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
    """Annotate a single JSON Schema property with degradation info.

    A property whose schema already carries ``x-degraded-from`` (the field's
    own ``json_schema_extra``) is left as the model publishes it.
    """
    if "x-degraded-from" in prop:
        return
    prop["x-degraded-from"] = df.original_type_repr
    desc = f"[degraded] Type was: {df.original_type_repr} (not serializable in config)"
    if "description" in prop:
        prop["description"] = f"{prop['description']} {desc}"
    else:
        prop["description"] = desc
