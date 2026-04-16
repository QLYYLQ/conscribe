"""Composed config builder for multi-layer YAML schema generation.

Builds a single config schema from multiple layer registrars, with
optional inline wiring: wired fields become the target layer's
discriminated union type instead of ``Literal[...]`` key selectors.

Typical usage::

    from conscribe.config.composed import build_composed_config

    result = build_composed_config(
        {"llm": LLM, "agent": Agent},
        inline_wiring=True,
    )
    schema = generate_composed_json_schema(result)
"""
from __future__ import annotations

import dataclasses
from collections import deque
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from conscribe.config.builder import LayerConfigResult, build_layer_config
from conscribe.wiring import parse_wiring


@dataclasses.dataclass(frozen=True)
class ComposedConfigResult:
    """Result of building a composed config from multiple layers.

    Attributes:
        layer_results: Dict mapping layer name to its ``LayerConfigResult``.
        top_level_type: Dynamically created Pydantic model with a
            ``list[UnionType]`` field per layer.
        dependency_order: Layers in topological order (leaves first).
        inline_wiring: Whether inline wiring replacement was applied.
    """

    layer_results: dict[str, LayerConfigResult]
    top_level_type: type[BaseModel]
    dependency_order: list[str]
    inline_wiring: bool


def build_composed_config(
    registrars: dict[str, Any],
    *,
    inline_wiring: bool = True,
) -> ComposedConfigResult:
    """Build a composed config from multiple layer registrars.

    Args:
        registrars: Dict mapping layer name to ``LayerRegistrar``.
        inline_wiring: If ``True``, wired fields become the target layer's
            discriminated union type. If ``False``, wired fields stay as
            ``Literal[...]`` key selectors (current behavior).

    Returns:
        A ``ComposedConfigResult`` with per-layer results and a top-level model.

    Raises:
        CircularWiringError: If wiring dependencies form a cycle.
    """
    graph = _build_dependency_graph(registrars)
    order = _topological_sort(graph, set(registrars.keys()))

    # Build each layer in dependency order (leaves first)
    built: dict[str, LayerConfigResult] = {}
    for layer_name in order:
        result = build_layer_config(registrars[layer_name])
        if inline_wiring:
            result = _replace_wired_literals_with_unions(result, built)
        built[layer_name] = result

    top_level = _build_top_level_model(built, order)

    return ComposedConfigResult(
        layer_results=built,
        top_level_type=top_level,
        dependency_order=order,
        inline_wiring=inline_wiring,
    )


# ---------------------------------------------------------------------------
# Dependency graph & topological sort
# ---------------------------------------------------------------------------


def _build_dependency_graph(
    registrars: dict[str, Any],
) -> dict[str, set[str]]:
    """Build a directed dependency graph from wiring declarations.

    For each registrar, iterates all registered classes and parses their
    ``__wiring__`` to find which other registries they reference.  Only
    edges where the target registry is also in ``registrars`` are included.

    Returns:
        Dict mapping layer name to set of layer names it depends on.
    """
    layer_names = set(registrars.keys())
    graph: dict[str, set[str]] = {name: set() for name in layer_names}

    for layer_name, registrar in registrars.items():
        all_classes = registrar.get_all()
        for cls in all_classes.values():
            specs = parse_wiring(cls)
            for spec in specs:
                if spec.registry_name and spec.registry_name in layer_names:
                    if spec.registry_name != layer_name:  # skip self-refs
                        graph[layer_name].add(spec.registry_name)

    return graph


def _topological_sort(
    graph: dict[str, set[str]],
    all_nodes: set[str],
) -> list[str]:
    """Kahn's algorithm. Returns nodes in dependency order (leaves first).

    Raises:
        CircularWiringError: If a cycle is detected.
    """
    from conscribe.exceptions import CircularWiringError

    # Compute in-degree (how many layers depend on each layer)
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}
    # Reverse edges: for topological sort we want "depended-on" layers first
    reverse: dict[str, set[str]] = {n: set() for n in all_nodes}
    for node, deps in graph.items():
        for dep in deps:
            reverse[dep].add(node)
            in_degree[node] += 1  # node depends on dep

    # Start with nodes that have no dependencies (leaves)
    queue: deque[str] = deque()
    for node in sorted(all_nodes):  # sorted for deterministic order
        if in_degree[node] == 0:
            queue.append(node)

    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for dependent in sorted(reverse[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(all_nodes):
        # Find cycle for error message
        remaining = all_nodes - set(result)
        cycle = sorted(remaining)
        raise CircularWiringError(cycle)

    return result


# ---------------------------------------------------------------------------
# Inline wiring replacement
# ---------------------------------------------------------------------------


def _replace_wired_literals_with_unions(
    result: LayerConfigResult,
    built_results: dict[str, LayerConfigResult],
) -> LayerConfigResult:
    """Replace Literal wired fields with target layer union types.

    For each per-key model that has ``__wired_fields__``, checks if the
    wired registry has already been built.  If so, replaces the
    ``Literal[...]`` field with the target layer's union type (or a subset
    union for Mode 2 wiring).

    Returns a new ``LayerConfigResult`` with rebuilt models and union type.
    """
    new_per_key: dict[str, type[BaseModel]] = {}
    changed = False

    for key, model in result.per_key_models.items():
        wired_fields: dict[str, str] = getattr(model, "__wired_fields__", {})
        if not wired_fields:
            new_per_key[key] = model
            continue

        replacements: dict[str, Any] = {}  # field_name -> new type
        for field_name, registry_name in wired_fields.items():
            if registry_name == "literal" or registry_name not in built_results:
                continue  # Mode 3 or target not in composed set

            target_result = built_results[registry_name]
            field_info = model.model_fields.get(field_name)
            if field_info is None:
                continue

            # Extract allowed keys from the current Literal type
            allowed_keys = _extract_literal_keys(field_info.annotation)
            if allowed_keys is None:
                continue

            # Build the inline union type (full or subset)
            inline_type = _build_inline_type(
                target_result, allowed_keys, field_info.annotation,
            )
            if inline_type is not None:
                replacements[field_name] = inline_type

        if not replacements:
            new_per_key[key] = model
            continue

        # Rebuild model with replaced field types
        changed = True
        new_model = _rebuild_model_with_replacements(model, replacements)
        new_per_key[key] = new_model

    if not changed:
        return result

    # Rebuild union type
    union_type = _rebuild_union_type(new_per_key, result)

    return LayerConfigResult(
        union_type=union_type,
        per_key_models=new_per_key,
        layer_name=result.layer_name,
        discriminator_field=result.discriminator_field,
        degraded_fields=result.degraded_fields,
        discriminator_fields=result.discriminator_fields,
        key_separator=result.key_separator,
        per_segment_models=result.per_segment_models,
    )


def _extract_literal_keys(annotation: Any) -> tuple[str, ...] | None:
    """Extract string keys from a Literal type annotation.

    Handles: ``Literal[...]``, ``Optional[Literal[...]]``,
    ``Union[Literal[...], None]``.

    Returns None if the annotation is not a Literal-based type.
    """
    origin = get_origin(annotation)

    if origin is Literal:
        args = get_args(annotation)
        if args and all(isinstance(a, str) for a in args):
            return tuple(args)
        return None

    # Optional[Literal[...]] or Union[Literal[...], None]
    if origin is Union:
        args = get_args(annotation)
        for arg in args:
            if arg is type(None):
                continue
            keys = _extract_literal_keys(arg)
            if keys is not None:
                return keys
        return None

    # Annotated[Literal[...], ...]
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            return _extract_literal_keys(args[0])

    return None


def _build_inline_type(
    target_result: LayerConfigResult,
    allowed_keys: tuple[str, ...],
    original_annotation: Any,
) -> Any | None:
    """Build the inline union type for a wired field.

    If allowed_keys covers all keys in the target, returns the full union.
    Otherwise, builds a subset union from matching per-key models.

    Preserves Optional wrapping from the original annotation.
    """
    target_keys = set(target_result.per_key_models.keys())
    allowed_set = set(allowed_keys)

    if allowed_set >= target_keys:
        # Full union
        inline_type = target_result.union_type
    else:
        # Subset union
        inline_type = _build_subset_union(target_result, allowed_keys)
        if inline_type is None:
            return None

    # Preserve Optional wrapping
    if _is_optional(original_annotation):
        return Union[inline_type, None]  # type: ignore[return-value]

    return inline_type


def _build_subset_union(
    target_result: LayerConfigResult,
    allowed_keys: tuple[str, ...],
) -> Any | None:
    """Build a discriminated union from a subset of a layer's models.

    Returns None if no matching models found.
    """
    subset_models = []
    for key in allowed_keys:
        model = target_result.per_key_models.get(key)
        if model is not None:
            subset_models.append(model)

    if not subset_models:
        return None

    if len(subset_models) == 1:
        return subset_models[0]

    return Annotated[
        Union[tuple(subset_models)],  # type: ignore[arg-type]
        Field(discriminator=target_result.discriminator_field),
    ]


def _is_optional(annotation: Any) -> bool:
    """Check if a type annotation is Optional (Union with None)."""
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            return _is_optional(args[0])
    return False


def _rebuild_model_with_replacements(
    model: type[BaseModel],
    replacements: dict[str, Any],
) -> type[BaseModel]:
    """Rebuild a Pydantic model with specific field types replaced.

    Preserves model config, other fields, __wired_fields__, and
    __degraded_fields__ metadata.
    """
    extra = model.model_config.get("extra", "forbid")

    field_definitions: dict[str, Any] = {}
    for field_name, field_info in model.model_fields.items():
        if field_name in replacements:
            new_type = replacements[field_name]
            # Preserve default
            if field_info.is_required():
                field_definitions[field_name] = (new_type, ...)
            else:
                field_definitions[field_name] = (new_type, field_info.default)
        else:
            field_definitions[field_name] = (field_info.annotation, field_info)

    base = type(
        f"_{model.__name__}Base",
        (BaseModel,),
        {"model_config": ConfigDict(extra=extra)},
    )

    new_model = create_model(model.__name__, __base__=base, **field_definitions)

    # Reattach metadata
    wired = getattr(model, "__wired_fields__", None)
    if wired:
        new_model.__wired_fields__ = wired  # type: ignore[attr-defined]
    degraded = getattr(model, "__degraded_fields__", None)
    if degraded:
        new_model.__degraded_fields__ = degraded  # type: ignore[attr-defined]

    return new_model


def _rebuild_union_type(
    per_key_models: dict[str, type[BaseModel]],
    original_result: LayerConfigResult,
) -> Any:
    """Rebuild the discriminated union type from updated per-key models."""
    if len(per_key_models) == 1:
        return next(iter(per_key_models.values()))

    return Annotated[
        Union[tuple(per_key_models.values())],  # type: ignore[arg-type]
        Field(discriminator=original_result.discriminator_field),
    ]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


def _build_top_level_model(
    layer_results: dict[str, LayerConfigResult],
    dependency_order: list[str],
) -> type[BaseModel]:
    """Build the top-level ComposedConfig model.

    Each layer becomes a ``list[union_type]`` field with an empty list default.
    """
    field_defs: dict[str, Any] = {}
    for layer_name in dependency_order:
        result = layer_results[layer_name]
        field_defs[layer_name] = (list[result.union_type], [])  # type: ignore[valid-type]

    return create_model("ComposedConfig", **field_defs)
