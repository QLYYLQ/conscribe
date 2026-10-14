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

Model construction rules (2.0):

- **Check, then validate, then build.**  :func:`build_layer_config` first
  refuses what cannot be built at all (not a registrar, no discriminator,
  an empty layer, nested-mode keys of the wrong length, declarations a
  nested-mode layer does not support), then resolves the wiring of every
  registered class (:func:`conscribe.wiring.validate_wiring`, every wiring
  problem of the layer in one
  :class:`~conscribe.exceptions.WiringResolutionError`), and only then
  builds.
- **Rebuild by inheritance.**  A per-key model *subclasses* the model it
  is built from (the extracted schema, a Tier-3 ``__config_schema__``, a
  user's BaseModel knob class) instead of copying its fields onto a fresh
  ``BaseModel``, so the source's validators, ``model_config`` and
  ``FieldInfo`` objects survive.  conscribe's extra policy is always set
  explicitly on the derived model: a source that does not set ``extra``
  itself stays ``"forbid"`` instead of drifting to pydantic's ``"ignore"``.
  A source model left incomplete by forward references is rebuilt in its
  own module first; one that stays incomplete is a
  :class:`~conscribe.exceptions.ConfigDeclarationError` naming the missing
  names.  A config model is never derived from a *registered* class -- the
  registrar metaclass would register the config model as an
  implementation (see :func:`_check_derivable`).
- **Discriminators.**  The discriminator field is ``Literal[key]`` with
  default ``key``; when the source model has an ``alias_generator`` it is
  declared with ``alias=<discriminator field>`` so the generator cannot
  rename it (a renamed discriminator makes the union unusable).  A config
  field whose name or alias is the discriminator field -- or, in nested
  mode, a level's discriminator, segment block or segment ``name`` -- is a
  :class:`~conscribe.exceptions.ConfigDeclarationError` instead of being
  silently replaced.
- **Tier-3 owners keep their wiring record.**  The per-key model of a class
  whose ``__config_schema__`` is used verbatim carries ``__wired_fields__``
  / ``__wired_keys__`` for the wired entries that are schema fields, so a
  composed build can inline them like extracted receptors.
- **Generated-model marker.**  Every model built here carries
  ``__conscribe_generated__ = True`` in its own ``__dict__`` (a plain class
  attribute, not a pydantic field).  Code generation defines marked models
  and imports everything else.
- **Names.**  Model class names are valid identifiers (characters that
  cannot appear in one become ``_`` separators; a name that would start
  with a digit, or whose key part is empty, starts with ``_``) and unique
  per build: a later model whose name is taken gets the next free numeric
  suffix (``WebSearchToolsConfig2``), in build order, so names are
  deterministic.
- **Empty layers raise** :class:`~conscribe.exceptions.EmptyLayerError`
  (citing the modules ``discover()`` failed to import).
- **Warnings point at user code.**  :class:`~conscribe.exceptions.DegradationWarning`
  and :class:`~conscribe.exceptions.ExtractionWarning` raised while a class
  is turned into a config model are reported at a class definition
  (``warnings.warn_explicit``), never at a line of conscribe: the registered
  class being built, or -- for what the nested builder itself reports -- the
  class that declares the parameter or the unreadable ``__init__``.

Nested mode
===========

Each registered key ``s0.s1...sn`` (exactly one segment per discriminator
field) gets a per-key model: the level-0 discriminator (``Literal[s0]``),
the level-0 parameters, and a block per further level holding that
segment's ``name`` (``Literal[si]``), its parameters and the next block.

* **Parameters follow the leaf's real constructor** (``walk_init_chain(cls)``:
  hand-written ``__init__`` and ``**kwargs`` forwarding, dataclass / attrs
  generated ``__init__``, pydantic model and pydantic dataclass fields).  A
  parameter's definition (type, ``Field``, default, factory) is the one the
  leaf's constructor uses; its *level* is the level of the root-most class
  in the MRO whose own constructor accepts it.  A class's level is the
  segment count of its ``__registry_key__`` (own or inherited) minus one;
  a class without one takes the level of the nearest keyed class on the
  leaf side.  When no parameter lands on level 0, the deepest level's
  parameters are promoted to level 0 (single-provider families such as
  ``google.default``).
* Fields are built by the extractor's field rules
  (:func:`conscribe.config.extractor._build_param_fields`): ``Field``
  carry, factories and the purity gate, ``None`` rule, annotation
  degradation and errors, ``__config_annotated_only__``, docstrings.
  Types pydantic cannot build a schema for degrade to ``Any`` when the
  level's model is created (reason ``"pydantic_incompatible"``).  Degraded
  fields are recorded in ``LayerConfigResult.degraded_fields`` and on the
  model holding the field (``__degraded_fields__``).
* A constructor whose signature cannot be read (or a Protocol placeholder
  that swallows its arguments) cuts the chain: an
  :class:`~conscribe.exceptions.ExtractionWarning`, and the models of that
  class's level and every level above it (towards level 0) accept unknown
  keys (``extra="allow"``).  Everything else is ``extra="forbid"``.
* A segment block is optional when none of its fields is required.
  Structurally identical segment models (same name, fields, declaring
  classes and nested block -- e.g. the parameter-less ``default`` provider
  of several families) are built once and shared.
* ``__wiring__``, ``__config_schema__`` and ``__config_from_param__`` are
  not supported on nested-mode classes
  (:class:`~conscribe.exceptions.ConfigDeclarationError`).

The compound discriminator (one function per layer, shared by every union
of the layer's models) joins the level-0 value and each segment block's
``name``; a missing name (block absent, or written without ``name``) reads
as ``"default"``, so a segment registered as ``default`` is selected by
omitting it.  A part that is not a string yields no tag (pydantic reports
``union_tag_not_found``).  Segment defaults stay reachable through the
union: when exactly one member under a prefix has a default for the next
block, the tag an omitted block produces (``anthropic.default``) is an
extra tag of that member, so the union accepts exactly the payloads the
published ``oneOf`` schema accepts (see :func:`_default_tags`).

See ``config-typing-design.md`` Section 5 for full specification.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import os
import re
import sys
import unicodedata
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Discriminator, Field, Tag, create_model
from pydantic.fields import FieldInfo

from conscribe.config._utils import is_pydantic_dataclass, is_pydantic_model, own_init
from conscribe.config.degradation import DegradedField, degrade_field_definitions
from conscribe.config.extractor import _build_param_fields, extract_config_schema
from conscribe.config.mro import InitChain, walk_init_chain
from conscribe.exceptions import (
    ConfigDeclarationError,
    ConscribeWarning,
    DegradationWarning,
    EmptyLayerError,
    ExtractionWarning,
)
from conscribe.introspect import config_from_param
from conscribe.wiring import resolve_wiring, validate_wiring

#: Class attribute set to ``True`` in the own ``__dict__`` of every model
#: conscribe builds.  Check it with ``model.__dict__.get(GENERATED_MARKER)``
#: so a user subclass of a generated model is not mistaken for one.
GENERATED_MARKER = "__conscribe_generated__"

#: Custom class attributes a derived model re-attaches from its source
#: model (inheritance would expose them anyway; re-attaching puts them in
#: the derived model's own ``__dict__``).  ``__wired_keys__`` records the
#: allowed keys of each wired field (set by the extractor).
_CARRIED_MODEL_ATTRS: tuple[str, ...] = (
    "__wired_fields__",
    "__wired_keys__",
    "__degraded_fields__",
)

#: Field name of the segment discriminator inside a nested-mode block.
_SEGMENT_NAME = "name"

#: Segment name the compound discriminator reads for a missing ``name``.
_DEFAULT_SEGMENT = "default"


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
            whose types were degraded to ``Any`` (in nested mode: the fields
            of every level of the key).  Empty dict when no degradation
            occurred (default).
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
    ``discriminator_fields`` set.  Before anything is built the layer is
    checked (:func:`_check_buildable`) and the wiring of every registered
    class is resolved (:func:`~conscribe.wiring.validate_wiring`), so all
    wiring problems of the layer are reported together.

    Args:
        registrar: A ``LayerRegistrar`` subclass (created via ``create_registrar``).

    Returns:
        A ``LayerConfigResult`` with the union type and per-key models.

    Raises:
        TypeError: If *registrar* is not a registrar created by
            ``create_registrar()``.
        ValueError: If the registrar has no discriminator configuration set.
        EmptyLayerError: If the layer has no registered classes.
        WiringResolutionError: If any registered class has wiring problems
            (every problem of every class is listed in ``problems``).
        ConfigDeclarationError: If a class declares something the layer
            cannot honour: a nested-mode key whose segment count does not
            match the levels; ``__wiring__`` / ``__config_schema__`` /
            ``__config_from_param__`` on a nested-mode class; a field named
            like a discriminator, segment block or segment ``name``; a
            ``RootModel`` or registered class as config schema; a config
            schema that stays incomplete (unresolvable forward references).
    """
    _check_buildable(registrar)
    validate_wiring(registrar)
    return _build_validated_layer(registrar)


def _check_buildable(registrar: Any) -> None:
    """Refuse a registrar that cannot produce a config union at all.

    Runs before wiring validation: a nested-mode class that declares
    ``__wiring__`` is reported as unsupported, not hidden behind problems
    of the wiring it should not declare.

    Raises:
        TypeError: *registrar* is not a registrar.
        ValueError: The registrar has no discriminator configuration.
        EmptyLayerError: The layer has no registered classes.
        ConfigDeclarationError: A nested-mode layer has a key whose segment
            count does not match its levels, or a class declaring
            ``__config_schema__`` / ``__config_from_param__`` / ``__wiring__``.
    """
    _require_registrar(registrar)
    layer_name = registrar._registry.name
    disc_fields = getattr(registrar, "discriminator_fields", None)
    if not disc_fields and not registrar.discriminator_field:
        raise ValueError(
            f"Registrar for layer {layer_name!r} has no "
            f"discriminator_field set. Cannot build config union."
        )
    classes = registrar.get_all()
    if not classes:
        raise EmptyLayerError([layer_name])
    if disc_fields:
        separator = registrar._key_separator
        _check_nested_keys(classes, layer_name, list(disc_fields), separator)
        _reject_nested_declarations(classes, layer_name, list(disc_fields))


def _require_registrar(registrar: Any) -> None:
    """Raise ``TypeError`` unless *registrar* was created by ``create_registrar()``."""
    from conscribe.registration.registrar import LayerRegistrar

    if isinstance(registrar, type) and issubclass(registrar, LayerRegistrar):
        return
    if isinstance(registrar, type):
        got = f"the class {_qualified_name(registrar)}"
    else:
        shown = repr(registrar)
        if len(shown) > 80:
            shown = shown[:77] + "..."
        got = f"{type(registrar).__name__} {shown}"
    raise TypeError(
        f"build_layer_config() expects a layer registrar created by "
        f"create_registrar(), got {got}."
    )


def _build_validated_layer(
    registrar: type,
    *,
    names: _ModelNames | None = None,
) -> LayerConfigResult:
    """Build a layer whose registrar passed :func:`_check_buildable` and whose
    wiring was validated (the composed builder validates every layer at once).

    Args:
        registrar: The layer's registrar.
        names: The model-name allocator of the build.  A composed build
            passes one allocator to every layer so model names are unique
            across the whole build; a fresh one is used when omitted.
    """
    names = names if names is not None else _ModelNames()
    if getattr(registrar, "discriminator_fields", None):
        return _build_nested_config(registrar, names)
    return _build_flat_config(registrar, names)


def _build_flat_config(registrar: type, names: _ModelNames) -> LayerConfigResult:
    """Build flat (single-level) discriminated union."""
    disc_field = registrar.discriminator_field
    layer_name = registrar._registry.name
    all_classes = registrar.get_all()

    per_key_models: dict[str, type[BaseModel]] = {}
    all_degraded: dict[str, list[Any]] = {}

    # Read MRO config from the registrar
    reg_mro_scope = getattr(registrar, "_mro_scope", "local")
    reg_mro_depth = getattr(registrar, "_mro_depth", None)

    for key, cls in all_classes.items():
        with _warnings_at(cls):
            schema = extract_config_schema(
                cls, mro_scope=reg_mro_scope, mro_depth=reg_mro_depth,
            )
        model_name = names.claim(_build_model_name(key, layer_name))

        if schema is None:
            # No extractable params — discriminator-only model
            model = _create_discriminator_only_model(model_name, disc_field, key)
        else:
            model = _inject_discriminator(
                schema, model_name, disc_field, key, owner=cls, layer_name=layer_name,
            )
            schema_degraded = getattr(schema, "__degraded_fields__", None)
            if schema_degraded:
                all_degraded[key] = list(schema_degraded)

        per_key_models[key] = model

    return LayerConfigResult(
        union_type=_flat_union(per_key_models, disc_field),
        per_key_models=per_key_models,
        layer_name=layer_name,
        discriminator_field=disc_field,
        degraded_fields=all_degraded,
    )


# ---------------------------------------------------------------------------
# Nested mode (compound discriminator)
# ---------------------------------------------------------------------------


def _build_nested_config(registrar: type, names: _ModelNames) -> LayerConfigResult:
    """Build nested (compound) discriminated union.

    Each key segment maps to a Pydantic sub-model level. Level 0 is flat
    at the top level; level 1+ are nested sub-models (see the module
    docstring for the parameter and level rules).

    Example for ``discriminator_fields=["model_type", "provider"]``:
    - ``"openai.azure"`` → flat ``model_type: Literal["openai"]`` + nested
      ``provider: AzureProviderConfig``
    """
    disc_fields = list(registrar.discriminator_fields)
    separator = registrar._key_separator
    layer_name = registrar._registry.name
    all_classes = registrar.get_all()
    n_levels = len(disc_fields)

    per_key_models: dict[str, type[BaseModel]] = {}
    per_segment_models: dict[str, dict[str, type[BaseModel]]] = {
        disc_fields[i]: {} for i in range(1, n_levels)
    }
    all_degraded: dict[str, list[Any]] = {}
    # Structurally identical segment models (e.g. the parameter-less
    # ``default`` provider of several families) are built once and shared.
    segment_models: dict[Any, type[BaseModel]] = {}

    for key in sorted(all_classes):
        cls = all_classes[key]
        segments = key.split(separator)
        params = _extract_params_by_level(cls, n_levels, separator)
        build = _NestedModelBuild(cls, key, layer_name, disc_fields, params, names)

        # Segment models, deepest level first.
        nested_model: type[BaseModel] | None = None
        for level in range(n_levels - 1, 0, -1):
            seg = segments[level]
            field_name = disc_fields[level]
            head: dict[str, Any] = {
                _SEGMENT_NAME: (Literal[seg], seg),  # type: ignore[valid-type]
            }
            tail: dict[str, Any] = {}
            if nested_model is not None:
                tail[disc_fields[level + 1]] = _segment_field(nested_model)
            seg_model = build.model(
                level, _segment_model_name(seg, field_name), head, tail, shared=segment_models,
            )
            per_segment_models.setdefault(field_name, {})[seg] = seg_model
            nested_model = seg_model

        # Combined model (level 0 flat + level 1 nested)
        head = {disc_fields[0]: (Literal[segments[0]], segments[0])}  # type: ignore[valid-type]
        tail = {}
        if nested_model is not None:
            tail[disc_fields[1]] = _segment_field(nested_model)
        per_key_models[key] = build.model(0, _build_model_name(key, layer_name), head, tail)
        if build.degraded:
            all_degraded[key] = list(build.degraded)

    union_type = _build_compound_union(per_key_models, disc_fields, separator, layer_name)

    return LayerConfigResult(
        union_type=union_type,
        per_key_models=per_key_models,
        layer_name=layer_name,
        discriminator_field=disc_fields[0],
        degraded_fields=all_degraded,
        discriminator_fields=disc_fields,
        key_separator=separator,
        per_segment_models=per_segment_models,
    )


def _check_nested_keys(
    all_classes: Mapping[str, type],
    layer_name: str,
    disc_fields: list[str],
    separator: str,
) -> None:
    """Refuse registered keys whose segment count does not match the levels.

    A key that is not ``<level 0>.<level 1>...`` has no place in the nested
    config; 1.x dropped such classes from the union silently.

    Raises:
        ConfigDeclarationError: Naming every offending key and its class.
    """
    n_levels = len(disc_fields)
    bad = [
        (key, cls)
        for key, cls in sorted(all_classes.items())
        if len(key.split(separator)) != n_levels
    ]
    if not bad:
        return
    shape = separator.join(f"<{field}>" for field in disc_fields)
    listed = ", ".join(
        f"{key!r} ({_qualified_name(cls)}, {len(key.split(separator))} segment"
        f"{'' if len(key.split(separator)) == 1 else 's'})"
        for key, cls in bad
    )
    raise ConfigDeclarationError(
        _qualified_name(bad[0][1]),
        f"layer {layer_name!r} is a nested-mode layer with {n_levels} levels "
        f"(discriminator_fields={disc_fields}, key_separator={separator!r}), so every "
        f"registered key must have exactly {n_levels} segments ({shape!r}); these do "
        f"not and would have no place in the config: {listed}. Register each class "
        f"under a {n_levels}-segment key, mark a shared base __abstract__ = True, or "
        f"move the class to another layer.",
    )


def _reject_nested_declarations(
    all_classes: Mapping[str, type],
    layer_name: str,
    disc_fields: list[str],
) -> None:
    """Refuse declarations the nested builder would otherwise ignore silently.

    The nested builder assembles each level from constructor parameters; it
    has no wiring pass, no Tier-3 path and no single-parameter flattening
    yet.  A class that declares one of them is refused instead of producing
    a config that quietly disagrees with the class.

    Raises:
        ConfigDeclarationError: For the first offending class (key order).
    """
    mode = (
        f"layer {layer_name!r} is a nested-mode layer "
        f"(discriminator_fields={list(disc_fields)})"
    )
    for key in sorted(all_classes):
        cls = all_classes[key]
        owner = _qualified_name(cls)
        schema = getattr(cls, "__config_schema__", None)
        if schema is not None:
            raise ConfigDeclarationError(
                owner,
                f"__config_schema__ ({getattr(schema, '__qualname__', schema)!r}) is not "
                f"supported for nested-mode layers yet: {mode}, whose levels are built "
                f"from __init__ parameters, so the schema would be ignored (key "
                f"{key!r}). Remove __config_schema__, or register the class in a "
                f"flat-mode layer.",
            )
        from_param = config_from_param(cls)
        if from_param is not None:
            raise ConfigDeclarationError(
                owner,
                f"__config_from_param__ = {from_param!r} is not supported for "
                f"nested-mode layers yet: {mode}, whose levels are built from the "
                f"constructor parameters, so the parameter could not be flattened "
                f"(key {key!r}). Remove __config_from_param__ (the parameter's model "
                f"becomes a nested block of its level), or register the class in a "
                f"flat-mode layer.",
            )
        entries = _declared_wiring_entries(cls)
        if entries:
            raise ConfigDeclarationError(
                owner,
                f"__wiring__ is not supported for nested-mode layers yet: {mode}, so "
                f"the wiring entries {entries} would be ignored (key {key!r}). Remove "
                f"them (set an inherited entry to None to exclude it), or register "
                f"the class in a flat-mode layer.",
            )
        for attr in ("__config_mro_scope__", "__config_mro_depth__"):
            if any(attr in vars(klass) for klass in cls.__mro__):
                raise ConfigDeclarationError(
                    owner,
                    f"{attr} is not supported for nested-mode layers: {mode}, whose "
                    f"levels always follow the leaf class's own constructor, so the "
                    f"setting would be ignored (key {key!r}). Remove it, or register "
                    f"the class in a flat-mode layer.",
                )


def _declared_wiring_entries(cls: type) -> list[str]:
    """Names of the ``__wiring__`` entries *cls* declares, merged along the MRO.

    ``None`` values exclude inherited entries, as in wiring resolution.  A
    ``__wiring__`` that is not a mapping counts as a declaration too (it is
    reported by name, not interpreted).
    """
    merged: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        raw = vars(klass).get("__wiring__")
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            return [f"<__wiring__ = {raw!r}>"]
        merged.update(raw)
    return sorted(name for name, value in merged.items() if value is not None)


def _segment_field(seg_model: type[BaseModel]) -> tuple[type[BaseModel], Any]:
    """Field definition for a nested segment sub-model (level 1+).

    The segment block is optional when every field of the sub-model has a
    default or a factory (``FieldInfo.is_required()``).  The default is an
    instance when all defaults are plain values (pydantic copies it per
    config); when any field uses a ``default_factory`` it is
    ``Field(default_factory=<segment model>)`` instead, so factories run
    once per config rather than once at build time.
    """
    fields = seg_model.model_fields.values()
    if any(fi.is_required() for fi in fields):
        return (seg_model, ...)
    if any(fi.default_factory is not None for fi in fields):
        return (seg_model, Field(default_factory=seg_model))
    return (seg_model, seg_model())


@dataclasses.dataclass
class _NestedParams:
    """The config fields of one nested-mode leaf class, split by key level.

    Attributes:
        fields: ``{level: {name: (type, FieldInfo)}}``.
        declared_on: ``{name: class}`` -- the class that declares each
            parameter (error messages, degradation records, warnings).
        degraded: Fields the extractor degraded (annotation not resolvable).
        open_levels: Levels whose models accept unknown keys
            (``extra="allow"``) because the constructor chain was cut.
    """

    fields: dict[int, dict[str, tuple[Any, FieldInfo]]]
    declared_on: dict[str, type]
    degraded: list[DegradedField]
    open_levels: frozenset[int] = frozenset()


def _extract_params_by_level(cls: type, n_levels: int, separator: str) -> _NestedParams:
    """Config fields of the leaf *cls*, split by key level (nested mode).

    Parameters come from the leaf's runtime constructor chain
    (``walk_init_chain(cls)``), so every definition is the one ``cls(...)``
    uses; a parameter's level is the level of the root-most class in the
    MRO whose own constructor accepts it (see :func:`_class_levels`).
    Within a level the fields keep that root-first declaration order.
    Fields are built by the extractor's field rules
    (:func:`~conscribe.config.extractor._build_param_fields`).  When no
    field lands on level 0, the deepest level's fields are promoted to
    level 0 (single-provider families such as ``google.default``).

    Raises:
        ConfigAnnotationError / ConfigDeclarationError: From the field rules.
    """
    leaf_level = n_levels - 1
    levels = _class_levels(cls, separator, leaf_level)
    chain = walk_init_chain(cls)
    params = {p.name: p for p in chain.params if p.keyword}

    # Root-first: the first class whose own constructor accepts a name fixes
    # its level and its position.
    level_of: dict[str, int] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        for name in _own_param_names(klass):
            if name in params and name not in level_of:
                level_of[name] = levels.get(klass, leaf_level)
    for name in params:  # accepted by the leaf, declared by no constructor of its own
        level_of.setdefault(name, leaf_level)

    ordered = [params[name] for name in level_of]
    with _warnings_at(cls):
        fields, degraded = _build_param_fields(cls, ordered, leaf=cls)

    by_level: dict[int, dict[str, tuple[Any, FieldInfo]]] = {}
    for name, level in level_of.items():
        if name in fields:
            by_level.setdefault(level, {})[name] = fields[name]
    # Promote leaf-only params to level 0 when no ancestor contributes any:
    # single-provider types (e.g. google.default) define every parameter on
    # the leaf class, and those appear flat in the config.
    if 0 not in by_level and by_level:
        by_level[0] = by_level.pop(max(by_level))

    open_levels = _warn_unreadable_chain(cls, chain, levels, leaf_level)
    return _NestedParams(
        fields=by_level,
        declared_on={name: param.declared_on for name, param in params.items()},
        degraded=list(degraded),
        open_levels=open_levels,
    )


def _class_levels(cls: type, separator: str, leaf_level: int) -> dict[type, int]:
    """The key level of every class in *cls*'s MRO (``object`` excluded).

    The leaf is at *leaf_level*.  Another class with a ``__registry_key__``
    (its own or inherited, keyed or abstract alike) is at that key's
    segment count minus one (capped at *leaf_level*); a class without one
    takes the level of the nearest keyed class on the leaf side.
    """
    levels: dict[type, int] = {}
    current = leaf_level
    for klass in cls.__mro__:
        if klass is object:
            continue
        if klass is not cls:
            key = getattr(klass, "__registry_key__", None)
            if isinstance(key, str) and key:
                current = min(key.count(separator), leaf_level)
        levels[klass] = current
    return levels


def _own_param_names(klass: type) -> tuple[str, ...]:
    """Names *klass*'s own constructor accepts by keyword, in signature order.

    Its own ``__init__`` (hand-written or generated), or its fields when it
    is a pydantic model / pydantic dataclass; empty when the class defines
    no constructor of its own.
    """
    if own_init(klass) is None and not is_pydantic_model(klass) and not is_pydantic_dataclass(klass):
        return ()
    return tuple(
        p.name for p in walk_init_chain(klass, depth=0).params if p.level == 0 and p.keyword
    )


def _warn_unreadable_chain(
    cls: type,
    chain: InitChain,
    levels: Mapping[type, int],
    leaf_level: int,
) -> frozenset[int]:
    """Report the parts of *cls*'s constructor conscribe cannot read.

    Emits an :class:`~conscribe.exceptions.ExtractionWarning` (at the class
    that cannot be read) for an ``__init__`` that is not a Python function
    (its parameters are typed ``Any``) and for a chain cut by an unreadable
    signature or a swallowing Protocol placeholder.

    Returns:
        The levels whose models must accept unknown keys: the level of the
        class where the chain was cut and every level towards level 0.
    """
    owner = _qualified_name(cls)
    seen: list[type] = []
    for param in chain.params:
        function = param.function
        if (
            param.source == "init"
            and function is not None
            and not hasattr(function, "__code__")
            and param.init_owner not in seen
        ):
            seen.append(param.init_owner)
            _warn_at(
                param.init_owner,
                f"{owner}: {_qualified_name(param.init_owner)}.__init__ is not a Python "
                f"function, so its annotations cannot be read; its parameters are typed "
                f"Any. Accept them in a Python __init__ to type them (nested-mode layers "
                f"do not support __config_schema__).",
                ExtractionWarning,
            )
    if chain.stop_reason not in ("unreadable", "swallowing_placeholder"):
        return frozenset()
    where = chain.stopped_at if chain.stopped_at is not None else cls
    level = levels.get(where, leaf_level)
    if chain.stop_reason == "unreadable":
        cause = f"cannot read the signature of {_qualified_name(where)}.__init__"
    else:
        cause = (
            f"{_qualified_name(where)}.__init__ is a Protocol placeholder that "
            f"swallows every argument (typing_extensions on Python < 3.12)"
        )
    _warn_at(
        where,
        f"{owner}: {cause}; the parameters it takes are not in the config, whose "
        f"models of level {level} and above (towards level 0) therefore accept "
        f"unknown keys (extra='allow'). Give the class a Python __init__ with a "
        f"readable signature.",
        ExtractionWarning,
    )
    return frozenset(range(level + 1))


class _NestedModelBuild:
    """Creates the models of one nested-mode key, level by level."""

    def __init__(
        self,
        cls: type,
        key: str,
        layer_name: str,
        disc_fields: list[str],
        params: _NestedParams,
        names: _ModelNames,
    ) -> None:
        self.cls = cls
        self.key = key
        self.layer_name = layer_name
        self.disc_fields = disc_fields
        self.params = params
        self.names = names
        #: Every degraded field of the key (all levels).
        self.degraded: list[DegradedField] = []

    def model(
        self,
        level: int,
        name: str,
        head: dict[str, Any],
        tail: dict[str, Any],
        *,
        shared: dict[Any, type[BaseModel]] | None = None,
    ) -> type[BaseModel]:
        """Create the model of *level*: ``head`` fields, the level's parameters, ``tail`` fields.

        Args:
            level: The key level.
            name: The model's name before de-duplication (see :class:`_ModelNames`).
            head: Fields before the level's parameters (its discriminator).
            tail: Fields after them (the next segment block).
            shared: Models already built in this layer by structural
                signature; a model with the same signature is reused instead
                of built again (segment models).

        Raises:
            ConfigDeclarationError: A parameter collides with a field the
                level reserves (its discriminator / segment ``name``, the next
                segment block).
        """
        level_fields = self.params.fields.get(level, {})
        self._check_reserved(level, level_fields, reserved=[*head, *tail])
        extra = "allow" if level in self.params.open_levels else "forbid"
        degraded = [d for d in self.params.degraded if d.field_name in level_fields]
        signature = None
        if shared is not None:
            signature = self._signature(name, extra, head, level_fields, tail, degraded)
            reused = shared.get(signature)
            if reused is not None:
                self.degraded.extend(reused.__dict__.get("__degraded_fields__", ()))
                return reused
        name = self.names.claim(name)
        defs = {**head, **level_fields, **tail}
        from pydantic import PydanticSchemaGenerationError, PydanticUserError

        try:
            model = _derive_model(name, BaseModel, defs, extra=extra, owner=self.cls)
        except (PydanticSchemaGenerationError, PydanticUserError):
            # Try-first, like the extractor: probe the parameters one by one
            # only when pydantic refused the model.
            cleaned, records = degrade_field_definitions(
                level_fields,
                source_class_name=_qualified_name(self.cls),
                source_classes={
                    n: _qualified_name(self.params.declared_on.get(n, self.cls))
                    for n in level_fields
                },
            )
            if not records:
                raise
            for record in records:
                _warn_at(
                    self.params.declared_on.get(record.field_name, self.cls),
                    f"{_qualified_name(self.cls)}.{record.field_name}: pydantic cannot "
                    f"build a schema for {record.original_type_repr!r} (declared on "
                    f"{record.source_class}); the field's type degrades to Any (default, "
                    f"description and constraints are kept).",
                    DegradationWarning,
                )
            degraded.extend(records)
            defs = {**head, **cleaned, **tail}
            model = _derive_model(name, BaseModel, defs, extra=extra, owner=self.cls)
        if degraded:
            model.__degraded_fields__ = list(degraded)  # type: ignore[attr-defined]
            self.degraded.extend(degraded)
        if shared is not None:
            shared[signature] = model
        return model

    def _signature(
        self,
        name: str,
        extra: str,
        head: Mapping[str, Any],
        level_fields: Mapping[str, tuple[Any, FieldInfo]],
        tail: Mapping[str, Any],
        degraded: list[DegradedField],
    ) -> tuple[Any, ...]:
        """What makes two segment models interchangeable.

        The name, extra policy and every field definition: types by
        equality (identity for classes), ``FieldInfo`` objects and defaults
        by ``repr`` (which shows factories and validators by identity), the
        nested block by identity, and the class declaring each parameter.
        """
        entries: list[tuple[Any, ...]] = []
        for field_name, (field_type, info) in {**head, **level_fields, **tail}.items():
            owner = self.params.declared_on.get(field_name) if field_name in level_fields else None
            entries.append((field_name, _hashable(field_type), repr(info), owner))
        return (name, extra, tuple(entries), tuple(degraded))

    def _check_reserved(
        self,
        level: int,
        level_fields: Mapping[str, tuple[Any, FieldInfo]],
        *,
        reserved: list[str],
    ) -> None:
        """Refuse a parameter named (or aliased) like a field the level reserves."""
        for field_name, definition in level_fields.items():
            info = definition[1] if isinstance(definition, tuple) else None
            clash = _clashing_input(field_name, info, reserved)
            if clash is None:
                continue
            if level == 0 and clash == self.disc_fields[0]:
                what = f"the level-0 discriminator field {clash!r}"
            elif clash == _SEGMENT_NAME and level > 0:
                what = (
                    f"the segment name field {clash!r} of the {self.disc_fields[level]!r} "
                    f"block"
                )
            else:
                what = f"the {clash!r} segment block"
            alias = "" if clash == field_name else f" (alias {clash!r})"
            raise ConfigDeclarationError(
                _qualified_name(self.params.declared_on.get(field_name, self.cls)),
                f"the parameter{alias} lands on level {level} of the config of key "
                f"{self.key!r} (layer {self.layer_name!r}, discriminator_fields="
                f"{self.disc_fields}), where it collides with {what}; conscribe writes "
                f"the discriminator there, so the parameter would be silently replaced. "
                f"Rename the parameter.",
                field_name,
            )


def _default_tags(
    per_key_models: Mapping[str, type[BaseModel]],
    disc_fields: list[str],
    separator: str,
) -> dict[str, str]:
    """Extra union tags for omitted segment blocks: ``{tag: leaf key}``.

    The compound discriminator reads a missing segment ``name`` as
    ``"default"``, so a payload that omits the block of level *i* under the
    prefix ``P`` is tagged ``P.default...`` (``default`` for level *i* and
    every deeper level).  When exactly one of *per_key_models* under ``P``
    has a default for that block -- the one branch of the published
    ``oneOf`` that admits the payload -- that tag selects it.  A tag that is
    a real key stays that key's (a segment registered as ``default`` is
    selected by omitting the name); a tag two leaves would claim selects
    neither.
    """
    n_levels = len(disc_fields)
    claims: dict[str, set[str]] = {}
    for key in sorted(per_key_models):
        segments = key.split(separator)
        model: Any = per_key_models[key]
        for level in range(1, n_levels):
            info = model.model_fields.get(disc_fields[level]) if _is_model(model) else None
            if info is None:
                break
            if not info.is_required():
                tag = separator.join([*segments[:level], *[_DEFAULT_SEGMENT] * (n_levels - level)])
                claims.setdefault(tag, set()).add(key)
            model = info.annotation
    return {
        tag: next(iter(keys))
        for tag, keys in sorted(claims.items())
        if len(keys) == 1 and tag not in per_key_models
    }


@functools.cache
def _compound_discriminator(
    disc_fields: tuple[str, ...],
    separator: str,
    layer_name: str,
) -> Callable[[Any], str | None]:
    """The compound discriminator function of a nested layer.

    Cached, so every union of one layer's models (the layer union, composed
    subset unions, rebuilt unions) shares one function object -- code
    generation matches it by identity.

    The tag is the level-0 value, then each segment block's ``name``
    (``"default"`` when the block or its ``name`` is missing), joined with
    the separator.  When any part is not a string the function returns
    ``None`` and pydantic reports ``union_tag_not_found`` (the parts of the
    original 1.x function, which code generation mirrors, plus that check).
    """
    head = disc_fields[0]

    def _discriminate(v: Any) -> str | None:
        parts = []
        for i, field_name in enumerate(disc_fields):
            if i == 0:
                # Level 0 is flat
                parts.append(v.get(head, "") if isinstance(v, dict) else getattr(v, head, ""))
            elif isinstance(v, dict):
                # Level 1+ is nested — walk into nested dicts/models
                nested: Any = v
                for j in range(1, i + 1):
                    nested = nested.get(disc_fields[j], {}) if isinstance(nested, dict) else {}
                parts.append(
                    nested.get(_SEGMENT_NAME, _DEFAULT_SEGMENT)
                    if isinstance(nested, dict)
                    else getattr(nested, _SEGMENT_NAME, _DEFAULT_SEGMENT)
                )
            else:
                obj = v
                for j in range(1, i + 1):
                    obj = getattr(obj, disc_fields[j], None)
                parts.append(getattr(obj, _SEGMENT_NAME, _DEFAULT_SEGMENT) if obj else _DEFAULT_SEGMENT)
        if not all(isinstance(part, str) for part in parts):
            return None
        return separator.join(parts)

    _discriminate.__qualname__ = f"_discriminate_{layer_name}"
    _discriminate.__name__ = f"_discriminate_{layer_name}"
    return _discriminate


def _build_compound_union(
    per_key_models: dict[str, type[BaseModel]],
    disc_fields: list[str],
    separator: str,
    layer_name: str,
) -> Any:
    """Build compound discriminated union with the layer's discriminator function.

    Members are tagged with their full keys (sorted), followed by the
    omitted-block tags of :func:`_default_tags` (the same model again under
    another tag; pydantic lists each model once in the JSON schema).  Also
    used for *subsets* of a nested layer (composed inline wiring): any
    subset of a layer's per-key models shares the layer's discriminator
    function, and its omitted-block tags are computed over the subset.
    """
    if len(per_key_models) == 1:
        return next(iter(per_key_models.values()))

    discriminate = _compound_discriminator(tuple(disc_fields), separator, layer_name)

    # Build annotated union members with Tag
    members = []
    for key in sorted(per_key_models.keys()):
        model = per_key_models[key]
        members.append(Annotated[model, Tag(key)])
    for tag, key in _default_tags(per_key_models, disc_fields, separator).items():
        members.append(Annotated[per_key_models[key], Tag(tag)])

    union_type = Annotated[
        Union[tuple(members)],  # type: ignore[arg-type]
        Discriminator(discriminate),
    ]

    return union_type


# ---------------------------------------------------------------------------
# Union helpers (shared with the composed builder)
# ---------------------------------------------------------------------------


def _flat_union(per_key_models: dict[str, type[BaseModel]], disc_field: str) -> Any:
    """``Annotated[Union[...], Field(discriminator=...)]``, or the model itself for one key."""
    if len(per_key_models) == 1:
        return next(iter(per_key_models.values()))
    return Annotated[
        Union[tuple(per_key_models.values())],  # type: ignore[arg-type]
        Field(discriminator=disc_field),
    ]


def _layer_union(
    result: LayerConfigResult,
    per_key_models: dict[str, type[BaseModel]],
) -> Any:
    """The union of *per_key_models* built the way *result*'s layer builds unions.

    Flat layers get ``Field(discriminator=...)``; nested layers get the
    compound ``Discriminator(callable)`` + ``Tag`` form with the layer's
    discriminator function.  Works for the full set of a layer's models and
    for any subset of them.
    """
    if result.discriminator_fields:
        return _build_compound_union(
            per_key_models,
            list(result.discriminator_fields),
            result.key_separator,
            result.layer_name,
        )
    return _flat_union(per_key_models, result.discriminator_field)


def _union_member_keys(result: LayerConfigResult) -> tuple[str, ...]:
    """Keys of ``result.union_type``'s members, in member order."""
    if result.discriminator_fields:
        return tuple(sorted(result.per_key_models))
    return tuple(result.per_key_models)


# ---------------------------------------------------------------------------
# Model derivation (rebuild by inheritance)
# ---------------------------------------------------------------------------


def _derive_model(
    name: str,
    base: type[BaseModel],
    fields: Mapping[str, Any],
    *,
    extra: str | None,
    module: str = __name__,
    carry_from: type[BaseModel] | None = None,
    attrs: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    validators: Mapping[str, Any] | None = None,
    owner: Any = None,
) -> type[BaseModel]:
    """Create a conscribe model by subclassing *base*.

    The single place where builder and composed create models.  The new
    model inherits *base*'s fields, validators and ``model_config``; only
    *fields* are (re)declared.  Never pass a ``FieldInfo`` taken from
    ``model_fields`` in *fields* -- build a fresh one (see
    :mod:`conscribe.config._fieldinfo`).

    A *base* left incomplete by forward references (a user model whose
    annotations name a class defined after it, a model derived from one) is
    rebuilt first -- its incomplete user-defined ancestors in their own
    modules, then *base* -- so the derived model resolves.

    Args:
        name: Class name of the new model.
        base: The model to inherit from (``BaseModel`` for a fresh one).
        fields: ``create_model`` field definitions to add or override.
        extra: Extra policy set explicitly on the new model (``None``
            leaves the inherited / pydantic default untouched).
        module: ``__module__`` of the new model.
        carry_from: Model whose custom attributes
            (``__wired_fields__``, ``__wired_keys__``,
            ``__degraded_fields__``) are re-attached.
        attrs: Further class attributes to set on the new model.
        config: Further ``model_config`` keys for the new model (e.g.
            ``json_schema_extra``), passed as class keywords exactly like
            *extra*, so ``model_config`` reports them.
        validators: Validators to add, as for ``create_model``'s
            ``__validators__`` (``{name: field_validator(...)(func)}``).
        owner: The registered class the model is built for (named in
            errors); *base* when omitted.

    Returns:
        The new model, marked with ``__conscribe_generated__ = True``.

    Raises:
        ConfigDeclarationError: The new model stays incomplete: a model it
            derives from names something that cannot be resolved in the
            module that declares it.
    """
    _complete_ancestors(base)
    cls_kwargs: dict[str, Any] = dict(config or {})
    if extra is not None:
        cls_kwargs["extra"] = extra
    model = create_model(  # type: ignore[call-overload]
        name,
        __base__=base,
        __module__=module,
        __cls_kwargs__=cls_kwargs,
        __validators__=dict(validators) if validators else None,
        **fields,
    )
    if carry_from is not None:
        for attr in _CARRIED_MODEL_ATTRS:
            value = getattr(carry_from, attr, None)
            if value:
                setattr(model, attr, type(value)(value) if isinstance(value, (dict, list)) else value)
    for attr, value in (attrs or {}).items():
        setattr(model, attr, value)
    _mark_generated(model)
    if not getattr(model, "__pydantic_complete__", True):
        _raise_incomplete(model, base, owner)
    return model


def _complete_ancestors(base: type[BaseModel]) -> None:
    """Rebuild the incomplete models of *base*'s MRO, root first (*base* last).

    pydantic resolves a model's forward references when the model is
    (re)built, and a subclass created while its parent was incomplete
    inherits the unresolved fields.  A user model is rebuilt in its own
    module; a model conscribe made (the extractor's flattened model, a
    per-key model) is rebuilt with the namespaces of the user modules above
    it, where the inherited annotations were written.
    """
    if not (_is_model(base) and not getattr(base, "__pydantic_complete__", True)):
        return
    _walk_rebuild(base, raise_errors=False)


def _walk_rebuild(model: type[BaseModel], *, raise_errors: bool) -> None:
    """Rebuild every incomplete model of *model*'s MRO, root first (see above)."""
    user_namespace: dict[str, Any] = {}
    for klass in reversed(model.__mro__):
        if not _is_model(klass):
            continue
        internal = _is_conscribe_module(klass.__module__)
        if not internal:
            module = sys.modules.get(klass.__module__)
            if module is not None:
                user_namespace.update(vars(module))
        if getattr(klass, "__pydantic_complete__", True):
            continue
        if internal:
            klass.model_rebuild(raise_errors=raise_errors, _types_namespace=dict(user_namespace))
        else:
            # Depth 0: resolve in the model's own module (and the namespace it
            # was defined in), not in this frame.
            klass.model_rebuild(raise_errors=raise_errors, _parent_namespace_depth=0)


def _is_conscribe_module(module: Any) -> bool:
    return isinstance(module, str) and (module == "conscribe" or module.startswith("conscribe."))


def _raise_incomplete(model: type[BaseModel], base: type[BaseModel], owner: Any) -> None:
    """Raise ``ConfigDeclarationError`` for a derived model that stays incomplete."""
    from pydantic.errors import PydanticUndefinedAnnotation

    found = "a forward reference"
    try:
        _walk_rebuild(model, raise_errors=True)
    except PydanticUndefinedAnnotation as exc:
        found = repr(exc.name)
    declaring = [
        _qualified_name(klass)
        for klass in reversed(model.__mro__)
        if _is_model(klass) and not _is_conscribe_module(klass.__module__)
    ]
    where = f" (declared on {', '.join(declaring)})" if declaring else ""
    raise ConfigDeclarationError(
        _qualified_name(owner if owner is not None else base),
        f"the config model {model.__name__} cannot be completed: it derives from "
        f"{base.__qualname__}, whose annotations{where} name {found}, which cannot be "
        f"resolved. pydantic resolves forward references in the module that declares "
        f"the model: define or import the name there (a name imported only under "
        f"'if TYPE_CHECKING:' does not count), then build again.",
    )


def _mark_generated(model: type[BaseModel]) -> None:
    """Set the generated-model marker in *model*'s own ``__dict__``."""
    setattr(model, GENERATED_MARKER, True)


def _hashable(value: Any) -> Any:
    """*value* when it is hashable (compared by ``==``), else its ``repr``."""
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return value


def _is_model(tp: Any) -> bool:
    """A strict subclass of ``pydantic.BaseModel``."""
    return isinstance(tp, type) and issubclass(tp, BaseModel) and tp is not BaseModel


def _check_derivable(source: type[BaseModel], owner: type) -> None:
    """Refuse to derive a per-key config model from *source* when that cannot work.

    Raises:
        ConfigDeclarationError: *source* is a ``RootModel`` (no named fields
            to add a discriminator to), or *source* is -- or inherits from --
            a registered class: subclassing it would run the registrar
            metaclass / ``__init_subclass__`` hook and register the config
            model as an implementation.
    """
    from pydantic import RootModel

    from conscribe.registration.meta_base import AutoRegistrarBase

    where = _qualified_name(owner)
    label = getattr(source, "__qualname__", repr(source))
    if issubclass(source, RootModel):
        raise ConfigDeclarationError(
            where,
            f"the config schema {label} is a RootModel; a per-key config model "
            f"needs named fields and a discriminator field. Use a BaseModel "
            f"subclass instead.",
        )
    if isinstance(source, AutoRegistrarBase) or any(
        "__registry_key__" in vars(klass) for klass in source.__mro__
    ):
        raise ConfigDeclarationError(
            where,
            f"the config schema {label} is (or inherits from) a registered class; "
            f"the per-key config model subclasses its schema, which would register "
            f"the config model as an implementation. Use a plain BaseModel "
            f"subclass as the config schema.",
        )


def _qualified_name(cls: Any) -> str:
    """``module.QualName`` of a class, for error messages."""
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", repr(cls))
    return f"{module}.{qualname}" if module else qualname


# ---------------------------------------------------------------------------
# Reserved field names
# ---------------------------------------------------------------------------


def _clashing_input(name: str, info: FieldInfo | None, reserved: list[str]) -> str | None:
    """The reserved name *name* (or one of its aliases) takes, if any."""
    for candidate in _input_names(name, info):
        if candidate in reserved:
            return candidate
    return None


def _input_names(name: str, info: FieldInfo | None) -> list[str]:
    """The field name and the top-level input keys of its aliases."""
    names = [name]
    if info is None:
        return names
    for alias in (info.alias, info.validation_alias):
        if isinstance(alias, str):
            names.append(alias)
        else:
            for choice in getattr(alias, "choices", None) or (alias,):
                if isinstance(choice, str):
                    names.append(choice)
                else:
                    path = getattr(choice, "path", None)
                    if path and isinstance(path[0], str):
                        names.append(path[0])
    return list(dict.fromkeys(names))


def _check_discriminator_clash(
    schema: type[BaseModel],
    disc_field: str,
    key: str,
    *,
    owner: Any,
    layer_name: str | None,
) -> None:
    """Refuse a source model with a field named (or aliased) like the discriminator.

    Raises:
        ConfigDeclarationError: Naming the class, the field and the fix.
    """
    for field_name, info in schema.model_fields.items():
        clash = _clashing_input(field_name, info, [disc_field])
        if clash is None:
            continue
        layer = f" of layer {layer_name!r}" if layer_name else ""
        alias = "" if clash == field_name else f" (alias {clash!r})"
        raise ConfigDeclarationError(
            _qualified_name(owner),
            f"the config field{alias} collides with the discriminator field "
            f"{disc_field!r}{layer}: the per-key config model writes the registry key "
            f"there (Literal[{key!r}]), so the field would be silently replaced. Rename "
            f"the field (the constructor parameter, wiring entry or field of "
            f"{schema.__qualname__}), or create the registrar with another "
            f"discriminator_field.",
            field_name,
        )


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------


class _ModelNames:
    """The model class names handed out in one build.

    :meth:`claim` returns the requested name, or -- when a model of the
    build already has it -- the name with the next free numeric suffix
    (``WebSearchToolsConfig2``).  Names are claimed in build order, so they
    are deterministic.  A composed build shares one instance across its
    layers.
    """

    def __init__(self, reserved: tuple[str, ...] = ()) -> None:
        self._taken: set[str] = set(reserved)

    def claim(self, name: str) -> str:
        """Return a name not handed out before in this build, based on *name*."""
        candidate = name
        suffix = 2
        while candidate in self._taken:
            candidate = f"{name}{suffix}"
            suffix += 1
        self._taken.add(candidate)
        return candidate


def _build_model_name(key: str, layer_name: str) -> str:
    """Build per-key model class name: ``{KeyPart}{LayerPart}Config``.

    KeyPart: the key's words each title-cased, joined (``browser_use`` →
    ``BrowserUse``, ``openai.azure`` → ``OpenaiAzure``, ``gpt-4o`` →
    ``Gpt4O``); every character that cannot appear in an identifier
    separates words, like ``_`` and ``.``.
    LayerPart: ≤3 chars → ALL_CAPS (``llm`` → ``LLM``); >3 chars → Title
    (``agent`` → ``Agent``); characters that cannot appear in an identifier
    become ``_``.

    The result is a valid identifier: it starts with ``_`` when it would
    start with a digit or the key has no word at all (``4o`` →
    ``_4OLLMConfig``).
    """
    key_part = "".join(word.title() for word in _words(key)) or "_"
    layer = _identifier_text(layer_name)
    layer_part = layer.upper() if len(layer) <= 3 else layer.title()
    return _as_class_name(f"{key_part}{layer_part}Config")


def _segment_model_name(segment: str, field_name: str) -> str:
    """Nested segment model class name: ``{Segment}{Field}Config`` (``AzureProviderConfig``).

    Both parts are title-cased; characters that cannot appear in an
    identifier become ``_`` (``gpt-4o`` → ``Gpt_4OProviderConfig``).
    """
    seg_part = _identifier_text(segment).title() or "_"
    return _as_class_name(f"{seg_part}{_identifier_text(field_name).title()}Config")


def _words(text: str) -> list[str]:
    """The identifier words of *text* (split at ``_`` and at every non-word character)."""
    return [word for word in re.split(r"[\W_]+", unicodedata.normalize("NFKC", text)) if word]


def _identifier_text(text: str) -> str:
    """*text* with every run of non-word characters replaced by ``_``."""
    return re.sub(r"\W+", "_", unicodedata.normalize("NFKC", text))


def _as_class_name(name: str) -> str:
    """*name* as a valid identifier (normalized as the parser would; ``_``-prefixed if needed)."""
    name = unicodedata.normalize("NFKC", name)
    if not name.isidentifier():
        name = "".join(ch for ch in name if ("_" + ch).isidentifier())
        if not name[:1].isidentifier():
            name = "_" + name
    return name


# ---------------------------------------------------------------------------
# Flat mode helpers
# ---------------------------------------------------------------------------


def _create_discriminator_only_model(
    name: str,
    disc_field: str,
    key: str,
) -> type[BaseModel]:
    """Create a model with only the discriminator field (extra='forbid')."""
    return _derive_model(
        name,
        BaseModel,
        {disc_field: (Literal[key], key)},  # type: ignore[valid-type]
        extra="forbid",
    )


def _inject_discriminator(
    schema: type[BaseModel],
    model_name: str,
    disc_field: str,
    key: str,
    *,
    owner: type | None = None,
    layer_name: str | None = None,
) -> type[BaseModel]:
    """Derive the per-key model: *schema* plus a ``Literal[key]`` discriminator.

    The per-key model subclasses *schema*, so its fields (``FieldInfo``
    objects untouched), validators and ``model_config`` all survive; the
    discriminator field is added as ``Literal[key]`` with default ``key``
    -- declared with ``alias=<disc_field>`` when *schema* has an
    ``alias_generator``, which would otherwise rename it.  The extra policy
    is set explicitly: *schema*'s own ``extra`` when it sets one,
    ``"forbid"`` otherwise (a Tier-3 schema without ``extra`` must not drift
    to pydantic's ``"ignore"``).

    When *schema* is *owner*'s Tier-3 ``__config_schema__`` (used verbatim),
    the per-key model records the wired entries that are schema fields in
    ``__wired_fields__`` / ``__wired_keys__`` (as the extractor does for the
    models it builds), so composed builds inline them.

    Args:
        schema: The extracted config schema (or Tier-3 ``__config_schema__``).
        model_name: Class name of the per-key model.
        disc_field: Discriminator field name.
        key: Registry key.
        owner: The registered class (error messages); defaults to *schema*.
        layer_name: The layer (error messages).

    Raises:
        ConfigDeclarationError: *schema* cannot be derived from (see
            :func:`_check_derivable`), has a field named or aliased like the
            discriminator, or stays incomplete.
    """
    target = owner if owner is not None else schema
    _check_derivable(schema, target)
    _check_discriminator_clash(schema, disc_field, key, owner=target, layer_name=layer_name)
    extra = schema.model_config.get("extra") or "forbid"
    disc: tuple[Any, Any] = (Literal[key], key)  # type: ignore[valid-type]
    if schema.model_config.get("alias_generator") is not None:
        disc = (Literal[key], Field(default=key, alias=disc_field))  # type: ignore[valid-type]
    return _derive_model(
        model_name,
        schema,
        {disc_field: disc},
        extra=extra,
        carry_from=schema,
        attrs=_tier3_wiring_attrs(owner, schema),
        owner=target,
    )


def _tier3_wiring_attrs(owner: type | None, schema: type[BaseModel]) -> dict[str, Any]:
    """``__wired_fields__`` / ``__wired_keys__`` of a Tier-3 owner's wired schema fields."""
    if owner is None or getattr(owner, "__config_schema__", None) is not schema:
        return {}
    wired = {
        name: rw for name, rw in resolve_wiring(owner).items() if name in schema.model_fields
    }
    if not wired:
        return {}
    keys: dict[str, tuple[str, ...]] = {}
    for name, rw in wired.items():
        allowed = tuple(dict.fromkeys(rw.allowed_keys))
        # A schema field typed with ``Literal[...]`` narrows the wiring: the
        # inline union must accept exactly what the per-layer model accepts.
        accepted = _literal_strings(schema.model_fields[name].annotation)
        if accepted is not None:
            allowed = tuple(k for k in allowed if k in accepted)
            if not allowed:
                raise ConfigDeclarationError(
                    owner.__qualname__,
                    f"the __config_schema__ field accepts {sorted(accepted)} but the "
                    f"wiring allows {list(dict.fromkeys(rw.allowed_keys))}; no key fits "
                    f"both. Align the schema's Literal with the __wiring__ keys.",
                    field=name,
                )
        keys[name] = allowed
    return {
        "__wired_fields__": {name: rw.registry_name or "literal" for name, rw in wired.items()},
        "__wired_keys__": keys,
    }


def _literal_strings(annotation: Any) -> frozenset[str] | None:
    """String values of the ``Literal[...]`` an annotation selects keys with.

    Looks through ``Annotated``, ``Optional`` / unions and containers
    (``list[Literal[...]]``).  ``None`` when the annotation holds no
    ``Literal`` of strings (e.g. a plain ``str`` field accepts any key).
    """
    origin = get_origin(annotation)
    if origin is Literal:
        values = [a for a in get_args(annotation) if isinstance(a, str)]
        return frozenset(values) if values else None
    found: set[str] = set()
    any_literal = False
    for arg in get_args(annotation) if origin is not None else ():
        sub = _literal_strings(arg)
        if sub is not None:
            any_literal = True
            found |= sub
    return frozenset(found) if any_literal else None


# ---------------------------------------------------------------------------
# Warnings at the user's class
# ---------------------------------------------------------------------------

#: Directory of the conscribe package: warnings located in it are moved to
#: the class being built.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def _warnings_at(cls: type) -> Iterator[None]:
    """Report conscribe's warnings raised while *cls* is processed at *cls*.

    Every :class:`~conscribe.exceptions.ConscribeWarning` raised in the
    block is held back and re-emitted when the block completes -- at *cls*'s
    definition (:func:`_warn_at`) when it points at a line of conscribe
    itself, unchanged otherwise -- so the user's filters apply to the final
    location (``"error"`` raises at that point).  Other warnings go through
    the user's filters as usual; those shown are shown unchanged.  When the
    block raises, the warnings are still reported, but the exception wins
    over a warning an ``"error"`` filter would raise.
    """
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConscribeWarning)
            yield
    except BaseException:
        with contextlib.suppress(Warning):
            _reemit(cls, caught)
        raise
    _reemit(cls, caught)


def _reemit(cls: type, caught: list[warnings.WarningMessage]) -> None:
    """Re-emit the warnings :func:`_warnings_at` held back (see there)."""
    for item in caught:
        if not issubclass(item.category, ConscribeWarning):
            # Already past the user's filters: show it as warn() would have.
            warnings.showwarning(
                item.message, item.category, item.filename, item.lineno, item.file, item.line,
            )
        elif _in_package(item.filename):
            _warn_at(cls, item.message, item.category)
        else:
            module = _module_of_file(item.filename)
            warnings.warn_explicit(
                item.message,
                item.category,
                item.filename,
                item.lineno,
                module=getattr(module, "__name__", None),
                registry=_warning_registry(module),
                source=item.source,
            )


def _in_package(filename: str) -> bool:
    try:
        path = os.path.abspath(filename)
    except (TypeError, ValueError):
        return False
    return path == _PACKAGE_DIR or path.startswith(_PACKAGE_DIR + os.sep)


def _module_of_file(filename: str) -> Any:
    """The imported module whose ``__file__`` is *filename*, or ``None``."""
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == filename:
            return module
    return None


def _warning_registry(module: Any) -> dict[str, Any] | None:
    """*module*'s ``__warningregistry__`` (created when missing), or ``None``."""
    namespace = getattr(module, "__dict__", None)
    if not isinstance(namespace, dict):
        return None
    registry = namespace.setdefault("__warningregistry__", {})
    return registry if isinstance(registry, dict) else None


def _warn_at(cls: Any, message: str | Warning, category: type[Warning]) -> None:
    """Emit a warning located at *cls*'s definition (``warnings.warn_explicit``).

    The location is the ``class`` statement when the source is available,
    else the class's own ``__init__``, else its module.  Filters see the
    class's module, and "once per location" de-duplication uses that
    module's ``__warningregistry__``.  (``module_globals`` is not passed:
    for a module created at runtime without ``__spec__.loader`` it makes
    ``warn_explicit`` raise.)
    """
    filename, lineno = _definition_site(cls)
    module_name = getattr(cls, "__module__", None)
    module = sys.modules.get(module_name) if isinstance(module_name, str) else None
    warnings.warn_explicit(
        message,
        category,
        filename,
        lineno,
        module=module_name if isinstance(module_name, str) else None,
        registry=_warning_registry(module),
    )


def _definition_site(cls: Any) -> tuple[str, int]:
    """``(filename, line)`` where *cls* is defined, as precisely as available."""
    try:
        filename = inspect.getsourcefile(cls)
        if filename:
            _, lineno = inspect.getsourcelines(cls)
            return filename, max(lineno, 1)
    except (OSError, TypeError):
        pass
    init = vars(cls).get("__init__") if isinstance(cls, type) else None
    code = getattr(inspect.unwrap(init), "__code__", None) if callable(init) else None
    if code is not None:
        return code.co_filename, code.co_firstlineno
    module = sys.modules.get(getattr(cls, "__module__", None) or "")
    filename = getattr(module, "__file__", None)
    if isinstance(filename, str):
        return filename, 1
    return f"<{getattr(cls, '__module__', None) or 'unknown'}>", 1
