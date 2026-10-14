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

Build order: argument checks (``registrars`` must map layer names to
registrars -- a ``TypeError`` otherwise -- and the layer names must be usable
as fields, see below), empty layers
(:class:`~conscribe.exceptions.EmptyLayerError`), each layer's own
declaration checks (discriminator configuration, nested-mode keys and
declarations: :func:`conscribe.config.builder.build_layer_config`), then the
wiring of every registered class of every layer is resolved at once
(:func:`conscribe.wiring.validate_wiring`) -- all wiring problems of the
build are reported in one :class:`~conscribe.exceptions.WiringResolutionError`
before anything is built -- then the layers are built leaves first.  Model
class names are unique across the whole build (a numeric suffix tells two
otherwise equal names apart, deterministically).

Layers are matched to wiring targets by their registrar's registry name
(``create_registrar(name, ...)``); the key a layer is passed under in
``registrars`` names its top-level section.  That key becomes a field of the
top-level model (and of the generated ``_composed.py``), so it must be a
valid identifier, not a keyword, not private (``_x``), and must not shadow a
``BaseModel`` attribute (``model_config``, ``schema``, ``json``, ``copy``,
``dict``, ``validate``, ...) or use pydantic's protected ``model_`` prefix
-- a :class:`~conscribe.exceptions.ConfigDeclarationError` names every bad
name with a replacement (the registry name, used by wiring, is unaffected).

Wiring dependencies between layers must be acyclic; a
:class:`~conscribe.exceptions.CircularWiringError` names one actual cycle
(``a -> b -> a``: ``a`` wires ``b``, ``b`` wires ``a``).  ``declare_only``
entries hold no configs and add no dependency.

Top level
---------

The top-level ``ComposedConfig`` model has one ``list[<layer union>]`` field
per layer and is strict by default (``top_level_extra="forbid"``): a
misspelled section such as ``hooks:`` is refused, and the JSON schema says
``additionalProperties: false``.  ``"ignore"`` / ``"allow"`` relax it.

**Inline-only layers** (``create_registrar(..., inline_only=True)``) have no
top-level section: their instances may only be written inside receptors of
other layers.  Their models are still built (receptors use them), but the
top-level model holds a hidden *phantom* field for each of them::

    archiver: SkipJsonSchema[Any] = Field(default=None, exclude=True)

whose ``mode="after"`` field validator refuses any value written there
(``[]`` and ``null`` included, in every ``top_level_extra`` mode) with a
pydantic error of type ``inline_only_layer`` at ``loc == (layer,)``,
reported together with every other error of the input::

    layer 'archiver' is inline-only; write its instances under: hook[trace_writer].archive

The locations are every receptor with a runtime slot (an ``__init__``
parameter or a class-level annotation; ``declare_only`` entries never
count) wired to the layer that holds inline configs in this build, rendered
``owner_layer[owner_key].receptor`` -- through inline-only owners the full
path, e.g. ``env[browser].hooks[trace_writer].archive`` -- and joined with
``" or "``.  A path through an inline-only owner only counts when every
receptor on it accepts the next owner's key (``env[browser].hooks`` must
allow ``trace_writer`` blocks); receptors no such path reaches are not
writable.  An inline-only layer left without a writable location is a
:class:`~conscribe.exceptions.ConfigDeclarationError`.  The writable
receptors are listed per layer in
:attr:`ComposedConfigResult.inline_only_layers`.

Every top-level schema extension lives in the model's
``ConfigDict(json_schema_extra=...)``, so ``top_level_type.model_json_schema()``
equals the published schema: ``x-composed-layers``, ``x-inline-wiring``,
``x-inline-only-layers`` (``{layer: [full receptor paths]}``) and, under
``ignore`` / ``allow``, a ``not: {anyOf: [{required: [layer]}, ...]}`` clause
that keeps refusing the inline-only names.

Inline wiring
-------------

Inline wiring rebuilds every per-key model that owns an inlined receptor
by *subclassing* it (see :func:`conscribe.config.builder._derive_model`):
untouched fields, validators and ``model_config`` are inherited, and each
retyped receptor gets a fresh ``Field`` built by
:func:`conscribe.config._fieldinfo.inline_field` (description, title,
aliases, deprecated marker, default / ``default_factory``; list-receptor
``min_length`` / ``max_length``; plain markers).  Validators inherited this
way see the inline value -- a nested config model, or a list of them --
where the per-layer model passed them a key string.  Wired fields are
read from the per-layer model's records (``__wired_fields__`` and
``__wired_keys__``); ``declare_only`` entries are never inlined -- they stay
the ``Optional[Literal[...]] = None`` key constraint of the per-layer model,
since nothing receives an instance for them.  Slot-less entries without
``declare_only`` are inlined (the framework injects those attributes).

Key-string constraints on a scalar receptor (``pattern`` / ``min_length`` /
``max_length``, also inside ``Optional[Annotated[str, ...]]`` or a union
member) narrow its allowed keys.  Constraints that reject every key are a
:class:`~conscribe.exceptions.ConfigDeclarationError` in both modes (no
config could ever select a key there).

Each rebuilt model records what was inlined in ``__inline_wired__``::

    {field_name: {"target_layer": "archiver",
                  "keys": ("net", "disk"),   # members of the in-process union, in order
                  "container": True,         # list receptor (list[...] of configs)
                  "optional": True}}         # receptor admits None

**Default instances.**  A single-instance receptor with a default key -- the
``"default"`` of its dict-form ``__wiring__`` entry, or a wired ``__init__``
parameter whose own default is a key string (``loop: str = "react"``) --
defaults to that key's config built from defaults::

    Field(default_factory=<final per-key model of the key>,
          json_schema_extra={"default": <model_dump(mode="json", by_alias=True) of it>})

The factory is the *final* model of the target layer (configs map back to
their class by ``type(cfg)``), so an omitted block validates to a config
indistinguishable from an explicit ``{<discriminator>: <key>}`` block.  The
target config must be buildable from defaults.  The receptor stays
``Optional`` only when its annotation admits ``None`` (omitted -> default
instance, ``null`` -> ``None``); otherwise ``null`` is refused.  The JSON
default lists the members of every set in a canonical order (sorted by
their JSON form), so it does not depend on hash order (``PYTHONHASHSEED``).
When the default instance cannot be serialized to JSON the schema shows no
default.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import keyword
import re
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticCustomError, PydanticSerializationError, PydanticUndefined

from conscribe.config._annotations import (
    UNION_ORIGINS,
    admits_none,
    annotation_shape,
    unwrap_annotated,
)
from conscribe.config._fieldinfo import inline_field, narrow_keys
from conscribe.config.builder import (
    LayerConfigResult,
    _build_validated_layer,
    _check_buildable,
    _derive_model,
    _layer_union,
    _ModelNames,
    _qualified_name,
    _union_member_keys,
)
from conscribe.exceptions import (
    CircularWiringError,
    ConfigDeclarationError,
    EmptyLayerError,
)
from conscribe.introspect import WiringSlot, wiring_slots
from conscribe.wiring import parse_wiring, validate_wiring

__all__ = [
    "INLINE_ONLY_ERROR_TYPE",
    "INLINE_ONLY_MESSAGE",
    "INLINE_WIRED_ATTR",
    "TOP_LEVEL_EXTRA_MODES",
    "ComposedConfigResult",
    "ReceptorLocation",
    "TopLevelExtra",
    "build_composed_config",
]

#: Class attribute recording the inlined receptors of a rebuilt per-key model.
INLINE_WIRED_ATTR = "__inline_wired__"

#: Pydantic error type of a value written under an inline-only layer's name.
INLINE_ONLY_ERROR_TYPE = "inline_only_layer"

#: Message template of that error (``PydanticCustomError`` context:
#: ``layer`` and ``where``, the receptor paths joined with ``" or "``).
INLINE_ONLY_MESSAGE = "layer '{layer}' is inline-only; write its instances under: {where}"

#: The ``top_level_extra`` modes of :func:`build_composed_config`: the extra
#: policy of the top-level model (``"forbid"`` refuses unknown top-level
#: keys, ``"ignore"`` drops them, ``"allow"`` keeps them in ``model_extra``).
TopLevelExtra = Literal["forbid", "ignore", "allow"]

#: The accepted ``top_level_extra`` values, in documentation order.
TOP_LEVEL_EXTRA_MODES: tuple[str, ...] = get_args(TopLevelExtra)

# Name of the field validator that refuses inline-only layer names.
_REFUSER_NAME = "_refuse_inline_only_layers"

# Class name of the top-level model (reserved in the build's model names).
_TOP_LEVEL_NAME = "ComposedConfig"

# Public ``BaseModel`` attributes a layer name (a top-level field) must not
# shadow: the running pydantic's, plus the names every supported pydantic
# version (2.7+) defines, so the rule does not depend on the version.
_BASEMODEL_ATTRS: frozenset[str] = frozenset(
    {name for name in dir(BaseModel) if not name.startswith("_")}
    | {
        "construct", "copy", "dict", "from_orm", "json", "parse_file", "parse_obj",
        "parse_raw", "schema", "schema_json", "update_forward_refs", "validate",
    }
)

# pydantic's protected namespace for model attributes (2.7-2.9 reserve the
# whole prefix; later versions keep ``model_validate*`` / ``model_dump*``).
_PROTECTED_PREFIX = "model_"


@dataclasses.dataclass(frozen=True)
class ReceptorLocation:
    """A receptor that holds instances of an inline-only layer inline.

    One receptor of one registered class, in one composed build.  ``str()``
    renders it the way config paths are written:
    ``hook[trace_writer].archive`` is the ``archive`` receptor inside a
    ``trace_writer`` block of the ``hook`` layer.

    Attributes:
        owner_layer: The layer (its name in the build) whose class owns the
            receptor.
        owner_key: The owning class's registry key in that layer.
        receptor: The receptor's name (the wired field).
        target_layer: The inline-only layer whose configs it holds.
        keys: The keys of *target_layer* the receptor accepts (the members
            of its inline union, in order) -- all keys of the layer, or the
            subset its wiring and key constraints allow.
    """

    owner_layer: str
    owner_key: str
    receptor: str
    target_layer: str
    keys: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.owner_layer}[{self.owner_key}].{self.receptor}"


@dataclasses.dataclass(frozen=True)
class ComposedConfigResult:
    """Result of building a composed config from multiple layers.

    Attributes:
        layer_results: Dict mapping layer name to its ``LayerConfigResult``
            (inline-only layers included).
        top_level_type: Dynamically created Pydantic model with a
            ``list[UnionType]`` field per top-level layer and a hidden
            phantom field per inline-only layer (see the module docstring);
            use :attr:`top_level_layers` to enumerate the sections.
        dependency_order: Every layer in topological order (leaves first).
        inline_wiring: Whether inline wiring replacement was applied.
        inline_only_layers: Each inline-only layer (dependency order) with
            the writable receptors that hold its instances inline (one hop
            each; the error message and ``x-inline-only-layers`` render the
            full paths through inline-only owners).  A receptor owned by an
            inline-only layer is listed only when some path from the top
            level accepts its owner's key.  Empty when the build has no
            inline-only layer.
        top_level_extra: The top-level model's extra policy (``"forbid"``,
            ``"ignore"`` or ``"allow"``, see :data:`TopLevelExtra`).
    """

    layer_results: dict[str, LayerConfigResult]
    top_level_type: type[BaseModel]
    dependency_order: list[str]
    inline_wiring: bool
    inline_only_layers: dict[str, tuple[ReceptorLocation, ...]] = dataclasses.field(
        default_factory=dict,
    )
    top_level_extra: str = "forbid"

    @property
    def top_level_layers(self) -> list[str]:
        """Layers with a top-level section: ``dependency_order`` minus the inline-only ones."""
        return [layer for layer in self.dependency_order if layer not in self.inline_only_layers]


def build_composed_config(
    registrars: Mapping[str, Any],
    *,
    inline_wiring: bool = True,
    top_level_extra: TopLevelExtra = "forbid",
) -> ComposedConfigResult:
    """Build a composed config from multiple layer registrars.

    Args:
        registrars: Mapping of layer name to registrar (the class
            ``create_registrar()`` returns).  The layer name is the
            top-level section -- a field of the top-level model, so a valid
            identifier that is not private and does not shadow a
            ``BaseModel`` attribute (see the module docstring); wiring
            entries reach a layer through its registrar's registry name.
        inline_wiring: If ``True``, wired fields become the target layer's
            discriminated union type (with default instances for receptors
            that declare a default key); ``declare_only`` entries stay key
            selectors.  If ``False``, wired fields stay ``Literal[...]`` key
            selectors.
        top_level_extra: Extra policy of the top-level model: ``"forbid"``
            (default: unknown top-level keys are refused), ``"ignore"``
            (dropped) or ``"allow"`` (kept in ``model_extra``).  Inline-only
            layer names are refused in every mode.

    Returns:
        A ``ComposedConfigResult`` with per-layer results and a top-level model.

    Raises:
        TypeError: If *registrars* is not a mapping, or maps a layer name to
            something that is not a registrar created by
            ``create_registrar()`` (every such layer is named).
        ValueError: If *top_level_extra* is not one of the accepted values,
            or a registrar has no discriminator configuration.
        EmptyLayerError: If any requested layer has no registered classes
            (every empty layer is named).
        WiringResolutionError: If any registered class of any layer has
            wiring problems (every problem is listed in ``problems``).
        CircularWiringError: If wiring dependencies form a cycle (one
            actual cycle is named).
        ConfigDeclarationError: If a layer name cannot be a top-level field
            (every such name is named, with a replacement); a layer cannot
            be built as declared (see
            :func:`~conscribe.config.builder.build_layer_config`); a wired
            field's key constraints reject every key it allows (both
            modes); an inline receptor's keys select no config model; a
            default key's config cannot be built from defaults (or a
            receptor's default cannot be a config default); an inline-only
            layer has no writable location in this build (no receptor holds
            its configs inline, or no path from the top level reaches one);
            ``inline_wiring=False`` is combined with an inline-only layer;
            or several layers share the registry a wiring entry targets.
    """
    _check_top_level_extra(top_level_extra)
    _check_registrars(registrars)
    _check_layer_names(registrars)
    _reject_empty_layers(registrars)
    # Layer-level declaration problems (no discriminator, nested-mode key
    # shapes, __wiring__ on a nested-mode class) come before wiring problems,
    # which they would otherwise hide.
    for registrar in registrars.values():
        _check_buildable(registrar)
    validate_wiring(registrars)
    if not inline_wiring:
        _reject_inline_only_without_inline_wiring(registrars)

    graph = _build_dependency_graph(registrars)
    order = _topological_sort(graph, set(registrars.keys()))
    layer_of = _layer_of_registry(registrars)
    slots_of = functools.cache(wiring_slots)
    # One name allocator for the whole build: model names are unique across
    # layers (a nested segment model and another layer's per-key model may
    # otherwise both be ``AzureProviderConfig``) and never the top level's.
    names = _ModelNames(reserved=(_TOP_LEVEL_NAME,))

    # Build each layer in dependency order (leaves first).  Wiring targets
    # are looked up by registry name.
    built: dict[str, LayerConfigResult] = {}
    targets: dict[str, LayerConfigResult] = {}
    for layer_name in order:
        registrar = registrars[layer_name]
        classes = registrar.get_all()
        result = _build_validated_layer(registrar, names=names)
        _check_key_constraints(result, classes)
        if inline_wiring:
            result = _replace_wired_literals_with_unions(
                result, targets,
                classes=classes,
                layer_of=layer_of,
                slots_of=slots_of,
            )
        built[layer_name] = result
        targets[_registry_name(registrar, layer_name)] = result

    inline_only = [layer for layer in order if _is_inline_only(registrars[layer])]
    locations, paths = _writable_inline_only_locations(
        registrars, built, order, inline_only, layer_of=layer_of, slots_of=slots_of,
    )
    top_level = _build_top_level_model(
        built,
        order,
        extra=top_level_extra,
        inline_only_paths=paths,
        inline_wiring=inline_wiring,
    )

    return ComposedConfigResult(
        layer_results=built,
        top_level_type=top_level,
        dependency_order=order,
        inline_wiring=inline_wiring,
        inline_only_layers=locations,
        top_level_extra=top_level_extra,
    )


def _check_top_level_extra(top_level_extra: Any) -> None:
    """Raise ``ValueError`` unless *top_level_extra* is an accepted mode."""
    if not isinstance(top_level_extra, str) or top_level_extra not in TOP_LEVEL_EXTRA_MODES:
        raise ValueError(
            f"build_composed_config(): top_level_extra must be one of "
            f"{list(TOP_LEVEL_EXTRA_MODES)}, got {top_level_extra!r}"
        )


# ---------------------------------------------------------------------------
# Argument checks: registrars and layer names
# ---------------------------------------------------------------------------


def _check_registrars(registrars: Any) -> None:
    """Raise ``TypeError`` unless *registrars* maps names to registrars.

    A registrar is a class returned by ``create_registrar()`` (a
    :class:`~conscribe.registration.registrar.LayerRegistrar` subclass with a
    registry).  Every layer holding something else is named.
    """
    if not isinstance(registrars, Mapping):
        raise TypeError(
            "build_composed_config() expects a dict mapping layer names to registrars "
            f"created by create_registrar(), got {_describe(registrars)}"
        )
    bad = [
        f"layer {name!r} is {_describe(registrar)}"
        for name, registrar in registrars.items()
        if not _is_registrar(registrar)
    ]
    if bad:
        raise TypeError(
            "build_composed_config() expects registrars created by create_registrar() "
            f"(the class it returns, not a registered class or an instance): "
            f"{'; '.join(bad)}"
        )


def _is_registrar(obj: Any) -> bool:
    """Whether *obj* is a registrar class returned by ``create_registrar()``."""
    from conscribe.registration.registrar import LayerRegistrar

    return (
        isinstance(obj, type)
        and issubclass(obj, LayerRegistrar)
        and obj is not LayerRegistrar
        and getattr(obj, "_registry", None) is not None
    )


def _describe(obj: Any) -> str:
    """``class pkg.Name`` / ``str 'x'`` -- what was passed, for an error message."""
    if isinstance(obj, type):
        return f"class {obj.__module__}.{obj.__qualname__}"
    text = repr(obj)
    if len(text) > 80:
        text = text[:77] + "..."
    return f"{type(obj).__name__} {text}"


def _check_layer_names(registrars: Mapping[Any, Any]) -> None:
    """Refuse layer names that cannot be fields of the top-level model.

    Raises:
        ConfigDeclarationError: Every bad name, why, and a replacement.
    """
    problems = []
    for name, registrar in registrars.items():
        why = _layer_name_problem(name)
        if why is None:
            continue
        problem = f"{name!r} {why}"
        suggestion = _suggest_layer_name(name)
        if suggestion is not None and suggestion not in registrars:
            problem += f" (pass it as {suggestion!r}"
            registry = _registry_name(registrar, str(name))
            if registry == name:
                problem += f"; wiring entries still reach it as registry {registry!r}"
            problem += ")"
        problems.append(problem)
    if problems:
        raise ConfigDeclarationError(
            "build_composed_config",
            f"invalid layer name(s): {'; '.join(problems)}. A layer name is the key its "
            f"registrar is passed under; it becomes a field of the top-level "
            f"ComposedConfig model and of the generated _composed.py, so it must be a "
            f"valid identifier that is not a keyword, not private (leading '_'), does "
            f"not shadow a BaseModel attribute and does not start with pydantic's "
            f"protected prefix {_PROTECTED_PREFIX!r}. Pass the registrar under another "
            f"name: only the section name changes, wiring entries find layers by their "
            f"registry name.",
        )


def _layer_name_problem(name: Any) -> str | None:
    """Why *name* cannot be a top-level field, or ``None`` when it can."""
    if not isinstance(name, str):
        return f"is of type {type(name).__name__}, not str"
    if not name.isidentifier():
        return "is not a valid identifier"
    if keyword.iskeyword(name):
        return "is a Python keyword"
    if name.startswith("_"):
        return "is private (pydantic does not make it a field)"
    if name in _BASEMODEL_ATTRS:
        return f"shadows the BaseModel attribute {name!r}"
    if name.startswith(_PROTECTED_PREFIX):
        return f"starts with pydantic's protected prefix {_PROTECTED_PREFIX!r}"
    return None


def _suggest_layer_name(name: Any) -> str | None:
    """A close name that passes :func:`_layer_name_problem`, if one is easy to find."""
    candidate = re.sub(r"\W+", "_", str(name)).strip("_")
    if candidate.startswith(_PROTECTED_PREFIX):
        candidate = candidate[len(_PROTECTED_PREFIX):].strip("_")
    if not candidate:
        candidate = "layer"
    if candidate[0].isdigit():
        candidate = f"layer_{candidate}"
    if _layer_name_problem(candidate) is not None:
        candidate = f"{candidate}_layer"
    return candidate if _layer_name_problem(candidate) is None else None


def _reject_empty_layers(registrars: dict[str, Any]) -> None:
    """Raise ``EmptyLayerError`` naming every layer without registered classes."""
    empty = [name for name, registrar in registrars.items() if not registrar.get_all()]
    if empty:
        raise EmptyLayerError(empty)


def _is_inline_only(registrar: Any) -> bool:
    """Whether the registrar's layer was declared ``inline_only=True``."""
    return bool(getattr(registrar, "inline_only", False))


def _reject_inline_only_without_inline_wiring(registrars: dict[str, Any]) -> None:
    """Refuse inline-only layers in a key-selector (``inline_wiring=False``) build."""
    inline_only = [layer for layer, registrar in registrars.items() if _is_inline_only(registrar)]
    if not inline_only:
        return
    names = ", ".join(repr(layer) for layer in inline_only)
    one = len(inline_only) == 1
    raise ConfigDeclarationError(
        "build_composed_config(inline_wiring=False)",
        f"{'layer' if one else 'layers'} {names} {'is' if one else 'are'} inline-only "
        f"(create_registrar(..., inline_only=True)): {'its' if one else 'their'} "
        f"instances can only be written inside receptors of other layers, and "
        f"receptors hold inline configs only with inline_wiring=True, so without it "
        f"they could not be written anywhere. Build with inline_wiring=True, or leave "
        f"{names} out of the registrars.",
    )


# ---------------------------------------------------------------------------
# Layers by registry name
# ---------------------------------------------------------------------------


def _registry_name(registrar: Any, layer_name: str) -> str:
    """The registry name wiring entries use for *registrar* (its layer name as fallback)."""
    name = getattr(getattr(registrar, "_registry", None), "name", None)
    return name if isinstance(name, str) and name else layer_name


def _layers_by_registry(registrars: dict[str, Any]) -> dict[str, list[str]]:
    """``{registry name: [layer names]}`` (several names when a registrar is passed twice)."""
    layers: dict[str, list[str]] = {}
    for layer_name, registrar in registrars.items():
        layers.setdefault(_registry_name(registrar, layer_name), []).append(layer_name)
    return layers


def _layer_of_registry(registrars: dict[str, Any]) -> dict[str, str]:
    """``{registry name: layer name}`` for the registries passed under one name."""
    return {
        registry: names[0]
        for registry, names in _layers_by_registry(registrars).items()
        if len(names) == 1
    }


# ---------------------------------------------------------------------------
# Dependency graph & topological sort
# ---------------------------------------------------------------------------


def _build_dependency_graph(
    registrars: dict[str, Any],
) -> dict[str, set[str]]:
    """Build a directed dependency graph from wiring declarations.

    For each registrar, iterates all registered classes and parses their
    ``__wiring__`` to find which other registries they reference.  Only
    edges where the target registry is also in ``registrars`` are included
    (self-references are skipped).  ``declare_only`` entries add no edge:
    they constrain keys and never hold the target layer's configs, so the
    target need not be built first.

    Returns:
        Dict mapping layer name to set of layer names it depends on.

    Raises:
        ConfigDeclarationError: A wiring entry targets a registry that the
            build holds under several layer names (its inline configs would
            be ambiguous).
    """
    layers_of = _layers_by_registry(registrars)
    graph: dict[str, set[str]] = {name: set() for name in registrars}

    for layer_name, registrar in registrars.items():
        for cls in registrar.get_all().values():
            for spec in parse_wiring(cls):
                if spec.declare_only:
                    continue  # a key constraint, never inline configs
                targets = layers_of.get(spec.registry_name) if spec.registry_name else None
                if not targets:
                    continue  # Mode 3, or a registry outside the build
                if len(targets) > 1:
                    raise ConfigDeclarationError(
                        _qualified_name(cls),
                        f"the wiring entry targets registry {spec.registry_name!r}, which "
                        f"this build holds under several layer names {targets}, so it is "
                        f"ambiguous which layer the field selects from. Pass that "
                        f"registrar once.",
                        spec.param_name,
                    )
                if targets[0] != layer_name:  # skip self-refs
                    graph[layer_name].add(targets[0])

    return graph


def _topological_sort(
    graph: dict[str, set[str]],
    all_nodes: set[str],
) -> list[str]:
    """Kahn's algorithm. Returns nodes in dependency order (leaves first).

    Raises:
        CircularWiringError: If a cycle is detected; it names one actual
            cycle of *graph* (see :func:`_find_cycle`).
    """
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
        raise CircularWiringError(_find_cycle(graph, all_nodes - set(result)))

    return result


def _find_cycle(graph: Mapping[str, set[str]], remaining: set[str]) -> list[str]:
    """One actual cycle among the nodes Kahn's algorithm could not order.

    Every such node depends on at least one other such node (otherwise it
    would have been ordered), so walking dependencies from the smallest node
    -- always to the smallest remaining dependency, for a deterministic
    answer -- revisits a node; the walk from that node's first visit is the
    cycle.

    Returns:
        The cycle's nodes in dependency direction (``["a", "b"]``: ``a``
        depends on ``b``, ``b`` on ``a``), without repeating the first.
    """
    path: list[str] = []
    seen: dict[str, int] = {}
    node = min(remaining)
    while node not in seen:
        seen[node] = len(path)
        path.append(node)
        node = min(dep for dep in graph.get(node, ()) if dep in remaining)
    return path[seen[node]:]


# ---------------------------------------------------------------------------
# Inline wiring replacement
# ---------------------------------------------------------------------------


def _replace_wired_literals_with_unions(
    result: LayerConfigResult,
    built_results: dict[str, LayerConfigResult],
    *,
    classes: dict[str, type] | None = None,
    layer_of: Mapping[str, str] | None = None,
    slots_of: Callable[[type], dict[str, WiringSlot]] | None = None,
) -> LayerConfigResult:
    """Replace Literal wired fields with target layer union types.

    For each per-key model that has ``__wired_fields__``, checks if the
    wired registry has already been built.  If so, replaces the
    ``Literal[...]`` field with the target layer's union type (or a subset
    union for Mode 2 wiring, or after key-string constraints narrowed the
    allowed keys), sets the default instance of a default key (see the
    module docstring), rebuilds the model by subclassing it and records the
    result in ``__inline_wired__``.  ``declare_only`` entries stay key
    selectors.

    Args:
        result: The owner layer, as built by ``build_layer_config``.
        built_results: Layers built so far (wiring targets), by registry name.
        classes: The owner layer's registered classes by key (error
            messages, ``__wiring__`` entries).
        layer_of: Registry name -> layer name of the wiring targets
            (``__inline_wired__["target_layer"]``); identity when omitted.
        slots_of: :func:`conscribe.introspect.wiring_slots` (or a cached
            version of it); ``declare_only`` entries and explicit
            ``__wiring__`` default keys are only known when it is given.

    Returns:
        A new ``LayerConfigResult`` with rebuilt models and union type, or
        *result* itself when nothing was inlined.

    Raises:
        ConfigDeclarationError: A wired field is missing from its config
            model, its key constraints reject every key, its keys select no
            config model, or its default cannot become a config default.
    """
    classes = classes or {}
    layer_of = layer_of or {}
    new_per_key: dict[str, type[BaseModel]] = {}
    changed = False

    for key, model in result.per_key_models.items():
        wired_fields: dict[str, str] = getattr(model, "__wired_fields__", None) or {}
        replacements: dict[str, Any] = {}  # field_name -> (type, FieldInfo)
        inline_record: dict[str, dict[str, Any]] = {}
        cls = classes.get(key)
        owner = _qualified_name(cls) if cls is not None else model.__name__
        entries = slots_of(cls) if (cls is not None and slots_of is not None) else {}

        for field_name, registry_name in wired_fields.items():
            if registry_name == "literal" or registry_name not in built_results:
                continue  # Mode 3, or a target outside the composed set: stays a key selector
            entry = entries.get(field_name)
            if entry is not None and entry.declare_only:
                continue  # a declaration-only key constraint: nothing receives a config

            target = built_results[registry_name]
            target_layer = layer_of.get(registry_name, registry_name)
            src = model.model_fields.get(field_name)
            if src is None:
                raise ConfigDeclarationError(
                    owner,
                    f"__wired_fields__ lists the field (wired to {registry_name!r}), but "
                    f"the config model {model.__name__} has no such field, so it cannot "
                    f"hold inline {target_layer!r} configs.",
                    field_name,
                )

            container = annotation_shape(src.annotation) != "scalar"
            optional = admits_none(src.annotation)
            allowed_keys = _receptor_keys(
                model, field_name, src, container=container, owner=owner,
            )
            inline_type, member_keys = _inline_union(
                target, allowed_keys, owner=owner, field=field_name,
            )
            if container:
                inline_type = list[inline_type]  # type: ignore[valid-type]
            if optional:
                inline_type = Optional[inline_type]

            default = _inline_default(
                src,
                entry,
                target=target,
                target_layer=target_layer,
                member_keys=member_keys,
                container=container,
                owner=owner,
                field=field_name,
            )
            replacements[field_name] = inline_field(
                src, inline_type, container=container, **default,
            )
            inline_record[field_name] = {
                "target_layer": target_layer,
                "keys": member_keys,
                "container": container,
                "optional": optional,
            }

        if not replacements:
            new_per_key[key] = model
            continue

        changed = True
        new_per_key[key] = _rebuild_model_with_replacements(
            model, replacements, inline_wired=inline_record,
        )

    if not changed:
        return result

    return dataclasses.replace(
        result,
        union_type=_layer_union(result, new_per_key),
        per_key_models=new_per_key,
    )


def _inline_union(
    target_result: LayerConfigResult,
    allowed_keys: tuple[str, ...],
    *,
    owner: str,
    field: str,
) -> tuple[Any, tuple[str, ...]]:
    """The inline union for a wired field and the keys of its members.

    If *allowed_keys* covers every config model of the target layer, the
    layer's own union is used.  Otherwise a subset union of the matching
    per-key models is built the way the target layer builds unions (flat
    ``Field(discriminator=...)``, or the nested compound ``Discriminator`` +
    ``Tag``).

    Returns:
        ``(inline_type, member_keys)`` -- *member_keys* are the keys of the
        models actually in the union, in member order.

    Raises:
        ConfigDeclarationError: None of *allowed_keys* has a config model in
            the target layer (e.g. only non-leaf keys of a nested layer).
    """
    target_keys = _union_member_keys(target_result)
    if set(allowed_keys) >= set(target_keys):
        return target_result.union_type, target_keys

    members = tuple(dict.fromkeys(k for k in allowed_keys if k in target_result.per_key_models))
    if not members:
        raise ConfigDeclarationError(
            owner,
            f"the wiring keys {list(allowed_keys)} select no config model of layer "
            f"{target_result.layer_name!r} (its config models: {sorted(target_keys)}), "
            f"so the field cannot hold an inline config. Wire keys that have config "
            f"models.",
            field,
        )
    subset = {k: target_result.per_key_models[k] for k in members}
    # A subset is built the way the target layer builds its own union: a
    # nested target needs the compound Discriminator + Tag form, because its
    # level-0 discriminator alone cannot tell ``openai.azure`` from
    # ``openai.official``.
    return _layer_union(target_result, subset), members


# ---------------------------------------------------------------------------
# Allowed keys and key-string constraints
# ---------------------------------------------------------------------------


def _receptor_keys(
    model: type[BaseModel],
    field_name: str,
    src: FieldInfo,
    *,
    container: bool,
    owner: str,
) -> tuple[str, ...]:
    """The keys a wired field accepts in the per-layer model.

    Every model with wired fields records their allowed keys in
    ``__wired_keys__``, always together with ``__wired_fields__`` (the
    extractor for the models it builds, the builder for a Tier-3 schema's
    wired fields).  On a scalar receptor, key-string constraints narrow them
    (they cannot apply to a nested config, so they are applied to the keys
    here); on a list receptor ``min_length`` / ``max_length`` count
    instances and keep their meaning on the inline list.

    Raises:
        ConfigDeclarationError: The key constraints reject every key.
    """
    keys = tuple(model.__wired_keys__[field_name])  # type: ignore[attr-defined]
    if container:
        return keys
    return _narrow_receptor_keys(keys, src, owner=owner, field=field_name)


def _check_key_constraints(result: LayerConfigResult, classes: Mapping[str, type]) -> None:
    """Refuse a layer whose wired scalar fields' constraints reject every key.

    Runs in both modes, on every wired field (inlined or not): a
    ``Literal[...]`` key selector whose ``pattern`` / ``min_length`` /
    ``max_length`` rejects all of its keys can never hold a key.

    Raises:
        ConfigDeclarationError: The first such field (see
            :func:`_narrow_receptor_keys`).
    """
    for key, model in result.per_key_models.items():
        wired_fields = getattr(model, "__wired_fields__", None) or {}
        if not wired_fields:
            continue
        cls = classes.get(key)
        owner = _qualified_name(cls) if cls is not None else model.__name__
        for field_name in wired_fields:
            src = model.model_fields.get(field_name)
            if src is None or annotation_shape(src.annotation) != "scalar":
                continue
            _receptor_keys(model, field_name, src, container=False, owner=owner)


def _narrow_receptor_keys(
    keys: tuple[str, ...],
    src: FieldInfo,
    *,
    owner: str,
    field: str,
) -> tuple[str, ...]:
    """Apply a scalar receptor's key-string constraints to its allowed keys.

    The constraints are the field's own (``FieldInfo.metadata``) plus those
    written inside its annotation: ``Optional[Annotated[str,
    StringConstraints(pattern=...)]]`` keeps them on the union member, out of
    ``FieldInfo.metadata``'s sight.  A key is allowed when the field's
    constraints and those of at least one non-``None`` union member accept
    it -- pydantic's union semantics, so the keys are exactly those the
    per-layer ``Literal[...]`` field accepts.

    Raises:
        ConfigDeclarationError: The constraints reject every key.
    """
    outer = list(src.metadata)
    alternatives = _member_metadata(src.annotation)
    if len(alternatives) == 1:
        return narrow_keys(keys, [*outer, *alternatives[0]], owner=owner, field=field)

    accepted: set[str] = set()
    for inner in alternatives:
        try:
            accepted.update(narrow_keys(keys, [*outer, *inner], owner=owner, field=field))
        except ConfigDeclarationError:
            continue  # this member accepts none of the keys; another one may
    kept = tuple(k for k in keys if k in accepted)
    if not kept:
        raise ConfigDeclarationError(
            owner,
            f"the key constraints of every member of its annotation {src.annotation!r} "
            f"reject every key the wiring allows ({list(keys)}), so no config could "
            f"select a {field!r}. Fix the constraints or the __wiring__ keys.",
            field,
        )
    return kept


def _member_metadata(annotation: Any) -> list[list[Any]]:
    """The metadata on the way to each non-``None`` member of *annotation*.

    ``Annotated[T, *meta]`` contributes *meta* (a ``FieldInfo`` in it its
    own ``metadata``); a union yields one list per non-``None`` member, each
    prefixed with the metadata around the union.  Any other annotation
    yields one list.
    """
    base, meta = unwrap_annotated(annotation)
    collected: list[Any] = []
    for item in meta:
        collected.extend(item.metadata if isinstance(item, FieldInfo) else [item])
    if get_origin(base) not in UNION_ORIGINS:
        return [collected]
    alternatives = [
        [*collected, *inner]
        for member in get_args(base)
        if member is not type(None)
        for inner in _member_metadata(member)
    ]
    return alternatives or [collected]


# ---------------------------------------------------------------------------
# Default instances (F4)
# ---------------------------------------------------------------------------


def _inline_default(
    src: FieldInfo,
    wiring_slot: WiringSlot | None,
    *,
    target: LayerConfigResult,
    target_layer: str,
    member_keys: tuple[str, ...],
    container: bool,
    owner: str,
    field: str,
) -> dict[str, Any]:
    """``inline_field`` default overrides for one inlined receptor.

    A single-instance receptor whose ``__wiring__`` entry declares a
    ``default`` key, or whose per-layer default is a key string (a wired
    ``__init__`` parameter such as ``loop: str = "react"``), gets the
    *final* per-key model of that key as ``default_factory`` and the dump
    of the default instance as the JSON-schema default.  Defaults that a
    field holding configs cannot honour are refused.  (A ``__wiring__``
    default on a list receptor or a slot-less entry never gets here:
    :func:`~conscribe.wiring.validate_wiring` reports it as ``bad_default``
    before anything is built.)

    Returns:
        ``{}`` to carry the source field's default, else
        ``{"default_factory": <model>, "json_schema_extra": {...}}``.

    Raises:
        ConfigDeclarationError: The default cannot become a config default
            (see :func:`_key_string_default`, :func:`_check_container_default`,
            :func:`_default_instance`), or the parameter default and the
            ``__wiring__`` default name different keys.
    """
    explicit = wiring_slot.default_key if wiring_slot is not None else None
    if container:
        _check_container_default(src, owner=owner, field=field, target_layer=target_layer)
        return {}

    implicit = _key_string_default(src, owner=owner, field=field, target_layer=target_layer)
    if explicit is not None and implicit is not None and implicit != explicit:
        raise ConfigDeclarationError(
            owner,
            f"two defaults name different keys: the parameter default {implicit!r} and "
            f"the __wiring__ default {explicit!r}. Declare the default once (in "
            f"__wiring__), or make both name the same key.",
            field,
        )
    key = explicit if explicit is not None else implicit
    if key is None:
        return {}

    source = "the __wiring__ default" if explicit is not None else "the parameter default"
    model = target.per_key_models.get(key)
    if model is None or key not in member_keys:
        raise ConfigDeclarationError(
            owner,
            f"{source} {key!r} selects no config this field accepts: it holds inline "
            f"{target_layer!r} configs for the keys {list(member_keys)} (key constraints "
            f"on the parameter narrow them; a nested-mode layer has config models for "
            f"its leaf keys only). Choose one of them as the default.",
            field,
        )
    instance = _default_instance(
        model, key=key, source=source, explicit=explicit is not None,
        target_layer=target_layer, owner=owner, field=field,
    )
    return {"default_factory": model, "json_schema_extra": _schema_default(instance)}


def _key_string_default(
    src: FieldInfo,
    *,
    owner: str,
    field: str,
    target_layer: str,
) -> str | None:
    """The key a scalar receptor's per-layer default names, or ``None``.

    ``None`` when the field is required or defaults to ``None``.

    Raises:
        ConfigDeclarationError: The default is neither a key string nor
            ``None`` (an object, a ``default_factory``): under inline wiring
            the field holds a config, which such a default cannot be.
    """
    if src.default_factory is not None:
        name = getattr(src.default_factory, "__qualname__", None) or repr(src.default_factory)
        raise ConfigDeclarationError(
            owner,
            f"the default_factory {name} cannot be honoured under inline wiring: the "
            f"field holds a {target_layer!r} config, which a user factory cannot build. "
            f"Declare the default as a key (the __wiring__ 'default', or a key-string "
            f"parameter default) or None.",
            field,
        )
    default = src.default
    if default is PydanticUndefined or default is None:
        return None
    if isinstance(default, str):
        return str.__str__(default)  # the key itself, also for str enums
    raise ConfigDeclarationError(
        owner,
        f"the default {default!r} is not a key: under inline wiring the field holds a "
        f"{target_layer!r} config, so its default must name the key whose config "
        f"(built from defaults) it stands for, or be None. Use a key string or None.",
        field,
    )


def _check_container_default(
    src: FieldInfo,
    *,
    owner: str,
    field: str,
    target_layer: str,
) -> None:
    """Refuse a non-empty plain default on a list receptor under inline wiring.

    The field holds a list of configs; a list of keys (or any other
    non-empty value) is not a valid default for it, and lists of default
    instances are not supported.  ``None``, empty collections and
    ``default_factory`` (``list`` and friends) are carried.

    Raises:
        ConfigDeclarationError: The default is a non-empty value.
    """
    if src.default_factory is not None:
        return
    default = src.default
    if default is PydanticUndefined or default is None:
        return
    if isinstance(default, (list, tuple, set, frozenset, dict)) and not default:
        return
    raise ConfigDeclarationError(
        owner,
        f"the default {default!r} cannot be the default of a list of inline "
        f"{target_layer!r} configs: the field holds configs, not keys, and a list of "
        f"default instances is not supported. Default to an empty collection or None "
        f"and write the blocks in config.",
        field,
    )


def _default_instance(
    model: type[BaseModel],
    *,
    key: str,
    source: str,
    explicit: bool,
    target_layer: str,
    owner: str,
    field: str,
) -> BaseModel:
    """Build *model* from defaults (the default instance of *key*).

    Raises:
        ConfigDeclarationError: The model cannot be built from defaults,
            naming its required fields.
    """
    try:
        return model()
    except ValidationError as exc:
        required = [name for name, info in model.model_fields.items() if info.is_required()]
        if required:
            why = f"its required field(s) {required} have no default"
            fix = "Give them defaults"
        else:
            reason = "; ".join(err["msg"] for err in exc.errors())
            why = f"building it from defaults fails: {reason}"
            fix = "Fix its defaults"
        if explicit:
            alternative = "choose a default key whose config can be built from defaults, or remove the default"
        else:
            alternative = (
                "choose a parameter default whose config can be built from defaults, "
                "remove the parameter default (the field becomes required), or build "
                "with inline_wiring=False"
            )
        raise ConfigDeclarationError(
            owner,
            f"{source} {key!r} makes the field default to the {model.__name__} config "
            f"(layer {target_layer!r}) built from defaults, but {why}. {fix}, "
            f"{alternative}.",
            field,
        ) from exc


def _schema_default(instance: BaseModel) -> dict[str, Any]:
    """``{"default": <JSON dump>}`` for the JSON schema, ``{}`` when not serializable.

    The members of every set in the dump are put in a canonical order (see
    :func:`_sorted_sets`), so the default -- and every artifact showing it
    (the in-process schema, the published schema, ``_composed.py``) -- does
    not depend on hash order (``PYTHONHASHSEED``).
    """
    try:
        dumped = instance.model_dump(mode="json", by_alias=True)
    except PydanticSerializationError:
        return {}
    return {"default": _sorted_sets(instance, dumped)}


def _sorted_sets(value: Any, dumped: Any) -> Any:
    """*dumped* -- pydantic's JSON form of *value* -- with every set's members sorted.

    Walks *value* and its JSON form side by side: models field by field
    (under their dump keys), mappings item by item, lists and tuples item by
    item, and set members in iteration order (pydantic serializes a set by
    iterating it, so member and JSON form line up).  Every JSON list that is
    the form of a ``set`` / ``frozenset`` is sorted by its members' type and
    JSON text (after the sets inside the members are sorted); lists and
    tuples keep their order.  Where the JSON form does not mirror the value
    (a custom serializer changed its shape) it is left as it is.
    """
    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        return _sorted_model_sets(value, dumped)
    if isinstance(value, (set, frozenset)) and isinstance(dumped, list):
        if len(value) == len(dumped):
            dumped = [_sorted_sets(member, form) for member, form in zip(value, dumped)]
        return sorted(dumped, key=_set_member_key)
    if isinstance(value, Mapping) and isinstance(dumped, dict) and len(value) == len(dumped):
        return {
            key: _sorted_sets(item, form)
            for item, (key, form) in zip(value.values(), dumped.items())
        }
    if isinstance(value, (list, tuple)) and isinstance(dumped, list) and len(value) == len(dumped):
        return [_sorted_sets(item, form) for item, form in zip(value, dumped)]
    return dumped


def _sorted_model_sets(model: BaseModel, dumped: dict[str, Any]) -> dict[str, Any]:
    """:func:`_sorted_sets` for a model's ``model_dump(mode="json", by_alias=True)``."""
    values: dict[str, Any] = {}
    for name, info in type(model).model_fields.items():
        values[info.serialization_alias or info.alias or name] = getattr(model, name, None)
    for name, computed in type(model).model_computed_fields.items():
        values[computed.alias or name] = getattr(model, name, None)
    values.update(model.model_extra or {})
    return {
        key: _sorted_sets(values[key], form) if key in values else form
        for key, form in dumped.items()
    }


def _set_member_key(form: Any) -> tuple[str, str]:
    """Sort key of a set member's JSON form: its type, then its canonical JSON text."""
    return (
        type(form).__name__,
        json.dumps(form, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )


def _rebuild_model_with_replacements(
    model: type[BaseModel],
    replacements: dict[str, Any],
    *,
    inline_wired: dict[str, dict[str, Any]] | None = None,
) -> type[BaseModel]:
    """Rebuild a per-key model with some fields redeclared.

    The new model *subclasses* *model* under the same name: every field not
    in *replacements* keeps its ``FieldInfo`` (inherited, untouched), and
    ``model_config``, validators, ``__wired_fields__``, ``__wired_keys__``
    and ``__degraded_fields__`` survive.  The extra policy is set explicitly.

    Args:
        model: The per-key model to rebuild.
        replacements: ``create_model`` definitions ``{field: (type, FieldInfo)}``
            for the redeclared fields; build them with
            :func:`~conscribe.config._fieldinfo.inline_field`, never from a
            ``model_fields`` FieldInfo.
        inline_wired: The ``__inline_wired__`` record to attach.

    Returns:
        The rebuilt model.
    """
    return _derive_model(
        model.__name__,
        model,
        replacements,
        extra=model.model_config.get("extra") or "forbid",
        module=__name__,
        carry_from=model,
        attrs={INLINE_WIRED_ATTR: inline_wired} if inline_wired else None,
    )


# ---------------------------------------------------------------------------
# Inline-only layers
# ---------------------------------------------------------------------------


def _writable_inline_only_locations(
    registrars: Mapping[str, Any],
    built: dict[str, LayerConfigResult],
    order: list[str],
    inline_only: list[str],
    *,
    layer_of: Mapping[str, str],
    slots_of: Callable[[type], dict[str, WiringSlot]],
) -> tuple[dict[str, tuple[ReceptorLocation, ...]], dict[str, tuple[str, ...]]]:
    """The writable receptors of each inline-only layer, and their full paths.

    A receptor is writable when it holds the layer's configs inline (see
    :func:`_inline_only_receptors`) and some path from the top level reaches
    it (see :func:`_inline_only_routes`).

    Returns:
        ``(locations, paths)`` per inline-only layer, in dependency order:
        the writable receptors (in the order of
        :func:`_inline_only_receptors`) and the full config path of every
        way to write one (``env[browser].hooks[trace_writer].archive``).

    Raises:
        ConfigDeclarationError: Inline-only layers without a writable
            location -- no receptor holds their configs inline (named with
            the entries that do not count), or no path from the top level
            reaches any of those receptors (each receptor named with the
            reason).  Every such layer is reported in one error.
    """
    if not inline_only:
        return {}, {}
    receptors, ignored = _inline_only_receptors(
        registrars, built, order, inline_only, layer_of=layer_of, slots_of=slots_of,
    )
    routes = _inline_only_routes(receptors)

    details = []
    for layer in inline_only:
        if not receptors[layer]:
            details.append(_no_receptor_detail(layer, registrars[layer], ignored[layer]))
        elif not routes[layer]:
            details.append(_unreachable_detail(layer, receptors[layer], routes))
    if details:
        raise ConfigDeclarationError("build_composed_config", "; ".join(details) + ".")

    locations = {
        layer: tuple(dict.fromkeys(location for _, location in routes[layer]))
        for layer in inline_only
    }
    paths = {
        layer: tuple(dict.fromkeys(path for path, _ in routes[layer]))
        for layer in inline_only
    }
    return locations, paths


def _inline_only_receptors(
    registrars: Mapping[str, Any],
    built: dict[str, LayerConfigResult],
    order: list[str],
    inline_only: list[str],
    *,
    layer_of: Mapping[str, str],
    slots_of: Callable[[type], dict[str, WiringSlot]],
) -> tuple[dict[str, list[ReceptorLocation]], dict[str, list[str]]]:
    """The receptors that hold each inline-only layer's configs inline (one hop).

    A receptor counts when it has a runtime slot (an ``__init__`` parameter
    or a class-level annotation; ``declare_only`` entries never count), its
    wiring targets the layer, and its per-key model holds inline configs of
    that layer in this build (``__inline_wired__``).  Locations are ordered
    by owner layer (dependency order), owner key (sorted) and receptor
    (``__wiring__`` order).

    Returns:
        ``(receptors, ignored)``: the counting receptors per layer, and the
        wiring entries targeting the layer that do not count (rendered with
        the reason, for error messages).
    """
    found: dict[str, list[ReceptorLocation]] = {layer: [] for layer in inline_only}
    ignored: dict[str, list[str]] = {layer: [] for layer in inline_only}

    for owner_layer in order:
        classes = registrars[owner_layer].get_all()
        per_key = built[owner_layer].per_key_models
        for owner_key in sorted(classes):
            model = per_key.get(owner_key)
            record = (vars(model).get(INLINE_WIRED_ATTR) or {}) if model is not None else {}
            for name, entry in slots_of(classes[owner_key]).items():
                target = layer_of.get(entry.spec.registry_name)
                if target not in found:
                    continue
                where = f"{owner_layer}[{owner_key}].{name}"
                inlined = record.get(name)
                if entry.declare_only:
                    ignored[target].append(f"{where} (a declare_only entry)")
                elif entry.slot is None:
                    ignored[target].append(
                        f"{where} (slot-less: no __init__ parameter or class-level "
                        f"annotation receives it)"
                    )
                elif inlined is None or inlined.get("target_layer") != target:
                    reason = (
                        "a receptor of the layer itself; a layer's configs never nest "
                        "inside its own"
                        if owner_layer == target
                        else "its config model holds no inline config for it"
                    )
                    ignored[target].append(f"{where} ({reason})")
                else:
                    found[target].append(
                        ReceptorLocation(owner_layer, owner_key, name, target, tuple(inlined["keys"]))
                    )
    return found, ignored


def _inline_only_routes(
    receptors: Mapping[str, list[ReceptorLocation]],
) -> dict[str, list[tuple[str, ReceptorLocation]]]:
    """Every way to reach each inline-only layer's receptors from the top level.

    A route is ``(full path, receptor)``.  A receptor owned by a top-level
    layer is reached directly (``hook[trace_writer].archive``: a top-level
    section accepts every key of its layer).  One owned by another
    inline-only layer is reached through each route of that layer whose
    receptor accepts the owner's key: ``env[browser].hooks[trace_writer].archive``
    exists only when ``env[browser].hooks`` allows ``trace_writer`` blocks.
    The wiring graph is acyclic and self-references never hold inline
    configs, so the recursion ends.
    """
    routes: dict[str, list[tuple[str, ReceptorLocation]]] = {}

    def resolve(layer: str) -> list[tuple[str, ReceptorLocation]]:
        if layer not in routes:
            found: list[tuple[str, ReceptorLocation]] = []
            for location in receptors[layer]:
                if location.owner_layer in receptors:  # an inline-only owner
                    suffix = f"[{location.owner_key}].{location.receptor}"
                    found.extend(
                        (path + suffix, location)
                        for path, via in resolve(location.owner_layer)
                        if location.owner_key in via.keys
                    )
                else:
                    found.append((str(location), location))
            routes[layer] = found
        return routes[layer]

    for layer in receptors:
        resolve(layer)
    return routes


def _no_receptor_detail(layer: str, registrar: Any, ignored: list[str]) -> str:
    """Error detail: no receptor of this build holds *layer*'s configs inline."""
    registry = _registry_name(registrar, layer)
    detail = (
        f"layer {layer!r} is inline-only (create_registrar(..., inline_only=True)), "
        f"but no receptor in this build holds its configs inline, so its "
        f"instances could not be written anywhere"
    )
    if ignored:
        detail += f"; entries that do not count: {', '.join(ignored)}"
    return detail + (
        f". Give a class of another layer in this build a receptor for it (a "
        f"__wiring__ entry targeting {registry!r} named like one of its __init__ "
        f"parameters or class-level annotations), include the owning layer in the "
        f"build, or create the registrar without inline_only=True"
    )


def _unreachable_detail(
    layer: str,
    receptors: list[ReceptorLocation],
    routes: Mapping[str, list[tuple[str, ReceptorLocation]]],
) -> str:
    """Error detail: *layer*'s receptors exist, but no path reaches any of them.

    Only receptors owned by inline-only layers can be unreachable (a
    top-level section accepts every key of its layer).
    """
    reasons = []
    for location in receptors:
        owner_routes = routes.get(location.owner_layer, [])
        if not owner_routes:
            why = f"layer {location.owner_layer!r} itself cannot be written anywhere"
        else:
            accepted = ", ".join(f"{path} accepts {list(via.keys)}" for path, via in owner_routes)
            why = (
                f"no path writes a {location.owner_key!r} block of layer "
                f"{location.owner_layer!r}: {accepted}"
            )
        reasons.append(f"{location} ({why})")
    return (
        f"layer {layer!r} is inline-only (create_registrar(..., inline_only=True)), "
        f"but no path from the top level reaches a receptor that holds its configs "
        f"inline, so its instances could not be written anywhere: {', '.join(reasons)}. "
        f"Allow the owning keys in the receptors that lead to them (their __wiring__ "
        f"keys), give a class of a top-level layer a receptor for it, or create the "
        f"registrar without inline_only=True"
    )


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


def _build_top_level_model(
    layer_results: dict[str, LayerConfigResult],
    dependency_order: list[str],
    *,
    extra: str = "forbid",
    inline_only_paths: Mapping[str, tuple[str, ...]] | None = None,
    inline_wiring: bool = True,
) -> type[BaseModel]:
    """Build the top-level ComposedConfig model.

    The one place the top-level model is built.  Each top-level layer
    becomes a ``list[union_type]`` field with an empty list default; each
    inline-only layer a hidden phantom field
    (``SkipJsonSchema[Any] = Field(default=None, exclude=True)``) refused by
    the ``_refuse_inline_only_layers`` field validator.  ``model_config``
    carries *extra* and the schema extensions (``json_schema_extra``).

    Args:
        layer_results: Built layers by name.
        dependency_order: Layer names, leaves first (field order).
        extra: The extra policy (``top_level_extra``).
        inline_only_paths: Full receptor paths per inline-only layer.
        inline_wiring: Recorded as ``x-inline-wiring``.
    """
    inline_only_paths = dict(inline_only_paths or {})
    field_defs: dict[str, Any] = {}
    for layer_name in dependency_order:
        if layer_name in inline_only_paths:
            field_defs[layer_name] = (SkipJsonSchema[Any], Field(default=None, exclude=True))
        else:
            union = layer_results[layer_name].union_type
            field_defs[layer_name] = (list[union], [])  # type: ignore[valid-type]

    validators: dict[str, Any] = {}
    if inline_only_paths:
        where = {layer: " or ".join(paths) for layer, paths in inline_only_paths.items()}
        validators[_REFUSER_NAME] = field_validator(*where, mode="after")(
            _inline_only_refuser(where)
        )

    return _derive_model(
        _TOP_LEVEL_NAME,
        BaseModel,
        field_defs,
        extra=extra,
        module=__name__,
        config={
            "json_schema_extra": _top_level_schema_extra(
                dependency_order, inline_wiring, inline_only_paths, extra,
            ),
        },
        validators=validators,
    )


def _inline_only_refuser(where: Mapping[str, str]) -> Callable[..., Any]:
    """The field validator body that refuses any value of an inline-only layer."""

    def _refuse_inline_only_layers(cls: Any, value: Any, info: Any) -> Any:
        raise PydanticCustomError(
            INLINE_ONLY_ERROR_TYPE,
            INLINE_ONLY_MESSAGE,
            {"layer": info.field_name, "where": where[info.field_name]},
        )

    return _refuse_inline_only_layers


def _top_level_schema_extra(
    dependency_order: list[str],
    inline_wiring: bool,
    inline_only_paths: Mapping[str, tuple[str, ...]],
    extra: str,
) -> dict[str, Any]:
    """The top-level ``json_schema_extra`` (JSON-serializable, deterministic)."""
    schema_extra: dict[str, Any] = {
        "x-composed-layers": list(dependency_order),
        "x-inline-wiring": inline_wiring,
    }
    if inline_only_paths:
        schema_extra["x-inline-only-layers"] = {
            layer: list(paths) for layer, paths in inline_only_paths.items()
        }
        if extra != "forbid":
            # Without additionalProperties: false, keep refusing the names.
            schema_extra["not"] = {
                "anyOf": [{"required": [layer]} for layer in inline_only_paths],
            }
    return schema_extra
