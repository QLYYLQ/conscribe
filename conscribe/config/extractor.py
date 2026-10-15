"""Config schema extractor.

Turns a registered implementation class into the pydantic model that
validates its config block (the builder then adds the discriminator).
Everything the extractor knows about a constructor comes from
:func:`conscribe.config.mro.walk_init_chain` and :mod:`conscribe.introspect`
-- the single rule for "which names reach the constructor" -- and every
annotation is resolved by :mod:`conscribe.config._annotations` in the
namespace of the class that *declares* it (the dataclass that declares a
field, the module of a hand-written ``__init__``).

Extraction paths
================

Every path reaches the wiring pass; nothing is dropped silently.

Tier 3 -- ``__config_schema__``
    The schema is returned verbatim (its validators and ``model_config``
    stay).  The class's wiring is still resolved (problems raise), and every
    merged ``__wiring__`` entry that has a runtime slot must be a field of
    the schema (:class:`~conscribe.exceptions.ConfigDeclarationError`
    otherwise); slot-less and ``declare_only`` entries need not be.  An
    entry's ``default`` must equal the schema field's default.

Flattened -- ``__config_from_param__ = "<param>"`` (opt-in, inherited)
    The config *is* that parameter's pydantic model: the extracted model
    subclasses it (its validators and ``model_config`` apply), takes field
    defaults from the parameter's default instance and adds the class's
    wiring receptors.  Validated strictly; the model carries
    ``__config_from_param__``.  Without the opt-in a single BaseModel
    parameter is an ordinary nested field (1.x flattened it implicitly).

Constructor reflection (Tier 1 / 1.5 / 2)
    Every parameter the constructor accepts, following the ``**kwargs``
    forwarding chain up the MRO as far as the MRO scope / depth allow:
    hand-written ``__init__`` methods, dataclass / attrs generated ones,
    pydantic models whose own ``__init__`` only forwards ``**data`` (their
    ``model_fields``) and pydantic dataclasses (their fields).  A class with
    no ``__init__`` anywhere in its MRO has no knobs (``extra="forbid"``).
    A registered pydantic model's config is rebuilt from its fields on a
    fresh model -- never derived from the registered class -- so the class's
    own validators do not run on its config.

Fields
======

* Each knob is built by :func:`conscribe.config._fieldinfo.config_field`:
  the whole ``Annotated[T, Field(...), ...]`` declaration is carried (title,
  description, examples, aliases, discriminator, constraints, markers...);
  ``exclude=`` is a :class:`~conscribe.exceptions.ConfigDeclarationError`;
  ``deprecated=`` becomes a JSON-schema marker; a docstring (Google / NumPy /
  reST ``Args:`` or ``Attributes:``) supplies the description when no
  ``Field`` does (Tier 1.5).
* Defaults: exactly one source (the signature or ``Field(default=...)`` /
  ``default_factory``; two different ones raise).  ``dataclasses.field(
  default_factory=...)``, attrs ``Factory`` and pydantic factories are read
  from the field itself -- the ``<factory>`` / ``NOTHING`` /
  ``PydanticUndefined`` sentinels never reach a model.  A *pure* factory (a
  builtin container constructor, or a zero-argument function / lambda with no
  closure that uses no global name but safe builtins) is evaluated twice and,
  when both results are equal, becomes a plain default; any other factory
  stays a ``default_factory``.  A ``Field(...)`` used as a default value
  (``x: int = Field(1, ge=0)``) is a field declaration, as in pydantic.
* None rule (every Python version): a ``None`` default makes the type admit
  ``None`` and ``Optional[Annotated[T, Field(...)]]`` keeps its ``FieldInfo``.
* Annotations that cannot be resolved: a name bound only under
  ``if TYPE_CHECKING:`` (or local to the function defining the class)
  degrades the field to ``Any`` -- default, description and constraints
  kept -- with a :class:`~conscribe.exceptions.DegradationWarning`
  (:class:`~conscribe.exceptions.ExtractionWarning`) and a
  ``__degraded_fields__`` record; a name found nowhere (a typo) or broken
  ``Annotated`` metadata raises
  :class:`~conscribe.exceptions.ConfigAnnotationError`.  Types pydantic
  cannot build a schema for degrade the same way (reason
  ``"pydantic_incompatible"``).
* ``__config_annotated_only__``: only parameters declared with a ``Field``
  (in ``Annotated`` metadata or as the default value) are knobs; wiring
  receptors are fields regardless.  Fields of pydantic models / dataclasses
  always count as declared.
* Extra policy: ``"allow"`` when the ``**kwargs`` chain was cut (MRO scope /
  depth, an ``__init__`` whose signature cannot be read, a Protocol
  placeholder that swallows its arguments -- the last two also emit an
  :class:`~conscribe.exceptions.ExtractionWarning`) or the constructor takes
  ``*args`` / ``**kwargs`` that go nowhere known; ``"forbid"`` otherwise.  A
  pydantic model / dataclass keeps its own ``extra`` (``"forbid"`` when
  unset).

Wiring
======

Every merged ``__wiring__`` entry becomes a field; its config value is a
registry key, so the field type is built from the wiring keys:

* The receptor's annotation decides the shape: every non-``None``,
  non-container member of a union maps to ``Literal[<keys>]`` (deduplicated),
  container members (``list``, ``dict``, ``set``...) to
  ``list[Literal[<keys>]]``, ``None`` stays; ``Annotated`` metadata is kept.
  Unresolvable annotations of receptors never raise: their shape is read
  from the annotation text.  A user-written ``Literal[...]`` member whose
  values are not all wiring keys is a ``ConfigDeclarationError``.
* Presence: a receptor with a runtime slot is required unless its default
  (the parameter's own, or ``None`` when a class-annotation slot admits
  ``None``) says otherwise; a slot-less or ``declare_only`` entry is
  ``Optional[Literal[...]] = None``.  A non-``None`` parameter default must
  be a valid value of the field.
* ``default`` (dict-form entries, single-instance receptors with a slot
  only): the field becomes optional with that key as default; when the
  receptor admits ``None`` it stays ``Optional`` (omitted -> the default key,
  ``null`` -> ``None``).  A parameter default naming another key is an error.
* Problems raise one :class:`~conscribe.exceptions.WiringResolutionError`
  listing every problem of the class (unknown keys and registries, grammar,
  ``bad_default``, ``contradiction`` for a ``declare_only`` entry that has a
  runtime slot); nothing is caught.

Model attributes
================

Models built here carry ``__conscribe_generated__ = True`` and, when
applicable:

``__wired_fields__``  ``{field: registry name}`` (``"literal"`` for a literal list)
``__wired_keys__``    ``{field: (key, ...)}`` -- the allowed keys, in ``Literal`` order
``__wired_defaults__`` ``{field: key}`` -- the default key of single-instance
                      receptors (an explicit wiring ``default``, or a parameter
                      default that is a key string)
``__degraded_fields__`` ``[DegradedField(...)]`` with a ``reason``
``__config_from_param__`` the flattened parameter (opt-in models only)
``__legacy_flat_keys__`` ``{"param": name, "keys": (...)}`` -- see below

Old-shape guard
===============

1.x flattened a class whose own ``__init__`` takes exactly one named
parameter typed with a BaseModel.  In 2.0 that parameter is a nested block,
so for one major version such models reject input that still uses the flat
keys (the BaseModel's field names / aliases that are not fields of the 2.0
model) with a pydantic error of type ``legacy_flat_config``:
``'<param>' is a nested block in conscribe 2.0; write {<param>: {...}} or
set __config_from_param__ = '<param>'``.  The check is a ``mode="before"``
model validator driven by ``__legacy_flat_keys__``, so models derived from
the extracted one (per-key models, composed rebuilds) inherit it.  That
parameter stays a field even under ``__config_annotated_only__`` (its
model's fields were knobs in 1.x, so the block must stay writable).
"""
from __future__ import annotations

import ast
import builtins
import dataclasses
import dis
import inspect
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Annotated, Any, ClassVar, Final, ForwardRef, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError, PydanticUndefined

from conscribe.config import _annotations as _ann
from conscribe.config._fieldinfo import CONFIG_FIELD_ATTRS, config_field, field_infos_of, narrow_keys
from conscribe.config._utils import is_pydantic_dataclass, own_init, qualified_name
from conscribe.config.degradation import DegradedField, degrade_field_definitions
from conscribe.config.docstring import parse_param_descriptions
from conscribe.config.mro import (
    InitChain,
    InitParam,
    mro_depth_problem,
    mro_scope_problem,
    walk_init_chain,
)
from conscribe.config.provenance import ConfigProvenance
from conscribe.exceptions import (
    ConfigDeclarationError,
    DegradationWarning,
    ExtractionWarning,
    InvalidConfigSchemaError,
    WiringProblem,
    WiringResolutionError,
)
from conscribe.introspect import ReceptorSlot, config_from_param

EMPTY: Any = inspect.Parameter.empty
NoneType = type(None)

#: Marker class attribute of every model conscribe builds (see
#: :data:`conscribe.config.builder.GENERATED_MARKER`).
_GENERATED_MARKER = "__conscribe_generated__"

#: Error type raised by the old-shape guard.
LEGACY_FLAT_ERROR = "legacy_flat_config"

__all__ = [
    "LEGACY_FLAT_ERROR", "extract_config_schema", "extract_config_provenance",
    "extract_own_init_params",
]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_config_schema(
    cls: type,
    mro_scope: str | list[str] = "local",
    mro_depth: int | None = None,
) -> type[BaseModel] | None:
    """Extract the config schema of a class.

    Extraction order:

    1. ``cls.__config_schema__`` -> returned verbatim (Tier 3), after its
       wiring checks.
    2. ``__config_from_param__ = "<param>"`` -> the flattened model of that
       parameter (opt-in).
    3. Constructor reflection -> a dynamic pydantic model (Tier 1 / 1.5 / 2),
       wiring applied.
    4. Nothing to emit (no knob, no wiring receptor, closed constructor) ->
       ``None``.

    The module docstring describes the field, wiring and default rules.

    A registered pydantic model (or pydantic dataclass) gets a config model
    rebuilt from its fields on a fresh ``BaseModel`` -- never derived from
    the registered class, which would register the config model and run the
    class's ``__init__`` -- so the class's own validators and
    ``model_config`` (apart from ``extra``) do not apply to its config.

    ``__config_from_param__`` is validated strictly: it must name a
    constructor parameter (within the MRO scope) whose annotation is a
    pydantic ``BaseModel`` subclass (not ``Optional`` / a union, not a
    ``RootModel``, not a registered class); the parameter must not be a
    wiring receptor; every other constructor parameter must be a wiring
    receptor; the parameter default may be absent or an instance of the
    model (its values become the field defaults), never ``None``.

    Args:
        cls: The class to extract config schema from.
        mro_scope: Scope for the ``**kwargs`` MRO walk: ``"local"``,
            ``"third_party"``, ``"all"``, or a ``list[str]`` of package
            names (local classes + only the listed packages).  Overridden
            per class by ``__config_mro_scope__``.
        mro_depth: Max parent ``__init__`` definers to visit (``None`` =
            unlimited).  Overridden per class by ``__config_mro_depth__``.

    Returns:
        A pydantic ``BaseModel`` subclass, or ``None``.

    Raises:
        InvalidConfigSchemaError: ``__config_schema__`` is not a
            ``BaseModel`` subclass (a ``TypeError``).
        WiringResolutionError: The class's ``__wiring__`` has problems (all
            of them are listed).
        ConfigAnnotationError: An annotation that a knob needs has a typo or
            broken ``Annotated`` metadata.
        ConfigDeclarationError: The class declares something contradictory
            (two defaults, ``exclude=``, a bad ``__config_from_param__``, a
            Tier-3 schema without a wired field, a ``Literal`` member that is
            not a wiring key, an invalid ``__config_mro_scope__`` /
            ``__config_mro_depth__``, a parameter no config can pass...).
    """
    schema = getattr(cls, "__config_schema__", None)
    if schema is not None:
        return _tier3(cls, schema)[0]
    scope = _mro_setting(cls, "__config_mro_scope__", "mro_scope", mro_scope, mro_scope_problem)
    depth = _mro_setting(cls, "__config_mro_depth__", "mro_depth", mro_depth, mro_depth_problem)
    return _Extraction(cls, walk_init_chain(cls, scope=scope, depth=depth)).run()


def extract_config_provenance(
    cls: type,
    mro_scope: str | list[str] = "local",
    mro_depth: int | None = None,
) -> ConfigProvenance:
    """Extract immutable declaration provenance using the schema extraction path.

    Accepts the same settings and raises the same declaration errors and
    warnings as :func:`extract_config_schema`. It performs extraction (and
    its normal factory handling); it is not a source-code-only static scan.
    It neither changes the returned schema nor adds JSON Schema extensions.
    Input explicit/default selection and actual runtime injection belong to
    the consuming framework, not this declaration record.
    """
    from conscribe.wiring import wiring_provenance

    schema = getattr(cls, "__config_schema__", None)
    if schema is not None:
        schema, wiring = _tier3(cls, schema)
        history = wiring_provenance(cls)
        fields = _schema_provenance_fields(schema, "schema", wiring, history)
        return ConfigProvenance(
            cls=cls, path="schema", fields=fields, exclusions=(),
            constructor_chain=None, wiring=history, source_model=schema,
            schema_declared_on=_attribute_owner(cls, "__config_schema__"),
            mro=cls.__mro__,
        )
    scope = _mro_setting(cls, "__config_mro_scope__", "mro_scope", mro_scope, mro_scope_problem)
    depth = _mro_setting(cls, "__config_mro_depth__", "mro_depth", mro_depth, mro_depth_problem)
    extraction = _Extraction(cls, walk_init_chain(cls, scope=scope, depth=depth))
    model = extraction.run()
    return extraction.provenance(model, scope=scope, depth=depth)


def _attribute_owner(cls: type, name: str) -> type | None:
    return next((owner for owner in cls.__mro__ if name in vars(owner)), None)


def _schema_provenance_fields(model: type[BaseModel], source: Any, wiring: _Wiring, history: Any) -> dict:
    from conscribe.config.provenance import ConfigFieldProvenance, FieldDeclaration, _receptor

    fields = {}
    for name in model.model_fields:
        owner = _ann.annotation_owner(model, name) or model
        declarations = tuple(
            FieldDeclaration(
                name, source, base,
                status="included" if base is owner else "overridden",
                reason=None if base is owner else "overridden",
            )
            for base in model.__mro__ if name in _ann.own_annotations(base)
        ) or (FieldDeclaration(name, source, owner),)
        fields[name] = ConfigFieldProvenance(
            name, source, owner, None, declarations,
            _receptor(wiring.slots.get(name)), history.get(name),
        )
    return fields


def _mro_setting(
    cls: type, attr: str, option: str, fallback: Any, check: Callable[[Any], str | None]
) -> Any:
    """The effective MRO scope / depth of *cls*, validated.

    The class attribute (inherited) wins over the registrar option.

    Raises:
        ConfigDeclarationError: The value is invalid; the message names where
            it was set.
    """
    declared_on = next((klass for klass in cls.__mro__ if attr in vars(klass)), None)
    value = vars(declared_on)[attr] if declared_on is not None else fallback
    problem = check(value)
    if problem is None:
        return value
    if declared_on is None:
        where = f"the registrar's {option}={value!r}"
    elif declared_on is cls:
        where = f"{attr} = {value!r}"
    else:
        where = f"{attr} = {value!r} (inherited from {qualified_name(declared_on)})"
    raise ConfigDeclarationError(qualified_name(cls), f"{where}: {problem}.")


def extract_own_init_params(
    cls: type,
) -> tuple[list[inspect.Parameter], dict[str, Any] | None]:
    """Parameters introduced by ``cls``'s own constructor (NOT inherited).

    Compatibility helper of the nested-mode builder.  Returns the named
    parameters of ``cls``'s own ``__init__`` (instance parameter, ``*args``
    and ``**kwargs`` excluded) that the nearest parent constructor does not
    already take, and their annotations resolved in the namespace of the
    class that declares each one (owner-aware: inherited dataclass fields
    resolve in their own module).  An annotation that cannot be resolved is
    returned in its raw form.  New code should use :func:`_build_param_fields`,
    which applies the full field rules (factories, warnings, errors).

    Returns:
        ``(params, hints)``: ``inspect.Parameter`` objects (a dataclass field
        with a ``default_factory`` shows the ``<factory>`` sentinel as its
        default) and ``{name: annotation}``.  ``([], None)`` when ``cls``
        does not define its own constructor.
    """
    unique = _own_init_params(cls)
    if unique is None:
        return [], None
    hints: dict[str, Any] = {}
    for param in unique:
        res = param.resolve()
        if res.status == "resolved":
            hints[param.name] = res.value
        elif res.status != "unannotated":
            hints[param.name] = res.raw
    return [p.parameter for p in unique], hints


# ---------------------------------------------------------------------------
# Nested-mode entry point (used by builder._extract_params_by_level)
# ---------------------------------------------------------------------------


def _build_param_fields(
    cls: type,
    params: Iterable[InitParam],
    *,
    leaf: type | None = None,
    annotated_only: bool | None = None,
) -> tuple[dict[str, tuple[Any, FieldInfo]], list[DegradedField]]:
    """Config field definitions of some constructor parameters of *cls*.

    Applies exactly the rules :func:`extract_config_schema` uses for a knob
    (owner-aware resolution, ``None`` rule, ``Field`` carry, factories and
    the purity gate, degradation warnings, annotation errors,
    ``__config_annotated_only__``).  Entry point for builders that assemble
    models from parameters themselves (nested mode).

    Args:
        cls: The class whose constructor takes *params* (docstrings are read
            along its MRO).
        params: Parameters from :func:`conscribe.config.mro.walk_init_chain`.
        leaf: The registered class (named in messages); defaults to *cls*.
        annotated_only: Override the leaf's ``__config_annotated_only__``.

    Returns:
        ``({name: (type, FieldInfo)}, degraded_fields)``; parameters skipped
        by ``__config_annotated_only__`` and parameters no config can pass
        (positional-only or ``_private`` ones that have a default) are
        absent.

    Raises:
        ConfigDeclarationError: Besides the field rules, a *required*
            parameter no config can pass, or a name that collides with a
            pydantic ``model_*`` member.
    """
    extraction = _Extraction(leaf or cls, walk_init_chain(cls, depth=0))
    if annotated_only is not None:
        extraction.annotated_only = annotated_only
    fields: dict[str, tuple[Any, FieldInfo]] = {}
    for param in params:
        if not extraction.configurable(param):
            continue
        definition = extraction.knob_field(extraction.knob(param))
        if definition is not None:
            fields[param.name] = definition
    return fields, list(extraction.degraded)


def _own_init_params(cls: type) -> list[InitParam] | None:
    """``InitParam`` objects introduced by *cls*'s own constructor."""
    if own_init(cls) is None:
        return None
    own = [p for p in walk_init_chain(cls, depth=0).params if p.level == 0]
    parent_names: set[str] = set()
    for base in cls.__mro__[1:]:
        if own_init(base) is not None:
            parent_names = {p.name for p in walk_init_chain(base, depth=0).params if p.level == 0}
            break
    return [p for p in own if p.name not in parent_names]


# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------


def _tier3(cls: type, schema: Any) -> tuple[type[BaseModel], _Wiring]:
    """Check a ``__config_schema__``; return it and the wiring evidence as read."""
    if not (isinstance(schema, type) and _safe_issubclass(schema, BaseModel)):
        raise InvalidConfigSchemaError(cls.__qualname__, schema)
    wiring = _Wiring.prepare(cls)
    owner = qualified_name(cls)
    fields = schema.model_fields
    missing = [
        name
        for name, rw in wiring.resolved.items()
        if not rw.declare_only and wiring.slots.get(name) is not None and name not in fields
    ]
    if missing:
        raise ConfigDeclarationError(
            owner,
            f"__config_schema__ {schema.__qualname__} has no field for the wired "
            f"receptor(s) {missing}; a Tier-3 schema is used verbatim, so it must "
            f"declare every __wiring__ entry the class has a runtime slot for. Add "
            f"the field(s) to {schema.__qualname__} (typed with the allowed keys) or "
            f"remove the entries from __wiring__.",
        )
    for name, rw in wiring.resolved.items():
        if rw.default_key is None or name not in fields:
            continue
        info = fields[name]
        if info.default_factory is not None or info.default is PydanticUndefined or (
            info.default != rw.default_key
        ):
            shown = (
                "no default"
                if info.default is PydanticUndefined and info.default_factory is None
                else (
                    f"default_factory {info.default_factory!r}"
                    if info.default_factory is not None
                    else f"default {info.default!r}"
                )
            )
            raise ConfigDeclarationError(
                owner,
                f"the wiring default {rw.default_key!r} disagrees with the field of "
                f"__config_schema__ {schema.__qualname__} ({shown}); a Tier-3 schema "
                f"is used verbatim, so give the field the default {rw.default_key!r} "
                f"or remove 'default' from the wiring entry.",
                name,
            )
    return schema, wiring


# ---------------------------------------------------------------------------
# Wiring preparation
# ---------------------------------------------------------------------------


@dataclass
class _Wiring:
    """The resolved ``__wiring__`` of one class plus its receptor slots."""

    cls: type
    owner_key: str | None
    resolved: dict[str, Any]  # name -> conscribe.wiring.ResolvedWiring
    slots: dict[str, ReceptorSlot | None]
    order: list[str]
    problems: list[WiringProblem] = dataclasses.field(default_factory=list)

    @classmethod
    def prepare(cls, owner: type) -> _Wiring:
        """Resolve *owner*'s wiring (grammar, registries, keys, defaults, slots).

        Raises one :class:`WiringResolutionError` listing every problem of
        the class, as :func:`conscribe.wiring.resolve_wiring` does (including
        the slot-dependent ``bad_default`` / ``contradiction`` checks).
        """
        from conscribe.wiring import _owner_key, _resolve_class_and_slots, collect_wiring_from_mro

        owner_key = _owner_key(owner)
        # The resolution pass already read the receptor slots of every
        # resolved entry; reuse them instead of resolving annotations twice.
        resolved, problems, slots = _resolve_class_and_slots(owner, owner_key)
        if not resolved and not problems:
            return cls(owner, owner_key, {}, {}, [])
        if problems:
            raise WiringResolutionError(problems=problems)
        order = [name for name in collect_wiring_from_mro(owner) if isinstance(name, str)]
        return cls(owner, owner_key, resolved, {n: slots.get(n) for n in resolved}, order)

    def problem(self, name: str, registry_name: str | None, kind: str, detail: str) -> WiringProblem:
        return WiringProblem(
            owner=self.cls.__qualname__,
            param_name=name,
            registry_name=registry_name or "",
            kind=kind,
            detail=detail,
            owner_key=self.owner_key,
        )

    def _ordered(self, problems: list[WiringProblem]) -> list[WiringProblem]:
        rank = {name: index for index, name in enumerate(self.order)}

        def key(item: tuple[int, WiringProblem]) -> tuple[int, int]:
            index, problem = item
            if not problem.param_name:
                return (-1, index)
            return (rank.get(problem.param_name, len(rank)), index)

        return [p for _, p in sorted(enumerate(problems), key=key)]

    def raise_problems(self) -> None:
        """Raise the problems found while building the wired fields, if any."""
        if self.problems:
            raise WiringResolutionError(problems=self._ordered(self.problems))


# ---------------------------------------------------------------------------
# One extraction
# ---------------------------------------------------------------------------


@dataclass
class _Knob:
    """One constructor parameter (or receptor slot) on its way to a field."""

    name: str
    #: Resolved annotation (``Annotated`` kept), degraded value, raw form when
    #: it could not be resolved, or ``EMPTY`` when unannotated.
    annotation: Any
    status: str
    res: _ann.ResolvedAnnotation | None
    owner: type
    source: str
    default: Any = EMPTY
    factory: Any = None
    #: ``FieldInfo`` objects declared outside the annotation (a ``Field(...)``
    #: default value, a pydantic model field).
    infos: tuple[FieldInfo, ...] = ()
    via: str = "init"


class _Extraction:
    """State of one :func:`extract_config_schema` run."""

    def __init__(self, cls: type, chain: InitChain) -> None:
        self.cls = cls
        self.owner = qualified_name(cls)
        self.chain = chain
        self.annotated_only = bool(getattr(cls, "__config_annotated_only__", False))
        self.fields: dict[str, tuple[Any, Any]] = {}
        #: ``{field: "module.QualName"}`` / ``{field: class}`` of the declaring class.
        self.sources: dict[str, str] = {}
        self.source_classes: dict[str, type] = {}
        self.degraded: list[DegradedField] = []
        self.wired_fields: dict[str, str] = {}
        self.wired_keys: dict[str, tuple[str, ...]] = {}
        self.wired_defaults: dict[str, str] = {}
        #: The parameter 1.x flattened implicitly: always a field, even under
        #: ``__config_annotated_only__`` (its model's fields were knobs in 1.x).
        self.legacy_param: str | None = None
        self._docs: dict[str, str] | None = None
        self._runtime: InitChain | None = None
        self._wiring: _Wiring | None = None
        self.exclusions: dict[str, str] = {}
        self.source_model: type[BaseModel] | None = None

    # -- driver ---------------------------------------------------------------

    def run(self) -> type[BaseModel] | None:
        self._warn_if_unreadable()
        wiring = _Wiring.prepare(self.cls)
        self._wiring = wiring
        from_param = config_from_param(self.cls)
        if from_param is not None:
            return self._flatten(from_param, wiring)

        guard = self._legacy_flat_guard(wiring)
        self.legacy_param = guard["param"] if guard is not None else None
        for param in self.chain.params:
            if param.name in wiring.resolved:
                self._add_wired(param.name, wiring)
                continue
            if not self.configurable(param):
                continue
            definition = self.knob_field(self.knob(param))
            if definition is not None:
                self.fields[param.name] = definition
                self.sources[param.name] = qualified_name(param.declared_on)
                self.source_classes[param.name] = param.declared_on
        for name in wiring.resolved:
            if name not in self.wired_fields:
                self._add_wired(name, wiring)
        wiring.raise_problems()

        extra = self._extra_policy()
        if not self.fields and extra == "forbid":
            return None  # the builder's discriminator-only model says the same
        return self._create(f"{self.cls.__name__}Config", extra, guard=guard)

    def provenance(self, model: type[BaseModel] | None, *, scope: Any, depth: int | None) -> ConfigProvenance:
        """Freeze the decisions made by this extraction, without re-extracting."""
        from conscribe.config.provenance import (
            ConfigFieldProvenance, FieldDeclaration, _constructor, _receptor,
        )
        from conscribe.wiring import wiring_provenance

        assert self._wiring is not None
        history = wiring_provenance(self.cls)
        constructor = _constructor(self.chain)
        fields = {}
        annotation_history: dict[str, list[tuple[type, bool]]] | None = None
        if self.source_model is not None:
            fields.update(_schema_provenance_fields(
                self.source_model, "flattened", self._wiring, history,
            ))
        for name in (() if model is None else model.model_fields):
            if name in fields:
                continue
            slot = self._wiring.slots.get(name)
            param = self.chain.get(name)
            if param is None and slot is not None and slot.via == "init":
                if self._runtime is None:
                    self._runtime = walk_init_chain(self.cls)
                param = self._runtime.get(name)
            entry = history.get(name)
            if param is not None:
                source, owner, init_owner = param.source, param.declared_on, param.init_owner
                chain = constructor if self.chain.get(name) is not None else _constructor(self._runtime)
                declarations = tuple(d for d in chain.declarations if d.name == name)
            elif slot is not None:
                source, owner, init_owner = "annotation", slot.declared_on or self.cls, None
                from conscribe.introspect import _annotation_slots

                if annotation_history is None:
                    annotation_history = {}
                    _annotation_slots(self.cls, history=annotation_history)
                declarations = tuple(
                    FieldDeclaration(
                        name, source, declared_on,
                        status="excluded" if pseudo else "included" if declared_on is owner else "overridden",
                        reason="pseudo_field" if pseudo else None if declared_on is owner else "overridden",
                    )
                    for declared_on, pseudo in reversed(annotation_history.get(name, []))
                )
            else:
                # Slot-less fields come from the wiring owner, not the leaf.
                assert entry is not None and not entry.excluded
                source, owner, init_owner = "wiring", entry.winner.declared_on, None
                declarations = (FieldDeclaration(name, source, owner),)
            fields[name] = ConfigFieldProvenance(
                name, source, owner, init_owner, declarations, _receptor(slot), entry,
            )
        exclusions = tuple(
            FieldDeclaration(
                p.name, p.source, p.declared_on, p.init_owner,
                "excluded", self.exclusions[p.name],
            )
            for p in self.chain.params if p.name in self.exclusions
        )
        flattened = config_from_param(self.cls)
        return ConfigProvenance(
            cls=self.cls, path="flattened" if flattened is not None else "reflection",
            fields=fields, exclusions=exclusions, constructor_chain=constructor,
            wiring=history, source_model=self.source_model, flattened_param=flattened,
            schema_declared_on=_attribute_owner(self.cls, "__config_from_param__") if flattened else None,
            mro_scope=scope if isinstance(scope, str) else tuple(sorted(scope)),
            mro_depth=depth,
            mro=self.cls.__mro__,
        )

    # -- knobs ----------------------------------------------------------------

    def configurable(self, param: InitParam) -> bool:
        """Whether a config can pass *param* to the constructor.

        A positional-only parameter cannot be passed by keyword, and pydantic
        never turns a name with a leading underscore into a field: such a
        parameter is skipped when it has a default (the constructor's default
        applies) and is a :class:`ConfigDeclarationError` when the
        constructor requires it.  A name that collides with a pydantic
        ``model_*`` member is an error too.
        """
        reason = None
        if not param.keyword:
            reason = "it is positional-only"
        elif param.name.startswith("_"):
            reason = "a config field cannot start with an underscore (pydantic ignores it)"
        if reason is None:
            _check_field_name(param.name, self.owner)
            return True
        declared = param.declared_default()
        required = (
            declared.value is EMPTY
            and declared.factory is None
            and not declared.takes_self
            and not (
                declared.field_info is not None
                and (
                    declared.field_info.default is not PydanticUndefined
                    or declared.field_info.default_factory is not None
                )
            )
        )
        if required:
            raise ConfigDeclarationError(
                self.owner,
                f"the constructor requires {param.name!r}, but {reason}, so no config "
                f"can pass it. Give it a default, make it a keyword parameter with a "
                f"public name, or declare __config_schema__ and construct the class "
                f"yourself.",
                param.name,
            )
        self.exclusions[param.name] = "positional_only" if not param.keyword else "private_name"
        return False

    def knob(self, param: InitParam) -> _Knob:
        """Collect everything a parameter declares (annotation, default, Field)."""
        info = param.field
        if param.source in ("model", "pydantic_dataclass") and isinstance(info, FieldInfo):
            declared = info.annotation
            if declared is not None and not _ann.contains_forward_refs(declared):
                return _Knob(
                    param.name, declared, "resolved", None, param.declared_on, param.source,
                    infos=(info,),
                )
            # The model is not complete yet (a forward reference to a later
            # class).  When pydantic evaluated the ``Annotated`` wrapper it
            # already merged its metadata into ``info``; when the whole
            # annotation stayed a string, ``info`` only holds what the
            # assignment declared and the metadata must come from the
            # annotation conscribe resolves.
            res = param.resolve()
            annotation = _knob_annotation(res)
            if res.status in ("resolved", "degraded") and not _annotation_left_to_us(declared):
                annotation = _ann.unwrap_annotated(annotation)[0]
            return _Knob(
                param.name, annotation, res.status, res, param.declared_on, param.source,
                infos=(info,),
            )

        res = param.resolve()
        annotation = _knob_annotation(res)
        if res.status == "resolved":
            # ``InitVar[T]`` / ``Final[T]`` dataclass fields are plain ``T`` knobs.
            annotation = _ann.unwrap_final(_ann.unwrap_initvar(annotation))
        declared_default = param.declared_default()
        if declared_default.takes_self:
            _ann.warn_at(
                f"{self.owner}.{param.name}: its attrs default is Factory(takes_self=True), "
                f"which needs the instance; a config model cannot express it, so the field "
                f"is required.",
                ExtractionWarning,
                param.declared_on,
                stacklevel=4,  # knob -> run -> extract_config_schema -> caller
            )
        field_info = declared_default.field_info
        return _Knob(
            param.name, annotation, res.status, res, param.declared_on, param.source,
            default=declared_default.value,
            factory=declared_default.factory,
            # ``x: int = Field(1, ge=0)`` declares the field, as in pydantic.
            infos=(field_info,) if field_info is not None else (),
        )

    def knob_field(self, knob: _Knob) -> tuple[Any, FieldInfo] | None:
        """The ``(type, FieldInfo)`` of an ordinary (non-wired) knob, or ``None``."""
        if (
            self.annotated_only
            and knob.source not in ("model", "pydantic_dataclass")
            and knob.name != self.legacy_param
            and not _declares_field(knob)
        ):
            self.exclusions[knob.name] = "annotated_only"
            return None
        if knob.status == "error" and knob.res is not None:
            knob.res.raise_if_error(self.cls)
        if knob.status == "degraded":
            self._record_degraded(knob)
        annotation = Any if knob.annotation is EMPTY else knob.annotation
        field_type, info = config_field(
            _with_infos(annotation, knob.infos),
            owner=qualified_name(knob.owner),
            field=knob.name,
            signature_default=knob.default,
            signature_factory=knob.factory,
            description=self._description(knob.name, knob.owner),
            model_field=knob.source in ("model", "pydantic_dataclass"),
        )
        return field_type, self._settle_factory(info, knob)

    def _settle_factory(self, info: FieldInfo, knob: _Knob) -> FieldInfo:
        """Turn a pure ``default_factory`` into a plain default."""
        factory = info.default_factory
        if factory is None or not _is_pure_factory(factory):
            return info
        stable, value = _evaluate_factory(factory, owner=qualified_name(knob.owner), field=knob.name)
        if not stable:
            return info
        return _rebuild_field(info, default=value)

    def _record_degraded(self, knob: _Knob) -> None:
        res = knob.res
        assert res is not None
        self.degraded.append(
            DegradedField(
                field_name=knob.name,
                original_type_repr=res.annotation_text,
                source_class=qualified_name(knob.owner),
                reason=res.reason or "type_checking_only",
            )
        )
        # warn -> _record_degraded -> knob_field -> run -> extract_config_schema -> caller
        res.warn(self.cls, stacklevel=5)

    def _description(self, name: str, owner: type | None) -> str | None:
        """Tier 1.5: the docstring description of *name* (``Args:`` / ``Attributes:``).

        Looked up in the constructor definers (the leaf-most first, as 1.x
        did), then in the class declaring the parameter or slot.  pydantic's
        own classes are skipped (``BaseModel``'s docstring documents its
        internals).
        """
        if self._docs is None:
            docs: dict[str, str] = {}
            classes = [self.chain.definer, *self.chain.definers, self.cls]
            for klass in classes:
                if klass is None or _is_pydantic_internal(klass):
                    continue
                for key, text in parse_param_descriptions(klass).items():
                    docs.setdefault(key, text)
            self._docs = docs
        found = self._docs.get(name)
        if found is None and owner is not None and not _is_pydantic_internal(owner):
            found = parse_param_descriptions(owner).get(name)
        return found

    # -- wiring ---------------------------------------------------------------

    def _add_wired(self, name: str, wiring: _Wiring) -> None:
        rw = wiring.resolved[name]
        slot = wiring.slots.get(name)
        keys = tuple(dict.fromkeys(rw.allowed_keys))
        literal = Literal[keys]  # type: ignore[valid-type]
        if name.startswith("_"):
            raise ConfigDeclarationError(
                self.owner,
                f"a config field cannot start with an underscore (pydantic ignores it), "
                f"so the __wiring__ entry {name!r} could never be configured. Rename it.",
                name,
            )
        _check_field_name(name, self.owner)
        self.wired_fields[name] = rw.registry_name or "literal"
        self.wired_keys[name] = keys
        if rw.declare_only or slot is None:
            description = self._description(name, None)
            info = Field(default=None, description=description) if description else Field(default=None)
            self.fields[name] = (Optional[literal], info)
            self.sources[name] = self.owner
            self.source_classes[name] = self.cls
            return

        knob = self._receptor_knob(name, slot)
        mapped = self._wired_type(knob, literal, keys)
        default, factory = knob.default, knob.factory
        if (
            knob.via == "annotation"
            and default is EMPTY
            and factory is None
            and not any(
                fi.default is not PydanticUndefined or fi.default_factory is not None
                for fi in (*knob.infos, *field_infos_of(mapped))
            )
        ):
            # A class-level slot without a class-level value: required, unless
            # the annotation admits None (then omitted means None).
            default = None if _ann.admits_none(mapped) else EMPTY
        field_type, info = config_field(
            _with_infos(mapped, knob.infos),
            owner=qualified_name(knob.owner),
            field=name,
            signature_default=default,
            signature_factory=factory,
            description=self._description(name, knob.owner),
        )
        info = self._settle_factory(info, knob)
        if slot.shape == "scalar":
            # Key-string constraints (``Field(pattern="^z")``) that reject every
            # allowed key leave a field that can only ever be None.
            narrow_keys(
                keys, _ann.unwrap_annotated(field_type)[1],
                owner=qualified_name(knob.owner), field=name,
            )
        info = self._wired_default(name, rw, wiring, field_type, info, scalar=slot.shape == "scalar")
        if info is None:
            return
        self.fields[name] = (field_type, info)
        self.sources[name] = qualified_name(knob.owner)
        self.source_classes[name] = knob.owner

    def _receptor_knob(self, name: str, slot: ReceptorSlot) -> _Knob:
        """The knob of a receptor slot (constructor parameter or class annotation).

        A class-annotation slot takes its default from the class-level value
        (``hooks: list[P] = []``; see :attr:`ReceptorSlot.default`); a
        ``Field(...)`` value declares the field.
        """
        if slot.via == "init":
            param = self.chain.get(name)
            if param is None:
                if self._runtime is None:
                    self._runtime = walk_init_chain(self.cls)
                param = self._runtime.get(name)
            if param is not None:
                knob = self.knob(param)
                if knob.status != "resolved":
                    owner = _ann.annotation_owner(self.cls, name)
                    from_class = _ann.resolve_class_annotation(owner, name) if owner else None
                    if from_class is not None and (
                        from_class.status == "resolved" or knob.status == "unannotated"
                    ):
                        knob = replace(
                            knob,
                            annotation=_knob_annotation(from_class),
                            status=from_class.status,
                            res=from_class,
                        )
                return knob
        owner = slot.declared_on or self.cls
        res = _ann.resolve_class_annotation(owner, name)
        try:
            value = inspect.getattr_static(self.cls, name)
        except AttributeError:
            value = None
        infos: tuple[FieldInfo, ...] = (value,) if isinstance(value, FieldInfo) else ()
        default = EMPTY if infos else slot.default
        factory = None if infos else slot.default_factory
        if res is None:  # pragma: no cover - receptor_slots found the annotation
            return _Knob(
                name, EMPTY, "unannotated", None, owner, "annotation",
                default=default, factory=factory, infos=infos, via="annotation",
            )
        return _Knob(
            name, _knob_annotation(res), res.status, res, owner, "annotation",
            default=default, factory=factory, infos=infos, via="annotation",
        )

    def _wired_type(self, knob: _Knob, literal: Any, keys: tuple[str, ...]) -> Any:
        """Map a receptor's annotation onto the wiring keys (D10)."""
        mapper = _WiredTypeMapper(self, knob.name, literal, keys)
        if knob.status in ("resolved", "unannotated"):
            return mapper.map_object(knob.annotation)
        assert knob.res is not None
        mapped = mapper.map_raw(knob.res.raw, knob.res.context)
        if knob.status == "degraded":
            _, metadata = _ann.unwrap_annotated(knob.res.value)
            if metadata:
                mapped = Annotated[(mapped, *metadata)]  # type: ignore[valid-type]
        return mapped

    def _wired_default(
        self,
        name: str,
        rw: Any,
        wiring: _Wiring,
        field_type: Any,
        info: FieldInfo,
        *,
        scalar: bool,
    ) -> FieldInfo | None:
        """Apply the wiring ``default`` / check the parameter default of a receptor."""
        default = info.default
        factory = info.default_factory
        registry = rw.registry_name
        if rw.default_key is not None:
            key = rw.default_key
            if factory is not None:
                wiring.problems.append(
                    wiring.problem(
                        name, registry, "bad_default",
                        f"the parameter declares a default_factory ({_callable_name(factory)}) "
                        f"and the wiring entry the default {key!r}; declare the default once.",
                    )
                )
                return None
            if default is not PydanticUndefined and default is not None and default != key:
                wiring.problems.append(
                    wiring.problem(
                        name, registry, "bad_default",
                        f"the parameter default {default!r} names a different key than the "
                        f"wiring default {key!r}; declare the default once (remove the "
                        f"parameter default or make the two equal).",
                    )
                )
                return None
            self.wired_defaults[name] = key
            return _rebuild_field(info, default=key)
        if factory is not None or default is PydanticUndefined or default is None:
            return info
        try:
            value = TypeAdapter(field_type).validate_python(default)
        except (ValidationError, TypeError, ValueError):
            allowed = list(self.wired_keys[name])
            wiring.problems.append(
                wiring.problem(
                    name, registry, "bad_default",
                    f"the parameter default {default!r} is not a valid value of the wired "
                    f"field: it accepts {'a list of ' if not scalar else ''}keys from "
                    f"{allowed}. Use one of them (or None), or declare the default with "
                    f"the dict-form wiring 'default'.",
                )
            )
            return None
        if scalar and isinstance(value, str):
            self.wired_defaults[name] = value
        if type(value) is not type(default):
            return _rebuild_field(info, default=value)
        return info

    # -- D6: flattening -------------------------------------------------------

    def _flatten(self, param_name: Any, wiring: _Wiring) -> type[BaseModel]:
        """Build the opt-in flattened model of ``__config_from_param__``."""
        cls = self.cls

        def fail(detail: str) -> ConfigDeclarationError:
            return ConfigDeclarationError(
                self.owner, f"__config_from_param__ = {param_name!r}: {detail}"
            )

        if not isinstance(param_name, str) or not param_name.isidentifier():
            raise fail("it must be the name of a constructor parameter (a str), or None to opt out.")
        param = self.chain.get(param_name)
        if param is None:
            raise fail(
                f"{cls.__qualname__}'s constructor has no parameter {param_name!r} "
                f"(parameters: {list(self.chain.names())}). Name the parameter that "
                f"receives the config model, or set __config_from_param__ = None."
            )
        if param_name in wiring.resolved:
            raise fail(
                f"{param_name!r} is a __wiring__ receptor; flattening never applies to a "
                f"receptor. Set __config_from_param__ = None."
            )
        knob = self.knob(param)
        if knob.status == "error" and knob.res is not None:
            knob.res.raise_if_error(cls)
        if knob.status != "resolved":
            raise fail(
                f"the annotation of {param_name!r} cannot be resolved "
                f"({knob.res.detail if knob.res else 'no annotation'}), so the config "
                f"model is unknown."
            )
        model, metadata = _ann.unwrap_annotated(knob.annotation)
        text = _ann.annotation_text(knob.annotation)
        if not (_is_model_class(model) and not _is_root_model(model)):
            raise fail(
                f"the parameter must be annotated with a pydantic BaseModel subclass "
                f"(not Optional / a union / a RootModel), got {text!r}."
            )
        if _is_registered(model):
            raise fail(
                f"{model.__qualname__} is (or inherits from) a registered class; the "
                f"flattened config subclasses the parameter's model, which would register "
                f"it. Use a plain BaseModel as the parameter type."
            )
        others = [
            p.name
            for p in self.chain.params
            if p.name != param_name and p.name not in wiring.resolved and self.configurable(p)
        ]
        if others:
            raise fail(
                f"the constructor also takes {others}, which a flattened config cannot "
                f"pass. Make them fields of {model.__qualname__}, declare them in "
                f"__wiring__, or set __config_from_param__ = None."
            )

        instance = self._flatten_default(knob, model, fail)
        self.source_model = model
        for name in wiring.resolved:
            if name in model.model_fields:
                raise fail(
                    f"the wiring receptor {name!r} collides with the field {name!r} of "
                    f"{model.__qualname__}. Rename one of them."
                )
            self._add_wired(name, wiring)
        wiring.raise_problems()

        overrides: dict[str, tuple[Any, Any]] = {}
        if instance is not None:
            for field_name, info in model.model_fields.items():
                if getattr(info, "init", None) is False:
                    continue
                value = getattr(instance, field_name)
                if info.default_factory is None and info.default is not PydanticUndefined and _same(
                    info.default, value
                ):
                    continue
                field_type = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation  # type: ignore[valid-type]
                overrides[field_name] = (field_type, _rebuild_field(info, default=value))
        self.fields = {**overrides, **self.fields}

        doc_infos = [*knob.infos, *(m for m in metadata if isinstance(m, FieldInfo))]
        description = next((fi.description for fi in doc_infos if fi.description), None) or (
            self._description(param_name, knob.owner)
        )
        extra = model.model_config.get("extra") or "forbid"
        flat = self._create(
            f"{cls.__name__}Config", extra, base=model, doc=description,
        )
        setattr(flat, "__config_from_param__", param_name)
        return flat

    def _flatten_default(
        self, knob: _Knob, model: type[BaseModel], fail: Callable[[str], Exception]
    ) -> BaseModel | None:
        """The default instance of a flattened parameter (``None`` = model defaults)."""
        sources = [
            fi for fi in (*knob.infos, *field_infos_of(knob.annotation))
            if fi.default is not PydanticUndefined or fi.default_factory is not None
        ]
        default, factory = knob.default, knob.factory
        if sources:
            if default is not EMPTY or factory is not None or len(sources) > 1:
                raise fail("the parameter declares its default more than once; declare it once.")
            default = sources[0].default
            factory = sources[0].default_factory
            if default is PydanticUndefined:
                default = EMPTY
        if factory is not None:
            if factory is model:
                return None
            if not _is_pure_factory(factory):
                raise fail(
                    f"its default comes from {_callable_name(factory)}, which conscribe "
                    f"cannot evaluate safely; give the parameter a {model.__qualname__}(...) "
                    f"default (or none)."
                )
            stable, default = _evaluate_factory(factory, owner=self.owner, field=knob.name)
            if not stable:
                raise fail(f"its default_factory {_callable_name(factory)} is not deterministic.")
        if default is EMPTY:
            return None
        if default is None:
            raise fail(
                "the parameter default is None, which a flattened config cannot express "
                f"(it always builds a {model.__qualname__}); give the parameter no default "
                f"or a {model.__qualname__}(...) default, or set __config_from_param__ = None."
            )
        if not isinstance(default, model):
            raise fail(
                f"the parameter default {default!r} is not a {model.__qualname__} instance."
            )
        return default

    # -- D6: old-shape guard --------------------------------------------------

    def _legacy_flat_guard(self, wiring: _Wiring) -> dict[str, Any] | None:
        """``__legacy_flat_keys__`` for classes 1.x flattened implicitly, else ``None``."""
        from conscribe.config._utils import find_init_definer

        cls = self.cls
        if _is_model_class(cls) and "__init__" not in cls.__dict__:
            return None  # 1.x took the BaseModel fast path
        definer = find_init_definer(cls)
        if definer is None:
            return None
        sig = _ann.safe_signature(definer.__dict__["__init__"])
        if sig is None:
            return None
        params = [
            p
            for p in sig.parameters.values()
            if p.name != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(params) != 1 or params[0].name in wiring.resolved:
            return None
        param = self.chain.get(params[0].name)
        if param is None:
            return None
        knob = self.knob(param)
        if knob.status != "resolved":
            return None
        model = _ann.unwrap_annotated(knob.annotation)[0]
        if not _is_model_class(model):
            return None
        keys: set[str] = set()
        for field_name, info in model.model_fields.items():
            keys.add(field_name)
            for alias in (info.alias, info.validation_alias):
                if isinstance(alias, str):
                    keys.add(alias)
        return {"param": param.name, "keys": tuple(sorted(keys))}

    # -- model ----------------------------------------------------------------

    def _extra_policy(self) -> str:
        chain = self.chain
        if chain.kind == "model" or BaseModel in (chain.terminal, chain.definer):
            # The constructor ends in BaseModel.__init__: the model's own setting.
            return self.cls.model_config.get("extra") or "forbid"  # type: ignore[attr-defined]
        if chain.kind == "pydantic_dataclass" or is_pydantic_dataclass(chain.terminal):
            owner = chain.definer if chain.kind == "pydantic_dataclass" else chain.terminal
            config = getattr(owner, "__pydantic_config__", None) or {}
            return config.get("extra") or "forbid"
        if not chain.fully_resolved:
            return "allow"
        if chain.var_keyword:
            return "forbid" if any(p.level > 0 for p in chain.params) else "allow"
        if chain.var_positional:
            return "allow"
        return "forbid"

    def _warn_if_unreadable(self) -> None:
        chain = self.chain
        unannotatable: list[type] = []
        for param in chain.params:
            function = param.function
            if (
                param.source == "init"
                and function is not None
                and not hasattr(function, "__code__")
                and param.init_owner not in unannotatable
            ):
                unannotatable.append(param.init_owner)
        for owner in unannotatable:
            # A readable signature without annotations (builtin / extension
            # ``__init__``): the parameters come from the raw signature.
            _ann.warn_at(
                f"{self.owner}: {qualified_name(owner)}.__init__ is not a Python function, so "
                f"its annotations cannot be read; its parameters are typed Any. Declare "
                f"__config_schema__ to describe the config explicitly.",
                ExtractionWarning,
                self.cls,
                stacklevel=4,
            )
        if chain.stop_reason not in ("unreadable", "swallowing_placeholder"):
            return
        where = qualified_name(chain.stopped_at) if chain.stopped_at is not None else "?"
        if chain.stop_reason == "unreadable":
            cause = f"cannot read the signature of {where}.__init__"
        else:
            cause = (
                f"{where}.__init__ is a Protocol placeholder that swallows every "
                f"argument (typing_extensions on Python < 3.12)"
            )
        _ann.warn_at(
            f"{self.owner}: {cause}; its parameters are not in the config, which "
            f"therefore accepts unknown keys (extra='allow'). Declare __config_schema__ "
            f"to describe the config explicitly.",
            ExtractionWarning,
            self.cls,
            stacklevel=4,
        )

    def _create(
        self,
        name: str,
        extra: str,
        *,
        base: type[BaseModel] | None = None,
        guard: dict[str, Any] | None = None,
        doc: str | None = None,
    ) -> type[BaseModel]:
        from pydantic import PydanticSchemaGenerationError, PydanticUserError

        validators = None
        if guard is not None:
            validators = {
                "_conscribe_reject_legacy_flat_keys": model_validator(mode="before")(
                    _reject_legacy_flat_keys
                )
            }
        fields = self.fields
        try:
            model = _create_dynamic_model(
                name, fields, extra, base=base, validators=validators, doc=doc,
            )
        except (PydanticSchemaGenerationError, PydanticUserError):
            fields, degraded = degrade_field_definitions(
                fields, source_class_name=self.owner, source_classes=self.sources,
            )
            wired = [d.field_name for d in degraded if d.field_name in self.wired_fields]
            if wired:
                raise ConfigDeclarationError(
                    self.owner,
                    f"the metadata of the wired field(s) {wired} cannot apply to a "
                    f"registry key (Literal[...]); remove it from the annotation.",
                ) from None
            for record in degraded:
                _ann.warn_at(
                    f"{self.owner}.{record.field_name}: pydantic cannot build a schema for "
                    f"{record.original_type_repr!r} (declared on {record.source_class}); "
                    f"the field's type degrades to Any (default, description and "
                    f"constraints are kept).",
                    DegradationWarning,
                    self.source_classes.get(record.field_name, self.cls),
                    stacklevel=4,
                )
            self.degraded.extend(degraded)
            model = _create_dynamic_model(
                name, fields, extra, base=base, validators=validators, doc=doc,
            )
        setattr(model, _GENERATED_MARKER, True)
        if self.degraded:
            model.__degraded_fields__ = list(self.degraded)  # type: ignore[attr-defined]
        if self.wired_fields:
            model.__wired_fields__ = dict(self.wired_fields)  # type: ignore[attr-defined]
            model.__wired_keys__ = dict(self.wired_keys)  # type: ignore[attr-defined]
        if self.wired_defaults:
            model.__wired_defaults__ = dict(self.wired_defaults)  # type: ignore[attr-defined]
        if guard is not None:
            model.__legacy_flat_keys__ = guard  # type: ignore[attr-defined]
        return model


# ---------------------------------------------------------------------------
# D10: wired field types
# ---------------------------------------------------------------------------


class _WiredTypeMapper:
    """Maps a receptor annotation onto ``Literal[<keys>]`` member by member."""

    def __init__(self, extraction: _Extraction, name: str, literal: Any, keys: tuple[str, ...]) -> None:
        self.extraction = extraction
        self.name = name
        self.literal = literal
        self.keys = keys

    # objects ------------------------------------------------------------------

    def map_object(self, tp: Any, depth: int = 0) -> Any:
        if tp is EMPTY or depth > 40:
            return self.literal
        if tp is None or tp is NoneType:
            return NoneType
        if isinstance(tp, (str, ForwardRef)):
            return self.map_raw(tp, _ann.class_context(self.extraction.cls))
        if isinstance(tp, dataclasses.InitVar):
            return self.map_object(tp.type, depth + 1)
        origin = get_origin(tp)
        if origin is Annotated:
            base, *metadata = get_args(tp)
            return Annotated[(self.map_object(base, depth + 1), *metadata)]  # type: ignore[valid-type]
        if origin in (ClassVar, Final):
            args = get_args(tp)
            return self.map_object(args[0], depth + 1) if args else self.literal
        if origin in _ann.UNION_ORIGINS:
            return self._union([self.map_object(a, depth + 1) for a in get_args(tp)])
        if origin is Literal:
            return self._literal(get_args(tp), _ann.annotation_text(tp))
        if _ann.annotation_shape(tp) != "scalar":
            self._check_nested_literals(tp, depth)
            return list[self.literal]  # type: ignore[name-defined]
        return self.literal

    def _check_nested_literals(self, tp: Any, depth: int) -> None:
        """Check ``Literal`` element types of a container receptor.

        Only the wired *instances* are registry keys: the element types of a
        list / set / tuple and the value type of a mapping.  Mapping keys
        (``dict[Literal["primary", "backup"], P]`` -- a role name per wired
        instance) are not, and are never checked.
        """
        if depth > 40:
            return
        origin = get_origin(tp)
        if origin is Literal:
            self._literal(get_args(tp), _ann.annotation_text(tp))
            return
        args = get_args(tp)
        if origin is Annotated:
            args = args[:1]
        elif origin not in _ann.UNION_ORIGINS and _ann.annotation_shape(tp) == "dict":
            args = args[-1:]  # the value type
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    self._check_nested_literals(item, depth + 1)
            elif arg is not Ellipsis:
                self._check_nested_literals(arg, depth + 1)

    # text -----------------------------------------------------------------------

    def map_raw(self, raw: Any, context: Any) -> Any:
        """Map an unresolved annotation (string, ``ForwardRef`` or typing object)."""
        if isinstance(raw, ForwardRef):
            raw = raw.__forward_arg__
        if not isinstance(raw, str):
            return self.map_object(raw)
        try:
            node = ast.parse(raw.strip(), mode="eval").body
        except SyntaxError:
            return self.literal
        return self._map_node(node, context, 0)

    def _map_node(self, node: ast.AST, context: Any, depth: int) -> Any:
        if depth > 40:
            return self.literal
        if _ann.is_none_node(node):
            return NoneType
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value.strip(), mode="eval").body
            except SyntaxError:
                return self.literal
            return self._map_node(inner, context, depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._union(
                [self._map_node(n, context, depth + 1) for n in _bitor_members(node)]
            )
        if isinstance(node, ast.Subscript):
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            head = _node_head(node.value, context)
            if head in ("Annotated", "ClassVar", "Final"):
                return self._map_node(elements[0], context, depth + 1) if elements else self.literal
            if head == "Optional":
                return self._union([self._map_node(elements[0], context, depth + 1), NoneType])
            if head == "Union":
                return self._union([self._map_node(e, context, depth + 1) for e in elements])
            if head == "Literal":
                values: list[Any] = []
                for element in elements:
                    try:
                        values.append(ast.literal_eval(element))
                    except ValueError:
                        return self.literal
                return self._literal(tuple(values), ast.unparse(node))
        if _ann.annotation_shape(ast.unparse(node), context) != "scalar":
            return list[self.literal]  # type: ignore[name-defined]
        return self.literal

    # shared ---------------------------------------------------------------------

    def _union(self, mapped: list[Any]) -> Any:
        has_none = any(m is NoneType for m in mapped)
        members: list[Any] = []
        for member in mapped:
            if member is NoneType:
                continue
            if not any(_same(member, existing) for existing in members):
                members.append(member)
        if not members:
            members = [self.literal]
        core = members[0] if len(members) == 1 else Union[tuple(members)]  # type: ignore[valid-type]
        return Optional[core] if has_none else core

    def _literal(self, values: tuple[Any, ...], text: str) -> Any:
        has_none = any(v is None for v in values)
        others = [v for v in values if v is not None]
        bad = [v for v in others if not (isinstance(v, str) and v in self.keys)]
        if bad:
            raise ConfigDeclarationError(
                self.extraction.owner,
                f"the wired field's annotation contains {text}, whose value(s) "
                f"{bad} are not keys of its __wiring__ entry ({list(self.keys)}); the "
                f"config value of a wired field is a registry key. Remove the Literal "
                f"(the wiring keys are used) or list only allowed keys.",
                self.name,
            )
        if not others:
            return NoneType
        return Optional[self.literal] if has_none else self.literal


def _bitor_members(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return [*_bitor_members(node.left), *_bitor_members(node.right)]
    return [node]


def _node_head(node: ast.AST, context: Any) -> str | None:
    """The typing construct a subscript head names (looked up, else spelled)."""
    spelled = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
    found, value = _ann._lookup_node(node, _ann._as_context(context))
    if found:
        for target, label in (
            (Annotated, "Annotated"),
            (Optional, "Optional"),
            (Union, "Union"),
            (Literal, "Literal"),
            (ClassVar, "ClassVar"),
            (Final, "Final"),
        ):
            if value is target:
                return label
        return None
    return spelled


# ---------------------------------------------------------------------------
# Old-shape guard validator
# ---------------------------------------------------------------------------


def _reject_legacy_flat_keys(cls: type[BaseModel], data: Any) -> Any:
    """``mode="before"`` validator: reject the 1.x flat keys of a nested block.

    Reads ``cls.__legacy_flat_keys__`` (``{"param": name, "keys": (...)}``)
    and raises a ``legacy_flat_config`` error when the input mapping uses one
    of those keys that is not a field (or alias) of *cls* itself.
    """
    spec = getattr(cls, "__legacy_flat_keys__", None)
    if not spec or not isinstance(data, Mapping):
        return data
    accepted: set[str] = set()
    for field_name, info in cls.model_fields.items():
        accepted.add(field_name)
        for alias in (info.alias, info.validation_alias):
            if isinstance(alias, str):
                accepted.add(alias)
    stale = [key for key in spec["keys"] if key in data and key not in accepted]
    if stale:
        param = spec["param"]
        raise PydanticCustomError(
            LEGACY_FLAT_ERROR,
            f"'{param}' is a nested block in conscribe 2.0; write {{{param}: {{...}}}} "
            f"or set __config_from_param__ = '{param}'",
            {"param": param, "keys": stale},
        )
    return data


# ---------------------------------------------------------------------------
# F5: factories and defaults
# ---------------------------------------------------------------------------

#: Factories that are always pure: builtin container constructors.
_PURE_CONTAINER_FACTORIES: tuple[type, ...] = (list, dict, set, tuple, frozenset)

#: Builtins a pure factory may use: side-effect free constructors and functions.
_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "abs", "all", "any", "bool", "bytearray", "bytes", "chr", "complex", "dict",
        "divmod", "enumerate", "filter", "float", "frozenset", "int", "len", "list",
        "map", "max", "min", "ord", "pow", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip",
    }
)

#: Opcodes that read a global / builtin name (allowed for safe builtins only).
_NAME_LOAD_OPCODES: frozenset[str] = frozenset(
    {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"}
)

#: Opcodes that reach outside the factory's own locals and the safe builtins
#: whatever their argument: imports (they run module code and expose
#: anything), class creation, writes to module globals, class-scope and
#: ``super()`` lookups.
_IMPURE_OPCODES: frozenset[str] = frozenset(
    {
        "IMPORT_NAME", "IMPORT_FROM", "IMPORT_STAR", "LOAD_BUILD_CLASS",
        "STORE_GLOBAL", "DELETE_GLOBAL", "STORE_NAME", "DELETE_NAME",
        "LOAD_CLASSDEREF", "LOAD_FROM_DICT_OR_DEREF", "LOAD_SUPER_ATTR",
    }
)

#: Attribute opcodes; a dunder attribute (``().__class__.__base__``) reaches
#: arbitrary objects from a harmless value.
_ATTR_OPCODES: frozenset[str] = frozenset(
    {"LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR", "DELETE_ATTR"}
)

_UNSAFE_CODE_FLAGS = (
    inspect.CO_VARARGS
    | inspect.CO_VARKEYWORDS
    | inspect.CO_GENERATOR
    | inspect.CO_COROUTINE
    | inspect.CO_ASYNC_GENERATOR
    | inspect.CO_ITERABLE_COROUTINE
)


def _code_globals(code: types.CodeType) -> set[str] | None:
    """Global names a code object (and the code objects nested in it) reads.

    ``None`` when the code does something no pure factory does (see
    ``_IMPURE_OPCODES`` and ``_ATTR_OPCODES``).
    """
    names: set[str] = set()
    for instruction in dis.get_instructions(code):
        opname, argval = instruction.opname, instruction.argval
        if opname in _IMPURE_OPCODES:
            return None
        if opname in _NAME_LOAD_OPCODES:
            if not isinstance(argval, str):
                return None
            names.add(argval)
        elif opname in _ATTR_OPCODES and isinstance(argval, str) and argval.startswith("__"):
            return None
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            nested = _code_globals(const)
            if nested is None:
                return None
            names |= nested
    return names


def _is_pure_factory(factory: Any) -> bool:
    """True for a factory conscribe may evaluate at build time.

    A builtin container constructor (``list``, ``dict``, ``set``, ``tuple``,
    ``frozenset``), or a plain zero-argument function / lambda with no
    closure that reads no global name except side-effect free builtins
    (none of them shadowed in its module) and reaches nothing else: no
    ``import`` (it would run module code and could read the machine's state,
    e.g. ``os.environ``), no class creation, no write to module globals, no
    dunder attribute access.
    """
    if any(factory is container for container in _PURE_CONTAINER_FACTORIES):
        return True
    if not isinstance(factory, types.FunctionType):
        return False
    code = factory.__code__
    if code.co_argcount or code.co_posonlyargcount or code.co_kwonlyargcount:
        return False
    if code.co_flags & _UNSAFE_CODE_FLAGS:
        return False
    if code.co_freevars or factory.__closure__:
        return False
    names = _code_globals(code)
    if names is None:
        return False
    globalns = factory.__globals__
    for name in names:
        if name not in _SAFE_BUILTINS:
            return False
        if name in globalns and globalns[name] is not getattr(builtins, name, None):
            return False  # a module-level name shadows the builtin
    return True


def _evaluate_factory(factory: Callable[[], Any], *, owner: str, field: str) -> tuple[bool, Any]:
    """Evaluate a pure factory twice: ``(True, value)`` when both results are equal."""
    try:
        first = factory()
        second = factory()
    except Exception as exc:  # noqa: BLE001 - reported to the user with the cause
        raise ConfigDeclarationError(
            owner,
            f"the default_factory {_callable_name(factory)} raised "
            f"{type(exc).__name__}: {exc}; the default can never be produced. Fix the "
            f"factory.",
            field,
        ) from exc
    if type(first) is not type(second) or not _same(first, second):
        return False, None
    return True, first


def _rebuild_field(source: FieldInfo, *, default: Any) -> FieldInfo:
    """A fresh ``Field`` with *source*'s carried attributes and a new default.

    *source* is a ``FieldInfo`` built by :func:`config_field` (its constraint
    metadata lives in the field type) or a ``model_fields`` entry of a
    flattened parameter's model (whose metadata the caller moves into the
    type; its ``exclude`` / ``deprecated`` settings are kept, as on the
    inherited fields of that model).
    """
    kwargs: dict[str, Any] = {}
    for attr in (*CONFIG_FIELD_ATTRS, "exclude", "deprecated"):
        value = getattr(source, attr, None)
        if attr == "repr":
            if value is False:
                kwargs["repr"] = False
        elif value is not None:
            kwargs[attr] = value
    if isinstance(kwargs.get("examples"), tuple):
        kwargs["examples"] = list(kwargs["examples"])
    return Field(default=default, **kwargs)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _knob_annotation(res: _ann.ResolvedAnnotation) -> Any:
    if res.status in ("resolved", "degraded"):
        return res.value
    if res.status == "unannotated":
        return EMPTY
    return res.raw


def _annotation_left_to_us(declared: Any) -> bool:
    """Is the stale annotation of an incomplete pydantic field the whole
    ``Annotated[...]`` text (so pydantic merged none of its metadata)?"""
    if isinstance(declared, ForwardRef):
        declared = declared.__forward_arg__
    return isinstance(declared, str) and _text_declares_field(declared)


def _check_field_name(name: str, owner: str) -> None:
    """Refuse a config field name that collides with a pydantic ``model_*`` member."""
    if name.startswith("model_") and hasattr(BaseModel, name):
        raise ConfigDeclarationError(
            owner,
            f"{name!r} collides with pydantic's BaseModel.{name}, so a config model "
            f"cannot have a field of that name. Rename the parameter, or declare "
            f"__config_schema__ (with a field alias) and construct the class yourself.",
            name,
        )


def _with_infos(annotation: Any, infos: tuple[FieldInfo, ...]) -> Any:
    if not infos:
        return annotation
    return Annotated[(annotation, *infos)]  # type: ignore[valid-type]


def _declares_field(knob: _Knob) -> bool:
    """``__config_annotated_only__``: does the parameter declare a ``Field``?"""
    if knob.infos:
        return True
    if knob.status in ("resolved", "degraded"):
        return bool(field_infos_of(knob.annotation))
    if knob.status == "error" and knob.res is not None:
        raw = knob.res.raw
        if isinstance(raw, ForwardRef):
            raw = raw.__forward_arg__
        if isinstance(raw, str):
            return _text_declares_field(raw)
        return bool(field_infos_of(raw))
    return False


def _text_declares_field(text: str) -> bool:
    """An unresolvable annotation spelled ``[Optional[]Annotated[T, <metadata>][]]``."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return False
    for _ in range(10):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value.strip(), mode="eval").body
            except SyntaxError:
                return False
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            members = [m for m in _bitor_members(node) if not _ann.is_none_node(m)]
            if len(members) != 1:
                return False
            node = members[0]
            continue
        if not isinstance(node, ast.Subscript):
            return False
        head = node.value.attr if isinstance(node.value, ast.Attribute) else getattr(node.value, "id", None)
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if head == "Annotated":
            return len(elements) >= 2
        if head in ("Optional", "Union"):
            members = [e for e in elements if not _ann.is_none_node(e)]
            if len(members) != 1:
                return False
            node = members[0]
            continue
        return False
    return False


def _safe_issubclass(tp: Any, parent: type) -> bool:
    """``issubclass`` that is False for non-classes (incl. 3.10 ``list[X]``)."""
    if not isinstance(tp, type) or isinstance(tp, types.GenericAlias):
        return False
    try:
        return issubclass(tp, parent)
    except TypeError:
        return False


def _is_model_class(tp: Any) -> bool:
    """A strict pydantic ``BaseModel`` subclass."""
    return _safe_issubclass(tp, BaseModel) and tp is not BaseModel


def _is_root_model(tp: Any) -> bool:
    from pydantic import RootModel

    return _safe_issubclass(tp, RootModel)


def _is_registered(tp: type) -> bool:
    """Is *tp* (or one of its bases) a registered implementation class?"""
    from conscribe.registration.meta_base import AutoRegistrarBase

    return isinstance(tp, AutoRegistrarBase) or any(
        "__registry_key__" in vars(klass) for klass in tp.__mro__
    )


def _is_pydantic_internal(klass: type) -> bool:
    module = getattr(klass, "__module__", "") or ""
    return module == "pydantic" or module.startswith("pydantic.")


def _callable_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if qualname:
        return f"{module}.{qualname}" if module else qualname
    return repr(obj)


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - incomparable values are simply different
        return False


def _create_dynamic_model(
    name: str,
    field_definitions: dict[str, Any],
    extra: str,
    *,
    base: type[BaseModel] | None = None,
    validators: dict[str, Any] | None = None,
    doc: str | None = None,
) -> type[BaseModel]:
    """Create a config model with the given fields and extra policy.

    The extra policy is passed as a class keyword (``extra=...``), so it
    lands in the model's own ``model_config`` without a helper base class.

    Args:
        name: Model class name.
        field_definitions: ``{field_name: (type, default_or_FieldInfo)}``.
        extra: Extra policy (``"forbid"``, ``"allow"`` or ``"ignore"``).
        base: Model to subclass (the flattened parameter's model); a fresh
            ``BaseModel`` otherwise.
        validators: ``create_model`` ``__validators__``.
        doc: Class docstring (the model's JSON-schema description).

    Returns:
        A dynamically created BaseModel subclass.
    """
    return create_model(  # type: ignore[call-overload,no-any-return]
        name,
        __base__=base or BaseModel,
        __module__=__name__,
        __cls_kwargs__={"extra": extra},
        __validators__=validators,
        __doc__=doc,
        **field_definitions,
    )
